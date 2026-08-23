"""Joint invariant/equivariant ligand message-passing backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ecloudflow.core.types import MolecularState
from ecloudflow.models.layers import (
    bipartite_radius_edges,
    radial_features,
    radius_edges,
    safe_rms,
    scatter_sum,
)
from ecloudflow.models.pocket_encoder import PocketEncoding


@dataclass(frozen=True)
class BackboneOutput:
    """Internal ligand invariant features and equivariant directions."""

    scalars: torch.Tensor
    directions: torch.Tensor
    pooled: torch.Tensor


class JointLigandBackbone(nn.Module):  # type: ignore[misc]
    """Fuse ligand state, time, task, and reusable pocket encoding."""

    def __init__(
        self, scalar_dim: int, num_blocks: int, *, cutoff: float = 8.0
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.state_projection = nn.Sequential(
            nn.Linear(8, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, scalar_dim)
        )
        self.time_projection = nn.Sequential(
            nn.Linear(3, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, scalar_dim)
        )
        self.task_projection = nn.Linear(8, scalar_dim)
        self.property_projection = nn.Linear(1, scalar_dim)
        self.cross_messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * scalar_dim + 9, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, scalar_dim),
            )
            for _ in range(num_blocks)
        )
        self.ligand_messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(scalar_dim + 8, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, scalar_dim),
            )
            for _ in range(num_blocks)
        )
        self.updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * scalar_dim, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, scalar_dim),
            )
            for _ in range(num_blocks)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(scalar_dim) for _ in range(num_blocks))
        self.direction_weights = nn.ModuleList(
            nn.Linear(scalar_dim, 2) for _ in range(num_blocks)
        )
        self.ligand_direction_weights = nn.ModuleList(
            nn.Linear(scalar_dim, 1) for _ in range(num_blocks)
        )

    def forward(
        self,
        state: MolecularState,
        time: torch.Tensor,
        pocket: PocketEncoding,
        electron_summary: torch.Tensor,
        task_features: torch.Tensor,
        property_features: torch.Tensor,
    ) -> BackboneOutput:
        """Return joint invariant features and equivariant node directions."""
        node_count = state.positions.shape[0]
        batch_size = time.shape[0]
        pocket_pooled = scatter_sum(pocket.scalars, pocket.batch, batch_size)
        pocket_counts = (
            torch.bincount(pocket.batch, minlength=batch_size)
            .clamp_min(1)
            .to(pocket.scalars.dtype)
        )
        pocket_pooled = pocket_pooled / pocket_counts[:, None]
        base_pooled = pocket_pooled + self.time_projection(_time_features(time))
        if not pocket.is_null:
            base_pooled = (
                base_pooled
                + self.task_projection(task_features)
                + self.property_projection(property_features)
            )
        if node_count == 0:
            return BackboneOutput(
                scalars=state.positions.new_zeros((0, self.scalar_dim)),
                directions=state.positions.new_zeros((0, 3)),
                pooled=base_pooled,
            )
        state_summary = torch.stack(
            (
                state.atom_logits.mean(-1),
                safe_rms(state.atom_logits, -1),
                state.charge_logits.mean(-1),
                safe_rms(state.charge_logits, -1),
                electron_summary[:, 0],
                electron_summary[:, 1],
                state.bond_logits.new_zeros(node_count),
                state.positions.new_ones(node_count),
            ),
            dim=-1,
        )
        if state.halfedge_index.shape[1]:
            source, target = state.halfedge_index
            bond_signal = safe_rms(state.bond_logits, -1)
            state_summary[:, 6].index_add_(0, source, bond_signal)
            state_summary[:, 6].index_add_(0, target, bond_signal)
        scalars = self.state_projection(state_summary) + base_pooled[state.node_batch]
        directions = state.positions.new_zeros((node_count, 3))
        cross = bipartite_radius_edges(
            pocket.positions,
            pocket.batch,
            state.positions,
            state.node_batch,
            self.cutoff,
        )
        ligand_edges = radius_edges(state.positions, state.node_batch, self.cutoff)
        for (
            message_network,
            ligand_message_network,
            update,
            norm,
            direction_head,
            ligand_direction_head,
        ) in zip(
            self.cross_messages,
            self.ligand_messages,
            self.updates,
            self.norms,
            self.direction_weights,
            self.ligand_direction_weights,
            strict=True,
        ):
            aggregated = torch.zeros_like(scalars)
            if cross.shape[1]:
                pocket_source, ligand_target = cross
                displacement = (
                    pocket.positions[pocket_source] - state.positions[ligand_target]
                )
                distance = displacement.norm(dim=-1).clamp_min(
                    torch.finfo(state.positions.dtype).eps
                )
                unit = displacement / distance[:, None]
                pocket_vector = pocket.vectors[pocket_source].mean(dim=1)
                alignment = (pocket_vector * unit).sum(dim=-1, keepdim=True)
                message = message_network(
                    torch.cat(
                        (
                            scalars[ligand_target],
                            pocket.scalars[pocket_source],
                            radial_features(distance, cutoff=self.cutoff),
                            alignment,
                        ),
                        dim=-1,
                    )
                )
                aggregated = aggregated + scatter_sum(
                    message, ligand_target, node_count
                )
                weights = direction_head(message)
                vector_message = weights[:, :1] * unit + weights[:, 1:] * pocket_vector
                directions = directions + scatter_sum(
                    vector_message, ligand_target, node_count
                )
            if ligand_edges.shape[1]:
                ligand_source, ligand_target = ligand_edges
                ligand_displacement = (
                    state.positions[ligand_source] - state.positions[ligand_target]
                )
                ligand_distance = ligand_displacement.norm(dim=-1).clamp_min(
                    torch.finfo(state.positions.dtype).eps
                )
                ligand_unit = ligand_displacement / ligand_distance[:, None]
                ligand_message = ligand_message_network(
                    torch.cat(
                        (
                            scalars[ligand_source],
                            radial_features(ligand_distance, cutoff=self.cutoff),
                        ),
                        dim=-1,
                    )
                )
                aggregated = aggregated + scatter_sum(
                    ligand_message, ligand_target, node_count
                )
                ligand_vector_message = (
                    ligand_direction_head(ligand_message) * ligand_unit
                )
                directions = directions + scatter_sum(
                    ligand_vector_message, ligand_target, node_count
                )
            scalars = norm(scalars + update(torch.cat((scalars, aggregated), dim=-1)))
        pooled = scatter_sum(scalars, state.node_batch, batch_size)
        counts = (
            torch.bincount(state.node_batch, minlength=batch_size)
            .clamp_min(1)
            .to(scalars.dtype)
        )
        pooled = base_pooled + pooled / counts[:, None]
        return BackboneOutput(scalars=scalars, directions=directions, pooled=pooled)


def _time_features(time: torch.Tensor) -> torch.Tensor:
    """Build deterministic invariant time features on the caller device."""
    return torch.stack(
        (time, torch.sin(torch.pi * time), torch.cos(torch.pi * time)), dim=-1
    )
