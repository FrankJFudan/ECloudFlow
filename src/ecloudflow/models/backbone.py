"""Joint parity-aware invariant/equivariant ligand message-passing backbone."""

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
    segment_softmax,
)
from ecloudflow.models.pocket_encoder import PocketEncoding


@dataclass(frozen=True)
class BackboneOutput:
    """Internal ligand invariant features and full Cartesian vector channels."""

    scalars: torch.Tensor
    vectors: torch.Tensor
    pooled: torch.Tensor


class JointLigandBackbone(nn.Module):  # type: ignore[misc]
    """Fuse ligand state, time, named targets, field moments, and pocket encoding."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        num_blocks: int,
        *,
        cutoff: float = 8.0,
    ) -> None:
        super().__init__()
        if min(scalar_dim, vector_dim, num_blocks) <= 0:
            raise ValueError("backbone dimensions and block count must be positive.")
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.state_projection = nn.Sequential(
            nn.Linear(8, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, scalar_dim)
        )
        self.time_projection = nn.Sequential(
            nn.Linear(3, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, scalar_dim)
        )
        self.task_projection = nn.Linear(8, scalar_dim)
        self.property_projection = nn.Linear(17, scalar_dim)
        self.interaction_projection = nn.Linear(2, scalar_dim)
        self.field_projection = nn.Linear(4, scalar_dim)
        self.field_vector_weights = nn.Linear(4, vector_dim)
        self.cross_messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * scalar_dim + 9, scalar_dim),
                nn.SiLU(),
                nn.Linear(scalar_dim, scalar_dim),
            )
            for _ in range(num_blocks)
        )
        self.cross_attention = nn.ModuleList(
            nn.Linear(scalar_dim, 1) for _ in range(num_blocks)
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
        self.cross_vector_weights = nn.ModuleList(
            nn.Linear(scalar_dim, 2 * vector_dim) for _ in range(num_blocks)
        )
        self.ligand_vector_weights = nn.ModuleList(
            nn.Linear(scalar_dim, vector_dim) for _ in range(num_blocks)
        )
        self.vector_gates = nn.ModuleList(
            nn.Linear(scalar_dim, vector_dim) for _ in range(num_blocks)
        )
        self.time_films = nn.ModuleList(
            nn.Linear(3, 2 * scalar_dim) for _ in range(num_blocks)
        )
        self.parity_updates = nn.ModuleList(
            nn.Linear(1, scalar_dim) for _ in range(num_blocks)
        )

    def forward(
        self,
        state: MolecularState,
        time: torch.Tensor,
        pocket: PocketEncoding,
        electron_summary: torch.Tensor,
        task_features: torch.Tensor,
        property_features: torch.Tensor,
        interaction_features: torch.Tensor,
        field_features: torch.Tensor,
        field_vectors: torch.Tensor,
    ) -> BackboneOutput:
        """Return parity-aware invariant features and ``[N,V,3]`` vectors."""
        node_count = state.positions.shape[0]
        batch_size = time.shape[0]
        time_features = _time_features(time)
        pocket_pooled = scatter_sum(pocket.scalars, pocket.batch, batch_size)
        pocket_counts = (
            torch.bincount(pocket.batch, minlength=batch_size)
            .clamp_min(1)
            .to(pocket.scalars.dtype)
        )
        pocket_pooled = pocket_pooled / pocket_counts[:, None]
        base_pooled = pocket_pooled + self.time_projection(time_features)
        if not pocket.is_null:
            base_pooled = (
                base_pooled
                + self.task_projection(task_features)
                + self.property_projection(property_features)
                + self.interaction_projection(interaction_features)
                + self.field_projection(field_features)
            )
        if node_count == 0:
            return BackboneOutput(
                scalars=state.positions.new_zeros((0, self.scalar_dim)),
                vectors=state.positions.new_zeros((0, self.vector_dim, 3)),
                pooled=base_pooled,
            )
        bond_node_signal = state.bond_logits.new_zeros(node_count)
        if state.halfedge_index.shape[1]:
            source, target = state.halfedge_index
            bond_signal = safe_rms(state.bond_logits, -1)
            bond_node_signal.index_add_(0, source, bond_signal)
            bond_node_signal.index_add_(0, target, bond_signal)
        state_summary = torch.stack(
            (
                state.atom_logits.mean(-1),
                safe_rms(state.atom_logits, -1),
                state.charge_logits.mean(-1),
                safe_rms(state.charge_logits, -1),
                electron_summary[:, 0],
                electron_summary[:, 1],
                bond_node_signal,
                state.positions.new_ones(node_count),
            ),
            dim=-1,
        )
        scalars = self.state_projection(state_summary) + base_pooled[state.node_batch]
        vectors = state.positions.new_zeros((node_count, self.vector_dim, 3))
        if not pocket.is_null:
            field_weights = self.field_vector_weights(field_features)
            vectors = (
                vectors
                + field_weights[state.node_batch, :, None]
                * field_vectors[state.node_batch, None, :]
            )
        cross = (
            bipartite_radius_edges(
                pocket.positions,
                pocket.batch,
                state.positions,
                state.node_batch,
                self.cutoff,
            )
            if not pocket.is_null
            else torch.empty((2, 0), dtype=torch.long, device=state.positions.device)
        )
        ligand_edges = radius_edges(state.positions, state.node_batch, self.cutoff)
        modules = zip(
            self.cross_messages,
            self.cross_attention,
            self.ligand_messages,
            self.updates,
            self.norms,
            self.cross_vector_weights,
            self.ligand_vector_weights,
            self.vector_gates,
            self.time_films,
            self.parity_updates,
            strict=True,
        )
        for (
            cross_network,
            attention_head,
            ligand_network,
            update,
            norm,
            cross_vector_head,
            ligand_vector_head,
            vector_gate,
            time_film,
            parity_update,
        ) in modules:
            scale, shift = time_film(time_features).chunk(2, dim=-1)
            modulated = scalars * (1.0 + 0.1 * torch.tanh(scale[state.node_batch]))
            modulated = modulated + shift[state.node_batch]
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
                pocket_vectors = pocket.vectors[pocket_source]
                alignment = (
                    (pocket_vectors * unit[:, None, :])
                    .sum(dim=-1)
                    .mean(dim=-1, keepdim=True)
                )
                message = cross_network(
                    torch.cat(
                        (
                            modulated[ligand_target],
                            pocket.scalars[pocket_source],
                            radial_features(distance, cutoff=self.cutoff),
                            alignment,
                        ),
                        dim=-1,
                    )
                )
                attention = segment_softmax(
                    attention_head(message).squeeze(-1), ligand_target, node_count
                )
                weighted_message = attention[:, None] * message
                aggregated = aggregated + scatter_sum(
                    weighted_message, ligand_target, node_count
                )
                vector_weights = cross_vector_head(message).reshape(
                    -1, 2, self.vector_dim
                )
                vector_message = (
                    vector_weights[:, 0, :, None] * unit[:, None, :]
                    + vector_weights[:, 1, :, None] * pocket_vectors
                )
                vectors = vectors + scatter_sum(
                    attention[:, None, None] * vector_message,
                    ligand_target,
                    node_count,
                )
            if ligand_edges.shape[1]:
                ligand_source, ligand_target = ligand_edges
                displacement = (
                    state.positions[ligand_source] - state.positions[ligand_target]
                )
                distance = displacement.norm(dim=-1).clamp_min(
                    torch.finfo(state.positions.dtype).eps
                )
                unit = displacement / distance[:, None]
                message = ligand_network(
                    torch.cat(
                        (
                            modulated[ligand_source],
                            radial_features(distance, cutoff=self.cutoff),
                        ),
                        dim=-1,
                    )
                )
                aggregated = aggregated + scatter_sum(
                    message, ligand_target, node_count
                )
                vector_message = (
                    ligand_vector_head(message)[..., None] * unit[:, None, :]
                )
                vectors = vectors + scatter_sum(
                    vector_message, ligand_target, node_count
                )
            chirality = state.positions.new_zeros((node_count, 1))
            if self.vector_dim >= 3:
                chirality = (
                    torch.linalg.cross(vectors[:, 0], vectors[:, 1], dim=-1)
                    * vectors[:, 2]
                ).sum(dim=-1, keepdim=True)
            scalars = norm(
                modulated
                + update(torch.cat((modulated, aggregated), dim=-1))
                + parity_update(chirality)
            )
            vectors = vectors * torch.sigmoid(vector_gate(scalars))[..., None]
        pooled = scatter_sum(scalars, state.node_batch, batch_size)
        counts = (
            torch.bincount(state.node_batch, minlength=batch_size)
            .clamp_min(1)
            .to(scalars.dtype)
        )
        pooled = base_pooled + pooled / counts[:, None]
        return BackboneOutput(scalars=scalars, vectors=vectors, pooled=pooled)


def _time_features(time: torch.Tensor) -> torch.Tensor:
    """Build deterministic invariant time features on the caller device."""
    return torch.stack(
        (time, torch.sin(torch.pi * time), torch.cos(torch.pi * time)), dim=-1
    )
