"""Sparse invariant/equivariant geometry utilities for ECloudFlow models."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import radius as pyg_radius  # type: ignore[import-untyped]
from torch_geometric.nn import (
    radius_graph as pyg_radius_graph,  # type: ignore[import-untyped]
)


def radius_edges(
    positions: torch.Tensor, batch: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Build deterministic directed radius edges without dense feature tensors.

    :param positions: Cartesian coordinates ``[N,3]`` in angstroms.
    :param batch: Complex indices ``[N]`` on the coordinate device.
    :param cutoff: Strictly positive radius in angstroms.
    :return: Directed non-self edges ``[2,E]`` with source then target rows.
    :rtype: torch.Tensor
    :raises ValueError: If shapes, devices, dtypes, or cutoff are invalid.

    Pair indices are generated independently per complex and filtered using
    relative displacement norms. No dense ``[N,N,C]`` channel tensor is
    allocated, and translations therefore cannot affect the edge set.
    """
    _validate_points_and_batch(positions, batch)
    if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or cutoff <= 0:
        raise ValueError("cutoff must be positive.")
    try:
        edges = pyg_radius_graph(
            positions,
            r=float(cutoff),
            batch=batch,
            loop=False,
            max_num_neighbors=max(positions.shape[0], 1),
            flow="source_to_target",
        )
    except ImportError:
        edges = _native_radius_edges(positions, batch, float(cutoff))
    if edges.shape[1] == 0:
        return edges
    order = torch.argsort(edges[0] * max(positions.shape[0], 1) + edges[1])
    return edges[:, order]


def _native_radius_edges(
    positions: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Provide bounded-memory sparse edges when PyG's compiled extra is absent.

    Candidate source rows are deterministically chunked so at most roughly
    65,536 pairs are resident by default. Batch boundaries, caller device,
    strict cutoff semantics, directed ordering, and exact edge values match the
    PyG path without a global Cartesian-product allocation.
    """
    pieces: list[torch.Tensor] = []
    for complex_index in torch.unique(batch, sorted=True):
        nodes = torch.nonzero(batch == complex_index, as_tuple=False).flatten()
        if nodes.numel() < 2:
            continue
        rows = chunk_size or max(1, 65536 // nodes.numel())
        for source, target in _candidate_pair_chunks(nodes, rows):
            keep = source != target
            source = source[keep]
            target = target[keep]
            distance = (positions[source] - positions[target]).norm(dim=-1)
            keep = distance < cutoff
            if bool(keep.any()):
                pieces.append(torch.stack((source[keep], target[keep])))
    if not pieces:
        return torch.empty((2, 0), dtype=torch.long, device=positions.device)
    return torch.cat(pieces, dim=1)


def _candidate_pair_chunks(nodes: torch.Tensor, chunk_size: int):
    """Yield deterministic Cartesian index chunks with bounded source rows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    for start in range(0, nodes.numel(), chunk_size):
        source_rows = nodes[start : start + chunk_size]
        yield (
            source_rows.repeat_interleave(nodes.numel()),
            nodes.repeat(source_rows.numel()),
        )


def _candidate_bipartite_chunks(
    source: torch.Tensor, target: torch.Tensor, max_candidates: int = 65536
):
    """Yield deterministic bipartite candidates under a fixed memory bound."""
    rows = max(1, max_candidates // max(target.numel(), 1))
    for start in range(0, source.numel(), rows):
        source_rows = source[start : start + rows]
        yield (
            source_rows.repeat_interleave(target.numel()),
            target.repeat(source_rows.numel()),
        )


def bipartite_radius_edges(
    source_positions: torch.Tensor,
    source_batch: torch.Tensor,
    target_positions: torch.Tensor,
    target_batch: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Build sparse source-to-target cross-radius edges within each complex."""
    _validate_points_and_batch(source_positions, source_batch)
    _validate_points_and_batch(target_positions, target_batch)
    if (
        source_positions.dtype != target_positions.dtype
        or source_positions.device != target_positions.device
    ):
        raise ValueError("source and target positions must share dtype and device.")
    if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or cutoff <= 0:
        raise ValueError("cutoff must be positive.")
    try:
        target_source = pyg_radius(
            source_positions,
            target_positions,
            r=float(cutoff),
            batch_x=source_batch,
            batch_y=target_batch,
            max_num_neighbors=max(source_positions.shape[0], 1),
        )
        edges = target_source.flip(0)
        if edges.shape[1] == 0:
            return edges
        order = torch.argsort(edges[0] * max(target_positions.shape[0], 1) + edges[1])
        return edges[:, order]
    except ImportError:
        pass
    pieces: list[torch.Tensor] = []
    common = torch.unique(source_batch, sorted=True)
    for complex_index in common:
        source = torch.nonzero(source_batch == complex_index, as_tuple=False).flatten()
        target = torch.nonzero(target_batch == complex_index, as_tuple=False).flatten()
        if source.numel() == 0 or target.numel() == 0:
            continue
        for source_grid, target_grid in _candidate_bipartite_chunks(source, target):
            distance = (
                source_positions[source_grid] - target_positions[target_grid]
            ).norm(dim=-1)
            keep = distance < float(cutoff)
            if bool(keep.any()):
                pieces.append(torch.stack((source_grid[keep], target_grid[keep])))
    if not pieces:
        return torch.empty((2, 0), dtype=torch.long, device=source_positions.device)
    return torch.cat(pieces, dim=1)


def radial_features(
    distance: torch.Tensor, channels: int = 8, cutoff: float = 8.0
) -> torch.Tensor:
    """Expand invariant distances in smooth Gaussian radial channels."""
    centers = torch.linspace(
        0.0, cutoff, channels, dtype=distance.dtype, device=distance.device
    )
    width = cutoff / max(channels - 1, 1)
    return torch.exp(-((distance[:, None] - centers[None, :]) / width).square())


def scatter_sum(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    """Sum sparse values into a caller-device leading dimension."""
    output = values.new_zeros((count, *values.shape[1:]))
    if index.numel():
        output.index_add_(0, index, values)
    return output


def segment_softmax(
    logits: torch.Tensor, index: torch.Tensor, count: int
) -> torch.Tensor:
    """Normalize one-dimensional logits independently for each destination."""
    if logits.ndim != 1 or index.shape != logits.shape or index.dtype != torch.long:
        raise ValueError("segment softmax expects logits [E] and long index [E].")
    output = torch.empty_like(logits)
    for destination in range(count):
        selected = index == destination
        if bool(selected.any()):
            output[selected] = torch.softmax(logits[selected], dim=0)
    return output


def safe_rms(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Return RMS values with a finite derivative at an all-zero input."""
    mean_square = values.square().mean(dim=dim)
    return mean_square.clamp_min(torch.finfo(values.dtype).eps).sqrt()


class EquivariantMessageBlock(nn.Module):  # type: ignore[misc]
    """Update invariant scalars and Cartesian vector copies from relative edges."""

    def __init__(self, scalar_dim: int, vector_dim: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.scalar_message = nn.Sequential(
            nn.Linear(scalar_dim + 8, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(2 * scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.vector_weight = nn.Linear(scalar_dim, vector_dim)
        self.norm = nn.LayerNorm(scalar_dim)

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one relative-geometry equivariant message-passing update."""
        if edge_index.shape[1] == 0:
            return self.norm(scalars), vectors
        source, target = edge_index
        displacement = positions[source] - positions[target]
        distance = displacement.norm(dim=-1).clamp_min(torch.finfo(positions.dtype).eps)
        message = self.scalar_message(
            torch.cat(
                (scalars[source], radial_features(distance, cutoff=self.cutoff)), dim=-1
            )
        )
        aggregated = scatter_sum(message, target, scalars.shape[0])
        updated_scalars = self.norm(
            scalars + self.scalar_update(torch.cat((scalars, aggregated), dim=-1))
        )
        unit = displacement / distance[:, None]
        vector_message = self.vector_weight(message)[..., None] * unit[:, None, :]
        updated_vectors = vectors + scatter_sum(
            vector_message, target, scalars.shape[0]
        )
        return updated_scalars, updated_vectors


def _validate_points_and_batch(positions: torch.Tensor, batch: torch.Tensor) -> None:
    """Validate flattened coordinate and batch inputs to sparse edge builders."""
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or not positions.is_floating_point()
    ):
        raise ValueError("positions must be a floating tensor with shape [N, 3].")
    if (
        batch.shape != (positions.shape[0],)
        or batch.dtype != torch.long
        or batch.device != positions.device
    ):
        raise ValueError(
            "batch must be a long tensor with shape [N] on the positions device."
        )
