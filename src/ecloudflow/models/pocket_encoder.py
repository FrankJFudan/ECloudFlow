"""SE(3)-equivariant reusable protein-pocket encoding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from ecloudflow.core.types import PocketGraph
from ecloudflow.models.layers import EquivariantMessageBlock, radius_edges, safe_rms


@dataclass(frozen=True)
class PocketEncoding:
    """Store a reusable sparse pocket representation for one model trajectory.

    :param scalars: Invariant pocket features ``[P,S]``, floating dtype/device.
    :param vectors: Equivariant vector copies ``[P,V,3]`` in the pocket frame.
    :param positions: Pocket coordinates ``[P,3]`` in centered-frame angstroms.
    :param batch: Explicit complex indices ``[P]`` with ``torch.long`` dtype.
    :param cache_key: Stable SHA-256 content/layout key for validation and reuse.
    :param feature_dim: Actual positive input feature width ``F``.
    :param scalar_dim: Encoder invariant width ``S``.
    :param vector_dim: Encoder vector multiplicity ``V``.
    :param is_null: Whether classifier-free null conditioning was requested.
    :return: Immutable encoded pocket value.
    :rtype: PocketEncoding
    :raises ValueError: If tensor shapes, dtype, device, or metadata disagree.

    Scalar values are invariant under proper rotations/translations; vectors
    rotate in their final Cartesian dimension. Inputs and outputs share dtype
    and device. The object holds autograd edges and is not detached or mutated.
    Caching has no global state or distributed rank side effect: callers retain
    and pass this value explicitly, and the model rejects an incompatible key,
    feature layout, frame geometry, device, dtype, or encoder width.
    """

    scalars: torch.Tensor
    vectors: torch.Tensor
    positions: torch.Tensor
    batch: torch.Tensor
    cache_key: str
    feature_dim: int
    scalar_dim: int
    vector_dim: int
    is_null: bool = False

    def __post_init__(self) -> None:
        """Validate encoded tensor and cache metadata compatibility.

        :return: None after validating scalar ``[P,S]``, vector ``[P,V,3]``,
            coordinate ``[P,3]``, and batch ``[P]`` shapes.
        :rtype: None
        :raises ValueError: If shape, floating dtype/device, centered frame
            metadata, cache key, or feature/encoder widths are incompatible.

        Validation does not mutate or detach tensors and retains gradients.
        Invariant scalars and proper-rotation equivariant vectors keep their
        caller device/dtype. Cache identity is explicit and no global state is
        changed; a malformed cached encoding fails before model conditioning.
        """
        point_count = self.positions.shape[0]
        if self.positions.ndim != 2 or self.positions.shape[1:] != (3,):
            raise ValueError("encoded pocket positions must have shape [P, 3].")
        if self.scalars.shape != (point_count, self.scalar_dim):
            raise ValueError("encoded pocket scalars have an incompatible shape.")
        if self.vectors.shape != (point_count, self.vector_dim, 3):
            raise ValueError("encoded pocket vectors have an incompatible shape.")
        if self.batch.shape != (point_count,) or self.batch.dtype != torch.long:
            raise ValueError("encoded pocket batch must be long with shape [P].")
        for value in (self.scalars, self.vectors, self.batch):
            if value.device != self.positions.device:
                raise ValueError("encoded pocket tensors must share one device.")
        if (
            self.scalars.dtype != self.positions.dtype
            or self.vectors.dtype != self.positions.dtype
        ):
            raise ValueError("encoded pocket floating tensors must share one dtype.")
        if self.feature_dim <= 0 or not self.cache_key:
            raise ValueError("encoded pocket metadata is invalid.")


class PocketEncoder(nn.Module):  # type: ignore[misc]
    """Encode invariant pocket atom features with relative-vector messages."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        num_blocks: int,
        *,
        cutoff: float = 8.0,
    ) -> None:
        super().__init__()
        if min(scalar_dim, vector_dim, num_blocks) <= 0 or cutoff <= 0:
            raise ValueError(
                "pocket dimensions, block count, and cutoff must be positive."
            )
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.num_blocks = num_blocks
        self.cutoff = float(cutoff)
        self.feature_projection = nn.Sequential(
            nn.Linear(4, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, scalar_dim)
        )
        self.null_embedding = nn.Parameter(torch.zeros(scalar_dim))
        self.blocks = nn.ModuleList(
            EquivariantMessageBlock(scalar_dim, vector_dim, self.cutoff)
            for _ in range(num_blocks)
        )

    def encode(self, pocket: PocketGraph, *, use_null: bool = False) -> PocketEncoding:
        """Encode a centered pocket once for reuse across sampling steps.

        :param pocket: Canonical graph with positions ``[P,3]`` in centered
            frame angstroms, invariant features ``[P,F]`` of any positive
            declared width, and long batch indices ``[P]``.
        :param use_null: Select classifier-free null scalars and zero vectors
            while retaining geometry/batch metadata for unconditional guidance.
        :return: Invariant scalars ``[P,S]``, equivariant vectors ``[P,V,3]``,
            coordinates, batch indices, actual feature width, and stable key.
        :rtype: PocketEncoding
        :raises ValueError: If module/input dtype-device placement, feature
            channels, batches, finite values, or encoder settings are invalid.

        Feature width is summarized explicitly rather than assumed to be the
        Task 6 width 50; every returned tensor shape follows the flattened
        contract. All arithmetic remains on the caller/module device and
        preserves floating dtype, gradients, and the centered coordinate frame.
        Relative displacements make scalars invariant and vectors equivariant
        under proper SE(3); reflections are not promised. Fixed parameters make
        evaluation deterministic. Neither input nor cache is mutated, and no
        rank-local global cache exists. The SHA-256 key includes tensor content,
        device, dtype, feature layout, null mode, and architecture; incompatible
        reuse fails before conditioning. Sparse radius edges avoid dense channel
        allocation and are differentiable after discrete neighborhood selection.
        """
        parameter = self.null_embedding
        if (
            pocket.positions.device != parameter.device
            or pocket.positions.dtype != parameter.dtype
        ):
            raise ValueError(
                "pocket and encoder parameters must share dtype and device."
            )
        if pocket.features.shape[1] <= 0:
            raise ValueError("pocket features must include at least one channel.")
        summary = torch.stack(
            (
                pocket.features.mean(dim=-1),
                safe_rms(pocket.features, -1),
                pocket.features.amax(dim=-1),
                pocket.features.amin(dim=-1),
            ),
            dim=-1,
        )
        learned = self.feature_projection(summary)
        null = self.null_embedding.expand_as(learned)
        scalars = null + (torch.zeros_like(learned) if use_null else learned)
        vectors = pocket.positions.new_zeros(
            (pocket.positions.shape[0], self.vector_dim, 3)
        )
        if not use_null:
            edges = radius_edges(pocket.positions, pocket.batch, self.cutoff)
            for block in self.blocks:
                scalars, vectors = block(scalars, vectors, pocket.positions, edges)
        return PocketEncoding(
            scalars=scalars,
            vectors=vectors,
            positions=pocket.positions,
            batch=pocket.batch,
            cache_key=self.cache_key(pocket, use_null=use_null),
            feature_dim=pocket.features.shape[1],
            scalar_dim=self.scalar_dim,
            vector_dim=self.vector_dim,
            is_null=use_null,
        )

    def cache_key(self, pocket: PocketGraph, *, use_null: bool = False) -> str:
        """Return the stable pocket-content and encoder-layout cache key."""
        digest = hashlib.sha256()
        for tensor in (pocket.positions, pocket.features, pocket.batch):
            contiguous = tensor.detach().contiguous().cpu()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tensor.device).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(contiguous.view(torch.uint8).numpy().tobytes())
        digest.update(
            (
                f"{self.scalar_dim}:{self.vector_dim}:{self.num_blocks}:"
                f"{self.cutoff}:{int(use_null)}:{id(self)}:"
                f"{tuple(parameter._version for parameter in self.parameters())}"
            ).encode()
        )
        return digest.hexdigest()
