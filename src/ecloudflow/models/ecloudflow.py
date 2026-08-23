"""Public pocket-conditioned joint ECloudFlow neural model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from ecloudflow.config import ModelConfig
from ecloudflow.core.types import GenerationCondition, MolecularState, PocketGraph
from ecloudflow.models.backbone import JointLigandBackbone
from ecloudflow.models.count_predictor import AtomCountPredictor
from ecloudflow.models.heads import ScalarHead, SymmetricPairHead
from ecloudflow.models.layers import safe_rms, scatter_sum
from ecloudflow.models.pocket_encoder import PocketEncoder, PocketEncoding


@dataclass(frozen=True)
class ModelPrediction:
    """Collect every typed output of the joint generative backbone.

    :param position_velocity: Cartesian flow velocity ``[N,3]`` in angstroms
        per unit path time, on the state floating dtype/device.
    :param position_score: Cartesian score ``[N,3]`` in inverse angstroms.
    :param electron_velocity: Packed equivariant latent flow ``[N,C]``.
    :param electron_score: Packed equivariant latent score ``[N,C]``.
    :param atom_logits: Invariant atom endpoint logits ``[N,A]``.
    :param charge_logits: Invariant formal-charge endpoint logits ``[N,Q]``.
    :param bond_logits: Invariant logits ``[E,K]`` for one canonical unordered
        halfedge per row; no dense pair tensor is materialized.
    :param count_logits: Normalized invariant log probabilities ``[B,M+1]``.
    :param affinity: Invariant scalar affinity auxiliary ``[B]``.
    :param interaction_logits: Invariant interaction auxiliary ``[B]``.
    :param pocket_cache_key: Validated stable key of the reused pocket encoding.
    :return: Immutable prediction container retaining all autograd edges.
    :rtype: ModelPrediction
    :raises ValueError: During model construction if any output contract fails.

    Proper rotations act on Cartesian and packed-irrep outputs while translations
    cancel from coordinate updates; scalar logits and auxiliaries are invariant.
    Fixed fragment masks are applied before construction, so downstream exact
    clamping sees zero coordinate motion and clean categorical fixed fields.
    Tensors are neither detached nor moved, and the record owns no cache or
    distributed side effect. Empty leading dimensions are valid.
    """

    position_velocity: torch.Tensor
    position_score: torch.Tensor
    electron_velocity: torch.Tensor
    electron_score: torch.Tensor
    atom_logits: torch.Tensor
    charge_logits: torch.Tensor
    bond_logits: torch.Tensor
    count_logits: torch.Tensor
    affinity: torch.Tensor
    interaction_logits: torch.Tensor
    pocket_cache_key: str

    def __post_init__(self) -> None:
        """Validate joint output shape, placement, and sparse graph semantics.

        :return: None after checking Cartesian ``[N,3]``, packed electron
            ``[N,C]``, node logits, sparse unordered-halfedge logits ``[E,K]``,
            and per-complex auxiliary shapes.
        :rtype: None
        :raises ValueError: If shapes, floating dtype/device placement, finite
            allowed values, or pocket cache identity are inconsistent.

        Position velocity/score are proper-rotation equivariant in the centered
        angstrom frame. Electron tensors retain packed irrep layout; categorical
        values, count logits, and auxiliaries are invariant. Fixed-field mask
        semantics are already applied by the model. This immutable check does
        not mutate/detach values or gradients, allocate a dense ``[N,N,C]``
        pair tensor, move devices/dtypes, or create distributed/global cache
        state. One bond row remains symmetric for each unordered halfedge.
        """
        node_count = self.position_velocity.shape[0]
        if self.position_velocity.shape != (
            node_count,
            3,
        ) or self.position_score.shape != (node_count, 3):
            raise ValueError("position prediction tensors must have shape [N, 3].")
        for name, value in (
            ("electron_velocity", self.electron_velocity),
            ("electron_score", self.electron_score),
            ("atom_logits", self.atom_logits),
            ("charge_logits", self.charge_logits),
        ):
            if value.ndim != 2 or value.shape[0] != node_count:
                raise ValueError(f"{name} must have shape [N, C].")
        if self.bond_logits.ndim != 2 or self.count_logits.ndim != 2:
            raise ValueError("bond and count logits must be rank-two tensors.")
        batch_size = self.count_logits.shape[0]
        if self.affinity.shape != (batch_size,) or self.interaction_logits.shape != (
            batch_size,
        ):
            raise ValueError("auxiliary predictions must have shape [B].")
        reference = self.position_velocity
        for value in (
            self.position_score,
            self.electron_velocity,
            self.electron_score,
            self.atom_logits,
            self.charge_logits,
            self.bond_logits,
            self.count_logits,
            self.affinity,
            self.interaction_logits,
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(
                    "all prediction tensors must share floating dtype and device."
                )
        if not self.pocket_cache_key:
            raise ValueError("pocket cache key must be non-empty.")


class _PackedElectronHead(nn.Module):  # type: ignore[misc]
    """Predict one packed-irrep tensor without mixing representation components."""

    def __init__(self, scalar_dim: int, vector_dim: int) -> None:
        super().__init__()
        self.vector_dim = vector_dim
        self.parameters_head = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, 3 + vector_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
        backbone_vectors: torch.Tensor,
        layout: tuple[int, int, tuple[tuple[int, int, int], ...]],
    ) -> torch.Tensor:
        """Apply scalar gates equally across each irrep component."""
        scalar_copies, vector_copies, higher = layout
        values = self.parameters_head(hidden)
        scalar_gate, scalar_shift, nonscalar_gate = values[:, :3].unbind(-1)
        vector_weights = values[:, 3:]
        direction = torch.einsum("nv,nvc->nc", vector_weights, backbone_vectors)
        pieces = [
            latent[:, :scalar_copies] * scalar_gate[:, None] + scalar_shift[:, None]
        ]
        offset = scalar_copies
        vector_width = 3 * vector_copies
        vectors = latent[:, offset : offset + vector_width].reshape(
            -1, vector_copies, 3
        )
        vectors = vectors * nonscalar_gate[:, None, None] + direction[:, None, :]
        pieces.append(vectors.reshape(-1, vector_width))
        offset += vector_width
        for _, multiplicity, width in higher:
            block_width = multiplicity * width
            block = latent[:, offset : offset + block_width]
            pieces.append(block * nonscalar_gate[:, None])
            offset += block_width
        return torch.cat(pieces, dim=-1)


class ECloudFlowModel(nn.Module):  # type: ignore[misc]
    """First joint SE(3)-equivariant pocket-conditioned ligand backbone.

    :param scalar_dim: Positive invariant hidden width.
    :param vector_dim: Pocket/backbone Cartesian vector multiplicity, at least
        three so the parity-sensitive scalar triple product is non-degenerate.
    :param num_blocks: Positive pocket and cross-message block count.
    :param lmax: Largest packed electron irrep order, in ``[0,4]``.
    :param electron_latent_dim: Exact packed electron channel width ``C``.
    :param electron_vector_dim: Multiplicity of the packed ``1o`` block. This
        is deliberately distinct from backbone ``vector_dim``; the default 8
        gives ``19x0e + 8x1o + 1x2e`` for ``C=48,lmax=2``.
    :param atom_classes: Positive configured atom vocabulary width ``A``.
    :param charge_classes: Positive configured charge vocabulary width ``Q``.
    :param bond_classes: Positive configured bond vocabulary width ``K``.
    :param max_atoms: Inclusive largest atom-count category.
    :param pocket_cutoff: Pocket radius cutoff in angstroms.
    :param cross_cutoff: Pocket-to-ligand radius cutoff in angstroms.
    :return: Device-agnostic PyTorch module for Tasks 11--13.
    :rtype: ECloudFlowModel
    :raises ValueError: If dimensions or the packed irrep layout are invalid.

    The module contains no rank-local global state and never calls ``cuda`` or
    transfers inputs, making it compatible with later Lightning DDP/FSDP
    wrapping. Proper SE(3) behavior is built from relative geometry, invariant
    contractions, and irrep-wise scalar gates; reflections are not required.
    """

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        num_blocks: int,
        lmax: int,
        *,
        electron_latent_dim: int = 48,
        electron_vector_dim: int = 8,
        atom_classes: int = 6,
        charge_classes: int = 4,
        bond_classes: int = 5,
        max_atoms: int = 64,
        pocket_cutoff: float = 8.0,
        cross_cutoff: float = 8.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("scalar_dim", scalar_dim),
            ("vector_dim", vector_dim),
            ("num_blocks", num_blocks),
            ("electron_latent_dim", electron_latent_dim),
            ("electron_vector_dim", electron_vector_dim),
            ("atom_classes", atom_classes),
            ("charge_classes", charge_classes),
            ("bond_classes", bond_classes),
            ("max_atoms", max_atoms),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if vector_dim < 3:
            raise ValueError(
                "vector_dim must be at least 3 for parity-sensitive chirality."
            )
        if not isinstance(lmax, int) or isinstance(lmax, bool) or not 0 <= lmax <= 4:
            raise ValueError("lmax must be an integer in [0, 4].")
        if pocket_cutoff <= 0 or cross_cutoff <= 0:
            raise ValueError("geometry cutoffs must be positive.")
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.num_blocks = num_blocks
        self.lmax = lmax
        self.electron_latent_dim = electron_latent_dim
        self.electron_vector_dim = electron_vector_dim
        self.max_atoms = max_atoms
        self.atom_classes = atom_classes
        self.charge_classes = charge_classes
        self.bond_classes = bond_classes
        self.electron_layout = _electron_layout(
            electron_latent_dim, electron_vector_dim, lmax
        )
        self.pocket_encoder = PocketEncoder(
            scalar_dim, vector_dim, num_blocks, cutoff=pocket_cutoff
        )
        self.backbone = JointLigandBackbone(
            scalar_dim, vector_dim, num_blocks, cutoff=cross_cutoff
        )
        self.position_velocity_head = nn.Linear(scalar_dim, vector_dim + 1)
        self.position_score_head = nn.Linear(scalar_dim, vector_dim + 1)
        self.electron_velocity_head = _PackedElectronHead(scalar_dim, vector_dim)
        self.electron_score_head = _PackedElectronHead(scalar_dim, vector_dim)
        self.atom_head = ScalarHead(scalar_dim, atom_classes)
        self.charge_head = ScalarHead(scalar_dim, charge_classes)
        self.bond_head = SymmetricPairHead(scalar_dim, bond_classes)
        self.count_predictor = AtomCountPredictor(scalar_dim, max_atoms=max_atoms)
        self.affinity_head = ScalarHead(scalar_dim)
        self.interaction_head = ScalarHead(scalar_dim)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        *,
        electron_latent_dim: int = 48,
        electron_vector_dim: int = 8,
        atom_classes: int = 6,
        charge_classes: int = 4,
        bond_classes: int = 5,
        max_atoms: int = 64,
        pocket_cutoff: float = 8.0,
        cross_cutoff: float = 8.0,
    ) -> ECloudFlowModel:
        """Construct from canonical widths plus explicit portable model defaults.

        :param config: Strict model name/width/block/order configuration. Its
            vector width must be at least three; canonical presets use at least four.
        :param electron_latent_dim: Overrideable packed electron width, default 48.
        :param electron_vector_dim: Overrideable ``1o`` multiplicity, default 8.
        :param atom_classes: Explicit atom endpoint class count, default 6.
        :param charge_classes: Explicit charge endpoint class count, default 4.
        :param bond_classes: Explicit bond endpoint class count, default 5.
        :param max_atoms: Inclusive count category maximum.
        :param pocket_cutoff: Pocket radius in angstroms.
        :param cross_cutoff: Pocket-ligand radius in angstroms.
        :return: Validated device-agnostic joint model.
        :rtype: ECloudFlowModel
        :raises ValueError: If any explicit default, vector width, or latent
            layout is invalid.

        No machine path, device, distributed rank, dtype, or global cache is
        inferred. Callers may override dataset vocabulary-independent values
        without altering :class:`ModelConfig` or checkpoint construction.
        """
        return cls(
            scalar_dim=config.scalar_dim,
            vector_dim=config.vector_dim,
            num_blocks=config.num_blocks,
            lmax=config.lmax,
            electron_latent_dim=electron_latent_dim,
            electron_vector_dim=electron_vector_dim,
            atom_classes=atom_classes,
            charge_classes=charge_classes,
            bond_classes=bond_classes,
            max_atoms=max_atoms,
            pocket_cutoff=pocket_cutoff,
            cross_cutoff=cross_cutoff,
        )

    def encode_pocket(
        self, pocket: PocketGraph, *, use_null: bool = False
    ) -> PocketEncoding:
        """Encode and key a pocket for explicit trajectory-level cache reuse.

        :param pocket: Canonical pocket positions ``[P,3]`` in centered-frame
            angstroms, invariant features ``[P,F]``, and batch indices ``[P]``.
        :param use_null: Produce real classifier-free null conditioning while
            retaining geometry and complex membership.
        :return: Reusable invariant scalars ``[P,S]``, equivariant vectors
            ``[P,V,3]``, coordinates, batch, actual feature layout, and key.
        :rtype: PocketEncoding
        :raises ValueError: If dtype/device, channels, geometry, or module
            placement is incompatible.

        Proper rotations rotate only vector outputs; translations cancel from
        learned features, and every tensor shape is explicit above. Evaluation
        is deterministic for fixed parameters and
        input. Tensors remain on the caller device/dtype with gradients intact.
        The model stores no mutable/global/rank-local cache: the caller owns the
        returned object. Its stable content key includes null mode and encoder
        layout, so another pocket, frame, device, or architecture is rejected.
        This method does not mutate the pocket and performs no distributed rank
        I/O. Autograd gradients through the encoded tensors remain intact.
        """
        return self.pocket_encoder.encode(pocket, use_null=use_null)

    def forward(
        self,
        state: MolecularState,
        time: torch.Tensor,
        condition: GenerationCondition,
        pocket_encoding: PocketEncoding | None = None,
    ) -> ModelPrediction:
        """Predict joint flow, score, categorical endpoints, and auxiliaries.

        :param state: Batched noisy flattened ligand with positions ``[N,3]``
            in centered-frame angstroms, atom/charge logits ``[N,A/Q]``, one
            unordered halfedge tensor ``[2,E]``, bond logits ``[E,K]``, packed
            electron irreps ``[N,C]``, and explicit node/halfedge batches.
        :param time: Floating per-complex path times ``[B]`` in ``[0,1]`` on
            the same dtype/device as state and pocket tensors.
        :param condition: Pocket, optional pocket field, named property values,
            invariant interaction targets, and optional exact fragment task
            condition. Property names use stable SHA-256 numeric identities in
            feature slots independent of their differentiable numeric values;
            joint name-value slots preserve each association before summation.
        :param pocket_encoding: Optional explicitly cached representation from
            :meth:`encode_pocket`; omitted values are computed for this call.
        :return: Coordinate/electron velocities and scores; invariant atom,
            charge, symmetric bond and count logits; affinity/interaction
            auxiliaries; and the validated pocket cache key.
        :rtype: ModelPrediction
        :raises TypeError: If state, time, condition, or cache has the wrong type.
        :raises ValueError: If time, batches, channels, packed irreps, devices,
            dtypes, fragment masks, or cached pocket identity are inconsistent.

        Coordinate outputs are scalar-weighted sums of all retained ligand
        vector channels ``[N,V,3]`` and packed ``1o`` contractions. Each block
        applies time FiLM scale/shift and normalized pocket cross-attention per
        ligand destination, so uneven neighbor counts do not rescale features.
        Translation cancels and proper rotations act exactly on Cartesian outputs.
        Cross products and a scalar triple product supply a chiral pseudoscalar:
        proper rotations preserve scalar predictions, while a reflection may
        change them as required by SE(3) rather than forced O(3) symmetry. Electron
        blocks use one learned scalar gate per irrep, preserving the planned
        ``19x0e + 8x1o + 1x2e`` transformation instead of treating orientation
        channels as scalars. Scalar logits/auxiliaries use invariant norms and
        contractions. Bond prediction consumes ``O(E)`` canonical pairs and is
        symmetric under endpoint exchange; no dense ``[N,N,C]`` allocation is
        made. Fragment fixed masks are applied before return and per-complex
        count probability is zero below fixed atom count. Pocket field invariant
        moments and translation-free polar moments, interaction targets, and
        property name/value pairs condition every block through explicit encoders.
        The null cache removes pocket geometry/features, pocket field, interaction
        target, property identity/value, and task signals for classifier-free
        guidance while retaining only batch cardinality. Empty ligands and
        multi-complex batches are supported.
        Calls are deterministic for fixed inputs/parameters, preserve finite
        autograd paths, never mutate inputs/cache, never move dtype/device, and
        hold no global or distributed rank state, making later Lightning
        DDP/FSDP wrapping safe. Incompatible cached frame/device/layout content
        fails before it can condition a different sampling trajectory.
        """
        _validate_public_inputs(self, state, time, condition, pocket_encoding)
        encoding = pocket_encoding or self.encode_pocket(condition.pocket)
        _validate_cache(self, condition.pocket, encoding)
        electron_summary = _electron_invariant_summary(
            state.electron_latent, self.electron_layout
        )
        batch_size = time.shape[0]
        task_features = _task_features(
            "de_novo" if condition.fragment is None else condition.fragment.task_id,
            batch_size,
            dtype=state.positions.dtype,
            device=state.positions.device,
        )
        property_features = _property_features(condition, batch_size, state.positions)
        interaction_features = _interaction_features(
            condition, batch_size, state.positions
        )
        field_features, field_vectors = _field_features(
            condition, batch_size, state.positions
        )
        hidden = self.backbone(
            state,
            time,
            encoding,
            electron_summary,
            task_features,
            property_features,
            interaction_features,
            field_features,
            field_vectors,
        )
        electron_vectors = _electron_vectors(
            state.electron_latent, self.electron_layout
        )
        electron_direction = (
            electron_vectors.mean(dim=1)
            if electron_vectors.shape[1]
            else state.positions.new_zeros((state.positions.shape[0], 3))
        )
        velocity_weights = self.position_velocity_head(hidden.scalars)
        score_weights = self.position_score_head(hidden.scalars)
        position_velocity = (
            torch.einsum(
                "nv,nvc->nc", velocity_weights[:, : self.vector_dim], hidden.vectors
            )
            + velocity_weights[:, -1:] * electron_direction
        )
        position_score = (
            torch.einsum(
                "nv,nvc->nc", score_weights[:, : self.vector_dim], hidden.vectors
            )
            + score_weights[:, -1:] * electron_direction
        )
        electron_velocity = self.electron_velocity_head(
            hidden.scalars,
            state.electron_latent,
            hidden.vectors,
            self.electron_layout,
        )
        electron_score = self.electron_score_head(
            hidden.scalars,
            state.electron_latent,
            hidden.vectors,
            self.electron_layout,
        )
        atom_logits = state.atom_logits + self.atom_head(hidden.scalars)
        charge_logits = state.charge_logits + self.charge_head(hidden.scalars)
        if state.halfedge_index.shape[1]:
            source, target = state.halfedge_index
            distance = (state.positions[source] - state.positions[target]).norm(dim=-1)
            bond_logits = state.bond_logits + self.bond_head(
                hidden.scalars[source], hidden.scalars[target], distance
            )
        else:
            bond_logits = state.bond_logits + hidden.scalars.sum() * 0.0
        fixed_counts = torch.zeros(
            batch_size, dtype=torch.long, device=state.positions.device
        )
        if condition.fragment is not None:
            fixed_counts.index_add_(
                0,
                state.node_batch,
                condition.fragment.fixed_atom_mask.to(dtype=torch.long),
            )
            fixed = condition.fragment.fixed_atom_mask
            fixed_bonds = condition.fragment.fixed_bond_mask
            position_velocity = position_velocity.masked_fill(fixed[:, None], 0.0)
            position_score = position_score.masked_fill(fixed[:, None], 0.0)
            atom_logits = torch.where(fixed[:, None], state.atom_logits, atom_logits)
            charge_logits = torch.where(
                fixed[:, None], state.charge_logits, charge_logits
            )
            bond_logits = torch.where(
                fixed_bonds[:, None], state.bond_logits, bond_logits
            )
        count_logits = self.count_predictor(hidden.pooled, fixed_counts).logits
        return ModelPrediction(
            position_velocity=position_velocity,
            position_score=position_score,
            electron_velocity=electron_velocity,
            electron_score=electron_score,
            atom_logits=atom_logits,
            charge_logits=charge_logits,
            bond_logits=bond_logits,
            count_logits=count_logits,
            affinity=self.affinity_head(hidden.pooled),
            interaction_logits=self.interaction_head(hidden.pooled),
            pocket_cache_key=encoding.cache_key,
        )


def _electron_layout(
    latent_dim: int, vector_dim: int, lmax: int
) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
    """Validate and return packed scalar/vector/higher-order block metadata."""
    higher = tuple((order, 1, 2 * order + 1) for order in range(2, lmax + 1))
    scalar_copies = latent_dim - 3 * vector_dim - sum(width for _, _, width in higher)
    if lmax == 0:
        scalar_copies = latent_dim
        vector_dim = 0
    elif scalar_copies <= 0:
        raise ValueError(
            "electron latent layout must retain at least one invariant scalar."
        )
    return scalar_copies, vector_dim, higher


def _electron_invariant_summary(
    latent: torch.Tensor,
    layout: tuple[int, int, tuple[tuple[int, int, int], ...]],
) -> torch.Tensor:
    """Reduce packed irreps only through invariant values and squared norms."""
    scalar_copies, vector_copies, higher = layout
    scalar = latent[:, :scalar_copies]
    invariants = [scalar.mean(dim=-1), safe_rms(scalar, -1)]
    offset = scalar_copies
    for multiplicity, width in ((vector_copies, 3), *((m, w) for _, m, w in higher)):
        block_width = multiplicity * width
        block = latent[:, offset : offset + block_width].reshape(
            -1, multiplicity, width
        )
        if multiplicity:
            block_norm = block.square().sum(dim=-1).mean(dim=-1)
            invariants[1] = (
                invariants[1]
                + block_norm.clamp_min(torch.finfo(block.dtype).eps).sqrt()
            )
        offset += block_width
    return torch.stack(invariants, dim=-1)


def _electron_vectors(
    latent: torch.Tensor,
    layout: tuple[int, int, tuple[tuple[int, int, int], ...]],
) -> torch.Tensor:
    """View the packed ``1o`` slice as Cartesian vector copies."""
    scalar_copies, vector_copies, _ = layout
    return latent[:, scalar_copies : scalar_copies + 3 * vector_copies].reshape(
        -1, vector_copies, 3
    )


def _task_features(
    task_id: str, batch_size: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Map a stable task identifier to invariant numeric conditioning."""
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()[:8]
    values = torch.tensor(list(digest), dtype=dtype, device=device) / 127.5 - 1.0
    return values.expand(batch_size, -1)


def _property_features(
    condition: GenerationCondition, batch_size: int, reference: torch.Tensor
) -> torch.Tensor:
    """Aggregate stable identity, value, and joint name-value features."""
    encoded = reference.new_zeros((batch_size, 17))
    if not condition.property_targets:
        return encoded
    for name, value in sorted(condition.property_targets.items()):
        if isinstance(value, torch.Tensor):
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(
                    "property target tensors must share the state dtype and device."
                )
            tensor = value
        else:
            tensor = reference.new_tensor(value)
        if tensor.ndim == 0:
            tensor = tensor.expand(batch_size)
        if tensor.shape != (batch_size,):
            raise ValueError(
                "property target tensors must be scalar or have shape [B]."
            )
        identity = _stable_identity(
            name, dtype=reference.dtype, device=reference.device
        )
        encoded[:, :8] = encoded[:, :8] + identity[None, :]
        encoded[:, 8] = encoded[:, 8] + tensor
        encoded[:, 9:] = encoded[:, 9:] + identity[None, :] * tensor[:, None]
    return encoded


def _interaction_features(
    condition: GenerationCondition, batch_size: int, reference: torch.Tensor
) -> torch.Tensor:
    """Encode arbitrary-width invariant interaction targets by stable moments."""
    target = condition.interaction_targets
    if target is None:
        return reference.new_zeros((batch_size, 2))
    if target.dtype != reference.dtype or target.device != reference.device:
        raise ValueError("interaction targets must share the state dtype and device.")
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2 or target.shape[0] != batch_size or target.shape[1] == 0:
        raise ValueError("interaction targets must have shape [B] or [B, I].")
    return torch.stack((target.mean(-1), safe_rms(target, -1)), dim=-1)


def _field_features(
    condition: GenerationCondition, batch_size: int, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode invariant field moments and one translation-free polar vector."""
    field = condition.pocket_field
    if field is None:
        return reference.new_zeros((batch_size, 4)), reference.new_zeros(
            (batch_size, 3)
        )
    if field.values.dtype != reference.dtype or field.values.device != reference.device:
        raise ValueError("pocket field must share the state dtype and device.")
    pocket = condition.pocket
    pocket_counts = (
        torch.bincount(pocket.batch, minlength=batch_size)
        .clamp_min(1)
        .to(reference.dtype)
    )
    centers = scatter_sum(pocket.positions, pocket.batch, batch_size)
    centers = centers / pocket_counts[:, None]
    valid = field.mask
    counts = torch.bincount(field.batch[valid], minlength=batch_size).to(
        reference.dtype
    )
    safe_counts = counts.clamp_min(1)
    point_mean = field.values.mean(-1)
    point_rms = safe_rms(field.values, -1)
    mean_sum = scatter_sum(point_mean[valid, None], field.batch[valid], batch_size)
    rms_sum = scatter_sum(point_rms[valid, None], field.batch[valid], batch_size)
    abs_sum = scatter_sum(point_mean[valid, None].abs(), field.batch[valid], batch_size)
    features = torch.cat(
        (
            mean_sum / safe_counts[:, None],
            rms_sum / safe_counts[:, None],
            abs_sum / safe_counts[:, None],
            (counts / pocket_counts)[:, None],
        ),
        dim=-1,
    )
    displacement = field.positions - centers[field.batch]
    moments = point_mean[:, None] * displacement
    vectors = scatter_sum(moments[valid], field.batch[valid], batch_size)
    vectors = vectors / safe_counts[:, None]
    return features, vectors


def _stable_identity(
    name: str, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Map a semantic condition name to a stable SHA-256 numeric identity."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()[:8]
    return torch.tensor(list(digest), dtype=dtype, device=device) / 127.5 - 1.0


def _validate_public_inputs(
    model: ECloudFlowModel,
    state: MolecularState,
    time: torch.Tensor,
    condition: GenerationCondition,
    cache: PocketEncoding | None,
) -> None:
    """Validate joint model boundary before any numerical conditioning."""
    if not isinstance(state, MolecularState) or not isinstance(
        condition, GenerationCondition
    ):
        raise TypeError("state and condition must use canonical contract types.")
    if not isinstance(time, torch.Tensor):
        raise TypeError("time must be a torch.Tensor.")
    pocket = condition.pocket
    if (
        pocket.positions.device != state.positions.device
        or pocket.positions.dtype != state.positions.dtype
    ):
        raise ValueError("state and pocket must share floating dtype and device.")
    if (
        time.ndim != 1
        or not time.is_floating_point()
        or time.dtype != state.positions.dtype
        or time.device != state.positions.device
        or not torch.isfinite(time).all()
        or bool(((time < 0) | (time > 1)).any())
    ):
        raise ValueError(
            "time must be finite floating [B] in [0, 1] on the state device and dtype."
        )
    batch_size = time.shape[0]
    expected_batches = torch.arange(batch_size, device=state.positions.device)
    if not torch.equal(torch.unique(pocket.batch, sorted=True), expected_batches):
        raise ValueError("pocket batch indices must be contiguous and match time [B].")
    if state.node_batch.numel() and not bool(
        torch.isin(torch.unique(state.node_batch), expected_batches).all()
    ):
        raise ValueError("node batch indices must belong to time [B].")
    parameter = next(model.parameters())
    if (
        parameter.device != state.positions.device
        or parameter.dtype != state.positions.dtype
    ):
        raise ValueError(
            "model parameters and state must share floating dtype and device."
        )
    if state.electron_latent.shape[1] != model.electron_latent_dim:
        raise ValueError(
            f"electron latent must have configured width {model.electron_latent_dim} and packed irrep layout."
        )
    if state.atom_logits.shape[1] != model.atom_classes:
        raise ValueError(
            f"atom class channels must equal configured width {model.atom_classes}."
        )
    if state.charge_logits.shape[1] != model.charge_classes:
        raise ValueError(
            f"charge class channels must equal configured width {model.charge_classes}."
        )
    if state.bond_logits.shape[1] != model.bond_classes:
        raise ValueError(
            f"bond class channels must equal configured width {model.bond_classes}."
        )
    if (
        condition.fragment is not None
        and condition.fragment.reference.positions.shape[0] != state.positions.shape[0]
    ):
        raise ValueError(
            "fragment reference and state must have the same flattened node count."
        )
    if cache is not None and not isinstance(cache, PocketEncoding):
        raise TypeError("pocket_encoding must be a PocketEncoding.")


def _validate_cache(
    model: ECloudFlowModel, pocket: PocketGraph, encoding: PocketEncoding
) -> None:
    """Reject cached encodings from another pocket, device, or architecture."""
    expected_key = model.pocket_encoder.cache_key(pocket, use_null=encoding.is_null)
    if (
        encoding.cache_key != expected_key
        or encoding.scalar_dim != model.scalar_dim
        or encoding.vector_dim != model.vector_dim
        or encoding.feature_dim != pocket.features.shape[1]
        or encoding.positions.device != pocket.positions.device
        or encoding.positions.dtype != pocket.positions.dtype
        or not torch.equal(encoding.batch, pocket.batch)
        or not torch.equal(encoding.positions, pocket.positions)
    ):
        raise ValueError(
            "cached pocket encoding is incompatible with this pocket, device, frame, or model layout."
        )
