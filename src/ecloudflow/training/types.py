"""Typed tensors crossing the ECloudFlow training boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingTargets:
    """Store exact path, endpoint, QM, chemistry, and auxiliary supervision."""

    position_velocity: torch.Tensor
    position_score: torch.Tensor
    electron_velocity: torch.Tensor
    electron_score: torch.Tensor
    atom_classes: torch.Tensor
    charge_classes: torch.Tensor
    bond_classes: torch.Tensor
    count_classes: torch.Tensor
    editable_atom_mask: torch.Tensor
    editable_bond_mask: torch.Tensor
    node_batch: torch.Tensor
    halfedge_index: torch.Tensor
    halfedge_batch: torch.Tensor
    count_mask: torch.Tensor
    qm_mask: torch.Tensor
    density: torch.Tensor | None = None
    density_gradient: torch.Tensor | None = None
    field_mask: torch.Tensor | None = None
    electron_count: torch.Tensor | None = None
    dipole: torch.Tensor | None = None
    latent_cycle: torch.Tensor | None = None
    valence_limits: torch.Tensor | None = None
    bond_order_values: torch.Tensor | None = None
    bond_length_mean: torch.Tensor | None = None
    bond_length_std: torch.Tensor | None = None
    nonbonded_halfedge_index: torch.Tensor | None = None
    protein_positions: torch.Tensor | None = None
    protein_batch: torch.Tensor | None = None
    ring_triplets: torch.Tensor | None = None
    ring_angle_mean: torch.Tensor | None = None
    ring_angle_std: torch.Tensor | None = None
    interaction: torch.Tensor | None = None
    interaction_mask: torch.Tensor | None = None
    affinity: torch.Tensor | None = None
    affinity_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class LossDiagnostics:
    """Typed subterm and mask diagnostics outside the six component mappings."""

    flow_fixed: torch.Tensor
    score_fixed: torch.Tensor
    affinity_log_variance_min: torch.Tensor
    affinity_log_variance_max: torch.Tensor
    subterms: dict[str, torch.Tensor]
    supervised_counts: dict[str, torch.Tensor]


@dataclass(frozen=True)
class LossBreakdown:
    """Hold exactly six raw, normalized, and weighted component losses."""

    total: torch.Tensor
    raw: dict[str, torch.Tensor]
    normalized: dict[str, torch.Tensor]
    weighted: dict[str, torch.Tensor]
    diagnostics: LossDiagnostics


@dataclass(frozen=True)
class ElectronDecoderContext:
    """Describe the real padded Task 8 decoder boundary for flattened tokens.

    :param centers: Padded atom centers ``[B,Nmax,3]`` in centered-frame angstroms.
    :param query_grid: Field query coordinates ``[B,G,3]`` in the same frames.
    :param atom_mask: Boolean physical-atom mask ``[B,Nmax]``.
    :param flat_index: Long mapping ``[B,Nmax]`` into flattened model nodes;
        padding may be ``-1`` only where ``atom_mask`` is false.
    :return: Immutable decoder context that fabricates no scientific values.
    :rtype: ElectronDecoderContext

    The training module gathers the model's differentiable first-order endpoint
    electron tokens through ``flat_index`` and calls the supplied real field
    decoder. All tensors retain caller dtype/device; validation occurs before
    decoding and no input is mutated or detached.
    """

    centers: torch.Tensor
    query_grid: torch.Tensor
    atom_mask: torch.Tensor
    flat_index: torch.Tensor


@dataclass(frozen=True)
class TrainingBatch:
    """Bundle model inputs and typed targets without changing graph layout."""

    state: object
    time: torch.Tensor
    condition: object
    targets: TrainingTargets
    decoder_context: ElectronDecoderContext | None = None
