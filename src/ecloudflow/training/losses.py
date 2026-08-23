"""Composite scientific losses and synchronized detached RMS normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional

from ecloudflow.config import LossConfig, WeightedLossConfig
from ecloudflow.models import ModelPrediction
from ecloudflow.training.types import LossBreakdown, LossDiagnostics, TrainingTargets

COMPONENT_NAMES = ("flow", "score", "discrete", "ecloud", "chem", "interaction")


class RunningLossScaler(nn.Module):  # type: ignore[misc]
    """Track synchronized detached running component mean squares.

    :param decay: EMA coefficient in ``[0,1)`` for previous mean-square state.
    :param epsilon: Positive finite lower bound inside the RMS square root.
    :return: Persistent six-component normalization module.
    :rtype: RunningLossScaler
    :raises ValueError: If decay or epsilon is outside its finite bound.

    ``update`` all-reduces detached squared rank-level component losses and
    presence counts whenever ``torch.distributed`` is initialized. Thus every
    rank applies identical state mutation; a component with no supervised rank
    is not decayed. This is a sufficient statistic for the RMS of rank component
    means, not an assertion about per-example variance. Buffers are float32 on
    the module device, checkpoint persistent, and never retain autograd. Dividing
    a whole component by one positive detached scalar preserves its internal and
    overall gradient direction. The class never selects a CUDA device manually.
    """

    component_names = COMPONENT_NAMES

    def __init__(self, decay: float = 0.99, epsilon: float = 1.0e-8) -> None:
        super().__init__()
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("running loss decay must be finite and in [0, 1).")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("running loss epsilon must be finite and positive.")
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.register_buffer(
            "mean_square", torch.ones(len(COMPONENT_NAMES)), persistent=True
        )
        self.register_buffer(
            "initialized",
            torch.zeros(len(COMPONENT_NAMES), dtype=torch.bool),
            persistent=True,
        )

    def update(
        self, values: Mapping[str, torch.Tensor], active: Mapping[str, bool]
    ) -> None:
        """Update detached statistics with optional distributed aggregation.

        :param values: Scalar finite component losses for the exact six names.
        :param active: Whether each local rank has genuine supervision for a name.
        :return: None after identical in-place persistent buffer updates per rank.
        :rtype: None
        :raises ValueError: If keys, scalar shapes, placement, or finiteness differ.

        Each rank contributes ``(loss.detach()**2, 1)`` only for active
        supervision. An initialized process group sums both tensors, giving all
        ranks the same sufficient statistics even for uneven local masks. Missing
        components contribute neither a zero observation nor decay. Updates use
        float32 under ``no_grad`` and have no autograd, optimizer, dtype-transfer,
        manual device, or random side effect. Persistent state resumes through
        normal checkpoint ``state_dict`` loading. Scalar shape validation precedes
        mutation, and deterministic updates leave missing-mask entries unchanged.
        """
        if set(values) != set(COMPONENT_NAMES) or set(active) != set(COMPONENT_NAMES):
            raise ValueError("running scaler requires exactly the six component names.")
        reference = next(iter(values.values()))
        if reference.device != self.mean_square.device:
            raise ValueError("running scaler buffers and losses must share one device.")
        _collective_finite_status("scaler", values, active)
        squares = torch.zeros_like(self.mean_square)
        counts = torch.zeros_like(self.mean_square)
        for index, name in enumerate(COMPONENT_NAMES):
            value = values[name]
            if value.ndim != 0 or value.device != reference.device:
                raise ValueError("running scaler values must be same-device scalars.")
            if active[name]:
                detached = value.detach().float()
                squares[index] = detached.square()
                counts[index] = 1.0
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(squares, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        with torch.no_grad():
            present = counts > 0
            observations = squares / counts.clamp_min(1.0)
            first = present & ~self.initialized
            continuing = present & self.initialized
            updated = self.mean_square * self.decay + observations * (1.0 - self.decay)
            self.mean_square.copy_(
                torch.where(
                    first,
                    observations,
                    torch.where(continuing, updated, self.mean_square),
                )
            )
            self.initialized[present] = True

    def normalize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Divide one component by its positive detached running RMS.

        :param name: One of the exact six stable component names.
        :param value: Scalar loss on the scaler buffer device.
        :return: Differentiable scalar divided by a detached positive RMS.
        :rtype: torch.Tensor
        :raises ValueError: If name, shape, or device is incompatible.

        The buffer is not mutated. Autograd sees multiplication by one positive
        constant, preserving gradient direction within the component. Float32
        RMS arithmetic is stable for mixed precision while the result follows
        PyTorch promotion rules on the caller device.
        """
        if name not in COMPONENT_NAMES or value.ndim != 0:
            raise ValueError("normalization requires a named scalar component.")
        if value.device != self.mean_square.device:
            raise ValueError("normalization value and scaler must share one device.")
        index = COMPONENT_NAMES.index(name)
        scale = torch.sqrt(self.mean_square[index].detach() + self.epsilon)
        return value / scale


def compute_ecloudflow_loss(
    prediction: ModelPrediction,
    targets: TrainingTargets,
    config: LossConfig,
    scaler: RunningLossScaler | None = None,
    *,
    step: int = 0,
) -> LossBreakdown:
    """Compute all normalized ECloudFlow training objectives.

    :param prediction: Joint outputs with flattened node tensors ``[N,*]``,
        sparse canonical bond rows ``[E,K]``, per-complex auxiliaries ``[B]``,
        first-order endpoint estimates, and optional real decoder reconstruction.
    :param targets: Exact flow/score targets, endpoint classes, availability
        masks, genuine-QM fields, geometry priors, and auxiliary labels. Position
        and length units are angstroms; density and gradient use electrons per
        cubic/fourth-power angstrom; dipoles use electron-angstroms.
    :param config: Frozen component/subterm weights, warm-ups, focal exponent,
        clash thresholds, variance clamps, and normalization controls.
    :param scaler: Optional persistent detached running RMS normalizer.
    :param step: Explicit non-negative optimizer-step context for deterministic
        linear component warm-ups; it is never inferred from rank-local state.
    :return: Total plus exactly six raw, normalized, and weighted mappings and
        typed diagnostics containing fixed-mask zeros and scientific subterms.
    :rtype: LossBreakdown
    :raises TypeError: If public typed contracts are not supplied.
    :raises ValueError: If shapes, dtypes, devices, classes, masks, sparse
        halfedges, or enabled-term prediction/context requirements are invalid.
    :raises FloatingPointError: If any active raw/normalized/weighted/total
        value is NaN or infinite, with component names in the diagnostic message.

    Fixed fragment atoms/halfedges are excluded from all generative reductions;
    empty/all-fixed/all-missing masks return finite differentiable zeros and do
    not update their running scales. ``qm_mask`` alone authorizes genuine QM
    density supervision, so approximate fields are never labeled QM. The
    endpoint geometry uses the model's documented ``x_t+(1-t)v`` first-order
    auxiliary: exact for straight deterministic paths, empirical for curved or
    stochastic paths, and never guaranteed to be a clean endpoint. Density
    terms require a differentiable real decoder reconstruction rather than a
    semantically unrelated tensor. Reductions accumulate in float32 for FP16/
    BF16, preserve caller device, and allocate only ``O(N+E+NP)`` tensors—never
    dense ``[N,N,C]`` bonds. Scaler mutation uses detached DDP-reduced sufficient
    statistics; input predictions/targets/config are not mutated. Warm-ups and
    losses are deterministic for fixed inputs and explicit step. Every tensor
    shape and dtype is checked in its coordinate frame, unordered halfedge
    storage stays sparse (no dense ``[N,N,C]``), and scaler checkpoint state
    resumes independently of the differentiable predictions.
    """
    if not isinstance(prediction, ModelPrediction):
        raise TypeError("prediction must be a ModelPrediction.")
    if not isinstance(targets, TrainingTargets):
        raise TypeError("targets must be TrainingTargets.")
    if not isinstance(config, LossConfig):
        raise TypeError("config must be LossConfig.")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative integer.")
    _validate_core_contract(prediction, targets)
    subterms: dict[str, torch.Tensor] = {}
    counts: dict[str, torch.Tensor] = {}

    editable_atoms = targets.editable_atom_mask
    editable_bonds = targets.editable_bond_mask
    flow_position = _masked_mse(
        prediction.position_velocity, targets.position_velocity, editable_atoms
    )
    flow_electron = _masked_mse(
        prediction.electron_velocity, targets.electron_velocity, editable_atoms
    )
    score_position = _masked_mse(
        prediction.position_score, targets.position_score, editable_atoms
    )
    score_electron = _masked_mse(
        prediction.electron_score, targets.electron_score, editable_atoms
    )
    subterms.update(
        flow_position=flow_position,
        flow_electron=flow_electron,
        score_position=score_position,
        score_electron=score_electron,
    )
    atom_count = editable_atoms.sum()
    bond_count = editable_bonds.sum()
    counts.update(
        flow_position=atom_count if config.flow.position else atom_count * 0,
        flow_electron=atom_count if config.flow.electron else atom_count * 0,
        score_position=atom_count if config.score.position else atom_count * 0,
        score_electron=atom_count if config.score.electron else atom_count * 0,
    )
    flow = config.flow.position * flow_position + config.flow.electron * flow_electron
    score = (
        config.score.position * score_position + config.score.electron * score_electron
    )

    atom = _masked_ce(prediction.atom_logits, targets.atom_classes, editable_atoms)
    charge = _masked_ce(
        prediction.charge_logits, targets.charge_classes, editable_atoms
    )
    bond = _masked_ce(prediction.bond_logits, targets.bond_classes, editable_bonds)
    count = _masked_ce(
        prediction.count_logits, targets.count_classes, targets.count_mask
    )
    subterms.update(
        discrete_atom=atom,
        discrete_charge=charge,
        discrete_bond=bond,
        discrete_count=count,
    )
    counts.update(
        discrete_atom=atom_count if config.discrete.atom else atom_count * 0,
        discrete_charge=atom_count if config.discrete.charge else atom_count * 0,
        discrete_bond=bond_count if config.discrete.bond else bond_count * 0,
        discrete_count=(
            targets.count_mask.sum()
            if config.discrete.count
            else targets.count_mask.sum() * 0
        ),
    )
    discrete = (
        config.discrete.atom * atom
        + config.discrete.charge * charge
        + config.discrete.bond * bond
        + config.discrete.count * count
    )

    ecloud_terms = _ecloud_terms(
        prediction,
        targets,
        config,
        subterms,
        counts,
        enabled=config.ecloud.weight > 0.0 and _warmup(config.ecloud, step) > 0.0,
    )
    ecloud = sum(ecloud_terms, _zero_from_prediction(prediction))

    chemistry_terms = _chemistry_terms(prediction, targets, config, subterms, counts)
    chem = sum(chemistry_terms, _zero_from_prediction(prediction))

    interaction, _ = _interaction_term(prediction, targets, config)
    subterms["interaction_focal"] = interaction
    counts["interaction"] = _mask_count(targets.interaction_mask, prediction.affinity)

    raw = {
        "flow": flow,
        "score": score,
        "discrete": discrete,
        "ecloud": ecloud,
        "chem": chem,
        "interaction": interaction,
    }
    component_configs: dict[str, WeightedLossConfig] = {
        "flow": config.flow,
        "score": config.score,
        "discrete": config.discrete,
        "ecloud": config.ecloud,
        "chem": config.chem,
        "interaction": config.interaction,
    }
    observed = {
        "flow": counts["flow_position"] + counts["flow_electron"],
        "score": counts["score_position"] + counts["score_electron"],
        "discrete": sum(
            (counts[name] for name in counts if name.startswith("discrete_")),
            atom_count * 0,
        ),
        "ecloud": sum(
            (counts[name] for name in counts if name.startswith("ecloud_")),
            atom_count * 0,
        ),
        "chem": sum(
            (counts[name] for name in counts if name.startswith("chem_")),
            atom_count * 0,
        ),
        "interaction": counts["interaction"],
    }
    active = {
        name: bool(observed[name] > 0)
        and component_configs[name].weight > 0.0
        and _warmup(component_configs[name], step) > 0.0
        for name in COMPONENT_NAMES
    }
    _collective_finite_status("raw", raw, active)
    if scaler is not None and config.normalization.enabled:
        scaler.update(raw, active)
        normalized = {
            name: scaler.normalize(name, raw[name]) for name in COMPONENT_NAMES
        }
    else:
        normalized = dict(raw)
    weighted: dict[str, torch.Tensor] = {}
    for name in COMPONENT_NAMES:
        factor = component_configs[name].weight * _warmup(component_configs[name], step)
        weighted[name] = (
            normalized[name] * factor
            if factor > 0.0
            else _zero_from_prediction(prediction)
        )
    total = sum(weighted.values(), _zero_from_prediction(prediction))
    _collective_finite_status("normalized", normalized, active)
    _collective_finite_status("weighted", weighted, active)
    total_values = {name: total for name in COMPONENT_NAMES}
    total_active = {name: name == "flow" for name in COMPONENT_NAMES}
    _collective_finite_status("total", total_values, total_active)
    fixed = ~editable_atoms
    flow_fixed = prediction.position_velocity[fixed].reshape(-1)[:0].sum()
    score_fixed = prediction.position_score[fixed].reshape(-1)[:0].sum()
    affinity_mask = targets.affinity_mask
    selected_log_variance = (
        prediction.affinity_log_variance[affinity_mask]
        if affinity_mask is not None
        else prediction.affinity_log_variance[:0]
    )
    log_variance = selected_log_variance.clamp(
        config.chem.affinity_log_variance_min,
        config.chem.affinity_log_variance_max,
    )
    zero = prediction.affinity.reshape(-1)[:0].sum()
    diagnostics = LossDiagnostics(
        flow_fixed=flow_fixed,
        score_fixed=score_fixed,
        affinity_log_variance_min=log_variance.min() if log_variance.numel() else zero,
        affinity_log_variance_max=log_variance.max() if log_variance.numel() else zero,
        subterms=subterms,
        supervised_counts=counts,
    )
    return LossBreakdown(total, raw, normalized, weighted, diagnostics)


def _validate_core_contract(
    prediction: ModelPrediction, targets: TrainingTargets
) -> None:
    """Validate flattened shapes, devices, masks, classes, and sparse topology."""
    reference = prediction.position_velocity
    node_count = reference.shape[0]
    edge_count = prediction.bond_logits.shape[0]
    batch_size = prediction.affinity.shape[0]
    pairs = (
        (
            targets.position_velocity,
            prediction.position_velocity.shape,
            "position_velocity",
        ),
        (targets.position_score, prediction.position_score.shape, "position_score"),
        (
            targets.electron_velocity,
            prediction.electron_velocity.shape,
            "electron_velocity",
        ),
        (targets.electron_score, prediction.electron_score.shape, "electron_score"),
    )
    for tensor, shape, name in pairs:
        if (
            tensor.shape != shape
            or tensor.device != reference.device
            or not tensor.is_floating_point()
        ):
            raise ValueError(
                f"{name} target must match prediction shape/device and be floating."
            )
    masks = (
        (targets.editable_atom_mask, (node_count,), "editable_atom_mask"),
        (targets.editable_bond_mask, (edge_count,), "editable_bond_mask"),
        (targets.count_mask, (batch_size,), "count_mask"),
        (targets.qm_mask, (batch_size,), "qm_mask"),
        (targets.interaction_mask, (batch_size,), "interaction_mask"),
        (targets.affinity_mask, (batch_size,), "affinity_mask"),
    )
    for tensor, shape, name in masks:
        if tensor is None and name in {"interaction_mask", "affinity_mask"}:
            continue
        assert tensor is not None
        if (
            tensor.shape != shape
            or tensor.dtype != torch.bool
            or tensor.device != reference.device
        ):
            raise ValueError(
                f"{name} must be a boolean tensor with shape {shape} on prediction device."
            )
    classes = (
        (targets.atom_classes, node_count, "atom"),
        (
            targets.charge_classes,
            node_count,
            "charge",
        ),
        (targets.bond_classes, edge_count, "bond"),
        (targets.count_classes, batch_size, "count"),
    )
    for tensor, size, name in classes:
        if (
            tensor.shape != (size,)
            or tensor.dtype != torch.long
            or tensor.device != reference.device
        ):
            raise ValueError(
                f"{name} class targets must be torch.long with the expected shape/device."
            )
    if (
        targets.node_batch.shape != (node_count,)
        or targets.node_batch.dtype != torch.long
        or targets.halfedge_batch.shape != (edge_count,)
        or targets.halfedge_batch.dtype != torch.long
        or targets.halfedge_index.shape != (2, edge_count)
        or targets.halfedge_index.dtype != torch.long
        or targets.node_batch.device != reference.device
        or targets.halfedge_batch.device != reference.device
        or targets.halfedge_index.device != reference.device
    ):
        raise ValueError(
            "node and halfedge batch/index tensors have invalid shapes or dtypes."
        )
    if targets.node_batch.numel() and (
        bool((targets.node_batch < 0).any())
        or bool((targets.node_batch >= batch_size).any())
    ):
        raise ValueError("node_batch contains an index outside the batch range.")
    if targets.halfedge_index.numel():
        source, target = targets.halfedge_index
        if bool((source >= target).any()):
            raise ValueError(
                "bond topology must contain one canonical unordered halfedge per pair."
            )
        if bool((source < 0).any()) or bool((target >= node_count).any()):
            raise ValueError("unordered halfedge endpoint is outside the node range.")
        if not torch.equal(targets.node_batch[source], targets.node_batch[target]):
            raise ValueError("unordered halfedge endpoints must belong to one complex.")
        if not torch.equal(targets.node_batch[source], targets.halfedge_batch):
            raise ValueError("halfedge_batch must match endpoint complex membership.")
        codes = source * node_count + target
        if codes.unique().numel() != codes.numel():
            raise ValueError("canonical unordered halfedges must be unique.")


def _ecloud_terms(
    prediction: ModelPrediction,
    targets: TrainingTargets,
    config: LossConfig,
    diagnostics: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
    *,
    enabled: bool,
) -> list[torch.Tensor]:
    """Compute genuine-QM reconstruction terms or an inactive differentiable zero."""
    zero = _zero_from_prediction(prediction)
    weights = {
        "density": config.ecloud.density,
        "gradient": config.ecloud.gradient,
        "electron_count": config.ecloud.electron_count,
        "dipole": config.ecloud.dipole,
        "cycle": config.ecloud.cycle,
    }
    if not enabled or not bool(targets.qm_mask.any()) or not any(weights.values()):
        for name in ("density", "gradient", "electron_count", "dipole", "cycle"):
            diagnostics[f"ecloud_{name}"] = zero
            counts[f"ecloud_{name}"] = targets.qm_mask.sum() * 0
        return [zero]
    reconstruction = prediction.electron_reconstruction
    if reconstruction is None:
        raise ValueError(
            "enabled QM terms require a real electron reconstruction prediction."
        )
    values: dict[str, torch.Tensor] = {}
    field_mask = targets.field_mask
    if weights["density"]:
        if targets.density is None or field_mask is None:
            raise ValueError(
                "enabled QM density requires density and field_mask targets."
            )
        values["density"] = _masked_mse(
            reconstruction.density,
            targets.density,
            targets.qm_mask[:, None] & field_mask,
        )
    if weights["gradient"]:
        if targets.density_gradient is None or field_mask is None:
            raise ValueError(
                "enabled QM density gradient requires gradient and field_mask targets."
            )
        values["gradient"] = _masked_mse(
            reconstruction.gradient,
            targets.density_gradient,
            targets.qm_mask[:, None] & field_mask,
        )
    if weights["electron_count"]:
        if targets.electron_count is None:
            raise ValueError("enabled QM electron count requires count targets.")
        values["electron_count"] = _masked_mse(
            reconstruction.electron_count, targets.electron_count, targets.qm_mask
        )
    if weights["dipole"]:
        if targets.dipole is None:
            raise ValueError("enabled QM dipole requires dipole targets.")
        values["dipole"] = _masked_mse(
            reconstruction.dipole, targets.dipole, targets.qm_mask
        )
    if weights["cycle"]:
        if targets.latent_cycle is None:
            raise ValueError("enabled electron cycle requires latent cycle targets.")
        cycle_mask = (
            targets.latent_cycle_mask
            if targets.latent_cycle_mask is not None
            else torch.ones(
                reconstruction.latent_round_trip.shape[:2],
                dtype=torch.bool,
                device=reconstruction.latent_round_trip.device,
            )
        )
        values["cycle"] = _masked_mse(
            reconstruction.latent_round_trip,
            targets.latent_cycle,
            targets.qm_mask[:, None] & cycle_mask,
        )
    for name in weights.keys() - values.keys():
        values[name] = zero
    for name, value in values.items():
        diagnostics[f"ecloud_{name}"] = value
        if name in {"density", "gradient"} and field_mask is not None:
            observed = (targets.qm_mask[:, None] & field_mask).sum()
        elif name == "cycle" and targets.latent_cycle_mask is not None:
            observed = (targets.qm_mask[:, None] & targets.latent_cycle_mask).sum()
        else:
            observed = targets.qm_mask.sum()
        counts[f"ecloud_{name}"] = observed if weights[name] else observed * 0
    return [weights[name] * value for name, value in values.items()]


def _chemistry_terms(
    prediction: ModelPrediction,
    targets: TrainingTargets,
    config: LossConfig,
    diagnostics: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    """Compute endpoint geometry, sparse graph, clash, and affinity surrogates."""
    editable = targets.editable_atom_mask
    zero = _zero_from_prediction(prediction)
    values: dict[str, torch.Tensor] = {
        name: zero
        for name in (
            "valence",
            "bond_length",
            "ligand_clash",
            "protein_clash",
            "ring_strain",
            "connectivity",
        )
    }
    weights = {
        "valence": config.chem.valence,
        "bond_length": config.chem.bond_length,
        "ligand_clash": config.chem.ligand_clash,
        "protein_clash": config.chem.protein_clash,
        "ring_strain": config.chem.ring_strain,
        "connectivity": config.chem.connectivity,
        "affinity": config.chem.affinity,
    }
    if bool(editable.any()):
        source, target = targets.halfedge_index
        active_edges = targets.editable_bond_mask
        active_source = source[active_edges]
        active_target = target[active_edges]
        selected_bond_logits = prediction.bond_logits[active_edges].float()
        bond_probability = selected_bond_logits.softmax(-1)
        needs_order = weights["valence"] or weights["connectivity"]
        if needs_order:
            if targets.bond_order_values is None:
                raise ValueError("enabled valence/connectivity requires bond orders.")
            if targets.bond_order_values.shape != (prediction.bond_logits.shape[1],):
                raise ValueError(
                    "bond_order_values must have one value per bond class."
                )
            if (
                not targets.bond_order_values.is_floating_point()
                or targets.bond_order_values.device != prediction.bond_logits.device
                or not bool(torch.isfinite(targets.bond_order_values).all())
            ):
                raise ValueError(
                    "bond_order_values must be finite floating model context."
                )
            expected_order = (
                (bond_probability * targets.bond_order_values.float().unsqueeze(0))
                .sum(dim=-1)
                .float()
            )
            valence = torch.zeros(
                prediction.position_velocity.shape[0],
                dtype=expected_order.dtype,
                device=expected_order.device,
            )
            valence.index_add_(0, active_source, expected_order)
            valence.index_add_(0, active_target, expected_order)
            if weights["valence"]:
                if targets.valence_limits is None:
                    raise ValueError(
                        "enabled valence overflow requires valence limits."
                    )
                if (
                    targets.valence_limits.shape != (editable.shape[0],)
                    or not targets.valence_limits.is_floating_point()
                    or targets.valence_limits.device != editable.device
                ):
                    raise ValueError(
                        "valence_limits must be floating [N] on model device."
                    )
                selected_limits = targets.valence_limits[editable]
                if not bool(torch.isfinite(selected_limits).all()):
                    raise ValueError("active valence limits must be finite.")
                values["valence"] = (
                    functional.relu(valence[editable] - selected_limits.float())
                    .square()
                    .mean()
                )
            if weights["connectivity"]:
                present_probability = 1.0 - bond_probability[:, 0]
                degree = torch.zeros_like(valence)
                degree.index_add_(0, active_source, present_probability)
                degree.index_add_(0, active_target, present_probability)
                values["connectivity"] = _masked_mean(
                    functional.relu(config.chem.minimum_degree - degree).square(),
                    editable,
                )
        geometry_enabled = any(
            weights[name]
            for name in ("bond_length", "ligand_clash", "protein_clash", "ring_strain")
        )
        if geometry_enabled and prediction.endpoint_positions is None:
            raise ValueError("enabled chemistry geometry requires endpoint prediction.")
        if weights["bond_length"]:
            if targets.bond_length_mean is None or targets.bond_length_std is None:
                raise ValueError("enabled bond length requires conditioned mean/std.")
            selected_mean = targets.bond_length_mean[active_edges]
            selected_std = targets.bond_length_std[active_edges]
            if (
                not selected_mean.is_floating_point()
                or not selected_std.is_floating_point()
            ):
                raise ValueError("bond length mean/std must be floating tensors.")
            if (
                not bool(torch.isfinite(selected_mean).all())
                or not bool(torch.isfinite(selected_std).all())
                or bool((selected_std <= 0).any())
            ):
                raise ValueError("active bond length std must be finite and positive.")
            distances = (
                (
                    prediction.endpoint_positions[active_source]
                    - prediction.endpoint_positions[active_target]
                )
                .float()
                .norm(dim=-1)
            )
            standardized = (distances - selected_mean.float()) / selected_std.float()
            values["bond_length"] = (
                standardized.square().mean() if standardized.numel() else zero
            )
        if weights["ligand_clash"]:
            if targets.nonbonded_halfedge_index is None:
                raise ValueError(
                    "enabled ligand clash requires sparse nonbonded pairs."
                )
            nonbonded = targets.nonbonded_halfedge_index
            _validate_sparse_pairs(
                nonbonded, editable.shape[0], "nonbonded_halfedge_index"
            )
            _validate_nonbonded_topology(nonbonded, targets)
            left, right = nonbonded
            pair_editable = editable[left] | editable[right]
            selected_left = left[pair_editable]
            selected_right = right[pair_editable]
            nonbonded_distance = (
                (
                    prediction.endpoint_positions[selected_left]
                    - prediction.endpoint_positions[selected_right]
                )
                .float()
                .norm(dim=-1)
            )
            values["ligand_clash"] = (
                functional.relu(config.chem.ligand_clash_distance - nonbonded_distance)
                .square()
                .mean()
                if nonbonded_distance.numel()
                else zero
            )
        if weights["protein_clash"]:
            if targets.protein_positions is None or targets.protein_batch is None:
                raise ValueError(
                    "enabled protein clash requires protein positions/batch."
                )
            values["protein_clash"] = _protein_clash(prediction, targets, config)
        if weights["ring_strain"]:
            if any(
                value is None
                for value in (
                    targets.ring_triplets,
                    targets.ring_angle_mean,
                    targets.ring_angle_std,
                )
            ):
                raise ValueError(
                    "enabled ring strain requires sparse ring angle priors."
                )
            values["ring_strain"] = _ring_strain(prediction, targets, config)
    affinity_mask = targets.affinity_mask
    if weights["affinity"] and _optional_mask_any(affinity_mask):
        if targets.affinity is None or affinity_mask is None:
            raise ValueError(
                "affinity availability requires per-example labels and mask."
            )
        selected_affinity = prediction.affinity[affinity_mask].float()
        selected_target = targets.affinity[affinity_mask].float()
        clamped = (
            prediction.affinity_log_variance[affinity_mask]
            .float()
            .clamp(
                config.chem.affinity_log_variance_min,
                config.chem.affinity_log_variance_max,
            )
        )
        residual = selected_affinity - selected_target
        nll = 0.5 * (torch.exp(-clamped) * residual.square() + clamped)
        values["affinity"] = nll.mean()
    else:
        values["affinity"] = zero
    for name, value in values.items():
        diagnostics[f"chem_{name}"] = value
        if name in {"valence", "connectivity"}:
            observed = editable.sum()
        elif name == "bond_length":
            observed = targets.editable_bond_mask.sum()
        elif name == "ligand_clash" and targets.nonbonded_halfedge_index is not None:
            left, right = targets.nonbonded_halfedge_index
            observed = (editable[left] | editable[right]).sum()
        elif name == "protein_clash" and targets.protein_batch is not None:
            observed = sum(
                int(
                    ((targets.node_batch == batch) & editable).sum()
                    * (targets.protein_batch == batch).sum()
                )
                for batch in torch.unique(targets.node_batch)
            )
            observed = torch.tensor(observed, device=editable.device)
        elif name == "ring_strain" and targets.ring_triplets is not None:
            left, center, right = targets.ring_triplets
            observed = (editable[left] | editable[center] | editable[right]).sum()
        elif name == "affinity" and affinity_mask is not None:
            observed = affinity_mask.sum()
        else:
            observed = editable.sum() * 0
        counts[f"chem_{name}"] = observed if weights[name] else observed * 0
    return [weights[name] * value for name, value in values.items()]


def _protein_clash(
    prediction: ModelPrediction, targets: TrainingTargets, config: LossConfig
) -> torch.Tensor:
    """Compute per-complex ligand-protein overlap without dense ligand bonds."""
    assert targets.protein_positions is not None
    assert targets.protein_batch is not None
    if targets.protein_positions.ndim != 2 or targets.protein_positions.shape[1] != 3:
        raise ValueError("protein_positions must have shape [P, 3].")
    protein_count = targets.protein_positions.shape[0]
    if (
        not targets.protein_positions.is_floating_point()
        or targets.protein_positions.device != prediction.endpoint_positions.device
        or targets.protein_batch.shape != (protein_count,)
        or targets.protein_batch.dtype != torch.long
        or targets.protein_batch.device != prediction.endpoint_positions.device
    ):
        raise ValueError(
            "protein positions/batch must be typed [P,3]/[P] on model device."
        )
    batch_size = prediction.affinity.shape[0]
    if targets.protein_batch.numel() and (
        bool((targets.protein_batch < 0).any())
        or bool((targets.protein_batch >= batch_size).any())
    ):
        raise ValueError("protein_batch contains an index outside the batch range.")
    losses: list[torch.Tensor] = []
    for complex_index in torch.unique(targets.node_batch):
        ligand_mask = (targets.node_batch == complex_index) & targets.editable_atom_mask
        protein_mask = targets.protein_batch == complex_index
        if bool(ligand_mask.any()) and bool(protein_mask.any()):
            selected_protein = targets.protein_positions[protein_mask]
            if not bool(torch.isfinite(selected_protein).all()):
                raise ValueError("active protein clash positions must be finite.")
            distances = torch.cdist(
                prediction.endpoint_positions[ligand_mask].float(),
                selected_protein.float(),
            )
            losses.append(
                functional.relu(config.chem.protein_clash_distance - distances)
                .square()
                .mean()
            )
    return torch.stack(losses).mean() if losses else _zero_from_prediction(prediction)


def _ring_strain(
    prediction: ModelPrediction, targets: TrainingTargets, config: LossConfig
) -> torch.Tensor:
    """Return standardized sparse ring-angle strain with stable collinear geometry."""
    del config
    assert targets.ring_triplets is not None
    assert targets.ring_angle_mean is not None
    assert targets.ring_angle_std is not None
    triplets = targets.ring_triplets
    if triplets.shape[0] != 3:
        raise ValueError("ring_triplets must have shape [3, R].")
    if triplets.shape[1] == 0:
        return _zero_from_prediction(prediction)
    _validate_sparse_indices(
        triplets, prediction.position_velocity.shape[0], "ring_triplets"
    )
    left, center, right = triplets
    if bool(((left == center) | (left == right) | (center == right)).any()):
        raise ValueError("ring triplets must contain three distinct nodes.")
    if not (
        torch.equal(targets.node_batch[left], targets.node_batch[center])
        and torch.equal(targets.node_batch[left], targets.node_batch[right])
    ):
        raise ValueError("ring triplets must stay within one complex.")
    ring_editable = (
        targets.editable_atom_mask[left]
        | targets.editable_atom_mask[center]
        | targets.editable_atom_mask[right]
    )
    expected_shape = (triplets.shape[1],)
    for prior, name in (
        (targets.ring_angle_mean, "ring_angle_mean"),
        (targets.ring_angle_std, "ring_angle_std"),
    ):
        if (
            prior.shape != expected_shape
            or not prior.is_floating_point()
            or prior.device != triplets.device
        ):
            raise ValueError(f"{name} must be floating [R] on the topology device.")
    selected_mean = targets.ring_angle_mean[ring_editable]
    selected_std = targets.ring_angle_std[ring_editable]
    if (
        not bool(torch.isfinite(selected_mean).all())
        or not bool(torch.isfinite(selected_std).all())
        or bool((selected_std <= 0).any())
    ):
        raise ValueError("active ring angle std must be finite and positive.")
    selected_left = left[ring_editable]
    selected_center = center[ring_editable]
    selected_right = right[ring_editable]
    first = (
        prediction.endpoint_positions[selected_left].float()
        - prediction.endpoint_positions[selected_center].float()
    )
    second = (
        prediction.endpoint_positions[selected_right].float()
        - prediction.endpoint_positions[selected_center].float()
    )
    cosine = functional.cosine_similarity(first, second, dim=-1, eps=1.0e-8).clamp(
        -1.0 + 1.0e-6, 1.0 - 1.0e-6
    )
    angles = torch.acos(cosine)
    standardized = (angles - selected_mean.float()) / selected_std.float()
    return (
        standardized.square().mean()
        if standardized.numel()
        else _zero_from_prediction(prediction)
    )


def _interaction_term(
    prediction: ModelPrediction, targets: TrainingTargets, config: LossConfig
) -> tuple[torch.Tensor, bool]:
    """Compute masked binary focal loss for per-complex interaction labels."""
    if not _optional_mask_any(targets.interaction_mask):
        return _zero_from_prediction(prediction), False
    if targets.interaction is None or targets.interaction_mask is None:
        raise ValueError("interaction availability requires labels and mask.")
    if targets.interaction.shape != prediction.interaction_logits.shape:
        raise ValueError("interaction labels must have shape [B].")
    labels = targets.interaction[targets.interaction_mask].float()
    if bool(((labels < 0.0) | (labels > 1.0)).any()):
        raise ValueError("interaction labels must be probabilities in [0, 1].")
    logits = prediction.interaction_logits[targets.interaction_mask].float()
    bce = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probability = torch.sigmoid(logits)
    correct_probability = labels * probability + (1.0 - labels) * (1.0 - probability)
    focal = (1.0 - correct_probability).pow(config.interaction.focal_gamma) * bce
    return focal.mean(), True


def _masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return float32-safe differentiable masked scalar-entry mean square error."""
    if (
        prediction.shape != target.shape
        or prediction.device != target.device
        or not prediction.is_floating_point()
        or not target.is_floating_point()
    ):
        raise ValueError(
            "masked MSE prediction and target must have identical shape/device and floating dtype."
        )
    expanded = _expanded_mask(mask, prediction)
    selected_prediction = prediction[expanded]
    selected_target = target[expanded]
    compute_prediction = (
        selected_prediction.float()
        if prediction.dtype in (torch.float16, torch.bfloat16)
        else selected_prediction
    )
    compute_target = (
        selected_target.float()
        if target.dtype in (torch.float16, torch.bfloat16)
        else selected_target
    )
    if compute_prediction.numel() == 0:
        return compute_prediction.sum()
    return (compute_prediction - compute_target).square().mean()


def _masked_ce(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return selected-row cross entropy or a differentiable empty zero."""
    selected_logits = logits[mask]
    selected_target = target[mask]
    compute_logits = (
        selected_logits.float()
        if logits.dtype in (torch.float16, torch.bfloat16)
        else selected_logits
    )
    if compute_logits.shape[0] == 0:
        return compute_logits.sum()
    if bool((selected_target < 0).any()) or bool(
        (selected_target >= logits.shape[1]).any()
    ):
        raise ValueError("class target is outside its configured vocabulary.")
    return functional.cross_entropy(compute_logits, selected_target)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average selected scalar entries after broadcasting a boolean prefix mask."""
    expanded = _expanded_mask(mask, values)
    selected = values[expanded]
    if selected.numel() == 0:
        return selected.sum()
    return selected.mean()


def _expanded_mask(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Expand a same-device boolean prefix mask without touching excluded values."""
    if (
        mask.dtype != torch.bool
        or mask.device != values.device
        or mask.ndim > values.ndim
    ):
        raise ValueError("loss mask must be a same-device boolean prefix tensor.")
    try:
        return torch.broadcast_to(
            mask.reshape(*mask.shape, *([1] * (values.ndim - mask.ndim))), values.shape
        )
    except RuntimeError as error:
        raise ValueError("loss mask does not broadcast to the value shape.") from error


def _zero_from_prediction(prediction: ModelPrediction) -> torch.Tensor:
    """Create a finite differentiable scalar zero on the prediction device."""
    return prediction.position_velocity.reshape(-1)[:0].sum()


def _mask_count(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    """Return an integer supervised count or a same-device zero."""
    return (
        mask.sum()
        if mask is not None
        else torch.zeros((), dtype=torch.long, device=reference.device)
    )


def _optional_mask_any(mask: torch.Tensor | None) -> bool:
    """Return whether an optional availability mask selects any example."""
    return mask is not None and bool(mask.any())


def _warmup(config: WeightedLossConfig, step: int) -> float:
    """Evaluate deterministic linear warm-up from explicit optimizer step."""
    if config.warmup_end == config.warmup_start:
        return float(step >= config.warmup_end)
    if step <= config.warmup_start:
        return 0.0
    if step >= config.warmup_end:
        return 1.0
    return (step - config.warmup_start) / (config.warmup_end - config.warmup_start)


def _collective_finite_status(
    stage: str, values: Mapping[str, torch.Tensor], active: Mapping[str, bool]
) -> dict[str, bool]:
    """Synchronize presence/nonfinite flags before any rank may raise.

    Every initialized rank performs exactly one fixed-shape reduction for this
    stage.  This prevents a locally invalid component from stranding peers in a
    later scaler collective.  Only detached boolean sufficient statistics cross
    ranks; differentiable losses never do.
    """
    reference = next(iter(values.values()))
    status = torch.zeros((len(COMPONENT_NAMES), 2), device=reference.device)
    for index, name in enumerate(COMPONENT_NAMES):
        present = bool(active[name])
        status[index, 0] = float(present)
        status[index, 1] = float(present and not bool(torch.isfinite(values[name])))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(status, op=dist.ReduceOp.SUM)
    bad = [name for index, name in enumerate(COMPONENT_NAMES) if status[index, 1] > 0]
    if bad:
        raise FloatingPointError(f"non-finite {stage} components: {', '.join(bad)}")
    return {
        name: bool(status[index, 0] > 0) for index, name in enumerate(COMPONENT_NAMES)
    }


def _validate_sparse_pairs(index: torch.Tensor, size: int, name: str) -> None:
    """Validate canonical unordered sparse pair indices."""
    if index.shape[0] != 2 or index.dtype != torch.long:
        raise ValueError(f"{name} must have shape [2, E] and torch.long dtype.")
    _validate_sparse_indices(index, size, name)
    if index.numel() and bool((index[0] >= index[1]).any()):
        raise ValueError(f"{name} must contain canonical unordered halfedges.")


def _validate_sparse_indices(index: torch.Tensor, size: int, name: str) -> None:
    """Validate a sparse integer index range without allocating dense topology."""
    if index.numel() and (bool((index < 0).any()) or bool((index >= size).any())):
        raise ValueError(f"{name} contains an index outside the node range.")


def _validate_nonbonded_topology(
    nonbonded: torch.Tensor, targets: TrainingTargets
) -> None:
    """Validate sparse nonbonded uniqueness, membership, and bond disjointness."""
    if nonbonded.shape[1] == 0:
        return
    left, right = nonbonded
    if not torch.equal(targets.node_batch[left], targets.node_batch[right]):
        raise ValueError("nonbonded pairs must stay within one complex.")
    node_count = targets.node_batch.shape[0]
    codes = left * node_count + right
    if codes.unique().numel() != codes.numel():
        raise ValueError("nonbonded pairs must be unique.")
    bond_codes = targets.halfedge_index[0] * node_count + targets.halfedge_index[1]
    if torch.isin(codes, bond_codes).any():
        raise ValueError("nonbonded pairs must be disjoint from bonded halfedges.")
