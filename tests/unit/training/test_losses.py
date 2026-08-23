"""Behavioral tests for scientifically typed composite training losses."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pydantic import ValidationError

from ecloudflow.config import AppConfig, LossConfig, load_config
from ecloudflow.ecloud.decoder import ElectronReconstruction
from ecloudflow.models import ModelPrediction
from ecloudflow.training import (
    RunningLossScaler,
    TrainingTargets,
    compute_ecloudflow_loss,
)


def _prediction(*, requires_grad: bool = True) -> ModelPrediction:
    def value(data: list[object]) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, requires_grad=requires_grad)

    return ModelPrediction(
        position_velocity=value([[1.0, 0.0, 0.0], [9.0, 9.0, 9.0], [0.0, 1.0, 0.0]]),
        position_score=value([[0.2, 0.0, 0.0], [8.0, 8.0, 8.0], [0.0, 0.2, 0.0]]),
        electron_velocity=value([[0.5, 0.0], [7.0, 7.0], [0.0, 0.5]]),
        electron_score=value([[0.1, 0.0], [6.0, 6.0], [0.0, 0.1]]),
        atom_logits=value([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]]),
        charge_logits=value([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]]),
        bond_logits=value([[0.0, 3.0], [3.0, 0.0]]),
        count_logits=torch.log_softmax(value([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]), -1),
        affinity=value([1.5, -2.0]),
        interaction_logits=value([2.0, -2.0]),
        pocket_cache_key="fixture",
        affinity_log_variance=value([0.2, -0.2]),
        endpoint_positions=value([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        endpoint_electron_latent=value([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        electron_reconstruction=ElectronReconstruction(
            density=value([[1.0, 2.0], [3.0, 4.0]]),
            gradient=value(
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]]
            ),
            electron_count=value([2.9, 7.0]),
            dipole=value([[0.1, 0.0, 0.0], [3.0, 3.0, 3.0]]),
            latent_round_trip=value(
                [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.0, 0.0]]]
            ),
        ),
    )


def _targets() -> TrainingTargets:
    return TrainingTargets(
        position_velocity=torch.zeros(3, 3),
        position_score=torch.zeros(3, 3),
        electron_velocity=torch.zeros(3, 2),
        electron_score=torch.zeros(3, 2),
        atom_classes=torch.tensor([0, 1, 0]),
        charge_classes=torch.tensor([0, 1, 0]),
        bond_classes=torch.tensor([1, 0]),
        count_classes=torch.tensor([1, 2]),
        editable_atom_mask=torch.tensor([True, False, True]),
        editable_bond_mask=torch.tensor([True, False]),
        node_batch=torch.tensor([0, 0, 0]),
        halfedge_index=torch.tensor([[0, 0], [1, 2]]),
        halfedge_batch=torch.tensor([0, 0]),
        count_mask=torch.tensor([True, True]),
        qm_mask=torch.tensor([True, False]),
        density=torch.tensor([[1.2, 1.8], [99.0, 99.0]]),
        density_gradient=torch.zeros(2, 2, 3),
        field_mask=torch.tensor([[True, True], [True, True]]),
        electron_count=torch.tensor([3.0, 99.0]),
        dipole=torch.zeros(2, 3),
        latent_cycle=torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        valence_limits=torch.tensor([4.0, 4.0, 4.0]),
        bond_order_values=torch.tensor([0.0, 1.0]),
        bond_length_mean=torch.tensor([1.4, 1.4]),
        bond_length_std=torch.tensor([0.1, 0.1]),
        nonbonded_halfedge_index=torch.tensor([[0], [2]]),
        protein_positions=torch.tensor([[3.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        protein_batch=torch.tensor([0, 1]),
        ring_triplets=torch.empty(3, 0, dtype=torch.long),
        ring_angle_mean=torch.empty(0),
        ring_angle_std=torch.empty(0),
        interaction=torch.tensor([1.0, 0.0]),
        interaction_mask=torch.tensor([True, True]),
        affinity=torch.tensor([1.0, 0.0]),
        affinity_mask=torch.tensor([True, False]),
    )


def test_composite_loss_has_exact_six_components_and_fixed_zero_diagnostics() -> None:
    """Mutation caught: adding fixed entries or a seventh component breaks the API."""
    result = compute_ecloudflow_loss(_prediction(), _targets(), LossConfig())

    expected = {"flow", "score", "discrete", "ecloud", "chem", "interaction"}
    assert set(result.raw) == expected
    assert set(result.normalized) == expected
    assert set(result.weighted) == expected
    assert result.diagnostics.flow_fixed.item() == 0.0
    assert result.diagnostics.score_fixed.item() == 0.0
    assert torch.isfinite(result.total)


def test_all_fixed_and_all_missing_supervision_is_differentiable_zero() -> None:
    """Mutation caught: mean-of-empty masks produce NaN or detach the zero loss."""
    targets = replace(
        _targets(),
        editable_atom_mask=torch.zeros(3, dtype=torch.bool),
        editable_bond_mask=torch.zeros(2, dtype=torch.bool),
        count_mask=torch.zeros(2, dtype=torch.bool),
        qm_mask=torch.zeros(2, dtype=torch.bool),
        interaction_mask=torch.zeros(2, dtype=torch.bool),
        affinity_mask=torch.zeros(2, dtype=torch.bool),
        nonbonded_halfedge_index=torch.empty(2, 0, dtype=torch.long),
    )
    result = compute_ecloudflow_loss(_prediction(), targets, LossConfig())
    result.total.backward()

    assert result.total.item() == 0.0
    assert torch.isfinite(result.total)


def test_empty_flattened_ligand_with_missing_labels_is_finite_zero() -> None:
    """Mutation caught: empty node/halfedge means create NaN or drop batch outputs."""
    prediction = ModelPrediction(
        position_velocity=torch.empty(0, 3, requires_grad=True),
        position_score=torch.empty(0, 3, requires_grad=True),
        electron_velocity=torch.empty(0, 2, requires_grad=True),
        electron_score=torch.empty(0, 2, requires_grad=True),
        atom_logits=torch.empty(0, 2, requires_grad=True),
        charge_logits=torch.empty(0, 2, requires_grad=True),
        bond_logits=torch.empty(0, 2, requires_grad=True),
        count_logits=torch.log_softmax(torch.zeros(2, 3, requires_grad=True), -1),
        affinity=torch.zeros(2, requires_grad=True),
        affinity_log_variance=torch.zeros(2, requires_grad=True),
        interaction_logits=torch.zeros(2, requires_grad=True),
        endpoint_positions=torch.empty(0, 3, requires_grad=True),
        endpoint_electron_latent=torch.empty(0, 2, requires_grad=True),
        pocket_cache_key="empty",
    )
    targets = TrainingTargets(
        position_velocity=torch.empty(0, 3),
        position_score=torch.empty(0, 3),
        electron_velocity=torch.empty(0, 2),
        electron_score=torch.empty(0, 2),
        atom_classes=torch.empty(0, dtype=torch.long),
        charge_classes=torch.empty(0, dtype=torch.long),
        bond_classes=torch.empty(0, dtype=torch.long),
        count_classes=torch.zeros(2, dtype=torch.long),
        editable_atom_mask=torch.empty(0, dtype=torch.bool),
        editable_bond_mask=torch.empty(0, dtype=torch.bool),
        node_batch=torch.empty(0, dtype=torch.long),
        halfedge_index=torch.empty(2, 0, dtype=torch.long),
        halfedge_batch=torch.empty(0, dtype=torch.long),
        count_mask=torch.zeros(2, dtype=torch.bool),
        qm_mask=torch.zeros(2, dtype=torch.bool),
    )

    result = compute_ecloudflow_loss(prediction, targets, LossConfig())
    result.total.backward()

    assert result.total.item() == 0.0
    assert torch.isfinite(result.total)


def test_mixed_qm_mask_excludes_non_qm_density_values() -> None:
    """Mutation caught: treating approximate/non-QM rows as QM changes the objective."""
    targets = _targets()
    first = compute_ecloudflow_loss(_prediction(), targets, LossConfig()).raw["ecloud"]
    changed = replace(
        targets,
        density=targets.density.index_put(
            (torch.tensor([1]),), torch.tensor([[1.0e6, -1.0e6]])
        ),
    )
    second = compute_ecloudflow_loss(_prediction(), changed, LossConfig()).raw["ecloud"]

    assert torch.equal(first, second)


def test_enabled_qm_term_requires_real_reconstruction() -> None:
    """Mutation caught: reusing an unrelated model tensor fabricates density prediction."""
    prediction = replace(_prediction(), electron_reconstruction=None)
    with pytest.raises(ValueError, match="electron reconstruction"):
        compute_ecloudflow_loss(prediction, _targets(), LossConfig())


def test_zero_weight_scientific_subterms_do_not_require_unused_context() -> None:
    """Mutation caught: disabled terms still demand labels or prediction auxiliaries."""
    base = LossConfig()
    ecloud = base.ecloud.model_copy(
        update={
            "density": 0.0,
            "gradient": 0.0,
            "electron_count": 0.0,
            "dipole": 0.0,
            "cycle": 0.0,
        }
    )
    chem = base.chem.model_copy(
        update={
            "valence": 0.0,
            "bond_length": 0.0,
            "ligand_clash": 0.0,
            "protein_clash": 0.0,
            "ring_strain": 0.0,
            "connectivity": 0.0,
        }
    )
    targets = replace(
        _targets(),
        density=None,
        density_gradient=None,
        field_mask=None,
        electron_count=None,
        dipole=None,
        latent_cycle=None,
        valence_limits=None,
        bond_order_values=None,
        bond_length_mean=None,
        bond_length_std=None,
        nonbonded_halfedge_index=None,
        protein_positions=None,
        protein_batch=None,
        ring_triplets=None,
        ring_angle_mean=None,
        ring_angle_std=None,
    )

    result = compute_ecloudflow_loss(
        replace(_prediction(), electron_reconstruction=None),
        targets,
        base.model_copy(update={"ecloud": ecloud, "chem": chem}),
    )

    assert result.raw["ecloud"].item() == 0.0
    assert torch.isfinite(result.raw["chem"])


def test_prediction_rejects_malformed_decoder_reconstruction_contract() -> None:
    """Mutation caught: detached wrong-dtype density crosses the typed prediction API."""
    prediction = _prediction()
    assert prediction.electron_reconstruction is not None
    malformed = prediction.electron_reconstruction._replace(
        density=prediction.electron_reconstruction.density.double()
    )
    with pytest.raises(ValueError, match="electron reconstruction"):
        replace(prediction, electron_reconstruction=malformed)


def test_qm_target_shape_cannot_broadcast_across_field_points() -> None:
    """Mutation caught: implicit broadcasting fabricates repeated density labels."""
    with pytest.raises(ValueError, match="shape"):
        compute_ecloudflow_loss(
            _prediction(), replace(_targets(), density=torch.ones(2, 1)), LossConfig()
        )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("atom_classes", torch.tensor([0, 2, 0])),
        ("bond_classes", torch.tensor([1, -1])),
    ],
)
def test_class_targets_are_range_validated(field: str, bad: torch.Tensor) -> None:
    """Mutation caught: out-of-vocabulary endpoint classes reach cross entropy."""
    with pytest.raises(ValueError, match="class"):
        compute_ecloudflow_loss(
            _prediction(), replace(_targets(), **{field: bad}), LossConfig()
        )


def test_sparse_halfedges_must_be_canonical_and_same_complex() -> None:
    """Mutation caught: reversed or cross-complex halfedges double count chemistry."""
    bad = replace(_targets(), halfedge_index=torch.tensor([[1, 0], [0, 2]]))
    with pytest.raises(ValueError, match="unordered halfedge"):
        compute_ecloudflow_loss(_prediction(), bad, LossConfig())


def test_zero_degree_connectivity_is_finite_and_has_bond_gradient() -> None:
    """Mutation caught: a division by degree makes isolated nodes non-finite."""
    prediction = replace(
        _prediction(),
        bond_logits=torch.tensor([[8.0, -8.0], [8.0, -8.0]], requires_grad=True),
    )
    result = compute_ecloudflow_loss(prediction, _targets(), LossConfig())
    result.raw["chem"].backward()

    assert torch.isfinite(result.raw["chem"])
    assert prediction.bond_logits.grad is not None
    assert torch.isfinite(prediction.bond_logits.grad).all()


def test_component_weight_and_warmup_zero_without_changing_raw_value() -> None:
    """Mutation caught: warm-up accidentally changes the reported scientific raw loss."""
    base = LossConfig()
    config = base.model_copy(
        update={
            "flow": base.flow.model_copy(update={"weight": 0.0}),
            "score": base.score.model_copy(update={"warmup_start": 4, "warmup_end": 8}),
        }
    )
    result = compute_ecloudflow_loss(_prediction(), _targets(), config, step=2)

    assert result.raw["flow"] > 0
    assert result.raw["score"] > 0
    assert result.weighted["flow"].item() == 0.0
    assert result.weighted["score"].item() == 0.0


def test_running_rms_uses_detached_state_and_preserves_component_gradient_direction() -> (
    None
):
    """Mutation caught: scaler statistics retain autograd or reverse a gradient."""
    prediction = _prediction()
    scaler = RunningLossScaler(decay=0.5, epsilon=1.0e-8)
    raw = compute_ecloudflow_loss(prediction, _targets(), LossConfig(), step=0).raw[
        "flow"
    ]
    result = compute_ecloudflow_loss(
        prediction, _targets(), LossConfig(), scaler=scaler, step=0
    )
    raw_grad = torch.autograd.grad(
        raw, prediction.position_velocity, retain_graph=True
    )[0]
    normalized_grad = torch.autograd.grad(
        result.normalized["flow"], prediction.position_velocity
    )[0]

    assert scaler.mean_square.grad_fn is None
    assert torch.sum(raw_grad * normalized_grad) > 0
    assert torch.all((raw_grad == 0) == (normalized_grad == 0))


def test_running_scaler_ignores_components_without_supervision_and_round_trips_state() -> (
    None
):
    """Mutation caught: missing-label batches decay scales or checkpoints lose them."""
    scaler = RunningLossScaler(decay=0.5, epsilon=1.0e-8)
    targets = replace(_targets(), qm_mask=torch.zeros(2, dtype=torch.bool))
    compute_ecloudflow_loss(_prediction(), targets, LossConfig(), scaler=scaler)
    ecloud_index = scaler.component_names.index("ecloud")
    state = scaler.state_dict()
    restored = RunningLossScaler(decay=0.5, epsilon=1.0e-8)
    restored.load_state_dict(state)

    assert not scaler.initialized[ecloud_index]
    assert torch.equal(restored.mean_square, scaler.mean_square)
    assert torch.equal(restored.initialized, scaler.initialized)


def test_running_scaler_applies_decay_after_first_observation() -> None:
    """Mutation caught: advanced-index updates modify a copy instead of the buffer."""
    scaler = RunningLossScaler(decay=0.5, epsilon=1.0e-8)
    zero = torch.tensor(0.0)
    first = {name: zero for name in scaler.component_names}
    second = dict(first)
    first["flow"] = torch.tensor(2.0)
    second["flow"] = torch.tensor(4.0)
    active = {name: name == "flow" for name in scaler.component_names}

    scaler.update(first, active)
    scaler.update(second, active)

    assert scaler.mean_square[0].item() == 10.0


def _distributed_scaler_worker(rank: int, init_file: str) -> None:
    """Assert two Gloo ranks converge from different local sufficient statistics."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=Path(init_file).as_uri()
    )
    try:
        scaler = RunningLossScaler(decay=0.5, epsilon=1.0e-8)
        zero = torch.tensor(0.0)
        values = {name: zero for name in scaler.component_names}
        values["flow"] = torch.tensor(1.0 if rank == 0 else 3.0)
        active = {name: name == "flow" for name in scaler.component_names}
        scaler.update(values, active)
        gathered = [torch.empty_like(scaler.mean_square) for _ in range(2)]
        dist.all_gather(gathered, scaler.mean_square)
        assert torch.equal(gathered[0], gathered[1])
        assert gathered[0][0].item() == 5.0
    finally:
        dist.destroy_process_group()


def test_running_scaler_synchronizes_detached_statistics_across_gloo_ranks(
    tmp_path: Path,
) -> None:
    """Mutation caught: rank-local scale updates silently diverge under DDP."""
    init_file = tmp_path / "gloo-init"
    mp.spawn(_distributed_scaler_worker, args=(str(init_file),), nprocs=2, join=True)


def test_affinity_log_variance_is_clamped_per_example() -> None:
    """Mutation caught: one extreme variance overflows heteroscedastic NLL."""
    prediction = replace(
        _prediction(),
        affinity_log_variance=torch.tensor([1000.0, -1000.0], requires_grad=True),
    )
    targets = replace(_targets(), affinity_mask=torch.tensor([True, True]))
    result = compute_ecloudflow_loss(prediction, targets, LossConfig())

    assert torch.isfinite(result.raw["chem"])
    assert result.diagnostics.affinity_log_variance_min.item() >= -10.0
    assert result.diagnostics.affinity_log_variance_max.item() <= 10.0


def test_nonfinite_active_component_fails_before_backward() -> None:
    """Mutation caught: NaN enters the optimizer through an active loss component."""
    prediction = replace(
        _prediction(),
        endpoint_positions=torch.tensor(
            [[float("nan"), 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
            requires_grad=True,
        ),
    )
    with pytest.raises(FloatingPointError, match="chem"):
        compute_ecloudflow_loss(prediction, _targets(), LossConfig())


def test_loss_config_is_frozen_serializable_bounded_and_forbids_unknown_keys() -> None:
    """Mutation caught: untyped loss overrides silently bypass validation."""
    config = AppConfig(loss={"interaction": {"weight": 0.25, "focal_gamma": 1.5}})

    assert config.loss.interaction.weight == 0.25
    assert LossConfig.model_validate_json(config.loss.model_dump_json()) == config.loss
    with pytest.raises(ValidationError):
        AppConfig(loss={"interaction": {"unknown": 1}})
    with pytest.raises(ValidationError):
        LossConfig(normalization={"decay": 1.0})
    with pytest.raises(ValidationError):
        LossConfig(flow={"warmup_start": 3, "warmup_end": 2})


def test_hydra_can_override_nested_loss_weights_without_add_syntax() -> None:
    """Mutation caught: schema-only loss config remains absent from Hydra composition."""
    config = load_config(["loss.interaction.weight=0.25", "loss.chem.affinity=0.0"])
    assert config.loss.interaction.weight == 0.25
    assert config.loss.chem.affinity == 0.0
