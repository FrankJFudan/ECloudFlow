"""Behavioral tests for scientifically typed composite training losses."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
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
        nonbonded_halfedge_index=torch.tensor([[1], [2]]),
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


def _leaf_with_row(tensor: torch.Tensor, row: int, value: float) -> torch.Tensor:
    """Return a fresh gradient leaf with one deliberately invalid excluded row."""
    changed = tensor.detach().clone()
    changed[row] = value
    return changed.requires_grad_(True)


def test_excluded_nonfinite_and_class_sentinel_rows_are_never_evaluated() -> None:
    """Mutation caught: multiplying post-arithmetic NaN by zero contaminates loss."""
    prediction = _prediction()
    assert prediction.electron_reconstruction is not None
    reconstruction = ElectronReconstruction(
        density=_leaf_with_row(
            prediction.electron_reconstruction.density, 1, float("nan")
        ),
        gradient=_leaf_with_row(
            prediction.electron_reconstruction.gradient, 1, float("inf")
        ),
        electron_count=_leaf_with_row(
            prediction.electron_reconstruction.electron_count, 1, float("nan")
        ),
        dipole=_leaf_with_row(
            prediction.electron_reconstruction.dipole, 1, float("inf")
        ),
        latent_round_trip=_leaf_with_row(
            prediction.electron_reconstruction.latent_round_trip, 1, float("nan")
        ),
    )
    prediction = replace(
        prediction,
        position_velocity=_leaf_with_row(prediction.position_velocity, 1, float("nan")),
        position_score=_leaf_with_row(prediction.position_score, 1, float("inf")),
        electron_velocity=_leaf_with_row(prediction.electron_velocity, 1, float("nan")),
        electron_score=_leaf_with_row(prediction.electron_score, 1, float("inf")),
        atom_logits=_leaf_with_row(prediction.atom_logits, 1, float("nan")),
        charge_logits=_leaf_with_row(prediction.charge_logits, 1, float("inf")),
        bond_logits=_leaf_with_row(prediction.bond_logits, 1, float("nan")),
        count_logits=_leaf_with_row(prediction.count_logits, 1, float("inf")),
        affinity=_leaf_with_row(prediction.affinity, 1, float("nan")),
        affinity_log_variance=_leaf_with_row(
            prediction.affinity_log_variance, 1, float("inf")
        ),
        interaction_logits=_leaf_with_row(
            prediction.interaction_logits, 1, float("nan")
        ),
        electron_reconstruction=reconstruction,
    )
    targets = replace(
        _targets(),
        atom_classes=torch.tensor([0, -1, 0]),
        charge_classes=torch.tensor([0, 999, 0]),
        bond_classes=torch.tensor([1, -1]),
        count_classes=torch.tensor([1, -1]),
        count_mask=torch.tensor([True, False]),
        density=torch.tensor([[1.2, 1.8], [float("nan"), float("inf")]]),
        density_gradient=torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[float("nan"), 0.0, 0.0], [float("inf"), 0.0, 0.0]],
            ]
        ),
        electron_count=torch.tensor([3.0, float("nan")]),
        dipole=torch.tensor([[0.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]]),
        latent_cycle=torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[float("nan"), 0.0], [float("inf"), 0.0]],
            ]
        ),
        interaction=torch.tensor([1.0, float("nan")]),
        interaction_mask=torch.tensor([True, False]),
        affinity=torch.tensor([1.0, float("nan")]),
        affinity_mask=torch.tensor([True, False]),
    )

    result = compute_ecloudflow_loss(prediction, targets, LossConfig())
    result.total.backward()

    assert torch.isfinite(result.total)
    for tensor in (
        prediction.position_velocity,
        prediction.position_score,
        prediction.electron_velocity,
        prediction.electron_score,
        prediction.atom_logits,
        prediction.charge_logits,
        prediction.bond_logits,
        prediction.count_logits,
        prediction.affinity,
        prediction.affinity_log_variance,
        prediction.interaction_logits,
        reconstruction.density,
        reconstruction.gradient,
        reconstruction.electron_count,
        reconstruction.dipole,
        reconstruction.latent_round_trip,
    ):
        assert tensor.grad is not None
        assert torch.equal(tensor.grad[1], torch.zeros_like(tensor.grad[1]))


def test_all_false_masks_ignore_nonfinite_placeholders_with_exact_zero_gradient() -> (
    None
):
    """Mutation caught: empty selection still sums nonfinite backing storage."""
    prediction = replace(
        _prediction(),
        position_velocity=torch.full((3, 3), float("nan"), requires_grad=True),
        atom_logits=torch.full((3, 2), float("inf"), requires_grad=True),
    )
    targets = replace(
        _targets(),
        editable_atom_mask=torch.zeros(3, dtype=torch.bool),
        editable_bond_mask=torch.zeros(2, dtype=torch.bool),
        count_mask=torch.zeros(2, dtype=torch.bool),
        qm_mask=torch.zeros(2, dtype=torch.bool),
        interaction_mask=torch.zeros(2, dtype=torch.bool),
        affinity_mask=torch.zeros(2, dtype=torch.bool),
    )

    result = compute_ecloudflow_loss(prediction, targets, LossConfig())
    result.total.backward()

    assert result.total.item() == 0.0
    assert torch.equal(
        prediction.position_velocity.grad,
        torch.zeros_like(prediction.position_velocity.grad),
    )
    assert torch.equal(
        prediction.atom_logits.grad, torch.zeros_like(prediction.atom_logits.grad)
    )


def test_missing_cycle_tokens_are_selected_before_nonfinite_arithmetic() -> None:
    """Mutation caught: padded cycle entries had no typed missing-label mask."""
    prediction = _prediction()
    assert prediction.electron_reconstruction is not None
    cycle = prediction.electron_reconstruction.latent_round_trip.detach().clone()
    cycle[0, 1] = float("nan")
    cycle = cycle.requires_grad_(True)
    prediction = replace(
        prediction,
        electron_reconstruction=prediction.electron_reconstruction._replace(
            latent_round_trip=cycle
        ),
    )
    target_cycle = _targets().latent_cycle
    assert target_cycle is not None
    target_cycle = target_cycle.clone()
    target_cycle[0, 1] = float("inf")
    targets = replace(
        _targets(),
        latent_cycle=target_cycle,
        latent_cycle_mask=torch.tensor([[True, False], [True, True]]),
    )

    result = compute_ecloudflow_loss(prediction, targets, LossConfig())
    result.raw["ecloud"].backward()

    assert torch.isfinite(result.raw["ecloud"])
    assert cycle.grad is not None
    assert torch.count_nonzero(cycle.grad[0, 1]) == 0


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


def test_enabled_field_terms_with_no_selected_points_require_no_reconstruction() -> (
    None
):
    """Mutation caught: a configured but unobserved field term forced decoding."""
    base = LossConfig()
    ecloud = base.ecloud.model_copy(
        update={"electron_count": 0.0, "dipole": 0.0, "cycle": 0.0}
    )
    targets = replace(
        _targets(),
        field_mask=torch.zeros(2, 2, dtype=torch.bool),
        density=None,
        density_gradient=None,
    )

    result = compute_ecloudflow_loss(
        replace(_prediction(), electron_reconstruction=None),
        targets,
        base.model_copy(update={"ecloud": ecloud}),
    )

    assert result.raw["ecloud"].item() == 0.0
    assert result.diagnostics.supervised_counts["ecloud_density"].item() == 0


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
    ("change", "message"),
    [
        ({"field_mask": torch.ones(2, 2)}, "field_mask"),
        ({"latent_cycle_mask": torch.ones(2, 2)}, "latent_cycle_mask"),
        ({"bond_length_mean": torch.tensor([1.4])}, "bond_length_mean"),
        ({"affinity": torch.ones(2, 1)}, "affinity target"),
        ({"affinity": torch.tensor([float("nan"), 0.0])}, "active affinity"),
    ],
)
def test_optional_contracts_raise_named_errors_before_arithmetic_or_indexing(
    change: dict[str, torch.Tensor], message: str
) -> None:
    """Mutation caught: optional tensor errors leaked from broadcasting/index kernels."""
    with pytest.raises(ValueError, match=message):
        compute_ecloudflow_loss(
            _prediction(), replace(_targets(), **change), LossConfig()
        )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("atom_classes", torch.tensor([2, 1, 0])),
        ("bond_classes", torch.tensor([-1, 0])),
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


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"nonbonded_halfedge_index": torch.tensor([[0], [1]])}, "disjoint"),
        (
            {"nonbonded_halfedge_index": torch.tensor([[1, 1], [2, 2]])},
            "unique",
        ),
        (
            {
                "ring_triplets": torch.tensor([[0], [0], [2]]),
                "ring_angle_mean": torch.tensor([1.0]),
                "ring_angle_std": torch.tensor([0.1]),
            },
            "distinct",
        ),
        (
            {
                "ring_triplets": torch.tensor([[1], [0], [2]]),
                "ring_angle_mean": torch.tensor([1.0]),
                "ring_angle_std": torch.tensor([-0.1]),
            },
            "positive",
        ),
    ],
)
def test_sparse_scientific_topology_is_validated_without_dense_allocation(
    change: dict[str, torch.Tensor], message: str
) -> None:
    """Mutation caught: invalid sparse scientific edges/triplets reach geometry math."""
    with pytest.raises(ValueError, match=message):
        compute_ecloudflow_loss(
            _prediction(), replace(_targets(), **change), LossConfig()
        )


def test_ring_triplets_reject_reversed_duplicates_and_missing_bond_arms() -> None:
    """Mutation caught: ring angles were accepted without canonical bonded support."""
    valid = replace(
        _targets(),
        ring_triplets=torch.tensor([[1], [0], [2]]),
        ring_angle_mean=torch.tensor([1.0]),
        ring_angle_std=torch.tensor([0.1]),
    )
    result = compute_ecloudflow_loss(_prediction(), valid, LossConfig())
    assert torch.isfinite(result.raw["chem"])

    duplicate = replace(
        valid,
        ring_triplets=torch.tensor([[1, 2], [0, 0], [2, 1]]),
        ring_angle_mean=torch.tensor([1.0, 1.0]),
        ring_angle_std=torch.tensor([0.1, 0.1]),
    )
    with pytest.raises(ValueError, match="unique"):
        compute_ecloudflow_loss(_prediction(), duplicate, LossConfig())

    missing_arm = replace(valid, ring_triplets=torch.tensor([[0], [1], [2]]))
    with pytest.raises(ValueError, match="bonded arms"):
        compute_ecloudflow_loss(_prediction(), missing_arm, LossConfig())


def test_ring_triplet_rejects_cross_complex_membership() -> None:
    """Mutation caught: ring topology could span complexes despite valid bond rows."""
    prediction = _prediction()

    def append_node(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat((tensor.detach(), tensor.detach()[:1]), dim=0).requires_grad_(
            tensor.requires_grad
        )

    prediction = replace(
        prediction,
        position_velocity=append_node(prediction.position_velocity),
        position_score=append_node(prediction.position_score),
        electron_velocity=append_node(prediction.electron_velocity),
        electron_score=append_node(prediction.electron_score),
        atom_logits=append_node(prediction.atom_logits),
        charge_logits=append_node(prediction.charge_logits),
        endpoint_positions=append_node(prediction.endpoint_positions),
        endpoint_electron_latent=append_node(prediction.endpoint_electron_latent),
    )
    targets = _targets()
    chem = LossConfig().chem.model_copy(
        update={
            "valence": 0.0,
            "bond_length": 0.0,
            "ligand_clash": 0.0,
            "protein_clash": 0.0,
            "connectivity": 0.0,
            "affinity": 0.0,
        }
    )
    config = LossConfig().model_copy(update={"chem": chem})
    targets = replace(
        targets,
        position_velocity=torch.cat((targets.position_velocity, torch.zeros(1, 3))),
        position_score=torch.cat((targets.position_score, torch.zeros(1, 3))),
        electron_velocity=torch.cat((targets.electron_velocity, torch.zeros(1, 2))),
        electron_score=torch.cat((targets.electron_score, torch.zeros(1, 2))),
        atom_classes=torch.tensor([0, 1, 0, 0]),
        charge_classes=torch.tensor([0, 1, 0, 0]),
        editable_atom_mask=torch.tensor([True, False, True, True]),
        node_batch=torch.tensor([0, 0, 1, 1]),
        halfedge_index=torch.tensor([[0, 2], [1, 3]]),
        halfedge_batch=torch.tensor([0, 1]),
        valence_limits=torch.full((4,), 4.0),
        ring_triplets=torch.tensor([[0], [1], [2]]),
        ring_angle_mean=torch.tensor([1.0]),
        ring_angle_std=torch.tensor([0.1]),
    )

    with pytest.raises(ValueError, match="one complex"):
        compute_ecloudflow_loss(prediction, targets, config)


def test_active_bond_prior_requires_positive_finite_stddev() -> None:
    """Mutation caught: clamping an invalid active prior silently changes supervision."""
    targets = replace(_targets(), bond_length_std=torch.tensor([-1.0, float("nan")]))
    with pytest.raises(ValueError, match="positive"):
        compute_ecloudflow_loss(_prediction(), targets, LossConfig())


def test_diagnostics_count_actual_enabled_observations_per_subterm() -> None:
    """Mutation caught: broad graph/QM masks over-report pair and point supervision."""
    result = compute_ecloudflow_loss(_prediction(), _targets(), LossConfig())
    counts = {
        name: int(value) for name, value in result.diagnostics.supervised_counts.items()
    }

    assert counts["flow_position"] == 2
    assert counts["flow_electron"] == 2
    assert counts["score_position"] == 2
    assert counts["discrete_bond"] == 1
    assert counts["ecloud_density"] == 2
    assert counts["ecloud_electron_count"] == 1
    assert counts["chem_bond_length"] == 1
    assert counts["chem_ligand_clash"] == 1
    assert counts["chem_ring_strain"] == 0
    assert counts["interaction"] == 2


def test_default_cycle_mask_counts_observed_tokens_not_qm_rows() -> None:
    """Mutation caught: default cycle diagnostics counted complexes instead of tokens."""
    result = compute_ecloudflow_loss(_prediction(), _targets(), LossConfig())

    assert result.diagnostics.supervised_counts["ecloud_cycle"].item() == 2


def test_disabled_component_does_not_initialize_or_decay_running_scale() -> None:
    """Mutation caught: broad masks update a scale for a zero-weight component."""
    base = LossConfig()
    config = base.model_copy(
        update={"ecloud": base.ecloud.model_copy(update={"weight": 0.0})}
    )
    scaler = RunningLossScaler(decay=0.5)
    compute_ecloudflow_loss(_prediction(), _targets(), config, scaler=scaler)

    assert not scaler.initialized[scaler.component_names.index("ecloud")]


def test_disabled_optional_components_require_no_context_and_report_zero_counts() -> (
    None
):
    """Mutation caught: component-zero chemistry/interaction still required labels."""
    base = LossConfig()
    config = base.model_copy(
        update={
            "flow": base.flow.model_copy(update={"weight": 0.0}),
            "chem": base.chem.model_copy(update={"weight": 0.0}),
            "interaction": base.interaction.model_copy(update={"weight": 0.0}),
        }
    )
    targets = replace(
        _targets(),
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
        affinity=None,
        interaction=None,
    )

    result = compute_ecloudflow_loss(_prediction(), targets, config)

    assert result.raw["chem"].item() == 0.0
    assert result.raw["interaction"].item() == 0.0
    assert result.diagnostics.supervised_counts["flow_position"].item() == 0
    assert all(
        value.item() == 0
        for name, value in result.diagnostics.supervised_counts.items()
        if name.startswith("chem_") or name == "interaction"
    )


def test_disabled_sparse_subterms_do_not_index_inactive_sentinels() -> None:
    """Mutation caught: diagnostics indexed disabled pair/triplet sentinel values."""
    base = LossConfig()
    chem = base.chem.model_copy(update={"ligand_clash": 0.0, "ring_strain": 0.0})
    targets = replace(
        _targets(),
        nonbonded_halfedge_index=torch.tensor([[-(10**9)], [10**9]]),
        ring_triplets=torch.tensor([[-(10**9)], [10**9], [10**9 + 1]]),
        ring_angle_mean=torch.tensor([float("nan")]),
        ring_angle_std=torch.tensor([float("inf")]),
    )

    result = compute_ecloudflow_loss(
        _prediction(), targets, base.model_copy(update={"chem": chem})
    )

    assert result.diagnostics.supervised_counts["chem_ligand_clash"].item() == 0
    assert result.diagnostics.supervised_counts["chem_ring_strain"].item() == 0


def test_zero_weight_ecloud_component_requires_no_decoder_prediction() -> None:
    """Mutation caught: inactive expensive reconstruction still required context/output."""
    base = LossConfig()
    config = base.model_copy(
        update={"ecloud": base.ecloud.model_copy(update={"weight": 0.0})}
    )
    prediction = replace(_prediction(), electron_reconstruction=None)

    result = compute_ecloudflow_loss(prediction, _targets(), config)

    assert result.raw["ecloud"].item() == 0.0
    assert result.weighted["ecloud"].item() == 0.0


def test_zero_component_factor_does_not_multiply_inactive_nonfinite_raw() -> None:
    """Mutation caught: IEEE NaN times a zero component factor remained NaN."""
    base = LossConfig()
    config = base.model_copy(
        update={"flow": base.flow.model_copy(update={"weight": 0.0})}
    )
    prediction = replace(
        _prediction(),
        position_velocity=torch.full((3, 3), float("nan"), requires_grad=True),
        electron_velocity=torch.full((3, 2), float("inf"), requires_grad=True),
    )

    result = compute_ecloudflow_loss(prediction, _targets(), config)

    assert result.weighted["flow"].item() == 0.0
    assert torch.isfinite(result.total)


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


def _distributed_nonfinite_worker(rank: int, init_file: str) -> None:
    """Require every rank to raise the same diagnostic without collective deadlock."""
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=Path(init_file).as_uri(),
        timeout=timedelta(seconds=5),
    )
    try:
        prediction = _prediction()
        if rank == 1:
            changed = prediction.position_velocity.detach().clone()
            changed[0, 0] = float("nan")
            prediction = replace(
                prediction, position_velocity=changed.requires_grad_(True)
            )
        scaler = RunningLossScaler()
        message = "no error"
        try:
            compute_ecloudflow_loss(prediction, _targets(), LossConfig(), scaler=scaler)
        except FloatingPointError as error:
            message = str(error)
        gathered: list[str | None] = [None, None]
        dist.all_gather_object(gathered, message)
        assert gathered[0] == gathered[1]
        assert gathered[0] is not None and "flow" in gathered[0]
    finally:
        dist.destroy_process_group()


def test_nonfinite_on_one_gloo_rank_raises_consistently_without_deadlock(
    tmp_path: Path,
) -> None:
    """Mutation caught: local fail-fast strands peers inside scaler all-reduce."""
    mp.spawn(
        _distributed_nonfinite_worker,
        args=(str(tmp_path / "gloo-nonfinite"),),
        nprocs=2,
        join=True,
    )


def _distributed_asymmetric_presence_worker(rank: int, init_file: str) -> None:
    """Verify a missing rank neither adds a zero observation nor blocks updates."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=Path(init_file).as_uri()
    )
    try:
        scaler = RunningLossScaler()
        zero = torch.tensor(0.0)
        values = {name: zero for name in scaler.component_names}
        values["flow"] = torch.tensor(3.0 if rank else 0.0)
        active = {name: name == "flow" and rank == 1 for name in scaler.component_names}
        scaler.update(values, active)
        assert scaler.initialized[0]
        assert scaler.mean_square[0].item() == 9.0
    finally:
        dist.destroy_process_group()


def test_gloo_scaler_handles_asymmetric_finite_and_missing_supervision(
    tmp_path: Path,
) -> None:
    """Mutation caught: missing-rank zeros bias globally present sufficient statistics."""
    mp.spawn(
        _distributed_asymmetric_presence_worker,
        args=(str(tmp_path / "gloo-asymmetric"),),
        nprocs=2,
        join=True,
    )


def _distributed_diagnostic_counts_worker(rank: int, init_file: str) -> None:
    """Require globally identical detached counts with asymmetric local masks."""
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=Path(init_file).as_uri()
    )
    try:
        targets = _targets()
        if rank == 1:
            targets = replace(
                targets,
                editable_atom_mask=torch.zeros(3, dtype=torch.bool),
                editable_bond_mask=torch.zeros(2, dtype=torch.bool),
                count_mask=torch.zeros(2, dtype=torch.bool),
                qm_mask=torch.zeros(2, dtype=torch.bool),
                interaction_mask=torch.zeros(2, dtype=torch.bool),
                affinity_mask=torch.zeros(2, dtype=torch.bool),
            )
        result = compute_ecloudflow_loss(_prediction(), targets, LossConfig())
        names = sorted(result.diagnostics.supervised_counts)
        vector = torch.stack(
            [result.diagnostics.supervised_counts[name].long() for name in names]
        )
        gathered = [torch.zeros_like(vector) for _ in range(2)]
        dist.all_gather(gathered, vector)
        assert torch.equal(gathered[0], gathered[1])
        assert vector[names.index("flow_position")].item() == 2
        assert vector[names.index("ecloud_density")].item() == 2
        assert vector[names.index("interaction")].item() == 2
    finally:
        dist.destroy_process_group()


def test_gloo_diagnostics_counts_are_globally_consistent(tmp_path: Path) -> None:
    """Mutation caught: rank-local diagnostic counts disagreed on uneven batches."""
    mp.spawn(
        _distributed_diagnostic_counts_worker,
        args=(str(tmp_path / "gloo-counts"),),
        nprocs=2,
        join=True,
    )


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
