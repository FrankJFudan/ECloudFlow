"""Behavior tests for the joint pocket-conditioned ligand model."""

from dataclasses import replace

import pytest
import torch

from ecloudflow.config import ModelConfig
from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import (
    ElectronField,
    FragmentCondition,
    GenerationCondition,
    MolecularState,
    PocketGraph,
)
from ecloudflow.models import ECloudFlowModel


def _state(*, requires_grad: bool = False, empty: bool = False) -> MolecularState:
    node_count = 0 if empty else 4
    positions = (
        torch.tensor(
            [[-0.8, 0.1, 0.2], [0.3, -0.7, 0.5], [1.0, 0.6, -0.4], [-0.2, 1.1, 0.9]],
            dtype=torch.float32,
        )[:node_count]
        .clone()
        .requires_grad_(requires_grad)
    )
    if empty:
        halfedge_index = torch.empty((2, 0), dtype=torch.long)
        node_batch = torch.empty(0, dtype=torch.long)
        halfedge_batch = torch.empty(0, dtype=torch.long)
    else:
        halfedge_index = torch.tensor([[0, 2], [1, 3]])
        node_batch = torch.tensor([0, 0, 1, 1])
        halfedge_batch = torch.tensor([0, 1])
    generator = torch.Generator().manual_seed(8)

    def values(rows: int, columns: int) -> torch.Tensor:
        return torch.randn(rows, columns, generator=generator).requires_grad_(
            requires_grad
        )

    return MolecularState(
        positions=positions,
        atom_logits=values(node_count, 6),
        charge_logits=values(node_count, 4),
        halfedge_index=halfedge_index,
        bond_logits=values(halfedge_index.shape[1], 5),
        electron_latent=values(node_count, 48),
        node_batch=node_batch,
        halfedge_batch=halfedge_batch,
    )


def _condition(
    state: MolecularState, *, task_id: str | None = None
) -> GenerationCondition:
    pocket = PocketGraph(
        positions=torch.tensor(
            [[-1.5, 0.0, 0.1], [0.2, 1.3, -0.2], [1.4, 0.1, 0.6], [0.5, -1.1, -0.7]]
        ),
        features=torch.arange(28, dtype=torch.float32).reshape(4, 7) / 10,
        batch=torch.tensor([0, 0, 1, 1]),
    )
    fragment = None
    if task_id is not None:
        frame = CoordinateFrame(origin=torch.zeros(3))
        pocket = replace(pocket, frame=frame)
        fixed = torch.tensor([True, False, True, True])
        fragment = FragmentCondition.from_atom_mask(
            fixed, state.replace(frame=frame), task_id=task_id
        )
    return GenerationCondition(pocket=pocket, fragment=fragment)


def _model() -> ECloudFlowModel:
    torch.manual_seed(9)
    return ECloudFlowModel.from_config(
        ModelConfig(name="tiny", scalar_dim=16, vector_dim=4, num_blocks=2, lmax=2),
        max_atoms=8,
        electron_vector_dim=8,
        atom_classes=6,
        charge_classes=4,
        bond_classes=5,
    )


def _rich_condition(state: MolecularState) -> GenerationCondition:
    """Build explicit field/property/interaction conditioning for ablations."""
    base = _condition(state)
    frame = CoordinateFrame(origin=torch.zeros(3))
    pocket = replace(base.pocket, frame=frame)
    field = ElectronField(
        positions=torch.tensor(
            [[-1.0, 0.2, 0.3], [0.6, 0.4, -0.2], [1.1, -0.8, 0.5], [0.2, 1.0, -0.7]]
        ),
        values=torch.tensor([[0.2, 1.0], [0.7, -0.3], [1.2, 0.4], [-0.5, 0.8]]),
        mask=torch.tensor([True, True, True, True]),
        batch=torch.tensor([0, 0, 1, 1]),
        frame=frame,
    )
    return GenerationCondition(
        pocket=pocket,
        pocket_field=field,
        property_targets={"affinity": torch.tensor([1.0, 2.0])},
        interaction_targets=torch.tensor([[0.2, 0.8], [0.5, -0.1]]),
    )


def test_forward_outputs_flattened_heads_and_applies_fragment_masks() -> None:
    """Mutation caught: applying masks after return leaks changes into fixed fields."""
    model = _model()
    state = _state()
    condition = _condition(state, task_id="grow")

    prediction = model(state, torch.tensor([0.2, 0.8]), condition)

    assert prediction.position_velocity.shape == (4, 3)
    assert prediction.position_score.shape == (4, 3)
    assert prediction.electron_velocity.shape == (4, 48)
    assert prediction.electron_score.shape == (4, 48)
    assert prediction.atom_logits.shape == (4, 6)
    assert prediction.charge_logits.shape == (4, 4)
    assert prediction.bond_logits.shape == (2, 5)
    assert prediction.count_logits.shape == (2, 9)
    assert prediction.affinity.shape == (2,)
    assert prediction.interaction_logits.shape == (2,)
    fixed = condition.fragment.fixed_atom_mask
    assert torch.equal(prediction.position_velocity[fixed], torch.zeros(3, 3))
    assert torch.equal(prediction.position_score[fixed], torch.zeros(3, 3))
    assert torch.equal(prediction.atom_logits[fixed], state.atom_logits[fixed])
    assert torch.equal(prediction.charge_logits[fixed], state.charge_logits[fixed])
    assert torch.equal(
        prediction.bond_logits[condition.fragment.fixed_bond_mask],
        state.bond_logits[condition.fragment.fixed_bond_mask],
    )
    probabilities = prediction.count_logits.exp()
    assert torch.equal(probabilities[0, :1], torch.zeros(1))
    assert torch.equal(probabilities[1, :2], torch.zeros(2))


def test_categorical_heads_change_probabilities_per_class_and_receive_gradients() -> (
    None
):
    """Mutation caught: one broadcast scalar leaves every categorical softmax unchanged."""
    model = _model()
    state = _state()
    prediction = model(state, torch.tensor([0.2, 0.8]), _condition(state))

    assert not torch.allclose(
        prediction.atom_logits.softmax(-1), state.atom_logits.softmax(-1)
    )
    assert not torch.allclose(
        prediction.charge_logits.softmax(-1), state.charge_logits.softmax(-1)
    )
    assert not torch.allclose(
        prediction.bond_logits.softmax(-1), state.bond_logits.softmax(-1)
    )
    class_weights = torch.arange(1, 7, dtype=state.positions.dtype)
    loss = (prediction.atom_logits * class_weights).sum()
    loss = (
        loss
        + (
            prediction.charge_logits * torch.arange(1, 5, dtype=state.positions.dtype)
        ).sum()
    )
    loss = (
        loss
        + (
            prediction.bond_logits * torch.arange(1, 6, dtype=state.positions.dtype)
        ).sum()
    )
    loss.backward()

    for head in (model.atom_head, model.charge_head, model.bond_head):
        final_weight = head.network[-1].weight
        assert final_weight.grad is not None
        assert torch.all(final_weight.grad.abs().sum(dim=1) > 0)


def test_model_rejects_state_vocabulary_width_mismatch() -> None:
    """Mutation caught: dynamic broadcasting silently accepts a wrong vocabulary."""
    state = _state()
    malformed = state.replace(atom_logits=state.atom_logits[:, :-1])

    with pytest.raises(ValueError, match="atom class"):
        _model()(malformed, torch.tensor([0.2, 0.8]), _condition(malformed))


@pytest.mark.parametrize(
    ("change", "label"),
    [
        ("field", "pocket field"),
        ("interaction", "interaction target"),
        ("property_name", "property name"),
        ("property_value", "property value"),
    ],
)
def test_each_explicit_condition_signal_changes_predictions(
    change: str, label: str
) -> None:
    """Mutation caught: dropping a named condition makes its ablation an alias."""
    model = _model().eval()
    state = _state()
    condition = _rich_condition(state)
    if change == "field":
        changed = replace(
            condition,
            pocket_field=replace(
                condition.pocket_field,
                values=condition.pocket_field.values
                + torch.tensor([[0.3, -0.1], [0.1, 0.2], [-0.2, 0.4], [0.5, 0.1]]),
            ),
        )
    elif change == "interaction":
        changed = replace(
            condition,
            interaction_targets=condition.interaction_targets
            + torch.tensor([[0.4, -0.2], [0.1, 0.3]]),
        )
    elif change == "property_name":
        changed = replace(
            condition,
            property_targets={"logp": torch.tensor([1.0, 2.0])},
        )
    else:
        changed = replace(
            condition,
            property_targets={"affinity": torch.tensor([2.0, 3.0])},
        )

    original = model(state, torch.tensor([0.3, 0.7]), condition)
    ablated = model(state, torch.tensor([0.3, 0.7]), changed)

    assert not torch.allclose(original.affinity, ablated.affinity), label


def test_condition_tensors_receive_finite_nonzero_gradients() -> None:
    """Mutation caught: detaching a condition encoder hides its training signal."""
    model = _model()
    state = _state()
    condition = _rich_condition(state)
    field_positions = condition.pocket_field.positions.detach().requires_grad_(True)
    field_values = condition.pocket_field.values.detach().requires_grad_(True)
    property_values = torch.tensor([1.0, 2.0], requires_grad=True)
    interaction = condition.interaction_targets.detach().requires_grad_(True)
    condition = replace(
        condition,
        pocket_field=replace(
            condition.pocket_field,
            positions=field_positions,
            values=field_values,
        ),
        property_targets={"affinity": property_values},
        interaction_targets=interaction,
    )

    prediction = model(state, torch.tensor([0.3, 0.7]), condition)
    loss = (
        prediction.position_velocity.square().sum()
        + prediction.atom_logits.square().sum()
        + prediction.affinity.square().sum()
    )
    loss.backward()

    for tensor in (field_positions, field_values, property_values, interaction):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.any(tensor.grad != 0)


@pytest.mark.parametrize("target_value", [-1.0, 0.0, 2.5])
def test_property_identity_is_independent_of_any_finite_value(
    target_value: float,
) -> None:
    """Mutation caught: a value-dependent identity aliases names at special values."""
    model = _model().eval()
    state = _state()
    condition = _rich_condition(state)
    first_value = torch.full((2,), target_value, requires_grad=True)
    second_value = torch.full((2,), target_value, requires_grad=True)
    first = replace(condition, property_targets={"affinity": first_value})
    second = replace(condition, property_targets={"logp": second_value})

    first_prediction = model(state, torch.tensor([0.3, 0.7]), first)
    second_prediction = model(state, torch.tensor([0.3, 0.7]), second)

    assert not torch.allclose(first_prediction.affinity, second_prediction.affinity)
    gradients = torch.autograd.grad(
        first_prediction.affinity.sum() + second_prediction.affinity.sum(),
        (first_value, second_value),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.any(gradient != 0)


@pytest.mark.parametrize("vector_dim", [1, 2])
def test_constructor_rejects_vector_width_without_chirality_capacity(
    vector_dim: int,
) -> None:
    """Mutation caught: fewer than three vectors makes the triple product zero."""
    with pytest.raises(ValueError, match="vector_dim must be at least 3"):
        ECloudFlowModel(
            scalar_dim=16,
            vector_dim=vector_dim,
            num_blocks=2,
            lmax=2,
        )


@pytest.mark.parametrize("vector_dim", [1, 2])
def test_from_config_rejects_vector_width_without_chirality_capacity(
    vector_dim: int,
) -> None:
    """Mutation caught: from_config bypasses the public chirality-width contract."""
    config = ModelConfig.model_construct(
        name="tiny", scalar_dim=16, vector_dim=vector_dim, num_blocks=2, lmax=2
    )

    with pytest.raises(ValueError, match="vector_dim must be at least 3"):
        ECloudFlowModel.from_config(config)


@pytest.mark.parametrize(
    ("condition", "match"),
    [
        (
            lambda state: replace(
                _rich_condition(state), property_targets={"bad": torch.ones(3)}
            ),
            r"shape \[B",
        ),
        (
            lambda state: replace(
                _rich_condition(state), interaction_targets=torch.ones(3, 2)
            ),
            r"shape \[B",
        ),
        (
            lambda state: replace(
                _rich_condition(state),
                interaction_targets=torch.ones(2, 2, dtype=torch.float64),
            ),
            "dtype and device",
        ),
    ],
)
def test_model_rejects_malformed_condition_batch_shapes(condition, match: str) -> None:
    """Mutation caught: condition broadcasting mixes targets between complexes."""
    state = _state()

    with pytest.raises(ValueError, match=match):
        _model()(state, torch.tensor([0.2, 0.8]), condition(state))


def test_null_branch_is_independent_of_all_pocket_and_target_content() -> None:
    """Mutation caught: null cross edges leak pocket geometry into unconditional CFG."""
    model = _model().eval()
    state = _state()
    first = _rich_condition(state)
    frame = first.pocket.frame
    assert frame is not None
    fixed = torch.tensor([True, False, True, True])
    reference = state.replace(frame=frame)
    first = replace(
        first,
        fragment=FragmentCondition.from_atom_mask(fixed, reference, task_id="grow"),
    )
    second = replace(
        first,
        pocket=replace(
            first.pocket,
            positions=first.pocket.positions * 3.0 + torch.tensor([4.0, -2.0, 1.0]),
            features=first.pocket.features.flip(-1) + 5.0,
        ),
        pocket_field=replace(
            first.pocket_field,
            positions=first.pocket_field.positions * -2.0,
            values=first.pocket_field.values + 7.0,
        ),
        property_targets={"different": torch.tensor([-5.0, 9.0])},
        interaction_targets=torch.tensor([[8.0, -3.0], [4.0, 6.0]]),
        fragment=FragmentCondition.from_atom_mask(fixed, reference, task_id="link"),
    )
    first_null = model.encode_pocket(first.pocket, use_null=True)
    second_null = model.encode_pocket(second.pocket, use_null=True)

    prediction = model(state, torch.tensor([0.4, 0.6]), first, first_null)
    changed = model(state, torch.tensor([0.4, 0.6]), second, second_null)

    for left, right in (
        (prediction.position_velocity, changed.position_velocity),
        (prediction.atom_logits, changed.atom_logits),
        (prediction.bond_logits, changed.bond_logits),
        (prediction.count_logits, changed.count_logits),
        (prediction.affinity, changed.affinity),
        (prediction.interaction_logits, changed.interaction_logits),
    ):
        assert torch.equal(left, right)


def test_ligand_backbone_retains_and_uses_all_vector_channels_and_time_films() -> None:
    """Mutation caught: collapsing vectors or one-time injection leaves channels/blocks unused."""
    model = _model()
    state = _state()
    captured = []
    handle = model.backbone.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.vectors)
    )
    prediction = model(state, torch.tensor([0.2, 0.8]), _condition(state))
    handle.remove()

    assert captured[0].shape == (4, 4, 3)
    assert torch.all(captured[0].square().sum(dim=(0, 2)) > 0)
    loss = (
        prediction.atom_logits.square().sum()
        + prediction.position_velocity.square().sum()
    )
    loss.backward()
    assert len(model.backbone.time_films) == 2
    assert all(
        film.weight.grad is not None and torch.any(film.weight.grad != 0)
        for film in model.backbone.time_films
    )


def test_cached_pocket_encoding_is_reused_and_wrong_cache_is_rejected() -> None:
    """Mutation caught: accepting a cache from another pocket cross-conditions a sample."""
    model = _model()
    state = _state()
    condition = _condition(state)
    encoded = model.encode_pocket(condition.pocket)

    first = model(state, torch.tensor([0.2, 0.3]), condition, encoded)
    second = model(state, torch.tensor([0.8, 0.9]), condition, encoded)
    assert encoded.cache_key == first.pocket_cache_key == second.pocket_cache_key

    wrong_pocket = replace(
        condition.pocket,
        positions=condition.pocket.positions + torch.tensor([0.1, 0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="cached pocket encoding"):
        model(
            state,
            torch.tensor([0.2, 0.3]),
            replace(condition, pocket=wrong_pocket),
            encoded,
        )


def test_cached_pocket_encoding_from_another_model_is_rejected() -> None:
    """Mutation caught: architecture-only cache keys accept stale learned features."""
    state = _state()
    condition = _condition(state)
    source_model = _model()
    target_model = _model()
    with torch.no_grad():
        next(target_model.parameters()).add_(0.5)
    encoded = source_model.encode_pocket(condition.pocket)

    with pytest.raises(ValueError, match="cached pocket encoding"):
        target_model(state, torch.tensor([0.2, 0.3]), condition, encoded)


def test_ligand_radius_geometry_conditions_node_predictions_without_pocket_edges() -> (
    None
):
    """Mutation caught: deleting ligand radius messages makes nonbonded geometry invisible."""
    model = _model().eval()
    state = _state()
    condition = _condition(state)
    far_condition = replace(
        condition,
        pocket=replace(
            condition.pocket,
            positions=condition.pocket.positions + torch.tensor([100.0, 0.0, 0.0]),
        ),
    )
    moved_state = state.replace(
        positions=state.positions
        + torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, -0.1, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
    )

    original = model(state, torch.tensor([0.3, 0.7]), far_condition)
    moved = model(moved_state, torch.tensor([0.3, 0.7]), far_condition)

    assert not torch.allclose(moved.atom_logits, original.atom_logits)


def test_null_conditioning_and_fragment_task_id_change_predictions() -> None:
    """Mutation caught: discarding CFG/task embeddings makes conditional branches aliases."""
    model = _model().eval()
    state = _state()
    grow = _condition(state, task_id="grow")
    link = replace(grow, fragment=replace(grow.fragment, task_id="link"))
    conditioned = model(state, torch.tensor([0.4, 0.6]), grow)
    null_encoding = model.encode_pocket(grow.pocket, use_null=True)
    unconditional = model(state, torch.tensor([0.4, 0.6]), grow, null_encoding)
    linked = model(state, torch.tensor([0.4, 0.6]), link)

    assert not torch.allclose(conditioned.affinity, unconditional.affinity)
    assert not torch.allclose(conditioned.interaction_logits, linked.interaction_logits)


def test_forward_supports_empty_ligand_and_keeps_per_complex_count_outputs() -> None:
    """Mutation caught: deriving batch size only from ligand nodes drops empty complexes."""
    model = _model()
    state = _state(empty=True)
    condition = _condition(state)

    prediction = model(state, torch.tensor([0.1, 0.9]), condition)

    assert prediction.position_velocity.shape == (0, 3)
    assert prediction.bond_logits.shape == (0, 5)
    assert prediction.count_logits.shape == (2, 9)
    assert torch.isfinite(prediction.count_logits[:, 0:]).all()


def test_forward_has_finite_gradients_through_all_joint_inputs() -> None:
    """Mutation caught: detach/no-grad in one joint stream removes its gradient."""
    model = _model()
    state = _state(requires_grad=True)
    condition = _condition(state)

    prediction = model(state, torch.tensor([0.25, 0.75]), condition)
    loss = (
        prediction.position_velocity.square().sum()
        + prediction.position_score.square().sum()
        + prediction.electron_velocity.square().sum()
        + prediction.electron_score.square().sum()
        + prediction.atom_logits.square().sum()
        + prediction.charge_logits.square().sum()
        + prediction.bond_logits.square().sum()
        + prediction.count_logits.square().sum()
        + prediction.affinity.square().sum()
        + prediction.interaction_logits.square().sum()
    )
    loss.backward()

    for tensor in (
        state.positions,
        state.atom_logits,
        state.charge_logits,
        state.bond_logits,
        state.electron_latent,
    ):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert parameter_gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in parameter_gradients
    )


def test_zero_invariant_features_have_finite_backward_derivatives() -> None:
    """Mutation caught: an unguarded RMS square root has an infinite derivative at zero."""
    model = _model()
    state = _state()
    zero_inputs = {
        name: torch.zeros_like(getattr(state, name), requires_grad=True)
        for name in ("atom_logits", "charge_logits", "bond_logits", "electron_latent")
    }
    state = state.replace(**zero_inputs)
    condition = _condition(state)
    pocket_features = torch.zeros_like(condition.pocket.features, requires_grad=True)
    condition = replace(
        condition, pocket=replace(condition.pocket, features=pocket_features)
    )

    prediction = model(state, torch.tensor([0.25, 0.75]), condition)
    loss = sum(
        value.square().sum()
        for value in (
            prediction.position_velocity,
            prediction.electron_velocity,
            prediction.atom_logits,
            prediction.charge_logits,
            prediction.bond_logits,
            prediction.count_logits,
        )
    )
    loss.backward()

    for tensor in (*zero_inputs.values(), pocket_features):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("bad_time", [torch.tensor([0.5]), torch.tensor([0.2, 1.1])])
def test_forward_rejects_malformed_or_out_of_range_times(
    bad_time: torch.Tensor,
) -> None:
    """Mutation caught: implicit time broadcasting mixes complex trajectories."""
    with pytest.raises(ValueError, match="time"):
        _model()(_state(), bad_time, _condition(_state()))


def test_forward_rejects_wrong_electron_irrep_width() -> None:
    """Mutation caught: treating all electron channels as scalars accepts bad layouts."""
    state = _state()
    malformed = state.replace(electron_latent=state.electron_latent[:, :-1])

    with pytest.raises(ValueError, match="electron latent"):
        _model()(malformed, torch.tensor([0.2, 0.8]), _condition(malformed))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_forward_runs_on_cuda_without_rank_local_or_cpu_state() -> None:
    """Mutation caught: creating CPU tensors or calling a fixed CUDA device breaks placement."""
    device = torch.device("cuda")
    model = _model().to(device)
    state = _state()
    state = state.replace(
        **{
            name: getattr(state, name).to(device)
            for name in (
                "positions",
                "atom_logits",
                "charge_logits",
                "halfedge_index",
                "bond_logits",
                "electron_latent",
                "node_batch",
                "halfedge_batch",
            )
        }
    )
    condition = _condition(state)
    condition = replace(
        condition,
        pocket=replace(
            condition.pocket,
            positions=condition.pocket.positions.to(device),
            features=condition.pocket.features.to(device),
            batch=condition.pocket.batch.to(device),
        ),
    )

    prediction = model(state, torch.tensor([0.2, 0.8], device=device), condition)

    assert prediction.position_velocity.device.type == "cuda"
    assert prediction.count_logits.device.type == "cuda"
