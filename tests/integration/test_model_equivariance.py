"""Integration tests for complete-model proper SE(3) behavior."""

from dataclasses import replace

import torch
from e3nn import o3

from ecloudflow.config import ModelConfig
from ecloudflow.core.types import GenerationCondition, MolecularState, PocketGraph
from ecloudflow.models import ECloudFlowModel


def _rotation() -> torch.Tensor:
    generator = torch.Generator().manual_seed(31)
    matrix = torch.randn(3, 3, generator=generator)
    q, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _batch() -> tuple[MolecularState, GenerationCondition]:
    generator = torch.Generator().manual_seed(32)
    positions = torch.randn(5, 3, generator=generator)
    state = MolecularState(
        positions=positions,
        atom_logits=torch.randn(5, 6, generator=generator),
        charge_logits=torch.randn(5, 4, generator=generator),
        halfedge_index=torch.tensor([[0, 0, 1, 3], [1, 2, 2, 4]]),
        bond_logits=torch.randn(4, 5, generator=generator),
        electron_latent=torch.randn(5, 48, generator=generator),
        node_batch=torch.zeros(5, dtype=torch.long),
        halfedge_batch=torch.zeros(4, dtype=torch.long),
    )
    pocket = PocketGraph(
        positions=torch.randn(6, 3, generator=generator),
        features=torch.randn(6, 11, generator=generator),
        batch=torch.zeros(6, dtype=torch.long),
    )
    return state, GenerationCondition(pocket=pocket)


def test_model_is_se3_equivariant_and_scalar_outputs_are_invariant() -> None:
    """Mutation caught: absolute geometry or component-wise mixing breaks SE(3)."""
    torch.manual_seed(33)
    model = ECloudFlowModel.from_config(
        ModelConfig(name="tiny", scalar_dim=16, vector_dim=4, num_blocks=2, lmax=2),
        electron_vector_dim=8,
        max_atoms=12,
    ).eval()
    state, condition = _batch()
    prediction = model(state, torch.tensor([0.4]), condition)
    rotation = _rotation()
    translation = torch.tensor([2.0, -1.0, 0.5])
    irreps = o3.Irreps("19x0e + 8x1o + 1x2e")
    representation = irreps.D_from_matrix(rotation)
    transformed_state = state.replace(
        positions=state.positions @ rotation.T + translation,
        electron_latent=state.electron_latent @ representation.T,
    )
    transformed_condition = replace(
        condition,
        pocket=replace(
            condition.pocket,
            positions=condition.pocket.positions @ rotation.T + translation,
        ),
    )
    moved = model(transformed_state, torch.tensor([0.4]), transformed_condition)

    assert torch.allclose(
        moved.position_velocity,
        prediction.position_velocity @ rotation.T,
        atol=5e-4,
        rtol=5e-4,
    )
    assert torch.allclose(
        moved.position_score,
        prediction.position_score @ rotation.T,
        atol=5e-4,
        rtol=5e-4,
    )
    assert torch.allclose(
        moved.electron_velocity,
        prediction.electron_velocity @ representation.T,
        atol=5e-4,
        rtol=5e-4,
    )
    assert torch.allclose(
        moved.electron_score,
        prediction.electron_score @ representation.T,
        atol=5e-4,
        rtol=5e-4,
    )
    for moved_scalar, original in (
        (moved.atom_logits, prediction.atom_logits),
        (moved.charge_logits, prediction.charge_logits),
        (moved.bond_logits, prediction.bond_logits),
        (moved.count_logits, prediction.count_logits),
        (moved.affinity, prediction.affinity),
        (moved.interaction_logits, prediction.interaction_logits),
    ):
        assert torch.allclose(moved_scalar, original, atol=5e-4, rtol=5e-4)
