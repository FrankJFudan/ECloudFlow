import torch

from ecloudflow.core.types import MolecularState
from ecloudflow.sampling.solver import EulerSolver, HeunSolver


def _state(x: float = 1.0):
    return MolecularState(
        positions=torch.tensor([[x, 0.0, 0.0]]),
        atom_logits=torch.tensor([[1.0, 0.0]]),
        charge_logits=torch.tensor([[1.0, 0.0]]),
        halfedge_index=torch.empty((2, 0), dtype=torch.long),
        bond_logits=torch.empty((0, 2)),
        electron_latent=torch.zeros((1, 1)),
        node_batch=torch.zeros(1, dtype=torch.long),
        halfedge_batch=torch.empty(0, dtype=torch.long),
    )


def _field(state, time):
    return {"positions": state.positions}


def test_heun_is_more_accurate_than_euler_on_linear_field():
    generator = torch.Generator().manual_seed(5)
    euler = EulerSolver(8).integrate(_state(), _field, (), generator)
    generator = torch.Generator().manual_seed(5)
    heun = HeunSolver(8).integrate(_state(), _field, (), generator)
    exact = torch.exp(torch.tensor(1.0))
    assert abs(heun.final.positions[0, 0] - exact) < abs(euler.final.positions[0, 0] - exact)
    assert euler.nfe == 8 and heun.nfe == 16

