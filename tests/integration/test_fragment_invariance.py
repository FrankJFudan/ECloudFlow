import torch

from ecloudflow.core import FragmentCondition, clamp_fragment
from ecloudflow.core.types import MolecularState
from ecloudflow.sampling.solver import EulerSolver


def _state(x: float = 1.0):
    return MolecularState(
        positions=torch.tensor([[x, 0.0, 0.0]]), atom_logits=torch.tensor([[1.0, 0.0]]),
        charge_logits=torch.tensor([[1.0, 0.0]]), halfedge_index=torch.empty((2, 0), dtype=torch.long),
        bond_logits=torch.empty((0, 2)), electron_latent=torch.zeros((1, 1)),
        node_batch=torch.zeros(1, dtype=torch.long), halfedge_batch=torch.empty(0, dtype=torch.long),
    )


def test_fixed_fragment_is_bitwise_equal_in_every_saved_frame():
    reference = _state(1.0)
    condition = FragmentCondition.from_atom_mask(torch.tensor([True]), reference)
    noisy = reference.replace(positions=torch.tensor([[3.0, 0.0, 0.0]]))
    trajectory = EulerSolver(4, save_every_step=True).integrate(
        noisy,
        lambda state, time: {"positions": torch.ones_like(state.positions)},
        (lambda state: clamp_fragment(state, condition),),
        torch.Generator().manual_seed(3),
    )
    for frame in trajectory.frames:
        assert torch.equal(frame.positions, reference.positions)
        assert torch.equal(frame.atom_logits, reference.atom_logits)
