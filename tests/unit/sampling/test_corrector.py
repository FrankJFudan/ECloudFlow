import torch

from ecloudflow.core.types import MolecularState
from ecloudflow.sampling.corrector import ScoreCorrector


def _state():
    return MolecularState(
        positions=torch.tensor([[1.0, 0.0, 0.0]]), atom_logits=torch.tensor([[1.0, 0.0]]),
        charge_logits=torch.tensor([[1.0, 0.0]]), halfedge_index=torch.empty((2, 0), dtype=torch.long),
        bond_logits=torch.empty((0, 2)), electron_latent=torch.zeros((1, 1)),
        node_batch=torch.zeros(1, dtype=torch.long), halfedge_batch=torch.empty(0, dtype=torch.long),
    )


def _field(state, time):
    return {"positions": state.positions}


def test_score_corrector_is_deterministic_with_caller_generator():
    first_generator = torch.Generator().manual_seed(7)
    first = ScoreCorrector(snr=0.1, steps=2).correct(_state(), _field, generator=first_generator)
    second_generator = torch.Generator().manual_seed(7)
    second = ScoreCorrector(snr=0.1, steps=2).correct(_state(), _field, generator=second_generator)
    assert torch.equal(first.final.positions, second.final.positions)
    assert first.nfe == 2
