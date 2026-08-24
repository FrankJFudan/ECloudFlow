import torch

from ecloudflow.core.types import MolecularState
from ecloudflow.sampling.corrector import ScoreCorrector


def _state():
    return MolecularState(
        positions=torch.tensor([[1.0, 0.0, 0.0]]),
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


def test_score_corrector_is_deterministic_with_caller_generator():
    first_generator = torch.Generator().manual_seed(7)
    first = ScoreCorrector(snr=0.1, steps=2).correct(
        _state(), _field, generator=first_generator
    )
    second_generator = torch.Generator().manual_seed(7)
    second = ScoreCorrector(snr=0.1, steps=2).correct(
        _state(), _field, generator=second_generator
    )
    assert torch.equal(first.final.positions, second.final.positions)
    assert first.nfe == 2


def test_score_corrector_dispatches_three_two_and_one_argument_hooks():
    seen = []

    def hook_three(state, time, generator):
        seen.append("three")
        return state

    def hook_two(state, time):
        seen.append("two")
        return state

    def hook_one(state):
        seen.append("one")
        return state

    ScoreCorrector(steps=1).correct(
        _state(),
        _field,
        (hook_three, hook_two, hook_one),
        torch.Generator().manual_seed(2),
    )
    assert seen == ["three", "two", "one"]


def test_score_corrector_rejects_invalid_step_overrides():
    corrector = ScoreCorrector(steps=1)
    for invalid in (-1, True, 1.5):
        try:
            corrector.correct(
                _state(), _field, generator=torch.Generator(), steps=invalid
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid correction step override was accepted")
