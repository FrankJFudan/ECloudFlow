import torch

from ecloudflow.sampling.priors import CavityAwarePrior


class _Cavity:
    center = (0.0, 0.0, 0.0)
    radius = 2.0

    def contains(self, points):
        return points.square().sum(-1) <= 4.0


class _Condition:
    cavity = _Cavity()


def test_cavity_prior_places_atoms_inside_supported_volume():
    prior = CavityAwarePrior(seed=11)
    state = prior.sample(_Condition(), 24)
    assert bool(_Condition.cavity.contains(state.positions).all())
    assert torch.allclose(state.atom_logits.sum(-1), torch.ones(24))

