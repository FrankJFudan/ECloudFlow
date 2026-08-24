import torch

from ecloudflow.sampling.priors import CavityAwarePrior


class _Cavity:
    center = (0.0, 0.0, 0.0)
    radius = 2.0

    def contains(self, points):
        return points.square().sum(-1) <= 4.0


class _Condition:
    cavity = _Cavity()


class _ImpossibleCavity:
    center = (0.0, 0.0, 0.0)
    radius = 1.0

    def contains(self, points):
        return torch.zeros(points.shape[0], dtype=torch.bool)


class _ImpossibleCondition:
    cavity = _ImpossibleCavity()


def test_cavity_prior_places_atoms_inside_supported_volume():
    prior = CavityAwarePrior(seed=11)
    state = prior.sample(_Condition(), 24)
    assert bool(_Condition.cavity.contains(state.positions).all())
    assert torch.allclose(state.atom_logits.sum(-1), torch.ones(24))


def test_cavity_prior_raises_after_bounded_rejection_exhaustion():
    prior = CavityAwarePrior(seed=11)
    try:
        prior.sample(_ImpossibleCondition(), 2)
    except ValueError as exc:
        assert "bounded attempts" in str(exc)
    else:
        raise AssertionError("unsupported cavity draws must fail explicitly")
