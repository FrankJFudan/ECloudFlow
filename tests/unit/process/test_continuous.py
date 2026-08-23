"""Tests for continuous stochastic-interpolant samples and targets."""

from __future__ import annotations

import pytest
import torch

from ecloudflow.process import ContinuousPath, LinearBridge


def _generator(seed: int) -> torch.Generator:
    """Return a deterministic CPU generator."""
    return torch.Generator().manual_seed(seed)


def test_continuous_path_endpoints_and_targets() -> None:
    """The sampled bridge has exact endpoints and finite interior targets."""
    path = ContinuousPath(LinearBridge(interior_noise=0.2))
    x0 = torch.zeros(4, 3)
    x1 = torch.ones(4, 3)
    assert torch.allclose(path.mean(x0, x1, torch.tensor(0.0)), x0)
    assert torch.allclose(path.mean(x0, x1, torch.tensor(1.0)), x1)
    sample = path.sample(x0, x1, torch.tensor(0.4), generator=_generator(4))
    velocity, score = path.targets(x0, x1, sample)
    assert velocity.shape == score.shape == x0.shape
    assert torch.isfinite(velocity).all() and torch.isfinite(score).all()


def test_velocity_is_the_exact_derivative_of_sampled_interpolation() -> None:
    """Velocity uses both analytic schedule derivatives under one noise draw."""
    path = ContinuousPath(LinearBridge(interior_noise=0.2))
    x0 = torch.randn(2, 3, dtype=torch.float64)
    x1 = torch.randn(2, 3, dtype=torch.float64)
    sample = path.sample(
        x0, x1, torch.tensor(0.37, dtype=torch.float64), generator=_generator(9)
    )
    velocity, _ = path.targets(x0, x1, sample)
    delta = torch.tensor(1e-5, dtype=torch.float64)
    before = (
        path.mean(x0, x1, sample.t - delta)
        + path.schedule.noise_scale(sample.t - delta) * sample.noise
    )
    after = (
        path.mean(x0, x1, sample.t + delta)
        + path.schedule.noise_scale(sample.t + delta) * sample.noise
    )
    assert torch.allclose(velocity, (after - before) / (2 * delta), atol=1e-5)


def test_continuous_path_rejects_endpoint_scores_and_preserves_inputs() -> None:
    """Endpoint scores are undefined and sampling never writes into endpoints."""
    path = ContinuousPath(LinearBridge())
    x0 = torch.zeros(2, 3)
    x1 = torch.ones(2, 3)
    sample = path.sample(x0, x1, torch.tensor(0.0), generator=_generator(1))
    with pytest.raises(ValueError, match="open interval"):
        path.targets(x0, x1, sample)
    assert torch.equal(x0, torch.zeros_like(x0))
    assert torch.equal(x1, torch.ones_like(x1))


def test_batch_times_and_antithetic_sampling_are_deterministic() -> None:
    """Batch time broadcasting and generator streams have stable semantics."""
    path = ContinuousPath(LinearBridge())
    x0 = torch.zeros(3, 2, 3)
    x1 = torch.ones_like(x0)
    sample = path.sample(
        x0, x1, torch.tensor([0.2, 0.5, 0.8]), generator=_generator(12)
    )
    assert sample.value.shape == x0.shape
    times = path.sample_times(4, generator=_generator(7), antithetic=True)
    assert torch.allclose(times[:2] + times[2:], torch.ones(2))
    assert torch.equal(
        times, path.sample_times(4, generator=_generator(7), antithetic=True)
    )


def test_bfloat16_reduction_accumulates_in_float32() -> None:
    """Masked losses remain finite and float32 under reduced precision."""
    path = ContinuousPath(LinearBridge())
    target = torch.ones(4, 3, dtype=torch.bfloat16)
    prediction = torch.zeros_like(target)
    loss = path.velocity_loss(
        prediction, target, edit_mask=torch.tensor([True, False, True, False])
    )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_targets_and_masked_loss_preserve_prediction_gradients() -> None:
    """Continuous targets and editable reductions retain autograd connections."""
    path = ContinuousPath(LinearBridge())
    x0 = torch.zeros(2, 3, requires_grad=True)
    x1 = torch.ones(2, 3, requires_grad=True)
    sample = path.sample(x0, x1, torch.tensor(0.4), generator=_generator(6))
    velocity, _ = path.targets(x0, x1, sample)
    prediction = torch.zeros_like(velocity, requires_grad=True)
    path.velocity_loss(
        prediction, velocity, edit_mask=torch.tensor([True, False])
    ).backward()
    assert prediction.grad is not None
    assert x0.grad is not None and x1.grad is not None
