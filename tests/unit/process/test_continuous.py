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


def test_score_threshold_excludes_unstable_times_and_keeps_exact_nearby_score() -> None:
    """Small nonzero gamma is excluded instead of silently denominator-clamped."""
    path = ContinuousPath(LinearBridge(), numerical_epsilon=1e-6)
    x0 = torch.zeros(2, 3)
    x1 = torch.ones_like(x0)
    excluded = path.sample(x0, x1, torch.tensor(1e-8), generator=_generator(2))
    with pytest.raises(ValueError, match="below numerical_epsilon"):
        path.targets(x0, x1, excluded)

    time = torch.tensor(2e-6)
    accepted = path.sample(x0, x1, time, generator=_generator(2))
    _, score = path.targets(x0, x1, accepted)
    expected = -accepted.noise / path.schedule.noise_scale(time)
    assert torch.equal(score, expected)


def test_float64_score_boundary_is_not_rounded_to_float32_safety() -> None:
    """Float64 threshold decisions retain the returned time's precision."""
    path = ContinuousPath(LinearBridge(), numerical_epsilon=1e-6)
    x0 = torch.zeros(1, 3, dtype=torch.float64)
    x1 = torch.ones_like(x0)
    boundary = torch.tensor(1.00000098e-6, dtype=torch.float64)
    assert path.schedule.noise_scale(boundary) < path.numerical_epsilon
    sample = path.sample(x0, x1, boundary, generator=_generator(4))
    with pytest.raises(ValueError, match="below numerical_epsilon"):
        path.targets(x0, x1, sample)

    accepted_time = torch.tensor(1.000002e-6, dtype=torch.float64)
    accepted = path.sample(x0, x1, accepted_time, generator=_generator(4))
    _, score = path.targets(x0, x1, accepted)
    assert torch.equal(
        score, -accepted.noise / path.schedule.noise_scale(accepted_time)
    )


def test_reduced_precision_retains_float32_epsilon_and_float32_targets() -> None:
    """BF16 targets use the original float32 epsilon from the sampled coupling."""
    path = ContinuousPath(LinearBridge(interior_noise=0.2))
    x0 = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
    x1 = torch.ones(2, 3, dtype=torch.bfloat16, requires_grad=True)
    time = torch.tensor(0.4)
    sample = path.sample(x0, x1, time, generator=_generator(10))
    velocity, score = path.targets(x0, x1, sample)
    assert sample.value.dtype == torch.bfloat16
    assert sample.noise.dtype == velocity.dtype == score.dtype == torch.float32
    expected_velocity = (x1.float() - x0.float()) + 0.2 * (
        1.0 - 2.0 * time
    ) * sample.noise
    expected_score = -sample.noise / (0.2 * time * (1.0 - time))
    assert torch.equal(velocity, expected_velocity)
    assert torch.equal(score, expected_score)
    (velocity.square().mean() + score.square().mean()).backward()
    assert x0.grad is not None and torch.isfinite(x0.grad).all()
    assert x1.grad is not None and torch.isfinite(x1.grad).all()


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
    assert bool((path.schedule.noise_scale(times) >= path.numerical_epsilon).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_score_time_samples_pass_targets_in_their_effective_dtype(
    dtype: torch.dtype,
) -> None:
    """Every default score-mode time is valid for a same-dtype endpoint path."""
    path = ContinuousPath(LinearBridge())
    times = path.sample_times(17, dtype=dtype, generator=_generator(21))
    x0 = torch.zeros(17, 2, dtype=dtype)
    x1 = torch.ones_like(x0)
    sample = path.sample(x0, x1, times, generator=_generator(22))
    velocity, score = path.targets(x0, x1, sample)
    assert times.dtype == dtype
    assert torch.isfinite(velocity).all() and torch.isfinite(score).all()


@pytest.mark.parametrize("shape", [1, 3, (1, 3)])
def test_odd_antithetic_score_times_are_deterministic_and_safe(
    shape: int | tuple[int, int],
) -> None:
    """Odd totals retain paired values plus one independently safe final value."""
    path = ContinuousPath(LinearBridge())
    times = path.sample_times(shape, antithetic=True, generator=_generator(23))
    repeated = path.sample_times(shape, antithetic=True, generator=_generator(23))
    flattened = times.reshape(-1)
    pairs = flattened.numel() // 2
    assert torch.equal(times, repeated)
    assert bool((path.schedule.noise_scale(flattened) >= path.numerical_epsilon).all())
    if pairs:
        assert torch.allclose(
            flattened[:pairs] + flattened[pairs : 2 * pairs], torch.ones(pairs)
        )
    x0 = torch.zeros(flattened.numel(), 2)
    x1 = torch.ones_like(x0)
    _, score = path.targets(x0, x1, path.sample(x0, x1, flattened))
    assert torch.isfinite(score).all()


def test_sample_times_rejects_non_shape_objects_with_typed_error() -> None:
    """Shape type errors are explicit rather than leaked tuple-conversion errors."""
    with pytest.raises(TypeError, match="integer or a sequence"):
        ContinuousPath(LinearBridge()).sample_times(1.5)  # type: ignore[arg-type]


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


@pytest.mark.parametrize("masked", [False, True])
def test_empty_velocity_loss_is_differentiable_finite_zero(masked: bool) -> None:
    """Empty arbitrary leading dimensions never reduce to a NaN mean."""
    path = ContinuousPath(LinearBridge())
    prediction = torch.empty(0, 3, requires_grad=True)
    target = torch.empty_like(prediction)
    loss = path.velocity_loss(
        prediction,
        target,
        edit_mask=torch.empty(0, dtype=torch.bool) if masked else None,
    )
    assert loss.dtype == torch.float32 and loss.item() == 0.0
    loss.backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad, torch.zeros_like(prediction.grad))
