"""Continuous stochastic-interpolant samples, targets, and masked losses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from ecloudflow.process.schedules import InterpolantSchedule, validate_time


@dataclass(frozen=True)
class ContinuousSample:
    """Store one noise-coupled sample of a continuous interpolant.

    ``value`` equals ``(1-a(t))x0 + a(t)x1 + g(t)noise``. ``noise`` has the
    endpoint tensor shape and is retained so conditional velocity and score
    targets use precisely the same random coupling used to create ``value``.
    For FP16/BF16 endpoints, ``noise`` is deliberately float32 rather than
    downcast, while ``value`` remains at endpoint dtype.
    """

    value: torch.Tensor
    time: torch.Tensor
    noise: torch.Tensor

    @property
    def t(self) -> torch.Tensor:
        """Return the schedule time under the concise conventional name."""
        return self.time


class ContinuousPath:
    """Sample an analytic Gaussian bridge from a prior at ``t=0`` to data at ``t=1``."""

    def __init__(
        self, schedule: InterpolantSchedule, *, numerical_epsilon: float = 1e-6
    ) -> None:
        """Initialize a continuous path with one analytic schedule.

        :param schedule: Prior-to-data schedule with analytic data/noise
            weights and derivatives.
        :param numerical_epsilon: Positive finite lower bound used only for a
            valid interior score denominator.
        :raises TypeError: If ``schedule`` does not implement the schedule API.
        :raises ValueError: If ``numerical_epsilon`` is not finite and positive.
        """
        if not isinstance(schedule, InterpolantSchedule):
            raise TypeError("schedule must be an InterpolantSchedule.")
        if not math.isfinite(numerical_epsilon) or numerical_epsilon <= 0.0:
            raise ValueError("numerical_epsilon must be finite and positive.")
        self.schedule = schedule
        self.numerical_epsilon = float(numerical_epsilon)

    def mean(
        self, x0: torch.Tensor, x1: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """Return the deterministic bridge mean ``(1-a(t))x0+a(t)x1``.

        :param x0: Prior endpoint with arbitrary non-empty or empty leading
            dimensions, floating dtype/device, and finite values.
        :param x1: Data endpoint with exactly ``x0`` shape, dtype, device, and
            finite values.
        :param time: Floating scalar or leading-prefix batch tensor on the
            endpoint device. A ``[B]`` time broadcasts to ``[B,...]``.
        :return: Mean tensor matching endpoint shape, dtype, and device.
        :rtype: torch.Tensor
        :raises ValueError: If endpoint/time shape, dtype, device, finiteness,
            or closed interval validation fails.

        The method neither mutates endpoints nor samples randomness. It is
        differentiable with respect to endpoints and time; BF16/FP16 arithmetic
        is accumulated in float32 before returning the endpoint dtype.
        """
        _validate_endpoints(x0, x1)
        expanded = _expand_time(time, x0)
        compute_dtype = _compute_dtype(x0.dtype)
        coefficient = self.schedule.data_weight(expanded.to(dtype=compute_dtype))
        mean = (1.0 - coefficient) * x0.to(compute_dtype) + coefficient * x1.to(
            compute_dtype
        )
        return mean.to(dtype=x0.dtype)

    def sample(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        time: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> ContinuousSample:
        """Draw one Gaussian-coupled continuous interpolant sample.

        :param x0: Finite floating prior tensor of arbitrary shape and device.
        :param x1: Finite floating data tensor matching ``x0`` exactly.
        :param time: Floating scalar or leading-prefix batch time on the same
            device in ``[0,1]``; zero and one produce exact endpoint means.
        :param generator: Optional torch generator controlling the sole normal
            draw. Reusing an equally seeded compatible generator is deterministic.
        :return: Immutable sample with endpoint-shaped ``value`` and ``noise``
            plus the unmodified caller time tensor. ``noise`` is float32 for
            FP16/BF16 endpoints and otherwise has endpoint dtype.
        :rtype: ContinuousSample
        :raises ValueError: If inputs have invalid shape, dtype, device,
            finiteness, or time range.
        :raises RuntimeError: If the supplied generator cannot generate on the
            endpoint device.

        No input is mutated and autograd remains connected to ``x0`` and ``x1``.
        The normal noise itself has no gradient. FP16/BF16 bridge arithmetic is
        performed in float32 and only ``value`` is cast back to endpoint dtype;
        retained epsilon is never downcast before target construction.
        """
        _validate_endpoints(x0, x1)
        expanded = _expand_time(time, x0)
        compute_dtype = _compute_dtype(x0.dtype)
        noise = torch.randn(
            x0.shape, dtype=compute_dtype, device=x0.device, generator=generator
        )
        expanded_compute = expanded.to(dtype=compute_dtype)
        coefficient = self.schedule.data_weight(expanded_compute)
        scale = self.schedule.noise_scale(expanded_compute)
        value = (
            (1.0 - coefficient) * x0.to(compute_dtype)
            + coefficient * x1.to(compute_dtype)
            + scale * noise
        )
        return ContinuousSample(value=value.to(dtype=x0.dtype), time=time, noise=noise)

    def targets(
        self, x0: torch.Tensor, x1: torch.Tensor, sample: ContinuousSample
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact conditional velocity and denoising-score targets.

        :param x0: Finite floating prior endpoint tensor of arbitrary shape.
        :param x1: Finite floating data endpoint matching ``x0`` shape, dtype,
            and device.
        :param sample: Coupled output from :meth:`sample`; its noise must match
            endpoint shape/device and uses float32 dtype for FP16/BF16
            endpoints (otherwise endpoint dtype); time must be strictly interior.
        :return: ``(velocity, score)`` tensors matching endpoint shape/device.
            Their dtype is float32 for FP16/BF16 endpoints and endpoint dtype
            otherwise. Velocity is ``a'(t)(x1-x0)+g'(t)epsilon`` and score is
            ``-epsilon/g(t)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        :raises TypeError: If ``sample`` is not a :class:`ContinuousSample`.
        :raises ValueError: If contracts disagree, time is not in the open
            interval, or ``abs(gamma(t))`` is below ``numerical_epsilon``.

        Endpoint scores are deliberately undefined and rejected. Interior times
        with a too-small nonzero noise scale are also excluded; accepted scores
        use the exact denominator, never a clamped surrogate. Inputs and the
        frozen sample are not mutated; endpoint gradients flow through velocity
        while score has the sampled-noise gradient semantics.
        """
        _validate_endpoints(x0, x1)
        if not isinstance(sample, ContinuousSample):
            raise TypeError("sample must be a ContinuousSample.")
        _validate_sample(sample, x0)
        expanded = _expand_time(sample.time, x0)
        if not bool(((expanded > 0.0) & (expanded < 1.0)).all()):
            raise ValueError("score targets require time in the open interval (0, 1).")
        compute_dtype = _compute_dtype(x0.dtype)
        time = expanded.to(dtype=compute_dtype)
        noise = sample.noise.to(dtype=compute_dtype)
        velocity = (
            self.schedule.data_weight_derivative(time)
            * (x1.to(compute_dtype) - x0.to(compute_dtype))
            + self.schedule.noise_scale_derivative(time) * noise
        )
        gamma = self.schedule.noise_scale(time)
        if not bool((gamma.abs() >= self.numerical_epsilon).all()):
            raise ValueError(
                "score targets exclude times whose abs(gamma) is below numerical_epsilon."
            )
        score = -noise / gamma
        return velocity, score

    def sample_times(
        self,
        shape: int | Sequence[int] | torch.Size,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
        mode: Literal["score", "flow"] = "score",
    ) -> torch.Tensor:
        """Draw reproducible open-interval training times, optionally paired.

        :param shape: Output shape, or a non-negative integer shorthand for one
            batch dimension.
        :param device: Optional output device; defaults to the generator/device
            default selected by :func:`torch.rand`.
        :param dtype: Floating output dtype.
        :param generator: Optional generator controlling all uniform draws.
        :param antithetic: Pair flattened draws as ``u, 1-u``; an odd final
            element is independently drawn.
        :param mode: ``"score"`` (default) rejects values with
            ``abs(gamma(t)) < numerical_epsilon``; explicit ``"flow"`` draws
            closed-interval bridge times without score-trainability filtering.
        :return: Finite times with ``shape``. Score mode yields only strictly
            interior, score-trainable values; flow mode yields ``[0,1]``.
        :rtype: torch.Tensor
        :raises ValueError: If shape, dtype, mode, or score-trainable support
            is invalid.
        :raises RuntimeError: If the supplied generator cannot generate on the
            requested device.

        The operation does not mutate a caller tensor. Antithetic layout is
        deterministic: the first half is ``u`` and the next half is ``1-u``;
        only the supplied generator determines randomness. No caller tensors
        exist to mutate, and sampled times are not differentiable random draws.
        """
        output_shape = _normalize_shape(shape)
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("dtype must be a floating torch dtype.")
        if mode not in {"score", "flow"}:
            raise ValueError("mode must be 'score' or 'flow'.")
        count = math.prod(output_shape)
        if mode == "flow":
            return _draw_uniform_times(
                output_shape,
                device=device,
                dtype=dtype,
                generator=generator,
                antithetic=antithetic,
            )
        values = _draw_score_times(
            self.schedule,
            count,
            device=device,
            dtype=dtype,
            generator=generator,
            antithetic=antithetic,
            threshold=self.numerical_epsilon,
        )
        return values.reshape(output_shape)

    def velocity_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        edit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return an unbiased editable-entry mean squared velocity loss.

        :param prediction: Predicted velocity with arbitrary finite floating
            shape, dtype, and device.
        :param target: Exact velocity target matching prediction shape, dtype,
            device, and finite values.
        :param edit_mask: Optional boolean editable prefix mask. ``[N]`` masks
            every feature of ``[N,C]``; true entries, not fixed entries, enter
            the reduction.
        :return: Scalar MSE in float32 for BF16/FP16 inputs, otherwise the
            native floating accumulation dtype. Empty tensors or an all-false
            mask return a differentiable finite zero.
        :rtype: torch.Tensor
        :raises ValueError: If tensors or mask violate shape/dtype/device/
            finiteness contracts.

        The denominator is the number of selected scalar entries, preventing
        fixed fragment entries from biasing the result. Arbitrary empty leading
        dimensions and a fully fixed mask return a differentiable zero instead
        of a mean-of-empty NaN. No input is mutated and prediction gradients
        remain intact under autocast.
        """
        _validate_endpoints(prediction, target)
        compute_dtype = _compute_dtype(prediction.dtype)
        squared = (prediction.to(compute_dtype) - target.to(compute_dtype)).square()
        if edit_mask is None:
            if squared.numel() == 0:
                return squared.sum() * 0.0
            return squared.mean()
        expanded = _expand_mask(edit_mask, prediction)
        weights = expanded.to(dtype=compute_dtype)
        denominator = weights.sum()
        if bool(denominator == 0):
            return squared.sum() * 0.0
        return (squared * weights).sum() / denominator


def _validate_endpoints(x0: torch.Tensor, x1: torch.Tensor) -> None:
    """Require compatible finite floating endpoint tensors."""
    if not isinstance(x0, torch.Tensor) or not x0.is_floating_point():
        raise ValueError("x0 must be a floating torch.Tensor.")
    if not isinstance(x1, torch.Tensor) or not x1.is_floating_point():
        raise ValueError("x1 must be a floating torch.Tensor.")
    if x0.shape != x1.shape or x0.dtype != x1.dtype or x0.device != x1.device:
        raise ValueError("x0 and x1 must have matching shape, dtype, and device.")
    if not bool(torch.isfinite(x0).all()) or not bool(torch.isfinite(x1).all()):
        raise ValueError("continuous endpoints must contain only finite values.")


def _expand_time(time: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or leading-prefix batch time to endpoint shape."""
    validate_time(time)
    if time.device != reference.device:
        raise ValueError("time must be on the endpoint device.")
    if time.ndim > reference.ndim:
        raise ValueError("time rank cannot exceed endpoint rank.")
    reshaped = time.reshape(*time.shape, *([1] * (reference.ndim - time.ndim)))
    try:
        return torch.broadcast_to(reshaped, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            "time must broadcast across endpoint leading dimensions."
        ) from error


def _validate_sample(sample: ContinuousSample, reference: torch.Tensor) -> None:
    """Check shape, dtype, device, and finite sample components."""
    expected_noise_dtype = _compute_dtype(reference.dtype)
    for name in ("value", "noise"):
        tensor = getattr(sample, name)
        expected_dtype = reference.dtype if name == "value" else expected_noise_dtype
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != reference.shape
            or tensor.dtype != expected_dtype
            or tensor.device != reference.device
            or not tensor.is_floating_point()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(
                f"sample.{name} must match endpoint shape, expected dtype, device, and finiteness."
            )


def _expand_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Expand a boolean prefix mask to every selected scalar entry."""
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("edit_mask must be a torch.bool tensor.")
    if mask.device != reference.device:
        raise ValueError("edit_mask must be on the prediction device.")
    if mask.ndim > reference.ndim:
        raise ValueError("edit_mask rank cannot exceed prediction rank.")
    reshaped = mask.reshape(*mask.shape, *([1] * (reference.ndim - mask.ndim)))
    try:
        return torch.broadcast_to(reshaped, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            "edit_mask must broadcast across prediction leading dimensions."
        ) from error


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use float32 accumulation for reduced-precision floating tensors."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _normalize_shape(shape: int | Sequence[int] | torch.Size) -> tuple[int, ...]:
    """Normalize one public sample-time shape into a validated tuple."""
    values: tuple[int, ...]
    if isinstance(shape, int):
        values = (shape,)
    else:
        values = tuple(shape)
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("shape must contain only non-negative integers.")
    return values


def _draw_uniform_times(
    shape: tuple[int, ...],
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
    generator: torch.Generator | None,
    antithetic: bool,
) -> torch.Tensor:
    """Draw flow times with deterministic optional antithetic ordering."""
    count = math.prod(shape)
    if not antithetic:
        return torch.rand(shape, device=device, dtype=dtype, generator=generator)
    pair_count = count // 2
    leading = torch.rand(pair_count, device=device, dtype=dtype, generator=generator)
    values = [leading, 1.0 - leading]
    if count % 2:
        values.append(torch.rand(1, device=device, dtype=dtype, generator=generator))
    return torch.cat(values).reshape(shape)


def _draw_score_times(
    schedule: InterpolantSchedule,
    count: int,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
    generator: torch.Generator | None,
    antithetic: bool,
    threshold: float,
) -> torch.Tensor:
    """Rejection-sample score-trainable times without changing their targets."""
    if count == 0:
        return torch.empty(0, device=device, dtype=dtype)
    if antithetic:
        pair_count = count // 2
        primary = _draw_valid_score_values(
            schedule,
            pair_count,
            device=device,
            dtype=dtype,
            generator=generator,
            threshold=threshold,
            require_antithetic=True,
        )
        values = [primary, 1.0 - primary]
        if count % 2:
            values.append(
                _draw_valid_score_values(
                    schedule,
                    1,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                    threshold=threshold,
                    require_antithetic=False,
                )
            )
        return torch.cat(values)
    return _draw_valid_score_values(
        schedule,
        count,
        device=device,
        dtype=dtype,
        generator=generator,
        threshold=threshold,
        require_antithetic=False,
    )


def _draw_valid_score_values(
    schedule: InterpolantSchedule,
    count: int,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
    generator: torch.Generator | None,
    threshold: float,
    require_antithetic: bool,
) -> torch.Tensor:
    """Draw primary score times, optionally requiring valid complementary times."""
    accepted: list[torch.Tensor] = []
    attempts = 0
    required = count
    while required:
        attempts += 1
        if attempts > 4096:
            raise ValueError("schedule has no reachable score-trainable time support.")
        candidate_count = max(required * 2, 32)
        candidates = torch.rand(
            candidate_count, device=device, dtype=dtype, generator=generator
        )
        stable_time = candidates.to(dtype=torch.float32)
        gamma = schedule.noise_scale(stable_time)
        valid = (stable_time > 0.0) & (stable_time < 1.0) & (gamma.abs() >= threshold)
        if require_antithetic:
            complement = 1.0 - stable_time
            complement_gamma = schedule.noise_scale(complement)
            valid &= (
                (complement > 0.0)
                & (complement < 1.0)
                & (complement_gamma.abs() >= threshold)
            )
        selected = candidates[valid][:required]
        if selected.numel():
            accepted.append(selected)
            required -= selected.numel()
    return torch.cat(accepted)
