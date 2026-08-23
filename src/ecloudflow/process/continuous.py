"""Continuous stochastic-interpolant samples, targets, and masked losses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ecloudflow.process.schedules import InterpolantSchedule, validate_time


@dataclass(frozen=True)
class ContinuousSample:
    """Store one noise-coupled sample of a continuous interpolant.

    ``value`` equals ``(1-a(t))x0 + a(t)x1 + g(t)noise``. ``noise`` has the
    endpoint tensor shape and is retained so conditional velocity and score
    targets use precisely the same random coupling used to create ``value``.
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
            plus the unmodified caller time tensor.
        :rtype: ContinuousSample
        :raises ValueError: If inputs have invalid shape, dtype, device,
            finiteness, or time range.

        No input is mutated and autograd remains connected to ``x0`` and ``x1``.
        The normal noise itself has no gradient. FP16/BF16 bridge arithmetic is
        performed in float32 and cast back to the endpoint dtype.
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
        return ContinuousSample(
            value=value.to(dtype=x0.dtype), time=time, noise=noise.to(dtype=x0.dtype)
        )

    def targets(
        self, x0: torch.Tensor, x1: torch.Tensor, sample: ContinuousSample
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact conditional velocity and denoising-score targets.

        :param x0: Finite floating prior endpoint tensor of arbitrary shape.
        :param x1: Finite floating data endpoint matching ``x0`` shape, dtype,
            and device.
        :param sample: Coupled output from :meth:`sample`; its noise must match
            endpoint shape/dtype/device and its time must be strictly interior.
        :return: ``(velocity, score)`` tensors matching endpoint shape, dtype,
            and device. Velocity is ``a'(t)(x1-x0)+g'(t)epsilon`` and score is
            ``-epsilon/g(t)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        :raises TypeError: If ``sample`` is not a :class:`ContinuousSample`.
        :raises ValueError: If contracts disagree, time is not in the open
            interval, or the schedule has non-positive interior noise.

        Endpoint scores are deliberately undefined and rejected, never made up
        by endpoint denominator clamping. At valid interior times only,
        ``numerical_epsilon`` bounds the denominator against underflow. Inputs
        and the frozen sample are not mutated; endpoint gradients flow through
        velocity while score has the sampled-noise gradient semantics.
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
        if not bool((gamma > 0.0).all()):
            raise ValueError("schedule noise scale must be positive at interior times.")
        score = -noise / gamma.clamp_min(self.numerical_epsilon)
        return velocity.to(dtype=x0.dtype), score.to(dtype=x0.dtype)

    def sample_times(
        self,
        shape: int | Sequence[int] | torch.Size,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        antithetic: bool = False,
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
        :return: Finite times with ``shape`` and values strictly in ``(0,1)``.
        :rtype: torch.Tensor
        :raises ValueError: If shape or dtype is invalid.

        The operation does not mutate a caller tensor. Antithetic layout is
        deterministic: the first half is ``u`` and the next half is ``1-u``;
        only the supplied generator determines randomness.
        """
        output_shape = _normalize_shape(shape)
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("dtype must be a floating torch dtype.")
        count = math.prod(output_shape)
        if not antithetic:
            raw = torch.rand(
                output_shape, device=device, dtype=dtype, generator=generator
            )
        else:
            pair_count = count // 2
            leading = torch.rand(
                pair_count, device=device, dtype=dtype, generator=generator
            )
            values = [leading, 1.0 - leading]
            if count % 2:
                values.append(
                    torch.rand(1, device=device, dtype=dtype, generator=generator)
                )
            raw = torch.cat(values).reshape(output_shape)
        epsilon = torch.finfo(dtype).eps
        return raw.clamp(min=epsilon, max=1.0 - epsilon)

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
            native floating accumulation dtype.
        :rtype: torch.Tensor
        :raises ValueError: If tensors or mask violate shape/dtype/device/
            finiteness contracts.

        The denominator is the number of selected scalar entries, preventing
        fixed fragment entries from biasing the result. A fully fixed mask
        returns a differentiable zero. No input is mutated and prediction
        gradients remain intact under autocast.
        """
        _validate_endpoints(prediction, target)
        compute_dtype = _compute_dtype(prediction.dtype)
        squared = (prediction.to(compute_dtype) - target.to(compute_dtype)).square()
        if edit_mask is None:
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
    for name in ("value", "noise"):
        tensor = getattr(sample, name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != reference.shape
            or tensor.dtype != reference.dtype
            or tensor.device != reference.device
            or not tensor.is_floating_point()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(
                f"sample.{name} must match endpoint shape, dtype, device, and finiteness."
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
    if isinstance(shape, int):
        values = (shape,)
    else:
        values = tuple(shape)
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("shape must contain only non-negative integers.")
    return values
