"""Analytic schedules for prior-to-data stochastic interpolants."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch


class InterpolantSchedule(ABC):
    """Define an analytic bridge with ``t=0`` prior and ``t=1`` data.

    Schedules parameterize ``x_t = (1-a(t)) x_0 + a(t) x_1 + g(t) epsilon``.
    They therefore provide the data weight ``a``, the interior noise scale
    ``g``, and their exact time derivatives. Implementations must use a
    non-negative finite scale with zero endpoint noise and positive interior
    noise, so a denoising score is defined only for ``0 < t < 1``.
    """

    @abstractmethod
    def data_weight(self, time: torch.Tensor) -> torch.Tensor:
        """Return the analytic data-endpoint coefficient ``a(t)``.

        :param time: Finite floating tensor of any shape, dtype, and device,
            with every value in the closed interval ``[0,1]``.
        :return: Coefficients matching ``time`` shape, dtype, and device, with
            ``a(0)=0`` and ``a(1)=1``.
        :rtype: torch.Tensor
        :raises ValueError: If time dtype, finiteness, shape, or endpoints are
            invalid for the schedule.

        Implementations do not mutate time or consume randomness, are
        deterministic for identical inputs, and retain autograd gradients with
        respect to floating time values.
        """

    @abstractmethod
    def data_weight_derivative(self, time: torch.Tensor) -> torch.Tensor:
        """Return the analytic derivative ``a'(t)`` of the data weight.

        :param time: Finite floating tensor of arbitrary shape, dtype, and
            device, containing times in ``[0,1]``.
        :return: Exact derivative values matching time shape, dtype, and device.
        :rtype: torch.Tensor
        :raises ValueError: If time dtype, finite values, shape, or interval is
            invalid.

        This method never mutates time or uses random state. It is deterministic
        and differentiable with respect to time wherever the schedule is.
        """

    @abstractmethod
    def noise_scale(self, time: torch.Tensor) -> torch.Tensor:
        """Return the analytic noise scale ``g(t)`` of the bridge.

        :param time: Finite floating tensor with arbitrary shape, dtype, and
            device, whose values lie in ``[0,1]``.
        :return: Noise scales matching time shape, dtype, and device; endpoint
            values are exactly zero and score training uses only stable interior
            values.
        :rtype: torch.Tensor
        :raises ValueError: If time dtype, finite values, shape, or interval is
            invalid.

        Implementations are deterministic, do not mutate input storage, and
        preserve autograd gradients for floating time tensors.
        """

    @abstractmethod
    def noise_scale_derivative(self, time: torch.Tensor) -> torch.Tensor:
        """Return the analytic derivative ``g'(t)`` of the noise scale.

        :param time: Finite floating time tensor with arbitrary shape, dtype,
            and device, constrained to the closed unit interval.
        :return: Exact derivative values matching time shape, dtype, and device.
        :rtype: torch.Tensor
        :raises ValueError: If time dtype, finiteness, shape, or interval is
            invalid.

        The derivative is deterministic, consumes no generator state, never
        mutates time, and remains differentiable with respect to time.
        """


class LinearBridge(InterpolantSchedule):
    """Use linear data interpolation with a smooth zero-endpoint noise bridge.

    The schedule is ``a(t)=t`` and ``g(t)=s t(1-t)``, where ``s`` is
    ``interior_noise``. This makes the prior and data endpoints exact while
    maintaining a positive noise scale at every valid score-training time.
    """

    def __init__(self, interior_noise: float = 1.0) -> None:
        """Initialize a linear prior-to-data bridge.

        :param interior_noise: Positive finite multiplier ``s`` for
            ``t(1-t)`` noise.
        :raises ValueError: If ``interior_noise`` is not finite and positive.
        """
        if not math.isfinite(interior_noise) or interior_noise <= 0.0:
            raise ValueError("interior_noise must be finite and positive.")
        self.interior_noise = float(interior_noise)

    def data_weight(self, time: torch.Tensor) -> torch.Tensor:
        """Return ``a(t)=t`` after validating the closed time interval.

        :param time: Finite floating time tensor on any device, with values in
            ``[0, 1]``.
        :return: Data weights with the input shape, dtype, and device.
        :rtype: torch.Tensor
        :raises ValueError: If time is non-floating, non-finite, or outside the
            closed interpolation interval.
        """
        _validate_time(time)
        return time

    def data_weight_derivative(self, time: torch.Tensor) -> torch.Tensor:
        """Return the exact constant derivative ``a'(t)=1``.

        :param time: Finite floating time tensor in ``[0, 1]``.
        :return: One tensor matching ``time`` dtype, device, and shape.
        :rtype: torch.Tensor
        :raises ValueError: If ``time`` is invalid.
        """
        _validate_time(time)
        return torch.ones_like(time)

    def noise_scale(self, time: torch.Tensor) -> torch.Tensor:
        """Return ``g(t)=s t(1-t)`` with exact zero endpoint values.

        :param time: Finite floating time tensor in ``[0, 1]``.
        :return: Non-negative noise scales matching ``time``.
        :rtype: torch.Tensor
        :raises ValueError: If ``time`` is invalid.
        """
        _validate_time(time)
        return self.interior_noise * time * (1.0 - time)

    def noise_scale_derivative(self, time: torch.Tensor) -> torch.Tensor:
        """Return ``g'(t)=s(1-2t)`` without numerical differencing.

        :param time: Finite floating time tensor in ``[0, 1]``.
        :return: Derivative tensor matching ``time``.
        :rtype: torch.Tensor
        :raises ValueError: If ``time`` is invalid.
        """
        _validate_time(time)
        return self.interior_noise * (1.0 - 2.0 * time)


class CosineBridge(InterpolantSchedule):
    """Use cosine-eased data interpolation and sinusoidal interior noise.

    The schedule is ``a(t)=sin(pi t / 2)`` and ``g(t)=s sin(pi t)``. Both
    derivatives are analytic, and ``g`` is exactly zero only at endpoints.
    """

    def __init__(self, interior_noise: float = 1.0) -> None:
        """Initialize a cosine prior-to-data bridge.

        :param interior_noise: Positive finite multiplier ``s`` for sinusoidal
            interior noise.
        :raises ValueError: If ``interior_noise`` is not finite and positive.
        """
        if not math.isfinite(interior_noise) or interior_noise <= 0.0:
            raise ValueError("interior_noise must be finite and positive.")
        self.interior_noise = float(interior_noise)

    def data_weight(self, time: torch.Tensor) -> torch.Tensor:
        """Return ``a(t)=sin(pi t / 2)`` on the closed interval.

        :param time: Finite floating time tensor in ``[0, 1]``.
        :return: Data weights matching ``time`` shape, dtype, and device.
        :rtype: torch.Tensor
        :raises ValueError: If ``time`` is invalid.
        """
        _validate_time(time)
        values = torch.sin(time * (math.pi / 2.0))
        return torch.where(
            time == 0.0,
            torch.zeros_like(values),
            torch.where(time == 1.0, torch.ones_like(values), values),
        )

    def data_weight_derivative(self, time: torch.Tensor) -> torch.Tensor:
        """Return ``a'(t)=pi cos(pi t/2)/2`` analytically.

        :param time: Finite floating time tensor in ``[0, 1]``.
        :return: Derivative tensor matching ``time``.
        :rtype: torch.Tensor
        :raises ValueError: If ``time`` is invalid.
        """
        _validate_time(time)
        return (math.pi / 2.0) * torch.cos(time * (math.pi / 2.0))

    def noise_scale(self, time: torch.Tensor) -> torch.Tensor:
        """Return ``g(t)=s sin(pi t)`` with exact zero endpoints.

        :param time: Finite floating time tensor in ``[0, 1]``.
        :return: Non-negative noise scales matching ``time``.
        :rtype: torch.Tensor
        :raises ValueError: If ``time`` is invalid.
        """
        _validate_time(time)
        values = self.interior_noise * torch.sin(math.pi * time)
        return torch.where(
            (time == 0.0) | (time == 1.0), torch.zeros_like(values), values
        )

    def noise_scale_derivative(self, time: torch.Tensor) -> torch.Tensor:
        """Return ``g'(t)=s pi cos(pi t)`` analytically.

        :param time: Finite floating time tensor in ``[0, 1]``.
        :return: Derivative tensor matching ``time``.
        :rtype: torch.Tensor
        :raises ValueError: If ``time`` is invalid.
        """
        _validate_time(time)
        return self.interior_noise * math.pi * torch.cos(math.pi * time)


def validate_time(time: torch.Tensor) -> None:
    """Validate a finite floating schedule time in the closed unit interval.

    :param time: Candidate scalar or tensor time on any torch device.
    :return: None.
    :rtype: None
    :raises ValueError: If the tensor has an invalid dtype, is empty or
        non-finite, or contains values outside ``[0, 1]``.
    """
    _validate_time(time)


def _validate_time(time: torch.Tensor) -> None:
    """Implement schedule-time validation shared by all public schedule calls."""
    if not isinstance(time, torch.Tensor) or not time.is_floating_point():
        raise ValueError("time must be a floating torch.Tensor.")
    if time.numel() == 0:
        raise ValueError("time must not be empty.")
    if not bool(torch.isfinite(time).all()):
        raise ValueError("time must contain only finite values.")
    if not bool(((time >= 0.0) & (time <= 1.0)).all()):
        raise ValueError("time must lie in [0, 1].")
