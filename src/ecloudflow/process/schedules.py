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
        """Return the analytic coefficient of the data endpoint."""

    @abstractmethod
    def data_weight_derivative(self, time: torch.Tensor) -> torch.Tensor:
        """Return the exact derivative of :meth:`data_weight`."""

    @abstractmethod
    def noise_scale(self, time: torch.Tensor) -> torch.Tensor:
        """Return the non-negative noise coefficient of the bridge."""

    @abstractmethod
    def noise_scale_derivative(self, time: torch.Tensor) -> torch.Tensor:
        """Return the exact derivative of :meth:`noise_scale`."""


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
