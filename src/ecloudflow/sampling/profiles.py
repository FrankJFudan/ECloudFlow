"""Named numerical presets for constrained molecular sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SamplingProfile:
    """Describe a reproducible solver/corrector configuration.

    :param name: Stable profile identifier.
    :param solver: Predictor integrator, either Euler or Heun.
    :param num_steps: Number of integration intervals.
    :param corrector_steps: Number of terminal Langevin correction steps.
    :return: Immutable numerical profile.
    :rtype: SamplingProfile
    """

    name: Literal["fast", "balanced", "quality"]
    solver: Literal["euler", "heun"]
    num_steps: int
    corrector_steps: int

    def __post_init__(self) -> None:
        if self.num_steps < 1 or self.corrector_steps < 0:
            raise ValueError(
                "num_steps must be positive and corrector_steps non-negative."
            )


def fast_profile() -> SamplingProfile:
    """Return the high-throughput 20-step Euler preset."""
    return SamplingProfile("fast", "euler", 20, 0)


def balanced_profile() -> SamplingProfile:
    """Return the default 40-step Heun preset with two corrections."""
    return SamplingProfile("balanced", "heun", 40, 2)


def quality_profile() -> SamplingProfile:
    """Return the high-quality 100-step Heun preset with eight corrections."""
    return SamplingProfile("quality", "heun", 100, 8)


def get_profile(name: str) -> SamplingProfile:
    """Resolve a named profile.

    :param name: Case-insensitive profile name.
    :return: Numerical profile.
    :rtype: SamplingProfile
    :raises ValueError: If ``name`` is unknown.
    """
    profiles = {
        "fast": fast_profile,
        "balanced": balanced_profile,
        "quality": quality_profile,
    }
    try:
        return profiles[name.lower()]()
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unknown sampling profile: {name!r}") from exc
