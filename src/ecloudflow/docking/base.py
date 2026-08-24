"""Typed interfaces for optional ligand docking backends."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class DockingStatus(str, Enum):
    """Stable status values for optional docking execution."""

    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass(frozen=True)
class DockingResult:
    """Record one docking score and its auditable execution provenance.

    :param score: Vina-like score in kcal/mol, or ``None`` when unavailable.
    :param status: Explicit backend status.
    :param backend: Backend name such as ``vina``.
    :param version: Backend version string when known.
    :param command: Argument vector used for the external process.
    :param raw_output: Captured stdout/stderr with credentials removed by the
        caller's path policy.
    :param elapsed_seconds: Wall-clock execution time.
    :param reason: Structured failure explanation.
    :param metadata: Additional JSON-safe backend details.
    :return: Immutable docking record.
    :rtype: DockingResult
    :raises ValueError: If a successful result has a non-finite score.
    """

    score: float | None
    status: DockingStatus | str
    backend: str = ""
    version: str = ""
    command: tuple[str, ...] = ()
    raw_output: str = ""
    elapsed_seconds: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate score/status semantics without inventing missing values."""
        status = (
            self.status
            if isinstance(self.status, DockingStatus)
            else DockingStatus(self.status)
        )
        object.__setattr__(self, "status", status)
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ValueError("docking score must be finite or None.")
        if status is DockingStatus.SUCCESS and self.score is None:
            raise ValueError("successful docking results require a score.")
        if not math.isfinite(float(self.elapsed_seconds)) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        object.__setattr__(self, "command", tuple(str(value) for value in self.command))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def available(self) -> bool:
        """Return whether a numeric score is available for ranking."""
        return self.status is DockingStatus.SUCCESS and self.score is not None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe backend summary."""
        return {
            "score": self.score,
            "status": self.status.value,
            "backend": self.backend,
            "version": self.version,
            "command": list(self.command),
            "raw_output": self.raw_output,
            "elapsed_seconds": self.elapsed_seconds,
            "reason": self.reason,
            **self.metadata,
        }


class DockingBackend(Protocol):
    """Protocol implemented by score-producing docking services."""

    name: str

    def score(self, molecule: Any, pocket: Any, **kwargs: Any) -> DockingResult:
        """Return one typed score without mutating molecule or pocket inputs."""


def validate_box(
    center: Sequence[float], size: Sequence[float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Validate and normalize a Vina search-box center and positive size."""
    if len(center) != 3 or len(size) != 3:
        raise ValueError("docking box center and size must contain three values.")
    normalized_center = tuple(float(value) for value in center)
    normalized_size = tuple(float(value) for value in size)
    if not all(math.isfinite(value) for value in normalized_center):
        raise ValueError("docking box center must be finite.")
    if not all(math.isfinite(value) and value > 0 for value in normalized_size):
        raise ValueError("docking box size must be finite and positive.")
    return normalized_center, normalized_size


__all__ = ["DockingBackend", "DockingResult", "DockingStatus", "validate_box"]
