"""Immutable contracts shared by all ECloudFlow evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol


class MetricStatus(str, Enum):
    """Explicit metric execution states."""

    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class MetricResult:
    """Return one scalar/vector metric with provenance and honest status.

    :param name: Stable metric name.
    :param group: One of the seven registry domains.
    :param value: Numeric, mapping, or sequence value; ``None`` when absent.
    :param status: Explicit success/unavailable/invalid/failed state.
    :param units: Human-readable units, if applicable.
    :param diagnostics: JSON-safe counts and failure details.
    :param per_item: Optional immutable per-molecule values.
    :return: Typed metric result.
    :rtype: MetricResult
    :raises ValueError: If a successful numeric value is non-finite.
    """

    name: str
    group: str
    value: Any = None
    status: MetricStatus | str = MetricStatus.SUCCESS
    units: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    per_item: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Validate status and freeze diagnostics without mutating inputs."""
        status = (
            self.status
            if isinstance(self.status, MetricStatus)
            else MetricStatus(self.status)
        )
        object.__setattr__(self, "status", status)
        if status is MetricStatus.SUCCESS:
            _validate_finite_value(self.value)
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )
        object.__setattr__(self, "per_item", tuple(self.per_item))

    @property
    def available(self) -> bool:
        """Return whether this result contains a successful value."""
        return self.status is MetricStatus.SUCCESS and self.value is not None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result summary."""
        return {
            "name": self.name,
            "group": self.group,
            "value": _json_safe(self.value),
            "status": self.status.value,
            "units": self.units,
            "diagnostics": _json_safe(dict(self.diagnostics)),
            "per_item": _json_safe(list(self.per_item)),
        }


@dataclass(frozen=True)
class EvaluationContext:
    """Read-only inputs passed to metric implementations.

    :param records: Valid or generated records for one or more pockets.
    :param ranked: Optional :class:`RankedMolecule` rows.
    :param pockets: Pocket objects keyed by pocket ID.
    :param references: Reference SMILES/index values for novelty metrics.
    :param electron_fields: Optional generated/reference electron fields.
    :param timings: NFE, wall-time, memory, and scaling observations.
    :param metadata: Additional immutable run provenance.
    :param pocket_id: Convenience ID for single-pocket contexts.
    :param raw_relaxed_policy: Geometry pose policy.
    :return: Immutable evaluation context.
    :rtype: EvaluationContext
    """

    records: tuple[Any, ...] = ()
    generated: tuple[Any, ...] = ()
    ranked: tuple[Any, ...] = ()
    pocket: Any = None
    reference: Any = None
    pockets: Mapping[str, Any] = field(default_factory=dict)
    references: Mapping[str, Any] = field(default_factory=dict)
    electron_fields: Mapping[str, Any] = field(default_factory=dict)
    timings: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    pocket_id: str | None = None
    raw_relaxed_policy: str = "raw"

    def __post_init__(self) -> None:
        """Normalize optional sequences and freeze mapping boundaries."""
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "generated", tuple(self.generated))
        object.__setattr__(self, "ranked", tuple(self.ranked))
        for name in ("pockets", "references", "electron_fields", "timings", "metadata"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if self.raw_relaxed_policy not in {"raw", "relaxed", "both"}:
            raise ValueError("raw_relaxed_policy must be raw, relaxed, or both.")

    @property
    def molecules(self) -> tuple[Any, ...]:
        """Return ranked rows when present, otherwise generation records."""
        return self.ranked or self.records or self.generated


class Metric(Protocol):
    """Protocol implemented by one named metric."""

    name: str
    group: str

    def compute(self, context: EvaluationContext | Any) -> MetricResult:
        """Compute a metric without mutating generated molecules."""


@dataclass(frozen=True)
class EvaluationResult:
    """Collect metric results and per-metric diagnostics for one run."""

    results: Mapping[str, MetricResult]
    context: EvaluationContext | None = None

    def __post_init__(self) -> None:
        """Freeze the named result mapping."""
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))

    @property
    def by_group(self) -> dict[str, dict[str, MetricResult]]:
        """Return a nested group-to-metric view for reports."""
        groups: dict[str, dict[str, MetricResult]] = {}
        for name, result in self.results.items():
            groups.setdefault(result.group, {})[name] = result
        return groups

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe evaluation summary."""
        return {name: result.as_dict() for name, result in self.results.items()}


def coerce_context(value: EvaluationContext | Any) -> EvaluationContext:
    """Wrap a single record or mapping in a minimal evaluation context."""
    if isinstance(value, EvaluationContext):
        return value
    if isinstance(value, Mapping):
        return EvaluationContext(metadata=value)
    return EvaluationContext(records=(value,))


def _validate_finite_value(value: Any) -> None:
    """Reject non-finite numeric leaves while allowing structured values."""
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("successful metric values must be finite.")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_value(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _validate_finite_value(item)


def _json_safe(value: Any) -> Any:
    """Convert nested metric values into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


__all__ = [
    "EvaluationContext",
    "EvaluationResult",
    "Metric",
    "MetricResult",
    "MetricStatus",
    "coerce_context",
]
