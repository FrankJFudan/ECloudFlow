"""Sampling efficiency, yield, NFE, and runtime metrics."""

from __future__ import annotations

from dataclasses import dataclass

from ecloudflow.evaluation._utils import (
    OptionalMetric,
    context_of,
    property_of,
    records_of,
    unavailable,
)
from ecloudflow.evaluation.types import MetricResult, MetricStatus


@dataclass(frozen=True)
class GenerationEfficiencyMetric:
    """Summarize valid yield, attempts, NFE, and wall-time diagnostics."""

    name: str = "generation_efficiency"
    group: str = "efficiency"

    def compute(self, context):
        """Return a structured efficiency value from explicit run metadata."""
        context = context_of(context)
        values = dict(context.timings)
        if not values:
            values = (
                dict(context.metadata.get("timings", {}))
                if isinstance(context.metadata.get("timings", {}), dict)
                else {}
            )
        records = records_of(context)
        if "valid_count" not in values and records:
            values["valid_count"] = len(records)
        if "valid_count" not in values:
            return unavailable(self.name, self.group, "timing/yield metadata is absent")
        target = values.get("target_count")
        if target is not None and float(target) > 0:
            values["yield"] = float(values["valid_count"]) / float(target)
        return MetricResult(self.name, self.group, values, MetricStatus.SUCCESS)


@dataclass(frozen=True)
class NFEMetric:
    """Return mean function evaluations from per-record diagnostics."""

    name: str = "nfe"
    group: str = "efficiency"

    def compute(self, context):
        """Return finite mean NFE or an unavailable state."""
        values = [
            float(value)
            for record in records_of(context_of(context))
            if (value := property_of(record, "nfe", "function_evaluations")) is not None
        ]
        if not values:
            return unavailable(self.name, self.group, "NFE diagnostics are absent")
        return MetricResult(
            self.name,
            self.group,
            sum(values) / len(values),
            MetricStatus.SUCCESS,
            units="evaluations",
        )


class MemoryMetric(OptionalMetric):
    """Expose peak-memory measurements from an optional profiler."""

    name = "peak_memory"
    group = "efficiency"


class ScalingMetric(OptionalMetric):
    """Expose multi-GPU scaling efficiency from benchmark artifacts."""

    name = "scaling_efficiency"
    group = "efficiency"


__all__ = [
    "GenerationEfficiencyMetric",
    "MemoryMetric",
    "NFEMetric",
    "ScalingMetric",
]
