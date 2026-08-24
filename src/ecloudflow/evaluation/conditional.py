"""Fragment-conditioned and pocket-conditional success metrics."""

from __future__ import annotations

from dataclasses import dataclass

from ecloudflow.evaluation._utils import (
    OptionalMetric,
    context_of,
    property_of,
    ratio_result,
    records_of,
    unavailable,
)


@dataclass(frozen=True)
class FragmentSuccessMetric:
    """Measure exact fixed-fragment preservation across generated poses."""

    name: str = "fragment_preservation_rate"
    group: str = "conditional"

    def compute(self, context):
        """Return the fraction with an explicit preservation diagnostic."""
        values = []
        for record in records_of(context_of(context)):
            value = property_of(
                record, "fragment_preserved", "fixed_fragment_preserved"
            )
            if value is not None:
                values.append(bool(value))
        if not values:
            return unavailable(self.name, self.group, "fragment diagnostics are absent")
        return ratio_result(self.name, self.group, values)


@dataclass(frozen=True)
class PocketConditionMetric:
    """Measure whether generated records retain a pocket-conditioning key."""

    name: str = "pocket_condition_coverage"
    group: str = "conditional"

    def compute(self, context):
        """Return the fraction with non-empty pocket provenance."""
        values = [
            bool(
                property_of(record, "pocket", "pocket_id")
                or context_of(context).pocket_id
            )
            for record in records_of(context_of(context))
        ]
        return ratio_result(self.name, self.group, values)


class PropertySuccessMetric(OptionalMetric):
    """Expose requested-property success from an optional property evaluator."""

    name = "property_success"
    group = "conditional"


__all__ = [
    "FragmentSuccessMetric",
    "PocketConditionMetric",
    "PropertySuccessMetric",
]
