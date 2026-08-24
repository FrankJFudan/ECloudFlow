"""Electron-cloud field quality metrics with explicit QM availability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ecloudflow.evaluation._utils import (
    OptionalMetric,
    context_of,
    mean_result,
    property_of,
    records_of,
)


@dataclass(frozen=True)
class ElectronCountMetric:
    """Measure mean absolute electron-count error from recorded fields."""

    name: str = "electron_count_error"
    group: str = "ecloud"

    def compute(self, context) -> Any:
        """Return a finite mean count error when target/prediction pairs exist."""
        context = context_of(context)
        values = []
        for record in records_of(context):
            prediction = property_of(
                record, "predicted_electron_count", "electron_count"
            )
            target = property_of(
                record, "target_electron_count", "reference_electron_count"
            )
            if prediction is not None and target is not None:
                values.append(abs(float(prediction) - float(target)))
        if not values:
            for field in context.electron_fields.values():
                if (
                    isinstance(field, dict)
                    and field.get("electron_count_error") is not None
                ):
                    values.append(float(field["electron_count_error"]))
        return mean_result(self.name, self.group, values, units="electrons")


@dataclass(frozen=True)
class DipoleErrorMetric:
    """Measure mean dipole-vector error from recorded diagnostics."""

    name: str = "dipole_error"
    group: str = "ecloud"

    def compute(self, context) -> Any:
        """Return mean scalar dipole error without fabricating QM labels."""
        values = [
            abs(float(value))
            for record in records_of(context_of(context))
            if (value := property_of(record, "dipole_error")) is not None
        ]
        return mean_result(self.name, self.group, values)


class ElectronDensityMetric(OptionalMetric):
    """Expose density-grid error when a field decoder/reference is available."""

    name = "electron_density_error"
    group = "ecloud"


class ElectronCycleMetric(OptionalMetric):
    """Expose latent density-cycle consistency as an optional metric."""

    name = "electron_cycle_error"
    group = "ecloud"


__all__ = [
    "DipoleErrorMetric",
    "ElectronCountMetric",
    "ElectronCycleMetric",
    "ElectronDensityMetric",
]
