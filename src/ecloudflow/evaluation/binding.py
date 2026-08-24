"""Binding-affinity and optional docking interaction metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ecloudflow.evaluation._utils import (
    OptionalMetric,
    context_of,
    mean_result,
    property_of,
    records_of,
    unavailable,
)
from ecloudflow.evaluation.types import MetricResult, MetricStatus


@dataclass(frozen=True)
class VinaScoreMetric:
    """Compute mean Vina score through an injected backend.

    Missing binaries/backends return ``UNAVAILABLE`` with ``value=None``;
    existing record properties are used only when a backend result is already
    explicitly marked successful.
    """

    backend: Any = None
    name: str = "vina_score"
    group: str = "binding"

    def compute(self, context) -> MetricResult:
        """Return mean finite docking score or an honest unavailable result."""
        context = context_of(context)
        if self.backend is None:
            return unavailable(self.name, self.group, "no Vina backend configured")
        scores = []
        for record in records_of(context):
            molecule = getattr(record, "molecule", None)
            pocket = property_of(record, "pocket") or context.pockets.get(
                context.pocket_id
            )
            try:
                result = self.backend.score(molecule, pocket)
            except Exception as error:  # noqa: BLE001 - optional tool boundary
                return MetricResult(
                    self.name,
                    self.group,
                    None,
                    MetricStatus.FAILED,
                    diagnostics={"reason": f"{type(error).__name__}: {error}"},
                )
            status = getattr(result, "status", None)
            score = getattr(
                result, "score", result if isinstance(result, (int, float)) else None
            )
            if (
                str(status) not in {"DockingStatus.SUCCESS", "success"}
                and status is not None
                and getattr(status, "value", status) != "success"
            ):
                continue
            if score is not None:
                scores.append(float(score))
        if not scores:
            return unavailable(self.name, self.group, "Vina produced no usable scores")
        return mean_result(self.name, self.group, scores, units="kcal/mol")


@dataclass(frozen=True)
class DockingSuccessMetric:
    """Measure the fraction of records with explicit successful docking."""

    name: str = "docking_success_rate"
    group: str = "binding"

    def compute(self, context) -> MetricResult:
        """Return success fraction from typed results or status properties."""
        values = []
        for record in records_of(context_of(context)):
            status = property_of(record, "docking_status", "status")
            values.append(status in {"success", "available", "DockingStatus.SUCCESS"})
        if not values:
            return unavailable(self.name, self.group, "no docking statuses")
        from ecloudflow.evaluation._utils import ratio_result

        return ratio_result(self.name, self.group, values)


@dataclass(frozen=True)
class PLIFMetric:
    """Read an optional protein-ligand interaction fingerprint score."""

    name: str = "plif_similarity"
    group: str = "binding"

    def compute(self, context) -> MetricResult:
        """Return mean recorded PLIF similarity, or unavailable."""
        values = [
            float(property_of(record, "plif", "plif_similarity"))
            for record in records_of(context_of(context))
            if property_of(record, "plif", "plif_similarity") is not None
        ]
        return mean_result(self.name, self.group, values)


class ShapeContactMetric(OptionalMetric):
    """Expose shape/contact similarity from an optional docking backend."""

    name = "shape_contact_similarity"
    group = "binding"


__all__ = [
    "DockingSuccessMetric",
    "PLIFMetric",
    "ShapeContactMetric",
    "VinaScoreMetric",
]
