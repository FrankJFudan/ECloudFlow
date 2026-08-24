"""Small read-only helpers shared by evaluation metric implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rdkit import Chem

from ecloudflow.evaluation.types import EvaluationContext, coerce_context


def context_of(value: EvaluationContext | Any) -> EvaluationContext:
    """Normalize a context or one-record fixture."""
    return coerce_context(value)


def records_of(context: EvaluationContext) -> tuple[Any, ...]:
    """Return ranked rows first, then raw generation records."""
    return context.molecules


def property_of(record: Any, *names: str) -> Any:
    """Read the first non-null property from a row or record."""
    properties = getattr(record, "properties", None)
    if isinstance(properties, Mapping):
        for name in names:
            if properties.get(name) is not None:
                return properties[name]
    for name in names:
        value = getattr(record, name, None)
        if value is not None:
            return value
    return None


def molecule_of(record: Any) -> Chem.Mol | None:
    """Return a defensive RDKit molecule or parse its canonical SMILES."""
    molecule = getattr(record, "molecule", None)
    if isinstance(molecule, Chem.Mol):
        return Chem.Mol(molecule)
    smiles = getattr(record, "canonical_smiles", None) or getattr(
        record, "smiles", None
    )
    if isinstance(smiles, str):
        return Chem.MolFromSmiles(smiles)
    return None


def unavailable(name: str, group: str, reason: str):
    """Construct a standard unavailable metric result."""
    from ecloudflow.evaluation.types import MetricResult, MetricStatus

    return MetricResult(
        name, group, None, MetricStatus.UNAVAILABLE, diagnostics={"reason": reason}
    )


def ratio_result(name: str, group: str, values: list[bool], *, units: str = "fraction"):
    """Construct a finite fraction result, rejecting empty input explicitly."""
    from ecloudflow.evaluation.types import MetricResult, MetricStatus

    if not values:
        return unavailable(name, group, "no evaluable records")
    valid = sum(bool(value) for value in values)
    return MetricResult(
        name,
        group,
        valid / len(values),
        MetricStatus.SUCCESS,
        units=units,
        diagnostics={"valid": valid, "total": len(values)},
        per_item=tuple(float(value) for value in values),
    )


def mean_result(
    name: str, group: str, values: list[float], *, units: str | None = None
):
    """Construct a finite mean result from non-empty numeric observations."""
    from ecloudflow.evaluation.types import MetricResult, MetricStatus

    if not values:
        return unavailable(name, group, "no evaluable records")
    return MetricResult(
        name,
        group,
        sum(values) / len(values),
        MetricStatus.SUCCESS,
        units=units,
        diagnostics={"count": len(values)},
        per_item=tuple(values),
    )


class OptionalMetric:
    """Base helper for optional metrics that require an external backend."""

    name = "optional_metric"
    group = "chemistry"

    def __init__(
        self, backend=None, *, name: str | None = None, group: str | None = None
    ):
        self.backend = backend
        if name is not None:
            self.name = name
        if group is not None:
            self.group = group

    def compute(self, context):
        """Return an explicit unavailable result when no compatible backend exists."""
        if self.backend is None:
            return unavailable(
                self.name, self.group, "optional backend is not configured"
            )
        return unavailable(
            self.name, self.group, "backend adapter has no compatible result"
        )


__all__ = [
    "OptionalMetric",
    "context_of",
    "mean_result",
    "molecule_of",
    "property_of",
    "ratio_result",
    "records_of",
    "unavailable",
]
