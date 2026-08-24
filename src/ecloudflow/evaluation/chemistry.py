"""RDKit-based chemical validity and graph quality metrics."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from ecloudflow.evaluation._utils import (
    OptionalMetric,
    context_of,
    molecule_of,
    ratio_result,
    records_of,
    unavailable,
)
from ecloudflow.evaluation.types import MetricResult, MetricStatus


@dataclass(frozen=True)
class RDKitValidityMetric:
    """Measure sanitized molecular graph validity without mutating records."""

    name: str = "rdkit_validity"
    group: str = "chemistry"

    def compute(self, context) -> MetricResult:
        """Return the fraction of records that RDKit can sanitize."""
        values = []
        for record in records_of(context_of(context)):
            molecule = molecule_of(record)
            if molecule is None:
                values.append(False)
                continue
            try:
                Chem.SanitizeMol(molecule)
                values.append(True)
            except (RuntimeError, ValueError):
                values.append(False)
        return ratio_result(self.name, self.group, values)


@dataclass(frozen=True)
class UniquenessMetric:
    """Measure unique canonical isomeric SMILES among generated records."""

    name: str = "uniqueness"
    group: str = "chemistry"

    def compute(self, context) -> MetricResult:
        """Return unique-count divided by total-count."""
        records = records_of(context_of(context))
        smiles = [getattr(record, "canonical_smiles", None) for record in records]
        smiles = [value for value in smiles if isinstance(value, str) and value]
        if not smiles:
            return unavailable(self.name, self.group, "no canonical SMILES")
        unique = len(set(smiles))
        return MetricResult(
            self.name,
            self.group,
            unique / len(smiles),
            MetricStatus.SUCCESS,
            units="fraction",
            diagnostics={"unique": unique, "total": len(smiles)},
        )


@dataclass(frozen=True)
class ConnectivityMetric:
    """Measure the fraction of records containing one connected component."""

    name: str = "connected_molecule_rate"
    group: str = "chemistry"

    def compute(self, context) -> MetricResult:
        """Return connected-graph fraction after defensive RDKit copying."""
        values = []
        for record in records_of(context_of(context)):
            molecule = molecule_of(record)
            values.append(molecule is not None and len(Chem.GetMolFrags(molecule)) == 1)
        return ratio_result(self.name, self.group, values)


@dataclass(frozen=True)
class ValenceMetric:
    """Measure sanitizable valence/formal-charge assignments."""

    name: str = "valence_validity"
    group: str = "chemistry"

    def compute(self, context) -> MetricResult:
        """Return the fraction whose atom valences pass RDKit sanitization."""
        return RDKitValidityMetric(self.name, self.group).compute(context)


class PoseBustersMetric(OptionalMetric):
    """Expose PoseBusters validity when the optional package is installed."""

    name = "posebusters_validity"
    group = "chemistry"


__all__ = [
    "ConnectivityMetric",
    "PoseBustersMetric",
    "RDKitValidityMetric",
    "UniquenessMetric",
    "ValenceMetric",
]
