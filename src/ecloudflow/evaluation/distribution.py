"""Reference-distribution, novelty, and simple diversity metrics."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from ecloudflow.evaluation._utils import (
    OptionalMetric,
    context_of,
    records_of,
    unavailable,
)
from ecloudflow.evaluation.types import MetricResult, MetricStatus


@dataclass(frozen=True)
class NoveltyMetric:
    """Measure canonical-SMILES novelty against an explicit reference index."""

    name: str = "novelty"
    group: str = "distribution"

    def compute(self, context) -> MetricResult:
        """Return generated fraction absent from ``context.references``."""
        context = context_of(context)
        generated = {
            getattr(record, "canonical_smiles", "")
            for record in records_of(context)
            if getattr(record, "canonical_smiles", None)
        }
        references = _reference_smiles(context)
        if not generated or not references:
            return unavailable(
                self.name, self.group, "generated or reference index is missing"
            )
        novel = generated - references
        return MetricResult(
            self.name,
            self.group,
            len(novel) / len(generated),
            MetricStatus.SUCCESS,
            units="fraction",
            diagnostics={"novel": len(novel), "generated": len(generated)},
        )


@dataclass(frozen=True)
class InternalDiversityMetric:
    """Measure one minus mean pairwise Morgan fingerprint similarity."""

    name: str = "internal_diversity"
    group: str = "distribution"

    def compute(self, context) -> MetricResult:
        """Return a deterministic pairwise Tanimoto diversity estimate."""
        molecules = []
        for record in records_of(context_of(context)):
            smiles = getattr(record, "canonical_smiles", None)
            molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if molecule is not None:
                molecules.append(molecule)
        if len(molecules) < 2:
            return unavailable(
                self.name, self.group, "at least two molecules are required"
            )
        fingerprints = [
            AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
            for molecule in molecules
        ]
        similarities = []
        for index, first in enumerate(fingerprints):
            for second in fingerprints[index + 1 :]:
                similarities.append(DataStructs.TanimotoSimilarity(first, second))
        return MetricResult(
            self.name,
            self.group,
            1.0 - sum(similarities) / len(similarities),
            MetricStatus.SUCCESS,
        )


def _reference_smiles(context) -> set[str]:
    """Normalize reference SMILES from common context representations."""
    if isinstance(context.references, (set, list, tuple)):
        values = context.references
    else:
        values = context.references.get(
            "smiles", context.references.get("canonical_smiles", ())
        )
    if isinstance(values, str):
        return {values}
    if isinstance(values, dict):
        values = values.keys()
    return {str(value) for value in values if value}


class FCDMetric(OptionalMetric):
    """Expose Frechet ChemNet Distance when a configured backend is present."""

    name = "fcd"
    group = "distribution"


class DescriptorJSDMetric(OptionalMetric):
    """Expose descriptor Jensen-Shannon divergence as an optional metric."""

    name = "descriptor_jsd"
    group = "distribution"


__all__ = [
    "DescriptorJSDMetric",
    "FCDMetric",
    "InternalDiversityMetric",
    "NoveltyMetric",
]
