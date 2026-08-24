"""Pose geometry, clash, and raw-versus-relaxed quality metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
class PoseGeometryMetric:
    """Measure availability of finite 3D binding-pose conformers."""

    name: str = "pose_geometry_validity"
    group: str = "geometry"

    def compute(self, context) -> MetricResult:
        """Return finite-conformer fraction and atom-level diagnostics."""
        values = []
        atom_counts = []
        for record in records_of(context_of(context)):
            molecule = molecule_of(record)
            valid = False
            if molecule is not None and molecule.GetNumConformers() > 0:
                positions = molecule.GetConformer(0).GetPositions()
                valid = all(
                    math.isfinite(float(value)) for row in positions for value in row
                )
                if valid:
                    atom_counts.append(molecule.GetNumAtoms())
            values.append(valid)
        result = ratio_result(self.name, self.group, values)
        if result.status is MetricStatus.SUCCESS:
            return MetricResult(
                result.name,
                result.group,
                result.value,
                result.status,
                units=result.units,
                diagnostics={
                    **dict(result.diagnostics),
                    "mean_atoms": _mean(atom_counts),
                },
                per_item=result.per_item,
            )
        return result


@dataclass(frozen=True)
class ClashMetric:
    """Measure the fraction of poses without severe intramolecular clashes."""

    name: str = "intramolecular_clash_free"
    group: str = "geometry"
    minimum_distance: float = 1.0

    def compute(self, context) -> MetricResult:
        """Return a clash-free fraction using non-bonded atom distances."""
        if self.minimum_distance <= 0:
            raise ValueError("minimum_distance must be positive.")
        values = []
        for record in records_of(context_of(context)):
            molecule = molecule_of(record)
            if molecule is None or molecule.GetNumConformers() == 0:
                values.append(False)
                continue
            conformer = molecule.GetConformer(0)
            clash = False
            for first in range(molecule.GetNumAtoms()):
                for second in range(first + 1, molecule.GetNumAtoms()):
                    if molecule.GetBondBetweenAtoms(first, second) is not None:
                        continue
                    distance = (
                        conformer.GetAtomPosition(first)
                        - conformer.GetAtomPosition(second)
                    ).Length()
                    if distance < self.minimum_distance:
                        clash = True
                        break
                if clash:
                    break
            values.append(not clash)
        return ratio_result(self.name, self.group, values)


@dataclass(frozen=True)
class RawRelaxedRMSDMetric:
    """Measure RMSD between raw and relaxed conformers when both are present."""

    name: str = "raw_relaxed_rmsd"
    group: str = "geometry"

    def compute(self, context) -> MetricResult:
        """Return mean aligned RMSD from record-provided pose pairs."""
        values = []
        for record in records_of(context_of(context)):
            value = getattr(record, "properties", {}).get("raw_relaxed_rmsd")
            if value is not None:
                values.append(float(value))
        if not values:
            return unavailable(
                self.name, self.group, "raw/relaxed RMSD is not recorded"
            )
        return MetricResult(
            self.name,
            self.group,
            sum(values) / len(values),
            MetricStatus.SUCCESS,
            units="angstrom",
        )


def _mean(values: list[int]) -> float | None:
    """Return a finite mean or an explicit missing value."""
    return sum(values) / len(values) if values else None


class BondLengthJSDMetric(OptionalMetric):
    """Expose bond-length distribution divergence when reference data exists."""

    name = "bond_length_jsd"
    group = "geometry"


class AngleJSDMetric(OptionalMetric):
    """Expose bond-angle distribution divergence as an optional metric."""

    name = "angle_jsd"
    group = "geometry"


class DihedralJSDMetric(OptionalMetric):
    """Expose dihedral distribution divergence as an optional metric."""

    name = "dihedral_jsd"
    group = "geometry"


__all__ = [
    "AngleJSDMetric",
    "BondLengthJSDMetric",
    "ClashMetric",
    "DihedralJSDMetric",
    "PoseGeometryMetric",
    "RawRelaxedRMSDMetric",
]
