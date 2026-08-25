"""Deterministic docking-aware ranking and molecule identifiers."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem
from rdkit.Chem import QED

from ecloudflow.sampling.results import GenerationRecord

_POCKET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class RankedMolecule:
    """Represent one successfully docked record in formal rank order.

    :param molecule_id: Stable ``<POCKET_ID>-<RANK:06d>`` identifier.
    :param pocket_id: Sanitized pocket identifier.
    :param rank: One-based formal rank.
    :param canonical_smiles: Canonical isomeric identity.
    :param temporary_id: Generation attempt identifier retained for joins.
    :param docking_score: Numeric Vina-like score in kcal/mol.
    :param qed: Quantitative estimate of drug-likeness, when available.
    :param sa_score: Conventional synthetic-accessibility score, when
        available; lower is easier.
    :param status: Ranking status, normally ``ranked``.
    :param record: Original immutable generation record.
    :return: Immutable ranked row.
    :rtype: RankedMolecule
    """

    molecule_id: str
    pocket_id: str
    rank: int
    canonical_smiles: str
    temporary_id: str
    docking_score: float
    qed: float | None = None
    sa_score: float | None = None
    status: str = "ranked"
    record: GenerationRecord | None = field(default=None, compare=False, repr=False)

    @property
    def vina_score(self) -> float:
        """Return the docking score under the common Vina spelling."""
        return self.docking_score

    @property
    def molecule(self) -> Any:
        """Return the source RDKit molecule when retained by the record."""
        return self.record.molecule if self.record is not None else None

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical tabular row."""
        record = self.record
        return {
            "rank": self.rank,
            "molecule_id": self.molecule_id,
            "pocket_id": self.pocket_id,
            "temporary_id": self.temporary_id,
            "smiles": self.canonical_smiles,
            "canonical_smiles": self.canonical_smiles,
            "isomeric_smiles": self.canonical_smiles,
            "sa": self.sa_score,
            "sa_score": self.sa_score,
            "qed": self.qed,
            "docking_score": self.docking_score,
            "vina_score": self.docking_score,
            "generation_status": self.status,
            "raw_path": str(record.raw_path) if record and record.raw_path else None,
            "relaxed_path": str(record.relaxed_path)
            if record and record.relaxed_path
            else None,
            "seed": record.seed if record else None,
            "model_checkpoint_hash": record.model_checkpoint_hash if record else "",
        }


def rank_molecules(
    pocket_id: str,
    records: Sequence[GenerationRecord],
) -> tuple[list[RankedMolecule], list[GenerationRecord]]:
    """Rank successfully docked unique molecules and assign formal IDs.

    :param pocket_id: Sanitized pocket identifier used as the ID prefix.
    :param records: Valid generation records.  Docking, QED, and SA values
        are read from their ``properties`` mapping under documented aliases.
    :return: ``(ranked, unranked)`` rows.  Missing/failed docking scores retain
        their generation records and never receive a formal rank ID.
    :rtype: tuple[list[RankedMolecule], list[GenerationRecord]]
    :raises ValueError: If the pocket identifier is unsafe or a supplied score
        is non-finite.

    Sorting is ascending docking score, descending QED, ascending conventional
    SA, and finally lexicographic canonical isomeric SMILES.  Missing QED/SA
    values sort after numeric values but do not fabricate a metric.
    """
    _validate_pocket_id(pocket_id)
    prepared: list[tuple[GenerationRecord, float, float | None, float | None]] = []
    unranked: list[GenerationRecord] = []
    for record in records:
        score = _metric(record, "docking_score", "vina_score", "vina", "dock_score")
        if score is None:
            unranked.append(record)
            continue
        qed = _metric(record, "qed", "QED")
        if qed is None:
            qed = _derived_qed(record.molecule)
        sa = _metric(record, "sa", "sa_score", "synthetic_accessibility", "SA")
        if sa is None:
            sa = _derived_sa(record.molecule)
        prepared.append((record, score, qed, sa))
    prepared.sort(
        key=lambda item: (
            item[1],
            -(item[2] if item[2] is not None else float("-inf")),
            item[3] if item[3] is not None else float("inf"),
            item[0].canonical_smiles,
        )
    )
    ranked = [
        RankedMolecule(
            molecule_id=f"{pocket_id}-{rank:06d}",
            pocket_id=pocket_id,
            rank=rank,
            canonical_smiles=record.canonical_smiles,
            temporary_id=record.temporary_id,
            docking_score=score,
            qed=qed,
            sa_score=sa,
            record=record,
        )
        for rank, (record, score, qed, sa) in enumerate(prepared, start=1)
    ]
    return ranked, unranked


def assign_rank_ids(
    pocket_id: str, records: Sequence[GenerationRecord]
) -> tuple[list[RankedMolecule], list[GenerationRecord]]:
    """Compatibility wrapper exposing the explicit rank-assignment operation."""
    return rank_molecules(pocket_id, records)


def _validate_pocket_id(pocket_id: str) -> None:
    """Reject empty/path-like identifiers before they enter artifact names."""
    if not isinstance(pocket_id, str) or not _POCKET_ID.fullmatch(pocket_id):
        raise ValueError("pocket_id must be a non-empty safe identifier.")


def _metric(record: GenerationRecord, *names: str) -> float | None:
    """Read one finite metric from record properties and explicit aliases."""
    values = record.properties
    for name in names:
        if name not in values or values[name] is None:
            continue
        try:
            value = float(values[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"metric {name!r} is not numeric") from error
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
        return value
    return None


def _derived_qed(molecule: Any) -> float | None:
    """Compute QED from a retained RDKit molecule when metadata is absent.

    :param molecule: Candidate molecule retained by a generation record.
    :return: Finite QED value, or ``None`` when no usable molecule exists.
    :rtype: float | None
    """
    if not isinstance(molecule, Chem.Mol):
        return None
    try:
        value = float(QED.qed(molecule))
    except (RuntimeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _derived_sa(molecule: Any) -> float | None:
    """Compute the conventional RDKit Ertl SA score when available.

    :param molecule: Candidate molecule retained by a generation record.
    :return: Conventional 1--10 synthetic-accessibility score, or ``None`` if
        the optional RDKit contribution is unavailable or cannot score it.
    :rtype: float | None

    The contribution is imported lazily because RDKit distributions may omit
    ``Contrib/SA_Score``. Missing optional code remains an explicit missing
    metric rather than a fabricated numeric value.
    """
    if not isinstance(molecule, Chem.Mol):
        return None
    try:
        from rdkit.Contrib.SA_Score import sascorer

        value = float(sascorer.calculateScore(molecule))
    except (ImportError, RuntimeError, ValueError, AttributeError):
        return None
    return value if math.isfinite(value) else None


__all__ = ["RankedMolecule", "assign_rank_ids", "rank_molecules"]
