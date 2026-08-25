"""Typed records for bounded molecular generation runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ecloudflow.sampling.profiles import SamplingProfile


class GenerationMode(str, Enum):
    """Supported de novo and fragment-conditioned generation objectives."""

    DE_NOVO = "de_novo"
    GROW = "grow"
    LINK = "link"
    REPLACE = "replace"
    MERGE = "merge"


class GenerationShortfallError(RuntimeError):
    """Raised when strict generation cannot meet its bounded target count.

    The completed :class:`GenerationResult` is attached to ``result`` so a
    caller can inspect every accepted, rejected, and failed attempt even when
    strict mode deliberately turns a shortfall into an exception.
    """

    def __init__(self, message: str, result: GenerationResult | None = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class GenerationRequest:
    """Bundle one public generation request for reuse across modes.

    :param pocket: Pocket path or already parsed pocket object.
    :param num_molecules: Number of valid unique molecules requested.
    :param fragment: Optional positioned fragment path/object or a non-empty
        sequence of paths/objects for link and merge tasks.
    :param mode: De novo, grow, link, replace, or merge objective.
    :param profile: Named sampling profile.
    :param max_attempts: Optional bounded attempt budget.
    :param output_dir: Optional artifact directory.
    :param seed: Master deterministic seed.
    :param strict_count: Raise :class:`GenerationShortfallError` on shortfall.
    :return: Immutable request.
    :rtype: GenerationRequest
    """

    pocket: Any
    num_molecules: int
    fragment: Any = None
    mode: GenerationMode = GenerationMode.DE_NOVO
    profile: str | SamplingProfile = "balanced"
    max_attempts: int | None = None
    output_dir: str | Path | None = None
    seed: int = 2026
    strict_count: bool = False

    def __post_init__(self) -> None:
        """Validate count, mode, and attempt-budget semantics."""
        if isinstance(self.num_molecules, bool) or not isinstance(
            self.num_molecules, int
        ):
            raise TypeError("num_molecules must be an integer.")
        if self.num_molecules < 1:
            raise ValueError("num_molecules must be positive.")
        try:
            mode = (
                self.mode
                if isinstance(self.mode, GenerationMode)
                else GenerationMode(self.mode)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown generation mode: {self.mode!r}") from error
        object.__setattr__(self, "mode", mode)
        if self.max_attempts is not None:
            if isinstance(self.max_attempts, bool) or not isinstance(
                self.max_attempts, int
            ):
                raise TypeError("max_attempts must be an integer.")
            if self.max_attempts < 1:
                raise ValueError("max_attempts must be positive.")
        if not isinstance(self.strict_count, bool):
            raise TypeError("strict_count must be boolean.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if isinstance(self.profile, SamplingProfile):
            profile: str | SamplingProfile = self.profile
        elif isinstance(self.profile, str):
            profile = self.profile.lower()
            if profile not in {"fast", "balanced", "quality"}:
                raise ValueError(f"unknown sampling profile: {self.profile!r}")
        else:
            raise TypeError("profile must be a string or SamplingProfile.")
        object.__setattr__(self, "profile", profile)
        if self.mode is GenerationMode.DE_NOVO and self.fragment is not None:
            raise ValueError("fragment is only valid for fragment-conditioned modes.")
        if self.mode is not GenerationMode.DE_NOVO and self.fragment is None:
            raise ValueError(f"mode {self.mode.value!r} requires a fragment.")


@dataclass(frozen=True)
class GenerationRecord:
    """Describe one valid unique generated molecule.

    :param canonical_smiles: Canonical isomeric SMILES used for uniqueness.
    :param attempt_id: Stable temporary attempt identifier.
    :param molecule: Optional defensive RDKit molecule containing the raw pose.
    :param mode: Generation objective used for this record.
    :param seed: Per-attempt seed, if available.
    :param raw_path: Raw pose artifact path, if written.
    :param relaxed_path: Separate relaxed pose artifact path, if written.
    :param properties: Additional serializable metrics/provenance.
    :param model_checkpoint_hash: Hash of the checkpoint used for generation.
    :return: Immutable record with read-only properties.
    :rtype: GenerationRecord
    """

    canonical_smiles: str
    attempt_id: str = ""
    molecule: Any = field(default=None, compare=False, repr=False)
    mode: GenerationMode = GenerationMode.DE_NOVO
    seed: int | None = None
    raw_path: Path | None = None
    relaxed_path: Path | None = None
    properties: Mapping[str, Any] = field(default_factory=dict, compare=False)
    model_checkpoint_hash: str = ""

    def __post_init__(self) -> None:
        """Validate stable identity fields and freeze the property mapping."""
        if not isinstance(self.canonical_smiles, str) or not self.canonical_smiles:
            raise ValueError("canonical_smiles must be a non-empty string.")
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("attempt_id must be a non-empty string.")
        mode = (
            self.mode
            if isinstance(self.mode, GenerationMode)
            else GenerationMode(self.mode)
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("record seed must be an integer or None.")
        if self.raw_path is not None:
            object.__setattr__(self, "raw_path", Path(self.raw_path))
        if self.relaxed_path is not None:
            object.__setattr__(self, "relaxed_path", Path(self.relaxed_path))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary without serializing the RDKit object."""
        return {
            "attempt_id": self.attempt_id,
            "smiles": self.canonical_smiles,
            "canonical_smiles": self.canonical_smiles,
            "mode": self.mode.value,
            "seed": self.seed,
            "raw_path": str(self.raw_path) if self.raw_path is not None else None,
            "relaxed_path": str(self.relaxed_path)
            if self.relaxed_path is not None
            else None,
            "model_checkpoint_hash": self.model_checkpoint_hash,
            **dict(self.properties),
        }

    @property
    def isomeric_smiles(self) -> str:
        """Return the canonical isomeric SMILES identity."""
        return self.canonical_smiles

    @property
    def smiles(self) -> str:
        """Return a compatibility alias for :attr:`canonical_smiles`."""
        return self.canonical_smiles

    @property
    def temporary_id(self) -> str:
        """Return the pre-ranking identifier used by docking outputs."""
        return self.attempt_id


@dataclass(frozen=True)
class GenerationAttempt:
    """Record every bounded attempt, including duplicate and failed attempts.

    :param attempt_id: Stable one-based attempt identifier.
    :param status: One of ``valid``, ``rejected``, or ``failed``.
    :param record: Valid record when status is ``valid``.
    :param reason: Structured rejection/failure reason.
    :param seed: Per-attempt deterministic seed.
    :param elapsed_seconds: Wall time spent on this attempt.
    :return: Immutable attempt record.
    :rtype: GenerationAttempt
    :raises ValueError: If status or record semantics are inconsistent.
    """

    attempt_id: str
    status: str
    record: GenerationRecord | None = None
    reason: str = ""
    seed: int | None = None
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validate status and valid-record invariants."""
        if self.status not in {"valid", "rejected", "failed"}:
            raise ValueError("attempt status must be valid, rejected, or failed.")
        if not self.attempt_id:
            raise ValueError("attempt_id must be non-empty.")
        if self.status == "valid" and self.record is None:
            raise ValueError("valid attempts require a GenerationRecord.")
        if self.status != "valid" and self.record is not None:
            raise ValueError("rejected/failed attempts cannot carry a valid record.")
        if not math.isfinite(float(self.elapsed_seconds)) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe attempt summary."""
        return {
            "attempt_id": self.attempt_id,
            "status": self.status,
            "reason": self.reason,
            "seed": self.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "record": self.record.as_dict() if self.record is not None else None,
        }


@dataclass(frozen=True)
class GenerationResult:
    """Summarize valid molecules and all bounded generation attempts.

    :param valid: Ordered tuple of unique valid records.
    :param attempt_records: Every completed attempt in execution order.
    :param target_count: Requested number of valid unique molecules.
    :param duplicate_count: Number of candidates rejected as duplicates.
    :param model_checkpoint_hash: Checkpoint provenance hash.
    :param output_dir: Optional run artifact directory.
    :param mode: Generation objective.
    :return: Immutable generation summary.
    :rtype: GenerationResult
    """

    valid: tuple[GenerationRecord, ...] = ()
    attempt_records: tuple[GenerationAttempt, ...] = ()
    target_count: int = 0
    duplicate_count: int = 0
    model_checkpoint_hash: str = ""
    output_dir: Path | None = None
    mode: GenerationMode = GenerationMode.DE_NOVO

    def __post_init__(self) -> None:
        """Normalize sequence fields and validate aggregate counts."""
        object.__setattr__(self, "valid", tuple(self.valid))
        object.__setattr__(self, "attempt_records", tuple(self.attempt_records))
        object.__setattr__(
            self,
            "mode",
            self.mode
            if isinstance(self.mode, GenerationMode)
            else GenerationMode(self.mode),
        )
        if self.target_count < 0 or self.duplicate_count < 0:
            raise ValueError("target_count and duplicate_count must be non-negative.")
        if self.duplicate_count > self.rejected_count:
            raise ValueError("duplicate_count cannot exceed rejected attempts.")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def attempts(self) -> int:
        """Return the number of completed bounded attempts."""
        return len(self.attempt_records)

    @property
    def valid_count(self) -> int:
        """Return the number of valid unique molecules."""
        return len(self.valid)

    @property
    def shortfall(self) -> int:
        """Return the number of requested molecules not obtained."""
        return max(0, self.target_count - self.valid_count)

    @property
    def rejected_count(self) -> int:
        """Return the number of rejected attempts."""
        return sum(attempt.status == "rejected" for attempt in self.attempt_records)

    @property
    def failed_count(self) -> int:
        """Return the number of failed attempts."""
        return sum(attempt.status == "failed" for attempt in self.attempt_records)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe complete run summary."""
        return {
            "target_count": self.target_count,
            "valid_count": self.valid_count,
            "shortfall": self.shortfall,
            "attempts": self.attempts,
            "duplicate_count": self.duplicate_count,
            "rejected_count": self.rejected_count,
            "failed_count": self.failed_count,
            "mode": self.mode.value,
            "model_checkpoint_hash": self.model_checkpoint_hash,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "valid": [record.as_dict() for record in self.valid],
            "attempt_records": [attempt.as_dict() for attempt in self.attempt_records],
        }

    def to_json(self, path: str | Path) -> Path:
        """Write the complete JSON-safe result using an atomic replacement.

        :param path: Destination JSON path; parent directories are created.
        :return: Written destination path.
        :rtype: pathlib.Path
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        temporary.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(destination)
        return destination

    def to_excel(self, path: str | Path) -> Path:
        """Write a compact ranked/attempt workbook.

        :param path: Destination workbook path.  Parent directories are created.
        :return: Written path.
        :rtype: pathlib.Path
        :raises RuntimeError: If neither pandas/openpyxl is installed.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = [record.as_dict() for record in self.valid]
        attempts = [attempt.as_dict() for attempt in self.attempt_records]
        try:
            import pandas as pd

            with pd.ExcelWriter(destination) as writer:
                pd.DataFrame(rows).to_excel(writer, sheet_name="valid", index=False)
                pd.DataFrame(attempts).to_excel(
                    writer, sheet_name="attempts", index=False
                )
            return destination
        except ImportError:
            try:
                from openpyxl import Workbook
            except ImportError as error:
                raise RuntimeError("to_excel requires pandas or openpyxl.") from error
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "valid"
            _write_rows(sheet, rows)
            attempt_sheet = workbook.create_sheet("attempts")
            _write_rows(attempt_sheet, attempts)
            workbook.save(destination)
            return destination


def _write_rows(sheet: Any, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to an openpyxl worksheet without pandas."""
    keys = list(rows[0]) if rows else ["status"]
    sheet.append(keys)
    for row in rows:
        sheet.append([row.get(key) for key in keys])


__all__ = [
    "GenerationAttempt",
    "GenerationMode",
    "GenerationRecord",
    "GenerationRequest",
    "GenerationResult",
    "GenerationShortfallError",
]
