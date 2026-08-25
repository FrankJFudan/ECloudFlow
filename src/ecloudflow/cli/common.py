"""Shared adapters for the command-line workflow.

The command modules deliberately keep orchestration thin.  This module owns
the small amount of file-format normalization needed to pass the same typed
records between generation, evaluation, and reporting commands.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem

from ecloudflow.sampling.results import GenerationMode, GenerationRecord


def merge_overrides(
    repeated: Sequence[str] | None = None, trailing: Sequence[str] | None = None
) -> list[str]:
    """Merge option-based and positional Hydra overrides in input order.

    :param repeated: Values supplied through one or more ``--override`` flags.
    :param trailing: Bare ``key=value`` arguments after the command options.
    :return: New list suitable for :func:`ecloudflow.config.load_config`.
    :rtype: list[str]
    :raises ValueError: If an override is not a Hydra assignment.
    """
    values = [str(item) for item in (repeated or ())] + [
        str(item) for item in (trailing or ())
    ]
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"configuration override must use key=value syntax: {value!r}"
            )
    # ``experiment`` is an optional Hydra defaults entry in this repository.
    # Accept the user-facing shorthand documented by the CLI while retaining
    # strict Hydra validation for every other key.
    return [
        f"+{value}" if value.startswith("experiment=") else value for value in values
    ]


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Write deterministic JSON through a sibling temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


def read_json(path: str | Path) -> Any:
    """Read one UTF-8 JSON document with a useful path-aware error."""
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read JSON document {source}: {error}") from error


def load_run_records(run_dir: str | Path) -> tuple[GenerationRecord, ...]:
    """Load one run's generation records without dropping unranked molecules.

    :param run_dir: Standard ECloudFlow output directory containing
        ``generation.json``, ``summary.json``, or ``samples.csv`` artifacts.
    :return: Immutable generation records in stable generation order.
    :rtype: tuple[GenerationRecord, ...]

    ``summary.json`` often contains only the docked or ranked subset of an
    earlier ``generation.json`` manifest.  This loader therefore treats the
    generation manifest as the authoritative ordered list of valid molecules
    and uses ranked rows only to enrich matching records with docking/ranking
    properties such as ``docking_score``, ``qed``, ``sa``, ``rank``, and
    ``molecule_id``.  Runs that stopped before ranking still load cleanly from
    ``generation.json`` or ``samples.csv``.  Duplicate rows across artifacts
    are coalesced rather than emitted twice.
    """
    directory = Path(run_dir)
    summary_path = directory / "summary.json"
    generation_path = directory / "generation.json"
    summary_rows: list[Mapping[str, Any]] = []
    if summary_path.is_file():
        payload = read_json(summary_path)
        rows = payload.get("ranked", []) if isinstance(payload, dict) else []
        if isinstance(rows, list):
            summary_rows = [row for row in rows if isinstance(row, Mapping)]
    if generation_path.is_file():
        payload = read_json(generation_path)
        rows = payload.get("valid", []) if isinstance(payload, dict) else []
        if isinstance(rows, list) and rows:
            generation_rows = [row for row in rows if isinstance(row, Mapping)]
            attempt_rows = (
                payload.get("attempt_records", [])
                if isinstance(payload, dict)
                else []
            )
            return _records_from_rows(
                _merge_generation_and_summary_rows(
                    _attach_valid_attempt_timings(generation_rows, attempt_rows),
                    summary_rows,
                ),
                directory,
            )
    if summary_rows:
        return _records_from_rows(summary_rows, directory)
    csv_path = directory / "samples.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return _records_from_rows(list(csv.DictReader(handle)), directory)
    return ()


def load_run_payload(run_dir: str | Path) -> dict[str, Any]:
    """Load report data while preserving normalized molecular measurement rows.

    :param run_dir: Standard ECloudFlow output directory.
    :return: The preferred JSON payload, optionally enriched with a ``rows``
        list reconstructed from generation and ranking artifacts.
    :rtype: dict[str, typing.Any]

    A completed evaluation already owns its explicitly computed rows and is
    returned unchanged. Before evaluation, the preferred ``summary.json``
    contains ranking fields but not per-attempt timing; an accompanying
    ``generation.json`` contains that timing only under ``attempt_records``.
    Reusing :func:`load_run_records` for non-evaluation payloads joins the two
    authoritative artifacts without replacing their original provenance or
    fabricating a speed measurement.
    """
    directory = Path(run_dir)
    for name in ("evaluation.json", "summary.json", "generation.json"):
        path = directory / name
        if path.is_file():
            payload = read_json(path)
            if isinstance(payload, dict):
                if name == "evaluation.json" and isinstance(payload.get("rows"), list):
                    return payload
                records = load_run_records(directory)
                if records:
                    enriched = dict(payload)
                    enriched["rows"] = [record.as_dict() for record in records]
                    return enriched
                return payload
    return {"rows": [record.as_dict() for record in load_run_records(directory)]}


def find_ranked_row(run_dir: str | Path, molecule_id: str) -> dict[str, Any] | None:
    """Find one ranked row by its formal ID without mutating run artifacts."""
    directory = Path(run_dir)
    payload: Any = {}
    summary = directory / "summary.json"
    if summary.is_file():
        payload = read_json(summary)
    if not isinstance(payload, dict) or not payload:
        payload = load_run_payload(directory)
    for key in ("ranked", "rows", "molecules"):
        rows = payload.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping) and row.get("molecule_id") == molecule_id:
                    return dict(row)
    return None


def json_safe(value: Any) -> Any:
    """Convert nested scientific values into finite JSON primitives."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def _records_from_rows(
    rows: Iterable[Mapping[str, Any]], directory: Path
) -> tuple[GenerationRecord, ...]:
    """Convert tabular rows into immutable generation records."""
    records: list[GenerationRecord] = []
    for index, row in enumerate(rows, start=1):
        smiles = (
            row.get("canonical_smiles")
            or row.get("isomeric_smiles")
            or row.get("smiles")
        )
        if not isinstance(smiles, str) or not smiles:
            continue
        attempt_id = str(
            row.get("temporary_id") or row.get("attempt_id") or f"row-{index:06d}"
        )
        mode_value = row.get("mode", GenerationMode.DE_NOVO.value)
        try:
            mode = GenerationMode(mode_value)
        except (TypeError, ValueError):
            mode = GenerationMode.DE_NOVO
        molecule = _load_molecule(row, directory, smiles)
        known = {
            "canonical_smiles",
            "isomeric_smiles",
            "smiles",
            "temporary_id",
            "attempt_id",
            "mode",
            "raw_path",
            "relaxed_path",
            "seed",
            "model_checkpoint_hash",
        }
        properties = {
            str(key): json_safe(value) for key, value in row.items() if key not in known
        }
        raw_path = _path_value(row.get("raw_path"), directory)
        relaxed_path = _path_value(row.get("relaxed_path"), directory)
        records.append(
            GenerationRecord(
                canonical_smiles=smiles,
                attempt_id=attempt_id,
                molecule=molecule,
                mode=mode,
                seed=_optional_int(row.get("seed")),
                raw_path=raw_path,
                relaxed_path=relaxed_path,
                properties=properties,
                model_checkpoint_hash=str(row.get("model_checkpoint_hash") or ""),
            )
        )
    return tuple(records)


def _merge_generation_and_summary_rows(
    generation_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Overlay ranked metadata onto generation rows without reordering them.

    :param generation_rows: Ordered valid rows from ``generation.json``.
    :param summary_rows: Ranked or docked subset rows from ``summary.json``.
    :return: Merged rows preserving generation order and appending only truly
        unmatched ranked rows.
    :rtype: tuple[Mapping[str, Any], ...]
    """
    if not summary_rows:
        return tuple(dict(row) for row in generation_rows)
    ranked_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    ranked_only: list[dict[str, Any]] = []
    for row in summary_rows:
        normalized = dict(row)
        key = _row_identifier(normalized)
        if key is None:
            ranked_only.append(normalized)
            continue
        ranked_by_key[key] = normalized
    merged: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in generation_rows:
        normalized = dict(row)
        key = _row_identifier(normalized)
        if key is not None and key in ranked_by_key:
            normalized = _merge_row(normalized, ranked_by_key[key])
            seen.add(key)
        elif key is not None:
            seen.add(key)
        merged.append(normalized)
    for row in ranked_only:
        key = _row_identifier(row)
        if key is None or key not in seen:
            merged.append(row)
            if key is not None:
                seen.add(key)
    for key, row in ranked_by_key.items():
        if key not in seen:
            merged.append(row)
            seen.add(key)
    return tuple(merged)


def _attach_valid_attempt_timings(
    generation_rows: Sequence[Mapping[str, Any]], attempt_rows: Any
) -> tuple[Mapping[str, Any], ...]:
    """Attach measured elapsed time from valid attempt events to molecule rows.

    :param generation_rows: Ordered valid molecule rows from the generation
        manifest.
    :param attempt_rows: Raw ``attempt_records`` payload from that manifest.
    :return: New valid rows with finite non-negative ``elapsed_seconds`` values
        overlaid by attempt identifier when the molecule row did not already
        carry one.
    :rtype: tuple[collections.abc.Mapping[str, typing.Any], ...]

    Generation records intentionally describe molecular outputs, while attempt
    records own wall-clock measurements. Joining them here preserves the
    authoritative generation order and gives evaluation/reporting a real speed
    observation without inventing an estimate for failed or missing attempts.
    Malformed timing entries are ignored because their source manifest remains
    available for forensic inspection and cannot be converted into a valid
    scientific measurement.
    """
    if not isinstance(attempt_rows, list):
        return tuple(dict(row) for row in generation_rows)
    timings: dict[str, float] = {}
    for attempt in attempt_rows:
        if not isinstance(attempt, Mapping) or attempt.get("status") != "valid":
            continue
        attempt_id = attempt.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            continue
        try:
            elapsed_seconds = float(attempt.get("elapsed_seconds"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(elapsed_seconds) and elapsed_seconds >= 0.0:
            timings[attempt_id] = elapsed_seconds
    enriched: list[Mapping[str, Any]] = []
    for row in generation_rows:
        normalized = dict(row)
        attempt_id = normalized.get("attempt_id")
        if (
            "elapsed_seconds" not in normalized
            and isinstance(attempt_id, str)
            and attempt_id in timings
        ):
            normalized["elapsed_seconds"] = timings[attempt_id]
        enriched.append(normalized)
    return tuple(enriched)


def _row_identifier(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return the strongest stable row identifier available for joins."""
    for name in ("temporary_id", "attempt_id"):
        value = row.get(name)
        if isinstance(value, str) and value:
            return "attempt_id", value
    molecule_id = row.get("molecule_id")
    if isinstance(molecule_id, str) and molecule_id:
        return "molecule_id", molecule_id
    smiles = row.get("canonical_smiles") or row.get("isomeric_smiles") or row.get("smiles")
    if isinstance(smiles, str) and smiles:
        return "canonical_smiles", smiles
    return None


def _merge_row(
    generation_row: Mapping[str, Any], summary_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge one ranked row into its generation source without losing fields."""
    merged = dict(generation_row)
    for key, value in summary_row.items():
        if value not in (None, "", "nan"):
            merged[str(key)] = value
    return merged


def _load_molecule(
    row: Mapping[str, Any], directory: Path, smiles: str
) -> Chem.Mol | None:
    """Load a raw SDF pose when available, otherwise parse the SMILES."""
    path = _path_value(row.get("raw_path"), directory)
    if path is not None and path.is_file():
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        molecule = next((item for item in supplier if item is not None), None)
        if molecule is not None:
            return molecule
    return Chem.MolFromSmiles(smiles)


def _path_value(value: Any, directory: Path) -> Path | None:
    """Resolve a serialized relative path against its run directory."""
    if value in (None, "", "nan"):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else directory / path


def _optional_int(value: Any) -> int | None:
    """Parse a nullable integer emitted by JSON or CSV."""
    if value in (None, "", "nan"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "atomic_write_json",
    "find_ranked_row",
    "json_safe",
    "load_run_payload",
    "load_run_records",
    "merge_overrides",
    "read_json",
]
