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
    """Load generated/ranked rows from the standard run artifact formats.

    Ranked rows are preferred when ``summary.json`` is present because they
    already carry docking, QED, and SA values.  A generation manifest or CSV
    remains a valid input for runs that stopped before docking.
    """
    directory = Path(run_dir)
    summary_path = directory / "summary.json"
    generation_path = directory / "generation.json"
    if summary_path.is_file():
        payload = read_json(summary_path)
        rows = payload.get("ranked", []) if isinstance(payload, dict) else []
        if rows:
            return _records_from_rows(rows, directory)
    if generation_path.is_file():
        payload = read_json(generation_path)
        rows = payload.get("valid", []) if isinstance(payload, dict) else []
        if rows:
            return _records_from_rows(rows, directory)
    csv_path = directory / "samples.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return _records_from_rows(list(csv.DictReader(handle)), directory)
    return ()


def load_run_payload(run_dir: str | Path) -> dict[str, Any]:
    """Load the most informative JSON artifact for report/evaluation input."""
    directory = Path(run_dir)
    for name in ("evaluation.json", "summary.json", "generation.json"):
        path = directory / name
        if path.is_file():
            payload = read_json(path)
            if isinstance(payload, dict):
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
            "rank",
            "molecule_id",
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
