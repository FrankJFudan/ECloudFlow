"""Evaluation command for completed generation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rdkit import Chem

from ecloudflow.cli.common import atomic_write_json, load_run_records, merge_overrides
from ecloudflow.config import load_config
from ecloudflow.evaluation import (
    EvaluationContext,
    MetricRegistry,
    bootstrap_macro_summary,
    evaluate_run,
)


def evaluate_command(
    run_dir: Annotated[Path, typer.Argument(help="Generation run directory.")],
    profile: Annotated[str, typer.Option("--profile")] = "default",
    overrides: Annotated[
        list[str] | None,
        typer.Option("--override", "-O", help="Repeatable Hydra key=value override."),
    ] = None,
    trailing: Annotated[
        list[str] | None,
        typer.Argument(help="Optional trailing key=value overrides."),
    ] = None,
) -> None:
    """Compute registered metrics for an existing run without mutating poses.

    :param run_dir: Directory containing generation, ranked, or CSV artifacts.
    :param profile: ``smoke`` limits the report to inexpensive local metrics;
        ``default`` enables all seven metric domains.
    :param overrides: Optional evaluation configuration overrides.
    :param trailing: Optional positional Hydra overrides.
    :return: None; writes ``evaluation.json`` and prints metric statuses.
    :rtype: None
    """
    if not run_dir.is_dir():
        raise typer.BadParameter(f"run directory does not exist: {run_dir}")
    try:
        override_values = merge_overrides(overrides, trailing)
        config = load_config(override_values)
    except (KeyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    records = load_run_records(run_dir)
    if not records:
        raise typer.BadParameter(
            "run directory contains no generation.json, summary.json ranked rows, or samples.csv"
        )
    if profile == "smoke":
        groups = ("chemistry", "conditional", "efficiency")
    elif profile in {"default", "full"}:
        groups = tuple(config.evaluation.groups)
    else:
        raise typer.BadParameter("profile must be default, full, or smoke")
    context = EvaluationContext(
        records=records,
        references=_load_references(config.evaluation.reference_path),
        timings=_timings_from_records(records),
        metadata={
            "run_dir": str(run_dir),
            "profile": profile,
            "optional_backends": dict(config.evaluation.optional_backends),
        },
        raw_relaxed_policy=config.evaluation.raw_relaxed_policy,
    )
    result = evaluate_run(
        context,
        registry=MetricRegistry.default(),
        groups=groups,
    )
    rows = [record.as_dict() for record in records]
    aggregate = _aggregate_docking(
        rows,
        config.evaluation.bootstrap_seed,
        config.evaluation.bootstrap_resamples,
    )
    destination = run_dir / "evaluation.json"
    atomic_write_json(
        destination,
        {
            **result.as_dict(),
            "rows": rows,
            "aggregate": aggregate,
            "config": config.evaluation.model_dump(mode="json"),
        },
    )
    typer.echo(f"evaluation: {destination}")
    for metric in result.results.values():
        typer.echo(f"{metric.name}: {metric.status.value}")


def _timings_from_records(records: tuple[Any, ...]) -> dict[str, Any]:
    """Extract explicit attempt timings without estimating missing values."""
    elapsed = [
        float(record.properties["elapsed_seconds"])
        for record in records
        if record.properties.get("elapsed_seconds") is not None
    ]
    return (
        {
            "valid_count": len(records),
            "mean_elapsed_seconds": sum(elapsed) / len(elapsed),
        }
        if elapsed
        else {"valid_count": len(records)}
    )


def _aggregate_docking(
    rows: list[dict[str, Any]], seed: int, resamples: int = 1000
) -> dict[str, Any]:
    """Return a pocket-macro docking summary when scores are available."""
    observations = []
    for row in rows:
        value = row.get("docking_score", row.get("vina_score"))
        if value is None:
            continue
        try:
            observations.append(
                {
                    "pocket_id": row.get("pocket_id", "default"),
                    "value": float(value),
                }
            )
        except (TypeError, ValueError):
            continue
    if not observations:
        return {"docking_score": None}
    return {
        "docking_score": bootstrap_macro_summary(
            observations, seed=seed, resamples=resamples, value="value"
        ).as_dict()
    }


def _load_references(path_value: str | None) -> dict[str, tuple[str, ...]]:
    """Load a canonical reference-SMILES index from common research formats.

    :param path_value: Optional JSON, CSV, SMI/TXT, or SDF file configured by
        ``evaluation.reference_path``.
    :return: Mapping containing a stable sorted tuple under ``"smiles"``.
    :rtype: dict[str, tuple[str, ...]]
    :raises typer.BadParameter: If the path, format, or molecule records are
        unreadable or contain no valid SMILES.

    JSON accepts either a top-level sequence or a mapping with ``smiles`` or
    ``canonical_smiles``. CSV selects either named column. SMI/TXT uses the
    first whitespace-delimited token on each non-comment line. SDF records are
    sanitized by RDKit. Every molecule is canonicalized isomerically and
    deduplicated; source files are read only and no reference is silently
    replaced after a parse failure.
    """
    if path_value is None:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise typer.BadParameter(f"evaluation reference file does not exist: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                values = payload.get("smiles", payload.get("canonical_smiles", ()))
            else:
                values = payload
            if isinstance(values, str):
                raw = [values]
            elif isinstance(values, (list, tuple, set)):
                raw = [str(value) for value in values]
            else:
                raise ValueError("JSON must contain a SMILES sequence")
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            raw = [
                str(row.get("canonical_smiles") or row.get("smiles") or "")
                for row in rows
            ]
        elif suffix in {".smi", ".smiles", ".txt"}:
            raw = [
                line.split()[0]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        elif suffix == ".sdf":
            molecules = list(Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True))
            if any(molecule is None for molecule in molecules):
                raise ValueError("SDF contains an invalid molecule record")
            raw = [
                Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
                for molecule in molecules
                if molecule is not None
            ]
        else:
            raise ValueError("supported suffixes are .json, .csv, .smi, .txt, and .sdf")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise typer.BadParameter(f"invalid evaluation reference file: {error}") from error
    canonical: set[str] = set()
    for value in raw:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise typer.BadParameter(f"invalid reference SMILES: {value!r}")
        canonical.add(
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        )
    if not canonical:
        raise typer.BadParameter("evaluation reference file contains no molecules")
    return {"smiles": tuple(sorted(canonical))}


__all__ = ["evaluate_command"]
