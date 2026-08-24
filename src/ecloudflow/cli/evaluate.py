"""Evaluation command for completed generation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

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
        timings=_timings_from_records(records),
        metadata={"run_dir": str(run_dir), "profile": profile},
        raw_relaxed_policy=config.evaluation.raw_relaxed_policy,
    )
    result = evaluate_run(
        context,
        registry=MetricRegistry.default(),
        groups=groups,
    )
    rows = [record.as_dict() for record in records]
    aggregate = _aggregate_docking(rows, config.evaluation.bootstrap_seed)
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


def _aggregate_docking(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
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
            observations, seed=seed, value="value"
        ).as_dict()
    }


__all__ = ["evaluate_command"]
