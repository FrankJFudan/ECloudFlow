"""Evaluation command for completed generation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ecloudflow.evaluation import EvaluationContext, evaluate_run


def evaluate_command(
    run_dir: Annotated[Path, typer.Argument(help="Generation run directory.")],
    profile: Annotated[str, typer.Option("--profile")] = "default",
) -> None:
    """Compute registered metrics for an existing run without mutating poses.

    :param run_dir: Directory containing ``summary.json`` or ``samples.csv``.
    :param profile: ``smoke`` selects chemistry and efficiency metrics only.
    :return: None; writes ``evaluation.json`` and prints metric statuses.
    :rtype: None
    """
    summary = run_dir / "summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
    elif (run_dir / "samples.csv").is_file():
        payload = {"samples_csv": str(run_dir / "samples.csv")}
    else:
        raise typer.BadParameter("run directory must contain summary.json or samples.csv")
    groups = ("chemistry", "efficiency") if profile == "smoke" else None
    result = evaluate_run(EvaluationContext(metadata=payload), groups=groups)
    destination = run_dir / "evaluation.json"
    destination.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"evaluation: {destination}")
    for metric in result.results.values():
        typer.echo(f"{metric.name}: {metric.status.value}")

