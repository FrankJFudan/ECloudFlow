"""HTML report command for completed evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ecloudflow.visualization import build_report


def report_command(
    run_dir: Annotated[Path, typer.Argument(help="Run directory containing evaluation.json.")],
    top_n: Annotated[int, typer.Option("--top-n", min=1)] = 20,
) -> None:
    """Build a self-contained HTML report from existing evaluation data.

    :param run_dir: Existing run directory.
    :param top_n: Number of rows shown in the report table.
    :return: None; report artifacts are written beside the run.
    :rtype: None
    """
    source = run_dir / "evaluation.json"
    if source.is_file():
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        summary = run_dir / "summary.json"
        if not summary.is_file():
            raise typer.BadParameter("run directory must contain evaluation.json or summary.json")
        payload = json.loads(summary.read_text(encoding="utf-8"))
    bundle = build_report(payload, run_dir, top_n=top_n)
    typer.echo(f"report: {bundle.html_path}")

