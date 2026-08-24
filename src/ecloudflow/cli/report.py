"""HTML and publication report command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ecloudflow.cli.common import load_run_payload
from ecloudflow.visualization import build_report


def report_command(
    run_dir: Annotated[
        Path, typer.Argument(help="Run directory containing evaluation artifacts.")
    ],
    top_n: Annotated[int, typer.Option("--top-n", min=1)] = 20,
    format_name: Annotated[
        str,
        typer.Option("--format", help="Report style: html or paper."),
    ] = "html",
) -> None:
    """Build a self-contained HTML report and publication figures.

    :param run_dir: Existing generation/evaluation run directory.
    :param top_n: Number of rows shown in the report table.
    :param format_name: ``paper`` keeps the same data while emphasizing static
        SVG/PDF/PNG artifacts; ``html`` is the default interactive handoff.
    :return: None; report artifacts are written beside the run.
    :rtype: None
    """
    if not run_dir.is_dir():
        raise typer.BadParameter(f"run directory does not exist: {run_dir}")
    if format_name not in {"html", "paper"}:
        raise typer.BadParameter("--format must be html or paper")
    payload = load_run_payload(run_dir)
    if not payload:
        raise typer.BadParameter("run directory contains no reportable JSON artifacts")
    bundle = build_report(payload, run_dir, top_n=top_n)
    typer.echo(f"report: {bundle.html_path}")
    if format_name == "paper":
        for path in bundle.paths:
            if path.suffix.lower() in {".svg", ".pdf", ".png"}:
                typer.echo(str(path))


__all__ = ["report_command"]
