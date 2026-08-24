"""Unified Typer entry point for ECloudFlow workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ecloudflow.cli.benchmark import benchmark_command
from ecloudflow.cli.config import app as config_app
from ecloudflow.cli.data import app as data_app
from ecloudflow.cli.doctor import doctor_command
from ecloudflow.cli.evaluate import evaluate_command
from ecloudflow.cli.report import report_command
from ecloudflow.cli.sample import sample_command
from ecloudflow.cli.train import train_command
from ecloudflow.cli.visualize import app as visualize_app

app = typer.Typer(
    help="Pocket-conditioned 3D molecular generation and evaluation.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(visualize_app, name="visualize")


@app.command("doctor")
def doctor(
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    dataset: Annotated[
        Path | None, typer.Option("--dataset", help="Dataset root or manifest path.")
    ] = None,
    checkpoint: Annotated[
        Path | None, typer.Option("--checkpoint", help="Checkpoint to validate.")
    ] = None,
    require_gpu: Annotated[
        bool, typer.Option("--require-gpu", help="Treat CUDA visibility as required.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable diagnostics.")
    ] = False,
) -> None:
    """Check required Python dependencies and optional scientific tools."""
    doctor_command(
        output_dir,
        as_json,
        dataset=dataset,
        checkpoint=checkpoint,
        require_gpu=require_gpu,
    )


app.command("train")(train_command)
app.command("benchmark")(benchmark_command)
app.command("sample")(sample_command)
app.command("evaluate")(evaluate_command)
app.command("report")(report_command)


if __name__ == "__main__":  # pragma: no cover
    app()
