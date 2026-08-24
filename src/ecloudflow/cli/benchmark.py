"""Command-line wrapper for distributed scaling measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ecloudflow.training.benchmark import benchmark_scaling


def benchmark_command(
    devices: Annotated[
        list[int] | None,
        typer.Option(
            "--devices", help="Device counts, for example --devices 1 --devices 2."
        ),
    ] = None,
    steps: Annotated[int, typer.Option("--steps", min=1)] = 10,
    config: Annotated[str, typer.Option("--config")] = "experiment=h100_smoke",
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "runs/benchmark"
    ),
    warmup_steps: Annotated[int | None, typer.Option("--warmup-steps", min=0)] = None,
    global_batch_size: Annotated[
        int | None, typer.Option("--global-batch-size", min=1)
    ] = None,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Use deterministic synthetic work and label the report as a dry run.",
        ),
    ] = False,
) -> None:
    """Measure fixed-workload throughput, memory, scaling, and NFE."""
    report = benchmark_scaling(
        devices or (1, 2, 4),
        steps,
        config,
        output_dir,
        warmup_steps=warmup_steps,
        global_batch_size=global_batch_size,
        profile=profile,
        dry_run=dry_run,
    )
    for row in report.rows:
        typer.echo(
            f"devices={row.devices} samples/s={row.samples_per_second:.3f} "
            f"speedup={row.speedup:.3f} efficiency={row.scaling_efficiency:.3f}"
        )
    for path in report.paths:
        typer.echo(str(path))


__all__ = ["benchmark_command"]
