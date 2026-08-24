"""Training command preflight and service handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ecloudflow.cli.common import atomic_write_json, merge_overrides
from ecloudflow.config import load_config


def train_command(
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "runs/train"
    ),
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1)] = None,
    overrides: Annotated[list[str] | None, typer.Option("--override", "-O")] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Write resolved settings without starting Lightning."
        ),
    ] = False,
    trailing: Annotated[
        list[str] | None, typer.Argument(help="Optional trailing key=value overrides.")
    ] = None,
) -> None:
    """Resolve training settings and hand off to the Lightning application service.

    :param output_dir: Checkpoint/log destination.
    :param max_steps: Optional bounded smoke-run step override.
    :param overrides: Optional Hydra configuration overrides.
    :param dry_run: Persist preflight only when true.
    :return: None; preflight metadata is always written.
    :rtype: None

    The command keeps dataset/model construction in the training service. This
    lightweight entry point is deterministic and smoke-friendly while making
    the resolved launch contract explicit.
    """
    try:
        config = load_config(merge_overrides(overrides, trailing))
    except (KeyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    if max_steps is not None:
        config = config.model_copy(
            update={
                "trainer": config.trainer.model_copy(update={"max_steps": max_steps})
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    launch = output_dir / "train-config.json"
    atomic_write_json(
        launch,
        {
            "dry_run": dry_run,
            "config": config.model_dump(mode="json"),
            "status": "preflight" if dry_run else "handoff",
        },
    )
    typer.echo(f"training preflight: {launch}")
    if not dry_run:
        typer.echo(
            "training service handoff is configured; invoke the Lightning runner for execution"
        )
