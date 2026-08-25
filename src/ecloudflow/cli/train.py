"""Executable training command with a side-effect-free dry-run preflight."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from ecloudflow.cli.common import atomic_write_json, merge_overrides
from ecloudflow.config import load_config
from ecloudflow.exceptions import DataValidationError
from ecloudflow.training import CheckpointStateError
from ecloudflow.training.runtime import (
    TrainingConfigurationError,
    run_training,
)


def train_command(
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "runs/train"
    ),
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1)] = None,
    resume_from: Annotated[
        Path | None,
        typer.Option(
            "--resume-from",
            help="Resume model, optimizer, EMA, RNG, and data progress from a checkpoint.",
        ),
    ] = None,
    init_from: Annotated[
        Path | None,
        typer.Option(
            "--init-from",
            help=(
                "Initialize model weights from a prior training stage; optimizer, "
                "EMA, loss normalization, RNG, and data progress start fresh."
            ),
        ),
    ] = None,
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
    """Resolve settings and run the complete staged Lightning application.

    :param output_dir: Checkpoint/log destination.
    :param max_steps: Optional bounded smoke-run step override.
    :param resume_from: Optional native Lightning checkpoint to resume strictly.
    :param init_from: Optional prior-stage checkpoint for model-only transfer.
    :param overrides: Optional Hydra configuration overrides.
    :param dry_run: Persist preflight only when true.
    :return: None; preflight metadata is always written.
    :rtype: None

    Dry-run mode performs strict Hydra/Pydantic resolution and writes provenance,
    but deliberately does not inspect the dataset, allocate a model, initialize
    CUDA/NCCL, or call ``Trainer.fit``. A real run validates the manifest and
    resume path before device launch, then constructs the configured field
    tokenizer, equivariant joint model, DataModule, callbacks, logger, and
    Trainer. ``trainer.devices=4``, ``accelerator=gpu``, ``strategy=ddp``, and
    ``precision=bf16-mixed`` therefore form one ordinary four-H100 command.
    """
    try:
        resolved_overrides = merge_overrides(overrides, trailing)
        config = load_config(resolved_overrides)
    except (KeyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    if init_from is not None and config.trainer.resume_from is not None:
        raise typer.BadParameter("--init-from cannot be combined with trainer.resume_from")
    if resume_from is not None and config.trainer.init_from is not None:
        raise typer.BadParameter("--resume-from cannot be combined with trainer.init_from")
    if max_steps is not None:
        config = config.model_copy(
            update={
                "trainer": config.trainer.model_copy(update={"max_steps": max_steps})
            }
        )
    if resume_from is not None and init_from is not None:
        raise typer.BadParameter("--resume-from and --init-from are mutually exclusive")
    if resume_from is not None:
        config = config.model_copy(
            update={
                "trainer": config.trainer.model_copy(
                    update={"resume_from": str(resume_from)}
                )
            }
        )
    if init_from is not None:
        config = config.model_copy(
            update={
                "trainer": config.trainer.model_copy(
                    update={"init_from": str(init_from)}
                )
            }
        )
    if not any(
        value.lstrip("+").startswith("trainer.checkpoint_dir=")
        for value in resolved_overrides
    ):
        config = config.model_copy(
            update={
                "trainer": config.trainer.model_copy(
                    update={"checkpoint_dir": str(output_dir / "checkpoints")}
                )
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    launch = output_dir / "train-config.json"
    payload = {
        "dry_run": dry_run,
        "config": config.model_dump(mode="json"),
        "status": "preflight" if dry_run else "starting",
    }
    if _is_global_zero():
        atomic_write_json(launch, payload)
        typer.echo(f"training configuration: {launch}")
    if dry_run:
        return
    try:
        runtime = run_training(config, output_dir)
    except (
        CheckpointStateError,
        DataValidationError,
        OSError,
        TrainingConfigurationError,
        TypeError,
        ValueError,
    ) as error:
        if _is_global_zero():
            atomic_write_json(
                launch,
                {**payload, "status": "failed", "error": str(error)},
            )
        raise typer.ClickException(f"training preflight failed: {error}") from error
    if _is_global_zero():
        atomic_write_json(
            launch,
            {
                **payload,
                "status": "completed",
                "global_step": int(runtime.trainer.global_step),
                "checkpoint_dir": str(runtime.checkpoint_dir),
            },
        )
        typer.echo(
            f"training completed at step {int(runtime.trainer.global_step)}; "
            f"checkpoints: {runtime.checkpoint_dir}"
        )


def _is_global_zero() -> bool:
    """Return true for a local launch or global rank zero under torchrun."""
    return os.environ.get("RANK", "0") == "0"
