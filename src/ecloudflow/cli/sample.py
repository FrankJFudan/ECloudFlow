"""Molecule generation command delegating to :class:`ECloudFlowPipeline`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ecloudflow import ECloudFlowPipeline
from ecloudflow.sampling.results import GenerationMode


def sample_command(
    pocket: Annotated[Path, typer.Argument(help="Pocket PDB path.")],
    num_molecules: Annotated[int, typer.Option("--num-molecules", "-n", min=1)] = 100,
    fragment: Annotated[Path | None, typer.Option("--fragment")] = None,
    mode: Annotated[GenerationMode, typer.Option("--mode")] = GenerationMode.DE_NOVO,
    profile: Annotated[str, typer.Option("--profile")] = "balanced",
    checkpoint: Annotated[Path, typer.Option("--checkpoint")] = Path("checkpoints/ecloudflow.ckpt"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("runs/sample"),
    seed: Annotated[int, typer.Option("--seed")] = 2026,
    max_attempts: Annotated[int | None, typer.Option("--max-attempts", min=1)] = None,
) -> None:
    """Generate, rank, and summarize ligands for one pocket.

    :param pocket: Input pocket PDB in the desired output coordinate frame.
    :param num_molecules: Target number of valid unique molecules.
    :param fragment: Optional positioned fragment SDF for fragment modes.
    :param mode: De novo, grow, link, replace, or merge generation objective.
    :param profile: Fast, balanced, or quality numerical profile.
    :param checkpoint: Trained checkpoint consumed by the pipeline.
    :param output_dir: Atomic run artifact directory.
    :param seed: Master deterministic generation seed.
    :param max_attempts: Optional bounded attempt count.
    :return: None; generation summary and artifact paths are printed.
    :rtype: None
    """
    if not pocket.is_file():
        raise typer.BadParameter(f"pocket does not exist: {pocket}", param_hint="pocket")
    if mode is not GenerationMode.DE_NOVO and fragment is None:
        raise typer.BadParameter("--fragment is required for fragment-conditioned modes")
    pipeline = ECloudFlowPipeline.from_pretrained(checkpoint)
    result = pipeline.generate(
        pocket=pocket,
        num_molecules=num_molecules,
        fragment=fragment,
        mode=mode,
        profile=profile,
        max_attempts=max_attempts,
        output_dir=output_dir,
        seed=seed,
    )
    typer.echo(f"generated {len(result.valid)} valid unique molecules after {result.attempts} attempts")
    if result.output_bundle is not None:
        for path in result.output_bundle.paths:
            typer.echo(path)

