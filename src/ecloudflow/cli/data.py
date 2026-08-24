"""Dataset preparation command adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ecloudflow.cli.common import atomic_write_json, merge_overrides
from ecloudflow.config import load_config

app = typer.Typer(help="Prepare and inspect dataset inputs.")


@app.command("prepare")
def prepare(
    dataset: Annotated[
        str, typer.Option("--dataset", help="Dataset family name.")
    ] = "pdbbind",
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "data/prepared"
    ),
    overrides: Annotated[list[str] | None, typer.Option("--override", "-O")] = None,
    pocket: Annotated[
        Path | None, typer.Option("--pocket", help="Optional source pocket PDB.")
    ] = None,
    ligand: Annotated[
        Path | None, typer.Option("--ligand", help="Optional source ligand SDF.")
    ] = None,
    sample_id: Annotated[str, typer.Option("--sample-id")] = "sample-000001",
    no_fields: Annotated[
        bool,
        typer.Option("--no-fields", help="Skip optional physical-field generation."),
    ] = False,
    trailing: Annotated[
        list[str] | None, typer.Argument(help="Optional trailing key=value overrides.")
    ] = None,
) -> None:
    """Validate preparation settings and write a reproducible run manifest.

    :param dataset: Named dataset family passed to data services.
    :param output_dir: Destination for the preparation manifest.
    :param overrides: Optional Hydra configuration overrides.
    :return: None; a preparation manifest is written to ``output_dir``.
    :rtype: None

    This command intentionally does not fabricate samples. Dataset-specific
    importers consume the manifest as their explicit next-stage input.
    """
    try:
        config = load_config(
            [*merge_overrides(overrides, trailing), f"data.dataset={dataset}"]
        )
    except (KeyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    if (pocket is None) != (ligand is None):
        raise typer.BadParameter("--pocket and --ligand must be supplied together")
    if pocket is not None and ligand is not None:
        from ecloudflow.data.parsers import build_complex_sample
        from ecloudflow.data.shards import ShardWriter

        if not pocket.is_file() or not ligand.is_file():
            raise typer.BadParameter("source pocket and ligand files must exist")
        try:
            sample = build_complex_sample(
                pocket,
                ligand,
                sample_id,
                build_fields=not no_fields,
            )
            manifest_obj = ShardWriter(
                target_shard_size_gb=config.data.target_shard_size_gb,
                max_samples_per_shard=1,
            ).write((sample,), output_dir)
        except Exception as error:
            raise typer.BadParameter(f"dataset preparation failed: {error}") from error
        typer.echo(f"manifest: {output_dir / 'manifest.json'}")
        typer.echo(f"samples: {len(manifest_obj.sample_ids)}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "prepare.json"
    atomic_write_json(
        manifest,
        {
            "dataset": dataset,
            "config": config.model_dump(mode="json"),
            "status": "preflight",
            "next_step": "provide --pocket and --ligand or use a dataset importer",
        },
    )
    typer.echo(f"preparation manifest: {manifest}")
