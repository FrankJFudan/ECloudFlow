"""Dataset preparation command adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ecloudflow.config import load_config

app = typer.Typer(help="Prepare and inspect dataset inputs.")


@app.command("prepare")
def prepare(
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset family name.")] = "pdbbind",
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("data/prepared"),
    overrides: Annotated[list[str] | None, typer.Option("--override", "-O")] = None,
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
    config = load_config([*(overrides or ()), f"data.dataset={dataset}"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "prepare.json"
    manifest.write_text(json.dumps({"dataset": dataset, "config": config.model_dump(mode="json")}, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"preparation manifest: {manifest}")

