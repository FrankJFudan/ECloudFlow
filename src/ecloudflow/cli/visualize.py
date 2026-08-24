"""Visualization commands for ranked poses and electron fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ecloudflow.cli.common import find_ranked_row, load_run_payload
from ecloudflow.visualization import render_complex_html, render_electron_field_html

app = typer.Typer(help="Render molecule and electron-cloud browser views.")


@app.command("molecule")
def molecule(
    run_dir: Annotated[Path, typer.Argument(help="Generation run directory.")],
    molecule_id: Annotated[
        str, typer.Option("--id", help="Formal ranked molecule ID.")
    ],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Render one ranked raw/relaxed complex as a self-contained HTML file."""
    row = find_ranked_row(run_dir, molecule_id)
    if row is None:
        raise typer.BadParameter(f"molecule ID not found: {molecule_id}")
    destination = output or run_dir / f"molecule_{molecule_id}.html"
    path = render_complex_html(row, destination)
    typer.echo(f"molecule viewer: {path}")


@app.command("ecloud")
def ecloud(
    run_dir: Annotated[Path, typer.Argument(help="Generation run directory.")],
    molecule_id: Annotated[
        str | None, typer.Option("--id", help="Optional molecule ID to annotate.")
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Render an electron-density payload as a self-contained HTML file."""
    payload = load_run_payload(run_dir)
    field_path = run_dir / "electron_field.json"
    if field_path.is_file():
        try:
            field = json.loads(field_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise typer.BadParameter(f"invalid electron_field.json: {error}") from error
    else:
        field = payload.get("electron_fields", payload.get("ecloud", payload))
    if molecule_id:
        field = {"molecule_id": molecule_id, "field": field}
    destination = output or run_dir / "electron_field.html"
    path = render_electron_field_html(field, destination)
    typer.echo(f"electron field viewer: {path}")


__all__ = ["app"]
