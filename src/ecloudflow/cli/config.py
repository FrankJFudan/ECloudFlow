"""Configuration inspection commands."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from ecloudflow.config import load_config

app = typer.Typer(help="Resolve and inspect strict ECloudFlow configuration.")


@app.command("show")
def show(
    overrides: Annotated[list[str] | None, typer.Option("--override", "-o", help="Hydra override such as sample.profile=fast.")] = None,
) -> None:
    """Print the fully resolved configuration as deterministic JSON.

    :param overrides: Optional repeated Hydra dot-list overrides.
    :return: None; resolved configuration is written to stdout.
    :rtype: None
    """
    config = load_config(overrides or ())
    typer.echo(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("explain")
def explain(field: str) -> None:
    """Explain one dotted Pydantic configuration field.

    :param field: Dotted field path such as ``sample.num_steps``.
    :return: None; field metadata is written to stdout.
    :rtype: None
    :raises typer.BadParameter: If the field path is unknown.
    """
    config = load_config()
    value: object = config
    for part in field.split("."):
        if not hasattr(value, part):
            raise typer.BadParameter(f"unknown configuration field: {field}")
        value = getattr(value, part)
    typer.echo(json.dumps({"field": field, "value": value}, default=str, indent=2))

