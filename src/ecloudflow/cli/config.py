"""Configuration inspection commands."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from ecloudflow.cli.common import merge_overrides
from ecloudflow.config import load_config

app = typer.Typer(help="Resolve and inspect strict ECloudFlow configuration.")


@app.command("show")
def show(
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--override",
            "-o",
            help="Hydra override such as sample.profile=fast. Repeatable.",
        ),
    ] = None,
    trailing: Annotated[
        list[str] | None,
        typer.Argument(help="Optional trailing key=value overrides."),
    ] = None,
) -> None:
    """Print the fully resolved configuration as deterministic JSON.

    :param overrides: Optional repeated Hydra dot-list overrides.
    :return: None; resolved configuration is written to stdout.
    :rtype: None
    """
    try:
        config = load_config(merge_overrides(overrides, trailing))
    except (KeyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("explain")
def explain(
    field: str,
    trailing: Annotated[
        list[str] | None,
        typer.Argument(help="Optional trailing key=value overrides."),
    ] = None,
) -> None:
    """Explain one dotted Pydantic configuration field.

    :param field: Dotted field path such as ``sample.num_steps``.
    :return: None; field metadata is written to stdout.
    :rtype: None
    :raises typer.BadParameter: If the field path is unknown.
    """
    try:
        config = load_config(merge_overrides(trailing=trailing))
    except (KeyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    try:
        model, value, metadata = _resolve_field(config, field)
    except (AttributeError, KeyError, TypeError) as error:
        raise typer.BadParameter(f"unknown configuration field: {field}") from error
    del model
    typer.echo(
        json.dumps(
            {
                "field": field,
                "value": value,
                "type": str(metadata.get("annotation", "")),
                "default": metadata.get("default"),
                "description": metadata.get("description"),
            },
            default=str,
            indent=2,
            sort_keys=True,
        )
    )


def _resolve_field(config: Any, field: str) -> tuple[Any, Any, dict[str, Any]]:
    """Resolve a Pydantic field and expose its declared metadata."""
    if not field or any(not part for part in field.split(".")):
        raise KeyError(field)
    model: Any = config
    metadata: dict[str, Any] = {}
    for part in field.split("."):
        fields = getattr(model, "model_fields", None)
        if not isinstance(fields, dict) or part not in fields:
            raise KeyError(part)
        info = fields[part]
        metadata = {
            "annotation": info.annotation,
            "default": None if info.is_required() else info.default,
            "description": info.description or f"Resolved configuration field {field}.",
        }
        model = getattr(model, part)
    return config, model, metadata
