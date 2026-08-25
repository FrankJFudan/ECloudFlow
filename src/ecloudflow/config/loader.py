"""Hydra composition and strict Pydantic validation for application settings."""

import os
from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import OmegaConf

from ecloudflow.config.schema import AppConfig

_SOURCE_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
_PACKAGED_CONFIG_DIR = Path(__file__).resolve().parent / "defaults"


def config_directory() -> Path:
    """Resolve the explicit, source-tree, or wheel-packaged configuration root.

    :return: Existing directory containing ``config.yaml`` and Hydra groups.
    :rtype: pathlib.Path
    :raises FileNotFoundError: If an explicit override or packaged defaults are
        absent.

    ``ECLOUDFLOW_CONFIG_DIR`` has highest priority so server operators can keep
    machine-specific presets outside the installed package. An editable source
    checkout then uses its top-level ``configs`` directory, which remains easy
    to inspect and modify. A regular wheel installation falls back to the
    read-only copy shipped under :mod:`ecloudflow.config`. Resolution performs
    no writes and does not mutate environment variables or Hydra global state.
    """
    configured = os.environ.get("ECLOUDFLOW_CONFIG_DIR")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not (candidate / "config.yaml").is_file():
            raise FileNotFoundError(
                "ECLOUDFLOW_CONFIG_DIR does not contain config.yaml: "
                f"{candidate}"
            )
        return candidate
    for candidate in (_SOURCE_CONFIG_DIR, _PACKAGED_CONFIG_DIR):
        if (candidate / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "ECloudFlow default configuration is missing from the source tree and package"
    )


def load_config(overrides: Sequence[str] = ()) -> AppConfig:
    """Compose configuration groups and validate their resolved values.

    :param overrides: Hydra overrides such as ``model=tiny`` or
        ``sample.num_molecules=12``.
    :return: Immutable, strictly validated application configuration.
    :rtype: AppConfig
    :raises KeyError: If an override names a configuration key that is unknown.
    :raises pydantic.ValidationError: If a resolved value violates the schema.
    """
    try:
        with initialize_config_dir(
            config_dir=str(config_directory()),
            version_base=None,
            job_name="ecloudflow",
        ):
            config = compose(config_name="config", overrides=list(overrides))
    except HydraException as error:
        raise KeyError(f"Invalid configuration override: {error}") from error

    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Resolved ECloudFlow configuration must be a mapping.")
    return AppConfig.model_validate(values)
