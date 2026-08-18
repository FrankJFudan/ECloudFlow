"""Hydra composition and strict Pydantic validation for application settings."""

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import OmegaConf

from ecloudflow.config.schema import AppConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


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
            config_dir=str(CONFIG_DIR), version_base=None, job_name="ecloudflow"
        ):
            config = compose(config_name="config", overrides=list(overrides))
    except HydraException as error:
        raise KeyError(f"Invalid configuration override: {error}") from error

    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Resolved ECloudFlow configuration must be a mapping.")
    return AppConfig.model_validate(values)
