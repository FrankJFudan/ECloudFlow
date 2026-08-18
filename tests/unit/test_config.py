"""Tests for strict Hydra and Pydantic configuration composition."""

import pytest
from pydantic import ValidationError

from ecloudflow.config.loader import load_config


def test_balanced_config_resolves_bounded_attempts():
    config = load_config([
        "model=tiny",
        "sample=balanced",
        "sample.num_molecules=12",
    ])
    assert config.model.name == "tiny"
    assert config.sample.num_molecules == 12
    assert config.sample.resolved_max_attempts == 60
    assert config.sample.solver == "heun"


def test_unknown_config_key_is_rejected():
    with pytest.raises((ValidationError, KeyError)):
        load_config(["sample.unknown_switch=true"])
