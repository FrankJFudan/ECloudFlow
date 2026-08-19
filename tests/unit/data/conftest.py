"""Fixtures shared by ECloudFlow data-system tests."""

from pathlib import Path

import pytest


@pytest.fixture
def fixture_dir() -> Path:
    """Return the repository fixture directory."""
    return Path(__file__).resolve().parents[2] / "fixtures"
