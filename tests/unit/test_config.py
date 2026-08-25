"""Tests for strict Hydra and Pydantic configuration composition."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ecloudflow.config.loader import config_directory, load_config
from ecloudflow.data.diffgui_lmdb import DiffGuiLMDBImporter
from ecloudflow.exceptions import DataValidationError


def test_balanced_config_resolves_bounded_attempts():
    config = load_config(
        [
            "model=tiny",
            "sample=balanced",
            "sample.num_molecules=12",
        ]
    )
    assert config.model.name == "tiny"
    assert config.sample.num_molecules == 12
    assert config.sample.resolved_max_attempts == 60
    assert config.sample.solver == "heun"


def test_unknown_config_key_is_rejected():
    with pytest.raises((ValidationError, KeyError)):
        load_config(["sample.unknown_switch=true"])


def test_packaged_config_copy_matches_editable_source_tree() -> None:
    """Wheel defaults must remain byte-identical to editable user configs."""
    source = Path("configs")
    packaged = Path("src/ecloudflow/config/defaults")
    assert config_directory().resolve() == source.resolve()
    source_files = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*.yaml")
    }
    packaged_files = {
        path.relative_to(packaged).as_posix(): path.read_bytes()
        for path in packaged.rglob("*.yaml")
    }
    assert packaged_files == source_files


def test_explicit_config_directory_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operators can select a complete external Hydra configuration tree."""
    override = tmp_path / "configs"
    override.mkdir()
    (override / "config.yaml").write_text(
        Path("configs/config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setenv("ECLOUDFLOW_CONFIG_DIR", str(override))
    assert config_directory() == override.resolve()


def test_diffgui_and_local_cache_paths_are_typed_hydra_overrides() -> None:
    """Portable data paths compose without machine-specific defaults."""
    default = load_config(["data=pdbbind"])
    assert default.data.local_cache_dir is None
    assert default.data.diffgui_lmdb is None
    assert default.data.diffgui_source_root is None
    configured = load_config(
        [
            "data=crossdocked",
            "data.local_cache_dir=cache/shards",
            "data.diffgui_lmdb=legacy/processed.lmdb",
            "data.diffgui_source_root=legacy/raw",
            "data.diffgui_build_fields=true",
        ]
    )
    assert configured.data.dataset == "crossdocked"
    assert configured.data.local_cache_dir == "cache/shards"
    assert configured.data.diffgui_lmdb == "legacy/processed.lmdb"
    assert configured.data.diffgui_source_root == "legacy/raw"
    assert configured.data.diffgui_build_fields is True
    importer = DiffGuiLMDBImporter.from_config(configured.data)
    assert importer.path.as_posix() == "legacy/processed.lmdb"
    assert importer.source_root is not None
    assert importer.source_root.as_posix() == "legacy/raw"
    with pytest.raises(DataValidationError, match="diffgui_lmdb"):
        DiffGuiLMDBImporter.from_config(default.data)
