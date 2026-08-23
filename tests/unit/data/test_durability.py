"""Tests for power-loss-aware filesystem publication primitives."""

from pathlib import Path

import pytest

from ecloudflow.data import durability


def test_durable_replace_flushes_both_directory_entries_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-directory rename flushes source and destination parents."""
    source_parent, destination_parent = tmp_path / "source", tmp_path / "destination"
    source_parent.mkdir()
    destination_parent.mkdir()
    source, destination = source_parent / "item", destination_parent / "item"
    source.write_text("payload", encoding="utf-8")
    events: list[tuple[str, Path]] = []
    original_replace = durability._atomic_replace

    def replace(left: Path, right: Path) -> None:
        events.append(("replace", right))
        original_replace(left, right)

    def flush(path: Path) -> None:
        events.append(("flush", path))

    monkeypatch.setattr(durability, "_atomic_replace", replace)
    monkeypatch.setattr(durability, "flush_directory", flush)
    durability.durable_replace(source, destination)
    assert events == [
        ("replace", destination),
        ("flush", source_parent),
        ("flush", destination_parent),
    ]


def test_durable_unlink_flushes_parent_after_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Active-marker deletion is followed by parent-directory durability."""
    marker = tmp_path / "active.json"
    marker.write_text("{}", encoding="utf-8")
    events: list[tuple[str, Path]] = []
    original_unlink = durability._atomic_unlink

    def unlink(path: Path) -> None:
        events.append(("unlink", path))
        original_unlink(path)

    def flush(path: Path) -> None:
        events.append(("flush", path))

    monkeypatch.setattr(durability, "_atomic_unlink", unlink)
    monkeypatch.setattr(durability, "flush_directory", flush)
    durability.durable_unlink(marker)
    assert events == [("unlink", marker), ("flush", tmp_path)]


def test_directory_flush_failure_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A platform that cannot flush directories may not claim durability."""
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_text("payload", encoding="utf-8")

    def unavailable(path: Path) -> None:
        raise durability.DurabilityError(f"cannot flush {path}")

    monkeypatch.setattr(durability, "flush_directory", unavailable)
    with pytest.raises(durability.DurabilityError, match="cannot flush"):
        durability.durable_replace(source, destination)


def test_durable_mkdir_flushes_new_ancestors_in_dependency_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nested directory entries become durable from ancestor to descendant."""
    target = tmp_path / "generation" / "nested"
    flushed: list[Path] = []
    monkeypatch.setattr(durability, "flush_directory", flushed.append)
    durability.durable_mkdir(target, parents=True)
    assert flushed == [tmp_path, tmp_path / "generation"]
