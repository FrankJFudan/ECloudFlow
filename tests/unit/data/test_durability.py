"""Tests for power-loss-aware filesystem publication primitives."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ecloudflow.data import durability


def test_durable_replace_flushes_both_directory_entries_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-directory rename makes the destination durable before the source."""
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
        ("flush", destination_parent),
        ("flush", source_parent),
    ]


def test_durable_replace_flushes_a_shared_parent_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-directory replacement requires exactly one parent flush."""
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_text("payload", encoding="utf-8")
    flushed: list[Path] = []
    monkeypatch.setattr(durability, "flush_directory", flushed.append)
    durability.durable_replace(source, destination)
    assert flushed == [tmp_path]


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


@pytest.mark.parametrize("failure_stage", ["flush", "fileno"])
def test_sync_file_wraps_non_os_file_object_failures(
    tmp_path: Path, failure_stage: str
) -> None:
    """Valid file-like objects expose typed failures from either sync stage."""
    cause = RuntimeError(f"broken {failure_stage}")

    class BrokenStream:
        name = tmp_path / "payload.partial"

        def flush(self) -> None:
            if failure_stage == "flush":
                raise cause

        def fileno(self) -> int:
            if failure_stage == "fileno":
                raise cause
            return 0

    with pytest.raises(durability.DurabilityError) as caught:
        durability.sync_file(BrokenStream())
    assert "payload.partial" in str(caught.value)
    assert caught.value.__cause__ is cause


def test_sync_file_rejects_an_invalid_stream_as_a_programmer_error() -> None:
    """An object without the documented file protocol remains a TypeError."""
    with pytest.raises(TypeError, match="flush.*fileno"):
        durability.sync_file(object())  # type: ignore[arg-type]


def test_sync_file_wraps_non_os_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kernel synchronization exceptions retain the stream path and cause."""
    cause = RuntimeError("fsync backend failed")

    class FakeStream:
        name = tmp_path / "payload.partial"

        @staticmethod
        def flush() -> None:
            return None

        @staticmethod
        def fileno() -> int:
            return 42

    def fail_fsync(_descriptor: int) -> None:
        raise cause

    monkeypatch.setattr(durability.os, "fsync", fail_fsync)
    with pytest.raises(durability.DurabilityError) as caught:
        durability.sync_file(FakeStream())
    assert str(FakeStream.name) in str(caught.value)
    assert caught.value.__cause__ is cause


def test_every_public_path_primitive_wraps_non_os_operational_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepted path arguments never leak arbitrary backend exceptions."""
    cause = RuntimeError("backend rejected operation")

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise cause

    monkeypatch.setattr(durability, "_flush_directory_windows", fail)
    monkeypatch.setattr(durability, "_flush_directory_posix", fail)
    with pytest.raises(durability.DurabilityError) as flush_error:
        durability.flush_directory(tmp_path)
    assert str(tmp_path) in str(flush_error.value)
    assert flush_error.value.__cause__ is cause

    monkeypatch.setattr(durability, "_atomic_replace", fail)
    with pytest.raises(durability.DurabilityError) as replace_error:
        durability.durable_replace(tmp_path / "source", tmp_path / "destination")
    assert "source" in str(replace_error.value)
    assert "destination" in str(replace_error.value)
    assert replace_error.value.__cause__ is cause

    monkeypatch.setattr(durability, "_atomic_unlink", fail)
    with pytest.raises(durability.DurabilityError) as unlink_error:
        durability.durable_unlink(tmp_path / "marker.json")
    assert "marker.json" in str(unlink_error.value)
    assert unlink_error.value.__cause__ is cause

    monkeypatch.setattr(Path, "exists", fail)
    with pytest.raises(durability.DurabilityError) as mkdir_error:
        durability.durable_mkdir(tmp_path / "nested", parents=True)
    assert "nested" in str(mkdir_error.value)
    assert mkdir_error.value.__cause__ is cause


@pytest.mark.parametrize(
    "operation",
    [
        lambda path: durability.flush_directory(object()),
        lambda path: durability.durable_replace(object(), path),
        lambda path: durability.durable_unlink(object()),
        lambda path: durability.durable_mkdir(object()),
    ],
)
def test_public_path_primitives_preserve_programmer_type_errors(
    tmp_path: Path, operation: Any
) -> None:
    """Invalid path argument types fail before an operational wrapper or mutation."""
    with pytest.raises(TypeError):
        operation(tmp_path)


def test_durable_unlink_missing_file_is_typed_unless_explicitly_allowed(
    tmp_path: Path,
) -> None:
    """Missing authoritative markers follow the public typed failure boundary."""
    missing = tmp_path / "missing.json"
    with pytest.raises(durability.DurabilityError) as caught:
        durability.durable_unlink(missing)
    assert str(missing) in str(caught.value)
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    durability.durable_unlink(missing, missing_ok=True)


class _FakeWindowsFunction:
    """Provide assignable ctypes metadata around a configurable fake call."""

    def __init__(self, *, result: Any = True, failure: Exception | None = None):
        self.result = result
        self.failure = failure
        self.argtypes: Any = None
        self.restype: Any = None

    def __call__(self, *_args: Any) -> Any:
        if self.failure is not None:
            raise self.failure
        return self.result


def _fake_windows_kernel(
    monkeypatch: pytest.MonkeyPatch,
    *,
    move: _FakeWindowsFunction | None = None,
    create: _FakeWindowsFunction | None = None,
    flush: _FakeWindowsFunction | None = None,
    close: _FakeWindowsFunction | None = None,
    windows_error: OSError | None = None,
) -> None:
    kernel = SimpleNamespace(
        MoveFileExW=move or _FakeWindowsFunction(),
        CreateFileW=create or _FakeWindowsFunction(result=1),
        FlushFileBuffers=flush or _FakeWindowsFunction(),
        CloseHandle=close or _FakeWindowsFunction(),
    )
    monkeypatch.setattr(
        durability.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: kernel,
        raising=False,
    )
    monkeypatch.setattr(durability.ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(
        durability.ctypes,
        "WinError",
        lambda _code: windows_error or OSError("fake Windows failure"),
        raising=False,
    )


def test_windows_move_wraps_non_os_ctypes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MoveFileExW invocation failures retain both paths and their cause."""
    cause = RuntimeError("ctypes invocation failed")
    _fake_windows_kernel(monkeypatch, move=_FakeWindowsFunction(failure=cause))
    source, destination = tmp_path / "source", tmp_path / "destination"
    with pytest.raises(durability.DurabilityError) as caught:
        durability._move_file_windows(source, destination)
    assert str(source) in str(caught.value)
    assert str(destination) in str(caught.value)
    assert caught.value.__cause__ is cause


def test_windows_directory_resolution_and_create_failures_are_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory resolution and CreateFileW errors share one typed boundary."""
    _fake_windows_kernel(monkeypatch)
    missing = tmp_path / "missing"
    with pytest.raises(durability.DurabilityError) as resolution_error:
        durability._flush_directory_windows(missing)
    assert str(missing) in str(resolution_error.value)
    assert isinstance(resolution_error.value.__cause__, FileNotFoundError)

    cause = RuntimeError("CreateFileW invocation failed")
    _fake_windows_kernel(monkeypatch, create=_FakeWindowsFunction(failure=cause))
    with pytest.raises(durability.DurabilityError) as create_error:
        durability._flush_directory_windows(tmp_path)
    assert str(tmp_path) in str(create_error.value)
    assert create_error.value.__cause__ is cause


def test_windows_flush_failure_is_typed_and_chained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A false FlushFileBuffers result retains the native error as its cause."""
    cause = OSError("FlushFileBuffers rejected directory")
    _fake_windows_kernel(
        monkeypatch,
        flush=_FakeWindowsFunction(result=False),
        windows_error=cause,
    )
    with pytest.raises(durability.DurabilityError) as caught:
        durability._flush_directory_windows(tmp_path)
    assert str(tmp_path) in str(caught.value)
    assert caught.value.__cause__ is cause


def test_posix_fsync_non_os_failure_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The POSIX helper wraps arbitrary fsync backend failures with its path."""
    cause = RuntimeError("POSIX fsync backend failed")
    monkeypatch.setattr(durability.os, "open", lambda *_args: 42)

    def fail_fsync(_descriptor: int) -> None:
        raise cause

    monkeypatch.setattr(durability.os, "fsync", fail_fsync)
    monkeypatch.setattr(durability.os, "close", lambda _descriptor: None)
    with pytest.raises(durability.DurabilityError) as caught:
        durability._flush_directory_posix(tmp_path)
    assert str(tmp_path) in str(caught.value)
    assert caught.value.__cause__ is cause
