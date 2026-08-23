"""Cross-platform power-loss-aware filesystem publication primitives."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import IO, Any


class DurabilityError(OSError):
    """Raise when the platform cannot prove a requested durable transition."""


def sync_file(stream: IO[Any]) -> None:
    """Flush buffered bytes and the underlying file to stable storage.

    :param stream: Open writable binary or text file exposing ``flush``/``fileno``.
    :return: None.
    :rtype: None
    :raises DurabilityError: If userspace or kernel file synchronization fails.

    This filesystem mutation flushes buffered content so acknowledged bytes
    survive power loss, but it does not publish a directory entry. Callers must
    follow it with :func:`durable_replace` when a partial file becomes authoritative.
    """
    try:
        stream.flush()
        os.fsync(stream.fileno())
    except OSError as error:
        raise DurabilityError(f"failed to synchronize file bytes: {error}") from error


def flush_directory(path: str | Path) -> None:
    """Synchronize directory-entry changes on POSIX or Windows.

    :param path: Existing directory whose entry updates must survive power loss.
    :return: None.
    :rtype: None
    :raises DurabilityError: If the platform API is absent or rejects the flush.

    POSIX opens the directory and calls ``fsync``. Windows opens a directory
    handle with ``FILE_FLAG_BACKUP_SEMANTICS`` and calls ``FlushFileBuffers``.
    The synchronization makes prior directory-entry mutations survive power
    loss. Failure is explicit; callers never silently downgrade to
    process-crash-only atomicity while claiming power-loss durability.
    """
    directory = Path(path)
    if os.name == "nt":
        _flush_directory_windows(directory)
    else:
        _flush_directory_posix(directory)


def durable_replace(source: str | Path, destination: str | Path) -> None:
    """Atomically replace a path and durably flush both affected parents.

    :param source: Existing fsynced file or complete directory to publish.
    :param destination: Final path on the same filesystem/volume.
    :return: None.
    :rtype: None
    :raises DurabilityError: If replacement or either parent flush fails.

    Source-parent durability is required for cross-directory generation
    promotion; destination-parent durability makes the new authoritative name
    survive power loss. Windows replacement additionally uses
    ``MOVEFILE_WRITE_THROUGH``. The operation never modifies file contents.
    """
    source_path, destination_path = Path(source), Path(destination)
    try:
        _atomic_replace(source_path, destination_path)
    except OSError as error:
        raise DurabilityError(
            f"failed to atomically replace {destination_path}: {error}"
        ) from error
    flush_directory(source_path.parent)
    if destination_path.parent != source_path.parent:
        flush_directory(destination_path.parent)


def durable_unlink(path: str | Path, *, missing_ok: bool = False) -> None:
    """Remove one file and durably synchronize its parent directory.

    :param path: File or marker to remove.
    :param missing_ok: Return without mutation when the path is already absent.
    :return: None.
    :rtype: None
    :raises DurabilityError: If deletion or parent-directory flush fails.

    Both platforms remove the file, then synchronize the parent entry table.
    On Windows that second step uses a directory handle and
    ``FlushFileBuffers``; an acknowledged active-marker removal therefore cannot
    reappear after power loss. Mutation is limited to the named file and its
    parent entry; unrelated contents are untouched.
    """
    target = Path(path)
    try:
        _atomic_unlink(target)
    except FileNotFoundError:
        if missing_ok:
            return
        raise
    except OSError as error:
        raise DurabilityError(f"failed to remove {target}: {error}") from error
    flush_directory(target.parent)


def durable_mkdir(
    path: str | Path, *, parents: bool = False, exist_ok: bool = True
) -> None:
    """Create directories and durably flush every newly added parent entry.

    :param path: Directory to create.
    :param parents: Create missing ancestors when true.
    :param exist_ok: Match :meth:`pathlib.Path.mkdir` existing-path semantics.
    :return: None.
    :rtype: None
    :raises DurabilityError: If creation or any parent-directory flush fails.

    Filesystem mutation is limited to missing path components. Each newly added
    parent entry is flushed in ancestor-to-descendant order so successful return
    covers directory creation across power loss; unsupported flushes fail closed.
    """
    target = Path(path)
    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        target.mkdir(parents=parents, exist_ok=exist_ok)
    except OSError as error:
        raise DurabilityError(
            f"failed to create directory {target}: {error}"
        ) from error
    for created in reversed(missing):
        flush_directory(created.parent)


def _atomic_replace(source: Path, destination: Path) -> None:
    """Perform the platform atomic replace before explicit parent flushing."""
    if os.name == "nt":
        _move_file_windows(source, destination)
    else:
        os.replace(source, destination)


def _atomic_unlink(path: Path) -> None:
    """Perform the platform file removal before explicit parent flushing."""
    path.unlink()


def _move_file_windows(source: Path, destination: Path | None) -> None:
    """Call ``MoveFileExW`` with write-through completion semantics."""
    try:
        from ctypes import wintypes
    except ImportError as error:  # pragma: no cover - Windows always supplies it.
        raise DurabilityError("Windows ctypes types are unavailable") from error
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file = kernel32.MoveFileExW
    move_file.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    move_file.restype = wintypes.BOOL
    flags = 0x1 | 0x8  # MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
    destination_text = (
        str(destination.resolve(strict=False)) if destination is not None else None
    )
    if not move_file(str(source.resolve(strict=False)), destination_text, flags):
        raise ctypes.WinError(ctypes.get_last_error())


def _flush_directory_windows(path: Path) -> None:
    """Flush one Windows directory handle using ``FlushFileBuffers``."""
    try:
        from ctypes import wintypes
    except ImportError as error:  # pragma: no cover - Windows always supplies it.
        raise DurabilityError("Windows ctypes types are unavailable") from error
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    flush = kernel32.FlushFileBuffers
    flush.argtypes = [wintypes.HANDLE]
    flush.restype = wintypes.BOOL
    close = kernel32.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    handle = create_file(
        str(path.resolve(strict=True)),
        0x40000000,  # GENERIC_WRITE
        0x1 | 0x2 | 0x4,  # FILE_SHARE_READ | WRITE | DELETE
        None,
        3,  # OPEN_EXISTING
        0x02000000 | 0x80000000,  # BACKUP_SEMANTICS | WRITE_THROUGH
        None,
    )
    invalid_handle = wintypes.HANDLE(-1).value
    if handle == invalid_handle:
        raise DurabilityError(
            f"failed to open directory for flush: {ctypes.WinError(ctypes.get_last_error())}"
        )
    try:
        if not flush(handle):
            raise DurabilityError(
                f"failed to flush directory: {ctypes.WinError(ctypes.get_last_error())}"
            )
    finally:
        if not close(handle):
            raise DurabilityError(
                f"failed to close directory handle: {ctypes.WinError(ctypes.get_last_error())}"
            )


def _flush_directory_posix(path: Path) -> None:
    """Open and fsync one POSIX directory entry table."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise DurabilityError(f"failed to flush directory {path}: {error}") from error
