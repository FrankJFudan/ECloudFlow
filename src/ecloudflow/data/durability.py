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
    :raises TypeError: If ``stream`` does not expose callable ``flush`` and
        ``fileno`` methods, before synchronization is attempted.
    :raises DurabilityError: If an accepted stream fails while flushing,
        returning its descriptor, or synchronizing through :func:`os.fsync`.

    This filesystem mutation flushes buffered content so acknowledged bytes
    survive power loss, but it does not publish a directory entry. Callers must
    follow it with :func:`durable_replace` when a partial file becomes
    authoritative. After the file protocol is validated, every operational
    exception is chained into ``DurabilityError`` with the stream name and
    failing operation; programmer argument errors remain ``TypeError``.
    """
    try:
        flush = stream.flush
        fileno = stream.fileno
    except AttributeError as error:
        raise TypeError(
            "stream must expose callable flush and fileno methods"
        ) from error
    except Exception as error:
        raise DurabilityError(
            f"failed to inspect file synchronization methods on "
            f"{_stream_context(stream)}: {error}"
        ) from error
    if not callable(flush) or not callable(fileno):
        raise TypeError("stream must expose callable flush and fileno methods")
    context = _stream_context(stream)
    try:
        flush()
    except Exception as error:
        raise DurabilityError(
            f"failed to flush buffered file bytes for {context}: {error}"
        ) from error
    try:
        descriptor = fileno()
    except Exception as error:
        raise DurabilityError(
            f"failed to obtain a file descriptor for {context}: {error}"
        ) from error
    try:
        os.fsync(descriptor)
    except Exception as error:
        raise DurabilityError(
            f"failed to fsync file bytes for {context}: {error}"
        ) from error


def flush_directory(path: str | Path) -> None:
    """Synchronize directory-entry changes on POSIX or Windows.

    :param path: Existing directory whose entry updates must survive power loss.
    :return: None.
    :rtype: None
    :raises TypeError: If ``path`` is not a string or path-like value; this is
        validated before any filesystem operation.
    :raises DurabilityError: If path handling or the platform API fails or
        rejects the flush after argument validation.

    POSIX opens the directory and calls ``fsync``. Windows opens a directory
    handle with ``FILE_FLAG_BACKUP_SEMANTICS`` and calls ``FlushFileBuffers``.
    The synchronization makes prior directory-entry mutations survive power
    loss. Failure is explicit; callers never silently downgrade to
    process-crash-only atomicity while claiming power-loss durability. All
    operational failures are chained with the directory path; invalid argument
    types remain ordinary programmer ``TypeError`` exceptions.
    """
    directory = Path(path)
    try:
        if os.name == "nt":
            _flush_directory_windows(directory)
        else:
            _flush_directory_posix(directory)
    except Exception as error:
        raise DurabilityError(
            f"failed to flush directory {directory}: {error}"
        ) from error


def durable_replace(source: str | Path, destination: str | Path) -> None:
    """Atomically replace a path and durably flush both affected parents.

    :param source: Existing fsynced file or complete directory to publish.
    :param destination: Final path on the same filesystem/volume.
    :return: None.
    :rtype: None
    :raises TypeError: If either path is not a string or path-like value; both
        are validated before filesystem mutation.
    :raises DurabilityError: If replacement or either parent flush fails after
        path validation.

    Destination-parent durability is established before source-parent
    durability for a cross-directory generation promotion. A power loss can
    therefore leave both durable names, but cannot lose both names because the
    destination name is durable before the source removal. Same-parent
    replacement flushes that directory once. Windows additionally uses
    ``MOVEFILE_WRITE_THROUGH``. The operation never modifies file contents.
    Operational exceptions are chained with both affected paths; invalid path
    argument types remain ordinary programmer ``TypeError`` exceptions.
    """
    source_path, destination_path = Path(source), Path(destination)
    try:
        _atomic_replace(source_path, destination_path)
    except Exception as error:
        raise DurabilityError(
            f"failed to atomically replace {source_path} with {destination_path}: "
            f"{error}"
        ) from error
    try:
        flush_directory(destination_path.parent)
    except Exception as error:
        raise DurabilityError(
            f"replaced {source_path} with {destination_path} but failed to flush "
            f"destination directory {destination_path.parent}: {error}"
        ) from error
    if destination_path.parent != source_path.parent:
        try:
            flush_directory(source_path.parent)
        except Exception as error:
            raise DurabilityError(
                f"replaced {source_path} with {destination_path} and flushed its "
                f"destination, but failed to flush source directory "
                f"{source_path.parent}: {error}"
            ) from error


def durable_unlink(path: str | Path, *, missing_ok: bool = False) -> None:
    """Remove one file and durably synchronize its parent directory.

    :param path: File or marker to remove.
    :param missing_ok: Return without mutation when the path is already absent.
    :return: None.
    :rtype: None
    :raises TypeError: If ``path`` is not a string/path-like value or
        ``missing_ok`` is not a boolean, before filesystem mutation.
    :raises DurabilityError: If deletion, a disallowed missing path, or the
        parent-directory flush fails after argument validation.

    Both platforms remove the file, then synchronize the parent entry table.
    On Windows that second step uses a directory handle and
    ``FlushFileBuffers``; an acknowledged active-marker removal therefore cannot
    reappear after power loss. Mutation is limited to the named file and its
    parent entry; unrelated contents are untouched. Every operational exception,
    including ``FileNotFoundError`` when ``missing_ok=False``, is chained with
    the target path. Programmer argument errors remain ``TypeError``.
    """
    target = Path(path)
    if not isinstance(missing_ok, bool):
        raise TypeError("missing_ok must be a bool")
    try:
        _atomic_unlink(target)
    except FileNotFoundError as error:
        if missing_ok:
            return
        raise DurabilityError(
            f"failed to remove missing path {target}: {error}"
        ) from error
    except Exception as error:
        raise DurabilityError(f"failed to remove {target}: {error}") from error
    try:
        flush_directory(target.parent)
    except Exception as error:
        raise DurabilityError(
            f"removed {target} but failed to flush parent directory "
            f"{target.parent}: {error}"
        ) from error


def durable_mkdir(
    path: str | Path, *, parents: bool = False, exist_ok: bool = True
) -> None:
    """Create directories and durably flush every newly added parent entry.

    :param path: Directory to create.
    :param parents: Create missing ancestors when true.
    :param exist_ok: Match :meth:`pathlib.Path.mkdir` existing-path semantics.
    :return: None.
    :rtype: None
    :raises TypeError: If ``path`` is not string/path-like or either flag is not
        boolean, before filesystem mutation.
    :raises DurabilityError: If path inspection, creation, or any parent flush
        fails after argument validation.

    Filesystem mutation is limited to missing path components. Each newly added
    parent entry is flushed in ancestor-to-descendant order so successful return
    covers directory creation across power loss; unsupported flushes fail closed.
    Operational exceptions are chained with the affected path, while invalid
    path and flag types remain ordinary programmer ``TypeError`` exceptions.
    """
    target = Path(path)
    if not isinstance(parents, bool):
        raise TypeError("parents must be a bool")
    if not isinstance(exist_ok, bool):
        raise TypeError("exist_ok must be a bool")
    missing: list[Path] = []
    cursor = target
    try:
        while not cursor.exists():
            missing.append(cursor)
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
    except Exception as error:
        raise DurabilityError(
            f"failed to inspect directory path {target}: {error}"
        ) from error
    try:
        target.mkdir(parents=parents, exist_ok=exist_ok)
    except Exception as error:
        raise DurabilityError(
            f"failed to create directory {target}: {error}"
        ) from error
    for created in reversed(missing):
        try:
            flush_directory(created.parent)
        except Exception as error:
            raise DurabilityError(
                f"created directory {created} but failed to flush parent "
                f"{created.parent}: {error}"
            ) from error


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
        raise DurabilityError(
            f"Windows ctypes types are unavailable for replacement from {source} "
            f"to {destination}"
        ) from error
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileExW
        move_file.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        move_file.restype = wintypes.BOOL
        flags = 0x1 | 0x8  # MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
        source_text = str(source.resolve(strict=False))
        destination_text = (
            str(destination.resolve(strict=False)) if destination is not None else None
        )
        moved = move_file(source_text, destination_text, flags)
    except Exception as error:
        raise DurabilityError(
            f"failed to invoke MoveFileExW from {source} to {destination}: {error}"
        ) from error
    if not moved:
        windows_error = _last_windows_error()
        raise DurabilityError(
            f"MoveFileExW rejected durable replacement from {source} to "
            f"{destination}: {windows_error}"
        ) from windows_error


def _flush_directory_windows(path: Path) -> None:
    """Flush one Windows directory handle using ``FlushFileBuffers``."""
    try:
        from ctypes import wintypes
    except ImportError as error:  # pragma: no cover - Windows always supplies it.
        raise DurabilityError(
            f"Windows ctypes types are unavailable for directory flush {path}"
        ) from error
    try:
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
        invalid_handle = wintypes.HANDLE(-1).value
        resolved = str(path.resolve(strict=True))
        handle = create_file(
            resolved,
            0x40000000,  # GENERIC_WRITE
            0x1 | 0x2 | 0x4,  # FILE_SHARE_READ | WRITE | DELETE
            None,
            3,  # OPEN_EXISTING
            0x02000000 | 0x80000000,  # BACKUP_SEMANTICS | WRITE_THROUGH
            None,
        )
    except Exception as error:
        raise DurabilityError(
            f"failed to resolve/open directory {path} with CreateFileW: {error}"
        ) from error
    if handle == invalid_handle:
        windows_error = _last_windows_error()
        raise DurabilityError(
            f"CreateFileW rejected directory flush handle for {path}: {windows_error}"
        ) from windows_error
    try:
        try:
            flushed = flush(handle)
        except Exception as error:
            raise DurabilityError(
                f"failed to invoke FlushFileBuffers for directory {path}: {error}"
            ) from error
        if not flushed:
            windows_error = _last_windows_error()
            raise DurabilityError(
                f"FlushFileBuffers rejected directory {path}: {windows_error}"
            ) from windows_error
    finally:
        try:
            closed = close(handle)
        except Exception as error:
            raise DurabilityError(
                f"failed to invoke CloseHandle for directory {path}: {error}"
            ) from error
        if not closed:
            windows_error = _last_windows_error()
            raise DurabilityError(
                f"CloseHandle rejected directory handle for {path}: {windows_error}"
            ) from windows_error


def _flush_directory_posix(path: Path) -> None:
    """Open and fsync one POSIX directory entry table."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except Exception as error:
        raise DurabilityError(f"failed to open directory {path}: {error}") from error
    try:
        try:
            os.fsync(descriptor)
        except Exception as error:
            raise DurabilityError(
                f"failed to fsync directory {path}: {error}"
            ) from error
    finally:
        try:
            os.close(descriptor)
        except Exception as error:
            raise DurabilityError(
                f"failed to close directory descriptor for {path}: {error}"
            ) from error


def _last_windows_error() -> Exception:
    """Return the current native Windows error, including retrieval failures."""
    try:
        return ctypes.WinError(ctypes.get_last_error())
    except Exception as error:  # noqa: BLE001 - ctypes backends may fail arbitrarily.
        return error


def _stream_context(stream: IO[Any]) -> str:
    """Return a non-throwing file name/type label for durability diagnostics."""
    try:
        name = getattr(stream, "name", None)
    except Exception:  # noqa: BLE001 - a diagnostic label must not mask sync failure.
        name = None
    return str(name) if name is not None else f"<{type(stream).__name__}>"
