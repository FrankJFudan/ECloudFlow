"""Safe subprocess adapter for AutoDock Vina-compatible binaries."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ecloudflow.docking.base import (
    DockingResult,
    DockingStatus,
    validate_box,
)

_RESULT_PATTERN = re.compile(
    r"(?:REMARK\s+VINA\s+RESULT:|VINA\s+RESULT:|^\s*[-+]?\d+(?:\.\d+)?\s+)[^\n]*",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


class VinaBackend:
    """Run a Vina-compatible executable with bounded, auditable arguments.

    :param executable: Binary name or absolute executable path.
    :param timeout_seconds: Positive subprocess timeout.
    :param exhaustiveness: Vina search exhaustiveness.
    :param seed: Optional deterministic Vina seed.
    :param version: Optional caller-supplied version string.
    :return: Configured backend.
    :rtype: VinaBackend

    The adapter accepts receptor paths and either ligand paths or RDKit
    molecules.  RDKit molecules are written to a temporary SDF; conversion to
    PDBQT is intentionally left to a configured upstream preparation step, so
    an unavailable converter is reported rather than silently bypassed.
    """

    name = "vina"

    def __init__(
        self,
        executable: str | Path = "vina",
        *,
        timeout_seconds: float = 120.0,
        exhaustiveness: int = 8,
        seed: int | None = 2026,
        version: str = "unknown",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if isinstance(exhaustiveness, bool) or exhaustiveness < 1:
            raise ValueError("exhaustiveness must be positive.")
        if seed is not None and (isinstance(seed, bool) or seed < 0):
            raise ValueError("seed must be non-negative or None.")
        self.executable = str(executable)
        self.timeout_seconds = float(timeout_seconds)
        self.exhaustiveness = int(exhaustiveness)
        self.seed = None if seed is None else int(seed)
        self.version = str(version)

    def score(
        self,
        molecule: Any,
        pocket: Any,
        *,
        box_center: Sequence[float] = (0.0, 0.0, 0.0),
        box_size: Sequence[float] = (20.0, 20.0, 20.0),
        ligand_path: str | Path | None = None,
        receptor_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> DockingResult:
        """Score one pose and return an explicit unavailable/failure status.

        :param molecule: RDKit molecule or an object carrying a ligand path.
        :param pocket: Pocket object or path; path-like pockets are preferred.
        :param box_center: Search-box center in angstroms.
        :param box_size: Positive search-box dimensions in angstroms.
        :param ligand_path: Optional prepared PDBQT ligand path.
        :param receptor_path: Optional prepared PDBQT receptor path.
        :param output_dir: Optional directory for Vina output files.
        :return: Typed score result; no missing score is replaced by zero.
        :rtype: DockingResult
        :raises ValueError: If required paths or box values are malformed.
        """
        center, size = validate_box(box_center, box_size)
        receptor = _resolve_path(receptor_path, pocket, "receptor")
        ligand = _resolve_path(ligand_path, molecule, "ligand")
        if receptor is None or ligand is None:
            return DockingResult(
                score=None,
                status=DockingStatus.UNAVAILABLE,
                backend=self.name,
                version=self.version,
                reason="prepared receptor_path and ligand_path are required",
            )
        executable = shutil.which(self.executable) or (
            self.executable if Path(self.executable).is_file() else None
        )
        if executable is None:
            return DockingResult(
                score=None,
                status=DockingStatus.UNAVAILABLE,
                backend=self.name,
                version=self.version,
                reason=f"executable not found: {self.executable}",
            )
        destination = Path(output_dir) if output_dir is not None else None
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(executable),
            "--receptor",
            str(receptor),
            "--ligand",
            str(ligand),
            "--center_x",
            _format_float(center[0]),
            "--center_y",
            _format_float(center[1]),
            "--center_z",
            _format_float(center[2]),
            "--size_x",
            _format_float(size[0]),
            "--size_y",
            _format_float(size[1]),
            "--size_z",
            _format_float(size[2]),
            "--exhaustiveness",
            str(self.exhaustiveness),
        ]
        if self.seed is not None:
            command.extend(["--seed", str(self.seed)])
        if destination is not None:
            command.extend(["--out", str(destination / "docked.pdbqt")])
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return DockingResult(
                score=None,
                status=DockingStatus.TIMEOUT,
                backend=self.name,
                version=self.version,
                command=tuple(command),
                raw_output=_capture_timeout(error),
                elapsed_seconds=time.monotonic() - started,
                reason="Vina subprocess timed out",
            )
        except OSError as error:
            return DockingResult(
                score=None,
                status=DockingStatus.FAILED,
                backend=self.name,
                version=self.version,
                command=tuple(command),
                elapsed_seconds=time.monotonic() - started,
                reason=f"failed to execute Vina: {error}",
            )
        raw_output = "\n".join(
            value for value in (completed.stdout or "", completed.stderr or "") if value
        )
        score = _parse_vina_score(raw_output)
        if completed.returncode != 0:
            return DockingResult(
                score=None,
                status=DockingStatus.FAILED,
                backend=self.name,
                version=self.version,
                command=tuple(command),
                raw_output=raw_output,
                elapsed_seconds=time.monotonic() - started,
                reason=f"Vina exited with code {completed.returncode}",
            )
        if score is None:
            return DockingResult(
                score=None,
                status=DockingStatus.FAILED,
                backend=self.name,
                version=self.version,
                command=tuple(command),
                raw_output=raw_output,
                elapsed_seconds=time.monotonic() - started,
                reason="Vina output did not contain a parseable score",
            )
        return DockingResult(
            score=score,
            status=DockingStatus.SUCCESS,
            backend=self.name,
            version=self.version,
            command=tuple(command),
            raw_output=raw_output,
            elapsed_seconds=time.monotonic() - started,
        )


def _resolve_path(explicit: str | Path | None, value: Any, label: str) -> Path | None:
    """Resolve an explicit path or a path-like object without writing inputs."""
    candidate = explicit
    if candidate is None and isinstance(value, (str, Path)):
        candidate = value
    if candidate is None:
        return None
    path = Path(candidate)
    if not path.is_file():
        return None
    return path


def _format_float(value: float) -> str:
    """Format a finite box coordinate compactly and deterministically."""
    return format(float(value), ".8g")


def _capture_timeout(error: subprocess.TimeoutExpired) -> str:
    """Capture partial timeout streams without assuming they are textual."""
    values = []
    for value in (error.stdout, error.stderr):
        if value:
            values.append(
                value.decode(errors="replace")
                if isinstance(value, bytes)
                else str(value)
            )
    return "\n".join(values)


def _parse_vina_score(output: str) -> float | None:
    """Parse the first Vina result energy from standard output."""
    for line in output.splitlines():
        if "VINA RESULT" in line.upper():
            values = _NUMBER_PATTERN.findall(line)
            if values:
                return float(values[-3] if len(values) >= 3 else values[0])
    for line in output.splitlines():
        match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s+", line)
        if match:
            return float(match.group(1))
    return None


__all__ = ["VinaBackend"]
