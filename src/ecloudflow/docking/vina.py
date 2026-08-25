"""Safe subprocess adapter for AutoDock Vina-compatible binaries."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem

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

    The adapter accepts either already prepared PDBQT paths or the normal CLI
    inputs of an RDKit ligand molecule plus a pocket PDB path.  When raw
    structures need preparation, the adapter uses a bounded temporary
    directory and a detected OpenBabel-compatible executable to prepare PDBQT
    files explicitly before launching Vina.  Missing preparation tools are
    reported as ``unavailable`` rather than silently inventing a score.
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
        box_center: Sequence[float] | None = None,
        box_size: Sequence[float] | None = None,
        ligand_path: str | Path | None = None,
        receptor_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> DockingResult:
        """Score one pose through explicit preparation and bounded Vina calls.

        :param molecule: Ligand input as either an RDKit molecule with one 3D
            conformer, a prepared ligand PDBQT path, or another file that
            OpenBabel can convert to PDBQT.
        :param pocket: Pocket input as either a prepared receptor PDBQT path
            or a coordinate file such as a pocket PDB.
        :param box_center: Optional explicit search-box center in angstroms.
            When omitted, the center is derived from available pocket and
            ligand coordinates.
        :param box_size: Optional explicit positive search-box size in
            angstroms.  When omitted, the size is derived from available
            coordinates with conservative padding and a minimum dimension.
        :param ligand_path: Optional explicit ligand structure path.  Prepared
            ``.pdbqt`` inputs are used directly; other formats are converted in
            a temporary directory when a converter is available.
        :param receptor_path: Optional explicit receptor structure path with
            the same preparation semantics as ``ligand_path``.
        :param output_dir: Optional directory for Vina output files.
        :return: Typed score result; unavailable preparation tools or failed
            conversions produce explicit statuses and never fabricate a score.
        :rtype: DockingResult
        :raises ValueError: If an explicit box contains malformed values.
        """
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
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="ecloudflow-vina-") as temporary:
            try:
                prepared = _prepare_vina_inputs(
                    molecule=molecule,
                    pocket=pocket,
                    ligand_path=ligand_path,
                    receptor_path=receptor_path,
                    box_center=box_center,
                    box_size=box_size,
                    temporary_dir=Path(temporary),
                    timeout_seconds=self.timeout_seconds,
                )
            except _PreparationError as error:
                return DockingResult(
                    score=None,
                    status=error.status,
                    backend=self.name,
                    version=self.version,
                    command=error.command,
                    raw_output=error.raw_output,
                    elapsed_seconds=time.monotonic() - started,
                    reason=error.reason,
                    metadata=error.metadata,
                )
            command = [
                str(executable),
                "--receptor",
                str(prepared.receptor_path),
                "--ligand",
                str(prepared.ligand_path),
                "--center_x",
                _format_float(prepared.box_center[0]),
                "--center_y",
                _format_float(prepared.box_center[1]),
                "--center_z",
                _format_float(prepared.box_center[2]),
                "--size_x",
                _format_float(prepared.box_size[0]),
                "--size_y",
                _format_float(prepared.box_size[1]),
                "--size_z",
                _format_float(prepared.box_size[2]),
                "--exhaustiveness",
                str(self.exhaustiveness),
            ]
            if self.seed is not None:
                command.extend(["--seed", str(self.seed)])
            if destination is not None:
                command.extend(["--out", str(destination / "docked.pdbqt")])
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
                    metadata=prepared.metadata,
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
                    metadata=prepared.metadata,
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
                metadata=prepared.metadata,
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
                metadata=prepared.metadata,
            )
        return DockingResult(
            score=score,
            status=DockingStatus.SUCCESS,
            backend=self.name,
            version=self.version,
            command=tuple(command),
            raw_output=raw_output,
            elapsed_seconds=time.monotonic() - started,
            metadata=prepared.metadata,
        )


@dataclass(frozen=True)
class _PreparedInputs:
    """Carry prepared structure paths and the resolved docking box."""

    receptor_path: Path
    ligand_path: Path
    box_center: tuple[float, float, float]
    box_size: tuple[float, float, float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _PreparationError(Exception):
    """Describe an unavailable or failed input-preparation boundary."""

    status: DockingStatus
    reason: str
    command: tuple[str, ...] = ()
    raw_output: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__init__(self.reason)
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


def _prepare_vina_inputs(
    *,
    molecule: Any,
    pocket: Any,
    ligand_path: str | Path | None,
    receptor_path: str | Path | None,
    box_center: Sequence[float] | None,
    box_size: Sequence[float] | None,
    temporary_dir: Path,
    timeout_seconds: float,
) -> _PreparedInputs:
    """Prepare ligand/receptor PDBQT files and resolve a docking box."""
    prep_commands: list[list[str]] = []
    receptor_source = _resolve_existing_source(receptor_path, pocket)
    if receptor_source is None:
        raise _PreparationError(
            DockingStatus.UNAVAILABLE,
            "receptor preparation requires a receptor_path or pocket file",
        )
    ligand_source = _resolve_existing_source(ligand_path, molecule)
    ligand_coords = _extract_coordinates(molecule, label="ligand")
    if ligand_coords is None and ligand_source is not None:
        ligand_coords = _extract_coordinates(ligand_source, label="ligand")
    receptor_coords = _extract_coordinates(receptor_source, label="receptor")
    resolved_center, resolved_size = _resolve_box(
        box_center=box_center,
        box_size=box_size,
        ligand_coords=ligand_coords,
        receptor_coords=receptor_coords,
    )
    receptor_prepared = _prepare_structure_path(
        label="receptor",
        source=receptor_source,
        value=pocket,
        temporary_dir=temporary_dir,
        converter_timeout=timeout_seconds,
        prep_commands=prep_commands,
    )
    ligand_prepared = _prepare_structure_path(
        label="ligand",
        source=ligand_source,
        value=molecule,
        temporary_dir=temporary_dir,
        converter_timeout=timeout_seconds,
        prep_commands=prep_commands,
    )
    metadata = {
        "box_center": list(resolved_center),
        "box_size": list(resolved_size),
        "preparation_commands": [list(command) for command in prep_commands],
    }
    return _PreparedInputs(
        receptor_path=receptor_prepared,
        ligand_path=ligand_prepared,
        box_center=resolved_center,
        box_size=resolved_size,
        metadata=metadata,
    )


def _resolve_existing_source(explicit: str | Path | None, value: Any) -> Path | None:
    """Resolve an existing path-like input without mutating it."""
    candidate = explicit
    if candidate is None and isinstance(value, (str, Path)):
        candidate = value
    if candidate is None:
        return None
    path = Path(candidate)
    return path if path.is_file() else None


def _prepare_structure_path(
    *,
    label: str,
    source: Path | None,
    value: Any,
    temporary_dir: Path,
    converter_timeout: float,
    prep_commands: list[list[str]],
) -> Path:
    """Return a prepared PDBQT path for one ligand or receptor input."""
    if source is not None:
        if source.suffix.lower() == ".pdbqt":
            return source
        target = temporary_dir / f"{label}.pdbqt"
        return _convert_to_pdbqt(
            source=source,
            target=target,
            timeout_seconds=converter_timeout,
            prep_commands=prep_commands,
        )
    if not isinstance(value, Chem.Mol):
        raise _PreparationError(
            DockingStatus.UNAVAILABLE,
            f"{label} preparation requires a file path or RDKit molecule",
        )
    if value.GetNumConformers() == 0:
        raise _PreparationError(
            DockingStatus.FAILED,
            f"{label} molecule must contain a 3D conformer for docking",
        )
    sdf_path = temporary_dir / f"{label}.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    try:
        writer.write(Chem.Mol(value))
    finally:
        writer.close()
    target = temporary_dir / f"{label}.pdbqt"
    return _convert_to_pdbqt(
        source=sdf_path,
        target=target,
        timeout_seconds=converter_timeout,
        prep_commands=prep_commands,
    )


def _convert_to_pdbqt(
    *,
    source: Path,
    target: Path,
    timeout_seconds: float,
    prep_commands: list[list[str]],
) -> Path:
    """Convert one structure file to PDBQT through OpenBabel-compatible tools."""
    converter = next(
        (candidate for candidate in ("obabel", "babel") if shutil.which(candidate)),
        None,
    )
    if converter is None:
        raise _PreparationError(
            DockingStatus.UNAVAILABLE,
            "OpenBabel/obabel is required to prepare non-PDBQT docking inputs",
        )
    command = [converter, str(source), "-O", str(target)]
    prep_commands.append(list(command))
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=min(timeout_seconds, 60.0),
        )
    except subprocess.TimeoutExpired as error:
        raise _PreparationError(
            DockingStatus.TIMEOUT,
            f"OpenBabel conversion timed out for {source.name}",
            command=tuple(command),
            raw_output=_capture_timeout(error),
            metadata={"preparation_commands": [list(item) for item in prep_commands]},
        ) from error
    except OSError as error:
        raise _PreparationError(
            DockingStatus.FAILED,
            f"failed to execute OpenBabel for {source.name}: {error}",
            command=tuple(command),
            metadata={"preparation_commands": [list(item) for item in prep_commands]},
        ) from error
    raw_output = "\n".join(
        value for value in (completed.stdout or "", completed.stderr or "") if value
    )
    if completed.returncode != 0:
        raise _PreparationError(
            DockingStatus.FAILED,
            f"OpenBabel exited with code {completed.returncode} while preparing {source.name}",
            command=tuple(command),
            raw_output=raw_output,
            metadata={"preparation_commands": [list(item) for item in prep_commands]},
        )
    if not target.is_file():
        raise _PreparationError(
            DockingStatus.FAILED,
            f"OpenBabel did not create prepared output for {source.name}",
            command=tuple(command),
            raw_output=raw_output,
            metadata={"preparation_commands": [list(item) for item in prep_commands]},
        )
    return target


def _resolve_box(
    *,
    box_center: Sequence[float] | None,
    box_size: Sequence[float] | None,
    ligand_coords: Sequence[tuple[float, float, float]] | None,
    receptor_coords: Sequence[tuple[float, float, float]] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolve an explicit or coordinate-derived Vina search box."""
    if box_center is not None and box_size is not None:
        return validate_box(box_center, box_size)
    coordinates = list(receptor_coords or ()) + list(ligand_coords or ())
    if not coordinates:
        raise _PreparationError(
            DockingStatus.UNAVAILABLE,
            "docking box requires explicit values or readable pocket/ligand coordinates",
        )
    minima = [min(point[index] for point in coordinates) for index in range(3)]
    maxima = [max(point[index] for point in coordinates) for index in range(3)]
    derived_center = tuple((lower + upper) / 2.0 for lower, upper in zip(minima, maxima))
    padding = 4.0 if receptor_coords else 8.0
    minimum_size = 12.0 if receptor_coords else 16.0
    derived_size = tuple(
        max(minimum_size, (upper - lower) + padding)
        for lower, upper in zip(minima, maxima)
    )
    center = derived_center if box_center is None else tuple(float(value) for value in box_center)
    size = derived_size if box_size is None else tuple(float(value) for value in box_size)
    return validate_box(center, size)


def _extract_coordinates(
    value: Any, *, label: str
) -> tuple[tuple[float, float, float], ...] | None:
    """Extract cartesian coordinates from an RDKit molecule or structure file."""
    if isinstance(value, Chem.Mol):
        if value.GetNumConformers() == 0:
            raise _PreparationError(
                DockingStatus.FAILED,
                f"{label} molecule must contain a 3D conformer for docking",
            )
        return tuple(
            tuple(float(component) for component in point)
            for point in value.GetConformer(0).GetPositions().tolist()
        )
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        molecule = next((item for item in supplier if item is not None), None)
        if molecule is None or molecule.GetNumConformers() == 0:
            raise _PreparationError(
                DockingStatus.FAILED,
                f"{label} structure {path.name} does not contain a usable conformer",
            )
        return tuple(
            tuple(float(component) for component in point)
            for point in molecule.GetConformer(0).GetPositions().tolist()
        )
    if suffix in {".mol", ".mol2"}:
        loader = Chem.MolFromMol2File if suffix == ".mol2" else Chem.MolFromMolFile
        molecule = loader(str(path), sanitize=False, removeHs=False)
        if molecule is None or molecule.GetNumConformers() == 0:
            raise _PreparationError(
                DockingStatus.FAILED,
                f"{label} structure {path.name} does not contain a usable conformer",
            )
        return tuple(
            tuple(float(component) for component in point)
            for point in molecule.GetConformer(0).GetPositions().tolist()
        )
    coordinates: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                coordinates.append(
                    (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                )
            except ValueError as error:
                raise _PreparationError(
                    DockingStatus.FAILED,
                    f"{label} structure {path.name} contains invalid atomic coordinates",
                ) from error
    if coordinates:
        return tuple(coordinates)
    return None


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
