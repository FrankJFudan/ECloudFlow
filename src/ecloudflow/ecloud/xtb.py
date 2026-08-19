"""Safe xTB electron-density adapter and minimal Gaussian cube reader.

Cube conventions were algorithmically adapted from ECloudGen as documented by
:data:`ecloudflow.ecloud.provenance.ECLOUDGEN_CUBE_ATTRIBUTION`. The upstream
snapshot has no stated code license (``NOASSERTION``); this module is an
independent, strict implementation with explicit units, frames, and failures.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from rdkit import Chem

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import ElectronField
from ecloudflow.ecloud.provenance import ToolProvenance

BOHR_PER_ANGSTROM = 1.8897259886
_XTB_INPUT = """$cube
  step=0.9448629943
$end
$write
  density=true
  spin density=false
$end
"""


class QMStatus(str, Enum):
    """Enumerate typed outcomes of one quantum-chemistry attempt."""

    SUCCESS = "success"
    TOOL_MISSING = "tool_missing"
    TIMEOUT = "timeout"
    NONZERO_EXIT = "nonzero_exit"
    MALFORMED_CUBE = "malformed_cube"


class QMFailureCategory(str, Enum):
    """Enumerate sanitized, credential-free xTB failure categories."""

    NONE = "none"
    TOOL_MISSING = "tool_missing"
    TIMEOUT = "timeout"
    SCC_NONCONVERGENCE = "scc_nonconvergence"
    EXECUTION = "execution"
    MALFORMED_CUBE = "malformed_cube"


@dataclass(frozen=True)
class CubeGrid:
    """Describe a parsed Gaussian cube lattice in explicit units.

    :param origin_angstrom: Global cube origin with shape ``[3]`` in Å.
    :param axes_angstrom: Three voxel-step vectors with shape ``[3, 3]`` in Å.
    :param shape: Positive voxel counts in cube axis order.
    :param voxel_volume_angstrom3: Positive voxel volume in Å³.
    :param density_scale: Conversion applied to source density, normally
        ``bohr**3 per angstrom**3`` for an atomic-unit xTB cube.
    :param source_sha256: SHA-256 hash of the exact cube bytes.
    :param integrated_electron_count: Integral of the accepted density over
        the cube volume, in electrons.
    :return: Immutable cube grid metadata.
    :rtype: CubeGrid
    """

    origin_angstrom: torch.Tensor
    axes_angstrom: torch.Tensor
    shape: tuple[int, int, int]
    voxel_volume_angstrom3: float
    density_scale: float
    source_sha256: str
    integrated_electron_count: float = 0.0

    def __post_init__(self) -> None:
        """Validate lattice shape, units, finiteness, and source hash.

        :return: None.
        :rtype: None
        :raises ValueError: If lattice metadata cannot describe a cube.
        """
        if self.origin_angstrom.shape != (3,) or self.axes_angstrom.shape != (3, 3):
            raise ValueError("cube origin and axes must have shapes [3] and [3, 3].")
        if (
            not self.origin_angstrom.is_floating_point()
            or not self.axes_angstrom.is_floating_point()
        ):
            raise ValueError("cube origin and axes must have floating dtypes.")
        if (
            not torch.isfinite(self.origin_angstrom).all()
            or not torch.isfinite(self.axes_angstrom).all()
        ):
            raise ValueError("cube origin and axes must be finite.")
        if len(self.shape) != 3 or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in self.shape
        ):
            raise ValueError("cube shape must contain three positive integers.")
        if (
            not math.isfinite(self.voxel_volume_angstrom3)
            or self.voxel_volume_angstrom3 <= 0.0
            or not math.isfinite(self.density_scale)
            or self.density_scale <= 0.0
        ):
            raise ValueError(
                "cube volume and density scale must be positive and finite."
            )
        if (
            not math.isfinite(self.integrated_electron_count)
            or self.integrated_electron_count <= 0.0
        ):
            raise ValueError(
                "cube integrated electron count must be positive and finite."
            )
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 hash.")


@dataclass(frozen=True)
class QMResult:
    """Represent success or typed failure of a ligand density calculation.

    :param status: Typed calculation outcome.
    :param density: Framed electron density in electrons/Å³, or ``None`` on failure.
    :param grid: Parsed cube metadata, or ``None`` when no valid cube exists.
    :param qm_mask: True only when ``density`` is genuine successful xTB output.
    :param provenance: Complete external-tool attempt provenance.
    :param message: Non-sensitive diagnostic summary; stderr itself is hashed.
    :param failure_category: Sanitized category derived from status and stderr.
    :return: Immutable quantum-chemistry result.
    :rtype: QMResult
    :raises ValueError: If status, mask, density, and grid disagree.
    """

    status: QMStatus
    density: ElectronField | None
    grid: CubeGrid | None
    qm_mask: bool
    provenance: ToolProvenance
    message: str = ""
    failure_category: QMFailureCategory = QMFailureCategory.NONE

    def __post_init__(self) -> None:
        """Reject inconsistent success and failure payloads.

        :return: None.
        :rtype: None
        :raises ValueError: If successful data are absent or failure data are faked.
        """
        if not isinstance(self.status, QMStatus):
            raise ValueError("status must be a QMStatus value.")  # noqa: TRY004
        if not isinstance(self.provenance, ToolProvenance):
            raise ValueError("provenance must be ToolProvenance.")  # noqa: TRY004
        if not isinstance(self.failure_category, QMFailureCategory):
            raise ValueError("failure_category must be a QMFailureCategory.")  # noqa: TRY004
        success = self.status is QMStatus.SUCCESS
        if success != self.qm_mask:
            raise ValueError("qm_mask must be true exactly for successful QM output.")
        if success != (self.density is not None and self.grid is not None):
            raise ValueError(
                "density and grid must exist exactly for successful QM output."
            )

    @property
    def integrated_electron_count(self) -> float | None:
        """Return the accepted cube electron count, or ``None`` on failure."""
        return None if self.grid is None else self.grid.integrated_electron_count


class XTBRunner:
    """Run isolated xTB ligand electron-density calculations."""

    def __init__(
        self,
        executable: str = "xtb",
        *,
        timeout: float = 120.0,
        work_root: Path | None = None,
    ) -> None:
        """Configure a safe xTB subprocess adapter.

        :param executable: Executable name or explicit path passed as argv[0].
        :param timeout: Positive finite timeout in seconds for each subprocess.
        :param work_root: Optional parent for unique temporary directories.
        :return: Configured runner; no process is started by construction.
        :rtype: None
        :raises ValueError: If executable or timeout is invalid.

        Temporary calculation directories are removed after every result.
        ``work_root`` itself is never deleted.
        """
        if not isinstance(executable, str) or not executable.strip():
            raise ValueError("executable must be a non-empty string.")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number.")
        self.executable = executable
        self.timeout = float(timeout)
        self.work_root = None if work_root is None else Path(work_root)

    def calculate_ligand(
        self,
        molecule: Chem.Mol,
        charge: int,
        multiplicity: int,
    ) -> QMResult:
        """Calculate a ligand electron-density cube with an isolated xTB run.

        :param molecule: Sanitized RDKit molecule with one three-dimensional
            conformer and explicit hydrogens.
        :param charge: Integer molecular charge passed to xTB.
        :param multiplicity: Positive spin multiplicity.
        :return: Density, grid metadata, status, mask, and complete provenance.
        :rtype: QMResult
        :raises ValueError: If molecule sanitation, coordinates, explicit
            hydrogens, charge, or multiplicity are invalid.

        The process runs in a unique temporary directory. It writes ``ligand.xyz``
        and ``xtb.inp`` there, captures stdout/stderr, uses no shell, and removes
        the directory afterward. Missing executables, timeouts, non-zero exits,
        and malformed cubes return typed failures with ``qm_mask=False`` and no
        density. No approximate density is ever substituted. Commands and source
        hashes are recorded; environment values and credentials are never stored.
        """
        _validate_calculation_input(molecule, charge, multiplicity)
        xyz = Chem.MolToXYZBlock(molecule)
        molecule_source = Chem.MolToMolBlock(molecule, includeStereo=True)
        command = (
            self.executable,
            "ligand.xyz",
            "--gfn",
            "2",
            "--chrg",
            str(charge),
            "--uhf",
            str(multiplicity - 1),
            "--norestart",
            "--input",
            "xtb.inp",
        )
        hashes = {
            "molecule": _sha256_text(molecule_source),
            "xyz": _sha256_text(xyz),
            "xtb_input": _sha256_text(_XTB_INPUT),
        }
        if self.work_root is not None:
            self.work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="ecloudflow-xtb-",
            dir=self.work_root,
        ) as temporary:
            workdir = Path(temporary)
            (workdir / "ligand.xyz").write_text(xyz, encoding="utf-8")
            (workdir / "xtb.inp").write_text(_XTB_INPUT, encoding="utf-8")
            try:
                completed = subprocess.run(
                    list(command),
                    cwd=workdir,
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                    check=True,
                    shell=False,
                )
            except FileNotFoundError as error:
                return self._failed(
                    QMStatus.TOOL_MISSING,
                    command,
                    charge,
                    multiplicity,
                    hashes,
                    f"xTB executable was not found: {error.filename or self.executable}",
                    category=QMFailureCategory.TOOL_MISSING,
                )
            except subprocess.TimeoutExpired as error:
                _hash_process_streams(hashes, error.stdout, error.stderr)
                return self._failed(
                    QMStatus.TIMEOUT,
                    command,
                    charge,
                    multiplicity,
                    hashes,
                    f"xTB exceeded the {self.timeout:g} second timeout.",
                    category=QMFailureCategory.TIMEOUT,
                )
            except subprocess.CalledProcessError as error:
                _hash_process_streams(hashes, error.stdout, error.stderr)
                return self._failed(
                    QMStatus.NONZERO_EXIT,
                    command,
                    charge,
                    multiplicity,
                    hashes,
                    f"xTB exited with code {error.returncode}.",
                    category=_classify_failure_category(error.stderr),
                )
            _hash_process_streams(hashes, completed.stdout, completed.stderr)
            if completed.returncode != 0:
                return self._failed(
                    QMStatus.NONZERO_EXIT,
                    command,
                    charge,
                    multiplicity,
                    hashes,
                    f"xTB exited with code {completed.returncode}.",
                    category=_classify_failure_category(completed.stderr),
                )
            version = _parse_xtb_version(completed.stdout + "\n" + completed.stderr)
            if version == "unavailable":
                version = self._query_version(workdir)
            cube_path = workdir / "density.cub"
            try:
                density, grid = read_density_cube(
                    cube_path,
                    expected_electrons=float(
                        sum(atom.GetAtomicNum() for atom in molecule.GetAtoms())
                        - charge
                    ),
                )
            except (OSError, ValueError) as error:
                try:
                    hashes["cube"] = hashlib.sha256(cube_path.read_bytes()).hexdigest()
                except OSError:
                    pass
                return self._failed(
                    QMStatus.MALFORMED_CUBE,
                    command,
                    charge,
                    multiplicity,
                    hashes,
                    f"xTB density cube is missing or malformed: {error}",
                    version=version,
                    category=QMFailureCategory.MALFORMED_CUBE,
                )
            hashes["cube"] = grid.source_sha256
            provenance = ToolProvenance(
                tool="xTB",
                version=version,
                executable=self.executable,
                command=command,
                charge=charge,
                multiplicity=multiplicity,
                source_hashes=hashes,
                integrated_electron_count=grid.integrated_electron_count,
                failure_category=QMFailureCategory.NONE.value,
            )
            return QMResult(
                status=QMStatus.SUCCESS,
                density=density,
                grid=grid,
                qm_mask=True,
                provenance=provenance,
            )

    def _query_version(self, workdir: Path) -> str:
        """Query xTB version safely in the current isolated directory.

        :param workdir: Existing unique calculation directory.
        :return: Parsed version or ``"unavailable"`` on any query failure.
        :rtype: str

        This second subprocess has the same timeout, capture, argv-list, and
        no-shell guarantees as the main calculation. Its text is not retained.
        """
        try:
            completed = subprocess.run(
                [self.executable, "--version"],
                cwd=workdir,
                timeout=self.timeout,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
        return _parse_xtb_version(completed.stdout + "\n" + completed.stderr)

    def _failed(
        self,
        status: QMStatus,
        command: tuple[str, ...],
        charge: int,
        multiplicity: int,
        hashes: dict[str, str],
        message: str,
        *,
        version: str = "unavailable",
        category: QMFailureCategory = QMFailureCategory.EXECUTION,
    ) -> QMResult:
        """Build a failure result that cannot carry density.

        :param status: Non-success typed outcome.
        :param command: Exact attempted argv tuple.
        :param charge: Validated molecular charge.
        :param multiplicity: Validated spin multiplicity.
        :param hashes: Available source and process-stream hashes.
        :param message: Concise non-sensitive failure summary.
        :param version: Parsed version when known.
        :return: Failed result with ``density=None`` and ``qm_mask=False``.
        :rtype: QMResult
        :raises ValueError: If called with a success status.
        """
        if status is QMStatus.SUCCESS:
            raise ValueError("_failed cannot create a successful result.")
        return QMResult(
            status=status,
            density=None,
            grid=None,
            qm_mask=False,
            provenance=ToolProvenance(
                tool="xTB",
                version=version,
                executable=self.executable,
                command=command,
                charge=charge,
                multiplicity=multiplicity,
                source_hashes=hashes,
                failure_category=category.value,
            ),
            message=message,
            failure_category=category,
        )


def read_density_cube(
    path: Path,
    *,
    expected_electrons: float | None = None,
    electron_count_tolerance: float = 0.20,
) -> tuple[ElectronField, CubeGrid]:
    """Read the minimal scalar Gaussian cube emitted by xTB.

    :param path: Existing cube path. Positive axis counts mean Bohr coordinates;
        negative counts mean Å coordinates according to Gaussian cube convention.
    :param expected_electrons: Optional molecular electron count used to reject
        a cube whose finite-box integral differs by more than the documented
        tolerance.
    :param electron_count_tolerance: Relative tolerance for the optional count
        check. A minimum absolute tolerance of 0.5 electrons accounts for
        finite-grid quadrature and cube truncation.
    :return: Framed flattened ``ElectronField`` in electrons/Å³ and immutable
        lattice metadata. Positions are local to a frame whose origin is the
        arithmetic mean of all global grid positions.
    :rtype: tuple[ElectronField, CubeGrid]
    :raises OSError: If the cube cannot be read.
    :raises ValueError: If headers, axes, counts, values, or finiteness are invalid.

    Only one scalar orbital/density value per voxel is supported. Non-orthogonal
    axes are preserved. Source values are treated as electrons/Bohr³, matching
    xTB density output, and multiplied by ``BOHR_PER_ANGSTROM**3``. The parser
    allocates no interpolation fallback and never repairs malformed data.
    """
    source = Path(path).read_bytes()
    try:
        lines = source.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("cube must be UTF-8 text.") from error
    if len(lines) < 6:
        raise ValueError("cube header must contain at least six lines.")
    atom_count, origin = _parse_header_vector(lines[2], "atom/origin")
    if atom_count < 0:
        raise ValueError("orbital cube files are not supported.")
    counts: list[int] = []
    axes: list[list[float]] = []
    signs: set[int] = set()
    for line, name in zip(lines[3:6], ("x axis", "y axis", "z axis"), strict=True):
        count, vector = _parse_header_vector(line, name)
        if count == 0:
            raise ValueError("cube axis counts must be non-zero.")
        signs.add(1 if count > 0 else -1)
        counts.append(abs(count))
        axes.append(vector)
    if len(signs) != 1:
        raise ValueError("cube axis count signs must use one coordinate unit.")
    data_start = 6 + atom_count
    if len(lines) < data_start:
        raise ValueError("cube atom records are truncated.")
    for atom_line in lines[6:data_start]:
        parts = atom_line.split()
        if len(parts) < 5:
            raise ValueError("cube atom records must contain five values.")
        try:
            [float(value) for value in parts[:5]]
        except ValueError as error:
            raise ValueError("cube atom records must be numeric.") from error
    value_tokens = " ".join(lines[data_start:]).split()
    expected = counts[0] * counts[1] * counts[2]
    if len(value_tokens) != expected:
        raise ValueError(
            f"cube contains {len(value_tokens)} values; expected {expected}."
        )
    try:
        values = torch.tensor(
            [float(value) for value in value_tokens], dtype=torch.float64
        )
    except ValueError as error:
        raise ValueError("cube density values must be numeric.") from error
    if not torch.isfinite(values).all():
        raise ValueError("cube density values must be finite.")
    if bool((values < 0).any()):
        raise ValueError("cube density values must be non-negative.")
    coordinate_scale = 1.0 / BOHR_PER_ANGSTROM if 1 in signs else 1.0
    density_scale = BOHR_PER_ANGSTROM**3
    origin_tensor = torch.tensor(origin, dtype=torch.float64) * coordinate_scale
    axes_tensor = torch.tensor(axes, dtype=torch.float64) * coordinate_scale
    if not torch.isfinite(origin_tensor).all() or not torch.isfinite(axes_tensor).all():
        raise ValueError("cube lattice metadata must be finite.")
    volume = float(abs(torch.linalg.det(axes_tensor)))
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("cube axes must span a positive finite volume.")
    integrated_electrons = float(values.sum()) * density_scale * volume
    if not math.isfinite(integrated_electrons) or integrated_electrons <= 1.0e-8:
        raise ValueError("cube density must have a positive integrated electron count.")
    if expected_electrons is not None:
        if (
            not math.isfinite(expected_electrons)
            or expected_electrons <= 0.0
            or not math.isfinite(electron_count_tolerance)
            or electron_count_tolerance < 0.0
        ):
            raise ValueError("expected electron count and tolerance must be valid.")
        allowed_error = max(0.5, expected_electrons * electron_count_tolerance)
        if abs(integrated_electrons - expected_electrons) > allowed_error:
            raise ValueError(
                "cube integrated electron count is outside the physical electron count "
                "tolerance."
            )
    indices = torch.cartesian_prod(
        torch.arange(counts[0], dtype=torch.float64),
        torch.arange(counts[1], dtype=torch.float64),
        torch.arange(counts[2], dtype=torch.float64),
    )
    global_positions = origin_tensor + indices @ axes_tensor
    frame = CoordinateFrame(origin=global_positions.mean(dim=0))
    field = ElectronField(
        positions=frame.to_local(global_positions),
        values=(values * density_scale).unsqueeze(-1),
        mask=torch.ones(expected, dtype=torch.bool),
        batch=torch.zeros(expected, dtype=torch.long),
        channel_names=("density",),
        frame=frame,
    )
    grid = CubeGrid(
        origin_angstrom=origin_tensor,
        axes_angstrom=axes_tensor,
        shape=(counts[0], counts[1], counts[2]),
        voxel_volume_angstrom3=volume,
        density_scale=density_scale,
        source_sha256=hashlib.sha256(source).hexdigest(),
        integrated_electron_count=integrated_electrons,
    )
    return field, grid


def interpolate_density_cube(
    density: ElectronField,
    grid: CubeGrid,
    query: ElectronField,
) -> ElectronField:
    """Interpolate a parsed cube onto framed query points with zero padding.

    :param density: Single-channel field returned by :func:`read_density_cube`.
    :param grid: Lattice metadata returned from the same parser call.
    :param query: Query point contract with an explicit frame, masks, batches,
        and arbitrary placeholder values. Query positions are in ``query.frame``.
    :return: One-channel ``"density"`` field on the exact query positions,
        mask, batches, and frame. Values use electrons/Å³ and are zero outside
        the cube or at masked query points.
    :rtype: ElectronField
    :raises TypeError: If inputs do not use the typed field/grid contracts.
    :raises ValueError: If source metadata, channel layout, frame provenance,
        dtype/device, or single-cube batch assumptions are incompatible.

    This is the minimum trilinear interpolation behavior adapted from
    ECloudGen's regular-grid interpolation convention. It supports oblique cube
    axes by solving fractional lattice coordinates, performs no file or process
    side effects, and never labels interpolated pocket heuristics as QM output.
    """
    if not isinstance(density, ElectronField):
        raise TypeError("density must be an ElectronField.")
    if not isinstance(grid, CubeGrid):
        raise TypeError("grid must be a CubeGrid.")
    if not isinstance(query, ElectronField):
        raise TypeError("query must be an ElectronField.")
    if density.frame is None or query.frame is None:
        raise ValueError("density and query must have explicit coordinate frames.")
    expected = math.prod(grid.shape)
    if (
        density.positions.shape[0] != expected
        or density.values.shape != (expected, 1)
        or density.channel_names != ("density",)
        or not bool(density.mask.all())
        or not bool((density.batch == 0).all())
    ):
        raise ValueError("density does not match the parsed single-cube contract.")
    _validate_density_grid_alignment(density, grid)
    if (
        density.positions.dtype != torch.float64
        or density.positions.device.type != "cpu"
    ):
        raise ValueError("parsed cube density must retain CPU float64 metadata.")
    if query.positions.device.type != "cpu":
        raise ValueError("cube interpolation currently requires CPU query points.")
    if bool((query.batch != 0).any()):
        raise ValueError("one cube may interpolate only query batch zero.")
    global_query = query.frame.to_global(query.positions)
    work_query = global_query.to(torch.float64)
    fractional = (work_query - grid.origin_angstrom) @ torch.linalg.inv(
        grid.axes_angstrom
    )
    lower = torch.floor(fractional).to(torch.long)
    fraction = fractional - lower.to(torch.float64)
    inside = query.mask.clone()
    for axis, size in enumerate(grid.shape):
        if size == 1:
            inside &= fractional[:, axis].abs() <= 1e-8
            lower[:, axis] = 0
            fraction[:, axis] = 0.0
        else:
            inside &= (fractional[:, axis] >= 0.0) & (fractional[:, axis] <= size - 1)
            lower[:, axis].clamp_(0, size - 2)
            fraction[:, axis] = fractional[:, axis] - lower[:, axis].to(torch.float64)
    cube = density.values[:, 0].reshape(grid.shape)
    output = torch.zeros(query.positions.shape[0], dtype=torch.float64)
    valid_indices = torch.nonzero(inside, as_tuple=False).flatten()
    if valid_indices.numel():
        base = lower[valid_indices]
        weights = fraction[valid_indices]
        interpolated = torch.zeros(valid_indices.numel(), dtype=torch.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    offsets = torch.tensor([dx, dy, dz], dtype=torch.long)
                    index = base + offsets
                    for axis, size in enumerate(grid.shape):
                        if size == 1:
                            index[:, axis] = 0
                    corner_weight = (
                        (weights[:, 0] if dx else 1.0 - weights[:, 0])
                        * (weights[:, 1] if dy else 1.0 - weights[:, 1])
                        * (weights[:, 2] if dz else 1.0 - weights[:, 2])
                    )
                    interpolated += (
                        corner_weight * cube[index[:, 0], index[:, 1], index[:, 2]]
                    )
        output[valid_indices] = interpolated
    return ElectronField(
        positions=query.positions,
        values=output.to(query.positions.dtype).unsqueeze(-1),
        mask=query.mask,
        batch=query.batch,
        channel_names=("density",),
        frame=query.frame,
    )


def _parse_header_vector(line: str, name: str) -> tuple[int, list[float]]:
    """Parse a cube integer and three-vector header line."""
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"cube {name} line must contain four values.")
    try:
        count = int(parts[0])
        vector = [float(value) for value in parts[1:4]]
    except ValueError as error:
        raise ValueError(f"cube {name} line must be numeric.") from error
    return count, vector


def _validate_calculation_input(
    molecule: Chem.Mol, charge: int, multiplicity: int
) -> None:
    """Validate chemistry and coordinates before any external side effect."""
    if not isinstance(molecule, Chem.Mol):
        raise ValueError("molecule must be an RDKit Mol.")  # noqa: TRY004
    if not isinstance(charge, int) or isinstance(charge, bool):
        raise ValueError("charge must be an integer.")  # noqa: TRY004
    if (
        not isinstance(multiplicity, int)
        or isinstance(multiplicity, bool)
        or multiplicity <= 0
    ):
        raise ValueError("multiplicity must be a positive integer.")
    sanitized = Chem.Mol(molecule)
    status = Chem.SanitizeMol(sanitized, catchErrors=True)
    if status != Chem.SanitizeFlags.SANITIZE_NONE:
        raise ValueError("molecule must be sanitizable with RDKit.")
    if molecule.GetNumConformers() != 1:
        raise ValueError("molecule must contain exactly one conformer.")
    conformer = molecule.GetConformer()
    if not conformer.Is3D():
        raise ValueError("molecule conformer must be three-dimensional.")
    coordinates = torch.tensor(conformer.GetPositions(), dtype=torch.float64)
    if (
        coordinates.shape != (molecule.GetNumAtoms(), 3)
        or not torch.isfinite(coordinates).all()
    ):
        raise ValueError("molecule coordinates must be finite Cartesian values.")
    if molecule.GetNumAtoms() == 0:
        raise ValueError("molecule must contain atoms.")
    if sanitized.GetNumHeavyAtoms() and not any(
        atom.GetAtomicNum() == 1 for atom in sanitized.GetAtoms()
    ):
        raise ValueError("molecule must contain explicit hydrogens.")
    implicit_hydrogen_atoms = [
        atom.GetIdx()
        for atom in sanitized.GetAtoms()
        if atom.GetAtomicNum() != 1 and atom.GetNumImplicitHs() != 0
    ]
    if implicit_hydrogen_atoms:
        raise ValueError(
            "molecule must not contain implicit hydrogen atoms; explicit hydrogen "
            f"validation failed for atom indices {implicit_hydrogen_atoms}."
        )


def _parse_xtb_version(text: str) -> str:
    """Extract a compact xTB semantic version from captured process text."""
    match = re.search(
        r"\b(?:xtb\s+)?version\s*:?[ \t]*v?(\d+\.\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    return match.group(1) if match else "unavailable"


def _classify_failure_category(stderr: str | bytes | None) -> QMFailureCategory:
    """Classify stderr without retaining or exposing its sensitive contents."""
    if stderr is None:
        return QMFailureCategory.EXECUTION
    text = (
        stderr.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes)
        else stderr
    )
    normalized = text.lower()
    if "scc" in normalized and ("converg" in normalized or "scf" in normalized):
        return QMFailureCategory.SCC_NONCONVERGENCE
    if "did not converge" in normalized or "failed to converge" in normalized:
        return QMFailureCategory.SCC_NONCONVERGENCE
    return QMFailureCategory.EXECUTION


def _sha256_text(text: str) -> str:
    """Hash one UTF-8 text source without retaining its contents."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_process_streams(
    hashes: dict[str, str], stdout: str | bytes | None, stderr: str | bytes | None
) -> None:
    """Hash captured process streams without recording sensitive text."""
    for name, value in (("stdout", stdout), ("stderr", stderr)):
        if value is None:
            continue
        encoded = value if isinstance(value, bytes) else value.encode("utf-8")
        hashes[name] = hashlib.sha256(encoded).hexdigest()


def _validate_density_grid_alignment(density: ElectronField, grid: CubeGrid) -> None:
    """Validate source field coordinates and frame against cube lattice metadata."""
    if density.frame is None:
        raise ValueError("density must have a coordinate frame for lattice validation.")
    indices = torch.cartesian_prod(
        torch.arange(grid.shape[0], dtype=torch.float64),
        torch.arange(grid.shape[1], dtype=torch.float64),
        torch.arange(grid.shape[2], dtype=torch.float64),
    )
    expected_global = grid.origin_angstrom + indices @ grid.axes_angstrom
    try:
        actual_global = density.frame.to_global(density.positions)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "density frame is incompatible with the cube lattice."
        ) from error
    if not torch.allclose(actual_global, expected_global, rtol=1.0e-7, atol=1.0e-8):
        raise ValueError("density positions do not match the cube lattice.")
    expected_origin = expected_global.mean(dim=0)
    if not torch.allclose(
        density.frame.origin, expected_origin, rtol=1.0e-7, atol=1.0e-8
    ):
        raise ValueError("density frame origin does not match the cube lattice.")
    if not torch.isfinite(density.values).all() or bool((density.values < 0).any()):
        raise ValueError("density values do not satisfy the physical cube contract.")
    integrated = float(density.values[:, 0].sum()) * grid.voxel_volume_angstrom3
    if not math.isclose(
        integrated,
        grid.integrated_electron_count,
        rel_tol=1.0e-7,
        abs_tol=1.0e-8,
    ):
        raise ValueError("density values do not match the cube lattice electron count.")
