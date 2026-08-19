from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from ecloudflow.core.types import ElectronField
from ecloudflow.ecloud.provenance import ToolProvenance
from ecloudflow.ecloud.xtb import (
    QMFailureCategory,
    QMStatus,
    XTBRunner,
    interpolate_density_cube,
    read_density_cube,
)


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parents[2] / "fixtures"


def methane_molecule() -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles("C"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=17) == 0
    return molecule


def carbon_dioxide_molecule() -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles("O=C=O"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=19) == 0
    assert all(atom.GetNumImplicitHs() == 0 for atom in molecule.GetAtoms())
    assert all(atom.GetAtomicNum() != 1 for atom in molecule.GetAtoms())
    return molecule


def test_xtb_runner_records_failed_qm_without_fake_density(tmp_path: Path):
    runner = XTBRunner(executable="missing-xtb", work_root=tmp_path)
    result = runner.calculate_ligand(methane_molecule(), charge=0, multiplicity=1)
    assert result.status is QMStatus.TOOL_MISSING
    assert result.density is None
    assert result.qm_mask is False
    assert "missing-xtb" in result.provenance.command
    assert result.provenance.charge == 0
    assert result.provenance.multiplicity == 1


def test_cube_fixture_is_parsed_to_framed_electron_field(fixture_dir: Path):
    density, grid = read_density_cube(fixture_dir / "xtb/success.cube")
    assert density.channel_names == ("density",)
    assert density.frame is not None
    assert density.positions.shape == (8, 3)
    assert density.values.shape == (8, 1)
    assert torch.all(density.mask)
    assert grid.shape == (2, 2, 2)
    assert grid.voxel_volume_angstrom3 > 0.0
    assert torch.allclose(
        density.values[:, 0],
        torch.full((8,), 1.25, dtype=torch.float64) * grid.density_scale,
    )
    assert grid.integrated_electron_count == pytest.approx(10.0)


def test_cube_interpolation_preserves_query_frame_and_zero_fills(fixture_dir: Path):
    density, grid = read_density_cube(fixture_dir / "xtb/success.cube")
    assert density.frame is not None
    positions = torch.cat(
        (
            density.positions[[0, 7]],
            torch.tensor([[50.0, 0.0, 0.0]], dtype=torch.float64),
        )
    )
    query = ElectronField(
        positions=positions,
        values=torch.zeros((3, 1), dtype=torch.float64),
        mask=torch.ones(3, dtype=torch.bool),
        batch=torch.zeros(3, dtype=torch.long),
        frame=density.frame,
    )
    result = interpolate_density_cube(density, grid, query)
    assert result.frame == query.frame
    assert torch.allclose(result.values[:2], density.values[[0, 7]])
    assert result.values[2, 0] == 0.0


def test_xtb_success_records_sources_and_uses_argument_list(
    tmp_path: Path, fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = (fixture_dir / "xtb/success.cube").read_text(encoding="utf-8")
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        assert isinstance(command, list)
        cwd = Path(str(kwargs["cwd"]))
        if "--version" not in command:
            (cwd / "density.cub").write_text(fixture, encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "* xtb version 6.7.1\n", "")
        return subprocess.CompletedProcess(command, 0, "xtb version 6.7.1\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = XTBRunner(executable="xtb-fixture", work_root=tmp_path).calculate_ligand(
        methane_molecule(), charge=0, multiplicity=1
    )
    assert result.status is QMStatus.SUCCESS
    assert result.qm_mask is True
    assert result.density is not None
    assert result.provenance.version == "6.7.1"
    assert result.integrated_electron_count == pytest.approx(10.0)
    assert result.provenance.integrated_electron_count == pytest.approx(10.0)
    assert {"molecule", "xyz", "xtb_input", "cube"} <= set(
        result.provenance.source_hashes
    )
    assert all(kwargs["capture_output"] is True for _, kwargs in calls)
    assert all(kwargs["shell"] is False for _, kwargs in calls)
    assert all("timeout" in kwargs for _, kwargs in calls)
    assert len({kwargs["cwd"] for _, kwargs in calls}) == 1


@pytest.mark.parametrize(
    ("exception", "status"),
    [
        (subprocess.TimeoutExpired(["xtb"], 1.0), QMStatus.TIMEOUT),
        (
            subprocess.CalledProcessError(3, ["xtb"], stderr="fixture failure"),
            QMStatus.NONZERO_EXIT,
        ),
    ],
)
def test_xtb_failures_are_typed_without_density(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    status: QMStatus,
):
    if isinstance(exception, subprocess.CalledProcessError):
        exception.stderr = (fixture_dir / "xtb/failure.stderr").read_text(
            encoding="utf-8"
        )

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise exception

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = XTBRunner(executable="xtb", work_root=tmp_path).calculate_ligand(
        methane_molecule(), charge=0, multiplicity=1
    )
    assert result.status is status
    assert result.density is None
    assert result.qm_mask is False


@pytest.mark.parametrize(
    ("stderr", "category"),
    [
        (
            "[ERROR] SCC did not converge after 100 iterations.",
            QMFailureCategory.SCC_NONCONVERGENCE,
        ),
        ("[ERROR] unable to open input file.", QMFailureCategory.EXECUTION),
    ],
)
def test_xtb_failure_category_sanitizes_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
    category: QMFailureCategory,
):
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(2, ["xtb"], stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = XTBRunner(executable="xtb", work_root=tmp_path).calculate_ligand(
        methane_molecule(), charge=0, multiplicity=1
    )
    assert result.failure_category is category
    assert result.provenance.failure_category == category.value
    assert stderr not in result.message


def test_partial_explicit_hydrogens_are_rejected_before_external_work(tmp_path: Path):
    molecule = methane_molecule()
    editable = Chem.RWMol(molecule)
    editable.RemoveAtom(molecule.GetNumAtoms() - 1)
    partial = editable.GetMol()
    Chem.SanitizeMol(partial)
    with pytest.raises(ValueError, match="implicit hydrogen"):
        XTBRunner(executable="missing-xtb", work_root=tmp_path).calculate_ligand(
            partial, charge=0, multiplicity=1
        )


def test_hydrogen_free_ligand_is_valid_for_xtb_input(tmp_path: Path):
    result = XTBRunner(executable="missing-xtb", work_root=tmp_path).calculate_ligand(
        carbon_dioxide_molecule(), charge=0, multiplicity=1
    )
    assert result.status is QMStatus.TOOL_MISSING


@pytest.mark.parametrize("replacement", ["-1.0", "0.0"])
def test_cube_rejects_nonphysical_density_values(
    tmp_path: Path, fixture_dir: Path, replacement: str
):
    source = (fixture_dir / "xtb/success.cube").read_text(encoding="utf-8")
    path = tmp_path / "density.cube"
    path.write_text(source.replace("1.250000E+00", replacement), encoding="utf-8")
    with pytest.raises(ValueError, match="density"):
        read_density_cube(path)


def test_cube_rejects_electron_count_mismatch(tmp_path: Path, fixture_dir: Path):
    path = tmp_path / "density.cube"
    path.write_text(
        (fixture_dir / "xtb/success.cube").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="electron count"):
        read_density_cube(path, expected_electrons=100.0)


def test_cube_interpolation_rejects_density_grid_frame_mismatch(fixture_dir: Path):
    density, grid = read_density_cube(fixture_dir / "xtb/success.cube")
    altered_positions = density.positions.clone()
    altered_positions[0, 0] += 0.25
    altered = ElectronField(
        positions=altered_positions,
        values=density.values,
        mask=density.mask,
        batch=density.batch,
        channel_names=density.channel_names,
        frame=density.frame,
    )
    query = ElectronField(
        positions=density.positions[:1],
        values=torch.zeros((1, 1), dtype=torch.float64),
        mask=torch.ones(1, dtype=torch.bool),
        batch=torch.zeros(1, dtype=torch.long),
        frame=density.frame,
    )
    with pytest.raises(ValueError, match="lattice"):
        interpolate_density_cube(altered, grid, query)


def test_cube_interpolation_rejects_same_integral_density_payload_mutation(
    fixture_dir: Path,
):
    density, grid = read_density_cube(fixture_dir / "xtb/success.cube")
    altered_values = density.values.clone()
    altered_values[0, 0] += 0.25
    altered_values[1, 0] -= 0.25
    altered = ElectronField(
        positions=density.positions,
        values=altered_values,
        mask=density.mask,
        batch=density.batch,
        channel_names=density.channel_names,
        frame=density.frame,
    )
    assert density.frame is not None
    query = ElectronField(
        positions=density.positions[:1],
        values=torch.zeros((1, 1), dtype=torch.float64),
        mask=torch.ones(1, dtype=torch.bool),
        batch=torch.zeros(1, dtype=torch.long),
        frame=density.frame,
    )
    with pytest.raises(ValueError, match="payload"):
        interpolate_density_cube(altered, grid, query)


def test_provenance_normalizes_sequence_fields():
    provenance = ToolProvenance(
        tool="xTB",
        version="unavailable",
        executable="xtb",
        command=["xtb", "--version"],
        charge=0,
        multiplicity=1,
        source_hashes={"source": "0" * 64},
    )
    assert provenance.command == ("xtb", "--version")


def test_xtb_temporary_directories_are_unique_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    directories: list[Path] = []

    def fake_run(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        workdir = Path(str(kwargs["cwd"]))
        directories.append(workdir)
        (workdir / "density.cub").write_text("not a cube\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "xtb version 6.7.1", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = XTBRunner(executable="xtb", work_root=tmp_path)
    runner.calculate_ligand(methane_molecule(), charge=0, multiplicity=1)
    runner.calculate_ligand(methane_molecule(), charge=0, multiplicity=1)
    assert len(directories) == 2
    assert directories[0] != directories[1]
    assert all(not directory.exists() for directory in directories)


def test_xtb_malformed_cube_is_a_typed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fake_run(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(str(kwargs["cwd"]), "density.cub").write_text(
            "not a cube\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "xtb version 6.7.1", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = XTBRunner(executable="xtb", work_root=tmp_path).calculate_ligand(
        methane_molecule(), charge=0, multiplicity=1
    )
    assert result.status is QMStatus.MALFORMED_CUBE
    assert result.density is None
    assert result.qm_mask is False
    assert "cube" in result.provenance.source_hashes


def test_xtb_rejects_invalid_input_before_external_work(tmp_path: Path):
    runner = XTBRunner(executable="xtb", work_root=tmp_path)
    molecule = methane_molecule()
    molecule.AddConformer(Chem.Conformer(molecule.GetConformer()), assignId=True)
    with pytest.raises(ValueError, match="exactly one"):
        runner.calculate_ligand(molecule, charge=0, multiplicity=1)
    with pytest.raises(ValueError, match="charge"):
        runner.calculate_ligand(methane_molecule(), charge=True, multiplicity=1)
    with pytest.raises(ValueError, match="multiplicity"):
        runner.calculate_ligand(methane_molecule(), charge=0, multiplicity=0)
