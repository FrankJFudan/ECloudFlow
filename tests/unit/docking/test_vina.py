import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from ecloudflow.docking import DockingStatus, VinaBackend


def test_vina_reports_missing_binary_without_fabricating_score(tmp_path):
    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.pdbqt"
    receptor.write_text("receptor", encoding="utf-8")
    ligand.write_text("ligand", encoding="utf-8")
    result = VinaBackend(executable="definitely-not-a-vina-binary").score(
        ligand,
        receptor,
    )
    assert result.status is DockingStatus.UNAVAILABLE
    assert result.score is None


def test_vina_score_parser_handles_standard_remark():
    from ecloudflow.docking.vina import _parse_vina_score

    assert _parse_vina_score("REMARK VINA RESULT: -8.40 0.0 0.0") == -8.4


def test_vina_prepares_rdkit_ligand_and_pocket_pdb_for_normal_cli_inputs(
    tmp_path, monkeypatch
):
    """RDKit ligands and pocket PDBs should be prepared before Vina scoring."""
    pocket = Path("tests/fixtures/complex/toy_pocket.pdb")
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=7) == 0
    commands: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        if name in {"vina", "obabel"}:
            return name
        return None

    def fake_run(command, check, capture_output, text, timeout):
        commands.append(list(command))
        executable = Path(str(command[0])).name
        if executable == "obabel":
            output = Path(command[command.index("-O") + 1])
            output.write_text(
                "ATOM      1  C   LIG A   1       1.000   2.000   3.000  1.00  0.00           C\nEND\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(
            command,
            0,
            "REMARK VINA RESULT: -7.50 0.0 0.0\n",
            "",
        )

    monkeypatch.setattr("ecloudflow.docking.vina.shutil.which", fake_which)
    monkeypatch.setattr("ecloudflow.docking.vina.subprocess.run", fake_run)

    result = VinaBackend(executable="vina", timeout_seconds=10.0).score(
        molecule,
        pocket,
        output_dir=tmp_path,
    )

    assert result.status is DockingStatus.SUCCESS
    assert result.score == -7.5
    assert len([command for command in commands if Path(command[0]).name == "obabel"]) == 2
    vina_command = next(command for command in commands if Path(command[0]).name == "vina")
    assert Path(vina_command[vina_command.index("--receptor") + 1]).suffix == ".pdbqt"
    assert Path(vina_command[vina_command.index("--ligand") + 1]).suffix == ".pdbqt"
    assert vina_command[vina_command.index("--size_x") + 1] == "12"
    assert vina_command[vina_command.index("--size_y") + 1] == "12"
    assert vina_command[vina_command.index("--size_z") + 1] == "12"
    assert (tmp_path / "docked.pdbqt") == Path(vina_command[vina_command.index("--out") + 1])


def test_vina_reports_unavailable_when_converter_is_missing_for_raw_inputs(
    monkeypatch,
):
    """Raw CLI inputs should fail clearly when no PDBQT converter is available."""
    pocket = Path("tests/fixtures/complex/toy_pocket.pdb")
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=11) == 0

    def fake_which(name: str) -> str | None:
        if name == "vina":
            return name
        return None

    monkeypatch.setattr("ecloudflow.docking.vina.shutil.which", fake_which)

    result = VinaBackend(executable="vina").score(molecule, pocket)

    assert result.status is DockingStatus.UNAVAILABLE
    assert result.score is None
    assert "OpenBabel/obabel" in result.reason
