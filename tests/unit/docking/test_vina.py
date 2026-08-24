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
