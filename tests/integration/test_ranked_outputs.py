import openpyxl
from rdkit import Chem

from ecloudflow.evaluation.outputs import write_ranked_outputs
from ecloudflow.evaluation.ranking import rank_molecules
from ecloudflow.sampling.results import GenerationRecord


def test_output_bundle_has_required_tables_and_sdf_order(tmp_path):
    records = [
        GenerationRecord(
            canonical_smiles="CCO",
            attempt_id="attempt-1",
            properties={"docking_score": -8.0, "qed": 0.6, "sa": 2.0},
        ),
        GenerationRecord(
            canonical_smiles="CCN",
            attempt_id="attempt-2",
            properties={"docking_score": -7.0, "qed": 0.7, "sa": 2.0},
        ),
    ]
    ranked, failed = rank_molecules("3ZTX", records)
    bundle = write_ranked_outputs(ranked, failed, tmp_path)
    assert {path.name for path in bundle.paths} >= {
        "samples.csv",
        "samples.parquet",
        "summary.xlsx",
        "ranked.sdf",
        "summary.json",
    }
    workbook = openpyxl.load_workbook(tmp_path / "summary.xlsx", read_only=True)
    try:
        assert set(workbook.sheetnames) == {"ranked", "failed", "aggregate"}
    finally:
        workbook.close()
    supplier = Chem.SDMolSupplier(str(tmp_path / "ranked.sdf"), sanitize=False)
    ids = [molecule.GetProp("molecule_id") for molecule in supplier if molecule]
    assert ids == ["3ZTX-000001", "3ZTX-000002"]
