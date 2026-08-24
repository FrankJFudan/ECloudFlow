from rdkit import Chem

from ecloudflow.evaluation.ranking import rank_molecules
from ecloudflow.sampling.results import GenerationRecord


def _record(name, score, qed, sa, status="success"):
    return GenerationRecord(
        canonical_smiles=name,
        attempt_id=f"attempt-{name}",
        properties={
            "docking_score": score,
            "qed": qed,
            "sa": sa,
            "status": status,
        },
    )


def test_ranking_uses_vina_qed_sa_smiles_and_no_eclf():
    records = [
        _record("B", -9.0, 0.7, 2.0),
        _record("A", -10.0, 0.6, 3.0),
        _record("C", -9.0, 0.8, 4.0),
        _record("D", None, 0.9, 1.0, status="dock_failed"),
    ]
    ranked, unranked = rank_molecules("3ZTX", records)
    assert [item.canonical_smiles for item in ranked] == ["A", "C", "B"]
    assert [item.molecule_id for item in ranked] == [
        "3ZTX-000001",
        "3ZTX-000002",
        "3ZTX-000003",
    ]
    assert all("ECLF" not in item.molecule_id for item in ranked)
    assert unranked[0].temporary_id == records[-1].temporary_id


def test_ranking_derives_qed_and_standard_sa_when_properties_are_missing():
    record = GenerationRecord(
        canonical_smiles="CCO",
        attempt_id="attempt-ethanol",
        molecule=Chem.MolFromSmiles("CCO"),
        properties={"docking_score": -5.0},
    )

    ranked, unranked = rank_molecules("POCKET", [record])

    assert not unranked
    assert ranked[0].qed is not None
    assert 0.0 < ranked[0].qed < 1.0
    assert ranked[0].sa_score is not None
    assert 1.0 <= ranked[0].sa_score <= 10.0
