import json
from pathlib import Path

from ecloudflow.cli.common import load_run_records
from ecloudflow.pipeline import ECloudFlowPipeline


class _SequenceGenerator:
    def __init__(self, sequence):
        self.sequence = iter(sequence)

    def __call__(self, **kwargs):
        return next(self.sequence)


def test_pipeline_returns_target_unique_valid_count_with_bounded_attempts(tmp_path):
    pipeline = ECloudFlowPipeline(
        candidate_generator=_SequenceGenerator(["CCO", "CCO", "CCN", "invalid"]),
        checkpoint_hash="stub-checkpoint",
    )
    result = pipeline.generate(
        pocket="toy-pocket.pdb",
        num_molecules=2,
        max_attempts=4,
        output_dir=tmp_path,
    )
    assert [record.canonical_smiles for record in result.valid] == ["CCO", "CCN"]
    assert result.attempts == 3
    assert result.duplicate_count == 1
    assert (tmp_path / "generation.json").exists()


def test_relative_output_dir_keeps_serialized_pose_paths_resolvable(
    tmp_path: Path, monkeypatch
) -> None:
    """Relative CLI output directories must preserve raw conformer artifacts."""
    monkeypatch.chdir(tmp_path)
    pipeline = ECloudFlowPipeline(
        candidate_generator=_SequenceGenerator(["CCO"]),
        checkpoint_hash="stub-checkpoint",
    )
    result = pipeline.generate(
        pocket="toy-pocket.pdb",
        num_molecules=1,
        output_dir=Path("relative-run"),
    )

    record = result.valid[0]
    assert record.raw_path is not None and record.raw_path.is_absolute()
    assert record.raw_path.is_file()
    loaded = load_run_records(Path("relative-run"))
    assert loaded and loaded[0].raw_path is not None
    assert loaded[0].raw_path.is_file()
    assert loaded[0].molecule is not None
    assert loaded[0].molecule.GetNumConformers() > 0


def test_load_run_records_keeps_unranked_valid_rows_while_enriching_ranked_subset(
    tmp_path: Path,
) -> None:
    """Summary metadata should enrich matching generation rows, not replace them."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "generation.json").write_text(
        json.dumps(
            {
                "valid": [
                    {
                        "attempt_id": "attempt-001",
                        "canonical_smiles": "CCO",
                        "mode": "de_novo",
                    },
                    {
                        "attempt_id": "attempt-002",
                        "canonical_smiles": "CCN",
                        "mode": "de_novo",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "ranked": [
                    {
                        "temporary_id": "attempt-001",
                        "canonical_smiles": "CCO",
                        "molecule_id": "POCKET-000001",
                        "rank": 1,
                        "docking_score": -8.1,
                        "qed": 0.72,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = load_run_records(run_dir)

    assert [record.attempt_id for record in records] == ["attempt-001", "attempt-002"]
    assert records[0].properties["molecule_id"] == "POCKET-000001"
    assert records[0].properties["rank"] == 1
    assert records[0].properties["docking_score"] == -8.1
    assert records[1].canonical_smiles == "CCN"
    assert "docking_score" not in records[1].properties


def test_load_run_records_retains_valid_attempt_elapsed_seconds(tmp_path: Path) -> None:
    """Per-molecule report rows must retain measured generation time."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "generation.json").write_text(
        json.dumps(
            {
                "valid": [
                    {
                        "attempt_id": "attempt-001",
                        "canonical_smiles": "CCO",
                        "mode": "de_novo",
                    },
                    {
                        "attempt_id": "attempt-002",
                        "canonical_smiles": "CCN",
                        "mode": "de_novo",
                    },
                ],
                "attempt_records": [
                    {
                        "attempt_id": "attempt-001",
                        "status": "valid",
                        "elapsed_seconds": 0.125,
                    },
                    {
                        "attempt_id": "attempt-002",
                        "status": "valid",
                        "elapsed_seconds": 0.25,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    records = load_run_records(run_dir)

    assert [record.properties["elapsed_seconds"] for record in records] == [
        0.125,
        0.25,
    ]
