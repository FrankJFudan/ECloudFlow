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
