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
