import pytest

from ecloudflow.sampling.results import (
    GenerationAttempt,
    GenerationMode,
    GenerationRecord,
    GenerationResult,
)


def test_generation_result_reports_attempt_counts_and_deduplicates_records():
    first = GenerationRecord(canonical_smiles="CCO", attempt_id="attempt-000001")
    second = GenerationRecord(canonical_smiles="CCN", attempt_id="attempt-000003")
    result = GenerationResult(
        valid=(first, second),
        attempt_records=(
            GenerationAttempt(attempt_id="attempt-000001", status="valid", record=first),
            GenerationAttempt(attempt_id="attempt-000002", status="rejected", reason="duplicate"),
            GenerationAttempt(attempt_id="attempt-000003", status="valid", record=second),
        ),
        target_count=2,
        duplicate_count=1,
        model_checkpoint_hash="abc",
    )
    assert result.attempts == 3
    assert result.valid_count == 2
    assert result.shortfall == 0
    assert GenerationMode("merge") is GenerationMode.MERGE


def test_generation_result_to_excel_writes_ranked_sheet(tmp_path):
    record = GenerationRecord(canonical_smiles="CCO", attempt_id="attempt-000001")
    result = GenerationResult(
        valid=(record,),
        attempt_records=(GenerationAttempt("attempt-000001", "valid", record=record),),
        target_count=1,
        model_checkpoint_hash="abc",
    )
    path = result.to_excel(tmp_path / "summary.xlsx")
    assert path.exists()
    assert path.stat().st_size > 0


def test_generation_attempt_rejects_unknown_status():
    with pytest.raises(ValueError):
        GenerationAttempt("attempt-000001", "unknown")
