from ecloudflow.evaluation import (
    EvaluationContext,
    MetricRegistry,
    MetricStatus,
    VinaScoreMetric,
    evaluate_run,
)
from ecloudflow.sampling.results import GenerationRecord


def test_registry_exposes_all_seven_metric_domains():
    registry = MetricRegistry.default()
    assert set(registry.groups) == {
        "chemistry",
        "distribution",
        "geometry",
        "binding",
        "ecloud",
        "conditional",
        "efficiency",
    }


def test_optional_vina_metric_reports_unavailable_without_fake_value():
    record = GenerationRecord(canonical_smiles="CCO", attempt_id="attempt-1")
    result = VinaScoreMetric(backend=None).compute(record)
    assert result.status == MetricStatus.UNAVAILABLE
    assert result.value is None


def test_evaluation_does_not_mutate_generation_records():
    record = GenerationRecord(canonical_smiles="CCO", attempt_id="attempt-1")
    before = dict(record.properties)
    evaluate_run(EvaluationContext(records=(record,)))
    assert dict(record.properties) == before
