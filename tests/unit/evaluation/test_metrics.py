from ecloudflow.evaluation import (
    EvaluationContext,
    MetricStatus,
    RDKitValidityMetric,
    UniquenessMetric,
)
from ecloudflow.sampling.results import GenerationRecord


def test_rdkit_validity_and_uniqueness_metrics_are_finite():
    records = (
        GenerationRecord(canonical_smiles="CCO", attempt_id="a"),
        GenerationRecord(canonical_smiles="CCO", attempt_id="b"),
        GenerationRecord(canonical_smiles="CCN", attempt_id="c"),
    )
    context = EvaluationContext(records=records)
    validity = RDKitValidityMetric().compute(context)
    uniqueness = UniquenessMetric().compute(context)
    assert validity.status is MetricStatus.SUCCESS
    assert validity.value == 1.0
    assert uniqueness.value == 2 / 3
