from ecloudflow.evaluation import EvaluationContext, MetricRegistry, evaluate_run
from ecloudflow.sampling.results import GenerationRecord


def test_default_evaluation_returns_named_results_for_all_domains():
    context = EvaluationContext(
        records=(GenerationRecord(canonical_smiles="CCO", attempt_id="a"),)
    )
    result = evaluate_run(context, registry=MetricRegistry.default())
    groups = {metric.group for metric in result.results.values()}
    assert groups == {
        "chemistry",
        "distribution",
        "geometry",
        "binding",
        "ecloud",
        "conditional",
        "efficiency",
    }
