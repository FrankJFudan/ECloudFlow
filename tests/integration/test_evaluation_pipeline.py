import json

from ecloudflow.cli.evaluate import _load_references
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


def test_reference_json_enables_canonical_novelty(tmp_path):
    """Configured reference molecules must reach the distribution registry."""
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"smiles": ["C(C)O"]}), encoding="utf-8")
    context = EvaluationContext(
        records=(
            GenerationRecord(canonical_smiles="CCO", attempt_id="a"),
            GenerationRecord(canonical_smiles="CCN", attempt_id="b"),
        ),
        references=_load_references(str(reference)),
    )

    result = evaluate_run(
        context,
        registry=MetricRegistry.default(),
        groups=("distribution",),
    )

    assert result.results["novelty"].value == 0.5
