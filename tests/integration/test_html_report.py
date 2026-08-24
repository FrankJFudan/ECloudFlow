from ecloudflow.evaluation import EvaluationContext, evaluate_run
from ecloudflow.sampling.results import GenerationRecord
from ecloudflow.visualization import build_report


def test_build_report_writes_self_contained_html_and_figures(tmp_path):
    evaluation = evaluate_run(
        EvaluationContext(
            records=(GenerationRecord(canonical_smiles="CCO", attempt_id="a"),)
        )
    )
    bundle = build_report(evaluation, tmp_path, top_n=5)
    assert bundle.html_path.exists()
    assert "ECloudFlow evaluation report" in bundle.html_path.read_text(
        encoding="utf-8"
    )
    assert (tmp_path / "metric_distribution.svg").exists()
