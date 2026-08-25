from ecloudflow.evaluation import EvaluationContext, evaluate_run
from ecloudflow.sampling.results import GenerationRecord
from ecloudflow.visualization import build_report, plot_quality_speed_pareto


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


def test_build_report_prefers_ranked_docking_scores_for_distribution(tmp_path) -> None:
    """Ranked report figures must not request a nonexistent generic column."""
    build_report(
        {
            "rows": [
                {"docking_score": -8.2, "qed": 0.63, "elapsed_seconds": 0.2},
                {"docking_score": -7.5, "qed": 0.52, "elapsed_seconds": 0.4},
            ]
        },
        tmp_path,
    )

    document = (tmp_path / "metric_distribution.svg").read_text(encoding="utf-8")
    assert "<!-- docking_score -->" in document


def test_build_report_reads_raw_generation_valid_rows(tmp_path) -> None:
    """Reports built before evaluation must still use generation molecule rows."""
    build_report(
        {
            "valid": [
                {"docking_score": -8.2, "qed": 0.63},
                {"docking_score": -7.5, "qed": 0.52},
            ]
        },
        tmp_path,
    )

    document = (tmp_path / "metric_distribution.svg").read_text(encoding="utf-8")
    assert "<!-- docking_score -->" in document


def test_build_report_uses_evaluation_context_molecule_rows(tmp_path) -> None:
    """The direct EvaluationResult API must preserve its molecular metrics."""
    evaluation = evaluate_run(
        EvaluationContext(
            records=(
                GenerationRecord(
                    canonical_smiles="CCO",
                    attempt_id="a",
                    properties={"docking_score": -8.2, "qed": 0.63},
                ),
            )
        )
    )

    build_report(evaluation, tmp_path)

    document = (tmp_path / "metric_distribution.svg").read_text(encoding="utf-8")
    assert "<!-- docking_score -->" in document


def test_quality_speed_pareto_uses_elapsed_seconds_axis(tmp_path) -> None:
    """Measured attempt duration must drive the public speed axis."""
    destination = plot_quality_speed_pareto(
        [
            {"qed": 0.63, "elapsed_seconds": 0.2},
            {"qed": 0.52, "elapsed_seconds": 0.4},
        ],
        tmp_path / "pareto.svg",
    )

    document = destination.read_text(encoding="utf-8")
    assert "<!-- Elapsed time (s) -->" in document
