from ecloudflow.evaluation.aggregate import bootstrap_macro_summary


def test_macro_average_weights_pockets_equally():
    rows = {"small-pocket": [1.0], "large-pocket": [0.0] * 20}
    summary = bootstrap_macro_summary(rows, value="valid", seed=19, resamples=500)
    assert summary.mean == 0.5
    assert summary.ci_low <= summary.mean <= summary.ci_high


def test_macro_summary_accepts_long_form_rows():
    summary = bootstrap_macro_summary(
        [
            {"pocket_id": "a", "valid": 1.0},
            {"pocket_id": "b", "valid": 0.0},
        ],
        value="valid",
        seed=1,
        resamples=20,
    )
    assert summary.mean == 0.5
