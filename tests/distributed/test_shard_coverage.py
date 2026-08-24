"""Static and local checks for benchmark/report artifact coverage."""

from pathlib import Path

from ecloudflow.training.benchmark import benchmark_scaling


def test_scaling_rows_keep_fixed_global_workload(tmp_path: Path) -> None:
    """Strong scaling rows must expose the same global batch size."""
    report = benchmark_scaling(
        devices=[1, 2],
        steps=1,
        config="experiment=h100_smoke",
        output_dir=tmp_path,
        warmup_steps=0,
        global_batch_size=4,
    )
    assert {row.global_batch_size for row in report.rows} == {4}
    assert report.rows[0].speedup == 1.0
