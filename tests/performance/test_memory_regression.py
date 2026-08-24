"""Local benchmark schema and artifact regression checks."""

import json

from ecloudflow.training.benchmark import benchmark_scaling


def test_scaling_report_contains_all_required_measurements(tmp_path) -> None:
    """A minimal local run still records throughput, memory, and scaling."""
    report = benchmark_scaling(
        devices=[1],
        steps=2,
        config="experiment=h100_smoke",
        output_dir=tmp_path,
        warmup_steps=1,
    )
    row = report.rows[0]
    assert row.devices == 1
    assert row.samples_per_second > 0
    assert row.peak_memory_bytes > 0
    assert row.peak_reserved_memory_bytes >= row.peak_memory_bytes
    assert row.speedup == 1.0
    assert row.scaling_efficiency == 1.0
    assert (tmp_path / "scaling.json").is_file()
    assert json.loads((tmp_path / "scaling.json").read_text())["rows"]
