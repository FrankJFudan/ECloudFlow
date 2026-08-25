import json
from pathlib import Path

import torch

import ecloudflow.training.benchmark as benchmark_module
from ecloudflow.config import BenchmarkConfig, DataConfig, ModelConfig
from ecloudflow.exceptions import BenchmarkError
from ecloudflow.models import ECloudFlowModel
from ecloudflow.training.benchmark import (
    benchmark_hashes,
    benchmark_scaling,
    merge_scaling_reports,
)


def test_benchmark_scaling_dry_run_writes_artifacts(tmp_path: Path) -> None:
    report = benchmark_scaling(
        [1, 2, 4],
        steps=2,
        config="experiment=h100_smoke",
        output_dir=tmp_path,
        dry_run=True,
    )
    assert [row.devices for row in report.rows] == [1, 2, 4]
    assert report.rows[0].nfe is None
    assert report.rows[0].valid_count is None
    assert report.rows[0].valid_yield is None
    assert report.rows[0].workload == "schema-simulation"
    assert report.rows[0].speedup == 1.0
    assert report.rows[0].peak_memory_bytes > 0
    assert report.rows[0].peak_reserved_memory_bytes >= report.rows[0].peak_memory_bytes
    assert report.rows[1].scaling_efficiency <= 1.0
    assert (tmp_path / "benchmark.json").exists()
    assert (tmp_path / "benchmark.csv").exists()
    assert (tmp_path / "environment.json").exists()


def test_benchmark_scripts_contain_nccl_contracts() -> None:
    smoke = Path("scripts/run_h100_smoke.sh").read_text(encoding="utf-8")
    scaling = Path("scripts/benchmark_scaling.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in smoke
    assert "--nproc_per_node=4" in smoke
    assert "ECLOUDFLOW_RUN_NCCL=1" in scaling
    assert "--nproc_per_node=1" in scaling
    assert "--nproc_per_node=2" in scaling
    assert "--nproc_per_node=4" in scaling
    assert "merge_scaling_reports" in scaling


def test_merge_scaling_reports_recomputes_cross_device_efficiency(
    tmp_path: Path,
) -> None:
    """Independent world sizes must publish one honest combined report."""
    source_dirs = []
    for count, rate in ((1, 100.0), (2, 180.0), (4, 300.0)):
        source = tmp_path / f"dev{count}"
        source.mkdir()
        payload = {
            "config": "experiment=h100_large",
            "benchmark": {"global_batch_size": 32},
            "rows": [
                {
                    "devices": count,
                    "samples_per_second": rate,
                    "speedup": 1.0,
                    "scaling_efficiency": 1.0,
                }
            ],
        }
        (source / "scaling.json").write_text(json.dumps(payload), encoding="utf-8")
        source_dirs.append(source / "scaling.json")

    paths = merge_scaling_reports(source_dirs, tmp_path / "combined")
    assert paths[0].is_file()
    rows = json.loads(paths[0].read_text())["rows"]
    assert [row["speedup"] for row in rows] == [1.0, 1.8, 3.0]
    assert [row["scaling_efficiency"] for row in rows] == [1.0, 0.9, 0.75]


def test_benchmark_data_hash_reads_declared_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """The data provenance field changes with the declared manifest bytes."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"samples": ["a"]}\n', encoding="utf-8")
    monkeypatch.setenv("ECLOUDFLOW_DATA_MANIFEST", str(manifest))
    hashes = benchmark_hashes("experiment=h100_smoke", BenchmarkConfig())
    assert hashes["data_source"] == str(manifest.resolve())
    assert hashes["data"] != hashes["config"]


def test_benchmark_data_hash_labels_missing_explicit_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """A configured missing manifest is visible instead of a silent fallback."""
    monkeypatch.delenv("ECLOUDFLOW_DATA_MANIFEST", raising=False)
    missing = tmp_path / "missing" / "manifest.json"
    hashes = benchmark_hashes(
        "experiment=h100_smoke",
        BenchmarkConfig(),
        DataConfig(manifest=str(missing)),
    )
    assert hashes["data_source"].startswith("manifest-missing:")


def test_relative_manifest_uses_repository_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    """A relative manifest may resolve from the repository when launch differs."""
    repository = tmp_path / "repository"
    module_path = repository / "src/ecloudflow/training/benchmark.py"
    module_path.parent.mkdir(parents=True)
    manifest = repository / "data/custom/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"samples": ["repository"]}\n', encoding="utf-8")
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    monkeypatch.setattr(benchmark_module, "__file__", str(module_path))

    hashes = benchmark_hashes(
        "experiment=h100_smoke",
        BenchmarkConfig(),
        DataConfig(manifest="data/custom/manifest.json"),
    )

    assert hashes["data_source"] == str(manifest.resolve())


def test_benchmark_scaling_normalizes_device_order_and_cpu_fallback(tmp_path: Path) -> None:
    """Unordered requests use the one-device row as the scaling baseline."""
    report = benchmark_scaling(
        devices={4, 1, 2},
        steps=1,
        config="experiment=h100_smoke",
        output_dir=tmp_path,
        dry_run=False,
    )
    assert [row.devices for row in report.rows] == [1, 2, 4]
    assert all(row.dry_run for row in report.rows)
    assert report.rows[0].speedup == 1.0


def test_merge_scaling_reports_rejects_fractional_device_count(tmp_path: Path) -> None:
    """Merging must not truncate a malformed fractional device count."""
    source = tmp_path / "fractional.json"
    source.write_text(
        json.dumps(
            {
                "config": "experiment=h100_smoke",
                "benchmark": {},
                "rows": [{"devices": 1.5, "samples_per_second": 10.0}],
            }
        ),
        encoding="utf-8",
    )
    try:
        merge_scaling_reports([source], tmp_path / "combined")
    except BenchmarkError:
        pass
    else:
        raise AssertionError("fractional device count should be rejected")


def test_merge_scaling_reports_rejects_dry_and_model_measurements(
    tmp_path: Path,
) -> None:
    """Estimated rows must never contribute to measured GPU speedup."""
    sources = []
    for name, dry_run, mode, backend in (
        ("estimate", True, "dry-run", "cpu-simulated"),
        ("model", False, "distributed", "cuda-ddp"),
    ):
        source = tmp_path / f"{name}.json"
        source.write_text(
            json.dumps(
                {
                    "config": "experiment=h100_large",
                    "benchmark": {"global_batch_size": 8},
                    "rows": [
                        {
                            "devices": 1 if dry_run else 2,
                            "samples_per_second": 10.0,
                            "dry_run": dry_run,
                            "mode": mode,
                            "backend": backend,
                            "workload": (
                                "schema-simulation"
                                if dry_run
                                else "ecloudflow-training-step"
                            ),
                            "input_source": (
                                "analytical-estimate"
                                if dry_run
                                else "deterministic-synthetic-complexes"
                            ),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        sources.append(source)

    try:
        merge_scaling_reports(sources, tmp_path / "combined")
    except BenchmarkError:
        pass
    else:
        raise AssertionError("dry-run and model measurements should not merge")


def test_config_fingerprint_includes_resolved_benchmark_values() -> None:
    """Equal override text cannot hide changed resolved scientific settings."""
    first = benchmark_hashes(
        "experiment=h100_smoke", BenchmarkConfig(global_batch_size=8)
    )
    second = benchmark_hashes(
        "experiment=h100_smoke", BenchmarkConfig(global_batch_size=16)
    )
    assert first["config"] != second["config"]


def test_model_workload_executes_ecloudflow_optimizer_step() -> None:
    """The measured workload must update the real joint model, not a toy MLP."""
    model = ECloudFlowModel.from_config(
        ModelConfig(scalar_dim=16, vector_dim=4, num_blocks=1, lmax=2),
        max_atoms=8,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    generator = torch.Generator().manual_seed(7)
    batch = benchmark_module._build_benchmark_batch(
        model,
        local_batch=1,
        ligand_nodes=3,
        pocket_nodes=4,
        device=torch.device("cpu"),
        generator=generator,
    )
    before = model.atom_head.network[-1].weight.detach().clone()

    benchmark_module._single_step(
        model, optimizer, batch, torch.device("cpu"), "32-true"
    )

    assert not torch.equal(before, model.atom_head.network[-1].weight)
