"""Benchmark harness for CPU dry-runs and NCCL-aware scaling measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import distributed as dist
from torch import nn, optim

from ecloudflow.config import AppConfig, BenchmarkConfig, DataConfig, load_config
from ecloudflow.core import GenerationCondition, MolecularState, PocketGraph
from ecloudflow.exceptions import BenchmarkError
from ecloudflow.models import ECloudFlowModel
from ecloudflow.sampling.profiles import get_profile

_DRY_RUN_REFERENCE_WIDTH = 256


@dataclass(frozen=True)
class BenchmarkRow:
    """Summarize one device-count benchmark row."""

    devices: int
    profile: str
    warmup_steps: int
    measurement_steps: int
    global_batch_size: int
    local_batch_size: int
    samples: int
    samples_per_second: float
    optimizer_steps_per_second: float
    peak_memory_bytes: int
    reserved_memory_bytes: int
    communication_time_seconds: float | None
    nfe: int | None
    valid_count: int | None
    valid_yield: float | None
    gpu_hours: float
    wall_time_seconds: float
    speedup: float = 1.0
    scaling_efficiency: float = 1.0
    dry_run: bool = False
    mode: str = "dry-run"
    config: str = ""
    workload: str = "unidentified"
    input_source: str = "unidentified"
    model_forward_calls: int = 0

    def __post_init__(self) -> None:
        """Reject malformed measurements before they reach JSON or CSV output."""
        integer_values = (
            self.devices,
            self.warmup_steps,
            self.measurement_steps,
            self.global_batch_size,
            self.local_batch_size,
            self.samples,
            self.peak_memory_bytes,
            self.reserved_memory_bytes,
            self.model_forward_calls,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_values
        ):
            raise ValueError("benchmark row integer fields must be integers")
        if (
            self.devices < 1
            or self.warmup_steps < 0
            or self.measurement_steps < 1
            or self.global_batch_size < 1
            or self.local_batch_size < 1
            or self.samples < 0
            or self.peak_memory_bytes <= 0
            or self.reserved_memory_bytes < self.peak_memory_bytes
        ):
            raise ValueError("benchmark row contains invalid integer measurements")
        for name, value in (("nfe", self.nfe), ("valid_count", self.valid_count)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        numeric = (
            self.samples_per_second,
            self.optimizer_steps_per_second,
            self.gpu_hours,
            self.wall_time_seconds,
            self.speedup,
            self.scaling_efficiency,
        )
        if any(not math.isfinite(float(value)) or value < 0 for value in numeric):
            raise ValueError("benchmark row contains non-finite metrics")
        if self.valid_yield is not None and (
            not math.isfinite(float(self.valid_yield))
            or not 0.0 <= self.valid_yield <= 1.0
        ):
            raise ValueError("valid_yield must be in [0, 1] or None")
        if not self.workload or not self.input_source:
            raise ValueError("workload and input_source must be non-empty")
        if self.communication_time_seconds is not None and (
            not math.isfinite(float(self.communication_time_seconds))
            or self.communication_time_seconds < 0
        ):
            raise ValueError("communication time must be finite and non-negative")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping for reports and fixtures."""
        return {
            "devices": self.devices,
            "profile": self.profile,
            "warmup_steps": self.warmup_steps,
            "measurement_steps": self.measurement_steps,
            "global_batch_size": self.global_batch_size,
            "local_batch_size": self.local_batch_size,
            "samples": self.samples,
            "samples_per_second": self.samples_per_second,
            "optimizer_steps_per_second": self.optimizer_steps_per_second,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_allocated_memory_bytes": self.peak_memory_bytes,
            "peak_reserved_memory_bytes": self.reserved_memory_bytes,
            "reserved_memory_bytes": self.reserved_memory_bytes,
            "communication_time_seconds": self.communication_time_seconds,
            "communication_seconds": self.communication_seconds,
            "nfe": self.nfe,
            "valid_count": self.valid_count,
            "valid_yield": self.valid_yield,
            "gpu_hours": self.gpu_hours,
            "wall_time_seconds": self.wall_time_seconds,
            "speedup": self.speedup,
            "scaling_efficiency": self.scaling_efficiency,
            "dry_run": self.dry_run,
            "mode": self.mode,
            "backend": self.backend,
            "config": self.config,
            "workload": self.workload,
            "input_source": self.input_source,
            "model_forward_calls": self.model_forward_calls,
        }

    @property
    def peak_reserved_memory_bytes(self) -> int:
        """Alias the reserved-memory field under the benchmark-report name."""
        return self.reserved_memory_bytes

    @property
    def peak_allocated_memory_bytes(self) -> int:
        """Alias the allocated-memory field under the benchmark-report name."""
        return self.peak_memory_bytes

    @property
    def communication_seconds(self) -> float | None:
        """Return measured communication time without fabricating zero."""
        return self.communication_time_seconds

    @property
    def backend(self) -> str:
        """Return a stable backend label for machine-readable reports."""
        if self.dry_run:
            return "cpu-simulated"
        return "cuda-ddp" if self.mode in {"local", "distributed"} else self.mode


ScalingRow = BenchmarkRow


@dataclass(frozen=True)
class ScalingReport:
    """Hold benchmark rows plus the resolved configuration and provenance."""

    rows: tuple[BenchmarkRow, ...]
    benchmark: BenchmarkConfig
    config: str
    metadata: dict[str, Any] = field(default_factory=dict)
    paths: tuple[Path, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe benchmark report."""
        return {
            "config": self.config,
            "benchmark": self.benchmark.model_dump(mode="json"),
            "metadata": dict(self.metadata),
            "rows": [row.as_dict() for row in self.rows],
            "paths": [str(path) for path in self.paths],
        }

    def write(
        self, output_dir: Path | str, *, stem: str = "scaling"
    ) -> tuple[Path, Path, Path]:
        """Write JSON, CSV, and environment metadata files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{stem}.json"
        csv_path = output_dir / f"{stem}.csv"
        env_path = output_dir / "environment.json"
        serialized = json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"
        json_path.write_text(serialized, encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(self.rows[0].as_dict().keys()) if self.rows else [],
            )
            if self.rows:
                writer.writeheader()
                for row in self.rows:
                    writer.writerow(row.as_dict())
        env_path.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Keep both historical names available while callers migrate to the
        # explicit ``scaling`` artifact name used by the server scripts.
        compatibility_stem = "benchmark" if stem == "scaling" else "scaling"
        if compatibility_stem != stem:
            compatibility_json = output_dir / f"{compatibility_stem}.json"
            compatibility_csv = output_dir / f"{compatibility_stem}.csv"
            compatibility_json.write_text(serialized, encoding="utf-8")
            compatibility_csv.write_bytes(csv_path.read_bytes())
        return json_path, csv_path, env_path


def measured_stub_nfe(profile: str) -> int:
    """Return the theoretical solver NFE for a named sampling profile.

    This helper describes a configured numerical schedule only. Benchmark
    reports never present it as an observed model or chemistry measurement.
    """
    resolved = get_profile(profile)
    return (
        resolved.num_steps * (2 if resolved.solver == "heun" else 1)
        + resolved.corrector_steps
    )


def benchmark_hashes(
    config: str,
    benchmark: BenchmarkConfig,
    data: DataConfig | None = None,
    app_config: AppConfig | None = None,
) -> dict[str, str]:
    """Return stable git/config/data fingerprints for shell scripts.

    ``data`` hashes the declared dataset manifest when one is available.  The
    resolved data configuration is accepted separately from the benchmark
    configuration so a benchmark cannot accidentally hash only its timing
    settings.  ``data_source`` records the path, an explicit missing-manifest
    marker, or the ``config-fallback`` label.

    :param config: Original Hydra override used to compose the run.
    :param benchmark: Resolved benchmark settings.
    :param data: Resolved dataset/shard settings, when available.
    :return: Git, configuration, and dataset provenance fingerprints.
    :rtype: dict[str, str]
    """
    normalized = _normalize_config_override(config)
    payload = json.dumps(
        {
            "config": normalized,
            "resolved": (
                app_config.model_dump(mode="json")
                if app_config is not None
                else {
                    "benchmark": benchmark.model_dump(mode="json"),
                    "data": data.model_dump(mode="json") if data is not None else None,
                }
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    data_hash, data_source = _dataset_fingerprint(payload, data=data)
    return {
        "git": _git_hash(),
        "config": _sha256_bytes(payload),
        "data": data_hash,
        "data_source": data_source,
    }


def _dataset_fingerprint(
    fallback_payload: bytes, *, data: DataConfig | None = None
) -> tuple[str, str]:
    """Hash the declared dataset manifest, or label a provenance fallback.

    The manifest path can be supplied through ``ECLOUDFLOW_DATA_MANIFEST`` or
    the resolved ``DataConfig.manifest`` field.  An explicitly configured but
    missing manifest is reported as ``manifest-missing:<path>`` and never
    silently replaced by a different candidate.  In the absence of an
    explicit path, the configured shard directory and conventional PDBBind
    location are checked relative to the repository and current working
    directory.  A configuration fallback keeps dry-run reports deterministic
    but is explicitly labeled as non-dataset provenance so it cannot be
    mistaken for a real data hash.
    """
    repo_root = Path(__file__).resolve().parents[3]
    configured = os.environ.get("ECLOUDFLOW_DATA_MANIFEST")
    if configured:
        return _fingerprint_explicit_manifest(Path(configured), fallback_payload)

    if data is not None and data.manifest:
        return _fingerprint_explicit_manifest(Path(data.manifest), fallback_payload)

    candidates: list[Path] = []
    if data is not None:
        candidates.extend(
            _relative_candidates(Path(data.shard_dir) / "manifest.json", repo_root)
        )
    candidates.extend(
        (
            repo_root / "data" / "processed" / "pdbbind" / "manifest.json",
            Path.cwd() / "data" / "processed" / "pdbbind" / "manifest.json",
        )
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return _sha256_bytes(candidate.read_bytes()), str(candidate.resolve())
        except OSError:
            continue
    return _sha256_bytes(fallback_payload), "config-fallback"


def _relative_candidates(path: Path, repo_root: Path) -> tuple[Path, ...]:
    """Return deterministic absolute/relative lookup candidates for a path."""
    if path.is_absolute():
        return (path,)
    # The current working directory is first because Hydra and the CLI resolve
    # relative data paths from the launch directory.  The repository root is a
    # useful fallback for callers launched from a different directory.
    current = Path.cwd() / path
    rooted = repo_root / path
    return (current, rooted) if current != rooted else (current,)


def _fingerprint_explicit_manifest(
    path: Path, fallback_payload: bytes
) -> tuple[str, str]:
    """Hash one explicitly selected manifest or report why it is unusable."""
    candidates = tuple(
        candidate.resolve()
        for candidate in _relative_candidates(
            path, Path(__file__).resolve().parents[3]
        )
    )
    resolved = next(
        (candidate for candidate in candidates if candidate.is_file()),
        candidates[0],
    )
    if not resolved.is_file():
        source = f"manifest-missing:{resolved}"
        # Include the marker and path in the fallback digest so two missing
        # manifests cannot accidentally share an apparently identical hash.
        return _sha256_bytes(fallback_payload + source.encode("utf-8")), source
    try:
        return _sha256_bytes(resolved.read_bytes()), str(resolved)
    except OSError:
        source = f"manifest-unreadable:{resolved}"
        return _sha256_bytes(fallback_payload + source.encode("utf-8")), source


def _sha256_bytes(payload: bytes) -> str:
    """Return a hexadecimal SHA-256 digest for an in-memory payload."""
    return hashlib.sha256(payload).hexdigest()


def _report_integer(value: Any, field: str) -> int:
    """Parse an integer report field without truncating fractional values."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("+").lstrip("-").isdigit():
            return int(text)
    raise TypeError(f"{field} must be an integer")


def _report_float(value: Any, field: str) -> float:
    """Parse a finite floating-point report field."""
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def merge_scaling_reports(
    report_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    stem: str = "scaling",
) -> tuple[Path, Path, Path]:
    """Merge independent one-world-size reports into one scaling artifact.

    :param report_paths: JSON reports produced by one, two, or four GPU jobs.
    :param output_dir: Destination for the combined JSON, CSV, and metadata.
    :param stem: Primary artifact basename; ``benchmark`` is written as an alias.
    :return: Primary JSON, CSV, and environment paths.
    :rtype: tuple[pathlib.Path, pathlib.Path, pathlib.Path]
    :raises BenchmarkError: If reports disagree or contain incomplete rows.

    NCCL strong-scaling jobs must run in separate process groups because one
    process group cannot change world size. This function is the explicit
    aggregation boundary and recomputes speedup/efficiency from the measured
    baseline rather than preserving each single-row report's default value.
    """
    if not report_paths:
        raise BenchmarkError("at least one scaling report is required")
    documents: list[tuple[Path, dict[str, Any]]] = []
    for value in report_paths:
        source = Path(value)
        if source.is_dir():
            source = source / f"{stem}.json"
            if not source.is_file():
                source = source.with_name("benchmark.json")
        if not source.is_file():
            raise BenchmarkError(f"scaling report does not exist: {source}")
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkError(f"invalid scaling report: {source}") from error
        if not isinstance(document, dict) or not isinstance(document.get("rows"), list):
            raise BenchmarkError(f"scaling report has no row list: {source}")
        documents.append((source, document))

    first = documents[0][1]
    config = first.get("config")
    benchmark = first.get("benchmark")
    if not isinstance(config, str) or not config:
        raise BenchmarkError("scaling report has no valid config override")
    if not isinstance(benchmark, dict):
        raise BenchmarkError("scaling report has no benchmark configuration")
    rows: list[dict[str, Any]] = []
    seen_devices: set[int] = set()
    source_hashes: list[dict[str, str] | None] = []
    shared_fields: dict[str, Any] = {}
    integer_fields = {
        "warmup_steps",
        "measurement_steps",
        "global_batch_size",
        "local_batch_size",
        "samples",
        "peak_memory_bytes",
        "peak_allocated_memory_bytes",
        "peak_reserved_memory_bytes",
        "reserved_memory_bytes",
        "model_forward_calls",
    }
    float_fields = {
        "samples_per_second",
        "optimizer_steps_per_second",
        "valid_yield",
        "gpu_hours",
        "wall_time_seconds",
        "speedup",
        "scaling_efficiency",
        "communication_time_seconds",
        "communication_seconds",
    }
    for source, document in documents:
        if document.get("config") != config or document.get("benchmark") != benchmark:
            raise BenchmarkError(f"scaling reports disagree on config: {source}")
        metadata = document.get("metadata")
        hashes: dict[str, str] | None = None
        if isinstance(metadata, dict) and "hashes" in metadata:
            raw_hashes = metadata.get("hashes")
            if not isinstance(raw_hashes, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in raw_hashes.items()
            ):
                raise BenchmarkError(f"invalid provenance hashes: {source}")
            hashes = dict(raw_hashes)
        source_hashes.append(hashes)
        source_rows = document["rows"]
        if len(source_rows) != 1 or not isinstance(source_rows[0], dict):
            raise BenchmarkError(
                f"each distributed report must contain exactly one row: {source}"
            )
        row = dict(source_rows[0])
        try:
            devices = _report_integer(row["devices"], "devices")
            rate = _report_float(row["samples_per_second"], "samples_per_second")
        except (KeyError, TypeError, ValueError) as error:
            raise BenchmarkError(
                f"scaling row is missing or malformed numeric fields: {source}"
            ) from error
        if devices < 1 or devices in seen_devices or rate <= 0:
            raise BenchmarkError(f"invalid or duplicate device row in {source}")
        row["devices"] = devices
        row["samples_per_second"] = rate
        for key in integer_fields.intersection(row):
            try:
                row[key] = _report_integer(row[key], key)
            except (TypeError, ValueError) as error:
                raise BenchmarkError(f"invalid {key} in scaling row: {source}") from error
            minimum = 1 if key in {
                "warmup_steps",
                "measurement_steps",
                "global_batch_size",
                "local_batch_size",
            } else 0
            if row[key] < minimum:
                raise BenchmarkError(f"invalid {key} in scaling row: {source}")
        for key in ("nfe", "valid_count"):
            if key not in row or row[key] is None:
                continue
            try:
                row[key] = _report_integer(row[key], key)
            except (TypeError, ValueError) as error:
                raise BenchmarkError(f"invalid {key} in scaling row: {source}") from error
            if row[key] < 0:
                raise BenchmarkError(f"invalid {key} in scaling row: {source}")
        for key in float_fields.intersection(row):
            if key in {
                "communication_time_seconds",
                "communication_seconds",
                "valid_yield",
            } and row[key] is None:
                continue
            try:
                row[key] = _report_float(row[key], key)
            except (TypeError, ValueError) as error:
                raise BenchmarkError(f"invalid {key} in scaling row: {source}") from error
            if row[key] < 0:
                raise BenchmarkError(f"invalid {key} in scaling row: {source}")
        if (
            "valid_yield" in row
            and row["valid_yield"] is not None
            and not 0.0 <= row["valid_yield"] <= 1.0
        ):
            raise BenchmarkError(f"invalid valid_yield in scaling row: {source}")
        if {
            "devices",
            "global_batch_size",
            "local_batch_size",
        }.issubset(row) and row["global_batch_size"] != row["local_batch_size"] * devices:
            raise BenchmarkError(f"global/local batch mismatch in scaling row: {source}")
        if {
            "valid_count",
            "samples",
        }.issubset(row) and row["valid_count"] is not None and row["valid_count"] > row["samples"]:
            raise BenchmarkError(f"valid count exceeds samples in scaling row: {source}")
        if {
            "peak_memory_bytes",
            "reserved_memory_bytes",
        }.issubset(row) and row["reserved_memory_bytes"] < row["peak_memory_bytes"]:
            raise BenchmarkError(f"reserved memory is below allocated memory: {source}")
        if {
            "global_batch_size",
            "measurement_steps",
            "samples",
        }.issubset(row) and row["samples"] != row["global_batch_size"] * row["measurement_steps"]:
            raise BenchmarkError(f"sample count mismatch in scaling row: {source}")
        compatibility = {
            "dry_run": row.get("dry_run"),
            "backend": row.get("backend"),
            "workload": row.get("workload"),
            "input_source": row.get("input_source"),
            "mode_family": _benchmark_mode_family(row.get("mode")),
        }
        for key, value in compatibility.items():
            shared_key = f"compatibility:{key}"
            if shared_key in shared_fields and value != shared_fields[shared_key]:
                raise BenchmarkError(
                    f"scaling reports disagree on {key}: {source}"
                )
            shared_fields[shared_key] = value
        for key in (
            "profile",
            "warmup_steps",
            "measurement_steps",
            "global_batch_size",
            "nfe",
        ):
            if key in row:
                if key in shared_fields and row[key] != shared_fields[key]:
                    raise BenchmarkError(
                        f"scaling reports disagree on {key}: {source}"
                    )
                shared_fields[key] = row[key]
        seen_devices.add(devices)
        rows.append(row)

    hashes_present = [item is not None for item in source_hashes]
    if any(hashes_present) and not all(hashes_present):
        raise BenchmarkError("scaling reports have incomplete provenance hashes")
    if all(hashes_present) and any(
        item != source_hashes[0] for item in source_hashes[1:]
    ):
        raise BenchmarkError("scaling reports disagree on provenance hashes")
    combined_hashes = source_hashes[0] if all(hashes_present) else {}

    rows.sort(key=lambda row: int(row["devices"]))
    baseline = float(rows[0]["samples_per_second"])
    for row in rows:
        speedup = float(row["samples_per_second"]) / baseline
        row["speedup"] = speedup
        row["scaling_efficiency"] = speedup / int(row["devices"])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{stem}.json"
    csv_path = destination / f"{stem}.csv"
    environment_path = destination / "environment.json"
    paths = (json_path, csv_path, environment_path)
    payload = {
        "config": config,
        "benchmark": benchmark,
        "metadata": {
            "combined": True,
            "device_counts": [int(row["devices"]) for row in rows],
            "source_reports": [str(source) for source, _ in documents],
            "hashes": combined_hashes,
            "data_source": combined_hashes.get(
                "data_source", "provenance-unavailable"
            ),
        },
        "paths": [str(path) for path in paths],
        "rows": rows,
    }
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    environment_path.write_text(
        json.dumps(payload["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compatibility_json = destination / "benchmark.json"
    compatibility_csv = destination / "benchmark.csv"
    if stem != "benchmark":
        compatibility_json.write_text(json_text, encoding="utf-8")
        compatibility_csv.write_bytes(csv_path.read_bytes())
    return paths


def _benchmark_mode_family(value: Any) -> str | None:
    """Normalize row modes that legitimately differ with DDP world size."""
    if value in {"local", "distributed"}:
        return "model-measurement"
    if value in {"dry-run", "simulation", "schema-only"}:
        return "schema-simulation"
    return str(value) if value is not None else None


def benchmark_scaling(
    devices: list[int] | tuple[int, ...] | set[int],
    steps: int,
    config: str,
    output_dir: Path | str,
    *,
    dry_run: bool | None = None,
    write: bool = True,
    stem: str = "scaling",
    metadata: dict[str, Any] | None = None,
    warmup_steps: int | None = None,
    global_batch_size: int | None = None,
    profile: str | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> ScalingReport:
    """Benchmark identical global work across requested device counts.

    :param devices: Positive device counts, normally ``[1, 2, 4]`` on H100.
    :param steps: Timed iterations used for throughput estimates.
    :param config: Hydra override such as ``experiment=h100_smoke``.
    :param output_dir: Destination for JSON, CSV, and environment artifacts.
    :param dry_run: Force deterministic synthetic timing instead of real runs.
    :return: Strong-scaling rows with speedup, efficiency, and NFE.
    :rtype: ScalingReport
    :param write: Write report artifacts to ``output_dir`` when true.
    :param stem: Basename used for JSON and CSV output files.
    :param metadata: Optional caller-supplied metadata merged into the report.
    :param warmup_steps: Optional compatibility override for the resolved benchmark.
    :param global_batch_size: Optional compatibility override for fixed global work.
    :param profile: Optional compatibility override for the sampling profile.
    :param seed: Optional compatibility override for deterministic workload seeding.
    :raises BenchmarkError: If inputs are invalid or a real multi-device run is
        requested without distributed initialization.
    """
    if kwargs:
        raise BenchmarkError(
            f"unknown benchmark_scaling options: {', '.join(sorted(kwargs))}"
        )
    _initialize_distributed_from_environment()
    resolved = _resolve_app_config(config)
    device_counts = _normalize_devices(devices)
    benchmark_values = resolved.benchmark.model_dump()
    compatibility_overrides = {
        "warmup_steps": warmup_steps,
        "global_batch_size": global_batch_size,
        "profile": profile,
        "seed": seed,
    }
    benchmark_values.update(
        {
            name: value
            for name, value in compatibility_overrides.items()
            if value is not None
        }
    )
    try:
        benchmark = BenchmarkConfig.model_validate(benchmark_values)
    except Exception as error:
        raise BenchmarkError(f"invalid benchmark override: {error}") from error
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise BenchmarkError("steps must be a positive integer.")
    if _distributed_ready() and device_counts != (dist.get_world_size(),):
        raise BenchmarkError(
            "torchrun benchmark must request exactly one device count matching "
            f"WORLD_SIZE={dist.get_world_size()}"
        )
    # A multi-device request can only be measured honestly inside the matching
    # torchrun process group.  Likewise, a CPU-only host cannot produce a real
    # GPU benchmark.  Force those fallbacks into the explicitly labeled
    # synthetic path even when a caller omitted ``--dry-run`` (or passed the
    # default false value from a CLI wrapper).
    simulation_required = (not torch.cuda.is_available()) or (
        any(count > 1 for count in device_counts) and not _distributed_ready()
    )
    dry_run = bool(dry_run) or simulation_required
    rows = [
        _benchmark_one_count(
            device_count=count,
            steps=steps,
            config=config,
            app_config=resolved,
            benchmark=benchmark,
            dry_run=dry_run,
        )
        for count in device_counts
    ]
    rows = _attach_scaling(rows)
    hashes = benchmark_hashes(
        config, benchmark, data=resolved.data, app_config=resolved
    )
    report_metadata = {
        "git": _git_hash(),
        "hashes": hashes,
        "data_source": hashes["data_source"],
        "data_manifest": resolved.data.manifest,
        "data_shard_dir": resolved.data.shard_dir,
        "dry_run": dry_run,
        "device_counts": list(device_counts),
        "world_size": _world_size(),
        "rank": _rank(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "benchmark": benchmark.model_dump(mode="json"),
        "workload": (
            "schema-simulation" if dry_run else "ecloudflow-training-step"
        ),
        "input_source": (
            "analytical-estimate"
            if dry_run
            else "deterministic-synthetic-complexes"
        ),
        "chemistry_metrics_measured": False,
    }
    if metadata:
        report_metadata.update(metadata)
    rank_zero_writer = _rank() == 0 or not _distributed_ready()
    output_path = Path(output_dir)
    report_paths = (
        (
            output_path / f"{stem}.json",
            output_path / f"{stem}.csv",
            output_path / "environment.json",
        )
        if write and rank_zero_writer
        else ()
    )
    report = ScalingReport(
        rows=tuple(rows),
        benchmark=benchmark,
        config=config,
        metadata=report_metadata,
        paths=report_paths,
    )
    if write and rank_zero_writer:
        report.write(output_path, stem=stem)
    if _distributed_ready():
        dist.barrier()
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark harness from ``python -m`` or shell scripts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment=h100_smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="*", default=None)
    parser.add_argument("--mode", choices=("smoke", "scaling"), default="scaling")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    resolved = _resolve_app_config(args.config)
    devices = args.devices if args.devices else list(resolved.benchmark.devices)
    steps = (
        args.steps if args.steps is not None else resolved.benchmark.measurement_steps
    )
    report = benchmark_scaling(
        devices=devices,
        steps=steps,
        config=args.config,
        output_dir=args.output,
        dry_run=True if args.dry_run else None,
    )
    if _rank() == 0:
        (args.output / "mode.txt").write_text(f"{args.mode}\n", encoding="utf-8")
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


def _benchmark_one_count(
    *,
    device_count: int,
    steps: int,
    config: str,
    app_config: AppConfig,
    benchmark: BenchmarkConfig,
    dry_run: bool,
) -> BenchmarkRow:
    if device_count < 1:
        raise BenchmarkError("device counts must be positive integers.")
    local_batch = _local_batch_size(benchmark.global_batch_size, device_count)
    if dry_run or (device_count > 1 and not _distributed_ready()):
        return _synthetic_row(
            device_count=device_count,
            steps=steps,
            config=config,
            benchmark=benchmark,
            local_batch=local_batch,
        )
    if device_count > 1:
        return _distributed_row(
            device_count=device_count,
            steps=steps,
            config=config,
            app_config=app_config,
            benchmark=benchmark,
            local_batch=local_batch,
        )
    return _local_row(
        device_count=device_count,
        steps=steps,
        config=config,
        app_config=app_config,
        benchmark=benchmark,
        local_batch=local_batch,
    )


def _synthetic_row(
    *,
    device_count: int,
    steps: int,
    config: str,
    benchmark: BenchmarkConfig,
    local_batch: int,
) -> BenchmarkRow:
    theoretical_nfe = measured_stub_nfe(benchmark.profile)
    base_step = 0.02 + 0.00015 * benchmark.global_batch_size + 0.00002 * theoretical_nfe
    scale = 1.0 + 0.8 * (device_count - 1)
    wall_time = steps * base_step / scale
    samples = benchmark.global_batch_size * steps
    samples_per_second = samples / wall_time
    optimizer_steps_per_second = steps / wall_time
    peak_memory = _estimated_memory_bytes(
        local_batch, device_count, benchmark, theoretical_nfe
    )
    reserved_memory = int(peak_memory * 1.1)
    communication = None
    gpu_hours = wall_time * device_count / 3600.0
    return BenchmarkRow(
        devices=device_count,
        profile=benchmark.profile,
        warmup_steps=benchmark.warmup_steps,
        measurement_steps=steps,
        global_batch_size=benchmark.global_batch_size,
        local_batch_size=local_batch,
        samples=samples,
        samples_per_second=samples_per_second,
        optimizer_steps_per_second=optimizer_steps_per_second,
        peak_memory_bytes=peak_memory,
        reserved_memory_bytes=reserved_memory,
        communication_time_seconds=communication,
        nfe=None,
        valid_count=None,
        valid_yield=None,
        gpu_hours=gpu_hours,
        wall_time_seconds=wall_time,
        dry_run=True,
        mode="dry-run",
        config=config,
        workload="schema-simulation",
        input_source="analytical-estimate",
        model_forward_calls=0,
    )


def _local_row(
    *,
    device_count: int,
    steps: int,
    config: str,
    app_config: AppConfig,
    benchmark: BenchmarkConfig,
    local_batch: int,
) -> BenchmarkRow:
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = benchmark.seed + device_count
    model = ECloudFlowModel.from_config(app_config.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    measurement = _run_measurement(
        model=model,
        optimizer=optimizer,
        device=device,
        local_batch=local_batch,
        warmup_steps=benchmark.warmup_steps,
        measurement_steps=steps,
        seed=seed,
        ligand_nodes=benchmark.ligand_nodes,
        pocket_nodes=benchmark.pocket_nodes,
        precision=app_config.trainer.precision,
    )
    return _row_from_measurement(
        device_count=device_count,
        benchmark=benchmark,
        config=config,
        local_batch=local_batch,
        steps=steps,
        measurement=measurement,
        mode="local",
        dry_run=False,
    )


def _distributed_row(
    *,
    device_count: int,
    steps: int,
    config: str,
    app_config: AppConfig,
    benchmark: BenchmarkConfig,
    local_batch: int,
) -> BenchmarkRow:
    if not _distributed_ready():
        raise BenchmarkError("multi-device benchmark requires torchrun initialization.")
    world_size = _world_size()
    if world_size != device_count:
        raise BenchmarkError(
            f"torchrun world size {world_size} does not match requested devices {device_count}."
        )
    local_rank = _rank()
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    seed = benchmark.seed + local_rank
    model = ECloudFlowModel.from_config(app_config.model).to(device)
    if device.type == "cuda":
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = nn.parallel.DistributedDataParallel(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    measurement = _run_measurement(
        model=model,
        optimizer=optimizer,
        device=device,
        local_batch=local_batch,
        warmup_steps=benchmark.warmup_steps,
        measurement_steps=steps,
        seed=seed,
        ligand_nodes=benchmark.ligand_nodes,
        pocket_nodes=benchmark.pocket_nodes,
        precision=app_config.trainer.precision,
    )
    # Strong scaling is reported from the slowest rank, not whichever rank
    # happened to publish the artifact first.
    timing = torch.tensor(
        [measurement.wall_time_seconds, measurement.compute_time_seconds],
        dtype=torch.float64,
        device=device,
    )
    memory = torch.tensor(
        [measurement.peak_memory_bytes, measurement.reserved_memory_bytes],
        dtype=torch.long,
        device=device,
    )
    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    measurement = _Measurement(
        wall_time_seconds=float(timing[0].item()),
        compute_time_seconds=float(timing[1].item()),
        peak_memory_bytes=int(memory[0].item()),
        reserved_memory_bytes=int(memory[1].item()),
    )
    return _row_from_measurement(
        device_count=device_count,
        benchmark=benchmark,
        config=config,
        local_batch=local_batch,
        steps=steps,
        measurement=measurement,
        mode="distributed",
        dry_run=False,
    )


@dataclass(frozen=True)
class _Measurement:
    wall_time_seconds: float
    compute_time_seconds: float
    peak_memory_bytes: int
    reserved_memory_bytes: int


@dataclass(frozen=True)
class _BenchmarkBatch:
    """Hold one deterministic model-shaped batch outside the timed region."""

    state: MolecularState
    time: torch.Tensor
    condition: GenerationCondition


def _run_measurement(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    local_batch: int,
    warmup_steps: int,
    measurement_steps: int,
    seed: int,
    ligand_nodes: int,
    pocket_nodes: int,
    precision: str,
) -> _Measurement:
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    batch = _build_benchmark_batch(
        model,
        local_batch=local_batch,
        ligand_nodes=ligand_nodes,
        pocket_nodes=pocket_nodes,
        device=device,
        generator=rng,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    model.train()
    for _ in range(warmup_steps):
        _single_step(model, optimizer, batch, device, precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    compute_total = 0.0
    for _ in range(measurement_steps):
        step_compute_start = time.perf_counter()
        _single_step(model, optimizer, batch, device, precision)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_end = time.perf_counter()
        compute_total += step_end - step_compute_start
    wall_time = time.perf_counter() - start
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        reserved_memory = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_memory = _estimated_memory_bytes(local_batch, 1, None, 0)
        reserved_memory = int(peak_memory * 1.1)
    return _Measurement(
        wall_time_seconds=wall_time,
        compute_time_seconds=compute_total,
        peak_memory_bytes=peak_memory,
        reserved_memory_bytes=reserved_memory,
    )


def _single_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: _BenchmarkBatch,
    device: torch.device,
    precision: str,
) -> None:
    """Run one real ECloudFlow forward/backward/optimizer benchmark step."""
    autocast_dtype = {
        "bf16-mixed": torch.bfloat16,
        "16-mixed": torch.float16,
    }.get(precision)
    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=autocast_dtype is not None,
    ):
        prediction = model(batch.state, batch.time, batch.condition)
        tensors = (
            prediction.position_velocity,
            prediction.position_score,
            prediction.electron_velocity,
            prediction.electron_score,
            prediction.atom_logits,
            prediction.charge_logits,
            prediction.bond_logits,
            prediction.count_logits,
            prediction.affinity,
            prediction.affinity_log_variance,
            prediction.interaction_logits,
        )
        terms = [value.float().square().mean() for value in tensors if value.numel()]
        loss = torch.stack(terms).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _build_benchmark_batch(
    model: nn.Module,
    *,
    local_batch: int,
    ligand_nodes: int,
    pocket_nodes: int,
    device: torch.device,
    generator: torch.Generator,
) -> _BenchmarkBatch:
    """Create deterministic contract-valid tensors for the actual model.

    :param model: Bare or DDP-wrapped :class:`ECloudFlowModel` whose declared
        categorical and electron widths define the generated tensors.
    :param local_batch: Number of independent complexes on this rank.
    :param ligand_nodes: Ligand nodes allocated per complex.
    :param pocket_nodes: Pocket atoms allocated per complex.
    :param device: Rank-local CUDA device.
    :param generator: Rank-seeded device generator used only before timing.
    :return: Immutable state, time, and pocket condition accepted by the model.
    :rtype: _BenchmarkBatch

    Inputs are synthetic and therefore cannot support chemistry-validity or
    binding claims. They do exercise the configured ECloudFlow message blocks,
    electron heads, categorical heads, gradients, optimizer, and DDP collectives.
    Their construction is deliberately outside the measured interval.
    """
    bare = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    if not isinstance(bare, ECloudFlowModel):
        raise TypeError("benchmark workload requires ECloudFlowModel")
    total_nodes = local_batch * ligand_nodes
    positions = torch.randn(
        (total_nodes, 3), device=device, generator=generator
    ) * 2.5
    node_batch = torch.arange(local_batch, device=device).repeat_interleave(
        ligand_nodes
    )
    base_edges = torch.triu_indices(
        ligand_nodes, ligand_nodes, offset=1, device=device
    )
    edges_per_complex = base_edges.shape[1]
    offsets = (
        torch.arange(local_batch, device=device) * ligand_nodes
    ).repeat_interleave(edges_per_complex)
    halfedge_index = base_edges.repeat(1, local_batch) + offsets[None, :]
    halfedge_batch = torch.arange(local_batch, device=device).repeat_interleave(
        edges_per_complex
    )
    state = MolecularState(
        positions=positions,
        atom_logits=torch.randn(
            (total_nodes, bare.atom_classes), device=device, generator=generator
        ),
        charge_logits=torch.randn(
            (total_nodes, bare.charge_classes), device=device, generator=generator
        ),
        halfedge_index=halfedge_index,
        bond_logits=torch.randn(
            (halfedge_index.shape[1], bare.bond_classes),
            device=device,
            generator=generator,
        ),
        electron_latent=torch.randn(
            (total_nodes, bare.electron_latent_dim),
            device=device,
            generator=generator,
        ),
        node_batch=node_batch,
        halfedge_batch=halfedge_batch,
    )
    total_pocket_nodes = local_batch * pocket_nodes
    pocket = PocketGraph(
        positions=torch.randn(
            (total_pocket_nodes, 3), device=device, generator=generator
        ) * 4.0,
        features=torch.randn(
            (total_pocket_nodes, 16), device=device, generator=generator
        ),
        batch=torch.arange(local_batch, device=device).repeat_interleave(
            pocket_nodes
        ),
    )
    return _BenchmarkBatch(
        state=state,
        time=torch.full((local_batch,), 0.5, device=device),
        condition=GenerationCondition(pocket=pocket),
    )


def _row_from_measurement(
    *,
    device_count: int,
    benchmark: BenchmarkConfig,
    config: str,
    local_batch: int,
    steps: int,
    measurement: _Measurement,
    mode: str,
    dry_run: bool,
) -> BenchmarkRow:
    samples = benchmark.global_batch_size * steps
    samples_per_second = samples / measurement.wall_time_seconds
    optimizer_steps_per_second = steps / measurement.wall_time_seconds
    # Backward includes NCCL collectives, so wall-minus-step timing cannot
    # isolate communication without a profiler trace. Leave it unmeasured.
    communication = None
    gpu_hours = measurement.wall_time_seconds * device_count / 3600.0
    return BenchmarkRow(
        devices=device_count,
        profile=benchmark.profile,
        warmup_steps=benchmark.warmup_steps,
        measurement_steps=steps,
        global_batch_size=benchmark.global_batch_size,
        local_batch_size=local_batch,
        samples=samples,
        samples_per_second=samples_per_second,
        optimizer_steps_per_second=optimizer_steps_per_second,
        peak_memory_bytes=measurement.peak_memory_bytes,
        reserved_memory_bytes=measurement.reserved_memory_bytes,
        communication_time_seconds=communication,
        nfe=None,
        valid_count=None,
        valid_yield=None,
        gpu_hours=gpu_hours,
        wall_time_seconds=measurement.wall_time_seconds,
        dry_run=dry_run,
        mode=mode,
        config=config,
        workload="ecloudflow-training-step",
        input_source="deterministic-synthetic-complexes",
        model_forward_calls=steps * device_count,
    )


def _attach_scaling(rows: list[BenchmarkRow]) -> list[BenchmarkRow]:
    if not rows:
        return rows
    # Reports are consumed as scaling tables, so make the baseline independent
    # of whether a caller supplied a list, tuple, or set of device counts.
    rows = sorted(rows, key=lambda row: row.devices)
    baseline = rows[0].samples_per_second
    adjusted = [replace(rows[0], speedup=1.0, scaling_efficiency=1.0)]
    for row in rows[1:]:
        speedup = row.samples_per_second / baseline if baseline > 0 else 0.0
        efficiency = speedup / row.devices if row.devices > 0 else 0.0
        adjusted.append(replace(row, speedup=speedup, scaling_efficiency=efficiency))
    return adjusted


def _estimated_memory_bytes(
    local_batch: int,
    device_count: int,
    benchmark: BenchmarkConfig | None,
    nfe: int,
) -> int:
    profile_width = 4 * _DRY_RUN_REFERENCE_WIDTH * (
        2 * _DRY_RUN_REFERENCE_WIDTH + 1
    )
    batch_term = local_batch * _DRY_RUN_REFERENCE_WIDTH * 8
    profile_term = (
        benchmark.global_batch_size if benchmark is not None else local_batch
    ) * max(1, device_count)
    return int(profile_width + batch_term + profile_term * 1024 + nfe * 4096)


def _local_batch_size(global_batch_size: int, device_count: int) -> int:
    if global_batch_size < 1:
        raise BenchmarkError("global_batch_size must be positive.")
    if global_batch_size % device_count != 0:
        raise BenchmarkError(
            "global_batch_size must be divisible by the requested device count."
        )
    return global_batch_size // device_count


def _normalize_devices(
    devices: list[int] | tuple[int, ...] | set[int],
) -> tuple[int, ...]:
    if not devices:
        raise BenchmarkError("at least one device count is required.")
    normalized = tuple(sorted(devices))
    if any(
        isinstance(device, bool) or not isinstance(device, int) or device < 1
        for device in normalized
    ):
        raise BenchmarkError("device counts must be positive integers.")
    if len(set(normalized)) != len(normalized):
        raise BenchmarkError("device counts must be unique.")
    return normalized


def _resolve_app_config(config: str) -> AppConfig:
    normalized = _normalize_config_override(config)
    return load_config([normalized])


def _normalize_config_override(config: str) -> str:
    config = config.strip()
    if not config:
        raise BenchmarkError("config override must be a non-empty string.")
    if config.startswith(("+", "~")):
        return config
    return f"+{config}"


def _distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _initialize_distributed_from_environment() -> None:
    """Initialize the torchrun process group when a multi-rank environment exists.

    The CLI intentionally stays a thin adapter, so this boundary owns the
    ``WORLD_SIZE``/``LOCAL_RANK`` contract required by NCCL benchmark jobs.
    Single-process local and CPU runs do not create a process group.
    """
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise BenchmarkError("WORLD_SIZE must be an integer") from error
    if world_size <= 1:
        return
    if not dist.is_available():
        raise BenchmarkError("torch.distributed is unavailable for torchrun")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                torch.cuda.set_device(local_rank)
            except (TypeError, ValueError, RuntimeError) as error:
                raise BenchmarkError("invalid LOCAL_RANK for NCCL benchmark") from error
        try:
            dist.init_process_group(backend=backend, init_method="env://")
        except (OSError, RuntimeError, ValueError) as error:
            raise BenchmarkError(
                "failed to initialize torchrun process group"
            ) from error
    if dist.get_world_size() != world_size:
        raise BenchmarkError(
            f"WORLD_SIZE={world_size} disagrees with process group "
            f"world size {dist.get_world_size()}"
        )


def _world_size() -> int:
    if _distributed_ready():
        return dist.get_world_size()
    return 1


def _rank() -> int:
    if _distributed_ready():
        return dist.get_rank()
    return 0


def _git_hash() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())


__all__ = [
    "AppConfig",
    "BenchmarkConfig",
    "BenchmarkError",
    "BenchmarkRow",
    "ScalingReport",
    "ScalingRow",
    "benchmark_hashes",
    "benchmark_scaling",
    "main",
    "measured_stub_nfe",
    "merge_scaling_reports",
]
