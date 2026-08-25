"""Environment diagnostics used by the ECloudFlow command line."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import typer


def collect_diagnostics(
    output_dir: Path | None = None,
    *,
    dataset: Path | None = None,
    checkpoint: Path | None = None,
    require_gpu: bool = False,
    min_gpus: int = 1,
    server: bool = False,
) -> dict[str, Any]:
    """Collect dependency, accelerator, executable, and filesystem checks.

    :param output_dir: Optional directory whose parent is checked for write
        access; no artifact is left behind by the diagnostic.
    :param dataset: Optional dataset root or manifest validated structurally.
    :param checkpoint: Optional checkpoint path checked for readable structure.
    :param require_gpu: Whether the configured CUDA minimum is mandatory.
    :param min_gpus: Positive minimum CUDA device count when GPUs are required.
    :param server: Require at least four GPUs plus BF16 and NCCL availability.
    :return: JSON-safe check mapping with ``ok`` and ``detail`` values.
    :rtype: dict[str, Any]
    :raises ValueError: If ``min_gpus`` is not a positive integer.

    Optional scientific executables are always reported but do not fail the
    ordinary diagnostic. Server mode promotes the accelerator, BF16, and NCCL
    records to required checks without initializing a process group or changing
    the active CUDA device.
    """
    if isinstance(min_gpus, bool) or not isinstance(min_gpus, int) or min_gpus < 1:
        raise ValueError("min_gpus must be a positive integer")
    required_gpu_count = max(min_gpus, 4 if server else min_gpus)
    checks: dict[str, Any] = {
        "python": {"ok": sys.version_info >= (3, 10), "detail": sys.version.split()[0]},
        "torch": _module_check("torch"),
        "rdkit": _module_check("rdkit"),
        "hydra": _module_check("hydra"),
        "lightning": _module_check("lightning"),
        "e3nn": _module_check("e3nn"),
        "biopython": _module_check("Bio"),
        "cuda": _cuda_check(required_gpu_count),
        "bf16": _bf16_check(),
        "nccl": _nccl_check(),
        "xtb": _executable_check("xtb"),
        "vina": _executable_check("vina"),
        "obabel": _executable_check("obabel"),
    }
    if output_dir is not None:
        checks["output_dir"] = _output_check(output_dir)
    if dataset is not None:
        checks["dataset"] = _dataset_check(dataset)
    if checkpoint is not None:
        checks["checkpoint"] = _checkpoint_check(checkpoint)
    required = {"python", "torch", "rdkit", "hydra", "lightning", "e3nn", "biopython"}
    if output_dir is not None:
        required.add("output_dir")
    if dataset is not None:
        required.add("dataset")
    if checkpoint is not None:
        required.add("checkpoint")
    if require_gpu or server:
        required.add("cuda")
    if server:
        required.update({"bf16", "nccl"})
    checks["ok"] = all(checks[key]["ok"] for key in required)
    return checks


def doctor_command(
    output_dir: Path | None = None,
    as_json: bool = False,
    *,
    dataset: Path | None = None,
    checkpoint: Path | None = None,
    require_gpu: bool = False,
    min_gpus: int = 1,
    server: bool = False,
) -> None:
    """Print environment checks and return a nonzero status only for essentials."""
    diagnostics = collect_diagnostics(
        output_dir,
        dataset=dataset,
        checkpoint=checkpoint,
        require_gpu=require_gpu,
        min_gpus=min_gpus,
        server=server,
    )
    if as_json:
        typer.echo(json.dumps(diagnostics, indent=2, sort_keys=True))
    else:
        for name, result in diagnostics.items():
            if name == "ok":
                continue
            marker = "ok" if result["ok"] else "missing"
            typer.echo(f"{name:12} {marker:8} {result['detail']}")
        typer.echo(
            "doctor: " + ("ready" if diagnostics["ok"] else "required checks failed")
        )
    if not diagnostics["ok"]:
        raise typer.Exit(code=1)


def _module_check(name: str) -> dict[str, Any]:
    """Return an import availability record without importing heavy modules."""
    available = importlib.util.find_spec(name) is not None
    return {"ok": available, "detail": "installed" if available else "not installed"}


def _cuda_check(min_gpus: int = 1) -> dict[str, Any]:
    """Inspect CUDA device count, identities, capabilities, and runtime versions."""
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = torch.cuda.device_count()
        devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": ".".join(
                    str(value) for value in torch.cuda.get_device_capability(index)
                ),
            }
            for index in range(count)
        ] if available else []
        return {
            "ok": available and count >= min_gpus,
            "detail": (
                f"available={available} devices={count} required={min_gpus} "
                f"torch={torch.__version__} cuda={torch.version.cuda} "
                f"hardware={json.dumps(devices, separators=(',', ':'))}"
            ),
        }
    except (
        ImportError,
        RuntimeError,
        OSError,
    ) as error:  # pragma: no cover - environment-specific
        return {"ok": False, "detail": f"unavailable: {type(error).__name__}"}


def _bf16_check() -> dict[str, Any]:
    """Report whether every visible CUDA device supports BF16 arithmetic."""
    try:
        import torch

        count = torch.cuda.device_count()
        supported = bool(torch.cuda.is_available()) and count > 0 and all(
            torch.cuda.get_device_capability(index)[0] >= 8
            for index in range(count)
        )
        return {"ok": supported, "detail": f"supported={supported} devices={count}"}
    except (ImportError, RuntimeError, OSError, TypeError) as error:
        return {"ok": False, "detail": f"unavailable: {type(error).__name__}"}


def _nccl_check() -> dict[str, Any]:
    """Report whether the installed PyTorch build exposes the NCCL backend."""
    try:
        import torch

        available = bool(
            torch.distributed.is_available()
            and torch.distributed.is_nccl_available()
        )
        version = (
            torch.cuda.nccl.version()
            if available and hasattr(torch.cuda, "nccl")
            else None
        )
        return {"ok": available, "detail": f"available={available} version={version}"}
    except (ImportError, RuntimeError, OSError) as error:
        return {"ok": False, "detail": f"unavailable: {type(error).__name__}"}


def _executable_check(name: str) -> dict[str, Any]:
    """Return an optional executable discovery record."""
    path = shutil.which(name)
    return {"ok": path is not None, "detail": path or "optional executable not found"}


def _output_check(path: Path) -> dict[str, Any]:
    """Check that an output directory can be created and written."""
    try:
        if path.exists() and not path.is_dir():
            return {"ok": False, "detail": f"not a directory: {path}"}
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".ecloudflow-doctor"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return {"ok": True, "detail": str(path)}
    except OSError as error:
        return {"ok": False, "detail": f"{type(error).__name__}: {error}"}


def _dataset_check(path: Path) -> dict[str, Any]:
    """Validate a dataset root or explicit manifest without opening shards."""
    candidate = path / "manifest.json" if path.is_dir() else path
    if not candidate.is_file():
        return {"ok": False, "detail": f"manifest not found: {candidate}"}
    try:
        from ecloudflow.data.manifest import DatasetManifest

        manifest = DatasetManifest.read(candidate)
        return {
            "ok": True,
            "detail": f"{candidate} samples={len(manifest.sample_ids)} hash={manifest.hash}",
        }
    except Exception as error:  # noqa: BLE001 - doctor reports validation state
        return {
            "ok": False,
            "detail": f"invalid manifest: {type(error).__name__}: {error}",
        }


def _checkpoint_check(path: Path) -> dict[str, Any]:
    """Check checkpoint existence and recognizable serialized structure."""
    if not path.is_file():
        return {"ok": False, "detail": f"checkpoint not found: {path}"}
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        recognized = callable(payload) or isinstance(payload, dict)
        return {
            "ok": recognized,
            "detail": "recognized" if recognized else "unsupported payload",
        }
    except Exception as error:  # noqa: BLE001 - malformed checkpoints are diagnostics
        return {"ok": False, "detail": f"unreadable: {type(error).__name__}: {error}"}
