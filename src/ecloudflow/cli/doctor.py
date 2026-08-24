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
) -> dict[str, Any]:
    """Collect dependency, accelerator, executable, and filesystem checks.

    :param output_dir: Optional directory whose parent is checked for write
        access; no artifact is left behind by the diagnostic.
    :return: JSON-safe check mapping with ``ok`` and ``detail`` values.
    :rtype: dict[str, Any]
    """
    checks: dict[str, Any] = {
        "python": {"ok": sys.version_info >= (3, 10), "detail": sys.version.split()[0]},
        "torch": _module_check("torch"),
        "rdkit": _module_check("rdkit"),
        "hydra": _module_check("hydra"),
        "lightning": _module_check("lightning"),
        "e3nn": _module_check("e3nn"),
        "biopython": _module_check("Bio"),
        "cuda": _cuda_check(),
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
    if require_gpu:
        required.add("cuda")
    checks["ok"] = all(checks[key]["ok"] for key in required)
    return checks


def doctor_command(
    output_dir: Path | None = None,
    as_json: bool = False,
    *,
    dataset: Path | None = None,
    checkpoint: Path | None = None,
    require_gpu: bool = False,
) -> None:
    """Print environment checks and return a nonzero status only for essentials."""
    diagnostics = collect_diagnostics(
        output_dir,
        dataset=dataset,
        checkpoint=checkpoint,
        require_gpu=require_gpu,
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


def _cuda_check() -> dict[str, Any]:
    """Inspect CUDA only when torch is importable."""
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "ok": available,
            "detail": f"available={available} devices={torch.cuda.device_count()}",
        }
    except (
        ImportError,
        RuntimeError,
        OSError,
    ) as error:  # pragma: no cover - environment-specific
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
