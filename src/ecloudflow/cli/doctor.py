"""Environment diagnostics used by the ECloudFlow command line."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import typer


def collect_diagnostics(output_dir: Path | None = None) -> dict[str, Any]:
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
        "cuda": _cuda_check(),
        "xtb": _executable_check("xtb"),
        "vina": _executable_check("vina"),
        "obabel": _executable_check("obabel"),
    }
    if output_dir is not None:
        checks["output_dir"] = _output_check(output_dir)
    checks["ok"] = all(value["ok"] for key, value in checks.items() if key in {"python", "torch", "rdkit", "hydra", "output_dir"})
    return checks


def doctor_command(output_dir: Path | None = None, as_json: bool = False) -> None:
    """Print environment checks and return a nonzero status only for essentials."""
    diagnostics = collect_diagnostics(output_dir)
    if as_json:
        typer.echo(json.dumps(diagnostics, indent=2, sort_keys=True))
    else:
        for name, result in diagnostics.items():
            if name == "ok":
                continue
            marker = "ok" if result["ok"] else "missing"
            typer.echo(f"{name:12} {marker:8} {result['detail']}")
        typer.echo("doctor: " + ("ready" if diagnostics["ok"] else "required checks failed"))
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

        return {"ok": True, "detail": f"available={torch.cuda.is_available()} devices={torch.cuda.device_count()}"}
    except (ImportError, RuntimeError, OSError) as error:  # pragma: no cover - environment-specific
        return {"ok": False, "detail": f"unavailable: {type(error).__name__}"}


def _executable_check(name: str) -> dict[str, Any]:
    """Return an optional executable discovery record."""
    path = shutil.which(name)
    return {"ok": path is not None, "detail": path or "optional executable not found"}


def _output_check(path: Path) -> dict[str, Any]:
    """Check that an output directory can be created and written."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".ecloudflow-doctor"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return {"ok": True, "detail": str(path)}
    except OSError as error:
        return {"ok": False, "detail": f"{type(error).__name__}: {error}"}
