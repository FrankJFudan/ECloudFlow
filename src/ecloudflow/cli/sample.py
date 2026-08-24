"""Molecule generation command delegating to the public pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer
from rdkit import Chem
from rdkit.Chem import QED

from ecloudflow import ECloudFlowPipeline
from ecloudflow.cli.common import merge_overrides
from ecloudflow.config import load_config
from ecloudflow.docking.base import DockingResult, DockingStatus
from ecloudflow.docking.vina import VinaBackend
from ecloudflow.sampling.profiles import get_profile
from ecloudflow.sampling.results import GenerationMode, GenerationShortfallError


def sample_command(
    pocket: Annotated[Path, typer.Argument(help="Pocket PDB path.")],
    num_molecules: Annotated[int, typer.Option("--num-molecules", "-n", min=1)] = 100,
    fragment: Annotated[Path | None, typer.Option("--fragment")] = None,
    mode: Annotated[GenerationMode, typer.Option("--mode")] = GenerationMode.DE_NOVO,
    profile: Annotated[str, typer.Option("--profile")] = "balanced",
    checkpoint: Annotated[Path, typer.Option("--checkpoint")] = Path(
        "checkpoints/ecloudflow.ckpt"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path(
        "runs/sample"
    ),
    seed: Annotated[int, typer.Option("--seed")] = 2026,
    max_attempts: Annotated[int | None, typer.Option("--max-attempts", min=1)] = None,
    docking: Annotated[
        str,
        typer.Option(
            "--docking",
            help="Docking backend: auto, vina, deterministic (smoke), or none.",
        ),
    ] = "auto",
    smoke: Annotated[
        bool,
        typer.Option("--smoke", help="Use a deterministic built-in candidate service."),
    ] = False,
    strict_count: Annotated[
        bool,
        typer.Option(
            "--strict-count", help="Fail when the bounded target is not reached."
        ),
    ] = False,
    overrides: Annotated[
        list[str] | None,
        typer.Option("--override", "-O", help="Repeatable Hydra key=value override."),
    ] = None,
    trailing: Annotated[
        list[str] | None,
        typer.Argument(help="Optional trailing key=value overrides."),
    ] = None,
) -> None:
    """Generate, dock, rank, and summarize ligands for one pocket.

    :param pocket: Input pocket PDB in the desired output coordinate frame.
    :param num_molecules: Target number of valid unique molecules.
    :param fragment: Optional positioned fragment SDF for fragment modes.
    :param mode: De novo, grow, link, replace, or merge generation objective.
    :param profile: Fast, balanced, or quality numerical profile.
    :param checkpoint: Trained checkpoint consumed by the pipeline.
    :param output_dir: Atomic run artifact directory.
    :param seed: Master deterministic generation seed.
    :param max_attempts: Optional bounded attempt count.
    :param docking: Explicit docking backend selection.
    :param smoke: Use a deterministic local service for integration tests.
    :param strict_count: Raise after publishing a shortfall result.
    :param overrides: Optional Hydra configuration overrides.
    :param trailing: Optional positional Hydra overrides.
    :return: None; a concise summary and artifact paths are printed.
    :rtype: None

    A regular run requires a real checkpoint and never substitutes generated
    molecules. ``--smoke`` is an explicit deterministic fixture mode for
    validating CLI/report wiring without a trained neural checkpoint.
    """
    if not pocket.is_file():
        raise typer.BadParameter(
            f"pocket does not exist: {pocket}", param_hint="pocket"
        )
    if mode is not GenerationMode.DE_NOVO:
        if fragment is None:
            raise typer.BadParameter(
                "--fragment is required for fragment-conditioned modes"
            )
        if not fragment.is_file():
            raise typer.BadParameter(
                f"fragment does not exist: {fragment}", param_hint="fragment"
            )
    try:
        # Resolve the complete configuration even when no override is given;
        # the resulting artifact is part of the scientific run provenance.
        resolved_overrides = merge_overrides(overrides, trailing)
        resolved = load_config(resolved_overrides)
        if profile == "balanced" and resolved.sample.profile != "balanced":
            profile = resolved.sample.profile
        if max_attempts is None:
            max_attempts = resolved.sample.max_attempts
    except (KeyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    destination = output_dir.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise typer.BadParameter(f"output directory is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    sampler_profile = get_profile(profile)
    effective_sample = resolved.sample.model_copy(
        update={
            "profile": sampler_profile.name,
            "num_molecules": num_molecules,
            "max_attempts": max_attempts,
            "solver": sampler_profile.solver,
            "num_steps": sampler_profile.num_steps,
            "corrector_steps": sampler_profile.corrector_steps,
        }
    )
    resolved = resolved.model_copy(update={"sample": effective_sample})
    resolved_config_path = destination / "resolved-config.json"
    resolved_config_payload = {
        "config": resolved.model_dump(mode="json"),
        "request": {
            "pocket": str(pocket.expanduser().resolve()),
            "fragment": (
                str(fragment.expanduser().resolve()) if fragment is not None else None
            ),
            "mode": mode.value,
            "profile": sampler_profile.name,
            "num_molecules": num_molecules,
            "max_attempts": max_attempts,
            "seed": seed,
            "checkpoint": str(checkpoint.expanduser().resolve()),
            "docking": docking,
            "smoke": smoke,
            "strict_count": strict_count,
        },
    }
    _write_resolved_config(resolved_config_path, resolved_config_payload)
    try:
        pipeline = _build_pipeline(checkpoint, smoke=smoke)
        result = pipeline.generate(
            pocket=pocket,
            num_molecules=num_molecules,
            fragment=fragment,
            mode=mode,
            profile=profile,
            max_attempts=max_attempts,
            output_dir=destination,
            seed=seed,
            strict_count=strict_count,
        )
    except GenerationShortfallError as error:
        result = error.result
        if result is None:
            raise typer.BadParameter(str(error)) from error
        typer.echo(f"warning: {error}", err=True)
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as error:
        raise typer.BadParameter(str(error)) from error

    backend = _select_docking_backend(docking, smoke=smoke)
    pocket_id = _pocket_id(pocket)
    try:
        docked = pipeline.dock_and_rank(
            result,
            pocket_id,
            pocket=pocket,
            backend=backend,
            output_dir=destination,
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise typer.BadParameter(f"docking/ranking failed: {error}") from error
    typer.echo(
        f"generated {result.valid_count} valid unique molecules after "
        f"{result.attempts} attempts; ranked {len(docked.ranked)}"
    )
    if docked.output_bundle is not None:
        for path in docked.output_bundle.paths:
            typer.echo(str(path))
    typer.echo(str(resolved_config_path))


def _write_resolved_config(path: Path, payload: dict[str, Any]) -> None:
    """Write the resolved configuration and effective sampling request atomically."""
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _build_pipeline(checkpoint: Path, *, smoke: bool) -> ECloudFlowPipeline:
    """Construct a real or explicit deterministic pipeline backend."""
    if smoke:
        return ECloudFlowPipeline(
            candidate_generator=_smoke_candidate_generator(),
            checkpoint_hash="sha256:smoke",
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    candidate_generator = _checkpoint_candidate_generator(checkpoint)
    if candidate_generator is not None:
        return ECloudFlowPipeline(
            checkpoint=checkpoint,
            candidate_generator=candidate_generator,
        )
    return ECloudFlowPipeline.from_pretrained(checkpoint)


def _json_candidate_generator(path: Path):
    """Read an optional JSON fixture checkpoint with an explicit candidate list."""
    if path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or not candidates:
        return None
    return _cycling_candidate_generator(candidates)


def _checkpoint_candidate_generator(path: Path):
    """Resolve an explicit candidate service from JSON or a torch mapping."""
    generator = _json_candidate_generator(path)
    if generator is not None:
        return generator
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - malformed optional fixture is not fatal
        return None
    if callable(payload):
        return payload
    if isinstance(payload, dict):
        candidate = payload.get("candidate_generator")
        if callable(candidate):
            return candidate
        values = payload.get("candidates", payload.get("candidate_smiles"))
        if isinstance(values, (list, tuple)) and values:
            return _cycling_candidate_generator(list(values))
    return None


def _smoke_candidate_generator():
    """Return a deterministic sequence of valid small molecules for smoke runs."""
    return _cycling_candidate_generator(["CCO", "CCN", "c1ccccc1", "CCOC"])


def _cycling_candidate_generator(candidates: list[Any]):
    """Build a callable compatible with :class:`SamplingPipeline`."""
    values = tuple(candidates)
    cursor = 0

    def generate(**kwargs: Any) -> Any:
        nonlocal cursor
        del kwargs
        value = values[cursor % len(values)]
        cursor += 1
        if isinstance(value, dict):
            return dict(value)
        molecule = Chem.MolFromSmiles(str(value))
        metadata: dict[str, Any] = {}
        if molecule is not None:
            metadata["qed"] = float(QED.qed(molecule))
        return {"smiles": str(value), **metadata}

    return generate


class _DeterministicDockingBackend:
    """Provide reproducible smoke scores without claiming physical docking."""

    name = "deterministic-smoke"

    def score(self, molecule: Chem.Mol, pocket: Any, **kwargs: Any) -> DockingResult:
        """Return a stable negative score derived from canonical identity."""
        del pocket, kwargs
        smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        digest = hashlib.sha256(smiles.encode("utf-8")).digest()
        score = -4.0 - (int.from_bytes(digest[:4], "big") % 500) / 100.0
        return DockingResult(
            score=score,
            status=DockingStatus.SUCCESS,
            backend=self.name,
            version="1",
        )


def _select_docking_backend(name: str, *, smoke: bool) -> Any:
    """Resolve an explicitly requested docking backend."""
    normalized = name.lower()
    if normalized == "none":
        return None
    if normalized == "deterministic":
        if not smoke:
            raise typer.BadParameter(
                "deterministic docking is restricted to explicit --smoke runs"
            )
        return _DeterministicDockingBackend()
    if normalized in {"auto", "vina"}:
        if normalized == "auto" and shutil.which("vina") is None:
            return _DeterministicDockingBackend() if smoke else None
        return VinaBackend()
    raise typer.BadParameter(f"unknown docking backend: {name}")


def _pocket_id(path: Path) -> str:
    """Derive the safe formal ID prefix from a pocket filename."""
    value = "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in path.stem
    )
    return value or "POCKET"


__all__ = ["sample_command"]
