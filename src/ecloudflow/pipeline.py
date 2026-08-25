"""Public bounded generation service for ECloudFlow.

The public service deliberately keeps model execution, chemical validation,
deduplication, and artifact publication in separate steps.  A learned model
can therefore be replaced by a deterministic test service without changing
the output contract, while every attempt remains auditable.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from ecloudflow.chemistry.relax import relax_molecule
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core.types import FragmentCondition
from ecloudflow.docking.base import DockingResult, DockingStatus
from ecloudflow.evaluation.outputs import OutputBundle, write_ranked_outputs
from ecloudflow.evaluation.ranking import RankedMolecule, rank_molecules
from ecloudflow.models import ECloudFlowModel
from ecloudflow.sampling.pipeline import (
    SamplingPipeline,
    build_fragment_condition,
    predict_atom_count,
)
from ecloudflow.sampling.profiles import SamplingProfile, get_profile
from ecloudflow.sampling.results import (
    GenerationAttempt,
    GenerationMode,
    GenerationRecord,
    GenerationRequest,
    GenerationResult,
    GenerationShortfallError,
)

CandidateGenerator = Callable[..., Any]


@dataclass(frozen=True)
class DockingRun:
    """Collect docking results, formal ranks, and optional output artifacts.

    :param ranked: Successfully scored rows in deterministic rank order.
    :param unranked: Records with missing or failed docking scores.
    :param docking_results: Backend results keyed by temporary attempt ID.
    :param output_bundle: Optional tabular/SDF publication manifest.
    :return: Immutable docking/ranking summary.
    :rtype: DockingRun
    """

    ranked: tuple[RankedMolecule, ...]
    unranked: tuple[GenerationRecord, ...]
    docking_results: Mapping[str, DockingResult]
    output_bundle: OutputBundle | None = None


@dataclass
class ECloudFlowPipeline:
    """Generate bounded sets of valid, unique ligands for one pocket.

    :param model: Optional learned model or callable sampler backend.
    :param checkpoint: Original checkpoint object or path, retained for
        provenance and optional lazy loading.
    :param candidate_generator: Optional injected service.  It may return an
        RDKit molecule, a SMILES string, a :class:`MolecularState`, a sampling
        trajectory, or a mapping containing one of those values.
    :param sampler: Optional low-level :class:`SamplingPipeline`.  When it is
        omitted, one is constructed from ``model`` or ``candidate_generator``.
    :param checkpoint_hash: Explicit provenance identifier.  A file-backed
        checkpoint uses a ``sha256:`` digest when this is omitted.
    :param relax_generated: Whether output runs also write a separately
        relaxed SDF beside every raw binding-pose SDF.
    :param relaxation_method: ``MMFF`` or ``UFF`` force-field method.
    :return: Mutable service object safe to reuse across generation requests.
    :rtype: ECloudFlowPipeline

    The pipeline never silently substitutes a candidate or loops beyond the
    configured attempt budget.  Canonical isomeric SMILES are the uniqueness
    key, and all accepted records retain their original generated pose in the
    raw artifact even when a relaxed copy is requested.
    """

    model: Any = None
    checkpoint: Any = None
    candidate_generator: CandidateGenerator | None = None
    sampler: SamplingPipeline | None = None
    checkpoint_hash: str | None = None
    relax_generated: bool = True
    relaxation_method: str = "MMFF"
    relaxation_iterations: int = 200
    docking_backend: Any = None

    def __post_init__(self) -> None:
        """Resolve service adapters and validate reusable pipeline settings."""
        if self.candidate_generator is not None and not callable(
            self.candidate_generator
        ):
            raise TypeError("candidate_generator must be callable.")
        if not isinstance(self.relax_generated, bool):
            raise TypeError("relax_generated must be boolean.")
        if self.relaxation_method.upper() not in {"MMFF", "UFF"}:
            raise ValueError("relaxation_method must be 'MMFF' or 'UFF'.")
        if (
            isinstance(self.relaxation_iterations, bool)
            or not isinstance(self.relaxation_iterations, int)
            or self.relaxation_iterations < 1
        ):
            raise ValueError("relaxation_iterations must be a positive integer.")
        self.relaxation_method = self.relaxation_method.upper()
        if not self.checkpoint_hash:
            self.checkpoint_hash = _checkpoint_hash(self.checkpoint)
        if not self.checkpoint_hash:
            self.checkpoint_hash = "uninitialized"
        if self.sampler is None:
            self.sampler = SamplingPipeline(
                candidate_generator=self.candidate_generator,
                model=self.model,
            )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path | bytes | Mapping[str, Any] | Any,
        *,
        model: Any = None,
        candidate_generator: CandidateGenerator | None = None,
        model_factory: Callable[..., Any] | None = None,
        map_location: str | torch.device = "cpu",
        load_weights: bool = True,
        **kwargs: Any,
    ) -> ECloudFlowPipeline:
        """Construct a pipeline from a checkpoint with a deterministic hash.

        :param checkpoint: Checkpoint path, serialized bytes, mapping, or an
            already-created backend object.
        :param model: Optional instantiated model receiving a checkpoint state
            dictionary through ``load_state_dict``.
        :param candidate_generator: Optional callable override for generation.
        :param model_factory: Optional callable receiving the loaded payload;
            a zero-argument factory is also accepted.
        :param map_location: Device passed to :func:`torch.load`.
        :param load_weights: Disable only when the caller wants lazy loading.
        :param kwargs: Additional :class:`ECloudFlowPipeline` constructor
            options such as relaxation settings.
        :return: Initialized pipeline with checkpoint provenance.
        :rtype: ECloudFlowPipeline
        :raises FileNotFoundError: If a path does not exist.
        :raises ValueError: If a serialized checkpoint cannot be read.

        Loading is intentionally conservative: a supplied model is populated
        only from a recognizable ``state_dict`` mapping, while a callable
        serialized object can serve directly as the candidate backend.
        """
        digest = _checkpoint_hash(checkpoint)
        payload: Any = None
        source = Path(checkpoint) if isinstance(checkpoint, (str, Path)) else None
        if source is not None:
            if not source.is_file():
                raise FileNotFoundError(f"checkpoint does not exist: {source}")
            if (
                load_weights
                or model_factory is not None
                or (model is None and candidate_generator is None)
            ):
                payload = _torch_load(source, map_location)
        elif isinstance(checkpoint, bytes):
            if load_weights or model_factory is not None:
                payload = _torch_load_bytes(checkpoint, map_location)
        elif isinstance(checkpoint, Mapping):
            payload = checkpoint
        else:
            payload = checkpoint

        built_from_payload = False
        if model_factory is not None:
            model = _call_factory(model_factory, payload)
        if (
            model is None
            and candidate_generator is None
            and isinstance(payload, Mapping)
        ):
            model = _build_checkpoint_model(payload, map_location=map_location)
            built_from_payload = True
        if (
            model is not None
            and payload is not None
            and load_weights
            and not built_from_payload
        ):
            _load_model_state(model, payload)
        if isinstance(model, ECloudFlowModel):
            model.eval()
        if candidate_generator is None and model is None and callable(payload):
            candidate_generator = payload
        if candidate_generator is None and model is None and callable(checkpoint):
            candidate_generator = checkpoint
        kwargs["checkpoint_hash"] = digest
        kwargs["checkpoint"] = checkpoint
        kwargs["model"] = model
        kwargs["candidate_generator"] = candidate_generator
        return cls(**kwargs)

    def generate(
        self,
        pocket: Any,
        num_molecules: int,
        fragment: Any = None,
        mode: GenerationMode = GenerationMode.DE_NOVO,
        profile: str | SamplingProfile = "balanced",
        max_attempts: int | None = None,
        output_dir: str | Path | None = None,
        seed: int = 2026,
        *,
        strict_count: bool = False,
    ) -> GenerationResult:
        """Generate valid unique ligands directly in a protein pocket.

        :param pocket: Pocket PDB path or parsed pocket object in the desired
            output coordinate frame.
        :param num_molecules: Target count of valid unique molecules after
            exact graph decoding and RDKit sanitization.
        :param fragment: Optional positioned fragment path/object for grow or
            replace generation, or a non-empty sequence of paths/objects for
            link and merge generation.
        :param mode: De novo or fragment-conditioned generation objective.
        :param profile: Named ``fast``, ``balanced``, or ``quality`` preset.
        :param max_attempts: Bounded attempts; defaults to five times target.
        :param output_dir: Optional directory for atomic run artifacts.
        :param seed: Master seed used to derive per-attempt generators.
        :param strict_count: Raise :class:`GenerationShortfallError` when the
            bounded budget cannot reach the requested target.
        :return: Valid records and every completed attempt.
        :rtype: GenerationResult
        :raises GenerationShortfallError: Only in strict-count mode after the
            completed result has been materialized and written.
        """
        request = GenerationRequest(
            pocket=pocket,
            num_molecules=num_molecules,
            fragment=fragment,
            mode=mode,
            profile=profile,
            max_attempts=max_attempts,
            output_dir=output_dir,
            seed=seed,
            strict_count=strict_count,
        )
        return self.generate_request(request)

    def generate_request(self, request: GenerationRequest) -> GenerationResult:
        """Execute one validated request with explicit attempt accounting.

        :param request: Immutable request shared by all supported generation
            modes.
        :return: Ordered unique records plus rejected/failed attempts.
        :rtype: GenerationResult
        :raises GenerationShortfallError: When ``request.strict_count`` is true
            and the bounded attempt budget is insufficient.
        """
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest.")
        profile = (
            request.profile
            if isinstance(request.profile, SamplingProfile)
            else get_profile(request.profile)
        )
        max_attempts = request.max_attempts or 5 * request.num_molecules
        destination = _prepare_output_dir(request.output_dir)
        fragment = _prepare_fragment_condition(
            request.pocket,
            request.fragment,
            request.mode,
            request.seed,
            self.model,
        )
        valid: list[GenerationRecord] = []
        attempts: list[GenerationAttempt] = []
        seen: set[str] = set()
        duplicate_count = 0

        for attempt_number in range(1, max_attempts + 1):
            if len(valid) >= request.num_molecules:
                break
            attempt_id = f"attempt-{attempt_number:06d}"
            attempt_seed = request.seed + attempt_number - 1
            started = time.monotonic()
            try:
                generator = _make_generator(attempt_seed)
                candidate = self.sampler.sample(
                    pocket=request.pocket,
                    fragment=fragment,
                    fixed=(
                        fragment
                        if isinstance(fragment, FragmentCondition)
                        else None
                    ),
                    mode=request.mode,
                    profile=profile,
                    seed=attempt_seed,
                    generator=generator,
                    attempt=attempt_number,
                )
                candidate, metadata = _unwrap_candidate(candidate)
                molecule = _coerce_molecule(candidate, attempt_seed)
                if isinstance(fragment, FragmentCondition):
                    _validate_fragment_output(molecule, fragment)
                canonical = Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=True
                )
                if not canonical:
                    raise _CandidateRejected("empty_canonical_smiles")
            except _CandidateRejected as error:
                attempts.append(
                    GenerationAttempt(
                        attempt_id=attempt_id,
                        status="rejected",
                        reason=error.reason,
                        seed=attempt_seed,
                        elapsed_seconds=_elapsed(started),
                    )
                )
                continue
            except Exception as error:  # noqa: BLE001 - boundary records failure
                attempts.append(
                    GenerationAttempt(
                        attempt_id=attempt_id,
                        status="failed",
                        reason=f"{type(error).__name__}: {error}",
                        seed=attempt_seed,
                        elapsed_seconds=_elapsed(started),
                    )
                )
                continue

            if canonical in seen:
                duplicate_count += 1
                attempts.append(
                    GenerationAttempt(
                        attempt_id=attempt_id,
                        status="rejected",
                        reason="duplicate_smiles",
                        seed=attempt_seed,
                        elapsed_seconds=_elapsed(started),
                    )
                )
                continue

            try:
                raw_path, relaxed_path = _publish_molecule(
                    molecule,
                    destination,
                    attempt_id,
                    relax_generated=self.relax_generated,
                    relaxation_method=self.relaxation_method,
                    relaxation_iterations=self.relaxation_iterations,
                    fixed_atom_mask=(
                        fragment.fixed_atom_mask
                        if isinstance(fragment, FragmentCondition)
                        else None
                    ),
                )
            except Exception as error:  # noqa: BLE001 - artifact boundary
                attempts.append(
                    GenerationAttempt(
                        attempt_id=attempt_id,
                        status="failed",
                        reason=f"artifact_error:{type(error).__name__}: {error}",
                        seed=attempt_seed,
                        elapsed_seconds=_elapsed(started),
                    )
                )
                continue

            properties = {
                "profile": profile.name,
                "pocket": _display_source(request.pocket),
                "fragment": _display_source(request.fragment),
                **metadata,
            }
            record = GenerationRecord(
                canonical_smiles=canonical,
                attempt_id=attempt_id,
                molecule=Chem.Mol(molecule),
                mode=request.mode,
                seed=attempt_seed,
                raw_path=raw_path,
                relaxed_path=relaxed_path,
                properties=properties,
                model_checkpoint_hash=self.checkpoint_hash or "uninitialized",
            )
            valid.append(record)
            seen.add(canonical)
            attempts.append(
                GenerationAttempt(
                    attempt_id=attempt_id,
                    status="valid",
                    record=record,
                    seed=attempt_seed,
                    elapsed_seconds=_elapsed(started),
                )
            )

        result = GenerationResult(
            valid=tuple(valid),
            attempt_records=tuple(attempts),
            target_count=request.num_molecules,
            duplicate_count=duplicate_count,
            model_checkpoint_hash=self.checkpoint_hash or "uninitialized",
            output_dir=destination,
            mode=request.mode,
        )
        if destination is not None:
            _write_manifest(destination / "generation.json", request, result)
        if self.model is not None and result.valid_count == 0:
            first_failure = next(
                (attempt.reason for attempt in attempts if attempt.reason),
                "no candidate was returned",
            )
            raise RuntimeError(
                "checkpoint-backed sampling produced no valid molecule; "
                f"first attempt: {first_failure}"
            )
        if request.strict_count and result.shortfall:
            raise GenerationShortfallError(
                f"requested {request.num_molecules} unique molecules but obtained "
                f"{result.valid_count} after {result.attempts} attempts",
                result,
            )
        return result

    def dock_and_rank(
        self,
        result: GenerationResult,
        pocket_id: str,
        *,
        pocket: Any = None,
        backend: Any = None,
        box_center: tuple[float, float, float] | None = None,
        box_size: tuple[float, float, float] | None = None,
        output_dir: str | Path | None = None,
    ) -> DockingRun:
        """Dock valid records, rank them, and optionally publish output files.

        :param result: Completed generation result from :meth:`generate`.
        :param pocket_id: Safe identifier used in formal molecule IDs.
        :param pocket: Original pocket path/object passed to the backend.
        :param backend: Optional docking backend overriding the configured one.
            When omitted, existing record properties are ranked without
            invoking a backend; records lacking scores remain unranked.
        :param box_center: Optional search-box center forwarded to the backend.
        :param box_size: Optional positive search-box size forwarded to backend.
        :param output_dir: Optional output directory for ranked artifacts.
        :return: Docking results, formal ranks, and output manifest.
        :rtype: DockingRun
        :raises TypeError: If ``result`` is not a :class:`GenerationResult`.

        A missing backend is represented by ``DISABLED`` results rather than a
        fabricated numeric score.  Ranking therefore remains deterministic and
        preserves every failure record for downstream evaluation.
        """
        if not isinstance(result, GenerationResult):
            raise TypeError("result must be a GenerationResult.")
        service = backend if backend is not None else self.docking_backend
        enriched: list[GenerationRecord] = []
        docking_results: dict[str, DockingResult] = {}
        for record in result.valid:
            properties = dict(record.properties)
            if service is None and not _has_docking_score(properties):
                docking = DockingResult(
                    score=None,
                    status=DockingStatus.DISABLED,
                    backend="disabled",
                    reason="no docking backend configured",
                )
            elif service is None:
                docking = DockingResult(
                    score=float(_get_docking_score(properties)),
                    status=DockingStatus.SUCCESS,
                    backend="record",
                )
            else:
                try:
                    docking = _invoke_docking(
                        service,
                        record.molecule,
                        pocket if pocket is not None else properties.get("pocket"),
                        box_center=box_center,
                        box_size=box_size,
                    )
                except Exception as error:  # noqa: BLE001 - optional tool boundary
                    docking = DockingResult(
                        score=None,
                        status=DockingStatus.FAILED,
                        backend=str(getattr(service, "name", type(service).__name__)),
                        reason=f"{type(error).__name__}: {error}",
                    )
            docking_results[record.temporary_id] = docking
            properties.update(
                {
                    "docking_score": docking.score,
                    "docking_status": docking.status.value,
                    "docking_backend": docking.backend,
                }
            )
            enriched.append(replace(record, properties=properties))
        ranked, unranked = rank_molecules(pocket_id, enriched)
        destination = Path(output_dir) if output_dir is not None else result.output_dir
        bundle = (
            write_ranked_outputs(ranked, unranked, destination)
            if destination is not None
            else None
        )
        return DockingRun(
            ranked=tuple(ranked),
            unranked=tuple(unranked),
            docking_results=docking_results,
            output_bundle=bundle,
        )

    def generate_and_rank(
        self,
        request: GenerationRequest,
        pocket_id: str,
        *,
        pocket: Any = None,
        backend: Any = None,
        box_center: tuple[float, float, float] | None = None,
        box_size: tuple[float, float, float] | None = None,
        output_dir: str | Path | None = None,
    ) -> DockingRun:
        """Run bounded generation followed by optional docking and publication."""
        result = self.generate_request(request)
        return self.dock_and_rank(
            result,
            pocket_id,
            pocket=pocket,
            backend=backend,
            box_center=box_center,
            box_size=box_size,
            output_dir=output_dir,
        )


class _CandidateRejected(ValueError):
    """Internal marker for candidates rejected by chemical normalization."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _prepare_fragment_condition(
    pocket: Any,
    fragment: Any,
    mode: GenerationMode,
    seed: int,
    model: Any,
) -> Any:
    """Convert an existing fragment file into the exact sampling contract.

    :param pocket: Pocket source defining the local binding frame.
    :param fragment: Fragment condition, SDF path/sequence of paths, or
        injected fixture value.
    :param mode: Requested generation objective.
    :param seed: Request seed used for deterministic editable-node placement.
    :param model: Optional loaded model whose device/dtype own inference.
    :return: Exact :class:`FragmentCondition` for real file inputs, otherwise
        the original injected value for backwards-compatible test services.
    :rtype: object
    :raises ValueError: If an existing fragment file cannot form a positioned
        graph condition in the pocket coordinate frame.

    Conversion happens once per public request, before bounded attempts.  The
    first fragment nodes preserve source atom order, identity, formal charge,
    internal bonds, and coordinates; editable capacity is appended after them.
    This prevents attempt-specific parsing or random masks from changing the
    conditioning problem.
    """
    if mode is GenerationMode.DE_NOVO or fragment is None:
        return fragment
    if isinstance(fragment, FragmentCondition):
        return fragment
    if not _fragment_sources_are_supported(fragment):
        return fragment
    device, dtype = _model_device_dtype(model)
    fallback_extra_atoms = {
        GenerationMode.GROW: 8,
        GenerationMode.LINK: 10,
        GenerationMode.REPLACE: 8,
        GenerationMode.MERGE: 10,
    }[mode]
    extra_atoms = fallback_extra_atoms
    if isinstance(model, ECloudFlowModel):
        fixed_only = build_fragment_condition(
            pocket,
            fragment,
            extra_atoms=0,
            device=device,
            dtype=dtype,
            seed=seed,
            task_id=mode.value,
            electron_latent_dim=int(model.electron_latent_dim),
        )
        predicted_count = predict_atom_count(model, pocket, fixed=fixed_only)
        fixed_count = int(fixed_only.fixed_atom_mask.sum())
        if predicted_count < fixed_count:
            raise ValueError(
                "model predicted fewer atoms than the fixed fragment"
            )
        # A count equal to the fixed scaffold is a valid no-growth outcome. Keep
        # the state size identical to the model's categorical prediction instead
        # of appending an artificial editable atom (which would violate the
        # learned count and can make a max-sized fragment impossible to sample).
        extra_atoms = predicted_count - fixed_count
    return build_fragment_condition(
        pocket,
        fragment,
        extra_atoms=extra_atoms,
        device=device,
        dtype=dtype,
        seed=seed,
        task_id=mode.value,
        electron_latent_dim=int(getattr(model, "electron_latent_dim", 48)),
    )


def _model_device_dtype(model: Any) -> tuple[torch.device, torch.dtype]:
    """Return model inference placement or the portable CPU default."""
    if model is None or not hasattr(model, "parameters"):
        return torch.device("cpu"), torch.float32
    try:
        parameter = next(model.parameters())
    except StopIteration:
        return torch.device("cpu"), torch.float32
    dtype = parameter.dtype
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return parameter.device, dtype


def _fragment_sources_are_supported(fragment: Any) -> bool:
    """Return whether a fragment source can form an exact positioned condition.

    Existing files are checked eagerly so arbitrary fixture strings remain
    available to injected candidate services. RDKit molecules are accepted
    directly, and mixed path/molecule sequences are normalized by
    :func:`build_fragment_condition` without changing their atom order.
    """
    if isinstance(fragment, Chem.Mol):
        return True
    if isinstance(fragment, (str, Path)):
        return Path(fragment).is_file()
    if isinstance(fragment, (list, tuple)) and fragment:
        return all(
            isinstance(value, Chem.Mol)
            or (isinstance(value, (str, Path)) and Path(value).is_file())
            for value in fragment
        )
    return False


def _validate_fragment_output(
    molecule: Chem.Mol, condition: FragmentCondition
) -> None:
    """Reject any final molecule that changed an exact fragment invariant.

    :param molecule: Sanitized candidate in the original pocket coordinate frame.
    :param condition: Local-frame exact fragment reference and masks.
    :return: None when fixed atom identity, formal charge, internal connectivity,
        bond order semantics, and global coordinates all match.
    :rtype: None
    :raises _CandidateRejected: If a fixed value changed or the conformer is absent.

    Atom indices are intentionally stable: fragment atoms occupy the first
    reference nodes and generated atoms are appended.  Aromatic RDKit bonds are
    accepted for fixed Kekule single/double ring edges because sanitization
    recovers aromaticity after the exact decoder has already enforced the stored
    alternating bond classes. Coordinate comparison uses a tight serialization
    tolerance only because RDKit conformers store C++ doubles, while all tensor
    clamping before reconstruction is bitwise exact.
    """
    reference = condition.reference
    fixed = condition.fixed_atom_mask.detach().cpu()
    indices = torch.nonzero(fixed, as_tuple=False).flatten().tolist()
    if molecule.GetNumAtoms() != reference.positions.shape[0]:
        raise _CandidateRejected("fragment_invariant_failed:atom_count_changed")
    vocabulary = ChemicalVocabulary.default_ligand()
    atoms = reference.atom_logits.argmax(dim=-1).detach().cpu()
    charges = reference.charge_logits.argmax(dim=-1).detach().cpu()
    for index in indices:
        atom = molecule.GetAtomWithIdx(index)
        if atom.GetSymbol() != vocabulary.atom_symbol(int(atoms[index])):
            raise _CandidateRejected(
                f"fragment_invariant_failed:atom_identity:{index}"
            )
        expected_charge = vocabulary.formal_charges[int(charges[index])]
        if atom.GetFormalCharge() != expected_charge:
            raise _CandidateRejected(
                f"fragment_invariant_failed:formal_charge:{index}"
            )
    if molecule.GetNumConformers() != 1:
        raise _CandidateRejected("fragment_invariant_failed:missing_pose")
    expected_positions = reference.positions.detach().cpu()
    if reference.frame is not None:
        expected_positions = reference.frame.to_global(reference.positions).detach().cpu()
    observed = torch.tensor(
        molecule.GetConformer().GetPositions(), dtype=expected_positions.dtype
    )
    if not torch.allclose(
        observed[fixed], expected_positions[fixed], atol=1.0e-5, rtol=0.0
    ):
        raise _CandidateRejected("fragment_invariant_failed:coordinates")
    source, target = reference.halfedge_index.detach().cpu()
    fixed_bonds = condition.fixed_bond_mask.detach().cpu()
    expected_classes = reference.bond_logits.argmax(dim=-1).detach().cpu()
    for edge_index in torch.nonzero(fixed_bonds, as_tuple=False).flatten().tolist():
        first, second = int(source[edge_index]), int(target[edge_index])
        expected_order = vocabulary.bond_orders[int(expected_classes[edge_index])]
        bond = molecule.GetBondBetweenAtoms(first, second)
        if expected_order == 0.0 and bond is None:
            continue
        if expected_order == 0.0:
            raise _CandidateRejected(
                f"fragment_invariant_failed:unexpected_internal_bond:{first}-{second}"
            )
        if bond is None:
            raise _CandidateRejected(
                f"fragment_invariant_failed:internal_bond_missing:{first}-{second}"
            )
        observed_order = float(bond.GetBondTypeAsDouble())
        aromatic_equivalent = bond.GetIsAromatic() and expected_order in {1.0, 2.0}
        if not aromatic_equivalent and observed_order != expected_order:
            raise _CandidateRejected(
                f"fragment_invariant_failed:internal_bond_order:{first}-{second}"
            )


def _coerce_molecule(candidate: Any, seed: int) -> Chem.Mol:
    """Normalize one candidate to a sanitized defensive RDKit molecule."""
    if isinstance(candidate, GenerationRecord):
        candidate = candidate.molecule
    if isinstance(candidate, (bytes, bytearray)):
        try:
            candidate = bytes(candidate).decode("utf-8")
        except UnicodeDecodeError as error:
            raise _CandidateRejected("invalid_utf8_candidate") from error
    if isinstance(candidate, str):
        if not candidate.strip():
            raise _CandidateRejected("empty_smiles")
        candidate = Chem.MolFromSmiles(candidate, sanitize=False)
        if candidate is None:
            raise _CandidateRejected("invalid_smiles")
    if not isinstance(candidate, Chem.Mol):
        raise _CandidateRejected(
            f"unsupported_candidate_type:{type(candidate).__name__}"
        )
    molecule = Chem.Mol(candidate)
    if molecule.GetNumAtoms() == 0:
        raise _CandidateRejected("empty_molecule")
    try:
        Chem.SanitizeMol(molecule)
    except (RuntimeError, ValueError) as error:
        raise _CandidateRejected(f"sanitization_failed:{error}") from error
    _ensure_conformer(molecule, seed)
    return molecule


def _unwrap_candidate(candidate: Any) -> tuple[Any, dict[str, Any]]:
    """Extract a candidate value and JSON-safe metadata from service output."""
    if isinstance(candidate, Mapping):
        value = None
        for key in ("molecule", "mol", "smiles", "state", "trajectory", "candidate"):
            if key in candidate:
                value = candidate[key]
                break
        if value is None:
            raise _CandidateRejected("candidate_mapping_has_no_value")
        metadata = {
            str(key): _json_safe(value)
            for key, value in candidate.items()
            if key
            not in {"molecule", "mol", "smiles", "state", "trajectory", "candidate"}
        }
        return value, metadata
    if isinstance(candidate, (tuple, list)) and candidate:
        metadata = (
            candidate[1]
            if len(candidate) > 1 and isinstance(candidate[1], Mapping)
            else {}
        )
        return candidate[0], {
            str(key): _json_safe(value) for key, value in metadata.items()
        }
    return candidate, {}


def _ensure_conformer(molecule: Chem.Mol, seed: int) -> None:
    """Ensure an accepted molecule has a usable conformer for SDF output."""
    if molecule.GetNumConformers() > 0:
        return
    random_seed = int(seed) & 0x7FFFFFFF
    try:
        status = AllChem.EmbedMolecule(molecule, randomSeed=random_seed)
    except (RuntimeError, ValueError):
        status = -1
    if status != 0 and molecule.GetNumConformers() == 0:
        try:
            AllChem.Compute2DCoords(molecule)
        except (RuntimeError, ValueError) as error:
            raise _CandidateRejected(f"conformer_generation_failed:{error}") from error


def _publish_molecule(
    molecule: Chem.Mol,
    destination: Path | None,
    attempt_id: str,
    *,
    relax_generated: bool,
    relaxation_method: str,
    relaxation_iterations: int,
    fixed_atom_mask: torch.Tensor | None,
) -> tuple[Path | None, Path | None]:
    """Write separate raw and relaxed artifacts with atomic replacement."""
    if destination is None:
        return None, None
    folder = destination / "molecules" / attempt_id
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / "raw.sdf"
    relaxed_path = folder / "relaxed.sdf"
    _atomic_write_sdf(molecule, raw_path)
    if relax_generated:
        relaxed = relax_molecule(
            molecule,
            fixed_atom_mask=fixed_atom_mask,
            method=relaxation_method,
            max_iterations=relaxation_iterations,
        )
        if not isinstance(relaxed, Chem.Mol):
            relaxed = relaxed.molecule
        _atomic_write_sdf(relaxed, relaxed_path)
    else:
        _atomic_write_sdf(molecule, relaxed_path)
    return raw_path, relaxed_path


def _atomic_write_sdf(molecule: Chem.Mol, path: Path) -> None:
    """Serialize one molecule through a sibling temporary file."""
    temporary = path.with_name(path.name + ".partial")
    writer = Chem.SDWriter(str(temporary))
    try:
        writer.write(Chem.Mol(molecule))
    finally:
        writer.close()
    temporary.replace(path)


def _write_manifest(
    path: Path, request: GenerationRequest, result: GenerationResult
) -> None:
    """Write request provenance and result records as an atomic JSON manifest."""
    payload = result.as_dict()
    payload["request"] = {
        "pocket": _display_source(request.pocket),
        "fragment": _display_source(request.fragment),
        "num_molecules": request.num_molecules,
        "mode": request.mode.value,
        "profile": (
            request.profile.name
            if isinstance(request.profile, SamplingProfile)
            else request.profile
        ),
        "max_attempts": request.max_attempts,
        "seed": request.seed,
        "strict_count": request.strict_count,
    }
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _prepare_output_dir(value: str | Path | None) -> Path | None:
    """Create and validate a run directory without touching absent output."""
    if value is None:
        return None
    # Resolve once at the public boundary so serialized pose paths remain
    # usable even when the caller supplied a relative run directory.
    destination = Path(value).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"output_dir is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _make_generator(seed: int) -> torch.Generator:
    """Create a caller-owned CPU generator for deterministic service calls."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return generator


def _elapsed(started: float) -> float:
    """Return a finite non-negative monotonic elapsed duration."""
    return max(0.0, float(time.monotonic() - started))


def _display_source(value: Any) -> Any:
    """Convert path-like and scalar request sources into stable text."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_display_source(item) for item in value]
    return type(value).__name__


def _json_safe(value: Any) -> Any:
    """Recursively convert provenance values into deterministic JSON values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return str(value)


def _checkpoint_hash(checkpoint: Any) -> str:
    """Compute a stable checkpoint identity without hashing model reprs."""
    if checkpoint is None:
        return ""
    if isinstance(checkpoint, bytes):
        return "sha256:" + hashlib.sha256(checkpoint).hexdigest()
    if isinstance(checkpoint, (str, Path)):
        source = Path(checkpoint)
        if source.is_file():
            return "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
        if source.is_dir():
            digest = hashlib.sha256()
            for child in sorted(path for path in source.rglob("*") if path.is_file()):
                digest.update(
                    str(child.relative_to(source)).replace("\\", "/").encode()
                )
                digest.update(child.read_bytes())
            return "sha256:" + digest.hexdigest()
        return str(checkpoint)
    if isinstance(checkpoint, Mapping):
        encoded = json.dumps(
            _json_safe(checkpoint), sort_keys=True, separators=(",", ":")
        )
        return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
    return getattr(checkpoint, "checkpoint_hash", "") or ""


def _torch_load(source: Path, map_location: str | torch.device) -> Any:
    """Load a torch checkpoint with a compatibility fallback for older builds."""
    try:
        return torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(source, map_location=map_location)


def _torch_load_bytes(payload: bytes, map_location: str | torch.device) -> Any:
    """Load serialized bytes through an in-memory buffer."""
    import io

    try:
        return torch.load(
            io.BytesIO(payload), map_location=map_location, weights_only=False
        )
    except TypeError:
        return torch.load(io.BytesIO(payload), map_location=map_location)


def _load_model_state(model: Any, payload: Any) -> None:
    """Load a recognizable state dictionary while preserving strict errors."""
    if not hasattr(model, "load_state_dict"):
        return
    state = payload
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint does not contain a recognizable state dictionary")
    if isinstance(model, ECloudFlowModel):
        state = _joint_backbone_state(
            {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
        )
    model.load_state_dict(state, strict=True)


def _build_checkpoint_model(
    payload: Mapping[str, Any], *, map_location: str | torch.device
) -> ECloudFlowModel:
    """Construct and strictly restore an ECloudFlow model from Lightning state.

    :param payload: Loaded Lightning or plain PyTorch checkpoint mapping.
    :param map_location: Requested inference device used for the restored model.
    :return: Evaluation-mode joint backbone with every expected tensor restored.
    :rtype: ECloudFlowModel
    :raises ValueError: If the checkpoint has no recognizable backbone state,
        contains an incomplete architecture, or its metadata conflicts with
        tensor-derived dimensions.

    Lightning stores training-module keys below ``joint_backbone.*`` whereas a
    model-only checkpoint stores them without a prefix.  The loader normalizes
    only these two documented layouts, derives all shape-identifiable widths
    from tensors, and reads ``lmax``/electron layout from the embedded resolved
    configuration when available.  Strict loading rejects missing, unexpected,
    or shape-incompatible tensors before sampling begins; auxiliary optimizer,
    tokenizer, decoder, EMA, and loss-scaler entries are deliberately excluded.
    """
    raw_state = _checkpoint_state(payload)
    state = _joint_backbone_state(raw_state)
    required = {
        "pocket_encoder.null_embedding",
        "position_velocity_head.weight",
        "atom_head.network.2.weight",
        "charge_head.network.2.weight",
        "bond_head.network.2.weight",
        "count_predictor.head.2.weight",
        "electron_velocity_head.parameters_head.2.weight",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(
            "checkpoint state_dict does not contain a complete ECloudFlow "
            f"joint_backbone; missing {', '.join(missing)}"
        )
    scalar_dim = int(state["pocket_encoder.null_embedding"].shape[0])
    vector_dim = int(state["position_velocity_head.weight"].shape[0]) - 1
    atom_classes = int(state["atom_head.network.2.weight"].shape[0])
    charge_classes = int(state["charge_head.network.2.weight"].shape[0])
    bond_classes = int(state["bond_head.network.2.weight"].shape[0])
    max_atoms = int(state["count_predictor.head.2.weight"].shape[0]) - 1
    electron_vector_dim = (
        int(state["electron_velocity_head.parameters_head.2.weight"].shape[0]) - 3
    )
    block_indices = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("pocket_encoder.blocks.") and key.split(".")[2].isdigit()
    }
    if not block_indices or block_indices != set(range(max(block_indices) + 1)):
        raise ValueError("checkpoint has an incomplete pocket encoder block sequence")
    architecture = _checkpoint_architecture(payload)
    lmax = int(architecture.get("lmax", 2))
    electron_latent_dim = int(architecture.get("electron_latent_dim", 48))
    configured = {
        "scalar_dim": scalar_dim,
        "vector_dim": vector_dim,
        "num_blocks": len(block_indices),
        "lmax": lmax,
    }
    for name, inferred in configured.items():
        if name in architecture and int(architecture[name]) != inferred:
            raise ValueError(
                f"checkpoint model metadata {name}={architecture[name]} conflicts "
                f"with state_dict value {inferred}"
            )
    vocabulary = ChemicalVocabulary.default_ligand()
    expected_channels = (
        len(vocabulary.atom_symbols),
        len(vocabulary.formal_charges),
        len(vocabulary.bond_classes),
    )
    if (atom_classes, charge_classes, bond_classes) != expected_channels:
        raise ValueError(
            "checkpoint atom/charge/bond channels do not match the canonical "
            f"ligand vocabulary: {(atom_classes, charge_classes, bond_classes)}"
        )
    model = ECloudFlowModel(
        scalar_dim=scalar_dim,
        vector_dim=vector_dim,
        num_blocks=len(block_indices),
        lmax=lmax,
        electron_latent_dim=electron_latent_dim,
        electron_vector_dim=electron_vector_dim,
        atom_classes=atom_classes,
        charge_classes=charge_classes,
        bond_classes=bond_classes,
        max_atoms=max_atoms,
        pocket_cutoff=float(architecture.get("pocket_cutoff", 8.0)),
        cross_cutoff=float(architecture.get("cross_cutoff", 8.0)),
    )
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as error:
        raise ValueError(f"checkpoint joint_backbone is incompatible: {error}") from error
    device = torch.device(map_location)
    model.to(device=device)
    model.eval()
    return model


def _checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Extract a tensor state mapping from documented checkpoint containers."""
    state: Any = payload
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint does not contain a non-empty state_dict")
    if not all(isinstance(key, str) for key in state):
        raise ValueError("checkpoint state_dict keys must be strings")
    tensors = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
    if not tensors:
        raise ValueError("checkpoint state_dict contains no tensors")
    return tensors


def _joint_backbone_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize model-only or Lightning joint-backbone key prefixes."""
    marker = "pocket_encoder.null_embedding"
    if marker in state:
        return dict(state)
    for prefix in ("joint_backbone.", "model.", "module.joint_backbone."):
        normalized = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if marker in normalized:
            return normalized
    raise ValueError(
        "checkpoint state_dict is not an ECloudFlow model-only or Lightning "
        "joint_backbone layout"
    )


def _checkpoint_architecture(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read optional architecture metadata without accepting arbitrary objects."""
    candidates: list[Any] = []
    reproducibility = payload.get("ecloudflow_reproducibility")
    if isinstance(reproducibility, Mapping):
        resolved = reproducibility.get("resolved_config")
        if isinstance(resolved, Mapping):
            candidates.append(resolved.get("model"))
    hyperparameters = payload.get("hyper_parameters")
    if isinstance(hyperparameters, Mapping):
        candidates.extend(
            (
                hyperparameters.get("model_config"),
                hyperparameters.get("model"),
            )
        )
        resolved = hyperparameters.get("resolved_config")
        if isinstance(resolved, Mapping):
            candidates.append(resolved.get("model"))
    candidates.append(payload.get("model_config"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def _call_factory(factory: Callable[..., Any], payload: Any) -> Any:
    """Call a model factory with its supported argument shape."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(payload)
    if not signature.parameters:
        return factory()
    return factory(payload)


def _has_docking_score(properties: Mapping[str, Any]) -> bool:
    """Return whether a record already carries a usable docking score."""
    return any(
        properties.get(name) is not None
        for name in ("docking_score", "vina_score", "vina")
    )


def _get_docking_score(properties: Mapping[str, Any]) -> float:
    """Read one existing finite docking score from record properties."""
    for name in ("docking_score", "vina_score", "vina"):
        if properties.get(name) is not None:
            value = float(properties[name])
            if not math.isfinite(value):
                raise ValueError(f"record {name} score must be finite")
            return value
    raise ValueError("record has no docking score")


def _invoke_docking(
    backend: Any,
    molecule: Any,
    pocket: Any,
    *,
    box_center: tuple[float, float, float] | None,
    box_size: tuple[float, float, float] | None,
) -> DockingResult:
    """Invoke injected backends with only the keyword arguments they accept."""
    kwargs: dict[str, Any] = {}
    if box_center is not None:
        kwargs["box_center"] = box_center
    if box_size is not None:
        kwargs["box_size"] = box_size
    service = backend.score if hasattr(backend, "score") else backend
    try:
        signature = inspect.signature(service)
    except (TypeError, ValueError):
        value = service(molecule, pocket, **kwargs)
    else:
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            value = service(molecule, pocket, **kwargs)
        else:
            accepted = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
            value = service(molecule, pocket, **accepted)
    if isinstance(value, DockingResult):
        return value
    if isinstance(value, Mapping):
        return DockingResult(**value)
    if value is None:
        return DockingResult(
            score=None,
            status=DockingStatus.FAILED,
            backend="custom",
            reason="backend returned None",
        )
    return DockingResult(
        score=float(value), status=DockingStatus.SUCCESS, backend="custom"
    )


__all__ = [
    "DockingRun",
    "ECloudFlowPipeline",
    "GenerationAttempt",
    "GenerationMode",
    "GenerationRecord",
    "GenerationRequest",
    "GenerationResult",
    "GenerationShortfallError",
]
