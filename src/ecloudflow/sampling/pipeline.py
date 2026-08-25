"""Low-level candidate service used by the public generation pipeline."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem

from ecloudflow.chemistry.decoder import BondDecodeProblem, ExactBondDecoder
from ecloudflow.chemistry.projector import ChemicalProjector
from ecloudflow.chemistry.reconstruct import reconstruct_rdkit_molecule
from ecloudflow.chemistry.standardize import standardize_molecule
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.masks import clamp_fragment
from ecloudflow.core.types import (
    ElectronField,
    FragmentCondition,
    GenerationCondition,
    MolecularState,
    PocketGraph,
)
from ecloudflow.data.features import (
    ligand_graph_from_molecule,
    pocket_graph_from_entity,
)
from ecloudflow.data.parsers import parse_ligand_sdf, parse_pocket_pdb
from ecloudflow.ecloud.pocket import PocketFieldBuilder
from ecloudflow.sampling.corrector import ScoreCorrector
from ecloudflow.sampling.profiles import SamplingProfile, get_profile
from ecloudflow.sampling.results import GenerationMode
from ecloudflow.sampling.solver import EulerSolver, HeunSolver, SamplingTrajectory

CandidateCallable = Callable[..., Any]


@dataclass
class SamplingPipeline:
    """Adapt model/sampler services to one normalized candidate interface.

    :param candidate_generator: Optional injected deterministic or learned
        service.  It receives keyword arguments ``pocket``, ``fragment``,
        ``mode``, ``profile``, ``seed``, ``generator``, and ``attempt``.
    :param model: Optional callable used when no explicit candidate generator
        is supplied.
    :param decoder: Exact graph decoder for MolecularState outputs.
    :param vocabulary: Ligand vocabulary used by reconstruction.
    :return: Candidate sampling adapter.
    :rtype: SamplingPipeline

    Production callers can inject a model-backed sampler while tests can use a
    deterministic sequence service.  The adapter never silently converts an
    invalid output into a fabricated molecule.
    """

    candidate_generator: CandidateCallable | None = None
    model: Any = None
    decoder: Any = None
    vocabulary: ChemicalVocabulary | None = None

    def __post_init__(self) -> None:
        """Fill the exact decoder lazily without importing optional tooling."""
        if self.decoder is None:
            self.decoder = ExactBondDecoder()
        if self.vocabulary is None:
            self.vocabulary = ChemicalVocabulary.default_ligand()
        if self.candidate_generator is not None and not callable(
            self.candidate_generator
        ):
            raise TypeError("candidate_generator must be callable.")

    def sample(
        self,
        *,
        pocket: Any,
        fragment: Any = None,
        fixed: Any = None,
        mode: GenerationMode = GenerationMode.DE_NOVO,
        profile: str | SamplingProfile = "balanced",
        seed: int = 2026,
        generator: torch.Generator | None = None,
        attempt: int = 1,
    ) -> Any:
        """Generate one raw candidate from an injected or learned service.

        :param pocket: Parsed pocket object or source path.
        :param fragment: Optional positioned fragment object/path.
        :param fixed: Optional exact :class:`FragmentCondition` used by the
            discrete decoder.  The public pipeline may pass a pre-built
            condition when fragment coordinates have already been parsed.
        :param mode: De novo or fragment-conditioned objective.
        :param profile: Named profile or resolved :class:`SamplingProfile`.
        :param seed: Deterministic per-attempt seed.
        :param generator: Optional caller-owned torch generator.
        :param attempt: One-based bounded attempt number.
        :return: RDKit molecule, SMILES, MolecularState, or trajectory.
        :rtype: object
        :raises RuntimeError: If no model/candidate service is configured.
        """
        resolved_profile = (
            profile if isinstance(profile, SamplingProfile) else get_profile(profile)
        )
        mode = mode if isinstance(mode, GenerationMode) else GenerationMode(mode)
        service = self.candidate_generator or self.model
        if service is None:
            raise RuntimeError("no candidate_generator or model is configured.")
        # ``ECloudFlowModel`` is a vector field, not a callable returning an
        # RDKit molecule.  Route it through the production state sampler so a
        # normal Lightning checkpoint has one unambiguous inference path.
        if self.candidate_generator is None and _is_ecloudflow_model(service):
            return _ModelSamplingService(service, self.decoder, self.vocabulary).sample(
                pocket=pocket,
                fragment=fragment,
                fixed=fixed,
                mode=mode,
                profile=resolved_profile,
                seed=seed,
                generator=generator,
                attempt=attempt,
            )
        kwargs = {
            "pocket": pocket,
            "fragment": fragment,
            "mode": mode,
            "profile": resolved_profile,
            "seed": seed,
            "generator": generator,
            "attempt": attempt,
        }
        output = _invoke_service(service, kwargs)
        return self._normalize_state_output(output, fixed=fixed)

    def _normalize_state_output(self, output: Any, *, fixed: Any = None) -> Any:
        """Convert a trajectory/state to an RDKit molecule at the decoder edge."""
        if isinstance(output, dict):
            for key in ("molecule", "mol", "smiles", "state", "trajectory"):
                if key in output:
                    normalized = self._normalize_state_output(output[key], fixed=fixed)
                    # Keep scalar metrics and provenance attached to a candidate;
                    # the public pipeline serializes these values into ranking
                    # tables instead of silently dropping them at this boundary.
                    return {
                        **output,
                        key: normalized,
                    }
        if isinstance(output, SamplingTrajectory):
            output = output.final
        if isinstance(output, MolecularState):
            decoded = self.decoder.decode(
                BondDecodeProblem(
                    state=output,
                    vocabulary=self.vocabulary,
                    fixed=fixed,
                    allowed_bond_mask=_fragment_allowed_bond_mask(output, fixed),
                )
            )
            molecule = reconstruct_rdkit_molecule(output, decoded, self.vocabulary)
            return _restore_global_pose(molecule, output.frame)
        return output


def _invoke_service(service: CandidateCallable, kwargs: dict[str, Any]) -> Any:
    """Call services with compatible keyword subsets without hiding failures."""
    try:
        signature = inspect.signature(service)
    except (TypeError, ValueError):
        return service(**kwargs)
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return service(**kwargs)
    accepted = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return service(**accepted)


def _is_ecloudflow_model(service: Any) -> bool:
    """Return whether ``service`` is the joint ECloudFlow neural backbone."""
    return service.__class__.__name__ == "ECloudFlowModel" and hasattr(
        service, "encode_pocket"
    )


class _ModelSamplingService:
    """Run a checkpoint-backed ECloudFlow model through constrained sampling.

    :param model: Loaded :class:`~ecloudflow.models.ECloudFlowModel`.
    :param decoder: Exact sparse valence/connectivity decoder.
    :param vocabulary: Model ligand vocabulary.
    :return: Reusable inference service.
    :rtype: _ModelSamplingService

    The service owns no global RNG.  Every request creates a device-matched
    generator from its attempt seed, builds a centered pocket state, integrates
    the joint flow, clamps fragment values after every substep, and performs
    exact graph decoding before returning an RDKit molecule.  A failed decode
    is raised to the public bounded pipeline and is recorded as a failed
    attempt; it is never replaced by a fixture molecule.
    """

    def __init__(self, model: Any, decoder: Any, vocabulary: ChemicalVocabulary) -> None:
        self.model = model
        self.decoder = decoder
        self.vocabulary = vocabulary
        self.projector = ChemicalProjector(vocabulary)

    def sample(
        self,
        *,
        pocket: Any,
        fragment: Any,
        fixed: FragmentCondition | None,
        mode: GenerationMode,
        profile: SamplingProfile,
        seed: int,
        generator: torch.Generator | None,
        attempt: int,
    ) -> dict[str, Any]:
        """Generate one chemically decoded candidate in the pocket frame.

        :param pocket: PDB path or canonical :class:`PocketGraph`.
        :param fragment: Original fragment source, retained for diagnostics.
        :param fixed: Exact full-state fragment condition, when applicable.
        :param mode: Generation objective.
        :param profile: Resolved solver/corrector profile.
        :param seed: Per-attempt deterministic seed.
        :param generator: Ignored when its device differs from the model; a
            device-local generator is always created to preserve solver rules.
        :param attempt: Bounded attempt number for provenance.
        :return: Mapping containing a sanitized molecule and sampling metadata.
        :rtype: dict[str, Any]
        :raises RuntimeError: If pocket conversion, model prediction, or exact
            graph decoding fails.
        """
        del generator
        device, dtype = _module_device_dtype(self.model)
        pocket_graph = _coerce_pocket_graph(pocket, device=device, dtype=dtype)
        pocket_field = _move_electron_field(
            PocketFieldBuilder.default().build(pocket_graph),
            device=device,
            dtype=dtype,
        )
        if (
            fixed is None
            and mode is not GenerationMode.DE_NOVO
            and _fragment_sources_are_supported(fragment)
        ):
            fixed = build_fragment_condition(
                pocket_graph,
                fragment,
                extra_atoms=_default_free_atoms(mode),
                device=device,
                dtype=dtype,
                seed=seed,
                task_id=mode.value,
                electron_latent_dim=int(getattr(self.model, "electron_latent_dim", 48)),
            )
        elif fixed is not None:
            fixed = _move_fragment_condition(fixed, device=device, dtype=dtype)
        state, condition = _initial_state(
            pocket_graph,
            pocket_field=pocket_field,
            fixed=fixed,
            model=self.model,
            mode=mode,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        rng = torch.Generator(device=device)
        rng.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
        hooks = self._hooks(fixed)
        vector_field = self._vector_field(condition)
        solver = (
            EulerSolver(profile.num_steps)
            if profile.solver == "euler"
            else HeunSolver(profile.num_steps)
        )
        trajectory = solver.integrate(state, vector_field, hooks, rng)
        corrected = trajectory
        if profile.corrector_steps:
            score = self._score_field(condition)
            corrected = ScoreCorrector(steps=profile.corrector_steps).correct(
                trajectory.final, score, hooks, rng
            )
        final_state = corrected.final
        problem = BondDecodeProblem(
            state=final_state,
            vocabulary=self.vocabulary,
            fixed=fixed,
            allowed_bond_mask=_fragment_allowed_bond_mask(final_state, fixed),
            require_connected=True,
        )
        decoded = self.decoder.decode(problem)
        if not decoded.feasible:
            raise RuntimeError(f"exact graph decoding failed: {decoded.reason or decoded.status.value}")
        molecule = reconstruct_rdkit_molecule(final_state, decoded, self.vocabulary)
        molecule = _restore_global_pose(molecule, final_state.frame)
        return {
            "molecule": molecule,
            "sampling_nfe": trajectory.nfe + corrected.nfe,
            "sampling_attempt": attempt,
            "sampling_profile": profile.name,
            "sampling_solver": profile.solver,
            "sampling_device": str(device),
        }

    def _hooks(self, fixed: FragmentCondition | None) -> tuple[Any, ...]:
        """Return exact fragment and chemical projection hooks in order."""
        if fixed is None:
            return (lambda state, *_: self.projector.project(state).state,)
        return (
            lambda state, *_: clamp_fragment(state, fixed),
            lambda state, *_: self.projector.project(state, fixed).state,
            lambda state, *_: clamp_fragment(state, fixed),
        )

    def _vector_field(self, condition: GenerationCondition) -> Any:
        """Create the model endpoint-to-flow adapter used by Euler/Heun."""
        def field(state: MolecularState, time: torch.Tensor) -> MolecularState:
            with torch.no_grad():
                prediction = self.model(state, time.reshape(1), condition)
            return state.replace(
                positions=prediction.position_velocity,
                atom_logits=prediction.atom_logits - state.atom_logits,
                charge_logits=prediction.charge_logits - state.charge_logits,
                bond_logits=prediction.bond_logits - state.bond_logits,
                electron_latent=prediction.electron_velocity,
            )

        return field

    def _score_field(self, condition: GenerationCondition) -> Any:
        """Create a score adapter for the terminal Langevin corrector."""
        def score(state: MolecularState, time: torch.Tensor) -> MolecularState:
            with torch.no_grad():
                prediction = self.model(state, time.reshape(1), condition)
            return state.replace(
                positions=prediction.position_score,
                atom_logits=prediction.atom_logits - state.atom_logits,
                charge_logits=prediction.charge_logits - state.charge_logits,
                bond_logits=prediction.bond_logits - state.bond_logits,
                electron_latent=prediction.electron_score,
            )

        return score


def build_fragment_condition(
    pocket: Any,
    fragment: Any,
    *,
    extra_atoms: int = 8,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 2026,
    task_id: str = "fragment",
    electron_latent_dim: int = 48,
) -> FragmentCondition:
    """Build a full-state exact condition from a positioned fragment file.

    :param pocket: Canonical pocket graph or PDB path defining the binding frame.
    :param fragment: SDF path, 3D RDKit molecule, or a non-empty sequence of
        such sources. Multiple sources are retained as separate fixed
        components so ``link`` and ``merge`` can add only cross-component
        bonds.
    :param extra_atoms: Number of editable placeholder atoms appended after the
        fixed fragment.  Their identities, charges, bonds, and positions remain
        generative; fixed atom indices and values never change.
    :param device: Destination tensor device.
    :param dtype: Floating tensor dtype for the state and frame.
    :param seed: Deterministic seed for editable placeholder coordinates.
    :param task_id: Stable grow/link/replace/merge condition identifier.
    :param electron_latent_dim: Packed ligand electron-channel width expected by
        the loaded joint backbone.
    :return: A :class:`FragmentCondition` whose reference has a sparse complete
        candidate graph and exact fixed atom/bond/coordinate masks.
    :rtype: FragmentCondition
    :raises ValueError: If the source fragment or pocket is malformed.

    Each source's original atom ordering is retained. The first ``F`` state
    nodes are fixed one-to-one to the concatenated source atoms, and all
    within-component source bonds and non-bonds are fixed by canonical
    halfedge identity. Cross-component fixed pairs remain editable in link and
    merge tasks. Additional nodes are explicit editable capacity, allowing
    grow/link/replace/merge inference while keeping supplied coordinates in the
    pocket's centered binding frame.
    """
    if isinstance(fragment, FragmentCondition):
        return fragment
    pocket_graph = _coerce_pocket_graph(pocket, device=device, dtype=dtype)
    sources = _normalize_fragment_sources(fragment)
    molecules = tuple(_load_fragment_molecule(source) for source in sources)
    if not molecules:
        raise ValueError("fragment must contain at least one molecule")
    if any(
        molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D()
        for molecule in molecules
    ):
        raise ValueError("every fragment must contain exactly one 3D conformer")
    frame = pocket_graph.frame
    if frame is None:
        frame = CoordinateFrame.from_pocket(pocket_graph.positions)
    frame = _move_frame(frame, device=device or pocket_graph.positions.device, dtype=dtype)
    cpu_frame = _move_frame(frame, device=torch.device("cpu"), dtype=torch.float32)
    graphs = tuple(
        _move_ligand_graph(
            ligand_graph_from_molecule(molecule, cpu_frame),
            device=pocket_graph.positions.device,
            dtype=dtype,
        )
        for molecule in molecules
    )
    fixed_positions = torch.cat([graph.positions for graph in graphs], dim=0)
    fixed_atom_types = torch.cat([graph.atom_types for graph in graphs], dim=0)
    fixed_charges = torch.cat([graph.formal_charges for graph in graphs], dim=0)
    fixed_edges: list[torch.Tensor] = []
    fixed_bond_types: list[torch.Tensor] = []
    component_chunks: list[torch.Tensor] = []
    attachment_chunks: list[torch.Tensor] = []
    atom_offset = 0
    component_offset = 0
    for molecule, graph in zip(molecules, graphs, strict=True):
        if graph.halfedge_index.shape[1]:
            fixed_edges.append(graph.halfedge_index + atom_offset)
            fixed_bond_types.append(graph.bond_types)
        components = Chem.GetMolFrags(molecule, asMols=False, sanitizeFrags=False)
        labels = torch.empty(
            graph.positions.shape[0], dtype=torch.long, device=graph.positions.device
        )
        for local_component, atom_indices in enumerate(components):
            labels[list(atom_indices)] = component_offset + local_component
        component_chunks.append(labels)
        attachment_chunks.append(
            _infer_attachment_mask(molecule).to(device=graph.positions.device)
        )
        atom_offset += graph.positions.shape[0]
        component_offset += max(1, len(components))
    fixed_count = fixed_positions.shape[0]
    fixed_edge_index = (
        torch.cat(fixed_edges, dim=1)
        if fixed_edges
        else torch.empty((2, 0), dtype=torch.long, device=fixed_positions.device)
    )
    fixed_bond_values = (
        torch.cat(fixed_bond_types)
        if fixed_bond_types
        else torch.empty(0, dtype=torch.long, device=fixed_positions.device)
    )
    free_count = max(0, int(extra_atoms))
    total = fixed_count + free_count
    positions = fixed_positions.new_zeros((total, 3))
    positions[:fixed_count] = fixed_positions
    if free_count:
        minimum = pocket_graph.positions.min(dim=0).values - 1.0
        maximum = pocket_graph.positions.max(dim=0).values + 1.0
        rng = torch.Generator(device=positions.device)
        rng.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
        positions[fixed_count:] = minimum + torch.rand(
            (free_count, 3),
            generator=rng,
            device=positions.device,
            dtype=positions.dtype,
        ) * (maximum - minimum).clamp_min(1.0)
    edges = torch.triu_indices(total, total, offset=1, device=positions.device)
    atom_channels = len(ChemicalVocabulary.default_ligand().atom_symbols)
    charge_channels = len(ChemicalVocabulary.default_ligand().formal_charges)
    bond_channels = len(ChemicalVocabulary.default_ligand().bond_classes)
    atom_logits = positions.new_full((total, atom_channels), -8.0)
    charge_logits = positions.new_full((total, charge_channels), -8.0)
    for index, (atom_type, charge) in enumerate(
        zip(fixed_atom_types.tolist(), fixed_charges.tolist(), strict=True)
    ):
        atom_logits[index, atom_type] = 8.0
        charge_logits[index, ChemicalVocabulary.default_ligand().charge_index(int(charge))] = 8.0
    if free_count:
        atom_logits[fixed_count:] = 0.0
        charge_logits[fixed_count:] = 0.0
    bond_logits = positions.new_full((edges.shape[1], bond_channels), -8.0)
    bond_logits[:, 0] = 8.0
    edge_lookup = {tuple(pair): index for index, pair in enumerate(edges.t().tolist())}
    for edge, bond_type in zip(
        fixed_edge_index.t().tolist(), fixed_bond_values.tolist(), strict=True
    ):
        row = edge_lookup[tuple(edge)]
        bond_logits[row, :] = -8.0
        bond_logits[row, int(bond_type)] = 8.0
    component_ids: torch.Tensor | None = None
    if component_offset > 1:
        component_ids = torch.cat(
            [
                torch.cat(component_chunks),
                torch.full(
                    (free_count,),
                    component_offset,
                    dtype=torch.long,
                    device=positions.device,
                ),
            ]
        )
    state = MolecularState(
        positions=positions,
        atom_logits=atom_logits,
        charge_logits=charge_logits,
        halfedge_index=edges,
        bond_logits=bond_logits,
        electron_latent=positions.new_zeros((total, electron_latent_dim)),
        node_batch=torch.zeros(total, dtype=torch.long, device=positions.device),
        halfedge_batch=torch.zeros(edges.shape[1], dtype=torch.long, device=positions.device),
        frame=frame,
    )
    fixed_mask = torch.zeros(total, dtype=torch.bool, device=positions.device)
    fixed_mask[:fixed_count] = True
    attachment = torch.zeros(total, dtype=torch.bool, device=positions.device)
    attachment[:fixed_count] = torch.cat(attachment_chunks)
    return FragmentCondition.from_atom_mask(
        fixed_mask,
        state,
        attachment_mask=attachment,
        component_ids=component_ids,
        task_id=task_id,
    )


def _normalize_fragment_sources(fragment: Any) -> tuple[Any, ...]:
    """Normalize one or multiple fragment source values without splitting Mols."""
    if isinstance(fragment, (str, Path, Chem.Mol)):
        return (fragment,)
    if isinstance(fragment, Sequence) and not isinstance(fragment, (bytes, bytearray)):
        values = tuple(fragment)
        if not values:
            raise ValueError("fragment sequence must not be empty")
        if not all(isinstance(value, (str, Path, Chem.Mol)) for value in values):
            raise TypeError("fragment sequence values must be SDF paths or RDKit Mols")
        return values
    raise TypeError("fragment must be an SDF path, RDKit Mol, or sequence of them")


def _load_fragment_molecule(source: Any) -> Chem.Mol:
    """Parse and standardize one fragment source while preserving atom order."""
    if isinstance(source, (str, Path)):
        try:
            return parse_ligand_sdf(source)
        except ValueError:
            return _parse_fragment_without_hydrogens(Path(source))
    if isinstance(source, Chem.Mol):
        molecule = Chem.Mol(source)
        if molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
            raise ValueError("fragment must contain exactly one 3D conformer")
        try:
            return standardize_molecule(molecule)
        except ValueError:
            # Explicit hydrogens are not ligand model nodes. Remove them only
            # after the first strict attempt so unsupported heavy-atom chemistry
            # still fails with the original vocabulary/standardization error.
            reduced = Chem.RemoveHs(molecule, sanitize=False)
            return standardize_molecule(reduced)
    raise TypeError("fragment source must be an SDF path or RDKit Mol")


def _infer_attachment_mask(molecule: Chem.Mol) -> torch.Tensor:
    """Derive a conservative fixed-atom attachment declaration from an SDF.

    :param molecule: Sanitized, hydrogen-suppressed fragment molecule with one
        conformer. Atom properties named ``ecloudflow_attachment``,
        ``attachment_site``, or ``attachment`` are treated as an explicit
        boolean declaration when at least one such property is present. If no
        declaration is present, only atoms with at least one implicit/explicit
        hydrogen are enabled; saturated atoms and heteroatoms without available
        valence remain forbidden. The mask is indexed in the source SDF atom
        order and never includes generated placeholder nodes.
    :return: Boolean attachment mask with shape ``[F]`` on the CPU.
    :rtype: torch.Tensor
    :raises ValueError: If an explicit atom property is present but cannot be
        parsed as a boolean value.

    This fail-closed policy prevents a complete candidate graph from turning
    every fixed atom into a linker site. Users can mark a chemically valid
    ring or scaffold position explicitly in the SDF when implicit hydrogens do
    not express the intended medicinal-chemistry cut. A source-level
    declaration is applied as an all-or-nothing mask so an omitted property
    cannot accidentally broaden a partially annotated fragment.
    """
    property_names = (
        "ecloudflow_attachment",
        "attachment_site",
        "attachment",
    )
    declared = [
        any(atom.HasProp(name) for name in property_names)
        for atom in molecule.GetAtoms()
    ]
    if any(declared):
        values: list[bool] = []
        for atom in molecule.GetAtoms():
            present = next(
                (name for name in property_names if atom.HasProp(name)), None
            )
            if present is None:
                values.append(False)
                continue
            raw = atom.GetProp(present).strip().lower()
            if raw in {"1", "true", "yes", "y", "on"}:
                values.append(True)
            elif raw in {"0", "false", "no", "n", "off"}:
                values.append(False)
            else:
                raise ValueError(
                    f"fragment atom {atom.GetIdx()} has invalid {present} value {raw!r}"
                )
        return torch.tensor(values, dtype=torch.bool)

    values = []
    for atom in molecule.GetAtoms():
        # ``GetTotalNumHs`` includes implicit and explicit hydrogens while
        # remaining compatible with SDFs that omit an explicit H count.
        values.append(int(atom.GetTotalNumHs()) > 0)
    return torch.tensor(values, dtype=torch.bool)


def predict_atom_count(
    model: Any,
    pocket: Any,
    *,
    fixed: FragmentCondition | None = None,
    pocket_field: ElectronField | None = None,
) -> int:
    """Predict a deterministic ligand node count from pocket conditioning.

    :param model: Loaded ECloudFlow backbone exposing canonical channel widths.
    :param pocket: Pocket graph, PDB path, or parsed entity defining the frame.
    :param fixed: Optional fixed-only fragment condition.  Its reference state
        supplies the fragment chemistry while count probabilities below the
        fixed atom count remain masked by the model.
    :param pocket_field: Optional precomputed physical pocket field. When absent,
        the deterministic default field is built from the canonical graph.
    :return: Highest-probability positive atom count, bounded by the model's
        inclusive ``max_atoms`` category.
    :rtype: int
    :raises RuntimeError: If the model returns malformed or non-finite count logits.

    Count inference uses a zero-node de novo state or the exact fixed fragment
    reference at path time zero.  Argmax, rather than stochastic resampling,
    makes repeated bounded attempts share the learned size decision while their
    cavity priors still differ by attempt seed.  No global RNG, gradient, model
    mutation, or CPU/device transfer occurs.
    """
    device, dtype = _module_device_dtype(model)
    pocket = _coerce_pocket_graph(pocket, device=device, dtype=dtype)
    pocket_field = (
        _move_electron_field(pocket_field, device=device, dtype=dtype)
        if pocket_field is not None
        else _move_electron_field(
            PocketFieldBuilder.default().build(pocket),
            device=device,
            dtype=dtype,
        )
    )
    if fixed is None:
        state = MolecularState(
            positions=torch.empty((0, 3), device=device, dtype=dtype),
            atom_logits=torch.empty(
                (0, int(model.atom_classes)), device=device, dtype=dtype
            ),
            charge_logits=torch.empty(
                (0, int(model.charge_classes)), device=device, dtype=dtype
            ),
            halfedge_index=torch.empty((2, 0), device=device, dtype=torch.long),
            bond_logits=torch.empty(
                (0, int(model.bond_classes)), device=device, dtype=dtype
            ),
            electron_latent=torch.empty(
                (0, int(model.electron_latent_dim)), device=device, dtype=dtype
            ),
            node_batch=torch.empty(0, device=device, dtype=torch.long),
            halfedge_batch=torch.empty(0, device=device, dtype=torch.long),
            frame=pocket.frame,
        )
    else:
        fixed = _move_fragment_condition(fixed, device=device, dtype=dtype)
        state = fixed.reference
    condition = GenerationCondition(
        pocket=pocket,
        pocket_field=pocket_field,
        fragment=fixed,
    )
    with torch.no_grad():
        prediction = model(
            state,
            torch.zeros(1, device=device, dtype=dtype),
            condition,
        )
    minimum = 1 if fixed is None else int(fixed.fixed_atom_mask.sum())
    minimum = max(1, minimum)
    if minimum > int(model.max_atoms):
        raise RuntimeError("fixed fragment exceeds the checkpoint maximum atom count")
    logits = prediction.count_logits
    # The count head intentionally masks categories below the admissible lower
    # bound with ``-inf``. Only the categories that can actually be selected
    # must be finite; rejecting the masked prefix makes every fixed-fragment
    # checkpoint request fail before sampling starts.
    if logits.shape != (1, int(model.max_atoms) + 1) or not bool(
        torch.isfinite(logits[:, minimum:]).all()
    ):
        raise RuntimeError("model returned malformed atom-count logits")
    return int(logits[0, minimum:].argmax()) + minimum


def _parse_fragment_without_hydrogens(path: Path) -> Chem.Mol:
    """Read one SDF and remove explicit hydrogens before vocabulary checks."""
    records = list(Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False))
    if len(records) != 1 or records[0] is None:
        raise ValueError("fragment SDF must contain exactly one valid molecule")
    molecule = Chem.RemoveHs(records[0], sanitize=False)
    return standardize_molecule(molecule)


def _default_free_atoms(mode: GenerationMode) -> int:
    """Return bounded editable capacity for one fragment objective."""
    return {GenerationMode.GROW: 8, GenerationMode.LINK: 10, GenerationMode.REPLACE: 8, GenerationMode.MERGE: 10}.get(mode, 8)


def _fragment_allowed_bond_mask(
    state: MolecularState,
    fixed: FragmentCondition | None,
) -> torch.Tensor | None:
    """Build the final sparse topology mask for fixed-fragment decoding.

    :param state: Candidate state supplying canonical unordered halfedges.
    :param fixed: Optional exact fragment contract.  ``None`` leaves all
        candidate edges available to the decoder.
    :return: Boolean ``[E]`` mask on ``state``'s device, or ``None`` for a
        de-novo state.
    :rtype: torch.Tensor | None
    :raises ValueError: If the fragment masks do not match the candidate
        topology or device.

    Fixed internal edges are always retained because their classes are already
    immutable in :class:`BondDecodeProblem`.  Free-to-free edges remain
    generative.  A fixed-to-free edge is retained only when its fixed endpoint
    is explicitly marked as an attachment site.  This mask is intentionally
    separate from the projector's valence-capacity mask: saturated edges may
    still be needed by the final discrete optimizer, while forbidden fragment
    crossings must be forced to ``none``.
    """
    if fixed is None:
        return None
    reference = fixed.reference
    if reference.positions.shape != state.positions.shape or reference.halfedge_index.shape != state.halfedge_index.shape:
        raise ValueError("fixed reference topology does not match state.")
    if not torch.equal(reference.halfedge_index, state.halfedge_index):
        raise ValueError("fixed reference halfedges do not match state.")
    if fixed.fixed_atom_mask.device != state.positions.device:
        raise ValueError("fixed fragment masks must be on the state device.")
    source, target = state.halfedge_index
    fixed_source = fixed.fixed_atom_mask[source]
    fixed_target = fixed.fixed_atom_mask[target]
    crossing = fixed_source ^ fixed_target
    attachment_crossing = (
        (fixed.attachment_mask[source] & fixed_source)
        | (fixed.attachment_mask[target] & fixed_target)
    )
    return (~crossing | attachment_crossing).to(device=state.positions.device)


def _fragment_is_file(value: Any) -> bool:
    """Return whether a fragment argument names an existing file."""
    return isinstance(value, (str, Path)) and Path(value).is_file()


def _fragment_sources_are_supported(value: Any) -> bool:
    """Return whether a source is an existing path or positioned RDKit molecule."""
    if isinstance(value, Chem.Mol):
        return True
    if _fragment_is_file(value):
        return True
    return isinstance(value, (list, tuple)) and bool(value) and all(
        isinstance(item, Chem.Mol) or _fragment_is_file(item) for item in value
    )


def _module_device_dtype(module: Any) -> tuple[torch.device, torch.dtype]:
    """Read a model's parameter placement, falling back to CPU float32."""
    try:
        parameter = next(module.parameters())
    except (AttributeError, StopIteration):
        return torch.device("cpu"), torch.float32
    dtype = parameter.dtype if parameter.dtype.is_floating_point else torch.float32
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return parameter.device, dtype


def _coerce_pocket_graph(
    pocket: Any, *, device: torch.device | str | None, dtype: torch.dtype
) -> PocketGraph:
    """Parse one pocket source and place its graph in the requested device."""
    if isinstance(pocket, PocketGraph):
        return _move_pocket_graph(pocket, device=device, dtype=dtype)
    if isinstance(pocket, (str, Path)):
        structure = parse_pocket_pdb(pocket)
        atoms = list(structure.get_atoms())
        global_positions = torch.stack(
            [torch.as_tensor(atom.get_coord(), dtype=dtype) for atom in atoms]
        )
        frame = CoordinateFrame.from_pocket(global_positions)
        graph = pocket_graph_from_entity(structure, frame)
        return _move_pocket_graph(graph, device=device, dtype=dtype)
    if hasattr(pocket, "get_atoms"):
        atoms = list(pocket.get_atoms())
        global_positions = torch.stack(
            [torch.as_tensor(atom.get_coord(), dtype=dtype) for atom in atoms]
        )
        frame = CoordinateFrame.from_pocket(global_positions)
        return _move_pocket_graph(pocket_graph_from_entity(pocket, frame), device=device, dtype=dtype)
    raise ValueError("pocket must be a PDB path, Biopython entity, or PocketGraph")


def _move_frame(frame: CoordinateFrame, *, device: torch.device | str, dtype: torch.dtype) -> CoordinateFrame:
    """Copy a coordinate frame without changing its rigid transform values."""
    target = torch.device(device)
    return CoordinateFrame(origin=frame.origin.to(device=target, dtype=dtype), rotation=frame.rotation.to(device=target, dtype=dtype))


def _move_pocket_graph(graph: PocketGraph, *, device: torch.device | str | None, dtype: torch.dtype) -> PocketGraph:
    """Move a pocket graph and its frame atomically."""
    target = torch.device(device) if device is not None else graph.positions.device
    frame = _move_frame(graph.frame, device=target, dtype=dtype) if graph.frame is not None else None
    return PocketGraph(
        positions=graph.positions.to(device=target, dtype=dtype),
        features=graph.features.to(device=target, dtype=dtype),
        batch=graph.batch.to(device=target),
        atom_numbers=graph.atom_numbers.to(device=target) if graph.atom_numbers is not None else None,
        frame=frame,
    )


def _move_electron_field(
    field: ElectronField,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> ElectronField:
    """Move a physical field and its frame without changing channel semantics.

    :param field: Validated electron field in a declared pocket frame.
    :param device: Destination device selected from the checkpoint model.
    :param dtype: Floating inference dtype shared with pocket coordinates.
    :return: New field preserving positions, values, masks, batches, and channels.
    :rtype: ElectronField

    The physical channel values are cast only to the model inference dtype;
    masks and batch labels retain boolean/long dtypes. No grid is regenerated,
    normalized, detached from provenance, or transformed into a different
    coordinate convention.
    """
    target = torch.device(device)
    frame = (
        _move_frame(field.frame, device=target, dtype=dtype)
        if field.frame is not None
        else None
    )
    return ElectronField(
        positions=field.positions.to(device=target, dtype=dtype),
        values=field.values.to(device=target, dtype=dtype),
        mask=field.mask.to(device=target),
        batch=field.batch.to(device=target),
        channel_names=field.channel_names,
        frame=frame,
    )


def _move_ligand_graph(graph: Any, *, device: torch.device, dtype: torch.dtype) -> Any:
    """Move a canonical ligand graph's tensors to one device and dtype."""
    return type(graph)(
        positions=graph.positions.to(device=device, dtype=dtype),
        atom_types=graph.atom_types.to(device=device),
        formal_charges=graph.formal_charges.to(device=device),
        halfedge_index=graph.halfedge_index.to(device=device),
        bond_types=graph.bond_types.to(device=device),
        batch=graph.batch.to(device=device),
    )


def _move_state(state: MolecularState, *, device: torch.device, dtype: torch.dtype) -> MolecularState:
    """Move every state tensor and frame while retaining canonical topology."""
    frame = _move_frame(state.frame, device=device, dtype=dtype) if state.frame is not None else None
    return state.replace(
        positions=state.positions.to(device=device, dtype=dtype),
        atom_logits=state.atom_logits.to(device=device, dtype=dtype),
        charge_logits=state.charge_logits.to(device=device, dtype=dtype),
        halfedge_index=state.halfedge_index.to(device=device),
        bond_logits=state.bond_logits.to(device=device, dtype=dtype),
        electron_latent=state.electron_latent.to(device=device, dtype=dtype),
        node_batch=state.node_batch.to(device=device),
        halfedge_batch=state.halfedge_batch.to(device=device),
        frame=frame,
    )


def _move_fragment_condition(condition: FragmentCondition, *, device: torch.device, dtype: torch.dtype) -> FragmentCondition:
    """Move a fragment reference and masks without relaxing exact values."""
    reference = _move_state(condition.reference, device=device, dtype=dtype)
    return FragmentCondition(
        reference=reference,
        fixed_atom_mask=condition.fixed_atom_mask.to(device=device),
        fixed_bond_mask=condition.fixed_bond_mask.to(device=device),
        fixed_coord_mask=condition.fixed_coord_mask.to(device=device),
        attachment_mask=condition.attachment_mask.to(device=device),
        component_ids=condition.component_ids.to(device=device) if condition.component_ids is not None else None,
        task_id=condition.task_id,
    )


def _initial_state(
    pocket: PocketGraph,
    *,
    pocket_field: ElectronField,
    fixed: FragmentCondition | None,
    model: Any,
    mode: GenerationMode,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[MolecularState, GenerationCondition]:
    """Create a cavity-supported state with a chemically conservative prior."""
    del mode
    if fixed is not None:
        reference = fixed.reference
        state = reference
        rng = torch.Generator(device=device).manual_seed(int(seed) & 0x7FFFFFFF)
        free = ~fixed.fixed_atom_mask
        if bool(free.any()):
            lower = pocket.positions.min(dim=0).values - 1.0
            upper = pocket.positions.max(dim=0).values + 1.0
            positions = state.positions.clone()
            positions[free] = lower + torch.rand((int(free.sum()), 3), generator=rng, device=device, dtype=dtype) * (upper - lower).clamp_min(1.0)
            state = state.replace(positions=positions)
        state = clamp_fragment(state, fixed)
        return state, GenerationCondition(
            pocket=pocket,
            pocket_field=pocket_field,
            fragment=fixed,
        )
    node_count = predict_atom_count(model, pocket, pocket_field=pocket_field)
    edges = torch.triu_indices(node_count, node_count, offset=1, device=device)
    positions = pocket.positions.new_zeros((node_count, 3))
    rng = torch.Generator(device=device).manual_seed(int(seed) & 0x7FFFFFFF)
    lower = pocket.positions.min(dim=0).values - 1.0
    upper = pocket.positions.max(dim=0).values + 1.0
    positions[:] = lower + torch.rand((node_count, 3), generator=rng, device=device, dtype=dtype) * (upper - lower).clamp_min(1.0)
    vocab = ChemicalVocabulary.default_ligand()
    atom_logits = positions.new_full((node_count, len(vocab.atom_symbols)), -6.0)
    atom_logits[:, vocab.atom_index("C")] = 6.0
    charge_logits = positions.new_full((node_count, len(vocab.formal_charges)), -6.0)
    charge_logits[:, vocab.charge_index(0)] = 6.0
    bond_logits = positions.new_full((edges.shape[1], len(vocab.bond_classes)), -8.0)
    bond_logits[:, vocab.bond_index("none")] = 8.0
    for index in range(node_count - 1):
        edge = (index, index + 1)
        row = (edges[0] == edge[0]) & (edges[1] == edge[1])
        bond_logits[row, vocab.bond_index("none")] = -8.0
        bond_logits[row, vocab.bond_index("single")] = 8.0
    state = MolecularState(
        positions=positions,
        atom_logits=atom_logits,
        charge_logits=charge_logits,
        halfedge_index=edges,
        bond_logits=bond_logits,
        electron_latent=positions.new_zeros(
            (node_count, int(getattr(model, "electron_latent_dim", 48)))
        ),
        node_batch=torch.zeros(node_count, dtype=torch.long, device=device),
        halfedge_batch=torch.zeros(edges.shape[1], dtype=torch.long, device=device),
        frame=_move_frame(pocket.frame, device=device, dtype=dtype) if pocket.frame is not None else None,
    )
    return state, GenerationCondition(pocket=pocket, pocket_field=pocket_field)


def _validate_fragment_molecule(molecule: Chem.Mol, condition: FragmentCondition) -> None:
    """Verify fixed atom identity, charge, internal bonds, and pose exactly."""
    fixed_indices = torch.nonzero(condition.fixed_atom_mask, as_tuple=False).flatten().tolist()
    if molecule.GetNumAtoms() < len(fixed_indices):
        raise RuntimeError("decoded molecule lost fixed fragment atoms")
    vocab = ChemicalVocabulary.default_ligand()
    reference = condition.reference
    atom_indices = reference.atom_logits.argmax(dim=-1)
    charge_indices = reference.charge_logits.argmax(dim=-1)
    for index in fixed_indices:
        atom = molecule.GetAtomWithIdx(index)
        if atom.GetSymbol() != vocab.atom_symbol(int(atom_indices[index])):
            raise RuntimeError(f"fixed atom identity changed at index {index}")
        if atom.GetFormalCharge() != int(vocab.formal_charges[int(charge_indices[index])]):
            raise RuntimeError(f"fixed atom formal charge changed at index {index}")
    conformer = molecule.GetConformer() if molecule.GetNumConformers() else None
    if conformer is not None:
        observed = torch.tensor(conformer.GetPositions(), dtype=reference.positions.dtype)
        if not torch.allclose(observed[condition.fixed_coord_mask.cpu()], reference.positions.cpu()[condition.fixed_coord_mask.cpu()], atol=1.0e-5, rtol=0.0):
            raise RuntimeError("fixed fragment coordinates changed")
    source, target = reference.halfedge_index
    bond_names = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    bond_orders = vocab.bond_orders
    for edge_index in torch.nonzero(condition.fixed_bond_mask, as_tuple=False).flatten().tolist():
        first, second = int(source[edge_index]), int(target[edge_index])
        expected = round(float(bond_orders[int(reference.bond_logits[edge_index].argmax())]))
        bond = molecule.GetBondBetweenAtoms(first, second)
        if expected == 0 and bond is None:
            continue
        if bond is None or expected == 0 or bond.GetBondType() != bond_names.get(expected):
            raise RuntimeError(f"fixed internal bond changed for ({first}, {second})")


def _restore_global_pose(
    molecule: Chem.Mol, frame: CoordinateFrame | None
) -> Chem.Mol:
    """Transform a decoded local conformer back to the input pocket frame."""
    if frame is None or molecule.GetNumConformers() != 1:
        return molecule
    restored = Chem.Mol(molecule)
    conformer = restored.GetConformer()
    local = torch.tensor(conformer.GetPositions(), dtype=frame.origin.dtype)
    global_positions = frame.to_global(local.to(device=frame.origin.device)).cpu()
    for index, point in enumerate(global_positions.tolist()):
        conformer.SetAtomPosition(index, tuple(float(value) for value in point))
    return restored


__all__ = ["SamplingPipeline"]
