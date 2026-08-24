"""Exact and transparent bond-graph decoding for sampled ligand states.

The trajectory representation is sparse and probabilistic.  This module is
the deliberately discrete boundary: atom and charge channels are selected,
unordered halfedges receive one Kekule bond class, and a bounded CP-SAT model
enforces valence and single-component connectivity.  RDKit reconstruction is
kept in :mod:`ecloudflow.chemistry.reconstruct` so that solver diagnostics are
available even when sanitization or an optional solver is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch

from ecloudflow.chemistry.valence import ValenceTable
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core.types import FragmentCondition, MolecularState


class DecodeStatus(str, Enum):
    """Stable status values emitted by exact and fallback decoders."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    FALLBACK_FEASIBLE = "fallback_feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


@dataclass(frozen=True)
class BondDecodeProblem:
    """Describe one final graph-decoding problem.

    :param state: Final flattened molecular state.  Atom, charge, and bond
        tensors are interpreted as logits (probabilities are also accepted).
    :param vocabulary: Ligand vocabulary defining categorical channel order.
    :param valence_table: Optional charge-conditioned maximum valence table.
        The vetted vocabulary-aligned default is used when omitted.
    :param fixed: Optional fragment contract.  Fixed atom/charge assignments,
        coordinates, and internal bond classes are taken from its reference.
    :param timeout_seconds: Positive CP-SAT wall-time bound.  A feasible result
        found before the bound is returned with ``timed_out=False``.
    :param allowed_bond_mask: Optional boolean ``[E]`` mask.  False mutable
        edges are forced to ``none``; fixed internal edges remain immutable.
    :param require_connected: Require all decoded atoms to form one component.
    :param atom_indices: Optional explicit ``[N]`` atom-channel assignments.
    :param charge_indices: Optional explicit ``[N]`` charge-channel
        assignments.
    :return: Validated immutable decode problem.
    :rtype: BondDecodeProblem
    :raises ValueError: If dimensions, devices, masks, or limits are invalid.

    The decoder never creates a dense ``[N, N]`` tensor.  Dense bond matrices
    are a reporting concern and can be materialized from the returned sparse
    halfedge orders by a caller that explicitly needs them.
    """

    state: MolecularState
    vocabulary: ChemicalVocabulary
    valence_table: ValenceTable | None = None
    fixed: FragmentCondition | None = None
    timeout_seconds: float = 2.0
    allowed_bond_mask: torch.Tensor | None = None
    require_connected: bool = True
    atom_indices: torch.Tensor | None = None
    charge_indices: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate the problem boundary without changing model tensors."""
        if self.vocabulary.domain != "ligand":
            raise ValueError("bond decoding requires a ligand vocabulary.")
        if not isinstance(self.require_connected, bool):
            raise TypeError("require_connected must be boolean.")
        if (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive.")
        if self.state.atom_logits.shape[1] != len(self.vocabulary.atom_symbols):
            raise ValueError("state atom channels do not match the vocabulary.")
        if self.state.charge_logits.shape[1] != len(self.vocabulary.formal_charges):
            raise ValueError("state charge channels do not match the vocabulary.")
        if self.state.bond_logits.shape[1] != len(self.vocabulary.bond_classes):
            raise ValueError("state bond channels do not match the vocabulary.")
        if self.fixed is not None:
            _validate_fragment_shape(self.state, self.fixed)
        if self.allowed_bond_mask is not None:
            mask = self.allowed_bond_mask
            if mask.dtype != torch.bool or mask.shape != (
                self.state.halfedge_index.shape[1],
            ):
                raise ValueError("allowed_bond_mask must have boolean shape [E].")
            if mask.device != self.state.positions.device:
                raise ValueError("allowed_bond_mask must be on the state device.")
        for name, value in (
            ("atom_indices", self.atom_indices),
            ("charge_indices", self.charge_indices),
        ):
            if value is None:
                continue
            if value.dtype != torch.long or value.shape != (
                self.state.positions.shape[0],
            ):
                raise ValueError(f"{name} must have torch.long shape [N].")
            if value.device != self.state.positions.device:
                raise ValueError(f"{name} must be on the state device.")
            if bool((value < 0).any()):
                raise ValueError(f"{name} must not contain negative indices.")
            upper_bound = (
                len(self.vocabulary.atom_symbols)
                if name == "atom_indices"
                else len(self.vocabulary.formal_charges)
            )
            if bool((value >= upper_bound).any()):
                raise ValueError(f"{name} contains an out-of-vocabulary index.")
        if torch.unique(self.state.node_batch).numel() > 1:
            raise ValueError("bond decoding expects one molecular graph per problem.")


@dataclass(frozen=True)
class BondDecodeResult:
    """Return a sparse decoded graph and auditable solver diagnostics.

    :param bond_orders: Floating bond orders with shape ``[E]`` aligned with
        ``problem.state.halfedge_index``.
    :param status: Exact, fallback, infeasible, or unavailable status.
    :param objective: Unscaled model log-probability objective of the returned
        graph, or ``-inf`` when no feasible graph exists.
    :param connected: Whether non-zero edges form one component.
    :param valence_valid: Whether every node satisfies its configured maximum.
    :param atom_indices: Selected atom channels, when decoding succeeded.
    :param charge_indices: Selected formal-charge channels, when successful.
    :param selected_classes: Integer bond-class indices with shape ``[E]``.
    :param reason: Human-readable bounded failure/diagnostic reason.
    :param timed_out: True when the exact solver hit its wall-time bound.
    :param solver_status: Raw backend status name, if available.
    :return: Immutable decode result.
    :rtype: BondDecodeResult
    """

    bond_orders: torch.Tensor
    status: DecodeStatus
    objective: float
    connected: bool
    valence_valid: bool
    atom_indices: torch.Tensor | None = None
    charge_indices: torch.Tensor | None = None
    selected_classes: torch.Tensor | None = None
    reason: str = ""
    timed_out: bool = False
    solver_status: str = ""

    @property
    def feasible(self) -> bool:
        """Return whether this result contains a chemically feasible graph."""
        return (
            self.connected
            and self.valence_valid
            and self.status
            in {
                DecodeStatus.OPTIMAL,
                DecodeStatus.FEASIBLE,
                DecodeStatus.FALLBACK_FEASIBLE,
            }
        )


class ExactBondDecoder:
    """Solve the highest-probability feasible graph with OR-Tools CP-SAT.

    :param timeout_seconds: Optional default wall-time bound.  A problem-level
        value takes precedence when the decoder default is omitted.
    :param num_search_workers: CP-SAT worker count.  One worker is deterministic
        and is the default for reproducible sampling reports.
    :param objective_scale: Positive factor converting floating log scores to
        CP-SAT integer coefficients.
    :return: Reusable exact decoder.
    :rtype: ExactBondDecoder
    :raises ValueError: If decoder limits are invalid.
    """

    def __init__(
        self,
        timeout_seconds: float | None = None,
        *,
        num_search_workers: int = 1,
        objective_scale: int = 100_000,
    ) -> None:
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive.")
        if (
            isinstance(num_search_workers, bool)
            or not isinstance(num_search_workers, int)
            or num_search_workers < 1
        ):
            raise ValueError("num_search_workers must be a positive integer.")
        if (
            isinstance(objective_scale, bool)
            or not isinstance(objective_scale, int)
            or objective_scale < 1
        ):
            raise ValueError("objective_scale must be positive.")
        self.timeout_seconds = timeout_seconds
        self.num_search_workers = int(num_search_workers)
        self.objective_scale = int(objective_scale)

    def decode(self, problem: BondDecodeProblem) -> BondDecodeResult:
        """Solve valence- and connectivity-constrained bond selection.

        :param problem: Atom/charge assignments, sparse pair log probabilities,
            valence rules, immutable fragment bonds, and decoder limits.
        :return: Highest-scoring exact result.  A feasible result found at a
            timeout is returned with ``status=FEASIBLE`` and ``timed_out=True``;
            no feasible solution is represented explicitly.
        :rtype: BondDecodeResult
        :raises ValueError: If fixed atom/charge/bond assignments already
            violate configured valence or if the problem is malformed.

        CP-SAT selects exactly one Kekule class per unordered pair.  Linear
        node constraints bound bond-order sums.  A single-commodity flow from
        atom zero enforces a connected graph without a dense adjacency tensor.
        """
        if self.timeout_seconds is not None:
            problem = _with_timeout(problem, self.timeout_seconds)
        prepared = _prepare(problem)
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            return BondDecodeResult(
                bond_orders=prepared.zero_orders,
                status=DecodeStatus.UNAVAILABLE,
                objective=float("-inf"),
                connected=False,
                valence_valid=False,
                atom_indices=prepared.atom_indices,
                charge_indices=prepared.charge_indices,
                selected_classes=prepared.none_classes,
                reason="OR-Tools CP-SAT is not installed.",
            )

        model = cp_model.CpModel()
        edge_count, node_count = prepared.edge_count, prepared.node_count
        class_count = len(problem.vocabulary.bond_classes)
        variables: list[list[cp_model.IntVar]] = []
        for edge in range(edge_count):
            row = [
                model.NewBoolVar(f"bond_{edge}_{klass}") for klass in range(class_count)
            ]
            model.Add(sum(row) == 1)
            allowed = prepared.allowed_classes[edge]
            for klass, variable in enumerate(row):
                if not allowed[klass]:
                    model.Add(variable == 0)
            variables.append(row)

        bond_orders = tuple(problem.vocabulary.bond_orders)
        for node in range(node_count):
            terms = []
            for edge, (source, target) in enumerate(prepared.edges):
                if source == node or target == node:
                    for klass, order in enumerate(bond_orders):
                        if order:
                            terms.append(round(order) * variables[edge][klass])
            model.Add(
                sum(terms) <= math.floor(float(prepared.maxima[node].item()) + 1.0e-8)
            )

        if prepared.require_connected and node_count > 1:
            self._add_connectivity_constraints(model, variables, prepared, bond_orders)

        scores = prepared.bond_log_probs
        objective_terms = []
        scale = self.objective_scale
        for edge, row in enumerate(variables):
            for klass, variable in enumerate(row):
                score = float(scores[edge, klass].item())
                coefficient = max(
                    -2_000_000_000, min(2_000_000_000, round(score * scale))
                )
                objective_terms.append(coefficient * variable)
        model.Maximize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(problem.timeout_seconds)
        solver.parameters.num_search_workers = self.num_search_workers
        solver.parameters.log_search_progress = False
        status_code = solver.Solve(model)
        status_name = solver.StatusName(status_code)
        feasible = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        if not feasible:
            timed_out = status_code == cp_model.UNKNOWN
            return BondDecodeResult(
                bond_orders=prepared.zero_orders,
                status=DecodeStatus.TIMEOUT if timed_out else DecodeStatus.INFEASIBLE,
                objective=float("-inf"),
                connected=False,
                valence_valid=False,
                atom_indices=prepared.atom_indices,
                charge_indices=prepared.charge_indices,
                selected_classes=prepared.none_classes,
                reason="CP-SAT found no feasible graph before the bound.",
                timed_out=timed_out,
                solver_status=status_name,
            )

        selected = torch.tensor(
            [
                next(
                    klass
                    for klass, variable in enumerate(row)
                    if solver.Value(variable)
                )
                for row in variables
            ],
            dtype=torch.long,
            device=problem.state.positions.device,
        )
        orders = torch.tensor(
            [bond_orders[int(klass)] for klass in selected.tolist()],
            dtype=problem.state.positions.dtype,
            device=problem.state.positions.device,
        )
        connected, valence_valid = _graph_flags(orders, prepared)
        objective = float(
            sum(float(scores[e, int(k)].item()) for e, k in enumerate(selected))
        )
        status = (
            DecodeStatus.OPTIMAL
            if status_code == cp_model.OPTIMAL
            else DecodeStatus.FEASIBLE
        )
        return BondDecodeResult(
            bond_orders=orders,
            status=status,
            objective=objective,
            connected=connected,
            valence_valid=valence_valid,
            atom_indices=prepared.atom_indices,
            charge_indices=prepared.charge_indices,
            selected_classes=selected,
            reason="",
            timed_out=status_code != cp_model.OPTIMAL,
            solver_status=status_name,
        )

    @staticmethod
    def _add_connectivity_constraints(model, variables, prepared, bond_orders) -> None:
        """Add a single-commodity flow rooted at atom zero."""
        node_count = prepared.node_count
        arcs: list[tuple[object, int, int]] = []
        for edge, (source, target) in enumerate(prepared.edges):
            selected = sum(
                variables[edge][klass]
                for klass, order in enumerate(bond_orders)
                if order > 0
            )
            forward = model.NewIntVar(0, node_count - 1, f"flow_{edge}_forward")
            reverse = model.NewIntVar(0, node_count - 1, f"flow_{edge}_reverse")
            model.Add(forward <= (node_count - 1) * selected)
            model.Add(reverse <= (node_count - 1) * selected)
            arcs.extend(((forward, source, target), (reverse, target, source)))
        for node in range(node_count):
            outgoing = [flow for flow, source, _ in arcs if source == node]
            incoming = [flow for flow, _, target in arcs if target == node]
            if node == 0:
                model.Add(sum(outgoing) - sum(incoming) == node_count - 1)
            else:
                model.Add(sum(incoming) - sum(outgoing) == 1)


class GreedyBondDecoder:
    """Constrained deterministic fallback when exact decoding is unavailable.

    The fallback uses the same atom, charge, fixed-bond, allowed-edge, valence,
    and connectivity checks as :class:`ExactBondDecoder`.  It always reports
    ``FALLBACK_FEASIBLE`` rather than claiming global optimality.
    """

    def decode(self, problem: BondDecodeProblem) -> BondDecodeResult:
        """Build a feasible graph by score-ordered edge additions.

        :param problem: Validated sparse decoding problem.
        :return: Feasible fallback result or an explicit infeasible status.
        :rtype: BondDecodeResult
        :raises ValueError: If immutable fixed assignments exceed valence.
        """
        prepared = _prepare(problem)
        orders = prepared.zero_orders.clone()
        selected = prepared.none_classes.clone()
        remaining = prepared.maxima.clone()
        for edge, klass in enumerate(prepared.fixed_classes.tolist()):
            if klass >= 0:
                selected[edge] = klass
                orders[edge] = problem.vocabulary.bond_orders[klass]
                source, target = prepared.edges[edge]
                remaining[source] -= orders[edge]
                remaining[target] -= orders[edge]
        if bool((remaining < -1.0e-6).any()):
            raise ValueError("fixed bonds exceed the configured maximum valence.")

        candidates = []
        for edge, allowed in enumerate(prepared.allowed_classes):
            if prepared.fixed_classes[edge] >= 0:
                continue
            source, target = prepared.edges[edge]
            ranked = sorted(
                (float(prepared.bond_log_probs[edge, klass]), klass)
                for klass, is_allowed in enumerate(allowed)
                if is_allowed and problem.vocabulary.bond_orders[klass] > 0
            )
            if ranked:
                candidates.append((max(ranked), edge, source, target))
        candidates.sort(key=lambda item: (-item[0][0], item[1]))

        parent = list(range(prepared.node_count))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: int, right: int) -> None:
            left, right = find(left), find(right)
            if left != right:
                parent[right] = left

        # First connect components, then greedily add any remaining high-score edges.
        for (_, klass), edge, source, target in candidates:
            if prepared.node_count > 1 and find(source) == find(target):
                continue
            order = problem.vocabulary.bond_orders[klass]
            if remaining[source] + 1.0e-6 < order or remaining[target] + 1.0e-6 < order:
                continue
            selected[edge], orders[edge] = klass, order
            remaining[source] -= order
            remaining[target] -= order
            union(source, target)
        for (_, klass), edge, source, target in candidates:
            if selected[edge] != 0:
                continue
            order = problem.vocabulary.bond_orders[klass]
            if remaining[source] + 1.0e-6 < order or remaining[target] + 1.0e-6 < order:
                continue
            selected[edge], orders[edge] = klass, order
            remaining[source] -= order
            remaining[target] -= order
        connected, valence_valid = _graph_flags(orders, prepared)
        if not connected or not valence_valid:
            return BondDecodeResult(
                bond_orders=orders,
                status=DecodeStatus.INFEASIBLE,
                objective=float("-inf"),
                connected=connected,
                valence_valid=valence_valid,
                atom_indices=prepared.atom_indices,
                charge_indices=prepared.charge_indices,
                selected_classes=selected,
                reason="greedy fallback could not satisfy connectivity and valence.",
            )
        objective = float(
            sum(
                float(prepared.bond_log_probs[e, int(k)].item())
                for e, k in enumerate(selected)
            )
        )
        return BondDecodeResult(
            bond_orders=orders,
            status=DecodeStatus.FALLBACK_FEASIBLE,
            objective=objective,
            connected=True,
            valence_valid=True,
            atom_indices=prepared.atom_indices,
            charge_indices=prepared.charge_indices,
            selected_classes=selected,
            reason="constrained greedy fallback; global optimality not established.",
        )


@dataclass(frozen=True)
class _PreparedProblem:
    atom_indices: torch.Tensor
    charge_indices: torch.Tensor
    bond_log_probs: torch.Tensor
    edges: tuple[tuple[int, int], ...]
    maxima: torch.Tensor
    allowed_classes: tuple[tuple[bool, ...], ...]
    fixed_classes: torch.Tensor
    none_classes: torch.Tensor
    zero_orders: torch.Tensor
    require_connected: bool

    @property
    def node_count(self) -> int:
        return int(self.atom_indices.shape[0])

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def _with_timeout(problem: BondDecodeProblem, timeout: float) -> BondDecodeProblem:
    """Return a problem copy with a decoder-level timeout override."""
    return BondDecodeProblem(
        state=problem.state,
        vocabulary=problem.vocabulary,
        valence_table=problem.valence_table,
        fixed=problem.fixed,
        timeout_seconds=timeout,
        allowed_bond_mask=problem.allowed_bond_mask,
        require_connected=problem.require_connected,
        atom_indices=problem.atom_indices,
        charge_indices=problem.charge_indices,
    )


def _prepare(problem: BondDecodeProblem) -> _PreparedProblem:
    """Resolve categorical assignments and immutable edge classes."""
    state = problem.state
    if state.positions.shape[0] < 1:
        raise ValueError("bond decoding requires at least one atom.")
    vocabulary = problem.vocabulary
    fixed = problem.fixed
    atom_indices = (
        state.atom_logits.argmax(dim=-1)
        if problem.atom_indices is None
        else problem.atom_indices
    )
    charge_indices = (
        state.charge_logits.argmax(dim=-1)
        if problem.charge_indices is None
        else problem.charge_indices
    )
    atom_indices = atom_indices.clone()
    charge_indices = charge_indices.clone()
    if fixed is not None:
        fixed_atoms = fixed.fixed_atom_mask
        atom_indices[fixed_atoms] = fixed.reference.atom_logits.argmax(dim=-1)[
            fixed_atoms
        ]
        charge_indices[fixed_atoms] = fixed.reference.charge_logits.argmax(dim=-1)[
            fixed_atoms
        ]
    table = problem.valence_table or ValenceTable.default(vocabulary)
    maxima = torch.tensor(
        [
            table.maximum(
                vocabulary.atom_symbol(int(atom)),
                vocabulary.formal_charges[int(charge)],
            )
            for atom, charge in zip(atom_indices.tolist(), charge_indices.tolist())
        ],
        dtype=state.positions.dtype,
        device=state.positions.device,
    )
    bond_log_probs = torch.log_softmax(state.bond_logits, dim=-1)
    edges = tuple(
        (int(source), int(target)) for source, target in state.halfedge_index.T.tolist()
    )
    class_count = len(vocabulary.bond_classes)
    none_index = vocabulary.bond_index("none")
    fixed_classes = torch.full(
        (len(edges),), -1, dtype=torch.long, device=state.positions.device
    )
    allowed_classes: list[tuple[bool, ...]] = []
    allowed_mask = problem.allowed_bond_mask
    fixed_bond_mask = (
        fixed.fixed_bond_mask
        if fixed is not None
        else torch.zeros(len(edges), dtype=torch.bool, device=state.positions.device)
    )
    fixed_reference_logits = fixed.reference.bond_logits if fixed is not None else None
    for edge in range(len(edges)):
        allowed = [True] * class_count
        if allowed_mask is not None and not bool(allowed_mask[edge]):
            allowed = [False] * class_count
            allowed[none_index] = True
        if bool(fixed_bond_mask[edge]):
            if fixed_reference_logits is None:
                raise ValueError("fixed bond mask requires a fragment reference.")
            klass = int(fixed_reference_logits[edge].argmax().item())
            if klass < 0 or klass >= class_count:
                raise ValueError("fixed bond class is outside the vocabulary.")
            fixed_classes[edge] = klass
            allowed = [index == klass for index in range(class_count)]
            bond_log_probs[edge] = torch.log_softmax(
                fixed_reference_logits[edge], dim=-1
            )
        allowed_classes.append(tuple(allowed))
    fixed_orders = torch.zeros(
        state.positions.shape[0],
        dtype=state.positions.dtype,
        device=state.positions.device,
    )
    for edge, klass in enumerate(fixed_classes.tolist()):
        if klass >= 0:
            order = vocabulary.bond_orders[klass]
            source, target = edges[edge]
            fixed_orders[source] += order
            fixed_orders[target] += order
    if bool((fixed_orders > maxima + 1.0e-6).any()):
        raise ValueError("fixed bonds exceed the configured maximum valence.")
    return _PreparedProblem(
        atom_indices=atom_indices,
        charge_indices=charge_indices,
        bond_log_probs=bond_log_probs,
        edges=edges,
        maxima=maxima,
        allowed_classes=tuple(allowed_classes),
        fixed_classes=fixed_classes,
        none_classes=torch.full_like(fixed_classes, none_index),
        zero_orders=torch.zeros(
            len(edges), dtype=state.positions.dtype, device=state.positions.device
        ),
        require_connected=problem.require_connected,
    )


def _validate_fragment_shape(state: MolecularState, fixed: FragmentCondition) -> None:
    """Check that fixed masks can be applied to the candidate topology."""
    reference = fixed.reference
    if (
        reference.positions.shape != state.positions.shape
        or reference.halfedge_index.shape != state.halfedge_index.shape
    ):
        raise ValueError("fixed reference topology does not match state.")
    if not torch.equal(reference.halfedge_index, state.halfedge_index):
        raise ValueError("fixed reference halfedges do not match state.")


def _graph_flags(orders: torch.Tensor, prepared: _PreparedProblem) -> tuple[bool, bool]:
    """Compute sparse connectivity and valence flags for one order vector."""
    remaining = prepared.maxima.clone()
    parent = list(range(prepared.node_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for order, (source, target) in zip(orders.tolist(), prepared.edges):
        if order <= 0:
            continue
        remaining[source] -= order
        remaining[target] -= order
        left, right = find(source), find(target)
        if left != right:
            parent[right] = left
    connected = (
        not prepared.require_connected
        or prepared.node_count <= 1
        or len({find(index) for index in range(prepared.node_count)}) == 1
    )
    return connected, bool((remaining >= -1.0e-5).all())


__all__ = [
    "BondDecodeProblem",
    "BondDecodeResult",
    "DecodeStatus",
    "ExactBondDecoder",
    "GreedyBondDecoder",
]
