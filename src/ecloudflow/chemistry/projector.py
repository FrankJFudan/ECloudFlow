"""Sparse differentiable valence projection for molecular trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ecloudflow.chemistry.valence import ValenceTable
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core import FragmentCondition, MolecularState, clamp_fragment


@dataclass(frozen=True)
class ProjectedState:
    """Collect a projected molecular state and sparse feasibility diagnostics.

    :param state: Immutable projected trajectory state retaining canonical
        unordered halfedges and flattened node/edge tensors.
    :param expected_valence: Differentiable per-node expected bond-order sums
        with shape ``[N]``, floating dtype/device matching ``state.positions``.
    :param maximum_valence: Differentiable atom/charge-distribution-weighted
        maximum valence with shape ``[N]``.
    :param allowed_new_bonds: Boolean per-halfedge feasibility vector with shape
        ``[E]``. It is never a dense ``[N, N]`` matrix.
    :return: Immutable projected-state view with convenience logit properties.
    :rtype: ProjectedState

    The boolean feasibility mask is diagnostic and non-differentiable. The
    projected logits and expected valence preserve gradients on all unmasked
    channels and do not discretize an atom, charge, or bond class.
    """

    state: MolecularState
    expected_valence: torch.Tensor
    maximum_valence: torch.Tensor
    allowed_new_bonds: torch.Tensor

    @property
    def atom_logits(self) -> torch.Tensor:
        """Return projected atom logits with shape ``[N, A]``.

        :return: Atom logits from the nested immutable molecular state.
        :rtype: torch.Tensor
        """
        return self.state.atom_logits

    @property
    def charge_logits(self) -> torch.Tensor:
        """Return projected charge logits with shape ``[N, Q]``.

        :return: Formal-charge logits from the projected state.
        :rtype: torch.Tensor
        """
        return self.state.charge_logits

    @property
    def bond_logits(self) -> torch.Tensor:
        """Return projected canonical-halfedge logits with shape ``[E, B]``.

        :return: Bond logits from the projected state.
        :rtype: torch.Tensor
        """
        return self.state.bond_logits

    @property
    def halfedge_index(self) -> torch.Tensor:
        """Return canonical unordered endpoints with shape ``[2, E]``.

        :return: Sparse halfedge index shared with the projected state.
        :rtype: torch.Tensor
        """
        return self.state.halfedge_index


class ChemicalProjector:
    """Project continuous bond distributions toward configured valence bounds.

    :param vocabulary: Fixed ligand channel vocabulary.
    :param valence_table: Optional charge-conditioned table. Omitted values use
        vetted defaults aligned with ``vocabulary``.
    :param tolerance: Non-negative numerical tolerance for valence comparisons.
    :return: Reusable sparse chemical projector.
    :rtype: ChemicalProjector
    :raises ValueError: If configuration is not ligand-scoped or tolerance is
        negative.
    """

    def __init__(
        self,
        vocabulary: ChemicalVocabulary,
        valence_table: ValenceTable | None = None,
        *,
        tolerance: float = 1.0e-6,
    ) -> None:
        """Initialize one vocabulary-aligned trajectory projector.

        :param vocabulary: Ligand atom, charge, and Kekule bond channels.
        :param valence_table: Optional charge-conditioned maximum-valence rules.
        :param tolerance: Non-negative comparison tolerance in bond-order units.
        :return: None.
        :rtype: None
        :raises ValueError: If the vocabulary or tolerance is invalid.
        """
        if vocabulary.domain != "ligand":
            raise ValueError("chemical projection requires a ligand vocabulary.")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        self.vocabulary = vocabulary
        self.valence_table = valence_table or ValenceTable.default(vocabulary)
        self.tolerance = tolerance

    def project(
        self,
        state: MolecularState,
        fixed: FragmentCondition | None = None,
    ) -> ProjectedState:
        """Mask bond updates that cannot satisfy configured valence rules.

        :param state: Current probabilistic molecular state. Atom, charge, and
            bond logits have shapes ``[N, A]``, ``[N, Q]``, and ``[E, B]``.
            Halfedges have canonical unordered shape ``[2, E]`` and remain
            sparse throughout projection.
        :param fixed: Optional fragment condition whose atom/charge logits,
            coordinates, and internal halfedge logits are restored exactly.
            Fixed-to-free new bonds are feasible only at attachment atoms.
        :return: State with continuous projected bond logits, per-node expected
            and maximum valence vectors of shape ``[N]``, and a boolean
            allowed-new-bond vector of shape ``[E]``.
        :rtype: ProjectedState
        :raises ValueError: If atom, charge, or bond channels do not match the
            chemical vocabulary, the fixed condition is incompatible, or
            immutable fixed bonds alone exceed a node's maximum valence.

        The projection reserves expected valence from immutable internal
        fragment bonds before applying a continuous capacity scale to mutable
        halfedges. It uses softmax expectations, is differentiable with respect
        to unmasked logits, does not choose categorical classes, and cannot
        guarantee final sanitization. Exact graph feasibility and RDKit
        aromaticity recovery belong to the final decoder. Boolean feasibility
        comparisons are control masks, not a discretized molecular graph.
        """
        self._validate_channels(state)
        working = clamp_fragment(state, fixed) if fixed is not None else state

        atom_probabilities = torch.softmax(working.atom_logits, dim=-1)
        charge_probabilities = torch.softmax(working.charge_logits, dim=-1)
        maximum_table = self.valence_table.tensor(
            self.vocabulary,
            dtype=working.positions.dtype,
            device=working.positions.device,
        )
        maximum_valence = torch.einsum(
            "na,aq,nq->n",
            atom_probabilities,
            maximum_table,
            charge_probabilities,
        )

        bond_probabilities = torch.softmax(working.bond_logits, dim=-1)
        bond_orders = working.bond_logits.new_tensor(self.vocabulary.bond_orders)
        expected_edge_orders = bond_probabilities @ bond_orders
        fixed_bond_mask = (
            fixed.fixed_bond_mask
            if fixed is not None
            else torch.zeros_like(expected_edge_orders, dtype=torch.bool)
        )
        immutable_edge_orders = torch.where(
            fixed_bond_mask, expected_edge_orders, torch.zeros_like(expected_edge_orders)
        )
        mutable_edge_orders = torch.where(
            fixed_bond_mask, torch.zeros_like(expected_edge_orders), expected_edge_orders
        )
        immutable_valence = self._sum_halfedges(working, immutable_edge_orders)
        mutable_valence = self._sum_halfedges(working, mutable_edge_orders)
        if bool((immutable_valence > maximum_valence + self.tolerance).any()):
            raise ValueError("fixed bonds exceed the configured maximum valence.")

        mutable_capacity = (maximum_valence - immutable_valence).clamp_min(0.0)
        denominator = mutable_valence.clamp_min(
            torch.finfo(mutable_valence.dtype).tiny
        )
        node_scale = (mutable_capacity / denominator).clamp(max=1.0)
        source, target = working.halfedge_index
        edge_scale = torch.minimum(node_scale[source], node_scale[target])
        edge_scale = torch.where(
            fixed_bond_mask, torch.ones_like(edge_scale), edge_scale
        )
        scaled_probabilities = bond_probabilities.clone()
        scaled_probabilities[:, 1:] = bond_probabilities[:, 1:] * edge_scale[:, None]
        scaled_probabilities[:, 0] = 1.0 - scaled_probabilities[:, 1:].sum(dim=-1)
        projected_logits = scaled_probabilities.clamp_min(
            torch.finfo(scaled_probabilities.dtype).tiny
        ).log()
        projected_logits = torch.where(
            (edge_scale < 1.0)[:, None], projected_logits, working.bond_logits
        )
        if fixed is not None:
            projected_logits = torch.where(
                fixed.fixed_bond_mask[:, None],
                fixed.reference.bond_logits,
                projected_logits,
            )
        projected = working.replace(bond_logits=projected_logits)

        projected_probabilities = torch.softmax(projected.bond_logits, dim=-1)
        projected_edge_orders = projected_probabilities @ bond_orders
        expected_valence = self._sum_halfedges(projected, projected_edge_orders)
        remaining = maximum_valence - expected_valence
        allowed_new_bonds = (remaining[source] + self.tolerance >= 1.0) & (
            remaining[target] + self.tolerance >= 1.0
        )
        if fixed is not None:
            fixed_source = fixed.fixed_atom_mask[source]
            fixed_target = fixed.fixed_atom_mask[target]
            internal = fixed_source & fixed_target
            crossing = fixed_source ^ fixed_target
            fixed_endpoint_is_attachment = (
                fixed.attachment_mask[source] & fixed_source
            ) | (fixed.attachment_mask[target] & fixed_target)
            allowed_new_bonds = allowed_new_bonds & ~internal
            allowed_new_bonds = allowed_new_bonds & (
                ~crossing | fixed_endpoint_is_attachment
            )

        return ProjectedState(
            state=projected,
            expected_valence=expected_valence,
            maximum_valence=maximum_valence,
            allowed_new_bonds=allowed_new_bonds,
        )

    def _validate_channels(self, state: MolecularState) -> None:
        """Validate categorical tensor widths against the vocabulary.

        :param state: Candidate flattened molecular state.
        :return: None.
        :rtype: None
        :raises ValueError: If atom, charge, or bond channel counts differ.
        """
        expected = {
            "atom_logits": len(self.vocabulary.atom_symbols),
            "charge_logits": len(self.vocabulary.formal_charges),
            "bond_logits": len(self.vocabulary.bond_classes),
        }
        for name, count in expected.items():
            actual = getattr(state, name).shape[1]
            if actual != count:
                raise ValueError(f"{name} must have {count} channels, received {actual}.")

    @staticmethod
    def _sum_halfedges(
        state: MolecularState, edge_values: torch.Tensor
    ) -> torch.Tensor:
        """Scatter scalar halfedge values to both unordered endpoints.

        :param state: Molecular state supplying ``[2, E]`` canonical endpoints.
        :param edge_values: Floating per-halfedge values with shape ``[E]``.
        :return: Per-node sum with shape ``[N]`` on the same dtype/device.
        :rtype: torch.Tensor

        Each halfedge contributes exactly once to each distinct endpoint. The
        operation uses ``O(N + E)`` memory and never creates a dense graph.
        """
        result = edge_values.new_zeros(state.positions.shape[0])
        source, target = state.halfedge_index
        result.index_add_(0, source, edge_values)
        result.index_add_(0, target, edge_values)
        return result
