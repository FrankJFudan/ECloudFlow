"""Exact fragment-mask operations for molecular sampling states."""

from __future__ import annotations

import torch

from ecloudflow.core.types import FragmentCondition, MolecularState
from ecloudflow.exceptions import FragmentInvariantError


def clamp_fragment(
    state: MolecularState, condition: FragmentCondition
) -> MolecularState:
    """Restore every fixed fragment field from its clean reference state.

    :param state: Current noisy or predicted molecular state with flattened
        coordinates ``[N, 3]`` in angstroms, atom/charge logits ``[N, A]`` and
        ``[N, Q]``, canonical bond logits ``[E, B]``, and a single shared value
        per unordered halfedge on the reference device.
    :param condition: Exact fragment contract whose reference supplies fixed
        coordinates, atom identities, charges, and internal bond values.
    :return: Revalidated state with coordinates selected by
        ``fixed_coord_mask``, atom and charge logits selected by
        ``fixed_atom_mask``, and bonds selected by ``fixed_bond_mask``.
    :rtype: MolecularState
    :raises FragmentInvariantError: If state and reference do not share shape,
        dtype/device, batch assignment, or canonical halfedge topology.

    All selections use :func:`torch.where`; fixed values are copied exactly from
    the reference after every solver or projection substep. Ligand electron
    tokens remain generative because the fragment contract fixes chemical graph
    fields and geometry, not a precomputed token realization. No dense bond
    matrix is created: one mask value clamps each canonical unordered halfedge.
    """
    reference = condition.reference
    _validate_clamp_compatibility(state, reference)
    return state.replace(
        positions=torch.where(
            condition.fixed_coord_mask[:, None], reference.positions, state.positions
        ),
        atom_logits=torch.where(
            condition.fixed_atom_mask[:, None], reference.atom_logits, state.atom_logits
        ),
        charge_logits=torch.where(
            condition.fixed_atom_mask[:, None],
            reference.charge_logits,
            state.charge_logits,
        ),
        bond_logits=torch.where(
            condition.fixed_bond_mask[:, None], reference.bond_logits, state.bond_logits
        ),
    )


def _validate_clamp_compatibility(
    state: MolecularState, reference: MolecularState
) -> None:
    """Ensure a fragment reference can safely clamp a candidate state.

    :param state: Candidate state to be clamped.
    :param reference: Clean fragment-reference state.
    :return: None.
    :rtype: None
    :raises FragmentInvariantError: If a tensor or topology difference would
        make fixed masks ambiguous or unsafe.
    """
    compared_names = (
        "positions",
        "atom_logits",
        "charge_logits",
        "bond_logits",
        "electron_latent",
        "node_batch",
        "halfedge_batch",
    )
    for name in compared_names:
        state_value = getattr(state, name)
        reference_value = getattr(reference, name)
        if (
            state_value.shape != reference_value.shape
            or state_value.dtype != reference_value.dtype
            or state_value.device != reference_value.device
        ):
            raise FragmentInvariantError(
                f"state and reference {name} must share shape, dtype, and device."
            )
    if not torch.equal(state.halfedge_index, reference.halfedge_index):
        raise FragmentInvariantError(
            "state and reference must have identical canonical halfedge_index tensors."
        )
    if not torch.equal(state.node_batch, reference.node_batch):
        raise FragmentInvariantError(
            "state and reference must have identical node_batch tensors."
        )
    if not torch.equal(state.halfedge_batch, reference.halfedge_batch):
        raise FragmentInvariantError(
            "state and reference must have identical halfedge_batch tensors."
        )
