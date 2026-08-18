"""Tests for exact fragment-condition masks and clamping."""

from collections.abc import Callable

import torch

from ecloudflow.core.masks import clamp_fragment
from ecloudflow.core.types import FragmentCondition, MolecularState


def find_halfedge(halfedge_index: torch.Tensor, source: int, target: int) -> int:
    """Return the index of one canonical unordered halfedge.

    :param halfedge_index: Canonical halfedges with shape ``[2, E]`` on CPU.
    :param source: One endpoint of the requested edge.
    :param target: Other endpoint of the requested edge.
    :return: Integer halfedge index for the unordered endpoint pair.
    :rtype: int
    :raises AssertionError: If the requested halfedge is absent.
    """
    lower, upper = sorted((source, target))
    matches = (halfedge_index[0] == lower) & (halfedge_index[1] == upper)
    indices = matches.nonzero(as_tuple=False).flatten()
    assert indices.numel() == 1
    return int(indices.item())


def test_clamp_fragment_restores_all_fixed_fields(
    molecular_state_factory: Callable[[int], MolecularState],
):
    reference = molecular_state_factory(num_atoms=3)
    noisy = reference.replace(
        positions=reference.positions + 7.0,
        atom_logits=reference.atom_logits.roll(1, dims=-1),
        charge_logits=reference.charge_logits.roll(1, dims=-1),
        bond_logits=reference.bond_logits.roll(1, dims=-1),
    )
    condition = FragmentCondition.from_atom_mask(
        torch.tensor([True, False, True]), reference
    )

    clamped = clamp_fragment(noisy, condition)
    edge_0_2 = find_halfedge(reference.halfedge_index, 0, 2)

    assert torch.equal(clamped.positions[[0, 2]], reference.positions[[0, 2]])
    assert torch.equal(clamped.atom_logits[[0, 2]], reference.atom_logits[[0, 2]])
    assert torch.equal(
        clamped.charge_logits[[0, 2]], reference.charge_logits[[0, 2]]
    )
    assert torch.equal(clamped.bond_logits[edge_0_2], reference.bond_logits[edge_0_2])


def test_fragment_mask_marks_only_complete_fixed_halfedges(
    molecular_state_factory: Callable[[int], MolecularState],
):
    reference = molecular_state_factory(num_atoms=3)
    condition = FragmentCondition.from_atom_mask(
        torch.tensor([True, False, True]), reference
    )

    assert torch.equal(condition.fixed_bond_mask, torch.tensor([False, True, False]))
    assert torch.equal(condition.fixed_coord_mask, condition.fixed_atom_mask)
