"""Tests for sparse differentiable trajectory valence projection."""

import pytest
import torch

from ecloudflow.chemistry.projector import ChemicalProjector
from ecloudflow.chemistry.valence import ValenceTable
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core import FragmentCondition, MolecularState


def _complete_halfedges(node_count: int) -> torch.Tensor:
    """Return deterministic canonical halfedges for one complete graph.

    :param node_count: Number of nodes in the complete graph.
    :return: Canonical halfedge index with shape ``[2, E]``.
    :rtype: torch.Tensor
    """
    return torch.triu_indices(node_count, node_count, offset=1)


def _carbon_with_four_single_bonds(vocab: ChemicalVocabulary) -> MolecularState:
    """Build a methane-like carbon center with explicit candidate halfedges.

    :param vocab: Ligand vocabulary defining the tensor channel counts.
    :return: Five-node state whose central carbon has expected valence four.
    :rtype: MolecularState
    """
    node_count = 5
    halfedges = _complete_halfedges(node_count)
    atom_logits = torch.full((node_count, len(vocab.atom_symbols)), -20.0)
    atom_logits[:, vocab.atom_index("C")] = 20.0
    charge_logits = torch.full((node_count, len(vocab.formal_charges)), -20.0)
    charge_logits[:, vocab.charge_index(0)] = 20.0
    bond_logits = torch.full((halfedges.shape[1], len(vocab.bond_classes)), -20.0)
    bond_logits[:, vocab.bond_index("none")] = 20.0
    central = (halfedges[0] == 0) | (halfedges[1] == 0)
    bond_logits[central, vocab.bond_index("none")] = -20.0
    bond_logits[central, vocab.bond_index("single")] = 20.0
    return MolecularState(
        positions=torch.zeros((node_count, 3)),
        atom_logits=atom_logits,
        charge_logits=charge_logits,
        halfedge_index=halfedges,
        bond_logits=bond_logits,
        electron_latent=torch.zeros((node_count, 2)),
        node_batch=torch.zeros(node_count, dtype=torch.long),
        halfedge_batch=torch.zeros(halfedges.shape[1], dtype=torch.long),
    )


def test_projector_masks_self_bonds_and_saturated_carbon_additions():
    vocab = ChemicalVocabulary.default_ligand()
    state = _carbon_with_four_single_bonds(vocab)

    projected = ChemicalProjector(vocab).project(state)
    touching_center = (state.halfedge_index[0] == 0) | (
        state.halfedge_index[1] == 0
    )

    assert (projected.halfedge_index[0] < projected.halfedge_index[1]).all()
    assert not (projected.halfedge_index[0] == projected.halfedge_index[1]).any()
    assert projected.allowed_new_bonds.shape == (state.halfedge_index.shape[1],)
    assert projected.allowed_new_bonds.dtype == torch.bool
    assert projected.allowed_new_bonds[touching_center].sum() == 0
    assert torch.allclose(projected.expected_valence[0], torch.tensor(4.0))


def test_projector_preserves_gradient_for_unmasked_bond_logits():
    vocab = ChemicalVocabulary.default_ligand()
    state = _carbon_with_four_single_bonds(vocab)
    bond_logits = state.bond_logits.clone().requires_grad_(True)
    state = state.replace(bond_logits=bond_logits)

    projected = ChemicalProjector(vocab).project(state)
    loss = projected.expected_valence.sum() + projected.bond_logits.sum() * 1.0e-4
    loss.backward()

    assert bond_logits.grad is not None
    assert torch.isfinite(bond_logits.grad).all()
    assert bond_logits.grad.abs().sum() > 0


def test_projector_restores_fixed_fragment_fields():
    vocab = ChemicalVocabulary.default_ligand()
    reference = _carbon_with_four_single_bonds(vocab)
    fixed_atoms = torch.tensor([True, True, False, False, False])
    condition = FragmentCondition.from_atom_mask(
        fixed_atoms,
        reference,
        attachment_mask=torch.tensor([True, False, False, False, False]),
    )
    noisy = reference.replace(
        atom_logits=reference.atom_logits.roll(1, dims=-1),
        charge_logits=reference.charge_logits.roll(1, dims=-1),
        bond_logits=reference.bond_logits.roll(1, dims=-1),
    )

    projected = ChemicalProjector(vocab).project(noisy, condition)

    assert torch.equal(projected.atom_logits[fixed_atoms], reference.atom_logits[fixed_atoms])
    assert torch.equal(
        projected.charge_logits[fixed_atoms], reference.charge_logits[fixed_atoms]
    )
    assert torch.equal(
        projected.bond_logits[condition.fixed_bond_mask],
        reference.bond_logits[condition.fixed_bond_mask],
    )


def test_projector_rejects_wrong_chemistry_channel_counts():
    vocab = ChemicalVocabulary.default_ligand()
    state = _carbon_with_four_single_bonds(vocab)

    with pytest.raises(ValueError, match="atom_logits"):
        ChemicalProjector(vocab).project(
            state.replace(atom_logits=state.atom_logits[:, :-1])
        )


def test_valence_table_accepts_counted_dataset_extensions_deterministically():
    vocab = ChemicalVocabulary.default_ligand()
    table = ValenceTable.default(vocab).with_dataset_counts(
        {("P", 0, 5): 12, ("P", 0, 6): 1}, minimum_count=2
    )

    assert table.maximum("P", 0) == 5.0

