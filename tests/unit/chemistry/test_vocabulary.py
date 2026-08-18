"""Tests for explicit ligand and pocket chemical vocabularies."""

import pytest

from ecloudflow.chemistry.vocabulary import ChemicalVocabulary


def test_default_ligand_vocabulary_has_binding_order():
    vocab = ChemicalVocabulary.default_ligand()

    assert vocab.atom_symbols == (
        "C",
        "N",
        "O",
        "S",
        "P",
        "F",
        "Cl",
        "Br",
        "I",
        "B",
        "Si",
        "Se",
    )
    assert vocab.formal_charges == (-2, -1, 0, 1, 2)
    assert vocab.bond_classes == ("none", "single", "double", "triple")


def test_vocabulary_rejects_unsupported_ligand_metal():
    vocab = ChemicalVocabulary.default_ligand()

    with pytest.raises(ValueError, match="unsupported ligand element"):
        vocab.atom_index("Fe")


def test_pocket_vocabulary_is_separate_and_expandable():
    ligand = ChemicalVocabulary.default_ligand()
    pocket = ChemicalVocabulary.default_pocket(extra_elements=("Xe",))

    assert "Fe" not in ligand.atom_symbols
    assert pocket.atom_symbol(pocket.atom_index("Fe")) == "Fe"
    assert pocket.atom_symbol(pocket.atom_index("Xe")) == "Xe"


def test_vocabulary_rejects_negative_atom_indices():
    vocab = ChemicalVocabulary.default_ligand()

    with pytest.raises(ValueError, match="outside vocabulary"):
        vocab.atom_symbol(-1)
