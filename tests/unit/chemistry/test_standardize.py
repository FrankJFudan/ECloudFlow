"""Tests for charge- and stereo-preserving molecule standardization."""

import pytest
from rdkit import Chem

from ecloudflow.chemistry.standardize import standardize_molecule


def test_standardization_preserves_charge_stereo_and_input_molecule():
    molecule = Chem.MolFromSmiles("C[C@H](O)C(=O)[O-]")
    assert molecule is not None
    input_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    input_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())

    standardized = standardize_molecule(molecule)

    assert standardized is not molecule
    assert sum(atom.GetFormalCharge() for atom in standardized.GetAtoms()) == input_charge
    assert standardized.GetProp("canonical_isomeric_smiles") == input_smiles
    assert "@" in standardized.GetProp("canonical_isomeric_smiles")
    assert Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) == input_smiles


def test_standardization_emits_only_kekule_model_bond_classes():
    molecule = Chem.MolFromSmiles("c1ccccc1")
    assert molecule is not None

    standardized = standardize_molecule(molecule)

    assert not any(bond.GetIsAromatic() for bond in standardized.GetBonds())
    assert {str(bond.GetBondType()) for bond in standardized.GetBonds()} == {
        "SINGLE",
        "DOUBLE",
    }
    assert standardized.GetProp("canonical_isomeric_smiles") == "c1ccccc1"


@pytest.mark.parametrize("smiles", ["F/C=C/F", "F/C=C\\F"])
def test_standardization_preserves_double_bond_stereochemistry(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    expected = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

    standardized = standardize_molecule(molecule)

    assert standardized.GetProp("canonical_isomeric_smiles") == expected
    assert any(
        bond.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}
        for bond in standardized.GetBonds()
    )


def test_standardization_rejects_unsupported_ligand_element():
    molecule = Chem.MolFromSmiles("[Fe]")
    assert molecule is not None

    with pytest.raises(ValueError, match="unsupported ligand element"):
        standardize_molecule(molecule)
