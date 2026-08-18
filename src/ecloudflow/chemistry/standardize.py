"""RDKit molecule standardization for charge-aware Kekule model targets."""

from __future__ import annotations

from rdkit import Chem

from ecloudflow.chemistry.vocabulary import ChemicalVocabulary

CANONICAL_ISOMERIC_SMILES_PROPERTY = "canonical_isomeric_smiles"


def standardize_molecule(
    molecule: Chem.Mol,
    vocabulary: ChemicalVocabulary | None = None,
) -> Chem.Mol:
    """Create a sanitized, stereo-preserving Kekule ligand target.

    :param molecule: RDKit molecule to standardize. The input is never mutated;
        atom ordering, conformers, formal charges, and stereochemical tags are
        copied into a new molecule before sanitization.
    :param vocabulary: Optional ligand vocabulary used for explicit element
        validation. Omitted values use :meth:`ChemicalVocabulary.default_ligand`.
    :return: Defensive RDKit molecule copy whose bonds are non-aromatic
        ``SINGLE``, ``DOUBLE``, or ``TRIPLE`` model targets. Its
        ``canonical_isomeric_smiles`` property retains the canonical aromatic,
        formal-charge, and stereo-aware identity computed before Kekulization.
    :rtype: rdkit.Chem.Mol
    :raises TypeError: If ``molecule`` is not an RDKit molecule.
    :raises ValueError: If the vocabulary is not ligand-scoped, an element is
        unsupported, sanitization fails, or the graph cannot be Kekulized.

    RDKit sanitization assigns valence and aromaticity on the private copy.
    Stereochemistry is then cleaned and forced before canonical isomeric SMILES
    generation. Kekulization clears aromatic flags only for model targets;
    downstream final decoding must run sanitization to perceive aromaticity.
    """
    if not isinstance(molecule, Chem.Mol):
        raise TypeError("molecule must be an RDKit Mol.")
    ligand_vocabulary = vocabulary or ChemicalVocabulary.default_ligand()
    if ligand_vocabulary.domain != "ligand":
        raise ValueError("molecule standardization requires a ligand vocabulary.")

    standardized = Chem.Mol(molecule)
    for atom in standardized.GetAtoms():
        ligand_vocabulary.atom_index(atom.GetSymbol())

    try:
        Chem.SanitizeMol(standardized)
        Chem.AssignStereochemistry(standardized, cleanIt=True, force=True)
        canonical_smiles = Chem.MolToSmiles(
            standardized, canonical=True, isomericSmiles=True
        )
        Chem.Kekulize(standardized, clearAromaticFlags=True)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"molecule sanitization or Kekulization failed: {error}") from error

    standardized.SetProp(CANONICAL_ISOMERIC_SMILES_PROPERTY, canonical_smiles)
    return standardized

