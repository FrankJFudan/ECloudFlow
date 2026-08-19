"""Conversion helpers from parsed structures to canonical tensor graphs."""

from __future__ import annotations

from typing import Any

import torch
from rdkit import Chem

from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import LigandGraph, PocketGraph

POCKET_FEATURE_NAMES = tuple(
    [
        f"element_{symbol}"
        for symbol in (
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
            "Na",
            "Mg",
            "K",
            "Ca",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
        )
    ]
    + [
        f"residue_{name}"
        for name in (
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "OTHER",
        )
    ]
    + [
        "backbone",
        "partial_charge",
        "donor",
        "acceptor",
        "aromatic",
        "hydrophobic",
        "metal",
    ]
)


def pocket_graph_from_entity(entity: Any, frame: CoordinateFrame) -> PocketGraph:
    """Build a pocket graph from a Biopython entity in a declared frame.

    :param entity: Biopython structure/model/chain exposing ``get_atoms``.
    :param frame: Centering frame constructed from the same global coordinates.
    :return: Pocket graph with one-hot element features and atom numbers.
    :rtype: PocketGraph
    :raises ValueError: If atom coordinates or elements are unavailable.
    """
    atoms = list(entity.get_atoms())
    if not atoms:
        raise ValueError("pocket contains no atoms")
    table = Chem.GetPeriodicTable()
    positions: list[torch.Tensor] = []
    numbers: list[int] = []
    for atom in atoms:
        element = str(getattr(atom, "element", "") or "").strip().title()
        number = int(table.GetAtomicNumber(element))
        if number <= 0:
            raise ValueError(f"unsupported pocket element: {element!r}")
        positions.append(torch.as_tensor(atom.get_coord(), dtype=torch.float32))
        numbers.append(number)
    global_positions = torch.stack(positions)
    local_positions = frame.to_local(global_positions)
    atomic_numbers = torch.tensor(numbers, dtype=torch.long)
    features = _pocket_features(atoms)
    return PocketGraph(
        positions=local_positions,
        features=features,
        batch=torch.zeros(len(numbers), dtype=torch.long),
        atom_numbers=atomic_numbers,
        frame=frame,
    )


def ligand_graph_from_molecule(
    molecule: Chem.Mol, frame: CoordinateFrame
) -> LigandGraph:
    """Build a canonical ligand graph from one sanitized 3D RDKit molecule.

    :param molecule: Sanitized molecule with exactly one finite conformer.
    :param frame: Pocket centering frame for the molecule's global coordinates.
    :return: Ligand graph with one-hot atom/charge/bond categories.
    :rtype: LigandGraph
    :raises ValueError: If the molecule lacks a valid 3D conformer or contains
        chemistry outside the fixed ECloudFlow ligand vocabulary.
    """
    if molecule.GetNumAtoms() == 0 or molecule.GetNumConformers() != 1:
        raise ValueError("ligand must contain atoms and exactly one conformer")
    conformer = molecule.GetConformer()
    if not conformer.Is3D():
        raise ValueError("ligand conformer must be three-dimensional")
    global_positions = torch.tensor(conformer.GetPositions(), dtype=torch.float32)
    positions = frame.to_local(global_positions)
    vocab = ChemicalVocabulary.default_ligand()
    atom_types = torch.tensor(
        [vocab.atom_index(atom.GetSymbol()) for atom in molecule.GetAtoms()],
        dtype=torch.long,
    )
    charges = torch.tensor(
        [atom.GetFormalCharge() for atom in molecule.GetAtoms()], dtype=torch.long
    )
    charge_values = torch.tensor(vocab.formal_charges, dtype=torch.long)
    if bool(((charges[:, None] == charge_values[None, :]).sum(dim=1) != 1).any()):
        raise ValueError("ligand contains a formal charge outside the vocabulary")
    edges: list[tuple[int, int]] = []
    bond_values: list[int] = []
    bond_names = {
        Chem.BondType.SINGLE: "single",
        Chem.BondType.DOUBLE: "double",
        Chem.BondType.TRIPLE: "triple",
    }
    for bond in molecule.GetBonds():
        src, dst = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((src, dst))
        try:
            bond_values.append(vocab.bond_index(bond_names[bond.GetBondType()]))
        except KeyError as error:
            raise ValueError(
                f"unsupported ligand bond type: {bond.GetBondType()}"
            ) from error
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return LigandGraph(
        positions=positions,
        atom_types=atom_types,
        formal_charges=charges,
        halfedge_index=edge_index,
        bond_types=torch.tensor(bond_values, dtype=torch.long),
        batch=torch.zeros(molecule.GetNumAtoms(), dtype=torch.long),
    )


def _pocket_features(atoms: list[Any]) -> torch.Tensor:
    """Return the stable biochemical pocket feature schema.

    :param atoms: Biopython atoms retaining residue and PDB atom metadata.
    :return: Features ordered exactly as :data:`POCKET_FEATURE_NAMES`.
    :rtype: torch.Tensor

    The schema preserves element identity, all standard amino-acid residue
    classes plus ``OTHER``, backbone membership, finite PQR charge (or neutral
    zero fallback), residue-aware donor/acceptor flags, aromaticity,
    hydrophobicity, and explicit metal identity. Feature columns are never
    silently dropped when metadata are absent; deterministic fallback values
    are encoded instead.
    """
    elements = (
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
        "Na",
        "Mg",
        "K",
        "Ca",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
    )
    residues = (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "OTHER",
    )
    rows: list[list[float]] = []
    backbone_names = {"N", "CA", "C", "O", "OXT"}
    donor_by_residue = {
        "ARG": {"N"},
        "LYS": {"N"},
        "HIS": {"ND1", "NE2"},
        "SER": {"OG"},
        "THR": {"OG1"},
        "TYR": {"OH"},
        "CYS": {"SG"},
        "ASN": {"ND2"},
        "GLN": {"NE2"},
        "TRP": {"NE1"},
    }
    acceptor_by_residue = {
        "ASP": {"OD1", "OD2"},
        "GLU": {"OE1", "OE2"},
        "ASN": {"OD1"},
        "GLN": {"OE1"},
        "SER": {"OG"},
        "THR": {"OG1"},
        "TYR": {"OH"},
        "CYS": {"SG"},
        "HIS": {"ND1", "NE2"},
    }
    hydrophobic_residues = {
        "ALA",
        "VAL",
        "ILE",
        "LEU",
        "MET",
        "PHE",
        "TRP",
        "TYR",
        "CYS",
        "PRO",
        "GLY",
    }
    aromatic_residues = {"PHE", "TYR", "TRP", "HIS"}
    metal_symbols = {"Na", "Mg", "K", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
    for atom in atoms:
        element = str(getattr(atom, "element", "") or "").strip().title()
        residue = str(atom.get_parent().get_resname()).strip().upper()
        residue = residue if residue in residues[:-1] else "OTHER"
        atom_name = str(atom.get_name()).strip().upper()
        pqr = getattr(atom, "pqr_charge", None)
        charge = (
            float(pqr)
            if isinstance(pqr, (int, float))
            and torch.isfinite(torch.tensor(float(pqr)))
            else 0.0
        )
        donor = float(atom_name in donor_by_residue.get(residue, set()))
        acceptor = float(atom_name in acceptor_by_residue.get(residue, set()))
        if atom_name == "N" and residue not in {"PRO", "OTHER"}:
            donor = 1.0
        if atom_name in {"O", "OXT"}:
            acceptor = 1.0
        if residue == "OTHER" and element in {"N", "O", "S"}:
            donor = float(element in {"N", "O", "S"} and atom_name not in {"O", "OXT"})
            acceptor = float(
                element in {"N", "O", "S"} and atom_name not in {"N", "NZ"}
            )
        rows.append(
            [float(element == item) for item in elements]
            + [float(residue == item) for item in residues]
            + [
                float(atom_name in backbone_names),
                charge,
                donor,
                acceptor,
                float(residue in aromatic_residues and element in {"C", "N"}),
                float(residue in hydrophobic_residues),
                float(element in metal_symbols),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)
