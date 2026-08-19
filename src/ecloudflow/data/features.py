"""Conversion helpers from parsed structures to canonical tensor graphs."""

from __future__ import annotations

from typing import Any

import torch
from rdkit import Chem

from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import LigandGraph, PocketGraph


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
    features = _pocket_features(atomic_numbers)
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


def _pocket_features(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Return deterministic one-hot plus coarse chemistry pocket features."""
    symbols = (6, 7, 8, 16, 15, 17, 35, 53, 26, 30)
    one_hot = torch.stack(
        [(atomic_numbers == number).to(torch.float32) for number in symbols], dim=1
    )
    donor = (
        torch.isin(atomic_numbers, torch.tensor([7, 8, 16]))
        .to(torch.float32)
        .unsqueeze(1)
    )
    acceptor = donor.clone()
    hydrophobic = (
        torch.isin(atomic_numbers, torch.tensor([6, 16])).to(torch.float32).unsqueeze(1)
    )
    return torch.cat((one_hot, donor, acceptor, hydrophobic), dim=1)
