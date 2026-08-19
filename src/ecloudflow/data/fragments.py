"""Seeded, ring-aware online fragment task construction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import FragmentCondition, LigandGraph, MolecularState


class FragmentMode(str, Enum):
    """Enumerate fragment-conditioned lead-optimization objectives."""

    GROW = "grow"
    LINK = "link"
    REPLACE = "replace"
    MERGE = "merge"


@dataclass(frozen=True)
class FragmentTask:
    """Expose an exact fixed-pose condition together with its task mode.

    :param mode: Requested lead-optimization task objective.
    :param condition: Exact immutable condition consumed by the sampler.
    :return: Fragment task with convenience mask accessors.
    :rtype: FragmentTask
    """

    mode: FragmentMode
    condition: FragmentCondition

    @property
    def fixed_atom_mask(self) -> torch.Tensor:
        """Return the exact fixed atom mask from the condition."""
        return self.condition.fixed_atom_mask

    @property
    def fixed_bond_mask(self) -> torch.Tensor:
        """Return the exact fixed internal-halfedge mask from the condition."""
        return self.condition.fixed_bond_mask

    @property
    def fixed_coord_mask(self) -> torch.Tensor:
        """Return the exact fixed coordinate mask from the condition."""
        return self.condition.fixed_coord_mask

    @property
    def attachment_mask(self) -> torch.Tensor:
        """Return fixed atoms permitted to attach generated atoms."""
        return self.condition.attachment_mask


class FragmentTaskSampler:
    """Construct seeded BRICS/Murcko/linker/ring-aware fragment tasks.

    :param seed: Seed for deterministic task and cut selection.
    :return: Stateful deterministic fragment-task sampler.
    :rtype: FragmentTaskSampler

    The sampler makes masks over one canonical :class:`MolecularState`; no atom
    reordering or generated 3D placement occurs. When given a graph, callers
    must provide its pocket frame so fixed coordinates remain in the binding pose.
    """

    def __init__(self, seed: int = 0) -> None:
        """Initialize a reproducible local random source.

        :param seed: Integer random seed.
        :return: None.
        :rtype: None
        """
        self._random = random.Random(seed)

    def sample(
        self,
        ligand: Chem.Mol | LigandGraph | MolecularState,
        *,
        forced_mode: FragmentMode | None = None,
        frame: CoordinateFrame | None = None,
    ) -> FragmentTask:
        """Create one exact fragment task without moving the input pose.

        :param ligand: RDKit molecule, canonical ligand graph, or molecular state.
        :param forced_mode: Optional task mode; omitted values use seeded choice.
        :param frame: Required pocket frame for a ``LigandGraph`` and optional
            for an RDKit molecule. A molecular state must already declare it.
        :return: Task with exact atom, bond, coordinate, and attachment masks.
        :rtype: FragmentTask
        :raises ValueError: If the ligand has no atoms, no valid frame, or its
            topology cannot create a non-empty fixed fragment.

        Ring bonds are never selected as graph cuts. Ring systems are expanded
        atomically after cut selection, preventing partially fixed aromatic or
        aliphatic rings. BRICS and linker candidates are preferred; Murcko
        scaffold selection supplies a robust replacement fallback.
        """
        mode = forced_mode or self._random.choice(tuple(FragmentMode))
        if not isinstance(mode, FragmentMode):
            mode = FragmentMode(mode)
        state, molecule = _coerce_state(ligand, frame)
        atom_count = state.positions.shape[0]
        if atom_count == 0:
            raise ValueError("fragment tasks require at least one ligand atom")
        fixed = _select_fixed_atoms(molecule, atom_count, mode, self._random)
        fixed = _expand_complete_rings(molecule, fixed)
        if not bool(fixed.any()):
            fixed[self._random.randrange(atom_count)] = True
        if bool(fixed.all()) and atom_count > 1:
            fixed[_select_free_atom(molecule, fixed, self._random)] = False
            fixed = _expand_complete_rings(molecule, fixed)
        attachments = _attachment_mask(state.halfedge_index, fixed)
        components = _component_ids(state.halfedge_index, fixed)
        condition = FragmentCondition.from_atom_mask(
            fixed,
            state,
            attachment_mask=attachments,
            component_ids=components,
            task_id=mode.value,
        )
        return FragmentTask(mode=mode, condition=condition)


def _coerce_state(
    ligand: Chem.Mol | LigandGraph | MolecularState, frame: CoordinateFrame | None
) -> tuple[MolecularState, Chem.Mol | None]:
    """Convert supported ligand input to an exact canonical state."""
    if isinstance(ligand, MolecularState):
        if ligand.frame is None:
            raise ValueError("fixed-pose molecular state must declare a pocket frame")
        return ligand, None
    if isinstance(ligand, LigandGraph):
        if frame is None:
            raise ValueError("LigandGraph fragment tasks require the pocket frame")
        return _state_from_graph(ligand, frame), None
    if not isinstance(ligand, Chem.Mol):
        raise TypeError("ligand must be an RDKit Mol, LigandGraph, or MolecularState")
    molecule = Chem.RemoveHs(Chem.Mol(ligand), sanitize=True)
    if molecule.GetNumConformers() != 1 or not molecule.GetConformer().Is3D():
        raise ValueError("RDKit fragment ligand must have exactly one 3D conformer")
    positions = torch.tensor(
        molecule.GetConformer().GetPositions(), dtype=torch.float32
    )
    used_frame = frame or CoordinateFrame.from_pocket(positions)
    graph = _graph_from_rdkit(molecule, used_frame)
    return _state_from_graph(graph, used_frame), molecule


def _graph_from_rdkit(molecule: Chem.Mol, frame: CoordinateFrame) -> LigandGraph:
    """Create a graph with fixed vocabulary channels from an RDKit molecule."""
    vocabulary = ChemicalVocabulary.default_ligand()
    positions = frame.to_local(
        torch.tensor(molecule.GetConformer().GetPositions(), dtype=torch.float32)
    )
    atom_types = torch.tensor(
        [vocabulary.atom_index(atom.GetSymbol()) for atom in molecule.GetAtoms()],
        dtype=torch.long,
    )
    charges = torch.tensor(
        [atom.GetFormalCharge() for atom in molecule.GetAtoms()], dtype=torch.long
    )
    bond_names = {
        Chem.BondType.SINGLE: "single",
        Chem.BondType.DOUBLE: "double",
        Chem.BondType.TRIPLE: "triple",
        Chem.BondType.AROMATIC: "double",
    }
    pairs: list[tuple[int, int]] = []
    bond_types: list[int] = []
    for bond in molecule.GetBonds():
        pairs.append(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))
        bond_types.append(vocabulary.bond_index(bond_names[bond.GetBondType()]))
    order = sorted(range(len(pairs)), key=pairs.__getitem__)
    pairs = [pairs[index] for index in order]
    bond_types = [bond_types[index] for index in order]
    return LigandGraph(
        positions=positions,
        atom_types=atom_types,
        formal_charges=charges,
        halfedge_index=torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.empty((2, 0), dtype=torch.long),
        bond_types=torch.tensor(bond_types, dtype=torch.long),
        batch=torch.zeros(molecule.GetNumAtoms(), dtype=torch.long),
    )


def _state_from_graph(graph: LigandGraph, frame: CoordinateFrame) -> MolecularState:
    """Encode discrete graph targets as deterministic one-hot state logits."""
    vocabulary = ChemicalVocabulary.default_ligand()
    atom_logits = torch.nn.functional.one_hot(
        graph.atom_types, len(vocabulary.atom_symbols)
    ).to(graph.positions.dtype)
    charge_indices = torch.tensor(
        [
            vocabulary.charge_index(int(value))
            for value in graph.formal_charges.tolist()
        ],
        dtype=torch.long,
    )
    charge_logits = torch.nn.functional.one_hot(
        charge_indices, len(vocabulary.formal_charges)
    ).to(graph.positions.dtype)
    bond_logits = torch.nn.functional.one_hot(
        graph.bond_types, len(vocabulary.bond_classes)
    ).to(graph.positions.dtype)
    return MolecularState(
        positions=graph.positions,
        atom_logits=atom_logits,
        charge_logits=charge_logits,
        halfedge_index=graph.halfedge_index,
        bond_logits=bond_logits,
        electron_latent=torch.zeros(
            (graph.positions.shape[0], 1), dtype=graph.positions.dtype
        ),
        node_batch=graph.batch,
        halfedge_batch=torch.zeros(graph.halfedge_index.shape[1], dtype=torch.long),
        frame=frame,
    )


def _select_fixed_atoms(
    molecule: Chem.Mol | None, count: int, mode: FragmentMode, rng: random.Random
) -> torch.Tensor:
    """Choose a mode-specific seed mask while preferring chemically meaningful cuts."""
    if molecule is None:
        fixed = torch.zeros(count, dtype=torch.bool)
        fixed[: max(1, count // 2)] = True
        return fixed
    if mode is FragmentMode.REPLACE:
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
        match = molecule.GetSubstructMatch(scaffold) if scaffold.GetNumAtoms() else ()
        if match:
            fixed = torch.zeros(count, dtype=torch.bool)
            fixed[list(match)] = True
            return fixed
    cut_bonds = _non_ring_cut_bonds(molecule)
    if mode is FragmentMode.GROW and cut_bonds:
        return _component_mask_after_cuts(
            molecule, [rng.choice(cut_bonds)], rng, largest=True
        )
    if mode is FragmentMode.LINK and len(cut_bonds) >= 2:
        selected = rng.sample(cut_bonds, 2)
        return _two_component_mask(molecule, selected)
    if mode is FragmentMode.MERGE and cut_bonds:
        return _two_component_mask(molecule, [rng.choice(cut_bonds)])
    fixed = torch.zeros(count, dtype=torch.bool)
    ring_atoms = [
        index for ring in molecule.GetRingInfo().AtomRings() for index in ring
    ]
    if ring_atoms:
        fixed[sorted(set(ring_atoms))] = True
    else:
        fixed[: max(1, count // 2)] = True
    return fixed


def _non_ring_cut_bonds(molecule: Chem.Mol) -> list[int]:
    """Return BRICS/linker candidates excluding bonds inside every ring."""
    brics = {
        bond_index
        for pair in BRICS.FindBRICSBonds(molecule)
        for bond_index in [_bond_index(molecule, pair[0][0], pair[0][1])]
        if bond_index is not None
    }
    linker = {
        bond.GetIdx()
        for bond in molecule.GetBonds()
        if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE
    }
    return sorted(brics | linker)


def _bond_index(molecule: Chem.Mol, first: int, second: int) -> int | None:
    """Find one RDKit bond index for a BRICS atom pair."""
    bond = molecule.GetBondBetweenAtoms(first, second)
    return None if bond is None or bond.IsInRing() else bond.GetIdx()


def _component_mask_after_cuts(
    molecule: Chem.Mol, cuts: list[int], rng: random.Random, *, largest: bool
) -> torch.Tensor:
    """Return one whole connected component after removing non-ring bonds."""
    fragmented = Chem.FragmentOnBonds(molecule, cuts, addDummies=False)
    components = [
        list(component)
        for component in Chem.GetMolFrags(fragmented, asMols=False, sanitizeFrags=False)
    ]
    selected = max(components, key=len) if largest else rng.choice(components)
    mask = torch.zeros(molecule.GetNumAtoms(), dtype=torch.bool)
    mask[selected] = True
    return mask


def _two_component_mask(molecule: Chem.Mol, cuts: list[int]) -> torch.Tensor:
    """Preserve two separated components for linker and integration tasks."""
    fragmented = Chem.FragmentOnBonds(molecule, cuts, addDummies=False)
    components = sorted(
        Chem.GetMolFrags(fragmented, asMols=False, sanitizeFrags=False),
        key=len,
        reverse=True,
    )
    mask = torch.zeros(molecule.GetNumAtoms(), dtype=torch.bool)
    for component in components[:2]:
        mask[list(component)] = True
    return mask


def _expand_complete_rings(
    molecule: Chem.Mol | None, fixed: torch.Tensor
) -> torch.Tensor:
    """Expand a partially selected ring to its complete atom set."""
    if molecule is None:
        return fixed
    expanded = fixed.clone()
    for ring in molecule.GetRingInfo().AtomRings():
        indices = list(ring)
        if bool(expanded[indices].any()):
            expanded[indices] = True
    return expanded


def _select_free_atom(
    molecule: Chem.Mol | None, fixed: torch.Tensor, rng: random.Random
) -> int:
    """Choose a non-ring free atom before ring expansion when possible."""
    if molecule is not None:
        candidates = [
            atom.GetIdx() for atom in molecule.GetAtoms() if not atom.IsInRing()
        ]
        if candidates:
            return rng.choice(candidates)
    return rng.randrange(fixed.numel())


def _attachment_mask(edges: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    """Mark fixed endpoints of exact fixed-to-free canonical halfedges."""
    attachments = torch.zeros_like(fixed)
    if edges.numel() == 0:
        return attachments
    source, target = edges
    crossing = fixed[source] ^ fixed[target]
    attachments[source[crossing & fixed[source]]] = True
    attachments[target[crossing & fixed[target]]] = True
    return attachments


def _component_ids(edges: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    """Return stable connected-component labels for fixed atoms and free atoms."""
    count = fixed.numel()
    parents = list(range(count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first, second in edges.t().tolist():
        if bool(fixed[first] and fixed[second]):
            union(first, second)
    labels = torch.zeros(count, dtype=torch.long)
    roots: dict[int, int] = {}
    for index in range(count):
        if bool(fixed[index]):
            root = find(index)
            roots.setdefault(root, len(roots))
            labels[index] = roots[root]
        else:
            labels[index] = len(roots)
    return labels
