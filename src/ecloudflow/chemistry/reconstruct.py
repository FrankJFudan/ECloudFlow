"""RDKit reconstruction at the discrete ECloudFlow boundary."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from rdkit import Chem
from rdkit.Chem import AssignStereochemistryFrom3D

from ecloudflow.chemistry.decoder import BondDecodeResult, DecodeStatus
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core.types import MolecularState

_BOND_TYPE_BY_ORDER = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


def reconstruct_rdkit_molecule(
    state: MolecularState,
    decoded: BondDecodeResult,
    vocabulary: ChemicalVocabulary | None = None,
    *,
    atom_indices: torch.Tensor | Sequence[int] | None = None,
    charge_indices: torch.Tensor | Sequence[int] | None = None,
    sanitize: bool = True,
) -> Chem.Mol:
    """Build and sanitize an RDKit molecule from a feasible Kekule graph.

    :param state: Final state supplying atom/charge channels, sparse canonical
        halfedges, and binding-pose coordinates in angstroms.
    :param decoded: Feasible result from an exact or constrained fallback
        decoder.  Its bond orders must align with ``state.halfedge_index``.
    :param vocabulary: Ligand vocabulary defining channel-to-element mappings.
        Defaults to the standard ECloudFlow ligand vocabulary.
    :param atom_indices: Optional explicit atom-channel assignments.  The
        result assignments, then state argmax assignments, are used otherwise.
    :param charge_indices: Optional explicit charge-channel assignments.
    :param sanitize: Whether to run RDKit sanitization and aromaticity
        perception.  Disabling it is intended only for debugging malformed
        intermediate graphs.
    :return: RDKit molecule with one conformer containing the sampled pose.
    :rtype: rdkit.Chem.Mol
    :raises ValueError: If the graph is infeasible, dimensions are inconsistent,
        atom/charge channels are unsupported, or RDKit sanitization fails.

    Model bonds are Kekule single/double/triple classes.  Sanitization is
    intentionally performed after all bonds are added; RDKit then perceives
    aromatic rings from the alternating Kekule representation.  Coordinates
    are copied into a conformer before 3D stereochemistry assignment and are
    never relaxed or overwritten by this function.
    """
    if not isinstance(state, MolecularState):
        raise TypeError("state must be a MolecularState.")
    if decoded.status not in {
        DecodeStatus.OPTIMAL,
        DecodeStatus.FEASIBLE,
        DecodeStatus.FALLBACK_FEASIBLE,
    }:
        raise ValueError(
            f"cannot reconstruct a non-feasible decode: {decoded.status.value}"
        )
    if decoded.bond_orders.shape != (state.halfedge_index.shape[1],):
        raise ValueError("decoded bond_orders must have shape [E].")
    if not bool(torch.isfinite(decoded.bond_orders).all()):
        raise ValueError("decoded bond_orders must contain finite values.")
    if not decoded.connected or not decoded.valence_valid:
        raise ValueError("decoded graph is not connected and valence-valid.")
    vocab = vocabulary or ChemicalVocabulary.default_ligand()
    if vocab.domain != "ligand":
        raise ValueError("RDKit ligand reconstruction requires a ligand vocabulary.")
    atom_choice = _resolve_indices(
        atom_indices, decoded.atom_indices, state.atom_logits
    )
    charge_choice = _resolve_indices(
        charge_indices, decoded.charge_indices, state.charge_logits
    )
    if (
        atom_choice.shape != (state.positions.shape[0],)
        or charge_choice.shape != atom_choice.shape
    ):
        raise ValueError("atom and charge assignments must have shape [N].")
    if bool((atom_choice < 0).any()) or bool(
        (atom_choice >= len(vocab.atom_symbols)).any()
    ):
        raise ValueError("atom assignments contain an out-of-vocabulary index.")
    if bool((charge_choice < 0).any()) or bool(
        (charge_choice >= len(vocab.formal_charges)).any()
    ):
        raise ValueError("charge assignments contain an out-of-vocabulary index.")

    editable = Chem.RWMol()
    for atom_channel, charge_channel in zip(
        atom_choice.tolist(), charge_choice.tolist()
    ):
        atom = Chem.Atom(vocab.atom_symbol(int(atom_channel)))
        atom.SetFormalCharge(int(vocab.formal_charges[int(charge_channel)]))
        editable.AddAtom(atom)
    for edge_index, order_value in enumerate(decoded.bond_orders.tolist()):
        order = round(float(order_value))
        if order == 0:
            continue
        if order not in _BOND_TYPE_BY_ORDER:
            raise ValueError(f"unsupported decoded bond order: {order_value}")
        source, target = (
            int(value) for value in state.halfedge_index[:, edge_index].tolist()
        )
        if source == target or source < 0 or target >= editable.GetNumAtoms():
            raise ValueError("decoded halfedge endpoint is invalid.")
        if editable.GetBondBetweenAtoms(source, target) is not None:
            raise ValueError("decoded graph contains duplicate bonds.")
        editable.AddBond(source, target, _BOND_TYPE_BY_ORDER[order])

    molecule = editable.GetMol()
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    positions = state.positions.detach().to(device="cpu", dtype=torch.float64)
    for index, point in enumerate(positions.tolist()):
        conformer.SetAtomPosition(index, tuple(float(value) for value in point))
    molecule.AddConformer(conformer, assignId=True)
    if sanitize:
        try:
            Chem.SanitizeMol(molecule)
        except (RuntimeError, ValueError) as error:
            problems = Chem.DetectChemistryProblems(molecule)
            details = "; ".join(problem.Message() for problem in problems)
            suffix = f" ({details})" if details else ""
            raise ValueError(f"RDKit sanitization failed{suffix}") from error
        try:
            AssignStereochemistryFrom3D(molecule, confId=0, replaceExistingTags=True)
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                f"3D stereochemistry assignment failed: {error}"
            ) from error
    return molecule


def _resolve_indices(
    explicit: torch.Tensor | Sequence[int] | None,
    result: torch.Tensor | None,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Resolve categorical assignments without retaining graph autograd state."""
    source: torch.Tensor | Sequence[int]
    if explicit is not None:
        source = explicit
    elif result is not None:
        source = result
    else:
        return logits.argmax(dim=-1).to(dtype=torch.long, device="cpu")
    if isinstance(source, torch.Tensor):
        if source.ndim != 1:
            raise ValueError("categorical assignments must be one-dimensional.")
        return source.detach().to(dtype=torch.long, device="cpu")
    return torch.tensor(tuple(int(value) for value in source), dtype=torch.long)


__all__ = ["reconstruct_rdkit_molecule"]
