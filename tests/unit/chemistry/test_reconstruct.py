import torch

from ecloudflow.chemistry.decoder import BondDecodeResult, DecodeStatus
from ecloudflow.chemistry.reconstruct import reconstruct_rdkit_molecule
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core import MolecularState


def test_reconstruct_kekule_graph_sanitizes_and_perceives_aromaticity():
    vocab = ChemicalVocabulary.default_ligand()
    n = 6
    atoms = torch.full((n, len(vocab.atom_symbols)), -20.0)
    atoms[:, vocab.atom_index("C")] = 20.0
    charges = torch.full((n, len(vocab.formal_charges)), -20.0)
    charges[:, vocab.charge_index(0)] = 20.0
    pairs = [(i, i + 1) for i in range(n - 1)] + [(0, n - 1)]
    edges = torch.tensor(pairs, dtype=torch.long).T
    state = MolecularState(
        positions=torch.zeros((n, 3)),
        atom_logits=atoms,
        charge_logits=charges,
        halfedge_index=edges,
        bond_logits=torch.zeros((n, 4)),
        electron_latent=torch.zeros((n, 1)),
        node_batch=torch.zeros(n, dtype=torch.long),
        halfedge_batch=torch.zeros(n, dtype=torch.long),
    )
    result = BondDecodeResult(
        bond_orders=torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]),
        status=DecodeStatus.OPTIMAL,
        objective=0.0,
        connected=True,
        valence_valid=True,
    )
    molecule = reconstruct_rdkit_molecule(state, result, vocab)
    assert molecule.GetNumAtoms() == n
    assert all(atom.GetIsAromatic() for atom in molecule.GetAtoms())
