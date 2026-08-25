import torch

from ecloudflow.chemistry.decoder import (
    BondDecodeProblem,
    DecodeStatus,
    ExactBondDecoder,
    GreedyBondDecoder,
)
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.core import FragmentCondition, MolecularState
from ecloudflow.sampling.pipeline import _fragment_allowed_bond_mask


def _problem() -> BondDecodeProblem:
    vocab = ChemicalVocabulary.default_ligand()
    atoms = torch.full((3, len(vocab.atom_symbols)), -20.0)
    atoms[:, vocab.atom_index("C")] = 20.0
    atoms[2, :] = -20.0
    atoms[2, vocab.atom_index("O")] = 20.0
    charges = torch.full((3, len(vocab.formal_charges)), -20.0)
    charges[:, vocab.charge_index(0)] = 20.0
    edges = torch.tensor([[0, 0, 1], [1, 2, 2]])
    bonds = torch.full((3, len(vocab.bond_classes)), -6.0)
    bonds[:, vocab.bond_index("none")] = 0.0
    bonds[0, vocab.bond_index("double")] = 6.0
    bonds[1:, vocab.bond_index("single")] = 5.0
    state = MolecularState(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]]),
        atom_logits=atoms,
        charge_logits=charges,
        halfedge_index=edges,
        bond_logits=bonds,
        electron_latent=torch.zeros((3, 1)),
        node_batch=torch.zeros(3, dtype=torch.long),
        halfedge_batch=torch.zeros(3, dtype=torch.long),
    )
    fixed = FragmentCondition.from_atom_mask(torch.tensor([True, True, False]), state)
    return BondDecodeProblem(
        state=state, vocabulary=vocab, fixed=fixed, timeout_seconds=2.0
    )


def test_exact_decoder_maximizes_feasible_graph_and_preserves_fixed_bond():
    problem = _problem()
    decoded = ExactBondDecoder(timeout_seconds=2.0).decode(problem)
    assert decoded.status == DecodeStatus.OPTIMAL
    assert decoded.bond_orders[0].item() == 2.0
    assert decoded.connected
    assert decoded.valence_valid


def test_greedy_fallback_is_explicitly_nonoptimal():
    decoded = GreedyBondDecoder().decode(_problem())
    assert decoded.status == DecodeStatus.FALLBACK_FEASIBLE
    assert decoded.status != DecodeStatus.OPTIMAL


def test_exact_decoder_rejects_overvalent_fixed_bonds():
    problem = _problem()
    bad_atom_logits = problem.fixed.reference.atom_logits.clone().fill_(-20.0)
    bad_atom_logits[:, problem.vocabulary.atom_index("O")] = 20.0
    bad_reference_bonds = problem.fixed.reference.bond_logits.clone().fill_(-20.0)
    bad_reference_bonds[0, problem.vocabulary.bond_index("triple")] = 20.0
    bad_reference = problem.fixed.reference.replace(
        atom_logits=bad_atom_logits,
        bond_logits=bad_reference_bonds,
    )
    bad_fixed = FragmentCondition.from_atom_mask(
        problem.fixed.fixed_atom_mask,
        bad_reference,
        attachment_mask=problem.fixed.attachment_mask,
        task_id=problem.fixed.task_id,
    )
    bad = problem.state.replace(
        bond_logits=problem.state.bond_logits.clone().fill_(-20.0)
    )
    problem = problem.__class__(
        state=bad,
        vocabulary=problem.vocabulary,
        fixed=bad_fixed,
        timeout_seconds=problem.timeout_seconds,
    )
    with __import__("pytest").raises(ValueError, match="fixed"):
        ExactBondDecoder().decode(
            problem.__class__(
                state=bad, vocabulary=problem.vocabulary, fixed=problem.fixed
            )
        )


def test_exact_decoder_respects_fragment_attachment_topology():
    """CP-SAT must not connect a free atom through a masked fixed atom.

    The fixed C=C scaffold contains two possible crossing edges to the free O
    atom.  Only fixed atom ``0`` is declared an attachment site.  Both edge
    logits intentionally prefer a single bond, so a successful result proves
    that the final topology mask, rather than model preference, controls the
    crossing choice.
    """
    base = _problem()
    fixed = FragmentCondition.from_atom_mask(
        base.fixed.fixed_atom_mask,
        base.fixed.reference,
        attachment_mask=torch.tensor([True, False, False]),
        task_id="grow",
    )
    allowed = _fragment_allowed_bond_mask(base.state, fixed)
    assert allowed is not None
    constrained = BondDecodeProblem(
        state=base.state,
        vocabulary=base.vocabulary,
        fixed=fixed,
        allowed_bond_mask=allowed,
        timeout_seconds=2.0,
    )

    decoded = ExactBondDecoder(timeout_seconds=2.0).decode(constrained)

    assert decoded.feasible
    # Edges are (0, 1), (0, 2), (1, 2); only (0, 2) may cross the scaffold.
    assert decoded.bond_orders[1].item() > 0.0
    assert decoded.bond_orders[2].item() == 0.0
