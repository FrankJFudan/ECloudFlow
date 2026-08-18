"""Tests for immutable canonical tensor contracts."""

from collections.abc import Callable

import pytest
import torch

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import (
    ComplexSample,
    ElectronField,
    FragmentCondition,
    GenerationCondition,
    LigandGraph,
    MolecularState,
    PocketGraph,
    SampleProvenance,
)
from ecloudflow.exceptions import ContractValidationError


def test_molecular_state_accepts_canonical_unordered_halfedges(
    molecular_state_factory: Callable[[int], MolecularState],
):
    state = molecular_state_factory()

    assert state.positions.shape == (3, 3)
    assert torch.equal(state.halfedge_index[0], torch.tensor([0, 0, 1]))


def test_molecular_state_rejects_self_edges_and_noncanonical_ordering(
    molecular_state_factory: Callable[[int], MolecularState],
):
    state = molecular_state_factory()

    with pytest.raises(ValueError, match="strictly smaller"):
        state.replace(halfedge_index=torch.tensor([[0, 1, 1], [1, 1, 2]]))
    with pytest.raises(ValueError, match="strictly smaller"):
        state.replace(halfedge_index=torch.tensor([[1, 0, 1], [0, 2, 2]]))


def test_molecular_state_rejects_cross_complex_halfedges(
    molecular_state_factory: Callable[[int], MolecularState],
):
    state = molecular_state_factory()

    with pytest.raises(ValueError, match="same complex"):
        state.replace(node_batch=torch.tensor([0, 0, 1], dtype=torch.long))


def test_ligand_graph_accepts_signed_formal_charges():
    ligand = LigandGraph(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atom_types=torch.tensor([6, 8], dtype=torch.long),
        formal_charges=torch.tensor([0, -1], dtype=torch.long),
        halfedge_index=torch.tensor([[0], [1]], dtype=torch.long),
        bond_types=torch.tensor([1], dtype=torch.long),
        batch=torch.tensor([0, 0], dtype=torch.long),
    )

    assert torch.equal(ligand.formal_charges, torch.tensor([0, -1]))


def test_generation_condition_rejects_field_dtype_and_batch_mismatches():
    pocket = PocketGraph(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        features=torch.tensor([[1.0], [2.0]]),
        batch=torch.tensor([0, 0], dtype=torch.long),
    )
    field = ElectronField(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        values=torch.tensor([[1.0]], dtype=torch.float64),
        mask=torch.tensor([True]),
        batch=torch.tensor([1], dtype=torch.long),
    )

    with pytest.raises(ContractValidationError, match="dtype"):
        GenerationCondition(pocket=pocket, pocket_field=field)


def test_generation_condition_rejects_field_frame_and_batch_mismatches():
    pocket_frame = CoordinateFrame(torch.zeros(3))
    pocket = PocketGraph(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        features=torch.tensor([[1.0], [2.0]]),
        batch=torch.tensor([0, 0], dtype=torch.long),
        frame=pocket_frame,
    )
    wrong_frame_field = ElectronField(
        positions=torch.tensor([[0.0, 0.0, 0.0]]),
        values=torch.tensor([[1.0]]),
        mask=torch.tensor([True]),
        batch=torch.tensor([0], dtype=torch.long),
        frame=CoordinateFrame(torch.ones(3)),
    )
    wrong_batch_field = ElectronField(
        positions=torch.tensor([[0.0, 0.0, 0.0]]),
        values=torch.tensor([[1.0]]),
        mask=torch.tensor([True]),
        batch=torch.tensor([1], dtype=torch.long),
        frame=pocket_frame,
    )

    with pytest.raises(ContractValidationError, match="frame"):
        GenerationCondition(pocket=pocket, pocket_field=wrong_frame_field)
    with pytest.raises(ContractValidationError, match="batch"):
        GenerationCondition(pocket=pocket, pocket_field=wrong_batch_field)


def test_generation_condition_rejects_fragment_in_a_different_coordinate_frame(
    molecular_state_factory: Callable[[int], MolecularState],
):
    pocket_frame = CoordinateFrame(torch.zeros(3))
    pocket = PocketGraph(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        features=torch.tensor([[1.0], [2.0]]),
        batch=torch.tensor([0, 0], dtype=torch.long),
        frame=pocket_frame,
    )
    reference = molecular_state_factory().replace(
        frame=CoordinateFrame(torch.tensor([5.0, 0.0, 0.0]))
    )
    fragment = FragmentCondition.from_atom_mask(
        torch.tensor([True, False, True]), reference
    )

    with pytest.raises(ContractValidationError, match="frame"):
        GenerationCondition(pocket=pocket, fragment=fragment)


def test_complex_sample_rejects_inconsistent_batch_membership():
    frame = CoordinateFrame(torch.zeros(3))
    pocket = PocketGraph(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        features=torch.tensor([[1.0], [2.0]]),
        batch=torch.tensor([0, 0], dtype=torch.long),
        frame=frame,
    )
    ligand = LigandGraph(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atom_types=torch.tensor([6, 8], dtype=torch.long),
        formal_charges=torch.tensor([0, -1], dtype=torch.long),
        halfedge_index=torch.tensor([[0], [1]], dtype=torch.long),
        bond_types=torch.tensor([1], dtype=torch.long),
        batch=torch.tensor([1, 1], dtype=torch.long),
    )

    with pytest.raises(ContractValidationError, match="batch"):
        ComplexSample(
            source_id="BATCH-MISMATCH",
            pocket=pocket,
            ligand=ligand,
            pocket_field=None,
            ligand_field=None,
            properties={},
            frame=frame,
            provenance=SampleProvenance(),
        )


def test_other_public_contracts_accept_consistent_local_complex():
    frame = CoordinateFrame(torch.zeros(3))
    pocket = PocketGraph(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        features=torch.tensor([[1.0], [2.0]]),
        batch=torch.tensor([0, 0], dtype=torch.long),
        frame=frame,
    )
    ligand = LigandGraph(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atom_types=torch.tensor([6, 8], dtype=torch.long),
        formal_charges=torch.tensor([0, -1], dtype=torch.long),
        halfedge_index=torch.tensor([[0], [1]], dtype=torch.long),
        bond_types=torch.tensor([1], dtype=torch.long),
        batch=torch.tensor([0, 0], dtype=torch.long),
    )
    field = ElectronField(
        positions=torch.tensor([[0.0, 0.0, 0.0]]),
        values=torch.tensor([[1.0]]),
        mask=torch.tensor([True]),
        batch=torch.tensor([0], dtype=torch.long),
        channel_names=("density",),
        frame=frame,
    )
    state = MolecularState(
        positions=ligand.positions,
        atom_logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        charge_logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        halfedge_index=ligand.halfedge_index,
        bond_logits=torch.tensor([[1.0, 0.0]]),
        electron_latent=torch.tensor([[0.0], [1.0]]),
        node_batch=ligand.batch,
        halfedge_batch=torch.tensor([0], dtype=torch.long),
        frame=frame,
    )
    fragment = FragmentCondition.from_atom_mask(torch.tensor([True, False]), state)
    condition = GenerationCondition(pocket=pocket, fragment=fragment)
    sample = ComplexSample(
        source_id="CONSISTENT",
        pocket=pocket,
        ligand=ligand,
        pocket_field=field,
        ligand_field=field,
        properties={"affinity": -7.5},
        frame=frame,
        provenance=SampleProvenance(),
        fragment=fragment,
    )

    assert condition.fragment is fragment
    assert sample.provenance.preprocessing_status == "complete"
