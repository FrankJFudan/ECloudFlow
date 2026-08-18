"""Tests for immutable canonical tensor contracts."""

from collections.abc import Callable

import pytest
import torch

from ecloudflow.core.types import MolecularState


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
