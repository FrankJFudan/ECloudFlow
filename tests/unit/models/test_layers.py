"""Behavior tests for sparse equivariant model layers."""

import torch

from ecloudflow.models.heads import SymmetricPairHead
from ecloudflow.models.layers import radius_edges


def test_radius_edges_exclude_cross_batch_and_out_of_cutoff_pairs() -> None:
    """Mutation caught: omitting either batch or cutoff filtering adds bad edges."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    batch = torch.tensor([0, 0, 1, 0])

    edge_index = radius_edges(positions, batch, cutoff=1.5)

    assert torch.equal(edge_index, torch.tensor([[0, 1], [1, 0]]))


def test_symmetric_pair_head_is_unchanged_when_endpoints_are_swapped() -> None:
    """Mutation caught: concatenating ordered endpoints breaks unordered bonds."""
    torch.manual_seed(2)
    head = SymmetricPairHead(scalar_dim=8)
    source = torch.randn(4, 8)
    target = torch.randn(4, 8)
    distance = torch.tensor([0.5, 1.0, 1.5, 2.0])

    forward = head(source, target, distance)
    reverse = head(target, source, distance)

    assert forward.shape == (4,)
    assert torch.equal(forward, reverse)
