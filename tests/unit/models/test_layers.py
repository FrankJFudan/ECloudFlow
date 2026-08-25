"""Behavior tests for sparse equivariant model layers."""

import torch

from ecloudflow.models.heads import SymmetricPairHead
from ecloudflow.models.layers import (
    _candidate_pair_chunks,
    _native_radius_edges,
    radius_edges,
    segment_softmax,
)


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


def test_symmetric_pair_head_produces_independent_class_channels() -> None:
    """Mutation caught: broadcasting one pair scalar cannot alter class probabilities."""
    torch.manual_seed(21)
    head = SymmetricPairHead(scalar_dim=8, output_dim=5)
    source = torch.randn(4, 8)
    target = torch.randn(4, 8)
    distance = torch.tensor([0.5, 1.0, 1.5, 2.0])

    forward = head(source, target, distance)
    reverse = head(target, source, distance)

    assert forward.shape == (4, 5)
    assert torch.equal(forward, reverse)
    assert torch.any(forward.softmax(-1) != torch.full_like(forward, 0.2))


def test_segment_softmax_normalizes_each_uneven_destination_group() -> None:
    """Mutation caught: unnormalized cross sums scale with neighbor count."""
    logits = torch.tensor([0.0, 0.0, 0.0, torch.log(torch.tensor(3.0))])
    destination = torch.tensor([0, 1, 1, 1])

    weights = segment_softmax(logits, destination, count=2)

    assert torch.allclose(weights, torch.tensor([1.0, 0.2, 0.2, 0.6]))
    assert torch.allclose(
        torch.stack((weights[destination == 0].sum(), weights[destination == 1].sum())),
        torch.ones(2),
    )


def test_segment_softmax_supports_bfloat16_indexed_assignment() -> None:
    """Mixed-precision model messages retain normalized BF16 weights."""
    logits = torch.tensor([0.0, 1.0, -0.5, 0.25], dtype=torch.bfloat16)
    destination = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    weights = segment_softmax(logits, destination, count=2)

    assert weights.dtype is torch.bfloat16
    assert torch.allclose(
        weights[destination == 0].float().sum(), torch.ones((), dtype=torch.float32), atol=2e-3
    )
    assert torch.allclose(
        weights[destination == 1].float().sum(), torch.ones((), dtype=torch.float32), atol=2e-3
    )


def test_native_radius_fallback_matches_hand_derived_edges() -> None:
    """Mutation caught: chunk boundaries may drop or duplicate exact radius pairs."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    batch = torch.tensor([0, 0, 0, 1])

    edges = _native_radius_edges(positions, batch, cutoff=1.0, chunk_size=1)

    assert torch.equal(edges, torch.tensor([[0, 1], [1, 0]]))


def test_candidate_chunks_have_bounded_size_for_large_pocket() -> None:
    """Mutation caught: one global Cartesian product exceeds the declared chunk bound."""
    nodes = torch.arange(2048)
    chunks = list(_candidate_pair_chunks(nodes, chunk_size=7))

    assert len(chunks) == 293
    assert max(source.numel() for source, _ in chunks) <= 7 * nodes.numel()
    assert sum(source.numel() for source, _ in chunks) == nodes.numel() ** 2
