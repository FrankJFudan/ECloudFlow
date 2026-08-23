"""Tests for categorical simplex paths and endpoint cross entropy."""

from __future__ import annotations

import pytest
import torch

from ecloudflow.process import CategoricalPath


def test_categorical_path_stays_on_simplex_and_ignores_fixed_nodes() -> None:
    """Interpolated probabilities normalize and fixed entries stay exact."""
    path = CategoricalPath(num_classes=5, prior=torch.ones(5) / 5)
    target = torch.tensor([1, 3, 2])
    fixed = torch.tensor([True, False, False])
    sample = path.sample(target, torch.tensor(0.6), fixed_mask=fixed)
    assert torch.allclose(sample.probabilities.sum(-1), torch.ones(3))
    assert sample.classes[0] == target[0]
    assert torch.equal(
        sample.probabilities[0], torch.nn.functional.one_hot(target[0], 5)
    )


def test_categorical_endpoints_and_seeded_class_draws() -> None:
    """Time zero is the prior, time one is data, and generator draws repeat."""
    path = CategoricalPath(num_classes=3, prior=torch.tensor([0.2, 0.3, 0.5]))
    target = torch.tensor([0, 2])
    start = path.sample(
        target, torch.tensor(0.0), generator=torch.Generator().manual_seed(4)
    )
    finish = path.sample(
        target, torch.tensor(1.0), generator=torch.Generator().manual_seed(4)
    )
    assert torch.allclose(
        start.probabilities, torch.tensor([[0.2, 0.3, 0.5]]).expand(2, -1)
    )
    assert torch.equal(finish.classes, target)
    first = path.sample(
        target, torch.tensor(0.4), generator=torch.Generator().manual_seed(8)
    )
    second = path.sample(
        target, torch.tensor(0.4), generator=torch.Generator().manual_seed(8)
    )
    assert torch.equal(first.classes, second.classes)


def test_endpoint_cross_entropy_masks_fixed_entries_without_biased_denominator() -> (
    None
):
    """Only editable rows contribute one averaged data-endpoint CE value."""
    path = CategoricalPath(num_classes=3, prior=torch.ones(3) / 3)
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    target = torch.tensor([1, 2])
    loss = path.endpoint_loss(logits, target, fixed_mask=torch.tensor([True, False]))
    expected = torch.nn.functional.cross_entropy(logits[1:], target[1:])
    assert loss.dtype == torch.float32
    assert torch.allclose(loss, expected)


def test_categorical_path_rejects_invalid_priors_classes_and_masks() -> None:
    """Probability and vocabulary contract violations are explicit."""
    with pytest.raises(ValueError, match="sum to one"):
        CategoricalPath(num_classes=3, prior=torch.ones(3))
    path = CategoricalPath(num_classes=3, prior=torch.ones(3) / 3)
    with pytest.raises(ValueError, match="outside"):
        path.sample(torch.tensor([3]), torch.tensor(0.5))
    with pytest.raises(ValueError, match="torch.bool"):
        path.sample(torch.tensor([1]), torch.tensor(0.5), fixed_mask=torch.tensor([1]))


def test_bfloat16_simplex_and_endpoint_loss_are_finite_with_gradients() -> None:
    """Reduced-precision probabilities normalize and CE accumulates stably."""
    path = CategoricalPath(num_classes=3, prior=torch.ones(3, dtype=torch.bfloat16) / 3)
    target = torch.tensor([0, 2])
    sample = path.sample(target, torch.tensor(0.4, dtype=torch.bfloat16))
    assert torch.allclose(
        sample.probabilities.float().sum(-1), torch.ones(2), atol=2e-2
    )
    logits = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
    loss = path.endpoint_loss(logits, target)
    assert loss.dtype == torch.float32
    loss.backward()
    assert logits.grad is not None
