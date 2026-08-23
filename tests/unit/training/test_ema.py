"""Behavioral tests for optimizer-success-coupled exponential moving averages."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ecloudflow.training import ExponentialMovingAverage


def _model() -> nn.Linear:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 3.0]]))
    return model


def test_ema_updates_with_hand_derived_convex_average() -> None:
    """Mutation caught: reversed decay coefficients produce the wrong shadow value."""
    model = _model()
    ema = ExponentialMovingAverage(model, decay=0.75)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[5.0, 7.0]]))
    ema.update(model)

    assert torch.equal(ema.shadow_parameters()[0], torch.tensor([[2.0, 4.0]]))
    assert ema.num_updates.item() == 1


def test_skipped_step_does_not_update_ema() -> None:
    """Mutation caught: invoking the skip path mutates EMA state."""
    model = _model()
    ema = ExponentialMovingAverage(model, decay=0.5)
    before = ema.shadow_parameters()[0].clone()
    ema.update_after_step(model, step_succeeded=False)

    assert torch.equal(ema.shadow_parameters()[0], before)
    assert ema.num_updates.item() == 0


def test_store_copy_restore_is_separate_and_exact() -> None:
    """Mutation caught: validation weight swapping destroys live training weights."""
    model = _model()
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(9.0)
    ema.store(model)
    ema.copy_to(model)
    assert torch.equal(model.weight, torch.tensor([[1.0, 3.0]]))
    ema.restore(model)
    assert torch.equal(model.weight, torch.full((1, 2), 9.0))
    with pytest.raises(RuntimeError, match="store"):
        ema.restore(model)


def test_checkpoint_round_trip_preserves_shadow_and_counter() -> None:
    """Mutation caught: EMA buffers are omitted from module state dictionaries."""
    model = _model()
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    restored = ExponentialMovingAverage(_model(), decay=0.5)
    restored.load_state_dict(ema.state_dict())

    assert torch.equal(restored.shadow_parameters()[0], ema.shadow_parameters()[0])
    assert restored.num_updates.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_ema_follows_model_device_and_dtype() -> None:
    """Mutation caught: CPU or float32-only shadow state breaks CUDA BF16 modules."""
    model = _model().to(device="cuda", dtype=torch.bfloat16)
    ema = ExponentialMovingAverage(model, decay=0.9).to(
        device="cuda", dtype=torch.bfloat16
    )
    ema.update(model)
    shadow = ema.shadow_parameters()[0]
    assert shadow.device.type == "cuda"
    assert shadow.dtype == torch.bfloat16
