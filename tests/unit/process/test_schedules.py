"""Tests for deterministic continuous interpolant schedules."""

from __future__ import annotations

import pytest
import torch

from ecloudflow.process import CosineBridge, LinearBridge


@pytest.mark.parametrize("schedule", [LinearBridge(0.2), CosineBridge(0.2)])
def test_bridges_have_exact_endpoints_and_analytic_derivatives(
    schedule: object,
) -> None:
    """All bridges share the prior-to-data endpoint convention."""
    bridge = schedule
    zero = torch.tensor(0.0)
    one = torch.tensor(1.0)
    assert torch.allclose(bridge.data_weight(zero), zero)  # type: ignore[union-attr]
    assert torch.allclose(bridge.data_weight(one), one)  # type: ignore[union-attr]
    assert torch.allclose(bridge.noise_scale(zero), zero)  # type: ignore[union-attr]
    assert torch.allclose(bridge.noise_scale(one), zero)  # type: ignore[union-attr]

    time = torch.tensor(0.37, dtype=torch.float64)
    step = torch.tensor(1e-5, dtype=torch.float64)
    numerical = (
        bridge.data_weight(time + step) - bridge.data_weight(time - step)  # type: ignore[union-attr]
    ) / (2 * step)
    assert torch.allclose(bridge.data_weight_derivative(time), numerical, atol=1e-5)  # type: ignore[union-attr]


def test_bridge_rejects_invalid_noise_and_times() -> None:
    """Schedule validation rejects invalid public probability-time inputs."""
    with pytest.raises(ValueError, match="interior_noise"):
        LinearBridge(0.0)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        CosineBridge().noise_scale(torch.tensor(1.1))
