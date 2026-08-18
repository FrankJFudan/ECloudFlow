"""Tests for centered pocket coordinate frames."""

import pytest
import torch

from ecloudflow.core.frames import CoordinateFrame


def test_coordinate_frame_round_trip_and_centering():
    points = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    frame = CoordinateFrame.from_pocket(points)

    assert torch.allclose(frame.to_global(frame.to_local(points)), points)
    assert torch.allclose(frame.to_local(points).mean(dim=0), torch.zeros(3))


def test_coordinate_frame_rejects_rank_or_nonfinite_centers():
    with pytest.raises(ValueError, match="shape"):
        CoordinateFrame(torch.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        CoordinateFrame(torch.tensor([0.0, float("nan"), 0.0]))
