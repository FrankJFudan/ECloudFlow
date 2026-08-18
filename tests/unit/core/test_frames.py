"""Tests for centered pocket coordinate frames."""

import pytest
import torch

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.exceptions import CoordinateFrameError


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


def test_coordinate_frame_round_trips_a_nonidentity_proper_rotation():
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    frame = CoordinateFrame(torch.tensor([2.0, -1.0, 0.5]), rotation)
    local_points = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.0, 1.0]])

    assert torch.allclose(frame.to_local(frame.to_global(local_points)), local_points)


def test_coordinate_frame_rejects_reflections_with_typed_exception():
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))

    with pytest.raises(CoordinateFrameError, match="proper rotation"):
        CoordinateFrame(torch.zeros(3), reflection)


def test_coordinate_frame_accepts_float16_cpu_without_raw_linalg_errors():
    frame = CoordinateFrame(torch.zeros(3, dtype=torch.float16))

    assert frame.rotation is not None
    assert frame.rotation.dtype is torch.float16
