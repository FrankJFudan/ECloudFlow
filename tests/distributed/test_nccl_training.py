"""Server-only NCCL acceptance marker with a local skip boundary."""

import os

import pytest
import torch


@pytest.mark.server
def test_four_h100_nccl_visibility() -> None:
    """Require four visible CUDA devices only when explicitly requested."""
    if os.environ.get("ECLOUDFLOW_REQUIRE_NCCL") != "1":
        pytest.skip("set ECLOUDFLOW_REQUIRE_NCCL=1 on the H100 server")
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 4
