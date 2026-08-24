"""Bounded two-rank Gloo smoke coverage for checkpoint collectives."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch import nn

from ecloudflow.training.checkpoint import ReproducibleCheckpoint, write_rank_zero_json


class _DataState:
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def state_dict(self):
        return {
            "epoch": 0,
            "consumed_batches": self.rank + 1,
            "manifest_hash": "sha256:" + "2" * 64,
            "preprocessing_version": "ddp-test-v1",
        }


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="requires torchrun with two ranks",
)
def test_two_rank_gloo_checkpoint_collectives_and_atomic_publication(
    tmp_path: Path,
) -> None:
    """Mutation caught: rank-zero-only collection strands peers or drops rank state."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=15))
    try:
        rank = dist.get_rank()
        parameter = nn.Parameter(torch.tensor([float(rank + 1)]))
        wrapped = nn.parallel.DistributedDataParallel(nn.Linear(1, 1))
        (wrapped(parameter).sum()).backward()
        trainer = type(
            "Trainer",
            (),
            {
                "global_rank": rank,
                "global_step": 1,
                "current_epoch": 0,
                "datamodule": _DataState(rank),
            },
        )()
        module = type(
            "Module",
            (nn.Module,),
            {"ema": nn.Linear(1, 1), "loss_scaler": nn.Linear(1, 1)},
        )()
        callback = ReproducibleCheckpoint(
            resolved_config={"model": {"name": "tiny"}, "trainer": {"max_steps": 1}}
        )
        checkpoint = {
            "state_dict": {
                "ema.weight": torch.ones(1),
                "loss_scaler.count": torch.ones(1),
            }
        }
        callback.on_save_checkpoint(trainer, module, checkpoint)
        metadata = checkpoint["ecloudflow_reproducibility"]
        assert len(metadata["rng_by_rank"]) == 2
        assert [state["consumed_batches"] for state in metadata["data_by_rank"]] == [
            1,
            2,
        ]
        artifact = tmp_path / "ddp-artifact.json"
        write_rank_zero_json(trainer, artifact, {"world_size": 2})
        assert artifact.is_file()
    finally:
        dist.destroy_process_group()


@pytest.mark.server
@pytest.mark.skip(reason="NCCL and four H100 GPUs are server-only validation")
def test_four_h100_nccl_training_is_server_only() -> None:
    """Document the production validation boundary without a local hardware claim."""
