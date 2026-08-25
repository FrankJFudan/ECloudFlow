"""Tests for executable Lightning runtime assembly and preflight behavior."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from ecloudflow.config import load_config
from ecloudflow.training import runtime as runtime_module
from ecloudflow.training.ema import ExponentialMovingAverage
from ecloudflow.training.runtime import TrainingConfigurationError, run_training


def test_run_training_invokes_fit_with_native_resume_path(tmp_path, monkeypatch):
    """Runtime execution must delegate full state restoration to Trainer.fit."""
    checkpoint = tmp_path / "resume.ckpt"
    checkpoint.write_bytes(b"checkpoint boundary is mocked in this unit test")
    config = load_config().model_copy(
        update={
            "trainer": load_config().trainer.model_copy(
                update={"resume_from": str(checkpoint)}
            )
        }
    )
    trainer = SimpleNamespace(fit=Mock())
    expected = SimpleNamespace(
        trainer=trainer,
        module=object(),
        datamodule=object(),
        checkpoint_dir=tmp_path / "checkpoints",
    )
    build = Mock(return_value=expected)
    monkeypatch.setattr(runtime_module, "build_training_runtime", build)

    observed = run_training(config, tmp_path / "run")

    assert observed is expected
    build.assert_called_once_with(config, tmp_path / "run")
    trainer.fit.assert_called_once_with(
        expected.module,
        datamodule=expected.datamodule,
        ckpt_path=str(checkpoint),
    )


def test_explicit_gpu_shortfall_fails_before_trainer_creation(tmp_path, monkeypatch):
    """An impossible four-GPU request must fail before DataModule or Trainer work."""
    base = load_config()
    config = base.model_copy(
        update={
            "trainer": base.trainer.model_copy(
                update={"accelerator": "gpu", "devices": 4}
            )
        }
    )
    trainer = Mock(side_effect=AssertionError("Trainer was constructed"))
    data = Mock(side_effect=AssertionError("DataModule was constructed"))
    monkeypatch.setattr(runtime_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime_module.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(runtime_module.L, "Trainer", trainer)
    monkeypatch.setattr(runtime_module, "ECloudDataModule", data)

    with pytest.raises(TrainingConfigurationError, match="requested 4 GPU"):
        runtime_module.build_training_runtime(config, tmp_path / "run")

    trainer.assert_not_called()
    data.assert_not_called()


class _TransferFixture(nn.Module):
    """Minimal three-group module matching the stage-transfer contract."""

    def __init__(self) -> None:
        super().__init__()
        self.joint_backbone = nn.Linear(2, 3)
        self.field_tokenizer = nn.Linear(3, 4)
        self.field_decoder = nn.Linear(4, 2)
        self.ema = ExponentialMovingAverage(self, decay=0.9)


def test_stage_initialization_loads_only_model_groups_and_resets_ema(tmp_path):
    """Prior-stage transfer starts fresh state around exact model weights."""
    source = _TransferFixture()
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(float(index))
    checkpoint = tmp_path / "stage.ckpt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    target = _TransferFixture()
    target.ema.num_updates.fill_(11)

    runtime_module._initialize_stage_weights(target, checkpoint)

    for group_name in runtime_module._TRANSFER_GROUPS:
        source_group = getattr(source, group_name)
        target_group = getattr(target, group_name)
        for expected, observed in zip(
            source_group.parameters(), target_group.parameters(), strict=True
        ):
            assert torch.equal(expected, observed)
    assert target.ema.num_updates.item() == 0
    for (_, parameter), shadow in zip(
        target.named_parameters(), target.ema.shadow_parameters(), strict=True
    ):
        assert torch.equal(parameter, shadow)


def test_stage_initialization_rejects_incomplete_checkpoint(tmp_path):
    """A missing scientific group must fail before Trainer construction."""
    checkpoint = tmp_path / "incomplete.ckpt"
    torch.save({"state_dict": {"joint_backbone.weight": torch.ones(3, 2)}}, checkpoint)

    with pytest.raises(TrainingConfigurationError, match="incompatible|missing"):
        runtime_module._initialize_stage_weights(_TransferFixture(), checkpoint)
