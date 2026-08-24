"""Integration tests for reproducible checkpoint metadata and stream resume."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import lightning as L
import pytest
import torch
from lightning import LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader

from ecloudflow.config import DataConfig, LossConfig
from ecloudflow.data.datamodule import ECloudDataModule
from ecloudflow.data.parsers import build_complex_sample
from ecloudflow.data.shards import ShardWriter
from ecloudflow.exceptions import DataValidationError
from ecloudflow.models import ModelPrediction
from ecloudflow.training.callbacks import NonFiniteDiagnostics
from ecloudflow.training.checkpoint import (
    CheckpointStateError,
    ReproducibleCheckpoint,
    assert_resume_compatible,
    atomic_write_json,
    capture_rng_state,
    restore_rng_state,
)
from ecloudflow.training.module import ECloudFlowTrainingModule
from ecloudflow.training.types import TrainingBatch, TrainingTargets


class TinyBackbone(nn.Module):
    """Return a complete prediction controlled by one trainable scalar."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, state, time, condition) -> ModelPrediction:
        del state, time, condition
        value = self.weight
        node = value.expand(1)
        xyz = value.expand(1, 3)
        return ModelPrediction(
            position_velocity=xyz,
            position_score=xyz,
            electron_velocity=node[:, None],
            electron_score=node[:, None],
            atom_logits=torch.stack((node, -node), dim=-1),
            charge_logits=torch.stack((node, -node), dim=-1),
            bond_logits=value.new_empty((0, 2)),
            count_logits=torch.log_softmax(torch.stack((node, -node), dim=-1), dim=-1),
            affinity=node,
            affinity_log_variance=torch.zeros_like(node),
            interaction_logits=node,
            endpoint_positions=xyz,
            endpoint_electron_latent=node[:, None],
            pocket_cache_key="tiny",
        )


def _batch(target: float = 1.0) -> TrainingBatch:
    zeros = torch.zeros(1)
    targets = TrainingTargets(
        position_velocity=torch.full((1, 3), target),
        position_score=torch.full((1, 3), target),
        electron_velocity=torch.full((1, 1), target),
        electron_score=torch.full((1, 1), target),
        atom_classes=torch.ones(1, dtype=torch.long),
        charge_classes=torch.ones(1, dtype=torch.long),
        bond_classes=torch.empty(0, dtype=torch.long),
        count_classes=torch.ones(1, dtype=torch.long),
        editable_atom_mask=torch.ones(1, dtype=torch.bool),
        editable_bond_mask=torch.empty(0, dtype=torch.bool),
        node_batch=torch.zeros(1, dtype=torch.long),
        halfedge_index=torch.empty(2, 0, dtype=torch.long),
        halfedge_batch=torch.empty(0, dtype=torch.long),
        count_mask=torch.ones(1, dtype=torch.bool),
        qm_mask=torch.zeros(1, dtype=torch.bool),
    )
    return TrainingBatch(
        state=object(), time=zeros, condition=object(), targets=targets
    )


class ReplayDataModule(LightningDataModule):
    """Small deterministic stateful loader for genuine Lightning resume tests."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
        self.consumed_batches = 0

    def train_dataloader(self):
        batches = [_batch(1.0), _batch(2.0), _batch(3.0)]
        return DataLoader(batches[self.consumed_batches :], batch_size=None)

    def mark_batch_consumed(self, count: int = 1) -> None:
        self.consumed_batches += count

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "consumed_batches": self.consumed_batches,
            "manifest_hash": "sha256:" + "1" * 64,
            "preprocessing_version": "test-v1",
        }

    def load_state_dict(self, state_dict):
        self.epoch = int(state_dict["epoch"])
        self.consumed_batches = int(state_dict["consumed_batches"])


def _run(root: Path, max_steps: int, checkpoint: Path | None = None):
    torch.manual_seed(19)
    base_loss = LossConfig()
    loss_config = base_loss.model_copy(
        update={
            "ecloud": base_loss.ecloud.model_copy(update={"weight": 0.0}),
            "chem": base_loss.chem.model_copy(update={"weight": 0.0}),
            "interaction": base_loss.interaction.model_copy(update={"weight": 0.0}),
        }
    )
    module = ECloudFlowTrainingModule(
        joint_backbone=TinyBackbone(),
        field_tokenizer=nn.Linear(1, 1),
        field_decoder=nn.Linear(1, 1),
        loss_config=loss_config,
        learning_rate=0.01,
    )
    data = ReplayDataModule()
    callback = ReproducibleCheckpoint(
        resolved_config={
            "model": {"name": "tiny"},
            "data": {"dataset": "test"},
            "loss": loss_config.model_dump(mode="json"),
            "trainer": {"max_steps": max_steps, "output_dir": str(root)},
        }
    )
    native_checkpoint = ModelCheckpoint(
        dirpath=root / "checkpoints",
        filename="step-{step}",
        every_n_train_steps=1,
        save_top_k=-1,
        save_last=True,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        logger=False,
        callbacks=[callback, native_checkpoint],
        enable_checkpointing=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=root,
    )
    trainer.fit(module, datamodule=data, ckpt_path=checkpoint)
    output = Path(native_checkpoint.last_model_path)
    assert output.is_file()
    return module, data, output


def test_resumed_two_step_run_matches_uninterrupted(tmp_path: Path) -> None:
    """Mutation caught: resume omits optimizer, EMA, scaler, RNG, or progress state."""
    full, full_data, _ = _run(tmp_path / "full", 2)
    _, _, partial_checkpoint = _run(tmp_path / "partial", 1)
    resumed, resumed_data, _ = _run(tmp_path / "resumed", 2, partial_checkpoint)

    for name, value in full.state_dict().items():
        torch.testing.assert_close(
            value, resumed.state_dict()[name], atol=1e-6, rtol=1e-6
        )
    assert full_data.epoch == resumed_data.epoch
    assert resumed_data.consumed_batches == 2


def test_resume_projection_allows_only_operational_and_termination_changes() -> None:
    """Mutation caught: a changed model/data/loss setting is accepted as compatible."""
    saved = {
        "model": {"width": 32},
        "data": {"dataset": "pdbbind"},
        "trainer": {"max_steps": 1, "output_dir": "old", "precision": "32-true"},
    }
    current = {
        "model": {"width": 32},
        "data": {"dataset": "pdbbind"},
        "trainer": {"max_steps": 2, "output_dir": "new", "precision": "32-true"},
    }
    assert_resume_compatible(saved, current)
    current["model"]["width"] = 64
    with pytest.raises(CheckpointStateError, match="semantic configuration"):
        assert_resume_compatible(saved, current)


def test_rng_round_trip_restores_cpu_sequence() -> None:
    """Mutation caught: resume restores metadata but not the rank-local CPU generator."""
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = torch.rand(4)
    torch.manual_seed(99)
    restore_rng_state(state)
    torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)


def test_atomic_json_failure_preserves_published_artifact(tmp_path: Path) -> None:
    """Mutation caught: a failed artifact serialization truncates the prior file."""
    path = tmp_path / "diagnostic.json"
    atomic_write_json(path, {"status": "complete"})
    with pytest.raises(TypeError):
        atomic_write_json(path, {"bad": object()})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "complete"}
    assert list(tmp_path.glob(".diagnostic.json.*.tmp")) == []


def test_nan_diagnostics_are_bounded_and_stop_at_threshold(tmp_path: Path) -> None:
    """Mutation caught: repeated nonfinite outputs emit unbounded files or continue."""
    callback = NonFiniteDiagnostics(
        artifact_dir=tmp_path, failure_threshold=2, max_artifacts=1
    )
    trainer = type("Trainer", (), {"global_rank": 0, "global_step": 3})()
    module = nn.Linear(1, 1)
    callback.on_train_batch_end(trainer, module, torch.tensor(float("nan")), None, 0)
    with pytest.raises(FloatingPointError, match="threshold"):
        callback.on_train_batch_end(
            trainer, module, torch.tensor(float("inf")), None, 1
        )
    assert len(list(tmp_path.glob("nonfinite-*.json"))) == 1


def test_multiworker_resume_replays_only_unconsumed_batches(tmp_path: Path) -> None:
    """Mutation caught: worker prefetch is mistaken for trainer-consumed progress."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures/complex"
    template = build_complex_sample(
        fixture / "toy_pocket.pdb",
        fixture / "toy_ligand.sdf",
        sample_id="resume-template",
        build_fields=False,
    )
    samples = [
        dataclasses.replace(template, source_id=f"resume-{index}") for index in range(8)
    ]
    ShardWriter(max_samples_per_shard=1).write(samples, tmp_path / "data")
    config = DataConfig(
        shard_dir=str(tmp_path / "data"),
        partition="all",
        batch_size=1,
        num_workers=2,
        persistent_workers=False,
        prefetch_factor=2,
        shuffle_buffer=0,
        pin_memory=False,
    )
    baseline = ECloudDataModule(config)
    expected = [batch[0].source_id for batch in baseline.train_dataloader()]

    interrupted = ECloudDataModule(config)
    iterator = iter(interrupted.train_dataloader())
    prefix = []
    for _ in range(3):
        prefix.append(next(iterator)[0].source_id)
        interrupted.mark_batch_consumed()
    state = interrupted.state_dict()
    assert state["consumed_batches"] == 3

    resumed = ECloudDataModule(config)
    resumed.load_state_dict(state)
    suffix = [batch[0].source_id for batch in resumed.train_dataloader()]
    assert prefix + suffix == expected


def test_preprocessing_mismatch_fails_before_resumed_iteration(tmp_path: Path) -> None:
    """Mutation caught: equal paths/manifests accept a changed preprocessing schema."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures/complex"
    sample = build_complex_sample(
        fixture / "toy_pocket.pdb",
        fixture / "toy_ligand.sdf",
        sample_id="preprocessing-version",
        build_fields=False,
    )
    ShardWriter(preprocessing_version="current-v2").write([sample], tmp_path)
    module = ECloudDataModule(
        DataConfig(shard_dir=str(tmp_path), num_workers=0, pin_memory=False)
    )
    module.load_state_dict(
        {
            "epoch": 0,
            "consumed_batches": 0,
            "manifest_hash": None,
            "preprocessing_version": "old-v1",
        }
    )
    with pytest.raises(DataValidationError, match="preprocessing version mismatch"):
        module.setup()
