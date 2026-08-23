"""Integration tests for the Lightning training boundary."""

from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecloudflow.config import LossConfig
from ecloudflow.models import ModelPrediction
from ecloudflow.training import ECloudFlowTrainingModule, TrainingBatch, TrainingTargets


class TinyJointBackbone(nn.Module):
    """Real differentiable backbone implementing the typed model boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self, state: object, time: torch.Tensor, condition: object
    ) -> ModelPrediction:
        del state, condition
        value = self.weight
        node = value.expand(2, 1)
        vector = value.expand(2, 3)
        return ModelPrediction(
            position_velocity=vector,
            position_score=vector,
            electron_velocity=node,
            electron_score=node,
            atom_logits=torch.stack((value.expand(2), -value.expand(2)), -1),
            charge_logits=torch.stack((value.expand(2), -value.expand(2)), -1),
            bond_logits=torch.stack((value.expand(1), -value.expand(1)), -1),
            count_logits=torch.log_softmax(
                torch.stack((-value.expand(1), value.expand(1), -value.expand(1)), -1),
                -1,
            ),
            affinity=value.expand(time.shape[0]),
            interaction_logits=value.expand(time.shape[0]),
            pocket_cache_key="tiny",
            affinity_log_variance=torch.zeros_like(time) + value * 0.0,
            endpoint_positions=vector,
            endpoint_electron_latent=node,
        )


@dataclass(frozen=True)
class _State:
    """Small state accepted by the explicit test backbone."""


def _batch() -> TrainingBatch:
    return TrainingBatch(
        state=_State(),
        time=torch.tensor([0.5]),
        condition=object(),
        targets=TrainingTargets(
            position_velocity=torch.ones(2, 3),
            position_score=torch.ones(2, 3),
            electron_velocity=torch.ones(2, 1),
            electron_score=torch.ones(2, 1),
            atom_classes=torch.zeros(2, dtype=torch.long),
            charge_classes=torch.zeros(2, dtype=torch.long),
            bond_classes=torch.zeros(1, dtype=torch.long),
            count_classes=torch.ones(1, dtype=torch.long),
            editable_atom_mask=torch.ones(2, dtype=torch.bool),
            editable_bond_mask=torch.ones(1, dtype=torch.bool),
            node_batch=torch.zeros(2, dtype=torch.long),
            halfedge_index=torch.tensor([[0], [1]]),
            halfedge_batch=torch.zeros(1, dtype=torch.long),
            count_mask=torch.ones(1, dtype=torch.bool),
            qm_mask=torch.zeros(1, dtype=torch.bool),
            interaction=torch.ones(1),
            interaction_mask=torch.ones(1, dtype=torch.bool),
            affinity=torch.ones(1),
            affinity_mask=torch.ones(1, dtype=torch.bool),
            valence_limits=torch.full((2,), 4.0),
            bond_order_values=torch.tensor([0.0, 1.0]),
            bond_length_mean=torch.ones(1),
            bond_length_std=torch.ones(1),
            nonbonded_halfedge_index=torch.empty(2, 0, dtype=torch.long),
            protein_positions=torch.empty(0, 3),
            protein_batch=torch.empty(0, dtype=torch.long),
            ring_triplets=torch.empty(3, 0, dtype=torch.long),
            ring_angle_mean=torch.empty(0),
            ring_angle_std=torch.empty(0),
        ),
    )


def test_real_cpu_lightning_step_updates_parameters_and_ema(tmp_path) -> None:
    """Mutation caught: detached loss or missing optimizer step leaves weights unchanged."""
    module = ECloudFlowTrainingModule(
        joint_backbone=TinyJointBackbone(),
        field_tokenizer=nn.Identity(),
        field_decoder=nn.Identity(),
        loss_config=LossConfig(),
        learning_rate=0.1,
    )
    before = module.joint_backbone.weight.detach().clone()
    loader = DataLoader([_batch()], batch_size=None)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=loader)

    assert not torch.equal(module.joint_backbone.weight.detach(), before)
    assert module.ema.num_updates.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_bf16_lightning_step_smoke(tmp_path) -> None:
    """Mutation caught: fixed CPU state or unsupported precision breaks accelerator training."""
    module = ECloudFlowTrainingModule(
        joint_backbone=TinyJointBackbone(),
        field_tokenizer=nn.Identity(),
        field_decoder=nn.Identity(),
        loss_config=LossConfig(),
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=DataLoader([_batch()], batch_size=None))
    assert module.ema.num_updates.item() == 1
