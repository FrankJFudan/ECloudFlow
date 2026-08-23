"""Integration tests for the Lightning training boundary."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecloudflow.config import LossConfig
from ecloudflow.ecloud.decoder import ElectronFieldDecoder, ElectronReconstruction
from ecloudflow.models import ModelPrediction
from ecloudflow.training import (
    ECloudFlowTrainingModule,
    ElectronDecoderContext,
    TrainingBatch,
    TrainingTargets,
    compute_ecloudflow_loss,
)


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


def _qm_batch() -> TrainingBatch:
    """Return one genuine-QM complex with every decoder objective observed."""
    batch = _batch()
    return TrainingBatch(
        state=batch.state,
        time=batch.time,
        condition=batch.condition,
        targets=dataclasses.replace(
            batch.targets,
            qm_mask=torch.ones(1, dtype=torch.bool),
            density=torch.zeros(1, 2),
            density_gradient=torch.zeros(1, 2, 3),
            field_mask=torch.ones(1, 2, dtype=torch.bool),
            electron_count=torch.zeros(1),
            dipole=torch.zeros(1, 3),
            latent_cycle=torch.zeros(1, 2, 1),
        ),
        decoder_context=ElectronDecoderContext(
            query_grid=torch.tensor([[[0.25, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            atom_mask=torch.ones(1, 2, dtype=torch.bool),
            flat_index=torch.tensor([[0, 1]]),
        ),
    )


class CountingDecoder(nn.Module):
    """Track whether inactive supervision causes an unnecessary decode."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def decode(self, *args: torch.Tensor) -> ElectronReconstruction:
        self.calls += 1
        raise AssertionError("inactive decoder must not be called")


class MixedQMBackbone(nn.Module):
    """Return two one-node complexes with independently differentiable centers."""

    def __init__(self) -> None:
        super().__init__()
        self.center_x = nn.Parameter(torch.tensor([0.2, 2.0]))
        self.latent = nn.Parameter(torch.tensor([[0.5], [0.7]]))

    def forward(
        self, state: object, time: torch.Tensor, condition: object
    ) -> ModelPrediction:
        del state, condition
        centers = torch.stack(
            (
                self.center_x,
                torch.zeros_like(self.center_x),
                torch.zeros_like(self.center_x),
            ),
            dim=-1,
        )
        zeros = centers * 0.0
        node_logits = torch.stack((self.center_x * 0.0, self.center_x * 0.0), -1)
        count_logits = torch.log_softmax(
            torch.stack(
                (self.center_x * 0.0, self.center_x * 0.0, self.center_x * 0.0),
                -1,
            ),
            -1,
        )
        return ModelPrediction(
            position_velocity=zeros,
            position_score=zeros,
            electron_velocity=self.latent * 0.0,
            electron_score=self.latent * 0.0,
            atom_logits=node_logits,
            charge_logits=node_logits,
            bond_logits=self.center_x.new_empty((0, 2)),
            count_logits=count_logits,
            affinity=self.center_x * 0.0,
            interaction_logits=self.center_x * 0.0,
            pocket_cache_key="mixed-qm",
            affinity_log_variance=self.center_x * 0.0,
            endpoint_positions=centers,
            endpoint_electron_latent=self.latent,
        )


def _mixed_qm_batch() -> TrainingBatch:
    """Return one observed and one masked QM row with invalid placeholders."""
    return TrainingBatch(
        state=_State(),
        time=torch.tensor([0.5, 0.5]),
        condition=object(),
        targets=TrainingTargets(
            position_velocity=torch.zeros(2, 3),
            position_score=torch.zeros(2, 3),
            electron_velocity=torch.zeros(2, 1),
            electron_score=torch.zeros(2, 1),
            atom_classes=torch.tensor([-1, -1]),
            charge_classes=torch.tensor([-1, -1]),
            bond_classes=torch.empty(0, dtype=torch.long),
            count_classes=torch.tensor([-1, -1]),
            editable_atom_mask=torch.zeros(2, dtype=torch.bool),
            editable_bond_mask=torch.empty(0, dtype=torch.bool),
            node_batch=torch.tensor([0, 1]),
            halfedge_index=torch.empty(2, 0, dtype=torch.long),
            halfedge_batch=torch.empty(0, dtype=torch.long),
            count_mask=torch.zeros(2, dtype=torch.bool),
            qm_mask=torch.tensor([True, False]),
            density=torch.tensor([[0.0, 0.0], [float("nan"), float("inf")]]),
            density_gradient=torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[float("nan"), 0.0, 0.0], [float("inf"), 0.0, 0.0]],
                ]
            ),
            field_mask=torch.ones(2, 2, dtype=torch.bool),
            electron_count=torch.tensor([0.0, float("nan")]),
            dipole=torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]),
            latent_cycle=torch.tensor([[[0.0]], [[float("nan")]]]),
            latent_cycle_mask=torch.ones(2, 1, dtype=torch.bool),
        ),
        decoder_context=ElectronDecoderContext(
            query_grid=torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[float("nan"), 0.0, 0.0], [float("inf"), 0.0, 0.0]],
                ]
            ),
            atom_mask=torch.ones(2, 1, dtype=torch.bool),
            flat_index=torch.tensor([[0], [1]]),
        ),
    )


def test_decoder_is_not_required_without_enabled_genuine_qm_supervision() -> None:
    """Mutation caught: merely supplying context allocates a decoder reconstruction."""
    decoder = CountingDecoder()
    module = ECloudFlowTrainingModule(
        joint_backbone=TinyJointBackbone(),
        field_tokenizer=nn.Identity(),
        field_decoder=decoder,
        loss_config=LossConfig(),
    )
    batch = _batch()
    inactive = TrainingBatch(
        state=batch.state,
        time=batch.time,
        condition=batch.condition,
        targets=batch.targets,
        decoder_context=ElectronDecoderContext(
            query_grid=torch.full((1, 2, 3), float("nan")),
            atom_mask=torch.ones(1, 2, dtype=torch.bool),
            flat_index=torch.tensor([[0, 1]]),
        ),
    )

    prediction = module(inactive)

    assert decoder.calls == 0
    assert prediction.electron_reconstruction is None


def test_enabled_qm_decoder_requires_explicit_context() -> None:
    """Mutation caught: active QM forward silently deferred a missing context error."""
    module = ECloudFlowTrainingModule(
        joint_backbone=TinyJointBackbone(),
        field_tokenizer=nn.Identity(),
        field_decoder=ElectronFieldDecoder(1, 0, 0, 1),
        loss_config=LossConfig(),
    )
    batch = _qm_batch()

    with pytest.raises(ValueError, match="decoder_context"):
        module(dataclasses.replace(batch, decoder_context=None))


def test_real_decoder_uses_predicted_centers_and_all_qm_terms_are_differentiable() -> (
    None
):
    """Mutation caught: external centers sever endpoint gradients from field losses."""
    decoder = ElectronFieldDecoder(n_radial=1, lmax=0, vector_dim=0, latent_dim=1)
    module = ECloudFlowTrainingModule(
        joint_backbone=TinyJointBackbone(),
        field_tokenizer=nn.Identity(),
        field_decoder=decoder,
        loss_config=LossConfig(),
    )
    batch = _qm_batch()
    prediction = module(batch)
    assert prediction.electron_reconstruction is not None
    endpoint = prediction.endpoint_positions
    endpoint.retain_grad()

    breakdown = compute_ecloudflow_loss(prediction, batch.targets, LossConfig())
    breakdown.raw["ecloud"].backward()

    assert endpoint.grad is not None
    assert torch.isfinite(endpoint.grad).all()
    assert torch.count_nonzero(endpoint.grad) > 0
    for name in ("density", "gradient", "electron_count", "dipole", "cycle"):
        assert breakdown.diagnostics.supervised_counts[f"ecloud_{name}"] > 0


def test_real_decoder_compacts_mixed_qm_rows_before_evaluation() -> None:
    """Mutation caught: non-QM invalid rows reached the real decoder and field math."""
    backbone = MixedQMBackbone()
    module = ECloudFlowTrainingModule(
        joint_backbone=backbone,
        field_tokenizer=nn.Identity(),
        field_decoder=ElectronFieldDecoder(1, 0, 0, 1),
        loss_config=LossConfig(),
    )
    batch = _mixed_qm_batch()
    prediction = module(batch)
    assert prediction.electron_reconstruction is not None
    assert torch.count_nonzero(prediction.electron_reconstruction.density[1]) == 0

    breakdown = compute_ecloudflow_loss(prediction, batch.targets, LossConfig())
    breakdown.raw["ecloud"].backward()

    assert torch.isfinite(breakdown.raw["ecloud"])
    assert backbone.center_x.grad is not None
    assert backbone.center_x.grad[0] != 0
    assert backbone.center_x.grad[1] == 0


def test_decoder_mapping_rejects_duplicates() -> None:
    """Mutation caught: duplicated flattened nodes silently fabricate padded atoms."""
    module = ECloudFlowTrainingModule(
        joint_backbone=TinyJointBackbone(),
        field_tokenizer=nn.Identity(),
        field_decoder=ElectronFieldDecoder(1, 0, 0, 1),
        loss_config=LossConfig(),
    )
    batch = _qm_batch()
    assert batch.decoder_context is not None
    duplicate = dataclasses.replace(
        batch.decoder_context, flat_index=torch.tensor([[0, 0]])
    )
    with pytest.raises(ValueError, match="duplicate"):
        module(dataclasses.replace(batch, decoder_context=duplicate))


@pytest.mark.parametrize(
    ("node_batch", "flat_index", "message"),
    [
        (torch.tensor([0, 1]), torch.tensor([[0, 1]]), "crosses"),
        (torch.tensor([0, 0]), torch.tensor([[0, -1]]), "misaligned"),
    ],
)
def test_decoder_mapping_rejects_cross_complex_and_incomplete_rows(
    node_batch: torch.Tensor, flat_index: torch.Tensor, message: str
) -> None:
    """Mutation caught: mapping validity was previously limited to index range."""
    module = ECloudFlowTrainingModule(
        joint_backbone=TinyJointBackbone(),
        field_tokenizer=nn.Identity(),
        field_decoder=ElectronFieldDecoder(1, 0, 0, 1),
        loss_config=LossConfig(),
    )
    batch = _qm_batch()
    assert batch.decoder_context is not None
    targets = dataclasses.replace(batch.targets, node_batch=node_batch)
    context = dataclasses.replace(
        batch.decoder_context,
        flat_index=flat_index,
        atom_mask=flat_index >= 0,
    )
    with pytest.raises(ValueError, match=message):
        module(dataclasses.replace(batch, targets=targets, decoder_context=context))


def test_real_decoder_runs_inside_lightning_optimizer_step(tmp_path) -> None:
    """Mutation caught: active decoder path worked manually but failed in Trainer closure."""
    decoder = ElectronFieldDecoder(1, 0, 0, 1)
    module = ECloudFlowTrainingModule(
        joint_backbone=TinyJointBackbone(),
        field_tokenizer=nn.Identity(),
        field_decoder=decoder,
        loss_config=LossConfig(),
        learning_rate=0.01,
    )
    before = next(decoder.parameters()).detach().clone()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=DataLoader([_qm_batch()], batch_size=None))

    assert not torch.equal(next(decoder.parameters()).detach(), before)
    assert module.ema.num_updates.item() == 1


class OverflowTrainingModule(ECloudFlowTrainingModule):
    """Produce an infinite gradient inside Lightning's real precision closure."""

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> torch.Tensor:
        del batch, batch_idx
        return self.joint_backbone.weight / torch.zeros(
            (), device=self.joint_backbone.weight.device
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA FP16 GradScaler unavailable"
)
def test_cuda_fp16_gradscaler_skip_does_not_update_ema(tmp_path) -> None:
    """Mutation caught: EMA advanced even when GradScaler skipped an overflow step."""
    module = OverflowTrainingModule(
        joint_backbone=TinyJointBackbone(),
        field_tokenizer=nn.Identity(),
        field_decoder=nn.Identity(),
        loss_config=LossConfig(),
    )
    before = module.joint_backbone.weight.detach().clone()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=DataLoader([_batch()], batch_size=None))

    assert torch.equal(module.joint_backbone.weight.detach().cpu(), before)
    assert module.ema.num_updates.item() == 0


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
