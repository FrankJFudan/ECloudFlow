"""Lightning training module for joint sparse ECloudFlow optimization."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import lightning as L
import torch
from torch import nn

from ecloudflow.config import LossConfig
from ecloudflow.ecloud.decoder import ElectronReconstruction
from ecloudflow.models import ModelPrediction
from ecloudflow.training.ema import ExponentialMovingAverage
from ecloudflow.training.losses import RunningLossScaler, compute_ecloudflow_loss
from ecloudflow.training.types import (
    ElectronDecoderContext,
    TrainingBatch,
    TrainingTargets,
)


class ECloudFlowTrainingModule(L.LightningModule):  # type: ignore[misc]
    """Train the joint model with checkpointable normalization and EMA state.

    :param joint_backbone: Task 10-compatible module accepting molecular state,
        per-complex time ``[B]``, and generation condition, and returning a
        :class:`ModelPrediction` with flattened nodes and sparse halfedges.
    :param field_tokenizer: Real Task 8 tokenizer or an explicitly compatible
        module retained under the stable staged-training group name.
    :param field_decoder: Real Task 8 decoder or compatible module exposing
        ``decode(latent, centers, query_grid, mask)`` when QM labels are active.
    :param loss_config: Frozen six-component scientific objective configuration.
    :param learning_rate: Positive finite AdamW learning rate.
    :param weight_decay: Non-negative finite AdamW decoupled weight decay.
    :param ema_decay: Finite EMA coefficient in ``[0,1)``.
    :return: Lightning 2.5-compatible device-agnostic training module.
    :rtype: ECloudFlowTrainingModule
    :raises TypeError: If trainable groups/config are not typed modules/config.
    :raises ValueError: If optimizer or EMA hyperparameters are invalid.

    The stable attributes ``field_tokenizer``, ``field_decoder``, and
    ``joint_backbone`` are explicit stage-freezing groups for Task 12. A decoder
    context gathers differentiable packed endpoint tokens from flattened nodes
    into only the real padded Task 8 boundary; it never invents density tensors.
    ``training_step`` logs every raw/normalized/weighted component with
    ``sync_dist=True``. Running scales are persistent buffers whose detached
    sufficient statistics all-reduce under initialized DDP. EMA is updated in
    ``optimizer_step`` only when Lightning actually invokes a step and all
    produced gradients are finite; exceptions and AMP-overflow/non-finite skips
    cannot mutate it. The module performs no manual CUDA/rank/device operation.
    Warm-ups use explicit ``global_step`` and resume deterministically through
    Lightning checkpoint state. Mixed precision loss reductions remain float32.
    """

    def __init__(
        self,
        *,
        joint_backbone: nn.Module,
        field_tokenizer: nn.Module,
        field_decoder: nn.Module,
        loss_config: LossConfig,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 0.0,
        ema_decay: float = 0.999,
    ) -> None:
        super().__init__()
        for name, module in (
            ("joint_backbone", joint_backbone),
            ("field_tokenizer", field_tokenizer),
            ("field_decoder", field_decoder),
        ):
            if not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be a torch.nn.Module.")
        if not isinstance(loss_config, LossConfig):
            raise TypeError("loss_config must be LossConfig.")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative.")
        self.field_tokenizer = field_tokenizer
        self.field_decoder = field_decoder
        self.joint_backbone = joint_backbone
        self.loss_config = loss_config
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.loss_scaler = RunningLossScaler(
            decay=loss_config.normalization.decay,
            epsilon=loss_config.normalization.epsilon,
        )
        self.ema = ExponentialMovingAverage(self, decay=ema_decay)
        self.save_hyperparameters(
            {
                "loss_config": loss_config.model_dump(mode="json"),
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "ema_decay": ema_decay,
            }
        )

    def forward(self, batch: TrainingBatch) -> ModelPrediction:
        """Run joint prediction and optional real electron-field decoding.

        :param batch: Typed state/time/condition/targets plus optional decoder
            context. State nodes remain flattened and bonds remain unordered
            sparse halfedges; decoder padding is created only at its Task 8 API.
        :return: Model prediction, optionally replaced with a differentiable
            density/gradient/count/dipole/cycle reconstruction.
        :rtype: ModelPrediction
        :raises TypeError: If the batch/model return or decoder API is incompatible.
        :raises ValueError: If decoder mapping shape, dtype, device, or range fails.

        Model and inputs stay on Lightning-managed dtype/device. The gather and
        decode are deterministic, preserve autograd, fabricate no supervision,
        mutate no input, and perform no rank/device transfer. Endpoint geometry
        retains its documented first-order rather than guaranteed-clean meaning.
        """
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be TrainingBatch.")
        prediction = self.joint_backbone(batch.state, batch.time, batch.condition)
        if not isinstance(prediction, ModelPrediction):
            raise TypeError("joint_backbone must return ModelPrediction.")
        if not self._decoder_is_observed(batch.targets):
            return prediction
        if batch.decoder_context is None:
            raise ValueError(
                "enabled observed QM reconstruction terms require decoder_context."
            )
        reconstruction = self._decode_electrons(
            prediction, batch.decoder_context, batch.targets
        )
        return replace(prediction, electron_reconstruction=reconstruction)

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> torch.Tensor:
        """Compute, validate, and synchronously log one training objective.

        :param batch: Typed flattened molecular training batch on the Lightning device.
        :param batch_idx: Zero-based local dataloader batch index used only by Lightning.
        :return: Finite differentiable scalar total for automatic optimization.
        :rtype: torch.Tensor
        :raises TypeError: If the typed batch or prediction boundary is violated.
        :raises ValueError: If a scientific target/mask/context contract is invalid.
        :raises FloatingPointError: Before optimizer mutation if any active term
            is non-finite.

        The loss function applies exact masks, float32 mixed-precision reductions,
        DDP-synchronized detached scaler mutation, and explicit global-step warmups.
        Logging uses ``sync_dist=True`` for all six raw/normalized/weighted terms
        and the total. It creates no dense bond matrix, manual device move, or
        rank-local scientific state, and does not mutate the batch. Tensor shape
        and dtype contracts preserve each unordered halfedge. Fixed inputs and a
        resumed checkpoint ``global_step`` give deterministic warmups.
        """
        del batch_idx
        prediction = self(batch)
        breakdown = compute_ecloudflow_loss(
            prediction,
            batch.targets,
            self.loss_config,
            self.loss_scaler if self.loss_config.normalization.enabled else None,
            step=int(self.global_step),
        )
        for stage, values in (
            ("raw", breakdown.raw),
            ("normalized", breakdown.normalized),
            ("weighted", breakdown.weighted),
        ):
            for name, value in values.items():
                self.log(
                    f"train/{stage}/{name}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=True,
                )
        self.log(
            "train/loss",
            breakdown.total,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        return breakdown.total

    def transfer_batch_to_device(
        self, batch: TrainingBatch, device: torch.device, dataloader_idx: int
    ) -> TrainingBatch:
        """Functionally transfer immutable typed batch contracts to a strategy device.

        :param batch: Frozen nested training/core dataclasses and tensor leaves.
        :param device: Device selected by the active Lightning strategy.
        :param dataloader_idx: Loader index supplied by Lightning and otherwise unused.
        :return: Revalidated dataclass tree with tensor leaves on ``device``.
        :rtype: TrainingBatch
        :raises TypeError: If the root is not :class:`TrainingBatch`.

        Lightning 2.5's default recursive mover rejects frozen dataclasses. This
        hook recursively uses ``dataclasses.replace`` so Task 7 immutable state,
        frame, condition, target, and decoder contracts remain validated rather
        than mutated. Tensor dtype and autograd connectivity are preserved; only
        the strategy-selected device changes. Mappings/sequences retain their
        container type where possible. No CUDA device, rank, collective, random
        state, mask, sparse topology, or scientific value is chosen manually.
        """
        del dataloader_idx
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be TrainingBatch.")
        moved = _move_immutable(batch, device)
        assert isinstance(moved, TrainingBatch)
        return moved

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Construct AdamW over the currently trainable staged groups.

        :return: AdamW using typed learning rate and weight decay.
        :rtype: torch.optim.Optimizer
        :raises ValueError: If stage freezing leaves no trainable parameter.

        The optimizer owns no manual device/rank state; Lightning/DDP handles
        placement and gradient synchronization. Parameter iteration is stable and
        excludes persistent loss-scaler/EMA buffers. Construction is deterministic
        and does not mutate model tensors.
        """
        parameters = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        if not parameters:
            raise ValueError("training stage has no trainable parameters.")
        return torch.optim.AdamW(
            parameters, lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any | None = None,
    ) -> None:
        """Perform one Lightning optimizer step and conditionally update EMA.

        :param epoch: Current epoch supplied by Lightning.
        :param batch_idx: Current batch index supplied by Lightning.
        :param optimizer: Lightning-managed optimizer on its existing device.
        :param optimizer_closure: Closure that computes loss and gradients.
        :return: None after optimizer and, only on finite success, EMA mutation.
        :rtype: None
        :raises Exception: Propagates closure/optimizer failures without EMA update.

        Lightning 2.5 invokes this hook only at an actual accumulated optimizer
        boundary. The superclass owns precision-plugin/AMP behavior. After it
        returns, EMA updates only if at least one gradient exists and every
        gradient is finite; overflow/skipped/non-finite and raised steps leave EMA
        exact. No CUDA, rank, distributed collective, or dtype conversion occurs.
        Checkpoint mutation is deterministic for fixed synchronized gradients.
        """
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        gradients = [
            parameter.grad
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        succeeded = bool(gradients) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        self.ema.update_after_step(self, step_succeeded=succeeded)

    def _decoder_is_observed(self, targets: TrainingTargets) -> bool:
        """Return whether an enabled decoder subterm has genuine-QM observations."""
        loss = self.loss_config.ecloud
        step = int(self.global_step)
        if loss.warmup_end == loss.warmup_start:
            warmup = float(step >= loss.warmup_end)
        elif step <= loss.warmup_start:
            warmup = 0.0
        elif step >= loss.warmup_end:
            warmup = 1.0
        else:
            warmup = (step - loss.warmup_start) / (loss.warmup_end - loss.warmup_start)
        if not bool(targets.qm_mask.any()) or loss.weight == 0.0 or warmup == 0.0:
            return False
        field_observed = targets.field_mask is None or bool(
            (targets.qm_mask[:, None] & targets.field_mask).any()
        )
        return any(
            (
                self.loss_config.ecloud.density and field_observed,
                self.loss_config.ecloud.gradient and field_observed,
                self.loss_config.ecloud.electron_count,
                self.loss_config.ecloud.dipole,
                self.loss_config.ecloud.cycle,
            )
        )

    def _decode_electrons(
        self,
        prediction: ModelPrediction,
        context: ElectronDecoderContext,
        targets: TrainingTargets,
    ) -> ElectronReconstruction:
        """Gather mapped predicted tokens/centers into active QM decoder rows."""
        flat_latent = prediction.endpoint_electron_latent
        flat_centers = prediction.endpoint_positions
        index = context.flat_index
        mask = context.atom_mask
        if (
            index.shape != mask.shape
            or index.dtype != torch.long
            or mask.dtype != torch.bool
            or index.device != flat_latent.device
            or mask.device != flat_latent.device
        ):
            raise ValueError(
                "decoder flat_index/mask must be same-device [B, N] long/bool tensors."
            )
        if bool((index[mask] < 0).any()) or bool(
            (index[mask] >= flat_latent.shape[0]).any()
        ):
            raise ValueError(
                "physical decoder flat_index is outside flattened node range."
            )
        if (
            context.query_grid.ndim != 3
            or context.query_grid.shape[0] != index.shape[0]
            or context.query_grid.shape[2] != 3
        ):
            raise ValueError("decoder query_grid must have shape [B, G, 3].")
        if (
            context.query_grid.device != flat_latent.device
            or not context.query_grid.is_floating_point()
        ):
            raise ValueError(
                "decoder query_grid must be floating on prediction device."
            )
        if index.shape[0] != prediction.affinity.shape[0]:
            raise ValueError(
                "decoder mapping batch dimension must equal prediction batch size."
            )
        selected_index = index[mask]
        if selected_index.unique().numel() != selected_index.numel():
            raise ValueError("decoder mapping contains duplicate flattened nodes.")
        expected = torch.arange(flat_latent.shape[0], device=index.device)
        if selected_index.numel() != expected.numel() or not torch.equal(
            selected_index.sort().values, expected
        ):
            raise ValueError(
                "decoder mapping is misaligned with flattened model nodes."
            )
        rows = torch.arange(index.shape[0], device=index.device)[:, None].expand_as(
            index
        )
        if not torch.equal(targets.node_batch[selected_index], rows[mask]):
            raise ValueError("decoder mapping crosses declared complex membership.")
        safe_index = index.clamp_min(0)
        padded = flat_latent[safe_index]
        centers = flat_centers[safe_index]
        padded = torch.where(mask[..., None], padded, torch.zeros_like(padded))
        centers = torch.where(mask[..., None], centers, torch.zeros_like(centers))
        active_rows = targets.qm_mask
        active_padded = padded[active_rows]
        active_centers = centers[active_rows]
        active_query = context.query_grid[active_rows]
        active_mask = mask[active_rows]
        decode = getattr(self.field_decoder, "decode", None)
        if not callable(decode):
            raise TypeError("field_decoder must expose a compatible decode method.")
        reconstruction = decode(
            active_padded, active_centers, active_query, active_mask
        )
        if not isinstance(reconstruction, ElectronReconstruction):
            raise TypeError("field_decoder.decode must return ElectronReconstruction.")
        batch_size = index.shape[0]
        active_index = active_rows.nonzero(as_tuple=False).flatten()

        def scatter(value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
            base = value.new_zeros(shape)
            return base.index_copy(0, active_index, value)

        return ElectronReconstruction(
            density=scatter(
                reconstruction.density,
                (batch_size, reconstruction.density.shape[1]),
            ),
            gradient=scatter(
                reconstruction.gradient,
                (batch_size, *reconstruction.gradient.shape[1:]),
            ),
            electron_count=scatter(reconstruction.electron_count, (batch_size,)),
            dipole=scatter(reconstruction.dipole, (batch_size, 3)),
            latent_round_trip=scatter(
                reconstruction.latent_round_trip,
                (batch_size, *reconstruction.latent_round_trip.shape[1:]),
            ),
        )


def _move_immutable(value: Any, device: torch.device) -> Any:
    """Recursively reconstruct frozen dataclasses with transferred tensor leaves."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {
            field.name: _move_immutable(getattr(value, field.name), device)
            for field in dataclasses.fields(value)
            if field.init
        }
        return dataclasses.replace(value, **changes)
    if isinstance(value, Mapping):
        return {key: _move_immutable(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return (
            type(value)(*(_move_immutable(item, device) for item in value))
            if hasattr(value, "_fields")
            else tuple(_move_immutable(item, device) for item in value)
        )
    if isinstance(value, list):
        return [_move_immutable(item, device) for item in value]
    return value
