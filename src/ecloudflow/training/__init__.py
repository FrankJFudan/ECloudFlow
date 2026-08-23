"""Training losses, stateful normalization, EMA, and Lightning integration."""

from ecloudflow.training.ema import ExponentialMovingAverage
from ecloudflow.training.losses import RunningLossScaler, compute_ecloudflow_loss
from ecloudflow.training.module import ECloudFlowTrainingModule
from ecloudflow.training.types import (
    ElectronDecoderContext,
    LossBreakdown,
    TrainingBatch,
    TrainingTargets,
)

__all__ = [
    "ECloudFlowTrainingModule",
    "ElectronDecoderContext",
    "ExponentialMovingAverage",
    "LossBreakdown",
    "RunningLossScaler",
    "TrainingBatch",
    "TrainingTargets",
    "compute_ecloudflow_loss",
]
