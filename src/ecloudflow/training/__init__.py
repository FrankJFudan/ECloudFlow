"""Training losses, stateful normalization, EMA, and Lightning integration."""

from ecloudflow.training.callbacks import AtomicArtifactWriter, NonFiniteDiagnostics
from ecloudflow.training.checkpoint import (
    CheckpointStateError,
    ReproducibleCheckpoint,
)
from ecloudflow.training.ema import ExponentialMovingAverage
from ecloudflow.training.losses import RunningLossScaler, compute_ecloudflow_loss
from ecloudflow.training.module import ECloudFlowTrainingModule
from ecloudflow.training.stages import TrainingStage, configure_stage
from ecloudflow.training.types import (
    ElectronDecoderContext,
    LossBreakdown,
    TrainingBatch,
    TrainingTargets,
)

__all__ = [
    "ECloudFlowTrainingModule",
    "AtomicArtifactWriter",
    "CheckpointStateError",
    "ElectronDecoderContext",
    "ExponentialMovingAverage",
    "LossBreakdown",
    "NonFiniteDiagnostics",
    "ReproducibleCheckpoint",
    "RunningLossScaler",
    "TrainingBatch",
    "TrainingTargets",
    "TrainingStage",
    "configure_stage",
    "compute_ecloudflow_loss",
]
