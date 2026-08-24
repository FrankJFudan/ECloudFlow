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
    "AtomicArtifactWriter",
    "BenchmarkConfig",
    "BenchmarkError",
    "CheckpointStateError",
    "ECloudFlowTrainingModule",
    "ElectronDecoderContext",
    "ExponentialMovingAverage",
    "LossBreakdown",
    "NonFiniteDiagnostics",
    "ReproducibleCheckpoint",
    "RunningLossScaler",
    "ScalingReport",
    "ScalingRow",
    "TrainingBatch",
    "TrainingStage",
    "TrainingTargets",
    "benchmark_scaling",
    "compute_ecloudflow_loss",
    "configure_stage",
    "measured_stub_nfe",
    "merge_scaling_reports",
]

_BENCHMARK_EXPORTS = {
    "BenchmarkConfig",
    "BenchmarkError",
    "ScalingReport",
    "ScalingRow",
    "benchmark_scaling",
    "merge_scaling_reports",
    "measured_stub_nfe",
}


def __getattr__(name: str):
    """Resolve benchmark helpers lazily to keep ``python -m`` warning-free."""
    if name in _BENCHMARK_EXPORTS:
        from ecloudflow.training import benchmark

        value = getattr(benchmark, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
