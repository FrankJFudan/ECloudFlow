"""Training losses, stateful normalization, EMA, and Lightning integration."""

from ecloudflow.training.batching import TrainingBatchBuilder
from ecloudflow.training.callbacks import AtomicArtifactWriter, NonFiniteDiagnostics
from ecloudflow.training.checkpoint import (
    CheckpointStateError,
    ReproducibleCheckpoint,
)
from ecloudflow.training.ema import ExponentialMovingAverage
from ecloudflow.training.losses import RunningLossScaler, compute_ecloudflow_loss
from ecloudflow.training.module import ECloudFlowTrainingModule
from ecloudflow.training.runtime import (
    TrainingConfigurationError,
    TrainingRuntime,
    build_training_runtime,
    run_training,
)
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
    "TrainingBatchBuilder",
    "TrainingConfigurationError",
    "TrainingRuntime",
    "TrainingStage",
    "TrainingTargets",
    "benchmark_scaling",
    "build_training_runtime",
    "compute_ecloudflow_loss",
    "configure_stage",
    "measured_stub_nfe",
    "merge_scaling_reports",
    "run_training",
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
