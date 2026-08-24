"""Bounded diagnostics and collective rank-zero artifact callbacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from lightning import Callback
from torch import distributed as dist

from ecloudflow.training.checkpoint import write_rank_zero_json


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    """Collect tensor leaves without traversing strings or arbitrary objects."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [tensor for item in value.values() for tensor in _tensor_leaves(item)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [tensor for item in value for tensor in _tensor_leaves(item)]
    return []


class AtomicArtifactWriter:
    """Publish structured artifacts collectively from rank zero.

    :param artifact_dir: Shared directory used for final artifact paths.
    :return: Stateless writer retaining only the resolved directory.
    :rtype: AtomicArtifactWriter

    Every rank must call :meth:`write` in the same order. JSON serialization is
    validated collectively before rank zero creates a same-directory temporary,
    flushes it, and atomically replaces the final path. Publication status is
    broadcast, so no peer silently continues after failure. Payloads contain CPU
    scalars rather than model tensors; model dtype/device and gradients are never
    changed. Equal inputs and filenames publish deterministic bytes.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def write(self, trainer: Any, filename: str, payload: object) -> Path:
        """Collectively publish one JSON artifact and return its final path.

        :param trainer: Active trainer exposing a stable global rank.
        :param filename: Relative leaf filename without parent traversal.
        :param payload: JSON-compatible structured diagnostic value.
        :return: Final path observed as complete by every rank.
        :rtype: pathlib.Path
        :raises ValueError: If filename is absolute or contains parent traversal.
        :raises CheckpointStateError: If serialization or atomic publication fails.

        Validation precedes the fixed distributed serialization/write-status
        protocol. Only rank zero mutates the filesystem. The function performs
        no tensor conversion, accelerator synchronization beyond metadata
        collectives, random operation, or model state mutation.
        """
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact filename must be a safe relative path")
        path = self.artifact_dir / relative
        write_rank_zero_json(trainer, path, payload)
        return path


class NonFiniteDiagnostics(Callback):
    """Bound nonfinite training diagnostics and fail at a fixed threshold.

    :param artifact_dir: Shared directory for atomic rank-zero JSON diagnostics.
    :param failure_threshold: Positive global nonfinite-batch count that stops fit.
    :param max_artifacts: Non-negative maximum number of diagnostic files.
    :return: Lightning callback with rank-consistent bounded counters.
    :rtype: NonFiniteDiagnostics
    :raises ValueError: If threshold or artifact bound is invalid.

    Each train-batch end reduces one nonfinite flag in a fixed collective order.
    Affected ranks never dump batch tensors, potentially sensitive structures, or
    unbounded arrays; the artifact contains only step/batch/rank/count metadata.
    At most ``max_artifacts`` atomic JSON files are written by rank zero, and all
    ranks raise together once ``failure_threshold`` is reached. Model tensors,
    device/dtype, gradients, RNG, and checkpoint state are not mutated.
    """

    def __init__(
        self,
        *,
        artifact_dir: str | Path,
        failure_threshold: int = 1,
        max_artifacts: int = 3,
    ) -> None:
        super().__init__()
        if isinstance(failure_threshold, bool) or failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if isinstance(max_artifacts, bool) or max_artifacts < 0:
            raise ValueError("max_artifacts must be nonnegative")
        self.failure_threshold = int(failure_threshold)
        self.max_artifacts = int(max_artifacts)
        self.failure_count = 0
        self.artifact_count = 0
        self.writer = AtomicArtifactWriter(artifact_dir)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Reduce nonfinite status, emit bounded metadata, and stop consistently.

        :param trainer: Active trainer with global rank and global step.
        :param pl_module: Module used only to select the strategy-managed device.
        :param outputs: Nested training output inspected for floating tensors.
        :param batch: Input batch, deliberately excluded from diagnostic artifacts.
        :param batch_idx: Rank-local batch index included as a bounded integer.
        :return: None for finite output or sub-threshold diagnostic handling.
        :rtype: None
        :raises FloatingPointError: On every rank when the threshold is reached.
        :raises CheckpointStateError: If collective atomic artifact writing fails.

        Floating leaves retain dtype/device and are inspected read-only. One flag
        all-reduce occurs on every initialized rank before conditional I/O; since
        the reduced result is identical, all ranks take the same publication and
        raise branches. Only rank zero writes. No batch/sample payload, gradient,
        optimizer, RNG, EMA, scaler, or checkpoint state is changed.
        """
        del batch
        tensors = _tensor_leaves(outputs)
        local_nonfinite = any(
            tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
            for tensor in tensors
        )
        device = tensors[0].device if tensors else next(pl_module.parameters()).device
        flag = torch.tensor(int(local_nonfinite), device=device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if not bool(flag):
            return
        self.failure_count += 1
        if self.artifact_count < self.max_artifacts:
            self.writer.write(
                trainer,
                f"nonfinite-{int(trainer.global_step):08d}-{self.failure_count:03d}.json",
                {
                    "batch_idx": int(batch_idx),
                    "failure_count": self.failure_count,
                    "global_step": int(trainer.global_step),
                    "nonfinite": True,
                },
            )
            self.artifact_count += 1
        if self.failure_count >= self.failure_threshold:
            raise FloatingPointError(
                f"nonfinite training output reached threshold {self.failure_threshold}"
            )
