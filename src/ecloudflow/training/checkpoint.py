"""Strict reproducibility metadata and atomic distributed publication."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import Callback
from torch import distributed as dist

from ecloudflow.exceptions import ECloudFlowError


class CheckpointStateError(ECloudFlowError, RuntimeError):
    """Raise when checkpoint identity, state, or atomic publication is unsafe."""


_OPERATIONAL_KEYS = frozenset(
    {
        "resume_from",
        "resume_from_checkpoint",
        "checkpoint_path",
        "ckpt_path",
        "checkpoint_dir",
        "dirpath",
        "output_dir",
        "artifact_dir",
        "logger",
        "loggers",
        "logger_settings",
        "max_steps",
        "max_epochs",
        "min_steps",
        "min_epochs",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "limit_predict_batches",
        "fast_dev_run",
    }
)


def _plain_config(config: object) -> dict[str, Any]:
    """Return a detached JSON-compatible resolved configuration mapping."""
    dump = getattr(config, "model_dump", None)
    value = dump(mode="json") if callable(dump) else config
    if not isinstance(value, Mapping):
        raise CheckpointStateError("resolved configuration must be a mapping")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise CheckpointStateError(
            f"resolved configuration is not JSON serializable: {error}"
        ) from error
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)
    return decoded


def training_semantic_projection(config: object) -> dict[str, Any]:
    """Return the documented scientific/training compatibility projection.

    :param config: Full resolved Pydantic model or JSON-compatible mapping.
    :return: Detached nested mapping with only operational resume differences removed.
    :rtype: dict[str, Any]
    :raises CheckpointStateError: If configuration is not a JSON mapping.

    Resume/checkpoint paths, output/artifact directories, logger settings, and
    termination bounds are removed by key at any nesting level. All remaining
    model, data, preprocessing, loss, stage, optimizer, precision, accumulation,
    clipping, and distributed semantics remain exact. The transformation is
    deterministic, device/dtype free, read-only, and performs no collective or
    filesystem mutation. It enables a one-step checkpoint to resume toward two
    steps without accepting a scientifically incompatible experiment.
    """

    def project(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: project(item)
                for key, item in sorted(value.items())
                if key not in _OPERATIONAL_KEYS
            }
        if isinstance(value, list):
            return [project(item) for item in value]
        return value

    return project(_plain_config(config))


def assert_resume_compatible(saved: object, current: object) -> None:
    """Fail closed unless canonical training semantics match exactly.

    :param saved: Full resolved configuration embedded in the checkpoint.
    :param current: Full resolved configuration requested for the resumed run.
    :return: None when the documented semantic projections are identical.
    :rtype: None
    :raises CheckpointStateError: If either config is malformed or semantics differ.

    Comparison uses canonical sorted JSON and is deterministic across ranks.
    There are no tensor/device/dtype conversions, mutations, collectives, or file
    writes. Operational paths/logger settings and training termination bounds may
    differ; all model/data/loss/stage/optimizer/runtime arithmetic semantics fail
    closed on mismatch.
    """
    if training_semantic_projection(saved) != training_semantic_projection(current):
        raise CheckpointStateError("checkpoint semantic configuration mismatch")


def capture_rng_state() -> dict[str, Any]:
    """Capture complete rank-local Python, NumPy, CPU, and CUDA RNG state.

    :return: Mapping containing Python/NumPy objects, a CPU RNG byte tensor, and
        one CPU byte tensor for every visible CUDA generator.
    :rtype: dict[str, Any]
    :raises CheckpointStateError: If CUDA is initialized but its RNG cannot be read.

    The function does not advance any generator. CPU and CUDA byte tensors are
    copied to CPU for portable Lightning serialization without numerical dtype
    conversion. Each distributed rank calls it independently before the fixed
    object gather; it performs no collective, filesystem mutation, or seeding.
    Determinism requires the same rank/world/device topology at restore.
    """
    try:
        cuda = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    except RuntimeError as error:
        raise CheckpointStateError(f"CUDA RNG state is unavailable: {error}") from error
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": cuda,
    }


def restore_rng_state(state: object) -> None:
    """Restore one rank's complete checkpointed RNG state without advancing it.

    :param state: Mapping returned by :func:`capture_rng_state`.
    :return: None after restoring Python, NumPy, CPU, and visible CUDA generators.
    :rtype: None
    :raises CheckpointStateError: If fields, tensor types, or CUDA topology differ.

    CPU byte tensors remain CPU; PyTorch transfers CUDA generator state to the
    corresponding devices internally without changing model dtype/device. This
    mutates process-local generators only, performs no distributed collective or
    filesystem I/O, and is deterministic when rank and visible-device topology
    match the saved run. Missing required state fails closed.
    """
    if not isinstance(state, Mapping):
        raise CheckpointStateError("RNG state must be a mapping")
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not required <= set(state):
        raise CheckpointStateError("RNG state is missing required generators")
    cpu = state["torch_cpu"]
    cuda = state["torch_cuda"]
    if (
        not isinstance(cpu, torch.Tensor)
        or not isinstance(cuda, list)
        or not all(isinstance(item, torch.Tensor) for item in cuda)
    ):
        raise CheckpointStateError("RNG tensor state is malformed")
    if len(cuda) != torch.cuda.device_count():
        raise CheckpointStateError("checkpoint CUDA RNG topology mismatch")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(cpu.cpu())
        if cuda:
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda])
    except (TypeError, ValueError, RuntimeError) as error:
        raise CheckpointStateError(f"RNG restoration failed: {error}") from error


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish already serialized text with same-directory replace semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise CheckpointStateError(
            f"atomic artifact publication failed for {path}: {error}"
        ) from error


def atomic_write_json(path: str | Path, payload: object) -> None:
    """Atomically publish one deterministic UTF-8 JSON artifact.

    :param path: Final artifact path; a temporary file is created beside it.
    :param payload: JSON-serializable diagnostic or provenance value.
    :return: None after flush, fsync, and atomic ``os.replace`` publication.
    :rtype: None
    :raises TypeError: If payload contains a non-JSON value.
    :raises ValueError: If payload contains an invalid JSON number/value.
    :raises CheckpointStateError: If directory, temporary write, sync, or replace fails.

    Serialization completes before touching the existing artifact, so malformed
    payloads cannot truncate a prior result. Publication is local and must be
    called only on rank zero or through :func:`write_rank_zero_json`. No tensors
    are moved or converted; callers should provide CPU scalar diagnostics. Equal
    mappings produce byte-identical sorted JSON independent of device/dtype.
    """
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _atomic_write_text(Path(path), text)


def write_rank_zero_json(trainer: Any, path: str | Path, payload: object) -> None:
    """Collectively validate and atomically publish JSON from rank zero.

    :param trainer: Active trainer exposing ``global_rank``.
    :param path: Shared final artifact path.
    :param payload: JSON-compatible value expected to be valid on every rank.
    :return: None after all ranks observe publication success.
    :rtype: None
    :raises CheckpointStateError: If any rank cannot serialize or rank zero cannot write.

    Every initialized rank executes serialization-status gather and publication-
    status broadcast in the same order. Only rank zero mutates the filesystem by
    same-directory temporary write and atomic replace. Failures are broadcast so
    peers do not hang or silently continue. Payload tensors are unsupported; no
    device/dtype transfer occurs. For equal inputs, bytes are deterministic.
    """
    serialization_error: str | None = None
    text: str | None = None
    try:
        text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        serialization_error = str(error)
    errors = _all_gather_objects(serialization_error)
    if any(error is not None for error in errors):
        raise CheckpointStateError(f"artifact serialization failed by rank: {errors}")
    publication_error: str | None = None
    if int(trainer.global_rank) == 0:
        assert text is not None
        try:
            _atomic_write_text(Path(path), text)
        except CheckpointStateError as error:
            publication_error = str(error)
    publication_error = _broadcast_object(publication_error, source=0)
    if publication_error is not None:
        raise CheckpointStateError(publication_error)


def _all_gather_objects(value: Any) -> list[Any]:
    """Gather one object per rank or return a one-element local list."""
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _broadcast_object(value: Any, *, source: int) -> Any:
    """Broadcast one object from source or return it in a local process."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    values = [value]
    dist.broadcast_object_list(values, src=source)
    return values[0]


def _git_revision() -> str:
    """Return the exact current Git revision or fail closed."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CheckpointStateError(f"Git revision is unavailable: {error}") from error
    revision = result.stdout.strip()
    if result.returncode != 0 or len(revision) != 40:
        raise CheckpointStateError("Git revision is unavailable or malformed")
    return revision


class ReproducibleCheckpoint(Callback):
    """Attach and restore strict per-rank reproducibility metadata.

    :param resolved_config: Full resolved configuration stored for provenance.
    :param reproducible_resume: Require every identity/state field and fail closed.
    :param git_revision: Optional exact 40-hex revision injection for packaged runs.
    :return: Lightning callback that complements native checkpoint state.
    :rtype: ReproducibleCheckpoint
    :raises CheckpointStateError: If configuration or revision is malformed.

    Lightning remains the sole owner of model, optimizer, scheduler, scaler
    plugin, module EMA, and loss-scaler tensor serialization. This callback
    validates the module EMA/scaler keys and adds only provenance, identities,
    and per-rank RNG/data state. Every rank performs RNG capture, data capture,
    and object gathers in the same fixed order before any rank-zero publication.
    Restored tensors retain Lightning-selected device/dtype; RNG payloads are CPU
    bytes. Rank-local data/RNG state is reapplied at fit start after Lightning's
    ordinary restore. Missing manifest/preprocessing/config/state fails closed.
    """

    def __init__(
        self,
        resolved_config: object,
        *,
        reproducible_resume: bool = True,
        git_revision: str | None = None,
    ) -> None:
        super().__init__()
        self.resolved_config = _plain_config(resolved_config)
        self.reproducible_resume = reproducible_resume
        self.git_revision = git_revision
        if git_revision is not None and len(git_revision) != 40:
            raise CheckpointStateError(
                "git_revision must contain exactly 40 characters"
            )
        self._pending_rng_state: object | None = None
        self._pending_data_state: object | None = None

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record one batch only after Lightning confirms successful consumption."""
        del pl_module, outputs, batch, batch_idx
        data = getattr(trainer, "datamodule", None)
        mark = getattr(data, "mark_batch_consumed", None)
        if callable(mark):
            mark()

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Publish a genuine Lightning epoch transition to resumable data state."""
        del pl_module
        data = getattr(trainer, "datamodule", None)
        set_epoch = getattr(data, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(trainer.current_epoch))

    def on_save_checkpoint(
        self,
        trainer: Any,
        pl_module: Any,
        checkpoint: dict[str, object],
    ) -> None:
        """Attach reproducibility state to a Lightning checkpoint.

        :param trainer: Active trainer providing rank, epoch, and data state.
        :param pl_module: ECloudFlow module containing EMA and loss scalers.
        :param checkpoint: Mutable checkpoint dictionary written by Lightning.
        :return: None. The input dictionary is updated in place on every rank.
        :rtype: None
        :raises CheckpointStateError: If identity, module state, data, or RNG fails.

        Every rank captures RNG then data state and contributes through two
        distributed object gathers in identical order. No filesystem write occurs
        here; Lightning owns native atomic checkpoint orchestration and rank-zero
        strategy semantics. Metadata tensors are CPU RNG bytes; native model,
        optimizer, scheduler, EMA, scaler, device, and dtype state is not duplicated.
        The full resolved config is retained while resume comparison uses the
        documented canonical training-semantic projection. Determinism requires
        the same world size and rank/device topology on restoration.
        """
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise CheckpointStateError("Lightning checkpoint state_dict is missing")
        keys = tuple(str(key) for key in state_dict)
        if not hasattr(pl_module, "ema") or not any(
            key.startswith("ema.") for key in keys
        ):
            raise CheckpointStateError("checkpoint EMA state is missing")
        if not hasattr(pl_module, "loss_scaler") or not any(
            key.startswith("loss_scaler.") for key in keys
        ):
            raise CheckpointStateError("checkpoint loss-scaler state is missing")
        rng_state = capture_rng_state()
        data_state = self._data_state(trainer)
        rng_by_rank = _all_gather_objects(rng_state)
        data_by_rank = _all_gather_objects(data_state)
        semantic = training_semantic_projection(self.resolved_config)
        semantic_hash = hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        checkpoint["ecloudflow_reproducibility"] = {
            "schema_version": 1,
            "global_step": int(trainer.global_step),
            "epoch": int(trainer.current_epoch),
            "world_size": len(rng_by_rank),
            "rng_by_rank": rng_by_rank,
            "data_by_rank": data_by_rank,
            "resolved_config": self.resolved_config,
            "semantic_config_hash": f"sha256:{semantic_hash}",
            "dataset_manifest_hash": data_state["manifest_hash"],
            "preprocessing_version": data_state["preprocessing_version"],
            "git_revision": self.git_revision or _git_revision(),
            "native_module_state": {"ema": True, "loss_scaler": True},
        }

    def on_load_checkpoint(
        self, trainer: Any, pl_module: Any, checkpoint: dict[str, object]
    ) -> None:
        """Validate metadata and stage rank-local RNG/data restoration.

        :param trainer: Active Lightning trainer and current DataModule.
        :param pl_module: Module whose EMA/loss-scaler native keys are required.
        :param checkpoint: Loaded Lightning checkpoint mapping.
        :return: None after validation and staging; no tensor weights are copied.
        :rtype: None
        :raises CheckpointStateError: If schema, config, identity, or rank state differs.

        Validation is deterministic and local after Lightning has loaded the CPU
        checkpoint payload. Model/optimizer/scheduler/EMA/scaler dtype and device
        restoration remain native Lightning behavior. Rank-local RNG/data values
        are selected by global rank and reapplied at fit start, after ordinary
        DataModule restoration. This hook performs no collective or filesystem
        mutation; all missing reproducible state fails closed.
        """
        del pl_module
        metadata = checkpoint.get("ecloudflow_reproducibility")
        if not isinstance(metadata, Mapping):
            if self.reproducible_resume:
                raise CheckpointStateError("reproducibility metadata is missing")
            return
        if metadata.get("schema_version") != 1:
            raise CheckpointStateError("checkpoint reproducibility schema mismatch")
        assert_resume_compatible(metadata.get("resolved_config"), self.resolved_config)
        rng_by_rank = metadata.get("rng_by_rank")
        data_by_rank = metadata.get("data_by_rank")
        if not isinstance(rng_by_rank, list) or not isinstance(data_by_rank, list):
            raise CheckpointStateError("per-rank checkpoint state is missing")
        rank = int(trainer.global_rank)
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if (
            len(rng_by_rank) != world_size
            or len(data_by_rank) != world_size
            or rank >= world_size
        ):
            raise CheckpointStateError("checkpoint distributed world-size mismatch")
        selected_data = data_by_rank[rank]
        if not isinstance(selected_data, Mapping):
            raise CheckpointStateError("rank-local data state is malformed")
        if selected_data.get("manifest_hash") != metadata.get("dataset_manifest_hash"):
            raise CheckpointStateError("checkpoint dataset manifest hash mismatch")
        if selected_data.get("preprocessing_version") != metadata.get(
            "preprocessing_version"
        ):
            raise CheckpointStateError("checkpoint preprocessing version mismatch")
        self._pending_rng_state = rng_by_rank[rank]
        self._pending_data_state = dict(selected_data)

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Reapply staged rank-local state after Lightning's native restoration."""
        del pl_module
        if self._pending_data_state is not None:
            data = getattr(trainer, "datamodule", None)
            load = getattr(data, "load_state_dict", None)
            if not callable(load):
                raise CheckpointStateError("resumed DataModule cannot restore state")
            load(self._pending_data_state)
        if self._pending_rng_state is not None:
            restore_rng_state(self._pending_rng_state)
        self._pending_data_state = None
        self._pending_rng_state = None

    def _data_state(self, trainer: Any) -> dict[str, Any]:
        """Return and validate the current rank's DataModule resume identity."""
        data = getattr(trainer, "datamodule", None)
        state_method = getattr(data, "state_dict", None)
        if not callable(state_method):
            raise CheckpointStateError(
                "reproducible checkpoint requires DataModule state"
            )
        state = state_method()
        if not isinstance(state, Mapping):
            raise CheckpointStateError("DataModule state must be a mapping")
        required = {
            "epoch",
            "consumed_batches",
            "manifest_hash",
            "preprocessing_version",
        }
        if not required <= set(state):
            raise CheckpointStateError("DataModule state is missing resume identity")
        if not isinstance(state["manifest_hash"], str) or not state["manifest_hash"]:
            raise CheckpointStateError("dataset manifest hash is unavailable")
        if (
            not isinstance(state["preprocessing_version"], str)
            or not state["preprocessing_version"]
        ):
            raise CheckpointStateError("preprocessing version is unavailable")
        return dict(state)
