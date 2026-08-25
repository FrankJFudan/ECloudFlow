"""Executable Lightning assembly for local and distributed model training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.config import AppConfig
from ecloudflow.data import ECloudDataModule
from ecloudflow.ecloud import ElectronFieldDecoder, EquivariantFieldTokenizer
from ecloudflow.models import ECloudFlowModel
from ecloudflow.training.batching import TrainingBatchBuilder
from ecloudflow.training.callbacks import NonFiniteDiagnostics
from ecloudflow.training.checkpoint import ReproducibleCheckpoint
from ecloudflow.training.module import ECloudFlowTrainingModule
from ecloudflow.training.stages import TrainingStage, configure_stage


class TrainingConfigurationError(ValueError):
    """Report a launch-time training configuration or environment mismatch."""


@dataclass(frozen=True)
class TrainingRuntime:
    """Expose all assembled objects used by one Lightning fit call.

    :param module: Joint staged Lightning optimization module.
    :param datamodule: Deterministic rank/worker-sharded clean-complex loader.
    :param trainer: Configured Lightning trainer owning devices and processes.
    :param checkpoint_dir: Resolved checkpoint artifact directory.
    :return: Immutable handles for inspection, testing, and execution.
    :rtype: TrainingRuntime

    Construction validates dataset and resume identities before accelerator
    workers are launched. The object itself does not start optimization;
    :func:`run_training` is the only public function that invokes ``fit``.
    """

    module: ECloudFlowTrainingModule
    datamodule: ECloudDataModule
    trainer: L.Trainer
    checkpoint_dir: Path


def build_training_runtime(config: AppConfig, output_dir: str | Path) -> TrainingRuntime:
    """Assemble a complete, validated Lightning training application.

    :param config: Fully resolved immutable ECloudFlow application configuration.
    :param output_dir: Run root for logs and bounded diagnostic artifacts.
    :return: Model, DataModule, callbacks, logger, Trainer, and checkpoint path.
    :rtype: TrainingRuntime
    :raises TypeError: If ``config`` is not the canonical Pydantic model.
    :raises TrainingConfigurationError: If GPU visibility, resume path, stage,
        or output/checkpoint paths cannot satisfy the requested run.
    :raises DataValidationError: If the configured manifest or shards are invalid.

    The joint model, tokenizer, and decoder are built from one explicit shape
    configuration so checkpoint tensors cannot silently disagree. The fixed
    ligand vocabulary supplies atom, charge, and Kekule bond channel counts.
    ``ddp`` with more than one device uses unused-parameter detection because
    optional QM, affinity, and interaction labels can legitimately be absent on
    a rank. Lightning remains the sole owner of process launch, BF16 autocast,
    gradient accumulation/clipping, optimizer state, and checkpoint I/O.

    Dataset setup is intentionally completed before the Trainer is returned.
    Missing manifests, invalid shard hashes, incompatible resume files, and
    insufficient visible GPUs therefore fail before a multi-process launch or
    partial checkpoint is created. No sample is consumed during this preflight.
    """
    if not isinstance(config, AppConfig):
        raise TypeError("config must be AppConfig.")
    run_root = Path(output_dir)
    if run_root.exists() and not run_root.is_dir():
        raise TrainingConfigurationError(
            f"training output path is not a directory: {run_root}"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    _validate_accelerator(config)
    checkpoint_dir = Path(config.trainer.checkpoint_dir)
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        raise TrainingConfigurationError(
            f"checkpoint path is not a directory: {checkpoint_dir}"
        )
    resume = config.trainer.resume_from
    if resume is not None and not Path(resume).is_file():
        raise TrainingConfigurationError(f"resume checkpoint does not exist: {resume}")
    init_from = config.trainer.init_from
    if resume is not None and init_from is not None:
        raise TrainingConfigurationError(
            "resume_from and init_from are mutually exclusive"
        )
    if init_from is not None and not Path(init_from).is_file():
        raise TrainingConfigurationError(
            f"initialization checkpoint does not exist: {init_from}"
        )

    datamodule = ECloudDataModule(config.data)
    datamodule.setup("fit")
    module = _build_module(config)
    if init_from is not None:
        _initialize_stage_weights(module, Path(init_from))
    callbacks = [
        NonFiniteDiagnostics(
            artifact_dir=run_root / "diagnostics",
            failure_threshold=config.trainer.nan_failure_threshold,
            max_artifacts=config.trainer.max_nan_artifacts,
        ),
        ReproducibleCheckpoint(
            config.model_dump(mode="json"),
            reproducible_resume=config.trainer.reproducible_resume,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="step-{step:08d}",
            auto_insert_metric_name=False,
            every_n_train_steps=config.trainer.every_n_train_steps,
            save_last=config.trainer.save_last,
            save_top_k=-1,
            save_on_train_epoch_end=False,
        ),
    ]
    logger = CSVLogger(save_dir=run_root / "logs", name="training")
    trainer = L.Trainer(
        accelerator=config.trainer.accelerator,
        strategy=_resolve_strategy(config),
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.trainer.gradient_clip_val,
        gradient_clip_algorithm="norm",
        max_steps=config.trainer.max_steps,
        max_epochs=config.trainer.max_epochs,
        default_root_dir=run_root,
        callbacks=callbacks,
        logger=logger,
        deterministic=config.trainer.deterministic,
        benchmark=config.trainer.benchmark,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        log_every_n_steps=config.trainer.log_every_n_steps,
        use_distributed_sampler=False,
        limit_val_batches=0,
    )
    return TrainingRuntime(module, datamodule, trainer, checkpoint_dir)


def run_training(config: AppConfig, output_dir: str | Path) -> TrainingRuntime:
    """Build the configured application and execute ``Lightning.Trainer.fit``.

    :param config: Fully resolved strict application configuration.
    :param output_dir: Run root containing logs, diagnostics, and CLI provenance.
    :return: Completed runtime handles; Trainer state exposes the final step and
        callback checkpoint paths.
    :rtype: TrainingRuntime
    :raises TrainingConfigurationError: If launch preflight cannot be satisfied.
    :raises DataValidationError: If dataset discovery or identity validation fails.
    :raises Exception: Propagates Lightning, model, optimizer, and checkpoint
        failures without converting a failed run into a successful CLI exit.

    ``seed_everything(..., workers=True)`` initializes Python, NumPy, CPU, CUDA,
    and DataLoader seeds before any model weights or stochastic path samples are
    created. H100 runs enable high float32 matrix-multiplication precision while
    retaining configured BF16 mixed precision. Resume is delegated through
    Lightning's native ``ckpt_path`` so model, optimizer, precision plugin, EMA,
    scaler, global step, epoch, and callback state restore together. The strict
    reproducibility callback additionally validates resolved configuration and
    dataset identity before resumed optimization proceeds.
    """
    if not isinstance(config, AppConfig):
        raise TypeError("config must be AppConfig.")
    L.seed_everything(config.seed, workers=True)
    if config.trainer.accelerator == "gpu":
        torch.set_float32_matmul_precision("high")
    runtime = build_training_runtime(config, output_dir)
    runtime.trainer.fit(
        runtime.module,
        datamodule=runtime.datamodule,
        ckpt_path=config.trainer.resume_from,
    )
    return runtime


def _build_module(config: AppConfig) -> ECloudFlowTrainingModule:
    """Construct shape-compatible joint, tokenizer, decoder, and stage policy."""
    vocabulary = ChemicalVocabulary.default_ligand()
    model = ECloudFlowModel.from_config(
        config.model,
        electron_latent_dim=config.model.electron_latent_dim,
        electron_vector_dim=config.model.electron_vector_dim,
        atom_classes=len(vocabulary.atom_symbols),
        charge_classes=len(vocabulary.formal_charges),
        bond_classes=len(vocabulary.bond_classes),
        max_atoms=config.model.max_atoms,
    )
    tokenizer = EquivariantFieldTokenizer(
        n_radial=config.model.field_n_radial,
        lmax=config.model.lmax,
        scalar_dim=config.model.scalar_dim,
        vector_dim=config.model.electron_vector_dim,
        latent_dim=config.model.electron_latent_dim,
        cutoff=config.model.field_cutoff,
        chunk_size=config.model.field_chunk_size,
    )
    # EMA requires every parameter to be materialized before it snapshots the
    # module. The runtime's stable ligand atom-plus-charge feature width is known
    # here, so initialize the tokenizer's deliberately generic LazyLinear once.
    with torch.no_grad():
        tokenizer.encode(
            torch.zeros(
                1,
                1,
                config.model.field_n_radial,
                (config.model.lmax + 1) ** 2,
            ),
            torch.zeros(
                1,
                1,
                len(vocabulary.atom_symbols) + len(vocabulary.formal_charges),
            ),
            torch.ones((1, 1), dtype=torch.bool),
        )
    decoder = ElectronFieldDecoder(
        n_radial=config.model.field_n_radial,
        lmax=config.model.lmax,
        vector_dim=config.model.electron_vector_dim,
        latent_dim=config.model.electron_latent_dim,
        cutoff=config.model.field_cutoff,
        chunk_size=config.model.field_chunk_size,
    )
    module = ECloudFlowTrainingModule(
        joint_backbone=model,
        field_tokenizer=tokenizer,
        field_decoder=decoder,
        loss_config=config.loss,
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
        ema_decay=config.trainer.ema_decay,
        batch_builder=TrainingBatchBuilder(config.model, config.trainer),
    )
    try:
        stage = TrainingStage(config.stage)
    except ValueError as error:
        raise TrainingConfigurationError(f"unknown training stage: {config.stage}") from error
    configure_stage(module, stage)
    return module


def _validate_accelerator(config: AppConfig) -> None:
    """Fail before Trainer launch when explicitly requested GPUs are unavailable."""
    if config.trainer.accelerator != "gpu":
        return
    visible = torch.cuda.device_count()
    if not torch.cuda.is_available() or visible < config.trainer.devices:
        raise TrainingConfigurationError(
            f"requested {config.trainer.devices} GPU devices but only {visible} are visible"
        )


def _resolve_strategy(config: AppConfig) -> str | Any:
    """Resolve DDP optional-label behavior while preserving other strategies."""
    if config.trainer.strategy == "ddp" and config.trainer.devices > 1:
        return DDPStrategy(find_unused_parameters=True)
    return config.trainer.strategy


_TRANSFER_GROUPS = ("joint_backbone", "field_tokenizer", "field_decoder")


def _initialize_stage_weights(
    module: ECloudFlowTrainingModule, checkpoint_path: Path
) -> None:
    """Load only scientific model groups from a prior curriculum stage.

    :param module: Fresh current-stage module with the requested architecture.
    :param checkpoint_path: Existing Lightning or model state-dict checkpoint.
    :return: None after strict group loading and EMA synchronization.
    :rtype: None
    :raises TrainingConfigurationError: If the checkpoint is unreadable,
        malformed, incomplete, or shape-incompatible with the current model.

    Stage transfer is deliberately distinct from resume. The three scientific
    parameter groups are loaded strictly, while optimizer state, gradient
    scaler, running loss normalization, RNG, DataLoader position, global step,
    and callbacks are ignored. EMA shadows are then copied from the initialized
    live parameters with a zero update count, preventing stale construction-time
    shadows. No tensor is moved off CPU here; Lightning owns the later device
    and precision transfer, and every DDP rank deterministically reads the same
    immutable checkpoint before fitting.
    """
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise TrainingConfigurationError(
            f"failed to read initialization checkpoint {checkpoint_path}: {error}"
        ) from error
    state: object = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping) or not state:
        raise TrainingConfigurationError(
            "initialization checkpoint does not contain a non-empty state_dict"
        )
    if not all(isinstance(key, str) for key in state):
        raise TrainingConfigurationError(
            "initialization checkpoint state_dict keys must be strings"
        )
    try:
        for group_name in _TRANSFER_GROUPS:
            prefix = f"{group_name}."
            group_state = {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if not group_state:
                raise TrainingConfigurationError(
                    f"initialization checkpoint is missing {group_name} weights"
                )
            group = getattr(module, group_name)
            group.load_state_dict(group_state, strict=True)
        parameters = tuple(module.named_parameters())
        shadows = module.ema.shadow_parameters()
        if len(parameters) != len(shadows):
            raise TrainingConfigurationError(
                "initialization checkpoint changed the EMA parameter layout"
            )
        with torch.no_grad():
            for (_, parameter), shadow in zip(parameters, shadows, strict=True):
                shadow.copy_(parameter)
            module.ema.num_updates.zero_()
    except TrainingConfigurationError:
        raise
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise TrainingConfigurationError(
            f"initialization checkpoint is incompatible with the model: {error}"
        ) from error
