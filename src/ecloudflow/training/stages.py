"""Explicit four-stage parameter and scientific-objective policies."""

from __future__ import annotations

from enum import Enum

from ecloudflow.training.module import ECloudFlowTrainingModule


class TrainingStage(str, Enum):
    """Stable names for the four independently resumable training stages."""

    ELECTRON_TOKENIZER = "electron_tokenizer"
    LIGAND_PRETRAIN = "ligand_pretrain"
    POCKET_MULTITASK = "pocket_multitask"
    HIGH_QUALITY_FINETUNE = "high_quality_finetune"


_GROUPS: dict[TrainingStage, frozenset[str]] = {
    TrainingStage.ELECTRON_TOKENIZER: frozenset({"field_tokenizer", "field_decoder"}),
    TrainingStage.LIGAND_PRETRAIN: frozenset(
        {"field_tokenizer", "field_decoder", "joint_backbone"}
    ),
    TrainingStage.POCKET_MULTITASK: frozenset(
        {"field_tokenizer", "field_decoder", "joint_backbone"}
    ),
    TrainingStage.HIGH_QUALITY_FINETUNE: frozenset(
        {"field_tokenizer", "field_decoder", "joint_backbone"}
    ),
}


def configure_stage(
    module: ECloudFlowTrainingModule, stage: TrainingStage
) -> tuple[str, ...]:
    """Apply one exact trainability and loss-availability policy.

    :param module: Task 11 module exposing the three stable parameter groups.
    :param stage: Typed curriculum stage; arbitrary strings are rejected.
    :return: Stable sorted names of groups made trainable.
    :rtype: tuple[str, ...]
    :raises TypeError: If ``module`` or ``stage`` has the wrong public type.

    Validation completes before mutation. Parameters retain their existing
    device, dtype, values, and buffers; only ``requires_grad`` flags and the
    frozen loss configuration reference change. The operation has no random,
    filesystem, optimizer, checkpoint, or distributed side effect and is
    deterministic on every rank. Optimizers must be constructed after calling
    this function. Tokenizer training enables only electron-cloud supervision;
    ligand pretraining excludes unavailable pocket interaction supervision;
    pocket multitask and fine-tuning require a positive interaction weight.
    """
    if not isinstance(module, ECloudFlowTrainingModule):
        raise TypeError("module must be ECloudFlowTrainingModule.")
    if not isinstance(stage, TrainingStage):
        raise TypeError("stage must be TrainingStage.")
    enabled = _GROUPS[stage]
    groups = {
        "field_tokenizer": module.field_tokenizer,
        "field_decoder": module.field_decoder,
        "joint_backbone": module.joint_backbone,
    }
    for name, group in groups.items():
        for parameter in group.parameters():
            parameter.requires_grad_(name in enabled)

    config = module.loss_config
    if stage is TrainingStage.ELECTRON_TOKENIZER:
        config = config.model_copy(
            update={
                "flow": config.flow.model_copy(update={"weight": 0.0}),
                "score": config.score.model_copy(update={"weight": 0.0}),
                "discrete": config.discrete.model_copy(update={"weight": 0.0}),
                "chem": config.chem.model_copy(update={"weight": 0.0}),
                "interaction": config.interaction.model_copy(update={"weight": 0.0}),
            }
        )
    elif stage is TrainingStage.LIGAND_PRETRAIN:
        config = config.model_copy(
            update={
                "flow": config.flow.model_copy(
                    update={"weight": max(config.flow.weight, 1.0)}
                ),
                "discrete": config.discrete.model_copy(
                    update={"weight": max(config.discrete.weight, 1.0)}
                ),
                "interaction": config.interaction.model_copy(update={"weight": 0.0}),
            }
        )
    elif config.interaction.weight <= 0.0:
        config = config.model_copy(
            update={
                "interaction": config.interaction.model_copy(update={"weight": 1.0})
            }
        )
    module.loss_config = config
    return tuple(sorted(enabled))
