"""Behavior tests for explicit staged-training policies."""

from torch import nn

from ecloudflow.config import LossConfig, load_config
from ecloudflow.training.module import ECloudFlowTrainingModule
from ecloudflow.training.stages import TrainingStage, configure_stage


def _module() -> ECloudFlowTrainingModule:
    """Build a module whose three stable groups all own parameters."""
    return ECloudFlowTrainingModule(
        joint_backbone=nn.Linear(2, 2),
        field_tokenizer=nn.Linear(2, 2),
        field_decoder=nn.Linear(2, 2),
        loss_config=LossConfig(),
    )


def _trainable_groups(module: ECloudFlowTrainingModule) -> set[str]:
    """Return stable groups containing at least one trainable parameter."""
    return {
        name
        for name in ("field_tokenizer", "field_decoder", "joint_backbone")
        if any(
            parameter.requires_grad for parameter in getattr(module, name).parameters()
        )
    }


def test_every_stage_has_an_exact_parameter_policy() -> None:
    """Mutation caught: an implicit/default group remains trainable in a stage."""
    expected = {
        TrainingStage.ELECTRON_TOKENIZER: {"field_tokenizer", "field_decoder"},
        TrainingStage.LIGAND_PRETRAIN: {
            "field_tokenizer",
            "field_decoder",
            "joint_backbone",
        },
        TrainingStage.POCKET_MULTITASK: {
            "field_tokenizer",
            "field_decoder",
            "joint_backbone",
        },
        TrainingStage.HIGH_QUALITY_FINETUNE: {
            "field_tokenizer",
            "field_decoder",
            "joint_backbone",
        },
    }
    module = _module()
    for stage, groups in expected.items():
        configure_stage(module, stage)
        assert _trainable_groups(module) == groups


def test_stage_loss_policies_exclude_semantically_unavailable_supervision() -> None:
    """Mutation caught: tokenizer or ligand-only training enables pocket interaction."""
    module = _module()
    configure_stage(module, TrainingStage.ELECTRON_TOKENIZER)
    assert module.loss_config.ecloud.weight > 0
    assert module.loss_config.flow.weight == 0
    assert module.loss_config.interaction.weight == 0

    configure_stage(module, TrainingStage.LIGAND_PRETRAIN)
    assert module.loss_config.flow.weight > 0
    assert module.loss_config.discrete.weight > 0
    assert module.loss_config.interaction.weight == 0

    configure_stage(module, TrainingStage.POCKET_MULTITASK)
    assert module.loss_config.interaction.weight > 0


def test_stage_configuration_rejects_untyped_values_without_partial_mutation() -> None:
    """Mutation caught: a misspelled stage silently selects a permissive default."""
    module = _module()
    before = tuple(parameter.requires_grad for parameter in module.parameters())
    try:
        configure_stage(module, "stage3")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("configure_stage accepted an untyped stage")
    assert tuple(parameter.requires_grad for parameter in module.parameters()) == before


def test_experiment_presets_separate_local_smoke_from_four_h100_production() -> None:
    """Mutation caught: the production preset silently falls back to local hardware."""
    smoke = load_config(["+experiment=smoke"])
    assert (smoke.trainer.accelerator, smoke.trainer.devices) == ("cpu", 1)
    assert smoke.trainer.max_steps == 2
    production = load_config(["+experiment=pdbbind_large"])
    assert (
        production.trainer.accelerator,
        production.trainer.strategy,
        production.trainer.devices,
        production.trainer.precision,
    ) == ("gpu", "ddp", 4, "bf16-mixed")
    assert production.trainer.accumulate_grad_batches == 8
    assert production.data.num_workers == 16
