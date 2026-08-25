"""Failure-oriented coverage for checkpoint and fragment production sampling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from rdkit import Chem
from typer.testing import CliRunner

from ecloudflow.chemistry.projector import ChemicalProjector
from ecloudflow.chemistry.vocabulary import ChemicalVocabulary
from ecloudflow.cli.main import app
from ecloudflow.core import FragmentCondition
from ecloudflow.models import ECloudFlowModel
from ecloudflow.pipeline import ECloudFlowPipeline, _prepare_fragment_condition
from ecloudflow.sampling.pipeline import (
    _fragment_allowed_bond_mask,
    build_fragment_condition,
)
from ecloudflow.sampling.profiles import SamplingProfile
from ecloudflow.sampling.results import GenerationMode

FIXTURES = Path(__file__).parents[1] / "fixtures" / "complex"


def _zero_checkpoint(path: Path) -> Path:
    """Write a small ordinary Lightning checkpoint with deterministic weights."""
    model = ECloudFlowModel(
        scalar_dim=16,
        vector_dim=4,
        num_blocks=1,
        lmax=2,
        electron_latent_dim=48,
        electron_vector_dim=8,
        atom_classes=12,
        charge_classes=5,
        bond_classes=4,
        max_atoms=32,
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    torch.save(
        {
            "state_dict": {
                f"joint_backbone.{name}": value
                for name, value in model.state_dict().items()
            }
        },
        path,
    )
    return path


def test_lightning_state_dict_constructs_model_and_samples(tmp_path: Path) -> None:
    """A normal checkpoint must execute model sampling instead of returning zero."""
    checkpoint = _zero_checkpoint(tmp_path / "model.ckpt")
    pipeline = ECloudFlowPipeline.from_pretrained(checkpoint)
    observed_fields = []
    original_forward = pipeline.model.forward

    def capture_field(state, time, condition):
        observed_fields.append(condition.pocket_field)
        return original_forward(state, time, condition)

    pipeline.model.forward = capture_field

    result = pipeline.generate(
        FIXTURES / "toy_pocket.pdb",
        1,
        profile=SamplingProfile("fast", "euler", 1, 0),
        max_attempts=1,
        output_dir=tmp_path / "run",
    )

    assert isinstance(pipeline.model, ECloudFlowModel)
    assert not pipeline.model.training
    assert observed_fields and all(field is not None for field in observed_fields)
    assert observed_fields[0].channel_names == (
        "density",
        "partial_charge",
        "donor",
        "acceptor",
        "hydrophobic",
        "aromatic",
    )
    assert result.valid_count == 1
    assert result.valid[0].molecule.GetNumConformers() == 1


def test_incomplete_lightning_state_dict_fails_before_sampling(tmp_path: Path) -> None:
    """Malformed production checkpoints cannot degrade into a zero-result run."""
    checkpoint = tmp_path / "broken.ckpt"
    torch.save({"state_dict": {"joint_backbone.unrelated": torch.ones(1)}}, checkpoint)

    with pytest.raises(ValueError, match="not an ECloudFlow"):
        ECloudFlowPipeline.from_pretrained(checkpoint)


def test_explicit_model_receives_lightning_prefixed_weights(tmp_path: Path) -> None:
    """Caller-created ECloudFlow models still receive strict checkpoint weights."""
    checkpoint = _zero_checkpoint(tmp_path / "model.ckpt")
    model = ECloudFlowModel(
        scalar_dim=16,
        vector_dim=4,
        num_blocks=1,
        lmax=2,
        electron_latent_dim=48,
        electron_vector_dim=8,
        atom_classes=12,
        charge_classes=5,
        bond_classes=4,
        max_atoms=32,
    )
    assert any(bool(parameter.detach().ne(0).any()) for parameter in model.parameters())

    pipeline = ECloudFlowPipeline.from_pretrained(checkpoint, model=model)

    assert pipeline.model is model
    assert all(bool(parameter.detach().eq(0).all()) for parameter in model.parameters())


def test_fragment_file_becomes_exact_condition_and_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    """File-backed fragment modes expose exact masks and reject changed output."""
    captured: list[FragmentCondition] = []

    def tampered_generator(*, fragment: FragmentCondition, **kwargs: object) -> str:
        del kwargs
        captured.append(fragment)
        return "CCO"

    pipeline = ECloudFlowPipeline(
        candidate_generator=tampered_generator,
        checkpoint_hash="fixture",
    )
    result = pipeline.generate(
        FIXTURES / "toy_pocket.pdb",
        1,
        fragment=FIXTURES / "toy_ligand.sdf",
        mode="grow",
        profile=SamplingProfile("fast", "euler", 1, 0),
        max_attempts=1,
        output_dir=tmp_path / "fragment-run",
    )

    assert len(captured) == 1
    condition = captured[0]
    assert condition.task_id == "grow"
    assert int(condition.fixed_atom_mask.sum()) == 6
    assert torch.equal(condition.fixed_coord_mask, condition.fixed_atom_mask)
    assert result.valid_count == 0
    assert "fragment_invariant_failed:atom_count_changed" in result.attempt_records[0].reason


def test_fragment_condition_preserves_source_identity_charge_bonds_and_pose() -> None:
    """SDF conversion keeps each supplied atom field in the pocket binding frame."""
    condition = build_fragment_condition(
        FIXTURES / "toy_pocket.pdb",
        FIXTURES / "toy_ligand.sdf",
        extra_atoms=2,
        seed=17,
        task_id="link",
    )
    molecule = next(
        item
        for item in Chem.SDMolSupplier(
            str(FIXTURES / "toy_ligand.sdf"), sanitize=True, removeHs=True
        )
        if item is not None
    )
    source_positions = torch.tensor(molecule.GetConformer().GetPositions())
    fixed_count = molecule.GetNumAtoms()
    reference = condition.reference

    assert condition.task_id == "link"
    assert condition.fixed_atom_mask.tolist() == [True] * fixed_count + [False, False]
    assert torch.allclose(
        reference.frame.to_global(reference.positions[:fixed_count]),
        source_positions.to(dtype=reference.positions.dtype),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert int(condition.fixed_bond_mask.sum()) == fixed_count * (fixed_count - 1) // 2


def test_fragment_attachment_mask_is_conservative_for_sdf_inputs() -> None:
    """Saturated fixed atoms cannot receive arbitrary generated attachments."""
    condition = build_fragment_condition(
        FIXTURES / "toy_pocket.pdb",
        FIXTURES / "toy_ligand.sdf",
        extra_atoms=1,
        task_id="grow",
    )

    # The ether oxygen in the fixture has no available hydrogen/valence site;
    # it must not be treated as a generic fixed-to-free attachment endpoint.
    assert not bool(condition.attachment_mask[2])
    assert bool(condition.attachment_mask[:6].any())
    allowed = _fragment_allowed_bond_mask(condition.reference, condition)
    assert allowed is not None
    edge_rows = {
        tuple(pair): int(index)
        for index, pair in enumerate(condition.reference.halfedge_index.t().tolist())
    }
    assert not bool(allowed[edge_rows[(2, 6)]])
    assert bool(allowed[edge_rows[(0, 6)]])


def test_multiple_fragment_sources_remain_separate_and_linkable() -> None:
    """Link tasks preserve two source components while exposing cross edges."""
    condition = build_fragment_condition(
        FIXTURES / "toy_pocket.pdb",
        [FIXTURES / "toy_ligand.sdf", FIXTURES / "toy_ligand.sdf"],
        extra_atoms=1,
        task_id="link",
    )

    assert condition.component_ids is not None
    assert int(condition.fixed_atom_mask.sum()) == 12
    assert set(condition.component_ids[:12].tolist()) == {0, 1}
    edge_rows = {
        tuple(pair): int(index)
        for index, pair in enumerate(condition.reference.halfedge_index.t().tolist())
    }
    # Internal source bonds/non-bonds remain fixed; a pair across the two
    # supplied components is deliberately left editable for linking.
    assert bool(condition.fixed_bond_mask[edge_rows[(0, 1)]])
    assert not bool(condition.fixed_bond_mask[edge_rows[(0, 6)]])
    allowed = _fragment_allowed_bond_mask(condition.reference, condition)
    assert allowed is not None and bool(allowed[edge_rows[(0, 6)]])


def test_projector_keeps_cross_component_fixed_pairs_editable() -> None:
    """Valence projection must not freeze a linker edge between fragments."""
    condition = build_fragment_condition(
        FIXTURES / "toy_pocket.pdb",
        [FIXTURES / "toy_ligand.sdf", FIXTURES / "toy_ligand.sdf"],
        extra_atoms=0,
        task_id="link",
    )
    projected = ChemicalProjector(ChemicalVocabulary.default_ligand()).project(
        condition.reference, condition
    )
    edge_rows = {
        tuple(pair): int(index)
        for index, pair in enumerate(condition.reference.halfedge_index.t().tolist())
    }

    assert not bool(condition.fixed_bond_mask[edge_rows[(0, 6)]])
    assert bool(projected.allowed_new_bonds[edge_rows[(0, 6)]])


@pytest.mark.parametrize("mixed_sources", [False, True])
def test_rdkit_fragment_sequence_is_normalized_by_public_pipeline(
    mixed_sources: bool,
) -> None:
    """The public pipeline normalizes Mol and mixed fragment sequences exactly."""
    source = next(
        item
        for item in Chem.SDMolSupplier(
            str(FIXTURES / "toy_ligand.sdf"), sanitize=True, removeHs=True
        )
        if item is not None
    )
    captured: list[FragmentCondition] = []

    def candidate_generator(*, fragment: FragmentCondition, **kwargs: object) -> str:
        del kwargs
        captured.append(fragment)
        return "CCO"

    pipeline = ECloudFlowPipeline(
        candidate_generator=candidate_generator,
        checkpoint_hash="fixture",
    )
    fragment = (
        [FIXTURES / "toy_ligand.sdf", Chem.Mol(source)]
        if mixed_sources
        else [Chem.Mol(source), Chem.Mol(source)]
    )
    pipeline.generate(
        FIXTURES / "toy_pocket.pdb",
        1,
        fragment=fragment,
        mode=GenerationMode.LINK,
        profile=SamplingProfile("fast", "euler", 1, 0),
        max_attempts=1,
    )

    assert len(captured) == 1
    assert int(captured[0].fixed_atom_mask.sum()) == 12
    assert captured[0].component_ids is not None


def test_model_fragment_count_can_equal_fixed_atom_count(monkeypatch) -> None:
    """A count head may validly choose a no-growth fixed-fragment state."""
    model = ECloudFlowModel(
        scalar_dim=16,
        vector_dim=4,
        num_blocks=1,
        lmax=2,
        electron_latent_dim=48,
        electron_vector_dim=8,
        atom_classes=12,
        charge_classes=5,
        bond_classes=4,
        max_atoms=6,
    )
    monkeypatch.setattr(
        "ecloudflow.pipeline.predict_atom_count",
        lambda *args, **kwargs: 6,
    )

    condition = _prepare_fragment_condition(
        FIXTURES / "toy_pocket.pdb",
        FIXTURES / "toy_ligand.sdf",
        GenerationMode.GROW,
        7,
        model,
    )

    assert isinstance(condition, FragmentCondition)
    assert condition.reference.positions.shape == (6, 3)
    assert int(condition.fixed_atom_mask.sum()) == 6


def test_cli_hydra_sampling_overrides_survive_default_options(tmp_path: Path) -> None:
    """Absent CLI options cannot overwrite explicit resolved sampling values."""
    result = CliRunner().invoke(
        app,
        [
            "sample",
            str(FIXTURES / "toy_pocket.pdb"),
            "--smoke",
            "--docking",
            "none",
            "--output-dir",
            str(tmp_path),
            "-O",
            "sample.num_molecules=2",
            "-O",
            "sample.num_steps=3",
            "-O",
            "sample.corrector_steps=4",
            "-O",
            "sample.solver=euler",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "resolved-config.json").read_text("utf-8"))
    assert payload["config"]["sample"] == {
        "corrector_steps": 4,
        "max_attempts": None,
        "num_molecules": 2,
        "num_steps": 3,
        "profile": "balanced",
        "resolved_max_attempts": 10,
        "solver": "euler",
    }


def test_cli_strict_count_publishes_artifacts_and_exits_nonzero(tmp_path: Path) -> None:
    """Strict shortfalls remain inspectable but are command failures."""
    result = CliRunner().invoke(
        app,
        [
            "sample",
            str(FIXTURES / "toy_pocket.pdb"),
            "-n",
            "2",
            "--max-attempts",
            "1",
            "--strict-count",
            "--smoke",
            "--docking",
            "none",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert (tmp_path / "generation.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    generation = json.loads((tmp_path / "generation.json").read_text("utf-8"))
    assert generation["valid_count"] == 1
    assert generation["shortfall"] == 1
