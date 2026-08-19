"""Tests for traceable protein and ligand parsing."""

from pathlib import Path

import pytest
import torch
from rdkit import Chem

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import ElectronField
from ecloudflow.data.features import POCKET_FEATURE_NAMES
from ecloudflow.data.parsers import (
    build_complex_sample,
    parse_ligand_sdf,
    parse_pocket_pdb,
)
from ecloudflow.exceptions import DataValidationError


def test_complex_parser_centers_and_restores_ligand_pose(fixture_dir: Path) -> None:
    """A centered ligand pose must invert exactly to source SDF coordinates."""
    sample = build_complex_sample(
        fixture_dir / "complex/toy_pocket.pdb",
        fixture_dir / "complex/toy_ligand.sdf",
        sample_id="TOY",
        build_fields=False,
    )
    restored = sample.frame.to_global(sample.ligand.positions)
    assert sample.provenance.original_ligand_positions is not None
    assert torch.allclose(
        restored, sample.provenance.original_ligand_positions, atol=1e-5
    )
    assert sample.pocket.features.shape[0] == sample.pocket.positions.shape[0]
    assert sample.pocket.frame == sample.frame


def test_ligand_parser_rejects_multiple_molecules(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """A ligand path with more than one record is never silently substituted."""
    source = (fixture_dir / "complex/toy_ligand.sdf").read_text()
    path = tmp_path / "two.sdf"
    path.write_text(source + source)
    with pytest.raises(DataValidationError, match="exactly one"):
        parse_ligand_sdf(path)


def test_pocket_feature_schema_preserves_biochemical_slices(fixture_dir: Path) -> None:
    """Pocket features expose stable element, residue, charge, and chemistry slices."""
    sample = build_complex_sample(
        fixture_dir / "complex/toy_pocket.pdb",
        fixture_dir / "complex/toy_ligand.sdf",
        sample_id="FEATURES",
        build_fields=False,
    )
    assert sample.pocket.features.shape[1] == len(POCKET_FEATURE_NAMES)
    assert POCKET_FEATURE_NAMES[-7:] == (
        "backbone",
        "partial_charge",
        "donor",
        "acceptor",
        "aromatic",
        "hydrophobic",
        "metal",
    )
    assert torch.isfinite(sample.pocket.features).all()
    for name in (
        "residue_ALA",
        "backbone",
        "partial_charge",
        "donor",
        "acceptor",
        "aromatic",
        "hydrophobic",
        "metal",
    ):
        assert name in POCKET_FEATURE_NAMES
    residue = POCKET_FEATURE_NAMES.index("residue_ALA")
    backbone = POCKET_FEATURE_NAMES.index("backbone")
    charge = POCKET_FEATURE_NAMES.index("partial_charge")
    donor = POCKET_FEATURE_NAMES.index("donor")
    acceptor = POCKET_FEATURE_NAMES.index("acceptor")
    aromatic = POCKET_FEATURE_NAMES.index("aromatic")
    hydrophobic = POCKET_FEATURE_NAMES.index("hydrophobic")
    metal = POCKET_FEATURE_NAMES.index("metal")
    element_zn = POCKET_FEATURE_NAMES.index("element_Zn")
    assert sample.pocket.features[0, residue] == 1
    assert sample.pocket.features[0, backbone] == 1
    assert sample.pocket.features[0, charge] == 0
    assert sample.pocket.features[0, donor] == 1
    assert sample.pocket.features[3, acceptor] == 1
    assert sample.pocket.features[1, aromatic] == 0
    assert sample.pocket.features[1, hydrophobic] == 1
    assert sample.pocket.features[-1, metal] == 1
    assert sample.pocket.features[-1, element_zn] == 1
    assert list(parse_pocket_pdb(fixture_dir / "complex/toy_pocket.pdb").get_atoms())


def test_float64_field_is_reframed_to_float32_sample(fixture_dir: Path) -> None:
    """A float64 physical builder field is converted before sample-frame transforms."""
    sample = build_complex_sample(
        fixture_dir / "complex/toy_pocket.pdb",
        fixture_dir / "complex/toy_ligand.sdf",
        sample_id="DTYPE",
        build_fields=False,
    )
    source_frame = CoordinateFrame(origin=torch.zeros(3, dtype=torch.float64))
    field = ElectronField(
        positions=torch.zeros((1, 3), dtype=torch.float64),
        values=torch.ones((1, 1), dtype=torch.float64),
        mask=torch.ones(1, dtype=torch.bool),
        batch=torch.zeros(1, dtype=torch.long),
        channel_names=("density",),
        frame=source_frame,
    )
    from ecloudflow.data.parsers import _reframe_field

    reframed = _reframe_field(field, sample.frame)
    assert reframed.positions.dtype == sample.frame.origin.dtype
    assert reframed.values.dtype == sample.frame.origin.dtype


def test_default_field_building_uses_injected_runner_and_keeps_heavy_pose(
    fixture_dir: Path,
) -> None:
    """Default field mode builds the pocket and sends an explicit-H xTB copy."""
    from ecloudflow.data.parsers import build_complex_sample

    class PocketBuilder:
        def build(self, pocket: object) -> ElectronField:
            assert hasattr(pocket, "positions")
            frame = CoordinateFrame(origin=torch.zeros(3, dtype=torch.float64))
            return ElectronField(
                positions=torch.zeros((1, 3), dtype=torch.float64),
                values=torch.ones((1, 1), dtype=torch.float64),
                mask=torch.ones(1, dtype=torch.bool),
                batch=torch.zeros(1, dtype=torch.long),
                channel_names=("density",),
                frame=frame,
            )

    class Runner:
        executable = "fake-xtb"

        def __init__(self) -> None:
            self.received: Chem.Mol | None = None

        def calculate_ligand(self, molecule: Chem.Mol, **kwargs: object) -> object:
            self.received = molecule
            raise ValueError("tool unavailable")

    runner = Runner()
    from ecloudflow.ecloud.provenance import FieldBuilderBundle

    sample = build_complex_sample(
        fixture_dir / "complex/toy_pocket.pdb",
        fixture_dir / "complex/toy_ligand.sdf",
        sample_id="FIELDS",
        field_builders=FieldBuilderBundle(PocketBuilder(), runner),
    )
    assert sample.pocket_field is not None
    assert sample.pocket_field.positions.dtype == sample.frame.origin.dtype
    assert runner.received is not None
    assert all(atom.GetAtomicNum() != 1 for atom in runner.received.GetAtoms()) is False
    assert sample.provenance.qm is not None
    assert sample.provenance.qm.qm_mask is False
    assert sample.provenance.qm.executable == "fake-xtb"
    assert sample.provenance.qm.command == ("fake-xtb",)
