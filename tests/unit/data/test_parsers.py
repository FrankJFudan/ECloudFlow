"""Tests for traceable protein and ligand parsing."""

from pathlib import Path

import pytest
import torch

from ecloudflow.data.parsers import build_complex_sample, parse_ligand_sdf
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
