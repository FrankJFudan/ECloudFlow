from __future__ import annotations

import torch
from Bio.PDB import Atom, Chain, Model, Residue, Structure
from rdkit import Chem
from rdkit.Chem import AllChem

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import PocketGraph
from ecloudflow.ecloud.pocket import POCKET_CHANNELS, PocketFieldBuilder
from ecloudflow.ecloud.provenance import ECLOUDGEN_CUBE_ATTRIBUTION, FieldBuilderBundle
from ecloudflow.ecloud.xtb import XTBRunner


def toy_pocket_with_donor_acceptor_and_metal() -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles("[Zn+2].c1ccccc1O.CC(=O)N"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=23) == 0
    return molecule


def test_pocket_field_is_deterministic_and_centered():
    pocket = toy_pocket_with_donor_acceptor_and_metal()
    builder = PocketFieldBuilder.default(spacing=1.0, padding=1.0)
    first = builder.build(pocket)
    second = builder.build(pocket)
    assert first.channel_names == (
        "density",
        "partial_charge",
        "donor",
        "acceptor",
        "hydrophobic",
        "aromatic",
    )
    assert first.frame == second.frame
    assert first.values.equal(second.values)
    assert torch.allclose(
        first.positions.mean(dim=0),
        torch.zeros(3, dtype=first.positions.dtype),
        atol=1e-6,
    )


def test_pocket_channels_are_finite_and_physically_distinct_with_metal():
    field = PocketFieldBuilder.default(spacing=1.0, padding=1.0).build(
        toy_pocket_with_donor_acceptor_and_metal()
    )
    assert field.channel_names == POCKET_CHANNELS
    assert torch.isfinite(field.values).all()
    assert bool((field.values[:, 0] > 0).any())
    for channel in range(1, len(POCKET_CHANNELS)):
        assert not torch.equal(field.values[:, 0], field.values[:, channel])
    assert bool((field.values[:, 2] > 0).any())
    assert bool((field.values[:, 3] > 0).any())
    assert bool((field.values[:, 5] > 0).any())


def test_pocket_graph_default_path_handles_metal():
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    frame = CoordinateFrame(torch.tensor([10.0, 0.0, 0.0]))
    pocket = PocketGraph(
        positions=positions,
        features=torch.zeros((2, 1)),
        batch=torch.zeros(2, dtype=torch.long),
        atom_numbers=torch.tensor([6, 30], dtype=torch.long),
        frame=frame,
    )
    field = PocketFieldBuilder.default(spacing=1.0, padding=1.0).build(pocket)
    assert torch.isfinite(field.values).all()
    assert field.positions.dtype == pocket.positions.dtype
    assert field.frame == frame


def test_field_builder_bundle_and_attribution_are_complete(tmp_path):
    bundle = FieldBuilderBundle.default(
        ligand_builder=XTBRunner(executable="xtb", work_root=tmp_path)
    )
    assert isinstance(bundle.pocket_builder, PocketFieldBuilder)
    assert isinstance(bundle.ligand_builder, XTBRunner)
    assert ECLOUDGEN_CUBE_ATTRIBUTION.license_identifier == "NOASSERTION"
    assert "9c0e3b1c" in ECLOUDGEN_CUBE_ATTRIBUTION.revision
    assert ECLOUDGEN_CUBE_ATTRIBUTION.source_url.startswith("https://")


def test_biopython_pocket_builds_physical_channels():
    structure = Structure.Structure("TOY")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    residue = Residue.Residue((" ", 1, " "), "TYR", " ")
    residue.add(Atom.Atom("OH", [0.0, 0.0, 0.0], 1.0, 1.0, " ", " OH ", 1, element="O"))
    residue.add(Atom.Atom("CA", [1.5, 0.0, 0.0], 1.0, 1.0, " ", " CA ", 2, element="C"))
    chain.add(residue)
    model.add(chain)
    structure.add(model)
    field = PocketFieldBuilder.default(spacing=1.0, padding=1.0).build(structure)
    assert field.channel_names == POCKET_CHANNELS
    assert field.frame is not None
    assert torch.isfinite(field.values).all()
    assert bool((field.values[:, 2] > 0).any())
