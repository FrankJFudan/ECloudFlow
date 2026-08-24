from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from ecloudflow.chemistry.relax import write_raw_and_relaxed


def _molecule() -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=11) == 0
    conf = molecule.GetConformer()
    point = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, (point.x + 1.5, point.y, point.z))
    return molecule


def _coordinates(path: Path) -> torch.Tensor:
    molecule = Chem.MolFromMolFile(str(path), removeHs=False)
    return torch.tensor(molecule.GetConformer().GetPositions())


def test_raw_pose_is_not_overwritten_by_relaxation(tmp_path):
    molecule = _molecule()
    raw_path, relaxed_path = write_raw_and_relaxed(
        molecule,
        tmp_path,
        fixed_atom_mask=torch.tensor([True] + [False] * (molecule.GetNumAtoms() - 1)),
    )
    raw = _coordinates(raw_path)
    relaxed = _coordinates(relaxed_path)
    assert torch.equal(_coordinates(raw_path), raw)
    assert torch.equal(raw[0], relaxed[0])
    assert not torch.equal(raw, relaxed)
    assert raw_path.name == "raw.sdf" and relaxed_path.name == "relaxed.sdf"
