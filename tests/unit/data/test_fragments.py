"""Tests for deterministic ring-aware fragment task construction."""

from rdkit import Chem
from rdkit.Chem import AllChem

from ecloudflow.data.fragments import FragmentMode, FragmentTaskSampler


def medicinal_ligand_fixture() -> Chem.Mol:
    """Return a small aromatic molecule with both ring and linker bonds."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccccc1C(=O)N"))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    return mol


def test_fragment_sampler_builds_all_four_optimization_modes() -> None:
    """Every supported optimization mode produces a non-empty fixed fragment."""
    ligand = medicinal_ligand_fixture()
    sampler = FragmentTaskSampler(seed=23)
    modes = {sampler.sample(ligand, forced_mode=mode).mode for mode in FragmentMode}
    assert modes == set(FragmentMode)
    for mode in FragmentMode:
        task = sampler.sample(ligand, forced_mode=mode)
        assert task.fixed_atom_mask.any()
        assert task.fixed_coord_mask.equal(task.fixed_atom_mask)
        assert task.attachment_mask[task.fixed_atom_mask].shape[0] == int(
            task.fixed_atom_mask.sum()
        )


def test_ring_atoms_are_not_split_by_ring_cut() -> None:
    """Ring-aware policies preserve complete ring systems when selecting cuts."""
    ligand = medicinal_ligand_fixture()
    task = FragmentTaskSampler(seed=7).sample(ligand, forced_mode=FragmentMode.REPLACE)
    for ring in ligand.GetRingInfo().AtomRings():
        ring_mask = task.fixed_atom_mask[list(ring)]
        assert bool(ring_mask.all()) or not bool(ring_mask.any())
