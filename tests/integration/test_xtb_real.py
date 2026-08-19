from __future__ import annotations

import shutil

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ecloudflow.ecloud.xtb import QMStatus, XTBRunner


@pytest.mark.external
def test_real_xtb_density_smoke():
    executable = shutil.which("xtb")
    if executable is None:
        pytest.skip("xTB executable is not available; external smoke test skipped.")
    molecule = Chem.AddHs(Chem.MolFromSmiles("C"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=31) == 0
    result = XTBRunner(executable=executable, timeout=180.0).calculate_ligand(
        molecule, charge=0, multiplicity=1
    )
    assert result.status is QMStatus.SUCCESS
    assert result.qm_mask is True
    assert result.density is not None
