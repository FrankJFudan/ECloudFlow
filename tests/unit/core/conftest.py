"""Deterministic local fixture factories for core-contract tests."""

from collections.abc import Callable

import pytest
import torch

from ecloudflow.core.types import MolecularState


@pytest.fixture
def molecular_state_factory() -> Callable[[int], MolecularState]:
    """Return a deterministic factory for canonical three-atom states.

    :return: Callable accepting ``num_atoms`` and returning a CPU state with
        canonical flattened unordered halfedges.
    :rtype: collections.abc.Callable[[int], MolecularState]
    """
    def build(num_atoms: int = 3) -> MolecularState:
        """Build one deterministic state for the requested fixture size.

        :param num_atoms: Number of atoms; only the three-atom fixture exists.
        :return: Canonical CPU molecular state with three unordered halfedges.
        :rtype: MolecularState
        :raises ValueError: If a size other than three is requested.
        """
        if num_atoms != 3:
            raise ValueError("The deterministic fixture contains exactly three atoms.")
        return MolecularState(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            atom_logits=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
            ),
            charge_logits=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
            ),
            halfedge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
            bond_logits=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
            ),
            electron_latent=torch.tensor(
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            ),
            node_batch=torch.zeros(3, dtype=torch.long),
            halfedge_batch=torch.zeros(3, dtype=torch.long),
        )

    return build
