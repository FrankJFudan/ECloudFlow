"""Integration smoke tests for the graph-to-electron-field latent bridge."""

import torch

from ecloudflow.data.features import POCKET_FEATURE_NAMES
from ecloudflow.ecloud.decoder import ElectronReconstruction
from ecloudflow.ecloud.tokenizer import EquivariantFieldTokenizer


def test_graph_field_autoencoder_step_uses_canonical_pocket_schema() -> None:
    torch.manual_seed(109)
    coefficients = torch.randn(2, 3, 2, 4, requires_grad=True)
    features = torch.randn(2, 3, len(POCKET_FEATURE_NAMES))
    mask = torch.tensor([[True, True, False], [True, True, True]])
    centers = torch.randn(2, 3, 3)
    query_grid = torch.randn(2, 13, 3)
    tokenizer = EquivariantFieldTokenizer(
        n_radial=2,
        lmax=1,
        scalar_dim=12,
        vector_dim=4,
        latent_dim=20,
        chunk_size=5,
    )
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1.0e-3)

    output = tokenizer.decode(
        tokenizer(coefficients, features, mask), centers, query_grid, mask
    )
    assert isinstance(output, ElectronReconstruction)
    assert output.density.shape == (2, 13)
    assert output.gradient.shape == (2, 13, 3)
    assert output.electron_count.shape == (2,)
    assert output.dipole.shape == (2, 3)
    assert output.latent_round_trip.shape == (2, 3, 20)
    loss = sum(value.square().mean() for value in output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()
