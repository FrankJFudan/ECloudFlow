"""Invariant node, pair, and auxiliary heads for the joint backbone."""

from __future__ import annotations

import torch
from torch import nn


class SymmetricPairHead(nn.Module):  # type: ignore[misc]
    """Map unordered endpoint features and distance to one invariant value."""

    def __init__(self, scalar_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * scalar_dim + 1, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, 1),
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, distance: torch.Tensor
    ) -> torch.Tensor:
        """Predict one value per pair using commutative endpoint statistics."""
        features = torch.cat(
            (source + target, (source - target).abs(), distance[:, None]), dim=-1
        )
        return self.network(features).squeeze(-1)


class ScalarHead(nn.Module):  # type: ignore[misc]
    """Map invariant hidden features to one invariant scalar per row."""

    def __init__(self, scalar_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one scalar for every input row."""
        return self.network(features).squeeze(-1)
