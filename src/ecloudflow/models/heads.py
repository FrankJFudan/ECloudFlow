"""Invariant node, pair, and auxiliary heads for the joint backbone."""

from __future__ import annotations

import torch
from torch import nn


class SymmetricPairHead(nn.Module):  # type: ignore[misc]
    """Map unordered endpoint features and distance to one invariant value."""

    def __init__(self, scalar_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        if scalar_dim <= 0 or output_dim <= 0:
            raise ValueError("scalar_dim and output_dim must be positive.")
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(2 * scalar_dim + 1, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, output_dim),
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, distance: torch.Tensor
    ) -> torch.Tensor:
        """Predict one value per pair using commutative endpoint statistics."""
        features = torch.cat(
            (source + target, (source - target).abs(), distance[:, None]), dim=-1
        )
        output = self.network(features)
        return output.squeeze(-1) if self.output_dim == 1 else output


class ScalarHead(nn.Module):  # type: ignore[misc]
    """Map invariant hidden features to one invariant scalar per row."""

    def __init__(self, scalar_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        if scalar_dim <= 0 or output_dim <= 0:
            raise ValueError("scalar_dim and output_dim must be positive.")
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one scalar for every input row."""
        output = self.network(features)
        return output.squeeze(-1) if self.output_dim == 1 else output
