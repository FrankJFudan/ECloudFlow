"""Fragment-constrained ligand atom-count prediction."""

from __future__ import annotations

import torch
from torch import nn


class AtomCountPredictor(nn.Module):  # type: ignore[misc]
    """Predict a categorical total atom count above each fixed-fragment size.

    :param scalar_dim: Positive pooled invariant conditioning width ``S``.
    :param max_atoms: Positive inclusive largest represented atom count.
    :return: Trainable categorical count predictor.
    :rtype: AtomCountPredictor
    :raises ValueError: If dimensions are not positive.

    Counts are invariant scalars. The predictor is stateless and device
    agnostic, so DDP/FSDP ranks receive only their local tensors.
    """

    def __init__(self, scalar_dim: int, max_atoms: int = 64) -> None:
        super().__init__()
        if scalar_dim <= 0 or max_atoms <= 0:
            raise ValueError("scalar_dim and max_atoms must be positive.")
        self.max_atoms = max_atoms
        self.head = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, max_atoms + 1),
        )

    def forward(
        self, pooled: torch.Tensor, fixed_fragment_counts: torch.Tensor
    ) -> torch.distributions.Categorical:
        """Return a categorical distribution with exact per-row lower bounds.

        :param pooled: Finite invariant complex features ``[B,S]``.
        :param fixed_fragment_counts: Long fixed-atom counts ``[B]`` on the
            same device, each in ``[0,max_atoms]``.
        :return: Batch-shaped categorical distribution over ``0..max_atoms``;
            every probability below its row's fixed count is exactly zero.
        :rtype: torch.distributions.Categorical
        :raises ValueError: If shape, dtype/device, finiteness, or bounds fail.

        Masking occurs on logits before normalization, is deterministic, does
        not mutate inputs, and retains finite gradients for every allowed count.
        The boolean mask and long count dtype stay on the pooled tensor device;
        no CPU transfer or distributed state is involved.
        """
        if (
            pooled.ndim != 2
            or not pooled.is_floating_point()
            or not torch.isfinite(pooled).all()
        ):
            raise ValueError("pooled count features must be finite floating [B, S].")
        if (
            fixed_fragment_counts.shape != (pooled.shape[0],)
            or fixed_fragment_counts.dtype != torch.long
            or fixed_fragment_counts.device != pooled.device
        ):
            raise ValueError(
                "fixed fragment counts must be long [B] on the pooled device."
            )
        if bool((fixed_fragment_counts < 0).any()) or bool(
            (fixed_fragment_counts > self.max_atoms).any()
        ):
            raise ValueError(
                "fixed fragment count must be within the configured atom range."
            )
        logits = self.head(pooled)
        counts = torch.arange(self.max_atoms + 1, device=pooled.device)
        logits = logits.masked_fill(
            counts[None, :] < fixed_fragment_counts[:, None], -torch.inf
        )
        return torch.distributions.Categorical(logits=logits)
