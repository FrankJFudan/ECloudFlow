"""Cavity-aware initial distributions for molecular trajectories."""

from __future__ import annotations

from typing import Any

import torch

from ecloudflow.core.types import MolecularState


class CavityAwarePrior:
    """Sample free atom coordinates from a pocket cavity support.

    The prior deliberately accepts a small structural protocol instead of a
    concrete cavity implementation: a cavity may provide ``sample`` or
    ``bounds``/``center``/``radius`` and a ``contains`` predicate. Rejection
    sampling is used for bounds so returned free atoms remain supported.

    :param seed: Optional seed used only to initialize this prior's generator.
    :param atom_channels: Number of atom simplex channels for generated states.
    :param charge_channels: Number of charge simplex channels.
    :param bond_channels: Number of bond simplex channels.
    :param latent_channels: Number of electron latent channels.
    :return: Reusable cavity-aware prior.
    :rtype: CavityAwarePrior
    """

    def __init__(
        self,
        seed: int | None = None,
        *,
        atom_channels: int = 2,
        charge_channels: int = 2,
        bond_channels: int = 2,
        latent_channels: int = 1,
    ) -> None:
        if min(atom_channels, charge_channels, bond_channels, latent_channels) < 1:
            raise ValueError("channel counts must be positive.")
        self.seed = seed
        self.atom_channels = atom_channels
        self.charge_channels = charge_channels
        self.bond_channels = bond_channels
        self.latent_channels = latent_channels

    def sample(
        self,
        condition: Any,
        num_atoms: int,
        *,
        generator: torch.Generator | None = None,
    ) -> MolecularState:
        """Draw an initial state in the cavity's centered coordinate frame.

        :param condition: Pocket condition with ``cavity`` and optional
            fragment/reference state attributes.
        :param num_atoms: Number of nodes in the returned flattened state.
        :param generator: Caller-owned device-matched random generator.
        :return: Molecular state with empty candidate halfedges and simplex
            categorical priors. Existing fixed fragment values are copied.
        :rtype: MolecularState
        :raises ValueError: If atom count, cavity geometry, or device metadata
            is invalid.
        """
        if not isinstance(num_atoms, int) or num_atoms < 1:
            raise ValueError("num_atoms must be a positive integer.")
        reference = getattr(getattr(condition, "fragment", None), "reference", None)
        if reference is not None:
            device, dtype = reference.positions.device, reference.positions.dtype
            atom_channels = reference.atom_logits.shape[1]
            charge_channels = reference.charge_logits.shape[1]
            bond_channels = reference.bond_logits.shape[1]
            latent_channels = reference.electron_latent.shape[1]
        else:
            device = getattr(
                getattr(condition, "pocket", None), "positions", torch.empty(0)
            ).device
            dtype = getattr(
                getattr(condition, "pocket", None),
                "positions",
                torch.empty(0, dtype=torch.float32),
            ).dtype
            atom_channels, charge_channels = self.atom_channels, self.charge_channels
            bond_channels, latent_channels = self.bond_channels, self.latent_channels
        rng = generator or torch.Generator(device=device)
        if generator is None and self.seed is not None:
            rng.manual_seed(self.seed)
        cavity = getattr(condition, "cavity", None)
        if cavity is None:
            cavity = getattr(getattr(condition, "pocket", None), "cavity", None)
        positions = self._sample_positions(cavity, num_atoms, dtype, device, rng)
        node_batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
        edges = torch.empty((2, 0), dtype=torch.long, device=device)
        state = MolecularState(
            positions=positions,
            atom_logits=torch.full(
                (num_atoms, atom_channels),
                1.0 / atom_channels,
                dtype=dtype,
                device=device,
            ),
            charge_logits=torch.full(
                (num_atoms, charge_channels),
                1.0 / charge_channels,
                dtype=dtype,
                device=device,
            ),
            halfedge_index=edges,
            bond_logits=torch.full(
                (0, bond_channels), 1.0 / bond_channels, dtype=dtype, device=device
            ),
            electron_latent=torch.zeros(
                (num_atoms, latent_channels), dtype=dtype, device=device
            ),
            node_batch=node_batch,
            halfedge_batch=torch.empty(0, dtype=torch.long, device=device),
            frame=getattr(reference, "frame", None),
        )
        if reference is not None and num_atoms == reference.positions.shape[0]:
            fixed = getattr(condition, "fragment", None)
            if fixed is not None:
                from ecloudflow.core.masks import clamp_fragment

                state = clamp_fragment(state, fixed)
        return state

    @staticmethod
    def _sample_positions(
        cavity: Any,
        count: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if cavity is not None and callable(getattr(cavity, "sample", None)):
            try:
                values = cavity.sample(count, generator=generator)
            except TypeError:
                values = cavity.sample(count)
            values = torch.as_tensor(values, dtype=dtype, device=device)
            if values.shape != (count, 3):
                raise ValueError("cavity.sample must return [N, 3] coordinates.")
            CavityAwarePrior._validate_support(cavity, values, count, device)
            return values
        bounds = getattr(cavity, "bounds", None)
        if bounds is not None:
            lower, upper = (
                torch.as_tensor(v, dtype=dtype, device=device) for v in bounds
            )
            if lower.shape != (3,) or upper.shape != (3,):
                raise ValueError("cavity bounds must contain two [3] vectors.")
            points = lower + torch.rand(
                (count, 3), generator=generator, device=device, dtype=dtype
            ) * (upper - lower)
            contains = getattr(cavity, "contains", None)
            if callable(contains):
                for _ in range(64):
                    mask = torch.as_tensor(
                        contains(points), device=device, dtype=torch.bool
                    )
                    if bool(mask.all()):
                        break
                    replacement = lower + torch.rand(
                        (count, 3), generator=generator, device=device, dtype=dtype
                    ) * (upper - lower)
                    points = torch.where(mask[:, None], points, replacement)
                CavityAwarePrior._validate_support(cavity, points, count, device)
            return points
        center = torch.as_tensor(
            getattr(cavity, "center", (0.0, 0.0, 0.0)), dtype=dtype, device=device
        )
        if center.numel() != 3:
            raise ValueError("cavity center must contain three coordinates.")
        radius = float(getattr(cavity, "radius", getattr(cavity, "half_extent", 5.0)))
        if radius <= 0:
            raise ValueError("cavity radius must be positive.")
        points = (
            center
            + (
                torch.rand((count, 3), generator=generator, device=device, dtype=dtype)
                * 2
                - 1
            )
            * radius
        )
        contains = getattr(cavity, "contains", None)
        if callable(contains):
            for _ in range(128):
                mask = torch.as_tensor(
                    contains(points), device=device, dtype=torch.bool
                )
                if bool(mask.all()):
                    break
                replacement = (
                    center
                    + (
                        torch.rand(
                            (count, 3), generator=generator, device=device, dtype=dtype
                        )
                        * 2
                        - 1
                    )
                    * radius
                )
                points = torch.where(mask[:, None], points, replacement)
            CavityAwarePrior._validate_support(cavity, points, count, device)
        return points

    @staticmethod
    def _validate_support(
        cavity: Any, points: torch.Tensor, count: int, device: torch.device
    ) -> None:
        """Reject a bounded rejection draw that still violates cavity support."""
        contains = getattr(cavity, "contains", None)
        if not callable(contains):
            return
        mask = torch.as_tensor(contains(points), device=device, dtype=torch.bool)
        if mask.shape != (count,):
            raise ValueError(
                "cavity.contains must return a boolean mask with shape [N]."
            )
        if not bool(mask.all()):
            raise ValueError(
                "cavity rejection sampling exhausted its bounded attempts."
            )
