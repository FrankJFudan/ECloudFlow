"""Langevin score correction with exact constraint hooks."""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch

from ecloudflow.core.types import MolecularState
from ecloudflow.sampling.solver import (
    SamplingTrajectory,
    StateHook,
    VectorFieldCallable,
    _call_field,
    _derivative_state,
    _validate_finite,
    apply_state_hooks,
)


class ScoreCorrector:
    """Apply configurable Langevin corrections to editable state channels.

    :param snr: Positive signal-to-noise ratio controlling adaptive step size.
    :param steps: Number of correction iterations.
    :param edit_mask: Optional boolean node mask; false nodes are unchanged.
    :return: Reusable score corrector.
    :rtype: ScoreCorrector
    """

    def __init__(
        self,
        snr: float = 0.16,
        steps: int = 1,
        *,
        edit_mask: torch.Tensor | None = None,
    ) -> None:
        if not torch.isfinite(torch.tensor(snr)) or snr <= 0:
            raise ValueError("snr must be finite and positive.")
        self._validate_steps(steps, "steps")
        self.snr, self.steps, self.edit_mask = float(snr), int(steps), edit_mask

    def correct(
        self,
        state: MolecularState,
        score: VectorFieldCallable,
        hooks: Sequence[StateHook] = (),
        generator: torch.Generator | None = None,
        *,
        steps: int | None = None,
    ) -> SamplingTrajectory:
        """Run Langevin score updates and invoke hooks after every substep.

        :param state: Current state in the centered pocket frame.
        :param score: Score callable returning a state-shaped derivative.
        :param hooks: Ordered exact-clamp and projection callbacks.
        :param generator: Device-matched caller-owned random generator.
        :param steps: Optional override for correction count.
        :return: Corrected trajectory with NFE and wall-time diagnostics.
        :rtype: SamplingTrajectory
        :raises SamplingNumericsError: If a score update is non-finite.
        """
        rng = generator or torch.Generator(device=state.positions.device)
        if rng.device != state.positions.device:
            raise ValueError("generator device must match state device.")
        count = (
            self.steps
            if steps is None
            else self._validate_steps(steps, "steps override")
        )
        initial_time = torch.tensor(
            1.0, dtype=state.positions.dtype, device=state.positions.device
        )
        current = apply_state_hooks(state, hooks, initial_time, rng)
        frames: list[MolecularState] = [current]
        started, nfe = time.perf_counter(), 0
        mask = self.edit_mask
        if mask is not None:
            if (
                mask.shape != (state.positions.shape[0],)
                or mask.device != state.positions.device
            ):
                raise ValueError("edit_mask must have shape [N] on the state device.")
            mask = mask[:, None]
        for index in range(count):
            t = torch.tensor(
                1.0, dtype=state.positions.dtype, device=state.positions.device
            )
            derivative = _derivative_state(_call_field(score, current, t), current)
            _validate_finite(derivative)
            nfe += 1
            # Adaptive Langevin step from the score/noise ratio. A zero score
            # has no deterministic direction, so it receives no random kick;
            # near-zero fields are capped to keep the update finite.
            grad_norm = derivative.positions.norm(dim=-1).mean()
            norm_floor = torch.tensor(
                torch.finfo(state.positions.dtype).eps,
                dtype=state.positions.dtype,
                device=state.positions.device,
            )
            noise_norm = torch.sqrt(
                torch.tensor(
                    3.0, dtype=state.positions.dtype, device=state.positions.device
                )
            )
            ratio = self.snr * noise_norm / grad_norm.clamp_min(norm_floor)
            step = (ratio.square() * 2.0).clamp(max=1.0)
            step = torch.where(grad_norm <= norm_floor, torch.zeros_like(step), step)
            noise = torch.randn(
                current.positions.shape,
                generator=rng,
                device=current.positions.device,
                dtype=current.positions.dtype,
            )
            position_delta = (
                step * derivative.positions + torch.sqrt(2.0 * step) * noise
            )
            if mask is not None:
                position_delta = position_delta * mask
            current = current.replace(positions=current.positions + position_delta)
            current = self._apply_other_channels(current, derivative, step, mask, rng)
            current = apply_state_hooks(current, hooks, t, rng)
            frames.append(current)
        return SamplingTrajectory(
            current,
            tuple(frames),
            nfe,
            time.perf_counter() - started,
            {"corrector_steps": count},
        )

    @staticmethod
    def _validate_steps(value: int, name: str) -> int:
        """Validate an integer correction count, excluding booleans."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return value

    @staticmethod
    def _apply_other_channels(
        state: MolecularState,
        derivative: MolecularState,
        step: torch.Tensor,
        mask: torch.Tensor | None,
        rng: torch.Generator,
    ) -> MolecularState:
        changes: dict[str, torch.Tensor] = {}
        for name in ("atom_logits", "charge_logits", "bond_logits", "electron_latent"):
            value = getattr(state, name) + step * getattr(derivative, name)
            if mask is not None and value.shape[0] == mask.shape[0]:
                value = torch.where(mask, value, getattr(state, name))
            changes[name] = value
        updated = state.replace(**changes)
        for name in ("atom_logits", "charge_logits", "bond_logits"):
            values = getattr(updated, name)
            original = getattr(state, name)
            if (
                values.numel()
                and bool((original >= 0).all())
                and bool(
                    torch.allclose(
                        original.sum(-1),
                        torch.ones_like(original.sum(-1)),
                        atol=1e-4,
                        rtol=1e-4,
                    )
                )
            ):
                values = values.clamp_min(0)
                values = values / values.sum(-1, keepdim=True).clamp_min(
                    torch.finfo(values.dtype).tiny
                )
                updated = updated.replace(**{name: values})
        return updated
