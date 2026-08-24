"""Deterministic Euler and Heun integration for molecular state paths."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ecloudflow.core.types import MolecularState
from ecloudflow.exceptions import SamplingNumericsError

VectorFieldCallable = Callable[..., Any]
StateHook = Callable[..., MolecularState]


def apply_state_hooks(
    state: MolecularState,
    hooks: Sequence[StateHook],
    time_value: torch.Tensor,
    generator: torch.Generator,
) -> MolecularState:
    """Apply the canonical 3/2/1-argument hook chain and finite check."""
    for hook in hooks:
        try:
            state = hook(state, time_value, generator)
        except TypeError as first:
            try:
                state = hook(state, time_value)
            except TypeError:
                try:
                    state = hook(state)
                except TypeError:
                    raise first
        if not isinstance(state, MolecularState):
            raise TypeError("state hooks must return MolecularState.")
    _validate_finite(state)
    return state


@dataclass(frozen=True)
class SamplingTrajectory:
    """Retain final state, optional frames, and numerical diagnostics."""

    final: MolecularState
    frames: tuple[MolecularState, ...] = ()
    nfe: int = 0
    wall_time: float = 0.0
    diagnostics: dict[str, Any] | None = None
    condition: Any = None

    @property
    def function_evaluations(self) -> int:
        """Return the number of vector-field evaluations."""
        return self.nfe

    @property
    def final_state(self) -> MolecularState:
        """Return the final state under the explicit API spelling."""
        return self.final

    @property
    def timings(self) -> dict[str, float]:
        """Return wall-clock timing diagnostics."""
        return {"wall_time": self.wall_time}


def _call_field(
    field: VectorFieldCallable, state: MolecularState, t: torch.Tensor
) -> Any:
    """Call common vector-field signatures without masking user errors."""
    try:
        return field(state, t)
    except TypeError as first:
        try:
            return field(t, state)
        except TypeError:
            raise first


def _derivative_state(value: Any, state: MolecularState) -> MolecularState:
    if isinstance(value, MolecularState):
        return value
    if isinstance(value, dict):
        zeros = {
            name: torch.zeros_like(getattr(state, name))
            for name in (
                "positions",
                "atom_logits",
                "charge_logits",
                "bond_logits",
                "electron_latent",
            )
        }
        zeros.update({k: v for k, v in value.items() if k in zeros})
        return state.replace(**zeros)
    if isinstance(value, (tuple, list)) and len(value) == 7:
        return state.replace(
            positions=value[0],
            atom_logits=value[1],
            charge_logits=value[2],
            bond_logits=value[3],
            electron_latent=value[4],
        )
    raise TypeError("vector field must return MolecularState or a state-field mapping.")


def _step(
    state: MolecularState, deriv: MolecularState, dt: torch.Tensor
) -> MolecularState:
    changes: dict[str, torch.Tensor] = {}
    for name in (
        "positions",
        "atom_logits",
        "charge_logits",
        "bond_logits",
        "electron_latent",
    ):
        candidate = getattr(state, name) + dt * getattr(deriv, name)
        # Preserve normalized categorical simplex semantics when inputs are probabilities.
        if (
            name != "positions"
            and candidate.numel()
            and bool((getattr(state, name) >= 0).all())
        ):
            old = getattr(state, name)
            if bool((old >= 0).all()) and bool(
                torch.allclose(
                    old.sum(-1), torch.ones_like(old.sum(-1)), atol=1e-4, rtol=1e-4
                )
            ):
                candidate = candidate.clamp_min(0)
                candidate = candidate / candidate.sum(-1, keepdim=True).clamp_min(
                    torch.finfo(candidate.dtype).tiny
                )
        changes[name] = candidate
    return state.replace(**changes)


def _validate_finite(state: MolecularState) -> None:
    for name in (
        "positions",
        "atom_logits",
        "charge_logits",
        "bond_logits",
        "electron_latent",
    ):
        if not bool(torch.isfinite(getattr(state, name)).all()):
            raise SamplingNumericsError(f"non-finite values in {name} during sampling.")


class _BaseSolver:
    def __init__(self, num_steps: int, *, save_every_step: bool = False) -> None:
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")
        self.num_steps = num_steps
        self.save_every_step = save_every_step

    def _prepare(self, state: MolecularState, generator: torch.Generator) -> None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator owned by the caller.")
        if generator.device != state.positions.device:
            raise ValueError(
                f"generator device {generator.device} does not match state device {state.positions.device}."
            )

    def _hook(
        self,
        state: MolecularState,
        hooks: Sequence[StateHook],
        t: torch.Tensor,
        generator: torch.Generator,
    ) -> MolecularState:
        return apply_state_hooks(state, hooks, t, generator)

    def integrate(
        self,
        state: MolecularState,
        vector_field: VectorFieldCallable,
        hooks: Sequence[StateHook] = (),
        generator: torch.Generator | None = None,
    ) -> SamplingTrajectory:
        """Integrate a molecular state with post-substep constraint hooks.

        :param state: Initial flattened molecular state.
        :param vector_field: Callable returning a state-shaped derivative.
        :param hooks: Ordered clamping/projection/recording callbacks.
        :param generator: Device-matched caller-owned random generator.
        :return: Trajectory with final state, optional frames, NFE and timing.
        :rtype: SamplingTrajectory
        :raises SamplingNumericsError: If any state tensor becomes non-finite.
        """
        rng = generator or torch.Generator(device=state.positions.device)
        self._prepare(state, rng)
        started = time.perf_counter()
        # Normalize the initial state through the same exact constraints used
        # after each substep, so retained trajectories have no unconstrained
        # frame at index zero.
        current = self._hook(
            state,
            hooks,
            torch.tensor(
                0.0, dtype=state.positions.dtype, device=state.positions.device
            ),
            rng,
        )
        frames: list[MolecularState] = [current] if self.save_every_step else []
        nfe = 0
        for index in range(self.num_steps):
            t = torch.tensor(
                index / self.num_steps,
                dtype=state.positions.dtype,
                device=state.positions.device,
            )
            t_next = torch.tensor(
                (index + 1) / self.num_steps,
                dtype=state.positions.dtype,
                device=state.positions.device,
            )
            current, evaluations = self._advance(current, vector_field, t, t_next)
            nfe += evaluations
            current = self._hook(current, hooks, t_next, rng)
            if self.save_every_step:
                frames.append(current)
        return SamplingTrajectory(
            current,
            tuple(frames),
            nfe,
            time.perf_counter() - started,
            {"num_steps": self.num_steps},
        )

    def _advance(
        self,
        state: MolecularState,
        field: VectorFieldCallable,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> tuple[MolecularState, int]:
        raise NotImplementedError


class EulerSolver(_BaseSolver):
    """Integrate with first-order explicit Euler updates."""

    def _advance(
        self,
        state: MolecularState,
        field: VectorFieldCallable,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> tuple[MolecularState, int]:
        derivative = _derivative_state(_call_field(field, state, t), state)
        _validate_finite(derivative)
        return _step(state, derivative, t_next - t), 1


class HeunSolver(_BaseSolver):
    """Integrate with second-order predictor-corrector Heun updates."""

    def _advance(
        self,
        state: MolecularState,
        field: VectorFieldCallable,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> tuple[MolecularState, int]:
        dt = t_next - t
        first = _derivative_state(_call_field(field, state, t), state)
        _validate_finite(first)
        predicted = _step(state, first, dt)
        _validate_finite(predicted)
        second = _derivative_state(_call_field(field, predicted, t_next), predicted)
        _validate_finite(second)
        average = state.replace(
            positions=0.5 * (first.positions + second.positions),
            atom_logits=0.5 * (first.atom_logits + second.atom_logits),
            charge_logits=0.5 * (first.charge_logits + second.charge_logits),
            bond_logits=0.5 * (first.bond_logits + second.bond_logits),
            electron_latent=0.5 * (first.electron_latent + second.electron_latent),
        )
        corrected = _step(state, average, dt)
        return corrected, 2
