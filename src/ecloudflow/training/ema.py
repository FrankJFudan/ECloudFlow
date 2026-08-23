"""Checkpointable exponential moving averages for successful optimizer steps."""

from __future__ import annotations

import math

import torch
from torch import nn


class ExponentialMovingAverage(nn.Module):  # type: ignore[misc]
    """Track persistent trainable-parameter averages as registered tensors.

    :param model: Module whose named parameters define the
        immutable EMA layout, shapes, dtypes, and devices.
    :param decay: Finite coefficient in ``[0,1)`` multiplying the old average.
    :return: Device/dtype-aware EMA module suitable for nested checkpoints.
    :rtype: ExponentialMovingAverage
    :raises ValueError: If decay is invalid or the model has no trainable parameter.

    Each update evaluates ``shadow = decay*shadow + (1-decay)*parameter`` under
    ``no_grad``. Shadows and the update counter are persistent buffers, so normal
    module device/dtype transfers and Lightning state dictionaries retain them.
    The separately stored live weights used by :meth:`restore` are deliberately
    ephemeral and never mistaken for resume state. No distributed collective is
    required because DDP optimizer parameters are already synchronized. Callers
    must use :meth:`update_after_step` only after a known successful optimizer step;
    skipped or failed steps leave every buffer exact and unchanged.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        super().__init__()
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be finite and in [0, 1).")
        parameters = tuple(model.named_parameters())
        if not parameters:
            raise ValueError("EMA requires at least one parameter.")
        self.decay = float(decay)
        self._parameter_names = tuple(name for name, _ in parameters)
        self._shadow_buffer_names = tuple(
            f"shadow_{index:06d}" for index in range(len(parameters))
        )
        for buffer_name, (_, parameter) in zip(
            self._shadow_buffer_names, parameters, strict=True
        ):
            self.register_buffer(
                buffer_name, parameter.detach().clone(), persistent=True
            )
        self.register_buffer(
            "num_updates", torch.zeros((), dtype=torch.long), persistent=True
        )
        self._stored: tuple[torch.Tensor, ...] | None = None

    def shadow_parameters(self) -> tuple[torch.Tensor, ...]:
        """Return registered shadow tensors in the retained parameter order."""
        return tuple(getattr(self, name) for name in self._shadow_buffer_names)

    def update(self, model: nn.Module) -> None:
        """Update every shadow after one confirmed optimizer mutation.

        :param model: Module with the same trainable named-parameter layout used
            at construction, on matching tensor devices and dtypes.
        :return: None after in-place detached buffer updates.
        :rtype: None
        :raises ValueError: If parameter names, shapes, devices, or dtypes differ.

        This method mutates only persistent EMA buffers and ``num_updates`` under
        ``torch.no_grad``. It performs no optimizer operation, rank/device move,
        or distributed communication. Consequently it must not be called on an
        AMP-overflow skip or failed step; :meth:`update_after_step` makes that
        condition explicit. Determinism follows the synchronized model values.
        Each shadow keeps the parameter shape and dtype on its device; no autograd
        gradient is retained, and checkpoint buffers update only after success.
        """
        parameters = self._validated_parameters(model)
        with torch.no_grad():
            for shadow, parameter in zip(
                self.shadow_parameters(), parameters, strict=True
            ):
                shadow.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
            self.num_updates.add_(1)

    def update_after_step(self, model: nn.Module, *, step_succeeded: bool) -> None:
        """Conditionally update only when an optimizer step succeeded.

        :param model: Model matching the EMA parameter layout.
        :param step_succeeded: Explicit success flag; false includes AMP overflow,
            missing gradients, a skipped closure, or any caller-detected failure.
        :return: None; false leaves all persistent state unchanged.
        :rtype: None
        :raises ValueError: If a successful update sees an incompatible model.

        The branch is deterministic, device independent, and has no mutation at
        all on the skipped path. On success it delegates to :meth:`update`, whose
        detached dtype/device and checkpoint semantics apply.
        """
        if step_succeeded:
            self.update(model)

    def store(self, model: nn.Module) -> None:
        """Store a separate ephemeral copy of current live model weights.

        :param model: Compatible model whose trainable parameters will later be
            restored after temporary EMA evaluation.
        :return: None after replacing the prior ephemeral store.
        :rtype: None
        :raises ValueError: If the model layout, shape, dtype, or device differs.

        Copies stay on each parameter device/dtype and are not registered in the
        EMA state dictionary. Persistent shadows and counters are not mutated.
        """
        self._stored = tuple(
            parameter.detach().clone()
            for parameter in self._validated_parameters(model)
        )

    def copy_to(self, model: nn.Module) -> None:
        """Copy persistent EMA shadows into compatible live parameters.

        :param model: Compatible model receiving shadow values in place.
        :return: None after a detached exact copy.
        :rtype: None
        :raises ValueError: If layout, shapes, dtype, or device differ.

        This mutates model parameters under ``no_grad`` but leaves EMA buffers,
        autograd graphs, optimizer state, and distributed rank state untouched.
        """
        parameters = self._validated_parameters(model)
        with torch.no_grad():
            for parameter, shadow in zip(
                parameters, self.shadow_parameters(), strict=True
            ):
                parameter.copy_(shadow)

    def restore(self, model: nn.Module) -> None:
        """Restore and consume the last separately stored live parameters.

        :param model: Compatible model receiving the ephemeral stored values.
        :return: None after exact restoration and deletion of the store.
        :rtype: None
        :raises RuntimeError: If :meth:`store` has not established a live copy.
        :raises ValueError: If the model layout, dtype, device, or shapes differ.

        Restoration mutates only live parameters under ``no_grad``; persistent
        checkpoint buffers remain unchanged. Consuming the store prevents stale
        nested validation contexts from silently restoring an earlier model.
        """
        if self._stored is None:
            raise RuntimeError("EMA restore requires a preceding store call.")
        parameters = self._validated_parameters(model)
        stored = self._stored
        with torch.no_grad():
            for parameter, value in zip(parameters, stored, strict=True):
                parameter.copy_(value)
        self._stored = None

    def _validated_parameters(self, model: nn.Module) -> tuple[nn.Parameter, ...]:
        """Return parameters after exact layout compatibility checks."""
        named = tuple(model.named_parameters())
        if tuple(name for name, _ in named) != self._parameter_names:
            raise ValueError("EMA model trainable parameter names do not match.")
        parameters = tuple(parameter for _, parameter in named)
        for parameter, shadow in zip(parameters, self.shadow_parameters(), strict=True):
            if (
                parameter.shape != shadow.shape
                or parameter.dtype != shadow.dtype
                or parameter.device != shadow.device
            ):
                raise ValueError(
                    "EMA parameter shape, dtype, or device does not match."
                )
        return parameters
