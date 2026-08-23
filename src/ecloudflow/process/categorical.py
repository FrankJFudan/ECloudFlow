"""Categorical probability paths on the normalized simplex."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from ecloudflow.process.schedules import validate_time


@dataclass(frozen=True)
class CategoricalSample:
    """Store simplex probabilities and one sampled categorical realization."""

    probabilities: torch.Tensor
    classes: torch.Tensor
    time: torch.Tensor

    @property
    def t(self) -> torch.Tensor:
        """Return the interpolation time under the conventional short name."""
        return self.time


class CategoricalPath:
    """Interpolate a categorical prior distribution to one-hot data targets."""

    def __init__(self, num_classes: int, prior: torch.Tensor) -> None:
        """Validate and retain an immutable categorical prior simplex.

        :param num_classes: Number of classes in the fixed categorical
            vocabulary, at least two.
        :param prior: Finite floating normalized simplex of shape ``[C]`` on
            the device that target class tensors will use. Reduced-precision
            priors are checked in float32 and normalized once in the retained
            probability dtype.
        :raises ValueError: If class count, prior shape/dtype/finiteness/device,
            non-negativity, or normalization is invalid.
        """
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two.")
        _validate_prior(prior, num_classes)
        self.num_classes = num_classes
        probability_dtype = _probability_dtype(prior.dtype)
        stable_prior = prior.to(dtype=probability_dtype)
        self._prior = (stable_prior / stable_prior.sum()).detach().clone()

    @property
    def prior(self) -> torch.Tensor:
        """Return a safe cloned view of the validated categorical prior."""
        return self._prior.clone()

    def probabilities(self, target: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Return ``(1-t) prior + t one_hot(target)`` on the class simplex.

        :param target: ``torch.long`` class indices of arbitrary shape on the
            configured prior device, each in ``[0,C)``.
        :param time: Floating scalar or target-prefix batch tensor on the same
            device with values in ``[0,1]``.
        :return: Normalized finite probabilities with shape ``target.shape+[C]``
            and float32 dtype for FP16/BF16/FP32 priors or float64 for float64
            priors, on the prior device.
        :rtype: torch.Tensor
        :raises ValueError: If classes, time, device, or finite simplex
            contracts are invalid.

        ``t=0`` is exactly the configured prior and ``t=1`` is exactly the data
        one-hot distribution. One canonical prior representation is used at
        endpoints and in the interior, avoiding a normalization discontinuity.
        The affine interpolation uses this retained representation directly;
        consequently its simplex sum is accurate to the retained dtype without
        an endpoint-specific renormalization step. Its time derivative is
        ``one_hot(target)-prior``, including at both endpoints. The calculation
        does not mutate either input and preserves autograd through a floating
        time tensor.
        """
        _validate_target(target, self.num_classes, self._prior.device)
        expanded_time = _expand_time(time, target)
        one_hot = functional.one_hot(target, self.num_classes).to(
            dtype=self._prior.dtype
        )
        prior = self._prior.reshape(*([1] * target.ndim), self.num_classes)
        weights = expanded_time.to(dtype=self._prior.dtype).unsqueeze(-1)
        probabilities = (1.0 - weights) * prior + weights * one_hot
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("categorical probabilities must remain finite.")
        return probabilities

    def sample(
        self,
        target: torch.Tensor,
        time: torch.Tensor,
        *,
        fixed_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> CategoricalSample:
        """Return simplex probabilities and a generator-controlled class draw.

        :param target: ``torch.long`` data classes of arbitrary shape on the
            prior device, with indices in the configured vocabulary.
        :param time: Floating scalar or target-prefix batch time in ``[0,1]``.
        :param fixed_mask: Optional ``torch.bool`` mask matching ``target``;
            true fragment entries are restored to their exact data one-hot
            probabilities and target classes at every time.
        :param generator: Optional torch generator controlling categorical
            draws; equal compatible seeds give deterministic class samples.
        :return: Immutable normalized simplex probabilities and class indices
            with target shape.
        :rtype: CategoricalSample
        :raises ValueError: If class, time, mask, shape, device, or dtype
            contracts are invalid.
        :raises RuntimeError: If the supplied generator is incompatible with
            the class-probability device.

        Sampling does not mutate target, masks, or the retained prior. Fixed
        entries are selected functionally, never approximately renormalized;
        generated class draws have no gradient while probability interpolation
        remains differentiable with respect to time.
        """
        probabilities = self.probabilities(target, time)
        fixed = _validate_fixed_mask(fixed_mask, target)
        target_one_hot = functional.one_hot(target, self.num_classes).to(
            dtype=probabilities.dtype
        )
        if fixed is not None:
            probabilities = torch.where(
                fixed.unsqueeze(-1), target_one_hot, probabilities
            )
        if target.numel() == 0:
            classes = torch.empty_like(target)
        else:
            classes = torch.multinomial(
                _multinomial_probabilities(probabilities).reshape(-1, self.num_classes),
                1,
                replacement=True,
                generator=generator,
            ).reshape_as(target)
        if fixed is not None:
            classes = torch.where(fixed, target, classes)
        return CategoricalSample(
            probabilities=probabilities, classes=classes, time=time
        )

    def endpoint_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        fixed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return editable mean cross entropy at the data endpoint ``t=1``.

        :param logits: Finite floating dtype unnormalized class scores with
            shape ``target.shape+[C]`` on the target/prior device.
        :param target: ``torch.long`` data-endpoint classes with arbitrary
            shape and indices in ``[0,C)``.
        :param fixed_mask: Optional ``torch.bool`` mask matching target; true
            fixed fragment entries contribute neither numerator nor denominator.
        :return: Scalar cross entropy, accumulated in float32 for BF16/FP16
            logits, or a differentiable zero if no entry is editable.
        :rtype: torch.Tensor
        :raises ValueError: If logits, classes, masks, devices, dtypes, shapes,
            or finite values violate the categorical contract.

        This intentionally defines CE only for the data endpoint: it compares
        logits to clean target classes, not a fabricated target at ``t=0``.
        The selected-row denominator prevents fixed entries from biasing the
        loss. Inputs are never mutated, and gradients flow only through logits.
        """
        _validate_target(target, self.num_classes, self._prior.device)
        if (
            not isinstance(logits, torch.Tensor)
            or not logits.is_floating_point()
            or logits.shape != (*target.shape, self.num_classes)
            or logits.device != target.device
            or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError(
                "logits must be finite floating target.shape+[num_classes] on target device."
            )
        fixed = _validate_fixed_mask(fixed_mask, target)
        compute_logits = (
            logits.float()
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits
        )
        row_loss = functional.cross_entropy(
            compute_logits.reshape(-1, self.num_classes),
            target.reshape(-1),
            reduction="none",
        ).reshape(target.shape)
        if fixed is None:
            return row_loss.mean() if row_loss.numel() else compute_logits.sum() * 0.0
        editable = ~fixed
        count = editable.sum()
        if bool(count == 0):
            return compute_logits.sum() * 0.0
        return (row_loss * editable.to(dtype=row_loss.dtype)).sum() / count


def _validate_prior(prior: torch.Tensor, num_classes: int) -> None:
    """Validate one normalized categorical prior without silent normalization."""
    if (
        not isinstance(prior, torch.Tensor)
        or not prior.is_floating_point()
        or prior.ndim != 1
        or prior.shape[0] != num_classes
        or prior.numel() == 0
        or not bool(torch.isfinite(prior).all())
    ):
        raise ValueError(
            "prior must be a finite floating tensor with shape [num_classes]."
        )
    if bool((prior < 0.0).any()):
        raise ValueError("prior probabilities must be non-negative.")
    stable_prior = prior.to(dtype=_probability_dtype(prior.dtype))
    tolerance = (
        1e-8
        if prior.dtype == torch.float64
        else 4e-3
        if prior.dtype in (torch.float16, torch.bfloat16)
        else 1e-5
    )
    if not bool(
        torch.isclose(
            stable_prior.sum(),
            stable_prior.new_tensor(1.0),
            atol=tolerance,
            rtol=0.0,
        )
    ):
        raise ValueError("prior probabilities must sum to one.")
    if not bool((prior > 0.0).any()):
        raise ValueError("prior must assign positive mass to at least one class.")


def _probability_dtype(dtype: torch.dtype) -> torch.dtype:
    """Choose stable simplex arithmetic while preserving float64 probabilities."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def _multinomial_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Preserve float64 probabilities and only promote reduced precision draws."""
    return (
        probabilities.float()
        if probabilities.dtype in (torch.float16, torch.bfloat16)
        else probabilities
    )


def _validate_target(
    target: torch.Tensor, num_classes: int, device: torch.device
) -> None:
    """Validate categorical data class indices and their configured device."""
    if not isinstance(target, torch.Tensor) or target.dtype != torch.long:
        raise ValueError("target must be a torch.long tensor.")
    if target.device != device:
        raise ValueError("target must be on the categorical prior device.")
    if target.numel() and (
        bool((target < 0).any()) or bool((target >= num_classes).any())
    ):
        raise ValueError("target contains a class outside the configured vocabulary.")


def _expand_time(time: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or target-prefix time tensor to target shape."""
    validate_time(time)
    if time.device != target.device:
        raise ValueError("time must be on the target device.")
    if time.ndim > target.ndim:
        raise ValueError("time rank cannot exceed target rank.")
    reshaped = time.reshape(*time.shape, *([1] * (target.ndim - time.ndim)))
    try:
        return torch.broadcast_to(reshaped, target.shape)
    except RuntimeError as error:
        raise ValueError(
            "time must broadcast across target leading dimensions."
        ) from error


def _validate_fixed_mask(
    mask: torch.Tensor | None, target: torch.Tensor
) -> torch.Tensor | None:
    """Validate an exact fixed-entry mask without changing caller storage."""
    if mask is None:
        return None
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.shape != target.shape
        or mask.device != target.device
    ):
        raise ValueError(
            "fixed_mask must be a torch.bool tensor matching target shape/device."
        )
    return mask
