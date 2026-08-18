"""Coordinate-frame definitions for centered molecular complexes."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ecloudflow.exceptions import CoordinateFrameError


@dataclass(frozen=True, eq=False)
class CoordinateFrame:
    """Describe a proper rigid local frame relative to global coordinates.

    ``rotation`` maps local column vectors to global column vectors. With row
    vector tensors, the equivalent equations are ``global = local @ rotation.T
    + origin`` and ``local = (global - origin) @ rotation``. Pocket centering
    uses an identity rotation, preserving the global axes while shifting the
    pocket centroid to the local origin.

    :param origin: Global-frame translation with shape ``[3]``, a finite
        floating tensor in angstroms on the frame device.
    :param rotation: Optional proper orthonormal matrix with shape ``[3, 3]``
        and the same floating dtype/device as ``origin``. ``None`` selects the
        identity rotation.
    :return: Immutable local-to-global coordinate transform.
    :rtype: CoordinateFrame
    :raises CoordinateFrameError: If tensor ranks, devices, dtypes, finite
        values, orthonormality, or handedness are invalid.

    The transform operates only on the final Cartesian dimension and does not
    modify input tensors. Floating-point round trips are accurate to ordinary
    matrix-multiplication precision; they are not guaranteed bitwise exact.
    """

    origin: torch.Tensor
    rotation: torch.Tensor | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate and normalize frame tensors.

        :return: None.
        :rtype: None
        :raises CoordinateFrameError: If the origin or rotation is not a finite
            proper three-dimensional floating-point transform.
        """
        _validate_vector(self.origin, "origin")
        rotation = self.rotation
        if rotation is None:
            rotation = torch.eye(3, dtype=self.origin.dtype, device=self.origin.device)
            object.__setattr__(self, "rotation", rotation)
        _validate_matrix(rotation, "rotation")
        if rotation.device != self.origin.device or rotation.dtype != self.origin.dtype:
            raise CoordinateFrameError(
                "rotation must have the same dtype and device as origin."
            )
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        if not torch.allclose(
            rotation.transpose(0, 1) @ rotation,
            identity,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise CoordinateFrameError("rotation must be orthonormal.")
        if not torch.allclose(
            torch.linalg.det(rotation),
            torch.ones((), dtype=rotation.dtype, device=rotation.device),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise CoordinateFrameError("rotation must be a proper rotation.")

    @classmethod
    def from_pocket(cls, positions: torch.Tensor) -> CoordinateFrame:
        """Construct an identity-orientation frame centered on pocket atoms.

        :param positions: Global pocket coordinates with shape ``[P, 3]`` for
            ``P >= 1``, floating dtype/device, and angstrom units.
        :return: Frame whose origin is the arithmetic pocket centroid in the
            global frame and whose local axes equal the global axes.
        :rtype: CoordinateFrame
        :raises CoordinateFrameError: If positions are empty, malformed,
            non-floating, or non-finite.

        The centroid is evaluated in the input dtype and device. This is a
        translation-only frame: no principal-axis alignment is performed, so
        rotations applied to an input complex remain explicit and traceable.
        """
        _validate_points(positions, "positions", require_nonempty=True)
        return cls(origin=positions.mean(dim=0))

    def to_local(self, points: torch.Tensor) -> torch.Tensor:
        """Transform global Cartesian points into this centered local frame.

        :param points: Global coordinates with shape ``[M, 3]``, the same
            floating dtype/device as the frame, and angstrom units.
        :return: Local coordinates with shape ``[M, 3]``, input dtype/device,
            and angstrom units.
        :rtype: torch.Tensor
        :raises CoordinateFrameError: If points are malformed, non-finite, or
            incompatible with the frame dtype/device.

        For row-vector tensors the computation is ``(points - origin) @
        rotation``. The operation is differentiable with respect to points,
        origin, and rotation when those tensors require gradients.
        """
        self._validate_points_for_transform(points)
        rotation = self.rotation
        if rotation is None:
            raise CoordinateFrameError("rotation must be initialized.")
        return (points - self.origin) @ rotation

    def to_global(self, points: torch.Tensor) -> torch.Tensor:
        """Transform local Cartesian points into the original global frame.

        :param points: Local coordinates with shape ``[M, 3]``, the same
            floating dtype/device as the frame, and angstrom units.
        :return: Global coordinates with shape ``[M, 3]``, input dtype/device,
            and angstrom units.
        :rtype: torch.Tensor
        :raises CoordinateFrameError: If points are malformed, non-finite, or
            incompatible with the frame dtype/device.

        For row-vector tensors the computation is ``points @ rotation.T +
        origin``. It is the numerical inverse of :meth:`to_local` for a valid
        proper orthonormal rotation.
        """
        self._validate_points_for_transform(points)
        rotation = self.rotation
        if rotation is None:
            raise CoordinateFrameError("rotation must be initialized.")
        return points @ rotation.transpose(0, 1) + self.origin

    def _validate_points_for_transform(self, points: torch.Tensor) -> None:
        """Validate transform input against this frame.

        :param points: Candidate Cartesian points with shape ``[M, 3]``.
        :return: None.
        :rtype: None
        :raises CoordinateFrameError: If points do not match this frame.
        """
        _validate_points(points, "points", require_nonempty=False)
        if points.dtype != self.origin.dtype or points.device != self.origin.device:
            raise CoordinateFrameError(
                "points must have the same dtype and device as the frame."
            )

    def __eq__(self, other: object) -> bool:
        """Compare frame tensor values without triggering tensor truth errors.

        :param other: Object to compare with this frame.
        :return: ``True`` when both frame tensors are exactly equal.
        :rtype: bool
        """
        if not isinstance(other, CoordinateFrame):
            return False
        self_rotation = self.rotation
        other_rotation = other.rotation
        return (
            self_rotation is not None
            and other_rotation is not None
            and torch.equal(self.origin, other.origin)
            and torch.equal(self_rotation, other_rotation)
        )


def _validate_vector(value: torch.Tensor, name: str) -> None:
    """Validate one finite Cartesian vector.

    :param value: Candidate tensor with expected shape ``[3]``.
    :param name: Human-readable tensor name for diagnostic messages.
    :return: None.
    :rtype: None
    :raises CoordinateFrameError: If the tensor is not finite floating ``[3]``.
    """
    if not isinstance(value, torch.Tensor):
        raise CoordinateFrameError(f"{name} must be a torch.Tensor.")
    if value.shape != (3,):
        raise CoordinateFrameError(f"{name} must have shape [3].")
    if not value.is_floating_point():
        raise CoordinateFrameError(f"{name} must have a floating dtype.")
    if not torch.isfinite(value).all():
        raise CoordinateFrameError(f"{name} must contain only finite values.")


def _validate_matrix(value: torch.Tensor, name: str) -> None:
    """Validate one finite Cartesian matrix.

    :param value: Candidate tensor with expected shape ``[3, 3]``.
    :param name: Human-readable tensor name for diagnostic messages.
    :return: None.
    :rtype: None
    :raises CoordinateFrameError: If the tensor is not finite floating ``[3, 3]``.
    """
    if not isinstance(value, torch.Tensor):
        raise CoordinateFrameError(f"{name} must be a torch.Tensor.")
    if value.shape != (3, 3):
        raise CoordinateFrameError(f"{name} must have shape [3, 3].")
    if not value.is_floating_point():
        raise CoordinateFrameError(f"{name} must have a floating dtype.")
    if not torch.isfinite(value).all():
        raise CoordinateFrameError(f"{name} must contain only finite values.")


def _validate_points(
    value: torch.Tensor, name: str, *, require_nonempty: bool
) -> None:
    """Validate one finite batch of Cartesian points.

    :param value: Candidate tensor with expected shape ``[M, 3]``.
    :param name: Human-readable tensor name for diagnostic messages.
    :param require_nonempty: Whether ``M`` must be positive.
    :return: None.
    :rtype: None
    :raises CoordinateFrameError: If the tensor fails coordinate requirements.
    """
    if not isinstance(value, torch.Tensor):
        raise CoordinateFrameError(f"{name} must be a torch.Tensor.")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise CoordinateFrameError(f"{name} must have shape [M, 3].")
    if require_nonempty and value.shape[0] == 0:
        raise CoordinateFrameError(f"{name} must contain at least one point.")
    if not value.is_floating_point():
        raise CoordinateFrameError(f"{name} must have a floating dtype.")
    if not torch.isfinite(value).all():
        raise CoordinateFrameError(f"{name} must contain only finite values.")
