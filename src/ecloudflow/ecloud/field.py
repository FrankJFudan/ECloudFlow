"""Differentiable projection and moment reductions for electron densities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import ElectronField
from ecloudflow.ecloud.basis import SphericalFieldBasis


class MultipoleMoments(NamedTuple):
    """Electron-number monopole, dipole, and traceless quadrupole tensors."""

    electron_count: torch.Tensor
    dipole: torch.Tensor
    quadrupole: torch.Tensor

    @property
    def monopole(self) -> torch.Tensor:
        """Return ``electron_count`` under the conventional multipole name."""
        return self.electron_count


class BatchedMultipoleMoments(NamedTuple):
    """Masked per-complex moments with explicit batch and frame provenance."""

    batch: torch.Tensor
    electron_count: torch.Tensor
    dipole: torch.Tensor
    quadrupole: torch.Tensor
    frame: CoordinateFrame


@dataclass(frozen=True)
class AtomCenteredFieldCoefficients:
    """Bind equivariant atom coefficients to batches, centers, frame, and basis.

    :param values: Coefficients with shape ``[N, R, (lmax+1)**2]``.
    :param centers: Atom centers with shape ``[N, 3]`` in angstroms.
    :param batch: Non-negative complex index per center with shape ``[N]`` and
        ``torch.long`` dtype.
    :param frame: Required coordinate frame shared by the source field and
        centers.
    :param basis: Exact radial/angular basis used for projection.
    :return: Immutable atom-centered field with numerical provenance.
    :rtype: AtomCenteredFieldCoefficients
    :raises TypeError: If ``frame`` or ``basis`` has an invalid type.
    :raises ValueError: If tensor shapes, finite values, dtypes, devices, or
        batch indices are incompatible with the frame and basis.

    Keeping frame, batch, and basis metadata with ``values`` prevents a valid
    coefficient tensor from being silently reconstructed in another complex,
    coordinate frame, or radial convention.
    """

    values: torch.Tensor
    centers: torch.Tensor
    batch: torch.Tensor
    frame: CoordinateFrame
    basis: SphericalFieldBasis

    def __post_init__(self) -> None:
        """Validate bound coefficient layout and provenance."""
        if not isinstance(self.frame, CoordinateFrame):
            raise TypeError("frame must be a CoordinateFrame.")
        if not isinstance(self.basis, SphericalFieldBasis):
            raise TypeError("basis must be a SphericalFieldBasis.")
        _validate_reconstruction_inputs(
            self.values, self.centers, self.centers, self.basis
        )
        _validate_batch_vector(self.batch, self.centers.shape[0], self.centers)
        if (
            self.frame.origin.dtype != self.centers.dtype
            or self.frame.origin.device != self.centers.device
        ):
            raise ValueError(
                "frame must have the same dtype and device as coefficient centers."
            )


def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    """Choose float32 accumulation for low-precision floating inputs."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _validate_float_tensor(tensor: torch.Tensor, name: str) -> None:
    """Validate one finite floating tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_chunk_size(chunk_size: int) -> None:
    """Validate a bounded positive chunk length."""
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")


def _validate_density_quadrature(
    density: torch.Tensor,
    integration_weights: torch.Tensor,
    *,
    require_nonnegative_density: bool,
) -> None:
    """Validate an unbatched scalar field and matching volume elements."""
    _validate_float_tensor(density, "density")
    _validate_float_tensor(integration_weights, "integration_weights")
    if density.ndim != 1:
        raise ValueError("density must have shape [G].")
    if integration_weights.shape != density.shape:
        raise ValueError("integration_weights must have shape [G] matching density.")
    if density.numel() == 0:
        raise ValueError("density and integration_weights must not be empty.")
    if density.device != integration_weights.device:
        raise ValueError("density and integration_weights must use the same device.")
    if density.dtype != integration_weights.dtype:
        raise ValueError("density and integration_weights must use the same dtype.")
    if bool((integration_weights <= 0).any()):
        raise ValueError("integration_weights must be strictly positive.")
    if require_nonnegative_density and bool((density < 0).any()):
        raise ValueError("density must be non-negative.")


def _validate_positions(
    positions: torch.Tensor,
    name: str,
    *,
    expected_count: int | None = None,
) -> None:
    """Validate finite unbatched coordinates in one three-dimensional frame."""
    _validate_float_tensor(positions, name)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3].")
    if expected_count is not None and positions.shape[0] != expected_count:
        raise ValueError(f"{name} must have shape [{expected_count}, 3].")


def _validate_batch_vector(
    batch: torch.Tensor, expected_count: int, reference: torch.Tensor
) -> None:
    """Validate non-negative long batch membership against a reference tensor."""
    if not isinstance(batch, torch.Tensor):
        raise TypeError("batch must be a torch.Tensor.")
    if batch.shape != (expected_count,):
        raise ValueError(f"batch must have shape [{expected_count}].")
    if batch.dtype != torch.long:
        raise ValueError("batch must have torch.long dtype.")
    if batch.device != reference.device:
        raise ValueError("batch must use the same device as coordinates.")
    if bool((batch < 0).any()):
        raise ValueError("batch indices must be non-negative.")


def _validate_field_weights(
    field: ElectronField, integration_weights: torch.Tensor
) -> None:
    """Validate quadrature metadata against an immutable electron field."""
    _validate_float_tensor(integration_weights, "integration_weights")
    if integration_weights.shape != field.mask.shape:
        raise ValueError("integration_weights must have shape [G] matching field.")
    if integration_weights.dtype != field.positions.dtype:
        raise ValueError("integration_weights and field must use the same dtype.")
    if integration_weights.device != field.positions.device:
        raise ValueError("integration_weights and field must use the same device.")
    if bool((integration_weights <= 0).any()):
        raise ValueError("integration_weights must be strictly positive.")


def _require_field_frame(field: ElectronField) -> CoordinateFrame:
    """Return required field frame provenance or reject its absence."""
    if field.frame is None:
        raise ValueError("ElectronField.frame is required for field numerics.")
    return field.frame


def _density_channel_index(field: ElectronField, density_channel: str | int) -> int:
    """Resolve one scalar density channel without guessing multi-channel data."""
    channel_count = field.values.shape[1]
    if isinstance(density_channel, bool):
        raise TypeError("density_channel must be an integer or channel name.")
    if isinstance(density_channel, int):
        if density_channel < 0 or density_channel >= channel_count:
            raise ValueError("density_channel index is out of range.")
        return density_channel
    if not isinstance(density_channel, str):
        raise TypeError("density_channel must be an integer or string.")
    if field.channel_names:
        try:
            return field.channel_names.index(density_channel)
        except ValueError as error:
            raise ValueError(
                f"ElectronField has no {density_channel!r} channel."
            ) from error
    if density_channel == "density" and channel_count == 1:
        return 0
    raise ValueError(
        "a named density channel requires channel_names, except a single channel "
        "may use the default 'density' name."
    )


def _validate_projection_inputs(
    density: torch.Tensor,
    grid: torch.Tensor,
    centers: torch.Tensor,
    integration_weights: torch.Tensor,
) -> None:
    """Validate projection tensor shapes and shared numerical placement."""
    _validate_density_quadrature(
        density, integration_weights, require_nonnegative_density=True
    )
    _validate_positions(grid, "grid", expected_count=density.shape[0])
    _validate_positions(centers, "centers")
    if centers.shape[0] == 0:
        raise ValueError("centers must contain at least one atom center.")
    tensors = (density, grid, centers, integration_weights)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError(
            "density, grid, centers, and weights must use the same device."
        )
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("density, grid, centers, and weights must use the same dtype.")


def _validate_reconstruction_inputs(
    coefficients: torch.Tensor,
    grid: torch.Tensor,
    centers: torch.Tensor,
    basis: SphericalFieldBasis,
) -> None:
    """Validate a packed coefficient tensor and its coordinate metadata."""
    _validate_float_tensor(coefficients, "coefficients")
    _validate_positions(grid, "grid")
    _validate_positions(centers, "centers")
    expected = (centers.shape[0], basis.n_radial, basis.harmonic_dim)
    if coefficients.shape != expected:
        raise ValueError(f"coefficients must have shape {expected}.")
    if centers.shape[0] == 0:
        raise ValueError("centers must contain at least one atom center.")
    if coefficients.device != grid.device or centers.device != grid.device:
        raise ValueError("coefficients, grid, and centers must use the same device.")
    if centers.dtype != grid.dtype:
        raise ValueError("grid and centers must use the same dtype.")
    stable_low_precision = (
        grid.dtype in (torch.float16, torch.bfloat16)
        and coefficients.dtype == torch.float32
    )
    if coefficients.dtype != grid.dtype and not stable_low_precision:
        raise ValueError(
            "coefficients must match the coordinate dtype, except stable float32 "
            "coefficients may accompany FP16/BF16 coordinates."
        )


def _atom_partition(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Return smooth atom weights including an implicit outside component."""
    scaled = distances / cutoff
    transition = ((scaled - 0.8) / 0.2).clamp(0.0, 1.0)
    smootherstep = (
        6.0 * transition.pow(5) - 15.0 * transition.pow(4) + 10.0 * transition.pow(3)
    )
    scores = torch.where(
        scaled <= 0.8,
        torch.ones_like(scaled),
        torch.where(scaled < 1.0, 1.0 - smootherstep, torch.zeros_like(scaled)),
    )
    normalization = scores.sum(dim=1, keepdim=True)
    coverage = 1.0 - torch.prod(1.0 - scores, dim=1, keepdim=True)
    return torch.where(
        normalization > 0,
        coverage * scores / normalization.clamp_min(torch.finfo(scores.dtype).tiny),
        torch.zeros_like(scores),
    )


def project_density_to_atoms(
    density: torch.Tensor,
    grid: torch.Tensor,
    centers: torch.Tensor,
    integration_weights: torch.Tensor,
    basis: SphericalFieldBasis,
) -> torch.Tensor:
    """Project a sampled density into atom-centered equivariant coefficients.

    :param density: Non-negative density values with shape ``[G]`` in
        electrons per cubic angstrom.
    :param grid: Grid coordinates with shape ``[G, 3]`` in angstroms.
    :param centers: Atom centers with shape ``[N, 3]`` in the same coordinate
        frame, floating dtype and device as ``grid``.
    :param integration_weights: Strictly positive quadrature volumes with
        shape ``[G]`` in cubic angstroms.
    :param basis: Radial and real-spherical-harmonic basis definition.
    :return: Coefficients with shape ``[N, R, (lmax + 1) ** 2]`` on the input
        device. FP16/BF16 inputs return float32 coefficients; other dtypes are
        preserved. With the normalized basis, coefficient units are electrons
        per angstrom to the power ``3/2``.
    :rtype: torch.Tensor
    :raises TypeError: If an input is not a tensor or ``basis`` has the wrong
        type.
    :raises ValueError: If density is negative or non-finite; shapes, dtypes,
        or devices disagree; weights are not positive; or centers are empty.

    For overlapping cutoff balls, smooth distance-only atom weights plus an
    implicit outside component assign each covered sample without electron
    double counting. The weights have a unit plateau through 80% of the cutoff
    and a quintic zero-value/zero-slope transition at the boundary. They remain
    invariant to joint rotations and translations.
    Coefficients in ``basis.l_slice(l)`` transform under the matching real
    e3nn irrep. Grid chunks cap intermediate storage at approximately
    ``[chunk_size, N, R, H]``. Autocast is disabled internally and FP16/BF16
    data accumulate in float32 to limit electron-count drift. A rank-one
    constrained-Galerkin correction along the basis-derived monopole-integral
    vector makes the continuous reconstructed count equal the quadrature count
    assigned to each center. This is the low-level unbatched tensor kernel;
    :func:`project_electron_field_to_atoms` is the binding contract path for
    masked, framed, or batched ``ElectronField`` data.
    """
    if not isinstance(basis, SphericalFieldBasis):
        raise TypeError("basis must be a SphericalFieldBasis.")
    _validate_projection_inputs(density, grid, centers, integration_weights)
    dtype = _working_dtype(density.dtype)
    with torch.autocast(device_type=density.device.type, enabled=False):
        work_density = density.to(dtype=dtype)
        work_grid = grid.to(dtype=dtype)
        work_centers = centers.to(dtype=dtype)
        work_weights = integration_weights.to(dtype=dtype)
        coefficients = torch.zeros(
            (centers.shape[0], basis.n_radial, basis.harmonic_dim),
            dtype=dtype,
            device=density.device,
        )
        assigned_electrons = torch.zeros(
            centers.shape[0], dtype=dtype, device=density.device
        )
        for start in range(0, grid.shape[0], basis.chunk_size):
            stop = min(start + basis.chunk_size, grid.shape[0])
            displacements = work_grid[start:stop, None, :] - work_centers[None, :, :]
            distances = torch.linalg.vector_norm(displacements, dim=-1)
            partition = _atom_partition(distances, basis.cutoff)
            basis_values = basis.evaluate(displacements)
            volume_density = work_density[start:stop] * work_weights[start:stop]
            contribution = torch.einsum(
                "g,gn,gnrh->nrh",
                volume_density,
                partition,
                basis_values,
            )
            coefficients = coefficients + contribution
            assigned_electrons = assigned_electrons + torch.einsum(
                "g,gn->n", volume_density, partition
            )
        monopole_integrals = basis.monopole_integrals(dtype, density.device)
        represented_electrons = coefficients[..., 0] @ monopole_integrals
        correction_direction = monopole_integrals / monopole_integrals.square().sum()
        corrected_monopoles = (
            coefficients[..., 0]
            + (assigned_electrons - represented_electrons).unsqueeze(-1)
            * correction_direction
        )
        return torch.cat(
            (corrected_monopoles.unsqueeze(-1), coefficients[..., 1:]), dim=-1
        )


def reconstruct_density(
    coefficients: torch.Tensor,
    grid: torch.Tensor,
    centers: torch.Tensor,
    basis: SphericalFieldBasis,
) -> torch.Tensor:
    """Reconstruct density samples from atom-centered equivariant coefficients.

    :param coefficients: Packed real-irrep coefficients with shape
        ``[N, R, H]``, where ``H=(lmax+1)**2`` and the ``l`` block occupies
        ``basis.l_slice(l)``.
    :param grid: Query coordinates with shape ``[G, 3]`` in angstroms.
    :param centers: The matching projection centers with shape ``[N, 3]`` in
        the same coordinate frame and on the same device as ``grid``.
    :param basis: Basis whose radial and angular sizes match ``coefficients``.
    :return: Reconstructed values with shape ``[G]`` in electrons per cubic
        angstrom. Low-precision inputs return float32; other dtypes are kept.
    :rtype: torch.Tensor
    :raises TypeError: If an input is not a tensor or ``basis`` is invalid.
    :raises ValueError: If layouts, dtypes, devices, coordinate shapes, or
        finite-value requirements are violated.

    Atom contributions are summed after smooth source assignment. Projection's
    conservation-constrained Galerkin correction and the basis-derived
    ``Y00`` volume integrals make each reconstructed monopole integrate to its
    assigned electron count; finite-grid agreement is therefore a real
    quadrature invariant rather than a fitted fixture constant. Signed values
    may occur in a truncated basis. Evaluation is differentiable, regular at
    atom centers, smooth at cutoffs, chunked over queries, and performed
    outside autocast with at least float32 arithmetic.
    Jointly rotating both ``grid`` and ``centers`` while transforming each
    coefficient ``l`` block leaves the scalar reconstruction invariant.
    """
    if not isinstance(basis, SphericalFieldBasis):
        raise TypeError("basis must be a SphericalFieldBasis.")
    _validate_reconstruction_inputs(coefficients, grid, centers, basis)
    dtype = (
        torch.float32
        if (
            coefficients.dtype in (torch.float16, torch.bfloat16)
            or grid.dtype in (torch.float16, torch.bfloat16)
        )
        else coefficients.dtype
    )
    with torch.autocast(device_type=grid.device.type, enabled=False):
        work_coefficients = coefficients.to(dtype=dtype)
        work_grid = grid.to(dtype=dtype)
        work_centers = centers.to(dtype=dtype)
        chunks: list[torch.Tensor] = []
        for start in range(0, grid.shape[0], basis.chunk_size):
            stop = min(start + basis.chunk_size, grid.shape[0])
            displacements = work_grid[start:stop, None, :] - work_centers[None, :, :]
            basis_values = basis.evaluate(displacements)
            chunks.append(torch.einsum("nrh,gnrh->g", work_coefficients, basis_values))
        return torch.cat(chunks, dim=0)


def integrated_electron_count(
    density: torch.Tensor,
    integration_weights: torch.Tensor,
    *,
    chunk_size: int = 65_536,
) -> torch.Tensor:
    """Integrate electron density with explicit quadrature volumes.

    :param density: Sampled scalar density with shape ``[G]`` in electrons per
        cubic angstrom. Signed truncated-basis reconstructions are accepted.
    :param integration_weights: Strictly positive volume elements with shape
        ``[G]`` in cubic angstroms, on the density dtype and device.
    :param chunk_size: Positive maximum samples reduced in one operation.
    :return: Scalar electron count on the input device. FP16/BF16 inputs are
        accumulated and returned in float32; other dtypes are preserved.
    :rtype: torch.Tensor
    :raises TypeError: If an input is not a tensor.
    :raises ValueError: If inputs are empty, non-finite, non-floating, have
        incompatible shape/dtype/device, or weights are not strictly positive.

    The result is ``sum_g density[g] * integration_weights[g]`` and remains
    connected to autograd. Chunk partials and the running total are evaluated
    with autocast disabled, preventing BF16/FP16 accumulation drift.
    """
    _validate_density_quadrature(
        density, integration_weights, require_nonnegative_density=False
    )
    _validate_chunk_size(chunk_size)
    dtype = _working_dtype(density.dtype)
    with torch.autocast(device_type=density.device.type, enabled=False):
        work_density = density.to(dtype=dtype)
        work_weights = integration_weights.to(dtype=dtype)
        count = torch.zeros((), dtype=dtype, device=density.device)
        for start in range(0, density.shape[0], chunk_size):
            stop = min(start + chunk_size, density.shape[0])
            count = count + torch.sum(
                work_density[start:stop] * work_weights[start:stop]
            )
        return count


def multipole_moments(
    density: torch.Tensor,
    grid: torch.Tensor,
    integration_weights: torch.Tensor,
    origin: torch.Tensor | None = None,
    *,
    chunk_size: int = 65_536,
) -> MultipoleMoments:
    """Reduce a sampled field to electron-number multipole moments.

    :param density: Scalar samples with shape ``[G]`` in electrons per cubic
        angstrom. Signed truncated-basis values are allowed.
    :param grid: Coordinates with shape ``[G, 3]`` in angstroms in one explicit
        coordinate frame, on the density dtype and device.
    :param integration_weights: Positive quadrature volumes with shape ``[G]``
        in cubic angstroms.
    :param origin: Optional moment origin with shape ``[3]`` in angstroms and
        the grid dtype/device. The coordinate-frame origin is used when omitted.
    :param chunk_size: Positive maximum samples reduced in one operation.
    :return: Named tuple containing scalar electron count in electrons, dipole
        shape ``[3]`` in electron-angstroms, and symmetric traceless quadrupole
        shape ``[3, 3]`` in electron-square-angstroms.
    :rtype: MultipoleMoments
    :raises TypeError: If a supplied numerical input is not a tensor.
    :raises ValueError: If shapes, finiteness, floating dtypes, devices,
        positive weights, or ``chunk_size`` are invalid.

    These are electron-number moments, not charge moments: no minus sign for
    the electron charge is applied. With ``x = grid - origin``, the dipole is
    ``integral rho*x`` and the traceless Cartesian quadrupole is
    ``integral rho*(3*x*x.T - |x|**2*I)``. All reductions are differentiable,
    chunked, and accumulated in float32 under BF16/FP16 autocast. This
    low-level API carries no batch/frame metadata; use
    :func:`electron_field_multipole_moments` for binding ``ElectronField``
    mask, batch, and coordinate-frame validation.
    """
    _validate_density_quadrature(
        density, integration_weights, require_nonnegative_density=False
    )
    _validate_positions(grid, "grid", expected_count=density.shape[0])
    _validate_chunk_size(chunk_size)
    if grid.device != density.device:
        raise ValueError(
            "density, grid, and integration_weights must use the same device."
        )
    if grid.dtype != density.dtype:
        raise ValueError(
            "density, grid, and integration_weights must use the same dtype."
        )
    if origin is not None:
        _validate_float_tensor(origin, "origin")
        if origin.shape != (3,):
            raise ValueError("origin must have shape [3].")
        if origin.device != grid.device:
            raise ValueError("origin and grid must use the same device.")
        if origin.dtype != grid.dtype:
            raise ValueError("origin and grid must use the same dtype.")
    dtype = _working_dtype(density.dtype)
    with torch.autocast(device_type=density.device.type, enabled=False):
        work_density = density.to(dtype=dtype)
        work_grid = grid.to(dtype=dtype)
        work_weights = integration_weights.to(dtype=dtype)
        work_origin = (
            torch.zeros(3, dtype=dtype, device=grid.device)
            if origin is None
            else origin.to(dtype=dtype)
        )
        count = torch.zeros((), dtype=dtype, device=density.device)
        dipole = torch.zeros(3, dtype=dtype, device=density.device)
        quadrupole = torch.zeros((3, 3), dtype=dtype, device=density.device)
        identity = torch.eye(3, dtype=dtype, device=density.device)
        for start in range(0, density.shape[0], chunk_size):
            stop = min(start + chunk_size, density.shape[0])
            relative = work_grid[start:stop] - work_origin
            mass = work_density[start:stop] * work_weights[start:stop]
            count = count + mass.sum()
            dipole = dipole + torch.einsum("g,gi->i", mass, relative)
            second = torch.einsum("g,gi,gj->ij", mass, relative, relative)
            radial_second = torch.sum(mass * relative.square().sum(dim=-1))
            quadrupole = quadrupole + 3.0 * second - radial_second * identity
        return MultipoleMoments(count, dipole, quadrupole)


def project_electron_field_to_atoms(
    field: ElectronField,
    centers: torch.Tensor,
    center_batch: torch.Tensor,
    integration_weights: torch.Tensor,
    basis: SphericalFieldBasis,
    *,
    centers_frame: CoordinateFrame,
    density_channel: str | int = "density",
) -> AtomCenteredFieldCoefficients:
    """Project a validated electron field without mixing masks, batches, or frames.

    :param field: Immutable sampled ``ElectronField`` with shape ``[G, C]``, a
        required coordinate frame, valid-point mask, and per-point batch IDs.
    :param centers: Atom centers with shape ``[N, 3]`` in angstroms.
    :param center_batch: Complex ID per center with shape ``[N]`` and
        ``torch.long`` dtype on the center device.
    :param integration_weights: Positive quadrature volumes with shape ``[G]``
        in cubic angstroms on the field dtype/device.
    :param basis: Regular radial and real-spherical-harmonic basis.
    :param centers_frame: Explicit frame provenance for ``centers``; it must
        exactly equal ``field.frame``.
    :param density_channel: Density column index or name. The default resolves
        ``"density"`` or the sole unnamed channel.
    :return: Coefficients ``[N, R, (lmax+1)**2]`` in electrons per angstrom to
        the power ``3/2``, bound immutably to centers, center batches, frame,
        and basis.
    :rtype: AtomCenteredFieldCoefficients
    :raises TypeError: If contracts, frame, basis, or channel selector have
        invalid Python types.
    :raises ValueError: If frame provenance is absent/different; masks leave a
        complex without data; batch sets disagree; or numerical metadata is
        invalid.

    Only ``field.mask`` points participate. Each distinct batch is projected
    independently against centers carrying the same batch ID, so coincident
    coordinates from different complexes cannot interact. This is the
    contract-safe public path; :func:`project_density_to_atoms` remains the
    unbatched tensor kernel. Accumulation, irreps, units, cutoff behavior, and
    mixed-precision guarantees are inherited from that kernel.
    """
    if not isinstance(field, ElectronField):
        raise TypeError("field must be an ElectronField.")
    if not isinstance(centers_frame, CoordinateFrame):
        raise TypeError("centers_frame must be a CoordinateFrame.")
    if not isinstance(basis, SphericalFieldBasis):
        raise TypeError("basis must be a SphericalFieldBasis.")
    field_frame = _require_field_frame(field)
    if field_frame != centers_frame:
        raise ValueError("centers_frame must exactly match ElectronField.frame.")
    _validate_positions(centers, "centers")
    if centers.shape[0] == 0:
        raise ValueError("centers must contain at least one atom center.")
    _validate_batch_vector(center_batch, centers.shape[0], centers)
    _validate_field_weights(field, integration_weights)
    if (
        centers.dtype != field.positions.dtype
        or centers.device != field.positions.device
    ):
        raise ValueError("centers and ElectronField positions must share dtype/device.")
    if (
        centers_frame.origin.dtype != centers.dtype
        or centers_frame.origin.device != centers.device
    ):
        raise ValueError("centers must share the centers_frame dtype and device.")
    channel = _density_channel_index(field, density_channel)
    valid_batches = torch.unique(field.batch[field.mask], sorted=True)
    center_batches = torch.unique(center_batch, sorted=True)
    if not torch.equal(valid_batches, center_batches):
        raise ValueError(
            "valid ElectronField batches must exactly match center batches."
        )
    dtype = _working_dtype(field.positions.dtype)
    values = torch.zeros(
        (centers.shape[0], basis.n_radial, basis.harmonic_dim),
        dtype=dtype,
        device=centers.device,
    )
    for batch_id in valid_batches:
        point_selection = field.mask & (field.batch == batch_id)
        center_selection = center_batch == batch_id
        center_indices = torch.nonzero(center_selection, as_tuple=False).flatten()
        batch_values = project_density_to_atoms(
            field.values[point_selection, channel],
            field.positions[point_selection],
            centers[center_selection],
            integration_weights[point_selection],
            basis,
        )
        values = values.index_copy(0, center_indices, batch_values)
    return AtomCenteredFieldCoefficients(
        values=values,
        centers=centers,
        batch=center_batch,
        frame=centers_frame,
        basis=basis,
    )


def reconstruct_electron_field(
    coefficients: AtomCenteredFieldCoefficients,
    query: ElectronField,
) -> ElectronField:
    """Reconstruct a masked batched field with binding frame provenance.

    :param coefficients: Atom-centered coefficients bound to their centers,
        center batches, frame, and basis.
    :param query: ``ElectronField`` supplying query positions, mask, batches,
        and required frame. Its existing value channels are ignored.
    :return: New ``ElectronField`` with one ``"density"`` channel, the query
        metadata, zero at masked positions, and independently reconstructed
        valid batches.
    :rtype: ElectronField
    :raises TypeError: If either argument has the wrong contract type.
    :raises ValueError: If query frame, dtype/device, or valid batch membership
        is incompatible with the coefficient provenance.

    Query positions and atom centers from different complexes never enter the
    same raw reconstruction call. Low-precision arithmetic still accumulates
    in float32, then casts to the query dtype because ``ElectronField`` binds
    values and positions to one dtype/device. The operation remains connected
    to coefficient, center, and query-position autograd graphs.
    """
    if not isinstance(coefficients, AtomCenteredFieldCoefficients):
        raise TypeError("coefficients must be AtomCenteredFieldCoefficients.")
    if not isinstance(query, ElectronField):
        raise TypeError("query must be an ElectronField.")
    query_frame = _require_field_frame(query)
    if query_frame != coefficients.frame:
        raise ValueError("query frame must exactly match coefficient frame.")
    if (
        query.positions.dtype != coefficients.centers.dtype
        or query.positions.device != coefficients.centers.device
    ):
        raise ValueError("query and coefficient centers must share dtype/device.")
    valid_batches = torch.unique(query.batch[query.mask], sorted=True)
    center_batches = torch.unique(coefficients.batch, sorted=True)
    if valid_batches.numel() and not bool(
        torch.isin(valid_batches, center_batches).all()
    ):
        raise ValueError("every valid query batch must have coefficient centers.")
    density = torch.zeros(
        query.positions.shape[0],
        dtype=query.positions.dtype,
        device=query.positions.device,
    )
    for batch_id in valid_batches:
        query_selection = query.mask & (query.batch == batch_id)
        center_selection = coefficients.batch == batch_id
        query_indices = torch.nonzero(query_selection, as_tuple=False).flatten()
        batch_density = reconstruct_density(
            coefficients.values[center_selection],
            query.positions[query_selection],
            coefficients.centers[center_selection],
            coefficients.basis,
        ).to(dtype=query.positions.dtype)
        density = density.index_copy(0, query_indices, batch_density)
    return ElectronField(
        positions=query.positions,
        values=density.unsqueeze(-1),
        mask=query.mask,
        batch=query.batch,
        channel_names=("density",),
        frame=query_frame,
    )


def electron_field_multipole_moments(
    field: ElectronField,
    integration_weights: torch.Tensor,
    *,
    density_channel: str | int = "density",
    chunk_size: int = 65_536,
) -> BatchedMultipoleMoments:
    """Reduce valid field points to independent per-complex multipoles.

    :param field: Immutable electron field with required frame, mask, batch,
        positions ``[G, 3]``, and density values ``[G, C]``.
    :param integration_weights: Positive quadrature volumes ``[G]`` in cubic
        angstroms on the field dtype/device.
    :param density_channel: Density column index or semantic channel name.
    :param chunk_size: Positive raw-reduction chunk size.
    :return: Sorted batch IDs ``[B]``, count ``[B]``, dipole ``[B, 3]``,
        quadrupole ``[B, 3, 3]``, and the exact source coordinate frame.
    :rtype: BatchedMultipoleMoments
    :raises TypeError: If field or selectors have invalid types.
    :raises ValueError: If frame provenance, weights, channel selection, or
        chunk size is invalid.

    Masked points are excluded before every reduction and batches are reduced
    independently. Moments use coordinates in ``field.frame`` and retain that
    provenance in the result. Units and sign convention match
    :func:`multipole_moments`; FP16/BF16 partials accumulate in float32.
    """
    if not isinstance(field, ElectronField):
        raise TypeError("field must be an ElectronField.")
    frame = _require_field_frame(field)
    _validate_field_weights(field, integration_weights)
    _validate_chunk_size(chunk_size)
    channel = _density_channel_index(field, density_channel)
    batch_ids = torch.unique(field.batch[field.mask], sorted=True)
    if batch_ids.numel() == 0:
        dtype = _working_dtype(field.positions.dtype)
        return BatchedMultipoleMoments(
            batch_ids,
            torch.empty(0, dtype=dtype, device=field.positions.device),
            torch.empty((0, 3), dtype=dtype, device=field.positions.device),
            torch.empty((0, 3, 3), dtype=dtype, device=field.positions.device),
            frame,
        )
    moments: list[MultipoleMoments] = []
    for batch_id in batch_ids:
        selection = field.mask & (field.batch == batch_id)
        moments.append(
            multipole_moments(
                field.values[selection, channel],
                field.positions[selection],
                integration_weights[selection],
                chunk_size=chunk_size,
            )
        )
    return BatchedMultipoleMoments(
        batch_ids,
        torch.stack([moment.electron_count for moment in moments]),
        torch.stack([moment.dipole for moment in moments]),
        torch.stack([moment.quadrupole for moment in moments]),
        frame,
    )
