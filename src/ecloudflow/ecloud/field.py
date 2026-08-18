"""Differentiable projection and moment reductions for electron densities."""

from __future__ import annotations

from typing import NamedTuple

import torch

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
    """Return a rotation-invariant smooth partition over covering centers."""
    support = distances <= cutoff
    scores = torch.exp(-2.0 * (distances / cutoff).square()) * support
    normalization = scores.sum(dim=1, keepdim=True)
    return torch.where(
        normalization > 0,
        scores / normalization.clamp_min(torch.finfo(scores.dtype).tiny),
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
        preserved.
    :rtype: torch.Tensor
    :raises TypeError: If an input is not a tensor or ``basis`` has the wrong
        type.
    :raises ValueError: If density is negative or non-finite; shapes, dtypes,
        or devices disagree; weights are not positive; or centers are empty.

    For overlapping cutoff balls, a smooth distance-only partition of unity
    assigns each sample among covering atoms before quadrature. This prevents
    electron double counting while remaining invariant to joint rotations and
    translations. Samples outside every cutoff ball have zero representation.
    Coefficients in ``basis.l_slice(l)`` transform under the matching real
    e3nn irrep. Grid chunks cap intermediate storage at approximately
    ``[chunk_size, N, R, H]``. Autocast is disabled internally and FP16/BF16
    data accumulate in float32 to limit electron-count drift. This tensor-only
    routine is unbatched; callers must select valid points from an
    ``ElectronField`` and keep grid and centers in its ``CoordinateFrame``.
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
        for start in range(0, grid.shape[0], basis.chunk_size):
            stop = min(start + basis.chunk_size, grid.shape[0])
            displacements = work_grid[start:stop, None, :] - work_centers[None, :, :]
            distances = torch.linalg.vector_norm(displacements, dim=-1)
            partition = _atom_partition(distances, basis.cutoff)
            radial = basis.radial_values(distances)
            harmonics = basis.spherical_harmonics(displacements)
            volume_density = work_density[start:stop] * work_weights[start:stop]
            contribution = torch.einsum(
                "g,gn,gnr,gnh->nrh",
                volume_density,
                partition,
                radial,
                harmonics,
            )
            coefficients = coefficients + contribution
        return coefficients


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

    Atom contributions are summed because projection used a partition of
    unity. The normalized constant radial function and integral-normalized
    ``Y00`` make the continuous integral of each reconstructed monopole equal
    to its assigned electron count; finite-grid agreement is therefore a real
    quadrature invariant rather than a fitted fixture constant. Signed values
    may occur in a truncated basis. Evaluation is differentiable, chunked over
    queries, and performed outside autocast with at least float32 arithmetic.
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
            distances = torch.linalg.vector_norm(displacements, dim=-1)
            radial = basis.radial_values(distances)
            harmonics = basis.spherical_harmonics(displacements)
            chunks.append(
                torch.einsum("nrh,gnr,gnh->g", work_coefficients, radial, harmonics)
            )
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
    chunked, and accumulated in float32 under BF16/FP16 autocast. This API is
    unbatched so moments from different coordinate frames cannot be mixed.
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
