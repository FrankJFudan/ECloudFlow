"""Compact radial functions and e3nn real spherical harmonics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from e3nn import o3


def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    """Choose a stable arithmetic dtype for basis evaluation."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _validate_float_tensor(tensor: torch.Tensor, name: str) -> None:
    """Reject non-tensor, non-floating, or non-finite numerical input."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values.")


@dataclass(frozen=True)
class SphericalFieldBasis:
    """Define an atom-centered radial and real-spherical-harmonic basis.

    :param n_radial: Positive number ``R`` of compact radial channels derived
        from regularized spherical-Bessel ``sinc(n r / cutoff)`` functions.
    :param lmax: Largest non-negative angular order. Harmonics are packed in
        e3nn order, with order ``l`` in ``[l**2:(l+1)**2]`` and total size
        ``H = (lmax + 1) ** 2``.
    :param cutoff: Positive finite support radius in angstroms.
    :param chunk_size: Positive maximum number of query-grid points processed
        by field projection or reconstruction at once.
    :return: Immutable basis specification evaluated on the input device.
    :rtype: SphericalFieldBasis
    :raises ValueError: If a count is invalid, ``cutoff`` is non-finite or
        non-positive, or ``chunk_size`` is not positive.

    For angular order ``l``, raw radial functions are proportional to
    ``(r/cutoff)**l * (1-(r/cutoff)**2)**2 * sinc(n*r/cutoff)``. They are
    orthonormalized under ``integral r**2 dr`` on ``[0, cutoff]``. The
    ``r**l`` factor makes their product with ``Y_lm`` a regular solid harmonic
    at atom centers, while the envelope gives zero value and first derivative
    at the cutoff. Harmonics use ``e3nn.o3.spherical_harmonics`` with integral
    normalization, so every ``l`` slice transforms as one real e3nn irrep
    with parity ``(-1)**l``. FP16 and BF16 inputs are evaluated in float32;
    float32 and float64 inputs retain their dtype. Radial normalization gives
    every channel units of inverse angstroms to the power ``3/2``.
    """

    n_radial: int
    lmax: int
    cutoff: float
    chunk_size: int = 8192

    def __post_init__(self) -> None:
        """Validate and canonicalize the immutable numerical configuration."""
        if (
            not isinstance(self.n_radial, int)
            or isinstance(self.n_radial, bool)
            or self.n_radial <= 0
        ):
            raise ValueError("n_radial must be a positive integer.")
        if (
            not isinstance(self.lmax, int)
            or isinstance(self.lmax, bool)
            or self.lmax < 0
        ):
            raise ValueError("lmax must be a non-negative integer.")
        if (
            isinstance(self.cutoff, bool)
            or not isinstance(self.cutoff, (int, float))
            or not math.isfinite(float(self.cutoff))
            or float(self.cutoff) <= 0.0
        ):
            raise ValueError("cutoff must be a positive finite number.")
        if (
            not isinstance(self.chunk_size, int)
            or isinstance(self.chunk_size, bool)
            or self.chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer.")
        object.__setattr__(self, "cutoff", float(self.cutoff))

    @property
    def harmonic_dim(self) -> int:
        """Return the packed harmonic dimension ``(lmax + 1) ** 2``."""
        return (self.lmax + 1) ** 2

    @property
    def irreps(self) -> o3.Irreps:
        """Return one e3nn real irrep for every represented angular order."""
        return o3.Irreps(
            [(1, (order, (-1) ** order)) for order in range(self.lmax + 1)]
        )

    def l_slice(self, order: int) -> slice:
        """Return the contiguous coefficient slice for one angular order.

        :param order: Angular order between zero and ``lmax`` inclusive.
        :return: Slice ``[order**2:(order+1)**2]`` in e3nn harmonic order.
        :rtype: slice
        :raises ValueError: If ``order`` is not a represented integer order.
        """
        if (
            not isinstance(order, int)
            or isinstance(order, bool)
            or order < 0
            or order > self.lmax
        ):
            raise ValueError("angular order must be an integer in [0, lmax].")
        return slice(order**2, (order + 1) ** 2)

    def radial_values(self, radii: torch.Tensor, order: int = 0) -> torch.Tensor:
        """Evaluate normalized compact radial functions for one angular order.

        :param radii: Non-negative radii with arbitrary shape in angstroms.
        :param order: Angular order between zero and ``lmax``. Every returned
            channel behaves as ``r**order`` at the origin.
        :return: Radial values with shape ``[*radii.shape, R]`` on the same
            device. FP16/BF16 inputs produce float32 output for stability;
            other floating dtypes are preserved.
        :rtype: torch.Tensor
        :raises TypeError: If ``radii`` is not a tensor.
        :raises ValueError: If radii are non-floating, non-finite, negative, or
            ``order`` is outside the represented range.

        A cached float64 midpoint quadrature constructs the Cholesky
        orthonormalizer for each ``l``. Evaluation itself stays differentiable
        with respect to radii. The polynomial cutoff envelope and Bessel
        candidates have zero radial derivative at the origin; the envelope
        and its first derivative vanish at ``cutoff``.
        """
        _validate_float_tensor(radii, "radii")
        self.l_slice(order)
        if bool((radii < 0).any()):
            raise ValueError("radii must be non-negative.")
        dtype = _working_dtype(radii.dtype)
        with torch.autocast(device_type=radii.device.type, enabled=False):
            work_radii = radii.to(dtype=dtype)
            scaled = work_radii / self.cutoff
            profiles = self._regular_profiles(scaled, order)
            values = profiles * scaled.unsqueeze(-1).pow(order)
            support = (work_radii < self.cutoff).unsqueeze(-1)
            return torch.where(support, values, torch.zeros_like(values))

    def monopole_integrals(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Return continuous-volume integrals of the ``l=0`` basis functions.

        :param dtype: Floating output dtype used by field accumulation.
        :param device: Output device used by the coefficient tensor.
        :return: Tensor with shape ``[R]`` containing
            ``integral R_n(r) Y_00 dV`` in angstroms to the power ``3/2``.
        :rtype: torch.Tensor
        :raises ValueError: If ``dtype`` is not floating.

        These analytically scaled, high-order numerical basis constants enable
        a conservation-constrained Galerkin correction. They depend only on
        the basis, never on a fixture or requested electron count.
        """
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point.")
        _, dimensionless_integrals = _cached_radial_data(self.n_radial, 0)
        scale = self.cutoff**1.5 * math.sqrt(4.0 * math.pi)
        return dimensionless_integrals.to(dtype=dtype, device=device) * scale

    def spherical_harmonics(self, vectors: torch.Tensor) -> torch.Tensor:
        """Evaluate e3nn real spherical harmonics through ``lmax``.

        :param vectors: Relative Cartesian vectors with shape ``[..., 3]`` in
            angstroms. Only their directions enter non-scalar harmonics.
        :return: Integral-normalized real harmonics with shape ``[..., H]`` in
            contiguous e3nn ``l`` slices on the input device. FP16/BF16 input
            is promoted to float32.
        :rtype: torch.Tensor
        :raises TypeError: If ``vectors`` is not a tensor.
        :raises ValueError: If the trailing shape, dtype, or values are invalid.

        Direction is undefined at zero displacement. There the invariant
        ``Y00`` retains its analytic value while every ``l>0`` component is
        defined as zero. This convention is rotation equivariant and avoids
        non-finite gradients from normalizing a zero vector.
        """
        _validate_float_tensor(vectors, "vectors")
        if vectors.ndim == 0 or vectors.shape[-1] != 3:
            raise ValueError("vectors must have shape [..., 3].")
        dtype = _working_dtype(vectors.dtype)
        with torch.autocast(device_type=vectors.device.type, enabled=False):
            work_vectors = vectors.to(dtype=dtype)
            norms = torch.linalg.vector_norm(work_vectors, dim=-1)
            at_origin = norms == 0
            fallback = torch.zeros_like(work_vectors)
            fallback[..., 0] = 1.0
            safe_vectors = torch.where(at_origin.unsqueeze(-1), fallback, work_vectors)
            harmonics = o3.spherical_harmonics(
                list(range(self.lmax + 1)),
                safe_vectors,
                normalize=True,
                normalization="integral",
            )
            if self.harmonic_dim == 1:
                return harmonics
            nonscalar = torch.where(
                at_origin.unsqueeze(-1),
                torch.zeros_like(harmonics[..., 1:]),
                harmonics[..., 1:],
            )
            return torch.cat((harmonics[..., :1], nonscalar), dim=-1)

    def evaluate(self, vectors: torch.Tensor) -> torch.Tensor:
        """Evaluate tensor-product basis functions on relative vectors.

        :param vectors: Relative coordinates with shape ``[..., 3]`` in
            angstroms, already expressed in one common coordinate frame.
        :return: Products ``R_n(r) Y_lm(r_hat)`` with shape ``[..., R, H]`` on
            the vector device and in the stable working dtype.
        :rtype: torch.Tensor
        :raises TypeError: If ``vectors`` is not a tensor.
        :raises ValueError: If vectors have invalid shape, dtype, or values.
        """
        _validate_float_tensor(vectors, "vectors")
        if vectors.ndim == 0 or vectors.shape[-1] != 3:
            raise ValueError("vectors must have shape [..., 3].")
        dtype = _working_dtype(vectors.dtype)
        with torch.autocast(device_type=vectors.device.type, enabled=False):
            work_vectors = vectors.to(dtype=dtype)
            scaled_vectors = work_vectors / self.cutoff
            scaled_radii = torch.linalg.vector_norm(scaled_vectors, dim=-1)
            solid_harmonics = o3.spherical_harmonics(
                list(range(self.lmax + 1)),
                scaled_vectors,
                normalize=False,
                normalization="integral",
            )
            blocks = []
            for order in range(self.lmax + 1):
                profiles = self._regular_profiles(scaled_radii, order)
                harmonic_block = solid_harmonics[..., self.l_slice(order)]
                blocks.append(profiles.unsqueeze(-1) * harmonic_block.unsqueeze(-2))
            return torch.cat(blocks, dim=-1)

    def _regular_profiles(self, scaled_radii: torch.Tensor, order: int) -> torch.Tensor:
        """Evaluate radial profiles multiplying regular solid harmonics."""
        raw = _dimensionless_profiles(scaled_radii, self.n_radial)
        transform, _ = _cached_radial_data(self.n_radial, order)
        work_transform = transform.to(
            dtype=scaled_radii.dtype, device=scaled_radii.device
        )
        profiles = raw @ work_transform / self.cutoff**1.5
        support = (scaled_radii < 1.0).unsqueeze(-1)
        return torch.where(support, profiles, torch.zeros_like(profiles))


def _dimensionless_profiles(scaled_radii: torch.Tensor, n_radial: int) -> torch.Tensor:
    """Evaluate smooth cutoff Bessel profiles before orthonormalization."""
    channels = torch.arange(
        1,
        n_radial + 1,
        dtype=scaled_radii.dtype,
        device=scaled_radii.device,
    )
    envelope = (1.0 - scaled_radii.square()).square().unsqueeze(-1)
    return envelope * torch.sinc(scaled_radii.unsqueeze(-1) * channels)


@lru_cache(maxsize=128)
def _cached_radial_data(n_radial: int, order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build float64 radial orthonormalizers and monopole integrals."""
    quadrature_size = 32_768
    scaled_radii = (
        torch.arange(quadrature_size, dtype=torch.float64) + 0.5
    ) / quadrature_size
    profiles = _dimensionless_profiles(scaled_radii, n_radial)
    raw_radial = profiles * scaled_radii.unsqueeze(-1).pow(order)
    weights = scaled_radii.square() / quadrature_size
    gram = raw_radial.T @ (raw_radial * weights.unsqueeze(-1))
    cholesky = torch.linalg.cholesky(gram)
    identity = torch.eye(n_radial, dtype=torch.float64)
    transform = torch.linalg.solve_triangular(cholesky.T, identity, upper=True)
    normalized = raw_radial @ transform
    integrals = torch.sum(normalized * weights.unsqueeze(-1), dim=0)
    return transform, integrals
