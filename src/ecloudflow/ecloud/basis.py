"""Compact radial functions and e3nn real spherical harmonics."""

from __future__ import annotations

import math
from dataclasses import dataclass

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

    :param n_radial: Positive number ``R`` of compact radial channels. Channel
        zero is the normalized constant on the cutoff ball; later channels
        are orthogonalized spherical-Bessel ``sin(k pi r / cutoff) / r``
        functions.
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

    Radial functions are orthonormal under ``integral r**2 dr`` on
    ``[0, cutoff]``. Harmonics use ``e3nn.o3.spherical_harmonics`` with
    ``normalization="integral"``. Consequently every ``l`` slice transforms
    as one real e3nn irrep ``l`` with parity ``(-1)**l``. The constant radial
    channel makes the continuous integral of the reconstructed ``l=0`` block
    equal to its projected electron count. FP16 and BF16 inputs are evaluated
    in float32; float32 and float64 inputs retain their dtype.
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

    def radial_values(self, radii: torch.Tensor) -> torch.Tensor:
        """Evaluate normalized compact radial functions.

        :param radii: Non-negative radii with arbitrary shape in angstroms.
        :return: Radial values with shape ``[*radii.shape, R]`` on the same
            device. FP16/BF16 inputs produce float32 output for stability;
            other floating dtypes are preserved.
        :rtype: torch.Tensor
        :raises TypeError: If ``radii`` is not a tensor.
        :raises ValueError: If radii are non-floating, non-finite, or negative.

        The raw Bessel candidates are analytically orthonormal among
        themselves. A Cholesky inverse of their analytic Gram matrix with the
        constant monopole performs deterministic Gram-Schmidt
        orthogonalization without sampled normalization error.
        """
        _validate_float_tensor(radii, "radii")
        if bool((radii < 0).any()):
            raise ValueError("radii must be non-negative.")
        dtype = _working_dtype(radii.dtype)
        with torch.autocast(device_type=radii.device.type, enabled=False):
            work_radii = radii.to(dtype=dtype)
            constant = math.sqrt(3.0 / self.cutoff**3)
            constant_values = torch.full_like(work_radii, constant).unsqueeze(-1)
            if self.n_radial == 1:
                values = constant_values
            else:
                orders = torch.arange(
                    1,
                    self.n_radial,
                    dtype=dtype,
                    device=radii.device,
                )
                frequencies = orders * (math.pi / self.cutoff)
                safe_radii = torch.where(
                    work_radii == 0,
                    torch.ones_like(work_radii),
                    work_radii,
                )
                arguments = work_radii.unsqueeze(-1) * frequencies
                bessel = math.sqrt(2.0 / self.cutoff) * (
                    torch.sin(arguments) / safe_radii.unsqueeze(-1)
                )
                limits = math.sqrt(2.0 / self.cutoff) * frequencies
                bessel = torch.where((work_radii == 0).unsqueeze(-1), limits, bessel)
                raw_values = torch.cat((constant_values, bessel), dim=-1)
                transform = self._radial_orthonormalizer(dtype, radii.device)
                values = raw_values @ transform
            support = (work_radii <= self.cutoff).unsqueeze(-1)
            return torch.where(support, values, torch.zeros_like(values))

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
        harmonics = self.spherical_harmonics(vectors)
        radii = torch.linalg.vector_norm(vectors.to(harmonics.dtype), dim=-1)
        radial = self.radial_values(radii)
        return radial.unsqueeze(-1) * harmonics.unsqueeze(-2)

    def _radial_orthonormalizer(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Build the analytic Cholesky inverse for the radial Gram matrix."""
        gram = torch.eye(self.n_radial, dtype=dtype, device=device)
        if self.n_radial > 1:
            orders = torch.arange(1, self.n_radial, dtype=dtype, device=device)
            signs = torch.where(
                orders.remainder(2) == 1,
                torch.ones_like(orders),
                -torch.ones_like(orders),
            )
            overlaps = math.sqrt(6.0) * signs / (math.pi * orders)
            gram[0, 1:] = overlaps
            gram[1:, 0] = overlaps
        cholesky = torch.linalg.cholesky(gram)
        identity = torch.eye(self.n_radial, dtype=dtype, device=device)
        return torch.linalg.solve_triangular(cholesky.T, identity, upper=True)
