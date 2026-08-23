"""Equivariant latent-to-field decoding and physical moment reductions."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from e3nn import o3
from torch import nn

from ecloudflow.ecloud.basis import SphericalFieldBasis


class ElectronReconstruction(NamedTuple):
    """Hold decoded scalar fields, derivatives, moments, and cycle tokens.

    :param density: Scalar electron-density samples ``[B, G]`` in electrons
        per cubic angstrom in the local coordinate frame.
    :param gradient: Covariant Cartesian density gradients ``[B, G, 3]`` in
        electrons per fourth-power angstrom; vectors rotate with the frame.
    :param electron_count: Invariant analytic monopoles ``[B]`` in electrons.
    :param dipole: Electron-number dipoles ``[B, 3]`` in electron-angstroms,
        measured about the local frame origin.
    :param latent_round_trip: Re-encoded tokens ``[B, N, C]`` in the decoder's
        documented packed e3nn irrep layout, with padding exactly zero.
    :return: Immutable tuple whose tensors remain on the input device.
    :rtype: ElectronReconstruction

    Low-precision decoding accumulates density and moments in float32. The
    object does not detach or copy results, so density, gradient, moments, and
    cycle terms retain gradients to latent tokens and decoder parameters.
    Joint proper rotation leaves density/count invariant and rotates gradient,
    dipole, and non-scalar token blocks. Under a translation ``t``, the dipole
    changes by ``electron_count * t``. No inputs are mutated; initialized
    parameters make evaluation deterministic for fixed tensors and module
    state. Numerical validation failures are raised by the decoder before work.
    """

    density: torch.Tensor
    gradient: torch.Tensor
    electron_count: torch.Tensor
    dipole: torch.Tensor
    latent_round_trip: torch.Tensor


class ElectronFieldDecoder(nn.Module):  # type: ignore[misc]
    """Decode packed scalar/vector atom tokens into a continuous density.

    :param n_radial: Positive number ``R`` of Task 4 radial basis functions.
    :param lmax: Non-negative maximum decoded harmonic order.
    :param vector_dim: Number of odd-parity vector copies in the latent layout.
    :param latent_dim: Total packed component count ``C``. After the invariant
        scalar and ``vector_dim`` vector blocks, one copy of every configured
        order ``l>=2`` is retained so orientation-bearing inputs are preserved.
    :param cutoff: Positive basis cutoff in angstroms.
    :param chunk_size: Positive maximum number of query points decoded at once.
    :return: Trainable equivariant decoder using e3nn linear maps.
    :rtype: ElectronFieldDecoder
    :raises ValueError: If dimensions cannot form the documented irreps or a
        basis setting is invalid.

    Coefficients are internally packed by angular order as
    ``R x 0e + R x 1o + ... + R x lmax`` and unpacked to boundary shape
    ``[B, N, R, (lmax+1)**2]``. This explicit conversion keeps padded tensors
    out of canonical flattened PyG model contracts. Query/center coordinates
    use one centered Cartesian frame in angstroms. Density is translation
    invariant; its gradient and the local ``l=1`` moment rotate covariantly.
    Parameter dtype/device define arithmetic placement, with FP16/BF16 inputs
    promoted to float32 accumulation. Forward calls do not mutate arguments;
    fixed parameters and inputs give deterministic results. Every operation,
    including returned gradients, remains differentiable unless the caller
    invokes it under ``torch.no_grad``. Boolean ``[B,N]`` masks suppress every
    padded token and center exactly before field or moment accumulation.
    """

    def __init__(
        self,
        n_radial: int,
        lmax: int,
        vector_dim: int,
        latent_dim: int,
        *,
        cutoff: float = 5.0,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        _validate_dimensions(n_radial, lmax, vector_dim, latent_dim)
        self.n_radial = n_radial
        self.lmax = lmax
        self.vector_dim = vector_dim
        self.latent_dim = latent_dim
        self.basis = SphericalFieldBasis(n_radial, lmax, cutoff, chunk_size)
        higher_entries = [(1, (order, (-1) ** order)) for order in range(2, lmax + 1)]
        higher_width = sum(2 * order + 1 for order in range(2, lmax + 1))
        scalar_copies = latent_dim - 3 * vector_dim - higher_width
        self.latent_irreps = o3.Irreps(
            [(scalar_copies, (0, 1)), (vector_dim, (1, -1)), *higher_entries]
        ).simplify()
        self.coefficient_irreps = o3.Irreps(
            [(n_radial, (order, (-1) ** order)) for order in range(lmax + 1)]
        )
        self.to_coefficients = o3.Linear(
            self.latent_irreps, self.coefficient_irreps, biases=True
        )
        self.to_round_trip = o3.Linear(
            self.coefficient_irreps, self.latent_irreps, biases=True
        )
        monopole = self.basis.monopole_integrals(torch.float64, torch.device("cpu"))
        self.register_buffer("monopole_integrals", monopole.to(torch.float32))
        self.register_buffer(
            "dipole_integrals", _dipole_integrals(self.basis).to(torch.float32)
        )

    def forward(
        self,
        latent: torch.Tensor,
        centers: torch.Tensor,
        query_grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> ElectronReconstruction:
        """Decode atom tokens at query coordinates and reduce their moments.

        :param latent: Padded tokens ``[B, N, C]`` on the module device in
            packed ``(C-3V)x0e + Vx1o`` layout. Scalars are unitless learned
            components and vector triples are expressed in the local frame.
        :param centers: Atom centers ``[B, N, 3]`` in angstroms in the same
            centered frame, floating dtype/device as ``query_grid``.
        :param query_grid: Query coordinates ``[B, G, 3]`` in angstroms.
        :param mask: Boolean physical-atom mask ``[B, N]`` on the same device;
            false atoms contribute exactly zero to all returned tensors.
        :return: Density ``[B,G]``, gradient ``[B,G,3]``, invariant electron
            count ``[B]``, frame-relative dipole ``[B,3]``, and packed cycle
            tokens ``[B,N,C]`` in an :class:`ElectronReconstruction`.
        :rtype: ElectronReconstruction
        :raises TypeError: If a numerical input is not a tensor.
        :raises ValueError: If shapes, masks, floating dtypes, finite values,
            devices, module placement, or packed irrep width disagree.

        Decoded coefficients have units of electrons per angstrom to power
        ``3/2`` under the Task 4 basis. Query chunks bound intermediates by
        ``[B, chunk_size, N, R, H]``. Autocast is disabled and density/moment
        sums use float32 when inputs or ambient mixed precision are lower.
        The density gradient is computed by differentiating each query chunk;
        it supports higher-order parameter gradients in ordinary grad mode.
        Inputs and masks are never mutated, no random operation or heuristic
        fallback is used, and padded latent/center values cannot influence the
        result. Translation leaves density/count/gradient unchanged and shifts
        dipole by count times translation; proper rotations act by the stated
        scalar, vector, and packed-irrep rules. Every returned tensor shape is
        explicit above, and evaluation is deterministic for fixed module state.
        """
        _validate_decode_inputs(self, latent, centers, query_grid, mask)
        parameter = next(self.parameters())
        dtype = parameter.dtype
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        with torch.autocast(device_type=latent.device.type, enabled=False):
            work_latent = latent.to(dtype=dtype)
            work_centers = centers.to(dtype=dtype)
            physical = mask.unsqueeze(-1).to(dtype=dtype)
            masked_latent = work_latent * physical
            packed_coefficients = self.to_coefficients(masked_latent)
            coefficients = _unpack_coefficients(
                packed_coefficients, self.n_radial, self.lmax
            )
            coefficients = coefficients * physical.unsqueeze(-1)
            round_trip = self.to_round_trip(packed_coefficients) * physical
            density, gradient = self._density_and_gradient(
                coefficients, work_centers, query_grid.to(dtype=dtype)
            )
            count, dipole = self._moments(coefficients, work_centers, mask)
        return ElectronReconstruction(
            density=density,
            gradient=gradient,
            electron_count=count,
            dipole=dipole,
            latent_round_trip=round_trip,
        )

    def decode(
        self,
        latent: torch.Tensor,
        centers: torch.Tensor,
        query_grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> ElectronReconstruction:
        """Decode tokens through :meth:`forward` without altering inputs.

        :param latent: Packed latent tensor ``[B, N, C]`` on the module device.
        :param centers: Atom centers ``[B, N, 3]`` in local-frame angstroms.
        :param query_grid: Local-frame query coordinates ``[B, G, 3]``.
        :param mask: Boolean physical-atom mask ``[B, N]``.
        :return: Differentiable density, vector gradient, invariant count,
            covariant dipole, and packed latent cycle terms.
        :rtype: ElectronReconstruction
        :raises TypeError: If any numerical argument is not a tensor.
        :raises ValueError: If shape, dtype, device, frame-layout, finiteness,
            or mask validation fails.

        Shapes, e3nn layout, units, frame conventions, float32 mixed-precision
        accumulation, chunking, deterministic behavior, mutation guarantees,
        padding semantics, irrep transformation, and gradients are identical
        to :meth:`forward`.
        """
        return self.forward(latent, centers, query_grid, mask)

    def _density_and_gradient(
        self,
        coefficients: torch.Tensor,
        centers: torch.Tensor,
        query_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode differentiable scalar density and query gradients by chunks."""
        density_chunks: list[torch.Tensor] = []
        gradient_chunks: list[torch.Tensor] = []
        create_graph = torch.is_grad_enabled()
        for start in range(0, query_grid.shape[1], self.basis.chunk_size):
            stop = min(start + self.basis.chunk_size, query_grid.shape[1])
            raw_queries = query_grid[:, start:stop]
            queries = (
                raw_queries
                if raw_queries.requires_grad
                else raw_queries.clone().requires_grad_(True)
            )
            with torch.enable_grad():
                displacement = queries[:, :, None, :] - centers[:, None, :, :]
                basis_values = self.basis.evaluate(displacement)
                density = torch.einsum("bnrh,bgnrh->bg", coefficients, basis_values)
                gradient = torch.autograd.grad(
                    density.sum(),
                    queries,
                    create_graph=create_graph,
                    retain_graph=True,
                )[0]
            density_chunks.append(density)
            gradient_chunks.append(gradient)
        return torch.cat(density_chunks, dim=1), torch.cat(gradient_chunks, dim=1)

    def _moments(
        self,
        coefficients: torch.Tensor,
        centers: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute analytic invariant monopoles and covariant dipoles."""
        monopole_integrals = self.monopole_integrals.to(coefficients)
        per_atom_count = coefficients[..., 0] @ monopole_integrals
        local_dipole = torch.zeros_like(centers)
        if self.lmax >= 1:
            local_dipole = torch.einsum(
                "bnrc,r->bnc",
                coefficients[..., self.basis.l_slice(1)],
                self.dipole_integrals.to(coefficients),
            )
        physical = mask.to(dtype=coefficients.dtype)
        count = torch.sum(per_atom_count * physical, dim=1)
        dipole = torch.sum(
            (local_dipole + per_atom_count.unsqueeze(-1) * centers)
            * physical.unsqueeze(-1),
            dim=1,
        )
        return count, dipole


def _dipole_integrals(basis: SphericalFieldBasis) -> torch.Tensor:
    """Numerically derive local l=1 Cartesian moment constants from the basis."""
    if basis.lmax < 1:
        return torch.zeros(basis.n_radial, dtype=torch.float64)
    sample_count = 16_384
    radii = (
        (torch.arange(sample_count, dtype=torch.float64) + 0.5)
        * basis.cutoff
        / sample_count
    )
    radial = basis.radial_values(radii, order=1)
    radial_integral = torch.sum(radial * radii.pow(3).unsqueeze(-1), dim=0) * (
        basis.cutoff / sample_count
    )
    return radial_integral * math.sqrt(4.0 * math.pi / 3.0)


def _unpack_coefficients(
    packed: torch.Tensor, n_radial: int, lmax: int
) -> torch.Tensor:
    """Convert e3nn order-grouped blocks to boundary ``[B,N,R,H]`` layout."""
    blocks: list[torch.Tensor] = []
    offset = 0
    for order in range(lmax + 1):
        width = n_radial * (2 * order + 1)
        blocks.append(
            packed[..., offset : offset + width].reshape(
                *packed.shape[:-1], n_radial, 2 * order + 1
            )
        )
        offset += width
    return torch.cat(blocks, dim=-1)


def _validate_dimensions(
    n_radial: int, lmax: int, vector_dim: int, latent_dim: int
) -> None:
    """Validate integer dimensions and a nonempty scalar latent block."""
    for name, value, minimum in (
        ("n_radial", n_radial, 1),
        ("lmax", lmax, 0),
        ("vector_dim", vector_dim, 0),
        ("latent_dim", latent_dim, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if lmax > 0 and vector_dim == 0:
        raise ValueError("vector_dim must be positive when lmax is nonzero.")
    higher_width = sum(2 * order + 1 for order in range(2, lmax + 1))
    if latent_dim - 3 * vector_dim - higher_width <= 0:
        raise ValueError(
            "latent_dim must retain every configured higher irrep and leave at "
            "least one invariant scalar component."
        )


def _validate_decode_inputs(
    module: ElectronFieldDecoder,
    latent: torch.Tensor,
    centers: torch.Tensor,
    query_grid: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Validate padded decoder-boundary tensors and module placement."""
    for name, tensor in (
        ("latent", latent),
        ("centers", centers),
        ("query_grid", query_grid),
        ("mask", mask),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
    if latent.ndim != 3 or latent.shape[-1] != module.latent_dim:
        raise ValueError(
            f"latent must have shape [B, N, {module.latent_dim}] in packed irrep layout."
        )
    if centers.shape != (*latent.shape[:2], 3):
        raise ValueError("centers must have shape [B, N, 3] matching latent.")
    if (
        query_grid.ndim != 3
        or query_grid.shape[0] != latent.shape[0]
        or query_grid.shape[2] != 3
    ):
        raise ValueError("query_grid must have shape [B, G, 3] matching latent batch.")
    if query_grid.shape[1] == 0:
        raise ValueError("query_grid must contain at least one query point.")
    if mask.shape != latent.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor with shape [B, N].")
    for name, tensor in (
        ("latent", latent),
        ("centers", centers),
        ("query_grid", query_grid),
    ):
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must have a floating dtype.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values.")
    if centers.dtype != query_grid.dtype:
        raise ValueError("centers and query_grid must have the same dtype.")
    devices = {latent.device, centers.device, query_grid.device, mask.device}
    if len(devices) != 1:
        raise ValueError("latent, centers, query_grid, and mask must share one device.")
    parameter = next(module.parameters())
    if parameter.device != latent.device:
        raise ValueError("decoder parameters and inputs must use the same device.")
