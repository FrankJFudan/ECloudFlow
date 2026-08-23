"""Equivariant atom-centered electron-field tokenization."""

from __future__ import annotations

import torch
from e3nn import o3
from torch import nn
from torch.nn import functional

from ecloudflow.ecloud.decoder import ElectronFieldDecoder, ElectronReconstruction


class EquivariantFieldTokenizer(nn.Module):  # type: ignore[misc]
    """Encode spherical coefficients into Task 10-compatible packed tokens.

    :param n_radial: Positive number ``R`` of Task 4 radial channels.
    :param lmax: Non-negative largest real-spherical-harmonic order.
    :param scalar_dim: Positive hidden invariant width used for field/atom
        conditioning; it is deliberately not an output multiplicity.
    :param vector_dim: Non-negative number ``V`` of latent ``1o`` vector copies.
    :param latent_dim: Total packed output width ``C``. One copy of every
        configured order ``l>=2`` is reserved, ``V`` copies are reserved for
        ``1o``, and all remaining components are even invariant scalars.
    :param cutoff: Positive decoder basis cutoff in angstroms.
    :param chunk_size: Positive maximum decoded query chunk size.
    :return: Trainable field encoder with a matching decoder.
    :rtype: EquivariantFieldTokenizer
    :raises ValueError: If dimensions do not leave at least one invariant
        scalar or the basis configuration is invalid.

    The public padded boundary is ``[B,N,R,H]`` to ``[B,N,C]``. Internally,
    coefficients are explicitly repacked as ``R x 0e + R x 1o + ...`` for
    :class:`e3nn.o3.Linear`; no padded form changes the canonical flattened
    ``MolecularState.electron_latent`` graph contract. For ``lmax=2``,
    ``latent_dim=48``, and ``vector_dim=8``, the exact layout is
    ``19x0e + 8x1o + 1x2e``. The retained higher-order copies prevent an
    orientation-bearing input order from being silently discarded.

    Coefficients use Task 4 units of electrons per angstrom to power ``3/2``.
    Atom features are invariant, unitless graph channels. Every tensor stays
    on the caller/module device; FP16/BF16 inputs are evaluated with float32
    module arithmetic. Masks are exact, inputs are never mutated, and fixed
    parameters make calls deterministic. Proper rotations transform each
    packed irrep while translations do not affect tokens. Gradients propagate
    through coefficients, invariant conditioning, gates, and all parameters;
    invalid shape/dtype/device/layout data raise before numerical evaluation.
    """

    def __init__(
        self,
        n_radial: int,
        lmax: int,
        scalar_dim: int,
        vector_dim: int,
        latent_dim: int,
        *,
        cutoff: float = 5.0,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if (
            not isinstance(scalar_dim, int)
            or isinstance(scalar_dim, bool)
            or scalar_dim <= 0
        ):
            raise ValueError("scalar_dim must be a positive integer hidden width.")
        self.decoder = ElectronFieldDecoder(
            n_radial=n_radial,
            lmax=lmax,
            vector_dim=vector_dim,
            latent_dim=latent_dim,
            cutoff=cutoff,
            chunk_size=chunk_size,
        )
        self.n_radial = n_radial
        self.lmax = lmax
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.latent_dim = latent_dim
        self.latent_irreps = self.decoder.latent_irreps
        self.coefficient_irreps = self.decoder.coefficient_irreps
        hidden_entries = [(scalar_dim, (0, 1))]
        hidden_entries.extend(
            (vector_dim, (order, (-1) ** order)) for order in range(1, lmax + 1)
        )
        self.hidden_irreps = o3.Irreps(hidden_entries)
        self.field_linear = o3.Linear(
            self.coefficient_irreps, self.hidden_irreps, biases=True
        )
        self.atom_conditioning = nn.LazyLinear(scalar_dim)
        self.gate_conditioning = (
            nn.Linear(scalar_dim, vector_dim * lmax) if lmax > 0 else None
        )
        self.to_latent = o3.Linear(self.hidden_irreps, self.latent_irreps, biases=True)

    def forward(
        self,
        coefficients: torch.Tensor,
        atom_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode atom-centered density coefficients into equivariant tokens.

        :param coefficients: Tensor ``[B, N, R, H]`` where
            ``H=(lmax+1)**2`` stores Task 4 real harmonics in contiguous
            ``[l**2:(l+1)**2]`` slices, with coefficient units of electrons
            per angstrom to power ``3/2``.
        :param atom_features: Finite invariant graph features ``[B, N, F]``;
            the Task 6 pocket schema has ``F=50``, while other explicit graph
            schemas may use a different stable positive width.
        :param mask: Boolean physical-atom mask ``[B, N]`` on the input device;
            false rows are mapped to exact zeros regardless of padded values.
        :return: Padded per-atom tokens ``[B, N, C]`` on the module device in
            :attr:`latent_irreps` order. For every ``l>=2`` one irrep copy is
            present, vectors occupy ``V`` contiguous ``1o`` triples, and the
            remainder are invariant scalars.
        :rtype: torch.Tensor
        :raises TypeError: If an input is not a tensor.
        :raises ValueError: If harmonic/radial layouts, feature shapes, masks,
            floating dtypes, finite values, devices, or module placement differ.

        The padded boundary is converted explicitly to order-grouped e3nn
        blocks and is not a canonical PyG state representation. Scalar SiLU
        and invariant sigmoid gates provide scalar/non-scalar activations;
        :class:`e3nn.o3.Linear` performs only same-irrep mixing. Atom features
        condition scalars and gates but cannot define a spatial direction.
        The output is translation invariant and SE(3)-equivariant under the
        packed irrep representation. Autocast is disabled and low-precision
        data use float32 arithmetic on the same device. The method is
        deterministic for fixed state, does not mutate/detach inputs, and
        preserves gradients to physical coefficient and feature rows; masked
        rows receive exactly zero gradient through their values. Input shape,
        dtype, and centered coordinate frame semantics are validated explicitly.
        """
        _validate_encode_inputs(self, coefficients, atom_features, mask)
        parameter = next(self.parameters())
        dtype = parameter.dtype
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        physical = mask.unsqueeze(-1).to(dtype=dtype)
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            packed = _pack_coefficients(
                coefficients.to(dtype=dtype), self.n_radial, self.lmax
            )
            hidden = self.field_linear(packed * physical)
            conditioned = self.atom_conditioning(atom_features.to(dtype=dtype))
            scalar_stop = self.scalar_dim
            scalars = functional.silu(hidden[..., :scalar_stop] + conditioned)
            blocks = [scalars]
            offset = scalar_stop
            if self.lmax > 0:
                assert self.gate_conditioning is not None
                gates = torch.sigmoid(self.gate_conditioning(conditioned))
                gate_offset = 0
                for order in range(1, self.lmax + 1):
                    width = self.vector_dim * (2 * order + 1)
                    block = hidden[..., offset : offset + width]
                    order_gates = gates[
                        ..., gate_offset : gate_offset + self.vector_dim
                    ].repeat_interleave(2 * order + 1, dim=-1)
                    blocks.append(block * order_gates)
                    offset += width
                    gate_offset += self.vector_dim
            activated = torch.cat(blocks, dim=-1) * physical
            return self.to_latent(activated) * physical

    def encode(
        self,
        coefficients: torch.Tensor,
        atom_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a padded field boundary using :meth:`forward`.

        :param coefficients: Task 4 coefficients ``[B,N,R,H]`` in electrons
            per angstrom to power ``3/2`` and the configured harmonic layout.
        :param atom_features: Invariant finite features ``[B,N,F]``.
        :param mask: Boolean physical-atom mask ``[B,N]``.
        :return: Equivariant tokens ``[B,N,C]`` in :attr:`latent_irreps` order,
            with exact zero padding and unchanged input device.
        :rtype: torch.Tensor
        :raises TypeError: If an input is not a tensor.
        :raises ValueError: If shape, mask, dtype/device, finite-value, frame
            representation, or irrep-layout validation fails.

        Units, explicit padded-to-packed conversion, float32 mixed-precision
        behavior, non-mutation, deterministic evaluation, SE(3) semantics,
        and gradient behavior are exactly those documented by :meth:`forward`.
        """
        return self.forward(coefficients, atom_features, mask)

    def decode(
        self,
        latent: torch.Tensor,
        centers: torch.Tensor,
        query_grid: torch.Tensor,
        mask: torch.Tensor,
    ) -> ElectronReconstruction:
        """Decode packed atom tokens into a continuous electron field.

        :param latent: Packed equivariant tokens ``[B,N,C]`` in
            :attr:`latent_irreps` order on the module device.
        :param centers: Atom centers ``[B,N,3]`` in local-frame angstroms.
        :param query_grid: Query coordinates ``[B,G,3]`` in the same frame.
        :param mask: Boolean physical-atom mask ``[B,N]``; padding is ignored.
        :return: Density ``[B,G]``, vector gradient ``[B,G,3]``, invariant
            count ``[B]``, covariant frame-origin dipole ``[B,3]``, and cycle
            tokens ``[B,N,C]``.
        :rtype: ElectronReconstruction
        :raises TypeError: If an input is not a tensor.
        :raises ValueError: If shape, packed layout, dtype/device, finiteness,
            frame-coordinate, or mask validation fails.

        The decoder processes query chunks, accumulates density and analytic
        moments in at least float32 under mixed precision, and retains gradients
        through density gradients and cycle tokens. It performs no random
        operation, mutation, silent fallback, or fake target construction.
        Rotation/translation behavior, units, and padding semantics follow
        :class:`ElectronFieldDecoder` and :class:`ElectronReconstruction`.
        Evaluation is deterministic for fixed parameters and tensor inputs.
        """
        return self.decoder.decode(latent, centers, query_grid, mask)


def _pack_coefficients(
    coefficients: torch.Tensor, n_radial: int, lmax: int
) -> torch.Tensor:
    """Convert boundary radial-major harmonics to e3nn order-grouped blocks."""
    return torch.cat(
        [
            coefficients[..., order**2 : (order + 1) ** 2].reshape(
                *coefficients.shape[:-2], n_radial * (2 * order + 1)
            )
            for order in range(lmax + 1)
        ],
        dim=-1,
    )


def _validate_encode_inputs(
    module: EquivariantFieldTokenizer,
    coefficients: torch.Tensor,
    atom_features: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Validate the only padded coefficient/token boundary."""
    for name, tensor in (
        ("coefficients", coefficients),
        ("atom_features", atom_features),
        ("mask", mask),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
    expected_harmonics = (module.lmax + 1) ** 2
    if coefficients.ndim != 4 or coefficients.shape[-2:] != (
        module.n_radial,
        expected_harmonics,
    ):
        raise ValueError(
            "coefficients have inconsistent radial or harmonic layout; expected "
            f"[B, N, {module.n_radial}, {expected_harmonics}]."
        )
    if (
        atom_features.ndim != 3
        or atom_features.shape[:2] != coefficients.shape[:2]
        or atom_features.shape[-1] == 0
    ):
        raise ValueError("atom feature shape must be [B, N, F] with F > 0.")
    if mask.shape != coefficients.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor with shape [B, N].")
    for name, tensor in (
        ("coefficients", coefficients),
        ("atom_features", atom_features),
    ):
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must have a floating dtype.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values.")
    if (
        coefficients.device != atom_features.device
        or mask.device != coefficients.device
    ):
        raise ValueError("coefficients, atom_features, and mask must share one device.")
    parameter = next(module.parameters())
    if parameter.device != coefficients.device:
        raise ValueError("tokenizer parameters and inputs must use the same device.")
