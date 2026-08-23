"""Scientific contract tests for the equivariant electron-field tokenizer."""

from __future__ import annotations

import pytest
import torch
from e3nn import o3

from ecloudflow.ecloud.tokenizer import EquivariantFieldTokenizer


def _inputs(*, requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(81)
    coefficients = torch.randn(2, 4, 3, 9, generator=generator) * 0.04
    coefficients.requires_grad_(requires_grad)
    atom_features = torch.randn(2, 4, 50, generator=generator)
    centers = torch.randn(2, 4, 3, generator=generator) * 0.4
    queries = torch.randn(2, 11, 3, generator=generator) * 0.7
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    return coefficients, atom_features, centers, queries, mask


def _rotation() -> torch.Tensor:
    axis = torch.tensor([0.3, -0.5, 0.8])
    axis = axis / torch.linalg.vector_norm(axis)
    return o3.axis_angle_to_matrix(axis, torch.tensor(0.63))


def _rotate_coefficients(
    coefficients: torch.Tensor, rotation: torch.Tensor
) -> torch.Tensor:
    blocks = []
    for order in range(3):
        block = coefficients[..., order**2 : (order + 1) ** 2]
        transform = o3.Irrep(order, (-1) ** order).D_from_matrix(rotation)
        blocks.append(block @ transform.T)
    return torch.cat(blocks, dim=-1)


def test_tokenizer_has_explicit_packed_irreps_and_masks_padding() -> None:
    coefficients, features, _, _, mask = _inputs()
    tokenizer = EquivariantFieldTokenizer(
        n_radial=3, lmax=2, scalar_dim=32, vector_dim=8, latent_dim=48
    )

    latent = tokenizer.encode(coefficients, features, mask)

    assert latent.shape == (2, 4, 48)
    assert tokenizer.latent_irreps == o3.Irreps("19x0e + 8x1o + 1x2e")
    assert torch.count_nonzero(latent[~mask]) == 0


def test_round_trip_is_differentiable_and_se3_equivariant() -> None:
    coefficients, features, centers, queries, mask = _inputs(requires_grad=True)
    tokenizer = EquivariantFieldTokenizer(
        n_radial=3,
        lmax=2,
        scalar_dim=32,
        vector_dim=8,
        latent_dim=48,
        cutoff=3.0,
        chunk_size=4,
    )
    rotation = _rotation()

    latent = tokenizer.encode(coefficients, features, mask)
    reconstruction = tokenizer.decode(latent, centers, queries, mask)
    rotated_latent = tokenizer.encode(
        _rotate_coefficients(coefficients, rotation), features, mask
    )
    rotated = tokenizer.decode(
        rotated_latent, centers @ rotation.T, queries @ rotation.T, mask
    )

    latent_transform = tokenizer.latent_irreps.D_from_matrix(rotation)
    assert torch.allclose(
        rotated_latent, latent @ latent_transform.T, atol=3e-4, rtol=3e-4
    )
    assert torch.allclose(rotated.density, reconstruction.density, atol=3e-4, rtol=3e-4)
    assert torch.allclose(
        rotated.gradient,
        reconstruction.gradient @ rotation.T,
        atol=5e-4,
        rtol=5e-4,
    )
    assert torch.allclose(
        rotated.electron_count, reconstruction.electron_count, atol=3e-4, rtol=3e-4
    )
    assert torch.allclose(
        rotated.dipole,
        reconstruction.dipole @ rotation.T,
        atol=5e-4,
        rtol=5e-4,
    )
    assert torch.allclose(
        rotated.latent_round_trip,
        reconstruction.latent_round_trip @ latent_transform.T,
        atol=3e-4,
        rtol=3e-4,
    )

    loss = (
        reconstruction.density.square().mean()
        + reconstruction.gradient.square().mean()
        + reconstruction.electron_count.square().mean()
        + reconstruction.dipole.square().mean()
        + reconstruction.latent_round_trip.square().mean()
    )
    loss.backward()
    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in tokenizer.parameters()
    )


def test_pure_quadrupole_input_reaches_latent_and_decoded_field() -> None:
    coefficients, features, centers, queries, mask = _inputs()
    coefficients.zero_()
    coefficients[:, :, :, 4:] = torch.randn_like(coefficients[:, :, :, 4:])
    tokenizer = EquivariantFieldTokenizer(
        n_radial=3, lmax=2, scalar_dim=16, vector_dim=4, latent_dim=24, cutoff=3.0
    )
    rotation = _rotation()

    latent = tokenizer(coefficients, features, mask)
    reconstruction = tokenizer.decode(latent, centers, queries, mask)
    scalar_only = tokenizer.decode(
        tokenizer(torch.zeros_like(coefficients), features, mask),
        centers,
        queries,
        mask,
    )
    rotated_latent = tokenizer(
        _rotate_coefficients(coefficients, rotation), features, mask
    )
    rotated = tokenizer.decode(
        rotated_latent, centers @ rotation.T, queries @ rotation.T, mask
    )

    tensor_slice = tokenizer.latent_irreps.slices()[-1]
    assert torch.count_nonzero(latent[..., tensor_slice]) > 0
    assert not torch.allclose(reconstruction.density, scalar_only.density)
    transform = tokenizer.latent_irreps.D_from_matrix(rotation)
    assert torch.allclose(rotated_latent, latent @ transform.T, atol=3e-4, rtol=3e-4)
    assert torch.allclose(rotated.density, reconstruction.density, atol=3e-4, rtol=3e-4)


def test_dipole_has_physical_translation_semantics() -> None:
    coefficients, features, centers, queries, mask = _inputs()
    tokenizer = EquivariantFieldTokenizer(
        n_radial=3, lmax=2, scalar_dim=16, vector_dim=4, latent_dim=24
    )
    latent = tokenizer(coefficients, features, mask)
    original = tokenizer.decode(latent, centers, queries, mask)
    translation = torch.tensor([1.2, -0.4, 0.7])

    shifted = tokenizer.decode(
        latent, centers + translation, queries + translation, mask
    )

    assert torch.allclose(shifted.density, original.density, atol=2e-5, rtol=2e-5)
    assert torch.allclose(shifted.electron_count, original.electron_count)
    expected_dipole = original.dipole + original.electron_count[:, None] * translation
    assert torch.allclose(shifted.dipole, expected_dipole, atol=2e-5, rtol=2e-5)


def test_padding_values_cannot_affect_any_reconstruction_output() -> None:
    coefficients, features, centers, queries, mask = _inputs()
    tokenizer = EquivariantFieldTokenizer(
        n_radial=3, lmax=2, scalar_dim=16, vector_dim=4, latent_dim=24
    )
    latent = tokenizer.encode(coefficients, features, mask)
    reference = tokenizer.decode(latent, centers, queries, mask)
    changed_coefficients = coefficients.clone()
    changed_features = features.clone()
    changed_centers = centers.clone()
    changed_coefficients[~mask] = 10_000.0
    changed_features[~mask] = -10_000.0
    changed_centers[~mask] = 1_000.0

    changed_latent = tokenizer.encode(changed_coefficients, changed_features, mask)
    changed = tokenizer.decode(changed_latent, changed_centers, queries, mask)

    for left, right in zip(reference, changed, strict=True):
        assert torch.equal(left, right)


def test_decoder_accumulates_float32_under_cpu_autocast() -> None:
    coefficients, features, centers, queries, mask = _inputs()
    tokenizer = EquivariantFieldTokenizer(
        n_radial=3, lmax=2, scalar_dim=16, vector_dim=4, latent_dim=24
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        reconstruction = tokenizer.decode(
            tokenizer.encode(coefficients, features, mask), centers, queries, mask
        )

    assert reconstruction.density.dtype == torch.float32
    assert reconstruction.gradient.dtype == torch.float32
    assert reconstruction.electron_count.dtype == torch.float32
    assert reconstruction.dipole.dtype == torch.float32


def test_query_chunk_size_does_not_change_decoding() -> None:
    coefficients, features, centers, queries, mask = _inputs()
    chunked = EquivariantFieldTokenizer(
        n_radial=3,
        lmax=2,
        scalar_dim=16,
        vector_dim=4,
        latent_dim=24,
        chunk_size=3,
    )
    unchunked = EquivariantFieldTokenizer(
        n_radial=3,
        lmax=2,
        scalar_dim=16,
        vector_dim=4,
        latent_dim=24,
        chunk_size=100,
    )
    chunked_latent = chunked(coefficients, features, mask)
    unchunked(coefficients, features, mask)
    unchunked.load_state_dict(chunked.state_dict())

    chunked_output = chunked.decode(chunked_latent, centers, queries, mask)
    unchunked_output = unchunked.decode(chunked_latent, centers, queries, mask)

    for left, right in zip(chunked_output, unchunked_output, strict=True):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("coefficients", torch.zeros(2, 4, 3, 8), "harmonic"),
        ("atom_features", torch.zeros(2, 5, 50), "feature"),
        ("mask", torch.ones(2, 4), "boolean"),
    ],
)
def test_tokenizer_rejects_inconsistent_boundary_tensors(
    field: str, replacement: torch.Tensor, message: str
) -> None:
    coefficients, features, _, _, mask = _inputs()
    tokenizer = EquivariantFieldTokenizer(
        n_radial=3, lmax=2, scalar_dim=16, vector_dim=4, latent_dim=24
    )
    values = {
        "coefficients": coefficients,
        "atom_features": features,
        "mask": mask,
    }
    values[field] = replacement
    with pytest.raises(ValueError, match=message):
        tokenizer.encode(**values)
