"""Scientific contract tests for the equivariant electron-field tokenizer."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
from e3nn import o3

from ecloudflow.ecloud.decoder import ElectronFieldDecoder
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
    lmax = math.isqrt(coefficients.shape[-1]) - 1
    assert (lmax + 1) ** 2 == coefficients.shape[-1]
    blocks = []
    for order in range(lmax + 1):
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


def test_pure_octupole_input_reaches_generic_lmax3_latent_and_field() -> None:
    generator = torch.Generator().manual_seed(97)
    coefficients = torch.zeros(1, 3, 2, 16)
    coefficients[..., 9:] = torch.randn(1, 3, 2, 7, generator=generator)
    features = torch.randn(1, 3, 9, generator=generator)
    centers = torch.randn(1, 3, 3, generator=generator) * 0.3
    queries = torch.randn(1, 9, 3, generator=generator) * 0.6
    mask = torch.tensor([[True, True, False]])
    tokenizer = EquivariantFieldTokenizer(
        n_radial=2,
        lmax=3,
        scalar_dim=12,
        vector_dim=2,
        latent_dim=24,
        cutoff=3.0,
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

    assert tokenizer.latent_irreps == o3.Irreps("6x0e + 2x1o + 1x2e + 1x3o")
    octupole_slice = tokenizer.latent_irreps.slices()[-1]
    assert torch.count_nonzero(latent[..., octupole_slice]) > 0
    assert not torch.allclose(reconstruction.density, scalar_only.density)
    transform = tokenizer.latent_irreps.D_from_matrix(rotation)
    assert torch.allclose(rotated_latent, latent @ transform.T, atol=4e-4, rtol=4e-4)
    assert torch.allclose(rotated.density, reconstruction.density, atol=4e-4, rtol=4e-4)


def test_layout_rejects_width_that_cannot_retain_lmax3_irreps() -> None:
    with pytest.raises(ValueError, match="retain every configured higher irrep"):
        EquivariantFieldTokenizer(
            n_radial=2,
            lmax=3,
            scalar_dim=12,
            vector_dim=2,
            latent_dim=18,
        )


def test_decoder_docs_bind_dynamic_layout_and_planned_example() -> None:
    class_doc = inspect.getdoc(ElectronFieldDecoder) or ""
    forward_doc = inspect.getdoc(ElectronFieldDecoder.forward) or ""
    for doc in (class_doc, forward_doc):
        assert "19x0e + 8x1o + 1x2e" in doc
        assert "every configured" in doc
        assert "(C-3V)x0e + Vx1o" not in doc


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


def test_extreme_finite_padding_is_sanitized_before_numerics_and_backward() -> None:
    coefficients, features, centers, queries, mask = _inputs()
    coefficients.requires_grad_()
    features.requires_grad_()
    centers.requires_grad_()
    tokenizer = EquivariantFieldTokenizer(
        n_radial=3, lmax=2, scalar_dim=16, vector_dim=4, latent_dim=24
    )
    reference_latent = tokenizer(coefficients, features, mask)
    reference = tokenizer.decode(reference_latent, centers, queries, mask)
    extreme = torch.finfo(coefficients.dtype).max

    changed_coefficients = torch.where(
        mask[..., None, None], coefficients.detach(), extreme
    ).requires_grad_()
    changed_features = torch.where(
        mask[..., None], features.detach(), extreme
    ).requires_grad_()
    changed_centers = torch.where(
        mask[..., None], centers.detach(), extreme
    ).requires_grad_()
    changed_latent = tokenizer(changed_coefficients, changed_features, mask)
    changed_latent = torch.where(
        mask[..., None], changed_latent, torch.full_like(changed_latent, extreme)
    )
    changed = tokenizer.decode(changed_latent, changed_centers, queries, mask)

    assert torch.equal(reference_latent, tokenizer(coefficients, features, mask))
    for left, right in zip(reference, changed, strict=True):
        assert torch.equal(left, right)
        assert torch.isfinite(right).all()

    reference_loss = sum(value.square().mean() for value in reference)
    changed_loss = sum(value.square().mean() for value in changed)
    reference_grads = torch.autograd.grad(
        reference_loss, (coefficients, features, centers, reference_latent)
    )
    changed_grads = torch.autograd.grad(
        changed_loss,
        (changed_coefficients, changed_features, changed_centers, changed_latent),
    )
    for reference_grad, changed_grad in zip(
        reference_grads, changed_grads, strict=True
    ):
        assert torch.equal(reference_grad, changed_grad)
        assert torch.isfinite(changed_grad).all()
    assert torch.count_nonzero(changed_grads[0][~mask]) == 0
    assert torch.count_nonzero(changed_grads[1][~mask]) == 0
    assert torch.count_nonzero(changed_grads[2][~mask]) == 0
    assert torch.count_nonzero(changed_grads[3][~mask]) == 0


@pytest.mark.parametrize("extreme", [float("inf"), float("nan")])
def test_nonfinite_padding_is_safe_substituted_without_zero_multiplication(
    extreme: float,
) -> None:
    coefficients, features, centers, queries, mask = _inputs()
    tokenizer = EquivariantFieldTokenizer(
        n_radial=3, lmax=2, scalar_dim=16, vector_dim=4, latent_dim=24
    )
    reference_latent = tokenizer(coefficients, features, mask)
    reference = tokenizer.decode(reference_latent, centers, queries, mask)
    changed_coefficients = torch.where(mask[..., None, None], coefficients, extreme)
    changed_features = torch.where(mask[..., None], features, extreme)
    changed_centers = torch.where(mask[..., None], centers, extreme)
    changed_latent = tokenizer(changed_coefficients, changed_features, mask)
    changed_latent = torch.where(mask[..., None], changed_latent, extreme)
    changed = tokenizer.decode(changed_latent, changed_centers, queries, mask)

    assert torch.equal(changed_latent[mask], reference_latent[mask])
    for left, right in zip(reference, changed, strict=True):
        assert torch.equal(left, right)
        assert torch.isfinite(right).all()


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
