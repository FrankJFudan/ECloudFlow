"""Scientific-invariant tests for electron-density field projection."""

import math

import pytest
import torch
from e3nn import o3

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import ElectronField
from ecloudflow.ecloud.basis import SphericalFieldBasis
from ecloudflow.ecloud.field import (
    electron_field_multipole_moments,
    integrated_electron_count,
    multipole_moments,
    project_density_to_atoms,
    project_electron_field_to_atoms,
    reconstruct_density,
    reconstruct_electron_field,
)


def _cartesian_grid(
    extent: float,
    points_per_axis: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-extent, extent, points_per_axis, dtype=dtype, device=device)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(
        -1, 3
    )
    spacing = 2.0 * extent / (points_per_axis - 1)
    weights = torch.full((grid.shape[0],), spacing**3, dtype=dtype, device=device)
    return grid, weights


def _gaussian_density(
    grid: torch.Tensor,
    *,
    electrons: float,
    center: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    displacement = grid - center
    normalization = electrons / ((2.0 * math.pi) ** 1.5 * sigma**3)
    return normalization * torch.exp(-0.5 * displacement.square().sum(-1) / sigma**2)


def test_gaussian_projection_has_requested_coefficient_layout():
    grid, weights = _cartesian_grid(2.5, 15)
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=grid.dtype)
    density = _gaussian_density(
        grid,
        electrons=6.0,
        center=torch.tensor([0.2, 0.0, 0.0], dtype=grid.dtype),
        sigma=0.6,
    )
    basis = SphericalFieldBasis(n_radial=4, lmax=2, cutoff=3.5)

    coefficients = project_density_to_atoms(density, grid, centers, weights, basis)

    assert coefficients.shape == (2, 4, 9)


def test_gaussian_reconstruction_preserves_integrated_electron_count():
    grid, weights = _cartesian_grid(4.0, 33)
    centers = torch.zeros((1, 3), dtype=grid.dtype)
    density = _gaussian_density(grid, electrons=6.0, center=centers[0], sigma=0.65)
    basis = SphericalFieldBasis(n_radial=6, lmax=2, cutoff=4.0, chunk_size=1024)

    coefficients = project_density_to_atoms(density, grid, centers, weights, basis)
    reconstructed = reconstruct_density(coefficients, grid, centers, basis)

    assert abs(integrated_electron_count(reconstructed, weights) - 6.0) < 0.15


def test_projection_coefficients_transform_in_nonzero_irrep_blocks():
    grid, weights = _cartesian_grid(2.5, 18)
    centers = torch.tensor([[0.1, -0.2, 0.0]], dtype=grid.dtype)
    gaussian_center = torch.tensor([0.65, -0.05, 0.3], dtype=grid.dtype)
    density = _gaussian_density(grid, electrons=3.0, center=gaussian_center, sigma=0.55)
    basis = SphericalFieldBasis(n_radial=5, lmax=2, cutoff=3.5, chunk_size=701)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=grid.dtype,
    )

    coefficients = project_density_to_atoms(density, grid, centers, weights, basis)
    rotated = project_density_to_atoms(
        density, grid @ rotation.T, centers @ rotation.T, weights, basis
    )
    alpha, beta, gamma = o3.matrix_to_angles(rotation)

    assert coefficients[..., basis.l_slice(1)].norm() > 0.05
    for order in range(basis.lmax + 1):
        representation = o3.wigner_D(order, alpha, beta, gamma)
        expected = coefficients[..., basis.l_slice(order)] @ representation.T
        assert torch.allclose(
            rotated[..., basis.l_slice(order)], expected, atol=3e-5, rtol=3e-5
        )


def test_reconstruction_is_invariant_under_consistent_proper_rotation():
    grid, weights = _cartesian_grid(2.5, 17)
    centers = torch.tensor([[0.1, -0.2, 0.0]], dtype=grid.dtype)
    density = _gaussian_density(
        grid,
        electrons=3.0,
        center=torch.tensor([0.65, -0.05, 0.3], dtype=grid.dtype),
        sigma=0.55,
    )
    basis = SphericalFieldBasis(n_radial=5, lmax=2, cutoff=3.5, chunk_size=509)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=grid.dtype,
    )

    coefficients = project_density_to_atoms(density, grid, centers, weights, basis)
    reconstruction = reconstruct_density(coefficients, grid, centers, basis)
    rotated_coefficients = project_density_to_atoms(
        density, grid @ rotation.T, centers @ rotation.T, weights, basis
    )
    rotated_reconstruction = reconstruct_density(
        rotated_coefficients, grid @ rotation.T, centers @ rotation.T, basis
    )

    assert torch.allclose(reconstruction, rotated_reconstruction, atol=3e-5, rtol=3e-5)


def test_multipole_count_matches_hand_derived_gaussian_value():
    grid, weights = _cartesian_grid(3.5, 31)
    electron_count = 2.5
    center = torch.tensor([0.4, -0.3, 0.2], dtype=grid.dtype)
    density = _gaussian_density(
        grid, electrons=electron_count, center=center, sigma=0.45
    )

    moments = multipole_moments(density, grid, weights)
    assert torch.allclose(
        moments.electron_count,
        torch.tensor(electron_count, dtype=grid.dtype),
        atol=2e-5,
    )


def test_multipole_dipole_matches_hand_derived_shifted_gaussian_value():
    grid, weights = _cartesian_grid(3.5, 31)
    electron_count = 2.5
    center = torch.tensor([0.4, -0.3, 0.2], dtype=grid.dtype)
    density = _gaussian_density(
        grid, electrons=electron_count, center=center, sigma=0.45
    )

    moments = multipole_moments(density, grid, weights)

    assert torch.allclose(moments.dipole, electron_count * center, atol=2e-5)


def test_multipole_quadrupole_matches_hand_derived_shifted_gaussian_value():
    grid, weights = _cartesian_grid(3.5, 31)
    electron_count = 2.5
    center = torch.tensor([0.4, -0.3, 0.2], dtype=grid.dtype)
    density = _gaussian_density(
        grid, electrons=electron_count, center=center, sigma=0.45
    )

    moments = multipole_moments(density, grid, weights)
    expected_quadrupole = electron_count * (
        3.0 * torch.outer(center, center)
        - center.square().sum() * torch.eye(3, dtype=grid.dtype)
    )

    assert torch.allclose(moments.quadrupole, expected_quadrupole, atol=8e-5)


def test_projection_reconstruction_and_moments_are_differentiable():
    grid, weights = _cartesian_grid(1.5, 9, dtype=torch.float32)
    centers = torch.tensor([[0.13, -0.07, 0.11]], requires_grad=True)
    raw_density = torch.linspace(0.2, 1.0, grid.shape[0], requires_grad=True)
    density = raw_density.square()
    basis = SphericalFieldBasis(n_radial=3, lmax=2, cutoff=3.0, chunk_size=101)

    coefficients = project_density_to_atoms(density, grid, centers, weights, basis)
    reconstruction = reconstruct_density(coefficients, grid, centers, basis)
    moments = multipole_moments(reconstruction, grid, weights)
    loss = reconstruction.square().mean() + moments.dipole.square().sum()
    loss.backward()

    assert raw_density.grad is not None and torch.isfinite(raw_density.grad).all()
    assert centers.grad is not None and torch.isfinite(centers.grad).all()


def test_chunk_size_does_not_change_projection_or_reconstruction():
    grid, weights = _cartesian_grid(1.5, 9, dtype=torch.float32)
    centers = torch.tensor([[0.1, 0.0, 0.0], [-0.4, 0.2, 0.1]])
    density = _gaussian_density(
        grid, electrons=4.0, center=torch.tensor([0.2, 0.0, 0.0]), sigma=0.6
    )
    small_chunks = SphericalFieldBasis(3, 2, 3.0, chunk_size=97)
    single_chunk = SphericalFieldBasis(3, 2, 3.0, chunk_size=grid.shape[0])

    small_coefficients = project_density_to_atoms(
        density, grid, centers, weights, small_chunks
    )
    large_coefficients = project_density_to_atoms(
        density, grid, centers, weights, single_chunk
    )
    small_reconstruction = reconstruct_density(
        small_coefficients, grid, centers, small_chunks
    )
    large_reconstruction = reconstruct_density(
        large_coefficients, grid, centers, single_chunk
    )

    assert torch.allclose(small_coefficients, large_coefficients, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        small_reconstruction, large_reconstruction, atol=2e-5, rtol=2e-5
    )


def test_projection_rejects_negative_density():
    grid = torch.zeros((2, 3))
    centers = torch.zeros((1, 3))
    weights = torch.ones(2)
    basis = SphericalFieldBasis(2, 1, 2.0)

    with pytest.raises(ValueError, match="non-negative"):
        project_density_to_atoms(
            torch.tensor([1.0, -0.1]), grid, centers, weights, basis
        )


def test_projection_rejects_nonpositive_quadrature_weights():
    grid = torch.zeros((2, 3))
    centers = torch.zeros((1, 3))
    density = torch.ones(2)
    basis = SphericalFieldBasis(2, 1, 2.0)

    with pytest.raises(ValueError, match="strictly positive"):
        project_density_to_atoms(
            density, grid, centers, torch.tensor([1.0, 0.0]), basis
        )


def test_projection_rejects_shape_mismatches():
    basis = SphericalFieldBasis(2, 1, 2.0)

    with pytest.raises(ValueError, match="shape"):
        project_density_to_atoms(
            torch.ones(2),
            torch.zeros((3, 3)),
            torch.zeros((1, 3)),
            torch.ones(2),
            basis,
        )


def test_projection_rejects_dtype_mismatches():
    basis = SphericalFieldBasis(2, 1, 2.0)

    with pytest.raises(ValueError, match="dtype"):
        project_density_to_atoms(
            torch.ones(2, dtype=torch.float64),
            torch.zeros((2, 3), dtype=torch.float32),
            torch.zeros((1, 3), dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
            basis,
        )


def test_field_reductions_reject_nonfinite_inputs():
    density = torch.tensor([1.0, float("nan")])
    weights = torch.ones(2)

    with pytest.raises(ValueError, match="finite"):
        integrated_electron_count(density, weights)


def test_reconstruction_rejects_invalid_coefficient_layout():
    basis = SphericalFieldBasis(2, 2, 3.0)

    with pytest.raises(ValueError, match="coefficients"):
        reconstruct_density(
            torch.zeros((1, 2, 8)),
            torch.zeros((2, 3)),
            torch.zeros((1, 3)),
            basis,
        )


def test_autocast_keeps_projection_and_reductions_in_float32():
    grid, weights = _cartesian_grid(1.0, 7, dtype=torch.float32)
    centers = torch.tensor([[0.1, 0.0, 0.0]])
    density = torch.ones(grid.shape[0])
    basis = SphericalFieldBasis(2, 1, 2.0, chunk_size=43)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        coefficients = project_density_to_atoms(density, grid, centers, weights, basis)
        reconstruction = reconstruct_density(coefficients, grid, centers, basis)
        count = integrated_electron_count(reconstruction, weights)

    assert coefficients.dtype == torch.float32
    assert reconstruction.dtype == torch.float32
    assert count.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_projection_matches_cpu():
    grid, weights = _cartesian_grid(1.5, 9, dtype=torch.float32)
    centers = torch.tensor([[0.1, -0.2, 0.0]])
    density = _gaussian_density(
        grid, electrons=3.0, center=torch.tensor([0.3, 0.1, -0.1]), sigma=0.55
    )
    basis = SphericalFieldBasis(3, 2, 3.0, chunk_size=113)
    expected = project_density_to_atoms(density, grid, centers, weights, basis)

    actual = project_density_to_atoms(
        density.cuda(), grid.cuda(), centers.cuda(), weights.cuda(), basis
    )

    assert torch.allclose(actual.cpu(), expected, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_projection_rejects_cross_device_inputs():
    grid, weights = _cartesian_grid(1.5, 5, dtype=torch.float32)
    centers = torch.tensor([[0.1, -0.2, 0.0]])
    density = torch.ones(grid.shape[0])
    basis = SphericalFieldBasis(3, 2, 3.0)

    with pytest.raises(ValueError, match="device"):
        project_density_to_atoms(density.cuda(), grid, centers, weights, basis)


def test_electron_field_projection_applies_masks_and_isolates_batches():
    frame = CoordinateFrame(torch.zeros(3, dtype=torch.float64))
    field = ElectronField(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        values=torch.tensor([[1.0], [3.0], [100.0]], dtype=torch.float64),
        mask=torch.tensor([True, True, False]),
        batch=torch.tensor([0, 1, 0], dtype=torch.long),
        channel_names=("density",),
        frame=frame,
    )
    centers = torch.zeros((2, 3), dtype=torch.float64)
    center_batch = torch.tensor([0, 1], dtype=torch.long)
    weights = torch.ones(3, dtype=torch.float64)
    basis = SphericalFieldBasis(2, 1, 2.0)

    projected = project_electron_field_to_atoms(
        field,
        centers,
        center_batch,
        weights,
        basis,
        centers_frame=frame,
    )
    expected_first = project_density_to_atoms(
        field.values[:1, 0], field.positions[:1], centers[:1], weights[:1], basis
    )
    expected_second = project_density_to_atoms(
        field.values[1:2, 0], field.positions[1:2], centers[1:], weights[1:2], basis
    )

    assert torch.allclose(projected.values[0], expected_first[0])
    assert torch.allclose(projected.values[1], expected_second[0])


def test_electron_field_projection_rejects_mismatched_center_frame():
    field_frame = CoordinateFrame(torch.zeros(3))
    field = ElectronField(
        positions=torch.zeros((1, 3)),
        values=torch.ones((1, 1)),
        mask=torch.tensor([True]),
        batch=torch.tensor([0], dtype=torch.long),
        channel_names=("density",),
        frame=field_frame,
    )
    wrong_frame = CoordinateFrame(torch.tensor([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="frame"):
        project_electron_field_to_atoms(
            field,
            torch.zeros((1, 3)),
            torch.tensor([0], dtype=torch.long),
            torch.ones(1),
            SphericalFieldBasis(2, 1, 2.0),
            centers_frame=wrong_frame,
        )


def test_electron_field_projection_requires_explicit_frame_provenance():
    field = ElectronField(
        positions=torch.zeros((1, 3)),
        values=torch.ones((1, 1)),
        mask=torch.tensor([True]),
        batch=torch.tensor([0], dtype=torch.long),
        channel_names=("density",),
        frame=None,
    )

    with pytest.raises(ValueError, match="frame"):
        project_electron_field_to_atoms(
            field,
            torch.zeros((1, 3)),
            torch.tensor([0], dtype=torch.long),
            torch.ones(1),
            SphericalFieldBasis(2, 1, 2.0),
            centers_frame=CoordinateFrame(torch.zeros(3)),
        )


def test_electron_field_reconstruction_rejects_query_in_another_frame():
    frame = CoordinateFrame(torch.zeros(3))
    field = ElectronField(
        positions=torch.zeros((1, 3)),
        values=torch.ones((1, 1)),
        mask=torch.tensor([True]),
        batch=torch.tensor([0], dtype=torch.long),
        channel_names=("density",),
        frame=frame,
    )
    projected = project_electron_field_to_atoms(
        field,
        torch.zeros((1, 3)),
        torch.tensor([0], dtype=torch.long),
        torch.ones(1),
        SphericalFieldBasis(2, 1, 2.0),
        centers_frame=frame,
    )
    query = ElectronField(
        positions=torch.zeros((1, 3)),
        values=torch.zeros((1, 1)),
        mask=torch.tensor([True]),
        batch=torch.tensor([0], dtype=torch.long),
        frame=CoordinateFrame(torch.tensor([1.0, 0.0, 0.0])),
    )

    with pytest.raises(ValueError, match="frame"):
        reconstruct_electron_field(projected, query)


def test_electron_field_reconstruction_applies_masks_and_isolates_batches():
    frame = CoordinateFrame(torch.zeros(3, dtype=torch.float64))
    source = ElectronField(
        positions=torch.zeros((2, 3), dtype=torch.float64),
        values=torch.tensor([[1.0], [3.0]], dtype=torch.float64),
        mask=torch.tensor([True, True]),
        batch=torch.tensor([0, 1], dtype=torch.long),
        channel_names=("density",),
        frame=frame,
    )
    centers = torch.zeros((2, 3), dtype=torch.float64)
    center_batch = torch.tensor([0, 1], dtype=torch.long)
    basis = SphericalFieldBasis(2, 1, 2.0)
    projected = project_electron_field_to_atoms(
        source,
        centers,
        center_batch,
        torch.ones(2, dtype=torch.float64),
        basis,
        centers_frame=frame,
    )
    query = ElectronField(
        positions=torch.tensor(
            [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        values=torch.zeros((3, 1), dtype=torch.float64),
        mask=torch.tensor([True, True, False]),
        batch=torch.tensor([0, 1, 1], dtype=torch.long),
        frame=frame,
    )

    reconstructed = reconstruct_electron_field(projected, query)
    expected_first = reconstruct_density(
        projected.values[:1], query.positions[:1], centers[:1], basis
    )
    expected_second = reconstruct_density(
        projected.values[1:], query.positions[1:2], centers[1:], basis
    )

    assert torch.allclose(reconstructed.values[:1, 0], expected_first)
    assert torch.allclose(reconstructed.values[1:2, 0], expected_second)
    assert reconstructed.values[2, 0] == 0.0


def test_electron_field_multipoles_apply_masks_and_reduce_each_batch():
    frame = CoordinateFrame(torch.zeros(3, dtype=torch.float64))
    field = ElectronField(
        positions=torch.tensor(
            [[1.0, 0.0, 0.0], [100.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        values=torch.tensor([[2.0], [100.0], [3.0]], dtype=torch.float64),
        mask=torch.tensor([True, False, True]),
        batch=torch.tensor([0, 0, 2], dtype=torch.long),
        channel_names=("density",),
        frame=frame,
    )

    moments = electron_field_multipole_moments(
        field, torch.tensor([0.5, 9.0, 0.25], dtype=torch.float64)
    )

    assert torch.equal(moments.batch, torch.tensor([0, 2], dtype=torch.long))
    assert torch.allclose(
        moments.electron_count,
        torch.tensor([1.0, 0.75], dtype=torch.float64),
    )
    assert moments.frame is frame


def test_atom_projection_vanishes_smoothly_as_center_crosses_cutoff():
    cutoff = 2.0
    basis = SphericalFieldBasis(2, 1, cutoff)

    def projected_norm(distance: float) -> tuple[float, float]:
        center = torch.tensor(
            [[distance, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        coefficients = project_density_to_atoms(
            torch.ones(1, dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.float64),
            center,
            torch.ones(1, dtype=torch.float64),
            basis,
        )
        value = coefficients.square().sum()
        gradient = torch.autograd.grad(value, center)[0]
        return value.item(), gradient.norm().item()

    inside_value, inside_gradient = projected_norm(cutoff - 1e-4)
    boundary_value, boundary_gradient = projected_norm(cutoff)
    outside_value, outside_gradient = projected_norm(cutoff + 1e-4)

    assert inside_value < 1e-10
    assert inside_gradient < 1e-5
    assert boundary_value == outside_value == 0.0
    assert boundary_gradient == outside_gradient == 0.0
