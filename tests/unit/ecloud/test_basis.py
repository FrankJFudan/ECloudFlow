"""Tests for the compact equivariant electron-density basis."""

import math

import pytest
import torch
from e3nn import o3

from ecloudflow.ecloud.basis import SphericalFieldBasis


def test_basis_exposes_contiguous_e3nn_irrep_slices():
    basis = SphericalFieldBasis(n_radial=4, lmax=3, cutoff=5.0)

    assert basis.harmonic_dim == 16
    assert basis.irreps == o3.Irreps("0e + 1o + 2e + 3o")
    assert [basis.l_slice(order) for order in range(4)] == [
        slice(0, 1),
        slice(1, 4),
        slice(4, 9),
        slice(9, 16),
    ]


def test_l0_radial_channels_are_smooth_at_atom_center():
    basis = SphericalFieldBasis(n_radial=3, lmax=1, cutoff=2.0)
    radii = torch.tensor([0.0, 1e-6], dtype=torch.float64, requires_grad=True)

    values = basis.radial_values(radii)
    gradient = torch.autograd.grad(values.sum(), radii)[0]

    assert torch.allclose(values[0], values[1], atol=1e-10, rtol=1e-10)
    assert gradient[0] == 0.0
    assert abs(gradient[1]) < 1e-3


def test_radial_channels_have_compact_cutoff_support():
    basis = SphericalFieldBasis(n_radial=3, lmax=1, cutoff=2.0)

    values = basis.radial_values(torch.tensor([2.01], dtype=torch.float64))

    assert values.equal(torch.zeros((1, 3), dtype=torch.float64))


@pytest.mark.parametrize("order", [0, 1, 2])
def test_radial_channels_are_orthonormal_under_spherical_volume_measure(order):
    basis = SphericalFieldBasis(n_radial=5, lmax=2, cutoff=3.5)
    step = basis.cutoff / 20_000
    radii = (torch.arange(20_000, dtype=torch.float64) + 0.5) * step
    values = basis.radial_values(radii, order=order)

    gram = values.T @ (values * (radii.square() * step).unsqueeze(-1))

    assert torch.allclose(gram, torch.eye(5, dtype=torch.float64), atol=2e-5)


def test_real_spherical_harmonics_have_integral_normalization():
    basis = SphericalFieldBasis(n_radial=2, lmax=2, cutoff=3.0)
    vectors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

    harmonics = basis.spherical_harmonics(vectors)

    assert torch.allclose(
        harmonics[:, 0],
        torch.full((1,), 1.0 / math.sqrt(4.0 * math.pi), dtype=torch.float64),
    )


def test_nonscalar_harmonics_vanish_at_zero_displacement():
    basis = SphericalFieldBasis(n_radial=2, lmax=2, cutoff=3.0)

    harmonics = basis.spherical_harmonics(torch.zeros((1, 3), dtype=torch.float64))

    assert harmonics[0, 1:].equal(torch.zeros(8, dtype=torch.float64))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_radial": 0, "lmax": 2, "cutoff": 4.0}, "n_radial"),
        ({"n_radial": 3, "lmax": -1, "cutoff": 4.0}, "lmax"),
        ({"n_radial": 3, "lmax": 2, "cutoff": float("nan")}, "cutoff"),
        ({"n_radial": 3, "lmax": 2, "cutoff": 0.0}, "cutoff"),
        (
            {"n_radial": 3, "lmax": 2, "cutoff": 4.0, "chunk_size": 0},
            "chunk_size",
        ),
    ],
)
def test_basis_rejects_improper_numerical_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SphericalFieldBasis(**kwargs)


def test_basis_rejects_out_of_range_angular_slice():
    basis = SphericalFieldBasis(n_radial=3, lmax=2, cutoff=4.0)

    with pytest.raises(ValueError, match="angular order"):
        basis.l_slice(3)


def test_nonscalar_basis_is_regular_with_bounded_gradient_at_atom_center():
    basis = SphericalFieldBasis(n_radial=3, lmax=2, cutoff=3.0)
    radii = torch.tensor([0.0, 1e-6, 2e-6], dtype=torch.float64)
    vectors = torch.zeros((3, 3), dtype=torch.float64)
    vectors[:, 0] = radii
    vectors.requires_grad_()

    l1_values = basis.evaluate(vectors)[:, 0, basis.l_slice(1)].sum(dim=-1)
    gradient = torch.autograd.grad(l1_values.sum(), vectors)[0]

    assert l1_values[0] == 0.0
    assert abs(l1_values[1]) < 1e-5
    assert torch.allclose(l1_values[2], 2.0 * l1_values[1], atol=1e-10, rtol=2e-5)
    assert torch.isfinite(gradient).all()
    assert gradient.norm(dim=-1).max() < 10.0


def test_radial_basis_and_gradient_vanish_continuously_at_cutoff():
    cutoff = 3.0
    basis = SphericalFieldBasis(n_radial=3, lmax=2, cutoff=cutoff)
    radii = torch.tensor(
        [cutoff - 1e-4, cutoff, cutoff + 1e-4],
        dtype=torch.float64,
        requires_grad=True,
    )

    values = basis.radial_values(radii, order=0)[:, 0]
    gradient = torch.autograd.grad(values.sum(), radii)[0]

    assert abs(values[0]) < 1e-6
    assert values[1:].equal(torch.zeros(2, dtype=torch.float64))
    assert abs(gradient[0]) < 1e-3
    assert gradient[1:].equal(torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize("order", [1, 2])
def test_radial_order_has_required_power_l_origin_behavior(order):
    basis = SphericalFieldBasis(n_radial=3, lmax=2, cutoff=3.0)
    radii = torch.tensor([0.0, 1e-5, 2e-5], dtype=torch.float64)

    values = basis.radial_values(radii, order=order)
    scaled = values[1:] / radii[1:, None] ** order

    assert values[0].equal(torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(scaled[0], scaled[1], atol=1e-5, rtol=1e-5)
