"""Behavior tests for pocket encoding and atom-count prediction."""

import pytest
import torch

from ecloudflow.core.types import PocketGraph
from ecloudflow.models import AtomCountPredictor, PocketEncoder


def _proper_rotation() -> torch.Tensor:
    axis = torch.tensor([1.0, 2.0, -1.0])
    axis = axis / axis.norm()
    angle = torch.tensor(0.71)
    cross = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return (
        torch.eye(3)
        + torch.sin(angle) * cross
        + (1.0 - torch.cos(angle)) * (cross @ cross)
    )


def _pocket(feature_dim: int = 7) -> PocketGraph:
    return PocketGraph(
        positions=torch.tensor(
            [[-1.0, 0.2, 0.0], [0.4, 1.1, -0.3], [1.2, -0.7, 0.8], [0.0, 0.0, 1.5]]
        ),
        features=torch.arange(4 * feature_dim, dtype=torch.float32).reshape(
            4, feature_dim
        )
        / 10,
        batch=torch.tensor([0, 0, 0, 0]),
    )


@pytest.mark.parametrize("feature_dim", [7, 50])
def test_pocket_encoder_accepts_declared_feature_layout_without_fixture_assumption(
    feature_dim: int,
) -> None:
    """Mutation caught: hard-coding Task 6's width 50 rejects other schemas."""
    encoder = PocketEncoder(scalar_dim=16, vector_dim=4, num_blocks=2)

    encoded = encoder.encode(_pocket(feature_dim))

    assert encoded.scalars.shape == (4, 16)
    assert encoded.vectors.shape == (4, 4, 3)
    assert encoded.feature_dim == feature_dim


def test_pocket_encoder_is_se3_equivariant_and_cache_key_is_content_stable() -> None:
    """Mutation caught: absolute coordinates leak translation into scalar features."""
    torch.manual_seed(3)
    encoder = PocketEncoder(scalar_dim=16, vector_dim=4, num_blocks=2).eval()
    pocket = _pocket()
    rotation = _proper_rotation()
    translation = torch.tensor([2.0, -1.0, 0.5])
    transformed = PocketGraph(
        positions=pocket.positions @ rotation.T + translation,
        features=pocket.features.clone(),
        batch=pocket.batch.clone(),
    )

    encoded = encoder.encode(pocket)
    cloned = encoder.encode(
        PocketGraph(
            pocket.positions.clone(), pocket.features.clone(), pocket.batch.clone()
        )
    )
    moved = encoder.encode(transformed)

    assert encoded.cache_key == cloned.cache_key
    assert torch.allclose(moved.scalars, encoded.scalars, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        moved.vectors, encoded.vectors @ rotation.T, atol=2e-5, rtol=2e-5
    )


def test_null_pocket_encoding_is_real_and_distinct_from_conditioned_encoding() -> None:
    """Mutation caught: ignoring the classifier-free branch makes both encodings equal."""
    torch.manual_seed(4)
    encoder = PocketEncoder(scalar_dim=16, vector_dim=4, num_blocks=1).eval()
    pocket = _pocket()

    conditioned = encoder.encode(pocket)
    null = encoder.encode(pocket, use_null=True)

    assert null.is_null
    assert null.cache_key != conditioned.cache_key
    assert not torch.allclose(null.scalars, conditioned.scalars)
    assert torch.count_nonzero(null.vectors) == 0


def test_pocket_cache_key_supports_bfloat16_without_dtype_conversion() -> None:
    """Mutation caught: NumPy conversion of BF16 cache content raises instead of hashing."""
    encoder = PocketEncoder(scalar_dim=16, vector_dim=4, num_blocks=1).to(
        dtype=torch.bfloat16
    )
    pocket = _pocket()
    pocket = PocketGraph(
        positions=pocket.positions.to(torch.bfloat16),
        features=pocket.features.to(torch.bfloat16),
        batch=pocket.batch,
    )

    encoded = encoder.encode(pocket)

    assert encoded.scalars.dtype == torch.bfloat16
    assert len(encoded.cache_key) == 64


def test_atom_count_distribution_has_exact_fragment_lower_bounds_per_complex() -> None:
    """Mutation caught: one global bound allows too few atoms in one batch item."""
    torch.manual_seed(5)
    predictor = AtomCountPredictor(scalar_dim=8, max_atoms=6)
    pooled = torch.randn(2, 8)
    fixed_counts = torch.tensor([2, 4])

    distribution = predictor(pooled, fixed_counts)

    assert distribution.probs.shape == (2, 7)
    assert torch.equal(distribution.probs[0, :2], torch.zeros(2))
    assert torch.equal(distribution.probs[1, :4], torch.zeros(4))
    assert torch.all(distribution.probs[0, 2:] > 0)
    assert torch.all(distribution.probs[1, 4:] > 0)


def test_atom_count_predictor_rejects_impossible_fragment_size() -> None:
    """Mutation caught: clipping an oversized fixed count silently violates conditioning."""
    predictor = AtomCountPredictor(scalar_dim=8, max_atoms=3)

    with pytest.raises(ValueError, match="fixed fragment count"):
        predictor(torch.zeros(1, 8), torch.tensor([4]))
