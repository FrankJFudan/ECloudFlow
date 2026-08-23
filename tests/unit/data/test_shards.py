"""Tests for atomic sharding and exact distributed sample coverage."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ecloudflow.core.types import ComplexSample, ElectronField
from ecloudflow.data.manifest import DatasetManifest
from ecloudflow.data.parsers import build_complex_sample
from ecloudflow.data.shards import (
    ShardWriter,
    bucketed_batches,
    sample_ids_for_partition,
    stream_samples,
)


def _samples(fixture_dir: Path, count: int) -> list[ComplexSample]:
    """Build lightweight canonical samples with distinct stable identifiers."""
    template = build_complex_sample(
        fixture_dir / "complex/toy_pocket.pdb",
        fixture_dir / "complex/toy_ligand.sdf",
        sample_id="template",
        build_fields=False,
    )
    return [replace(template, source_id=f"sample-{index}") for index in range(count)]


def test_atomic_shards_round_trip_manifest_and_tensors(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """Canonical immutable samples survive atomic serialization exactly."""
    samples = _samples(fixture_dir, 4)
    field = ElectronField(
        positions=samples[0].pocket.positions[:2],
        values=torch.ones((2, 2), dtype=samples[0].pocket.positions.dtype),
        mask=torch.ones(2, dtype=torch.bool),
        batch=torch.zeros(2, dtype=torch.long),
        channel_names=("density", "esp"),
        frame=samples[0].frame,
    )
    samples[0] = replace(
        samples[0],
        pocket_field=field,
        properties={"affinity": torch.tensor(-7.5)},
    )
    manifest = ShardWriter(max_samples_per_shard=2).write(samples, tmp_path)
    assert manifest.sample_ids == tuple(sample.source_id for sample in samples)
    assert len(manifest.shards) == 2
    assert manifest.hash.startswith("sha256:")
    assert all(record.sha256.startswith("sha256:") for record in manifest.shards)
    assert not list(tmp_path.glob("*.partial"))
    restored_manifest = DatasetManifest.read(tmp_path / "manifest.json")
    assert restored_manifest == manifest
    restored = list(stream_samples(manifest.shard_paths(tmp_path)))
    assert [sample.source_id for sample in restored] == list(manifest.sample_ids)
    assert torch.equal(restored[0].pocket.features, samples[0].pocket.features)
    assert torch.equal(
        restored[0].ligand.halfedge_index, samples[0].ligand.halfedge_index
    )
    assert dict(restored[0].provenance.file_hashes) == dict(
        samples[0].provenance.file_hashes
    )
    assert restored[0].pocket_field is not None
    assert torch.equal(restored[0].pocket_field.values, field.values)
    assert torch.equal(restored[0].properties["affinity"], torch.tensor(-7.5))


def test_webdataset_rank_worker_partition_has_exact_coverage(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """Rank partitioning followed by worker partitioning covers each ID once."""
    manifest = ShardWriter(max_samples_per_shard=3).write(
        _samples(fixture_dir, 10), tmp_path
    )
    paths = manifest.shard_paths(tmp_path)
    seen: list[str] = []
    for rank in range(2):
        for worker in range(2):
            seen.extend(sample_ids_for_partition(paths, rank, 2, worker, 2))
    assert sorted(seen) == [f"sample-{index}" for index in range(10)]
    assert len(seen) == len(set(seen))


def test_skips_are_stable_and_shuffle_and_buckets_are_deterministic(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """Failed inputs are recorded while bounded ordering remains reproducible."""
    samples = _samples(fixture_dir, 6)
    manifest = ShardWriter(max_samples_per_shard=6).write(
        [samples[0], object(), *samples[1:]],  # type: ignore[list-item]
        tmp_path,
    )
    assert len(manifest.skips) == 1
    assert manifest.skips[0].sample_id == "unknown"
    paths = manifest.shard_paths(tmp_path)
    first = [s.source_id for s in stream_samples(paths, seed=4, shuffle_buffer=3)]
    second = [s.source_id for s in stream_samples(paths, seed=4, shuffle_buffer=3)]
    assert first == second
    batches = list(
        bucketed_batches(stream_samples(paths), batch_size=2, bucket_width=8)
    )
    assert [len(batch) for batch in batches] == [2, 2, 2]
    with pytest.raises(ValueError, match="rank"):
        list(stream_samples(paths, rank=2, world_size=2))
