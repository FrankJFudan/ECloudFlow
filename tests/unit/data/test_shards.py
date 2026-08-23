"""Tests for atomic sharding and exact distributed sample coverage."""

import hashlib
import io
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ecloudflow.core.types import ComplexSample, ElectronField
from ecloudflow.data.manifest import DatasetManifest
from ecloudflow.data.parsers import build_complex_sample
from ecloudflow.data.shards import (
    ShardReadError,
    ShardWriteError,
    ShardWriter,
    bucketed_batches,
    sample_ids_for_partition,
    stream_samples,
)
from ecloudflow.data.splits import build_grouped_split


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
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rank partitioning followed by worker partitioning covers each ID once."""
    manifest = ShardWriter(max_samples_per_shard=3).write(
        _samples(fixture_dir, 10), tmp_path
    )
    paths = manifest.shard_paths(tmp_path)
    import ecloudflow.data.shards as shard_module

    decode_count = 0
    original_decode = shard_module._deserialize_sample

    def counting_decode(payload: bytes) -> ComplexSample:
        nonlocal decode_count
        decode_count += 1
        return original_decode(payload)

    monkeypatch.setattr(shard_module, "_deserialize_sample", counting_decode)
    seen: list[str] = []
    for rank in range(2):
        for worker in range(2):
            seen.extend(sample_ids_for_partition(paths, rank, 2, worker, 2))
    assert sorted(seen) == [f"sample-{index}" for index in range(10)]
    assert len(seen) == len(set(seen))
    assert decode_count == 10


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


def test_interrupted_generation_resumes_and_old_manifest_stays_valid(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """Only a complete immutable generation may replace the dataset manifest."""
    writer = ShardWriter(max_samples_per_shard=2)
    old_manifest = writer.write(_samples(fixture_dir, 2), tmp_path)
    old_manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    old_shards = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in old_manifest.shard_paths(tmp_path)
    }
    replacement = [
        replace(sample, source_id=f"replacement-{index}")
        for index, sample in enumerate(_samples(fixture_dir, 6))
    ]

    def interrupted():
        for index, sample in enumerate(replacement):
            if index == 3:
                raise RuntimeError("simulated interruption")
            yield sample

    with pytest.raises(RuntimeError, match="simulated interruption"):
        writer.write(interrupted(), tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == old_manifest_bytes
    resumed = writer.write(replacement, tmp_path)
    assert resumed.generation_id != old_manifest.generation_id
    assert resumed.sample_ids == tuple(sample.source_id for sample in replacement)
    for path, digest in old_shards.items():
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_size_boundary_checkpoint_never_claims_an_unwritten_candidate(
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume reprocesses the candidate that triggered a pre-append flush."""
    samples = _samples(fixture_dir, 3)
    writer = ShardWriter()
    writer.target_size_bytes = 1
    import ecloudflow.data.shards as shard_module

    original = shard_module._write_generation_journal
    interrupted = False

    def fail_after_first_checkpoint(*args, **kwargs) -> None:
        nonlocal interrupted
        original(*args, **kwargs)
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated checkpoint interruption")

    monkeypatch.setattr(
        shard_module, "_write_generation_journal", fail_after_first_checkpoint
    )
    with pytest.raises(RuntimeError, match="checkpoint interruption"):
        writer.write(samples, tmp_path)

    manifest = writer.write(samples, tmp_path)
    assert manifest.sample_ids == tuple(sample.source_id for sample in samples)
    assert [
        sample.source_id for sample in stream_samples(manifest.shard_paths(tmp_path))
    ] == [sample.source_id for sample in samples]


def test_failed_identifier_is_reserved_before_duplicate_serialization(
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later duplicate cannot replace the first candidate after its failure."""
    samples = _samples(fixture_dir, 2)
    samples[1] = replace(samples[1], source_id=samples[0].source_id)
    import ecloudflow.data.shards as shard_module

    original = shard_module._serialize_sample
    calls = 0

    def fail_first(sample: ComplexSample) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic serialization failure")
        return original(sample)

    monkeypatch.setattr(shard_module, "_serialize_sample", fail_first)
    manifest = ShardWriter().write(samples, tmp_path)
    assert manifest.sample_ids == ()
    assert [skip.category for skip in manifest.skips] == [
        "RuntimeError",
        "DuplicateSampleError",
    ]
    assert calls == 1


def test_verified_local_cache_hit_refill_and_corruption_failure(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """Cache hits survive source loss, while corrupt hits are safely refilled."""
    dataset_dir, cache_dir = tmp_path / "dataset", tmp_path / "cache"
    manifest = ShardWriter().write(_samples(fixture_dir, 2), dataset_dir)
    source = manifest.shard_paths(dataset_dir)[0]
    source_bytes = source.read_bytes()
    expected = {source.name: manifest.shards[0].sha256}
    ids = {source.name: manifest.shards[0].sample_ids}
    assert (
        len(
            list(
                stream_samples(
                    [source],
                    expected_hashes=expected,
                    shard_sample_ids=ids,
                    cache_dir=cache_dir,
                )
            )
        )
        == 2
    )
    cached = next(cache_dir.glob("*.tar"))
    source.write_bytes(b"corrupt source")
    assert (
        len(
            list(
                stream_samples(
                    [source],
                    expected_hashes=expected,
                    shard_sample_ids=ids,
                    cache_dir=cache_dir,
                )
            )
        )
        == 2
    )
    source.write_bytes(source_bytes)
    cached.write_bytes(b"corrupt cache")
    assert (
        len(
            list(
                stream_samples(
                    [source],
                    expected_hashes=expected,
                    shard_sample_ids=ids,
                    cache_dir=cache_dir,
                )
            )
        )
        == 2
    )
    source.write_bytes(b"corrupt source again")
    cached.write_bytes(b"corrupt cache again")
    with pytest.raises(ShardReadError, match="source shard hash mismatch"):
        list(
            stream_samples(
                [source],
                expected_hashes=expected,
                shard_sample_ids=ids,
                cache_dir=cache_dir,
            )
        )


def test_corrupt_restricted_payload_is_wrapped_as_shard_read_error(
    tmp_path: Path,
) -> None:
    """Restricted-load and unpickling failures never escape as raw exceptions."""
    path = tmp_path / "corrupt.tar"
    payload = b"not a torch archive"
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo("00000000.sample.pt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ShardReadError, match="invalid canonical sample payload"):
        list(stream_samples([path]))


def test_grouped_manifest_round_trips_complete_audit_and_exact_coverage(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """Grouping methods and canonical split inputs remain recoverable from JSON."""
    samples = _samples(fixture_dir, 2)
    records = [
        {
            "sample_id": sample.source_id,
            "source_identifier": f"pdbbind:{sample.source_id}",
            "protein_id": f"protein-{index}",
            "sequence_cluster": f"cluster-{index}",
            "ligand_id": f"ligand-{index}",
            "ligand_group": f"scaffold-{index}",
        }
        for index, sample in enumerate(samples)
    ]
    split = build_grouped_split(records, seed=31)
    manifest = ShardWriter().write(samples, tmp_path, split=split)
    restored = DatasetManifest.read(tmp_path / "manifest.json")
    assert restored.hash == manifest.hash
    assert restored.sample_groups == split.sample_groups
    assert restored.split_audit == split.audit
    assert restored.split_audit is not None
    assert restored.split_audit.source_identifiers["sample-0"] == "pdbbind:sample-0"
    extra_split = build_grouped_split(
        [
            *records,
            {
                "sample_id": "extra",
                "protein_id": "protein-extra",
                "sequence_cluster": "cluster-extra",
                "ligand_id": "ligand-extra",
                "ligand_group": "scaffold-extra",
            },
        ]
    )
    with pytest.raises(ShardWriteError, match="exactly match"):
        ShardWriter().write(samples, tmp_path / "bad-split", split=extra_split)
