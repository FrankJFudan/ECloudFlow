"""Atomic WebDataset-compatible shards and distributed streaming."""

from __future__ import annotations

import hashlib
import io
import os
import random
import tarfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

from ecloudflow.core.frames import CoordinateFrame
from ecloudflow.core.types import (
    ComplexSample,
    ElectronField,
    FragmentCondition,
    LigandGraph,
    MolecularState,
    PocketGraph,
    QMProvenance,
    SampleProvenance,
)
from ecloudflow.data.manifest import (
    DatasetManifest,
    ShardRecord,
    SkipRecord,
)
from ecloudflow.data.splits import GroupedSplit

_GIB = 1024**3
_SERIALIZATION_VERSION = 1


class ShardWriteError(RuntimeError):
    """Raise when a shard cannot be finalized by atomic rename."""


class ShardReadError(RuntimeError):
    """Raise when a shard or canonical sample payload is invalid."""


class ShardWriter:
    """Serialize validated canonical samples into atomic tar shards.

    :param target_shard_size_gb: Target size between 0.5 and 2 GiB.
    :param max_samples_per_shard: Optional deterministic test/operations bound;
        production sizing is primarily byte based.
    :param preprocessing_version: Version embedded in the dataset manifest.
    :return: Stateful writer configuration reusable across datasets.
    :rtype: ShardWriter

    Individual samples are encoded before they enter a shard. A serialization
    failure is recorded as a skip and cannot corrupt or renumber an already
    accepted sample. Each tar is written under ``.partial``, fsynced, hashed,
    and atomically renamed before the manifest is published.
    """

    def __init__(
        self,
        *,
        target_shard_size_gb: float = 1.0,
        max_samples_per_shard: int | None = None,
        preprocessing_version: str = "1",
    ) -> None:
        if not 0.5 <= target_shard_size_gb <= 2.0:
            raise ValueError("target_shard_size_gb must lie in [0.5, 2.0]")
        if max_samples_per_shard is not None and max_samples_per_shard < 1:
            raise ValueError("max_samples_per_shard must be positive when supplied")
        if not preprocessing_version:
            raise ValueError("preprocessing_version must be non-empty")
        self.target_size_bytes = int(target_shard_size_gb * _GIB)
        self.max_samples_per_shard = max_samples_per_shard
        self.preprocessing_version = preprocessing_version

    def write(
        self,
        samples: Iterable[ComplexSample],
        output_dir: Path,
        *,
        split: GroupedSplit | None = None,
    ) -> DatasetManifest:
        """Serialize validated samples into atomic WebDataset tar shards.

        :param samples: Stream of canonical complex samples.
        :param output_dir: Destination containing shards and ``manifest.json``.
        :param split: Optional leakage-controlled sample/entity split metadata.
        :return: Manifest with sample IDs, source hashes, shard hashes, skips,
            preprocessing version, and split metadata.
        :rtype: DatasetManifest
        :raises ShardWriteError: If tar creation, synchronization, hashing, or
            atomic finalization fails.

        Shards are first written with a ``.partial`` suffix, hashed, fsynced,
        and renamed. Failed samples are recorded and never replaced by another
        sample. Distributed rank/worker partitioning is a read-time operation.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_ids: list[str] = []
        source_hashes: dict[str, str] = {}
        skips: list[SkipRecord] = []
        shard_records: list[ShardRecord] = []
        pending: list[tuple[ComplexSample, bytes]] = []
        pending_bytes = 0
        seen_ids: set[str] = set()

        def flush() -> None:
            nonlocal pending, pending_bytes
            if not pending:
                return
            record = self._write_shard(pending, output_dir, len(shard_records))
            shard_records.append(record)
            pending = []
            pending_bytes = 0

        for candidate in samples:
            sample_id = getattr(candidate, "source_id", "unknown")
            try:
                if not isinstance(candidate, ComplexSample):
                    raise TypeError("record is not a ComplexSample")
                if candidate.source_id in seen_ids:
                    raise ValueError("duplicate sample identifier")
                encoded = _serialize_sample(candidate)
            except (TypeError, ValueError, RuntimeError, OSError) as error:
                skips.append(_skip_record(str(sample_id), error))
                continue
            if pending and (
                pending_bytes + len(encoded) > self.target_size_bytes
                or (
                    self.max_samples_per_shard is not None
                    and len(pending) >= self.max_samples_per_shard
                )
            ):
                flush()
            pending.append((candidate, encoded))
            pending_bytes += len(encoded)
            seen_ids.add(candidate.source_id)
            sample_ids.append(candidate.source_id)
            for role, digest in candidate.provenance.file_hashes.items():
                source_hashes[f"{candidate.source_id}:{role}"] = digest
        flush()
        metadata = split.to_metadata() if split is not None else {}
        manifest = DatasetManifest(
            sample_ids=tuple(sample_ids),
            shards=tuple(shard_records),
            source_hashes=dict(sorted(source_hashes.items())),
            skips=tuple(skips),
            preprocessing_version=self.preprocessing_version,
            split_hash=metadata.get("hash"),
            sample_partitions=metadata.get("sample_partitions", {}),
            entity_partitions=metadata.get("entity_partitions", {}),
            entity_groups=metadata.get("entity_groups", {}),
        )
        try:
            manifest.write(output_dir / "manifest.json")
        except OSError as error:
            raise ShardWriteError(
                f"failed to finalize dataset manifest: {error}"
            ) from error
        return manifest

    def _write_shard(
        self,
        samples: list[tuple[ComplexSample, bytes]],
        output_dir: Path,
        shard_index: int,
    ) -> ShardRecord:
        """Write, synchronize, hash, and atomically publish one tar shard."""
        target = output_dir / f"shard-{shard_index:06d}.tar"
        partial = target.with_suffix(".tar.partial")
        try:
            with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as archive:
                for member_index, (sample, payload) in enumerate(samples):
                    member = tarfile.TarInfo(f"{member_index:08d}.sample.pt")
                    member.size = len(payload)
                    member.mtime = 0
                    member.uid = member.gid = 0
                    member.uname = member.gname = ""
                    archive.addfile(member, io.BytesIO(payload))
            # Windows requires a writable descriptor for ``fsync``.
            with partial.open("r+b") as stream:
                os.fsync(stream.fileno())
            digest = _sha256_file(partial)
            size_bytes = partial.stat().st_size
            os.replace(partial, target)
        except (OSError, tarfile.TarError) as error:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            raise ShardWriteError(
                f"failed to atomically write shard {shard_index}: {error}"
            ) from error
        return ShardRecord(
            path=target.name,
            sha256="sha256:" + digest,
            size_bytes=size_bytes,
            sample_ids=tuple(sample.source_id for sample, _ in samples),
        )


def stream_samples(
    paths: Iterable[str | Path],
    *,
    rank: int = 0,
    world_size: int = 1,
    worker_id: int = 0,
    num_workers: int = 1,
    seed: int = 0,
    shuffle_buffer: int = 0,
    allowed_sample_ids: set[str] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
) -> Iterator[ComplexSample]:
    """Stream canonical samples after rank-then-worker partitioning.

    :param paths: Tar shard paths in any input order.
    :param rank: Distributed rank in ``[0, world_size)``.
    :param world_size: Number of distributed ranks.
    :param worker_id: DataLoader worker in ``[0, num_workers)``.
    :param num_workers: Workers participating within this rank.
    :param seed: Deterministic bounded-shuffle seed.
    :param shuffle_buffer: Maximum samples held for shuffle; zero or one keeps
        storage order.
    :param allowed_sample_ids: Optional production sample-level split filter.
    :param expected_hashes: Optional expected SHA-256 values keyed by path name.
    :return: Lazy iterator of validated canonical samples.
    :rtype: Iterator[ComplexSample]
    :raises ValueError: If rank/worker arguments or the shuffle size are invalid.
    :raises ShardReadError: If a shard hash, tar member, or sample payload fails.

    Filtering occurs before the global ordinal is assigned. The ordinal is
    first modulo-partitioned across ranks; the rank-local ordinal is then
    modulo-partitioned across workers. This order guarantees exact once-only
    coverage for every selected sample without loading a shard into memory.
    """
    _validate_partition(rank, world_size, worker_id, num_workers, shuffle_buffer)
    ordered_paths = sorted((Path(path) for path in paths), key=lambda path: path.name)

    def selected() -> Iterator[ComplexSample]:
        ordinal = 0
        for path in ordered_paths:
            if expected_hashes is not None:
                expected = expected_hashes.get(path.name)
                actual = "sha256:" + _sha256_file(path)
                if expected is None or actual != expected:
                    raise ShardReadError(f"shard hash mismatch: {path.name}")
            try:
                with tarfile.open(path, "r:*") as archive:
                    for member in archive:
                        if not member.isfile() or not member.name.endswith(
                            ".sample.pt"
                        ):
                            continue
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise ShardReadError(
                                f"cannot read tar member {member.name}"
                            )
                        sample = _deserialize_sample(extracted.read())
                        if (
                            allowed_sample_ids is not None
                            and sample.source_id not in allowed_sample_ids
                        ):
                            continue
                        global_ordinal = ordinal
                        ordinal += 1
                        if global_ordinal % world_size != rank:
                            continue
                        rank_ordinal = global_ordinal // world_size
                        if rank_ordinal % num_workers == worker_id:
                            yield sample
            except (OSError, tarfile.TarError) as error:
                raise ShardReadError(
                    f"failed to stream shard {path}: {error}"
                ) from error

    yield from _buffered_shuffle(
        selected(), shuffle_buffer, seed + rank * 1_000_003 + worker_id
    )


def sample_ids_for_partition(
    paths: Iterable[str | Path],
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
) -> list[str]:
    """Return IDs selected by the canonical rank/worker partition.

    :param paths: Tar shard paths.
    :param rank: Distributed rank.
    :param world_size: Distributed rank count.
    :param worker_id: Rank-local worker index.
    :param num_workers: Rank-local worker count.
    :return: Storage-ordered selected sample IDs.
    :rtype: list[str]
    """
    return [
        sample.source_id
        for sample in stream_samples(
            paths,
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            num_workers=num_workers,
        )
    ]


def bucketed_batches(
    samples: Iterable[ComplexSample],
    *,
    batch_size: int,
    bucket_width: int = 32,
) -> Iterator[list[ComplexSample]]:
    """Group samples by pocket and ligand node-count buckets.

    :param samples: Canonical sample stream after rank/worker partitioning.
    :param batch_size: Maximum samples per yielded batch.
    :param bucket_width: Node-count width for both pocket and ligand axes.
    :return: Deterministic iterator of non-empty sample lists.
    :rtype: Iterator[list[ComplexSample]]
    :raises ValueError: If ``batch_size`` or ``bucket_width`` is not positive.

    The two-dimensional bucket key limits padding variance on large pockets
    while keeping all pending storage bounded by one partial batch per observed
    bucket. Full buckets are emitted immediately; partial buckets are drained
    in sorted key order at end of epoch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if bucket_width < 1:
        raise ValueError("bucket_width must be positive")
    buckets: dict[tuple[int, int], list[ComplexSample]] = {}
    for sample in samples:
        key = (
            sample.pocket.positions.shape[0] // bucket_width,
            sample.ligand.positions.shape[0] // bucket_width,
        )
        bucket = buckets.setdefault(key, [])
        bucket.append(sample)
        if len(bucket) == batch_size:
            yield list(bucket)
            del buckets[key]
    for key in sorted(buckets):
        yield buckets[key]


def _buffered_shuffle(
    samples: Iterable[ComplexSample], buffer_size: int, seed: int
) -> Iterator[ComplexSample]:
    """Shuffle a stream deterministically with bounded resident storage."""
    if buffer_size <= 1:
        yield from samples
        return
    random_generator = random.Random(seed)
    buffer: list[ComplexSample] = []
    for sample in samples:
        buffer.append(sample)
        if len(buffer) >= buffer_size:
            yield buffer.pop(random_generator.randrange(len(buffer)))
    while buffer:
        yield buffer.pop(random_generator.randrange(len(buffer)))


def _validate_partition(
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
    shuffle_buffer: int,
) -> None:
    """Validate distributed streaming coordinates."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank must lie in [0, world_size)")
    if num_workers < 1 or not 0 <= worker_id < num_workers:
        raise ValueError("worker_id must lie in [0, num_workers)")
    if shuffle_buffer < 0:
        raise ValueError("shuffle_buffer must be nonnegative")


def _skip_record(sample_id: str, error: Exception) -> SkipRecord:
    """Create a bounded single-line skip record without raw environment data."""
    message = " ".join(str(error).split())[:240] or type(error).__name__
    return SkipRecord(
        sample_id=sample_id or "unknown",
        category=type(error).__name__,
        message=message,
    )


def _sha256_file(path: Path) -> str:
    """Hash a file incrementally for multi-gigabyte shard support."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serialize_sample(sample: ComplexSample) -> bytes:
    """Encode an immutable canonical sample into safe primitive payloads."""
    stream = io.BytesIO()
    torch.save(_sample_to_payload(sample), stream)
    return stream.getvalue()


def _deserialize_sample(payload: bytes) -> ComplexSample:
    """Decode and revalidate one canonical sample payload."""
    try:
        values = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if values.get("serialization_version") != _SERIALIZATION_VERSION:
            raise ValueError("unsupported sample serialization version")
        return _sample_from_payload(values)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ShardReadError(f"invalid canonical sample payload: {error}") from error


def _tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    """Detach tensors onto CPU before persistent dataset serialization."""
    return value.detach().cpu() if value is not None else None


def _frame_to_payload(frame: CoordinateFrame | None) -> dict[str, Any] | None:
    """Convert a coordinate frame to tensor primitives."""
    if frame is None:
        return None
    return {"origin": _tensor(frame.origin), "rotation": _tensor(frame.rotation)}


def _frame_from_payload(values: Mapping[str, Any] | None) -> CoordinateFrame | None:
    """Reconstruct and validate a coordinate frame from primitives."""
    if values is None:
        return None
    return CoordinateFrame(origin=values["origin"], rotation=values["rotation"])


def _field_to_payload(field: ElectronField | None) -> dict[str, Any] | None:
    """Convert an optional electron field to primitive tensors."""
    if field is None:
        return None
    return {
        "positions": _tensor(field.positions),
        "values": _tensor(field.values),
        "mask": _tensor(field.mask),
        "batch": _tensor(field.batch),
        "channel_names": tuple(field.channel_names),
        "frame": _frame_to_payload(field.frame),
    }


def _field_from_payload(values: Mapping[str, Any] | None) -> ElectronField | None:
    """Reconstruct and validate an optional electron field."""
    if values is None:
        return None
    return ElectronField(
        positions=values["positions"],
        values=values["values"],
        mask=values["mask"],
        batch=values["batch"],
        channel_names=tuple(values["channel_names"]),
        frame=_frame_from_payload(values["frame"]),
    )


def _state_to_payload(state: MolecularState) -> dict[str, Any]:
    """Convert a molecular state to primitive tensors."""
    return {
        name: _tensor(getattr(state, name))
        for name in (
            "positions",
            "atom_logits",
            "charge_logits",
            "halfedge_index",
            "bond_logits",
            "electron_latent",
            "node_batch",
            "halfedge_batch",
        )
    } | {"frame": _frame_to_payload(state.frame)}


def _state_from_payload(values: Mapping[str, Any]) -> MolecularState:
    """Reconstruct and validate a molecular state."""
    return MolecularState(
        positions=values["positions"],
        atom_logits=values["atom_logits"],
        charge_logits=values["charge_logits"],
        halfedge_index=values["halfedge_index"],
        bond_logits=values["bond_logits"],
        electron_latent=values["electron_latent"],
        node_batch=values["node_batch"],
        halfedge_batch=values["halfedge_batch"],
        frame=_frame_from_payload(values["frame"]),
    )


def _fragment_to_payload(fragment: FragmentCondition | None) -> dict[str, Any] | None:
    """Convert an optional exact fragment condition to primitives."""
    if fragment is None:
        return None
    return {
        "reference": _state_to_payload(fragment.reference),
        "fixed_atom_mask": _tensor(fragment.fixed_atom_mask),
        "fixed_bond_mask": _tensor(fragment.fixed_bond_mask),
        "fixed_coord_mask": _tensor(fragment.fixed_coord_mask),
        "attachment_mask": _tensor(fragment.attachment_mask),
        "component_ids": _tensor(fragment.component_ids),
        "task_id": fragment.task_id,
    }


def _fragment_from_payload(
    values: Mapping[str, Any] | None,
) -> FragmentCondition | None:
    """Reconstruct and validate an optional exact fragment condition."""
    if values is None:
        return None
    return FragmentCondition(
        reference=_state_from_payload(values["reference"]),
        fixed_atom_mask=values["fixed_atom_mask"],
        fixed_bond_mask=values["fixed_bond_mask"],
        fixed_coord_mask=values["fixed_coord_mask"],
        attachment_mask=values["attachment_mask"],
        component_ids=values["component_ids"],
        task_id=values["task_id"],
    )


def _qm_to_payload(qm: QMProvenance | None) -> dict[str, Any] | None:
    """Convert optional QM provenance to credential-free primitives."""
    if qm is None:
        return None
    return {
        "status": qm.status,
        "qm_mask": qm.qm_mask,
        "tool": qm.tool,
        "version": qm.version,
        "executable": qm.executable,
        "command": tuple(qm.command),
        "charge": qm.charge,
        "multiplicity": qm.multiplicity,
        "failure_category": qm.failure_category,
        "source_hashes": dict(qm.source_hashes),
        "integrated_electron_count": qm.integrated_electron_count,
    }


def _qm_from_payload(values: Mapping[str, Any] | None) -> QMProvenance | None:
    """Reconstruct and validate optional QM provenance."""
    if values is None:
        return None
    return QMProvenance(
        status=values["status"],
        qm_mask=values["qm_mask"],
        tool=values["tool"],
        version=values["version"],
        executable=values["executable"],
        command=tuple(values["command"]),
        charge=values["charge"],
        multiplicity=values["multiplicity"],
        failure_category=values["failure_category"],
        source_hashes=dict(values["source_hashes"]),
        integrated_electron_count=values["integrated_electron_count"],
    )


def _sample_to_payload(sample: ComplexSample) -> dict[str, Any]:
    """Convert every canonical sample field to safe tensor primitives."""
    properties = {
        key: _tensor(value) if isinstance(value, torch.Tensor) else value
        for key, value in sample.properties.items()
    }
    return {
        "serialization_version": _SERIALIZATION_VERSION,
        "source_id": sample.source_id,
        "pocket": {
            "positions": _tensor(sample.pocket.positions),
            "features": _tensor(sample.pocket.features),
            "batch": _tensor(sample.pocket.batch),
            "atom_numbers": _tensor(sample.pocket.atom_numbers),
            "frame": _frame_to_payload(sample.pocket.frame),
        },
        "ligand": {
            "positions": _tensor(sample.ligand.positions),
            "atom_types": _tensor(sample.ligand.atom_types),
            "formal_charges": _tensor(sample.ligand.formal_charges),
            "halfedge_index": _tensor(sample.ligand.halfedge_index),
            "bond_types": _tensor(sample.ligand.bond_types),
            "batch": _tensor(sample.ligand.batch),
        },
        "pocket_field": _field_to_payload(sample.pocket_field),
        "ligand_field": _field_to_payload(sample.ligand_field),
        "properties": properties,
        "frame": _frame_to_payload(sample.frame),
        "provenance": {
            "source_paths": dict(sample.provenance.source_paths),
            "file_hashes": dict(sample.provenance.file_hashes),
            "tool_versions": dict(sample.provenance.tool_versions),
            "preprocessing_status": sample.provenance.preprocessing_status,
            "original_ligand_positions": _tensor(
                sample.provenance.original_ligand_positions
            ),
            "qm": _qm_to_payload(sample.provenance.qm),
        },
        "fragment": _fragment_to_payload(sample.fragment),
    }


def _sample_from_payload(values: Mapping[str, Any]) -> ComplexSample:
    """Reconstruct all canonical contracts, triggering their validators."""
    pocket_values = values["pocket"]
    ligand_values = values["ligand"]
    provenance_values = values["provenance"]
    pocket = PocketGraph(
        positions=pocket_values["positions"],
        features=pocket_values["features"],
        batch=pocket_values["batch"],
        atom_numbers=pocket_values["atom_numbers"],
        frame=_frame_from_payload(pocket_values["frame"]),
    )
    ligand = LigandGraph(
        positions=ligand_values["positions"],
        atom_types=ligand_values["atom_types"],
        formal_charges=ligand_values["formal_charges"],
        halfedge_index=ligand_values["halfedge_index"],
        bond_types=ligand_values["bond_types"],
        batch=ligand_values["batch"],
    )
    provenance = SampleProvenance(
        source_paths=dict(provenance_values["source_paths"]),
        file_hashes=dict(provenance_values["file_hashes"]),
        tool_versions=dict(provenance_values["tool_versions"]),
        preprocessing_status=provenance_values["preprocessing_status"],
        original_ligand_positions=provenance_values["original_ligand_positions"],
        qm=_qm_from_payload(provenance_values["qm"]),
    )
    frame = _frame_from_payload(values["frame"])
    if frame is None:
        raise ValueError("complex sample frame is absent")
    return ComplexSample(
        source_id=values["source_id"],
        pocket=pocket,
        ligand=ligand,
        pocket_field=_field_from_payload(values["pocket_field"]),
        ligand_field=_field_from_payload(values["ligand_field"]),
        properties=dict(values["properties"]),
        frame=frame,
        provenance=provenance,
        fragment=_fragment_from_payload(values["fragment"]),
    )
