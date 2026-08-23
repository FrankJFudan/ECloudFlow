"""Atomic WebDataset-compatible shards and distributed streaming."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import tarfile
import uuid
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


class DuplicateSampleError(ValueError):
    """Raise when a reserved candidate identifier appears more than once."""


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
        """Build or resume one content-verified immutable dataset generation.

        :param samples: Deterministic stream of canonical complexes. Pocket,
            ligand, and optional field coordinates use ``[N, 3]`` local-frame
            angstrom tensors; graph tensors, dtypes, devices, batch indices,
            chemical masks, and fragment fixed masks are preserved exactly.
        :param output_dir: Dataset root containing ``.staging``, immutable
            ``generations/<id>`` directories, and the current ``manifest.json``.
        :param split: Optional frozen leakage-controlled sample/entity split.
            Grouped assignments must cover exactly every successfully serialized
            sample; ``None`` publishes explicit unpartitioned mode.
        :return: Fully validated manifest for the newly published or recovered
            generation, including hashes, skips, provenance, and split audit.
        :rtype: DatasetManifest
        :raises ShardWriteError: If resume input changes, grouped coverage is
            incomplete, a staged/promoted shard fails size/hash validation, or
            tar, checkpoint, promotion, synchronization, or publication fails.
        :raises RuntimeError: If the input iterable itself aborts preprocessing.

        Publication is a recoverable state machine: ``STAGING`` journals only
        durable candidate/shard prefixes; ``READY`` adds a self-contained
        publication descriptor; ``PROMOTED`` atomically renames that directory
        to its immutable generation path; ``PUBLISHED`` atomically replaces the
        top-level manifest before clearing the active marker. Recovery validates
        the descriptor and all shard sizes/hashes in either READY or PROMOTED
        location and finishes publication before touching ``samples``. Thus a
        crash immediately before promotion, in the rename-to-manifest window,
        or after manifest replacement is idempotent and never reserializes a
        completed generation; the previous manifest always references unchanged
        shards until the replacement is durable.

        Candidate IDs are reserved before type validation/serialization. Failed
        candidates become bounded typed skips, and a duplicate can never replace
        the first occurrence. Anonymous invalid objects are recorded but do not
        create a reusable synthetic ID. Serialization performs filesystem I/O
        only, does not mutate input samples, move tensors between CPU/GPU, change
        coordinate frames/units, or apply distributed partitioning.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_fingerprint = _generation_config_fingerprint(
            self.target_size_bytes,
            self.max_samples_per_shard,
            self.preprocessing_version,
            split,
        )
        recovered = _recover_ready_generation(output_dir)
        if recovered is not None:
            return recovered
        generation_id, stage_dir, journal = _open_generation(
            output_dir, config_fingerprint
        )
        completed_candidates = list(journal.get("candidates", []))
        staged_shards = tuple(
            _shard_record_from_dict(record) for record in journal.get("shards", [])
        )
        _validate_completed_shards(stage_dir, staged_shards)
        sample_ids: list[str] = []
        source_hashes: dict[str, str] = {}
        skips: list[SkipRecord] = []
        shard_records: list[ShardRecord] = list(staged_shards)
        pending: list[tuple[ComplexSample, bytes]] = []
        pending_bytes = 0
        seen_ids: set[str] = set()
        candidate_records: list[dict[str, str]] = []
        replay_index = 0

        def flush() -> None:
            nonlocal pending, pending_bytes
            if not pending:
                return
            record = self._write_shard(pending, stage_dir, len(shard_records))
            shard_records.append(record)
            pending = []
            pending_bytes = 0
            _write_generation_journal(
                stage_dir,
                config_fingerprint,
                candidate_records,
                shard_records,
            )

        for candidate in samples:
            sample, encoded, skip, candidate_record = _process_candidate(
                candidate, seen_ids
            )
            if replay_index < len(completed_candidates):
                if candidate_record != completed_candidates[replay_index]:
                    raise ShardWriteError(
                        "resumed candidate stream differs from the staged generation"
                    )
                candidate_records.append(candidate_record)
                replay_index += 1
                if skip is not None:
                    skips.append(skip)
                elif sample is not None:
                    _append_sample_metadata(sample, sample_ids, source_hashes)
                continue
            if skip is not None:
                candidate_records.append(candidate_record)
                skips.append(skip)
                continue
            if sample is None or encoded is None:
                raise ShardWriteError("candidate processing returned no outcome")
            if pending and (pending_bytes + len(encoded) > self.target_size_bytes):
                # The current candidate is deliberately excluded from the
                # checkpoint until its payload belongs to a durable shard.
                flush()
            candidate_records.append(candidate_record)
            pending.append((sample, encoded))
            pending_bytes += len(encoded)
            _append_sample_metadata(sample, sample_ids, source_hashes)
            if (
                self.max_samples_per_shard is not None
                and len(pending) >= self.max_samples_per_shard
            ):
                flush()
        if replay_index != len(completed_candidates):
            raise ShardWriteError(
                "resumed candidate stream ended before staged records"
            )
        flush()
        _write_generation_journal(
            stage_dir,
            config_fingerprint,
            candidate_records,
            shard_records,
        )
        if split is not None and set(split.sample_partitions) != set(sample_ids):
            raise ShardWriteError(
                "grouped split assignments must exactly match serialized sample IDs"
            )
        metadata = split.to_metadata() if split is not None else {}
        published_shards = tuple(
            ShardRecord(
                path=(Path("generations") / generation_id / record.path).as_posix(),
                sha256=record.sha256,
                size_bytes=record.size_bytes,
                sample_ids=record.sample_ids,
            )
            for record in shard_records
        )
        manifest = DatasetManifest(
            sample_ids=tuple(sample_ids),
            shards=published_shards,
            source_hashes=dict(sorted(source_hashes.items())),
            skips=tuple(skips),
            preprocessing_version=self.preprocessing_version,
            generation_id=generation_id,
            partition_mode="grouped" if split is not None else "unpartitioned",
            split_hash=metadata.get("hash"),
            sample_partitions=metadata.get("sample_partitions", {}),
            sample_groups=metadata.get("sample_groups", {}),
            entity_partitions=metadata.get("entity_partitions", {}),
            entity_groups=metadata.get("entity_groups", {}),
            split_audit=split.audit if split is not None else None,
        )
        _write_publication_descriptor(manifest, stage_dir)
        _set_active_generation(
            output_dir, generation_id, config_fingerprint, state="ready"
        )
        generation_dir = output_dir / "generations" / generation_id
        _promote_generation(stage_dir, generation_dir)
        _set_active_generation(
            output_dir, generation_id, config_fingerprint, state="promoted"
        )
        _publish_dataset_manifest(manifest, output_dir)
        _set_active_generation(
            output_dir, generation_id, config_fingerprint, state="published"
        )
        _clear_active_generation(output_dir, generation_id)
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


def _recover_ready_generation(output_dir: Path) -> DatasetManifest | None:
    """Finish a durable READY/PROMOTED/PUBLISHED generation without replay.

    :param output_dir: Dataset root containing the active marker and generation.
    :return: Recovered published manifest, or ``None`` when no complete
        publication descriptor exists and ordinary STAGING must continue.
    :rtype: DatasetManifest | None
    :raises ShardWriteError: If a descriptor, generation identity, shard path,
        byte size, digest, promotion, or final manifest publication is invalid.

    The active marker is a CPU/filesystem control record only. A descriptor in
    ``.staging/<id>`` denotes READY; the same descriptor in
    ``generations/<id>`` denotes PROMOTED. Every shard is re-hashed before any
    top-level manifest replacement. An already matching manifest denotes
    PUBLISHED and only marker cleanup remains. The function never opens the
    caller's sample iterable, deserializes tensors, changes device/dtype/frame,
    or mutates an older immutable generation.
    """
    active_path = output_dir / ".staging" / "active.json"
    if not active_path.is_file():
        return None
    try:
        active = json.loads(active_path.read_text(encoding="utf-8"))
        generation_id = str(active["generation_id"])
        config_fingerprint = str(active["config_fingerprint"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    if not _portable_generation_id(generation_id) or not _valid_sha256(
        config_fingerprint
    ):
        return None

    stage_dir = output_dir / ".staging" / generation_id
    generation_dir = output_dir / "generations" / generation_id
    staged_descriptor = stage_dir / "publication.json"
    promoted_descriptor = generation_dir / "publication.json"
    if promoted_descriptor.is_file():
        descriptor = promoted_descriptor
        shard_root = generation_dir
        promoted = True
    elif staged_descriptor.is_file():
        descriptor = staged_descriptor
        shard_root = stage_dir
        promoted = False
    else:
        return None

    try:
        manifest = DatasetManifest.read(descriptor)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ShardWriteError(
            f"invalid generation publication descriptor: {error}"
        ) from error
    if manifest.generation_id != generation_id:
        raise ShardWriteError("publication descriptor generation ID mismatch")
    _validate_publication_shards(manifest, shard_root, generation_id)

    if not promoted:
        _set_active_generation(
            output_dir, generation_id, config_fingerprint, state="ready"
        )
        _promote_generation(stage_dir, generation_dir)
    _set_active_generation(
        output_dir, generation_id, config_fingerprint, state="promoted"
    )

    published_path = output_dir / "manifest.json"
    already_published = False
    if published_path.is_file():
        try:
            already_published = (
                DatasetManifest.read(published_path).hash == manifest.hash
            )
        except (OSError, ValueError, TypeError, KeyError):
            already_published = False
    if not already_published:
        _publish_dataset_manifest(manifest, output_dir)
    _set_active_generation(
        output_dir, generation_id, config_fingerprint, state="published"
    )
    _clear_active_generation(output_dir, generation_id)
    return manifest


def _write_publication_descriptor(manifest: DatasetManifest, stage_dir: Path) -> None:
    """Persist a self-contained recovery descriptor before directory promotion."""
    try:
        manifest.write(stage_dir / "publication.json")
    except OSError as error:
        raise ShardWriteError(
            f"failed to checkpoint generation publication: {error}"
        ) from error


def _promote_generation(stage_dir: Path, generation_dir: Path) -> None:
    """Atomically rename a READY stage into its immutable generation path."""
    generation_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(stage_dir, generation_dir)
    except OSError as error:
        raise ShardWriteError(
            f"failed to promote dataset generation: {error}"
        ) from error


def _publish_dataset_manifest(manifest: DatasetManifest, output_dir: Path) -> None:
    """Atomically make one promoted generation the current dataset."""
    try:
        manifest.write(output_dir / "manifest.json")
    except OSError as error:
        raise ShardWriteError(
            f"failed to finalize dataset manifest: {error}"
        ) from error


def _set_active_generation(
    output_dir: Path,
    generation_id: str,
    config_fingerprint: str,
    *,
    state: str,
) -> None:
    """Atomically checkpoint one publication state-machine transition."""
    if state not in {"staging", "ready", "promoted", "published"}:
        raise ValueError("unknown generation publication state")
    _write_json_atomic(
        output_dir / ".staging" / "active.json",
        {
            "generation_id": generation_id,
            "config_fingerprint": config_fingerprint,
            "state": state,
        },
    )


def _clear_active_generation(output_dir: Path, generation_id: str) -> None:
    """Remove the matching active marker after manifest publication."""
    active_path = output_dir / ".staging" / "active.json"
    try:
        active = json.loads(active_path.read_text(encoding="utf-8"))
        if active.get("generation_id") == generation_id:
            active_path.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return


def _validate_publication_shards(
    manifest: DatasetManifest, shard_root: Path, generation_id: str
) -> None:
    """Verify descriptor paths, byte sizes, and hashes before publication."""
    expected_parent = Path("generations") / generation_id
    for record in manifest.shards:
        relative = Path(record.path)
        if relative.parent != expected_parent:
            raise ShardWriteError(
                f"publication shard path escaped its generation: {record.path}"
            )
        path = shard_root / relative.name
        if (
            not path.is_file()
            or path.stat().st_size != record.size_bytes
            or "sha256:" + _sha256_file(path) != record.sha256
        ):
            raise ShardWriteError(f"publication shard validation failed: {record.path}")


def _portable_generation_id(value: str) -> bool:
    """Return whether an active generation ID is one safe path component."""
    return bool(value) and all(
        character.isalnum() or character in "-_" for character in value
    )


def _generation_config_fingerprint(
    target_size_bytes: int,
    max_samples_per_shard: int | None,
    preprocessing_version: str,
    split: GroupedSplit | None,
) -> str:
    """Hash settings that must remain identical across a resumed generation."""
    payload = {
        "target_size_bytes": target_size_bytes,
        "max_samples_per_shard": max_samples_per_shard,
        "preprocessing_version": preprocessing_version,
        "split_hash": split.hash if split is not None else None,
    }
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _open_generation(
    output_dir: Path, config_fingerprint: str
) -> tuple[str, Path, dict[str, Any]]:
    """Resume a compatible staged generation or create a new isolated stage."""
    staging_root = output_dir / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    active_path = staging_root / "active.json"
    if active_path.is_file():
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
            generation_id = str(active["generation_id"])
            stage_dir = staging_root / generation_id
            if (
                active.get("config_fingerprint") == config_fingerprint
                and stage_dir.is_dir()
            ):
                journal_path = stage_dir / "journal.json"
                journal = (
                    json.loads(journal_path.read_text(encoding="utf-8"))
                    if journal_path.is_file()
                    else {"candidates": [], "shards": []}
                )
                if journal.get("config_fingerprint") not in (
                    None,
                    config_fingerprint,
                ):
                    raise ShardWriteError(
                        "staged generation configuration fingerprint mismatch"
                    )
                return generation_id, stage_dir, journal
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            pass
    generation_id = uuid.uuid4().hex
    stage_dir = staging_root / generation_id
    stage_dir.mkdir(parents=False, exist_ok=False)
    _set_active_generation(
        output_dir,
        generation_id,
        config_fingerprint,
        state="staging",
    )
    journal = {
        "config_fingerprint": config_fingerprint,
        "candidates": [],
        "shards": [],
    }
    _write_json_atomic(stage_dir / "journal.json", journal)
    return generation_id, stage_dir, journal


def _write_generation_journal(
    stage_dir: Path,
    config_fingerprint: str,
    candidates: list[dict[str, str]],
    shards: list[ShardRecord],
) -> None:
    """Atomically checkpoint completed candidates and validated staged shards."""
    _write_json_atomic(
        stage_dir / "journal.json",
        {
            "config_fingerprint": config_fingerprint,
            "candidates": candidates,
            "shards": [_shard_record_to_dict(record) for record in shards],
        },
    )


def _write_json_atomic(path: Path, values: Mapping[str, Any]) -> None:
    """Write one fsynced JSON control file through an atomic replacement."""
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(values, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _shard_record_to_dict(record: ShardRecord) -> dict[str, Any]:
    """Convert a staged shard record to journal primitives."""
    return {
        "path": record.path,
        "sha256": record.sha256,
        "size_bytes": record.size_bytes,
        "sample_ids": list(record.sample_ids),
    }


def _shard_record_from_dict(values: Mapping[str, Any]) -> ShardRecord:
    """Reconstruct a staged shard record from journal primitives."""
    return ShardRecord(
        path=str(values["path"]),
        sha256=str(values["sha256"]),
        size_bytes=int(values["size_bytes"]),
        sample_ids=tuple(str(value) for value in values["sample_ids"]),
    )


def _validate_completed_shards(
    stage_dir: Path, shards: tuple[ShardRecord, ...]
) -> None:
    """Verify every journaled staged shard before accepting resume progress."""
    for shard in shards:
        path = stage_dir / shard.path
        if (
            not path.is_file()
            or path.stat().st_size != shard.size_bytes
            or "sha256:" + _sha256_file(path) != shard.sha256
        ):
            raise ShardWriteError(f"staged shard validation failed: {shard.path}")


def _process_candidate(
    candidate: object, seen_ids: set[str]
) -> tuple[
    ComplexSample | None,
    bytes | None,
    SkipRecord | None,
    dict[str, str],
]:
    """Reserve an ID before serialization and return a journalable outcome."""
    try:
        raw_sample_id = getattr(candidate, "source_id", None)
    # Candidate-supplied descriptors may raise arbitrary exceptions; isolation
    # converts them into an anonymous skip without aborting the dataset build.
    except Exception as error:  # noqa: BLE001
        skip = _skip_record("anonymous", error)
        return (
            None,
            None,
            skip,
            {
                "sample_id": skip.sample_id,
                "status": "skip",
                "category": skip.category,
                "message": skip.message,
            },
        )
    sample_id = (
        raw_sample_id
        if isinstance(raw_sample_id, str) and raw_sample_id.strip()
        else None
    )
    try:
        if sample_id is not None:
            if sample_id in seen_ids:
                raise DuplicateSampleError(
                    "duplicate sample identifier was already reserved"
                )
            seen_ids.add(sample_id)
        if not isinstance(candidate, ComplexSample):
            raise TypeError("record is not a ComplexSample")
        if sample_id is None:
            raise ValueError("ComplexSample source_id is not a usable identifier")
        encoded = _serialize_sample(candidate)
    except (TypeError, ValueError, RuntimeError, OSError) as error:
        skip = _skip_record(sample_id or "anonymous", error)
        return (
            None,
            None,
            skip,
            {
                "sample_id": skip.sample_id,
                "status": "skip",
                "category": skip.category,
                "message": skip.message,
            },
        )
    payload_hash = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return (
        candidate,
        encoded,
        None,
        {
            "sample_id": candidate.source_id,
            "status": "success",
            "payload_hash": payload_hash,
        },
    )


def _append_sample_metadata(
    sample: ComplexSample,
    sample_ids: list[str],
    source_hashes: dict[str, str],
) -> None:
    """Append one accepted sample ID and its role-qualified source hashes."""
    sample_ids.append(sample.source_id)
    for role, digest in sample.provenance.file_hashes.items():
        source_hashes[f"{sample.source_id}:{role}"] = digest


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
    shard_sample_ids: Mapping[str, tuple[str, ...]] | None = None,
    cache_dir: str | Path | None = None,
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
    :param shard_sample_ids: Optional manifest member index keyed by path name;
        required for split filtering without decoding non-owned members.
    :param cache_dir: Optional portable local read-through cache directory. Cache
        use requires expected hashes and verifies every hit and fill.
    :return: Lazy iterator of validated canonical samples whose coordinate
        tensors retain ``[N, 3]`` local binding-frame angstrom values, dtype,
        device, graph/field masks, chemical attributes, and provenance.
    :rtype: Iterator[ComplexSample]
    :raises ValueError: If rank/worker arguments or the shuffle size are invalid.
    :raises ShardReadError: If a shard hash, tar member, or sample payload fails.

    When shards outnumber consumers, whole-shard ownership is assigned rank
    first and worker second so non-owners never open the tar. Sparse shard sets
    use manifest/member indices to choose ownership before extraction and
    deserialization. Filtering occurs before within-shard ordinals, preserving
    exact once-only coverage without N-times sample decoding. Path order and
    the seed/rank/worker tuple completely determine output order. Hash-verified
    cache fills mutate only ``cache_dir`` through fsynced atomic replacement;
    shards and reconstructed samples are never modified or moved across devices.
    """
    _validate_partition(rank, world_size, worker_id, num_workers, shuffle_buffer)
    ordered_paths = sorted((Path(path) for path in paths), key=lambda path: path.name)
    consumer_count = world_size * num_workers
    if cache_dir is not None and expected_hashes is None:
        raise ValueError("cache_dir requires expected_hashes for verified reads")
    if allowed_sample_ids is not None and shard_sample_ids is None:
        raise ValueError("allowed_sample_ids requires shard_sample_ids")

    def selected() -> Iterator[ComplexSample]:
        ordinal = 0
        whole_shard_ownership = len(ordered_paths) >= consumer_count
        for shard_index, source_path in enumerate(ordered_paths):
            if whole_shard_ownership and (
                shard_index % world_size != rank
                or (shard_index // world_size) % num_workers != worker_id
            ):
                continue
            expected = _lookup_by_path(expected_hashes, source_path)
            if expected_hashes is not None and expected is None:
                raise ShardReadError(f"missing expected shard hash: {source_path.name}")
            path = _resolve_cached_shard(source_path, cache_dir, expected)
            if expected is not None and "sha256:" + _sha256_file(path) != expected:
                raise ShardReadError(f"shard hash mismatch: {source_path.name}")
            indexed_ids = _lookup_by_path(shard_sample_ids, source_path)
            try:
                with tarfile.open(path, "r:*") as archive:
                    member_index = 0
                    for member in archive:
                        if not member.isfile() or not member.name.endswith(
                            ".sample.pt"
                        ):
                            continue
                        indexed_id = (
                            indexed_ids[member_index]
                            if indexed_ids is not None
                            and member_index < len(indexed_ids)
                            else None
                        )
                        member_index += 1
                        if indexed_ids is not None and member_index > len(indexed_ids):
                            raise ShardReadError(
                                f"manifest member index is shorter than {source_path.name}"
                            )
                        if (
                            allowed_sample_ids is not None
                            and indexed_id not in allowed_sample_ids
                        ):
                            continue
                        if not whole_shard_ownership:
                            global_ordinal = ordinal
                            ordinal += 1
                            if global_ordinal % world_size != rank:
                                continue
                            rank_ordinal = global_ordinal // world_size
                            if rank_ordinal % num_workers != worker_id:
                                continue
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise ShardReadError(
                                f"cannot read tar member {member.name}"
                            )
                        sample = _deserialize_sample(extracted.read())
                        if indexed_id is not None and sample.source_id != indexed_id:
                            raise ShardReadError(
                                f"manifest member ID mismatch in {source_path.name}"
                            )
                        yield sample
                    if indexed_ids is not None and member_index != len(indexed_ids):
                        raise ShardReadError(
                            f"manifest member index length mismatch: {source_path.name}"
                        )
            except (OSError, tarfile.TarError) as error:
                raise ShardReadError(
                    f"failed to stream shard {path}: {error}"
                ) from error

    yield from _buffered_shuffle(
        selected(), shuffle_buffer, seed + rank * 1_000_003 + worker_id
    )


def _lookup_by_path(values: Mapping[str, Any] | None, path: Path) -> Any | None:
    """Find manifest metadata by absolute string, POSIX string, or basename."""
    if values is None:
        return None
    for key in (str(path), path.as_posix(), path.name):
        if key in values:
            return values[key]
    return None


def _resolve_cached_shard(
    source_path: Path,
    cache_dir: str | Path | None,
    expected_hash: str | None,
) -> Path:
    """Resolve one shard through a content-addressed verified read-through cache.

    :param source_path: Immutable source tar path used on a cache miss/corruption.
    :param cache_dir: Optional caller-chosen local cache root; ``None`` disables
        all cache filesystem mutation and returns ``source_path`` directly.
    :param expected_hash: Canonical ``sha256:<hex>`` content identity. It is
        mandatory when caching so untrusted names never become cache paths.
    :return: Source path when caching is disabled, otherwise a verified cached tar.
    :rtype: pathlib.Path
    :raises ShardReadError: If the hash is absent/malformed, source/cache I/O
        fails, or copied bytes do not match the expected digest.

    A valid cache hit never opens the source. Corrupt hits are replaced from a
    uniquely named partial file after copy, ``fsync``, and SHA-256 verification.
    Concurrent workers may race safely because every successful replacement has
    identical content. No tar member is decoded and no tensor dtype, device,
    coordinate frame, units, or mask is changed here.
    """
    if cache_dir is None:
        return source_path
    if expected_hash is None or not _valid_sha256(expected_hash):
        raise ShardReadError(f"missing cache verification hash: {source_path.name}")
    destination_root = Path(cache_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{expected_hash.removeprefix('sha256:')}.tar"
    if destination.is_file() and "sha256:" + _sha256_file(destination) == expected_hash:
        return destination
    partial = destination.with_suffix(f".{uuid.uuid4().hex}.partial")
    try:
        with source_path.open("rb") as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if "sha256:" + _sha256_file(partial) != expected_hash:
            raise ShardReadError(f"source shard hash mismatch: {source_path.name}")
        os.replace(partial, destination)
    except OSError as error:
        raise ShardReadError(
            f"failed to fill shard cache: {source_path.name}"
        ) from error
    finally:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


def _valid_sha256(value: str) -> bool:
    """Validate a cache-key digest before using it as a filename."""
    if not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError:
        return False
    return True


def sample_ids_for_partition(
    paths: Iterable[str | Path],
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
) -> list[str]:
    """Return IDs selected by the canonical rank/worker partition.

    :param paths: Immutable tar shard paths in arbitrary input order.
    :param rank: Distributed rank in ``[0, world_size)``.
    :param world_size: Positive distributed rank count.
    :param worker_id: Rank-local worker index in ``[0, num_workers)``.
    :param num_workers: Positive rank-local worker count.
    :return: Storage-ordered selected sample IDs.
    :rtype: list[str]
    :raises ValueError: If rank/world-size or worker/count bounds are invalid.
    :raises ShardReadError: If any owned shard or canonical payload is invalid.

    This audit helper applies the same rank-first, worker-second ownership as
    :func:`stream_samples` with shuffling disabled. Across all valid consumers,
    each sample ID appears exactly once. It performs CPU shard decoding but does
    not mutate files/samples or change tensor dtype, device, frame, units, or masks.
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
    in sorted key order at end of epoch. No Cartesian coordinates, features,
    bond/fragment masks, dtype, device, local binding frame, or angstrom units
    are inspected or changed; only bounded lists of sample references mutate.
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
    except Exception as error:
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
