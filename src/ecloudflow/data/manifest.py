"""Versioned, content-addressed dataset manifest records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ecloudflow.data import durability
from ecloudflow.data.splits import SplitAudit


@dataclass(frozen=True)
class ShardRecord:
    """Describe one finalized immutable tar shard.

    :param path: Manifest-relative tar path.
    :param sha256: File digest prefixed by ``"sha256:"``.
    :param size_bytes: Final tar size in bytes.
    :param sample_ids: Stable sample identifiers stored in the shard.
    :return: Immutable content-addressed shard descriptor.
    :rtype: ShardRecord
    """

    path: str
    sha256: str
    size_bytes: int
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate a relative path, digest, size, and unique sample IDs."""
        candidate = Path(self.path)
        if (
            not self.path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.suffix != ".tar"
        ):
            raise ValueError("shard path must be a safe relative tar path")
        if not _is_sha256(self.sha256):
            raise ValueError("shard sha256 must be a prefixed hexadecimal digest")
        if self.size_bytes <= 0:
            raise ValueError("shard size_bytes must be positive")
        if not self.sample_ids or len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("shard sample_ids must be non-empty and unique")


@dataclass(frozen=True)
class SkipRecord:
    """Describe one source record that was not written.

    :param sample_id: Known source ID or ``"unknown"``.
    :param category: Stable exception category without sensitive text.
    :param message: Short sanitized diagnostic.
    :return: Immutable bounded preprocessing failure descriptor.
    :rtype: SkipRecord
    """

    sample_id: str
    category: str
    message: str

    def __post_init__(self) -> None:
        """Require bounded non-empty skip diagnostics."""
        if not self.sample_id or not self.category or not self.message:
            raise ValueError("skip record fields must be non-empty")
        if len(self.message) > 240 or "\n" in self.message or "\r" in self.message:
            raise ValueError("skip record message must be bounded and single-line")


@dataclass(frozen=True)
class DatasetManifest:
    """Index atomic shards, source hashes, skips, and split metadata.

    :param sample_ids: Ordered identifiers of successfully serialized samples.
    :param shards: Finalized tar shard records.
    :param source_hashes: Source hashes keyed by ``"sample_id:role"``.
    :param skips: Typed failures that were never replaced by another sample.
    :param preprocessing_version: Schema/preprocessing version string.
    :param generation_id: Immutable dataset-generation directory identifier.
    :param partition_mode: Explicit ``"grouped"`` or ``"unpartitioned"`` mode.
    :param split_hash: Optional grouped-split fingerprint.
    :param sample_partitions: Production sample-to-partition lookup.
    :param sample_groups: Production sample-to-leakage-group lookup.
    :param entity_partitions: Entity lookup retained for leakage auditing.
    :param entity_groups: Entity-to-connected-group lookup for auditing.
    :param split_audit: Recoverable grouping algorithm and input metadata.
    :return: Immutable, JSON-serializable manifest.
    :rtype: DatasetManifest
    """

    sample_ids: tuple[str, ...]
    shards: tuple[ShardRecord, ...]
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    skips: tuple[SkipRecord, ...] = ()
    preprocessing_version: str = "1"
    generation_id: str = "legacy"
    partition_mode: str = "unpartitioned"
    split_hash: str | None = None
    sample_partitions: Mapping[str, str] = field(default_factory=dict)
    sample_groups: Mapping[str, str] = field(default_factory=dict)
    entity_partitions: Mapping[str, str] = field(default_factory=dict)
    entity_groups: Mapping[str, str] = field(default_factory=dict)
    split_audit: SplitAudit | None = None

    def __post_init__(self) -> None:
        """Validate cross-record coverage and freeze manifest mappings."""
        if not self.preprocessing_version:
            raise ValueError("preprocessing_version must be non-empty")
        if not self.generation_id or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in self.generation_id
        ):
            raise ValueError("generation_id must be a portable path component")
        if self.partition_mode not in {"grouped", "unpartitioned"}:
            raise ValueError("partition_mode must be grouped or unpartitioned")
        if len(self.sample_ids) != len(set(self.sample_ids)) or any(
            not sample_id for sample_id in self.sample_ids
        ):
            raise ValueError("manifest sample_ids must be unique and non-empty")
        shard_ids = tuple(
            sample_id for shard in self.shards for sample_id in shard.sample_ids
        )
        if shard_ids != self.sample_ids:
            raise ValueError("manifest sample_ids must exactly match ordered shard IDs")
        if self.split_hash is not None and not _is_sha256(self.split_hash):
            raise ValueError("split_hash must use a sha256 prefix")
        for name in (
            "source_hashes",
            "sample_partitions",
            "sample_groups",
            "entity_partitions",
            "entity_groups",
        ):
            values = dict(getattr(self, name))
            if any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or not value
                for key, value in values.items()
            ):
                raise ValueError(f"{name} must contain non-empty string pairs")
            object.__setattr__(self, name, MappingProxyType(values))
        sample_id_set = set(self.sample_ids)
        if self.partition_mode == "grouped":
            if set(self.sample_partitions) != sample_id_set:
                raise ValueError(
                    "grouped sample_partitions must exactly cover serialized sample IDs"
                )
            if set(self.sample_groups) != sample_id_set:
                raise ValueError(
                    "grouped sample_groups must exactly cover serialized sample IDs"
                )
            if self.split_hash is None or self.split_audit is None:
                raise ValueError(
                    "grouped manifests require split hash and audit metadata"
                )
            if set(self.split_audit.input_hashes) != sample_id_set:
                raise ValueError(
                    "split audit inputs must exactly cover serialized sample IDs"
                )
        elif any(
            (
                self.sample_partitions,
                self.sample_groups,
                self.entity_partitions,
                self.entity_groups,
                self.split_hash,
                self.split_audit,
            )
        ):
            raise ValueError("unpartitioned manifests must not contain split metadata")

    @property
    def hash(self) -> str:
        """Return a stable SHA-256 fingerprint of the manifest payload.

        :return: Lowercase hexadecimal digest prefixed by ``"sha256:"``.
        :rtype: str

        The hash covers generation paths, shard/source hashes, skips, explicit
        partition mode, all sample/entity groups, and recoverable split audit
        metadata. It excludes only its own derived hash field.
        """
        payload = json.dumps(
            self.to_dict(include_hash=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        """Convert typed records to stable JSON-compatible primitives.

        :param include_hash: Whether to include the derived manifest hash.
        :return: Mapping suitable for JSON serialization.
        :rtype: dict[str, Any]
        """
        data: dict[str, Any] = {
            "sample_ids": list(self.sample_ids),
            "shards": [
                {
                    "path": shard.path,
                    "sha256": shard.sha256,
                    "size_bytes": shard.size_bytes,
                    "sample_ids": list(shard.sample_ids),
                }
                for shard in self.shards
            ],
            "source_hashes": dict(self.source_hashes),
            "skips": [
                {
                    "sample_id": skip.sample_id,
                    "category": skip.category,
                    "message": skip.message,
                }
                for skip in self.skips
            ],
            "preprocessing_version": self.preprocessing_version,
            "generation_id": self.generation_id,
            "partition_mode": self.partition_mode,
            "split_hash": self.split_hash,
            "sample_partitions": dict(self.sample_partitions),
            "sample_groups": dict(self.sample_groups),
            "entity_partitions": dict(self.entity_partitions),
            "entity_groups": dict(self.entity_groups),
            "split_audit": self.split_audit.to_dict()
            if self.split_audit is not None
            else None,
        }
        if include_hash:
            data["hash"] = self.hash
        return data

    def write(self, path: str | Path) -> None:
        """Publish this immutable dataset descriptor as stable fsynced JSON.

        :param path: Final manifest path; missing parent directories are created.
        :return: None.
        :rtype: None
        :raises OSError: If bytes cannot be written, synchronized, or renamed.

        Canonical JSON includes the derived content hash, ordered sample/shard
        membership, byte sizes, SHA-256 values, split audit metadata, skips, and
        immutable generation paths. Bytes are written to a sibling ``.partial``
        file, flushed and ``fsync``-ed, then atomically replaced. Readers see
        either the previous complete manifest or this complete manifest, never
        a mixed generation after process failure. The destination directory is
        then durably flushed (POSIX ``fsync`` or Windows
        ``FlushFileBuffers``/write-through replacement), so successful return
        also covers the manifest directory entry across power loss. Unsupported
        durability fails explicitly through ``OSError``. The method mutates only
        the destination filesystem; it does not alter this frozen record, shard
        tensors, coordinate frames, devices, dtypes, masks, or source data.
        """
        destination = Path(path)
        durability.durable_mkdir(destination.parent, parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        encoded = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            durability.sync_file(stream)
        durability.durable_replace(temporary, destination)

    @classmethod
    def read(cls, path: str | Path) -> DatasetManifest:
        """Load and fully validate a published immutable dataset descriptor.

        :param path: Existing UTF-8 JSON manifest emitted by :meth:`write`.
        :return: Frozen typed manifest with ordered shards and mapping proxies.
        :rtype: DatasetManifest
        :raises OSError: If the manifest cannot be read from the filesystem.
        :raises json.JSONDecodeError: If the file is not valid JSON.
        :raises KeyError: If a required shard, skip, or audit field is absent.
        :raises TypeError: If stored values cannot reconstruct typed records.
        :raises ValueError: If hashes, paths, partition coverage, generation
            identity, or the stored manifest fingerprint are inconsistent.

        Loading performs no shard decoding, tensor/device transfer, or file
        mutation. Constructor validation requires grouped assignments and audit
        inputs to cover exactly the serialized sample IDs and rejects unsafe
        shard paths before callers resolve them. Canonical JSON and ordered
        records reconstruct deterministically to the same manifest hash.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_hash = data.pop("hash", None)
        data["sample_ids"] = tuple(data.get("sample_ids", ()))
        data["shards"] = tuple(
            ShardRecord(
                path=item["path"],
                sha256=item["sha256"],
                size_bytes=int(item["size_bytes"]),
                sample_ids=tuple(item["sample_ids"]),
            )
            for item in data.get("shards", ())
        )
        data["skips"] = tuple(
            SkipRecord(
                sample_id=item["sample_id"],
                category=item["category"],
                message=item["message"],
            )
            for item in data.get("skips", ())
        )
        if data.get("split_audit") is not None:
            data["split_audit"] = SplitAudit.from_dict(data["split_audit"])
        manifest = cls(**data)
        if expected_hash is not None and manifest.hash != expected_hash:
            raise ValueError("dataset manifest hash does not match its content")
        return manifest

    def shard_paths(self, base_dir: str | Path) -> tuple[Path, ...]:
        """Resolve immutable generation shard paths without filesystem mutation.

        :param base_dir: Dataset root containing versioned generation directories.
        :return: Manifest-ordered absolute or root-relative ``Path`` objects.
        :rtype: tuple[pathlib.Path, ...]

        The method performs no existence checks or directory creation. Callers
        retain manifest order because it defines deterministic sample ordering.
        """
        root = Path(base_dir)
        return tuple(root / shard.path for shard in self.shards)


def _is_sha256(value: str) -> bool:
    """Return whether ``value`` is a canonical prefixed hexadecimal digest."""
    if not value.startswith("sha256:") or len(value) != 71:
        return False
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError:
        return False
    return True
