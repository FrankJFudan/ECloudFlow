"""Versioned, content-addressed dataset manifest records."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ShardRecord:
    """Describe one finalized immutable tar shard.

    :param path: Manifest-relative tar path.
    :param sha256: File digest prefixed by ``"sha256:"``.
    :param size_bytes: Final tar size in bytes.
    :param sample_ids: Stable sample identifiers stored in the shard.
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
        if not self.sha256.startswith("sha256:") or len(self.sha256) != 71:
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
    :param split_hash: Optional grouped-split fingerprint.
    :param sample_partitions: Production sample-to-partition lookup.
    :param entity_partitions: Entity lookup retained for leakage auditing.
    :param entity_groups: Entity-to-connected-group lookup for auditing.
    :return: Immutable, JSON-serializable manifest.
    :rtype: DatasetManifest
    """

    sample_ids: tuple[str, ...]
    shards: tuple[ShardRecord, ...]
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    skips: tuple[SkipRecord, ...] = ()
    preprocessing_version: str = "1"
    split_hash: str | None = None
    sample_partitions: Mapping[str, str] = field(default_factory=dict)
    entity_partitions: Mapping[str, str] = field(default_factory=dict)
    entity_groups: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate cross-record coverage and freeze manifest mappings."""
        if not self.preprocessing_version:
            raise ValueError("preprocessing_version must be non-empty")
        if len(self.sample_ids) != len(set(self.sample_ids)) or any(
            not sample_id for sample_id in self.sample_ids
        ):
            raise ValueError("manifest sample_ids must be unique and non-empty")
        shard_ids = tuple(
            sample_id for shard in self.shards for sample_id in shard.sample_ids
        )
        if shard_ids != self.sample_ids:
            raise ValueError("manifest sample_ids must exactly match ordered shard IDs")
        if self.split_hash is not None and not self.split_hash.startswith("sha256:"):
            raise ValueError("split_hash must use a sha256 prefix")
        for name in (
            "source_hashes",
            "sample_partitions",
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

    @property
    def hash(self) -> str:
        """Return a stable SHA-256 fingerprint of the manifest payload."""
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
            "split_hash": self.split_hash,
            "sample_partitions": dict(self.sample_partitions),
            "entity_partitions": dict(self.entity_partitions),
            "entity_groups": dict(self.entity_groups),
        }
        if include_hash:
            data["hash"] = self.hash
        return data

    def write(self, path: str | Path) -> None:
        """Atomically write and fsync this manifest as stable JSON.

        :param path: Final manifest path in an existing or new directory.
        :return: None.
        :rtype: None
        :raises OSError: If bytes cannot be written, synchronized, or renamed.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        encoded = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)

    @classmethod
    def read(cls, path: str | Path) -> DatasetManifest:
        """Read and verify a manifest emitted by :meth:`write`.

        :param path: Existing JSON manifest path.
        :return: Reconstructed typed manifest.
        :rtype: DatasetManifest
        :raises ValueError: If the stored manifest fingerprint is inconsistent.
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
        manifest = cls(**data)
        if expected_hash is not None and manifest.hash != expected_hash:
            raise ValueError("dataset manifest hash does not match its content")
        return manifest

    def shard_paths(self, base_dir: str | Path) -> tuple[Path, ...]:
        """Resolve all recorded shard paths relative to ``base_dir``."""
        root = Path(base_dir)
        return tuple(root / shard.path for shard in self.shards)
