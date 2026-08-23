"""Lightning DataModule for deterministic sharded ECloudFlow datasets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lightning import LightningDataModule
from torch import distributed
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ecloudflow.config.schema import DataConfig
from ecloudflow.core.types import ComplexSample
from ecloudflow.data.manifest import DatasetManifest
from ecloudflow.data.shards import bucketed_batches, stream_samples
from ecloudflow.exceptions import DataValidationError


class _ShardBatchDataset(IterableDataset[list[ComplexSample]]):
    """Worker-aware iterable over already bucketed canonical batches."""

    def __init__(
        self,
        paths: tuple[Path, ...],
        manifest: DatasetManifest,
        config: DataConfig,
        partition: str,
        epoch: int,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.manifest = manifest
        self.config = config
        self.partition = partition
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[ComplexSample]]:
        """Partition rank first, worker second, then bucket local samples."""
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        if distributed.is_available() and distributed.is_initialized():
            rank, world_size = distributed.get_rank(), distributed.get_world_size()
        else:
            rank, world_size = 0, 1
        allowed_ids: set[str] | None = None
        if self.manifest.sample_partitions and self.partition != "all":
            allowed_ids = {
                sample_id
                for sample_id, assigned in self.manifest.sample_partitions.items()
                if assigned == self.partition
            }
        expected_hashes = None
        if self.config.verify_shard_hashes:
            expected_hashes = {
                record.path: record.sha256 for record in self.manifest.shards
            }
        stream = stream_samples(
            self.paths,
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            num_workers=worker_count,
            seed=self.config.seed + self.epoch,
            shuffle_buffer=self.config.shuffle_buffer,
            allowed_sample_ids=allowed_ids,
            expected_hashes=expected_hashes,
        )
        yield from bucketed_batches(
            stream,
            batch_size=self.config.batch_size,
            bucket_width=self.config.bucket_width,
        )


class ECloudDataModule(LightningDataModule):
    """Expose leakage-safe shards through reproducible Lightning loaders.

    :param config: Strict frozen dataset and worker configuration.
    :return: DataModule whose iterable datasets own distributed partitioning.
    :rtype: ECloudDataModule

    The tar stream assigns samples to distributed rank before DataLoader worker,
    then applies deterministic bounded shuffling and node-count bucketing. The
    loader itself uses ``batch_size=None`` because batches are already formed by
    the iterable dataset. Epoch state is explicit and checkpoint serializable.
    """

    def __init__(self, config: DataConfig) -> None:
        super().__init__()
        self.config = config
        self.epoch = 0
        self.manifest: DatasetManifest | None = None
        self.paths: tuple[Path, ...] = ()

    def setup(self, stage: str | None = None) -> None:
        """Read the immutable manifest and resolve its shard paths.

        :param stage: Optional Lightning stage; shard discovery is stage agnostic.
        :return: None.
        :rtype: None
        :raises DataValidationError: If the manifest or recorded shards are absent.
        """
        del stage
        shard_root = Path(self.config.shard_dir)
        manifest_path = (
            Path(self.config.manifest)
            if self.config.manifest is not None
            else shard_root / "manifest.json"
        )
        if not manifest_path.is_file():
            raise DataValidationError(
                f"dataset manifest does not exist: {manifest_path}"
            )
        try:
            manifest = DatasetManifest.read(manifest_path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise DataValidationError(f"invalid dataset manifest: {error}") from error
        paths = manifest.shard_paths(shard_root)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise DataValidationError(f"dataset shards do not exist: {missing}")
        self.manifest = manifest
        self.paths = paths

    def train_dataloader(self) -> DataLoader[list[ComplexSample]]:
        """Return the configured sample partition as bucketed training batches."""
        return self._loader(self.config.partition)

    def val_dataloader(self) -> DataLoader[list[ComplexSample]]:
        """Return the validation partition when split metadata are available."""
        return self._loader("val")

    def test_dataloader(self) -> DataLoader[list[ComplexSample]]:
        """Return the test partition when split metadata are available."""
        return self._loader("test")

    def _loader(self, partition: str) -> DataLoader[list[ComplexSample]]:
        """Construct one DataLoader without applying a second batch collation."""
        if self.manifest is None:
            self.setup()
        if self.manifest is None:
            raise DataValidationError("dataset manifest was not initialized")
        dataset = _ShardBatchDataset(
            self.paths, self.manifest, self.config, partition, self.epoch
        )
        options: dict[str, Any] = {
            "batch_size": None,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": bool(
                self.config.num_workers and self.config.persistent_workers
            ),
        }
        if self.config.num_workers:
            options["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(dataset, **options)

    def set_epoch(self, epoch: int) -> None:
        """Set the nonnegative epoch used to derive deterministic shuffle seeds."""
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, Any]:
        """Return resumable deterministic stream metadata."""
        return {
            "epoch": self.epoch,
            "manifest_hash": self.manifest.hash if self.manifest is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore epoch state after validating the dataset fingerprint."""
        expected = state_dict.get("manifest_hash")
        if self.manifest is not None and expected not in (None, self.manifest.hash):
            raise DataValidationError("checkpoint dataset manifest hash mismatch")
        self.set_epoch(int(state_dict.get("epoch", 0)))
