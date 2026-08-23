"""Lightning DataModule for deterministic sharded ECloudFlow datasets."""

from __future__ import annotations

from collections.abc import Iterator
from multiprocessing import Value
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
        epoch_state: Any,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.sample_partitions = dict(manifest.sample_partitions)
        self.shard_hashes = {
            Path(record.path).name: record.sha256 for record in manifest.shards
        }
        self.shard_sample_ids = {
            Path(record.path).name: tuple(record.sample_ids)
            for record in manifest.shards
        }
        self.config = config
        self.partition = partition
        self.epoch_state = epoch_state

    def __iter__(self) -> Iterator[list[ComplexSample]]:
        """Build one deterministic rank/worker-local epoch iterator.

        :return: Iterator of node-count bucketed canonical sample lists.
        :rtype: Iterator[list[ComplexSample]]
        :raises ValueError: If rank/worker, batching, or shuffle settings are invalid.
        :raises ShardReadError: If an owned shard, cache entry, or payload fails
            hash, archive, member-index, or canonical validation.

        Distributed identity is read inside each worker. The shared epoch value
        is read when iteration begins, so persistent worker processes observe a
        later :meth:`ECloudDataModule.set_epoch` call without loader recreation.
        Whole-shard ownership is preferred; sparse shards use indexed members
        and never deserialize samples owned by another rank or worker. Returned
        complexes are canonical CPU records with ``[N, 3]`` local binding-frame
        coordinates in angstroms; floating dtype, shape, graph features,
        chemical/field masks, and provenance are preserved. Accelerator transfer
        is a later training-loop responsibility. The worker mutates only its
        iterator/bucket state and optional verified cache; manifest mappings and
        samples remain immutable.
        """
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        if distributed.is_available() and distributed.is_initialized():
            rank, world_size = distributed.get_rank(), distributed.get_world_size()
        else:
            rank, world_size = 0, 1
        allowed_ids: set[str] | None = None
        if self.sample_partitions and self.partition != "all":
            allowed_ids = {
                sample_id
                for sample_id, assigned in self.sample_partitions.items()
                if assigned == self.partition
            }
        expected_hashes = None
        if self.config.verify_shard_hashes or self.config.local_cache_dir is not None:
            expected_hashes = self.shard_hashes
        epoch = int(self.epoch_state.value)
        stream = stream_samples(
            self.paths,
            rank=rank,
            world_size=world_size,
            worker_id=worker_id,
            num_workers=worker_count,
            seed=self.config.seed + epoch,
            shuffle_buffer=self.config.shuffle_buffer,
            allowed_sample_ids=allowed_ids,
            expected_hashes=expected_hashes,
            shard_sample_ids=self.shard_sample_ids,
            cache_dir=self.config.local_cache_dir,
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
        self._epoch_state = Value("q", 0, lock=True)
        self._pending_manifest_hash: str | None = None
        self.manifest: DatasetManifest | None = None
        self.paths: tuple[Path, ...] = ()

    def setup(self, stage: str | None = None) -> None:
        """Validate and transactionally publish immutable dataset state.

        :param stage: Optional Lightning lifecycle stage. Discovery is identical
            for ``fit``, ``validate``, ``test``, ``predict``, and ``None``.
        :return: None.
        :rtype: None
        :raises DataValidationError: If the manifest/shards are absent or
            invalid, or if a restore-before-setup checkpoint hash disagrees.

        The manifest and resolved generation paths are held in local variables
        until every existence, schema, exact-partition, and pending-checkpoint
        validation succeeds. Entry clears previously published module state;
        therefore any caught failure leaves loaders unusable until a later valid
        setup. This CPU-only operation reads metadata/path existence but never
        opens shard tensors, changes dtype/device/frame/angstrom coordinates or
        masks, writes files, or mutates the frozen manifest.
        """
        del stage
        self.manifest = None
        self.paths = ()
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
        if (
            self._pending_manifest_hash is not None
            and self._pending_manifest_hash != manifest.hash
        ):
            raise DataValidationError("checkpoint dataset manifest hash mismatch")
        self.manifest = manifest
        self.paths = paths

    def train_dataloader(self) -> DataLoader[list[ComplexSample]]:
        """Create the configured training-partition loader.

        :return: Iterable DataLoader yielding pre-bucketed sample lists without
            an additional collation step.
        :rtype: torch.utils.data.DataLoader
        :raises DataValidationError: If setup, manifest, or shard validation fails.

        Rank and worker ownership are resolved lazily inside worker processes.
        Persistent workers share epoch state but never mutate the manifest.
        """
        return self._loader(self.config.partition)

    def val_dataloader(self) -> DataLoader[list[ComplexSample]]:
        """Create a validation loader using sample-level split assignments.

        :return: Bucketed validation DataLoader; explicit unpartitioned
            manifests yield all samples.
        :rtype: torch.utils.data.DataLoader
        :raises DataValidationError: If dataset setup or validation fails.
        """
        return self._loader("val")

    def test_dataloader(self) -> DataLoader[list[ComplexSample]]:
        """Create a test loader using sample-level split assignments.

        :return: Bucketed test DataLoader; explicit unpartitioned manifests
            yield all samples.
        :rtype: torch.utils.data.DataLoader
        :raises DataValidationError: If dataset setup or validation fails.
        """
        return self._loader("test")

    def _loader(self, partition: str) -> DataLoader[list[ComplexSample]]:
        """Construct one DataLoader without applying a second batch collation."""
        if self.manifest is None:
            self.setup()
        if self.manifest is None:
            raise DataValidationError("dataset manifest was not initialized")
        dataset = _ShardBatchDataset(
            self.paths, self.manifest, self.config, partition, self._epoch_state
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
        """Publish a deterministic epoch to current persistent workers.

        :param epoch: Nonnegative epoch number mixed with the configured seed.
        :return: None.
        :rtype: None
        :raises ValueError: If ``epoch`` is negative.

        The synchronized CPU scalar is worker-visible and contains no device
        tensors. Equal seed/epoch pairs reproduce order; a changed epoch changes
        bounded-shuffle order without recreating persistent workers.
        Updating the scalar does not mutate samples, manifests, shard files,
        coordinate frames, tensor dtype/device, graph features, or masks.
        """
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)
        with self._epoch_state.get_lock():
            self._epoch_state.value = self.epoch

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint-safe epoch and dataset identity metadata.

        :return: Plain mapping containing epoch and the loaded manifest hash.
        :rtype: dict[str, Any]

        The method performs no filesystem mutation. A manifest hash is absent
        only when Lightning requests state before :meth:`setup`.
        Returned values contain no tensors/devices and are deterministically
        stable across ranks for equal epoch and manifest; callers receive a new
        mutable dictionary.
        """
        return {
            "epoch": self.epoch,
            "manifest_hash": self.manifest.hash if self.manifest is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore epoch and defer manifest validation when setup is later.

        :param state_dict: Checkpoint metadata returned by :meth:`state_dict`.
        :return: None.
        :rtype: None
        :raises DataValidationError: If a loaded manifest disagrees with the
            checkpoint hash, either immediately or during subsequent setup.

        Lightning may restore state before calling ``setup``. The expected hash
        is retained and checked as soon as the immutable manifest is loaded.
        If state arrives after setup, a mismatch fails before the epoch changes.
        Otherwise the synchronized CPU epoch scalar is updated deterministically
        for persistent workers. No shard/sample tensor, dtype, device, binding
        frame, angstrom coordinate, feature, or mask is mutated.
        """
        expected = state_dict.get("manifest_hash")
        if expected is not None and not isinstance(expected, str):
            self.manifest = None
            self.paths = ()
            raise DataValidationError("checkpoint manifest hash must be textual")
        self._pending_manifest_hash = expected
        if self.manifest is not None and expected not in (None, self.manifest.hash):
            self.manifest = None
            self.paths = ()
            raise DataValidationError("checkpoint dataset manifest hash mismatch")
        self.set_epoch(int(state_dict.get("epoch", 0)))
