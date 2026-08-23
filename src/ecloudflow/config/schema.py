"""Strict typed configuration schemas for ECloudFlow."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class StrictModel(BaseModel):
    """Base model that rejects undeclared configuration fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    """Model-width settings shared by all ECloudFlow backbones."""

    name: Literal["tiny", "base", "large"] = "tiny"
    scalar_dim: int = Field(default=64, ge=16)
    vector_dim: int = Field(default=16, ge=4)
    num_blocks: int = Field(default=3, ge=1)
    lmax: int = Field(default=2, ge=0, le=4)


class SampleConfig(StrictModel):
    """Sampling profile settings and bounded generation controls."""

    profile: Literal["fast", "balanced", "quality"] = "balanced"
    num_molecules: int = Field(default=100, ge=1)
    max_attempts: int | None = Field(default=None, ge=1)
    solver: Literal["euler", "heun"] = "heun"
    num_steps: int = Field(default=40, ge=1)
    corrector_steps: int = Field(default=2, ge=0)

    @computed_field
    @property
    def resolved_max_attempts(self) -> int:
        """Return the explicit attempt bound or five times the target count.

        :return: Positive maximum number of generation attempts.
        :rtype: int
        """
        return self.max_attempts or 5 * self.num_molecules


class DataConfig(StrictModel):
    """Dataset and distributed streaming settings.

    :param dataset: Dataset family name used by importers and manifests.
    :param manifest: Optional manifest path; defaults to ``manifest.json`` in
        ``shard_dir``.
    :param shard_dir: Directory containing WebDataset-compatible tar shards.
    :param partition: Sample-level partition selected by a DataLoader.
    :param batch_size: Per-device batch size.
    :param num_workers: DataLoader worker count.
    :param seed: Epoch/shuffle seed.
    :param shuffle_buffer: Deterministic bounded shuffle buffer size.
    :param bucket_width: Pocket/ligand node-count bucket width.
    :param target_shard_size_gb: Target atomic shard size in gigabytes.
    :param persistent_workers: Reuse workers across epochs when workers exist.
    :param prefetch_factor: Batches prefetched by each worker.
    :param pin_memory: Pin CPU tensors before accelerator transfer.
    :param verify_shard_hashes: Verify manifest hashes when opening shards.
    """

    dataset: str = "pdbbind"
    manifest: str | None = None
    shard_dir: str = "data/shards"
    partition: Literal["train", "val", "test", "all"] = "train"
    batch_size: int = Field(default=8, ge=1)
    num_workers: int = Field(default=8, ge=0)
    seed: int = 2026
    shuffle_buffer: int = Field(default=2048, ge=0)
    bucket_width: int = Field(default=32, ge=1)
    target_shard_size_gb: float = Field(default=1.0, ge=0.5, le=2.0)
    persistent_workers: bool = True
    prefetch_factor: int = Field(default=2, ge=1)
    pin_memory: bool = True
    verify_shard_hashes: bool = False


class AppConfig(StrictModel):
    """Top-level configuration composed from model and sampling groups."""

    seed: int = 2026
    model: ModelConfig = ModelConfig()
    sample: SampleConfig = SampleConfig()
    data: DataConfig = DataConfig()
