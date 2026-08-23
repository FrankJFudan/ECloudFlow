"""Strict typed configuration schemas for ECloudFlow."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


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
    :param local_cache_dir: Optional portable local read-through shard cache;
        ``None`` disables caching without assuming a machine-specific path.
    :param diffgui_lmdb: Optional existing DiffGui processed LMDB path.
    :param diffgui_source_root: Optional read-only root for official DiffGui
        protein and ligand source filenames.
    :param diffgui_build_fields: Build physical fields while importing DiffGui.
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
    local_cache_dir: str | None = None
    diffgui_lmdb: str | None = None
    diffgui_source_root: str | None = None
    diffgui_build_fields: bool = False


class WeightedLossConfig(StrictModel):
    """Common component weight and deterministic linear warm-up bounds."""

    weight: float = Field(default=1.0, ge=0.0, le=1.0e6)
    warmup_start: int = Field(default=0, ge=0)
    warmup_end: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_warmup(self) -> "WeightedLossConfig":
        """Require a nondecreasing inclusive warm-up interval."""
        if self.warmup_end < self.warmup_start:
            raise ValueError(
                "warmup_end must be greater than or equal to warmup_start."
            )
        return self


class FlowLossConfig(WeightedLossConfig):
    """Weights for exact Cartesian and packed-electron velocity targets."""

    position: float = Field(default=1.0, ge=0.0, le=1.0e6)
    electron: float = Field(default=1.0, ge=0.0, le=1.0e6)


class ScoreLossConfig(WeightedLossConfig):
    """Weights for exact Cartesian and packed-electron score targets."""

    position: float = Field(default=1.0, ge=0.0, le=1.0e6)
    electron: float = Field(default=1.0, ge=0.0, le=1.0e6)


class DiscreteLossConfig(WeightedLossConfig):
    """Weights for node, sparse-halfedge, and per-complex endpoint classes."""

    atom: float = Field(default=1.0, ge=0.0, le=1.0e6)
    charge: float = Field(default=1.0, ge=0.0, le=1.0e6)
    bond: float = Field(default=1.0, ge=0.0, le=1.0e6)
    count: float = Field(default=1.0, ge=0.0, le=1.0e6)


class ECloudLossConfig(WeightedLossConfig):
    """Weights for genuine-QM field reconstruction and latent cycle terms."""

    density: float = Field(default=1.0, ge=0.0, le=1.0e6)
    gradient: float = Field(default=0.1, ge=0.0, le=1.0e6)
    electron_count: float = Field(default=0.1, ge=0.0, le=1.0e6)
    dipole: float = Field(default=0.1, ge=0.0, le=1.0e6)
    cycle: float = Field(default=0.1, ge=0.0, le=1.0e6)


class ChemistryLossConfig(WeightedLossConfig):
    """Weights and stable constants for differentiable chemistry surrogates."""

    valence: float = Field(default=0.1, ge=0.0, le=1.0e6)
    bond_length: float = Field(default=0.1, ge=0.0, le=1.0e6)
    ligand_clash: float = Field(default=0.1, ge=0.0, le=1.0e6)
    protein_clash: float = Field(default=0.1, ge=0.0, le=1.0e6)
    ring_strain: float = Field(default=0.1, ge=0.0, le=1.0e6)
    connectivity: float = Field(default=0.1, ge=0.0, le=1.0e6)
    affinity: float = Field(default=0.1, ge=0.0, le=1.0e6)
    ligand_clash_distance: float = Field(default=1.2, gt=0.0, le=10.0)
    protein_clash_distance: float = Field(default=1.5, gt=0.0, le=10.0)
    minimum_degree: float = Field(default=1.0, ge=0.0, le=16.0)
    affinity_log_variance_min: float = Field(default=-10.0, ge=-30.0, le=30.0)
    affinity_log_variance_max: float = Field(default=10.0, ge=-30.0, le=30.0)
    epsilon: float = Field(default=1.0e-8, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_variance_bounds(self) -> "ChemistryLossConfig":
        """Require ordered finite heteroscedastic log-variance bounds."""
        if self.affinity_log_variance_max < self.affinity_log_variance_min:
            raise ValueError("affinity log-variance bounds must be ordered.")
        return self


class InteractionLossConfig(WeightedLossConfig):
    """Weight and focusing exponent for per-complex interaction supervision."""

    focal_gamma: float = Field(default=2.0, ge=0.0, le=10.0)


class LossNormalizationConfig(StrictModel):
    """Detached distributed running-RMS normalization controls."""

    enabled: bool = True
    decay: float = Field(default=0.99, ge=0.0, lt=1.0)
    epsilon: float = Field(default=1.0e-8, gt=0.0, le=1.0)


class LossConfig(StrictModel):
    """Frozen, serializable configuration for all six scientific components."""

    flow: FlowLossConfig = FlowLossConfig()
    score: ScoreLossConfig = ScoreLossConfig()
    discrete: DiscreteLossConfig = DiscreteLossConfig()
    ecloud: ECloudLossConfig = ECloudLossConfig()
    chem: ChemistryLossConfig = ChemistryLossConfig()
    interaction: InteractionLossConfig = InteractionLossConfig()
    normalization: LossNormalizationConfig = LossNormalizationConfig()


class AppConfig(StrictModel):
    """Top-level configuration composed from model and sampling groups."""

    seed: int = 2026
    model: ModelConfig = ModelConfig()
    sample: SampleConfig = SampleConfig()
    data: DataConfig = DataConfig()
    loss: LossConfig = LossConfig()
