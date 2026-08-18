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


class AppConfig(StrictModel):
    """Top-level configuration composed from model and sampling groups."""

    seed: int = 2026
    model: ModelConfig = ModelConfig()
    sample: SampleConfig = SampleConfig()
