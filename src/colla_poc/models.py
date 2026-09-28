"""Validated experiment settings."""
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator

class EmbeddingConfig(BaseModel):
    model: str
    revision: str = 'main'
    batch_size: int = Field(gt=0)

class DuplicateConfig(BaseModel):
    phash_max_distance: int = Field(ge=0, le=64)
    cosine_similarity_threshold: float = Field(ge=-1, le=1)
    candidate_time_window_minutes: float = Field(gt=0)

class MomentConfig(BaseModel):
    coarse_time_gap_minutes: float = Field(gt=0)
    time_weight: float = Field(ge=0)
    visual_weight: float = Field(ge=0)
    gps_weight: float = Field(ge=0)
    max_time_distance_minutes: float = Field(gt=0)
    max_gps_distance_meters: float = Field(gt=0)
    split_similarity_threshold: float = Field(ge=0, le=1)

    @model_validator(mode='after')
    def check_weights(self) -> 'MomentConfig':
        if self.time_weight + self.visual_weight + self.gps_weight <= 0:
            raise ValueError('At least one moment signal needs a positive weight')
        return self

class QualityConfig(BaseModel):
    sharpness_weight: float = Field(ge=0)
    exposure_weight: float = Field(ge=0)
    contrast_weight: float = Field(ge=0)
    resolution_weight: float = Field(ge=0)

    @model_validator(mode='after')
    def check_weights(self) -> 'QualityConfig':
        total = self.sharpness_weight + self.exposure_weight + self.contrast_weight + self.resolution_weight
        if abs(total - 1) > 1e-6:
            raise ValueError('Quality weights must add to 1')
        return self

class RankingConfig(BaseModel):
    quality_weight: float = Field(ge=0)
    similarity_penalty: float = Field(ge=0)
    contributor_balance_weight: float = Field(default=0, ge=0)
    top_k: int = Field(gt=0)

class SemanticsConfig(BaseModel):
    confidence_threshold: float = Field(ge=0, le=1)
    labels: dict[str, str]

class Config(BaseModel):
    embedding: EmbeddingConfig
    duplicate: DuplicateConfig
    moment: MomentConfig
    quality: QualityConfig
    ranking: RankingConfig
    semantics: SemanticsConfig
