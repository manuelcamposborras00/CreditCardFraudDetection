"""
config_schema.py — Pydantic models for config.yaml validation.

Validates structure and types before the pipeline starts,
surfacing configuration errors early with clear messages.
"""

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    file: str


class ModelConfig(BaseModel):
    mi_sample_size: int = Field(gt=0)


class VisualizationConfig(BaseModel):
    boxplot_ylim: tuple[float, float]
    outlier_threshold_amount: float = Field(gt=0)
    class_labels: dict[int, str]


class AppConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    visualization: VisualizationConfig
