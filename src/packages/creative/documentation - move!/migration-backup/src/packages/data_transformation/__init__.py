"""
Pynomaly Data Transformation Package

A comprehensive data transformation and feature engineering package for anomaly detection.
Provides tools for data integration, cleaning, preparation, and advanced feature engineering.
"""

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "team@monorepo.io"

from .application.use_cases.data_pipeline import DataPipelineUseCase
from .domain.entities.transformation_pipeline import TransformationPipeline
from .domain.value_objects.pipeline_config import (
    PipelineConfig, SourceType, CleaningStrategy, ScalingMethod, EncodingStrategy
)
from .infrastructure.adapters.data_source_adapter import DataSourceAdapter

__all__ = [
    "DataPipelineUseCase",
    "TransformationPipeline",
    "PipelineConfig",
    "SourceType",
    "CleaningStrategy", 
    "ScalingMethod",
    "EncodingStrategy",
    "DataSourceAdapter",
]