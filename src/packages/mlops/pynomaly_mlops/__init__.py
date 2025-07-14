"""Pynomaly MLOps Platform

A comprehensive MLOps platform for model lifecycle management,
experiment tracking, and deployment orchestration.
"""

from .application import ModelRegistryService, ExperimentTrackingService, ExperimentAnalysisService
from .domain import (
    Model, ModelStatus, ModelType,
    Experiment, ExperimentRun, ExperimentStatus, ExperimentRunStatus,
    SemanticVersion, ModelMetrics,
    ModelRepository, ExperimentRepository, ModelPromotionService
)
from .infrastructure import (
    MLOpsContainer, MLOpsSettings,
    ArtifactStorageService, S3ArtifactStorage, LocalArtifactStorage,
    SqlAlchemyModelRepository, SqlAlchemyExperimentRepository
)

__version__ = "1.0.0"

__all__ = [
    # Application services
    "ModelRegistryService",
    "ExperimentTrackingService",
    "ExperimentAnalysisService",
    
    # Domain entities and value objects
    "Model", "ModelStatus", "ModelType",
    "Experiment", "ExperimentRun", "ExperimentStatus", "ExperimentRunStatus",
    "SemanticVersion", "ModelMetrics",
    
    # Domain contracts
    "ModelRepository", "ExperimentRepository", "ModelPromotionService",
    
    # Infrastructure
    "MLOpsContainer", "MLOpsSettings",
    "ArtifactStorageService", "S3ArtifactStorage", "LocalArtifactStorage",
    "SqlAlchemyModelRepository", "SqlAlchemyExperimentRepository",
]