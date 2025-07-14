"""Pynomaly MLOps Platform

A comprehensive MLOps platform for model lifecycle management,
experiment tracking, and deployment orchestration.
"""

from .application import (
    ModelRegistryService, ExperimentTrackingService, ExperimentAnalysisService,
    PipelineOrchestrationService
)
from .domain import (
    Model, ModelStatus, ModelType,
    Experiment, ExperimentRun, ExperimentStatus, ExperimentRunStatus,
    Pipeline, PipelineStep, PipelineRun, PipelineStatus, StepStatus, StepType,
    SemanticVersion, ModelMetrics,
    ModelRepository, ExperimentRepository, PipelineRepository, ModelPromotionService
)
from .infrastructure import (
    MLOpsContainer, MLOpsSettings,
    ArtifactStorageService, S3ArtifactStorage, LocalArtifactStorage,
    SqlAlchemyModelRepository, SqlAlchemyExperimentRepository,
    SqlAlchemyPipelineRepository, SqlAlchemyPipelineRunRepository,
    PipelineExecutor, PipelineScheduler
)

__version__ = "1.0.0"

__all__ = [
    # Application services
    "ModelRegistryService",
    "ExperimentTrackingService",
    "ExperimentAnalysisService",
    "PipelineOrchestrationService",
    
    # Domain entities and value objects
    "Model", "ModelStatus", "ModelType",
    "Experiment", "ExperimentRun", "ExperimentStatus", "ExperimentRunStatus",
    "Pipeline", "PipelineStep", "PipelineRun", "PipelineStatus", "StepStatus", "StepType",
    "SemanticVersion", "ModelMetrics",
    
    # Domain contracts
    "ModelRepository", "ExperimentRepository", "PipelineRepository", "ModelPromotionService",
    
    # Infrastructure
    "MLOpsContainer", "MLOpsSettings",
    "ArtifactStorageService", "S3ArtifactStorage", "LocalArtifactStorage",
    "SqlAlchemyModelRepository", "SqlAlchemyExperimentRepository",
    "SqlAlchemyPipelineRepository", "SqlAlchemyPipelineRunRepository",
    "PipelineExecutor", "PipelineScheduler",
]