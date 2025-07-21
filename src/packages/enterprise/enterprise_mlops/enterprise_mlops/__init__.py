"""
Enterprise MLOps Package

Provides comprehensive MLOps integrations including experiment tracking,
model management, deployment orchestration, and monitoring for enterprise
anomaly detection workflows.

Key Features:
- MLflow integration for experiment tracking and model registry
- Kubeflow integration for ML pipelines and model serving
- Datadog integration for metrics and monitoring
- New Relic integration for application observability
- Centralized orchestration service for coordinating MLOps operations
- Enterprise-grade security, multi-tenancy, and scalability

Usage:
    from enterprise_mlops import MLOpsOrchestrationService
    from enterprise_mlops.infrastructure.mlops.mlflow import MLflowIntegration
    from enterprise_mlops.infrastructure.monitoring.datadog import DatadogIntegration
    
    # Initialize integrations
    mlflow = MLflowIntegration("http://mlflow-server:5000")
    datadog = DatadogIntegration("api-key", "app-key")
    
    # Create orchestration service
    mlops = MLOpsOrchestrationService(
        mlflow_integration=mlflow,
        datadog_integration=datadog
    )
"""

from .application.services.mlops_orchestration_service import MLOpsOrchestrationService
from .domain.entities.mlops import (
    MLExperiment, MLModel, ModelDeployment, MLPipeline,
    ExperimentStatus, ModelStatus, DeploymentStatus, PipelineStatus
)

__version__ = "1.0.0"
__all__ = [
    "MLOpsOrchestrationService",
    "MLExperiment",
    "MLModel", 
    "ModelDeployment",
    "MLPipeline",
    "ExperimentStatus",
    "ModelStatus", 
    "DeploymentStatus",
    "PipelineStatus"
]