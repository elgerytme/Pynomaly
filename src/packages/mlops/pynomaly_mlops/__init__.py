"""
Pynomaly MLOps Package

A comprehensive MLOps platform for machine learning operations including:
- Model lifecycle management
- Automated training and deployment pipelines  
- Real-time monitoring and observability
- Experiment tracking and management
- Data versioning and lineage
- Governance and compliance
- AutoML and hyperparameter optimization
"""

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "dev@pynomaly.io"

from pynomaly_mlops.application.services.mlops_service import MLOpsService
from pynomaly_mlops.domain.entities.model import Model, ModelStatus, ModelType
from pynomaly_mlops.domain.entities.experiment import Experiment, ExperimentRun
from pynomaly_mlops.domain.entities.pipeline import Pipeline, PipelineStep, PipelineRun
from pynomaly_mlops.domain.entities.deployment import Deployment, DeploymentEnvironment
from pynomaly_mlops.infrastructure.persistence.model_repository import ModelRepository
from pynomaly_mlops.infrastructure.monitoring.metrics_collector import MetricsCollector

__all__ = [
    "MLOpsService",
    "Model",
    "ModelStatus", 
    "ModelType",
    "Experiment",
    "ExperimentRun",
    "Pipeline",
    "PipelineStep",
    "PipelineRun", 
    "Deployment",
    "DeploymentEnvironment",
    "ModelRepository",
    "MetricsCollector",
]