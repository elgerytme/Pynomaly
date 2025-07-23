"""MLOps domain interfaces for cross-package communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..domain.abstractions import ServiceInterface


class ExperimentTrackingInterface(ServiceInterface):
    """Interface for experiment tracking services."""
    
    @abstractmethod
    async def create_experiment(self, name: str, description: Optional[str] = None) -> UUID:
        """Create a new experiment."""
        pass
    
    @abstractmethod
    async def log_metric(self, experiment_id: UUID, metric_name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric for an experiment."""
        pass
    
    @abstractmethod
    async def log_parameter(self, experiment_id: UUID, param_name: str, value: Any) -> None:
        """Log a parameter for an experiment."""
        pass
    
    @abstractmethod
    async def log_artifact(self, experiment_id: UUID, artifact_path: str, artifact_data: bytes) -> None:
        """Log an artifact for an experiment."""
        pass
    
    @abstractmethod
    async def get_experiment(self, experiment_id: UUID) -> Optional[Dict[str, Any]]:
        """Get experiment details."""
        pass
    
    @abstractmethod
    async def list_experiments(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all experiments."""
        pass
    
    @abstractmethod
    async def delete_experiment(self, experiment_id: UUID) -> bool:
        """Delete an experiment."""
        pass


class ModelRegistryInterface(ServiceInterface):
    """Interface for model registry services."""
    
    @abstractmethod
    async def register_model(self, name: str, version: str, model_data: bytes, metadata: Optional[Dict[str, Any]] = None) -> UUID:
        """Register a new model version."""
        pass
    
    @abstractmethod
    async def get_model(self, model_id: UUID) -> Optional[Dict[str, Any]]:
        """Get model details."""
        pass
    
    @abstractmethod
    async def get_model_by_name_version(self, name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get model by name and version."""
        pass
    
    @abstractmethod
    async def list_models(self, name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models with optional name filter."""
        pass
    
    @abstractmethod
    async def promote_model(self, model_id: UUID, stage: str) -> bool:
        """Promote model to a stage (staging, production, etc.)."""
        pass
    
    @abstractmethod
    async def deprecate_model(self, model_id: UUID, reason: Optional[str] = None) -> bool:
        """Deprecate a model version."""
        pass
    
    @abstractmethod
    async def download_model(self, model_id: UUID) -> bytes:
        """Download model artifacts."""
        pass


class ModelDeploymentInterface(ServiceInterface):
    """Interface for model deployment services."""
    
    @abstractmethod
    async def deploy_model(self, model_id: UUID, deployment_config: Dict[str, Any]) -> UUID:
        """Deploy a model."""
        pass
    
    @abstractmethod
    async def get_deployment(self, deployment_id: UUID) -> Optional[Dict[str, Any]]:
        """Get deployment details."""
        pass
    
    @abstractmethod
    async def list_deployments(self, model_id: Optional[UUID] = None) -> List[Dict[str, Any]]:
        """List deployments, optionally filtered by model."""
        pass
    
    @abstractmethod
    async def update_deployment(self, deployment_id: UUID, config: Dict[str, Any]) -> bool:
        """Update deployment configuration."""
        pass
    
    @abstractmethod
    async def scale_deployment(self, deployment_id: UUID, replicas: int) -> bool:
        """Scale deployment to specified number of replicas."""
        pass
    
    @abstractmethod
    async def stop_deployment(self, deployment_id: UUID) -> bool:
        """Stop a deployment."""
        pass
    
    @abstractmethod
    async def get_deployment_metrics(self, deployment_id: UUID) -> Dict[str, Any]:
        """Get deployment performance metrics."""
        pass


class PipelineInterface(ServiceInterface):
    """Interface for ML pipeline services."""
    
    @abstractmethod
    async def create_pipeline(self, name: str, definition: Dict[str, Any]) -> UUID:
        """Create a new pipeline."""
        pass
    
    @abstractmethod
    async def run_pipeline(self, pipeline_id: UUID, parameters: Optional[Dict[str, Any]] = None) -> UUID:
        """Run a pipeline with optional parameters."""
        pass
    
    @abstractmethod
    async def get_pipeline_run(self, run_id: UUID) -> Optional[Dict[str, Any]]:
        """Get pipeline run details."""
        pass
    
    @abstractmethod
    async def list_pipeline_runs(self, pipeline_id: UUID) -> List[Dict[str, Any]]:
        """List runs for a pipeline."""
        pass
    
    @abstractmethod
    async def cancel_pipeline_run(self, run_id: UUID) -> bool:
        """Cancel a running pipeline."""
        pass


class AutoMLInterface(ServiceInterface):
    """Interface for AutoML services."""
    
    @abstractmethod
    async def start_automl_job(self, dataset_id: UUID, target_column: str, problem_type: str, config: Dict[str, Any]) -> UUID:
        """Start an AutoML job."""
        pass
    
    @abstractmethod
    async def get_automl_job(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        """Get AutoML job details."""
        pass
    
    @abstractmethod
    async def list_automl_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List AutoML jobs with optional status filter."""
        pass
    
    @abstractmethod
    async def get_automl_results(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        """Get AutoML job results including best models."""
        pass
    
    @abstractmethod
    async def cancel_automl_job(self, job_id: UUID) -> bool:
        """Cancel a running AutoML job."""
        pass


class FeatureStoreInterface(ServiceInterface):
    """Interface for feature store services."""
    
    @abstractmethod
    async def create_feature_group(self, name: str, features: List[Dict[str, Any]]) -> UUID:
        """Create a feature group."""
        pass
    
    @abstractmethod
    async def get_features(self, feature_group_id: UUID, entity_ids: List[str]) -> Dict[str, Any]:
        """Get features for specific entities."""
        pass
    
    @abstractmethod
    async def ingest_features(self, feature_group_id: UUID, feature_data: Dict[str, Any]) -> bool:
        """Ingest feature data."""
        pass
    
    @abstractmethod
    async def get_feature_statistics(self, feature_group_id: UUID) -> Dict[str, Any]:
        """Get feature statistics and data quality metrics."""
        pass


class ModelMonitoringInterface(ServiceInterface):
    """Interface for model monitoring services."""
    
    @abstractmethod
    async def setup_monitoring(self, deployment_id: UUID, monitoring_config: Dict[str, Any]) -> UUID:
        """Set up monitoring for a deployed model."""
        pass
    
    @abstractmethod
    async def log_prediction(self, deployment_id: UUID, input_data: Any, prediction: Any, actual: Optional[Any] = None) -> None:
        """Log a prediction for monitoring."""
        pass
    
    @abstractmethod
    async def get_model_metrics(self, deployment_id: UUID, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """Get model performance metrics."""
        pass
    
    @abstractmethod
    async def detect_drift(self, deployment_id: UUID) -> Dict[str, Any]:
        """Detect data or concept drift."""
        pass
    
    @abstractmethod
    async def get_alerts(self, deployment_id: UUID) -> List[Dict[str, Any]]:
        """Get active alerts for a deployment."""
        pass