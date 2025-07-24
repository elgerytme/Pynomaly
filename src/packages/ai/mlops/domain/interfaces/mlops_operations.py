"""MLOps Operations Interfaces (Ports).

This module defines the abstract interfaces for MLOps operations that the
anomaly detection domain requires. These interfaces represent the "ports"
in hexagonal architecture, defining contracts for external MLOps services
without coupling to specific implementations.

Following DDD principles, these interfaces belong to the domain layer and
define what the domain needs from external MLOps capabilities like experiment
tracking, model deployment, monitoring, and lifecycle management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid

from ..entities.model import Model
from ..entities.detection_result import DetectionResult


class ExperimentStatus(Enum):
    """Status of an MLOps experiment."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(Enum):
    """Status of an experiment run."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class DeploymentStatus(Enum):
    """Status of model deployment."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ModelStage(Enum):
    """Stage of model in MLOps lifecycle."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ExperimentMetadata:
    """Metadata for an MLOps experiment."""
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    tags: Dict[str, str]
    parameters: Dict[str, Any]


@dataclass
class RunMetadata:
    """Metadata for an experiment run."""
    run_id: str
    experiment_id: str
    name: str
    status: RunStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    created_by: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    tags: Dict[str, str]


@dataclass
class ModelVersionMetadata:
    """Metadata for a model version in MLOps."""
    version_id: str
    model_id: str
    version: str
    stage: ModelStage
    created_at: datetime
    created_by: str
    description: str
    run_id: Optional[str]
    source_path: str
    performance_metrics: Dict[str, float]
    deployment_config: Dict[str, Any]
    tags: Dict[str, str]


@dataclass
class DeploymentMetadata:
    """Metadata for model deployment."""
    deployment_id: str
    model_version_id: str
    name: str
    status: DeploymentStatus
    endpoint_url: Optional[str]
    created_at: datetime
    updated_at: datetime
    created_by: str
    configuration: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    health_check_config: Dict[str, Any]


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for model monitoring."""
    model_version_id: str
    timestamp: datetime
    request_count: int
    average_response_time_ms: float
    error_rate: float
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    custom_metrics: Dict[str, float]


class MLOpsExperimentTrackingPort(ABC):
    """Port for MLOps experiment tracking operations.
    
    This interface defines the contract for tracking experiments, runs,
    parameters, metrics, and artifacts in an MLOps platform.
    """

    @abstractmethod
    async def create_experiment(
        self, 
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new experiment for tracking.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: Optional tags for categorization
            created_by: User creating the experiment
            
        Returns:
            Unique experiment identifier
            
        Raises:
            ExperimentCreationError: If experiment creation fails
        """
        pass

    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Retrieve experiment metadata.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            Experiment metadata if found, None otherwise
            
        Raises:
            ExperimentRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def list_experiments(
        self, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ExperimentMetadata]:
        """List experiments with optional filtering.
        
        Args:
            filters: Optional filters (status, created_by, tags, etc.)
            
        Returns:
            List of experiment metadata
            
        Raises:
            ExperimentQueryError: If listing fails
        """
        pass

    @abstractmethod
    async def start_run(
        self,
        experiment_id: str,
        run_name: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Start a new experiment run.
        
        Args:
            experiment_id: Parent experiment identifier
            run_name: Name for the run
            parameters: Initial parameters
            tags: Optional tags
            created_by: User starting the run
            
        Returns:
            Unique run identifier
            
        Raises:
            RunCreationError: If run creation fails
        """
        pass

    @abstractmethod
    async def end_run(self, run_id: str, status: RunStatus = RunStatus.COMPLETED) -> None:
        """End an experiment run.
        
        Args:
            run_id: Run identifier
            status: Final status of the run
            
        Raises:
            RunUpdateError: If run ending fails
        """
        pass

    @abstractmethod
    async def log_parameter(self, run_id: str, key: str, value: Any) -> None:
        """Log a parameter for a run.
        
        Args:
            run_id: Run identifier
            key: Parameter name
            value: Parameter value
            
        Raises:
            ParameterLoggingError: If parameter logging fails
        """
        pass

    @abstractmethod
    async def log_parameters(self, run_id: str, parameters: Dict[str, Any]) -> None:
        """Log multiple parameters for a run.
        
        Args:
            run_id: Run identifier
            parameters: Dictionary of parameters
            
        Raises:
            ParameterLoggingError: If parameter logging fails
        """
        pass

    @abstractmethod
    async def log_metric(
        self, 
        run_id: str, 
        key: str, 
        value: float, 
        step: Optional[int] = None
    ) -> None:
        """Log a metric for a run.
        
        Args:
            run_id: Run identifier
            key: Metric name
            value: Metric value
            step: Optional step number
            
        Raises:
            MetricLoggingError: If metric logging fails
        """
        pass

    @abstractmethod
    async def log_metrics(
        self, 
        run_id: str, 
        metrics: Dict[str, float], 
        step: Optional[int] = None
    ) -> None:
        """Log multiple metrics for a run.
        
        Args:
            run_id: Run identifier
            metrics: Dictionary of metrics
            step: Optional step number
            
        Raises:
            MetricLoggingError: If metric logging fails
        """
        pass

    @abstractmethod
    async def log_artifact(
        self, 
        run_id: str, 
        artifact_path: str, 
        artifact_name: str = ""
    ) -> str:
        """Log an artifact for a run.
        
        Args:
            run_id: Run identifier
            artifact_path: Path to the artifact
            artifact_name: Optional name for the artifact
            
        Returns:
            URI of the logged artifact
            
        Raises:
            ArtifactLoggingError: If artifact logging fails
        """
        pass

    @abstractmethod
    async def get_run(self, run_id: str) -> Optional[RunMetadata]:
        """Retrieve run metadata.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run metadata if found, None otherwise
            
        Raises:
            RunRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def list_runs(
        self, 
        experiment_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RunMetadata]:
        """List runs for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            filters: Optional filters (status, created_by, etc.)
            
        Returns:
            List of run metadata
            
        Raises:
            RunQueryError: If listing fails
        """
        pass


class MLOpsModelRegistryPort(ABC):
    """Port for MLOps model registry operations.
    
    This interface defines the contract for managing model versions,
    stages, and metadata in an MLOps model registry.
    """

    @abstractmethod
    async def register_model(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Register a new model in the registry.
        
        Args:
            name: Model name
            description: Model description
            tags: Optional tags
            created_by: User registering the model
            
        Returns:
            Unique model identifier
            
        Raises:
            ModelRegistrationError: If registration fails
        """
        pass

    @abstractmethod
    async def create_model_version(
        self,
        model_id: str,
        version: str,
        run_id: str,
        source_path: str,
        description: str = "",
        performance_metrics: Optional[Dict[str, float]] = None,
        deployment_config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new version of a model.
        
        Args:
            model_id: Model identifier
            version: Version string
            run_id: Associated run identifier
            source_path: Path to model artifacts
            description: Version description
            performance_metrics: Model performance metrics
            deployment_config: Deployment configuration
            tags: Optional tags
            created_by: User creating the version
            
        Returns:
            Unique version identifier
            
        Raises:
            VersionCreationError: If version creation fails
        """
        pass

    @abstractmethod
    async def get_model_version(
        self, 
        model_id: str, 
        version: str
    ) -> Optional[ModelVersionMetadata]:
        """Retrieve model version metadata.
        
        Args:
            model_id: Model identifier
            version: Version string
            
        Returns:
            Model version metadata if found, None otherwise
            
        Raises:
            VersionRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def list_model_versions(
        self, 
        model_id: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersionMetadata]:
        """List versions of a model.
        
        Args:
            model_id: Model identifier
            stage: Optional stage filter
            
        Returns:
            List of model version metadata
            
        Raises:
            VersionQueryError: If listing fails
        """
        pass

    @abstractmethod
    async def transition_model_stage(
        self,
        model_id: str,
        version: str,
        stage: ModelStage,
        comment: str = "",
        archive_existing: bool = True
    ) -> None:
        """Transition model version to a different stage.
        
        Args:
            model_id: Model identifier
            version: Version string
            stage: Target stage
            comment: Optional comment
            archive_existing: Whether to archive existing model in target stage
            
        Raises:
            StageTransitionError: If transition fails
        """
        pass

    @abstractmethod
    async def get_latest_model_version(
        self, 
        model_id: str, 
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersionMetadata]:
        """Get the latest version of a model in a specific stage.
        
        Args:
            model_id: Model identifier
            stage: Optional stage filter
            
        Returns:
            Latest model version metadata if found, None otherwise
            
        Raises:
            VersionRetrievalError: If retrieval fails
        """
        pass


class MLOpsModelDeploymentPort(ABC):
    """Port for MLOps model deployment operations.
    
    This interface defines the contract for deploying, managing, and
    monitoring model deployments in an MLOps platform.
    """

    @abstractmethod
    async def deploy_model(
        self,
        model_version_id: str,
        deployment_name: str,
        configuration: Dict[str, Any],
        resource_requirements: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> str:
        """Deploy a model version.
        
        Args:
            model_version_id: Model version identifier
            deployment_name: Name for the deployment
            configuration: Deployment configuration
            resource_requirements: Optional resource requirements
            created_by: User deploying the model
            
        Returns:
            Unique deployment identifier
            
        Raises:
            DeploymentError: If deployment fails
        """
        pass

    @abstractmethod
    async def get_deployment(self, deployment_id: str) -> Optional[DeploymentMetadata]:
        """Retrieve deployment metadata.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Deployment metadata if found, None otherwise
            
        Raises:
            DeploymentRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def list_deployments(
        self,
        model_id: Optional[str] = None,
        status: Optional[DeploymentStatus] = None
    ) -> List[DeploymentMetadata]:
        """List model deployments.
        
        Args:
            model_id: Optional model identifier filter
            status: Optional status filter
            
        Returns:
            List of deployment metadata
            
        Raises:
            DeploymentQueryError: If listing fails
        """
        pass

    @abstractmethod
    async def update_deployment(
        self,
        deployment_id: str,
        configuration: Dict[str, Any]
    ) -> None:
        """Update deployment configuration.
        
        Args:
            deployment_id: Deployment identifier
            configuration: New configuration
            
        Raises:
            DeploymentUpdateError: If update fails
        """
        pass

    @abstractmethod
    async def stop_deployment(self, deployment_id: str) -> None:
        """Stop a model deployment.
        
        Args:
            deployment_id: Deployment identifier
            
        Raises:
            DeploymentStopError: If stopping fails
        """
        pass

    @abstractmethod
    async def get_deployment_endpoint(self, deployment_id: str) -> Optional[str]:
        """Get the endpoint URL for a deployment.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Endpoint URL if deployment is ready, None otherwise
            
        Raises:
            EndpointRetrievalError: If retrieval fails
        """
        pass


class MLOpsModelMonitoringPort(ABC):
    """Port for MLOps model monitoring operations.
    
    This interface defines the contract for monitoring model performance,
    data drift, and system health in production.
    """

    @abstractmethod
    async def log_prediction_request(
        self,
        model_version_id: str,
        request_data: Dict[str, Any],
        prediction: Any,
        response_time_ms: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log a prediction request for monitoring.
        
        Args:
            model_version_id: Model version identifier
            request_data: Input data for prediction
            prediction: Model prediction
            response_time_ms: Response time in milliseconds
            timestamp: Optional timestamp (defaults to now)
            
        Raises:
            PredictionLoggingError: If logging fails
        """
        pass

    @abstractmethod
    async def log_performance_metrics(
        self,
        model_version_id: str,
        metrics: ModelPerformanceMetrics
    ) -> None:
        """Log performance metrics for a model.
        
        Args:
            model_version_id: Model version identifier
            metrics: Performance metrics
            
        Raises:
            MetricsLoggingError: If logging fails
        """
        pass

    @abstractmethod
    async def get_performance_metrics(
        self,
        model_version_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[ModelPerformanceMetrics]:
        """Retrieve performance metrics for a time range.
        
        Args:
            model_version_id: Model version identifier
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of performance metrics
            
        Raises:
            MetricsRetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    async def detect_data_drift(
        self,
        model_version_id: str,
        reference_data: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect data drift for a model.
        
        Args:
            model_version_id: Model version identifier
            reference_data: Reference data (e.g., training data)
            current_data: Current production data
            
        Returns:
            Drift detection results
            
        Raises:
            DriftDetectionError: If drift detection fails
        """
        pass

    @abstractmethod
    async def create_alert(
        self,
        model_version_id: str,
        alert_type: str,
        message: str,
        severity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create an alert for model monitoring.
        
        Args:
            model_version_id: Model version identifier
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (low, medium, high, critical)
            metadata: Optional additional metadata
            
        Returns:
            Alert identifier
            
        Raises:
            AlertCreationError: If alert creation fails
        """
        pass

    @abstractmethod
    async def setup_monitoring_dashboard(
        self,
        model_version_id: str,
        dashboard_config: Dict[str, Any]
    ) -> str:
        """Setup monitoring dashboard for a model.
        
        Args:
            model_version_id: Model version identifier
            dashboard_config: Dashboard configuration
            
        Returns:
            Dashboard URL or identifier
            
        Raises:
            DashboardSetupError: If dashboard setup fails
        """
        pass


# Custom exceptions for MLOps operations
class MLOpsOperationError(Exception):
    """Base exception for MLOps operation errors."""
    pass


class ExperimentCreationError(MLOpsOperationError):
    """Exception raised during experiment creation."""
    pass


class ExperimentRetrievalError(MLOpsOperationError):
    """Exception raised during experiment retrieval."""
    pass


class ExperimentQueryError(MLOpsOperationError):
    """Exception raised during experiment queries."""
    pass


class RunCreationError(MLOpsOperationError):
    """Exception raised during run creation."""
    pass


class RunUpdateError(MLOpsOperationError):
    """Exception raised during run updates."""
    pass


class RunRetrievalError(MLOpsOperationError):
    """Exception raised during run retrieval."""
    pass


class RunQueryError(MLOpsOperationError):
    """Exception raised during run queries."""
    pass


class ParameterLoggingError(MLOpsOperationError):
    """Exception raised during parameter logging."""
    pass


class MetricLoggingError(MLOpsOperationError):
    """Exception raised during metric logging."""
    pass


class ArtifactLoggingError(MLOpsOperationError):
    """Exception raised during artifact logging."""
    pass


class ModelRegistrationError(MLOpsOperationError):
    """Exception raised during model registration."""
    pass


class VersionCreationError(MLOpsOperationError):
    """Exception raised during version creation."""
    pass


class VersionRetrievalError(MLOpsOperationError):
    """Exception raised during version retrieval."""
    pass


class VersionQueryError(MLOpsOperationError):
    """Exception raised during version queries."""
    pass


class StageTransitionError(MLOpsOperationError):
    """Exception raised during stage transitions."""
    pass


class DeploymentError(MLOpsOperationError):
    """Exception raised during model deployment."""
    pass


class DeploymentRetrievalError(MLOpsOperationError):
    """Exception raised during deployment retrieval."""
    pass


class DeploymentQueryError(MLOpsOperationError):
    """Exception raised during deployment queries."""
    pass


class DeploymentUpdateError(MLOpsOperationError):
    """Exception raised during deployment updates."""
    pass


class DeploymentStopError(MLOpsOperationError):
    """Exception raised during deployment stopping."""
    pass


class EndpointRetrievalError(MLOpsOperationError):
    """Exception raised during endpoint retrieval."""
    pass


class PredictionLoggingError(MLOpsOperationError):
    """Exception raised during prediction logging."""
    pass


class MetricsLoggingError(MLOpsOperationError):
    """Exception raised during metrics logging."""
    pass


class MetricsRetrievalError(MLOpsOperationError):
    """Exception raised during metrics retrieval."""
    pass


class DriftDetectionError(MLOpsOperationError):
    """Exception raised during drift detection."""
    pass


class AlertCreationError(MLOpsOperationError):
    """Exception raised during alert creation."""
    pass


class DashboardSetupError(MLOpsOperationError):
    """Exception raised during dashboard setup."""
    pass