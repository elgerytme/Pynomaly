"""
Training Data Transfer Objects

DTOs for training requests, responses, and configuration that provide
type-safe data transfer between application layers with comprehensive
validation and serialization support.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pynomaly.application.dto.optimization_dto import OptimizationConfigDTO
from pynomaly.domain.value_objects.hyperparameters import (
    HyperparameterSet,
    HyperparameterSpace,
)


class TrainingTrigger(Enum):
    """Training trigger types."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    DATA_DRIFT = "data_drift"
    NEW_DATA = "new_data"
    API_REQUEST = "api_request"


class TrainingPriorityLevel(Enum):
    """Training priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ResourceConstraintsDTO:
    """Resource constraints for training."""

    # Time constraints
    max_training_time_seconds: Optional[int] = None
    max_time_per_trial: Optional[int] = None

    # Resource limits
    max_memory_mb: Optional[int] = None
    max_cpu_cores: Optional[int] = None
    max_gpu_memory_mb: Optional[int] = None
    enable_gpu: bool = False

    # Concurrency
    n_jobs: int = 1
    max_concurrent_trials: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_training_time_seconds": self.max_training_time_seconds,
            "max_time_per_trial": self.max_time_per_trial,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_cores": self.max_cpu_cores,
            "max_gpu_memory_mb": self.max_gpu_memory_mb,
            "enable_gpu": self.enable_gpu,
            "n_jobs": self.n_jobs,
            "max_concurrent_trials": self.max_concurrent_trials,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceConstraintsDTO":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AutoMLConfigDTO:
    """AutoML configuration for training."""

    enable_automl: bool = True
    optimization_objective: str = "auc"  # auc, precision, recall, f1_score, etc.
    max_algorithms: int = 3
    enable_ensemble: bool = True
    ensemble_method: str = "voting"  # voting, stacking, blending

    # Algorithm selection
    algorithm_whitelist: Optional[List[str]] = None
    algorithm_blacklist: Optional[List[str]] = None

    # Model selection criteria
    model_selection_strategy: str = (
        "best_single_metric"  # best_single_metric, weighted_score, pareto_optimal
    )
    metric_weights: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_automl": self.enable_automl,
            "optimization_objective": self.optimization_objective,
            "max_algorithms": self.max_algorithms,
            "enable_ensemble": self.enable_ensemble,
            "ensemble_method": self.ensemble_method,
            "algorithm_whitelist": self.algorithm_whitelist,
            "algorithm_blacklist": self.algorithm_blacklist,
            "model_selection_strategy": self.model_selection_strategy,
            "metric_weights": self.metric_weights,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoMLConfigDTO":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ValidationConfigDTO:
    """Validation configuration for training."""

    validation_strategy: str = "holdout"  # holdout, cross_validation, time_series_split
    validation_split: float = 0.2
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, kfold, group, time_series

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_metric: str = "validation_score"
    early_stopping_threshold: float = 0.001

    # Data validation
    enable_data_validation: bool = True
    min_samples_required: int = 100
    max_missing_ratio: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validation_strategy": self.validation_strategy,
            "validation_split": self.validation_split,
            "cv_folds": self.cv_folds,
            "cv_strategy": self.cv_strategy,
            "enable_early_stopping": self.enable_early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_metric": self.early_stopping_metric,
            "early_stopping_threshold": self.early_stopping_threshold,
            "enable_data_validation": self.enable_data_validation,
            "min_samples_required": self.min_samples_required,
            "max_missing_ratio": self.max_missing_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationConfigDTO":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class NotificationConfigDTO:
    """Notification configuration for training."""

    enable_notifications: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["websocket"])

    # Email notifications
    email_recipients: List[str] = field(default_factory=list)
    email_on_completion: bool = True
    email_on_failure: bool = True

    # Webhook notifications
    webhook_urls: List[str] = field(default_factory=list)
    webhook_events: List[str] = field(default_factory=lambda: ["completion", "failure"])

    # Slack notifications
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_notifications": self.enable_notifications,
            "notification_channels": self.notification_channels,
            "email_recipients": self.email_recipients,
            "email_on_completion": self.email_on_completion,
            "email_on_failure": self.email_on_failure,
            "webhook_urls": self.webhook_urls,
            "webhook_events": self.webhook_events,
            "slack_webhook_url": self.slack_webhook_url,
            "slack_channel": self.slack_channel,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationConfigDTO":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingConfigDTO:
    """
    Complete training configuration DTO.

    Comprehensive configuration object that contains all settings needed
    for automated training including AutoML, validation, optimization,
    and resource management.
    """

    # Core training settings
    experiment_name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Dataset and model settings
    dataset_id: Optional[str] = None
    detector_id: Optional[UUID] = None
    target_algorithms: Optional[List[str]] = None

    # Configuration components
    automl_config: AutoMLConfigDTO = field(default_factory=AutoMLConfigDTO)
    validation_config: ValidationConfigDTO = field(default_factory=ValidationConfigDTO)
    optimization_config: Optional[OptimizationConfigDTO] = None
    resource_constraints: ResourceConstraintsDTO = field(
        default_factory=ResourceConstraintsDTO
    )
    notification_config: NotificationConfigDTO = field(
        default_factory=NotificationConfigDTO
    )

    # Scheduling settings
    schedule_cron: Optional[str] = None
    schedule_enabled: bool = False

    # Performance monitoring
    performance_monitoring_enabled: bool = True
    retrain_threshold: float = 0.05
    performance_window_days: int = 7

    # Model management
    auto_deploy_best_model: bool = False
    model_versioning_enabled: bool = True
    keep_model_versions: int = 5

    # Advanced settings
    enable_model_explainability: bool = True
    enable_drift_detection: bool = True
    enable_feature_selection: bool = True

    # Metadata
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    priority: TrainingPriorityLevel = TrainingPriorityLevel.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags,
            "dataset_id": self.dataset_id,
            "detector_id": str(self.detector_id) if self.detector_id else None,
            "target_algorithms": self.target_algorithms,
            "automl_config": self.automl_config.to_dict(),
            "validation_config": self.validation_config.to_dict(),
            "optimization_config": (
                self.optimization_config.to_dict() if self.optimization_config else None
            ),
            "resource_constraints": self.resource_constraints.to_dict(),
            "notification_config": self.notification_config.to_dict(),
            "schedule_cron": self.schedule_cron,
            "schedule_enabled": self.schedule_enabled,
            "performance_monitoring_enabled": self.performance_monitoring_enabled,
            "retrain_threshold": self.retrain_threshold,
            "performance_window_days": self.performance_window_days,
            "auto_deploy_best_model": self.auto_deploy_best_model,
            "model_versioning_enabled": self.model_versioning_enabled,
            "keep_model_versions": self.keep_model_versions,
            "enable_model_explainability": self.enable_model_explainability,
            "enable_drift_detection": self.enable_drift_detection,
            "enable_feature_selection": self.enable_feature_selection,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "priority": self.priority.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfigDTO":
        """Create from dictionary representation."""
        data = data.copy()

        # Convert UUID field
        if "detector_id" in data and data["detector_id"]:
            data["detector_id"] = UUID(data["detector_id"])

        # Convert datetime field
        if "created_at" in data and data["created_at"]:
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        # Convert enum field
        if "priority" in data:
            data["priority"] = TrainingPriorityLevel(data["priority"])

        # Convert component configurations
        if "automl_config" in data:
            data["automl_config"] = AutoMLConfigDTO.from_dict(data["automl_config"])

        if "validation_config" in data:
            data["validation_config"] = ValidationConfigDTO.from_dict(
                data["validation_config"]
            )

        if "optimization_config" in data and data["optimization_config"]:
            data["optimization_config"] = OptimizationConfigDTO.from_dict(
                data["optimization_config"]
            )

        if "resource_constraints" in data:
            data["resource_constraints"] = ResourceConstraintsDTO.from_dict(
                data["resource_constraints"]
            )

        if "notification_config" in data:
            data["notification_config"] = NotificationConfigDTO.from_dict(
                data["notification_config"]
            )

        return cls(**data)


@dataclass
class TrainingRequestDTO:
    """Training request DTO."""

    # Required fields
    detector_id: UUID
    dataset_id: str

    # Configuration
    config: Optional[TrainingConfigDTO] = None

    # Trigger information
    trigger_type: TrainingTrigger = TrainingTrigger.MANUAL
    trigger_metadata: Dict[str, Any] = field(default_factory=dict)

    # Request metadata
    requested_by: Optional[str] = None
    requested_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detector_id": str(self.detector_id),
            "dataset_id": self.dataset_id,
            "config": self.config.to_dict() if self.config else None,
            "trigger_type": self.trigger_type.value,
            "trigger_metadata": self.trigger_metadata,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingRequestDTO":
        """Create from dictionary."""
        data = data.copy()

        # Convert UUID and datetime fields
        data["detector_id"] = UUID(data["detector_id"])
        if "requested_at" in data:
            data["requested_at"] = datetime.fromisoformat(data["requested_at"])

        # Convert enum field
        if "trigger_type" in data:
            data["trigger_type"] = TrainingTrigger(data["trigger_type"])

        # Convert config
        if "config" in data and data["config"]:
            data["config"] = TrainingConfigDTO.from_dict(data["config"])

        return cls(**data)


@dataclass
class ModelMetricsDTO:
    """Model evaluation metrics DTO."""

    # Core metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None

    # Anomaly detection specific
    anomaly_score_mean: Optional[float] = None
    anomaly_score_std: Optional[float] = None
    contamination_detected: Optional[float] = None

    # Cross-validation metrics
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None

    # Training metrics
    training_time_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    model_size_mb: Optional[float] = None

    # Additional metrics
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "anomaly_score_mean": self.anomaly_score_mean,
            "anomaly_score_std": self.anomaly_score_std,
            "contamination_detected": self.contamination_detected,
            "cv_scores": self.cv_scores,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "training_time_seconds": self.training_time_seconds,
            "inference_time_ms": self.inference_time_ms,
            "model_size_mb": self.model_size_mb,
            "confusion_matrix": self.confusion_matrix,
            "feature_importance": self.feature_importance,
            "additional_metrics": self.additional_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetricsDTO":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingResultDTO:
    """Training result DTO."""

    # Identification
    training_id: str
    detector_id: UUID
    dataset_id: str

    # Training metadata
    experiment_name: Optional[str] = None
    trigger_type: TrainingTrigger = TrainingTrigger.MANUAL
    status: str = "completed"

    # Results
    best_algorithm: Optional[str] = None
    best_hyperparameters: Optional[Dict[str, Any]] = None
    best_metrics: Optional[ModelMetricsDTO] = None

    # Model information
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    model_path: Optional[str] = None

    # Training process
    total_trials: Optional[int] = None
    successful_trials: Optional[int] = None
    training_duration_seconds: Optional[float] = None
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)

    # Performance comparison
    baseline_metrics: Optional[ModelMetricsDTO] = None
    performance_improvement: Optional[float] = None

    # Resource usage
    peak_memory_mb: Optional[float] = None
    total_cpu_hours: Optional[float] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Messages and logs
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "training_id": self.training_id,
            "detector_id": str(self.detector_id),
            "dataset_id": self.dataset_id,
            "experiment_name": self.experiment_name,
            "trigger_type": self.trigger_type.value,
            "status": self.status,
            "best_algorithm": self.best_algorithm,
            "best_hyperparameters": self.best_hyperparameters,
            "best_metrics": self.best_metrics.to_dict() if self.best_metrics else None,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "training_duration_seconds": self.training_duration_seconds,
            "optimization_history": self.optimization_history,
            "baseline_metrics": (
                self.baseline_metrics.to_dict() if self.baseline_metrics else None
            ),
            "performance_improvement": self.performance_improvement,
            "peak_memory_mb": self.peak_memory_mb,
            "total_cpu_hours": self.total_cpu_hours,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "error_message": self.error_message,
            "warnings": self.warnings,
            "logs": self.logs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingResultDTO":
        """Create from dictionary."""
        data = data.copy()

        # Convert UUID field
        data["detector_id"] = UUID(data["detector_id"])

        # Convert datetime fields
        for field_name in ["started_at", "completed_at"]:
            if field_name in data and data[field_name]:
                data[field_name] = datetime.fromisoformat(data[field_name])

        # Convert enum field
        if "trigger_type" in data:
            data["trigger_type"] = TrainingTrigger(data["trigger_type"])

        # Convert metrics
        if "best_metrics" in data and data["best_metrics"]:
            data["best_metrics"] = ModelMetricsDTO.from_dict(data["best_metrics"])

        if "baseline_metrics" in data and data["baseline_metrics"]:
            data["baseline_metrics"] = ModelMetricsDTO.from_dict(
                data["baseline_metrics"]
            )

        return cls(**data)


@dataclass
class TrainingStatusDTO:
    """Training status DTO for real-time updates."""

    training_id: str
    status: str
    progress_percentage: float
    current_step: str

    # Current operation details
    current_algorithm: Optional[str] = None
    current_trial: Optional[int] = None
    total_trials: Optional[int] = None
    current_score: Optional[float] = None
    best_score: Optional[float] = None

    # Timing
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None

    # Resource usage
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    # Messages
    current_message: Optional[str] = None
    recent_logs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "training_id": self.training_id,
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "current_step": self.current_step,
            "current_algorithm": self.current_algorithm,
            "current_trial": self.current_trial,
            "total_trials": self.total_trials,
            "current_score": self.current_score,
            "best_score": self.best_score,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "estimated_completion": (
                self.estimated_completion.isoformat()
                if self.estimated_completion
                else None
            ),
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "current_message": self.current_message,
            "recent_logs": self.recent_logs,
            "warnings": self.warnings,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingStatusDTO":
        """Create from dictionary."""
        data = data.copy()

        # Convert datetime fields
        for field_name in ["started_at", "estimated_completion"]:
            if field_name in data and data[field_name]:
                data[field_name] = datetime.fromisoformat(data[field_name])

        # Remove computed fields
        data.pop("timestamp", None)

        return cls(**data)
