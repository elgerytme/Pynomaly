"""ML Model domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class ModelStatus(str, Enum):
    """Model status enumeration."""
    
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"


class ModelType(str, Enum):
    """Model type enumeration."""
    
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    FORECASTING = "forecasting"
    RECOMMENDATION = "recommendation"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    auc_roc: float | None = None
    rmse: float | None = None
    mae: float | None = None
    r2_score: float | None = None
    custom_metrics: dict[str, float] = field(default_factory=dict)
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a custom metric."""
        self.custom_metrics[name] = value
    
    def get_primary_metric(self, model_type: ModelType) -> float | None:
        """Get the primary metric for the model type."""
        primary_metrics = {
            ModelType.CLASSIFICATION: self.f1_score or self.accuracy,
            ModelType.REGRESSION: self.rmse or self.mae,
            ModelType.CLUSTERING: self.custom_metrics.get("silhouette_score"),
            ModelType.ANOMALY_DETECTION: self.auc_roc or self.f1_score,
            ModelType.FORECASTING: self.rmse or self.mae,
            ModelType.RECOMMENDATION: self.custom_metrics.get("ndcg"),
        }
        return primary_metrics.get(model_type)


@dataclass
class HyperParameters:
    """Model hyperparameters."""
    
    parameters: dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a hyperparameter value."""
        return self.parameters.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a hyperparameter value."""
        self.parameters[key] = value
    
    def update(self, params: dict[str, Any]) -> None:
        """Update multiple hyperparameters."""
        self.parameters.update(params)


@dataclass
class Model:
    """ML Model domain entity."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    model_type: ModelType = ModelType.CLASSIFICATION
    algorithm: str = ""
    version: str = "1.0.0"
    status: ModelStatus = ModelStatus.TRAINING
    
    # Model artifacts
    hyperparameters: HyperParameters = field(default_factory=HyperParameters)
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    
    # Metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    trained_at: datetime | None = None
    deployed_at: datetime | None = None
    
    # Training information
    training_dataset_id: UUID | None = None
    training_duration_seconds: float | None = None
    training_samples: int | None = None
    validation_samples: int | None = None
    
    # Deployment information
    deployment_environment: str | None = None
    deployment_version: str | None = None
    endpoint_url: str | None = None
    
    # Model artifacts paths
    model_path: str | None = None
    config_path: str | None = None
    
    # Performance tracking
    prediction_count: int = 0
    last_prediction_at: datetime | None = None
    
    def update_status(self, status: ModelStatus) -> None:
        """Update model status with timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()
        
        if status == ModelStatus.TRAINED:
            self.trained_at = datetime.utcnow()
        elif status == ModelStatus.DEPLOYED:
            self.deployed_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def update_metrics(self, metrics: ModelMetrics) -> None:
        """Update model metrics."""
        self.metrics = metrics
        self.updated_at = datetime.utcnow()
    
    def update_hyperparameters(self, params: dict[str, Any]) -> None:
        """Update model hyperparameters."""
        self.hyperparameters.update(params)
        self.updated_at = datetime.utcnow()
    
    def record_prediction(self) -> None:
        """Record a prediction event."""
        self.prediction_count += 1
        self.last_prediction_at = datetime.utcnow()
    
    def is_ready_for_deployment(self) -> bool:
        """Check if model is ready for deployment."""
        return (
            self.status == ModelStatus.TRAINED and
            self.model_path is not None and
            self.metrics.get_primary_metric(self.model_type) is not None
        )
    
    def get_age_days(self) -> int:
        """Get model age in days."""
        return (datetime.utcnow() - self.created_at).days
    
    def summary(self) -> dict[str, Any]:
        """Get model summary."""
        primary_metric = self.metrics.get_primary_metric(self.model_type)
        
        return {
            "id": str(self.id),
            "name": self.name,
            "type": self.model_type.value,
            "algorithm": self.algorithm,
            "version": self.version,
            "status": self.status.value,
            "primary_metric": primary_metric,
            "prediction_count": self.prediction_count,
            "age_days": self.get_age_days(),
            "tags": self.tags,
            "deployment_environment": self.deployment_environment,
        }