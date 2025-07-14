"""Data Science Model entity for versioning and metadata management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_entity import BaseEntity


class ModelType(str, Enum):
    """Types of data science models."""
    
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    
    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"


class DataScienceModel(BaseEntity):
    """Entity representing a data science model with versioning and metadata.
    
    This entity manages the complete lifecycle of data science models including
    training, validation, deployment, and monitoring.
    
    Attributes:
        name: Human-readable name for the model
        model_type: Type of model (statistical, ML, DL, etc.)
        algorithm: Algorithm or framework used
        version_number: Semantic version of the model
        status: Current lifecycle status
        description: Detailed description of the model
        hyperparameters: Model configuration parameters
        training_dataset_id: Reference to training dataset
        validation_dataset_id: Reference to validation dataset
        performance_metrics: Model performance indicators
        features: List of features used by the model
        target_variable: Name of the target variable
        model_size_bytes: Size of the serialized model
        training_duration_seconds: Time taken to train the model
        trained_at: Timestamp when training completed
        deployed_at: Timestamp when model was deployed
        artifact_uri: Location of the serialized model
        experiment_id: Reference to the experiment that produced this model
        parent_model_id: Reference to parent model (for model evolution)
        tags: Searchable tags for model organization
        business_context: Business use case and requirements
        monitoring_config: Configuration for model monitoring
    """
    
    name: str = Field(..., min_length=1, max_length=255)
    model_type: ModelType
    algorithm: str = Field(..., min_length=1, max_length=100)
    version_number: str = Field(..., regex=r'^\d+\.\d+\.\d+$')
    status: ModelStatus = Field(default=ModelStatus.DRAFT)
    description: Optional[str] = Field(None, max_length=2000)
    
    # Model configuration
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    features: list[str] = Field(default_factory=list)
    target_variable: Optional[str] = None
    
    # Dataset references
    training_dataset_id: Optional[str] = None
    validation_dataset_id: Optional[str] = None
    
    # Performance metrics
    performance_metrics: dict[str, float] = Field(default_factory=dict)
    
    # Model artifacts
    model_size_bytes: Optional[int] = Field(None, ge=0)
    artifact_uri: Optional[str] = None
    
    # Timing information
    training_duration_seconds: Optional[float] = Field(None, ge=0)
    trained_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    
    # Experiment tracking
    experiment_id: Optional[str] = None
    parent_model_id: Optional[str] = None
    
    # Organization and context
    tags: list[str] = Field(default_factory=list)
    business_context: dict[str, Any] = Field(default_factory=dict)
    monitoring_config: dict[str, Any] = Field(default_factory=dict)
    
    @validator('features')
    def validate_features(cls, v: list[str]) -> list[str]:
        """Validate feature list."""
        if len(v) != len(set(v)):
            raise ValueError("Feature names must be unique")
        
        for feature in v:
            if not feature.strip():
                raise ValueError("Feature names cannot be empty")
                
        return v
    
    @validator('tags')
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate tags list."""
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    @validator('performance_metrics')
    def validate_performance_metrics(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate performance metrics."""
        for metric_name, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Performance metric '{metric_name}' must be numeric")
            
            if not metric_name.strip():
                raise ValueError("Performance metric names cannot be empty")
                
        return v
    
    @validator('hyperparameters')
    def validate_hyperparameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate hyperparameters."""
        for param_name in v.keys():
            if not param_name.strip():
                raise ValueError("Hyperparameter names cannot be empty")
                
        return v
    
    def mark_as_training(self) -> None:
        """Mark model as currently training."""
        self.status = ModelStatus.TRAINING
        self.mark_as_updated()
    
    def mark_as_trained(self, performance_metrics: dict[str, float], 
                       training_duration: float) -> None:
        """Mark model as successfully trained."""
        self.status = ModelStatus.TRAINED
        self.performance_metrics.update(performance_metrics)
        self.training_duration_seconds = training_duration
        self.trained_at = datetime.utcnow()
        self.mark_as_updated()
    
    def mark_as_validated(self, validation_metrics: dict[str, float]) -> None:
        """Mark model as validated with metrics."""
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model must be trained before validation")
            
        self.status = ModelStatus.VALIDATED
        self.performance_metrics.update(validation_metrics)
        self.mark_as_updated()
    
    def mark_as_deployed(self, deployment_config: dict[str, Any]) -> None:
        """Mark model as deployed."""
        if self.status not in [ModelStatus.VALIDATED, ModelStatus.DEPLOYED]:
            raise ValueError("Model must be validated before deployment")
            
        self.status = ModelStatus.DEPLOYED
        self.deployed_at = datetime.utcnow()
        self.monitoring_config.update(deployment_config)
        self.mark_as_updated()
    
    def mark_as_failed(self, error_message: str) -> None:
        """Mark model as failed with error details."""
        self.status = ModelStatus.FAILED
        self.metadata["error_message"] = error_message
        self.metadata["failed_at"] = datetime.utcnow().isoformat()
        self.mark_as_updated()
    
    def add_performance_metric(self, name: str, value: float) -> None:
        """Add a performance metric."""
        if not name.strip():
            raise ValueError("Metric name cannot be empty")
            
        self.performance_metrics[name] = value
        self.mark_as_updated()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        tag = tag.strip().lower()
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.mark_as_updated()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model."""
        tag = tag.strip().lower()
        if tag in self.tags:
            self.tags.remove(tag)
            self.mark_as_updated()
    
    def get_metric(self, name: str) -> Optional[float]:
        """Get a specific performance metric."""
        return self.performance_metrics.get(name)
    
    def is_deployed(self) -> bool:
        """Check if model is currently deployed."""
        return self.status == ModelStatus.DEPLOYED
    
    def is_trainable(self) -> bool:
        """Check if model can be trained."""
        return self.status in [ModelStatus.DRAFT, ModelStatus.FAILED]
    
    def is_deployable(self) -> bool:
        """Check if model can be deployed."""
        return self.status == ModelStatus.VALIDATED
    
    def get_lineage_depth(self) -> int:
        """Get the depth in the model lineage chain."""
        # This would need to be implemented with repository access
        # For now, return 0 if no parent, 1 if has parent
        return 1 if self.parent_model_id else 0
    
    def calculate_model_score(self) -> float:
        """Calculate an overall model quality score."""
        if not self.performance_metrics:
            return 0.0
            
        # Simple averaging of available metrics
        # In practice, this would be more sophisticated
        scores = [v for v in self.performance_metrics.values() if 0 <= v <= 1]
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_training_summary(self) -> dict[str, Any]:
        """Get a summary of model training information."""
        return {
            "model_id": str(self.id),
            "name": self.name,
            "algorithm": self.algorithm,
            "version": self.version_number,
            "status": self.status.value,
            "features_count": len(self.features),
            "training_duration": self.training_duration_seconds,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "performance_metrics": self.performance_metrics,
            "model_score": self.calculate_model_score(),
        }
    
    def validate_invariants(self) -> None:
        """Validate domain invariants."""
        super().validate_invariants()
        
        # Business rule: Deployed models must have validation metrics
        if self.status == ModelStatus.DEPLOYED and not self.performance_metrics:
            raise ValueError("Deployed models must have performance metrics")
        
        # Business rule: Trained models must have training timestamp
        if self.status in [ModelStatus.TRAINED, ModelStatus.VALIDATED, ModelStatus.DEPLOYED]:
            if not self.trained_at:
                raise ValueError("Trained models must have training timestamp")
        
        # Business rule: Models with target variable must be supervised learning
        if self.target_variable and self.model_type in [ModelType.CLUSTERING]:
            raise ValueError("Clustering models should not have target variable")
        
        # Business rule: Feature list cannot be empty for trained models
        if self.status in [ModelStatus.TRAINED, ModelStatus.VALIDATED, ModelStatus.DEPLOYED]:
            if not self.features:
                raise ValueError("Trained models must have feature list")