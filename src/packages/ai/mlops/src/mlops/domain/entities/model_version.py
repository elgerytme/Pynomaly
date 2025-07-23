"""Model Version domain entity for MLOps package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4


class ModelVersionStatus(str, Enum):
    """Model version status enumeration."""
    
    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Model Version domain entity.
    
    Represents a specific version of a machine learning model,
    following Domain-Driven Design principles.
    """
    
    id: UUID = field(default_factory=uuid4)
    model_id: UUID = field(default_factory=uuid4)
    version: str = "1.0.0"
    status: ModelVersionStatus = ModelVersionStatus.CREATED
    
    # Version metadata
    description: str = ""
    changelog: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Model artifacts
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    requirements_path: Optional[str] = None
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Training information
    training_config: Dict[str, Any] = field(default_factory=dict)
    training_dataset_id: Optional[UUID] = None
    validation_dataset_id: Optional[UUID] = None
    
    # Model signature
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Deployment tracking
    deployments: List[UUID] = field(default_factory=list)
    is_production_ready: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    trained_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    
    # User tracking
    created_by: str = ""
    team: str = ""
    
    # Model lineage
    parent_version_id: Optional[UUID] = None
    experiment_id: Optional[UUID] = None
    
    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        self._validate_model_version()
    
    def _validate_model_version(self) -> None:
        """Validate model version state and business rules."""
        if not self.version or len(self.version.strip()) == 0:
            raise ValueError("Model version cannot be empty")
        
        if not self._is_valid_version_format(self.version):
            raise ValueError(f"Invalid version format: {self.version}")
        
        if self.status not in ModelVersionStatus:
            raise ValueError(f"Invalid model version status: {self.status}")
    
    def _is_valid_version_format(self, version: str) -> bool:
        """Validate semantic version format (x.y.z)."""
        parts = version.split('.')
        if len(parts) != 3:
            return False
        
        try:
            for part in parts:
                int(part)
            return True
        except ValueError:
            return False
    
    def start_training(self, training_config: Dict[str, Any]) -> None:
        """Start model training.
        
        Args:
            training_config: Configuration for training
            
        Raises:
            ValueError: If model version is not in valid state for training
        """
        if self.status != ModelVersionStatus.CREATED:
            raise ValueError(f"Cannot start training from {self.status} status")
        
        self.status = ModelVersionStatus.TRAINING
        self.training_config = training_config
        self.updated_at = datetime.utcnow()
    
    def complete_training(
        self, 
        model_path: str,
        metrics: Dict[str, float],
        config_path: Optional[str] = None
    ) -> None:
        """Complete model training.
        
        Args:
            model_path: Path to trained model artifact
            metrics: Training metrics
            config_path: Optional path to model configuration
            
        Raises:
            ValueError: If model version is not in TRAINING status
        """
        if self.status != ModelVersionStatus.TRAINING:
            raise ValueError(f"Cannot complete training from {self.status} status")
        
        self.status = ModelVersionStatus.TRAINED
        self.model_path = model_path
        self.config_path = config_path
        self.metrics = metrics
        self.trained_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def start_validation(self, validation_dataset_id: UUID) -> None:
        """Start model validation.
        
        Args:
            validation_dataset_id: ID of validation dataset
            
        Raises:
            ValueError: If model version is not trained
        """
        if self.status != ModelVersionStatus.TRAINED:
            raise ValueError(f"Cannot start validation from {self.status} status")
        
        self.status = ModelVersionStatus.VALIDATING
        self.validation_dataset_id = validation_dataset_id
        self.updated_at = datetime.utcnow()
    
    def complete_validation(self, validation_metrics: Dict[str, float]) -> None:
        """Complete model validation.
        
        Args:
            validation_metrics: Validation performance metrics
            
        Raises:
            ValueError: If model version is not validating
        """
        if self.status != ModelVersionStatus.VALIDATING:
            raise ValueError(f"Cannot complete validation from {self.status} status")
        
        self.status = ModelVersionStatus.VALIDATED
        self.validation_metrics = validation_metrics
        self.updated_at = datetime.utcnow()
        
        # Check if model is production ready based on validation metrics
        self._evaluate_production_readiness()
    
    def _evaluate_production_readiness(self) -> None:
        """Evaluate if model version is ready for production."""
        # This is a simplified check - in practice, this would involve
        # more sophisticated business rules and thresholds
        if self.validation_metrics:
            accuracy = self.validation_metrics.get('accuracy', 0.0)
            precision = self.validation_metrics.get('precision', 0.0)
            recall = self.validation_metrics.get('recall', 0.0)
            
            # Example thresholds - these would be configurable in practice
            min_accuracy = 0.8
            min_precision = 0.7
            min_recall = 0.7
            
            self.is_production_ready = (
                accuracy >= min_accuracy and
                precision >= min_precision and
                recall >= min_recall
            )
    
    def deploy(self, deployment_id: UUID) -> None:
        """Mark model version as deployed.
        
        Args:
            deployment_id: ID of the deployment
        """
        if self.status not in [ModelVersionStatus.VALIDATED, ModelVersionStatus.DEPLOYED]:
            raise ValueError(f"Cannot deploy from {self.status} status")
        
        if deployment_id not in self.deployments:
            self.deployments.append(deployment_id)
        
        if self.status != ModelVersionStatus.DEPLOYED:
            self.status = ModelVersionStatus.DEPLOYED
            self.deployed_at = datetime.utcnow()
        
        self.updated_at = datetime.utcnow()
    
    def undeploy(self, deployment_id: UUID) -> None:
        """Remove deployment from model version.
        
        Args:
            deployment_id: ID of the deployment to remove
        """
        if deployment_id in self.deployments:
            self.deployments.remove(deployment_id)
            
            # If no more deployments, change status back to validated
            if not self.deployments and self.status == ModelVersionStatus.DEPLOYED:
                self.status = ModelVersionStatus.VALIDATED
            
            self.updated_at = datetime.utcnow()
    
    def archive(self, reason: str = "") -> None:
        """Archive the model version.
        
        Args:
            reason: Optional reason for archiving
        """
        if self.status == ModelVersionStatus.ARCHIVED:
            raise ValueError("Model version is already archived")
        
        self.status = ModelVersionStatus.ARCHIVED
        self.archived_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        if reason:
            self.training_config["archive_reason"] = reason
    
    def fail(self, error_message: str) -> None:
        """Mark model version as failed.
        
        Args:
            error_message: Error message describing the failure
        """
        self.status = ModelVersionStatus.FAILED
        self.updated_at = datetime.utcnow()
        self.training_config["error_message"] = error_message
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model version.
        
        Args:
            tag: Tag to add
        """
        if not tag or not isinstance(tag, str):
            raise ValueError("Tag must be a non-empty string")
        
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model version.
        
        Args:
            tag: Tag to remove
        """
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """Update model metrics.
        
        Args:
            new_metrics: New metrics to add or update
        """
        self.metrics.update(new_metrics)
        self.updated_at = datetime.utcnow()
    
    def set_model_signature(
        self, 
        input_schema: Dict[str, Any], 
        output_schema: Dict[str, Any]
    ) -> None:
        """Set the model input/output signature.
        
        Args:
            input_schema: Input schema specification
            output_schema: Output schema specification
        """
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.updated_at = datetime.utcnow()
    
    @property
    def is_deployed(self) -> bool:
        """Check if model version is currently deployed."""
        return self.status == ModelVersionStatus.DEPLOYED and bool(self.deployments)
    
    @property
    def is_trainable(self) -> bool:
        """Check if model version can be trained."""
        return self.status == ModelVersionStatus.CREATED
    
    @property
    def is_deployable(self) -> bool:
        """Check if model version can be deployed."""
        return (
            self.status == ModelVersionStatus.VALIDATED and 
            self.is_production_ready and 
            self.model_path is not None
        )
    
    @property
    def deployment_count(self) -> int:
        """Get number of active deployments."""
        return len(self.deployments)
    
    @property
    def age_days(self) -> float:
        """Get age of model version in days."""
        return (datetime.utcnow() - self.created_at).total_seconds() / 86400
    
    def get_best_metric(self, metric_name: str, higher_is_better: bool = True) -> Optional[float]:
        """Get the best value for a specific metric.
        
        Args:
            metric_name: Name of the metric
            higher_is_better: Whether higher values are better
            
        Returns:
            Best metric value from training or validation metrics
        """
        training_value = self.metrics.get(metric_name)
        validation_value = self.validation_metrics.get(metric_name)
        
        values = [v for v in [training_value, validation_value] if v is not None]
        if not values:
            return None
        
        return max(values) if higher_is_better else min(values)
    
    def compare_with(self, other_version: ModelVersion, metric_name: str) -> Optional[float]:
        """Compare this version with another version on a specific metric.
        
        Args:
            other_version: Other model version to compare with
            metric_name: Metric to compare
            
        Returns:
            Difference (this - other), None if metric not available in both
        """
        this_value = self.get_best_metric(metric_name)
        other_value = other_version.get_best_metric(metric_name)
        
        if this_value is None or other_value is None:
            return None
        
        return this_value - other_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model version to dictionary representation."""
        return {
            "id": str(self.id),
            "model_id": str(self.model_id),
            "version": self.version,
            "status": self.status.value,
            "description": self.description,
            "changelog": self.changelog,
            "tags": self.tags,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "requirements_path": self.requirements_path,
            "metrics": self.metrics,
            "validation_metrics": self.validation_metrics,
            "training_config": self.training_config,
            "training_dataset_id": str(self.training_dataset_id) if self.training_dataset_id else None,
            "validation_dataset_id": str(self.validation_dataset_id) if self.validation_dataset_id else None,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "deployments": [str(dep_id) for dep_id in self.deployments],
            "is_production_ready": self.is_production_ready,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "created_by": self.created_by,
            "team": self.team,
            "parent_version_id": str(self.parent_version_id) if self.parent_version_id else None,
            "experiment_id": str(self.experiment_id) if self.experiment_id else None,
            "is_deployed": self.is_deployed,
            "is_deployable": self.is_deployable,
            "deployment_count": self.deployment_count,
            "age_days": self.age_days,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelVersion:
        """Create model version from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ModelVersion instance
        """
        model_version = cls(
            id=UUID(data["id"]) if data.get("id") else uuid4(),
            model_id=UUID(data["model_id"]) if data.get("model_id") else uuid4(),
            version=data.get("version", "1.0.0"),
            status=ModelVersionStatus(data.get("status", ModelVersionStatus.CREATED)),
            description=data.get("description", ""),
            changelog=data.get("changelog", ""),
            tags=data.get("tags", []),
            model_path=data.get("model_path"),
            config_path=data.get("config_path"),
            requirements_path=data.get("requirements_path"),
            metrics=data.get("metrics", {}),
            validation_metrics=data.get("validation_metrics", {}),
            training_config=data.get("training_config", {}),
            training_dataset_id=UUID(data["training_dataset_id"]) if data.get("training_dataset_id") else None,
            validation_dataset_id=UUID(data["validation_dataset_id"]) if data.get("validation_dataset_id") else None,
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            deployments=[UUID(dep_id) for dep_id in data.get("deployments", [])],
            is_production_ready=data.get("is_production_ready", False),
            created_by=data.get("created_by", ""),
            team=data.get("team", ""),
            parent_version_id=UUID(data["parent_version_id"]) if data.get("parent_version_id") else None,
            experiment_id=UUID(data["experiment_id"]) if data.get("experiment_id") else None,
        )
        
        # Handle datetime fields
        if data.get("created_at"):
            model_version.created_at = datetime.fromisoformat(data["created_at"])
        
        if data.get("updated_at"):
            model_version.updated_at = datetime.fromisoformat(data["updated_at"])
        
        if data.get("trained_at"):
            model_version.trained_at = datetime.fromisoformat(data["trained_at"])
        
        if data.get("deployed_at"):
            model_version.deployed_at = datetime.fromisoformat(data["deployed_at"])
        
        if data.get("archived_at"):
            model_version.archived_at = datetime.fromisoformat(data["archived_at"])
        
        return model_version
    
    def __str__(self) -> str:
        """String representation of the model version."""
        return f"ModelVersion(id={self.id.hex[:8]}, version='{self.version}', status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model version."""
        return (
            f"ModelVersion("
            f"id={self.id}, "
            f"model_id={self.model_id}, "
            f"version='{self.version}', "
            f"status={self.status.value}, "
            f"deployments={len(self.deployments)}"
            f")"
        )