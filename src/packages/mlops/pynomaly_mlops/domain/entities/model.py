"""Model Entity

Core domain entity representing a machine learning model with versioning,
lifecycle management, and metadata tracking capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from pynomaly_mlops.domain.value_objects.semantic_version import SemanticVersion
from pynomaly_mlops.domain.value_objects.model_metrics import ModelMetrics


class ModelStatus(Enum):
    """Model lifecycle status enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelType(Enum):
    """Model type classification."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    ENSEMBLE = "ensemble"


@dataclass
class Model:
    """Core Model entity with versioning and lifecycle management.
    
    This entity represents a trained machine learning model with comprehensive
    metadata, versioning, and lifecycle tracking capabilities.
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = field()
    version: SemanticVersion = field()
    
    # Model Information
    model_type: ModelType = field()
    framework: str = field()  # e.g., "sklearn", "pytorch", "tensorflow"
    algorithm: str = field()  # e.g., "RandomForest", "LinearRegression"
    
    # Lifecycle
    status: ModelStatus = field(default=ModelStatus.DEVELOPMENT)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = field()
    
    # Metadata
    description: Optional[str] = field(default=None)
    tags: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    metrics: Optional[ModelMetrics] = field(default=None)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Artifacts
    artifact_uri: Optional[str] = field(default=None)
    model_size_bytes: Optional[int] = field(default=None)
    checksum: Optional[str] = field(default=None)
    
    # Lineage
    parent_model_id: Optional[UUID] = field(default=None)
    experiment_id: Optional[UUID] = field(default=None)
    dataset_version: Optional[str] = field(default=None)
    
    # Deployment Information
    last_deployed_at: Optional[datetime] = field(default=None)
    deployment_count: int = field(default=0)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.name:
            raise ValueError("Model name cannot be empty")
        
        if not self.created_by:
            raise ValueError("Model must have a creator")
            
        # Ensure tags are unique
        self.tags = list(set(self.tags))
    
    def promote_to_status(self, new_status: ModelStatus, promoted_by: str) -> None:
        """Promote model to a new lifecycle status.
        
        Args:
            new_status: Target status for promotion
            promoted_by: User performing the promotion
            
        Raises:
            ValueError: If promotion is not allowed
        """
        if not self._is_valid_promotion(new_status):
            raise ValueError(f"Invalid promotion from {self.status} to {new_status}")
        
        self.status = new_status
        self.updated_at = datetime.utcnow()
        
        # Add promotion tag
        promotion_tag = f"promoted_to_{new_status.value}_by_{promoted_by}"
        if promotion_tag not in self.tags:
            self.tags.append(promotion_tag)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def update_metrics(self, metrics: ModelMetrics) -> None:
        """Update model performance metrics."""
        self.metrics = metrics
        self.updated_at = datetime.utcnow()
    
    def mark_deployed(self) -> None:
        """Mark model as deployed and update deployment statistics."""
        self.last_deployed_at = datetime.utcnow()
        self.deployment_count += 1
        self.updated_at = datetime.utcnow()
    
    def create_child_version(self, new_version: SemanticVersion, created_by: str) -> "Model":
        """Create a new model version based on this model.
        
        Args:
            new_version: Version for the new model
            created_by: User creating the new version
            
        Returns:
            New Model instance as child version
        """
        return Model(
            name=self.name,
            version=new_version,
            model_type=self.model_type,
            framework=self.framework,
            algorithm=self.algorithm,
            created_by=created_by,
            description=self.description,
            parent_model_id=self.id,
            experiment_id=self.experiment_id,
        )
    
    def _is_valid_promotion(self, target_status: ModelStatus) -> bool:
        """Check if promotion to target status is valid."""
        valid_transitions = {
            ModelStatus.DEVELOPMENT: [ModelStatus.TESTING, ModelStatus.ARCHIVED],
            ModelStatus.TESTING: [ModelStatus.STAGING, ModelStatus.DEVELOPMENT, ModelStatus.ARCHIVED],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION, ModelStatus.TESTING, ModelStatus.ARCHIVED],
            ModelStatus.PRODUCTION: [ModelStatus.DEPRECATED, ModelStatus.ARCHIVED],
            ModelStatus.DEPRECATED: [ModelStatus.ARCHIVED],
            ModelStatus.ARCHIVED: [],  # Terminal state
        }
        
        return target_status in valid_transitions.get(self.status, [])
    
    @property
    def is_production_ready(self) -> bool:
        """Check if model is ready for production deployment."""
        return (
            self.status == ModelStatus.PRODUCTION and
            self.metrics is not None and
            self.artifact_uri is not None and
            self.checksum is not None
        )
    
    @property
    def version_string(self) -> str:
        """Get string representation of model version."""
        return str(self.version)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "version": str(self.version),
            "model_type": self.model_type.value,
            "framework": self.framework,
            "algorithm": self.algorithm,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags,
            "hyperparameters": self.hyperparameters,
            "training_config": self.training_config,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "validation_metrics": self.validation_metrics,
            "artifact_uri": self.artifact_uri,
            "model_size_bytes": self.model_size_bytes,
            "checksum": self.checksum,
            "parent_model_id": str(self.parent_model_id) if self.parent_model_id else None,
            "experiment_id": str(self.experiment_id) if self.experiment_id else None,
            "dataset_version": self.dataset_version,
            "last_deployed_at": self.last_deployed_at.isoformat() if self.last_deployed_at else None,
            "deployment_count": self.deployment_count,
        }