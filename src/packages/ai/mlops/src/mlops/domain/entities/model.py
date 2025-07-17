"""Model entity for MLOps domain."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import dataclasses

from ..value_objects.model_value_objects import (
    ModelId, ModelStatus, ModelMetrics, ModelMetadata, ModelVersion, ModelType
)


@dataclass(frozen=True)
class Model:
    """
    Domain entity representing a machine learning model.
    
    A model is a trained ML algorithm that can make predictions.
    It includes metadata, performance metrics, and version information.
    """
    
    model_id: ModelId
    name: str
    description: str
    model_type: ModelType
    algorithm: str
    version: ModelVersion
    status: ModelStatus
    metrics: Optional[ModelMetrics] = None
    metadata: Optional[ModelMetadata] = None
    tags: List[str] = field(default_factory=list)
    created_by: str = ""
    model_path: Optional[str] = None
    experiment_id: Optional[str] = None
    parent_model_id: Optional[ModelId] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate model after initialization."""
        if not self.name.strip():
            raise ValueError("Model name cannot be empty")
        
        if not self.algorithm.strip():
            raise ValueError("Model algorithm cannot be empty")
    
    def update_status(self, status: ModelStatus) -> "Model":
        """Update model status."""
        return dataclasses.replace(
            self,
            status=status,
            updated_at=datetime.utcnow()
        )
    
    def update_metrics(self, metrics: ModelMetrics) -> "Model":
        """Update model metrics."""
        return dataclasses.replace(
            self,
            metrics=metrics,
            updated_at=datetime.utcnow()
        )
    
    def update_metadata(self, metadata: ModelMetadata) -> "Model":
        """Update model metadata."""
        return dataclasses.replace(
            self,
            metadata=metadata,
            updated_at=datetime.utcnow()
        )
    
    def add_tag(self, tag: str) -> "Model":
        """Add a tag to the model."""
        if tag not in self.tags:
            new_tags = list(self.tags) + [tag]
            return dataclasses.replace(
                self,
                tags=new_tags,
                updated_at=datetime.utcnow()
            )
        return self
    
    def remove_tag(self, tag: str) -> "Model":
        """Remove a tag from the model."""
        if tag in self.tags:
            new_tags = [t for t in self.tags if t != tag]
            return dataclasses.replace(
                self,
                tags=new_tags,
                updated_at=datetime.utcnow()
            )
        return self
    
    def increment_version(self, version_type: str = "patch") -> "Model":
        """Increment model version."""
        if version_type == "major":
            new_version = self.version.increment_major()
        elif version_type == "minor":
            new_version = self.version.increment_minor()
        else:  # patch
            new_version = self.version.increment_patch()
        
        return dataclasses.replace(
            self,
            version=new_version,
            updated_at=datetime.utcnow()
        )
    
    def is_ready_for_deployment(self) -> bool:
        """Check if model is ready for deployment."""
        return (
            self.status == ModelStatus.VALIDATED and
            self.metrics is not None and
            self.model_path is not None
        )
    
    def is_production_ready(self) -> bool:
        """Check if model meets production requirements."""
        if not self.is_ready_for_deployment():
            return False
        
        # Check if model has minimum required metrics
        if self.metrics:
            if self.model_type == ModelType.CLASSIFICATION:
                return (
                    self.metrics.accuracy is not None and
                    self.metrics.precision is not None and
                    self.metrics.recall is not None
                )
            elif self.model_type == ModelType.REGRESSION:
                return (
                    self.metrics.mse is not None and
                    self.metrics.r2_score is not None
                )
        
        return True
    
    def get_performance_score(self) -> Optional[float]:
        """Get overall performance score."""
        if not self.metrics:
            return None
        
        if self.model_type == ModelType.CLASSIFICATION:
            return self.metrics.f1_score or self.metrics.accuracy
        elif self.model_type == ModelType.REGRESSION:
            return self.metrics.r2_score
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "model_id": str(self.model_id),
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type.value,
            "algorithm": self.algorithm,
            "version": str(self.version),
            "status": self.status.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "tags": self.tags,
            "created_by": self.created_by,
            "model_path": self.model_path,
            "experiment_id": self.experiment_id,
            "parent_model_id": str(self.parent_model_id) if self.parent_model_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Model":
        """Create model from dictionary."""
        return cls(
            model_id=ModelId(data["model_id"]),
            name=data["name"],
            description=data["description"],
            model_type=ModelType(data["model_type"]),
            algorithm=data["algorithm"],
            version=ModelVersion.from_string(data["version"]),
            status=ModelStatus(data["status"]),
            metrics=ModelMetrics.from_dict(data["metrics"]) if data.get("metrics") else None,
            metadata=ModelMetadata.from_dict(data["metadata"]) if data.get("metadata") else None,
            tags=data.get("tags", []),
            created_by=data.get("created_by", ""),
            model_path=data.get("model_path"),
            experiment_id=data.get("experiment_id"),
            parent_model_id=ModelId(data["parent_model_id"]) if data.get("parent_model_id") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
    
    def __str__(self) -> str:
        return f"Model(name='{self.name}', version='{self.version}', status='{self.status.value}')"
    
    def __repr__(self) -> str:
        return self.__str__()