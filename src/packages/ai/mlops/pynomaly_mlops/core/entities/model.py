"""
Model Entity

Represents a machine learning model in the MLOps system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from ..abstractions import BaseEntity
from ..value_objects import ModelStatus, ModelMetrics, ModelMetadata


@dataclass
class Model(BaseEntity):
    """
    Represents a machine learning model.
    
    A model is a trained ML algorithm that can make predictions.
    It includes metadata, performance metrics, and version information.
    """
    
    name: str
    description: Optional[str] = None
    algorithm: Optional[str] = None
    version: str = "1.0.0"
    status: ModelStatus = ModelStatus.DRAFT
    metrics: Optional[ModelMetrics] = None
    metadata: Optional[ModelMetadata] = None
    tags: List[str] = None
    created_by: Optional[str] = None
    model_path: Optional[str] = None
    experiment_id: Optional[UUID] = None
    
    def __post_init__(self):
        """
        Initialize model with default values.
        """
        super().__post_init__()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = ModelMetadata()
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the model.
        
        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """
        Remove a tag from the model.
        
        Args:
            tag: Tag to remove
        """
        if tag in self.tags:
            self.tags.remove(tag)
    
    def update_metrics(self, metrics: ModelMetrics) -> None:
        """
        Update model metrics.
        
        Args:
            metrics: New metrics to set
        """
        self.metrics = metrics
        self.updated_at = datetime.utcnow()
    
    def update_status(self, status: ModelStatus) -> None:
        """
        Update model status.
        
        Args:
            status: New status to set
        """
        self.status = status
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary.
        
        Returns:
            Dictionary representation of the model
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "algorithm": self.algorithm,
            "version": self.version,
            "status": self.status.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "tags": self.tags,
            "created_by": self.created_by,
            "model_path": self.model_path,
            "experiment_id": str(self.experiment_id) if self.experiment_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Model':
        """
        Create model from dictionary.
        
        Args:
            data: Dictionary containing model data
            
        Returns:
            Model instance
        """
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            description=data.get("description"),
            algorithm=data.get("algorithm"),
            version=data.get("version", "1.0.0"),
            status=ModelStatus(data.get("status", ModelStatus.DRAFT.value)),
            metrics=ModelMetrics.from_dict(data["metrics"]) if data.get("metrics") else None,
            metadata=ModelMetadata.from_dict(data["metadata"]) if data.get("metadata") else None,
            tags=data.get("tags", []),
            created_by=data.get("created_by"),
            model_path=data.get("model_path"),
            experiment_id=UUID(data["experiment_id"]) if data.get("experiment_id") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
    
    def __str__(self) -> str:
        return f"Model(name='{self.name}', version='{self.version}', status='{self.status.value}')"
    
    def __repr__(self) -> str:
        return self.__str__()