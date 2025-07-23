"""Experiment domain entity for MLOps package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from ..value_objects.model_value_objects import ModelMetrics


class ExperimentStatus(str, Enum):
    """Experiment status enumeration."""
    
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Experiment:
    """Experiment domain entity.
    
    Represents an ML experiment with tracking capabilities,
    following Domain-Driven Design principles.
    """
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.CREATED
    
    # Experiment configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Model association
    model_id: Optional[UUID] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # User tracking
    created_by: str = ""
    team: str = ""
    
    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        if not self.name:
            self.name = f"experiment_{self.id.hex[:8]}"
        
        self._validate_experiment()
    
    def _validate_experiment(self) -> None:
        """Validate experiment state and business rules."""
        if not self.name or len(self.name.strip()) == 0:
            raise ValueError("Experiment name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Experiment name cannot exceed 100 characters")
        
        if self.status not in ExperimentStatus:
            raise ValueError(f"Invalid experiment status: {self.status}")
    
    def start(self) -> None:
        """Start the experiment.
        
        Business logic for experiment startup.
        
        Raises:
            ValueError: If experiment is not in CREATED status
        """
        if self.status != ExperimentStatus.CREATED:
            raise ValueError(f"Cannot start experiment in {self.status} status")
        
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def complete(self, final_metrics: Optional[Dict[str, float]] = None) -> None:
        """Complete the experiment with final results.
        
        Args:
            final_metrics: Final metrics to record
            
        Raises:
            ValueError: If experiment is not in RUNNING status
        """
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot complete experiment in {self.status} status")
        
        if final_metrics:
            self.metrics.update(final_metrics)
        
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def fail(self, error_message: str = "") -> None:
        """Mark experiment as failed.
        
        Args:
            error_message: Optional error message
        """
        self.status = ExperimentStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        if error_message:
            self.artifacts["error_message"] = error_message
    
    def cancel(self) -> None:
        """Cancel the experiment.
        
        Raises:
            ValueError: If experiment is already completed or failed
        """
        if self.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
            raise ValueError(f"Cannot cancel experiment in {self.status} status")
        
        self.status = ExperimentStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def add_metric(self, name: str, value: float) -> None:
        """Add or update a metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if not name or not isinstance(name, str):
            raise ValueError("Metric name must be a non-empty string")
        
        if not isinstance(value, (int, float)):
            raise ValueError("Metric value must be a number")
        
        self.metrics[name] = float(value)
        self.updated_at = datetime.utcnow()
    
    def add_artifact(self, name: str, path: str) -> None:
        """Add an artifact reference.
        
        Args:
            name: Artifact name
            path: Artifact path or URL
        """
        if not name or not isinstance(name, str):
            raise ValueError("Artifact name must be a non-empty string")
        
        if not path or not isinstance(path, str):
            raise ValueError("Artifact path must be a non-empty string")
        
        self.artifacts[name] = path
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment.
        
        Args:
            tag: Tag to add
        """
        if not tag or not isinstance(tag, str):
            raise ValueError("Tag must be a non-empty string")
        
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the experiment.
        
        Args:
            tag: Tag to remove
        """
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    @property
    def duration(self) -> Optional[float]:
        """Get experiment duration in seconds.
        
        Returns:
            Duration in seconds if experiment has started, None otherwise
        """
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    @property
    def is_active(self) -> bool:
        """Check if experiment is currently active."""
        return self.status == ExperimentStatus.RUNNING
    
    @property
    def is_finished(self) -> bool:
        """Check if experiment is finished (completed, failed, or cancelled)."""
        return self.status in [
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED
        ]
    
    def get_best_metric(self, metric_name: str, higher_is_better: bool = True) -> Optional[float]:
        """Get the best value for a specific metric.
        
        Args:
            metric_name: Name of the metric
            higher_is_better: Whether higher values are better
            
        Returns:
            Best metric value or None if metric doesn't exist
        """
        if metric_name not in self.metrics:
            return None
        
        return self.metrics[metric_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary representation.
        
        Returns:
            Dictionary representation of the experiment
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "tags": self.tags,
            "model_id": str(self.model_id) if self.model_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by": self.created_by,
            "team": self.team,
            "duration": self.duration,
            "is_active": self.is_active,
            "is_finished": self.is_finished,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Experiment:
        """Create experiment from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Experiment instance
        """
        experiment = cls(
            id=UUID(data["id"]) if data.get("id") else uuid4(),
            name=data.get("name", ""),
            description=data.get("description", ""),
            status=ExperimentStatus(data.get("status", ExperimentStatus.CREATED)),
            parameters=data.get("parameters", {}),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
            tags=data.get("tags", []),
            model_id=UUID(data["model_id"]) if data.get("model_id") else None,
            created_by=data.get("created_by", ""),
            team=data.get("team", ""),
        )
        
        # Handle datetime fields
        if data.get("created_at"):
            experiment.created_at = datetime.fromisoformat(data["created_at"])
        
        if data.get("updated_at"):
            experiment.updated_at = datetime.fromisoformat(data["updated_at"])
        
        if data.get("started_at"):
            experiment.started_at = datetime.fromisoformat(data["started_at"])
        
        if data.get("completed_at"):
            experiment.completed_at = datetime.fromisoformat(data["completed_at"])
        
        return experiment
    
    def __str__(self) -> str:
        """String representation of the experiment."""
        return f"Experiment(id={self.id.hex[:8]}, name='{self.name}', status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the experiment."""
        return (
            f"Experiment("
            f"id={self.id}, "
            f"name='{self.name}', "
            f"status={self.status.value}, "
            f"metrics_count={len(self.metrics)}, "
            f"artifacts_count={len(self.artifacts)}"
            f")"
        )