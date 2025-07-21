"""Value objects for ML experiments."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from enum import Enum


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExperimentType(Enum):
    """Experiment type enumeration."""
    TRAINING = "training"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    MODEL_COMPARISON = "model_comparison"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_VALIDATION = "data_validation"
    A_B_TESTING = "a_b_testing"
    PERFORMANCE_TESTING = "performance_testing"


@dataclass(frozen=True)
class ExperimentId:
    """Unique identifier for ML experiments."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Configuration for ML experiments."""
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    data_splits: Dict[str, float] = field(default_factory=dict)
    cross_validation: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    feature_selection: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: List[str] = field(default_factory=list)
    early_stopping: Dict[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "algorithm": self.algorithm,
            "hyperparameters": self.hyperparameters,
            "data_splits": self.data_splits,
            "cross_validation": self.cross_validation,
            "preprocessing_steps": self.preprocessing_steps,
            "feature_selection": self.feature_selection,
            "evaluation_metrics": self.evaluation_metrics,
            "early_stopping": self.early_stopping,
            "random_seed": self.random_seed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfiguration":
        """Create configuration from dictionary."""
        return cls(
            algorithm=data["algorithm"],
            hyperparameters=data.get("hyperparameters", {}),
            data_splits=data.get("data_splits", {}),
            cross_validation=data.get("cross_validation", {}),
            preprocessing_steps=data.get("preprocessing_steps", []),
            feature_selection=data.get("feature_selection", {}),
            evaluation_metrics=data.get("evaluation_metrics", []),
            early_stopping=data.get("early_stopping", {}),
            random_seed=data.get("random_seed"),
        )


@dataclass(frozen=True)
class ExperimentResults:
    """Results of ML experiments."""
    final_metrics: Dict[str, float] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    validation_scores: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[List[List[int]]] = None
    prediction_samples: List[Dict[str, Any]] = field(default_factory=list)
    model_artifacts: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "final_metrics": self.final_metrics,
            "best_params": self.best_params,
            "training_history": self.training_history,
            "validation_scores": self.validation_scores,
            "feature_importance": self.feature_importance,
            "confusion_matrix": self.confusion_matrix,
            "prediction_samples": self.prediction_samples,
            "model_artifacts": self.model_artifacts,
            "logs": self.logs,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResults":
        """Create results from dictionary."""
        return cls(
            final_metrics=data.get("final_metrics", {}),
            best_params=data.get("best_params", {}),
            training_history=data.get("training_history", []),
            validation_scores=data.get("validation_scores", []),
            feature_importance=data.get("feature_importance", {}),
            confusion_matrix=data.get("confusion_matrix"),
            prediction_samples=data.get("prediction_samples", []),
            model_artifacts=data.get("model_artifacts", []),
            logs=data.get("logs", []),
        )


@dataclass(frozen=True)
class ExperimentMetadata:
    """Metadata for ML experiments."""
    created_by: str
    project_name: str
    experiment_name: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    compute_resources: Dict[str, Any] = field(default_factory=dict)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    dataset_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "created_by": self.created_by,
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags,
            "environment": self.environment,
            "compute_resources": self.compute_resources,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "dataset_version": self.dataset_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetadata":
        """Create metadata from dictionary."""
        return cls(
            created_by=data["created_by"],
            project_name=data["project_name"],
            experiment_name=data["experiment_name"],
            description=data.get("description"),
            tags=data.get("tags", []),
            environment=data.get("environment", {}),
            compute_resources=data.get("compute_resources", {}),
            git_commit=data.get("git_commit"),
            git_branch=data.get("git_branch"),
            dataset_version=data.get("dataset_version"),
        )