"""Value objects for ML models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from enum import Enum


class ModelStatus(Enum):
    """Model status enumeration."""
    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REGISTERED = "registered"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class ModelType(Enum):
    """Model type enumeration."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERATIVE = "generative"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"


@dataclass(frozen=True)
class ModelId:
    """Unique identifier for ML models."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ModelMetrics:
    """Performance metrics for ML models."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2_score": self.r2_score,
            "confusion_matrix": self.confusion_matrix,
            "custom_metrics": self.custom_metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetrics":
        """Create metrics from dictionary."""
        return cls(
            accuracy=data.get("accuracy"),
            precision=data.get("precision"),
            recall=data.get("recall"),
            f1_score=data.get("f1_score"),
            auc_roc=data.get("auc_roc"),
            mse=data.get("mse"),
            rmse=data.get("rmse"),
            mae=data.get("mae"),
            r2_score=data.get("r2_score"),
            confusion_matrix=data.get("confusion_matrix"),
            custom_metrics=data.get("custom_metrics", {}),
        )


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata for ML models."""
    framework: Optional[str] = None
    framework_version: Optional[str] = None
    python_version: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    data_schema: Dict[str, Any] = field(default_factory=dict)
    training_duration: Optional[float] = None
    training_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    test_samples: Optional[int] = None
    model_size_bytes: Optional[int] = None
    environment: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "framework": self.framework,
            "framework_version": self.framework_version,
            "python_version": self.python_version,
            "dependencies": self.dependencies,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "data_schema": self.data_schema,
            "training_duration": self.training_duration,
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
            "model_size_bytes": self.model_size_bytes,
            "environment": self.environment,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        return cls(
            framework=data.get("framework"),
            framework_version=data.get("framework_version"),
            python_version=data.get("python_version"),
            dependencies=data.get("dependencies", {}),
            hyperparameters=data.get("hyperparameters", {}),
            feature_names=data.get("feature_names", []),
            target_names=data.get("target_names", []),
            data_schema=data.get("data_schema", {}),
            training_duration=data.get("training_duration"),
            training_samples=data.get("training_samples"),
            validation_samples=data.get("validation_samples"),
            test_samples=data.get("test_samples"),
            model_size_bytes=data.get("model_size_bytes"),
            environment=data.get("environment", {}),
        )


@dataclass(frozen=True)
class ModelVersion:
    """Version information for ML models."""
    major: int
    minor: int
    patch: int
    build: Optional[str] = None
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    def __lt__(self, other: "ModelVersion") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: "ModelVersion") -> bool:
        """Check version equality."""
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        """Create version from string."""
        if "+" in version_str:
            version_part, build = version_str.split("+", 1)
        else:
            version_part, build = version_str, None
            
        parts = version_part.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
            
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
            build=build
        )
    
    def increment_patch(self) -> "ModelVersion":
        """Increment patch version."""
        return ModelVersion(self.major, self.minor, self.patch + 1, self.build)
    
    def increment_minor(self) -> "ModelVersion":
        """Increment minor version."""
        return ModelVersion(self.major, self.minor + 1, 0, self.build)
    
    def increment_major(self) -> "ModelVersion":
        """Increment major version."""
        return ModelVersion(self.major + 1, 0, 0, self.build)