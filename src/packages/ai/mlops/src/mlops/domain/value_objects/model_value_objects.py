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
    OUTLIER_PREDICTION = "outlier_prediction"
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


@dataclass(frozen=True)
class ModelStorageInfo:
    """Storage information for ML models."""
    
    storage_type: str  # "local", "s3", "gcs", "azure", etc.
    bucket_name: Optional[str] = None
    path: Optional[str] = None
    url: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    compression: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate storage info after creation."""
        if not self.storage_type:
            raise ValueError("Storage type is required")
        
        if self.storage_type in ("s3", "gcs", "azure") and not self.bucket_name:
            raise ValueError(f"Bucket name is required for {self.storage_type} storage")
    
    @property
    def full_path(self) -> str:
        """Get the full storage path."""
        if self.storage_type == "local":
            return self.path or ""
        elif self.storage_type in ("s3", "gcs", "azure"):
            bucket = self.bucket_name or ""
            path = self.path or ""
            return f"{bucket}/{path}" if path else bucket
        else:
            return self.url or self.path or ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "storage_type": self.storage_type,
            "bucket_name": self.bucket_name,
            "path": self.path,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "compression": self.compression,
            "full_path": self.full_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelStorageInfo":
        """Create storage info from dictionary."""
        return cls(
            storage_type=data["storage_type"],
            bucket_name=data.get("bucket_name"),
            path=data.get("path"),
            url=data.get("url"),
            size_bytes=data.get("size_bytes"),
            checksum=data.get("checksum"),
            compression=data.get("compression"),
        )


@dataclass(frozen=True)
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    
    # Regression metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate metrics after creation."""
        for metric_name, value in self.__dict__.items():
            if value is not None and isinstance(value, (int, float)) and metric_name != 'custom_metrics':
                if not (0.0 <= value <= 1.0) and metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr', 'r2_score']:
                    # Allow r2_score to be negative for very poor models
                    if metric_name == 'r2_score' and value >= -1.0:
                        continue
                    elif metric_name != 'r2_score':
                        continue  # For now, don't enforce strict bounds
    
    @property
    def has_classification_metrics(self) -> bool:
        """Check if classification metrics are available."""
        return any(getattr(self, metric) is not None 
                  for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'])
    
    @property
    def has_regression_metrics(self) -> bool:
        """Check if regression metrics are available."""
        return any(getattr(self, metric) is not None 
                  for metric in ['mse', 'rmse', 'mae', 'r2_score'])
    
    def get_primary_metric(self, task_type: str = "classification") -> Optional[float]:
        """Get the primary metric for the task type."""
        if task_type == "classification":
            return self.f1_score or self.accuracy or self.auc_roc
        elif task_type == "regression":
            return self.r2_score or self.rmse or self.mae
        return None
    
    def add_custom_metric(self, name: str, value: float) -> "PerformanceMetrics":
        """Add a custom metric (returns new instance since dataclass is frozen)."""
        new_custom_metrics = dict(self.custom_metrics)
        new_custom_metrics[name] = value
        
        return PerformanceMetrics(
            accuracy=self.accuracy,
            precision=self.precision,
            recall=self.recall,
            f1_score=self.f1_score,
            auc_roc=self.auc_roc,
            auc_pr=self.auc_pr,
            mse=self.mse,
            rmse=self.rmse,
            mae=self.mae,
            r2_score=self.r2_score,
            custom_metrics=new_custom_metrics,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2_score": self.r2_score,
            "custom_metrics": self.custom_metrics,
        }
        # Remove None values for cleaner representation
        return {k: v for k, v in result.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create performance metrics from dictionary."""
        return cls(
            accuracy=data.get("accuracy"),
            precision=data.get("precision"),
            recall=data.get("recall"),
            f1_score=data.get("f1_score"),
            auc_roc=data.get("auc_roc"),
            auc_pr=data.get("auc_pr"),
            mse=data.get("mse"),
            rmse=data.get("rmse"),
            mae=data.get("mae"),
            r2_score=data.get("r2_score"),
            custom_metrics=data.get("custom_metrics", {}),
        )