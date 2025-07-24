"""Model-related value objects."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import uuid


class ModelStatus(str, Enum):
    """Model status enumeration."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class SerializationFormat(str, Enum):
    """Model serialization format enumeration."""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    ONNX = "onnx"
    JSON = "json"


@dataclass(frozen=True)
class ModelVersion:
    """Model version identifier."""
    major: int
    minor: int
    patch: int
    
    def __post_init__(self):
        if any(v < 0 for v in [self.major, self.minor, self.patch]):
            raise ValueError("Version numbers must be non-negative")
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        """Create version from string like '1.2.3'."""
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError("Version string must be in format 'major.minor.patch'")
        
        try:
            major, minor, patch = map(int, parts)
            return cls(major, minor, patch)
        except ValueError:
            raise ValueError("Version parts must be integers")


@dataclass(frozen=True)
class ModelConfiguration:
    """Model configuration parameters."""
    algorithm: str
    hyperparameters: Dict[str, Any]
    contamination: float = 0.1
    preprocessing_steps: Optional[List[str]] = None
    feature_selection: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.preprocessing_steps is None:
            object.__setattr__(self, 'preprocessing_steps', [])
        if self.feature_selection is None:
            object.__setattr__(self, 'feature_selection', {})
        
        if not 0.0 <= self.contamination <= 0.5:
            raise ValueError("Contamination must be between 0.0 and 0.5")


@dataclass(frozen=True)
class TrainingConfiguration:
    """Training configuration parameters."""
    batch_size: int = 1000
    validation_split: float = 0.2
    early_stopping: bool = True
    max_iterations: Optional[int] = None
    convergence_threshold: float = 1e-6
    cross_validation_folds: int = 5
    
    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not 0.0 < self.validation_split < 1.0:
            raise ValueError("Validation split must be between 0 and 1")
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.cross_validation_folds < 2:
            raise ValueError("Cross validation folds must be at least 2")


@dataclass(frozen=True)
class PerformanceMetrics:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    training_time_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    def __post_init__(self):
        # Validate metrics are in valid ranges
        for metric_name, value in [
            ("accuracy", self.accuracy),
            ("precision", self.precision), 
            ("recall", self.recall),
            ("f1_score", self.f1_score),
            ("auc_roc", self.auc_roc)
        ]:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{metric_name} must be between 0.0 and 1.0")
        
        if self.training_time_seconds is not None and self.training_time_seconds < 0:
            raise ValueError("Training time must be non-negative")
        if self.inference_time_ms is not None and self.inference_time_ms < 0:
            raise ValueError("Inference time must be non-negative")
        if self.memory_usage_mb is not None and self.memory_usage_mb < 0:
            raise ValueError("Memory usage must be non-negative")
    
    @property
    def has_classification_metrics(self) -> bool:
        """Check if classification metrics are available."""
        return any([
            self.accuracy is not None,
            self.precision is not None,
            self.recall is not None,
            self.f1_score is not None
        ])
    
    @property
    def has_performance_metrics(self) -> bool:
        """Check if performance metrics are available."""
        return any([
            self.training_time_seconds is not None,
            self.inference_time_ms is not None,
            self.memory_usage_mb is not None
        ])


@dataclass(frozen=True)
class ModelMetadata:
    """Complete model metadata."""
    model_id: str
    name: str
    version: ModelVersion
    status: ModelStatus
    configuration: ModelConfiguration
    training_config: TrainingConfiguration
    performance: PerformanceMetrics
    created_at: datetime
    updated_at: Optional[datetime] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    creator: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            object.__setattr__(self, 'tags', [])
        
        # Validate model_id is a valid UUID format
        try:
            uuid.UUID(self.model_id)
        except ValueError:
            raise ValueError("model_id must be a valid UUID")
    
    @property
    def is_deployable(self) -> bool:
        """Check if model is ready for deployment."""
        return (
            self.status == ModelStatus.TRAINED and
            self.performance.has_classification_metrics
        )
    
    @classmethod
    def create_new(
        cls,
        name: str,
        configuration: ModelConfiguration,
        training_config: TrainingConfiguration,
        description: Optional[str] = None,
        creator: Optional[str] = None
    ) -> "ModelMetadata":
        """Create new model metadata."""
        return cls(
            model_id=str(uuid.uuid4()),
            name=name,
            version=ModelVersion(1, 0, 0),
            status=ModelStatus.TRAINING,
            configuration=configuration,
            training_config=training_config,
            performance=PerformanceMetrics(),
            created_at=datetime.utcnow(),
            description=description,
            creator=creator
        )