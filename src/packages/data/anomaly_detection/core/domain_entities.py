"""
Domain Entities for Anomaly Detection
=====================================

Core domain entities and value objects for the anomaly detection domain.
These entities define the fundamental business objects and their behaviors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid

class AnomalyType(Enum):
    """Anomaly type enumeration."""
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    TREND_ANOMALY = "trend_anomaly"

class DataType(Enum):
    """Data type enumeration."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TIME_SERIES = "time_series"
    TEXT = "text"
    IMAGE = "image"
    MULTIVARIATE = "multivariate"

class ModelStatus(Enum):
    """Model status enumeration."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"

@dataclass
class DataPoint:
    """Individual data point for anomaly detection."""
    id: Optional[str] = None
    values: Union[List[float], Dict[str, Any]] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'values': self.values,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata,
            'labels': self.labels
        }

@dataclass
class Dataset:
    """Dataset entity for anomaly detection."""
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    data_points: List[DataPoint] = field(default_factory=list)
    data_type: DataType = DataType.NUMERICAL
    features: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def add_data_point(self, data_point: DataPoint):
        """Add data point to dataset."""
        self.data_points.append(data_point)
        self.updated_at = datetime.now()
    
    def get_size(self) -> int:
        """Get dataset size."""
        return len(self.data_points)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.features.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'data_type': self.data_type.value,
            'features': self.features,
            'target_column': self.target_column,
            'size': self.get_size(),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'version': self.version,
            'tags': self.tags
        }

@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    data_point_id: str
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: AnomalyType = AnomalyType.POINT_ANOMALY
    confidence: float = 0.0
    explanation: Optional[str] = None
    features_importance: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'data_point_id': self.data_point_id,
            'is_anomaly': self.is_anomaly,
            'anomaly_score': self.anomaly_score,
            'anomaly_type': self.anomaly_type.value,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'features_importance': self.features_importance,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata
        }

@dataclass
class DetectionResult:
    """Complete detection result for a dataset."""
    id: Optional[str] = None
    dataset_id: str = ""
    detector_id: str = ""
    anomaly_results: List[AnomalyResult] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def get_anomaly_count(self) -> int:
        """Get total number of anomalies detected."""
        return sum(1 for result in self.anomaly_results if result.is_anomaly)
    
    def get_anomaly_rate(self) -> float:
        """Get anomaly rate (percentage of anomalies)."""
        if not self.anomaly_results:
            return 0.0
        return (self.get_anomaly_count() / len(self.anomaly_results)) * 100.0
    
    def get_average_anomaly_score(self) -> float:
        """Get average anomaly score."""
        if not self.anomaly_results:
            return 0.0
        return sum(result.anomaly_score for result in self.anomaly_results) / len(self.anomaly_results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'detector_id': self.detector_id,
            'anomaly_count': self.get_anomaly_count(),
            'total_points': len(self.anomaly_results),
            'anomaly_rate': self.get_anomaly_rate(),
            'average_score': self.get_average_anomaly_score(),
            'summary_statistics': self.summary_statistics,
            'performance_metrics': self.performance_metrics,
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata
        }

@dataclass
class DetectorConfiguration:
    """Configuration for anomaly detectors."""
    algorithm: str = "isolation_forest"
    parameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    postprocessing_steps: List[str] = field(default_factory=list)
    threshold: Optional[float] = None
    auto_threshold: bool = True
    validation_split: float = 0.2
    random_state: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'algorithm': self.algorithm,
            'parameters': self.parameters,
            'preprocessing_steps': self.preprocessing_steps,
            'postprocessing_steps': self.postprocessing_steps,
            'threshold': self.threshold,
            'auto_threshold': self.auto_threshold,
            'validation_split': self.validation_split,
            'random_state': self.random_state
        }

@dataclass
class DetectorModel:
    """Trained detector model entity."""
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    algorithm: str = ""
    configuration: DetectorConfiguration = field(default_factory=DetectorConfiguration)
    training_dataset_id: Optional[str] = None
    model_data: Optional[bytes] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    status: ModelStatus = ModelStatus.TRAINING
    version: str = "1.0.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    trained_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def mark_as_trained(self):
        """Mark model as trained."""
        self.status = ModelStatus.TRAINED
        self.trained_at = datetime.now()
        self.updated_at = datetime.now()
    
    def mark_as_deployed(self):
        """Mark model as deployed."""
        self.status = ModelStatus.DEPLOYED
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'algorithm': self.algorithm,
            'configuration': self.configuration.to_dict(),
            'training_dataset_id': self.training_dataset_id,
            'performance_metrics': self.performance_metrics,
            'feature_names': self.feature_names,
            'status': self.status.value,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'tags': self.tags
        }

@dataclass
class TrainingJob:
    """Training job entity."""
    id: Optional[str] = None
    name: str = ""
    dataset_id: str = ""
    configuration: DetectorConfiguration = field(default_factory=DetectorConfiguration)
    status: str = "pending"
    progress: float = 0.0
    logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    result_model_id: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def start_training(self):
        """Start training job."""
        self.status = "running"
        self.started_at = datetime.now()
    
    def complete_training(self, model_id: str):
        """Complete training job."""
        self.status = "completed"
        self.progress = 100.0
        self.result_model_id = model_id
        self.completed_at = datetime.now()
    
    def fail_training(self, error_message: str):
        """Fail training job."""
        self.status = "failed"
        self.error_message = error_message
        self.completed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'dataset_id': self.dataset_id,
            'configuration': self.configuration.to_dict(),
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'result_model_id': self.result_model_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata
        }

# Domain protocols (interfaces)
class DetectorProtocol(ABC):
    """Protocol for anomaly detectors."""
    
    @abstractmethod
    def fit(self, dataset: Dataset) -> DetectorModel:
        """Train the detector on dataset.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Trained detector model
        """
        pass
    
    @abstractmethod
    def predict(self, dataset: Dataset, model: DetectorModel) -> DetectionResult:
        """Predict anomalies in dataset.
        
        Args:
            dataset: Dataset to analyze
            model: Trained detector model
            
        Returns:
            Detection results
        """
        pass
    
    @abstractmethod
    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported algorithms.
        
        Returns:
            List of algorithm names
        """
        pass

class DatasetRepositoryProtocol(ABC):
    """Protocol for dataset repository."""
    
    @abstractmethod
    async def save(self, dataset: Dataset) -> Dataset:
        """Save dataset.
        
        Args:
            dataset: Dataset to save
            
        Returns:
            Saved dataset
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, dataset_id: str) -> Optional[Dataset]:
        """Get dataset by ID.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dataset or None if not found
        """
        pass
    
    @abstractmethod
    async def delete(self, dataset_id: str) -> bool:
        """Delete dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            True if deleted successfully
        """
        pass
    
    @abstractmethod
    async def list_all(self) -> List[Dataset]:
        """List all datasets.
        
        Returns:
            List of datasets
        """
        pass

class DetectorModelRepositoryProtocol(ABC):
    """Protocol for detector model repository."""
    
    @abstractmethod
    async def save(self, model: DetectorModel) -> DetectorModel:
        """Save detector model.
        
        Args:
            model: Model to save
            
        Returns:
            Saved model
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, model_id: str) -> Optional[DetectorModel]:
        """Get model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model or None if not found
        """
        pass
    
    @abstractmethod
    async def delete(self, model_id: str) -> bool:
        """Delete model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deleted successfully
        """
        pass
    
    @abstractmethod
    async def list_all(self) -> List[DetectorModel]:
        """List all models.
        
        Returns:
            List of models
        """
        pass

class DetectionResultRepositoryProtocol(ABC):
    """Protocol for detection result repository."""
    
    @abstractmethod
    async def save(self, result: DetectionResult) -> DetectionResult:
        """Save detection result.
        
        Args:
            result: Detection result to save
            
        Returns:
            Saved result
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, result_id: str) -> Optional[DetectionResult]:
        """Get detection result by ID.
        
        Args:
            result_id: Result identifier
            
        Returns:
            Detection result or None if not found
        """
        pass
    
    @abstractmethod
    async def get_by_dataset_id(self, dataset_id: str) -> List[DetectionResult]:
        """Get detection results by dataset ID.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            List of detection results
        """
        pass