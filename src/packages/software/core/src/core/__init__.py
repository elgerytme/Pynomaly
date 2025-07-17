"""
Core Package - Shared Core Components and Abstractions

This package provides foundational components used across the entire system:
- Base entities and value objects
- Core abstractions and protocols
- Shared utilities and types
- Common exceptions and error handling
- Domain-agnostic business logic
"""

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "support@pynomaly.com"

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar, Dict, List, Optional, Protocol
from uuid import UUID, uuid4

# Core abstractions with fallback implementations
class BaseEntity:
    """Base entity with lifecycle management."""
    
    def __init__(self, **data: Any):
        self.id: UUID = data.get('id', uuid4())
        self.created_at: datetime = data.get('created_at', datetime.utcnow())
        self.updated_at: datetime = data.get('updated_at', datetime.utcnow())
        self.version: int = data.get('version', 1)
        self.metadata: Dict[str, Any] = data.get('metadata', {})
        
    def __hash__(self) -> int:
        return hash(self.id)
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseEntity):
            return False
        return self.id == other.id
        
    def mark_as_updated(self) -> None:
        self.updated_at = datetime.utcnow()
        self.version += 1

class BaseRepository(ABC):
    """Base repository interface."""
    
    @abstractmethod
    def save(self, entity: BaseEntity) -> BaseEntity:
        pass
        
    @abstractmethod
    def find_by_id(self, entity_id: UUID) -> Optional[BaseEntity]:
        pass
        
    @abstractmethod
    def find_all(self) -> List[BaseEntity]:
        pass
        
    @abstractmethod
    def delete(self, entity_id: UUID) -> bool:
        pass

class BaseService(ABC):
    """Base service interface."""
    pass

class BaseValueObject:
    """Base value object."""
    
    def __init__(self, **data: Any):
        self.__dict__.update(data)
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__
        
    def __hash__(self) -> int:
        return hash(frozenset(self.__dict__.items()))

class Specification(ABC):
    """Base specification pattern."""
    
    @abstractmethod
    def is_satisfied_by(self, candidate: Any) -> bool:
        pass

# Value Objects
class PerformanceMetrics(BaseValueObject):
    """Performance metrics value object."""
    
    def __init__(self, accuracy: float = 0.0, precision: float = 0.0, 
                 recall: float = 0.0, f1_score: float = 0.0, **kwargs):
        super().__init__(accuracy=accuracy, precision=precision, 
                        recall=recall, f1_score=f1_score, **kwargs)

class ModelMetrics(BaseValueObject):
    """Model metrics value object."""
    
    def __init__(self, training_time: float = 0.0, inference_time: float = 0.0, 
                 model_size: int = 0, **kwargs):
        super().__init__(training_time=training_time, inference_time=inference_time,
                        model_size=model_size, **kwargs)

class SemanticVersion(BaseValueObject):
    """Semantic version value object."""
    
    def __init__(self, major: int = 0, minor: int = 1, patch: int = 0, **kwargs):
        super().__init__(major=major, minor=minor, patch=patch, **kwargs)

class ConfidenceInterval(BaseValueObject):
    """Confidence interval value object."""
    
    def __init__(self, lower: float = 0.0, upper: float = 1.0, confidence: float = 0.95, **kwargs):
        super().__init__(lower=lower, upper=upper, confidence=confidence, **kwargs)

class ContaminationRate(BaseValueObject):
    """Contamination rate value object."""
    
    def __init__(self, rate: float = 0.1, **kwargs):
        super().__init__(rate=rate, **kwargs)

class Hyperparameters(BaseValueObject):
    """Hyperparameters value object."""
    
    def __init__(self, parameters: Dict[str, Any] = None, **kwargs):
        super().__init__(parameters=parameters or {}, **kwargs)

class ThresholdConfig(BaseValueObject):
    """Threshold configuration value object."""
    
    def __init__(self, threshold: float = 0.5, auto_adjust: bool = True, **kwargs):
        super().__init__(threshold=threshold, auto_adjust=auto_adjust, **kwargs)

class SeverityScore(BaseValueObject):
    """Severity score value object."""
    
    def __init__(self, score: float = 0.0, level: str = "low", **kwargs):
        super().__init__(score=score, level=level, **kwargs)

class ModelStorageInfo(BaseValueObject):
    """Model storage information value object."""
    
    def __init__(self, path: str = "", size: int = 0, format: str = "pickle", **kwargs):
        super().__init__(path=path, size=size, format=format, **kwargs)

class StorageCredentials(BaseValueObject):
    """Storage credentials value object."""
    
    def __init__(self, access_key: str = "", secret_key: str = "", **kwargs):
        super().__init__(access_key=access_key, secret_key=secret_key, **kwargs)

# Entities
class GenericDetector(BaseEntity):
    """Generic detector entity."""
    
    def __init__(self, name: str = "generic", algorithm: str = "isolation_forest", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.algorithm = algorithm
        self.is_trained = False
        self.model = None

# Shared utilities
class ErrorHandling:
    """Error handling utilities."""
    
    @staticmethod
    def handle_error(error: Exception, context: str = "") -> None:
        """Handle error with context."""
        print(f"Error in {context}: {error}")

class UnifiedException(Exception):
    """Unified exception for the system."""
    
    def __init__(self, message: str, error_code: str = "UNKNOWN", **kwargs):
        super().__init__(message)
        self.error_code = error_code
        self.context = kwargs

class RecoveryStrategy:
    """Recovery strategy for error handling."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        
    def execute(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                continue

class MonitoringService:
    """Monitoring service for system observability."""
    
    def __init__(self):
        self.metrics = {}
        
    def log_metric(self, name: str, value: Any) -> None:
        """Log a metric."""
        self.metrics[name] = value
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return self.metrics

class ResilienceService:
    """Resilience service for system stability."""
    
    def __init__(self):
        self.circuit_breakers = {}
        
    def add_circuit_breaker(self, name: str, threshold: int = 5) -> None:
        """Add circuit breaker."""
        self.circuit_breakers[name] = {'threshold': threshold, 'failures': 0}

# Protocols
class DataLoaderProtocol(Protocol):
    """Data loader protocol."""
    
    def load(self, source: str) -> Any:
        """Load data from source."""
        ...

class DetectorProtocol(Protocol):
    """Detector protocol."""
    
    def detect(self, data: Any) -> Any:
        """Detect anomalies in data."""
        ...

class RepositoryProtocol(Protocol):
    """Repository protocol."""
    
    def save(self, entity: BaseEntity) -> BaseEntity:
        """Save entity."""
        ...

class ExportProtocol(Protocol):
    """Export protocol."""
    
    def export(self, data: Any, destination: str) -> bool:
        """Export data to destination."""
        ...

class ImportProtocol(Protocol):
    """Import protocol."""
    
    def import_data(self, source: str) -> Any:
        """Import data from source."""
        ...

class GenericDetectionProtocol(Protocol):
    """Generic detection protocol."""
    
    def detect_anomalies(self, data: Any) -> Any:
        """Detect anomalies generically."""
        ...

# Types
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, float]
ParameterDict = Dict[str, Any]
ResultDict = Dict[str, Any]

__all__ = [
    # Abstractions
    "BaseEntity",
    "BaseRepository",
    "BaseService",
    "BaseValueObject",
    "Specification",
    
    # Value Objects
    "PerformanceMetrics",
    "ModelMetrics",
    "SemanticVersion",
    "ConfidenceInterval",
    "ContaminationRate",
    "Hyperparameters",
    "ThresholdConfig",
    "SeverityScore",
    "ModelStorageInfo",
    "StorageCredentials",
    
    # Entities
    "GenericDetector",
    
    # Shared utilities
    "ErrorHandling",
    "UnifiedException",
    "RecoveryStrategy",
    "MonitoringService",
    "ResilienceService",
    
    # Protocols
    "DataLoaderProtocol",
    "DetectorProtocol",
    "RepositoryProtocol",
    "ExportProtocol",
    "ImportProtocol",
    "GenericDetectionProtocol",
    
    # Types
    "ConfigDict",
    "MetricsDict",
    "ParameterDict",
    "ResultDict",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]