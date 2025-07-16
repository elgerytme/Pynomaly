"""Interface for anomaly detection domain communication."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum


class AnomalyType(Enum):
    """Types of anomalies."""
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    DRIFT = "drift"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    anomaly_id: str
    dataset_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float
    confidence: float
    description: str
    affected_features: List[str]
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class DetectionRequest:
    """Request for anomaly detection."""
    dataset_id: str
    algorithm: str
    parameters: Dict[str, Any]
    threshold: float
    include_explanations: bool = False


class AnomalyDetectionProtocol(Protocol):
    """Protocol for anomaly detection service interactions."""
    
    def detect_anomalies(self, request: DetectionRequest) -> List[AnomalyResult]:
        """Detect anomalies in data."""
        ...
    
    def get_detection_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of detection job."""
        ...


class AnomalyDetectionInterface(ABC):
    """Abstract interface for anomaly detection operations."""
    
    @abstractmethod
    def run_detection(self, request: DetectionRequest) -> List[AnomalyResult]:
        """Run anomaly detection on dataset."""
        pass
    
    @abstractmethod
    def get_anomaly_details(self, anomaly_id: str) -> Optional[AnomalyResult]:
        """Get details of specific anomaly."""
        pass
    
    @abstractmethod
    def get_detection_history(self, dataset_id: str) -> List[AnomalyResult]:
        """Get detection history for dataset."""
        pass
    
    @abstractmethod
    def validate_detection_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate detection parameters."""
        pass