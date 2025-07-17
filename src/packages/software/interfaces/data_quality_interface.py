"""Interface for data quality domain communication."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class QualityIssue:
    """Data quality issue representation."""
    type: str
    severity: float
    description: str
    affected_columns: List[str]
    recommendation: str


@dataclass
class QualityReport:
    """Data quality assessment report."""
    overall_score: float
    quality_level: QualityLevel
    issues: List[QualityIssue]
    recommendations: List[str]
    metadata: Dict[str, Any]


class DataQualityProtocol(Protocol):
    """Protocol for data quality service interactions."""
    
    def assess_quality(self, data: Any) -> QualityReport:
        """Assess data quality and return report."""
        ...
    
    def get_quality_metrics(self, data: Any) -> Dict[str, float]:
        """Get quality measurements for data."""
        ...


class DataQualityInterface(ABC):
    """Abstract interface for data quality operations."""
    
    @abstractmethod
    def assess_dataset_quality(self, dataset_id: str) -> QualityReport:
        """Assess quality of a data_collection."""
        pass
    
    @abstractmethod
    def monitor_quality_trends(self, dataset_id: str) -> List[QualityReport]:
        """Monitor quality trends over time."""
        pass
    
    @abstractmethod
    def get_quality_recommendations(self, dataset_id: str) -> List[str]:
        """Get recommendations for improving data quality."""
        pass