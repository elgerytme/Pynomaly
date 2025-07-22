"""Anomaly entity for representing individual anomalous data points."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    UNKNOWN = "unknown"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Represents a single anomalous data point.
    
    Contains information about the anomaly including its location,
    confidence, features, and metadata for explanation and action.
    """
    
    index: int
    confidence_score: float
    anomaly_type: AnomalyType = AnomalyType.UNKNOWN
    severity: Optional[AnomalySeverity] = None
    feature_values: Optional[npt.NDArray[np.floating]] = None
    feature_contributions: Optional[Dict[str, float]] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Initialize derived fields after object creation."""
        if self.metadata is None:
            self.metadata = {}
            
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
            
        # Determine severity based on confidence if not provided
        if self.severity is None:
            self.severity = self._calculate_severity()
    
    def _calculate_severity(self) -> AnomalySeverity:
        """Calculate severity based on confidence score."""
        if self.confidence_score >= 0.9:
            return AnomalySeverity.CRITICAL
        elif self.confidence_score >= 0.75:
            return AnomalySeverity.HIGH
        elif self.confidence_score >= 0.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def get_top_contributing_features(self, n: int = 5) -> List[tuple[str, float]]:
        """Get top N features contributing to the anomaly.
        
        Args:
            n: Number of top contributing features to return
            
        Returns:
            List of (feature_name, contribution_score) tuples
        """
        if not self.feature_contributions:
            return []
            
        sorted_features = sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_features[:n]
    
    def is_high_priority(self) -> bool:
        """Check if anomaly requires immediate attention."""
        return self.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly to dictionary for serialization."""
        return {
            "index": self.index,
            "confidence_score": float(self.confidence_score),
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value if self.severity else None,
            "feature_values": self.feature_values.tolist() if self.feature_values is not None else None,
            "feature_contributions": self.feature_contributions,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
            "explanation": self.explanation,
            "is_high_priority": self.is_high_priority()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Anomaly:
        """Create anomaly from dictionary."""
        return cls(
            index=data["index"],
            confidence_score=data["confidence_score"],
            anomaly_type=AnomalyType(data.get("anomaly_type", "unknown")),
            severity=AnomalySeverity(data["severity"]) if data.get("severity") else None,
            feature_values=np.array(data["feature_values"]) if data.get("feature_values") else None,
            feature_contributions=data.get("feature_contributions"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            metadata=data.get("metadata", {}),
            explanation=data.get("explanation")
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Anomaly(index={self.index}, confidence={self.confidence_score:.3f}, "
            f"type={self.anomaly_type.value}, severity={self.severity.value if self.severity else 'unknown'})"
        )
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return self.__str__()