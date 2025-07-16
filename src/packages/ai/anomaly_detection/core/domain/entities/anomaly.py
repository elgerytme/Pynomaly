"""Enhanced anomaly entity with advanced classification support."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pynomaly.domain.value_objects import AnomalyScore

# Import advanced classification types
try:
    from pynomaly.domain.value_objects.anomaly_classification import (
        AdvancedAnomalyClassification,
        ClassificationResult,
        ConfidenceLevel,
    )
    from pynomaly.domain.value_objects.anomaly_type import AnomalyType
    from pynomaly.domain.value_objects.severity_score import SeverityLevel
except ImportError:
    # Fallback for backward compatibility
    AdvancedAnomalyClassification = None
    ClassificationResult = None
    ConfidenceLevel = None
    AnomalyType = None
    SeverityLevel = None


@dataclass
class Anomaly:
    """Enhanced anomaly entity with advanced classification support."""

    score: float | AnomalyScore
    data_point: dict[str, Any]
    detector_name: str
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: str | None = None
    
    # Advanced classification fields
    classification: AdvancedAnomalyClassification | None = None
    anomaly_type: AnomalyType | None = None
    severity_level: SeverityLevel | None = None
    confidence_level: ConfidenceLevel | None = None

    def __post_init__(self) -> None:
        """Validate anomaly after initialization."""
        if not isinstance(self.score, (int, float, AnomalyScore)):
            raise TypeError(
                f"Score must be a number or AnomalyScore, got {type(self.score)}"
            )

        if not self.detector_name:
            raise ValueError("Detector name cannot be empty")

        if not isinstance(self.data_point, dict):
            raise TypeError(
                f"Data point must be a dictionary, got {type(self.data_point)}"
            )

    @property
    def severity(self) -> str:
        """Categorize anomaly severity based on advanced classification or score."""
        # Use advanced classification if available
        if self.classification:
            return self.classification.severity_classification
        
        # Use severity_level if available
        if self.severity_level and SeverityLevel:
            return self.severity_level.value.lower()
        
        # Fallback to score-based severity
        score_value = (
            self.score.value if isinstance(self.score, AnomalyScore) else self.score
        )

        if score_value > 0.9:
            return "critical"
        elif score_value > 0.7:
            return "high"
        elif score_value > 0.5:
            return "medium"
        else:
            return "low"

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the anomaly."""
        self.metadata[key] = value
    
    def get_anomaly_type(self) -> str:
        """Get the anomaly type as string."""
        if self.anomaly_type and AnomalyType:
            return self.anomaly_type.value.lower()
        if self.classification and self.classification.hierarchical_classification:
            return self.classification.hierarchical_classification.primary_category.lower()
        return "unknown"
    
    def get_confidence_score(self) -> float:
        """Get the confidence score for this anomaly."""
        if self.classification:
            return self.classification.get_confidence_score()
        # Default confidence based on score magnitude
        score_value = (
            self.score.value if isinstance(self.score, AnomalyScore) else self.score
        )
        return min(abs(score_value), 1.0)
    
    def get_classification_details(self) -> dict[str, Any]:
        """Get detailed classification information."""
        if not self.classification:
            return {"type": "basic", "classification": self.get_anomaly_type()}
        
        details = {
            "type": "advanced",
            "primary_class": self.classification.get_primary_class(),
            "confidence_score": self.classification.get_confidence_score(),
            "severity": self.classification.severity_classification,
        }
        
        if self.classification.is_hierarchical():
            hierarchy = self.classification.hierarchical_classification
            details["hierarchy"] = {
                "primary": hierarchy.primary_category,
                "secondary": hierarchy.secondary_category,
                "tertiary": hierarchy.tertiary_category,
                "path": hierarchy.get_full_path(),
                "depth": hierarchy.get_hierarchy_depth(),
            }
        
        if self.classification.is_multi_class():
            multi_class = self.classification.multi_class_classification
            details["multi_class"] = {
                "primary": multi_class.primary_result.predicted_class,
                "alternatives": [
                    {
                        "class": result.predicted_class,
                        "confidence": result.confidence_score,
                    }
                    for result in multi_class.alternative_results
                ],
                "strategy": multi_class.multi_class_strategy,
            }
        
        if self.classification.has_spatial_context():
            details["spatial_context"] = self.classification.spatial_classification
        
        return details
    
    def is_highly_confident(self) -> bool:
        """Check if this anomaly classification is highly confident."""
        confidence = self.get_confidence_score()
        return confidence >= 0.8
    
    def is_critical_severity(self) -> bool:
        """Check if this anomaly is of critical severity."""
        return self.severity == "critical"
    
    def has_advanced_classification(self) -> bool:
        """Check if this anomaly has advanced classification."""
        return self.classification is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert anomaly to dictionary representation."""
        # Get the numeric value from score
        score_value = (
            self.score.value if isinstance(self.score, AnomalyScore) else self.score
        )

        result = {
            "id": str(self.id),
            "score": score_value,
            "detector_name": self.detector_name,
            "timestamp": self.timestamp.isoformat(),
            "data_point": self.data_point,
            "metadata": self.metadata,
            "severity": self.severity,
            "explanation": self.explanation,
            "anomaly_type": self.get_anomaly_type(),
            "confidence_score": self.get_confidence_score(),
            "has_advanced_classification": self.has_advanced_classification(),
        }
        
        # Add advanced classification details if available
        if self.has_advanced_classification():
            result["classification_details"] = self.get_classification_details()
        
        # Add confidence level if available
        if self.confidence_level and ConfidenceLevel:
            result["confidence_level"] = self.confidence_level.value.lower()
        
        return result

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, Anomaly):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)
