"""Quality anomaly detection entity for ML-enhanced quality detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from uuid import uuid4

from ..value_objects.quality_scores import QualityScores


class AnomalyType(Enum):
    """Types of quality anomalies."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_DEVIATION = "pattern_deviation"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    DRIFT_DETECTION = "drift_detection"
    CLUSTERING_ANOMALY = "clustering_anomaly"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    CORRELATION_ANOMALY = "correlation_anomaly"
    DISTRIBUTION_SHIFT = "distribution_shift"


class AnomalySeverity(Enum):
    """Severity levels for quality anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyStatus(Enum):
    """Status of quality anomalies."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class QualityAnomalyId:
    """Unique identifier for quality anomaly."""
    value: str = field(default_factory=lambda: str(uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class AnomalyDetectionResult:
    """Result of anomaly detection analysis."""
    anomaly_score: float
    confidence_score: float
    feature_contributions: Dict[str, float]
    detection_method: str
    model_version: str
    detected_at: datetime
    
    def is_anomaly(self, threshold: float = 0.5) -> bool:
        """Check if result indicates an anomaly."""
        return self.anomaly_score > threshold
    
    def get_top_contributing_features(self, n: int = 5) -> List[tuple[str, float]]:
        """Get top N features contributing to anomaly."""
        sorted_features = sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:n]


@dataclass(frozen=True)
class AnomalyPattern:
    """Pattern information for quality anomaly."""
    pattern_type: str
    pattern_strength: float
    pattern_description: str
    historical_occurrences: int
    last_occurrence: Optional[datetime] = None
    
    def is_recurring_pattern(self) -> bool:
        """Check if this is a recurring pattern."""
        return self.historical_occurrences > 2


@dataclass
class QualityAnomaly:
    """Entity representing a quality anomaly detected by ML models."""
    
    anomaly_id: QualityAnomalyId
    dataset_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    status: AnomalyStatus
    title: str
    description: str
    detection_result: AnomalyDetectionResult
    affected_columns: List[str]
    affected_records: int
    quality_impact: QualityScores
    root_cause_analysis: Dict[str, Any]
    patterns: List[AnomalyPattern] = field(default_factory=list)
    
    # Temporal information
    detected_at: datetime = field(default_factory=datetime.now)
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    occurrence_count: int = 1
    
    # Investigation and resolution
    investigated_by: Optional[str] = None
    investigated_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # ML model information
    model_name: str = "default_anomaly_detector"
    model_version: str = "1.0.0"
    training_data_period: Optional[tuple[datetime, datetime]] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.first_occurrence is None:
            self.first_occurrence = self.detected_at
        if self.last_occurrence is None:
            self.last_occurrence = self.detected_at
        
        # Ensure severity is consistent with anomaly score
        if self.detection_result.anomaly_score >= 0.9:
            self.severity = AnomalySeverity.CRITICAL
        elif self.detection_result.anomaly_score >= 0.7:
            self.severity = AnomalySeverity.HIGH
        elif self.detection_result.anomaly_score >= 0.5:
            self.severity = AnomalySeverity.MEDIUM
        else:
            self.severity = AnomalySeverity.LOW
    
    def start_investigation(self, investigator: str) -> None:
        """Start investigation of the anomaly."""
        self.status = AnomalyStatus.INVESTIGATING
        self.investigated_by = investigator
        self.investigated_at = datetime.now()
    
    def confirm_anomaly(self, notes: str = None) -> None:
        """Confirm the anomaly as a true positive."""
        self.status = AnomalyStatus.CONFIRMED
        if notes:
            self.resolution_notes = notes
    
    def mark_false_positive(self, notes: str = None) -> None:
        """Mark the anomaly as a false positive."""
        self.status = AnomalyStatus.FALSE_POSITIVE
        if notes:
            self.resolution_notes = notes
    
    def resolve_anomaly(self, resolution_notes: str, resolver: str = None) -> None:
        """Resolve the anomaly."""
        self.status = AnomalyStatus.RESOLVED
        self.resolution_notes = resolution_notes
        self.resolved_at = datetime.now()
        if resolver:
            self.investigated_by = resolver
    
    def add_occurrence(self, detected_at: datetime = None) -> None:
        """Add a new occurrence of this anomaly."""
        if detected_at is None:
            detected_at = datetime.now()
        
        self.occurrence_count += 1
        self.last_occurrence = detected_at
        
        # Reset status if resolved
        if self.status == AnomalyStatus.RESOLVED:
            self.status = AnomalyStatus.DETECTED
    
    def add_pattern(self, pattern: AnomalyPattern) -> None:
        """Add a pattern to the anomaly."""
        self.patterns.append(pattern)
    
    def get_severity_score(self) -> float:
        """Get numerical severity score."""
        severity_scores = {
            AnomalySeverity.LOW: 0.25,
            AnomalySeverity.MEDIUM: 0.5,
            AnomalySeverity.HIGH: 0.75,
            AnomalySeverity.CRITICAL: 1.0
        }
        return severity_scores[self.severity]
    
    def get_quality_impact_score(self) -> float:
        """Get overall quality impact score."""
        return self.quality_impact.overall_score
    
    def is_recurring(self) -> bool:
        """Check if this is a recurring anomaly."""
        return self.occurrence_count > 1
    
    def is_critical(self) -> bool:
        """Check if anomaly is critical."""
        return self.severity == AnomalySeverity.CRITICAL
    
    def is_resolved(self) -> bool:
        """Check if anomaly is resolved."""
        return self.status == AnomalyStatus.RESOLVED
    
    def get_time_to_resolution(self) -> Optional[float]:
        """Get time to resolution in hours."""
        if self.resolved_at and self.detected_at:
            delta = self.resolved_at - self.detected_at
            return delta.total_seconds() / 3600
        return None
    
    def get_top_contributing_features(self, n: int = 5) -> List[tuple[str, float]]:
        """Get top contributing features to the anomaly."""
        return self.detection_result.get_top_contributing_features(n)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of the anomaly."""
        return {
            'anomaly_id': str(self.anomaly_id),
            'dataset_id': self.dataset_id,
            'type': self.anomaly_type.value,
            'severity': self.severity.value,
            'status': self.status.value,
            'title': self.title,
            'description': self.description,
            'anomaly_score': self.detection_result.anomaly_score,
            'confidence_score': self.detection_result.confidence_score,
            'affected_columns': self.affected_columns,
            'affected_records': self.affected_records,
            'quality_impact': self.quality_impact.overall_score,
            'detected_at': self.detected_at.isoformat(),
            'occurrence_count': self.occurrence_count,
            'is_recurring': self.is_recurring(),
            'is_resolved': self.is_resolved(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'top_contributing_features': self.get_top_contributing_features(),
            'patterns': [
                {
                    'type': pattern.pattern_type,
                    'strength': pattern.pattern_strength,
                    'description': pattern.pattern_description,
                    'historical_occurrences': pattern.historical_occurrences
                }
                for pattern in self.patterns
            ],
            'tags': self.tags
        }
    
    def get_investigation_details(self) -> Dict[str, Any]:
        """Get investigation details."""
        return {
            'anomaly_id': str(self.anomaly_id),
            'status': self.status.value,
            'investigated_by': self.investigated_by,
            'investigated_at': self.investigated_at.isoformat() if self.investigated_at else None,
            'resolution_notes': self.resolution_notes,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'time_to_resolution_hours': self.get_time_to_resolution(),
            'root_cause_analysis': self.root_cause_analysis,
            'detection_details': {
                'method': self.detection_result.detection_method,
                'model_version': self.detection_result.model_version,
                'anomaly_score': self.detection_result.anomaly_score,
                'confidence_score': self.detection_result.confidence_score,
                'feature_contributions': self.detection_result.feature_contributions
            }
        }
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the anomaly."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the anomaly."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata for the anomaly."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)