"""Model Performance Degradation domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from ..value_objects.performance_degradation_metrics import PerformanceDegradationMetrics


class DegradationStatus(Enum):
    """Status of performance degradation."""
    
    MONITORING = "monitoring"
    DETECTED = "detected"
    ALERT_RAISED = "alert_raised"
    INVESTIGATING = "investigating"
    RECOVERING = "recovering"
    RECOVERED = "recovered"
    ACKNOWLEDGED = "acknowledged"
    IGNORED = "ignored"


class RecoveryAction(Enum):
    """Actions taken to recover from performance degradation."""
    
    NONE = "none"
    RETRAIN_MODEL = "retrain_model"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_QUALITY_FIX = "data_quality_fix"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ARCHITECTURE_CHANGE = "architecture_change"
    ROLLBACK_MODEL = "rollback_model"
    SCALE_RESOURCES = "scale_resources"
    MANUAL_INTERVENTION = "manual_intervention"


class DegradationSeverity(Enum):
    """Severity levels for performance degradation."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelPerformanceDegradation:
    """Entity representing model performance degradation detection and tracking."""
    
    # Core identification
    id: UUID = field(default_factory=uuid4)
    model_id: str = ""
    model_version: str = ""
    
    # Degradation details
    status: DegradationStatus = DegradationStatus.MONITORING
    severity: DegradationSeverity = DegradationSeverity.LOW
    degradation_type: str = ""  # e.g., "accuracy_drop", "latency_increase", "concept_drift"
    
    # Metrics and measurements
    baseline_metrics: PerformanceDegradationMetrics = field(default_factory=PerformanceDegradationMetrics)
    current_metrics: PerformanceDegradationMetrics = field(default_factory=PerformanceDegradationMetrics)
    degradation_percentage: float = 0.0
    
    # Detection information
    detected_at: Optional[datetime] = None
    detection_method: str = ""  # e.g., "statistical_test", "threshold_check", "ml_detector"
    confidence_score: float = 0.0
    
    # Recovery tracking
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    recovery_started_at: Optional[datetime] = None
    recovery_completed_at: Optional[datetime] = None
    
    # Contextual information
    affected_features: List[str] = field(default_factory=list)
    potential_causes: List[str] = field(default_factory=list)
    business_impact: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    
    # User tracking
    created_by: str = ""
    acknowledged_by: str = ""
    assigned_to: str = ""
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    alert_rules_triggered: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate entity state after initialization."""
        if not self.model_id:
            raise ValueError("Model ID is required")
        
        if self.degradation_percentage < 0 or self.degradation_percentage > 100:
            raise ValueError("Degradation percentage must be between 0 and 100")
        
        if self.confidence_score < 0 or self.confidence_score > 1:
            raise ValueError("Confidence score must be between 0 and 1")
    
    def is_active(self) -> bool:
        """Check if degradation is currently active."""
        return self.status in [
            DegradationStatus.DETECTED,
            DegradationStatus.ALERT_RAISED,
            DegradationStatus.INVESTIGATING,
            DegradationStatus.RECOVERING
        ]
    
    def is_recovered(self) -> bool:
        """Check if degradation has been recovered."""
        return self.status == DegradationStatus.RECOVERED
    
    def is_critical(self) -> bool:
        """Check if degradation is critical severity."""
        return self.severity == DegradationSeverity.CRITICAL
    
    def get_duration_hours(self) -> Optional[float]:
        """Get duration of degradation in hours."""
        if not self.detected_at:
            return None
        
        end_time = self.recovery_completed_at or datetime.utcnow()
        duration = end_time - self.detected_at
        return duration.total_seconds() / 3600
    
    def add_recovery_action(self, action: RecoveryAction) -> None:
        """Add a recovery action to the degradation."""
        if action not in self.recovery_actions:
            self.recovery_actions.append(action)
            self.updated_at = datetime.utcnow()
            
            if not self.recovery_started_at:
                self.recovery_started_at = datetime.utcnow()
    
    def mark_as_recovered(self, recovery_time: Optional[datetime] = None) -> None:
        """Mark the degradation as recovered."""
        self.status = DegradationStatus.RECOVERED
        self.recovery_completed_at = recovery_time or datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def acknowledge(self, acknowledged_by: str, acknowledged_at: Optional[datetime] = None) -> None:
        """Acknowledge the degradation."""
        self.status = DegradationStatus.ACKNOWLEDGED
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = acknowledged_at or datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def escalate_severity(self) -> None:
        """Escalate the degradation severity."""
        severity_order = [
            DegradationSeverity.LOW,
            DegradationSeverity.MEDIUM,
            DegradationSeverity.HIGH,
            DegradationSeverity.CRITICAL
        ]
        
        current_index = severity_order.index(self.severity)
        if current_index < len(severity_order) - 1:
            self.severity = severity_order[current_index + 1]
            self.updated_at = datetime.utcnow()
    
    def add_potential_cause(self, cause: str) -> None:
        """Add a potential cause for the degradation."""
        if cause not in self.potential_causes:
            self.potential_causes.append(cause)
            self.updated_at = datetime.utcnow()
    
    def update_metrics(self, current_metrics: PerformanceDegradationMetrics) -> None:
        """Update current performance metrics."""
        self.current_metrics = current_metrics
        self.updated_at = datetime.utcnow()
        
        # Recalculate degradation percentage if baseline exists
        if self.baseline_metrics.primary_metric_value > 0:
            baseline_value = self.baseline_metrics.primary_metric_value
            current_value = current_metrics.primary_metric_value
            self.degradation_percentage = abs((baseline_value - current_value) / baseline_value) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "id": str(self.id),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "status": self.status.value,
            "severity": self.severity.value,
            "degradation_type": self.degradation_type,
            "degradation_percentage": self.degradation_percentage,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "detection_method": self.detection_method,
            "confidence_score": self.confidence_score,
            "recovery_actions": [action.value for action in self.recovery_actions],
            "recovery_started_at": self.recovery_started_at.isoformat() if self.recovery_started_at else None,
            "recovery_completed_at": self.recovery_completed_at.isoformat() if self.recovery_completed_at else None,
            "affected_features": self.affected_features,
            "potential_causes": self.potential_causes,
            "business_impact": self.business_impact,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "created_by": self.created_by,
            "acknowledged_by": self.acknowledged_by,
            "assigned_to": self.assigned_to,
            "tags": self.tags,
            "metadata": self.metadata,
            "alert_rules_triggered": self.alert_rules_triggered,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "current_metrics": self.current_metrics.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelPerformanceDegradation:
        """Create entity from dictionary representation."""
        # Parse datetime fields
        detected_at = datetime.fromisoformat(data["detected_at"]) if data.get("detected_at") else None
        recovery_started_at = datetime.fromisoformat(data["recovery_started_at"]) if data.get("recovery_started_at") else None
        recovery_completed_at = datetime.fromisoformat(data["recovery_completed_at"]) if data.get("recovery_completed_at") else None
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        acknowledged_at = datetime.fromisoformat(data["acknowledged_at"]) if data.get("acknowledged_at") else None
        
        # Parse enums
        status = DegradationStatus(data["status"])
        severity = DegradationSeverity(data["severity"])
        recovery_actions = [RecoveryAction(action) for action in data.get("recovery_actions", [])]
        
        # Create metrics objects
        baseline_metrics = PerformanceDegradationMetrics.from_dict(data.get("baseline_metrics", {}))
        current_metrics = PerformanceDegradationMetrics.from_dict(data.get("current_metrics", {}))
        
        return cls(
            id=UUID(data["id"]),
            model_id=data["model_id"],
            model_version=data["model_version"],
            status=status,
            severity=severity,
            degradation_type=data["degradation_type"],
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            degradation_percentage=data["degradation_percentage"],
            detected_at=detected_at,
            detection_method=data["detection_method"],
            confidence_score=data["confidence_score"],
            recovery_actions=recovery_actions,
            recovery_started_at=recovery_started_at,
            recovery_completed_at=recovery_completed_at,
            affected_features=data.get("affected_features", []),
            potential_causes=data.get("potential_causes", []),
            business_impact=data.get("business_impact", ""),
            created_at=created_at,
            updated_at=updated_at,
            acknowledged_at=acknowledged_at,
            created_by=data.get("created_by", ""),
            acknowledged_by=data.get("acknowledged_by", ""),
            assigned_to=data.get("assigned_to", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            alert_rules_triggered=data.get("alert_rules_triggered", [])
        )