"""Shared domain entities used across packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from .abstractions import BaseEntity


class AlertType(str, Enum):
    """Types of alerts that can be raised."""
    
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    SYSTEM_ERROR = "system_error"
    SECURITY_INCIDENT = "security_incident"
    RESOURCE_THRESHOLD = "resource_threshold"
    BUSINESS_METRIC = "business_metric"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""
    
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"
    
    @property
    def priority(self) -> int:
        """Get numeric priority for sorting."""
        priorities = {
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3, 
            AlertSeverity.CRITICAL: 4
        }
        return priorities[self]


class AlertStatus(str, Enum):
    """Status of an alert."""
    
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"


@dataclass
class AlertCondition:
    """Condition that triggers an alert."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    metric_name: str = ""
    operator: str = ""  # >, <, >=, <=, ==, !=
    threshold_value: float = 0.0
    current_value: Optional[float] = None
    evaluation_window_minutes: int = 5
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate condition after initialization."""
        if not self.name:
            raise ValueError("Condition name cannot be empty")
        
        if not self.metric_name:
            raise ValueError("Metric name cannot be empty")
        
        valid_operators = [">", "<", ">=", "<=", "==", "!="]
        if self.operator not in valid_operators:
            raise ValueError(f"Invalid operator. Must be one of: {valid_operators}")
        
        if self.evaluation_window_minutes <= 0:
            raise ValueError("Evaluation window must be positive")
    
    def evaluate(self, current_value: float) -> bool:
        """Evaluate if the condition is met."""
        self.current_value = current_value
        
        if self.operator == ">":
            return current_value > self.threshold_value
        elif self.operator == "<":
            return current_value < self.threshold_value
        elif self.operator == ">=":
            return current_value >= self.threshold_value
        elif self.operator == "<=":
            return current_value <= self.threshold_value
        elif self.operator == "==":
            return current_value == self.threshold_value
        elif self.operator == "!=":
            return current_value != self.threshold_value
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert condition to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "metric_name": self.metric_name,
            "operator": self.operator,
            "threshold_value": self.threshold_value,
            "current_value": self.current_value,
            "evaluation_window_minutes": self.evaluation_window_minutes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class Alert(BaseEntity):
    """Alert domain entity for system notifications."""
    
    # Alert identification
    title: str = ""
    message: str = ""
    alert_type: AlertType = AlertType.OPERATIONAL
    severity: AlertSeverity = AlertSeverity.MEDIUM
    status: AlertStatus = AlertStatus.ACTIVE
    
    # Source information
    source_system: str = ""
    source_component: str = ""
    source_instance: Optional[str] = None
    
    # Alert conditions
    conditions: List[AlertCondition] = field(default_factory=list)
    trigger_value: Optional[float] = None
    
    # Timing information
    first_occurred_at: datetime = field(default_factory=datetime.utcnow)
    last_occurred_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Assignment and escalation
    assigned_to: Optional[str] = None
    assigned_team: Optional[str] = None
    escalated_to: Optional[str] = None
    escalation_level: int = 0
    
    # Additional context
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    related_alerts: List[UUID] = field(default_factory=list)
    
    # Resolution information
    resolution_notes: str = ""
    resolution_action_taken: str = ""
    resolved_by: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate alert after initialization."""
        super().__post_init__()
        
        if not self.title:
            raise ValueError("Alert title cannot be empty")
        
        if not self.message:
            raise ValueError("Alert message cannot be empty")
        
        if not self.source_system:
            raise ValueError("Source system cannot be empty")
        
        if self.escalation_level < 0:
            raise ValueError("Escalation level cannot be negative")
    
    def acknowledge(self, acknowledged_by: str) -> None:
        """Acknowledge the alert."""
        if self.status == AlertStatus.RESOLVED:
            raise ValueError("Cannot acknowledge resolved alert")
        
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        self.assigned_to = acknowledged_by
        self.updated_at = datetime.utcnow()
    
    def resolve(self, resolved_by: str, resolution_notes: str = "", action_taken: str = "") -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.resolved_by = resolved_by
        self.resolution_notes = resolution_notes
        self.resolution_action_taken = action_taken
        self.updated_at = datetime.utcnow()
    
    def dismiss(self, dismissed_by: str, reason: str = "") -> None:
        """Dismiss the alert."""
        self.status = AlertStatus.DISMISSED
        self.resolved_at = datetime.utcnow()
        self.resolved_by = dismissed_by
        self.resolution_notes = f"Dismissed: {reason}"
        self.updated_at = datetime.utcnow()
    
    def escalate(self, escalated_to: str, reason: str = "") -> None:
        """Escalate the alert."""
        self.status = AlertStatus.ESCALATED
        self.escalated_to = escalated_to
        self.escalation_level += 1
        self.context["escalation_reason"] = reason
        self.context["escalation_timestamp"] = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow()
    
    def add_condition(self, condition: AlertCondition) -> None:
        """Add a condition to the alert."""
        if condition not in self.conditions:
            self.conditions.append(condition)
            self.updated_at = datetime.utcnow()
    
    def remove_condition(self, condition_id: UUID) -> bool:
        """Remove a condition from the alert."""
        for i, condition in enumerate(self.conditions):
            if condition.id == condition_id:
                self.conditions.pop(i)
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the alert."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the alert."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if alert has a specific tag."""
        return tag in self.tags
    
    def add_context(self, key: str, value: Any) -> None:
        """Add context information to the alert."""
        self.context[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context information from the alert."""
        return self.context.get(key, default)
    
    def is_active(self) -> bool:
        """Check if alert is active."""
        return self.status == AlertStatus.ACTIVE
    
    def is_resolved(self) -> bool:
        """Check if alert is resolved."""
        return self.status == AlertStatus.RESOLVED
    
    def is_acknowledged(self) -> bool:
        """Check if alert is acknowledged."""
        return self.status == AlertStatus.ACKNOWLEDGED
    
    def is_critical(self) -> bool:
        """Check if alert is critical severity."""
        return self.severity == AlertSeverity.CRITICAL
    
    def is_high_priority(self) -> bool:
        """Check if alert is high priority (high or critical)."""
        return self.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    def get_duration(self) -> Optional[float]:
        """Get alert duration in seconds."""
        if self.resolved_at:
            return (self.resolved_at - self.first_occurred_at).total_seconds()
        return (datetime.utcnow() - self.first_occurred_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "title": self.title,
            "message": self.message,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "source_system": self.source_system,
            "source_component": self.source_component,
            "source_instance": self.source_instance,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "trigger_value": self.trigger_value,
            "first_occurred_at": self.first_occurred_at.isoformat(),
            "last_occurred_at": self.last_occurred_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "assigned_to": self.assigned_to,
            "assigned_team": self.assigned_team,
            "escalated_to": self.escalated_to,
            "escalation_level": self.escalation_level,
            "tags": self.tags,
            "context": self.context,
            "related_alerts": [str(alert_id) for alert_id in self.related_alerts],
            "resolution_notes": self.resolution_notes,
            "resolution_action_taken": self.resolution_action_taken,
            "resolved_by": self.resolved_by,
            "is_active": self.is_active(),
            "is_resolved": self.is_resolved(),
            "is_acknowledged": self.is_acknowledged(),
            "is_critical": self.is_critical(),
            "is_high_priority": self.is_high_priority(),
            "duration_seconds": self.get_duration()
        })
        return base_dict
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"Alert('{self.title}', {self.severity.value}, {self.status.value})"
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return (
            f"Alert(id={self.id}, title='{self.title}', "
            f"severity={self.severity.value}, status={self.status.value})"
        )