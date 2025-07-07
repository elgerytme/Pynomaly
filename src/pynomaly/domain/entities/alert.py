"""Alert entity for monitoring and notification system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class AlertSeverity(Enum):
    """Severity level of an alert."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Status of an alert."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


class AlertType(Enum):
    """Type of alert."""

    ANOMALY_DETECTION = "anomaly_detection"
    MODEL_PERFORMANCE = "model_performance"
    DATA_QUALITY = "data_quality"
    SYSTEM_HEALTH = "system_health"
    PIPELINE_FAILURE = "pipeline_failure"
    THRESHOLD_BREACH = "threshold_breach"
    DATA_DRIFT = "data_drift"
    MODEL_DRIFT = "model_drift"
    RESOURCE_USAGE = "resource_usage"
    SECURITY = "security"
    CUSTOM = "custom"


class AlertCategory(Enum):
    """Category classification for alerts."""

    OPERATIONAL = "operational"
    TECHNICAL = "technical"
    BUSINESS = "business"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"


@dataclass
class AlertMetadata:
    """Metadata for alerts."""
    
    metadata_id: str = field(default_factory=lambda: str(uuid4()))
    alert_id: str = ""
    key: str = ""
    value: Any = None
    value_type: str = "string"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AlertCorrelation:
    """Alert correlation information."""

    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    primary_alert_id: str = ""
    related_alert_ids: list[str] = field(default_factory=list)
    correlation_score: float = 0.0
    correlation_type: str = "pattern"
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate correlation data."""
        if not (0.0 <= self.correlation_score <= 1.0):
            raise ValueError("Correlation score must be between 0.0 and 1.0")


class NotificationChannel(Enum):
    """Available notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    DISCORD = "discord"


@dataclass
class AlertCondition:
    """Represents a condition that triggers an alert."""

    metric_name: str
    operator: str  # "gt", "lt", "eq", "gte", "lte", "ne"
    threshold: float
    time_window_minutes: int = 5
    consecutive_breaches: int = 1
    description: str = ""

    def __post_init__(self) -> None:
        """Validate condition after initialization."""
        valid_operators = ["gt", "lt", "eq", "gte", "lte", "ne"]
        if self.operator not in valid_operators:
            raise ValueError(f"Operator must be one of {valid_operators}")

        if self.time_window_minutes <= 0:
            raise ValueError("Time window must be positive")

        if self.consecutive_breaches <= 0:
            raise ValueError("Consecutive breaches must be positive")

    def evaluate(self, value: float) -> bool:
        """Evaluate if the condition is met."""
        if self.operator == "gt":
            return value > self.threshold
        elif self.operator == "lt":
            return value < self.threshold
        elif self.operator == "eq":
            return value == self.threshold
        elif self.operator == "gte":
            return value >= self.threshold
        elif self.operator == "lte":
            return value <= self.threshold
        elif self.operator == "ne":
            return value != self.threshold
        return False

    def get_description(self) -> str:
        """Get human-readable description of the condition."""
        if self.description:
            return self.description

        operator_text = {
            "gt": "greater than",
            "lt": "less than",
            "eq": "equal to",
            "gte": "greater than or equal to",
            "lte": "less than or equal to",
            "ne": "not equal to",
        }

        return f"{self.metric_name} {operator_text[self.operator]} {self.threshold}"


@dataclass
class AlertNotification:
    """Represents a notification sent for an alert."""

    id: UUID = field(default_factory=uuid4)
    alert_id: UUID = field(default=uuid4)
    channel: NotificationChannel = NotificationChannel.EMAIL
    recipient: str = ""
    sent_at: datetime | None = None
    delivered_at: datetime | None = None
    status: str = "pending"  # pending, sent, delivered, failed
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_delivered(self) -> bool:
        """Check if notification was delivered."""
        return self.status == "delivered"

    @property
    def is_failed(self) -> bool:
        """Check if notification failed."""
        return self.status == "failed"

    def mark_sent(self) -> None:
        """Mark notification as sent."""
        self.sent_at = datetime.utcnow()
        self.status = "sent"

    def mark_delivered(self) -> None:
        """Mark notification as delivered."""
        self.delivered_at = datetime.utcnow()
        self.status = "delivered"

    def mark_failed(self, error_message: str) -> None:
        """Mark notification as failed."""
        self.status = "failed"
        self.error_message = error_message


@dataclass
class Alert:
    """Represents a system alert for monitoring and notifications.

    An Alert monitors specific conditions and triggers notifications
    when those conditions are met. It tracks the alert lifecycle
    from creation to resolution.

    Attributes:
        id: Unique identifier for the alert
        name: Human-readable name for the alert
        description: Detailed description of what the alert monitors
        alert_type: Type of alert (anomaly, performance, etc.)
        severity: Severity level of the alert
        condition: The condition that triggers this alert
        created_at: When the alert was created
        created_by: User who created the alert
        status: Current status of the alert
        triggered_at: When the alert was last triggered
        acknowledged_at: When the alert was acknowledged
        resolved_at: When the alert was resolved
        acknowledged_by: User who acknowledged the alert
        resolved_by: User who resolved the alert
        source: Source system or component generating the alert
        tags: Semantic tags for organization
        metadata: Additional metadata about the alert
        notifications: List of notifications sent for this alert
        suppression_rules: Rules for suppressing duplicate alerts
        escalation_rules: Rules for escalating unacknowledged alerts
    """

    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: AlertCondition
    created_by: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: AlertStatus = AlertStatus.ACTIVE
    triggered_at: datetime | None = None
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    acknowledged_by: str | None = None
    resolved_by: str | None = None
    source: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    notifications: list[AlertNotification] = field(default_factory=list)
    suppression_rules: dict[str, Any] = field(default_factory=dict)
    escalation_rules: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alert after initialization."""
        if not self.name:
            raise ValueError("Alert name cannot be empty")

        if not self.description:
            raise ValueError("Alert description cannot be empty")

        if not isinstance(self.alert_type, AlertType):
            raise TypeError(
                f"Alert type must be AlertType, got {type(self.alert_type)}"
            )

        if not isinstance(self.severity, AlertSeverity):
            raise TypeError(
                f"Severity must be AlertSeverity, got {type(self.severity)}"
            )

        if not isinstance(self.condition, AlertCondition):
            raise TypeError(
                f"Condition must be AlertCondition, got {type(self.condition)}"
            )

        if not self.created_by:
            raise ValueError("Created by cannot be empty")

    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == AlertStatus.ACTIVE

    @property
    def is_acknowledged(self) -> bool:
        """Check if alert has been acknowledged."""
        return self.status == AlertStatus.ACKNOWLEDGED

    @property
    def is_resolved(self) -> bool:
        """Check if alert has been resolved."""
        return self.status == AlertStatus.RESOLVED

    @property
    def is_critical(self) -> bool:
        """Check if alert is critical severity."""
        return self.severity == AlertSeverity.CRITICAL

    @property
    def duration_minutes(self) -> float | None:
        """Get alert duration in minutes."""
        if not self.triggered_at:
            return None

        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.triggered_at).total_seconds() / 60

    @property
    def response_time_minutes(self) -> float | None:
        """Get time to acknowledgment in minutes."""
        if not self.triggered_at or not self.acknowledged_at:
            return None

        return (self.acknowledged_at - self.triggered_at).total_seconds() / 60

    @property
    def resolution_time_minutes(self) -> float | None:
        """Get time to resolution in minutes."""
        if not self.triggered_at or not self.resolved_at:
            return None

        return (self.resolved_at - self.triggered_at).total_seconds() / 60

    def trigger(
        self, triggered_by: str = "system", context: dict[str, Any] | None = None
    ) -> None:
        """Trigger the alert."""
        if self.status in [AlertStatus.RESOLVED, AlertStatus.EXPIRED]:
            # Reset alert for new trigger
            self.acknowledged_at = None
            self.acknowledged_by = None
            self.resolved_at = None
            self.resolved_by = None

        self.status = AlertStatus.ACTIVE
        self.triggered_at = datetime.utcnow()

        # Store trigger context
        self.metadata["triggered_by"] = triggered_by
        self.metadata["trigger_context"] = context or {}
        self.metadata["trigger_count"] = self.metadata.get("trigger_count", 0) + 1

    def acknowledge(self, acknowledged_by: str, notes: str = "") -> None:
        """Acknowledge the alert."""
        if self.status != AlertStatus.ACTIVE:
            raise ValueError("Can only acknowledge active alerts")

        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = acknowledged_by

        if notes:
            self.metadata["acknowledgment_notes"] = notes

    def resolve(self, resolved_by: str, resolution_notes: str = "") -> None:
        """Resolve the alert."""
        if self.status in [AlertStatus.RESOLVED, AlertStatus.EXPIRED]:
            raise ValueError("Alert is already resolved")

        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.resolved_by = resolved_by

        if resolution_notes:
            self.metadata["resolution_notes"] = resolution_notes

        # Calculate resolution metrics
        if self.triggered_at:
            self.metadata["total_duration_minutes"] = self.duration_minutes
            if self.acknowledged_at:
                self.metadata["response_time_minutes"] = self.response_time_minutes
            self.metadata["resolution_time_minutes"] = self.resolution_time_minutes

    def suppress(
        self,
        suppressed_by: str,
        reason: str = "",
        duration_minutes: int | None = None,
    ) -> None:
        """Suppress the alert."""
        self.status = AlertStatus.SUPPRESSED
        self.metadata["suppressed_by"] = suppressed_by
        self.metadata["suppression_reason"] = reason
        self.metadata["suppressed_at"] = datetime.utcnow().isoformat()

        if duration_minutes:
            expiry_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
            self.metadata["suppression_expires_at"] = expiry_time.isoformat()

    def unsuppress(self) -> None:
        """Remove suppression from the alert."""
        if self.status == AlertStatus.SUPPRESSED:
            self.status = AlertStatus.ACTIVE
            self.metadata.pop("suppressed_by", None)
            self.metadata.pop("suppression_reason", None)
            self.metadata.pop("suppressed_at", None)
            self.metadata.pop("suppression_expires_at", None)

    def expire(self) -> None:
        """Mark alert as expired."""
        self.status = AlertStatus.EXPIRED
        self.metadata["expired_at"] = datetime.utcnow().isoformat()

    def add_notification(self, notification: AlertNotification) -> None:
        """Add a notification to the alert."""
        notification.alert_id = self.id
        self.notifications.append(notification)

    def get_notification_summary(self) -> dict[str, int]:
        """Get summary of notification statuses."""
        summary = {"pending": 0, "sent": 0, "delivered": 0, "failed": 0}

        for notification in self.notifications:
            status = notification.status
            if status in summary:
                summary[status] += 1

        return summary

    def should_escalate(self) -> bool:
        """Check if alert should be escalated."""
        if not self.escalation_rules or self.status != AlertStatus.ACTIVE:
            return False

        escalation_time = self.escalation_rules.get("escalation_time_minutes", 30)
        if not self.triggered_at:
            return False

        minutes_since_trigger = (
            datetime.utcnow() - self.triggered_at
        ).total_seconds() / 60
        return minutes_since_trigger >= escalation_time and not self.acknowledged_at

    def should_auto_resolve(self) -> bool:
        """Check if alert should be auto-resolved."""
        auto_resolve_time = self.metadata.get("auto_resolve_time_minutes")
        if not auto_resolve_time or not self.triggered_at:
            return False

        minutes_since_trigger = (
            datetime.utcnow() - self.triggered_at
        ).total_seconds() / 60
        return minutes_since_trigger >= auto_resolve_time

    def is_suppression_expired(self) -> bool:
        """Check if suppression has expired."""
        if self.status != AlertStatus.SUPPRESSED:
            return False

        expiry_str = self.metadata.get("suppression_expires_at")
        if not expiry_str:
            return False

        expiry_time = datetime.fromisoformat(expiry_str)
        return datetime.utcnow() >= expiry_time

    def add_tag(self, tag: str) -> None:
        """Add a tag to the alert."""
        if tag and tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the alert."""
        if tag in self.tags:
            self.tags.remove(tag)

    def update_condition(self, new_condition: AlertCondition) -> None:
        """Update the alert condition."""
        self.condition = new_condition
        self.metadata["condition_updated_at"] = datetime.utcnow().isoformat()

    def set_escalation_rules(
        self, escalation_time_minutes: int, escalation_contacts: list[str]
    ) -> None:
        """Set escalation rules for the alert."""
        self.escalation_rules = {
            "escalation_time_minutes": escalation_time_minutes,
            "escalation_contacts": escalation_contacts,
            "escalation_enabled": True,
        }

    def set_suppression_rules(
        self, duplicate_window_minutes: int = 5, max_notifications_per_hour: int = 10
    ) -> None:
        """Set suppression rules to prevent alert spam."""
        self.suppression_rules = {
            "duplicate_window_minutes": duplicate_window_minutes,
            "max_notifications_per_hour": max_notifications_per_hour,
            "suppression_enabled": True,
        }

    def get_info(self) -> dict[str, Any]:
        """Get comprehensive information about the alert."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "triggered_at": (
                self.triggered_at.isoformat() if self.triggered_at else None
            ),
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_by": self.resolved_by,
            "source": self.source,
            "duration_minutes": self.duration_minutes,
            "response_time_minutes": self.response_time_minutes,
            "resolution_time_minutes": self.resolution_time_minutes,
            "condition": {
                "metric_name": self.condition.metric_name,
                "operator": self.condition.operator,
                "threshold": self.condition.threshold,
                "description": self.condition.get_description(),
            },
            "notification_summary": self.get_notification_summary(),
            "should_escalate": self.should_escalate(),
            "tags": self.tags.copy(),
            "metadata": self.metadata.copy(),
        }

    def get_timeline(self) -> list[dict[str, Any]]:
        """Get chronological timeline of alert events."""
        timeline = []

        timeline.append(
            {
                "timestamp": self.created_at.isoformat(),
                "event": "created",
                "user": self.created_by,
                "details": f"Alert '{self.name}' created",
            }
        )

        if self.triggered_at:
            timeline.append(
                {
                    "timestamp": self.triggered_at.isoformat(),
                    "event": "triggered",
                    "user": self.metadata.get("triggered_by", "system"),
                    "details": f"Alert triggered - {self.condition.get_description()}",
                }
            )

        if self.acknowledged_at and self.acknowledged_by:
            timeline.append(
                {
                    "timestamp": self.acknowledged_at.isoformat(),
                    "event": "acknowledged",
                    "user": self.acknowledged_by,
                    "details": self.metadata.get(
                        "acknowledgment_notes", "Alert acknowledged"
                    ),
                }
            )

        if self.resolved_at and self.resolved_by:
            timeline.append(
                {
                    "timestamp": self.resolved_at.isoformat(),
                    "event": "resolved",
                    "user": self.resolved_by,
                    "details": self.metadata.get("resolution_notes", "Alert resolved"),
                }
            )

        # Add notification events
        for notification in self.notifications:
            if notification.sent_at:
                timeline.append(
                    {
                        "timestamp": notification.sent_at.isoformat(),
                        "event": "notification_sent",
                        "user": "system",
                        "details": f"Notification sent via {notification.channel.value} to {notification.recipient}",
                    }
                )

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        return timeline

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Alert('{self.name}', {self.severity.value}, "
            f"status={self.status.value}, type={self.alert_type.value})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Alert(id={self.id}, name='{self.name}', "
            f"severity={self.severity.value}, status={self.status.value})"
        )
