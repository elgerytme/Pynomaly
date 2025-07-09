"""
Domain entities for third-party integrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pynomaly.shared.types import TenantId, UserId


class IntegrationType(str, Enum):
    """Types of supported integrations."""
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    DISCORD = "discord"
    EMAIL = "email"
    WEBHOOK = "webhook"
    JIRA = "jira"
    SERVICENOW = "servicenow"
    SPLUNK = "splunk"
    DATADOG = "datadog"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"


class IntegrationStatus(str, Enum):
    """Integration status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING_AUTH = "pending_auth"
    SUSPENDED = "suspended"


class NotificationLevel(str, Enum):
    """Notification severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TriggerType(str, Enum):
    """Types of triggers that can activate integrations."""
    ANOMALY_DETECTED = "anomaly_detected"
    SYSTEM_ERROR = "system_error"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    MODEL_DRIFT = "model_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COMPLIANCE_VIOLATION = "compliance_violation"
    CUSTOM_ALERT = "custom_alert"
    SCHEDULED_REPORT = "scheduled_report"


@dataclass
class IntegrationCredentials:
    """Secure storage for integration credentials."""
    encrypted_data: str
    encryption_key_id: str
    expires_at: datetime | None = None
    scopes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationConfig:
    """Configuration for a specific integration."""
    # Basic settings
    enabled: bool = True
    notification_levels: list[NotificationLevel] = field(default_factory=list)
    triggers: list[TriggerType] = field(default_factory=list)

    # Delivery settings
    retry_count: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 30
    rate_limit_per_minute: int = 60

    # Content settings
    template_id: str | None = None
    custom_template: str | None = None
    include_charts: bool = False
    include_raw_data: bool = False

    # Filtering
    filters: dict[str, Any] = field(default_factory=dict)

    # Integration-specific settings
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class Integration:
    """Third-party integration entity."""
    id: str
    name: str
    integration_type: IntegrationType
    tenant_id: TenantId
    created_by: UserId
    status: IntegrationStatus
    config: IntegrationConfig
    credentials: IntegrationCredentials | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered: datetime | None = None
    last_error: str | None = None
    trigger_count: int = 0
    success_count: int = 0
    error_count: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.trigger_count == 0:
            return 0.0
        return (self.success_count / self.trigger_count) * 100

    def is_healthy(self) -> bool:
        """Check if integration is healthy."""
        return (
            self.status == IntegrationStatus.ACTIVE and
            self.success_rate >= 90.0 and
            self.error_count < 10
        )


@dataclass
class NotificationPayload:
    """Payload for sending notifications."""
    trigger_type: TriggerType
    level: NotificationLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tenant_id: TenantId | None = None
    user_id: UserId | None = None
    data: dict[str, Any] = field(default_factory=dict)
    attachments: list[dict[str, Any]] = field(default_factory=list)

    def to_slack_format(self) -> dict[str, Any]:
        """Convert to Slack message format."""
        color_map = {
            NotificationLevel.INFO: "#36a64f",      # Green
            NotificationLevel.WARNING: "#ff9500",   # Orange
            NotificationLevel.ERROR: "#ff0000",     # Red
            NotificationLevel.CRITICAL: "#8b0000"   # Dark red
        }

        return {
            "text": self.title,
            "attachments": [
                {
                    "color": color_map.get(self.level, "#36a64f"),
                    "fields": [
                        {
                            "title": "Message",
                            "value": self.message,
                            "short": False
                        },
                        {
                            "title": "Level",
                            "value": self.level.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        }
                    ]
                }
            ] + self.attachments
        }

    def to_teams_format(self) -> dict[str, Any]:
        """Convert to Microsoft Teams message format."""
        theme_color_map = {
            NotificationLevel.INFO: "0078D4",      # Blue
            NotificationLevel.WARNING: "FF8C00",   # Orange
            NotificationLevel.ERROR: "DC143C",     # Crimson
            NotificationLevel.CRITICAL: "8B0000"   # Dark red
        }

        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": theme_color_map.get(self.level, "0078D4"),
            "summary": self.title,
            "sections": [
                {
                    "activityTitle": self.title,
                    "activitySubtitle": f"Level: {self.level.value.upper()}",
                    "activityImage": "https://example.com/pynomaly-icon.png",
                    "facts": [
                        {
                            "name": "Message",
                            "value": self.message
                        },
                        {
                            "name": "Time",
                            "value": self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
                        }
                    ]
                }
            ]
        }

    def to_pagerduty_format(self) -> dict[str, Any]:
        """Convert to PagerDuty event format."""
        severity_map = {
            NotificationLevel.INFO: "info",
            NotificationLevel.WARNING: "warning",
            NotificationLevel.ERROR: "error",
            NotificationLevel.CRITICAL: "critical"
        }

        return {
            "payload": {
                "summary": self.title,
                "source": "pynomaly",
                "severity": severity_map.get(self.level, "info"),
                "timestamp": self.timestamp.isoformat(),
                "custom_details": {
                    "message": self.message,
                    "trigger_type": self.trigger_type.value,
                    "tenant_id": str(self.tenant_id) if self.tenant_id else None,
                    **self.data
                }
            },
            "event_action": "trigger"
        }

    def to_webhook_format(self) -> dict[str, Any]:
        """Convert to generic webhook format."""
        return {
            "event": "notification",
            "trigger_type": self.trigger_type.value,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "data": self.data,
            "attachments": self.attachments
        }


@dataclass
class NotificationTemplate:
    """Template for formatting notifications."""
    id: str
    name: str
    integration_type: IntegrationType
    trigger_types: list[TriggerType]
    title_template: str
    message_template: str
    tenant_id: TenantId
    created_by: UserId
    is_default: bool = False
    variables: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def render(self, context: dict[str, Any]) -> dict[str, str]:
        """Render template with provided context."""
        try:
            title = self.title_template.format(**context)
            message = self.message_template.format(**context)
            return {"title": title, "message": message}
        except KeyError as e:
            return {
                "title": "Template Error",
                "message": f"Missing template variable: {e}"
            }


@dataclass
class NotificationHistory:
    """History record of sent notifications."""
    id: str
    integration_id: str
    payload: NotificationPayload
    response_status: int
    response_body: str
    sent_at: datetime = field(default_factory=datetime.utcnow)
    delivery_time_ms: int | None = None
    retry_count: int = 0
    error_message: str | None = None

    @property
    def was_successful(self) -> bool:
        """Check if notification was successfully delivered."""
        return 200 <= self.response_status < 300


# Default notification templates
DEFAULT_TEMPLATES = {
    IntegrationType.SLACK: {
        TriggerType.ANOMALY_DETECTED: NotificationTemplate(
            id="slack_anomaly_default",
            name="Slack Anomaly Alert",
            integration_type=IntegrationType.SLACK,
            trigger_types=[TriggerType.ANOMALY_DETECTED],
            title_template="🚨 Anomaly Detected in {dataset_name}",
            message_template="Detected {anomaly_count} anomalies in dataset '{dataset_name}' using detector '{detector_name}'. Confidence: {confidence}%",
            tenant_id="",  # Will be set per tenant
            created_by="",  # System template
            is_default=True,
            variables=["dataset_name", "anomaly_count", "detector_name", "confidence"]
        ),
        TriggerType.SYSTEM_ERROR: NotificationTemplate(
            id="slack_error_default",
            name="Slack System Error",
            integration_type=IntegrationType.SLACK,
            trigger_types=[TriggerType.SYSTEM_ERROR],
            title_template="⚠️ System Error in Pynomaly",
            message_template="System error occurred: {error_message}. Component: {component}. Time: {timestamp}",
            tenant_id="",
            created_by="",
            is_default=True,
            variables=["error_message", "component", "timestamp"]
        )
    },
    IntegrationType.PAGERDUTY: {
        TriggerType.ANOMALY_DETECTED: NotificationTemplate(
            id="pagerduty_anomaly_default",
            name="PagerDuty Anomaly Incident",
            integration_type=IntegrationType.PAGERDUTY,
            trigger_types=[TriggerType.ANOMALY_DETECTED],
            title_template="Anomaly Detected: {dataset_name}",
            message_template="Critical anomalies detected in {dataset_name}. Immediate investigation required. Confidence: {confidence}%",
            tenant_id="",
            created_by="",
            is_default=True,
            variables=["dataset_name", "confidence"]
        )
    },
    IntegrationType.TEAMS: {
        TriggerType.ANOMALY_DETECTED: NotificationTemplate(
            id="teams_anomaly_default",
            name="Teams Anomaly Alert",
            integration_type=IntegrationType.TEAMS,
            trigger_types=[TriggerType.ANOMALY_DETECTED],
            title_template="Anomaly Alert: {dataset_name}",
            message_template="Anomalies detected in {dataset_name}. Count: {anomaly_count}. Detector: {detector_name}",
            tenant_id="",
            created_by="",
            is_default=True,
            variables=["dataset_name", "anomaly_count", "detector_name"]
        )
    }
}


@dataclass
class IntegrationMetrics:
    """Metrics for integration performance."""
    integration_id: str
    total_notifications: int = 0
    successful_notifications: int = 0
    failed_notifications: int = 0
    average_delivery_time_ms: float = 0.0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    uptime_percentage: float = 100.0
    rate_limit_hits: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_notifications == 0:
            return 0.0
        return (self.successful_notifications / self.total_notifications) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        return 100.0 - self.success_rate
