"""Intelligent alerting engine for enterprise anomaly detection monitoring.

This module provides advanced alerting capabilities with multi-channel notifications,
context-aware routing, and intelligent alert management for enterprise deployments.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Optional dependencies for notification channels
try:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels with escalation priority."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(Enum):
    """Categories of alerts for intelligent routing."""

    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    BUSINESS = "business"
    INFRASTRUCTURE = "infrastructure"
    DATA_QUALITY = "data_quality"


class NotificationChannel(Enum):
    """Available notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    DASHBOARD = "dashboard"


@dataclass
class AlertRule:
    """Advanced alert rule with intelligent routing."""

    id: str
    name: str
    description: str
    category: AlertCategory
    severity: AlertSeverity
    condition: str
    threshold_value: float
    comparison_operator: str  # ">", "<", ">=", "<=", "==", "!="
    evaluation_window_minutes: int = 5
    cooldown_minutes: int = 15
    escalation_delay_minutes: int = 30
    notification_channels: list[NotificationChannel] = field(default_factory=list)
    business_hours_only: bool = False
    environment_filters: list[str] = field(default_factory=list)
    tag_filters: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    # Advanced features
    require_acknowledgment: bool = False
    auto_resolve_minutes: int | None = None
    escalation_rules: list[dict[str, Any]] = field(default_factory=list)
    suppression_rules: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Alert:
    """Intelligent alert with context and routing information."""

    id: str
    rule_id: str
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    timestamp: datetime
    metric_name: str
    metric_value: float
    threshold_value: float

    # Context information
    source_service: str
    environment: str
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Alert lifecycle
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    resolved: bool = False
    resolved_by: str | None = None
    resolved_at: datetime | None = None

    # Escalation tracking
    escalation_level: int = 0
    escalated_at: datetime | None = None
    notification_attempts: list[dict[str, Any]] = field(default_factory=list)

    # Correlation and grouping
    correlation_id: str | None = None
    related_alerts: list[str] = field(default_factory=list)


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""

    @abstractmethod
    async def send_notification(
        self, alert: Alert, recipients: list[str], **kwargs
    ) -> bool:
        """Send notification through this provider."""
        pass

    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate provider configuration."""
        pass


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        use_tls: bool = True,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls

        self.logger = logging.getLogger(__name__)

    async def send_notification(
        self, alert: Alert, recipients: list[str], **kwargs
    ) -> bool:
        """Send email notification."""

        if not EMAIL_AVAILABLE:
            self.logger.error("Email functionality not available")
            return False

        try:
            subject = f"[{alert.severity.value.upper()}] {alert.title}"

            body = f"""
Alert: {alert.title}
Severity: {alert.severity.value.upper()}
Category: {alert.category.value}
Time: {alert.timestamp.isoformat()}

Description: {alert.description}

Metric: {alert.metric_name}
Current Value: {alert.metric_value}
Threshold: {alert.threshold_value}

Service: {alert.source_service}
Environment: {alert.environment}

Alert ID: {alert.id}
Rule ID: {alert.rule_id}

This is an automated message from Pynomaly Enterprise Monitoring.
"""

            msg = MIMEMultipart()
            msg["From"] = self.username
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)

            for recipient in recipients:
                msg["To"] = recipient
                server.send_message(msg)
                self.logger.info(f"Email sent to {recipient} for alert {alert.id}")

            server.quit()
            return True

        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False

    def validate_configuration(self) -> bool:
        """Validate email configuration."""
        return all([self.smtp_server, self.smtp_port, self.username, self.password])


class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)

    async def send_notification(
        self, alert: Alert, recipients: list[str], **kwargs
    ) -> bool:
        """Send Slack notification."""

        if not REQUESTS_AVAILABLE:
            self.logger.error("Requests library not available for Slack notifications")
            return False

        try:
            # Slack message color based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",  # Green
                AlertSeverity.WARNING: "#ffcc00",  # Yellow
                AlertSeverity.ERROR: "#ff9900",  # Orange
                AlertSeverity.CRITICAL: "#ff0000",  # Red
                AlertSeverity.EMERGENCY: "#990000",  # Dark Red
            }

            color = color_map.get(alert.severity, "#808080")

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{alert.severity.value.upper()}: {alert.title}",
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Metric",
                                "value": f"{alert.metric_name}: {alert.metric_value}",
                                "short": True,
                            },
                            {
                                "title": "Threshold",
                                "value": str(alert.threshold_value),
                                "short": True,
                            },
                            {
                                "title": "Service",
                                "value": alert.source_service,
                                "short": True,
                            },
                            {
                                "title": "Environment",
                                "value": alert.environment,
                                "short": True,
                            },
                        ],
                        "timestamp": int(alert.timestamp.timestamp()),
                        "footer": "Pynomaly Enterprise Monitoring",
                        "footer_icon": "https://example.com/pynomaly-icon.png",
                    }
                ]
            }

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            self.logger.info(f"Slack notification sent for alert {alert.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False

    def validate_configuration(self) -> bool:
        """Validate Slack configuration."""
        return bool(
            self.webhook_url and self.webhook_url.startswith("https://hooks.slack.com/")
        )


class WebhookNotificationProvider(NotificationProvider):
    """Generic webhook notification provider."""

    def __init__(self, webhook_url: str, headers: dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.logger = logging.getLogger(__name__)

    async def send_notification(
        self, alert: Alert, recipients: list[str], **kwargs
    ) -> bool:
        """Send webhook notification."""

        if not REQUESTS_AVAILABLE:
            self.logger.error(
                "Requests library not available for webhook notifications"
            )
            return False

        try:
            payload = {
                "alert_id": alert.id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "category": alert.category.value,
                "timestamp": alert.timestamp.isoformat(),
                "metric": {
                    "name": alert.metric_name,
                    "value": alert.metric_value,
                    "threshold": alert.threshold_value,
                },
                "source": {
                    "service": alert.source_service,
                    "environment": alert.environment,
                },
                "tags": alert.tags,
                "metadata": alert.metadata,
            }

            response = requests.post(
                self.webhook_url, json=payload, headers=self.headers, timeout=10
            )
            response.raise_for_status()

            self.logger.info(f"Webhook notification sent for alert {alert.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False

    def validate_configuration(self) -> bool:
        """Validate webhook configuration."""
        return bool(self.webhook_url)


class IntelligentAlertingEngine:
    """Advanced alerting engine with intelligent routing and correlation."""

    def __init__(
        self,
        enable_alert_correlation: bool = True,
        enable_intelligent_suppression: bool = True,
        max_alerts_per_hour: int = 100,
        correlation_window_minutes: int = 10,
    ):
        """Initialize intelligent alerting engine.

        Args:
            enable_alert_correlation: Enable alert correlation and grouping
            enable_intelligent_suppression: Enable intelligent alert suppression
            max_alerts_per_hour: Maximum alerts per hour to prevent spam
            correlation_window_minutes: Time window for alert correlation
        """
        self.enable_alert_correlation = enable_alert_correlation
        self.enable_intelligent_suppression = enable_intelligent_suppression
        self.max_alerts_per_hour = max_alerts_per_hour
        self.correlation_window_minutes = correlation_window_minutes

        # Alert management
        self.active_alerts: dict[str, Alert] = {}
        self.alert_rules: dict[str, AlertRule] = {}
        self.alert_history: deque = deque(maxlen=10000)

        # Notification providers
        self.notification_providers: dict[
            NotificationChannel, NotificationProvider
        ] = {}

        # Intelligent features
        self.alert_rate_limiter: deque = deque(maxlen=1000)
        self.suppressed_alerts: set[str] = set()
        self.correlation_groups: dict[str, list[str]] = {}

        # Background processing
        self.processing_thread: threading.Thread | None = None
        self.processing_active: bool = False

        self.logger = logging.getLogger(__name__)
        self.logger.info("Intelligent Alerting Engine initialized")

    def add_notification_provider(
        self, channel: NotificationChannel, provider: NotificationProvider
    ):
        """Add a notification provider for a channel."""

        if provider.validate_configuration():
            self.notification_providers[channel] = provider
            self.logger.info(f"Added notification provider for {channel.value}")
        else:
            self.logger.error(f"Invalid configuration for {channel.value} provider")

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule to the engine."""

        self.alert_rules[rule.id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""

        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")

    async def evaluate_metric(
        self,
        metric_name: str,
        metric_value: float,
        source_service: str,
        environment: str = "production",
        tags: dict[str, str] = None,
        metadata: dict[str, Any] = None,
    ):
        """Evaluate a metric against all applicable alert rules."""

        tags = tags or {}
        metadata = metadata or {}

        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            # Check environment filters
            if rule.environment_filters and environment not in rule.environment_filters:
                continue

            # Check tag filters
            if rule.tag_filters:
                if not all(
                    tags.get(key) == value for key, value in rule.tag_filters.items()
                ):
                    continue

            # Check business hours constraint
            if rule.business_hours_only and not self._is_business_hours():
                continue

            # Evaluate condition
            if self._evaluate_condition(
                metric_value, rule.comparison_operator, rule.threshold_value
            ):
                # Check cooldown period
                if self._is_in_cooldown(rule.id):
                    continue

                # Check rate limiting
                if self._is_rate_limited():
                    self.logger.warning("Alert rate limit exceeded, suppressing alert")
                    continue

                # Create alert
                alert = Alert(
                    id=f"alert_{rule.id}_{int(time.time() * 1000)}",
                    rule_id=rule.id,
                    title=f"{rule.name}: {metric_name}",
                    description=f"{rule.description}. Current value: {metric_value}, Threshold: {rule.threshold_value}",
                    severity=rule.severity,
                    category=rule.category,
                    timestamp=datetime.now(),
                    metric_name=metric_name,
                    metric_value=metric_value,
                    threshold_value=rule.threshold_value,
                    source_service=source_service,
                    environment=environment,
                    tags=tags,
                    metadata=metadata,
                )

                # Process alert with intelligent features
                await self._process_alert(alert, rule)

    def _evaluate_condition(
        self, value: float, operator: str, threshold: float
    ) -> bool:
        """Evaluate alert condition."""

        try:
            if operator == ">":
                return value > threshold
            elif operator == "<":
                return value < threshold
            elif operator == ">=":
                return value >= threshold
            elif operator == "<=":
                return value <= threshold
            elif operator == "==":
                return abs(value - threshold) < 0.001
            elif operator == "!=":
                return abs(value - threshold) >= 0.001
            else:
                self.logger.warning(f"Unknown operator: {operator}")
                return False
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False

    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""

        now = datetime.now()
        # Business hours: Monday-Friday, 9 AM - 5 PM
        if now.weekday() >= 5:  # Weekend
            return False

        hour = now.hour
        return 9 <= hour <= 17

    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period."""

        # Look for recent alerts from this rule
        cutoff = datetime.now() - timedelta(
            minutes=self.alert_rules[rule_id].cooldown_minutes
        )

        for alert in self.alert_history:
            if (
                alert.rule_id == rule_id
                and hasattr(alert, "timestamp")
                and alert.timestamp > cutoff
            ):
                return True

        return False

    def _is_rate_limited(self) -> bool:
        """Check if alert rate limit is exceeded."""

        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        # Count alerts in the last hour
        recent_alerts = [
            timestamp for timestamp in self.alert_rate_limiter if timestamp > cutoff
        ]

        return len(recent_alerts) >= self.max_alerts_per_hour

    async def _process_alert(self, alert: Alert, rule: AlertRule):
        """Process alert with intelligent features."""

        # Add to rate limiter
        self.alert_rate_limiter.append(datetime.now())

        # Check for intelligent suppression
        if self.enable_intelligent_suppression and self._should_suppress_alert(
            alert, rule
        ):
            self.suppressed_alerts.add(alert.id)
            self.logger.info(f"Alert suppressed: {alert.id}")
            return

        # Check for correlation
        if self.enable_alert_correlation:
            correlation_id = self._correlate_alert(alert)
            if correlation_id:
                alert.correlation_id = correlation_id

        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)

        # Send notifications
        await self._send_notifications(alert, rule)

        # Schedule escalation if required
        if rule.escalation_rules:
            self._schedule_escalation(alert, rule)

        # Schedule auto-resolution if configured
        if rule.auto_resolve_minutes:
            self._schedule_auto_resolve(alert, rule)

        self.logger.info(f"Alert processed: {alert.id} [{alert.severity.value}]")

    def _should_suppress_alert(self, alert: Alert, rule: AlertRule) -> bool:
        """Determine if alert should be suppressed using intelligent logic."""

        # Check suppression rules
        for suppression_rule in rule.suppression_rules:
            condition = suppression_rule.get("condition", "")

            if condition == "similar_alert_exists":
                # Check for similar active alerts
                for active_alert in self.active_alerts.values():
                    if (
                        active_alert.category == alert.category
                        and active_alert.source_service == alert.source_service
                        and not active_alert.resolved
                    ):
                        return True

            elif condition == "maintenance_window":
                # Check if in maintenance window (would be configured separately)
                if self._is_maintenance_window(alert.source_service):
                    return True

        return False

    def _is_maintenance_window(self, service: str) -> bool:
        """Check if service is in maintenance window."""
        # Placeholder for maintenance window logic
        return False

    def _correlate_alert(self, alert: Alert) -> str | None:
        """Correlate alert with existing alerts."""

        correlation_window = datetime.now() - timedelta(
            minutes=self.correlation_window_minutes
        )

        # Look for related alerts
        for existing_alert in self.active_alerts.values():
            if (
                existing_alert.timestamp > correlation_window
                and existing_alert.category == alert.category
                and existing_alert.source_service == alert.source_service
            ):
                # Create or join correlation group
                if existing_alert.correlation_id:
                    correlation_id = existing_alert.correlation_id
                else:
                    correlation_id = f"correlation_{int(time.time() * 1000)}"
                    existing_alert.correlation_id = correlation_id
                    self.correlation_groups[correlation_id] = [existing_alert.id]

                # Add new alert to correlation group
                self.correlation_groups[correlation_id].append(alert.id)
                alert.related_alerts = self.correlation_groups[correlation_id].copy()

                return correlation_id

        return None

    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications through configured channels."""

        for channel in rule.notification_channels:
            if channel not in self.notification_providers:
                self.logger.warning(f"No provider configured for {channel.value}")
                continue

            provider = self.notification_providers[channel]

            # Get recipients (would be configured separately)
            recipients = self._get_recipients_for_channel(channel, alert.severity)

            if not recipients:
                continue

            try:
                success = await provider.send_notification(alert, recipients)

                # Record notification attempt
                attempt = {
                    "channel": channel.value,
                    "timestamp": datetime.now().isoformat(),
                    "success": success,
                    "recipients": recipients,
                }
                alert.notification_attempts.append(attempt)

                if success:
                    self.logger.info(
                        f"Notification sent via {channel.value} for alert {alert.id}"
                    )
                else:
                    self.logger.error(
                        f"Failed to send notification via {channel.value} for alert {alert.id}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Exception sending notification via {channel.value}: {e}"
                )

    def _get_recipients_for_channel(
        self, channel: NotificationChannel, severity: AlertSeverity
    ) -> list[str]:
        """Get recipients for a notification channel based on severity."""

        # This would be configured based on your organization's structure
        # For now, return example recipients

        if channel == NotificationChannel.EMAIL:
            if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                return ["alerts@company.com", "oncall@company.com"]
            else:
                return ["monitoring@company.com"]

        elif channel == NotificationChannel.SLACK:
            return ["#alerts", "#monitoring"]

        return []

    def _schedule_escalation(self, alert: Alert, rule: AlertRule):
        """Schedule alert escalation."""
        # In a real implementation, this would use a scheduler
        # For now, we'll just log the intention
        self.logger.info(
            f"Escalation scheduled for alert {alert.id} in {rule.escalation_delay_minutes} minutes"
        )

    def _schedule_auto_resolve(self, alert: Alert, rule: AlertRule):
        """Schedule automatic alert resolution."""
        # In a real implementation, this would use a scheduler
        self.logger.info(
            f"Auto-resolve scheduled for alert {alert.id} in {rule.auto_resolve_minutes} minutes"
        )

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""

        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()

            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True

        return False

    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert."""

        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_by = resolved_by
            alert.resolved_at = datetime.now()

            # Remove from active alerts
            del self.active_alerts[alert_id]

            self.logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
            return True

        return False

    def get_active_alerts(self, category: AlertCategory | None = None) -> list[Alert]:
        """Get active alerts, optionally filtered by category."""

        alerts = list(self.active_alerts.values())

        if category:
            alerts = [alert for alert in alerts if alert.category == category]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_statistics(self) -> dict[str, Any]:
        """Get alerting engine statistics."""

        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)

        recent_alerts = [
            alert
            for alert in self.alert_history
            if hasattr(alert, "timestamp") and alert.timestamp > last_hour
        ]

        daily_alerts = [
            alert
            for alert in self.alert_history
            if hasattr(alert, "timestamp") and alert.timestamp > last_day
        ]

        return {
            "active_alerts": len(self.active_alerts),
            "alerts_last_hour": len(recent_alerts),
            "alerts_last_day": len(daily_alerts),
            "suppressed_alerts": len(self.suppressed_alerts),
            "correlation_groups": len(self.correlation_groups),
            "configured_rules": len(self.alert_rules),
            "notification_channels": len(self.notification_providers),
            "rate_limit_status": {
                "current_rate": len(
                    [t for t in self.alert_rate_limiter if t > last_hour]
                ),
                "limit": self.max_alerts_per_hour,
            },
        }


def create_default_alert_rules() -> list[AlertRule]:
    """Create default alert rules for enterprise monitoring."""

    return [
        AlertRule(
            id="high_error_rate",
            name="High Error Rate",
            description="Error rate exceeds acceptable threshold",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.ERROR,
            condition="error_rate > 5.0",
            threshold_value=5.0,
            comparison_operator=">",
            notification_channels=[
                NotificationChannel.EMAIL,
                NotificationChannel.SLACK,
            ],
            escalation_delay_minutes=30,
        ),
        AlertRule(
            id="critical_performance_degradation",
            name="Critical Performance Degradation",
            description="Detection time exceeds critical threshold",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.CRITICAL,
            condition="avg_detection_time > 60.0",
            threshold_value=60.0,
            comparison_operator=">",
            notification_channels=[
                NotificationChannel.EMAIL,
                NotificationChannel.PAGERDUTY,
            ],
            require_acknowledgment=True,
        ),
        AlertRule(
            id="security_anomaly_detected",
            name="Security Anomaly Detected",
            description="Unusual authentication or access patterns detected",
            category=AlertCategory.SECURITY,
            severity=AlertSeverity.CRITICAL,
            condition="security_score < 70.0",
            threshold_value=70.0,
            comparison_operator="<",
            notification_channels=[
                NotificationChannel.EMAIL,
                NotificationChannel.TEAMS,
            ],
            business_hours_only=False,
        ),
        AlertRule(
            id="compliance_violation",
            name="Compliance Violation",
            description="Regulatory compliance threshold breached",
            category=AlertCategory.COMPLIANCE,
            severity=AlertSeverity.ERROR,
            condition="compliance_score < 95.0",
            threshold_value=95.0,
            comparison_operator="<",
            notification_channels=[NotificationChannel.EMAIL],
            require_acknowledgment=True,
        ),
        AlertRule(
            id="data_quality_degradation",
            name="Data Quality Degradation",
            description="Data quality metrics below acceptable levels",
            category=AlertCategory.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            condition="data_quality_score < 85.0",
            threshold_value=85.0,
            comparison_operator="<",
            notification_channels=[
                NotificationChannel.EMAIL,
                NotificationChannel.SLACK,
            ],
        ),
    ]
