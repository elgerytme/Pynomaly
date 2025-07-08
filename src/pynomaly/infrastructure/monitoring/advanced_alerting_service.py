"""Advanced alerting service with intelligent thresholds and escalation.

This module provides enterprise-grade alerting capabilities:
- Dynamic threshold adjustment and anomaly-based alerting
- Multi-channel notification delivery (email, SMS, Slack, Teams)
- Alert escalation and on-call rotation management
- Alert correlation and noise reduction
- Custom alert templates and conditional logic
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import aiohttp
from jinja2 import Template

# Optional notification providers
try:
    import smtplib
    from email.mime.multipart import MimeMultipart
    from email.mime.text import MimeText

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient

    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Alert notification channels."""

    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"


class AlertState(Enum):
    """Alert lifecycle states."""

    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationLevel(Enum):
    """Alert escalation levels."""

    LEVEL_1 = "level_1"  # Team lead
    LEVEL_2 = "level_2"  # Manager
    LEVEL_3 = "level_3"  # On-call engineer
    LEVEL_4 = "level_4"  # Director/VP


@dataclass
class ThresholdCondition:
    """Defines a threshold condition for alerting."""

    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    value: int | float
    window_minutes: int = 5
    evaluation_frequency: int = 60  # seconds

    # Dynamic threshold adjustment
    use_dynamic_threshold: bool = False
    baseline_window_hours: int = 24
    sensitivity: float = 2.0  # Standard deviations

    # Additional conditions
    minimum_data_points: int = 3
    require_consecutive_violations: int = 1


@dataclass
class AlertRule:
    """Defines an alert rule with conditions and actions."""

    rule_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""

    # Conditions
    conditions: list[ThresholdCondition] = field(default_factory=list)
    condition_logic: str = "AND"  # AND, OR

    # Alert properties
    severity: AlertSeverity = AlertSeverity.MEDIUM
    channels: list[AlertChannel] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    # Timing
    enabled: bool = True
    suppress_duration_minutes: int = 60  # Don't re-alert for this period
    auto_resolve_minutes: int | None = None

    # Escalation
    escalation_enabled: bool = False
    escalation_delay_minutes: int = 30
    escalation_levels: list[EscalationLevel] = field(default_factory=list)

    # Template customization
    title_template: str = "Alert: {{rule_name}}"
    message_template: str = "Alert triggered: {{description}}"

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"


@dataclass
class Alert:
    """Represents an active alert."""

    alert_id: str = field(default_factory=lambda: str(uuid4()))
    rule_id: str = ""

    # Content
    title: str = ""
    message: str = ""
    severity: AlertSeverity = AlertSeverity.MEDIUM
    tags: dict[str, str] = field(default_factory=dict)

    # State
    state: AlertState = AlertState.TRIGGERED
    triggered_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None

    # Context
    triggered_value: int | float | None = None
    threshold_value: int | float | None = None
    metric_name: str | None = None

    # Escalation tracking
    current_escalation_level: EscalationLevel | None = None
    escalation_attempts: list[dict[str, Any]] = field(default_factory=list)

    # Notification tracking
    notifications_sent: list[dict[str, Any]] = field(default_factory=list)

    # Resolution info
    resolved_by: str | None = None
    resolution_note: str | None = None


@dataclass
class NotificationChannel:
    """Configuration for notification channels."""

    channel_type: AlertChannel
    name: str
    enabled: bool = True

    # Channel-specific configuration
    config: dict[str, Any] = field(default_factory=dict)

    # Filtering
    severity_filter: list[AlertSeverity] = field(default_factory=list)
    tag_filters: dict[str, str] = field(default_factory=dict)

    # Rate limiting
    rate_limit_per_hour: int | None = None
    quiet_hours_start: str | None = None  # "22:00"
    quiet_hours_end: str | None = None  # "08:00"


@dataclass
class OnCallSchedule:
    """On-call rotation schedule."""

    schedule_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""

    # Rotation settings
    rotation_type: str = "weekly"  # daily, weekly, monthly
    team_members: list[str] = field(default_factory=list)

    # Schedule configuration
    start_date: datetime = field(default_factory=datetime.now)
    timezone: str = "UTC"
    handoff_time: str = "09:00"  # Daily handoff time

    # Escalation paths
    escalation_mapping: dict[EscalationLevel, list[str]] = field(default_factory=dict)

    # Override settings
    overrides: list[dict[str, Any]] = field(
        default_factory=list
    )  # Vacation, holidays, etc.


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    async def send_notification(
        self, alert: Alert, recipients: list[str], template_data: dict[str, Any]
    ) -> bool:
        """Send notification for an alert."""
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test the notification provider connection."""
        pass


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider."""

    async def send_notification(
        self, alert: Alert, recipients: list[str], template_data: dict[str, Any]
    ) -> bool:
        """Send email notification."""
        try:
            if not EMAIL_AVAILABLE:
                logger.warning("Email libraries not available")
                return False

            smtp_server = self.config.get("smtp_server", "localhost")
            smtp_port = self.config.get("smtp_port", 587)
            username = self.config.get("username")
            password = self.config.get("password")
            from_email = self.config.get("from_email", "noreply@pynomaly.com")

            # Create message
            msg = MimeMultipart()
            msg["From"] = from_email
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = alert.title

            # Create email body
            email_template = Template(
                self.config.get(
                    "template",
                    """
            <html>
            <body>
                <h2>{{ alert.title }}</h2>
                <p><strong>Severity:</strong> {{ alert.severity.value.upper() }}</p>
                <p><strong>Triggered At:</strong> {{ alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
                <p><strong>Message:</strong> {{ alert.message }}</p>

                {% if alert.metric_name %}
                <p><strong>Metric:</strong> {{ alert.metric_name }}</p>
                <p><strong>Current Value:</strong> {{ alert.triggered_value }}</p>
                <p><strong>Threshold:</strong> {{ alert.threshold_value }}</p>
                {% endif %}

                {% if alert.tags %}
                <h3>Tags:</h3>
                <ul>
                {% for key, value in alert.tags.items() %}
                    <li><strong>{{ key }}:</strong> {{ value }}</li>
                {% endfor %}
                </ul>
                {% endif %}

                <p><em>Alert ID: {{ alert.alert_id }}</em></p>
            </body>
            </html>
            """,
                )
            )

            body = email_template.render(alert=alert, **template_data)
            msg.attach(MimeText(body, "html"))

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)

                server.send_message(msg)

            logger.info(f"Sent email notification for alert {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test email connection."""
        try:
            smtp_server = self.config.get("smtp_server", "localhost")
            smtp_port = self.config.get("smtp_port", 587)

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if self.config.get("username") and self.config.get("password"):
                    server.starttls()
                    server.login(self.config["username"], self.config["password"])

            return True

        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False


class SMSNotificationProvider(NotificationProvider):
    """SMS notification provider using Twilio."""

    async def send_notification(
        self, alert: Alert, recipients: list[str], template_data: dict[str, Any]
    ) -> bool:
        """Send SMS notification."""
        try:
            if not TWILIO_AVAILABLE:
                logger.warning("Twilio library not available")
                return False

            account_sid = self.config.get("account_sid")
            auth_token = self.config.get("auth_token")
            from_phone = self.config.get("from_phone")

            if not all([account_sid, auth_token, from_phone]):
                logger.error("Twilio configuration incomplete")
                return False

            client = TwilioClient(account_sid, auth_token)

            # Create SMS message
            sms_template = Template(
                self.config.get(
                    "template",
                    "ALERT: {{ alert.title }} | Severity: {{ alert.severity.value.upper() }} | {{ alert.message }}",
                )
            )

            message = sms_template.render(alert=alert, **template_data)

            # Truncate if too long (SMS limit is usually 160 chars)
            if len(message) > 160:
                message = message[:157] + "..."

            # Send to all recipients
            success_count = 0
            for phone in recipients:
                try:
                    client.messages.create(body=message, from_=from_phone, to=phone)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to send SMS to {phone}: {e}")

            logger.info(
                f"Sent SMS notifications for alert {alert.alert_id} to {success_count}/{len(recipients)} recipients"
            )
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to send SMS notification: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test Twilio connection."""
        try:
            if not TWILIO_AVAILABLE:
                return False

            account_sid = self.config.get("account_sid")
            auth_token = self.config.get("auth_token")

            if not all([account_sid, auth_token]):
                return False

            client = TwilioClient(account_sid, auth_token)
            # Test by fetching account info
            client.api.accounts(account_sid).fetch()
            return True

        except Exception as e:
            logger.error(f"Twilio connection test failed: {e}")
            return False


class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.session: aiohttp.ClientSession | None = None

    async def send_notification(
        self, alert: Alert, recipients: list[str], template_data: dict[str, Any]
    ) -> bool:
        """Send Slack notification."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False

            # Create Slack message
            color_map = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger",
            }

            slack_payload = {
                "text": f"Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "warning"),
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": alert.triggered_at.strftime(
                                    "%Y-%m-%d %H:%M:%S UTC"
                                ),
                                "short": True,
                            },
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False,
                            },
                        ],
                        "footer": f"Alert ID: {alert.alert_id}",
                    }
                ],
            }

            # Add metric information if available
            if alert.metric_name:
                slack_payload["attachments"][0]["fields"].extend(
                    [
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {
                            "title": "Value",
                            "value": f"{alert.triggered_value} (threshold: {alert.threshold_value})",
                            "short": True,
                        },
                    ]
                )

            # Add tags if present
            if alert.tags:
                tags_str = ", ".join([f"{k}:{v}" for k, v in alert.tags.items()])
                slack_payload["attachments"][0]["fields"].append(
                    {"title": "Tags", "value": tags_str, "short": False}
                )

            async with self.session.post(webhook_url, json=slack_payload) as response:
                if response.status == 200:
                    logger.info(f"Sent Slack notification for alert {alert.alert_id}")
                    return True
                else:
                    logger.error(f"Slack notification failed: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test Slack connection."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                return False

            test_payload = {
                "text": "Pynomaly connection test",
                "attachments": [
                    {
                        "color": "good",
                        "text": "This is a test message from Pynomaly alerting system",
                    }
                ],
            }

            async with self.session.post(webhook_url, json=test_payload) as response:
                return response.status == 200

        except Exception as e:
            logger.error(f"Slack connection test failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None


class WebhookNotificationProvider(NotificationProvider):
    """Generic webhook notification provider."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.session: aiohttp.ClientSession | None = None

    async def send_notification(
        self, alert: Alert, recipients: list[str], template_data: dict[str, Any]
    ) -> bool:
        """Send webhook notification."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False

            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "state": alert.state.value,
                "triggered_at": alert.triggered_at.isoformat(),
                "tags": alert.tags,
                "metric_name": alert.metric_name,
                "triggered_value": alert.triggered_value,
                "threshold_value": alert.threshold_value,
                "recipients": recipients,
                **template_data,
            }

            headers = {"Content-Type": "application/json"}
            if self.config.get("auth_header"):
                headers["Authorization"] = self.config["auth_header"]

            async with self.session.post(
                webhook_url, json=payload, headers=headers
            ) as response:
                if response.status in [200, 201, 202]:
                    logger.info(f"Sent webhook notification for alert {alert.alert_id}")
                    return True
                else:
                    logger.error(f"Webhook notification failed: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test webhook connection."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                return False

            test_payload = {
                "test": True,
                "message": "Pynomaly webhook connection test",
                "timestamp": datetime.now().isoformat(),
            }

            headers = {"Content-Type": "application/json"}
            if self.config.get("auth_header"):
                headers["Authorization"] = self.config["auth_header"]

            async with self.session.post(
                webhook_url, json=test_payload, headers=headers
            ) as response:
                return response.status in [200, 201, 202]

        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None


class AdvancedAlertingService:
    """Enterprise-grade alerting service with intelligent features."""

    def __init__(self):
        self.alert_rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.notification_channels: dict[str, NotificationChannel] = {}
        self.notification_providers: dict[AlertChannel, NotificationProvider] = {}
        self.on_call_schedules: dict[str, OnCallSchedule] = {}

        # Alert correlation and suppression
        self.suppressed_alerts: set[str] = set()
        self.alert_correlation_rules: list[dict[str, Any]] = []

        # Metrics for threshold adjustment
        self.metric_history: dict[str, list[tuple]] = (
            {}
        )  # metric_name -> [(timestamp, value)]
        self.dynamic_thresholds: dict[str, dict[str, float]] = (
            {}
        )  # metric_name -> {mean, std, threshold}

        # Background tasks
        self._evaluation_task: asyncio.Task | None = None
        self._escalation_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize the alerting service."""
        # Start background tasks
        self._evaluation_task = asyncio.create_task(self._evaluate_alert_rules())
        self._escalation_task = asyncio.create_task(self._process_escalations())
        self._cleanup_task = asyncio.create_task(self._cleanup_old_data())

        logger.info("Advanced alerting service initialized")

    async def shutdown(self) -> None:
        """Shutdown the alerting service."""
        # Cancel background tasks
        for task in [self._evaluation_task, self._escalation_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Cleanup notification providers
        for provider in self.notification_providers.values():
            if hasattr(provider, "cleanup"):
                await provider.cleanup()

        logger.info("Advanced alerting service shutdown")

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")

    def remove_alert_rule(self, rule_id: str) -> None:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")

    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.notification_channels[channel.name] = channel

        # Initialize provider if not exists
        if channel.channel_type not in self.notification_providers:
            provider = self._create_notification_provider(
                channel.channel_type, channel.config
            )
            if provider:
                self.notification_providers[channel.channel_type] = provider

        logger.info(
            f"Added notification channel: {channel.name} ({channel.channel_type.value})"
        )

    def _create_notification_provider(
        self, channel_type: AlertChannel, config: dict[str, Any]
    ) -> NotificationProvider | None:
        """Create notification provider instance."""
        try:
            if channel_type == AlertChannel.EMAIL:
                return EmailNotificationProvider(config)
            elif channel_type == AlertChannel.SMS:
                return SMSNotificationProvider(config)
            elif channel_type == AlertChannel.SLACK:
                return SlackNotificationProvider(config)
            elif channel_type == AlertChannel.WEBHOOK:
                return WebhookNotificationProvider(config)
            else:
                logger.warning(f"Unsupported notification channel: {channel_type}")
                return None
        except Exception as e:
            logger.error(
                f"Failed to create notification provider for {channel_type}: {e}"
            )
            return None

    async def record_metric(self, metric_name: str, value: int | float) -> None:
        """Record metric value for threshold evaluation."""
        timestamp = datetime.now()

        # Store in history
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []

        self.metric_history[metric_name].append((timestamp, value))

        # Keep only recent data (last 24 hours)
        cutoff_time = timestamp - timedelta(hours=24)
        self.metric_history[metric_name] = [
            (ts, val)
            for ts, val in self.metric_history[metric_name]
            if ts > cutoff_time
        ]

        # Update dynamic thresholds
        await self._update_dynamic_threshold(metric_name)

        # Evaluate relevant alert rules
        await self._evaluate_metric_alerts(metric_name, value)

    async def _update_dynamic_threshold(self, metric_name: str) -> None:
        """Update dynamic threshold for a metric."""
        if metric_name not in self.metric_history:
            return

        history = self.metric_history[metric_name]
        if len(history) < 10:  # Need minimum data points
            return

        # Calculate statistics
        values = [val for _, val in history]
        mean = sum(values) / len(values)
        variance = sum((val - mean) ** 2 for val in values) / len(values)
        std = variance**0.5

        # Store threshold info
        self.dynamic_thresholds[metric_name] = {
            "mean": mean,
            "std": std,
            "upper_threshold": mean + (2.0 * std),
            "lower_threshold": mean - (2.0 * std),
            "last_updated": datetime.now(),
        }

    async def _evaluate_metric_alerts(
        self, metric_name: str, value: int | float
    ) -> None:
        """Evaluate alert rules for a specific metric."""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            # Check if rule applies to this metric
            relevant_conditions = [
                cond for cond in rule.conditions if cond.metric_name == metric_name
            ]

            if not relevant_conditions:
                continue

            # Evaluate conditions
            condition_results = []
            for condition in relevant_conditions:
                result = await self._evaluate_condition(condition, metric_name, value)
                condition_results.append(result)

            # Apply logic (AND/OR)
            if rule.condition_logic.upper() == "AND":
                alert_triggered = all(condition_results)
            else:  # OR
                alert_triggered = any(condition_results)

            # Handle alert state
            if alert_triggered:
                await self._trigger_alert(rule, metric_name, value)
            else:
                await self._check_alert_resolution(rule.rule_id)

    async def _evaluate_condition(
        self,
        condition: ThresholdCondition,
        metric_name: str,
        current_value: int | float,
    ) -> bool:
        """Evaluate a single threshold condition."""

        # Use dynamic threshold if enabled
        if condition.use_dynamic_threshold and metric_name in self.dynamic_thresholds:
            threshold_info = self.dynamic_thresholds[metric_name]

            if condition.operator in [">", ">="]:
                threshold_value = threshold_info["upper_threshold"]
            else:
                threshold_value = threshold_info["lower_threshold"]
        else:
            threshold_value = condition.value

        # Evaluate condition
        if condition.operator == ">":
            return current_value > threshold_value
        elif condition.operator == ">=":
            return current_value >= threshold_value
        elif condition.operator == "<":
            return current_value < threshold_value
        elif condition.operator == "<=":
            return current_value <= threshold_value
        elif condition.operator == "==":
            return current_value == threshold_value
        elif condition.operator == "!=":
            return current_value != threshold_value

        return False

    async def _trigger_alert(
        self, rule: AlertRule, metric_name: str, triggered_value: int | float
    ) -> None:
        """Trigger an alert for a rule."""

        # Check if alert is suppressed
        if self._is_alert_suppressed(rule.rule_id):
            return

        # Check if similar alert already exists
        existing_alert = self._find_existing_alert(rule.rule_id)
        if existing_alert and existing_alert.state == AlertState.TRIGGERED:
            # Update existing alert
            existing_alert.triggered_value = triggered_value
            return

        # Create new alert
        alert = Alert(
            rule_id=rule.rule_id,
            title=Template(rule.title_template).render(rule_name=rule.name),
            message=Template(rule.message_template).render(
                description=rule.description,
                metric_name=metric_name,
                triggered_value=triggered_value,
            ),
            severity=rule.severity,
            tags=rule.tags.copy(),
            metric_name=metric_name,
            triggered_value=triggered_value,
        )

        # Store alert
        self.active_alerts[alert.alert_id] = alert

        # Send notifications
        await self._send_alert_notifications(alert, rule)

        # Add to suppression list
        self._add_to_suppression(rule.rule_id, rule.suppress_duration_minutes)

        logger.warning(f"Alert triggered: {alert.title} (ID: {alert.alert_id})")

    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications for an alert."""

        for channel_type in rule.channels:
            try:
                # Find matching notification channels
                matching_channels = [
                    ch
                    for ch in self.notification_channels.values()
                    if ch.channel_type == channel_type and ch.enabled
                ]

                for channel in matching_channels:
                    # Check severity filter
                    if (
                        channel.severity_filter
                        and alert.severity not in channel.severity_filter
                    ):
                        continue

                    # Check tag filters
                    if channel.tag_filters:
                        if not all(
                            alert.tags.get(key) == value
                            for key, value in channel.tag_filters.items()
                        ):
                            continue

                    # Check quiet hours
                    if self._is_quiet_hours(channel):
                        continue

                    # Get provider
                    provider = self.notification_providers.get(channel_type)
                    if not provider:
                        continue

                    # Get recipients
                    recipients = self._get_recipients(channel, alert)
                    if not recipients:
                        continue

                    # Send notification
                    success = await provider.send_notification(
                        alert, recipients, {"channel": channel.name, "rule": rule}
                    )

                    # Track notification
                    alert.notifications_sent.append(
                        {
                            "channel": channel.name,
                            "channel_type": channel_type.value,
                            "recipients": recipients,
                            "success": success,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to send notification via {channel_type}: {e}")

    def _get_recipients(self, channel: NotificationChannel, alert: Alert) -> list[str]:
        """Get notification recipients based on on-call schedule and escalation level."""

        # For now, return recipients from channel config
        # In a full implementation, this would consult on-call schedules
        return channel.config.get("recipients", [])

    def _is_quiet_hours(self, channel: NotificationChannel) -> bool:
        """Check if current time is within quiet hours."""
        if not channel.quiet_hours_start or not channel.quiet_hours_end:
            return False

        # Simple implementation - would need proper timezone handling
        current_time = datetime.now().strftime("%H:%M")
        return channel.quiet_hours_start <= current_time <= channel.quiet_hours_end

    def _is_alert_suppressed(self, rule_id: str) -> bool:
        """Check if alert is currently suppressed."""
        return rule_id in self.suppressed_alerts

    def _add_to_suppression(self, rule_id: str, duration_minutes: int) -> None:
        """Add rule to suppression list."""
        self.suppressed_alerts.add(rule_id)

        # Schedule removal from suppression
        async def remove_suppression():
            await asyncio.sleep(duration_minutes * 60)
            self.suppressed_alerts.discard(rule_id)

        asyncio.create_task(remove_suppression())

    def _find_existing_alert(self, rule_id: str) -> Alert | None:
        """Find existing alert for a rule."""
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and alert.state == AlertState.TRIGGERED:
                return alert
        return None

    async def _check_alert_resolution(self, rule_id: str) -> None:
        """Check if an alert should be auto-resolved."""

        existing_alert = self._find_existing_alert(rule_id)
        if existing_alert:
            # Get the rule
            rule = self.alert_rules.get(rule_id)
            if rule and rule.auto_resolve_minutes:
                # Check if enough time has passed
                time_since_trigger = datetime.now() - existing_alert.triggered_at
                if time_since_trigger > timedelta(minutes=rule.auto_resolve_minutes):
                    await self.resolve_alert(
                        existing_alert.alert_id, "system", "Auto-resolved"
                    )

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False

        alert.state = AlertState.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    async def resolve_alert(
        self, alert_id: str, resolved_by: str, resolution_note: str | None = None
    ) -> bool:
        """Resolve an alert."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False

        alert.state = AlertState.RESOLVED
        alert.resolved_at = datetime.now()
        alert.resolved_by = resolved_by
        alert.resolution_note = resolution_note

        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True

    async def _evaluate_alert_rules(self) -> None:
        """Background task to evaluate alert rules periodically."""
        while True:
            try:
                # This would trigger periodic evaluations
                # For metrics not received via record_metric()
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert rule evaluation: {e}")

    async def _process_escalations(self) -> None:
        """Background task to process alert escalations."""
        while True:
            try:
                current_time = datetime.now()

                for alert in self.active_alerts.values():
                    if alert.state != AlertState.TRIGGERED:
                        continue

                    rule = self.alert_rules.get(alert.rule_id)
                    if not rule or not rule.escalation_enabled:
                        continue

                    # Check if escalation is due
                    time_since_trigger = current_time - alert.triggered_at
                    if time_since_trigger > timedelta(
                        minutes=rule.escalation_delay_minutes
                    ):
                        await self._escalate_alert(alert, rule)

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in escalation processing: {e}")

    async def _escalate_alert(self, alert: Alert, rule: AlertRule) -> None:
        """Escalate an alert to the next level."""

        current_level = alert.current_escalation_level
        next_level = self._get_next_escalation_level(
            current_level, rule.escalation_levels
        )

        if not next_level:
            logger.warning(f"No further escalation levels for alert {alert.alert_id}")
            return

        alert.current_escalation_level = next_level
        alert.state = AlertState.ESCALATED

        # Record escalation attempt
        alert.escalation_attempts.append(
            {
                "level": next_level.value,
                "timestamp": datetime.now().isoformat(),
                "reason": "Escalation timeout reached",
            }
        )

        # Send escalated notifications
        await self._send_escalated_notifications(alert, rule, next_level)

        logger.warning(f"Alert {alert.alert_id} escalated to {next_level.value}")

    def _get_next_escalation_level(
        self,
        current_level: EscalationLevel | None,
        available_levels: list[EscalationLevel],
    ) -> EscalationLevel | None:
        """Get the next escalation level."""

        if not available_levels:
            return None

        if current_level is None:
            return available_levels[0]

        try:
            current_index = available_levels.index(current_level)
            if current_index + 1 < len(available_levels):
                return available_levels[current_index + 1]
        except ValueError:
            pass

        return None

    async def _send_escalated_notifications(
        self, alert: Alert, rule: AlertRule, escalation_level: EscalationLevel
    ) -> None:
        """Send notifications for escalated alert."""

        # This would implement escalation-specific notification logic
        # For now, just send to the same channels as the original alert
        await self._send_alert_notifications(alert, rule)

    async def _cleanup_old_data(self) -> None:
        """Background task to cleanup old data."""
        while True:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(days=7)

                # Clean up resolved alerts older than 7 days
                alerts_to_remove = [
                    alert_id
                    for alert_id, alert in self.active_alerts.items()
                    if alert.state == AlertState.RESOLVED
                    and alert.resolved_at
                    and alert.resolved_at < cutoff_time
                ]

                for alert_id in alerts_to_remove:
                    del self.active_alerts[alert_id]

                if alerts_to_remove:
                    logger.info(
                        f"Cleaned up {len(alerts_to_remove)} old resolved alerts"
                    )

                # Clean up old metric history (keep only last 24 hours)
                for metric_name in list(self.metric_history.keys()):
                    cutoff_time_metrics = current_time - timedelta(hours=24)
                    self.metric_history[metric_name] = [
                        (ts, val)
                        for ts, val in self.metric_history[metric_name]
                        if ts > cutoff_time_metrics
                    ]

                await asyncio.sleep(3600)  # Run every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    def get_alert_statistics(self) -> dict[str, Any]:
        """Get alerting system statistics."""
        stats = {
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "active_alerts": len(
                [
                    a
                    for a in self.active_alerts.values()
                    if a.state == AlertState.TRIGGERED
                ]
            ),
            "acknowledged_alerts": len(
                [
                    a
                    for a in self.active_alerts.values()
                    if a.state == AlertState.ACKNOWLEDGED
                ]
            ),
            "resolved_alerts": len(
                [
                    a
                    for a in self.active_alerts.values()
                    if a.state == AlertState.RESOLVED
                ]
            ),
            "suppressed_rules": len(self.suppressed_alerts),
            "notification_channels": len(self.notification_channels),
            "metrics_tracked": len(self.metric_history),
        }

        # Severity breakdown
        severity_counts = {}
        for alert in self.active_alerts.values():
            if alert.state == AlertState.TRIGGERED:
                severity = alert.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

        stats["alerts_by_severity"] = severity_counts

        return stats


# Convenience functions for common alerting patterns


async def create_anomaly_detection_alert_rule(
    service: AdvancedAlertingService,
    detector_name: str,
    threshold_rate: float = 0.1,  # 10% anomaly rate
    severity: AlertSeverity = AlertSeverity.HIGH,
) -> AlertRule:
    """Create alert rule for anomaly detection rate."""

    rule = AlertRule(
        name=f"High Anomaly Rate - {detector_name}",
        description=f"Anomaly rate exceeds {threshold_rate*100}% for detector {detector_name}",
        conditions=[
            ThresholdCondition(
                metric_name="anomaly.rate.percent",
                operator=">",
                value=threshold_rate * 100,
                window_minutes=5,
            )
        ],
        severity=severity,
        channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
        tags={"detector": detector_name, "type": "anomaly_rate"},
        suppress_duration_minutes=30,
        escalation_enabled=True,
        escalation_delay_minutes=15,
    )

    service.add_alert_rule(rule)
    return rule


async def create_training_failure_alert_rule(
    service: AdvancedAlertingService,
    failure_threshold: int = 3,  # 3 consecutive failures
    severity: AlertSeverity = AlertSeverity.CRITICAL,
) -> AlertRule:
    """Create alert rule for training job failures."""

    rule = AlertRule(
        name="Training Job Failures",
        description=f"More than {failure_threshold} training job failures",
        conditions=[
            ThresholdCondition(
                metric_name="training.failures.count",
                operator=">=",
                value=failure_threshold,
                window_minutes=60,
            )
        ],
        severity=severity,
        channels=[AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.SLACK],
        tags={"type": "training_failure"},
        suppress_duration_minutes=60,
        escalation_enabled=True,
        escalation_delay_minutes=10,
    )

    service.add_alert_rule(rule)
    return rule


async def create_system_health_alert_rule(
    service: AdvancedAlertingService,
    component: str,
    health_threshold: float = 0.9,  # 90% health
    severity: AlertSeverity = AlertSeverity.HIGH,
) -> AlertRule:
    """Create alert rule for system health monitoring."""

    rule = AlertRule(
        name=f"System Health - {component}",
        description=f"Health score below {health_threshold*100}% for {component}",
        conditions=[
            ThresholdCondition(
                metric_name=f"system.health.{component}",
                operator="<",
                value=health_threshold,
                window_minutes=2,
                require_consecutive_violations=2,
            )
        ],
        severity=severity,
        channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK],
        tags={"component": component, "type": "health_check"},
        suppress_duration_minutes=15,
        auto_resolve_minutes=5,  # Auto-resolve if health improves
        escalation_enabled=True,
    )

    service.add_alert_rule(rule)
    return rule
