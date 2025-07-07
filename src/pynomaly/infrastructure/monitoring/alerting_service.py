"""Alerting and notification service for monitoring."""

from __future__ import annotations

import asyncio
import json
import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import requests


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert definition."""

    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    source: str = "pynomaly"
    tags: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    suppressed_until: datetime | None = None


class AlertNotifier(ABC):
    """Abstract base class for alert notifiers."""

    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert notification.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully, False otherwise
        """
        pass


class EmailNotifier(AlertNotifier):
    """Email alert notifier."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: str | None = None,
        password: str | None = None,
        from_email: str | None = None,
        to_emails: list[str] | None = None,
        use_tls: bool = True,
    ):
        """Initialize email notifier.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
            to_emails: List of recipient email addresses
            use_tls: Whether to use TLS
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email or username
        self.to_emails = to_emails or []
        self.use_tls = use_tls
        self.logger = logging.getLogger(__name__)

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()

                if self.username and self.password:
                    server.login(self.username, self.password)

                server.send_message(msg)

            self.logger.info(f"Alert email sent: {alert.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send alert email: {e}")
            return False

    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.EMERGENCY: "#6f42c1",
        }

        color = severity_colors.get(alert.severity, "#6c757d")

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: {color}; color: white; padding: 10px; border-radius: 5px;">
                <h2>{alert.title}</h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div style="margin: 20px 0;">
                <h3>Description</h3>
                <p>{alert.description}</p>
            </div>
            
            <div style="margin: 20px 0;">
                <h3>Details</h3>
                <ul>
                    <li><strong>Alert ID:</strong> {alert.id}</li>
                    <li><strong>Source:</strong> {alert.source}</li>
                    <li><strong>Status:</strong> {alert.status.value}</li>
                </ul>
            </div>
            
            {self._format_metrics_html(alert.metrics) if alert.metrics else ""}
            {self._format_tags_html(alert.tags) if alert.tags else ""}
            
            <div style="margin-top: 30px; font-size: 12px; color: #666;">
                <p>This alert was generated by Pynomaly monitoring system.</p>
            </div>
        </body>
        </html>
        """

    def _format_metrics_html(self, metrics: dict[str, Any]) -> str:
        """Format metrics as HTML."""
        items = "".join(
            [
                f"<li><strong>{key}:</strong> {value}</li>"
                for key, value in metrics.items()
            ]
        )
        return f"<div><h3>Metrics</h3><ul>{items}</ul></div>"

    def _format_tags_html(self, tags: dict[str, str]) -> str:
        """Format tags as HTML."""
        items = "".join(
            [f"<li><strong>{key}:</strong> {value}</li>" for key, value in tags.items()]
        )
        return f"<div><h3>Tags</h3><ul>{items}</ul></div>"


class SlackNotifier(AlertNotifier):
    """Slack alert notifier."""

    def __init__(self, webhook_url: str):
        """Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            # Create Slack message
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9f00",
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.EMERGENCY: "#800080",
            }

            payload = {
                "text": f"Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True,
                            },
                            {
                                "title": "Status",
                                "value": alert.status.value,
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": alert.created_at.strftime(
                                    "%Y-%m-%d %H:%M:%S UTC"
                                ),
                                "short": True,
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True,
                            },
                            {
                                "title": "Description",
                                "value": alert.description,
                                "short": False,
                            },
                        ],
                    }
                ],
            }

            # Add metrics if present
            if alert.metrics:
                metrics_text = "\n".join(
                    [f"• {key}: {value}" for key, value in alert.metrics.items()]
                )
                payload["attachments"][0]["fields"].append(
                    {
                        "title": "Metrics",
                        "value": f"```{metrics_text}```",
                        "short": False,
                    }
                )

            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()

            self.logger.info(f"Alert sent to Slack: {alert.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False


class WebhookNotifier(AlertNotifier):
    """Generic webhook alert notifier."""

    def __init__(self, webhook_url: str, headers: dict[str, str] | None = None):
        """Initialize webhook notifier.

        Args:
            webhook_url: Webhook URL
            headers: Additional HTTP headers
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.logger = logging.getLogger(__name__)

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            # Create payload
            payload = {
                "id": alert.id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "created_at": alert.created_at.isoformat(),
                "resolved_at": (
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ),
                "source": alert.source,
                "tags": alert.tags,
                "metrics": alert.metrics,
            }

            # Send webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10,
            )
            response.raise_for_status()

            self.logger.info(f"Alert sent via webhook: {alert.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False


class AlertingService:
    """Main alerting service."""

    def __init__(self):
        """Initialize alerting service."""
        self.notifiers: list[AlertNotifier] = []
        self.alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.suppression_rules: list[dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def add_notifier(self, notifier: AlertNotifier) -> None:
        """Add an alert notifier."""
        self.notifiers.append(notifier)

    def add_suppression_rule(
        self,
        tags: dict[str, str],
        duration_minutes: int = 60,
    ) -> None:
        """Add alert suppression rule.

        Args:
            tags: Tags that must match for suppression
            duration_minutes: Suppression duration in minutes
        """
        self.suppression_rules.append(
            {
                "tags": tags,
                "duration": timedelta(minutes=duration_minutes),
            }
        )

    async def send_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        tags: dict[str, str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        """Send an alert.

        Args:
            title: Alert title
            description: Alert description
            severity: Alert severity
            tags: Alert tags
            metrics: Alert metrics

        Returns:
            Alert ID
        """
        # Create alert
        alert_id = f"alert_{int(datetime.utcnow().timestamp() * 1000000)}"
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            tags=tags or {},
            metrics=metrics or {},
        )

        # Check suppression rules
        if self._is_suppressed(alert):
            alert.status = AlertStatus.SUPPRESSED
            self.logger.info(f"Alert suppressed: {alert_id}")
            return alert_id

        # Store alert
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Send notifications
        await self._send_notifications(alert)

        self.logger.info(f"Alert created: {alert_id}")
        return alert_id

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if resolved successfully
        """
        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()

        # Send resolution notification
        await self._send_notifications(alert)

        # Remove from active alerts
        del self.alerts[alert_id]

        self.logger.info(f"Alert resolved: {alert_id}")
        return True

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts."""
        return list(self.alerts.values())

    def get_alert_history(self, hours: int = 24) -> list[Alert]:
        """Get alert history.

        Args:
            hours: Hours of history to retrieve

        Returns:
            List of alerts from the specified time period
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.created_at >= cutoff]

    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed."""
        for rule in self.suppression_rules:
            # Check if tags match
            rule_tags = rule["tags"]
            if all(alert.tags.get(key) == value for key, value in rule_tags.items()):
                # Check if we're within suppression window
                suppression_end = datetime.utcnow() + rule["duration"]
                alert.suppressed_until = suppression_end
                return True

        return False

    async def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications to all configured notifiers."""
        if not self.notifiers:
            self.logger.warning("No notifiers configured")
            return

        # Send to all notifiers concurrently
        tasks = [notifier.send_alert(alert) for notifier in self.notifiers]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results
        success_count = sum(1 for result in results if result is True)
        self.logger.info(
            f"Alert {alert.id} sent to {success_count}/{len(self.notifiers)} notifiers"
        )


def create_alerting_service(
    email_config: dict[str, Any] | None = None,
    slack_webhook: str | None = None,
    webhook_url: str | None = None,
) -> AlertingService:
    """Create alerting service with configured notifiers.

    Args:
        email_config: Email configuration
        slack_webhook: Slack webhook URL
        webhook_url: Generic webhook URL

    Returns:
        Configured alerting service
    """
    service = AlertingService()

    # Add email notifier if configured
    if email_config:
        notifier = EmailNotifier(**email_config)
        service.add_notifier(notifier)

    # Add Slack notifier if configured
    if slack_webhook:
        notifier = SlackNotifier(slack_webhook)
        service.add_notifier(notifier)

    # Add webhook notifier if configured
    if webhook_url:
        notifier = WebhookNotifier(webhook_url)
        service.add_notifier(notifier)

    return service
