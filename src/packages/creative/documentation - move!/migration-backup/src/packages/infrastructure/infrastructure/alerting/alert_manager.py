#!/usr/bin/env python3
"""
Real-time Alert Manager for Pynomaly.
This module provides comprehensive alerting capabilities with multiple notification channels.
"""

import json
import os
import smtplib
import uuid
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import aiohttp
import structlog
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Database setup
Base = declarative_base()


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status types."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Notification channel types."""

    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"


class AlertRule(Base):
    """Alert rule database model."""

    __tablename__ = "alert_rules"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Rule configuration
    metric_name = Column(String(255), nullable=False)
    condition = Column(String(255), nullable=False)  # >, <, >=, <=, ==, !=
    threshold = Column(String(255), nullable=False)  # Can be numeric or string
    duration = Column(Integer, default=0)  # Duration in seconds

    # Alert properties
    severity = Column(String(50), nullable=False)
    enabled = Column(Boolean, default=True)

    # Notification settings
    notification_channels = Column(Text, nullable=False)  # JSON array
    notification_template = Column(Text, nullable=True)

    # Timing
    cooldown_period = Column(Integer, default=300)  # 5 minutes default

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=True)


class Alert(Base):
    """Alert database model."""

    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    rule_id = Column(String(36), nullable=False)

    # Alert details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(String(50), nullable=False)
    status = Column(String(50), default=AlertStatus.ACTIVE.value)

    # Trigger information
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(String(255), nullable=False)
    threshold = Column(String(255), nullable=False)

    # Timing
    triggered_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Metadata
    metadata = Column(Text, nullable=True)  # JSON
    fingerprint = Column(String(255), nullable=True)  # For deduplication

    # Notification tracking
    notifications_sent = Column(Integer, default=0)
    last_notification_at = Column(DateTime, nullable=True)


# Pydantic models
class AlertRuleCreate(BaseModel):
    """Alert rule creation request."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    metric_name: str = Field(..., min_length=1, max_length=255)
    condition: str = Field(..., regex=r'^(>|<|>=|<=|==|!=)$')
    threshold: str = Field(..., min_length=1, max_length=255)
    duration: int = Field(default=0, ge=0)
    severity: AlertSeverity
    enabled: bool = True
    notification_channels: list[NotificationChannel]
    notification_template: str | None = None
    cooldown_period: int = Field(default=300, ge=0)


class AlertRuleUpdate(BaseModel):
    """Alert rule update request."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    metric_name: str | None = Field(None, min_length=1, max_length=255)
    condition: str | None = Field(None, regex=r'^(>|<|>=|<=|==|!=)$')
    threshold: str | None = Field(None, min_length=1, max_length=255)
    duration: int | None = Field(None, ge=0)
    severity: AlertSeverity | None = None
    enabled: bool | None = None
    notification_channels: list[NotificationChannel] | None = None
    notification_template: str | None = None
    cooldown_period: int | None = Field(None, ge=0)


class AlertInfo(BaseModel):
    """Alert information."""

    id: str
    rule_id: str
    name: str
    description: str | None
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    metric_value: str
    threshold: str
    triggered_at: datetime
    acknowledged_at: datetime | None
    resolved_at: datetime | None
    metadata: dict[str, Any] | None
    fingerprint: str | None
    notifications_sent: int
    last_notification_at: datetime | None


class NotificationConfig(BaseModel):
    """Notification configuration."""

    # Email configuration
    email_smtp_host: str | None = None
    email_smtp_port: int | None = 587
    email_smtp_username: str | None = None
    email_smtp_password: str | None = None
    email_from: str | None = None
    email_to: list[str] = []

    # Slack configuration
    slack_webhook_url: str | None = None
    slack_channel: str | None = None
    slack_username: str | None = "Pynomaly"

    # Discord configuration
    discord_webhook_url: str | None = None

    # Teams configuration
    teams_webhook_url: str | None = None

    # Generic webhook configuration
    webhook_url: str | None = None
    webhook_headers: dict[str, str] = {}

    # SMS configuration (placeholder)
    sms_provider: str | None = None
    sms_api_key: str | None = None
    sms_from: str | None = None
    sms_to: list[str] = []

    # PagerDuty configuration
    pagerduty_integration_key: str | None = None


class AlertManager:
    """Comprehensive alert management system."""

    def __init__(self, database_url: str, notification_config: NotificationConfig):
        """Initialize alert manager."""
        self.database_url = database_url
        self.notification_config = notification_config

        # Database setup
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

        # Alert tracking
        self.active_alerts: dict[str, Alert] = {}
        self.rule_cooldowns: dict[str, datetime] = {}

        # Notification clients
        self.http_session: aiohttp.ClientSession | None = None

        logger.info("Alert manager initialized")

    async def start(self):
        """Start alert manager."""
        self.http_session = aiohttp.ClientSession()
        logger.info("Alert manager started")

    async def stop(self):
        """Stop alert manager."""
        if self.http_session:
            await self.http_session.close()
        logger.info("Alert manager stopped")

    def get_db(self):
        """Get database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Alert Rule Management
    async def create_alert_rule(self, rule_data: AlertRuleCreate) -> str:
        """Create a new alert rule."""
        db = next(self.get_db())

        try:
            rule = AlertRule(
                name=rule_data.name,
                description=rule_data.description,
                metric_name=rule_data.metric_name,
                condition=rule_data.condition,
                threshold=rule_data.threshold,
                duration=rule_data.duration,
                severity=rule_data.severity.value,
                enabled=rule_data.enabled,
                notification_channels=json.dumps([ch.value for ch in rule_data.notification_channels]),
                notification_template=rule_data.notification_template,
                cooldown_period=rule_data.cooldown_period,
            )

            db.add(rule)
            db.commit()
            db.refresh(rule)

            logger.info(f"Created alert rule: {rule.name}")
            return rule.id

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create alert rule: {e}")
            raise
        finally:
            db.close()

    async def get_alert_rules(self) -> list[AlertRule]:
        """Get all alert rules."""
        db = next(self.get_db())

        try:
            rules = db.query(AlertRule).all()
            return rules
        finally:
            db.close()

    async def update_alert_rule(self, rule_id: str, update_data: AlertRuleUpdate) -> bool:
        """Update an alert rule."""
        db = next(self.get_db())

        try:
            rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
            if not rule:
                return False

            # Update fields
            update_dict = update_data.dict(exclude_unset=True)
            for key, value in update_dict.items():
                if key == "severity" and value:
                    value = value.value
                elif key == "notification_channels" and value:
                    value = json.dumps([ch.value for ch in value])
                setattr(rule, key, value)

            rule.updated_at = datetime.utcnow()
            db.commit()

            logger.info(f"Updated alert rule: {rule.name}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update alert rule: {e}")
            raise
        finally:
            db.close()

    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule."""
        db = next(self.get_db())

        try:
            rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
            if not rule:
                return False

            db.delete(rule)
            db.commit()

            logger.info(f"Deleted alert rule: {rule.name}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete alert rule: {e}")
            raise
        finally:
            db.close()

    # Alert Processing
    async def process_metric(self, metric_name: str, value: float, metadata: dict[str, Any] = None):
        """Process a metric value against all alert rules."""
        db = next(self.get_db())

        try:
            # Get all enabled rules for this metric
            rules = db.query(AlertRule).filter(
                AlertRule.metric_name == metric_name,
                AlertRule.enabled == True
            ).all()

            for rule in rules:
                await self._evaluate_rule(rule, value, metadata or {})

        except Exception as e:
            logger.error(f"Failed to process metric {metric_name}: {e}")
        finally:
            db.close()

    async def _evaluate_rule(self, rule: AlertRule, value: float, metadata: dict[str, Any]):
        """Evaluate a single alert rule."""
        try:
            # Check cooldown
            if rule.id in self.rule_cooldowns:
                if datetime.utcnow() < self.rule_cooldowns[rule.id]:
                    return

            # Parse threshold
            try:
                threshold = float(rule.threshold)
            except ValueError:
                # String comparison
                threshold = rule.threshold

            # Evaluate condition
            triggered = False
            if rule.condition == ">":
                triggered = value > threshold
            elif rule.condition == "<":
                triggered = value < threshold
            elif rule.condition == ">=":
                triggered = value >= threshold
            elif rule.condition == "<=":
                triggered = value <= threshold
            elif rule.condition == "==":
                triggered = value == threshold
            elif rule.condition == "!=":
                triggered = value != threshold

            if triggered:
                await self._trigger_alert(rule, value, metadata)
            else:
                await self._resolve_alert(rule.id)

        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule.name}: {e}")

    async def _trigger_alert(self, rule: AlertRule, value: float, metadata: dict[str, Any]):
        """Trigger an alert."""
        try:
            # Create fingerprint for deduplication
            fingerprint = f"{rule.id}:{rule.metric_name}:{rule.threshold}"

            # Check if alert already exists
            db = next(self.get_db())
            existing_alert = db.query(Alert).filter(
                Alert.rule_id == rule.id,
                Alert.status == AlertStatus.ACTIVE.value,
                Alert.fingerprint == fingerprint
            ).first()

            if existing_alert:
                # Update existing alert
                existing_alert.metric_value = str(value)
                existing_alert.metadata = json.dumps(metadata)
                db.commit()
                db.close()
                return

            # Create new alert
            alert = Alert(
                rule_id=rule.id,
                name=rule.name,
                description=rule.description,
                severity=rule.severity,
                status=AlertStatus.ACTIVE.value,
                metric_name=rule.metric_name,
                metric_value=str(value),
                threshold=rule.threshold,
                metadata=json.dumps(metadata),
                fingerprint=fingerprint,
            )

            db.add(alert)
            db.commit()
            db.refresh(alert)
            db.close()

            # Send notifications
            await self._send_notifications(alert, rule)

            # Set cooldown
            self.rule_cooldowns[rule.id] = datetime.utcnow() + timedelta(seconds=rule.cooldown_period)

            logger.warning(f"Alert triggered: {alert.name}")

        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")

    async def _resolve_alert(self, rule_id: str):
        """Resolve active alerts for a rule."""
        db = next(self.get_db())

        try:
            alerts = db.query(Alert).filter(
                Alert.rule_id == rule_id,
                Alert.status == AlertStatus.ACTIVE.value
            ).all()

            for alert in alerts:
                alert.status = AlertStatus.RESOLVED.value
                alert.resolved_at = datetime.utcnow()

                logger.info(f"Alert resolved: {alert.name}")

            db.commit()

        except Exception as e:
            logger.error(f"Failed to resolve alerts: {e}")
        finally:
            db.close()

    # Notification System
    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for an alert."""
        try:
            channels = json.loads(rule.notification_channels)

            for channel in channels:
                if channel == NotificationChannel.EMAIL.value:
                    await self._send_email_notification(alert, rule)
                elif channel == NotificationChannel.SLACK.value:
                    await self._send_slack_notification(alert, rule)
                elif channel == NotificationChannel.DISCORD.value:
                    await self._send_discord_notification(alert, rule)
                elif channel == NotificationChannel.TEAMS.value:
                    await self._send_teams_notification(alert, rule)
                elif channel == NotificationChannel.WEBHOOK.value:
                    await self._send_webhook_notification(alert, rule)
                elif channel == NotificationChannel.PAGERDUTY.value:
                    await self._send_pagerduty_notification(alert, rule)

            # Update notification tracking
            db = next(self.get_db())
            db_alert = db.query(Alert).filter(Alert.id == alert.id).first()
            if db_alert:
                db_alert.notifications_sent += 1
                db_alert.last_notification_at = datetime.utcnow()
                db.commit()
            db.close()

        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")

    async def _send_email_notification(self, alert: Alert, rule: AlertRule):
        """Send email notification."""
        try:
            if not self.notification_config.email_smtp_host:
                return

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.notification_config.email_from
            msg['To'] = ', '.join(self.notification_config.email_to)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.name}"

            # Email body
            body = f"""
Alert: {alert.name}
Severity: {alert.severity}
Metric: {alert.metric_name}
Value: {alert.metric_value}
Threshold: {alert.threshold}
Triggered: {alert.triggered_at}
Description: {alert.description or 'No description provided'}
            """

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            with smtplib.SMTP(self.notification_config.email_smtp_host, self.notification_config.email_smtp_port) as server:
                server.starttls()
                server.login(self.notification_config.email_smtp_username, self.notification_config.email_smtp_password)
                server.send_message(msg)

            logger.info(f"Email notification sent for alert: {alert.name}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    async def _send_slack_notification(self, alert: Alert, rule: AlertRule):
        """Send Slack notification."""
        try:
            if not self.notification_config.slack_webhook_url:
                return

            # Color mapping
            colors = {
                AlertSeverity.CRITICAL.value: "#FF0000",
                AlertSeverity.HIGH.value: "#FF8000",
                AlertSeverity.MEDIUM.value: "#FFFF00",
                AlertSeverity.LOW.value: "#00FF00",
                AlertSeverity.INFO.value: "#0080FF",
            }

            # Create Slack message
            message = {
                "username": self.notification_config.slack_username,
                "attachments": [
                    {
                        "color": colors.get(alert.severity, "#808080"),
                        "title": f"🚨 {alert.name}",
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Metric",
                                "value": f"{alert.metric_name}: {alert.metric_value}",
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": alert.threshold,
                                "short": True
                            },
                            {
                                "title": "Triggered",
                                "value": alert.triggered_at.isoformat(),
                                "short": True
                            }
                        ],
                        "footer": "Pynomaly Alert System",
                        "ts": int(alert.triggered_at.timestamp())
                    }
                ]
            }

            # Send notification
            async with self.http_session.post(
                self.notification_config.slack_webhook_url,
                json=message
            ) as response:
                if response.status == 200:
                    logger.info(f"Slack notification sent for alert: {alert.name}")
                else:
                    logger.error(f"Failed to send Slack notification: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    async def _send_discord_notification(self, alert: Alert, rule: AlertRule):
        """Send Discord notification."""
        try:
            if not self.notification_config.discord_webhook_url:
                return

            # Color mapping
            colors = {
                AlertSeverity.CRITICAL.value: 0xFF0000,
                AlertSeverity.HIGH.value: 0xFF8000,
                AlertSeverity.MEDIUM.value: 0xFFFF00,
                AlertSeverity.LOW.value: 0x00FF00,
                AlertSeverity.INFO.value: 0x0080FF,
            }

            # Create Discord message
            message = {
                "embeds": [
                    {
                        "title": f"🚨 {alert.name}",
                        "description": alert.description or "No description provided",
                        "color": colors.get(alert.severity, 0x808080),
                        "fields": [
                            {
                                "name": "Severity",
                                "value": alert.severity.upper(),
                                "inline": True
                            },
                            {
                                "name": "Metric",
                                "value": f"{alert.metric_name}: {alert.metric_value}",
                                "inline": True
                            },
                            {
                                "name": "Threshold",
                                "value": alert.threshold,
                                "inline": True
                            }
                        ],
                        "timestamp": alert.triggered_at.isoformat(),
                        "footer": {
                            "text": "Pynomaly Alert System"
                        }
                    }
                ]
            }

            # Send notification
            async with self.http_session.post(
                self.notification_config.discord_webhook_url,
                json=message
            ) as response:
                if response.status == 204:
                    logger.info(f"Discord notification sent for alert: {alert.name}")
                else:
                    logger.error(f"Failed to send Discord notification: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

    async def _send_teams_notification(self, alert: Alert, rule: AlertRule):
        """Send Microsoft Teams notification."""
        try:
            if not self.notification_config.teams_webhook_url:
                return

            # Create Teams message
            message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "FF0000" if alert.severity == AlertSeverity.CRITICAL.value else "FF8000",
                "summary": f"Pynomaly Alert: {alert.name}",
                "sections": [
                    {
                        "activityTitle": f"🚨 {alert.name}",
                        "activitySubtitle": f"Severity: {alert.severity.upper()}",
                        "facts": [
                            {
                                "name": "Metric",
                                "value": f"{alert.metric_name}: {alert.metric_value}"
                            },
                            {
                                "name": "Threshold",
                                "value": alert.threshold
                            },
                            {
                                "name": "Triggered",
                                "value": alert.triggered_at.isoformat()
                            }
                        ]
                    }
                ]
            }

            # Send notification
            async with self.http_session.post(
                self.notification_config.teams_webhook_url,
                json=message
            ) as response:
                if response.status == 200:
                    logger.info(f"Teams notification sent for alert: {alert.name}")
                else:
                    logger.error(f"Failed to send Teams notification: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")

    async def _send_webhook_notification(self, alert: Alert, rule: AlertRule):
        """Send generic webhook notification."""
        try:
            if not self.notification_config.webhook_url:
                return

            # Create webhook payload
            payload = {
                "alert": {
                    "id": alert.id,
                    "name": alert.name,
                    "description": alert.description,
                    "severity": alert.severity,
                    "status": alert.status,
                    "metric_name": alert.metric_name,
                    "metric_value": alert.metric_value,
                    "threshold": alert.threshold,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "metadata": json.loads(alert.metadata) if alert.metadata else {}
                },
                "rule": {
                    "id": rule.id,
                    "name": rule.name,
                    "condition": rule.condition,
                    "duration": rule.duration
                }
            }

            # Send notification
            async with self.http_session.post(
                self.notification_config.webhook_url,
                json=payload,
                headers=self.notification_config.webhook_headers
            ) as response:
                if response.status == 200:
                    logger.info(f"Webhook notification sent for alert: {alert.name}")
                else:
                    logger.error(f"Failed to send webhook notification: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")

    async def _send_pagerduty_notification(self, alert: Alert, rule: AlertRule):
        """Send PagerDuty notification."""
        try:
            if not self.notification_config.pagerduty_integration_key:
                return

            # Create PagerDuty event
            event = {
                "routing_key": self.notification_config.pagerduty_integration_key,
                "event_action": "trigger",
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": f"{alert.name} - {alert.severity.upper()}",
                    "source": "monorepo",
                    "severity": alert.severity,
                    "component": alert.metric_name,
                    "custom_details": {
                        "metric_value": alert.metric_value,
                        "threshold": alert.threshold,
                        "triggered_at": alert.triggered_at.isoformat()
                    }
                }
            }

            # Send notification
            async with self.http_session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=event
            ) as response:
                if response.status == 202:
                    logger.info(f"PagerDuty notification sent for alert: {alert.name}")
                else:
                    logger.error(f"Failed to send PagerDuty notification: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")

    # Alert Management
    async def get_active_alerts(self) -> list[AlertInfo]:
        """Get all active alerts."""
        db = next(self.get_db())

        try:
            alerts = db.query(Alert).filter(Alert.status == AlertStatus.ACTIVE.value).all()

            return [
                AlertInfo(
                    id=alert.id,
                    rule_id=alert.rule_id,
                    name=alert.name,
                    description=alert.description,
                    severity=AlertSeverity(alert.severity),
                    status=AlertStatus(alert.status),
                    metric_name=alert.metric_name,
                    metric_value=alert.metric_value,
                    threshold=alert.threshold,
                    triggered_at=alert.triggered_at,
                    acknowledged_at=alert.acknowledged_at,
                    resolved_at=alert.resolved_at,
                    metadata=json.loads(alert.metadata) if alert.metadata else None,
                    fingerprint=alert.fingerprint,
                    notifications_sent=alert.notifications_sent,
                    last_notification_at=alert.last_notification_at
                )
                for alert in alerts
            ]

        finally:
            db.close()

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        db = next(self.get_db())

        try:
            alert = db.query(Alert).filter(Alert.id == alert_id).first()
            if not alert:
                return False

            alert.status = AlertStatus.ACKNOWLEDGED.value
            alert.acknowledged_at = datetime.utcnow()
            db.commit()

            logger.info(f"Alert acknowledged: {alert.name}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to acknowledge alert: {e}")
            raise
        finally:
            db.close()

    async def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        db = next(self.get_db())

        try:
            alert = db.query(Alert).filter(Alert.id == alert_id).first()
            if not alert:
                return False

            alert.status = AlertStatus.RESOLVED.value
            alert.resolved_at = datetime.utcnow()
            db.commit()

            logger.info(f"Alert resolved: {alert.name}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to resolve alert: {e}")
            raise
        finally:
            db.close()


# Global alert manager instance
alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get alert manager instance."""
    global alert_manager
    if alert_manager is None:
        database_url = os.getenv("DATABASE_URL", "sqlite:///alerts.db")

        # Create notification config from environment
        notification_config = NotificationConfig(
            email_smtp_host=os.getenv("EMAIL_SMTP_HOST"),
            email_smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "587")),
            email_smtp_username=os.getenv("EMAIL_SMTP_USERNAME"),
            email_smtp_password=os.getenv("EMAIL_SMTP_PASSWORD"),
            email_from=os.getenv("EMAIL_FROM"),
            email_to=os.getenv("EMAIL_TO", "").split(",") if os.getenv("EMAIL_TO") else [],
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
            slack_channel=os.getenv("SLACK_CHANNEL"),
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
            teams_webhook_url=os.getenv("TEAMS_WEBHOOK_URL"),
            webhook_url=os.getenv("WEBHOOK_URL"),
            pagerduty_integration_key=os.getenv("PAGERDUTY_INTEGRATION_KEY"),
        )

        alert_manager = AlertManager(database_url, notification_config)

    return alert_manager


# Make components available for import
__all__ = [
    "AlertManager",
    "AlertSeverity",
    "AlertStatus",
    "NotificationChannel",
    "AlertRuleCreate",
    "AlertRuleUpdate",
    "AlertInfo",
    "NotificationConfig",
    "get_alert_manager",
]
