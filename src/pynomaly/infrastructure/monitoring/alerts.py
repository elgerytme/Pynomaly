#!/usr/bin/env python3
"""
Real-time Monitoring and Alerting System for Pynomaly

This module provides comprehensive monitoring and alerting capabilities
for production environments.
"""

import json
import logging
import smtplib
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any

import psutil
import requests
import yaml


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    timestamp: datetime
    source: str
    tags: dict[str, str]
    threshold_value: float | None = None
    current_value: float | None = None
    resolution_time: datetime | None = None
    acknowledged_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "tags": self.tags,
            "threshold_value": self.threshold_value,
            "current_value": self.current_value,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "acknowledged_by": self.acknowledged_by
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    description: str
    condition: str
    severity: AlertSeverity
    threshold: float
    duration: int  # seconds
    cooldown: int  # seconds
    enabled: bool = True
    tags: dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class MetricsCollector:
    """Collect system and application metrics."""

    def __init__(self):
        self.metrics = {}
        self.running = False
        self.collection_interval = 30  # seconds
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start metrics collection."""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        self.logger.info("Metrics collection started")

    def stop(self):
        """Stop metrics collection."""
        self.running = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join()
        self.logger.info("Metrics collection stopped")

    def _collect_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")

    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free

            # Network metrics
            net_io = psutil.net_io_counters()

            # Load average
            load_avg = psutil.getloadavg()

            self.metrics.update({
                "system.cpu.percent": cpu_percent,
                "system.cpu.count": cpu_count,
                "system.memory.percent": memory_percent,
                "system.memory.available": memory_available,
                "system.disk.percent": disk_percent,
                "system.disk.free": disk_free,
                "system.network.bytes_sent": net_io.bytes_sent,
                "system.network.bytes_recv": net_io.bytes_recv,
                "system.load.1min": load_avg[0],
                "system.load.5min": load_avg[1],
                "system.load.15min": load_avg[2],
                "timestamp": datetime.now()
            })

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # Database connection pool
            self.metrics["app.db.connections"] = self._get_db_connections()

            # Cache hit rate
            self.metrics["app.cache.hit_rate"] = self._get_cache_hit_rate()

            # API response times
            self.metrics["app.api.response_time"] = self._get_api_response_time()

            # Error rates
            self.metrics["app.error.rate"] = self._get_error_rate()

            # Queue sizes
            self.metrics["app.queue.size"] = self._get_queue_size()

        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")

    def _get_db_connections(self) -> int:
        """Get database connection count."""
        # Placeholder - implement based on your database
        return 10

    def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        # Placeholder - implement based on your cache
        return 0.95

    def _get_api_response_time(self) -> float:
        """Get average API response time."""
        # Placeholder - implement based on your API metrics
        return 150.0  # milliseconds

    def _get_error_rate(self) -> float:
        """Get error rate."""
        # Placeholder - implement based on your error tracking
        return 0.01  # 1%

    def _get_queue_size(self) -> int:
        """Get queue size."""
        # Placeholder - implement based on your queue system
        return 5

    def get_metric(self, name: str) -> Any | None:
        """Get metric value."""
        return self.metrics.get(name)

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics."""
        return self.metrics.copy()


class AlertManager:
    """Manage alerts and notifications."""

    def __init__(self, config_path: str = "config/monitoring/alert_rules.yml"):
        self.config_path = Path(config_path)
        self.rules: list[AlertRule] = []
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.metrics_collector = MetricsCollector()
        self.notification_channels: list[NotificationChannel] = []
        self.running = False
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self._load_configuration()

        # Setup notification channels
        self._setup_notification_channels()

    def _load_configuration(self):
        """Load alert rules from configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    config = yaml.safe_load(f)

                self.rules = []
                for rule_data in config.get('rules', []):
                    rule = AlertRule(
                        name=rule_data['name'],
                        description=rule_data['description'],
                        condition=rule_data['condition'],
                        severity=AlertSeverity(rule_data['severity']),
                        threshold=rule_data['threshold'],
                        duration=rule_data.get('duration', 60),
                        cooldown=rule_data.get('cooldown', 300),
                        enabled=rule_data.get('enabled', True),
                        tags=rule_data.get('tags', {})
                    )
                    self.rules.append(rule)

                self.logger.info(f"Loaded {len(self.rules)} alert rules")
            else:
                self.logger.warning(f"Alert rules file not found: {self.config_path}")

        except Exception as e:
            self.logger.error(f"Error loading alert configuration: {e}")

    def _setup_notification_channels(self):
        """Setup notification channels."""
        # Email notifications
        email_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "alerts@pynomaly.com",
            "password": "your_app_password",
            "recipients": ["admin@pynomaly.com", "ops@pynomaly.com"]
        }
        self.notification_channels.append(EmailNotificationChannel(email_config))

        # Slack notifications
        slack_config = {
            "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
            "channel": "#alerts"
        }
        self.notification_channels.append(SlackNotificationChannel(slack_config))

        # PagerDuty notifications
        pagerduty_config = {
            "integration_key": "your_pagerduty_integration_key"
        }
        self.notification_channels.append(PagerDutyNotificationChannel(pagerduty_config))

    def start(self):
        """Start alert monitoring."""
        self.running = True
        self.metrics_collector.start()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        self.logger.info("Alert monitoring started")

    def stop(self):
        """Stop alert monitoring."""
        self.running = False
        self.metrics_collector.stop()

        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join()

        self.logger.info("Alert monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._evaluate_rules()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    def _evaluate_rules(self):
        """Evaluate alert rules."""
        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                self._evaluate_rule(rule)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.name}: {e}")

    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        # Get current metric value
        current_value = self._get_metric_value(rule.condition)

        if current_value is None:
            return

        # Check if threshold is exceeded
        threshold_exceeded = self._check_threshold(current_value, rule.threshold, rule.condition)

        alert_id = f"{rule.name}_{hash(rule.condition)}"

        if threshold_exceeded:
            if alert_id not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    id=alert_id,
                    title=rule.name,
                    description=rule.description,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    timestamp=datetime.now(),
                    source="pynomaly_monitoring",
                    tags=rule.tags,
                    threshold_value=rule.threshold,
                    current_value=current_value
                )

                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)

                # Send notifications
                self._send_alert_notifications(alert)

                self.logger.warning(f"Alert triggered: {rule.name}")
        else:
            # Resolve alert if it exists
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolution_time = datetime.now()

                del self.active_alerts[alert_id]

                # Send resolution notification
                self._send_resolution_notifications(alert)

                self.logger.info(f"Alert resolved: {rule.name}")

    def _get_metric_value(self, condition: str) -> float | None:
        """Get metric value from condition."""
        # Parse condition to extract metric name
        # This is a simplified implementation
        if "cpu.percent" in condition:
            return self.metrics_collector.get_metric("system.cpu.percent")
        elif "memory.percent" in condition:
            return self.metrics_collector.get_metric("system.memory.percent")
        elif "disk.percent" in condition:
            return self.metrics_collector.get_metric("system.disk.percent")
        elif "error.rate" in condition:
            return self.metrics_collector.get_metric("app.error.rate")
        elif "response_time" in condition:
            return self.metrics_collector.get_metric("app.api.response_time")

        return None

    def _check_threshold(self, current_value: float, threshold: float, condition: str) -> bool:
        """Check if threshold is exceeded."""
        if ">" in condition:
            return current_value > threshold
        elif "<" in condition:
            return current_value < threshold
        elif "==" in condition:
            return abs(current_value - threshold) < 0.001

        return False

    def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications."""
        for channel in self.notification_channels:
            try:
                channel.send_alert(alert)
            except Exception as e:
                self.logger.error(f"Error sending alert notification: {e}")

    def _send_resolution_notifications(self, alert: Alert):
        """Send resolution notifications."""
        for channel in self.notification_channels:
            try:
                channel.send_resolution(alert)
            except Exception as e:
                self.logger.error(f"Error sending resolution notification: {e}")

    def get_active_alerts(self) -> list[Alert]:
        """Get active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self) -> list[Alert]:
        """Get alert history."""
        return self.alert_history

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by

            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")


class NotificationChannel:
    """Base class for notification channels."""

    def send_alert(self, alert: Alert):
        """Send alert notification."""
        raise NotImplementedError

    def send_resolution(self, alert: Alert):
        """Send resolution notification."""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def send_alert(self, alert: Alert):
        """Send alert via email."""
        try:
            subject = f"🚨 Pynomaly Alert: {alert.title}"
            body = self._format_alert_message(alert)

            self._send_email(subject, body)
            self.logger.info(f"Alert email sent: {alert.id}")

        except Exception as e:
            self.logger.error(f"Error sending alert email: {e}")

    def send_resolution(self, alert: Alert):
        """Send resolution via email."""
        try:
            subject = f"✅ Pynomaly Alert Resolved: {alert.title}"
            body = self._format_resolution_message(alert)

            self._send_email(subject, body)
            self.logger.info(f"Resolution email sent: {alert.id}")

        except Exception as e:
            self.logger.error(f"Error sending resolution email: {e}")

    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert message."""
        return f"""
Pynomaly Alert Notification

Alert ID: {alert.id}
Title: {alert.title}
Description: {alert.description}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Source: {alert.source}

Threshold: {alert.threshold_value}
Current Value: {alert.current_value}

Tags: {json.dumps(alert.tags, indent=2)}

Please investigate and take appropriate action.
"""

    def _format_resolution_message(self, alert: Alert) -> str:
        """Format resolution message."""
        duration = alert.resolution_time - alert.timestamp
        return f"""
Pynomaly Alert Resolution

Alert ID: {alert.id}
Title: {alert.title}
Resolution Time: {alert.resolution_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration}

The alert has been automatically resolved.
"""

    def _send_email(self, subject: str, body: str):
        """Send email."""
        msg = MIMEMultipart()
        msg['From'] = self.config['username']
        msg['To'] = ', '.join(self.config['recipients'])
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
        server.starttls()
        server.login(self.config['username'], self.config['password'])

        text = msg.as_string()
        server.sendmail(self.config['username'], self.config['recipients'], text)
        server.quit()


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def send_alert(self, alert: Alert):
        """Send alert to Slack."""
        try:
            message = self._format_slack_alert(alert)
            self._send_slack_message(message)
            self.logger.info(f"Alert sent to Slack: {alert.id}")

        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {e}")

    def send_resolution(self, alert: Alert):
        """Send resolution to Slack."""
        try:
            message = self._format_slack_resolution(alert)
            self._send_slack_message(message)
            self.logger.info(f"Resolution sent to Slack: {alert.id}")

        except Exception as e:
            self.logger.error(f"Error sending Slack resolution: {e}")

    def _format_slack_alert(self, alert: Alert) -> dict[str, Any]:
        """Format Slack alert message."""
        color = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "danger"
        }

        return {
            "channel": self.config['channel'],
            "username": "Pynomaly Alerts",
            "icon_emoji": ":rotating_light:",
            "attachments": [
                {
                    "color": color[alert.severity],
                    "title": f"🚨 {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True},
                        {"title": "Current Value", "value": str(alert.current_value), "short": True}
                    ],
                    "footer": "Pynomaly Monitoring",
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }

    def _format_slack_resolution(self, alert: Alert) -> dict[str, Any]:
        """Format Slack resolution message."""
        duration = alert.resolution_time - alert.timestamp

        return {
            "channel": self.config['channel'],
            "username": "Pynomaly Alerts",
            "icon_emoji": ":white_check_mark:",
            "attachments": [
                {
                    "color": "good",
                    "title": f"✅ Resolved: {alert.title}",
                    "text": f"Alert resolved after {duration}",
                    "fields": [
                        {"title": "Alert ID", "value": alert.id, "short": True},
                        {"title": "Resolution Time", "value": alert.resolution_time.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "footer": "Pynomaly Monitoring",
                    "ts": int(alert.resolution_time.timestamp())
                }
            ]
        }

    def _send_slack_message(self, message: dict[str, Any]):
        """Send message to Slack."""
        response = requests.post(
            self.config['webhook_url'],
            json=message,
            timeout=30
        )
        response.raise_for_status()


class PagerDutyNotificationChannel(NotificationChannel):
    """PagerDuty notification channel."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def send_alert(self, alert: Alert):
        """Send alert to PagerDuty."""
        try:
            # Only send critical alerts to PagerDuty
            if alert.severity != AlertSeverity.CRITICAL:
                return

            payload = {
                "routing_key": self.config['integration_key'],
                "event_action": "trigger",
                "dedup_key": alert.id,
                "payload": {
                    "summary": f"Pynomaly Alert: {alert.title}",
                    "severity": alert.severity.value,
                    "source": alert.source,
                    "timestamp": alert.timestamp.isoformat(),
                    "custom_details": {
                        "description": alert.description,
                        "threshold": alert.threshold_value,
                        "current_value": alert.current_value,
                        "tags": alert.tags
                    }
                }
            }

            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            self.logger.info(f"Alert sent to PagerDuty: {alert.id}")

        except Exception as e:
            self.logger.error(f"Error sending PagerDuty alert: {e}")

    def send_resolution(self, alert: Alert):
        """Send resolution to PagerDuty."""
        try:
            payload = {
                "routing_key": self.config['integration_key'],
                "event_action": "resolve",
                "dedup_key": alert.id
            }

            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            self.logger.info(f"Resolution sent to PagerDuty: {alert.id}")

        except Exception as e:
            self.logger.error(f"Error sending PagerDuty resolution: {e}")


class HealthChecker:
    """Health check monitoring."""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
        self.running = False

    def start(self):
        """Start health checking."""
        self.running = True
        self.health_thread = threading.Thread(target=self._health_check_loop)
        self.health_thread.daemon = True
        self.health_thread.start()
        self.logger.info("Health checker started")

    def stop(self):
        """Stop health checking."""
        self.running = False
        if hasattr(self, 'health_thread'):
            self.health_thread.join()
        self.logger.info("Health checker stopped")

    def _health_check_loop(self):
        """Main health check loop."""
        while self.running:
            try:
                self._check_database_health()
                self._check_cache_health()
                self._check_api_health()
                self._check_external_services()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")

    def _check_database_health(self):
        """Check database health."""
        try:
            # Placeholder - implement database health check
            # Example: execute simple query and measure response time
            response_time = 50  # milliseconds

            if response_time > 1000:  # 1 second threshold
                self._create_health_alert(
                    "database_slow",
                    "Database Response Slow",
                    f"Database response time: {response_time}ms",
                    AlertSeverity.WARNING
                )

        except Exception as e:
            self._create_health_alert(
                "database_error",
                "Database Connection Error",
                f"Database connection failed: {e}",
                AlertSeverity.CRITICAL
            )

    def _check_cache_health(self):
        """Check cache health."""
        try:
            # Placeholder - implement cache health check
            pass
        except Exception as e:
            self._create_health_alert(
                "cache_error",
                "Cache Connection Error",
                f"Cache connection failed: {e}",
                AlertSeverity.ERROR
            )

    def _check_api_health(self):
        """Check API health."""
        try:
            # Placeholder - implement API health check
            pass
        except Exception as e:
            self._create_health_alert(
                "api_error",
                "API Health Check Failed",
                f"API health check failed: {e}",
                AlertSeverity.ERROR
            )

    def _check_external_services(self):
        """Check external services health."""
        try:
            # Placeholder - implement external service health checks
            pass
        except Exception as e:
            self._create_health_alert(
                "external_service_error",
                "External Service Error",
                f"External service check failed: {e}",
                AlertSeverity.WARNING
            )

    def _create_health_alert(self, alert_id: str, title: str, description: str, severity: AlertSeverity):
        """Create health check alert."""
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            status=AlertStatus.ACTIVE,
            timestamp=datetime.now(),
            source="health_checker",
            tags={"component": "health_check"}
        )

        # Send to alert manager
        self.alert_manager.active_alerts[alert_id] = alert
        self.alert_manager.alert_history.append(alert)
        self.alert_manager._send_alert_notifications(alert)


def main():
    """Main function for testing."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create alert manager
    alert_manager = AlertManager()

    # Create health checker
    health_checker = HealthChecker(alert_manager)

    try:
        # Start monitoring
        alert_manager.start()
        health_checker.start()

        print("Monitoring started. Press Ctrl+C to stop.")

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        alert_manager.stop()
        health_checker.stop()
        print("Monitoring stopped.")


if __name__ == "__main__":
    main()
