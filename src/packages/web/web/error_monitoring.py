"""
Error monitoring and alerting system for web UI
Provides real-time error tracking, alerting, and analysis
"""

import asyncio
import json
import smtplib
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from ...infrastructure.config.settings import get_settings
from .error_handling import ErrorCode, WebUIError, get_web_ui_logger


@dataclass
class ErrorAlert:
    """Error alert configuration"""
    error_code: ErrorCode
    threshold: int
    time_window: int  # minutes
    severity: str
    recipients: list[str]
    enabled: bool = True


@dataclass
class ErrorMetrics:
    """Error metrics for monitoring"""
    error_count: int = 0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    last_error_time: datetime | None = None
    error_types: dict[str, int] = None
    error_levels: dict[str, int] = None

    def __post_init__(self):
        if self.error_types is None:
            self.error_types = {}
        if self.error_levels is None:
            self.error_levels = {}


class ErrorMonitor:
    """Real-time error monitoring system"""

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_web_ui_logger()
        self.error_buffer = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        self.error_timestamps = defaultdict(list)
        self.metrics = ErrorMetrics()
        self.running = False
        self.monitor_thread = None

        # Alert configuration
        self.alert_configs = self._load_alert_configs()

        # Email configuration
        self.email_enabled = self._configure_email()

        # Monitoring intervals
        self.monitoring_interval = 60  # seconds
        self.cleanup_interval = 300  # 5 minutes

    def _load_alert_configs(self) -> list[ErrorAlert]:
        """Load alert configurations"""
        default_alerts = [
            ErrorAlert(
                error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                threshold=5,
                time_window=10,
                severity="critical",
                recipients=["admin@example.com"]
            ),
            ErrorAlert(
                error_code=ErrorCode.SECURITY_VIOLATION_ERROR,
                threshold=1,
                time_window=5,
                severity="critical",
                recipients=["security@example.com"]
            ),
            ErrorAlert(
                error_code=ErrorCode.DATABASE_CONNECTION_ERROR,
                threshold=3,
                time_window=5,
                severity="high",
                recipients=["ops@example.com"]
            ),
            ErrorAlert(
                error_code=ErrorCode.PERFORMANCE_THRESHOLD_ERROR,
                threshold=10,
                time_window=15,
                severity="medium",
                recipients=["dev@example.com"]
            )
        ]

        # Load from configuration file if exists
        config_file = Path("config/error_alerts.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                    return [ErrorAlert(**alert) for alert in config_data]
            except Exception as e:
                self.logger.logger.warning(f"Failed to load alert config: {e}")

        return default_alerts

    def _configure_email(self) -> bool:
        """Configure email settings"""
        return (
            hasattr(self.settings, 'smtp_server') and self.settings.smtp_server is not None and
            hasattr(self.settings, 'smtp_username') and self.settings.smtp_username is not None and
            hasattr(self.settings, 'smtp_password') and self.settings.smtp_password is not None and
            hasattr(self.settings, 'sender_email') and self.settings.sender_email is not None
        )

    def start_monitoring(self):
        """Start the error monitoring system"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.logger.info("Error monitoring started")

    def stop_monitoring(self):
        """Stop the error monitoring system"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.logger.info("Error monitoring stopped")

    def record_error(self, error: WebUIError, request_info: dict[str, Any] | None = None):
        """Record an error for monitoring"""
        error_data = {
            "error_id": error.error_id,
            "error_code": error.error_code.value,
            "error_level": error.error_level.value,
            "message": error.message,
            "timestamp": error.timestamp,
            "request_info": request_info or {}
        }

        # Add to buffer
        self.error_buffer.append(error_data)

        # Update counts
        self.error_counts[error.error_code] += 1
        self.error_timestamps[error.error_code].append(error.timestamp)

        # Update metrics
        self._update_metrics(error_data)

        # Check for alerts
        self._check_alerts(error.error_code, error.timestamp)

    def _update_metrics(self, error_data: dict[str, Any]):
        """Update error metrics"""
        self.metrics.error_count += 1
        self.metrics.last_error_time = error_data["timestamp"]

        # Update error types
        error_code = error_data["error_code"]
        self.metrics.error_types[error_code] = self.metrics.error_types.get(error_code, 0) + 1

        # Update error levels
        error_level = error_data["error_level"]
        self.metrics.error_levels[error_level] = self.metrics.error_levels.get(error_level, 0) + 1

        # Calculate error rate (errors per minute in last hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_errors = [e for e in self.error_buffer if e["timestamp"] > cutoff_time]
        self.metrics.error_rate = len(recent_errors) / 60  # errors per minute

    def _check_alerts(self, error_code: ErrorCode, timestamp: datetime):
        """Check if any alerts should be triggered"""
        for alert in self.alert_configs:
            if not alert.enabled or alert.error_code != error_code:
                continue

            # Check if threshold is exceeded within time window
            cutoff_time = timestamp - timedelta(minutes=alert.time_window)
            recent_errors = [
                t for t in self.error_timestamps[error_code]
                if t > cutoff_time
            ]

            if len(recent_errors) >= alert.threshold:
                self._trigger_alert(alert, len(recent_errors), timestamp)

    def _trigger_alert(self, alert: ErrorAlert, error_count: int, timestamp: datetime):
        """Trigger an alert"""
        alert_data = {
            "alert_id": f"alert_{int(timestamp.timestamp())}",
            "error_code": alert.error_code.value,
            "error_count": error_count,
            "threshold": alert.threshold,
            "time_window": alert.time_window,
            "severity": alert.severity,
            "timestamp": timestamp.isoformat()
        }

        # Log alert
        self.logger.logger.critical(
            f"Error alert triggered: {alert.error_code.value}",
            **alert_data
        )

        # Send email alert if configured
        if self.email_enabled:
            asyncio.create_task(self._send_email_alert(alert, alert_data))

        # Could add other alert channels here (Slack, PagerDuty, etc.)

    async def _send_email_alert(self, alert: ErrorAlert, alert_data: dict[str, Any]):
        """Send email alert"""
        try:
            subject = f"[{alert.severity.upper()}] Pynomaly Error Alert: {alert.error_code.value}"

            body = f"""
            Error Alert - Pynomaly Web UI

            Alert Details:
            - Error Code: {alert.error_code.value}
            - Error Count: {alert_data['error_count']} in {alert.time_window} minutes
            - Threshold: {alert.threshold}
            - Severity: {alert.severity}
            - Time: {alert_data['timestamp']}

            Recent Error Summary:
            {self._get_recent_error_summary()}

            System Metrics:
            - Total Errors: {self.metrics.error_count}
            - Error Rate: {self.metrics.error_rate:.2f} errors/minute
            - Last Error: {self.metrics.last_error_time}

            Please investigate immediately.
            """

            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.settings.sender_email
            msg['To'] = ', '.join(alert.recipients)
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            with smtplib.SMTP(self.settings.smtp_server, self.settings.smtp_port) as server:
                if self.settings.smtp_use_tls:
                    server.starttls()

                server.login(self.settings.smtp_username, self.settings.smtp_password)
                server.send_message(msg)

            self.logger.logger.info(f"Alert email sent for {alert.error_code.value}")

        except Exception as e:
            self.logger.logger.error(f"Failed to send alert email: {e}")

    def _get_recent_error_summary(self) -> str:
        """Get summary of recent errors"""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_errors = [e for e in self.error_buffer if e["timestamp"] > cutoff_time]

        if not recent_errors:
            return "No recent errors"

        # Count by error code
        error_counts = defaultdict(int)
        for error in recent_errors:
            error_counts[error["error_code"]] += 1

        summary = f"Recent errors (last hour): {len(recent_errors)} total\n"
        for error_code, count in error_counts.items():
            summary += f"  - {error_code}: {count}\n"

        return summary

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Cleanup old timestamps
                self._cleanup_old_timestamps()

                # Generate monitoring report
                self._generate_monitoring_report()

                # Sleep until next interval
                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _cleanup_old_timestamps(self):
        """Clean up old timestamps to prevent memory leaks"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)

        for error_code in self.error_timestamps:
            self.error_timestamps[error_code] = [
                t for t in self.error_timestamps[error_code]
                if t > cutoff_time
            ]

    def _generate_monitoring_report(self):
        """Generate periodic monitoring report"""
        if len(self.error_buffer) == 0:
            return

        # Calculate metrics for last hour
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_errors = [e for e in self.error_buffer if e["timestamp"] > cutoff_time]

        if recent_errors:
            error_types = defaultdict(int)
            error_levels = defaultdict(int)

            for error in recent_errors:
                error_types[error["error_code"]] += 1
                error_levels[error["error_level"]] += 1

            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_errors": len(recent_errors),
                "error_rate": len(recent_errors) / 60,  # per minute
                "error_types": dict(error_types),
                "error_levels": dict(error_levels),
                "top_errors": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
            }

            self.logger.logger.info("Monitoring report", **report)

    def get_metrics(self) -> dict[str, Any]:
        """Get current error metrics"""
        return {
            "error_count": self.metrics.error_count,
            "error_rate": self.metrics.error_rate,
            "last_error_time": self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
            "error_types": self.metrics.error_types,
            "error_levels": self.metrics.error_levels,
            "buffer_size": len(self.error_buffer),
            "monitoring_active": self.running
        }

    def get_error_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get error history for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            {
                "error_id": e["error_id"],
                "error_code": e["error_code"],
                "error_level": e["error_level"],
                "message": e["message"],
                "timestamp": e["timestamp"].isoformat(),
                "request_info": e["request_info"]
            }
            for e in self.error_buffer
            if e["timestamp"] > cutoff_time
        ]

    def get_error_trends(self, hours: int = 24) -> dict[str, Any]:
        """Get error trends over time"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_buffer if e["timestamp"] > cutoff_time]

        if not recent_errors:
            return {"trend": "stable", "hourly_counts": [], "change_rate": 0}

        # Group errors by hour
        hourly_counts = defaultdict(int)
        for error in recent_errors:
            hour = error["timestamp"].replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour] += 1

        # Calculate trend
        hours_list = sorted(hourly_counts.keys())
        counts = [hourly_counts[h] for h in hours_list]

        if len(counts) < 2:
            trend = "stable"
            change_rate = 0
        else:
            recent_avg = sum(counts[-3:]) / min(3, len(counts))
            earlier_avg = sum(counts[:3]) / min(3, len(counts))

            if recent_avg > earlier_avg * 1.2:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"

            change_rate = ((recent_avg - earlier_avg) / earlier_avg) * 100 if earlier_avg > 0 else 0

        return {
            "trend": trend,
            "hourly_counts": [{"hour": h.isoformat(), "count": c} for h, c in hourly_counts.items()],
            "change_rate": change_rate
        }


# Global error monitor instance
_error_monitor: ErrorMonitor | None = None


def get_error_monitor() -> ErrorMonitor:
    """Get global error monitor instance"""
    global _error_monitor
    if _error_monitor is None:
        _error_monitor = ErrorMonitor()
    return _error_monitor


def start_error_monitoring():
    """Start global error monitoring"""
    monitor = get_error_monitor()
    monitor.start_monitoring()


def stop_error_monitoring():
    """Stop global error monitoring"""
    monitor = get_error_monitor()
    monitor.stop_monitoring()


def record_error_for_monitoring(error: WebUIError, request_info: dict[str, Any] | None = None):
    """Record error for monitoring"""
    monitor = get_error_monitor()
    monitor.record_error(error, request_info)
