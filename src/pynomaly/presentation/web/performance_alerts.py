"""Automated Performance Monitoring and Alerting System."""

import asyncio
import json
import logging
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Performance metric types."""
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_USAGE = "disk_usage"
    CORE_WEB_VITAL = "core_web_vital"
    API_LATENCY = "api_latency"
    DATABASE_PERFORMANCE = "database_performance"


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: dict[str, str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    evaluation_window: int = 300  # 5 minutes in seconds
    minimum_samples: int = 5
    comparison_operator: str = "gt"  # gt, lt, eq
    enabled: bool = True
    tags_filter: dict[str, str] = None

    def __post_init__(self):
        if self.tags_filter is None:
            self.tags_filter = {}


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    alert_id: str
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    current_value: float
    threshold_value: float
    triggered_at: datetime
    resolved_at: datetime | None = None
    tags: dict[str, str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}


class PerformanceMonitoringSystem:
    """Automated performance monitoring and alerting system."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or "config/monitoring/performance_alerts.json"
        self.metrics_buffer: list[PerformanceMetric] = []
        self.active_alerts: dict[str, PerformanceAlert] = {}
        self.alert_history: list[PerformanceAlert] = []
        self.thresholds: dict[str, AlertThreshold] = {}
        self.alert_handlers: list[Callable] = []
        self.monitoring_active = False
        self.monitoring_interval = 30  # Check every 30 seconds
        self.metrics_retention_hours = 24
        self.max_buffer_size = 10000

        # Load configuration
        self.load_configuration()

        # Setup default thresholds
        self.setup_default_thresholds()

        # Performance statistics
        self.performance_stats = {
            'total_alerts': 0,
            'alerts_by_severity': {severity.value: 0 for severity in AlertSeverity},
            'alerts_by_type': {metric.value: 0 for metric in MetricType},
            'average_resolution_time': 0
        }

    def load_configuration(self):
        """Load monitoring configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

                # Load thresholds
                if 'thresholds' in config:
                    for threshold_data in config['thresholds']:
                        threshold = AlertThreshold(**threshold_data)
                        self.thresholds[f"{threshold.metric_type.value}_{hash(str(threshold.tags_filter))}"] = threshold

                # Load other settings
                self.monitoring_interval = config.get('monitoring_interval', 30)
                self.metrics_retention_hours = config.get('metrics_retention_hours', 24)
                self.max_buffer_size = config.get('max_buffer_size', 10000)

                logger.info(f"Loaded monitoring configuration from {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to load monitoring configuration: {e}")

    def setup_default_thresholds(self):
        """Setup default performance thresholds."""
        default_thresholds = [
            AlertThreshold(
                metric_type=MetricType.RESPONSE_TIME,
                warning_threshold=1000,  # 1 second
                critical_threshold=3000,  # 3 seconds
                evaluation_window=300,
                minimum_samples=10
            ),
            AlertThreshold(
                metric_type=MetricType.ERROR_RATE,
                warning_threshold=0.05,  # 5%
                critical_threshold=0.10,  # 10%
                evaluation_window=300,
                minimum_samples=10
            ),
            AlertThreshold(
                metric_type=MetricType.MEMORY_USAGE,
                warning_threshold=0.80,  # 80%
                critical_threshold=0.90,  # 90%
                evaluation_window=300,
                minimum_samples=5
            ),
            AlertThreshold(
                metric_type=MetricType.CPU_USAGE,
                warning_threshold=0.80,  # 80%
                critical_threshold=0.90,  # 90%
                evaluation_window=300,
                minimum_samples=5
            ),
            AlertThreshold(
                metric_type=MetricType.CORE_WEB_VITAL,
                warning_threshold=2500,  # LCP threshold
                critical_threshold=4000,
                evaluation_window=300,
                minimum_samples=5,
                tags_filter={"vital": "LCP"}
            ),
            AlertThreshold(
                metric_type=MetricType.CORE_WEB_VITAL,
                warning_threshold=100,  # FID threshold
                critical_threshold=300,
                evaluation_window=300,
                minimum_samples=5,
                tags_filter={"vital": "FID"}
            ),
            AlertThreshold(
                metric_type=MetricType.CORE_WEB_VITAL,
                warning_threshold=0.1,  # CLS threshold
                critical_threshold=0.25,
                evaluation_window=300,
                minimum_samples=5,
                tags_filter={"vital": "CLS"}
            ),
            AlertThreshold(
                metric_type=MetricType.API_LATENCY,
                warning_threshold=500,  # 500ms
                critical_threshold=1000,  # 1 second
                evaluation_window=300,
                minimum_samples=10
            ),
            AlertThreshold(
                metric_type=MetricType.DATABASE_PERFORMANCE,
                warning_threshold=200,  # 200ms
                critical_threshold=500,  # 500ms
                evaluation_window=300,
                minimum_samples=10
            )
        ]

        for threshold in default_thresholds:
            key = f"{threshold.metric_type.value}_{hash(str(threshold.tags_filter))}"
            if key not in self.thresholds:
                self.thresholds[key] = threshold

    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)

    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        self.metrics_buffer.append(metric)

        # Maintain buffer size
        if len(self.metrics_buffer) > self.max_buffer_size:
            self.metrics_buffer = self.metrics_buffer[-self.max_buffer_size:]

        # Clean old metrics
        self.cleanup_old_metrics()

    def cleanup_old_metrics(self):
        """Remove old metrics from buffer."""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        self.metrics_buffer = [
            metric for metric in self.metrics_buffer
            if metric.timestamp > cutoff_time
        ]

    def get_metrics_for_evaluation(self, threshold: AlertThreshold) -> list[PerformanceMetric]:
        """Get metrics for threshold evaluation."""
        cutoff_time = datetime.now() - timedelta(seconds=threshold.evaluation_window)

        # Filter metrics by type, time, and tags
        filtered_metrics = []
        for metric in self.metrics_buffer:
            if (metric.metric_type == threshold.metric_type and
                metric.timestamp > cutoff_time):

                # Check tags filter
                if threshold.tags_filter:
                    if all(metric.tags.get(k) == v for k, v in threshold.tags_filter.items()):
                        filtered_metrics.append(metric)
                else:
                    filtered_metrics.append(metric)

        return filtered_metrics

    def evaluate_threshold(self, threshold: AlertThreshold) -> PerformanceAlert | None:
        """Evaluate a threshold and return alert if triggered."""
        if not threshold.enabled:
            return None

        metrics = self.get_metrics_for_evaluation(threshold)

        if len(metrics) < threshold.minimum_samples:
            return None

        # Calculate metric value based on comparison operator
        values = [metric.value for metric in metrics]

        if threshold.comparison_operator == "gt":
            current_value = statistics.mean(values)
            warning_triggered = current_value > threshold.warning_threshold
            critical_triggered = current_value > threshold.critical_threshold
        elif threshold.comparison_operator == "lt":
            current_value = statistics.mean(values)
            warning_triggered = current_value < threshold.warning_threshold
            critical_triggered = current_value < threshold.critical_threshold
        else:  # eq
            current_value = statistics.mean(values)
            warning_triggered = abs(current_value - threshold.warning_threshold) < 0.01
            critical_triggered = abs(current_value - threshold.critical_threshold) < 0.01

        # Determine alert severity
        if critical_triggered:
            severity = AlertSeverity.CRITICAL
            threshold_value = threshold.critical_threshold
        elif warning_triggered:
            severity = AlertSeverity.MEDIUM
            threshold_value = threshold.warning_threshold
        else:
            return None

        # Create alert
        alert_id = f"{threshold.metric_type.value}_{hash(str(threshold.tags_filter))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        alert = PerformanceAlert(
            alert_id=alert_id,
            severity=severity,
            metric_type=threshold.metric_type,
            message=self.generate_alert_message(threshold, current_value, threshold_value, severity),
            current_value=current_value,
            threshold_value=threshold_value,
            triggered_at=datetime.now(),
            tags=threshold.tags_filter.copy(),
            metadata={
                'samples_count': len(metrics),
                'evaluation_window': threshold.evaluation_window,
                'values_min': min(values),
                'values_max': max(values),
                'values_std': statistics.stdev(values) if len(values) > 1 else 0
            }
        )

        return alert

    def generate_alert_message(self, threshold: AlertThreshold, current_value: float,
                             threshold_value: float, severity: AlertSeverity) -> str:
        """Generate alert message."""
        metric_name = threshold.metric_type.value.replace('_', ' ').title()

        # Format values based on metric type
        if threshold.metric_type in [MetricType.RESPONSE_TIME, MetricType.API_LATENCY, MetricType.DATABASE_PERFORMANCE]:
            current_str = f"{current_value:.0f}ms"
            threshold_str = f"{threshold_value:.0f}ms"
        elif threshold.metric_type == MetricType.ERROR_RATE:
            current_str = f"{current_value:.2%}"
            threshold_str = f"{threshold_value:.2%}"
        elif threshold.metric_type in [MetricType.MEMORY_USAGE, MetricType.CPU_USAGE]:
            current_str = f"{current_value:.1%}"
            threshold_str = f"{threshold_value:.1%}"
        else:
            current_str = f"{current_value:.2f}"
            threshold_str = f"{threshold_value:.2f}"

        # Add tags info
        tags_str = ""
        if threshold.tags_filter:
            tags_str = f" ({', '.join(f'{k}={v}' for k, v in threshold.tags_filter.items())})"

        return f"{severity.value.upper()} - {metric_name}{tags_str}: {current_str} exceeds threshold {threshold_str}"

    def process_alert(self, alert: PerformanceAlert):
        """Process a new alert."""
        # Check if similar alert is already active
        similar_alert_key = f"{alert.metric_type.value}_{hash(str(alert.tags))}"

        if similar_alert_key in self.active_alerts:
            # Update existing alert
            existing_alert = self.active_alerts[similar_alert_key]
            existing_alert.current_value = alert.current_value
            existing_alert.triggered_at = alert.triggered_at
            existing_alert.metadata.update(alert.metadata)

            # Escalate if severity increased
            if alert.severity.value != existing_alert.severity.value:
                existing_alert.severity = alert.severity
                existing_alert.message = alert.message
                self.notify_alert_handlers(existing_alert)
        else:
            # New alert
            self.active_alerts[similar_alert_key] = alert
            self.alert_history.append(alert)

            # Update statistics
            self.performance_stats['total_alerts'] += 1
            self.performance_stats['alerts_by_severity'][alert.severity.value] += 1
            self.performance_stats['alerts_by_type'][alert.metric_type.value] += 1

            # Notify handlers
            self.notify_alert_handlers(alert)

    def notify_alert_handlers(self, alert: PerformanceAlert):
        """Notify all alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def resolve_alert(self, alert_key: str):
        """Resolve an active alert."""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved_at = datetime.now()

            # Update resolution time statistics
            resolution_time = (alert.resolved_at - alert.triggered_at).total_seconds()
            current_avg = self.performance_stats['average_resolution_time']
            total_resolved = sum(1 for a in self.alert_history if a.resolved_at)

            if total_resolved > 0:
                self.performance_stats['average_resolution_time'] = (
                    (current_avg * (total_resolved - 1) + resolution_time) / total_resolved
                )

            # Remove from active alerts
            del self.active_alerts[alert_key]

            logger.info(f"Alert resolved: {alert.alert_id}")

    def check_alert_resolution(self):
        """Check if any active alerts should be resolved."""
        for alert_key, alert in list(self.active_alerts.items()):
            # Find corresponding threshold
            threshold_key = f"{alert.metric_type.value}_{hash(str(alert.tags))}"
            threshold = self.thresholds.get(threshold_key)

            if not threshold:
                continue

            # Get recent metrics
            metrics = self.get_metrics_for_evaluation(threshold)

            if len(metrics) < threshold.minimum_samples:
                continue

            # Check if metric is back to normal
            values = [metric.value for metric in metrics]
            current_value = statistics.mean(values)

            is_resolved = False
            if threshold.comparison_operator == "gt":
                is_resolved = current_value <= threshold.warning_threshold
            elif threshold.comparison_operator == "lt":
                is_resolved = current_value >= threshold.warning_threshold
            else:  # eq
                is_resolved = abs(current_value - threshold.warning_threshold) >= 0.01

            if is_resolved:
                self.resolve_alert(alert_key)

    async def monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Evaluate all thresholds
                for threshold in self.thresholds.values():
                    alert = self.evaluate_threshold(threshold)
                    if alert:
                        self.process_alert(alert)

                # Check for alert resolution
                self.check_alert_resolution()

                # Clean up old metrics
                self.cleanup_old_metrics()

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.monitoring_interval)

    def start_monitoring(self):
        """Start the monitoring system."""
        if not self.monitoring_active:
            self.monitoring_active = True
            asyncio.create_task(self.monitoring_loop())
            logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")

    def get_active_alerts(self) -> list[PerformanceAlert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> list[PerformanceAlert]:
        """Get alert history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.triggered_at > cutoff_time]

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance monitoring statistics."""
        return {
            **self.performance_stats,
            'active_alerts_count': len(self.active_alerts),
            'monitoring_active': self.monitoring_active,
            'metrics_buffer_size': len(self.metrics_buffer),
            'thresholds_count': len(self.thresholds)
        }

    def add_threshold(self, threshold: AlertThreshold):
        """Add a new threshold."""
        key = f"{threshold.metric_type.value}_{hash(str(threshold.tags_filter))}"
        self.thresholds[key] = threshold

    def update_threshold(self, threshold_key: str, threshold: AlertThreshold):
        """Update an existing threshold."""
        self.thresholds[threshold_key] = threshold

    def remove_threshold(self, threshold_key: str):
        """Remove a threshold."""
        if threshold_key in self.thresholds:
            del self.thresholds[threshold_key]

    def export_configuration(self, file_path: str):
        """Export current configuration to file."""
        config = {
            'monitoring_interval': self.monitoring_interval,
            'metrics_retention_hours': self.metrics_retention_hours,
            'max_buffer_size': self.max_buffer_size,
            'thresholds': [asdict(threshold) for threshold in self.thresholds.values()]
        }

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        logger.info(f"Configuration exported to {file_path}")


# Default alert handlers
def console_alert_handler(alert: PerformanceAlert):
    """Console alert handler."""
    print(f"[{alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}] {alert.message}")


def email_alert_handler(alert: PerformanceAlert):
    """Email alert handler (implementation needed)."""
    # This would integrate with email service
    logger.info(f"Email alert: {alert.message}")


def slack_alert_handler(alert: PerformanceAlert):
    """Slack alert handler (implementation needed)."""
    # This would integrate with Slack API
    logger.info(f"Slack alert: {alert.message}")


def webhook_alert_handler(alert: PerformanceAlert):
    """Webhook alert handler (implementation needed)."""
    # This would send HTTP POST to webhook URL
    logger.info(f"Webhook alert: {alert.message}")


# Global monitoring instance
performance_monitor = PerformanceMonitoringSystem()

# Add default handlers
performance_monitor.add_alert_handler(console_alert_handler)
performance_monitor.add_alert_handler(email_alert_handler)
performance_monitor.add_alert_handler(slack_alert_handler)
performance_monitor.add_alert_handler(webhook_alert_handler)
