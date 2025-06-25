"""Enterprise monitoring and alerting for autonomous anomaly detection.

This module provides comprehensive monitoring, metrics collection, and alerting
for autonomous detection systems in production environments.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Metrics and monitoring imports
try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to collect."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""

    name: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 10", "== 0"
    threshold: float
    severity: AlertSeverity
    description: str
    cooldown_seconds: int = 300  # 5 minutes default cooldown


@dataclass
class Alert:
    """An active alert."""

    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_value: float
    resolved: bool = False
    resolved_timestamp: datetime | None = None


@dataclass
class PerformanceMetric:
    """Performance metric with metadata."""

    name: str
    value: int | float
    unit: str
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AutonomousDetectionMonitor:
    """Comprehensive monitoring system for autonomous detection."""

    def __init__(
        self,
        enable_prometheus: bool = True,
        enable_logging: bool = True,
        metrics_retention_hours: int = 24,
        alert_cooldown_seconds: int = 300,
    ):
        """Initialize monitoring system.

        Args:
            enable_prometheus: Enable Prometheus metrics collection
            enable_logging: Enable detailed logging
            metrics_retention_hours: How long to retain metrics in memory
            alert_cooldown_seconds: Default cooldown between alerts
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_logging = enable_logging
        self.metrics_retention_hours = metrics_retention_hours
        self.alert_cooldown_seconds = alert_cooldown_seconds

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if enable_logging:
            self._setup_logging()

        # Metrics storage
        self.metrics_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self.current_metrics: dict[str, PerformanceMetric] = {}

        # Alert management
        self.alert_rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.last_alert_times: dict[str, datetime] = {}

        # Performance tracking
        self.detection_stats = {
            "total_detections": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "total_execution_time": 0.0,
            "algorithm_usage": defaultdict(int),
            "dataset_sizes_processed": [],
        }

        # Prometheus metrics (if available)
        if self.enable_prometheus:
            self._setup_prometheus_metrics()

        # System resource monitoring
        self.system_monitor_active = False
        self.system_monitor_thread = None

        self.logger.info("Autonomous Detection Monitor initialized")

    def _setup_logging(self):
        """Setup structured logging for monitoring."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors."""
        self.registry = CollectorRegistry()

        # Core detection metrics
        self.detection_counter = Counter(
            "autonomous_detections_total",
            "Total number of autonomous detections",
            ["status", "algorithm"],
            registry=self.registry,
        )

        self.detection_duration = Histogram(
            "autonomous_detection_duration_seconds",
            "Time spent on autonomous detection",
            ["algorithm"],
            registry=self.registry,
        )

        self.anomalies_found = Histogram(
            "autonomous_anomalies_found",
            "Number of anomalies found per detection",
            ["algorithm"],
            registry=self.registry,
        )

        self.algorithm_selection_accuracy = Gauge(
            "autonomous_algorithm_selection_accuracy",
            "Accuracy of algorithm selection",
            registry=self.registry,
        )

        # System resource metrics
        self.memory_usage = Gauge(
            "autonomous_memory_usage_bytes",
            "Memory usage during detection",
            registry=self.registry,
        )

        self.cpu_usage = Gauge(
            "autonomous_cpu_usage_percent",
            "CPU usage during detection",
            registry=self.registry,
        )

        # Data processing metrics
        self.dataset_size = Histogram(
            "autonomous_dataset_size_rows",
            "Size of datasets processed",
            registry=self.registry,
        )

        self.preprocessing_time = Histogram(
            "autonomous_preprocessing_duration_seconds",
            "Time spent on data preprocessing",
            registry=self.registry,
        )

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule to the monitoring system."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")

    def record_metric(
        self,
        name: str,
        value: int | float,
        unit: str = "",
        labels: dict[str, str] = None,
        metadata: dict[str, Any] = None,
    ):
        """Record a custom metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            labels=labels or {},
            metadata=metadata or {},
        )

        self.current_metrics[name] = metric
        self.metrics_history[name].append(metric)

        # Check alert rules
        self._check_alert_rules(name, value)

        self.logger.debug(f"Recorded metric: {name} = {value} {unit}")

    def record_detection_start(
        self, dataset_info: dict[str, Any], config: dict[str, Any]
    ) -> str:
        """Record the start of an autonomous detection."""
        detection_id = f"detection_{int(time.time() * 1000)}"

        self.detection_stats["total_detections"] += 1

        # Record dataset size
        dataset_size = dataset_info.get("n_samples", 0)
        self.detection_stats["dataset_sizes_processed"].append(dataset_size)

        if self.enable_prometheus:
            self.dataset_size.observe(dataset_size)

        self.record_metric(
            "detection_started",
            1,
            "count",
            labels={"detection_id": detection_id},
            metadata={"dataset_info": dataset_info, "config": config},
        )

        self.logger.info(f"Detection started: {detection_id}")
        return detection_id

    def record_detection_end(
        self,
        detection_id: str,
        success: bool,
        execution_time: float,
        algorithms_used: list[str],
        anomalies_found: int,
        best_algorithm: str = None,
        error: str = None,
    ):
        """Record the completion of an autonomous detection."""

        # Update stats
        if success:
            self.detection_stats["successful_detections"] += 1
        else:
            self.detection_stats["failed_detections"] += 1

        self.detection_stats["total_execution_time"] += execution_time

        # Update algorithm usage
        for algo in algorithms_used:
            self.detection_stats["algorithm_usage"][algo] += 1

        # Record metrics
        self.record_metric(
            "detection_duration",
            execution_time,
            "seconds",
            labels={
                "detection_id": detection_id,
                "status": "success" if success else "failure",
                "best_algorithm": best_algorithm or "unknown",
            },
        )

        self.record_metric(
            "anomalies_detected",
            anomalies_found,
            "count",
            labels={
                "detection_id": detection_id,
                "algorithm": best_algorithm or "unknown",
            },
        )

        # Prometheus metrics
        if self.enable_prometheus:
            status = "success" if success else "failure"
            for algo in algorithms_used:
                self.detection_counter.labels(status=status, algorithm=algo).inc()
                if success:
                    self.detection_duration.labels(algorithm=algo).observe(
                        execution_time
                    )
                    if algo == best_algorithm:
                        self.anomalies_found.labels(algorithm=algo).observe(
                            anomalies_found
                        )

        # Log completion
        if success:
            self.logger.info(
                f"Detection completed: {detection_id} in {execution_time:.2f}s, "
                f"{anomalies_found} anomalies found with {best_algorithm}"
            )
        else:
            self.logger.error(f"Detection failed: {detection_id}, error: {error}")

    def record_algorithm_selection(
        self,
        selected_algorithms: list[str],
        confidence_scores: dict[str, float],
        data_characteristics: dict[str, Any],
    ):
        """Record algorithm selection decision."""

        # Calculate selection quality metrics
        avg_confidence = (
            sum(confidence_scores.values()) / len(confidence_scores)
            if confidence_scores
            else 0
        )
        top_confidence = max(confidence_scores.values()) if confidence_scores else 0

        self.record_metric(
            "algorithm_selection_confidence",
            avg_confidence,
            "ratio",
            labels={
                "top_algorithm": selected_algorithms[0]
                if selected_algorithms
                else "none"
            },
        )

        self.record_metric(
            "algorithm_selection_top_confidence", top_confidence, "ratio"
        )

        if self.enable_prometheus:
            self.algorithm_selection_accuracy.set(avg_confidence)

        self.logger.info(
            f"Algorithm selection: {selected_algorithms[:3]} "
            f"(avg confidence: {avg_confidence:.2f})"
        )

    def record_preprocessing_metrics(
        self,
        preprocessing_time: float,
        quality_improvement: float,
        steps_applied: list[str],
    ):
        """Record data preprocessing metrics."""

        self.record_metric("preprocessing_duration", preprocessing_time, "seconds")

        self.record_metric("data_quality_improvement", quality_improvement, "ratio")

        self.record_metric("preprocessing_steps_count", len(steps_applied), "count")

        if self.enable_prometheus:
            self.preprocessing_time.observe(preprocessing_time)

        self.logger.info(
            f"Preprocessing completed in {preprocessing_time:.2f}s, "
            f"quality improved by {quality_improvement:.1%}"
        )

    def record_system_resources(self):
        """Record current system resource usage."""
        if not PSUTIL_AVAILABLE:
            return

        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("system_memory_usage", memory.percent, "percent")
            self.record_metric("system_memory_available", memory.available, "bytes")

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system_cpu_usage", cpu_percent, "percent")

            # Disk usage
            disk = psutil.disk_usage("/")
            self.record_metric("system_disk_usage", disk.percent, "percent")

            if self.enable_prometheus:
                self.memory_usage.set(memory.used)
                self.cpu_usage.set(cpu_percent)

        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")

    def _check_alert_rules(self, metric_name: str, value: float):
        """Check if any alert rules are triggered by the new metric value."""

        for rule_name, rule in self.alert_rules.items():
            if rule.metric_name != metric_name:
                continue

            # Check cooldown
            if rule_name in self.last_alert_times:
                time_since_last = datetime.now() - self.last_alert_times[rule_name]
                if time_since_last.total_seconds() < rule.cooldown_seconds:
                    continue

            # Evaluate condition
            triggered = self._evaluate_condition(value, rule.condition, rule.threshold)

            if triggered:
                self._trigger_alert(rule, value)

    def _evaluate_condition(
        self, value: float, condition: str, threshold: float
    ) -> bool:
        """Evaluate an alert condition."""
        try:
            if condition.startswith(">"):
                return value > threshold
            elif condition.startswith("<"):
                return value < threshold
            elif condition.startswith(">="):
                return value >= threshold
            elif condition.startswith("<="):
                return value <= threshold
            elif condition.startswith("=="):
                return abs(value - threshold) < 0.001  # Float equality
            elif condition.startswith("!="):
                return abs(value - threshold) >= 0.001
            else:
                self.logger.warning(f"Unknown condition operator: {condition}")
                return False
        except Exception as e:
            self.logger.error(f"Error evaluating condition {condition}: {e}")
            return False

    def _trigger_alert(self, rule: AlertRule, metric_value: float):
        """Trigger an alert."""
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=f"{rule.description}. Current value: {metric_value:.2f}",
            timestamp=datetime.now(),
            metric_value=metric_value,
        )

        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        self.last_alert_times[rule.name] = datetime.now()

        # Log alert
        self.logger.warning(
            f"ALERT [{rule.severity.value.upper()}] {rule.name}: {alert.message}"
        )

        # TODO: Implement external alerting (email, Slack, PagerDuty, etc.)
        self._send_external_alert(alert)

    def _send_external_alert(self, alert: Alert):
        """Send alert to external systems (placeholder for integration)."""
        # This would integrate with external alerting systems
        # For now, just log at appropriate level
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.ERROR:
            self.logger.error(f"ERROR ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            self.logger.warning(f"WARNING ALERT: {alert.message}")
        else:
            self.logger.info(f"INFO ALERT: {alert.message}")

    def resolve_alert(self, rule_name: str):
        """Mark an alert as resolved."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolved_timestamp = datetime.now()
            del self.active_alerts[rule_name]

            self.logger.info(f"Alert resolved: {rule_name}")

    def get_active_alerts(self) -> list[Alert]:
        """Get all currently active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> list[Alert]:
        """Get alert history for the specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff]

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics and system status."""

        # Calculate success rate
        total_detections = self.detection_stats["total_detections"]
        success_rate = (
            self.detection_stats["successful_detections"] / total_detections
            if total_detections > 0
            else 0
        )

        # Calculate average execution time
        avg_execution_time = self.detection_stats["total_execution_time"] / max(
            1, self.detection_stats["successful_detections"]
        )

        # Top algorithms by usage
        top_algorithms = sorted(
            self.detection_stats["algorithm_usage"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Recent metrics
        recent_metrics = {
            name: metric.value for name, metric in self.current_metrics.items()
        }

        return {
            "detection_stats": {
                "total_detections": total_detections,
                "success_rate": success_rate,
                "avg_execution_time_seconds": avg_execution_time,
                "top_algorithms": top_algorithms,
            },
            "system_status": {
                "active_alerts": len(self.active_alerts),
                "monitoring_enabled": True,
                "prometheus_enabled": self.enable_prometheus,
            },
            "recent_metrics": recent_metrics,
            "active_alerts": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in self.active_alerts.values()
            ],
        }

    def start_system_monitoring(self, interval_seconds: int = 60):
        """Start continuous system resource monitoring."""
        if self.system_monitor_active:
            return

        self.system_monitor_active = True

        def monitor_loop():
            while self.system_monitor_active:
                try:
                    self.record_system_resources()
                    time.sleep(interval_seconds)
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}")
                    time.sleep(interval_seconds)

        self.system_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.system_monitor_thread.start()

        self.logger.info(f"Started system monitoring (interval: {interval_seconds}s)")

    def stop_system_monitoring(self):
        """Stop continuous system resource monitoring."""
        self.system_monitor_active = False
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5)

        self.logger.info("Stopped system monitoring")

    def export_metrics_to_file(self, file_path: str, format: str = "json"):
        """Export metrics to file for analysis."""

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics_summary": self.get_metrics_summary(),
            "metrics_history": {
                name: [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "labels": m.labels,
                        "metadata": m.metadata,
                    }
                    for m in list(history)  # Convert deque to list
                ]
                for name, history in self.metrics_history.items()
            },
            "alert_history": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_value": alert.metric_value,
                    "resolved": alert.resolved,
                    "resolved_timestamp": (
                        alert.resolved_timestamp.isoformat()
                        if alert.resolved_timestamp
                        else None
                    ),
                }
                for alert in self.alert_history
            ],
        }

        if format.lower() == "json":
            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Metrics exported to {file_path}")


def create_default_alert_rules() -> list[AlertRule]:
    """Create a set of default alert rules for autonomous detection."""

    return [
        AlertRule(
            name="high_failure_rate",
            metric_name="detection_failure_rate",
            condition="> 0.2",
            threshold=0.2,
            severity=AlertSeverity.WARNING,
            description="Detection failure rate above 20%",
        ),
        AlertRule(
            name="slow_detection",
            metric_name="detection_duration",
            condition="> 300",
            threshold=300,
            severity=AlertSeverity.WARNING,
            description="Detection taking longer than 5 minutes",
        ),
        AlertRule(
            name="high_memory_usage",
            metric_name="system_memory_usage",
            condition="> 90",
            threshold=90,
            severity=AlertSeverity.ERROR,
            description="System memory usage above 90%",
        ),
        AlertRule(
            name="high_cpu_usage",
            metric_name="system_cpu_usage",
            condition="> 95",
            threshold=95,
            severity=AlertSeverity.WARNING,
            description="System CPU usage above 95%",
        ),
        AlertRule(
            name="low_algorithm_confidence",
            metric_name="algorithm_selection_confidence",
            condition="< 0.5",
            threshold=0.5,
            severity=AlertSeverity.INFO,
            description="Low confidence in algorithm selection",
        ),
        AlertRule(
            name="no_anomalies_found",
            metric_name="anomalies_detected",
            condition="== 0",
            threshold=0,
            severity=AlertSeverity.INFO,
            description="No anomalies detected in recent runs",
        ),
    ]


# Global monitor instance (singleton pattern)
_global_monitor: AutonomousDetectionMonitor | None = None


def get_monitor() -> AutonomousDetectionMonitor:
    """Get or create the global monitor instance."""
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = AutonomousDetectionMonitor()

        # Add default alert rules
        default_rules = create_default_alert_rules()
        for rule in default_rules:
            _global_monitor.add_alert_rule(rule)

    return _global_monitor


def initialize_monitoring(
    enable_prometheus: bool = True,
    enable_system_monitoring: bool = True,
    system_monitor_interval: int = 60,
) -> AutonomousDetectionMonitor:
    """Initialize the global monitoring system."""
    global _global_monitor

    _global_monitor = AutonomousDetectionMonitor(
        enable_prometheus=enable_prometheus, enable_logging=True
    )

    # Add default alert rules
    default_rules = create_default_alert_rules()
    for rule in default_rules:
        _global_monitor.add_alert_rule(rule)

    # Start system monitoring if requested
    if enable_system_monitoring:
        _global_monitor.start_system_monitoring(system_monitor_interval)

    return _global_monitor
