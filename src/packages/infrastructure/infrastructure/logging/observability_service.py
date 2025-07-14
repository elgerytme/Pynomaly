"""Comprehensive observability service that orchestrates logging, metrics, and tracing."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .log_aggregator import LogAggregator, LogEntry, LogFilter, LogStreamType
from .log_analysis import AnomalyDetector, LogAnalyzer, LogPattern, PatternRule
from .metrics_collector import MetricsCollector
from .structured_logger import LogLevel, StructuredLogger
from .tracing_manager import TracingManager


@dataclass
class ObservabilityConfig:
    """Configuration for observability service."""

    # Storage paths
    logs_storage_path: Path | None = None
    metrics_storage_path: Path | None = None
    traces_storage_path: Path | None = None

    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    enable_console_logging: bool = True
    enable_json_logging: bool = True
    max_log_file_size_mb: int = 100
    log_backup_count: int = 5

    # Metrics configuration
    enable_system_metrics: bool = True
    metrics_flush_interval: int = 60
    max_metrics_in_memory: int = 10000

    # Tracing configuration
    enable_tracing: bool = True
    trace_sampling_rate: float = 0.1
    jaeger_endpoint: str | None = None

    # Analysis configuration
    enable_log_analysis: bool = True
    analysis_interval: int = 60
    enable_anomaly_detection: bool = True
    anomaly_sensitivity: float = 2.0

    # Aggregation configuration
    enable_log_aggregation: bool = True
    aggregation_interval: int = 300
    max_log_streams: int = 100

    # Performance settings
    enable_background_tasks: bool = True
    max_workers: int = 4

    # Alerting
    enable_alerts: bool = False
    alert_webhook_url: str | None = None
    critical_pattern_alert: bool = True


@dataclass
class ObservabilityMetrics:
    """Comprehensive metrics for observability service."""

    # Service metrics
    service_uptime: float = 0.0
    total_log_entries: int = 0
    total_metrics_collected: int = 0
    total_traces_created: int = 0

    # Component status
    logger_status: str = "unknown"
    metrics_collector_status: str = "unknown"
    tracer_status: str = "unknown"
    analyzer_status: str = "unknown"
    aggregator_status: str = "unknown"

    # Performance metrics
    log_processing_rate: float = 0.0
    metrics_processing_rate: float = 0.0
    average_analysis_time: float = 0.0

    # Pattern detection
    patterns_detected: int = 0
    anomalies_detected: int = 0
    critical_alerts: int = 0

    # Error counts
    logging_errors: int = 0
    metrics_errors: int = 0
    tracing_errors: int = 0
    analysis_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_uptime": self.service_uptime,
            "total_log_entries": self.total_log_entries,
            "total_metrics_collected": self.total_metrics_collected,
            "total_traces_created": self.total_traces_created,
            "logger_status": self.logger_status,
            "metrics_collector_status": self.metrics_collector_status,
            "tracer_status": self.tracer_status,
            "analyzer_status": self.analyzer_status,
            "aggregator_status": self.aggregator_status,
            "log_processing_rate": self.log_processing_rate,
            "metrics_processing_rate": self.metrics_processing_rate,
            "average_analysis_time": self.average_analysis_time,
            "patterns_detected": self.patterns_detected,
            "anomalies_detected": self.anomalies_detected,
            "critical_alerts": self.critical_alerts,
            "logging_errors": self.logging_errors,
            "metrics_errors": self.metrics_errors,
            "tracing_errors": self.tracing_errors,
            "analysis_errors": self.analysis_errors,
        }


class ObservabilityService:
    """Comprehensive observability service that orchestrates all monitoring components."""

    def __init__(self, config: ObservabilityConfig):
        """Initialize observability service.

        Args:
            config: Observability configuration
        """
        self.config = config
        self.start_time = time.time()

        # Core components
        self.logger: StructuredLogger | None = None
        self.metrics_collector: MetricsCollector | None = None
        self.tracer: TracingManager | None = None
        self.log_analyzer: LogAnalyzer | None = None
        self.log_aggregator: LogAggregator | None = None
        self.anomaly_detector: AnomalyDetector | None = None

        # Service state
        self._initialized = False
        self._running = False
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()

        # Metrics
        self.metrics = ObservabilityMetrics()

        # Background tasks
        self._monitoring_thread: threading.Thread | None = None
        self._alert_callbacks: list[Callable[[dict[str, Any]], None]] = []

        # Performance tracking
        self._log_count_last_minute = 0
        self._metrics_count_last_minute = 0
        self._last_minute_timestamp = time.time()

    def initialize(self):
        """Initialize all observability components."""
        with self._lock:
            if self._initialized:
                return

            try:
                self._initialize_logger()
                self._initialize_metrics_collector()
                self._initialize_tracer()
                self._initialize_log_aggregator()
                self._initialize_log_analyzer()
                self._initialize_anomaly_detector()

                # Start background monitoring
                if self.config.enable_background_tasks:
                    self._start_monitoring_thread()

                self._initialized = True
                self._running = True

                if self.logger:
                    self.logger.info("Observability service initialized successfully")

            except Exception as e:
                if self.logger:
                    self.logger.error(
                        "Failed to initialize observability service", error=e
                    )
                raise

    def _initialize_logger(self):
        """Initialize structured logger."""
        try:
            self.logger = StructuredLogger(
                name="pynomaly.observability",
                level=self.config.log_level,
                output_path=(
                    self.config.logs_storage_path / "observability.log"
                    if self.config.logs_storage_path
                    else None
                ),
                enable_console=self.config.enable_console_logging,
                enable_json=self.config.enable_json_logging,
                max_file_size_mb=self.config.max_log_file_size_mb,
                backup_count=self.config.log_backup_count,
            )
            self.metrics.logger_status = "active"
        except Exception:
            self.metrics.logger_status = "error"
            self.metrics.logging_errors += 1
            raise

    def _initialize_metrics_collector(self):
        """Initialize metrics collector."""
        try:
            self.metrics_collector = MetricsCollector(
                storage_path=self.config.metrics_storage_path,
                enable_system_metrics=self.config.enable_system_metrics,
                flush_interval_seconds=self.config.metrics_flush_interval,
                max_metrics_in_memory=self.config.max_metrics_in_memory,
            )
            self.metrics.metrics_collector_status = "active"
        except Exception as e:
            self.metrics.metrics_collector_status = "error"
            self.metrics.metrics_errors += 1
            if self.logger:
                self.logger.error("Failed to initialize metrics collector", error=e)
            raise

    def _initialize_tracer(self):
        """Initialize tracing manager."""
        if not self.config.enable_tracing:
            self.metrics.tracer_status = "disabled"
            return

        try:
            self.tracer = TracingManager(
                service_name="pynomaly",
                sampling_rate=self.config.trace_sampling_rate,
                jaeger_endpoint=self.config.jaeger_endpoint,
            )
            self.metrics.tracer_status = "active"
        except Exception as e:
            self.metrics.tracer_status = "error"
            self.metrics.tracing_errors += 1
            if self.logger:
                self.logger.error("Failed to initialize tracer", error=e)
            # Don't raise - tracing is optional

    def _initialize_log_aggregator(self):
        """Initialize log aggregator."""
        if not self.config.enable_log_aggregation:
            self.metrics.aggregator_status = "disabled"
            return

        try:
            self.log_aggregator = LogAggregator(
                storage_path=self.config.logs_storage_path,
                max_streams=self.config.max_log_streams,
                aggregation_interval=self.config.aggregation_interval,
                enable_persistence=self.config.logs_storage_path is not None,
            )

            # Create default streams
            self._create_default_log_streams()
            self.metrics.aggregator_status = "active"
        except Exception as e:
            self.metrics.aggregator_status = "error"
            if self.logger:
                self.logger.error("Failed to initialize log aggregator", error=e)
            # Don't raise - aggregation is optional

    def _initialize_log_analyzer(self):
        """Initialize log analyzer."""
        if not self.config.enable_log_analysis:
            self.metrics.analyzer_status = "disabled"
            return

        try:
            self.log_analyzer = LogAnalyzer(
                analysis_interval=self.config.analysis_interval,
                enable_realtime=True,
                enable_background_analysis=self.config.enable_background_tasks,
            )
            self.metrics.analyzer_status = "active"
        except Exception as e:
            self.metrics.analyzer_status = "error"
            self.metrics.analysis_errors += 1
            if self.logger:
                self.logger.error("Failed to initialize log analyzer", error=e)
            # Don't raise - analysis is optional

    def _initialize_anomaly_detector(self):
        """Initialize anomaly detector."""
        if not self.config.enable_anomaly_detection:
            return

        try:
            self.anomaly_detector = AnomalyDetector(
                sensitivity=self.config.anomaly_sensitivity
            )
        except Exception as e:
            if self.logger:
                self.logger.error("Failed to initialize anomaly detector", error=e)

    def _create_default_log_streams(self):
        """Create default log streams."""
        if not self.log_aggregator:
            return

        default_streams = [
            ("errors", LogStreamType.REALTIME, LogFilter(levels=["ERROR", "CRITICAL"])),
            ("warnings", LogStreamType.REALTIME, LogFilter(levels=["WARNING"])),
            ("security", LogStreamType.REALTIME, LogFilter(tags=["security"])),
            ("performance", LogStreamType.BATCH, LogFilter(tags=["performance"])),
            ("audit", LogStreamType.BATCH, LogFilter(tags=["audit"])),
            ("all_logs", LogStreamType.BATCH, LogFilter()),
        ]

        for name, stream_type, filter_config in default_streams:
            try:
                self.log_aggregator.create_stream(name, stream_type, filter_config)
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"Failed to create default stream '{name}'", error=e
                    )

    def _start_monitoring_thread(self):
        """Start background monitoring thread."""

        def monitoring_worker():
            while not self._shutdown_event.wait(30):  # Check every 30 seconds
                try:
                    self._update_service_metrics()
                    self._check_for_alerts()
                except Exception as e:
                    if self.logger:
                        self.logger.error("Error in monitoring thread", error=e)

        self._monitoring_thread = threading.Thread(
            target=monitoring_worker, daemon=True
        )
        self._monitoring_thread.start()

    def _update_service_metrics(self):
        """Update service-level metrics."""
        current_time = time.time()

        # Update uptime
        self.metrics.service_uptime = current_time - self.start_time

        # Update processing rates
        time_since_last = current_time - self._last_minute_timestamp
        if time_since_last >= 60:  # Update every minute
            self.metrics.log_processing_rate = (
                self._log_count_last_minute / time_since_last
            )
            self.metrics.metrics_processing_rate = (
                self._metrics_count_last_minute / time_since_last
            )

            # Reset counters
            self._log_count_last_minute = 0
            self._metrics_count_last_minute = 0
            self._last_minute_timestamp = current_time

        # Collect metrics from components
        if self.metrics_collector:
            collector_stats = self.metrics_collector.get_stats()
            self.metrics.total_metrics_collected = collector_stats.get(
                "metrics_collected", 0
            )
            self.metrics.metrics_errors += collector_stats.get("collection_errors", 0)

        if self.log_analyzer:
            analyzer_stats = self.log_analyzer.get_stats()
            self.metrics.patterns_detected = analyzer_stats["analyzer_stats"][
                "patterns_detected"
            ]
            self.metrics.analysis_errors += analyzer_stats["analyzer_stats"][
                "analysis_errors"
            ]

        if self.tracer:
            tracer_stats = self.tracer.get_stats()
            self.metrics.total_traces_created = tracer_stats.get("spans_created", 0)

    def _check_for_alerts(self):
        """Check for alert conditions."""
        if not self.config.enable_alerts:
            return

        alerts = []

        # Check for critical patterns
        if self.log_analyzer and self.config.critical_pattern_alert:
            critical_patterns = self.log_analyzer.get_patterns(
                severity=None,  # Will be filtered below
                since=datetime.utcnow() - timedelta(minutes=5),
            )

            for pattern in critical_patterns:
                if pattern.severity.value in ["high", "critical"]:
                    alerts.append(
                        {
                            "type": "critical_pattern",
                            "severity": pattern.severity.value,
                            "title": pattern.title,
                            "description": pattern.description,
                            "pattern_id": pattern.id,
                            "occurrence_count": pattern.occurrence_count,
                            "timestamp": pattern.last_occurrence.isoformat(),
                        }
                    )

        # Check for anomalies
        if self.anomaly_detector:
            anomalies = self.anomaly_detector.detect_anomalies()
            for anomaly in anomalies:
                if anomaly.get("severity") in ["high", "critical"]:
                    alerts.append(
                        {
                            "type": "anomaly",
                            "severity": anomaly["severity"],
                            "title": f"Anomaly: {anomaly['metric']}",
                            "description": f"Detected statistical anomaly in {anomaly['metric']} (z-score: {anomaly['z_score']:.2f})",
                            "metric_id": anomaly["metric_id"],
                            "value": anomaly["value"],
                            "z_score": anomaly["z_score"],
                            "timestamp": anomaly["timestamp"],
                        }
                    )

        # Process alerts
        for alert in alerts:
            self._process_alert(alert)

    def _process_alert(self, alert: dict[str, Any]):
        """Process an alert."""
        self.metrics.critical_alerts += 1

        if self.logger:
            self.logger.critical(
                f"ALERT: {alert['title']}",
                alert_type=alert["type"],
                severity=alert["severity"],
                description=alert["description"],
            )

        # Notify alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                if self.logger:
                    self.logger.error("Error in alert callback", error=e)

        # Send webhook if configured
        if self.config.alert_webhook_url:
            self._send_alert_webhook(alert)

    def _send_alert_webhook(self, alert: dict[str, Any]):
        """Send alert to webhook URL."""
        # This would typically use requests or httpx
        # For now, just log the intent
        if self.logger:
            self.logger.info(
                "Alert webhook triggered",
                webhook_url=self.config.alert_webhook_url,
                alert_type=alert["type"],
                severity=alert["severity"],
            )

    # Public API methods

    def log(self, level: str, message: str, **kwargs):
        """Log a message through the observability service."""
        if not self.logger:
            return

        self._log_count_last_minute += 1
        self.metrics.total_log_entries += 1

        # Add observability context
        kwargs.update(
            {
                "observability_service": True,
                "service_uptime": self.metrics.service_uptime,
            }
        )

        # Log through structured logger
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, **kwargs)

        # Create log entry for analysis
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level.upper(),
            logger_name="pynomaly.observability",
            message=message,
            context=kwargs,
            tags=kwargs.get("tags", []),
        )

        # Feed to aggregator and analyzer
        if self.log_aggregator:
            self.log_aggregator.add_log_entry(log_entry)

        if self.log_analyzer:
            self.log_analyzer.analyze_entry(log_entry)

    def record_metric(
        self, name: str, value: float, metric_type: str = "gauge", **labels
    ):
        """Record a metric."""
        if not self.metrics_collector:
            return

        self._metrics_count_last_minute += 1

        if metric_type == "counter":
            self.metrics_collector.counter(name, value, labels)
        elif metric_type == "gauge":
            self.metrics_collector.gauge(name, value, labels)
        elif metric_type == "histogram":
            self.metrics_collector.histogram(name, value, labels)
        elif metric_type == "timer":
            self.metrics_collector.timer(name, value, labels)

    def start_trace(self, operation_name: str, **attributes):
        """Start a new trace span."""
        if not self.tracer:
            return None

        return self.tracer.start_span(operation_name, **attributes)

    def add_pattern_rule(self, rule: PatternRule):
        """Add a custom pattern detection rule."""
        if self.log_analyzer:
            self.log_analyzer.add_rule(rule)

    def get_patterns(self, **filters) -> list[LogPattern]:
        """Get detected patterns."""
        if not self.log_analyzer:
            return []

        return self.log_analyzer.get_patterns(**filters)

    def add_alert_callback(self, callback: Callable[[dict[str, Any]], None]):
        """Add alert callback function."""
        self._alert_callbacks.append(callback)

    def get_service_metrics(self) -> dict[str, Any]:
        """Get comprehensive service metrics."""
        self._update_service_metrics()

        result = self.metrics.to_dict()

        # Add component metrics
        if self.metrics_collector:
            result["metrics_collector"] = self.metrics_collector.get_stats()

        if self.log_analyzer:
            result["log_analyzer"] = self.log_analyzer.get_stats()

        if self.log_aggregator:
            result["log_aggregator"] = self.log_aggregator.get_stats()

        if self.tracer:
            result["tracer"] = self.tracer.get_stats()

        if self.anomaly_detector:
            result["anomaly_detector"] = self.anomaly_detector.get_stats()

        return result

    def get_health_status(self) -> dict[str, Any]:
        """Get service health status."""
        status = "healthy"
        issues = []

        # Check component health
        if self.metrics.logger_status == "error":
            status = "degraded"
            issues.append("Logger component has errors")

        if self.metrics.metrics_collector_status == "error":
            status = "degraded"
            issues.append("Metrics collector has errors")

        if self.metrics.tracer_status == "error":
            status = "degraded"
            issues.append("Tracer has errors")

        # Check error rates
        total_errors = (
            self.metrics.logging_errors
            + self.metrics.metrics_errors
            + self.metrics.tracing_errors
            + self.metrics.analysis_errors
        )

        if total_errors > 10:
            status = "degraded"
            issues.append(f"High error count: {total_errors}")

        return {
            "status": status,
            "uptime": self.metrics.service_uptime,
            "initialized": self._initialized,
            "running": self._running,
            "issues": issues,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def shutdown(self):
        """Shutdown observability service."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._shutdown_event.set()

            if self.logger:
                self.logger.info("Shutting down observability service")

            # Shutdown components
            if self.metrics_collector:
                self.metrics_collector.shutdown()

            if self.log_aggregator:
                self.log_aggregator.shutdown()

            if self.log_analyzer:
                self.log_analyzer.shutdown()

            if self.tracer:
                self.tracer.shutdown()

            # Wait for monitoring thread
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5)

            if self.logger:
                self.logger.info("Observability service shutdown complete")


# Global service instance
_global_service: ObservabilityService | None = None


def get_observability_service() -> ObservabilityService | None:
    """Get global observability service."""
    return _global_service


def configure_observability(config: ObservabilityConfig) -> ObservabilityService:
    """Configure global observability service."""
    global _global_service
    _global_service = ObservabilityService(config)
    _global_service.initialize()
    return _global_service


def shutdown_observability():
    """Shutdown global observability service."""
    global _global_service
    if _global_service:
        _global_service.shutdown()
        _global_service = None
