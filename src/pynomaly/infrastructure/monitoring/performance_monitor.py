"""Real-time performance monitoring system.

This module provides comprehensive performance monitoring capabilities for
anomaly detection operations, integrating with the feature flag system for
controlled deployment during Phase 2.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import psutil

from ...infrastructure.config.feature_flags import require_feature


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""

    # Timing metrics
    execution_time: float = 0.0
    setup_time: float = 0.0
    cleanup_time: float = 0.0

    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_peak: float = 0.0

    # Throughput metrics
    samples_processed: int = 0
    samples_per_second: float = 0.0

    # Quality metrics
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None

    # System metrics
    timestamp: datetime = field(default_factory=datetime.utcnow)
    operation_name: str = ""
    algorithm_name: str = ""
    dataset_size: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "execution_time": self.execution_time,
            "setup_time": self.setup_time,
            "cleanup_time": self.cleanup_time,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "memory_peak": self.memory_peak,
            "samples_processed": self.samples_processed,
            "samples_per_second": self.samples_per_second,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "timestamp": self.timestamp.isoformat(),
            "operation_name": self.operation_name,
            "algorithm_name": self.algorithm_name,
            "dataset_size": self.dataset_size,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceAlert:
    """Container for performance alerts."""

    alert_type: str  # 'threshold_exceeded', 'anomaly_detected', 'degradation'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold_value: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    operation_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "operation_name": self.operation_name,
        }


class PerformanceMonitor:
    """Real-time performance monitoring system."""

    def __init__(
        self,
        max_history: int = 1000,
        alert_thresholds: dict[str, float] | None = None,
        monitoring_interval: float = 1.0,
    ):
        """Initialize performance monitor.

        Args:
            max_history: Maximum number of metrics to keep in history
            alert_thresholds: Dictionary of metric name to threshold values
            monitoring_interval: Interval between monitoring checks (seconds)
        """
        self.max_history = max_history
        self.monitoring_interval = monitoring_interval

        # Performance data storage
        self.metrics_history: deque = deque(maxlen=max_history)
        self.current_operations: dict[str, dict] = {}
        self.operation_statistics: dict[str, list[PerformanceMetrics]] = defaultdict(
            list
        )

        # Alert system
        self.alert_thresholds = alert_thresholds or {
            "execution_time": 30.0,  # seconds
            "memory_usage": 1000.0,  # MB
            "cpu_usage": 80.0,  # percentage
            "samples_per_second": 100.0,  # minimum throughput
        }
        self.active_alerts: list[PerformanceAlert] = []
        self.alert_callbacks: list[Callable] = []

        # Real-time monitoring
        self._monitoring_active = False
        self._monitoring_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Statistics
        self.total_operations = 0
        self.failed_operations = 0
        self.average_execution_time = 0.0

    @require_feature("performance_monitoring")
    def start_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """Stop real-time performance monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._stop_event.set()

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)

        self._monitoring_thread = None

    @require_feature("performance_monitoring")
    def start_operation(
        self,
        operation_name: str,
        algorithm_name: str = "",
        dataset_size: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start monitoring a performance operation.

        Args:
            operation_name: Name of the operation being monitored
            algorithm_name: Name of the algorithm being used
            dataset_size: Size of dataset being processed
            metadata: Additional metadata

        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"

        # Get baseline metrics
        process = psutil.Process()
        memory_info = process.memory_info()

        self.current_operations[operation_id] = {
            "operation_name": operation_name,
            "algorithm_name": algorithm_name,
            "dataset_size": dataset_size,
            "metadata": metadata or {},
            "start_time": time.time(),
            "start_memory": memory_info.rss / 1024 / 1024,  # MB
            "peak_memory": memory_info.rss / 1024 / 1024,
            "start_timestamp": datetime.utcnow(),
        }

        return operation_id

    @require_feature("performance_monitoring")
    def end_operation(
        self,
        operation_id: str,
        samples_processed: int = 0,
        quality_metrics: dict[str, float] | None = None,
    ) -> PerformanceMetrics:
        """End monitoring a performance operation.

        Args:
            operation_id: ID of the operation to end
            samples_processed: Number of samples processed
            quality_metrics: Quality metrics (accuracy, precision, etc.)

        Returns:
            Complete performance metrics for the operation
        """
        if operation_id not in self.current_operations:
            raise ValueError(f"Operation {operation_id} not found")

        operation_data = self.current_operations.pop(operation_id)
        end_time = time.time()

        # Calculate timing metrics
        execution_time = end_time - operation_data["start_time"]

        # Get current resource usage
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        current_memory = memory_info.rss / 1024 / 1024  # MB
        memory_usage = current_memory - operation_data["start_memory"]
        memory_peak = operation_data["peak_memory"]

        # Calculate throughput
        samples_per_second = (
            samples_processed / execution_time if execution_time > 0 else 0
        )

        # Create performance metrics
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            cpu_usage=cpu_percent,
            memory_usage=memory_usage,
            memory_peak=memory_peak,
            samples_processed=samples_processed,
            samples_per_second=samples_per_second,
            operation_name=operation_data["operation_name"],
            algorithm_name=operation_data["algorithm_name"],
            dataset_size=operation_data["dataset_size"],
            metadata=operation_data["metadata"],
        )

        # Add quality metrics if provided
        if quality_metrics:
            metrics.accuracy = quality_metrics.get("accuracy")
            metrics.precision = quality_metrics.get("precision")
            metrics.recall = quality_metrics.get("recall")
            metrics.f1_score = quality_metrics.get("f1_score")

        # Store metrics
        self.metrics_history.append(metrics)
        self.operation_statistics[operation_data["operation_name"]].append(metrics)

        # Update statistics
        self.total_operations += 1
        self._update_average_execution_time(execution_time)

        # Check for alerts
        self._check_alerts(metrics)

        return metrics

    @require_feature("performance_monitoring")
    def get_real_time_metrics(self) -> dict[str, Any]:
        """Get current real-time system metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_usage": cpu_percent,
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "active_operations": len(self.current_operations),
            "total_operations": self.total_operations,
            "failed_operations": self.failed_operations,
            "average_execution_time": self.average_execution_time,
            "active_alerts": len(self.active_alerts),
        }

    @require_feature("performance_monitoring")
    def get_operation_statistics(
        self,
        operation_name: str | None = None,
        time_window: timedelta | None = None,
    ) -> dict[str, Any]:
        """Get statistics for operations.

        Args:
            operation_name: Specific operation to analyze (None for all)
            time_window: Time window for analysis (None for all history)

        Returns:
            Statistical analysis of operations
        """
        # Filter metrics based on criteria
        if operation_name:
            metrics_list = self.operation_statistics.get(operation_name, [])
        else:
            metrics_list = list(self.metrics_history)

        if time_window:
            cutoff_time = datetime.utcnow() - time_window
            metrics_list = [m for m in metrics_list if m.timestamp >= cutoff_time]

        if not metrics_list:
            return {"message": "No metrics available for specified criteria"}

        # Calculate statistics
        execution_times = [m.execution_time for m in metrics_list]
        memory_usages = [m.memory_usage for m in metrics_list]
        cpu_usages = [m.cpu_usage for m in metrics_list]
        throughputs = [
            m.samples_per_second for m in metrics_list if m.samples_per_second > 0
        ]

        stats = {
            "operation_count": len(metrics_list),
            "time_range": {
                "start": min(m.timestamp for m in metrics_list).isoformat(),
                "end": max(m.timestamp for m in metrics_list).isoformat(),
            },
            "execution_time": {
                "mean": np.mean(execution_times),
                "median": np.median(execution_times),
                "std": np.std(execution_times),
                "min": np.min(execution_times),
                "max": np.max(execution_times),
                "p95": np.percentile(execution_times, 95),
                "p99": np.percentile(execution_times, 99),
            },
            "memory_usage": {
                "mean": np.mean(memory_usages),
                "median": np.median(memory_usages),
                "std": np.std(memory_usages),
                "min": np.min(memory_usages),
                "max": np.max(memory_usages),
            },
            "cpu_usage": {
                "mean": np.mean(cpu_usages),
                "median": np.median(cpu_usages),
                "std": np.std(cpu_usages),
                "min": np.min(cpu_usages),
                "max": np.max(cpu_usages),
            },
        }

        if throughputs:
            stats["throughput"] = {
                "mean": np.mean(throughputs),
                "median": np.median(throughputs),
                "std": np.std(throughputs),
                "min": np.min(throughputs),
                "max": np.max(throughputs),
            }

        return stats

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add a callback function for performance alerts."""
        self.alert_callbacks.append(callback)

    def get_active_alerts(self) -> list[PerformanceAlert]:
        """Get list of currently active alerts."""
        return self.active_alerts.copy()

    def clear_alerts(self, alert_type: str | None = None) -> None:
        """Clear alerts, optionally filtered by type."""
        if alert_type:
            self.active_alerts = [
                a for a in self.active_alerts if a.alert_type != alert_type
            ]
        else:
            self.active_alerts.clear()

    def export_metrics(
        self, format_type: str = "json", time_window: timedelta | None = None
    ) -> str | dict:
        """Export metrics in specified format.

        Args:
            format_type: 'json' or 'csv'
            time_window: Time window for export (None for all)

        Returns:
            Metrics data in requested format
        """
        # Filter metrics by time window
        metrics_list = list(self.metrics_history)
        if time_window:
            cutoff_time = datetime.utcnow() - time_window
            metrics_list = [m for m in metrics_list if m.timestamp >= cutoff_time]

        if format_type == "json":
            return {
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_metrics": len(metrics_list),
                "metrics": [m.to_dict() for m in metrics_list],
            }
        elif format_type == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "timestamp",
                    "operation_name",
                    "algorithm_name",
                    "execution_time",
                    "memory_usage",
                    "cpu_usage",
                    "samples_processed",
                    "samples_per_second",
                ],
            )
            writer.writeheader()

            for metric in metrics_list:
                writer.writerow(
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "operation_name": metric.operation_name,
                        "algorithm_name": metric.algorithm_name,
                        "execution_time": metric.execution_time,
                        "memory_usage": metric.memory_usage,
                        "cpu_usage": metric.cpu_usage,
                        "samples_processed": metric.samples_processed,
                        "samples_per_second": metric.samples_per_second,
                    }
                )

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while not self._stop_event.is_set():
            try:
                # Update peak memory for active operations
                if self.current_operations:
                    process = psutil.Process()
                    current_memory = process.memory_info().rss / 1024 / 1024

                    for operation_data in self.current_operations.values():
                        operation_data["peak_memory"] = max(
                            operation_data["peak_memory"], current_memory
                        )

                # Sleep until next monitoring interval
                self._stop_event.wait(self.monitoring_interval)

            except Exception as e:
                # Log error but continue monitoring
                print(f"Monitoring error: {e}")
                self._stop_event.wait(self.monitoring_interval)

    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check if metrics trigger any alerts."""
        alerts_to_add = []

        # Check execution time threshold
        if metrics.execution_time > self.alert_thresholds.get(
            "execution_time", float("inf")
        ):
            alert = PerformanceAlert(
                alert_type="threshold_exceeded",
                severity="medium",
                message=f"Execution time {metrics.execution_time:.2f}s exceeds threshold",
                metric_name="execution_time",
                current_value=metrics.execution_time,
                threshold_value=self.alert_thresholds["execution_time"],
                operation_name=metrics.operation_name,
            )
            alerts_to_add.append(alert)

        # Check memory usage threshold
        if metrics.memory_usage > self.alert_thresholds.get(
            "memory_usage", float("inf")
        ):
            alert = PerformanceAlert(
                alert_type="threshold_exceeded",
                severity="high",
                message=f"Memory usage {metrics.memory_usage:.1f}MB exceeds threshold",
                metric_name="memory_usage",
                current_value=metrics.memory_usage,
                threshold_value=self.alert_thresholds["memory_usage"],
                operation_name=metrics.operation_name,
            )
            alerts_to_add.append(alert)

        # Check CPU usage threshold
        if metrics.cpu_usage > self.alert_thresholds.get("cpu_usage", float("inf")):
            alert = PerformanceAlert(
                alert_type="threshold_exceeded",
                severity="medium",
                message=f"CPU usage {metrics.cpu_usage:.1f}% exceeds threshold",
                metric_name="cpu_usage",
                current_value=metrics.cpu_usage,
                threshold_value=self.alert_thresholds["cpu_usage"],
                operation_name=metrics.operation_name,
            )
            alerts_to_add.append(alert)

        # Check throughput threshold (minimum)
        min_throughput = self.alert_thresholds.get("samples_per_second", 0)
        if (
            metrics.samples_per_second > 0
            and metrics.samples_per_second < min_throughput
        ):
            alert = PerformanceAlert(
                alert_type="threshold_exceeded",
                severity="medium",
                message=f"Throughput {metrics.samples_per_second:.1f} samples/s below threshold",
                metric_name="samples_per_second",
                current_value=metrics.samples_per_second,
                threshold_value=min_throughput,
                operation_name=metrics.operation_name,
            )
            alerts_to_add.append(alert)

        # Add alerts and trigger callbacks
        for alert in alerts_to_add:
            self.active_alerts.append(alert)
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Alert callback error: {e}")

    def _update_average_execution_time(self, execution_time: float) -> None:
        """Update running average execution time."""
        if self.total_operations == 1:
            self.average_execution_time = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_execution_time = (
                alpha * execution_time + (1 - alpha) * self.average_execution_time
            )


class PerformanceTracker:
    """Context manager for tracking operation performance."""

    def __init__(
        self,
        monitor: PerformanceMonitor,
        operation_name: str,
        algorithm_name: str = "",
        dataset_size: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize performance tracker."""
        self.monitor = monitor
        self.operation_name = operation_name
        self.algorithm_name = algorithm_name
        self.dataset_size = dataset_size
        self.metadata = metadata
        self.operation_id: str | None = None
        self.samples_processed = 0
        self.quality_metrics: dict[str, float] = {}

    def __enter__(self) -> PerformanceTracker:
        """Start performance tracking."""
        self.operation_id = self.monitor.start_operation(
            self.operation_name, self.algorithm_name, self.dataset_size, self.metadata
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End performance tracking."""
        if self.operation_id:
            try:
                self.monitor.end_operation(
                    self.operation_id, self.samples_processed, self.quality_metrics
                )
            except Exception as e:
                print(f"Error ending performance tracking: {e}")

    def set_samples_processed(self, count: int) -> None:
        """Set number of samples processed."""
        self.samples_processed = count

    def set_quality_metrics(self, metrics: dict[str, float]) -> None:
        """Set quality metrics for the operation."""
        self.quality_metrics = metrics


# Utility functions and decorators
def monitor_performance(
    monitor: PerformanceMonitor, operation_name: str, algorithm_name: str = ""
):
    """Decorator for monitoring function performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceTracker(monitor, operation_name, algorithm_name) as tracker:
                result = func(*args, **kwargs)

                # Try to extract metrics from result if it's a detection result
                if hasattr(result, "metadata") and isinstance(result.metadata, dict):
                    if "samples_processed" in result.metadata:
                        tracker.set_samples_processed(
                            result.metadata["samples_processed"]
                        )

                    quality_metrics = {}
                    for metric in ["accuracy", "precision", "recall", "f1_score"]:
                        if metric in result.metadata:
                            quality_metrics[metric] = result.metadata[metric]

                    if quality_metrics:
                        tracker.set_quality_metrics(quality_metrics)

                return result

        return wrapper

    return decorator


async def monitor_async_performance(
    monitor: PerformanceMonitor, operation_name: str, algorithm_name: str = ""
):
    """Async context manager for monitoring async operations."""
    tracker = PerformanceTracker(monitor, operation_name, algorithm_name)
    await asyncio.to_thread(tracker.__enter__)
    try:
        yield tracker
    finally:
        await asyncio.to_thread(tracker.__exit__, None, None, None)
