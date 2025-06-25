"""Comprehensive metrics collection and aggregation system."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import psutil


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: int | float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: dict[str, str] = field(default_factory=dict)
    unit: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "unit": self.unit,
            "description": self.description
        }


@dataclass
class MetricsSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    std: float
    percentiles: dict[str, float]  # e.g., {"p50": 100, "p95": 200, "p99": 300}
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """High-performance metrics collection system."""

    def __init__(
        self,
        max_metrics_in_memory: int = 10000,
        flush_interval_seconds: int = 60,
        auto_flush: bool = True,
        enable_system_metrics: bool = True,
        system_metrics_interval: int = 30,
        storage_path: Path | None = None
    ):
        """Initialize metrics collector.

        Args:
            max_metrics_in_memory: Maximum metrics to keep in memory
            flush_interval_seconds: How often to flush metrics to storage
            auto_flush: Whether to automatically flush metrics
            enable_system_metrics: Whether to collect system metrics
            system_metrics_interval: Interval for system metrics collection
            storage_path: Path to store metrics data
        """
        self.max_metrics_in_memory = max_metrics_in_memory
        self.flush_interval_seconds = flush_interval_seconds
        self.auto_flush = auto_flush
        self.enable_system_metrics = enable_system_metrics
        self.system_metrics_interval = system_metrics_interval
        self.storage_path = storage_path

        # Metrics storage
        self._metrics: deque = deque(maxlen=max_metrics_in_memory)
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._timers: dict[str, list[float]] = defaultdict(list)

        # Aggregated metrics
        self._metric_summaries: dict[str, MetricsSummary] = {}

        # Threading
        self._lock = threading.RLock()
        self._flush_thread: threading.Thread | None = None
        self._system_metrics_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Performance tracking
        self.stats = {
            "metrics_collected": 0,
            "metrics_flushed": 0,
            "flush_errors": 0,
            "last_flush_time": None,
            "collection_errors": 0
        }

        # Start background tasks
        if self.auto_flush:
            self._start_flush_thread()

        if self.enable_system_metrics:
            self._start_system_metrics_thread()

    def _start_flush_thread(self):
        """Start background thread for flushing metrics."""
        def flush_worker():
            while not self._shutdown_event.wait(self.flush_interval_seconds):
                try:
                    self.flush_metrics()
                except Exception as e:
                    self.stats["flush_errors"] += 1
                    print(f"Error flushing metrics: {e}")

        self._flush_thread = threading.Thread(target=flush_worker, daemon=True)
        self._flush_thread.start()

    def _start_system_metrics_thread(self):
        """Start background thread for collecting system metrics."""
        def system_metrics_worker():
            while not self._shutdown_event.wait(self.system_metrics_interval):
                try:
                    self._collect_system_metrics()
                except Exception as e:
                    self.stats["collection_errors"] += 1
                    print(f"Error collecting system metrics: {e}")

        self._system_metrics_thread = threading.Thread(target=system_metrics_worker, daemon=True)
        self._system_metrics_thread.start()

    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.gauge("system.cpu.usage_percent", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.gauge("system.memory.usage_percent", memory.percent)
            self.gauge("system.memory.available_mb", memory.available / (1024 * 1024))
            self.gauge("system.memory.used_mb", memory.used / (1024 * 1024))

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.gauge("system.disk.usage_percent", (disk.used / disk.total) * 100)
            self.gauge("system.disk.free_gb", disk.free / (1024 * 1024 * 1024))

            # Network metrics
            network = psutil.net_io_counters()
            self.counter("system.network.bytes_sent", network.bytes_sent)
            self.counter("system.network.bytes_recv", network.bytes_recv)

            # Process metrics
            process = psutil.Process()
            self.gauge("process.cpu.percent", process.cpu_percent())
            self.gauge("process.memory.rss_mb", process.memory_info().rss / (1024 * 1024))
            self.gauge("process.memory.vms_mb", process.memory_info().vms / (1024 * 1024))
            self.gauge("process.num_threads", process.num_threads())

        except Exception as e:
            self.stats["collection_errors"] += 1
            print(f"Error collecting system metrics: {e}")

    def counter(self, name: str, value: int | float = 1, labels: dict[str, str] | None = None):
        """Increment a counter metric."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)

        with self._lock:
            self._counters[metric_key] += value

            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.COUNTER,
                labels=labels
            )
            self._add_metric(metric)

    def gauge(self, name: str, value: int | float, labels: dict[str, str] | None = None):
        """Set a gauge metric value."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)

        with self._lock:
            self._gauges[metric_key] = value

            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                labels=labels
            )
            self._add_metric(metric)

    def histogram(self, name: str, value: int | float, labels: dict[str, str] | None = None):
        """Add a value to a histogram metric."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)

        with self._lock:
            self._histograms[metric_key].append(float(value))

            # Keep only recent values to prevent memory bloat
            if len(self._histograms[metric_key]) > 1000:
                self._histograms[metric_key] = self._histograms[metric_key][-1000:]

            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                labels=labels
            )
            self._add_metric(metric)

    def timer(self, name: str, value: int | float, labels: dict[str, str] | None = None):
        """Record a timing metric."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)

        with self._lock:
            self._timers[metric_key].append(float(value))

            # Keep only recent values
            if len(self._timers[metric_key]) > 1000:
                self._timers[metric_key] = self._timers[metric_key][-1000:]

            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.TIMER,
                labels=labels,
                unit="milliseconds"
            )
            self._add_metric(metric)

    def _get_metric_key(self, name: str, labels: dict[str, str]) -> str:
        """Generate unique key for metric with labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _add_metric(self, metric: Metric):
        """Add metric to collection."""
        self._metrics.append(metric)
        self.stats["metrics_collected"] += 1

    def get_counter_value(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current counter value."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)
        with self._lock:
            return self._counters.get(metric_key, 0)

    def get_gauge_value(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        """Get current gauge value."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)
        with self._lock:
            return self._gauges.get(metric_key)

    def get_histogram_summary(self, name: str, labels: dict[str, str] | None = None) -> MetricsSummary | None:
        """Get histogram summary statistics."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)

        with self._lock:
            values = self._histograms.get(metric_key)
            if not values:
                return None

            return self._calculate_summary(name, values, labels)

    def get_timer_summary(self, name: str, labels: dict[str, str] | None = None) -> MetricsSummary | None:
        """Get timer summary statistics."""
        labels = labels or {}
        metric_key = self._get_metric_key(name, labels)

        with self._lock:
            values = self._timers.get(metric_key)
            if not values:
                return None

            return self._calculate_summary(name, values, labels)

    def _calculate_summary(self, name: str, values: list[float], labels: dict[str, str]) -> MetricsSummary:
        """Calculate summary statistics for a list of values."""
        if not values:
            return MetricsSummary(
                name=name,
                count=0,
                sum=0,
                min=0,
                max=0,
                mean=0,
                std=0,
                percentiles={},
                labels=labels
            )

        import numpy as np

        values_array = np.array(values)

        return MetricsSummary(
            name=name,
            count=len(values),
            sum=float(np.sum(values_array)),
            min=float(np.min(values_array)),
            max=float(np.max(values_array)),
            mean=float(np.mean(values_array)),
            std=float(np.std(values_array)),
            percentiles={
                "p50": float(np.percentile(values_array, 50)),
                "p75": float(np.percentile(values_array, 75)),
                "p90": float(np.percentile(values_array, 90)),
                "p95": float(np.percentile(values_array, 95)),
                "p99": float(np.percentile(values_array, 99))
            },
            labels=labels
        )

    def get_all_metrics(self) -> list[Metric]:
        """Get all collected metrics."""
        with self._lock:
            return list(self._metrics)

    def get_metrics_by_name(self, name: str) -> list[Metric]:
        """Get all metrics with a specific name."""
        with self._lock:
            return [m for m in self._metrics if m.name == name]

    def get_metrics_by_type(self, metric_type: MetricType) -> list[Metric]:
        """Get all metrics of a specific type."""
        with self._lock:
            return [m for m in self._metrics if m.metric_type == metric_type]

    def get_metrics_since(self, since: datetime) -> list[Metric]:
        """Get all metrics since a specific timestamp."""
        with self._lock:
            return [m for m in self._metrics if m.timestamp >= since]

    def clear_metrics(self):
        """Clear all collected metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._metric_summaries.clear()

    def flush_metrics(self):
        """Flush metrics to storage."""
        if not self.storage_path:
            return

        with self._lock:
            metrics_to_flush = list(self._metrics)
            self._metrics.clear()

        if not metrics_to_flush:
            return

        try:
            # Ensure storage directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = self.storage_path.parent / f"metrics_{timestamp}.json"

            # Write metrics to file
            with open(filename, 'w') as f:
                metrics_data = [metric.to_dict() for metric in metrics_to_flush]
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics_count": len(metrics_data),
                    "metrics": metrics_data
                }, f, indent=2)

            self.stats["metrics_flushed"] += len(metrics_to_flush)
            self.stats["last_flush_time"] = datetime.utcnow()

        except Exception:
            self.stats["flush_errors"] += 1
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get collector statistics."""
        with self._lock:
            return {
                "metrics_in_memory": len(self._metrics),
                "counters_count": len(self._counters),
                "gauges_count": len(self._gauges),
                "histograms_count": len(self._histograms),
                "timers_count": len(self._timers),
                **self.stats
            }

    def shutdown(self):
        """Shutdown the metrics collector."""
        self._shutdown_event.set()

        # Final flush
        if self.auto_flush:
            try:
                self.flush_metrics()
            except Exception:
                pass

        # Wait for threads to finish
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5)

        if self._system_metrics_thread and self._system_metrics_thread.is_alive():
            self._system_metrics_thread.join(timeout=5)


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, labels: dict[str, str] | None = None):
        """Initialize timer context.

        Args:
            collector: Metrics collector instance
            name: Timer metric name
            labels: Optional labels for the metric
        """
        self.collector = collector
        self.name = name
        self.labels = labels or {}
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record metric."""
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.collector.timer(self.name, duration_ms, self.labels)


def timer_decorator(collector: MetricsCollector, name: str | None = None, labels: dict[str, str] | None = None):
    """Decorator for automatic timing of functions."""
    def decorator(func: Callable):
        metric_name = name or f"{func.__module__}.{func.__name__}.duration"

        def wrapper(*args, **kwargs):
            with TimerContext(collector, metric_name, labels):
                return func(*args, **kwargs)

        return wrapper
    return decorator


# Global metrics collector instance
_default_collector: MetricsCollector | None = None


def get_default_collector() -> MetricsCollector:
    """Get or create the default metrics collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


def configure_metrics(
    storage_path: Path | None = None,
    enable_system_metrics: bool = True,
    flush_interval_seconds: int = 60,
    max_metrics_in_memory: int = 10000
) -> MetricsCollector:
    """Configure the global metrics collector."""
    global _default_collector
    _default_collector = MetricsCollector(
        storage_path=storage_path,
        enable_system_metrics=enable_system_metrics,
        flush_interval_seconds=flush_interval_seconds,
        max_metrics_in_memory=max_metrics_in_memory
    )
    return _default_collector
