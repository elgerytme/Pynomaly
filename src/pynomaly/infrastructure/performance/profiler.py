"""Advanced performance profiling and optimization system."""

from __future__ import annotations

import cProfile
import functools
import gc
import io
import pstats
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class ProfileResult:
    """Profiling result with comprehensive metrics."""

    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    cpu_percent: float
    memory_used_mb: float
    memory_peak_mb: float
    function_calls: int
    primitive_calls: int
    top_functions: list[tuple[str, float]] = field(default_factory=list)
    memory_allocations: dict[str, Any] | None = None
    gc_stats: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "cpu_percent": self.cpu_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_peak_mb": self.memory_peak_mb,
            "function_calls": self.function_calls,
            "primitive_calls": self.primitive_calls,
            "top_functions": self.top_functions,
            "memory_allocations": self.memory_allocations,
            "gc_stats": self.gc_stats,
            "context": self.context,
        }


class PerformanceProfiler:
    """Advanced performance profiler with multiple profiling modes."""

    def __init__(
        self,
        enable_cpu_profiling: bool = True,
        enable_memory_profiling: bool = True,
        enable_line_profiling: bool = False,
        max_results: int = 1000,
        output_dir: Path | None = None,
    ):
        """Initialize performance profiler.

        Args:
            enable_cpu_profiling: Enable CPU profiling
            enable_memory_profiling: Enable memory profiling
            enable_line_profiling: Enable line-by-line profiling
            max_results: Maximum number of results to keep
            output_dir: Directory for profile output files
        """
        self.enable_cpu_profiling = enable_cpu_profiling
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_line_profiling = enable_line_profiling
        self.max_results = max_results
        self.output_dir = output_dir

        # Profile storage
        self._results: deque[ProfileResult] = deque(maxlen=max_results)
        self._active_profiles: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

        # Performance metrics
        self._metrics: dict[str, deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # System monitoring
        self._process = psutil.Process()
        self._baseline_memory = None

        # Initialize memory tracing if enabled
        if self.enable_memory_profiling:
            if not tracemalloc.is_tracing():
                tracemalloc.start()

    @contextmanager
    def profile(
        self,
        operation_name: str,
        context: dict[str, Any] | None = None,
        save_to_file: bool = False,
    ):
        """Context manager for profiling operations.

        Args:
            operation_name: Name of operation being profiled
            context: Additional context information
            save_to_file: Whether to save detailed profile to file
        """
        profile_id = str(uuid4())
        context = context or {}

        # Start profiling
        profile_data = self._start_profiling(profile_id, operation_name, context)

        try:
            yield profile_data
        finally:
            # End profiling and store results
            result = self._end_profiling(profile_id, save_to_file)
            if result:
                with self._lock:
                    self._results.append(result)

    def _start_profiling(
        self, profile_id: str, operation_name: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Start profiling for an operation."""
        start_time = datetime.utcnow()

        profile_data = {
            "operation_name": operation_name,
            "start_time": start_time,
            "context": context,
            "cpu_profiler": None,
            "memory_snapshot": None,
            "start_memory": None,
            "start_cpu_times": None,
        }

        # CPU profiling
        if self.enable_cpu_profiling:
            cpu_profiler = cProfile.Profile()
            cpu_profiler.enable()
            profile_data["cpu_profiler"] = cpu_profiler

        # Memory profiling
        if self.enable_memory_profiling and tracemalloc.is_tracing():
            profile_data["memory_snapshot"] = tracemalloc.take_snapshot()
            profile_data["start_memory"] = self._process.memory_info().rss

        # CPU times
        profile_data["start_cpu_times"] = self._process.cpu_times()

        with self._lock:
            self._active_profiles[profile_id] = profile_data

        return profile_data

    def _end_profiling(
        self, profile_id: str, save_to_file: bool = False
    ) -> ProfileResult | None:
        """End profiling and generate results."""
        with self._lock:
            if profile_id not in self._active_profiles:
                return None

            profile_data = self._active_profiles.pop(profile_id)

        end_time = datetime.utcnow()
        duration = (end_time - profile_data["start_time"]).total_seconds()

        # CPU profiling results
        cpu_stats = {}
        if profile_data["cpu_profiler"]:
            profiler = profile_data["cpu_profiler"]
            profiler.disable()

            # Get statistics
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats("cumulative")

            cpu_stats = {
                "total_calls": stats.total_calls,
                "primitive_calls": stats.prim_calls,
                "top_functions": self._extract_top_functions(stats),
            }

            # Save to file if requested
            if save_to_file and self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                filename = (
                    self.output_dir
                    / f"profile_{profile_data['operation_name']}_{int(time.time())}.prof"
                )
                stats.dump_stats(str(filename))

        # Memory profiling results
        memory_stats = {}
        if self.enable_memory_profiling and tracemalloc.is_tracing():
            current_snapshot = tracemalloc.take_snapshot()

            if profile_data["memory_snapshot"]:
                top_stats = current_snapshot.compare_to(
                    profile_data["memory_snapshot"], "lineno"
                )

                memory_stats = {
                    "allocations": [
                        {
                            "filename": stat.traceback.format()[0],
                            "size_mb": stat.size_diff / 1024 / 1024,
                            "count": stat.count_diff,
                        }
                        for stat in top_stats[:10]
                    ],
                    "total_size_diff_mb": sum(stat.size_diff for stat in top_stats)
                    / 1024
                    / 1024,
                }

        # System resource usage
        current_memory = self._process.memory_info().rss
        memory_used_mb = (
            (current_memory - profile_data.get("start_memory", current_memory))
            / 1024
            / 1024
        )
        memory_peak_mb = self._process.memory_info().rss / 1024 / 1024

        # CPU usage
        cpu_percent = self._process.cpu_percent()

        # GC statistics
        gc_stats = {"collections": gc.get_stats(), "objects": len(gc.get_objects())}

        # Create result
        result = ProfileResult(
            operation_name=profile_data["operation_name"],
            start_time=profile_data["start_time"],
            end_time=end_time,
            duration_seconds=duration,
            cpu_percent=cpu_percent,
            memory_used_mb=memory_used_mb,
            memory_peak_mb=memory_peak_mb,
            function_calls=cpu_stats.get("total_calls", 0),
            primitive_calls=cpu_stats.get("primitive_calls", 0),
            top_functions=cpu_stats.get("top_functions", []),
            memory_allocations=memory_stats,
            gc_stats=gc_stats,
            context=profile_data["context"],
        )

        return result

    def _extract_top_functions(
        self, stats: pstats.Stats, limit: int = 10
    ) -> list[tuple[str, float]]:
        """Extract top functions from profile statistics."""
        top_functions = []

        # Get function statistics
        for func, stat in stats.stats.items():
            filename, line_num, func_name = func
            cumulative_time = stat[3]  # cumulative time

            # Format function name
            func_display = f"{func_name} ({Path(filename).name}:{line_num})"
            top_functions.append((func_display, cumulative_time))

        # Sort by cumulative time and return top N
        top_functions.sort(key=lambda x: x[1], reverse=True)
        return top_functions[:limit]

    def profile_function(
        self,
        func: Callable | None = None,
        operation_name: str | None = None,
        save_to_file: bool = False,
    ):
        """Decorator for profiling functions."""

        def decorator(f: Callable) -> Callable:
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{f.__module__}.{f.__name__}"

                with self.profile(op_name, save_to_file=save_to_file):
                    return f(*args, **kwargs)

            return wrapper

        if func is None:
            return decorator
        else:
            return decorator(func)

    def add_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        context: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        """Add a performance metric."""
        metric = PerformanceMetric(
            name=name, value=value, unit=unit, context=context or {}, tags=tags or []
        )

        with self._lock:
            self._metrics[name].append(metric)

    def get_results(
        self,
        operation_name: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[ProfileResult]:
        """Get profiling results."""
        with self._lock:
            results = list(self._results)

        # Filter by operation name
        if operation_name:
            results = [r for r in results if r.operation_name == operation_name]

        # Filter by time
        if since:
            results = [r for r in results if r.start_time >= since]

        # Sort by start time (most recent first)
        results.sort(key=lambda x: x.start_time, reverse=True)

        # Apply limit
        if limit:
            results = results[:limit]

        return results

    def get_metrics(
        self, metric_name: str | None = None, since: datetime | None = None
    ) -> dict[str, list[PerformanceMetric]]:
        """Get performance metrics."""
        with self._lock:
            if metric_name:
                metrics = {metric_name: list(self._metrics.get(metric_name, []))}
            else:
                metrics = {name: list(values) for name, values in self._metrics.items()}

        # Filter by time
        if since:
            filtered_metrics = {}
            for name, metric_list in metrics.items():
                filtered_list = [m for m in metric_list if m.timestamp >= since]
                if filtered_list:
                    filtered_metrics[name] = filtered_list
            metrics = filtered_metrics

        return metrics

    def get_summary(self, operation_name: str | None = None) -> dict[str, Any]:
        """Get performance summary statistics."""
        results = self.get_results(operation_name)

        if not results:
            return {"message": "No profiling results available"}

        # Calculate statistics
        durations = [r.duration_seconds for r in results]
        memory_usage = [r.memory_used_mb for r in results]
        cpu_usage = [r.cpu_percent for r in results]

        import statistics

        summary = {
            "operation_name": operation_name or "all",
            "total_runs": len(results),
            "duration_stats": {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
                "stdev": statistics.stdev(durations) if len(durations) > 1 else 0,
            },
            "memory_stats": {
                "mean_mb": statistics.mean(memory_usage),
                "median_mb": statistics.median(memory_usage),
                "min_mb": min(memory_usage),
                "max_mb": max(memory_usage),
            },
            "cpu_stats": {
                "mean_percent": statistics.mean(cpu_usage),
                "median_percent": statistics.median(cpu_usage),
                "min_percent": min(cpu_usage),
                "max_percent": max(cpu_usage),
            },
            "time_range": {
                "start": min(r.start_time for r in results).isoformat(),
                "end": max(r.end_time for r in results).isoformat(),
            },
        }

        return summary

    def clear_results(self, operation_name: str | None = None):
        """Clear profiling results."""
        with self._lock:
            if operation_name:
                # Remove specific operation results
                self._results = deque(
                    (r for r in self._results if r.operation_name != operation_name),
                    maxlen=self.max_results,
                )
            else:
                # Clear all results
                self._results.clear()

    def export_results(
        self, filepath: Path, format: str = "json", operation_name: str | None = None
    ):
        """Export profiling results to file."""
        results = self.get_results(operation_name)

        if format.lower() == "json":
            import json

            data = {
                "export_time": datetime.utcnow().isoformat(),
                "operation_name": operation_name,
                "total_results": len(results),
                "results": [result.to_dict() for result in results],
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        elif format.lower() == "csv":
            import csv

            with open(filepath, "w", newline="") as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
                    writer.writeheader()
                    for result in results:
                        writer.writerow(result.to_dict())

        else:
            raise ValueError(f"Unsupported export format: {format}")


class SystemMonitor:
    """System-wide performance monitoring."""

    def __init__(
        self,
        monitoring_interval: int = 5,
        history_size: int = 1000,
        enable_alerts: bool = False,
    ):
        """Initialize system monitor.

        Args:
            monitoring_interval: Monitoring interval in seconds
            history_size: Number of monitoring samples to keep
            enable_alerts: Enable performance alerts
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_alerts = enable_alerts

        self._metrics_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._monitoring_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()

        # Alert thresholds
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "response_time_ms": 1000.0,
        }

        self.alert_callbacks: list[Callable[[str, dict[str, Any]], None]] = []

    def start_monitoring(self):
        """Start background monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        def monitoring_worker():
            while not self._shutdown_event.wait(self.monitoring_interval):
                try:
                    self._collect_system_metrics()
                except Exception as e:
                    print(f"Error in system monitoring: {e}")

        self._monitoring_thread = threading.Thread(
            target=monitoring_worker, daemon=True
        )
        self._monitoring_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._shutdown_event.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)

    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = datetime.utcnow()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0, 0, 0)

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk metrics
        disk = psutil.disk_usage("/")

        # Network metrics
        network = psutil.net_io_counters()

        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()

        metrics = {
            "timestamp": timestamp,
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "load_avg_1min": load_avg[0],
            "load_avg_5min": load_avg[1],
            "load_avg_15min": load_avg[2],
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_percent": memory.percent,
            "swap_total_gb": swap.total / (1024**3),
            "swap_used_gb": swap.used / (1024**3),
            "swap_percent": swap.percent,
            "disk_total_gb": disk.total / (1024**3),
            "disk_used_gb": disk.used / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "disk_usage_percent": (disk.used / disk.total) * 100,
            "network_bytes_sent": network.bytes_sent,
            "network_bytes_recv": network.bytes_recv,
            "process_memory_rss_mb": process_memory.rss / (1024**2),
            "process_memory_vms_mb": process_memory.vms / (1024**2),
            "process_cpu_percent": process.cpu_percent(),
        }

        # Store metrics
        with self._lock:
            for key, value in metrics.items():
                if key != "timestamp":
                    self._metrics_history[key].append((timestamp, value))

        # Check alerts
        if self.enable_alerts:
            self._check_alerts(metrics)

    def _check_alerts(self, metrics: dict[str, Any]):
        """Check for alert conditions."""
        alerts = []

        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics and metrics[metric_name] > threshold:
                alert = {
                    "metric": metric_name,
                    "value": metrics[metric_name],
                    "threshold": threshold,
                    "timestamp": metrics["timestamp"],
                    "severity": (
                        "high" if metrics[metric_name] > threshold * 1.1 else "medium"
                    ),
                }
                alerts.append(alert)

        # Notify alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback("performance_alert", alert)
                except Exception as e:
                    print(f"Error in alert callback: {e}")

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current system metrics."""
        # Collect immediate metrics
        self._collect_system_metrics()

        # Return latest values
        with self._lock:
            current = {}
            for metric_name, history in self._metrics_history.items():
                if history:
                    current[metric_name] = history[-1][1]  # Get latest value
            return current

    def get_metric_history(
        self, metric_name: str, since: datetime | None = None, limit: int | None = None
    ) -> list[tuple[datetime, float]]:
        """Get metric history."""
        with self._lock:
            history = list(self._metrics_history.get(metric_name, []))

        # Filter by time
        if since:
            history = [(ts, value) for ts, value in history if ts >= since]

        # Apply limit
        if limit:
            history = history[-limit:]

        return history

    def add_alert_callback(self, callback: Callable[[str, dict[str, Any]], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)

    def get_summary(self) -> dict[str, Any]:
        """Get system performance summary."""
        current = self.get_current_metrics()

        # Calculate trends for key metrics
        trends = {}
        for metric in ["cpu_percent", "memory_percent", "disk_usage_percent"]:
            history = self.get_metric_history(metric, limit=10)
            if len(history) >= 2:
                recent_avg = sum(value for _, value in history[-3:]) / 3
                older_avg = (
                    sum(value for _, value in history[-6:-3]) / 3
                    if len(history) >= 6
                    else recent_avg
                )
                trends[metric] = {
                    "current": history[-1][1],
                    "trend": "increasing" if recent_avg > older_avg else "decreasing",
                    "change_percent": (
                        ((recent_avg - older_avg) / older_avg * 100)
                        if older_avg > 0
                        else 0
                    ),
                }

        return {
            "current_metrics": current,
            "trends": trends,
            "monitoring_active": self._monitoring_thread
            and self._monitoring_thread.is_alive(),
            "alert_thresholds": self.alert_thresholds,
            "metrics_collected": sum(
                len(history) for history in self._metrics_history.values()
            ),
        }


# Global instances
_global_profiler: PerformanceProfiler | None = None
_global_monitor: SystemMonitor | None = None


def get_profiler() -> PerformanceProfiler | None:
    """Get global profiler instance."""
    return _global_profiler


def get_monitor() -> SystemMonitor | None:
    """Get global monitor instance."""
    return _global_monitor


def configure_profiler(**kwargs) -> PerformanceProfiler:
    """Configure global profiler."""
    global _global_profiler
    _global_profiler = PerformanceProfiler(**kwargs)
    return _global_profiler


def configure_monitor(**kwargs) -> SystemMonitor:
    """Configure global monitor."""
    global _global_monitor
    _global_monitor = SystemMonitor(**kwargs)
    return _global_monitor


def profile(operation_name: str | None = None, save_to_file: bool = False):
    """Decorator for profiling functions using global profiler."""

    def decorator(func: Callable) -> Callable:
        if not _global_profiler:
            return func  # Return original function if no profiler configured

        return _global_profiler.profile_function(func, operation_name, save_to_file)

    return decorator
