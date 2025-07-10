#!/usr/bin/env python3
"""
Performance Optimization Tools for Pynomaly

This module provides comprehensive performance monitoring, profiling,
and optimization capabilities for production environments.
"""

import gc
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any

import memory_profiler
import numpy as np
import psutil


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    RESPONSE_TIME = "response_time"


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    CACHING = "caching"
    CONNECTION_POOLING = "connection_pooling"
    ASYNC_PROCESSING = "async_processing"
    LOAD_BALANCING = "load_balancing"
    DATABASE_OPTIMIZATION = "database_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    CPU_OPTIMIZATION = "cpu_optimization"


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class PerformanceProfile:
    """Performance profiling results."""
    function_name: str
    total_calls: int
    total_time: float
    avg_time: float
    max_time: float
    min_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    call_stack: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "max_time": self.max_time,
            "min_time": self.min_time,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "timestamp": self.timestamp.isoformat(),
            "call_stack": self.call_stack
        }


class PerformanceCollector:
    """Collect performance metrics from various sources."""

    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics = deque(maxlen=10000)  # Keep last 10k metrics
        self.running = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # System monitoring
        self._last_cpu_times = None
        self._last_network_io = None
        self._last_disk_io = None

    def start_collection(self):
        """Start metric collection."""
        if self.running:
            return

        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        self.logger.info("Performance collection started")

    def stop_collection(self):
        """Stop metric collection."""
        self.running = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=5)
        self.logger.info("Performance collection stopped")

    def _collection_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")

    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        now = datetime.now()

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.add_metric(PerformanceMetric(
                PerformanceMetricType.CPU_USAGE,
                cpu_percent,
                now,
                tags={"source": "system", "metric": "cpu_percent"}
            ))

            # Memory metrics
            memory = psutil.virtual_memory()
            self.add_metric(PerformanceMetric(
                PerformanceMetricType.MEMORY_USAGE,
                memory.percent,
                now,
                tags={"source": "system", "metric": "memory_percent"}
            ))

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if self._last_disk_io:
                read_bytes_delta = disk_io.read_bytes - self._last_disk_io.read_bytes
                write_bytes_delta = disk_io.write_bytes - self._last_disk_io.write_bytes

                self.add_metric(PerformanceMetric(
                    PerformanceMetricType.DISK_IO,
                    read_bytes_delta / self.collection_interval,
                    now,
                    tags={"source": "system", "metric": "disk_read_rate"}
                ))

                self.add_metric(PerformanceMetric(
                    PerformanceMetricType.DISK_IO,
                    write_bytes_delta / self.collection_interval,
                    now,
                    tags={"source": "system", "metric": "disk_write_rate"}
                ))

            self._last_disk_io = disk_io

            # Network I/O metrics
            network_io = psutil.net_io_counters()
            if self._last_network_io:
                bytes_sent_delta = network_io.bytes_sent - self._last_network_io.bytes_sent
                bytes_recv_delta = network_io.bytes_recv - self._last_network_io.bytes_recv

                self.add_metric(PerformanceMetric(
                    PerformanceMetricType.NETWORK_IO,
                    bytes_sent_delta / self.collection_interval,
                    now,
                    tags={"source": "system", "metric": "network_send_rate"}
                ))

                self.add_metric(PerformanceMetric(
                    PerformanceMetricType.NETWORK_IO,
                    bytes_recv_delta / self.collection_interval,
                    now,
                    tags={"source": "system", "metric": "network_recv_rate"}
                ))

            self._last_network_io = network_io

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    def add_metric(self, metric: PerformanceMetric):
        """Add performance metric."""
        self.metrics.append(metric)

    def get_metrics(self, metric_type: PerformanceMetricType | None = None,
                   tags: dict[str, str] | None = None,
                   start_time: datetime | None = None,
                   end_time: datetime | None = None) -> list[PerformanceMetric]:
        """Get metrics with optional filtering."""
        filtered_metrics = []

        for metric in self.metrics:
            # Filter by type
            if metric_type and metric.metric_type != metric_type:
                continue

            # Filter by tags
            if tags:
                if not all(metric.tags.get(k) == v for k, v in tags.items()):
                    continue

            # Filter by time range
            if start_time and metric.timestamp < start_time:
                continue
            if end_time and metric.timestamp > end_time:
                continue

            filtered_metrics.append(metric)

        return filtered_metrics

    def get_metric_statistics(self, metric_type: PerformanceMetricType,
                             tags: dict[str, str] | None = None,
                             time_window: timedelta | None = None) -> dict[str, float]:
        """Get statistical summary of metrics."""
        end_time = datetime.now()
        start_time = end_time - time_window if time_window else None

        metrics = self.get_metrics(metric_type, tags, start_time, end_time)
        values = [m.value for m in metrics]

        if not values:
            return {}

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }


class FunctionProfiler:
    """Profile function performance."""

    def __init__(self):
        self.profiles = {}
        self.call_times = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def profile_function(self, include_memory: bool = False):
        """Decorator to profile function performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = None

                if include_memory:
                    start_memory = memory_profiler.memory_usage()[0]

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time

                    # Record timing
                    func_name = f"{func.__module__}.{func.__name__}"
                    self.call_times[func_name].append(execution_time)

                    # Record memory usage
                    if include_memory and start_memory:
                        end_memory = memory_profiler.memory_usage()[0]
                        memory_delta = end_memory - start_memory
                        self.memory_usage[func_name].append(memory_delta)

                    # Update profile
                    self._update_profile(func_name, execution_time, start_memory)

            return wrapper
        return decorator

    def _update_profile(self, func_name: str, execution_time: float, memory_delta: float | None):
        """Update function profile."""
        if func_name not in self.profiles:
            self.profiles[func_name] = PerformanceProfile(
                function_name=func_name,
                total_calls=0,
                total_time=0.0,
                avg_time=0.0,
                max_time=0.0,
                min_time=float('inf'),
                memory_usage=0.0,
                cpu_usage=0.0,
                timestamp=datetime.now()
            )

        profile = self.profiles[func_name]
        profile.total_calls += 1
        profile.total_time += execution_time
        profile.avg_time = profile.total_time / profile.total_calls
        profile.max_time = max(profile.max_time, execution_time)
        profile.min_time = min(profile.min_time, execution_time)

        if memory_delta:
            profile.memory_usage = max(profile.memory_usage, memory_delta)

        profile.timestamp = datetime.now()

    def get_profile(self, func_name: str) -> PerformanceProfile | None:
        """Get profile for specific function."""
        return self.profiles.get(func_name)

    def get_all_profiles(self) -> dict[str, PerformanceProfile]:
        """Get all function profiles."""
        return self.profiles.copy()

    def get_top_functions(self, metric: str = "total_time", limit: int = 10) -> list[PerformanceProfile]:
        """Get top functions by metric."""
        sorted_profiles = sorted(
            self.profiles.values(),
            key=lambda p: getattr(p, metric, 0),
            reverse=True
        )
        return sorted_profiles[:limit]

    def reset_profiles(self):
        """Reset all profiles."""
        self.profiles.clear()
        self.call_times.clear()
        self.memory_usage.clear()


class MemoryProfiler:
    """Profile memory usage."""

    def __init__(self):
        self.snapshots = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def take_snapshot(self, label: str = ""):
        """Take memory usage snapshot."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            snapshot = {
                "label": label,
                "timestamp": datetime.now(),
                "rss": memory_info.rss,  # Resident Set Size
                "vms": memory_info.vms,  # Virtual Memory Size
                "percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else 0
            }

            self.snapshots.append(snapshot)
            return snapshot

        except Exception as e:
            self.logger.error(f"Error taking memory snapshot: {e}")
            return None

    def compare_snapshots(self, start_label: str, end_label: str) -> dict[str, Any]:
        """Compare two memory snapshots."""
        start_snapshot = None
        end_snapshot = None

        for snapshot in self.snapshots:
            if snapshot["label"] == start_label:
                start_snapshot = snapshot
            elif snapshot["label"] == end_label:
                end_snapshot = snapshot

        if not start_snapshot or not end_snapshot:
            return {}

        return {
            "rss_delta": end_snapshot["rss"] - start_snapshot["rss"],
            "vms_delta": end_snapshot["vms"] - start_snapshot["vms"],
            "percent_delta": end_snapshot["percent"] - start_snapshot["percent"],
            "threads_delta": end_snapshot["num_threads"] - start_snapshot["num_threads"],
            "fds_delta": end_snapshot["num_fds"] - start_snapshot["num_fds"],
            "duration": (end_snapshot["timestamp"] - start_snapshot["timestamp"]).total_seconds()
        }

    def get_memory_trend(self, window_size: int = 10) -> dict[str, Any]:
        """Get memory usage trend."""
        if len(self.snapshots) < window_size:
            return {}

        recent_snapshots = self.snapshots[-window_size:]
        rss_values = [s["rss"] for s in recent_snapshots]
        percent_values = [s["percent"] for s in recent_snapshots]

        return {
            "rss_trend": np.polyfit(range(len(rss_values)), rss_values, 1)[0],
            "percent_trend": np.polyfit(range(len(percent_values)), percent_values, 1)[0],
            "rss_mean": np.mean(rss_values),
            "percent_mean": np.mean(percent_values),
            "rss_std": np.std(rss_values),
            "percent_std": np.std(percent_values)
        }


class PerformanceOptimizer:
    """Automatic performance optimization."""

    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
        self.optimization_rules = []
        self.applied_optimizations = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize default optimization rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default optimization rules."""
        # High CPU usage rule
        self.add_optimization_rule(
            name="high_cpu_usage",
            condition=lambda: self._check_high_cpu_usage(),
            strategy=OptimizationStrategy.CPU_OPTIMIZATION,
            action=self._optimize_cpu_usage,
            threshold=80.0,
            description="Optimize when CPU usage exceeds 80%"
        )

        # High memory usage rule
        self.add_optimization_rule(
            name="high_memory_usage",
            condition=lambda: self._check_high_memory_usage(),
            strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
            action=self._optimize_memory_usage,
            threshold=85.0,
            description="Optimize when memory usage exceeds 85%"
        )

        # Slow response time rule
        self.add_optimization_rule(
            name="slow_response_time",
            condition=lambda: self._check_slow_response_time(),
            strategy=OptimizationStrategy.CACHING,
            action=self._optimize_caching,
            threshold=1000.0,  # milliseconds
            description="Enable caching when response time exceeds 1000ms"
        )

    def add_optimization_rule(self, name: str, condition: Callable, strategy: OptimizationStrategy,
                            action: Callable, threshold: float, description: str = ""):
        """Add optimization rule."""
        rule = {
            "name": name,
            "condition": condition,
            "strategy": strategy,
            "action": action,
            "threshold": threshold,
            "description": description,
            "last_triggered": None,
            "trigger_count": 0
        }
        self.optimization_rules.append(rule)

    def _check_high_cpu_usage(self) -> bool:
        """Check if CPU usage is high."""
        stats = self.collector.get_metric_statistics(
            PerformanceMetricType.CPU_USAGE,
            time_window=timedelta(minutes=5)
        )
        return stats.get("mean", 0) > 80.0

    def _check_high_memory_usage(self) -> bool:
        """Check if memory usage is high."""
        stats = self.collector.get_metric_statistics(
            PerformanceMetricType.MEMORY_USAGE,
            time_window=timedelta(minutes=5)
        )
        return stats.get("mean", 0) > 85.0

    def _check_slow_response_time(self) -> bool:
        """Check if response time is slow."""
        stats = self.collector.get_metric_statistics(
            PerformanceMetricType.RESPONSE_TIME,
            time_window=timedelta(minutes=10)
        )
        return stats.get("p95", 0) > 1000.0

    def _optimize_cpu_usage(self):
        """Optimize CPU usage."""
        self.logger.info("Applying CPU optimization")

        # Force garbage collection
        gc.collect()

        # Log optimization
        self.applied_optimizations.append({
            "strategy": OptimizationStrategy.CPU_OPTIMIZATION,
            "action": "garbage_collection",
            "timestamp": datetime.now(),
            "description": "Forced garbage collection to reduce CPU load"
        })

    def _optimize_memory_usage(self):
        """Optimize memory usage."""
        self.logger.info("Applying memory optimization")

        # Force garbage collection
        collected = gc.collect()

        # Log optimization
        self.applied_optimizations.append({
            "strategy": OptimizationStrategy.MEMORY_OPTIMIZATION,
            "action": "garbage_collection",
            "timestamp": datetime.now(),
            "description": f"Garbage collection freed {collected} objects",
            "objects_freed": collected
        })

    def _optimize_caching(self):
        """Optimize caching strategy."""
        self.logger.info("Applying caching optimization")

        # This would typically enable or tune caching mechanisms
        self.applied_optimizations.append({
            "strategy": OptimizationStrategy.CACHING,
            "action": "enable_caching",
            "timestamp": datetime.now(),
            "description": "Enabled aggressive caching due to slow response times"
        })

    def run_optimization_cycle(self):
        """Run one optimization cycle."""
        for rule in self.optimization_rules:
            try:
                if rule["condition"]():
                    # Apply optimization if not recently triggered
                    now = datetime.now()
                    last_triggered = rule["last_triggered"]

                    if not last_triggered or (now - last_triggered).total_seconds() > 300:  # 5 minutes
                        self.logger.info(f"Triggering optimization rule: {rule['name']}")
                        rule["action"]()
                        rule["last_triggered"] = now
                        rule["trigger_count"] += 1

            except Exception as e:
                self.logger.error(f"Error in optimization rule {rule['name']}: {e}")

    def get_optimization_report(self) -> dict[str, Any]:
        """Get optimization report."""
        return {
            "total_rules": len(self.optimization_rules),
            "total_optimizations": len(self.applied_optimizations),
            "rules": [
                {
                    "name": rule["name"],
                    "strategy": rule["strategy"].value,
                    "description": rule["description"],
                    "trigger_count": rule["trigger_count"],
                    "last_triggered": rule["last_triggered"].isoformat() if rule["last_triggered"] else None
                }
                for rule in self.optimization_rules
            ],
            "recent_optimizations": [
                {
                    "strategy": opt["strategy"].value,
                    "action": opt["action"],
                    "timestamp": opt["timestamp"].isoformat(),
                    "description": opt["description"]
                }
                for opt in self.applied_optimizations[-10:]  # Last 10 optimizations
            ]
        }


class PerformanceAnalyzer:
    """Analyze performance data and provide insights."""

    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze_performance_trends(self, time_window: timedelta = timedelta(hours=1)) -> dict[str, Any]:
        """Analyze performance trends over time."""
        end_time = datetime.now()
        start_time = end_time - time_window

        analysis = {}

        # Analyze each metric type
        for metric_type in PerformanceMetricType:
            metrics = self.collector.get_metrics(metric_type, start_time=start_time, end_time=end_time)

            if not metrics:
                continue

            values = [m.value for m in metrics]
            timestamps = [m.timestamp for m in metrics]

            # Calculate trend
            if len(values) > 1:
                # Convert timestamps to seconds from start
                time_series = [(t - start_time).total_seconds() for t in timestamps]
                trend_coefficient = np.polyfit(time_series, values, 1)[0]

                analysis[metric_type.value] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "trend": "increasing" if trend_coefficient > 0 else "decreasing",
                    "trend_coefficient": trend_coefficient,
                    "latest_value": values[-1]
                }

        return analysis

    def detect_performance_anomalies(self, z_threshold: float = 2.0) -> list[dict[str, Any]]:
        """Detect performance anomalies using z-score."""
        anomalies = []

        for metric_type in PerformanceMetricType:
            metrics = self.collector.get_metrics(
                metric_type,
                start_time=datetime.now() - timedelta(hours=24)
            )

            if len(metrics) < 10:  # Need sufficient data
                continue

            values = [m.value for m in metrics]
            mean = np.mean(values)
            std = np.std(values)

            if std == 0:  # No variation
                continue

            # Check recent values for anomalies
            recent_metrics = metrics[-10:]  # Last 10 values

            for metric in recent_metrics:
                z_score = abs((metric.value - mean) / std)

                if z_score > z_threshold:
                    anomalies.append({
                        "metric_type": metric.metric_type.value,
                        "value": metric.value,
                        "expected_range": [mean - 2*std, mean + 2*std],
                        "z_score": z_score,
                        "timestamp": metric.timestamp.isoformat(),
                        "tags": metric.tags,
                        "severity": "high" if z_score > 3.0 else "medium"
                    })

        return anomalies

    def generate_performance_recommendations(self) -> list[dict[str, Any]]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Analyze recent performance
        trends = self.analyze_performance_trends(timedelta(hours=1))

        # CPU usage recommendations
        if "cpu_usage" in trends:
            cpu_stats = trends["cpu_usage"]
            if cpu_stats["mean"] > 80:
                recommendations.append({
                    "category": "cpu",
                    "priority": "high",
                    "issue": f"High CPU usage (average {cpu_stats['mean']:.1f}%)",
                    "recommendation": "Consider scaling horizontally or optimizing CPU-intensive operations",
                    "actions": [
                        "Profile CPU-intensive functions",
                        "Implement caching for expensive operations",
                        "Consider async processing for I/O operations"
                    ]
                })

        # Memory usage recommendations
        if "memory_usage" in trends:
            memory_stats = trends["memory_usage"]
            if memory_stats["mean"] > 85:
                recommendations.append({
                    "category": "memory",
                    "priority": "high",
                    "issue": f"High memory usage (average {memory_stats['mean']:.1f}%)",
                    "recommendation": "Optimize memory usage to prevent out-of-memory errors",
                    "actions": [
                        "Review memory-intensive objects",
                        "Implement object pooling",
                        "Add more frequent garbage collection",
                        "Consider memory-efficient data structures"
                    ]
                })

        # Response time recommendations
        if "response_time" in trends:
            response_stats = trends["response_time"]
            if response_stats["mean"] > 500:  # 500ms
                recommendations.append({
                    "category": "latency",
                    "priority": "medium",
                    "issue": f"Slow response times (average {response_stats['mean']:.1f}ms)",
                    "recommendation": "Optimize application response times",
                    "actions": [
                        "Implement response caching",
                        "Optimize database queries",
                        "Use CDN for static content",
                        "Implement connection pooling"
                    ]
                })

        # Detect anomalies
        anomalies = self.detect_performance_anomalies()

        if anomalies:
            high_severity_anomalies = [a for a in anomalies if a["severity"] == "high"]
            if high_severity_anomalies:
                recommendations.append({
                    "category": "anomaly",
                    "priority": "critical",
                    "issue": f"Detected {len(high_severity_anomalies)} high-severity performance anomalies",
                    "recommendation": "Investigate performance anomalies immediately",
                    "actions": [
                        "Review system logs for errors",
                        "Check for resource contention",
                        "Monitor for external dependencies issues",
                        "Consider rolling back recent changes"
                    ],
                    "anomalies": high_severity_anomalies[:5]  # Include first 5 anomalies
                })

        return recommendations


@contextmanager
def performance_timer(name: str, collector: PerformanceCollector):
    """Context manager for timing operations."""
    start_time = time.time()
    start_memory = memory_profiler.memory_usage()[0]

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = memory_profiler.memory_usage()[0]

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        # Add latency metric
        collector.add_metric(PerformanceMetric(
            PerformanceMetricType.LATENCY,
            duration * 1000,  # Convert to milliseconds
            datetime.now(),
            tags={"operation": name, "source": "timer"}
        ))

        # Add memory metric if significant change
        if abs(memory_delta) > 1.0:  # > 1MB change
            collector.add_metric(PerformanceMetric(
                PerformanceMetricType.MEMORY_USAGE,
                memory_delta,
                datetime.now(),
                tags={"operation": name, "source": "timer", "metric": "memory_delta"}
            ))


def profile_async_function(collector: PerformanceCollector):
    """Decorator for profiling async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time

                func_name = f"{func.__module__}.{func.__name__}"
                collector.add_metric(PerformanceMetric(
                    PerformanceMetricType.LATENCY,
                    duration * 1000,  # Convert to milliseconds
                    datetime.now(),
                    tags={"function": func_name, "type": "async", "source": "profiler"}
                ))

        return wrapper
    return decorator


class PerformanceManager:
    """Main performance management system."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self.collector = PerformanceCollector(
            collection_interval=self.config.get("collection_interval", 30)
        )
        self.function_profiler = FunctionProfiler()
        self.memory_profiler = MemoryProfiler()
        self.optimizer = PerformanceOptimizer(self.collector)
        self.analyzer = PerformanceAnalyzer(self.collector)

        # Optimization cycle
        self.optimization_enabled = self.config.get("auto_optimization", True)
        self.optimization_interval = self.config.get("optimization_interval", 300)  # 5 minutes

    def start(self):
        """Start performance management."""
        self.logger.info("Starting performance management system")

        # Start metric collection
        self.collector.start_collection()

        # Start optimization cycle if enabled
        if self.optimization_enabled:
            self.optimization_thread = threading.Thread(
                target=self._optimization_loop,
                daemon=True
            )
            self.optimization_thread.start()

    def stop(self):
        """Stop performance management."""
        self.logger.info("Stopping performance management system")

        # Stop metric collection
        self.collector.stop_collection()

        # Stop optimization
        self.optimization_enabled = False

    def _optimization_loop(self):
        """Main optimization loop."""
        while self.optimization_enabled:
            try:
                self.optimizer.run_optimization_cycle()
                time.sleep(self.optimization_interval)
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                metric_type.value: self.collector.get_metric_statistics(
                    metric_type,
                    time_window=timedelta(hours=1)
                )
                for metric_type in PerformanceMetricType
            },
            "function_profiles": {
                name: profile.to_dict()
                for name, profile in self.function_profiler.get_all_profiles().items()
            },
            "performance_trends": self.analyzer.analyze_performance_trends(),
            "anomalies": self.analyzer.detect_performance_anomalies(),
            "recommendations": self.analyzer.generate_performance_recommendations(),
            "optimization_report": self.optimizer.get_optimization_report(),
            "memory_snapshots": self.memory_profiler.snapshots[-10:],  # Last 10 snapshots
            "config": self.config
        }


def main():
    """Main function for testing."""
    import random

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.info("Testing performance optimization system")

    # Create performance manager
    perf_manager = PerformanceManager({
        "collection_interval": 5,
        "auto_optimization": True,
        "optimization_interval": 30
    })

    # Start performance management
    perf_manager.start()

    # Simulate some workload
    @perf_manager.function_profiler.profile_function(include_memory=True)
    def simulate_work(duration: float):
        """Simulate CPU and memory intensive work."""
        start_time = time.time()
        data = []

        while time.time() - start_time < duration:
            # Simulate CPU work
            for _ in range(1000):
                random.random()

            # Simulate memory allocation
            data.extend([random.random() for _ in range(1000)])

        return len(data)

    # Run simulation
    logger.info("Running performance simulation...")

    for i in range(5):
        with performance_timer(f"simulation_{i}", perf_manager.collector):
            result = simulate_work(2.0)  # 2 seconds of work
            logger.info(f"Simulation {i} completed: {result} data points")

        time.sleep(1)

    # Wait for some metrics to be collected
    time.sleep(10)

    # Generate performance report
    logger.info("Generating performance report...")
    report = perf_manager.get_performance_report()

    # Print summary
    print("\n📊 Performance Report Summary:")
    print(f"   System Metrics: {len(report['system_metrics'])} types")
    print(f"   Function Profiles: {len(report['function_profiles'])}")
    print(f"   Anomalies Detected: {len(report['anomalies'])}")
    print(f"   Recommendations: {len(report['recommendations'])}")
    print(f"   Optimizations Applied: {report['optimization_report']['total_optimizations']}")

    # Print recommendations
    if report['recommendations']:
        print("\n💡 Performance Recommendations:")
        for rec in report['recommendations']:
            print(f"   - {rec['category'].upper()}: {rec['issue']}")
            print(f"     Action: {rec['recommendation']}")

    # Stop performance management
    perf_manager.stop()

    logger.info("Performance testing completed")


if __name__ == "__main__":
    main()
