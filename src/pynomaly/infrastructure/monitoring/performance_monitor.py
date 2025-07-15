"""
Advanced Performance Monitoring and Bottleneck Detection System.

This module provides comprehensive performance monitoring, real-time bottleneck detection,
and automated performance optimization recommendations for the Pynomaly system.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import psutil

try:
    import cProfile
    import pstats
    from io import StringIO

    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

try:
    import line_profiler

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""

    alert_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: str = "medium"  # low, medium, high, critical
    category: str = "performance"  # performance, memory, cpu, io, bottleneck

    title: str = ""
    description: str = ""
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0

    component: str = ""  # Which component triggered the alert
    suggested_actions: list[str] = field(default_factory=list)

    resolved: bool = False
    resolved_timestamp: datetime | None = None
    resolution_notes: str = ""


@dataclass
class BottleneckAnalysis:
    """Bottleneck analysis results."""

    analysis_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Identified bottlenecks
    bottlenecks: list[dict[str, Any]] = field(default_factory=list)

    # Resource utilization
    cpu_bottleneck: bool = False
    memory_bottleneck: bool = False
    io_bottleneck: bool = False
    algorithm_bottleneck: bool = False

    # Performance metrics
    overall_performance_score: float = 0.0
    bottleneck_severity: str = "low"  # low, medium, high, critical

    # Optimization recommendations
    immediate_actions: list[str] = field(default_factory=list)
    optimization_opportunities: list[str] = field(default_factory=list)

    # Detailed analysis
    profiling_data: dict[str, Any] = field(default_factory=dict)
    resource_analysis: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    metrics_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Execution metrics
    execution_time: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0

    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    disk_io_rate: float = 0.0
    network_io_rate: float = 0.0

    # System metrics
    load_average: list[float] = field(default_factory=list)
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0

    # Performance scores
    overall_score: float = 0.0
    efficiency_score: float = 0.0
    reliability_score: float = 0.0

    # Custom metrics
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    # Metadata
    operation_name: str = ""
    component: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class PerformanceThresholds:
    """Performance monitoring thresholds."""

    # Execution time thresholds (seconds)
    max_execution_time: float = 300.0
    warning_execution_time: float = 60.0

    # Memory thresholds (MB)
    max_memory_usage: float = 4096.0
    warning_memory_usage: float = 2048.0
    memory_growth_rate_threshold: float = 100.0  # MB per minute

    # CPU thresholds (percentage)
    max_cpu_usage: float = 95.0
    warning_cpu_usage: float = 80.0
    cpu_sustained_threshold: float = 60.0  # seconds at high CPU

    # I/O thresholds
    max_disk_io_rate: float = 1000.0  # MB/s
    warning_disk_io_rate: float = 500.0  # MB/s

    # Throughput thresholds
    min_throughput: float = 10.0  # samples/second
    throughput_degradation_threshold: float = 0.5  # 50% degradation

    # Cache thresholds
    min_cache_hit_ratio: float = 0.7
    warning_cache_hit_ratio: float = 0.8

    # GC thresholds
    max_gc_time_ratio: float = 0.1  # 10% of total time
    max_gc_frequency: int = 100  # collections per minute


class RealTimeResourceMonitor:
    """Real-time system resource monitoring."""

    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.monitoring_active = False

        # Data storage
        self.cpu_history = deque(maxlen=300)  # 5 minutes at 1s intervals
        self.memory_history = deque(maxlen=300)
        self.io_history = deque(maxlen=300)
        self.network_history = deque(maxlen=300)

        # Monitoring thread
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Current metrics
        self.current_metrics = {}
        self._metrics_lock = threading.Lock()

        logger.info("Real-time resource monitor initialized")

    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._stop_event.clear()

        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()

        logger.info("Real-time monitoring started")

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        logger.info("Real-time monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.sampling_interval):
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()

                # Store in history
                timestamp = time.time()

                self.cpu_history.append((timestamp, metrics["cpu_percent"]))
                self.memory_history.append((timestamp, metrics["memory_mb"]))
                self.io_history.append((timestamp, metrics["disk_io_mb_per_sec"]))
                self.network_history.append(
                    (timestamp, metrics["network_bytes_per_sec"])
                )

                # Update current metrics
                with self._metrics_lock:
                    self.current_metrics = metrics

            except Exception as e:
                logger.warning(f"Error in monitoring loop: {e}")

    def _collect_system_metrics(self) -> dict[str, Any]:
        """Collect current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_times = psutil.cpu_times()

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()

        # Network metrics
        network_io = psutil.net_io_counters()

        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            # CPU
            "cpu_percent": cpu_percent,
            "cpu_user_time": cpu_times.user,
            "cpu_system_time": cpu_times.system,
            "cpu_idle_time": cpu_times.idle,
            # Memory
            "memory_mb": memory.used / 1024 / 1024,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / 1024 / 1024,
            "swap_mb": swap.used / 1024 / 1024,
            "swap_percent": swap.percent,
            # Process memory
            "process_memory_mb": process_memory.rss / 1024 / 1024,
            "process_memory_vms_mb": process_memory.vms / 1024 / 1024,
            # Disk I/O
            "disk_read_mb": disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
            "disk_write_mb": disk_io.write_bytes / 1024 / 1024 if disk_io else 0,
            "disk_io_mb_per_sec": self._calculate_io_rate(disk_io) if disk_io else 0,
            # Network
            "network_bytes_sent": network_io.bytes_sent if network_io else 0,
            "network_bytes_recv": network_io.bytes_recv if network_io else 0,
            "network_bytes_per_sec": self._calculate_network_rate(network_io)
            if network_io
            else 0,
            # Load average (Unix-like systems)
            "load_average": os.getloadavg() if hasattr(os, "getloadavg") else [0, 0, 0],
            # Garbage collection
            "gc_stats": gc.get_stats() if hasattr(gc, "get_stats") else [],
        }

    def _calculate_io_rate(self, current_io) -> float:
        """Calculate current I/O rate."""
        # This would track previous values to calculate rate
        # Simplified implementation
        return 0.0

    def _calculate_network_rate(self, current_net) -> float:
        """Calculate current network rate."""
        # This would track previous values to calculate rate
        # Simplified implementation
        return 0.0

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current system metrics."""
        with self._metrics_lock:
            return self.current_metrics.copy()

    def get_cpu_trend(self, duration_minutes: int = 5) -> dict[str, Any]:
        """Get CPU usage trend over specified duration."""
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_data = [(t, v) for t, v in self.cpu_history if t >= cutoff_time]

        if not recent_data:
            return {"trend": "no_data"}

        values = [v for _, v in recent_data]

        return {
            "trend": "increasing" if values[-1] > values[0] else "decreasing",
            "average": np.mean(values),
            "maximum": max(values),
            "minimum": min(values),
            "current": values[-1],
            "data_points": len(values),
        }

    def get_memory_trend(self, duration_minutes: int = 5) -> dict[str, Any]:
        """Get memory usage trend over specified duration."""
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_data = [(t, v) for t, v in self.memory_history if t >= cutoff_time]

        if not recent_data:
            return {"trend": "no_data"}

        values = [v for _, v in recent_data]

        return {
            "trend": "increasing" if values[-1] > values[0] else "decreasing",
            "average": np.mean(values),
            "maximum": max(values),
            "minimum": min(values),
            "current": values[-1],
            "growth_rate_mb_per_min": (values[-1] - values[0]) / duration_minutes
            if len(values) > 1
            else 0,
            "data_points": len(values),
        }


class BottleneckDetector:
    """Advanced bottleneck detection and analysis."""

    def __init__(
        self,
        resource_monitor: RealTimeResourceMonitor,
        thresholds: PerformanceThresholds,
    ):
        self.resource_monitor = resource_monitor
        self.thresholds = thresholds

        # Detection state
        self.active_bottlenecks: dict[str, BottleneckAnalysis] = {}
        self.bottleneck_history: list[BottleneckAnalysis] = []

        logger.info("Bottleneck detector initialized")

    async def detect_bottlenecks(
        self,
        execution_metrics: dict[str, Any] | None = None,
        profiling_data: dict[str, Any] | None = None,
    ) -> BottleneckAnalysis:
        """Detect and analyze performance bottlenecks."""
        analysis = BottleneckAnalysis()

        # Get current system state
        current_metrics = self.resource_monitor.get_current_metrics()
        cpu_trend = self.resource_monitor.get_cpu_trend()
        memory_trend = self.resource_monitor.get_memory_trend()

        # Analyze different bottleneck categories
        analysis.cpu_bottleneck = self._detect_cpu_bottleneck(
            current_metrics, cpu_trend
        )
        analysis.memory_bottleneck = self._detect_memory_bottleneck(
            current_metrics, memory_trend
        )
        analysis.io_bottleneck = self._detect_io_bottleneck(current_metrics)

        if execution_metrics:
            analysis.algorithm_bottleneck = self._detect_algorithm_bottleneck(
                execution_metrics
            )

        # Identify specific bottlenecks
        bottlenecks = []

        if analysis.cpu_bottleneck:
            bottlenecks.extend(
                self._analyze_cpu_bottlenecks(current_metrics, cpu_trend)
            )

        if analysis.memory_bottleneck:
            bottlenecks.extend(
                self._analyze_memory_bottlenecks(current_metrics, memory_trend)
            )

        if analysis.io_bottleneck:
            bottlenecks.extend(self._analyze_io_bottlenecks(current_metrics))

        if analysis.algorithm_bottleneck and execution_metrics:
            bottlenecks.extend(self._analyze_algorithm_bottlenecks(execution_metrics))

        analysis.bottlenecks = bottlenecks

        # Calculate overall performance score
        analysis.overall_performance_score = self._calculate_performance_score(analysis)
        analysis.bottleneck_severity = self._determine_severity(analysis)

        # Generate recommendations
        analysis.immediate_actions = self._generate_immediate_actions(analysis)
        analysis.optimization_opportunities = self._generate_optimization_opportunities(
            analysis
        )

        # Store profiling data if available
        if profiling_data:
            analysis.profiling_data = profiling_data

        # Store resource analysis
        analysis.resource_analysis = {
            "current_metrics": current_metrics,
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
        }

        # Update state
        self.bottleneck_history.append(analysis)
        if len(self.bottleneck_history) > 100:  # Keep last 100 analyses
            self.bottleneck_history.pop(0)

        return analysis

    def _detect_cpu_bottleneck(
        self, metrics: dict[str, Any], trend: dict[str, Any]
    ) -> bool:
        """Detect CPU bottlenecks."""
        current_cpu = metrics.get("cpu_percent", 0)
        avg_cpu = trend.get("average", 0)

        return (
            current_cpu > self.thresholds.max_cpu_usage
            or avg_cpu > self.thresholds.warning_cpu_usage
        )

    def _detect_memory_bottleneck(
        self, metrics: dict[str, Any], trend: dict[str, Any]
    ) -> bool:
        """Detect memory bottlenecks."""
        current_memory = metrics.get("memory_mb", 0)
        memory_percent = metrics.get("memory_percent", 0)
        growth_rate = trend.get("growth_rate_mb_per_min", 0)

        return (
            current_memory > self.thresholds.max_memory_usage
            or memory_percent > 90
            or growth_rate > self.thresholds.memory_growth_rate_threshold
        )

    def _detect_io_bottleneck(self, metrics: dict[str, Any]) -> bool:
        """Detect I/O bottlenecks."""
        io_rate = metrics.get("disk_io_mb_per_sec", 0)

        return io_rate > self.thresholds.max_disk_io_rate

    def _detect_algorithm_bottleneck(self, execution_metrics: dict[str, Any]) -> bool:
        """Detect algorithm-specific bottlenecks."""
        execution_time = execution_metrics.get("execution_time", 0)
        throughput = execution_metrics.get("throughput", float("inf"))

        return (
            execution_time > self.thresholds.max_execution_time
            or throughput < self.thresholds.min_throughput
        )

    def _analyze_cpu_bottlenecks(
        self, metrics: dict[str, Any], trend: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Analyze CPU bottlenecks in detail."""
        bottlenecks = []

        cpu_percent = metrics.get("cpu_percent", 0)
        avg_cpu = trend.get("average", 0)

        if cpu_percent > self.thresholds.max_cpu_usage:
            bottlenecks.append(
                {
                    "type": "cpu_overload",
                    "severity": "high",
                    "description": f"CPU usage at {cpu_percent:.1f}% exceeds threshold of {self.thresholds.max_cpu_usage}%",
                    "current_value": cpu_percent,
                    "threshold": self.thresholds.max_cpu_usage,
                    "impact": "severe_performance_degradation",
                }
            )

        if avg_cpu > self.thresholds.warning_cpu_usage:
            bottlenecks.append(
                {
                    "type": "cpu_sustained_high_usage",
                    "severity": "medium",
                    "description": f"Average CPU usage at {avg_cpu:.1f}% indicates sustained high load",
                    "current_value": avg_cpu,
                    "threshold": self.thresholds.warning_cpu_usage,
                    "impact": "performance_degradation",
                }
            )

        return bottlenecks

    def _analyze_memory_bottlenecks(
        self, metrics: dict[str, Any], trend: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Analyze memory bottlenecks in detail."""
        bottlenecks = []

        memory_mb = metrics.get("memory_mb", 0)
        memory_percent = metrics.get("memory_percent", 0)
        growth_rate = trend.get("growth_rate_mb_per_min", 0)

        if memory_mb > self.thresholds.max_memory_usage:
            bottlenecks.append(
                {
                    "type": "memory_overload",
                    "severity": "high",
                    "description": f"Memory usage at {memory_mb:.1f}MB exceeds threshold of {self.thresholds.max_memory_usage}MB",
                    "current_value": memory_mb,
                    "threshold": self.thresholds.max_memory_usage,
                    "impact": "risk_of_out_of_memory",
                }
            )

        if memory_percent > 90:
            bottlenecks.append(
                {
                    "type": "system_memory_pressure",
                    "severity": "high",
                    "description": f"System memory usage at {memory_percent:.1f}% indicates memory pressure",
                    "current_value": memory_percent,
                    "threshold": 90,
                    "impact": "system_instability_risk",
                }
            )

        if growth_rate > self.thresholds.memory_growth_rate_threshold:
            bottlenecks.append(
                {
                    "type": "memory_leak_suspected",
                    "severity": "medium",
                    "description": f"Memory growing at {growth_rate:.1f}MB/min suggests potential memory leak",
                    "current_value": growth_rate,
                    "threshold": self.thresholds.memory_growth_rate_threshold,
                    "impact": "progressive_degradation",
                }
            )

        return bottlenecks

    def _analyze_io_bottlenecks(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze I/O bottlenecks in detail."""
        bottlenecks = []

        io_rate = metrics.get("disk_io_mb_per_sec", 0)

        if io_rate > self.thresholds.max_disk_io_rate:
            bottlenecks.append(
                {
                    "type": "disk_io_overload",
                    "severity": "medium",
                    "description": f"Disk I/O rate at {io_rate:.1f}MB/s exceeds threshold",
                    "current_value": io_rate,
                    "threshold": self.thresholds.max_disk_io_rate,
                    "impact": "io_wait_performance_degradation",
                }
            )

        return bottlenecks

    def _analyze_algorithm_bottlenecks(
        self, execution_metrics: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Analyze algorithm-specific bottlenecks."""
        bottlenecks = []

        execution_time = execution_metrics.get("execution_time", 0)
        throughput = execution_metrics.get("throughput", float("inf"))

        if execution_time > self.thresholds.max_execution_time:
            bottlenecks.append(
                {
                    "type": "algorithm_slow_execution",
                    "severity": "high",
                    "description": f"Algorithm execution time {execution_time:.1f}s exceeds threshold",
                    "current_value": execution_time,
                    "threshold": self.thresholds.max_execution_time,
                    "impact": "user_experience_degradation",
                }
            )

        if throughput < self.thresholds.min_throughput:
            bottlenecks.append(
                {
                    "type": "algorithm_low_throughput",
                    "severity": "medium",
                    "description": f"Algorithm throughput {throughput:.1f} samples/s below threshold",
                    "current_value": throughput,
                    "threshold": self.thresholds.min_throughput,
                    "impact": "processing_capacity_limitation",
                }
            )

        return bottlenecks

    def _calculate_performance_score(self, analysis: BottleneckAnalysis) -> float:
        """Calculate overall performance score (0-100)."""
        base_score = 100.0

        # Deduct points for each bottleneck
        for bottleneck in analysis.bottlenecks:
            severity = bottleneck.get("severity", "low")
            if severity == "critical":
                base_score -= 40
            elif severity == "high":
                base_score -= 25
            elif severity == "medium":
                base_score -= 15
            elif severity == "low":
                base_score -= 5

        return max(0.0, base_score)

    def _determine_severity(self, analysis: BottleneckAnalysis) -> str:
        """Determine overall bottleneck severity."""
        if analysis.overall_performance_score < 30:
            return "critical"
        elif analysis.overall_performance_score < 50:
            return "high"
        elif analysis.overall_performance_score < 70:
            return "medium"
        else:
            return "low"

    def _generate_immediate_actions(self, analysis: BottleneckAnalysis) -> list[str]:
        """Generate immediate action recommendations."""
        actions = []

        if analysis.cpu_bottleneck:
            actions.append(
                "Consider reducing CPU-intensive operations or enabling parallel processing"
            )

        if analysis.memory_bottleneck:
            actions.append("Implement garbage collection or reduce memory allocation")

        if analysis.io_bottleneck:
            actions.append("Optimize I/O operations or implement caching")

        if analysis.algorithm_bottleneck:
            actions.append(
                "Review algorithm implementation for optimization opportunities"
            )

        # Add severity-specific actions
        if analysis.bottleneck_severity == "critical":
            actions.insert(0, "URGENT: Consider stopping non-essential processes")

        return actions

    def _generate_optimization_opportunities(
        self, analysis: BottleneckAnalysis
    ) -> list[str]:
        """Generate optimization opportunity recommendations."""
        opportunities = []

        for bottleneck in analysis.bottlenecks:
            bottleneck_type = bottleneck.get("type", "")

            if "cpu" in bottleneck_type:
                opportunities.extend(
                    [
                        "Implement multi-threading or multiprocessing",
                        "Use vectorized operations where possible",
                        "Consider GPU acceleration for compute-intensive tasks",
                    ]
                )

            elif "memory" in bottleneck_type:
                opportunities.extend(
                    [
                        "Implement memory pooling",
                        "Use memory-mapped files for large datasets",
                        "Optimize data structures for memory efficiency",
                    ]
                )

            elif "io" in bottleneck_type:
                opportunities.extend(
                    [
                        "Implement asynchronous I/O operations",
                        "Use compression to reduce I/O volume",
                        "Implement intelligent caching strategies",
                    ]
                )

        # Remove duplicates while preserving order
        return list(dict.fromkeys(opportunities))


class PerformanceMonitor:
    """Main performance monitoring coordinator."""

    def __init__(
        self,
        thresholds: PerformanceThresholds | None = None,
        storage_path: Path | None = None,
        enable_alerts: bool = True,
    ):
        self.thresholds = thresholds or PerformanceThresholds()
        self.storage_path = storage_path or Path("performance_monitoring")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Core components
        self.resource_monitor = RealTimeResourceMonitor()
        self.bottleneck_detector = BottleneckDetector(
            self.resource_monitor, self.thresholds
        )

        # State management
        self.monitoring_active = False
        self.alerts: list[PerformanceAlert] = []
        self.enable_alerts = enable_alerts

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []

        logger.info("Performance monitor initialized")

    async def start_monitoring(self):
        """Start comprehensive performance monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.resource_monitor.start_monitoring()

        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        self.resource_monitor.stop_monitoring()

        logger.info("Performance monitoring stopped")

    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring specific operations."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Enable tracemalloc if available
        if tracemalloc.is_tracing():
            trace_start = tracemalloc.take_snapshot()
        else:
            tracemalloc.start()
            trace_start = None

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Calculate metrics
            execution_time = end_time - start_time
            memory_growth = end_memory - start_memory

            # Get trace data
            if tracemalloc.is_tracing():
                trace_end = tracemalloc.take_snapshot()
                if trace_start:
                    top_stats = trace_end.compare_to(trace_start, "lineno")
                else:
                    top_stats = trace_end.statistics("lineno")
            else:
                top_stats = []

            # Create execution metrics
            execution_metrics = {
                "operation_name": operation_name,
                "execution_time": execution_time,
                "memory_growth_mb": memory_growth,
                "start_memory_mb": start_memory,
                "end_memory_mb": end_memory,
                "trace_stats": top_stats[:10]
                if top_stats
                else [],  # Top 10 memory allocations
            }

            # Trigger bottleneck detection
            asyncio.create_task(self._analyze_operation_performance(execution_metrics))

    async def _analyze_operation_performance(self, execution_metrics: dict[str, Any]):
        """Analyze performance of a monitored operation."""
        try:
            # Detect bottlenecks
            analysis = await self.bottleneck_detector.detect_bottlenecks(
                execution_metrics=execution_metrics
            )

            # Generate alerts if needed
            if self.enable_alerts:
                await self._generate_performance_alerts(analysis, execution_metrics)

            # Store performance data
            self.performance_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "execution_metrics": execution_metrics,
                    "analysis": analysis,
                }
            )

            # Limit history size
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]

        except Exception as e:
            logger.error(f"Error analyzing operation performance: {e}")

    async def _generate_performance_alerts(
        self, analysis: BottleneckAnalysis, execution_metrics: dict[str, Any]
    ):
        """Generate performance alerts based on analysis."""
        operation_name = execution_metrics.get("operation_name", "unknown")

        # Generate alerts for each bottleneck
        for bottleneck in analysis.bottlenecks:
            alert = PerformanceAlert(
                severity=bottleneck.get("severity", "medium"),
                category="bottleneck",
                title=f"Performance Bottleneck in {operation_name}",
                description=bottleneck.get("description", ""),
                metric_name=bottleneck.get("type", ""),
                current_value=bottleneck.get("current_value", 0),
                threshold_value=bottleneck.get("threshold", 0),
                component=operation_name,
                suggested_actions=analysis.immediate_actions,
            )

            self.alerts.append(alert)

            # Log alert
            logger.warning(f"Performance Alert: {alert.title} - {alert.description}")

        # Generate alert for overall performance if score is low
        if analysis.overall_performance_score < 50:
            alert = PerformanceAlert(
                severity="high"
                if analysis.overall_performance_score < 30
                else "medium",
                category="performance",
                title=f"Low Performance Score in {operation_name}",
                description=f"Overall performance score: {analysis.overall_performance_score:.1f}/100",
                metric_name="performance_score",
                current_value=analysis.overall_performance_score,
                threshold_value=50,
                component=operation_name,
                suggested_actions=analysis.optimization_opportunities,
            )

            self.alerts.append(alert)

    def get_active_alerts(self, severity: str | None = None) -> list[PerformanceAlert]:
        """Get active performance alerts."""
        active_alerts = [alert for alert in self.alerts if not alert.resolved]

        if severity:
            active_alerts = [
                alert for alert in active_alerts if alert.severity == severity
            ]

        return active_alerts

    def resolve_alert(self, alert_id: UUID, resolution_notes: str = ""):
        """Resolve a performance alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_timestamp = datetime.utcnow()
                alert.resolution_notes = resolution_notes
                logger.info(f"Alert {alert_id} resolved: {resolution_notes}")
                break

    def get_performance_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get performance summary for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # Filter recent data
        recent_history = [
            entry
            for entry in self.performance_history
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff_time
        ]

        if not recent_history:
            return {"message": "No performance data available"}

        # Calculate summary statistics
        execution_times = [
            entry["execution_metrics"].get("execution_time", 0)
            for entry in recent_history
        ]

        memory_growths = [
            entry["execution_metrics"].get("memory_growth_mb", 0)
            for entry in recent_history
        ]

        performance_scores = [
            entry["analysis"].overall_performance_score for entry in recent_history
        ]

        # Count bottlenecks by type
        bottleneck_counts = defaultdict(int)
        for entry in recent_history:
            for bottleneck in entry["analysis"].bottlenecks:
                bottleneck_counts[bottleneck.get("type", "unknown")] += 1

        # Count alerts by severity
        alert_counts = defaultdict(int)
        recent_alerts = [
            alert for alert in self.alerts if alert.timestamp >= cutoff_time
        ]
        for alert in recent_alerts:
            alert_counts[alert.severity] += 1

        return {
            "period_hours": hours,
            "total_operations": len(recent_history),
            "execution_time_stats": {
                "average": np.mean(execution_times),
                "median": np.median(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "std": np.std(execution_times),
            },
            "memory_growth_stats": {
                "average": np.mean(memory_growths),
                "median": np.median(memory_growths),
                "total": sum(memory_growths),
            },
            "performance_score_stats": {
                "average": np.mean(performance_scores),
                "median": np.median(performance_scores),
                "min": min(performance_scores),
                "max": max(performance_scores),
            },
            "bottleneck_counts": dict(bottleneck_counts),
            "alert_counts": dict(alert_counts),
            "current_system_metrics": self.resource_monitor.get_current_metrics(),
        }

    async def generate_performance_report(self, output_path: Path) -> Path:
        """Generate comprehensive performance report."""
        report_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "monitoring_active": self.monitoring_active,
            "thresholds": {
                "max_execution_time": self.thresholds.max_execution_time,
                "max_memory_usage": self.thresholds.max_memory_usage,
                "max_cpu_usage": self.thresholds.max_cpu_usage,
                "min_throughput": self.thresholds.min_throughput,
            },
            "summary_24h": self.get_performance_summary(24),
            "summary_7d": self.get_performance_summary(24 * 7),
            "active_alerts": [
                {
                    "alert_id": str(alert.alert_id),
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "title": alert.title,
                    "description": alert.description,
                    "component": alert.component,
                }
                for alert in self.get_active_alerts()
            ],
            "system_info": {
                "cpu_cores": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "disk_space_gb": psutil.disk_usage("/").total / 1024**3,
            },
        }

        # Save report
        report_path = (
            output_path
            / f"performance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            import json

            json.dump(report_data, f, indent=2)

        logger.info(f"Performance report generated: {report_path}")
        return report_path


class PerformanceTracker:
    """Simple performance tracker for operations."""

    def __init__(self):
        self.metrics: list[PerformanceMetrics] = []
        self.start_time: float | None = None
        self.start_memory: float | None = None

    def start(self, operation_name: str = ""):
        """Start tracking an operation."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.operation_name = operation_name

    def stop(self) -> PerformanceMetrics:
        """Stop tracking and return metrics."""
        if self.start_time is None:
            raise ValueError("Tracker not started")

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        metrics = PerformanceMetrics(
            operation_name=getattr(self, "operation_name", ""),
            execution_time=end_time - self.start_time,
            memory_usage_mb=end_memory - self.start_memory,
            timestamp=datetime.utcnow(),
        )

        self.metrics.append(metrics)
        self.start_time = None
        self.start_memory = None

        return metrics

    def get_metrics(self) -> list[PerformanceMetrics]:
        """Get all tracked metrics."""
        return self.metrics.copy()

    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()


# Context manager functions for performance monitoring
@contextmanager
def monitor_performance(operation_name: str = ""):
    """Context manager for monitoring operation performance."""
    tracker = PerformanceTracker()
    tracker.start(operation_name)
    try:
        yield tracker
    finally:
        tracker.stop()


@contextmanager
def monitor_async_performance(operation_name: str = ""):
    """Context manager for monitoring async operation performance."""
    tracker = PerformanceTracker()
    tracker.start(operation_name)
    try:
        yield tracker
    finally:
        tracker.stop()


# Factory function for easy instantiation
def create_performance_monitor(
    storage_path: Path | None = None,
    enable_alerts: bool = True,
    custom_thresholds: dict[str, float] | None = None,
) -> PerformanceMonitor:
    """Create a performance monitor with sensible defaults."""
    thresholds = PerformanceThresholds()

    if custom_thresholds:
        for key, value in custom_thresholds.items():
            if hasattr(thresholds, key):
                setattr(thresholds, key, value)

    return PerformanceMonitor(
        thresholds=thresholds, storage_path=storage_path, enable_alerts=enable_alerts
    )
