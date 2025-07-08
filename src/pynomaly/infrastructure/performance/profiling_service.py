"""Advanced performance profiling and optimization service for anomaly detection."""

from __future__ import annotations

import asyncio
import cProfile
import functools
import gc
import inspect
import logging
import pstats
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import psutil

# Optional profiling libraries
try:
    import line_profiler

    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    import memory_profiler

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import py_spy

    PY_SPY_AVAILABLE = True
except ImportError:
    PY_SPY_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ProfilerType(Enum):
    """Types of profilers available."""

    CPROFILE = "cprofile"
    LINE_PROFILER = "line_profiler"
    MEMORY_PROFILER = "memory_profiler"
    TRACEMALLOC = "tracemalloc"
    RESOURCE_MONITOR = "resource_monitor"
    ASYNC_PROFILER = "async_profiler"


class MetricType(Enum):
    """Types of performance metrics."""

    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    IO_OPERATIONS = "io_operations"
    CACHE_HITS = "cache_hits"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    context: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Result of performance profiling."""

    profile_id: str
    profiler_type: ProfilerType
    function_name: str
    execution_time_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    metrics: list[PerformanceMetric]
    stats_summary: str
    detailed_stats: dict[str, Any]
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PerformanceConfig:
    """Configuration for performance profiling."""

    # Profiling settings
    enable_profiling: bool = True
    default_profiler: ProfilerType = ProfilerType.CPROFILE
    sample_rate: float = 0.01  # Sample 1% of requests
    max_profile_duration_seconds: int = 300

    # Memory tracking
    enable_memory_tracking: bool = True
    memory_profiler_precision: int = 3
    tracemalloc_top_stats: int = 10

    # Resource monitoring
    enable_resource_monitoring: bool = True
    monitoring_interval_seconds: float = 1.0
    resource_alert_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_io_mb_per_sec": 100.0,
            "network_io_mb_per_sec": 50.0,
        }
    )

    # Output settings
    output_directory: str = "/tmp/pynomaly_profiles"
    save_detailed_profiles: bool = True
    max_profile_files: int = 100

    # Optimization settings
    enable_auto_optimization: bool = False
    optimization_trigger_threshold: float = 2.0  # 2x performance degradation
    gc_optimization: bool = True

    # Async profiling
    enable_async_profiling: bool = True
    async_stack_depth: int = 50


class FunctionProfiler:
    """Profiler for individual functions."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._active_profiles: dict[str, Any] = {}

    def profile_function(
        self,
        profiler_type: ProfilerType = ProfilerType.CPROFILE,
        save_stats: bool = True,
    ):
        """Decorator for profiling function execution."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.config.enable_profiling:
                    return func(*args, **kwargs)

                # Sample based on rate
                if np.random.random() > self.config.sample_rate:
                    return func(*args, **kwargs)

                profile_id = str(uuid4())
                start_time = time.time()
                start_memory = self._get_memory_usage()

                try:
                    if profiler_type == ProfilerType.CPROFILE:
                        result = self._profile_with_cprofile(
                            func, args, kwargs, profile_id, save_stats
                        )
                    elif profiler_type == ProfilerType.LINE_PROFILER:
                        result = self._profile_with_line_profiler(
                            func, args, kwargs, profile_id, save_stats
                        )
                    elif profiler_type == ProfilerType.MEMORY_PROFILER:
                        result = self._profile_with_memory_profiler(
                            func, args, kwargs, profile_id, save_stats
                        )
                    else:
                        # Default execution
                        result = func(*args, **kwargs)

                    # Calculate metrics
                    execution_time = time.time() - start_time
                    end_memory = self._get_memory_usage()
                    memory_delta = end_memory - start_memory

                    # Log performance metrics
                    self.logger.info(
                        f"Function {func.__name__} executed in {execution_time:.3f}s, "
                        f"memory delta: {memory_delta:.2f}MB"
                    )

                    return result

                except Exception as e:
                    self.logger.error(f"Error profiling function {func.__name__}: {e}")
                    # Return original function result on profiling error
                    return func(*args, **kwargs)

            # Async wrapper
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if (
                    not self.config.enable_profiling
                    or not self.config.enable_async_profiling
                ):
                    return await func(*args, **kwargs)

                if np.random.random() > self.config.sample_rate:
                    return await func(*args, **kwargs)

                profile_id = str(uuid4())
                start_time = time.time()
                start_memory = self._get_memory_usage()

                try:
                    result = await self._profile_async_function(
                        func, args, kwargs, profile_id
                    )

                    execution_time = time.time() - start_time
                    end_memory = self._get_memory_usage()
                    memory_delta = end_memory - start_memory

                    self.logger.info(
                        f"Async function {func.__name__} executed in {execution_time:.3f}s, "
                        f"memory delta: {memory_delta:.2f}MB"
                    )

                    return result

                except Exception as e:
                    self.logger.error(
                        f"Error profiling async function {func.__name__}: {e}"
                    )
                    return await func(*args, **kwargs)

            # Return appropriate wrapper based on function type
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper

        return decorator

    def _profile_with_cprofile(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        profile_id: str,
        save_stats: bool,
    ) -> Any:
        """Profile function with cProfile."""
        profiler = cProfile.Profile()

        profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()

        if save_stats:
            self._save_cprofile_stats(profiler, func.__name__, profile_id)

        return result

    def _profile_with_line_profiler(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        profile_id: str,
        save_stats: bool,
    ) -> Any:
        """Profile function with line_profiler."""
        if not LINE_PROFILER_AVAILABLE:
            self.logger.warning("line_profiler not available, falling back to cProfile")
            return self._profile_with_cprofile(
                func, args, kwargs, profile_id, save_stats
            )

        profiler = line_profiler.LineProfiler()
        profiler.add_function(func)

        profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()

        if save_stats:
            self._save_line_profiler_stats(profiler, func.__name__, profile_id)

        return result

    def _profile_with_memory_profiler(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        profile_id: str,
        save_stats: bool,
    ) -> Any:
        """Profile function with memory_profiler."""
        if not MEMORY_PROFILER_AVAILABLE:
            self.logger.warning("memory_profiler not available, using tracemalloc")
            return self._profile_with_tracemalloc(
                func, args, kwargs, profile_id, save_stats
            )

        # Use memory_profiler's memory usage tracking
        start_memory = memory_profiler.memory_usage()[0]

        result = func(*args, **kwargs)

        end_memory = memory_profiler.memory_usage()[0]
        memory_delta = end_memory - start_memory

        if save_stats:
            self._save_memory_profile_stats(
                func.__name__, profile_id, start_memory, end_memory, memory_delta
            )

        return result

    def _profile_with_tracemalloc(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        profile_id: str,
        save_stats: bool,
    ) -> Any:
        """Profile function with tracemalloc."""
        tracemalloc.start()

        try:
            result = func(*args, **kwargs)
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        if save_stats:
            self._save_tracemalloc_stats(func.__name__, profile_id, current, peak)

        return result

    async def _profile_async_function(
        self, func: Callable, args: tuple, kwargs: dict, profile_id: str
    ) -> Any:
        """Profile async function execution."""
        # Track task creation and completion
        task_start = time.time()
        loop = asyncio.get_event_loop()

        # Monitor event loop lag
        loop_start = loop.time()

        result = await func(*args, **kwargs)

        loop_end = loop.time()
        task_end = time.time()

        # Calculate async-specific metrics
        execution_time = task_end - task_start
        loop_time = loop_end - loop_start

        self.logger.debug(
            f"Async function {func.__name__}: "
            f"execution_time={execution_time:.3f}s, "
            f"loop_time={loop_time:.3f}s"
        )

        return result

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        else:
            # Fallback to basic measurement
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def _save_cprofile_stats(
        self, profiler: cProfile.Profile, func_name: str, profile_id: str
    ) -> None:
        """Save cProfile statistics."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save binary stats
            stats_file = output_dir / f"{func_name}_{profile_id}.prof"
            profiler.dump_stats(str(stats_file))

            # Save human-readable stats
            stats_text_file = output_dir / f"{func_name}_{profile_id}_stats.txt"
            with open(stats_text_file, "w") as f:
                stats = pstats.Stats(profiler, stream=f)
                stats.sort_stats("cumulative")
                stats.print_stats(20)  # Top 20 functions

            self.logger.debug(f"Saved cProfile stats to {stats_file}")

        except Exception as e:
            self.logger.error(f"Error saving cProfile stats: {e}")

    def _save_line_profiler_stats(
        self, profiler, func_name: str, profile_id: str
    ) -> None:
        """Save line_profiler statistics."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            stats_file = output_dir / f"{func_name}_{profile_id}_line_profile.txt"

            with open(stats_file, "w") as f:
                profiler.print_stats(stream=f)

            self.logger.debug(f"Saved line profiler stats to {stats_file}")

        except Exception as e:
            self.logger.error(f"Error saving line profiler stats: {e}")

    def _save_memory_profile_stats(
        self,
        func_name: str,
        profile_id: str,
        start_memory: float,
        end_memory: float,
        memory_delta: float,
    ) -> None:
        """Save memory profiling statistics."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            stats_file = output_dir / f"{func_name}_{profile_id}_memory.txt"

            with open(stats_file, "w") as f:
                f.write(f"Memory Profile for {func_name}\n")
                f.write(f"Profile ID: {profile_id}\n")
                f.write(f"Start Memory: {start_memory:.2f} MB\n")
                f.write(f"End Memory: {end_memory:.2f} MB\n")
                f.write(f"Memory Delta: {memory_delta:.2f} MB\n")
                f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")

            self.logger.debug(f"Saved memory profile stats to {stats_file}")

        except Exception as e:
            self.logger.error(f"Error saving memory profile stats: {e}")

    def _save_tracemalloc_stats(
        self, func_name: str, profile_id: str, current_memory: int, peak_memory: int
    ) -> None:
        """Save tracemalloc statistics."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            stats_file = output_dir / f"{func_name}_{profile_id}_tracemalloc.txt"

            with open(stats_file, "w") as f:
                f.write(f"Tracemalloc Profile for {func_name}\n")
                f.write(f"Profile ID: {profile_id}\n")
                f.write(f"Current Memory: {current_memory / 1024 / 1024:.2f} MB\n")
                f.write(f"Peak Memory: {peak_memory / 1024 / 1024:.2f} MB\n")
                f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")

            self.logger.debug(f"Saved tracemalloc stats to {stats_file}")

        except Exception as e:
            self.logger.error(f"Error saving tracemalloc stats: {e}")


class ResourceMonitor:
    """Real-time resource monitoring service."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Monitoring state
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._metrics_history: deque = deque(maxlen=1000)

        # Alert tracking
        self._alert_cooldowns: dict[str, datetime] = {}
        self._alert_callbacks: list[Callable] = []

    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="ResourceMonitor"
        )
        self._monitor_thread.start()

        self.logger.info("Resource monitoring started")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

        self.logger.info("Resource monitoring stopped")

    def add_alert_callback(self, callback: Callable[[str, float, float], None]) -> None:
        """Add callback for resource alerts."""
        self._alert_callbacks.append(callback)

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self._metrics_history.append(metrics)
                self._check_alerts(metrics)

                time.sleep(self.config.monitoring_interval_seconds)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)  # Brief pause on error

    def _collect_metrics(self) -> dict[str, float]:
        """Collect current resource metrics."""
        metrics = {}

        if PSUTIL_AVAILABLE:
            try:
                # CPU metrics
                metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
                metrics["cpu_count"] = psutil.cpu_count()

                # Memory metrics
                memory = psutil.virtual_memory()
                metrics["memory_percent"] = memory.percent
                metrics["memory_available_mb"] = memory.available / 1024 / 1024
                metrics["memory_used_mb"] = memory.used / 1024 / 1024

                # Disk I/O metrics
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    # Calculate I/O rate (simplified)
                    metrics["disk_read_mb"] = disk_io.read_bytes / 1024 / 1024
                    metrics["disk_write_mb"] = disk_io.write_bytes / 1024 / 1024

                # Network I/O metrics
                net_io = psutil.net_io_counters()
                if net_io:
                    metrics["network_sent_mb"] = net_io.bytes_sent / 1024 / 1024
                    metrics["network_recv_mb"] = net_io.bytes_recv / 1024 / 1024

                # Process-specific metrics
                process = psutil.Process()
                metrics["process_memory_mb"] = process.memory_info().rss / 1024 / 1024
                metrics["process_cpu_percent"] = process.cpu_percent()
                metrics["process_threads"] = process.num_threads()

                # System load
                try:
                    load_avg = psutil.getloadavg()
                    metrics["load_1min"] = load_avg[0]
                    metrics["load_5min"] = load_avg[1]
                    metrics["load_15min"] = load_avg[2]
                except AttributeError:
                    # Windows doesn't have load average
                    pass

            except Exception as e:
                self.logger.error(f"Error collecting psutil metrics: {e}")

        # Python-specific metrics
        metrics["gc_objects"] = len(gc.get_objects())
        metrics["gc_collections"] = sum(
            gc.get_stats()[i]["collections"] for i in range(3)
        )

        # Add timestamp
        metrics["timestamp"] = time.time()

        return metrics

    def _check_alerts(self, metrics: dict[str, float]) -> None:
        """Check for alert conditions."""
        current_time = datetime.utcnow()

        for metric_name, threshold in self.config.resource_alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]

                if value > threshold:
                    # Check cooldown
                    cooldown_key = f"alert_{metric_name}"
                    last_alert = self._alert_cooldowns.get(cooldown_key)

                    if (
                        last_alert is None
                        or (current_time - last_alert).total_seconds() > 60
                    ):  # 1 minute cooldown

                        self._trigger_alert(metric_name, value, threshold)
                        self._alert_cooldowns[cooldown_key] = current_time

    def _trigger_alert(self, metric_name: str, value: float, threshold: float) -> None:
        """Trigger resource alert."""
        self.logger.warning(
            f"Resource alert: {metric_name} = {value:.2f} exceeds threshold {threshold:.2f}"
        )

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(metric_name, value, threshold)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

    def get_current_metrics(self) -> dict[str, float] | None:
        """Get latest collected metrics."""
        if self._metrics_history:
            return self._metrics_history[-1]
        return None

    def get_metrics_history(self, minutes: int = 10) -> list[dict[str, float]]:
        """Get metrics history for specified time period."""
        if not self._metrics_history:
            return []

        cutoff_time = time.time() - (minutes * 60)

        return [
            metrics
            for metrics in self._metrics_history
            if metrics.get("timestamp", 0) >= cutoff_time
        ]


class PerformanceOptimizer:
    """Automatic performance optimization service."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Optimization history
        self._optimization_history: list[dict[str, Any]] = []
        self._baseline_metrics: dict[str, float] = {}

    def analyze_performance(self, profile_results: list[ProfileResult]) -> list[str]:
        """Analyze profile results and generate optimization recommendations."""
        recommendations = []

        if not profile_results:
            return recommendations

        # Analyze execution times
        execution_times = [r.execution_time_seconds for r in profile_results]
        avg_execution_time = np.mean(execution_times)
        max_execution_time = np.max(execution_times)

        if max_execution_time > avg_execution_time * 3:
            recommendations.append(
                f"Performance outlier detected: max execution time "
                f"({max_execution_time:.3f}s) is 3x higher than average "
                f"({avg_execution_time:.3f}s). Consider profiling specific cases."
            )

        # Analyze memory usage
        memory_usage = [r.memory_usage_mb for r in profile_results]
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)

        if max_memory > avg_memory * 2:
            recommendations.append(
                f"Memory usage spike detected: max usage ({max_memory:.1f}MB) "
                f"is 2x higher than average ({avg_memory:.1f}MB). "
                f"Consider memory profiling and optimization."
            )

        # Analyze CPU usage patterns
        cpu_usage = [r.cpu_percent for r in profile_results]
        avg_cpu = np.mean(cpu_usage)

        if avg_cpu > 80:
            recommendations.append(
                f"High CPU usage detected (avg: {avg_cpu:.1f}%). "
                f"Consider parallelization or algorithm optimization."
            )

        # Function-specific analysis
        function_stats = defaultdict(list)
        for result in profile_results:
            function_stats[result.function_name].append(result)

        for func_name, results in function_stats.items():
            if len(results) > 1:
                times = [r.execution_time_seconds for r in results]
                std_time = np.std(times)
                mean_time = np.mean(times)

                if std_time / mean_time > 0.5:  # High variability
                    recommendations.append(
                        f"Function {func_name} shows high execution time variability "
                        f"(CV: {std_time/mean_time:.2f}). Consider input-dependent optimization."
                    )

        return recommendations

    def optimize_garbage_collection(self) -> dict[str, Any]:
        """Optimize Python garbage collection settings."""
        if not self.config.gc_optimization:
            return {}

        optimization_result = {
            "optimizations_applied": [],
            "before_stats": self._get_gc_stats(),
        }

        try:
            # Adjust GC thresholds for better performance
            current_thresholds = gc.get_threshold()

            # For ML workloads, reduce frequency of generation 2 collection
            new_thresholds = (
                current_thresholds[0],
                current_thresholds[1],
                max(current_thresholds[2] * 2, 2000),
            )

            gc.set_threshold(*new_thresholds)
            optimization_result["optimizations_applied"].append(
                f"Adjusted GC thresholds from {current_thresholds} to {new_thresholds}"
            )

            # Force a collection to start fresh
            collected = gc.collect()
            optimization_result["optimizations_applied"].append(
                f"Forced GC collection, freed {collected} objects"
            )

            optimization_result["after_stats"] = self._get_gc_stats()

            self.logger.info(
                f"GC optimization completed: {len(optimization_result['optimizations_applied'])} changes"
            )

        except Exception as e:
            self.logger.error(f"Error optimizing garbage collection: {e}")
            optimization_result["error"] = str(e)

        return optimization_result

    def _get_gc_stats(self) -> dict[str, Any]:
        """Get current garbage collection statistics."""
        return {
            "threshold": gc.get_threshold(),
            "counts": gc.get_count(),
            "stats": gc.get_stats(),
            "objects": len(gc.get_objects()),
        }

    def optimize_numpy_operations(self) -> list[str]:
        """Provide NumPy optimization recommendations."""
        recommendations = []

        try:
            # Check NumPy configuration
            config = np.__config__.show()

            # Check BLAS configuration
            if "openblas" in str(config).lower():
                recommendations.append(
                    "OpenBLAS detected. Consider setting OMP_NUM_THREADS for optimal performance."
                )
            elif "mkl" in str(config).lower():
                recommendations.append(
                    "Intel MKL detected. Consider setting MKL_NUM_THREADS for optimal performance."
                )
            else:
                recommendations.append(
                    "No optimized BLAS library detected. Consider installing OpenBLAS or Intel MKL."
                )

            # Check for optimal settings
            recommendations.extend(
                [
                    "Use np.float32 instead of np.float64 when precision allows for better cache performance",
                    "Prefer vectorized operations over loops",
                    "Use np.einsum for complex tensor operations",
                    "Consider memory layout (C vs Fortran order) for cache efficiency",
                    "Use numpy.memmap for large arrays that don't fit in memory",
                ]
            )

        except Exception as e:
            self.logger.error(f"Error analyzing NumPy configuration: {e}")

        return recommendations


class ProfilingService:
    """Main profiling service coordinating all profiling activities."""

    def __init__(self, config: PerformanceConfig | None = None):
        """Initialize profiling service."""
        self.config = config or PerformanceConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.function_profiler = FunctionProfiler(self.config)
        self.resource_monitor = ResourceMonitor(self.config)
        self.optimizer = PerformanceOptimizer(self.config)

        # Profile storage
        self._profile_results: list[ProfileResult] = []
        self._max_stored_results = 1000

        # Start monitoring if enabled
        if self.config.enable_resource_monitoring:
            self.resource_monitor.start_monitoring()

    def profile(
        self,
        profiler_type: ProfilerType = ProfilerType.CPROFILE,
        save_stats: bool = True,
    ):
        """Decorator for profiling functions."""
        return self.function_profiler.profile_function(profiler_type, save_stats)

    @contextmanager
    def profile_context(
        self,
        name: str,
        profiler_type: ProfilerType = ProfilerType.CPROFILE,
    ) -> Generator[str, None, None]:
        """Context manager for profiling code blocks."""
        profile_id = str(uuid4())
        start_time = time.time()
        start_memory = self.function_profiler._get_memory_usage()

        if profiler_type == ProfilerType.TRACEMALLOC:
            tracemalloc.start()

        profiler = None
        if profiler_type == ProfilerType.CPROFILE:
            profiler = cProfile.Profile()
            profiler.enable()

        try:
            yield profile_id
        finally:
            # Stop profiling
            if profiler:
                profiler.disable()
                self.function_profiler._save_cprofile_stats(profiler, name, profile_id)

            if profiler_type == ProfilerType.TRACEMALLOC:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self.function_profiler._save_tracemalloc_stats(
                    name, profile_id, current, peak
                )

            # Calculate final metrics
            execution_time = time.time() - start_time
            end_memory = self.function_profiler._get_memory_usage()
            memory_delta = end_memory - start_memory

            self.logger.info(
                f"Profile context '{name}' completed in {execution_time:.3f}s, "
                f"memory delta: {memory_delta:.2f}MB"
            )

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        current_metrics = self.resource_monitor.get_current_metrics()
        metrics_history = self.resource_monitor.get_metrics_history(10)

        summary = {
            "current_metrics": current_metrics,
            "profile_count": len(self._profile_results),
            "optimization_recommendations": [],
        }

        if self._profile_results:
            # Analyze recent profiles
            recent_profiles = self._profile_results[-100:]
            recommendations = self.optimizer.analyze_performance(recent_profiles)
            summary["optimization_recommendations"] = recommendations

            # Add performance statistics
            execution_times = [p.execution_time_seconds for p in recent_profiles]
            memory_usage = [p.memory_usage_mb for p in recent_profiles]

            summary["performance_stats"] = {
                "avg_execution_time": np.mean(execution_times),
                "p95_execution_time": np.percentile(execution_times, 95),
                "avg_memory_usage": np.mean(memory_usage),
                "max_memory_usage": np.max(memory_usage),
            }

        if metrics_history:
            # Calculate trends
            cpu_values = [m.get("cpu_percent", 0) for m in metrics_history]
            memory_values = [m.get("memory_percent", 0) for m in metrics_history]

            summary["resource_trends"] = {
                "cpu_trend": (
                    "increasing" if cpu_values[-1] > cpu_values[0] else "stable"
                ),
                "memory_trend": (
                    "increasing" if memory_values[-1] > memory_values[0] else "stable"
                ),
                "avg_cpu": np.mean(cpu_values),
                "avg_memory": np.mean(memory_values),
            }

        return summary

    def cleanup_old_profiles(self) -> int:
        """Clean up old profile files and data."""
        cleaned_count = 0

        try:
            output_dir = Path(self.config.output_directory)
            if output_dir.exists():
                # Get all profile files
                profile_files = list(output_dir.glob("*.prof")) + list(
                    output_dir.glob("*.txt")
                )

                # Sort by modification time
                profile_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

                # Keep only the most recent files
                if len(profile_files) > self.config.max_profile_files:
                    files_to_delete = profile_files[self.config.max_profile_files :]

                    for file_path in files_to_delete:
                        file_path.unlink()
                        cleaned_count += 1

            # Clean up in-memory profile results
            if len(self._profile_results) > self._max_stored_results:
                excess_count = len(self._profile_results) - self._max_stored_results
                self._profile_results = self._profile_results[
                    -self._max_stored_results :
                ]
                cleaned_count += excess_count

            if cleaned_count > 0:
                self.logger.info(
                    f"Cleaned up {cleaned_count} old profile files/records"
                )

        except Exception as e:
            self.logger.error(f"Error cleaning up profiles: {e}")

        return cleaned_count

    def shutdown(self) -> None:
        """Shutdown profiling service."""
        self.resource_monitor.stop_monitoring()
        self.cleanup_old_profiles()
        self.logger.info("Profiling service shutdown complete")


# Global profiling service instance
_profiling_service: ProfilingService | None = None


def get_profiling_service() -> ProfilingService:
    """Get global profiling service instance."""
    global _profiling_service
    if _profiling_service is None:
        _profiling_service = ProfilingService()
    return _profiling_service


def profile(profiler_type: ProfilerType = ProfilerType.CPROFILE):
    """Convenience decorator for profiling functions."""
    return get_profiling_service().profile(profiler_type)


def profile_context(name: str, profiler_type: ProfilerType = ProfilerType.CPROFILE):
    """Convenience context manager for profiling code blocks."""
    return get_profiling_service().profile_context(name, profiler_type)
