"""Advanced memory management and optimization for anomaly detection workloads."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil


@dataclass
class MemoryUsage:
    """Memory usage statistics."""

    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    process_memory_mb: float
    peak_memory_mb: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryOptimizationResult:
    """Result of memory optimization operation."""

    strategy: str
    memory_before_mb: float
    memory_after_mb: float
    memory_saved_mb: float
    success: bool
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)


class AdaptiveMemoryManager:
    """Adaptive memory manager with intelligent optimization strategies."""

    def __init__(
        self,
        target_memory_percent: float = 80.0,
        warning_threshold_percent: float = 85.0,
        critical_threshold_percent: float = 95.0,
        optimization_interval_seconds: float = 30.0,
        enable_automatic_optimization: bool = True,
    ):
        """Initialize adaptive memory manager.

        Args:
            target_memory_percent: Target memory usage percentage
            warning_threshold_percent: Warning threshold for memory usage
            critical_threshold_percent: Critical threshold triggering aggressive optimization
            optimization_interval_seconds: Interval between automatic optimizations
            enable_automatic_optimization: Enable automatic memory optimization
        """
        self.target_memory_percent = target_memory_percent
        self.warning_threshold_percent = warning_threshold_percent
        self.critical_threshold_percent = critical_threshold_percent
        self.optimization_interval_seconds = optimization_interval_seconds
        self.enable_automatic_optimization = enable_automatic_optimization

        self.logger = logging.getLogger(__name__)

        # Memory tracking
        self._memory_history: List[MemoryUsage] = []
        self._optimization_history: List[MemoryOptimizationResult] = []
        self._last_optimization_time = 0.0

        # Optimization strategies (ordered by aggressiveness)
        self._optimization_strategies = [
            self._optimize_garbage_collection,
            self._optimize_pandas_dtypes,
            self._optimize_numpy_arrays,
            self._compress_large_objects,
            self._offload_to_disk,
            self._reduce_cache_sizes,
        ]

        # Monitored objects for optimization
        self._monitored_objects: Dict[str, Any] = {}

        # Performance metrics
        self._optimization_metrics = {
            "total_optimizations": 0,
            "total_memory_saved_mb": 0.0,
            "average_optimization_time": 0.0,
        }

        # Background task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

    async def start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Memory monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Memory monitoring stopped")

    def get_memory_usage(self) -> MemoryUsage:
        """Get current memory usage statistics."""
        # System memory
        memory = psutil.virtual_memory()

        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()

        return MemoryUsage(
            total_mb=memory.total / (1024 * 1024),
            available_mb=memory.available / (1024 * 1024),
            used_mb=memory.used / (1024 * 1024),
            percent_used=memory.percent,
            process_memory_mb=process_memory.rss / (1024 * 1024),
            peak_memory_mb=(
                process_memory.peak_wset / (1024 * 1024)
                if hasattr(process_memory, "peak_wset")
                else 0.0
            ),
        )

    async def optimize_memory_usage(
        self, force: bool = False
    ) -> List[MemoryOptimizationResult]:
        """Optimize memory usage using available strategies.

        Args:
            force: Force optimization even if not needed

        Returns:
            List of optimization results
        """
        current_usage = self.get_memory_usage()

        if not force and current_usage.percent_used <= self.target_memory_percent:
            return []

        self.logger.info(
            f"Starting memory optimization (current usage: {current_usage.percent_used:.1f}%)"
        )

        optimization_results = []

        # Determine optimization aggressiveness
        if current_usage.percent_used >= self.critical_threshold_percent:
            strategies_to_use = self._optimization_strategies  # Use all strategies
            self.logger.warning(
                "Critical memory usage - applying all optimization strategies"
            )
        elif current_usage.percent_used >= self.warning_threshold_percent:
            strategies_to_use = self._optimization_strategies[
                :4
            ]  # Use moderate strategies
            self.logger.info(
                "High memory usage - applying moderate optimization strategies"
            )
        else:
            strategies_to_use = self._optimization_strategies[
                :2
            ]  # Use gentle strategies
            self.logger.info(
                "Moderate memory usage - applying gentle optimization strategies"
            )

        # Apply optimization strategies
        for strategy in strategies_to_use:
            try:
                result = await strategy()
                optimization_results.append(result)

                # Check if we've reached target
                new_usage = self.get_memory_usage()
                if new_usage.percent_used <= self.target_memory_percent:
                    self.logger.info(
                        f"Target memory usage reached: {new_usage.percent_used:.1f}%"
                    )
                    break

            except Exception as e:
                self.logger.error(
                    f"Optimization strategy {strategy.__name__} failed: {e}"
                )
                optimization_results.append(
                    MemoryOptimizationResult(
                        strategy=strategy.__name__,
                        memory_before_mb=current_usage.process_memory_mb,
                        memory_after_mb=current_usage.process_memory_mb,
                        memory_saved_mb=0.0,
                        success=False,
                        duration_seconds=0.0,
                        details={"error": str(e)},
                    )
                )

        # Update metrics
        self._optimization_history.extend(optimization_results)
        self._update_optimization_metrics(optimization_results)
        self._last_optimization_time = time.time()

        total_saved = sum(r.memory_saved_mb for r in optimization_results if r.success)
        self.logger.info(f"Memory optimization completed - saved {total_saved:.2f} MB")

        return optimization_results

    def register_object(self, name: str, obj: Any) -> None:
        """Register object for memory monitoring and optimization.

        Args:
            name: Object name/identifier
            obj: Object to monitor
        """
        self._monitored_objects[name] = obj
        self.logger.debug(f"Registered object for monitoring: {name}")

    def unregister_object(self, name: str) -> None:
        """Unregister object from monitoring.

        Args:
            name: Object name/identifier to remove
        """
        if name in self._monitored_objects:
            del self._monitored_objects[name]
            self.logger.debug(f"Unregistered object: {name}")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Collect memory usage
                usage = self.get_memory_usage()
                self._memory_history.append(usage)

                # Limit history size
                if len(self._memory_history) > 1000:
                    self._memory_history = self._memory_history[-500:]

                # Check if optimization is needed
                if (
                    self.enable_automatic_optimization
                    and usage.percent_used > self.warning_threshold_percent
                    and time.time() - self._last_optimization_time
                    > self.optimization_interval_seconds
                ):

                    await self.optimize_memory_usage()

                # Sleep until next check
                await asyncio.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Memory monitoring loop error: {e}")
                await asyncio.sleep(10.0)  # Longer sleep on error

    async def _optimize_garbage_collection(self) -> MemoryOptimizationResult:
        """Optimize using garbage collection."""
        start_time = time.time()
        memory_before = self.get_memory_usage()

        # Force garbage collection
        collected = gc.collect()

        # Force collection of all generations
        for generation in range(gc.get_count().__len__()):
            gc.collect(generation)

        memory_after = self.get_memory_usage()
        duration = time.time() - start_time

        return MemoryOptimizationResult(
            strategy="garbage_collection",
            memory_before_mb=memory_before.process_memory_mb,
            memory_after_mb=memory_after.process_memory_mb,
            memory_saved_mb=memory_before.process_memory_mb
            - memory_after.process_memory_mb,
            success=True,
            duration_seconds=duration,
            details={"objects_collected": collected},
        )

    async def _optimize_pandas_dtypes(self) -> MemoryOptimizationResult:
        """Optimize pandas DataFrame dtypes."""
        start_time = time.time()
        memory_before = self.get_memory_usage()

        optimized_count = 0
        memory_saved = 0.0

        for name, obj in self._monitored_objects.items():
            if isinstance(obj, pd.DataFrame):
                original_memory = obj.memory_usage(deep=True).sum()

                # Optimize dtypes
                for col in obj.columns:
                    if obj[col].dtype == "int64":
                        obj[col] = pd.to_numeric(obj[col], downcast="integer")
                    elif obj[col].dtype == "float64":
                        obj[col] = pd.to_numeric(obj[col], downcast="float")
                    elif obj[col].dtype == "object":
                        # Check if categorical conversion would be beneficial
                        if obj[col].nunique() / len(obj) < 0.5:
                            obj[col] = obj[col].astype("category")

                new_memory = obj.memory_usage(deep=True).sum()
                memory_saved += (original_memory - new_memory) / (1024 * 1024)  # MB
                optimized_count += 1

        memory_after = self.get_memory_usage()
        duration = time.time() - start_time

        return MemoryOptimizationResult(
            strategy="pandas_dtypes",
            memory_before_mb=memory_before.process_memory_mb,
            memory_after_mb=memory_after.process_memory_mb,
            memory_saved_mb=memory_saved,
            success=True,
            duration_seconds=duration,
            details={"dataframes_optimized": optimized_count},
        )

    async def _optimize_numpy_arrays(self) -> MemoryOptimizationResult:
        """Optimize NumPy arrays."""
        start_time = time.time()
        memory_before = self.get_memory_usage()

        optimized_count = 0
        memory_saved = 0.0

        for name, obj in self._monitored_objects.items():
            if isinstance(obj, np.ndarray):
                original_memory = obj.nbytes

                # Convert float64 to float32 if precision allows
                if obj.dtype == np.float64:
                    # Check if conversion would significantly affect precision
                    if np.allclose(obj, obj.astype(np.float32), rtol=1e-6):
                        obj = obj.astype(np.float32)
                        self._monitored_objects[name] = obj

                        new_memory = obj.nbytes
                        memory_saved += (original_memory - new_memory) / (
                            1024 * 1024
                        )  # MB
                        optimized_count += 1

        memory_after = self.get_memory_usage()
        duration = time.time() - start_time

        return MemoryOptimizationResult(
            strategy="numpy_arrays",
            memory_before_mb=memory_before.process_memory_mb,
            memory_after_mb=memory_after.process_memory_mb,
            memory_saved_mb=memory_saved,
            success=True,
            duration_seconds=duration,
            details={"arrays_optimized": optimized_count},
        )

    async def _compress_large_objects(self) -> MemoryOptimizationResult:
        """Compress large objects to save memory."""
        start_time = time.time()
        memory_before = self.get_memory_usage()

        compressed_count = 0
        memory_saved = 0.0

        # This is a placeholder for object compression
        # In practice, you might compress large DataFrames or arrays
        # using libraries like blosc, lz4, or zstd

        memory_after = self.get_memory_usage()
        duration = time.time() - start_time

        return MemoryOptimizationResult(
            strategy="compress_objects",
            memory_before_mb=memory_before.process_memory_mb,
            memory_after_mb=memory_after.process_memory_mb,
            memory_saved_mb=memory_saved,
            success=True,
            duration_seconds=duration,
            details={"objects_compressed": compressed_count},
        )

    async def _offload_to_disk(self) -> MemoryOptimizationResult:
        """Offload large objects to disk."""
        start_time = time.time()
        memory_before = self.get_memory_usage()

        offloaded_count = 0
        memory_saved = 0.0

        # This is a placeholder for disk offloading
        # In practice, you might save large DataFrames to parquet
        # or numpy arrays to .npy files

        memory_after = self.get_memory_usage()
        duration = time.time() - start_time

        return MemoryOptimizationResult(
            strategy="offload_to_disk",
            memory_before_mb=memory_before.process_memory_mb,
            memory_after_mb=memory_after.process_memory_mb,
            memory_saved_mb=memory_saved,
            success=True,
            duration_seconds=duration,
            details={"objects_offloaded": offloaded_count},
        )

    async def _reduce_cache_sizes(self) -> MemoryOptimizationResult:
        """Reduce cache sizes to free memory."""
        start_time = time.time()
        memory_before = self.get_memory_usage()

        caches_cleared = 0

        # Clear various caches
        try:
            # Clear pandas string cache
            if hasattr(pd.core.strings, "_str_map_cache"):
                pd.core.strings._str_map_cache.clear()
                caches_cleared += 1
        except:
            pass

        # Force garbage collection after cache clearing
        gc.collect()

        memory_after = self.get_memory_usage()
        duration = time.time() - start_time

        return MemoryOptimizationResult(
            strategy="reduce_caches",
            memory_before_mb=memory_before.process_memory_mb,
            memory_after_mb=memory_after.process_memory_mb,
            memory_saved_mb=memory_before.process_memory_mb
            - memory_after.process_memory_mb,
            success=True,
            duration_seconds=duration,
            details={"caches_cleared": caches_cleared},
        )

    def _update_optimization_metrics(
        self, results: List[MemoryOptimizationResult]
    ) -> None:
        """Update optimization metrics."""
        successful_results = [r for r in results if r.success]

        self._optimization_metrics["total_optimizations"] += len(successful_results)
        self._optimization_metrics["total_memory_saved_mb"] += sum(
            r.memory_saved_mb for r in successful_results
        )

        if successful_results:
            avg_time = sum(r.duration_seconds for r in successful_results) / len(
                successful_results
            )
            current_avg = self._optimization_metrics["average_optimization_time"]
            total_ops = self._optimization_metrics["total_optimizations"]

            # Update running average
            self._optimization_metrics["average_optimization_time"] = (
                current_avg * (total_ops - len(successful_results))
                + avg_time * len(successful_results)
            ) / total_ops

    def get_memory_trends(self, hours: int = 1) -> Dict[str, Any]:
        """Get memory usage trends over specified time period.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dictionary with trend analysis
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_usage = [u for u in self._memory_history if u.timestamp >= cutoff_time]

        if not recent_usage:
            return {"error": "No recent data available"}

        usage_values = [u.percent_used for u in recent_usage]

        return {
            "time_period_hours": hours,
            "data_points": len(recent_usage),
            "current_usage_percent": recent_usage[-1].percent_used,
            "min_usage_percent": min(usage_values),
            "max_usage_percent": max(usage_values),
            "average_usage_percent": sum(usage_values) / len(usage_values),
            "trend": self._calculate_trend(usage_values),
            "optimization_count": len(
                [
                    r
                    for r in self._optimization_history
                    if r.memory_before_mb > 0
                    and any(u.timestamp >= cutoff_time for u in self._memory_history)
                ]
            ),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear trend
        n = len(values)
        x = list(range(n))

        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        else:
            return "stable"

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization activities."""
        return {
            "metrics": self._optimization_metrics.copy(),
            "recent_optimizations": len(
                [
                    r
                    for r in self._optimization_history
                    if time.time() - r.duration_seconds < 3600
                ]
            ),  # Last hour
            "monitored_objects": len(self._monitored_objects),
            "monitoring_active": self._running,
            "last_optimization": self._last_optimization_time,
        }


class MemoryProfiler:
    """Memory profiler for detailed analysis of memory usage patterns."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._profiles: Dict[str, Dict[str, Any]] = {}

    def profile_function(self, func_name: str):
        """Decorator to profile memory usage of functions."""

        def decorator(func: Callable) -> Callable:
            async def async_wrapper(*args, **kwargs):
                return await self._profile_async_function(
                    func_name, func, *args, **kwargs
                )

            def sync_wrapper(*args, **kwargs):
                return self._profile_sync_function(func_name, func, *args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    async def _profile_async_function(
        self, func_name: str, func: Callable, *args, **kwargs
    ):
        """Profile async function memory usage."""
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)

            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            end_time = time.time()

            self._record_profile(
                func_name, start_memory, end_memory, end_time - start_time, True
            )

            return result

        except Exception as e:
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            end_time = time.time()

            self._record_profile(
                func_name, start_memory, end_memory, end_time - start_time, False
            )
            raise

    def _profile_sync_function(self, func_name: str, func: Callable, *args, **kwargs):
        """Profile sync function memory usage."""
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)

            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            end_time = time.time()

            self._record_profile(
                func_name, start_memory, end_memory, end_time - start_time, True
            )

            return result

        except Exception as e:
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            end_time = time.time()

            self._record_profile(
                func_name, start_memory, end_memory, end_time - start_time, False
            )
            raise

    def _record_profile(
        self,
        func_name: str,
        start_memory: float,
        end_memory: float,
        duration: float,
        success: bool,
    ):
        """Record profiling data."""
        if func_name not in self._profiles:
            self._profiles[func_name] = {
                "calls": 0,
                "total_duration": 0.0,
                "total_memory_delta": 0.0,
                "max_memory_delta": 0.0,
                "min_memory_delta": float("inf"),
                "failures": 0,
            }

        profile = self._profiles[func_name]
        memory_delta = end_memory - start_memory

        profile["calls"] += 1
        profile["total_duration"] += duration
        profile["total_memory_delta"] += memory_delta
        profile["max_memory_delta"] = max(profile["max_memory_delta"], memory_delta)
        profile["min_memory_delta"] = min(profile["min_memory_delta"], memory_delta)

        if not success:
            profile["failures"] += 1

    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of all function profiles."""
        summary = {}

        for func_name, profile in self._profiles.items():
            calls = profile["calls"]
            summary[func_name] = {
                "total_calls": calls,
                "average_duration": (
                    profile["total_duration"] / calls if calls > 0 else 0
                ),
                "average_memory_delta_mb": (
                    profile["total_memory_delta"] / calls if calls > 0 else 0
                ),
                "max_memory_delta_mb": profile["max_memory_delta"],
                "min_memory_delta_mb": (
                    profile["min_memory_delta"]
                    if profile["min_memory_delta"] != float("inf")
                    else 0
                ),
                "failure_rate": profile["failures"] / calls if calls > 0 else 0,
            }

        return summary
