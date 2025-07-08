"""Memory usage optimization utilities and monitoring."""

from __future__ import annotations

import gc
import logging
import tracemalloc
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional
from weakref import WeakSet

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    timestamp: float
    process_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    tracemalloc_current_mb: Optional[float] = None
    tracemalloc_peak_mb: Optional[float] = None
    gc_counts: Optional[tuple[int, int, int]] = None


class MemoryMonitor:
    """Advanced memory usage monitoring and optimization."""

    def __init__(self, enable_tracemalloc: bool = True,
                 warning_threshold_mb: float = 1000.0,
                 critical_threshold_mb: float = 2000.0):
        """Initialize memory monitor."""
        self.enable_tracemalloc = enable_tracemalloc
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self._snapshots: list[MemorySnapshot] = []
        self._tracked_objects: WeakSet = WeakSet()

        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("Started tracemalloc for detailed memory tracking")

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        import time

        # Get process memory info
        process = psutil.Process()
        memory_info = process.memory_info()

        # Get system memory info
        system_memory = psutil.virtual_memory()

        # Get tracemalloc info if enabled
        tracemalloc_current = None
        tracemalloc_peak = None

        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_current = current / 1024 / 1024  # Convert to MB
            tracemalloc_peak = peak / 1024 / 1024

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            process_memory_mb=memory_info.rss / 1024 / 1024,
            available_memory_mb=system_memory.available / 1024 / 1024,
            memory_percent=system_memory.percent,
            tracemalloc_current_mb=tracemalloc_current,
            tracemalloc_peak_mb=tracemalloc_peak,
            gc_counts=gc.get_count()
        )

        self._snapshots.append(snapshot)

        # Keep only last 100 snapshots
        if len(self._snapshots) > 100:
            self._snapshots = self._snapshots[-100:]

        # Check thresholds
        self._check_memory_thresholds(snapshot)

        return snapshot

    def _check_memory_thresholds(self, snapshot: MemorySnapshot) -> None:
        """Check if memory usage exceeds thresholds."""
        if snapshot.process_memory_mb > self.critical_threshold_mb:
            logger.critical(
                f"CRITICAL: Memory usage {snapshot.process_memory_mb:.1f}MB "
                f"exceeds critical threshold {self.critical_threshold_mb}MB"
            )
            self.emergency_cleanup()
        elif snapshot.process_memory_mb > self.warning_threshold_mb:
            logger.warning(
                f"WARNING: Memory usage {snapshot.process_memory_mb:.1f}MB "
                f"exceeds warning threshold {self.warning_threshold_mb}MB"
            )

    def get_memory_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self._snapshots:
            return {}

        latest = self._snapshots[-1]

        # Calculate trends if we have multiple snapshots
        trend_data = {}
        if len(self._snapshots) >= 2:
            first = self._snapshots[0]
            trend_data = {
                'memory_trend_mb': latest.process_memory_mb - first.process_memory_mb,
                'time_span_seconds': latest.timestamp - first.timestamp,
                'average_memory_mb': sum(s.process_memory_mb for s in self._snapshots) / len(self._snapshots)
            }

        stats = {
            'current_memory_mb': latest.process_memory_mb,
            'available_memory_mb': latest.available_memory_mb,
            'memory_percent': latest.memory_percent,
            'gc_counts': latest.gc_counts,
            'snapshot_count': len(self._snapshots),
            'tracked_objects': len(self._tracked_objects),
        }

        if latest.tracemalloc_current_mb is not None:
            stats.update({
                'tracemalloc_current_mb': latest.tracemalloc_current_mb,
                'tracemalloc_peak_mb': latest.tracemalloc_peak_mb,
            })

        stats.update(trend_data)
        return stats

    def get_top_memory_consumers(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get top memory consuming objects using tracemalloc."""
        if not (self.enable_tracemalloc and tracemalloc.is_tracing()):
            return []

        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            consumers = []
            for index, stat in enumerate(top_stats[:limit]):
                consumers.append({
                    'rank': index + 1,
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count,
                    'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                })

            return consumers
        except Exception as e:
            logger.error(f"Error getting memory consumers: {e}")
            return []

    def emergency_cleanup(self) -> dict[str, int]:
        """Perform emergency memory cleanup."""
        logger.warning("Performing emergency memory cleanup")

        cleanup_stats = {
            'gc_collected': 0,
            'objects_before': len(gc.get_objects()),
            'cache_cleared': 0,
        }

        # Force garbage collection
        cleanup_stats['gc_collected'] = gc.collect()

        # Clear tracked objects
        self._tracked_objects.clear()

        # Clear function caches (if any)
        try:

            # Clear lru_cache decorated functions
            for obj in gc.get_objects():
                if hasattr(obj, 'cache_clear'):
                    obj.cache_clear()
                    cleanup_stats['cache_cleared'] += 1
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")

        cleanup_stats['objects_after'] = len(gc.get_objects())
        cleanup_stats['objects_freed'] = cleanup_stats['objects_before'] - cleanup_stats['objects_after']

        logger.info(f"Emergency cleanup completed: {cleanup_stats}")
        return cleanup_stats

    def track_object(self, obj: Any) -> None:
        """Track an object for memory monitoring."""
        try:
            self._tracked_objects.add(obj)
        except TypeError:
            # Object is not weakly referenceable
            pass

    @contextmanager
    def memory_profile(self, operation_name: str = "operation"):
        """Context manager for profiling memory usage of an operation."""
        start_snapshot = self.take_snapshot()
        start_time = start_snapshot.timestamp

        logger.debug(f"Starting memory profile for: {operation_name}")

        try:
            yield
        finally:
            end_snapshot = self.take_snapshot()

            memory_delta = end_snapshot.process_memory_mb - start_snapshot.process_memory_mb
            time_delta = end_snapshot.timestamp - start_time

            logger.info(
                f"Memory profile for '{operation_name}': "
                f"Δ{memory_delta:+.2f}MB over {time_delta:.2f}s"
            )


class DataFrameOptimizer:
    """Pandas DataFrame memory optimization utilities."""

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage."""
        if df.empty:
            return df

        original_memory = df.memory_usage(deep=True).sum()

        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert to category if it's beneficial
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')

        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100

        if verbose:
            logger.info(
                f"DataFrame memory optimized: {original_memory/1024/1024:.2f}MB → "
                f"{optimized_memory/1024/1024:.2f}MB ({reduction:.1f}% reduction)"
            )

        return df

    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, max_memory_mb: float = 100.0) -> Generator[pd.DataFrame, None, None]:
        """Yield DataFrame chunks based on memory limit."""
        if df.empty:
            yield df
            return

        # Estimate memory per row
        sample_size = min(1000, len(df))
        sample_memory = df.head(sample_size).memory_usage(deep=True).sum()
        memory_per_row = sample_memory / sample_size

        # Calculate chunk size
        max_memory_bytes = max_memory_mb * 1024 * 1024
        chunk_size = max(1, int(max_memory_bytes / memory_per_row))

        logger.debug(f"Chunking DataFrame into chunks of {chunk_size} rows")

        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            yield df.iloc[start:end].copy()

    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> dict[str, Any]:
        """Get detailed memory usage information for DataFrame."""
        if df.empty:
            return {'total_mb': 0, 'columns': {}}

        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()

        column_info = {}
        for col in df.columns:
            col_memory = memory_usage[col]
            col_info = {
                'memory_mb': col_memory / 1024 / 1024,
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique() if df[col].dtype != 'object' else None
            }

            # Add additional info for object columns
            if df[col].dtype == 'object':
                try:
                    col_info['avg_string_length'] = df[col].astype(str).str.len().mean()
                    col_info['max_string_length'] = df[col].astype(str).str.len().max()
                except:
                    pass

            column_info[col] = col_info

        return {
            'total_mb': total_memory / 1024 / 1024,
            'shape': df.shape,
            'columns': column_info,
            'index_memory_mb': memory_usage['Index'] / 1024 / 1024,
        }


class ArrayOptimizer:
    """NumPy array memory optimization utilities."""

    @staticmethod
    def optimize_array_dtype(arr: np.ndarray, preserve_precision: bool = True) -> np.ndarray:
        """Optimize NumPy array dtype to reduce memory usage."""
        if arr.size == 0:
            return arr

        original_dtype = arr.dtype

        # For integer arrays
        if np.issubdtype(arr.dtype, np.integer):
            # Find the minimum integer type that can hold the data
            min_val, max_val = arr.min(), arr.max()

            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                info = np.iinfo(dtype)
                if min_val >= info.min and max_val <= info.max:
                    if arr.dtype != dtype:
                        logger.debug(f"Optimizing array dtype: {original_dtype} → {dtype}")
                    return arr.astype(dtype)

        # For floating point arrays
        elif np.issubdtype(arr.dtype, np.floating):
            if not preserve_precision:
                # Try float32 if values fit
                if arr.dtype == np.float64:
                    arr_f32 = arr.astype(np.float32)
                    if np.allclose(arr, arr_f32, equal_nan=True):
                        logger.debug(f"Optimizing array dtype: {original_dtype} → float32")
                        return arr_f32

        return arr

    @staticmethod
    def get_array_memory_info(arr: np.ndarray) -> dict[str, Any]:
        """Get memory information for NumPy array."""
        return {
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'size_mb': arr.nbytes / 1024 / 1024,
            'elements': arr.size,
            'c_contiguous': arr.flags['C_CONTIGUOUS'],
            'f_contiguous': arr.flags['F_CONTIGUOUS'],
            'writable': arr.flags['WRITEABLE'],
        }


# Memory optimization decorators

def memory_limit(max_memory_mb: float):
    """Decorator to limit function memory usage."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = MemoryMonitor()

            with monitor.memory_profile(func.__name__):
                start_snapshot = monitor.take_snapshot()

                try:
                    result = func(*args, **kwargs)

                    end_snapshot = monitor.take_snapshot()
                    memory_used = end_snapshot.process_memory_mb - start_snapshot.process_memory_mb

                    if memory_used > max_memory_mb:
                        logger.warning(
                            f"Function {func.__name__} used {memory_used:.2f}MB "
                            f"(limit: {max_memory_mb}MB)"
                        )

                    return result

                except MemoryError:
                    logger.error(f"Memory error in {func.__name__}")
                    monitor.emergency_cleanup()
                    raise

        return wrapper
    return decorator


def optimize_dataframe_memory(func):
    """Decorator to automatically optimize DataFrame memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        if isinstance(result, pd.DataFrame):
            result = DataFrameOptimizer.optimize_dtypes(result, verbose=False)
        elif isinstance(result, dict):
            # Optimize any DataFrames in the result dictionary
            for key, value in result.items():
                if isinstance(value, pd.DataFrame):
                    result[key] = DataFrameOptimizer.optimize_dtypes(value, verbose=False)

        return result
    return wrapper


def gc_after_function(func):
    """Decorator to run garbage collection after function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            collected = gc.collect()
            if collected > 0:
                logger.debug(f"Garbage collected {collected} objects after {func.__name__}")
    return wrapper


class MemoryPool:
    """Simple memory pool for reusing objects."""

    def __init__(self, factory_func: callable, max_size: int = 100):
        """Initialize memory pool."""
        self.factory_func = factory_func
        self.max_size = max_size
        self._pool: list[Any] = []
        self._in_use: WeakSet = WeakSet()

    def acquire(self) -> Any:
        """Acquire an object from the pool."""
        if self._pool:
            obj = self._pool.pop()
            self._in_use.add(obj)
            logger.debug(f"Acquired object from pool (pool size: {len(self._pool)})")
            return obj
        else:
            obj = self.factory_func()
            self._in_use.add(obj)
            logger.debug("Created new object (pool empty)")
            return obj

    def release(self, obj: Any) -> None:
        """Release an object back to the pool."""
        if obj in self._in_use and len(self._pool) < self.max_size:
            # Reset object state if it has a reset method
            if hasattr(obj, 'reset'):
                obj.reset()

            self._pool.append(obj)
            self._in_use.discard(obj)
            logger.debug(f"Released object to pool (pool size: {len(self._pool)})")

    @contextmanager
    def get_object(self):
        """Context manager for acquiring and releasing objects."""
        obj = self.acquire()
        try:
            yield obj
        finally:
            self.release(obj)

    def get_stats(self) -> dict[str, int]:
        """Get pool statistics."""
        return {
            'pool_size': len(self._pool),
            'in_use': len(self._in_use),
            'max_size': self.max_size,
        }


# Global memory monitor instance
_global_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor()
    return _global_memory_monitor


def configure_memory_monitoring(enable_tracemalloc: bool = True,
                              warning_threshold_mb: float = 1000.0,
                              critical_threshold_mb: float = 2000.0) -> MemoryMonitor:
    """Configure global memory monitoring."""
    global _global_memory_monitor
    _global_memory_monitor = MemoryMonitor(
        enable_tracemalloc=enable_tracemalloc,
        warning_threshold_mb=warning_threshold_mb,
        critical_threshold_mb=critical_threshold_mb
    )
    logger.info("Configured memory monitoring")
    return _global_memory_monitor
