"""Memory optimization utilities for efficient data processing."""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_memory: int
    available_memory: int
    used_memory: int
    used_percentage: float
    process_memory: int
    process_percentage: float


class MemoryMonitor:
    """Monitor memory usage and provide optimization recommendations."""

    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        """Initialize memory monitor."""
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._memory_history: list[MemoryStats] = []

    def start_monitoring(self, interval_seconds: int = 5):
        """Start continuous memory monitoring."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval_seconds,), daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Memory monitoring started (interval: {interval_seconds}s)")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Memory monitoring stopped")

    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        system_memory = psutil.virtual_memory()

        # Process memory
        process = psutil.Process()
        process_info = process.memory_info()

        return MemoryStats(
            total_memory=system_memory.total,
            available_memory=system_memory.available,
            used_memory=system_memory.used,
            used_percentage=system_memory.percent,
            process_memory=process_info.rss,
            process_percentage=process_info.rss / system_memory.total * 100,
        )

    def get_memory_history(self) -> list[MemoryStats]:
        """Get memory usage history."""
        return self._memory_history.copy()

    def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                stats = self.get_current_stats()
                self._memory_history.append(stats)

                # Keep only last 1000 entries
                if len(self._memory_history) > 1000:
                    self._memory_history = self._memory_history[-1000:]

                # Check thresholds
                if stats.used_percentage > self.critical_threshold * 100:
                    logger.critical(
                        f"Critical memory usage: {stats.used_percentage:.1f}%"
                    )
                elif stats.used_percentage > self.warning_threshold * 100:
                    logger.warning(f"High memory usage: {stats.used_percentage:.1f}%")

                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(interval_seconds)


class DataFrameOptimizer:
    """Optimize pandas DataFrame memory usage."""

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage."""
        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype

            # Optimize numeric columns
            if pd.api.types.is_numeric_dtype(col_type):
                optimized_df[col] = DataFrameOptimizer._optimize_numeric_column(
                    optimized_df[col], aggressive
                )

            # Optimize string columns
            elif pd.api.types.is_object_dtype(col_type):
                optimized_df[col] = DataFrameOptimizer._optimize_string_column(
                    optimized_df[col], aggressive
                )

        return optimized_df

    @staticmethod
    def _optimize_numeric_column(
        series: pd.Series, aggressive: bool = False
    ) -> pd.Series:
        """Optimize numeric column data type."""
        if pd.api.types.is_integer_dtype(series):
            # Integer optimization
            min_val, max_val = series.min(), series.max()

            if min_val >= 0:  # Unsigned integers
                if max_val < 256:
                    return series.astype(np.uint8)
                elif max_val < 65536:
                    return series.astype(np.uint16)
                elif max_val < 4294967296:
                    return series.astype(np.uint32)
                else:
                    return series.astype(np.uint64)
            else:  # Signed integers
                if min_val >= -128 and max_val < 128:
                    return series.astype(np.int8)
                elif min_val >= -32768 and max_val < 32768:
                    return series.astype(np.int16)
                elif min_val >= -2147483648 and max_val < 2147483648:
                    return series.astype(np.int32)
                else:
                    return series.astype(np.int64)

        elif pd.api.types.is_float_dtype(series):
            # Float optimization
            if aggressive:
                # Check if we can use float32 without significant precision loss
                float32_series = series.astype(np.float32)
                if np.allclose(series.dropna(), float32_series.dropna(), rtol=1e-6):
                    return float32_series

            return series.astype(np.float64)

        return series

    @staticmethod
    def _optimize_string_column(
        series: pd.Series, aggressive: bool = False
    ) -> pd.Series:
        """Optimize string column data type."""
        try:
            # Try to convert to category if it saves memory
            unique_ratio = series.nunique() / len(series)

            if unique_ratio < 0.5:  # Less than 50% unique values
                return series.astype("category")

            # For highly unique strings, keep as object
            return series

        except Exception:
            return series

    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> dict[str, Any]:
        """Get detailed memory usage information for DataFrame."""
        memory_usage = df.memory_usage(deep=True)

        return {
            "total_memory_mb": memory_usage.sum() / 1024**2,
            "column_memory": {
                col: memory_usage[col] / 1024**2 for col in memory_usage.index
            },
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
        }


class ChunkedProcessor:
    """Process large datasets in memory-efficient chunks."""

    def __init__(self, chunk_size: int = 10000, memory_limit_mb: int = 1000):
        """Initialize chunked processor."""
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.memory_monitor = MemoryMonitor()

    def process_dataframe_chunks(
        self, df: pd.DataFrame, processor_func, *args, **kwargs
    ) -> Generator[Any, None, None]:
        """Process DataFrame in chunks to manage memory usage."""
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size

        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i : i + self.chunk_size].copy()

            # Check memory usage
            stats = self.memory_monitor.get_current_stats()
            if stats.process_memory / 1024**2 > self.memory_limit_mb:
                logger.warning(
                    f"Memory limit approached: {stats.process_memory / 1024**2:.1f}MB"
                )
                gc.collect()  # Force garbage collection

            chunk_num = i // self.chunk_size + 1
            logger.debug(f"Processing chunk {chunk_num}/{total_chunks}")

            try:
                result = processor_func(chunk, *args, **kwargs)
                yield result
            finally:
                # Clean up chunk from memory
                del chunk
                gc.collect()

    def process_file_chunks(
        self, file_path: str, processor_func, file_type: str = "csv", *args, **kwargs
    ) -> Generator[Any, None, None]:
        """Process file in chunks without loading entire file into memory."""
        if file_type.lower() == "csv":
            chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size)
        elif file_type.lower() == "parquet":
            # For parquet, we need to handle chunks differently
            df = pd.read_parquet(file_path)
            chunk_iter = (
                df.iloc[i : i + self.chunk_size]
                for i in range(0, len(df), self.chunk_size)
            )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        for chunk_num, chunk in enumerate(chunk_iter, 1):
            # Check memory usage
            stats = self.memory_monitor.get_current_stats()
            if stats.process_memory / 1024**2 > self.memory_limit_mb:
                logger.warning(
                    f"Memory limit approached: {stats.process_memory / 1024**2:.1f}MB"
                )
                gc.collect()

            logger.debug(f"Processing file chunk {chunk_num}")

            try:
                result = processor_func(chunk, *args, **kwargs)
                yield result
            finally:
                del chunk
                gc.collect()


class ModelMemoryOptimizer:
    """Optimize machine learning model memory usage."""

    @staticmethod
    def optimize_model_storage(model: Any, compression_level: int = 1) -> bytes:
        """Optimize model storage using compression."""
        import lzma
        import pickle

        # Serialize model
        model_bytes = pickle.dumps(model)

        # Compress based on level
        if compression_level == 0:
            return model_bytes
        elif compression_level == 1:
            import zlib

            return zlib.compress(model_bytes)
        elif compression_level == 2:
            return lzma.compress(model_bytes, preset=lzma.PRESET_DEFAULT)
        else:
            return lzma.compress(model_bytes, preset=lzma.PRESET_EXTREME)

    @staticmethod
    def load_optimized_model(compressed_data: bytes, compression_level: int = 1) -> Any:
        """Load model from optimized storage."""
        import lzma
        import pickle

        # Decompress based on level
        if compression_level == 0:
            model_bytes = compressed_data
        elif compression_level == 1:
            import zlib

            model_bytes = zlib.decompress(compressed_data)
        else:
            model_bytes = lzma.decompress(compressed_data)

        # Deserialize model
        return pickle.loads(model_bytes)

    @staticmethod
    def get_model_memory_usage(model: Any) -> dict[str, Any]:
        """Get model memory usage information."""
        import pickle
        import sys

        # Get object size
        size_bytes = sys.getsizeof(model)

        # Get serialized size
        try:
            serialized_size = len(pickle.dumps(model))
        except Exception:
            serialized_size = None

        # Get compressed size
        try:
            compressed_size = len(ModelMemoryOptimizer.optimize_model_storage(model, 1))
        except Exception:
            compressed_size = None

        return {
            "object_size_mb": size_bytes / 1024**2,
            "serialized_size_mb": serialized_size / 1024**2
            if serialized_size
            else None,
            "compressed_size_mb": compressed_size / 1024**2
            if compressed_size
            else None,
            "compression_ratio": serialized_size / compressed_size
            if serialized_size and compressed_size
            else None,
        }


@contextmanager
def memory_profiler(description: str = "Operation"):
    """Context manager for memory profiling."""
    memory_monitor = MemoryMonitor()

    # Get initial memory
    initial_stats = memory_monitor.get_current_stats()
    logger.info(
        f"{description} - Initial memory: {initial_stats.process_memory / 1024**2:.1f}MB"
    )

    try:
        yield memory_monitor
    finally:
        # Get final memory
        final_stats = memory_monitor.get_current_stats()
        memory_diff = (
            final_stats.process_memory - initial_stats.process_memory
        ) / 1024**2

        logger.info(
            f"{description} - Final memory: {final_stats.process_memory / 1024**2:.1f}MB"
        )
        logger.info(f"{description} - Memory change: {memory_diff:+.1f}MB")

        # Force garbage collection
        gc.collect()


class MemoryPool:
    """Memory pool for reusing large objects."""

    def __init__(self, max_size: int = 10):
        """Initialize memory pool."""
        self.max_size = max_size
        self._pools: dict[str, list[Any]] = {}
        self._lock = threading.Lock()

    def get_array(self, shape: tuple, dtype: type = np.float64) -> np.ndarray:
        """Get array from pool or create new one."""
        key = f"array_{shape}_{dtype}"

        with self._lock:
            if key in self._pools and self._pools[key]:
                array = self._pools[key].pop()
                array.fill(0)  # Clear previous data
                return array

        # Create new array
        return np.zeros(shape, dtype=dtype)

    def return_array(self, array: np.ndarray):
        """Return array to pool for reuse."""
        key = f"array_{array.shape}_{array.dtype}"

        with self._lock:
            if key not in self._pools:
                self._pools[key] = []

            if len(self._pools[key]) < self.max_size:
                self._pools[key].append(array)

    def get_dataframe_buffer(self, shape: tuple) -> pd.DataFrame:
        """Get DataFrame buffer from pool."""
        key = f"dataframe_{shape}"

        with self._lock:
            if key in self._pools and self._pools[key]:
                return self._pools[key].pop()

        # Create new DataFrame buffer
        return pd.DataFrame(index=range(shape[0]), columns=range(shape[1]))

    def return_dataframe_buffer(self, df: pd.DataFrame):
        """Return DataFrame buffer to pool."""
        key = f"dataframe_{df.shape}"

        with self._lock:
            if key not in self._pools:
                self._pools[key] = []

            if len(self._pools[key]) < self.max_size:
                # Clear DataFrame
                df.iloc[:, :] = 0
                self._pools[key].append(df)

    def clear(self):
        """Clear all pools."""
        with self._lock:
            self._pools.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "pool_count": len(self._pools),
                "total_objects": sum(len(pool) for pool in self._pools.values()),
                "pools": {key: len(pool) for key, pool in self._pools.items()},
            }


class MemoryOptimizationManager:
    """Central manager for memory optimization."""

    def __init__(self):
        """Initialize memory optimization manager."""
        self.monitor = MemoryMonitor()
        self.dataframe_optimizer = DataFrameOptimizer()
        self.chunked_processor = ChunkedProcessor()
        self.model_optimizer = ModelMemoryOptimizer()
        self.memory_pool = MemoryPool()

    def start_monitoring(self):
        """Start memory monitoring."""
        self.monitor.start_monitoring()

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitor.stop_monitoring()

    def optimize_dataframe(
        self, df: pd.DataFrame, aggressive: bool = False
    ) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        return self.dataframe_optimizer.optimize_dtypes(df, aggressive)

    def process_large_dataset(self, df: pd.DataFrame, processor_func, *args, **kwargs):
        """Process large dataset in memory-efficient chunks."""
        return self.chunked_processor.process_dataframe_chunks(
            df, processor_func, *args, **kwargs
        )

    def get_optimization_report(self) -> dict[str, Any]:
        """Get comprehensive memory optimization report."""
        stats = self.monitor.get_current_stats()
        pool_stats = self.memory_pool.get_stats()

        return {
            "current_memory": {
                "process_memory_mb": stats.process_memory / 1024**2,
                "system_memory_percent": stats.used_percentage,
                "available_memory_mb": stats.available_memory / 1024**2,
            },
            "memory_pool": pool_stats,
            "recommendations": self._get_recommendations(stats),
            "timestamp": time.time(),
        }

    def _get_recommendations(self, stats: MemoryStats) -> list[str]:
        """Get memory optimization recommendations."""
        recommendations = []

        if stats.used_percentage > 90:
            recommendations.append(
                "Critical: System memory usage over 90%. Consider reducing batch sizes."
            )
        elif stats.used_percentage > 80:
            recommendations.append(
                "Warning: High system memory usage. Monitor closely."
            )

        if stats.process_percentage > 50:
            recommendations.append(
                "Process using significant memory. Consider data chunking."
            )

        if not recommendations:
            recommendations.append("Memory usage is within acceptable limits.")

        return recommendations
