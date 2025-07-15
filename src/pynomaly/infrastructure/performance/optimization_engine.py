"""
Performance Optimization Engine for Pynomaly.

This module provides comprehensive performance optimization features including
intelligent caching, batch processing, parallel execution, and memory optimization.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import hashlib
import logging
import multiprocessing
import pickle
import queue
import threading
import time
import weakref
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization features."""

    # Caching configuration
    enable_caching: bool = True
    cache_size_mb: int = 512
    cache_ttl_seconds: int = 3600
    cache_strategy: str = "lru"  # lru, lfu, ttl
    cache_persistence: bool = True
    cache_compression: bool = True

    # Parallel processing configuration
    enable_parallel_processing: bool = True
    max_workers: int = field(
        default_factory=lambda: min(8, multiprocessing.cpu_count())
    )
    chunk_size: int = 1000
    thread_pool_size: int = 4
    process_pool_size: int = field(default_factory=lambda: multiprocessing.cpu_count())

    # Batch processing configuration
    enable_batch_processing: bool = True
    batch_size: int = 1000
    max_batch_size: int = 10000
    auto_batch_sizing: bool = True
    batch_timeout_seconds: float = 30.0

    # Memory optimization
    enable_memory_optimization: bool = True
    memory_threshold_mb: float = 1024.0
    gc_threshold: int = 1000
    object_pooling: bool = True
    lazy_loading: bool = True

    # I/O optimization
    enable_io_optimization: bool = True
    io_buffer_size: int = 8192
    async_io: bool = True
    compression_level: int = 6

    # Algorithm-specific optimizations
    enable_algorithm_optimization: bool = True
    use_vectorization: bool = True
    use_gpu_acceleration: bool = False
    numerical_precision: str = "float32"  # float16, float32, float64


class CacheKey:
    """Intelligent cache key generator."""

    @staticmethod
    def generate_key(func_name: str, args: tuple, kwargs: dict[str, Any]) -> str:
        """Generate a unique cache key for function calls."""
        # Create a deterministic representation
        key_parts = [func_name]

        # Handle args
        for arg in args:
            if isinstance(arg, (pd.DataFrame, np.ndarray)):
                # Use hash of data for large objects
                if hasattr(arg, "values"):
                    key_parts.append(f"df_{hash(arg.values.tobytes())}")
                else:
                    key_parts.append(f"arr_{hash(arg.tobytes())}")
            elif hasattr(arg, "__dict__"):
                # For objects with attributes
                key_parts.append(f"obj_{hash(str(sorted(arg.__dict__.items())))}")
            else:
                key_parts.append(str(arg))

        # Handle kwargs
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (pd.DataFrame, np.ndarray)):
                if hasattr(v, "values"):
                    key_parts.append(f"{k}_df_{hash(v.values.tobytes())}")
                else:
                    key_parts.append(f"{k}_arr_{hash(v.tobytes())}")
            else:
                key_parts.append(f"{k}_{v}")

        # Generate hash
        key_string = "_".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()


class IntelligentCache:
    """Advanced caching system with multiple strategies and optimization."""

    def __init__(self, config: OptimizationConfig, storage_path: Path | None = None):
        self.config = config
        self.storage_path = storage_path or Path("cache")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Cache storage
        self._cache: dict[str, dict[str, Any]] = {}
        self._access_times: dict[str, float] = {}
        self._access_counts: dict[str, int] = defaultdict(int)
        self._cache_size: int = 0
        self._max_size: int = config.cache_size_mb * 1024 * 1024  # Convert to bytes

        # Locks for thread safety
        self._lock = threading.RLock()

        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()

        if config.cache_persistence:
            self._start_cleanup_thread()

        logger.info(
            f"Intelligent cache initialized with {config.cache_size_mb}MB capacity"
        )

    def get(self, key: str) -> Any | None:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                return None

            cache_entry = self._cache[key]

            # Check TTL
            if (time.time() - cache_entry["timestamp"]) > self.config.cache_ttl_seconds:
                self._remove_entry(key)
                return None

            # Update access statistics
            self._access_times[key] = time.time()
            self._access_counts[key] += 1

            return cache_entry["data"]

    def set(self, key: str, value: Any, size_hint: int | None = None) -> bool:
        """Set item in cache."""
        with self._lock:
            # Calculate size
            if size_hint:
                entry_size = size_hint
            else:
                entry_size = self._estimate_size(value)

            # Check if we need to make space
            while (self._cache_size + entry_size) > self._max_size and self._cache:
                self._evict_item()

            # Add to cache if there's space
            if (self._cache_size + entry_size) <= self._max_size:
                cache_entry = {
                    "data": value,
                    "size": entry_size,
                    "timestamp": time.time(),
                }

                # Remove existing entry if updating
                if key in self._cache:
                    self._remove_entry(key)

                self._cache[key] = cache_entry
                self._cache_size += entry_size
                self._access_times[key] = time.time()
                self._access_counts[key] += 1

                return True

            return False

    def _evict_item(self) -> None:
        """Evict item based on configured strategy."""
        if not self._cache:
            return

        if self.config.cache_strategy == "lru":
            # Least Recently Used
            oldest_key = min(
                self._access_times.keys(), key=lambda k: self._access_times[k]
            )
            self._remove_entry(oldest_key)
        elif self.config.cache_strategy == "lfu":
            # Least Frequently Used
            least_used_key = min(
                self._access_counts.keys(), key=lambda k: self._access_counts[k]
            )
            self._remove_entry(least_used_key)
        elif self.config.cache_strategy == "ttl":
            # Time To Live - remove oldest
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k]["timestamp"]
            )
            self._remove_entry(oldest_key)

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry_size = self._cache[key]["size"]
            del self._cache[key]
            self._cache_size -= entry_size

        if key in self._access_times:
            del self._access_times[key]

        if key in self._access_counts:
            del self._access_counts[key]

    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (str, bytes)):
                return len(obj)
            else:
                # Fallback to pickle size estimation
                return len(pickle.dumps(obj))
        except Exception:
            # Conservative estimate
            return 1024

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""

        def cleanup_worker():
            while not self._stop_cleanup.wait(60):  # Check every minute
                try:
                    self._cleanup_expired()
                except Exception as e:
                    logger.warning(f"Cache cleanup error: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for key, cache_entry in self._cache.items():
                if (
                    current_time - cache_entry["timestamp"]
                ) > self.config.cache_ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._cache_size = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "total_entries": len(self._cache),
                "total_size_mb": self._cache_size / (1024 * 1024),
                "max_size_mb": self._max_size / (1024 * 1024),
                "hit_ratio": self._calculate_hit_ratio(),
                "most_accessed": self._get_most_accessed(5),
                "strategy": self.config.cache_strategy,
            }

    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        # This would be tracked with additional counters in a full implementation
        return 0.8  # Placeholder

    def _get_most_accessed(self, limit: int) -> list[tuple[str, int]]:
        """Get most accessed cache entries."""
        return sorted(self._access_counts.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]


class BatchProcessor:
    """Intelligent batch processing system."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._batch_queues: dict[str, queue.Queue] = {}
        self._batch_timers: dict[str, threading.Timer] = {}
        self._processors: dict[str, Callable] = {}
        self._lock = threading.Lock()

        logger.info("Batch processor initialized")

    def register_processor(self, name: str, processor_func: Callable) -> None:
        """Register a batch processor function."""
        self._processors[name] = processor_func
        self._batch_queues[name] = queue.Queue()
        logger.debug(f"Registered batch processor: {name}")

    async def add_to_batch(self, processor_name: str, item: Any) -> Any:
        """Add item to batch for processing."""
        if processor_name not in self._processors:
            raise ValueError(f"Unknown processor: {processor_name}")

        batch_queue = self._batch_queues[processor_name]

        # Create future for result
        result_future = asyncio.Future()
        batch_item = {"data": item, "future": result_future, "timestamp": time.time()}

        batch_queue.put(batch_item)

        # Start timer if needed
        self._start_batch_timer(processor_name)

        # Check if batch is full
        if batch_queue.qsize() >= self.config.batch_size:
            await self._process_batch(processor_name)

        return await result_future

    def _start_batch_timer(self, processor_name: str) -> None:
        """Start timer for batch processing."""
        with self._lock:
            if processor_name not in self._batch_timers:
                timer = threading.Timer(
                    self.config.batch_timeout_seconds,
                    lambda: asyncio.create_task(self._process_batch(processor_name)),
                )
                timer.start()
                self._batch_timers[processor_name] = timer

    async def _process_batch(self, processor_name: str) -> None:
        """Process accumulated batch."""
        batch_queue = self._batch_queues[processor_name]
        processor_func = self._processors[processor_name]

        # Extract batch items
        batch_items = []
        while not batch_queue.empty() and len(batch_items) < self.config.max_batch_size:
            try:
                batch_items.append(batch_queue.get_nowait())
            except queue.Empty:
                break

        if not batch_items:
            return

        # Cancel timer
        with self._lock:
            if processor_name in self._batch_timers:
                self._batch_timers[processor_name].cancel()
                del self._batch_timers[processor_name]

        try:
            # Extract data for processing
            batch_data = [item["data"] for item in batch_items]

            # Process batch
            if asyncio.iscoroutinefunction(processor_func):
                results = await processor_func(batch_data)
            else:
                results = processor_func(batch_data)

            # Distribute results
            for item, result in zip(batch_items, results, strict=False):
                if not item["future"].done():
                    item["future"].set_result(result)

        except Exception as e:
            # Set exception for all futures
            for item in batch_items:
                if not item["future"].done():
                    item["future"].set_exception(e)

            logger.error(f"Batch processing error for {processor_name}: {e}")


class ParallelExecutor:
    """Advanced parallel execution engine."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._thread_pool = ThreadPoolExecutor(max_workers=config.thread_pool_size)
        self._process_pool = ProcessPoolExecutor(max_workers=config.process_pool_size)
        self._semaphore = asyncio.Semaphore(config.max_workers)

        logger.info(
            f"Parallel executor initialized with {config.max_workers} max workers"
        )

    async def execute_parallel(
        self,
        func: Callable,
        items: list[Any],
        use_processes: bool = False,
        chunk_size: int | None = None,
    ) -> list[Any]:
        """Execute function in parallel across items."""
        if not items:
            return []

        chunk_size = chunk_size or self.config.chunk_size

        # Split items into chunks
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

        # Choose execution method
        if use_processes:
            executor = self._process_pool
        else:
            executor = self._thread_pool

        # Execute chunks in parallel
        loop = asyncio.get_event_loop()
        tasks = []

        for chunk in chunks:
            async with self._semaphore:
                if use_processes:
                    # For process pool, function must be picklable
                    task = loop.run_in_executor(
                        executor, self._process_chunk, func, chunk
                    )
                else:
                    task = loop.run_in_executor(
                        executor, self._execute_chunk, func, chunk
                    )
                tasks.append(task)

        # Wait for all tasks to complete
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, Exception):
                raise chunk_result
            results.extend(chunk_result)

        return results

    def _execute_chunk(self, func: Callable, chunk: list[Any]) -> list[Any]:
        """Execute function on a chunk of items."""
        return [func(item) for item in chunk]

    def _process_chunk(self, func: Callable, chunk: list[Any]) -> list[Any]:
        """Process chunk in separate process."""
        # This would be used for CPU-intensive tasks
        return [func(item) for item in chunk]

    async def execute_concurrent(
        self, tasks: list[Callable], max_concurrent: int | None = None
    ) -> list[Any]:
        """Execute multiple different tasks concurrently."""
        max_concurrent = max_concurrent or self.config.max_workers
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(task):
            async with semaphore:
                if asyncio.iscoroutinefunction(task):
                    return await task()
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, task)

        return await asyncio.gather(*[execute_with_semaphore(task) for task in tasks])

    def cleanup(self):
        """Clean up executor resources."""
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)


class MemoryOptimizer:
    """Memory optimization utilities."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._object_pools: dict[type, list[Any]] = defaultdict(list)
        self._weak_refs: set[weakref.ref] = set()
        self._gc_counter = 0

        logger.info("Memory optimizer initialized")

    def get_pooled_object(self, obj_type: type, *args, **kwargs) -> Any:
        """Get object from pool or create new one."""
        if not self.config.object_pooling:
            return obj_type(*args, **kwargs)

        pool = self._object_pools[obj_type]

        if pool:
            obj = pool.pop()
            # Reset object if it has a reset method
            if hasattr(obj, "reset"):
                obj.reset(*args, **kwargs)
            return obj

        return obj_type(*args, **kwargs)

    def return_to_pool(self, obj: Any) -> None:
        """Return object to pool for reuse."""
        if not self.config.object_pooling:
            return

        obj_type = type(obj)
        pool = self._object_pools[obj_type]

        # Limit pool size
        if len(pool) < 100:  # Configurable limit
            pool.append(obj)

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        memory = psutil.virtual_memory()
        used_mb = (memory.total - memory.available) / (1024 * 1024)

        return used_mb > self.config.memory_threshold_mb

    def trigger_gc_if_needed(self) -> None:
        """Trigger garbage collection if needed."""
        self._gc_counter += 1

        if self._gc_counter >= self.config.gc_threshold or self.check_memory_pressure():
            collected = gc.collect()
            self._gc_counter = 0

            if collected > 0:
                logger.debug(f"Garbage collection freed {collected} objects")

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if not self.config.enable_memory_optimization:
            return df

        optimized_df = df.copy()

        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=["int64"]).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()

            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif (
                col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max
            ):
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif (
                col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max
            ):
                optimized_df[col] = optimized_df[col].astype(np.int32)

        # Optimize float columns
        for col in optimized_df.select_dtypes(include=["float64"]).columns:
            if self.config.numerical_precision == "float32":
                optimized_df[col] = optimized_df[col].astype(np.float32)
            elif self.config.numerical_precision == "float16":
                optimized_df[col] = optimized_df[col].astype(np.float16)

        # Optimize categorical columns
        for col in optimized_df.select_dtypes(include=["object"]).columns:
            num_unique_values = len(optimized_df[col].unique())
            num_total_values = len(optimized_df[col])

            if num_unique_values / num_total_values < 0.5:
                optimized_df[col] = optimized_df[col].astype("category")

        memory_reduction = (
            df.memory_usage(deep=True).sum()
            - optimized_df.memory_usage(deep=True).sum()
        )
        if memory_reduction > 0:
            logger.debug(
                f"DataFrame memory optimization saved {memory_reduction / (1024*1024):.2f} MB"
            )

        return optimized_df


class PerformanceOptimizationEngine:
    """Main performance optimization engine that coordinates all optimization features."""

    def __init__(self, config: OptimizationConfig, storage_path: Path | None = None):
        self.config = config
        self.storage_path = storage_path or Path("optimization")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.cache = IntelligentCache(config, self.storage_path / "cache")
        self.batch_processor = BatchProcessor(config)
        self.parallel_executor = ParallelExecutor(config)
        self.memory_optimizer = MemoryOptimizer(config)

        # Performance tracking
        self.optimization_stats = defaultdict(int)
        self.performance_history = deque(maxlen=1000)

        logger.info("Performance optimization engine initialized")

    def cached(self, ttl: int | None = None, key_func: Callable | None = None):
        """Decorator for caching function results."""

        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.config.enable_caching:
                    return await func(*args, **kwargs)

                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = CacheKey.generate_key(func.__name__, args, kwargs)

                # Try to get from cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.optimization_stats["cache_hits"] += 1
                    return cached_result

                # Execute function and cache result
                result = await func(*args, **kwargs)
                self.cache.set(cache_key, result)
                self.optimization_stats["cache_misses"] += 1

                return result

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.config.enable_caching:
                    return func(*args, **kwargs)

                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = CacheKey.generate_key(func.__name__, args, kwargs)

                # Try to get from cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.optimization_stats["cache_hits"] += 1
                    return cached_result

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.cache.set(cache_key, result)
                self.optimization_stats["cache_misses"] += 1

                return result

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def batched(self, processor_name: str, batch_size: int | None = None):
        """Decorator for batch processing."""

        def decorator(func: Callable):
            # Register the function as a batch processor
            self.batch_processor.register_processor(processor_name, func)

            @functools.wraps(func)
            async def wrapper(item):
                if not self.config.enable_batch_processing:
                    return (
                        await func([item])
                        if asyncio.iscoroutinefunction(func)
                        else func([item])
                    )

                return await self.batch_processor.add_to_batch(processor_name, item)

            return wrapper

        return decorator

    def parallel(self, use_processes: bool = False, chunk_size: int | None = None):
        """Decorator for parallel execution."""

        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(items: list[Any], **kwargs):
                if not self.config.enable_parallel_processing or len(items) == 1:
                    if asyncio.iscoroutinefunction(func):
                        return [await func(item, **kwargs) for item in items]
                    else:
                        return [func(item, **kwargs) for item in items]

                # Create partial function with kwargs
                partial_func = functools.partial(func, **kwargs) if kwargs else func

                return await self.parallel_executor.execute_parallel(
                    partial_func, items, use_processes, chunk_size
                )

            return wrapper

        return decorator

    def memory_optimized(self, optimize_dataframes: bool = True):
        """Decorator for memory optimization."""

        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Optimize input DataFrames
                if optimize_dataframes:
                    optimized_args = []
                    for arg in args:
                        if isinstance(arg, pd.DataFrame):
                            optimized_args.append(
                                self.memory_optimizer.optimize_dataframe(arg)
                            )
                        else:
                            optimized_args.append(arg)
                    args = tuple(optimized_args)

                # Trigger GC if needed before execution
                self.memory_optimizer.trigger_gc_if_needed()

                # Execute function
                result = await func(*args, **kwargs)

                # Optimize output DataFrames
                if optimize_dataframes and isinstance(result, pd.DataFrame):
                    result = self.memory_optimizer.optimize_dataframe(result)

                # Trigger GC after execution
                self.memory_optimizer.trigger_gc_if_needed()

                return result

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Similar optimization for sync functions
                if optimize_dataframes:
                    optimized_args = []
                    for arg in args:
                        if isinstance(arg, pd.DataFrame):
                            optimized_args.append(
                                self.memory_optimizer.optimize_dataframe(arg)
                            )
                        else:
                            optimized_args.append(arg)
                    args = tuple(optimized_args)

                self.memory_optimizer.trigger_gc_if_needed()
                result = func(*args, **kwargs)

                if optimize_dataframes and isinstance(result, pd.DataFrame):
                    result = self.memory_optimizer.optimize_dataframe(result)

                self.memory_optimizer.trigger_gc_if_needed()
                return result

            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        cache_stats = self.cache.get_stats()

        return {
            "cache_stats": cache_stats,
            "optimization_counters": dict(self.optimization_stats),
            "memory_status": {
                "system_memory_mb": psutil.virtual_memory().used / (1024 * 1024),
                "memory_pressure": self.memory_optimizer.check_memory_pressure(),
            },
            "parallel_execution": {
                "max_workers": self.config.max_workers,
                "thread_pool_size": self.config.thread_pool_size,
                "process_pool_size": self.config.process_pool_size,
            },
        }

    def cleanup(self):
        """Cleanup optimization engine resources."""
        self.parallel_executor.cleanup()
        self.cache.clear()
        logger.info("Performance optimization engine cleaned up")


# Factory function for easy instantiation
def create_optimization_engine(
    cache_size_mb: int = 512,
    max_workers: int | None = None,
    enable_all_optimizations: bool = True,
    storage_path: Path | None = None,
) -> PerformanceOptimizationEngine:
    """Create a performance optimization engine with sensible defaults."""
    config = OptimizationConfig(
        cache_size_mb=cache_size_mb,
        max_workers=max_workers or min(8, multiprocessing.cpu_count()),
        enable_caching=enable_all_optimizations,
        enable_parallel_processing=enable_all_optimizations,
        enable_batch_processing=enable_all_optimizations,
        enable_memory_optimization=enable_all_optimizations,
    )

    return PerformanceOptimizationEngine(config, storage_path)
