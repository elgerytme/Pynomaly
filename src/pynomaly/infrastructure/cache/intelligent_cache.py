"""Intelligent caching layer with predictive prefetching and adaptive optimization."""

from __future__ import annotations

import asyncio
import logging
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from .redis_cache import RedisCache

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheStrategy(Enum):
    """Cache strategy types."""

    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"


class CompressionType(Enum):
    """Compression types for cache values."""

    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


class SerializationFormat(Enum):
    """Serialization formats for cache values."""

    PICKLE = "pickle"
    JSON = "json"
    MSGPACK = "msgpack"
    NUMPY = "numpy"
    ARROW = "arrow"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    size: int
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    ttl: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    compression: CompressionType = CompressionType.NONE
    serialization: SerializationFormat = SerializationFormat.PICKLE


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    prefetches: int = 0
    compressions: int = 0
    decompressions: int = 0
    serializations: int = 0
    deserializations: int = 0
    total_size: int = 0
    hit_rate: float = 0.0
    avg_access_time: float = 0.0
    compression_ratio: float = 0.0

    def update_hit_rate(self) -> None:
        """Update hit rate calculation."""
        total_requests = self.hits + self.misses
        if total_requests > 0:
            self.hit_rate = self.hits / total_requests


@dataclass
class AccessPattern:
    """Cache access pattern tracking."""

    key: str
    access_times: deque = field(default_factory=lambda: deque(maxlen=100))
    access_frequency: float = 0.0
    last_access: float = field(default_factory=time.time)
    prediction_confidence: float = 0.0

    def record_access(self) -> None:
        """Record access event."""
        current_time = time.time()
        self.access_times.append(current_time)
        self.last_access = current_time

        # Calculate access frequency (accesses per hour)
        if len(self.access_times) > 1:
            time_span = current_time - self.access_times[0]
            if time_span > 0:
                self.access_frequency = len(self.access_times) / (time_span / 3600)

    def predict_next_access(self) -> float:
        """Predict next access time."""
        if len(self.access_times) < 2:
            return time.time() + 3600  # Default 1 hour

        # Calculate average interval between accesses
        intervals = []
        for i in range(1, len(self.access_times)):
            intervals.append(self.access_times[i] - self.access_times[i - 1])

        if intervals:
            avg_interval = statistics.mean(intervals)
            # Adjust confidence based on consistency
            std_dev = statistics.stdev(intervals) if len(intervals) > 1 else 0
            self.prediction_confidence = max(0.1, 1.0 - (std_dev / avg_interval))

            return self.last_access + avg_interval

        return time.time() + 3600


class IntelligentCacheManager:
    """Intelligent cache manager with predictive prefetching and optimization."""

    def __init__(
        self,
        redis_cache: RedisCache,
        max_memory_size: int = 100 * 1024 * 1024,  # 100MB
        compression_threshold: int = 1024,  # 1KB
        prefetch_enabled: bool = True,
        adaptive_ttl: bool = True,
        max_prefetch_threads: int = 4,
    ):
        """Initialize intelligent cache manager.

        Args:
            redis_cache: Redis cache instance
            max_memory_size: Maximum memory cache size in bytes
            compression_threshold: Compress values larger than this
            prefetch_enabled: Enable predictive prefetching
            adaptive_ttl: Enable adaptive TTL based on access patterns
            max_prefetch_threads: Maximum prefetch threads
        """
        self.redis_cache = redis_cache
        self.max_memory_size = max_memory_size
        self.compression_threshold = compression_threshold
        self.prefetch_enabled = prefetch_enabled
        self.adaptive_ttl = adaptive_ttl

        # In-memory L1 cache
        self.memory_cache: dict[str, CacheEntry] = {}
        self.memory_cache_lock = threading.RLock()

        # Statistics and monitoring
        self.stats = CacheStats()
        self.access_patterns: dict[str, AccessPattern] = {}

        # Prefetching
        self.prefetch_executor = ThreadPoolExecutor(max_workers=max_prefetch_threads)
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prefetch_task: asyncio.Task | None = None

        # Optimization
        self.last_optimization = time.time()
        self.optimization_interval = 300  # 5 minutes

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        """Start background tasks for optimization and prefetching."""
        if self.prefetch_enabled:
            self.prefetch_task = asyncio.create_task(self._prefetch_worker())

        # Schedule periodic optimization
        asyncio.create_task(self._periodic_optimization())

    async def get(
        self,
        key: str,
        default: T = None,
        loader: Callable[[], T] | None = None,
        ttl: int | None = None,
    ) -> T:
        """Get value from cache with intelligent optimization.

        Args:
            key: Cache key
            default: Default value if not found
            loader: Function to load value if not cached
            ttl: TTL for new values

        Returns:
            Cached or loaded value
        """
        start_time = time.time()

        try:
            # Track access pattern
            self._track_access(key)

            # Check L1 memory cache first
            memory_entry = self._get_from_memory(key)
            if memory_entry is not None:
                self.stats.hits += 1
                return memory_entry.value

            # Check L2 Redis cache
            redis_value = self.redis_cache.get(key)
            if redis_value is not None:
                self.stats.hits += 1

                # Promote to L1 cache if frequently accessed
                if self._should_promote_to_l1(key):
                    await self._set_in_memory(key, redis_value, ttl)

                return redis_value

            # Cache miss
            self.stats.misses += 1

            # Load value if loader provided
            if loader is not None:
                value = await self._load_value(loader)
                if value is not None:
                    await self.set(key, value, ttl)
                    return value

            # Schedule prefetch if pattern detected
            if self.prefetch_enabled and self._should_prefetch(key):
                await self._schedule_prefetch(key)

            return default

        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return default
        finally:
            # Update access time statistics
            access_time = time.time() - start_time
            self._update_access_time(access_time)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH,
    ) -> bool:
        """Set value in cache with intelligent optimization.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds
            strategy: Cache strategy

        Returns:
            Success status
        """
        try:
            self.stats.sets += 1

            # Determine optimal serialization and compression
            serialization = self._choose_serialization(value)
            compression = self._choose_compression(value)

            # Adaptive TTL based on access patterns
            if self.adaptive_ttl and ttl is None:
                ttl = self._calculate_adaptive_ttl(key)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=self._calculate_size(value),
                timestamp=time.time(),
                ttl=ttl,
                compression=compression,
                serialization=serialization,
            )

            # Execute caching strategy
            if strategy == CacheStrategy.WRITE_THROUGH:
                # Write to both L1 and L2 synchronously
                await self._set_in_memory(key, value, ttl)
                success = self.redis_cache.set(key, value, ttl)
                return success

            elif strategy == CacheStrategy.WRITE_BEHIND:
                # Write to L1 immediately, L2 asynchronously
                await self._set_in_memory(key, value, ttl)
                asyncio.create_task(self._async_redis_set(key, value, ttl))
                return True

            elif strategy == CacheStrategy.WRITE_AROUND:
                # Write only to L2, bypass L1
                return self.redis_cache.set(key, value, ttl)

            elif strategy == CacheStrategy.CACHE_ASIDE:
                # Application manages cache manually
                return self.redis_cache.set(key, value, ttl)

            else:
                # Default to write-through
                await self._set_in_memory(key, value, ttl)
                return self.redis_cache.set(key, value, ttl)

        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            Success status
        """
        try:
            self.stats.deletes += 1

            # Remove from L1 cache
            with self.memory_cache_lock:
                self.memory_cache.pop(key, None)

            # Remove from L2 cache
            redis_success = self.redis_cache.delete(key)

            # Remove access pattern
            self.access_patterns.pop(key, None)

            return redis_success

        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern.

        Args:
            pattern: Key pattern

        Returns:
            Number of keys deleted
        """
        try:
            # Delete from L1 cache
            deleted_count = 0
            with self.memory_cache_lock:
                keys_to_delete = [
                    key
                    for key in self.memory_cache.keys()
                    if self._matches_pattern(key, pattern)
                ]

                for key in keys_to_delete:
                    self.memory_cache.pop(key, None)
                    deleted_count += 1

            # Delete from L2 cache
            redis_deleted = self.redis_cache.delete_pattern(pattern)

            # Remove access patterns
            pattern_keys = [
                key
                for key in self.access_patterns.keys()
                if self._matches_pattern(key, pattern)
            ]

            for key in pattern_keys:
                self.access_patterns.pop(key, None)

            return deleted_count + redis_deleted

        except Exception as e:
            logger.error(f"Cache delete pattern failed for {pattern}: {e}")
            return 0

    async def warm_cache(self, keys_and_loaders: list[tuple[str, Callable]]) -> int:
        """Warm cache with preloaded data.

        Args:
            keys_and_loaders: List of (key, loader_function) tuples

        Returns:
            Number of keys warmed
        """
        warmed_count = 0

        for key, loader in keys_and_loaders:
            try:
                # Skip if already cached
                if await self.exists(key):
                    continue

                # Load and cache value
                value = await self._load_value(loader)
                if value is not None:
                    await self.set(key, value)
                    warmed_count += 1

            except Exception as e:
                logger.error(f"Cache warming failed for key {key}: {e}")

        logger.info(f"Cache warmed with {warmed_count} keys")
        return warmed_count

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        # Check L1 first
        with self.memory_cache_lock:
            if key in self.memory_cache:
                return True

        # Check L2
        return self.redis_cache.exists(key)

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Cache statistics
        """
        self.stats.update_hit_rate()

        # Memory cache statistics
        memory_stats = {
            "entries": len(self.memory_cache),
            "size_bytes": sum(entry.size for entry in self.memory_cache.values()),
            "utilization": sum(entry.size for entry in self.memory_cache.values())
            / self.max_memory_size,
        }

        # Access pattern statistics
        pattern_stats = {
            "tracked_keys": len(self.access_patterns),
            "avg_access_frequency": statistics.mean(
                [p.access_frequency for p in self.access_patterns.values()]
            )
            if self.access_patterns
            else 0,
            "high_confidence_predictions": sum(
                1
                for p in self.access_patterns.values()
                if p.prediction_confidence > 0.8
            ),
        }

        return {
            "cache_stats": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "sets": self.stats.sets,
                "deletes": self.stats.deletes,
                "evictions": self.stats.evictions,
                "prefetches": self.stats.prefetches,
                "hit_rate": self.stats.hit_rate,
                "avg_access_time": self.stats.avg_access_time,
                "compression_ratio": self.stats.compression_ratio,
            },
            "memory_cache": memory_stats,
            "access_patterns": pattern_stats,
            "configuration": {
                "max_memory_size": self.max_memory_size,
                "compression_threshold": self.compression_threshold,
                "prefetch_enabled": self.prefetch_enabled,
                "adaptive_ttl": self.adaptive_ttl,
            },
        }

    def _track_access(self, key: str) -> None:
        """Track access pattern for key."""
        if key not in self.access_patterns:
            self.access_patterns[key] = AccessPattern(key=key)

        self.access_patterns[key].record_access()

    def _get_from_memory(self, key: str) -> CacheEntry | None:
        """Get entry from memory cache."""
        with self.memory_cache_lock:
            entry = self.memory_cache.get(key)
            if entry is None:
                return None

            # Check TTL
            if entry.ttl and time.time() > entry.timestamp + entry.ttl:
                self.memory_cache.pop(key, None)
                self.stats.evictions += 1
                return None

            # Update access info
            entry.access_count += 1
            entry.last_accessed = time.time()

            return entry

    async def _set_in_memory(self, key: str, value: Any, ttl: int | None) -> None:
        """Set entry in memory cache."""
        with self.memory_cache_lock:
            # Check if we need to evict
            while self._should_evict():
                self._evict_lru()

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=self._calculate_size(value),
                timestamp=time.time(),
                ttl=ttl,
            )

            self.memory_cache[key] = entry

    def _should_promote_to_l1(self, key: str) -> bool:
        """Check if key should be promoted to L1 cache."""
        pattern = self.access_patterns.get(key)
        if not pattern:
            return False

        # Promote if frequently accessed
        return pattern.access_frequency > 10  # 10 accesses per hour

    def _should_evict(self) -> bool:
        """Check if memory cache needs eviction."""
        current_size = sum(entry.size for entry in self.memory_cache.values())
        return current_size > self.max_memory_size

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.memory_cache:
            return

        # Find LRU entry
        lru_key = min(
            self.memory_cache.keys(), key=lambda k: self.memory_cache[k].last_accessed
        )

        self.memory_cache.pop(lru_key, None)
        self.stats.evictions += 1

    def _choose_serialization(self, value: Any) -> SerializationFormat:
        """Choose optimal serialization format."""
        if isinstance(value, (dict, list, str, int, float, bool)):
            return SerializationFormat.JSON

        # Check for numpy arrays
        if hasattr(value, "__array__"):
            return SerializationFormat.NUMPY

        # Default to pickle
        return SerializationFormat.PICKLE

    def _choose_compression(self, value: Any) -> CompressionType:
        """Choose optimal compression type."""
        size = self._calculate_size(value)

        if size < self.compression_threshold:
            return CompressionType.NONE

        # For large values, use compression
        return CompressionType.ZLIB

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value."""
        try:
            import sys

            return sys.getsizeof(value)
        except Exception:
            return 1024  # Default estimate

    def _calculate_adaptive_ttl(self, key: str) -> int:
        """Calculate adaptive TTL based on access patterns."""
        pattern = self.access_patterns.get(key)
        if not pattern:
            return 3600  # Default 1 hour

        # Base TTL on access frequency
        if pattern.access_frequency > 100:  # Very frequent
            return 7200  # 2 hours
        elif pattern.access_frequency > 10:  # Frequent
            return 3600  # 1 hour
        else:  # Infrequent
            return 1800  # 30 minutes

    def _should_prefetch(self, key: str) -> bool:
        """Check if key should be prefetched."""
        pattern = self.access_patterns.get(key)
        if not pattern:
            return False

        # Prefetch if we have good prediction confidence
        return pattern.prediction_confidence > 0.6

    async def _schedule_prefetch(self, key: str) -> None:
        """Schedule prefetch for key."""
        if self.prefetch_queue.full():
            return

        await self.prefetch_queue.put(key)

    async def _prefetch_worker(self) -> None:
        """Worker for prefetching cache entries."""
        while True:
            try:
                key = await self.prefetch_queue.get()

                # Skip if already cached
                if await self.exists(key):
                    continue

                # Try to predict and prefetch
                await self._perform_prefetch(key)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")

    async def _perform_prefetch(self, key: str) -> None:
        """Perform prefetch for key."""
        # This is a placeholder - in real implementation,
        # this would use ML models or heuristics to prefetch
        # related keys based on access patterns
        logger.debug(f"Prefetching key: {key}")
        self.stats.prefetches += 1

    async def _periodic_optimization(self) -> None:
        """Periodic cache optimization."""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)
                await self._optimize_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")

    async def _optimize_cache(self) -> None:
        """Optimize cache based on access patterns."""
        # Clean up old access patterns
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hour

        keys_to_remove = [
            key
            for key, pattern in self.access_patterns.items()
            if pattern.last_access < cutoff_time
        ]

        for key in keys_to_remove:
            self.access_patterns.pop(key, None)

        # Update statistics
        self.stats.update_hit_rate()

        logger.debug(
            f"Cache optimization completed, removed {len(keys_to_remove)} old patterns"
        )

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern."""
        # Simple glob-style pattern matching
        import fnmatch

        return fnmatch.fnmatch(key, pattern)

    def _update_access_time(self, access_time: float) -> None:
        """Update access time statistics."""
        if self.stats.hits + self.stats.misses == 0:
            self.stats.avg_access_time = access_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_access_time = (
                alpha * access_time + (1 - alpha) * self.stats.avg_access_time
            )

    async def _async_redis_set(self, key: str, value: Any, ttl: int | None) -> None:
        """Asynchronous Redis set operation."""
        try:
            self.redis_cache.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Async Redis set failed for key {key}: {e}")

    async def _load_value(self, loader: Callable) -> Any:
        """Load value using loader function."""
        try:
            if asyncio.iscoroutinefunction(loader):
                return await loader()
            else:
                return loader()
        except Exception as e:
            logger.error(f"Value loader failed: {e}")
            return None

    async def close(self) -> None:
        """Close cache manager and cleanup resources."""
        try:
            # Cancel prefetch task
            if self.prefetch_task:
                self.prefetch_task.cancel()
                try:
                    await self.prefetch_task
                except asyncio.CancelledError:
                    pass

            # Shutdown prefetch executor
            self.prefetch_executor.shutdown(wait=False)

            # Clear memory cache
            with self.memory_cache_lock:
                self.memory_cache.clear()

            # Clear access patterns
            self.access_patterns.clear()

            logger.info("Intelligent cache manager closed")

        except Exception as e:
            logger.error(f"Error closing cache manager: {e}")


# Global intelligent cache manager
_intelligent_cache_manager: IntelligentCacheManager | None = None


def get_intelligent_cache_manager(
    redis_cache: RedisCache | None = None,
    max_memory_size: int = 100 * 1024 * 1024,
    compression_threshold: int = 1024,
    prefetch_enabled: bool = True,
    adaptive_ttl: bool = True,
    max_prefetch_threads: int = 4,
) -> IntelligentCacheManager:
    """Get or create global intelligent cache manager."""
    global _intelligent_cache_manager

    if _intelligent_cache_manager is None:
        if redis_cache is None:
            raise ValueError(
                "Redis cache instance required for intelligent cache manager"
            )

        _intelligent_cache_manager = IntelligentCacheManager(
            redis_cache=redis_cache,
            max_memory_size=max_memory_size,
            compression_threshold=compression_threshold,
            prefetch_enabled=prefetch_enabled,
            adaptive_ttl=adaptive_ttl,
            max_prefetch_threads=max_prefetch_threads,
        )

    return _intelligent_cache_manager


async def close_intelligent_cache_manager() -> None:
    """Close global intelligent cache manager."""
    global _intelligent_cache_manager

    if _intelligent_cache_manager:
        await _intelligent_cache_manager.close()
        _intelligent_cache_manager = None
