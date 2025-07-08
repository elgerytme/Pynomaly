"""Advanced caching system for performance optimization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass

    @abstractmethod
    async def clear(self, pattern: str | None = None) -> int:
        """Clear cache entries, optionally by pattern."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache implementation with LRU eviction."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """Initialize memory cache."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, dict[str, Any]] = {}
        self._access_times: dict[str, float] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def get(self, key: str) -> Any | None:
        """Get value from memory cache."""
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        # Check expiration
        if entry["expires_at"] and time.time() > entry["expires_at"]:
            await self.delete(key)
            self._misses += 1
            return None

        # Update access time for LRU
        self._access_times[key] = time.time()
        self._hits += 1

        return entry["value"]

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in memory cache."""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None

        # Evict if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            await self._evict_lru()

        self._cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time(),
        }
        self._access_times[key] = time.time()

        return True

    async def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        if key in self._cache:
            del self._cache[key]
            self._access_times.pop(key, None)
            return True
        return False

    async def clear(self, pattern: str | None = None) -> int:
        """Clear cache entries."""
        if pattern is None:
            count = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            return count

        # Pattern matching
        keys_to_delete = [k for k in self._cache.keys() if pattern in k]
        for key in keys_to_delete:
            await self.delete(key)

        return len(keys_to_delete)

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return await self.get(key) is not None

    def get_stats(self) -> dict[str, Any]:
        """Get memory cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "backend": "memory",
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "evictions": self._evictions,
            "utilization_percent": round(len(self._cache) / self.max_size * 100, 2),
        }

    async def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._access_times:
            return

        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        await self.delete(lru_key)
        self._evictions += 1


class RedisCache(CacheBackend):
    """Redis cache implementation (requires redis-py)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        default_ttl: int = 3600,
        key_prefix: str = "pynomaly:",
    ):
        """Initialize Redis cache."""
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._redis = None
        self._hits = 0
        self._misses = 0

    async def _get_redis(self):
        """Get Redis connection (lazy initialization)."""
        if self._redis is None:
            try:
                import redis.asyncio as redis

                self._redis = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=False,  # We handle serialization manually
                )
                await self._redis.ping()
                logger.info("Connected to Redis cache")
            except ImportError:
                logger.warning("redis-py not available, falling back to memory cache")
                return None
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                return None

        return self._redis

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Any | None:
        """Get value from Redis cache."""
        redis = await self._get_redis()
        if not redis:
            return None

        try:
            prefixed_key = self._make_key(key)
            data = await redis.get(prefixed_key)

            if data is None:
                self._misses += 1
                return None

            self._hits += 1
            return pickle.loads(data)

        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            self._misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in Redis cache."""
        redis = await self._get_redis()
        if not redis:
            return False

        try:
            prefixed_key = self._make_key(key)
            serialized_value = pickle.dumps(value)
            ttl = ttl or self.default_ttl

            result = await redis.setex(prefixed_key, ttl, serialized_value)
            return bool(result)

        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        redis = await self._get_redis()
        if not redis:
            return False

        try:
            prefixed_key = self._make_key(key)
            result = await redis.delete(prefixed_key)
            return bool(result)

        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

    async def clear(self, pattern: str | None = None) -> int:
        """Clear Redis cache entries."""
        redis = await self._get_redis()
        if not redis:
            return 0

        try:
            if pattern is None:
                # Clear all keys with our prefix
                pattern = f"{self.key_prefix}*"
            else:
                pattern = f"{self.key_prefix}*{pattern}*"

            keys = await redis.keys(pattern)
            if keys:
                result = await redis.delete(*keys)
                return result
            return 0

        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        redis = await self._get_redis()
        if not redis:
            return False

        try:
            prefixed_key = self._make_key(key)
            result = await redis.exists(prefixed_key)
            return bool(result)

        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get Redis cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "backend": "redis",
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "connected": self._redis is not None,
        }


class CacheManager:
    """High-level cache manager with multiple backends and strategies."""

    def __init__(self, backend: CacheBackend, enable_metrics: bool = True):
        """Initialize cache manager."""
        self.backend = backend
        self.enable_metrics = enable_metrics
        self._operation_times: list[float] = []

    async def get(self, key: str) -> Any | None:
        """Get value from cache with metrics."""
        start_time = time.perf_counter()

        try:
            result = await self.backend.get(key)

            if self.enable_metrics:
                operation_time = time.perf_counter() - start_time
                self._operation_times.append(operation_time)

                # Keep only last 1000 operations for metrics
                if len(self._operation_times) > 1000:
                    self._operation_times = self._operation_times[-1000:]

            return result

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with metrics."""
        start_time = time.perf_counter()

        try:
            result = await self.backend.set(key, value, ttl)

            if self.enable_metrics:
                operation_time = time.perf_counter() - start_time
                self._operation_times.append(operation_time)

            return result

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def get_or_set(
        self, key: str, factory: Callable, ttl: int | None = None
    ) -> Any:
        """Get value from cache or set it using factory function."""
        # Try to get from cache first
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value

        # Generate value using factory
        try:
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()

            # Cache the value
            await self.set(key, value, ttl)
            return value

        except Exception as e:
            logger.error(f"Factory function error for key {key}: {e}")
            raise

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return await self.backend.delete(key)

    async def clear(self, pattern: str | None = None) -> int:
        """Clear cache entries."""
        return await self.backend.clear(pattern)

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return await self.backend.exists(key)

    def invalidate_pattern(self, pattern: str) -> asyncio.Task:
        """Asynchronously invalidate cache entries matching pattern."""
        return asyncio.create_task(self.clear(pattern))

    def get_performance_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        backend_stats = self.backend.get_stats()

        if self._operation_times:
            avg_time = sum(self._operation_times) / len(self._operation_times)
            max_time = max(self._operation_times)
            min_time = min(self._operation_times)
        else:
            avg_time = max_time = min_time = 0.0

        backend_stats.update(
            {
                "avg_operation_time_ms": round(avg_time * 1000, 3),
                "max_operation_time_ms": round(max_time * 1000, 3),
                "min_operation_time_ms": round(min_time * 1000, 3),
                "total_operations": len(self._operation_times),
            }
        )

        return backend_stats


class CacheKey:
    """Utility class for generating consistent cache keys."""

    @staticmethod
    def generate(namespace: str, *args, **kwargs) -> str:
        """Generate a cache key from namespace and arguments."""
        # Create deterministic string from args and kwargs
        key_data = {
            "namespace": namespace,
            "args": [str(arg) for arg in args],
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        }

        # Serialize and hash for consistent keys
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        return f"{namespace}:{key_hash}"

    @staticmethod
    def for_detector(detector_id: str | UUID, operation: str = "default") -> str:
        """Generate cache key for detector operations."""
        return CacheKey.generate("detector", str(detector_id), operation)

    @staticmethod
    def for_dataset(dataset_id: str | UUID, operation: str = "default") -> str:
        """Generate cache key for dataset operations."""
        return CacheKey.generate("dataset", str(dataset_id), operation)

    @staticmethod
    def for_detection_result(
        detector_id: str | UUID,
        dataset_id: str | UUID,
        operation: str = "result",
    ) -> str:
        """Generate cache key for detection results."""
        return CacheKey.generate(
            "detection", str(detector_id), str(dataset_id), operation
        )

    @staticmethod
    def for_user(user_id: str | UUID, operation: str = "profile") -> str:
        """Generate cache key for user operations."""
        return CacheKey.generate("user", str(user_id), operation)

    @staticmethod
    def for_stats(entity_type: str, time_window: str = "1h") -> str:
        """Generate cache key for statistical data."""
        return CacheKey.generate("stats", entity_type, time_window)


# Cache decorators


def cached(
    ttl: int = 3600,
    key_prefix: str = "func",
    cache_manager: CacheManager | None = None,
):
    """Decorator to cache function results."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            nonlocal cache_manager
            if cache_manager is None:
                # Use default memory cache
                cache_manager = CacheManager(MemoryCache())

            # Generate cache key
            cache_key = CacheKey.generate(
                f"{key_prefix}:{func.__name__}", *args, **kwargs
            )

            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")

            return result

        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, run in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def cache_invalidate(pattern: str, cache_manager: CacheManager | None = None):
    """Decorator to invalidate cache entries after function execution."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            nonlocal cache_manager
            if cache_manager is None:
                cache_manager = CacheManager(MemoryCache())

            # Execute function first
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Invalidate cache pattern
            invalidated_count = await cache_manager.clear(pattern)
            logger.debug(
                f"Invalidated {invalidated_count} cache entries with pattern: {pattern}"
            )

            return result

        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global cache instance
_global_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        # Default to memory cache
        _global_cache_manager = CacheManager(MemoryCache(max_size=5000))
    return _global_cache_manager


def configure_cache(cache_type: str = "memory", **kwargs) -> CacheManager:
    """Configure global cache manager."""
    global _global_cache_manager

    if cache_type == "memory":
        backend = MemoryCache(
            max_size=kwargs.get("max_size", 5000),
            default_ttl=kwargs.get("default_ttl", 3600),
        )
    elif cache_type == "redis":
        backend = RedisCache(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 6379),
            db=kwargs.get("db", 0),
            password=kwargs.get("password"),
            default_ttl=kwargs.get("default_ttl", 3600),
            key_prefix=kwargs.get("key_prefix", "pynomaly:"),
        )
    else:
        raise ValueError(f"Unsupported cache type: {cache_type}")

    _global_cache_manager = CacheManager(
        backend, enable_metrics=kwargs.get("enable_metrics", True)
    )
    logger.info(f"Configured {cache_type} cache manager")

    return _global_cache_manager
