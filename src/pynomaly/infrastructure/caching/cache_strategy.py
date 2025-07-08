"""Advanced caching strategies for Pynomaly performance optimization."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Configuration for caching system."""

    redis_url: str = "redis://localhost:6379/0"
    default_ttl: int = 3600  # 1 hour
    max_memory_cache_size: int = 1000  # items
    enable_compression: bool = True
    enable_metrics: bool = True
    key_prefix: str = "pynomaly"


class CacheMetrics(BaseModel):
    """Cache performance metrics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with LRU eviction."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """Initialize memory cache."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self.metrics = CacheMetrics()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key not in self._cache:
            self.metrics.misses += 1
            return None

        entry = self._cache[key]

        # Check expiration
        if entry["expires_at"] and datetime.utcnow() > entry["expires_at"]:
            await self.delete(key)
            self.metrics.misses += 1
            return None

        # Update access order (LRU)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        self.metrics.hits += 1
        return entry["value"]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl > 0 else None

        # Evict if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            await self._evict_lru()

        self._cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.utcnow(),
        }

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        self.metrics.sets += 1
        self.metrics.total_size = len(self._cache)
        return True

    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self.metrics.deletes += 1
            self.metrics.total_size = len(self._cache)
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        return key in self._cache

    async def clear(self) -> bool:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        self.metrics.total_size = 0
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "backend": "memory",
            "size": len(self._cache),
            "max_size": self.max_size,
            "metrics": self.metrics.dict(),
        }

    async def _evict_lru(self):
        """Evict least recently used item."""
        if self._access_order:
            lru_key = self._access_order[0]
            await self.delete(lru_key)
            self.metrics.evictions += 1


class RedisCache(CacheBackend):
    """Redis cache backend with advanced features."""

    def __init__(
        self, redis_url: str, default_ttl: int = 3600, enable_compression: bool = True
    ):
        """Initialize Redis cache."""
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.metrics = CacheMetrics()

        # Initialize Redis connection
        self._redis = redis.from_url(redis_url, decode_responses=False)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            data = self._redis.get(key)
            if data is None:
                self.metrics.misses += 1
                return None

            # Decompress if enabled
            if self.enable_compression:
                import zlib

                data = zlib.decompress(data)

            # Deserialize
            value = pickle.loads(data)
            self.metrics.hits += 1
            return value

        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self.metrics.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            ttl = ttl or self.default_ttl

            # Serialize
            data = pickle.dumps(value)

            # Compress if enabled
            if self.enable_compression:
                import zlib

                data = zlib.compress(data)

            # Store in Redis
            result = (
                self._redis.setex(key, ttl, data)
                if ttl > 0
                else self._redis.set(key, data)
            )

            if result:
                self.metrics.sets += 1
                return True
            return False

        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            result = self._redis.delete(key)
            if result:
                self.metrics.deletes += 1
                return True
            return False

        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            return bool(self._redis.exists(key))
        except Exception as e:
            logger.error(f"Redis cache exists error: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            self._redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self._redis.info()
            return {
                "backend": "redis",
                "redis_info": {
                    "memory_usage": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                },
                "metrics": self.metrics.dict(),
            }
        except Exception as e:
            logger.error(f"Redis cache stats error: {e}")
            return {"backend": "redis", "error": str(e)}


class HybridCache(CacheBackend):
    """Hybrid cache combining memory and Redis backends."""

    def __init__(
        self,
        l1_cache: MemoryCache,
        l2_cache: RedisCache,
        l1_ttl_ratio: float = 0.1,  # L1 cache has 10% of L2 TTL
    ):
        """Initialize hybrid cache."""
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.l1_ttl_ratio = l1_ttl_ratio
        self.metrics = CacheMetrics()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from hybrid cache (L1 first, then L2)."""
        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            self.metrics.hits += 1
            return value

        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            # Populate L1 cache
            l1_ttl = int(self.l2_cache.default_ttl * self.l1_ttl_ratio)
            await self.l1_cache.set(key, value, l1_ttl)
            self.metrics.hits += 1
            return value

        self.metrics.misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in both cache layers."""
        ttl = ttl or self.l2_cache.default_ttl
        l1_ttl = int(ttl * self.l1_ttl_ratio)

        # Set in both caches
        l1_success = await self.l1_cache.set(key, value, l1_ttl)
        l2_success = await self.l2_cache.set(key, value, ttl)

        if l1_success or l2_success:
            self.metrics.sets += 1
            return True
        return False

    async def delete(self, key: str) -> bool:
        """Delete value from both cache layers."""
        l1_success = await self.l1_cache.delete(key)
        l2_success = await self.l2_cache.delete(key)

        if l1_success or l2_success:
            self.metrics.deletes += 1
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in either cache layer."""
        return await self.l1_cache.exists(key) or await self.l2_cache.exists(key)

    async def clear(self) -> bool:
        """Clear both cache layers."""
        l1_success = await self.l1_cache.clear()
        l2_success = await self.l2_cache.clear()
        return l1_success and l2_success

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics from both cache layers."""
        l1_stats = await self.l1_cache.get_stats()
        l2_stats = await self.l2_cache.get_stats()

        return {
            "backend": "hybrid",
            "l1_cache": l1_stats,
            "l2_cache": l2_stats,
            "combined_metrics": self.metrics.dict(),
        }


class CacheKeyGenerator:
    """Generate consistent cache keys."""

    @staticmethod
    def generate_key(prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Convert all arguments to strings
        key_parts = [prefix]

        for arg in args:
            if isinstance(arg, (UUID, str)):
                key_parts.append(str(arg))
            elif isinstance(arg, (dict, list)):
                key_parts.append(
                    hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest()
                )
            else:
                key_parts.append(str(arg))

        for k, v in sorted(kwargs.items()):
            if isinstance(v, (dict, list)):
                v_str = hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest()
            else:
                v_str = str(v)
            key_parts.append(f"{k}:{v_str}")

        return ":".join(key_parts)

    @staticmethod
    def detector_key(detector_id: UUID) -> str:
        """Generate key for detector caching."""
        return CacheKeyGenerator.generate_key("detector", detector_id)

    @staticmethod
    def dataset_key(dataset_id: UUID) -> str:
        """Generate key for dataset caching."""
        return CacheKeyGenerator.generate_key("dataset", dataset_id)

    @staticmethod
    def detection_result_key(result_id: UUID) -> str:
        """Generate key for detection result caching."""
        return CacheKeyGenerator.generate_key("result", result_id)

    @staticmethod
    def algorithm_summary_key() -> str:
        """Generate key for algorithm summary caching."""
        return CacheKeyGenerator.generate_key("algorithm_summary")

    @staticmethod
    def search_key(
        entity_type: str, filters: Dict[str, Any], page: int, page_size: int
    ) -> str:
        """Generate key for search result caching."""
        return CacheKeyGenerator.generate_key(
            "search", entity_type, filters=filters, page=page, page_size=page_size
        )


class CacheManager:
    """Central cache management system."""

    def __init__(self, config: CacheConfig):
        """Initialize cache manager."""
        self.config = config
        self.backend = self._create_backend()
        self.key_generator = CacheKeyGenerator()

    def _create_backend(self) -> CacheBackend:
        """Create appropriate cache backend."""
        try:
            # Try Redis backend first
            redis_cache = RedisCache(
                self.config.redis_url,
                self.config.default_ttl,
                self.config.enable_compression,
            )

            # Test Redis connection
            redis_cache._redis.ping()

            # Create hybrid cache with memory L1 and Redis L2
            memory_cache = MemoryCache(
                self.config.max_memory_cache_size,
                int(self.config.default_ttl * 0.1),  # L1 has shorter TTL
            )

            return HybridCache(memory_cache, redis_cache)

        except Exception as e:
            logger.warning(f"Redis not available, falling back to memory cache: {e}")
            return MemoryCache(
                self.config.max_memory_cache_size, self.config.default_ttl
            )

    async def get_detector(self, detector_id: UUID) -> Optional[Any]:
        """Get cached detector."""
        key = self.key_generator.detector_key(detector_id)
        return await self.backend.get(key)

    async def set_detector(
        self, detector_id: UUID, detector: Any, ttl: Optional[int] = None
    ) -> bool:
        """Cache detector."""
        key = self.key_generator.detector_key(detector_id)
        return await self.backend.set(key, detector, ttl)

    async def get_dataset(self, dataset_id: UUID) -> Optional[Any]:
        """Get cached dataset."""
        key = self.key_generator.dataset_key(dataset_id)
        return await self.backend.get(key)

    async def set_dataset(
        self, dataset_id: UUID, dataset: Any, ttl: Optional[int] = None
    ) -> bool:
        """Cache dataset."""
        key = self.key_generator.dataset_key(dataset_id)
        return await self.backend.set(key, dataset, ttl)

    async def get_detection_result(self, result_id: UUID) -> Optional[Any]:
        """Get cached detection result."""
        key = self.key_generator.detection_result_key(result_id)
        return await self.backend.get(key)

    async def set_detection_result(
        self, result_id: UUID, result: Any, ttl: Optional[int] = None
    ) -> bool:
        """Cache detection result."""
        key = self.key_generator.detection_result_key(result_id)
        return await self.backend.set(key, result, ttl)

    async def get_search_results(
        self, entity_type: str, filters: Dict[str, Any], page: int, page_size: int
    ) -> Optional[Any]:
        """Get cached search results."""
        key = self.key_generator.search_key(entity_type, filters, page, page_size)
        return await self.backend.get(key)

    async def set_search_results(
        self,
        entity_type: str,
        filters: Dict[str, Any],
        page: int,
        page_size: int,
        results: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache search results."""
        key = self.key_generator.search_key(entity_type, filters, page, page_size)
        # Search results have shorter TTL
        search_ttl = ttl or (self.config.default_ttl // 4)
        return await self.backend.set(key, results, search_ttl)

    async def invalidate_entity(self, entity_type: str, entity_id: UUID):
        """Invalidate all cached data for an entity."""
        if entity_type == "detector":
            key = self.key_generator.detector_key(entity_id)
            await self.backend.delete(key)
            # Also invalidate algorithm summary
            await self.backend.delete(self.key_generator.algorithm_summary_key())
        elif entity_type == "dataset":
            key = self.key_generator.dataset_key(entity_id)
            await self.backend.delete(key)
        elif entity_type == "result":
            key = self.key_generator.detection_result_key(entity_id)
            await self.backend.delete(key)

    async def invalidate_searches(self, entity_type: str):
        """Invalidate all search results for an entity type."""
        # Note: This is a simplified implementation
        # In production, you might want to track search keys or use Redis patterns
        logger.info(f"Search invalidation requested for {entity_type}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return await self.backend.get_stats()

    async def clear_cache(self) -> bool:
        """Clear all cache entries."""
        return await self.backend.clear()


# Cache decorators for easy use


def cache_result(cache_manager: CacheManager, ttl: Optional[int] = None):
    """Decorator to cache function results."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = CacheKeyGenerator.generate_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = await cache_manager.backend.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )
            await cache_manager.backend.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


def invalidate_cache(cache_manager: CacheManager, entity_type: str):
    """Decorator to invalidate cache after function execution."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )

            # Extract entity ID from result or arguments
            entity_id = None
            if hasattr(result, "id"):
                entity_id = result.id
            elif args and hasattr(args[0], "id"):
                entity_id = args[0].id

            if entity_id:
                await cache_manager.invalidate_entity(entity_type, entity_id)

            return result

        return wrapper

    return decorator
