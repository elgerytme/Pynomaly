"""Redis-based caching implementation for Pynomaly.

This module provides a Redis caching layer for:
- Model artifacts
- Detection results
- Computed features
- API responses
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import timedelta
from typing import Any, Optional, TypeVar, Union

import redis
from redis.exceptions import ConnectionError, RedisError

from pynomaly.domain.entities import DetectionResult, Detector
from pynomaly.domain.exceptions import CacheError
from pynomaly.infrastructure.config import Settings
# Temporarily disabled telemetry
# from pynomaly.infrastructure.monitoring import get_telemetry, trace_method

# Simple no-op decorator for trace_method while telemetry is disabled
def trace_method(operation_name: str):
    """No-op decorator for telemetry tracing."""
    def decorator(func):
        return func
    return decorator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RedisCache:
    """Redis-based cache implementation with automatic serialization."""
    
    def __init__(self, settings: Settings):
        """Initialize Redis cache with settings.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.enabled = settings.cache_enabled and settings.redis_url is not None
        self._client: Optional[redis.Redis] = None
        # self._telemetry = get_telemetry()  # Temporarily disabled
        
        if self.enabled:
            self._connect()
    
    def _connect(self) -> None:
        """Establish Redis connection."""
        try:
            self._client = redis.from_url(
                self.settings.redis_url,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self._client.ping()
            logger.info("Successfully connected to Redis cache")
            
        except (ConnectionError, RedisError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.enabled = False
            self._client = None
    
    # @trace_method("cache.get")  # Temporarily disabled
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        if not self.enabled or not self._client:
            return default
        
        try:
            value = self._client.get(key)
            
            if value is None:
                if self._telemetry:
                    self._telemetry.record_cache_miss("redis")
                return default
            
            if self._telemetry:
                self._telemetry.record_cache_hit("redis")
            
            # Deserialize based on key prefix
            return self._deserialize(key, value)
            
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return default
    
    # @trace_method("cache.set")  # Temporarily disabled
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default from settings)
            
        Returns:
            Success status
        """
        if not self.enabled or not self._client:
            return False
        
        try:
            # Serialize value
            serialized = self._serialize(key, value)
            
            # Set with TTL
            if ttl is None:
                ttl = self.settings.cache_ttl
            
            if ttl > 0:
                self._client.setex(key, ttl, serialized)
            else:
                self._client.set(key, serialized)
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    # @trace_method("cache.delete")  # Temporarily disabled
    def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        if not self.enabled or not self._client:
            return False
        
        try:
            self._client.delete(key)
            return True
            
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "detector:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self._client:
            return 0
        
        try:
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.warning(f"Cache delete pattern failed for {pattern}: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        if not self.enabled or not self._client:
            return False
        
        try:
            return bool(self._client.exists(key))
            
        except Exception:
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Update key expiration time.
        
        Args:
            key: Cache key
            ttl: New TTL in seconds
            
        Returns:
            Success status
        """
        if not self.enabled or not self._client:
            return False
        
        try:
            return bool(self._client.expire(key, ttl))
            
        except Exception:
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries.
        
        Returns:
            Success status
        """
        if not self.enabled or not self._client:
            return False
        
        try:
            self._client.flushdb()
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def _serialize(self, key: str, value: Any) -> bytes:
        """Serialize value based on key prefix.
        
        Args:
            key: Cache key
            value: Value to serialize
            
        Returns:
            Serialized bytes
        """
        # JSON serialization for simple types
        if key.startswith(("config:", "metadata:", "stats:")):
            return json.dumps(value).encode('utf-8')
        
        # Pickle for complex objects
        return pickle.dumps(value)
    
    def _deserialize(self, key: str, data: bytes) -> Any:
        """Deserialize value based on key prefix.
        
        Args:
            key: Cache key
            data: Serialized data
            
        Returns:
            Deserialized value
        """
        # JSON deserialization
        if key.startswith(("config:", "metadata:", "stats:")):
            return json.loads(data.decode('utf-8'))
        
        # Pickle for complex objects
        return pickle.loads(data)
    
    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Redis connection closed")


class CacheKeys:
    """Cache key generation utilities."""
    
    @staticmethod
    def detector(detector_id: str) -> str:
        """Generate detector cache key."""
        return f"detector:{detector_id}"
    
    @staticmethod
    def detector_model(detector_id: str) -> str:
        """Generate detector model cache key."""
        return f"detector:model:{detector_id}"
    
    @staticmethod
    def detection_result(detector_id: str, dataset_id: str) -> str:
        """Generate detection result cache key."""
        return f"result:{detector_id}:{dataset_id}"
    
    @staticmethod
    def dataset_features(dataset_id: str) -> str:
        """Generate dataset features cache key."""
        return f"features:{dataset_id}"
    
    @staticmethod
    def dataset_stats(dataset_id: str) -> str:
        """Generate dataset statistics cache key."""
        return f"stats:{dataset_id}"
    
    @staticmethod
    def api_response(method: str, path: str, params_hash: str) -> str:
        """Generate API response cache key."""
        return f"api:{method}:{path}:{params_hash}"
    
    @staticmethod
    def experiment(experiment_id: str) -> str:
        """Generate experiment cache key."""
        return f"experiment:{experiment_id}"
    
    @staticmethod
    def algorithm_info(algorithm: str) -> str:
        """Generate algorithm info cache key."""
        return f"algorithm:info:{algorithm}"


class CachedRepository:
    """Mixin for adding caching to repositories."""
    
    def __init__(self, cache: RedisCache, *args, **kwargs):
        """Initialize with cache.
        
        Args:
            cache: Redis cache instance
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        self._cache = cache
    
    def _get_from_cache(self, key: str, loader_func, ttl: Optional[int] = None):
        """Get from cache or load and cache.
        
        Args:
            key: Cache key
            loader_func: Function to load data if not cached
            ttl: Cache TTL
            
        Returns:
            Cached or loaded value
        """
        # Try cache first
        value = self._cache.get(key)
        if value is not None:
            return value
        
        # Load data
        value = loader_func()
        
        # Cache for future
        if value is not None:
            self._cache.set(key, value, ttl)
        
        return value
    
    def _invalidate_cache(self, patterns: list[str]) -> None:
        """Invalidate cache entries.
        
        Args:
            patterns: List of key patterns to invalidate
        """
        for pattern in patterns:
            self._cache.delete_pattern(pattern)


class DetectorCacheDecorator:
    """Decorator for caching detector operations."""
    
    def __init__(self, cache: RedisCache):
        """Initialize decorator with cache.
        
        Args:
            cache: Redis cache instance
        """
        self.cache = cache
    
    def cache_detection(self, ttl: Optional[int] = None):
        """Decorator to cache detection results.
        
        Args:
            ttl: Cache TTL in seconds
            
        Returns:
            Decorator function
        """
        def decorator(func):
            async def wrapper(self, detector_id: str, dataset_id: str, *args, **kwargs):
                # Generate cache key
                key = CacheKeys.detection_result(detector_id, dataset_id)
                
                # Check cache
                cached = self.cache.get(key)
                if cached is not None:
                    logger.debug(f"Detection result cache hit: {key}")
                    return cached
                
                # Execute detection
                result = await func(self, detector_id, dataset_id, *args, **kwargs)
                
                # Cache result
                if result and ttl:
                    self.cache.set(key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def invalidate_detection(self, detector_id: str):
        """Invalidate detection results for a detector.
        
        Args:
            detector_id: Detector ID
        """
        pattern = f"result:{detector_id}:*"
        deleted = self.cache.delete_pattern(pattern)
        logger.info(f"Invalidated {deleted} detection results for detector {detector_id}")


# Global cache instance
_cache: Optional[RedisCache] = None


def init_cache(settings: Settings) -> RedisCache:
    """Initialize global cache instance.
    
    Args:
        settings: Application settings
        
    Returns:
        Cache instance
    """
    global _cache
    _cache = RedisCache(settings)
    return _cache


def get_cache() -> Optional[RedisCache]:
    """Get global cache instance.
    
    Returns:
        Cache instance or None
    """
    return _cache