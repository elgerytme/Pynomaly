"""Cache adapter implementations for enterprise applications.

This module provides adapters for various caching systems including
Redis, Memcached, and in-memory caches.
"""

from __future__ import annotations

import json
import logging
import pickle
from abc import abstractmethod
from typing import Any

from enterprise_core import Cache, InfrastructureError
from pydantic import Field

from .base import AdapterConfiguration, BaseAdapter, adapter

logger = logging.getLogger(__name__)


class CacheConfiguration(AdapterConfiguration):
    """Configuration for cache adapters."""

    adapter_type: str = Field(..., description="Cache adapter type")
    key_prefix: str = Field(
        default="", description="Key prefix for all cache operations"
    )
    serializer: str = Field(
        default="json", description="Serialization method (json, pickle)"
    )
    default_ttl: int | None = Field(default=3600, description="Default TTL in seconds")
    max_connections: int = Field(
        default=100, description="Maximum number of connections"
    )


class CacheAdapter(BaseAdapter, Cache):
    """Base class for cache adapters."""

    def __init__(self, config: CacheConfiguration) -> None:
        super().__init__(config)
        self.cache_config = config
        self._serializer = self._get_serializer(config.serializer)
        self._deserializer = self._get_deserializer(config.serializer)

    def _get_serializer(self, serializer_type: str):
        """Get the appropriate serializer function."""
        if serializer_type == "json":
            return json.dumps
        elif serializer_type == "pickle":
            return pickle.dumps
        else:
            raise InfrastructureError(
                f"Unsupported serializer: {serializer_type}",
                error_code="UNSUPPORTED_SERIALIZER",
            )

    def _get_deserializer(self, serializer_type: str):
        """Get the appropriate deserializer function."""
        if serializer_type == "json":
            return json.loads
        elif serializer_type == "pickle":
            return pickle.loads
        else:
            raise InfrastructureError(
                f"Unsupported deserializer: {serializer_type}",
                error_code="UNSUPPORTED_DESERIALIZER",
            )

    def _build_key(self, key: str) -> str:
        """Build the full cache key with prefix."""
        if self.cache_config.key_prefix:
            return f"{self.cache_config.key_prefix}:{key}"
        return key

    def _serialize_value(self, value: Any) -> str | bytes:
        """Serialize a value for storage."""
        try:
            return self._serializer(value)
        except Exception as e:
            raise InfrastructureError(
                f"Failed to serialize value: {e}",
                error_code="SERIALIZATION_FAILED",
                cause=e,
            ) from e

    def _deserialize_value(self, value: str | bytes) -> Any:
        """Deserialize a value from storage."""
        try:
            return self._deserializer(value)
        except Exception as e:
            raise InfrastructureError(
                f"Failed to deserialize value: {e}",
                error_code="DESERIALIZATION_FAILED",
                cause=e,
            ) from e

    @abstractmethod
    async def _raw_get(self, key: str) -> str | bytes | None:
        """Get raw value from cache."""
        pass

    @abstractmethod
    async def _raw_set(
        self, key: str, value: str | bytes, ttl: int | None = None
    ) -> None:
        """Set raw value in cache."""
        pass

    @abstractmethod
    async def _raw_delete(self, key: str) -> None:
        """Delete raw value from cache."""
        pass

    @abstractmethod
    async def _raw_exists(self, key: str) -> bool:
        """Check if raw key exists in cache."""
        pass

    @abstractmethod
    async def _raw_clear(self) -> None:
        """Clear all raw values from cache."""
        pass

    # Cache protocol implementation
    async def get(self, key: str) -> Any | None:
        """Get a value from the cache."""
        full_key = self._build_key(key)

        async with self.with_retry():
            try:
                raw_value = await self._raw_get(full_key)
                if raw_value is None:
                    return None

                return self._deserialize_value(raw_value)
            except Exception as e:
                logger.warning(f"Cache get failed for key {key}: {e}")
                return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in the cache with optional TTL."""
        full_key = self._build_key(key)
        effective_ttl = ttl or self.cache_config.default_ttl

        async with self.with_retry():
            try:
                serialized_value = self._serialize_value(value)
                await self._raw_set(full_key, serialized_value, effective_ttl)
            except Exception as e:
                raise InfrastructureError(
                    f"Cache set failed for key {key}: {e}",
                    error_code="CACHE_SET_FAILED",
                    details={"key": key, "ttl": effective_ttl},
                    cause=e,
                ) from e

    async def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        full_key = self._build_key(key)

        async with self.with_retry():
            try:
                await self._raw_delete(full_key)
            except Exception as e:
                raise InfrastructureError(
                    f"Cache delete failed for key {key}: {e}",
                    error_code="CACHE_DELETE_FAILED",
                    details={"key": key},
                    cause=e,
                ) from e

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        full_key = self._build_key(key)

        async with self.with_retry():
            try:
                return await self._raw_exists(full_key)
            except Exception as e:
                logger.warning(f"Cache exists check failed for key {key}: {e}")
                return False

    async def clear(self) -> None:
        """Clear all cached values."""
        async with self.with_retry():
            try:
                await self._raw_clear()
            except Exception as e:
                raise InfrastructureError(
                    f"Cache clear failed: {e}",
                    error_code="CACHE_CLEAR_FAILED",
                    cause=e,
                ) from e


@adapter("redis")
class RedisAdapter(CacheAdapter):
    """Redis adapter for caching."""

    def __init__(self, config: CacheConfiguration) -> None:
        super().__init__(config)
        self._redis: Any | None = None

    async def _create_connection(self) -> Any:
        """Create Redis connection."""
        try:
            import redis.asyncio as redis
        except ImportError:
            raise InfrastructureError(
                "Redis not installed. Install with: pip install 'enterprise-adapters[cache]'",
                error_code="DEPENDENCY_MISSING",
            ) from None

        # Build connection parameters
        if self.config.connection_string:
            self._redis = redis.from_url(
                self.config.connection_string,
                max_connections=self.cache_config.max_connections,
                socket_timeout=self.config.timeout,
                retry_on_timeout=True,
            )
        else:
            self._redis = redis.Redis(
                host=self.config.host or "localhost",
                port=self.config.port or 6379,
                password=self.config.password,
                db=int(self.config.database or 0),
                max_connections=self.cache_config.max_connections,
                socket_timeout=self.config.timeout,
                retry_on_timeout=True,
                ssl=self.config.ssl_enabled,
                ssl_cert_reqs="required" if self.config.ssl_verify else None,
            )

        # Test connection
        await self._redis.ping()
        return self._redis

    async def _close_connection(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def _test_connection(self) -> bool:
        """Test the Redis connection."""
        if not self._redis:
            return False

        try:
            await self._redis.ping()
            return True
        except Exception:
            return False

    async def _raw_get(self, key: str) -> str | bytes | None:
        """Get raw value from Redis."""
        if not self._redis:
            raise InfrastructureError("Redis not connected", error_code="NOT_CONNECTED")

        return await self._redis.get(key)

    async def _raw_set(
        self, key: str, value: str | bytes, ttl: int | None = None
    ) -> None:
        """Set raw value in Redis."""
        if not self._redis:
            raise InfrastructureError("Redis not connected", error_code="NOT_CONNECTED")

        await self._redis.set(key, value, ex=ttl)

    async def _raw_delete(self, key: str) -> None:
        """Delete raw value from Redis."""
        if not self._redis:
            raise InfrastructureError("Redis not connected", error_code="NOT_CONNECTED")

        await self._redis.delete(key)

    async def _raw_exists(self, key: str) -> bool:
        """Check if raw key exists in Redis."""
        if not self._redis:
            raise InfrastructureError("Redis not connected", error_code="NOT_CONNECTED")

        result = await self._redis.exists(key)
        return bool(result)

    async def _raw_clear(self) -> None:
        """Clear all raw values from Redis."""
        if not self._redis:
            raise InfrastructureError("Redis not connected", error_code="NOT_CONNECTED")

        await self._redis.flushdb()

    # Redis-specific methods
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in Redis."""
        if not self._redis:
            raise InfrastructureError("Redis not connected", error_code="NOT_CONNECTED")

        full_key = self._build_key(key)
        async with self.with_retry():
            try:
                return await self._redis.incrby(full_key, amount)
            except Exception as e:
                raise InfrastructureError(
                    f"Redis increment failed for key {key}: {e}",
                    error_code="REDIS_INCR_FAILED",
                    details={"key": key, "amount": amount},
                    cause=e,
                ) from e

    async def set_expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for an existing key."""
        if not self._redis:
            raise InfrastructureError("Redis not connected", error_code="NOT_CONNECTED")

        full_key = self._build_key(key)
        async with self.with_retry():
            try:
                return await self._redis.expire(full_key, ttl)
            except Exception as e:
                raise InfrastructureError(
                    f"Redis expire failed for key {key}: {e}",
                    error_code="REDIS_EXPIRE_FAILED",
                    details={"key": key, "ttl": ttl},
                    cause=e,
                ) from e

    async def get_ttl(self, key: str) -> int:
        """Get the time-to-live for a key."""
        if not self._redis:
            raise InfrastructureError("Redis not connected", error_code="NOT_CONNECTED")

        full_key = self._build_key(key)
        async with self.with_retry():
            try:
                return await self._redis.ttl(full_key)
            except Exception as e:
                raise InfrastructureError(
                    f"Redis TTL check failed for key {key}: {e}",
                    error_code="REDIS_TTL_FAILED",
                    details={"key": key},
                    cause=e,
                ) from e


@adapter("memcached")
class MemcachedAdapter(CacheAdapter):
    """Memcached adapter for caching."""

    def __init__(self, config: CacheConfiguration) -> None:
        super().__init__(config)
        self._client: Any | None = None

    async def _create_connection(self) -> Any:
        """Create Memcached connection."""
        try:
            import aiomcache
        except ImportError:
            raise InfrastructureError(
                "aiomcache not installed. Install with: pip install 'enterprise-adapters[cache]'",
                error_code="DEPENDENCY_MISSING",
            ) from None

        host = self.config.host or "localhost"
        port = self.config.port or 11211

        self._client = aiomcache.Client(host, port)

        # Test connection
        await self._client.get("test_connection")
        return self._client

    async def _close_connection(self) -> None:
        """Close Memcached connection."""
        if self._client:
            self._client.close()
            self._client = None

    async def _test_connection(self) -> bool:
        """Test the Memcached connection."""
        if not self._client:
            return False

        try:
            await self._client.get("test_connection")
            return True
        except Exception:
            return False

    async def _raw_get(self, key: str) -> str | bytes | None:
        """Get raw value from Memcached."""
        if not self._client:
            raise InfrastructureError(
                "Memcached not connected", error_code="NOT_CONNECTED"
            )

        return await self._client.get(key.encode())

    async def _raw_set(
        self, key: str, value: str | bytes, ttl: int | None = None
    ) -> None:
        """Set raw value in Memcached."""
        if not self._client:
            raise InfrastructureError(
                "Memcached not connected", error_code="NOT_CONNECTED"
            )

        if isinstance(value, str):
            value = value.encode()

        await self._client.set(key.encode(), value, exptime=ttl or 0)

    async def _raw_delete(self, key: str) -> None:
        """Delete raw value from Memcached."""
        if not self._client:
            raise InfrastructureError(
                "Memcached not connected", error_code="NOT_CONNECTED"
            )

        await self._client.delete(key.encode())

    async def _raw_exists(self, key: str) -> bool:
        """Check if raw key exists in Memcached."""
        if not self._client:
            raise InfrastructureError(
                "Memcached not connected", error_code="NOT_CONNECTED"
            )

        result = await self._client.get(key.encode())
        return result is not None

    async def _raw_clear(self) -> None:
        """Clear all raw values from Memcached."""
        if not self._client:
            raise InfrastructureError(
                "Memcached not connected", error_code="NOT_CONNECTED"
            )

        await self._client.flush_all()


@adapter("memory")
class InMemoryCacheAdapter(CacheAdapter):
    """In-memory cache adapter for development and testing."""

    def __init__(self, config: CacheConfiguration) -> None:
        super().__init__(config)
        self._cache: dict[str, dict[str, Any]] = {}

    async def _create_connection(self) -> Any:
        """Create in-memory cache (no actual connection needed)."""
        return self._cache

    async def _close_connection(self) -> None:
        """Close in-memory cache."""
        self._cache.clear()

    async def _test_connection(self) -> bool:
        """Test the in-memory cache (always available)."""
        return True

    async def _raw_get(self, key: str) -> str | bytes | None:
        """Get raw value from memory cache."""
        import time

        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check expiration
        if entry["expires_at"] and time.time() > entry["expires_at"]:
            del self._cache[key]
            return None

        return entry["value"]

    async def _raw_set(
        self, key: str, value: str | bytes, ttl: int | None = None
    ) -> None:
        """Set raw value in memory cache."""
        import time

        expires_at = None
        if ttl:
            expires_at = time.time() + ttl

        self._cache[key] = {
            "value": value,
            "expires_at": expires_at,
        }

    async def _raw_delete(self, key: str) -> None:
        """Delete raw value from memory cache."""
        self._cache.pop(key, None)

    async def _raw_exists(self, key: str) -> bool:
        """Check if raw key exists in memory cache."""
        return await self._raw_get(key) is not None

    async def _raw_clear(self) -> None:
        """Clear all raw values from memory cache."""
        self._cache.clear()
