"""Advanced caching service with multi-level strategies and intelligent eviction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np

# Optional caching libraries
try:
    import redis
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import lz4.frame

    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd

    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import msgpack

    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

T = TypeVar("T")


class CacheLevel(Enum):
    """Cache levels for multi-tier caching."""

    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


class CompressionAlgorithm(Enum):
    """Compression algorithms for cache data."""

    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    GZIP = "gzip"


class SerializationFormat(Enum):
    """Serialization formats for cache data."""

    PICKLE = "pickle"
    JSON = "json"
    MSGPACK = "msgpack"
    NUMPY = "numpy"


@dataclass
class CacheConfig:
    """Configuration for caching service."""

    # Cache levels
    enable_l1_memory: bool = True
    enable_l2_redis: bool = True
    enable_l3_disk: bool = False

    # Memory cache (L1)
    l1_max_size_mb: int = 256
    l1_max_entries: int = 10000
    l1_ttl_seconds: int = 300

    # Redis cache (L2)
    redis_url: str = "redis://localhost:6379"
    redis_cluster_nodes: list[str] | None = None
    l2_ttl_seconds: int = 3600
    redis_pool_size: int = 20

    # Disk cache (L3)
    l3_cache_dir: str = "/tmp/pynomaly_cache"
    l3_max_size_gb: int = 10
    l3_ttl_seconds: int = 86400

    # Eviction policies
    l1_eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    l2_eviction_policy: EvictionPolicy = EvictionPolicy.TTL
    l3_eviction_policy: EvictionPolicy = EvictionPolicy.LRU

    # Compression
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4
    compression_min_size_bytes: int = 1024
    compression_level: int = 1

    # Serialization
    default_serialization: SerializationFormat = SerializationFormat.PICKLE
    model_serialization: SerializationFormat = SerializationFormat.PICKLE
    feature_serialization: SerializationFormat = SerializationFormat.NUMPY

    # Performance
    async_operations: bool = True
    batch_operations: bool = True
    prefetch_enabled: bool = True
    write_through: bool = False
    write_behind: bool = True

    # Monitoring
    enable_metrics: bool = True
    metrics_interval_seconds: int = 60
    enable_cache_warming: bool = True


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""

    key: str
    value: T
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    def touch(self) -> None:
        """Update access information."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    # Hit/miss statistics
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0

    # Size statistics
    l1_size_bytes: int = 0
    l1_entry_count: int = 0
    l2_size_bytes: int = 0
    l2_entry_count: int = 0
    l3_size_bytes: int = 0
    l3_entry_count: int = 0

    # Performance statistics
    avg_get_latency_ms: float = 0.0
    avg_set_latency_ms: float = 0.0
    compression_ratio: float = 1.0
    eviction_count: int = 0

    # Operations per second
    gets_per_second: float = 0.0
    sets_per_second: float = 0.0

    @property
    def l1_hit_rate(self) -> float:
        """Calculate L1 hit rate."""
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        """Calculate L2 hit rate."""
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        """Calculate overall hit rate."""
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        total_requests = total_hits + max(
            self.l1_misses, self.l2_misses, self.l3_misses
        )
        return total_hits / total_requests if total_requests > 0 else 0.0


class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> bool:
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
        """Clear all entries from cache."""
        pass

    @abstractmethod
    async def get_size(self) -> tuple[int, int]:
        """Get cache size (bytes, entries)."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with LRU eviction."""

    def __init__(self, max_size_mb: int = 256, max_entries: int = 10000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self._cache: dict[str, CacheEntry[bytes]] = {}
        self._access_order: list[str] = []
        self._current_size_bytes = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> bytes | None:
        """Get value from memory cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                if entry.is_expired:
                    await self._remove_entry(key)
                    return None

                entry.touch()
                self._update_access_order(key)
                return entry.value

            return None

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> bool:
        """Set value in memory cache."""
        async with self._lock:
            value_size = len(value)

            # Check if we need to evict entries
            await self._ensure_capacity(value_size)

            # Remove existing entry if present
            if key in self._cache:
                await self._remove_entry(key)

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                size_bytes=value_size,
                ttl_seconds=ttl_seconds,
            )

            self._cache[key] = entry
            self._access_order.append(key)
            self._current_size_bytes += value_size

            return True

    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired:
                    await self._remove_entry(key)
                    return False
                return True
            return False

    async def clear(self) -> bool:
        """Clear all entries from memory cache."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_size_bytes = 0
            return True

    async def get_size(self) -> tuple[int, int]:
        """Get memory cache size."""
        async with self._lock:
            return self._current_size_bytes, len(self._cache)

    async def _ensure_capacity(self, required_bytes: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Evict by size
        while (
            self._current_size_bytes + required_bytes > self.max_size_bytes
            and self._access_order
        ):
            oldest_key = self._access_order[0]
            await self._remove_entry(oldest_key)

        # Evict by count
        while len(self._cache) >= self.max_entries and self._access_order:
            oldest_key = self._access_order[0]
            await self._remove_entry(oldest_key)

    async def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_size_bytes -= entry.size_bytes
            del self._cache[key]

            if key in self._access_order:
                self._access_order.remove(key)

    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)


class RedisCache(CacheBackend):
    """Redis cache backend with clustering support."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._client: aioredis.Redis | None = None
        self._cluster_client = None

    async def _get_client(self) -> aioredis.Redis:
        """Get Redis client."""
        if self._client is None:
            if not REDIS_AVAILABLE:
                raise RuntimeError("Redis not available")

            if self.config.redis_cluster_nodes:
                # Use Redis Cluster
                self._cluster_client = aioredis.RedisCluster.from_url(
                    self.config.redis_url,
                    max_connections=self.config.redis_pool_size,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                )
                self._client = self._cluster_client
            else:
                # Use single Redis instance
                self._client = aioredis.from_url(
                    self.config.redis_url,
                    max_connections=self.config.redis_pool_size,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                )

        return self._client

    async def get(self, key: str) -> bytes | None:
        """Get value from Redis cache."""
        client = await self._get_client()
        value = await client.get(key)
        return value if value else None

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> bool:
        """Set value in Redis cache."""
        client = await self._get_client()

        if ttl_seconds:
            result = await client.setex(key, ttl_seconds, value)
        else:
            result = await client.set(key, value)

        return bool(result)

    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        client = await self._get_client()
        result = await client.delete(key)
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        client = await self._get_client()
        result = await client.exists(key)
        return result > 0

    async def clear(self) -> bool:
        """Clear all entries from Redis cache."""
        client = await self._get_client()
        await client.flushdb()
        return True

    async def get_size(self) -> tuple[int, int]:
        """Get Redis cache size."""
        client = await self._get_client()
        info = await client.info()

        used_memory = info.get("used_memory", 0)
        db_keys = 0

        # Count keys in all databases
        for key, value in info.items():
            if key.startswith("db") and "keys" in str(value):
                # Parse "keys=X,expires=Y" format
                keys_info = str(value).split(",")[0]
                if "=" in keys_info:
                    db_keys += int(keys_info.split("=")[1])

        return used_memory, db_keys

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()


class CompressionService:
    """Service for compressing/decompressing cache data."""

    def __init__(self, algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4):
        self.algorithm = algorithm

    def compress(self, data: bytes) -> bytes:
        """Compress data using configured algorithm."""
        if self.algorithm == CompressionAlgorithm.NONE:
            return data

        elif self.algorithm == CompressionAlgorithm.LZ4:
            if not LZ4_AVAILABLE:
                return data
            return lz4.frame.compress(data)

        elif self.algorithm == CompressionAlgorithm.ZSTD:
            if not ZSTD_AVAILABLE:
                return data
            compressor = zstd.ZstdCompressor()
            return compressor.compress(data)

        elif self.algorithm == CompressionAlgorithm.GZIP:
            import gzip

            return gzip.compress(data)

        return data

    def decompress(self, data: bytes) -> bytes:
        """Decompress data using configured algorithm."""
        if self.algorithm == CompressionAlgorithm.NONE:
            return data

        try:
            if self.algorithm == CompressionAlgorithm.LZ4:
                if not LZ4_AVAILABLE:
                    return data
                return lz4.frame.decompress(data)

            elif self.algorithm == CompressionAlgorithm.ZSTD:
                if not ZSTD_AVAILABLE:
                    return data
                decompressor = zstd.ZstdDecompressor()
                return decompressor.decompress(data)

            elif self.algorithm == CompressionAlgorithm.GZIP:
                import gzip

                return gzip.decompress(data)

        except Exception:
            # If decompression fails, assume data is not compressed
            return data

        return data


class SerializationService:
    """Service for serializing/deserializing cache data."""

    def serialize(self, obj: Any, format_type: SerializationFormat) -> bytes:
        """Serialize object to bytes."""
        try:
            if format_type == SerializationFormat.PICKLE:
                return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

            elif format_type == SerializationFormat.JSON:
                # Handle numpy arrays and other special types
                if isinstance(obj, np.ndarray):
                    obj = {"__numpy_array__": obj.tolist(), "dtype": str(obj.dtype)}
                return json.dumps(obj, default=str).encode("utf-8")

            elif format_type == SerializationFormat.MSGPACK:
                if not MSGPACK_AVAILABLE:
                    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
                return msgpack.packb(obj, use_bin_type=True)

            elif format_type == SerializationFormat.NUMPY:
                if isinstance(obj, np.ndarray):
                    return obj.tobytes()
                else:
                    # Fallback to pickle for non-numpy objects
                    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception:
            # Fallback to pickle
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, data: bytes, format_type: SerializationFormat) -> Any:
        """Deserialize bytes to object."""
        try:
            if format_type == SerializationFormat.PICKLE:
                return pickle.loads(data)

            elif format_type == SerializationFormat.JSON:
                obj = json.loads(data.decode("utf-8"))
                # Handle numpy arrays
                if isinstance(obj, dict) and "__numpy_array__" in obj:
                    return np.array(obj["__numpy_array__"], dtype=obj["dtype"])
                return obj

            elif format_type == SerializationFormat.MSGPACK:
                if not MSGPACK_AVAILABLE:
                    return pickle.loads(data)
                return msgpack.unpackb(data, raw=False)

            elif format_type == SerializationFormat.NUMPY:
                # This requires knowing the original shape and dtype
                # For now, try to load as pickle fallback
                return pickle.loads(data)

        except Exception:
            # Fallback to pickle
            return pickle.loads(data)

        return pickle.loads(data)


class AdvancedCacheService:
    """Advanced multi-level caching service with intelligent strategies."""

    def __init__(self, config: CacheConfig | None = None):
        """Initialize advanced cache service."""
        self.config = config or CacheConfig()
        self.logger = logging.getLogger(__name__)

        # Cache backends
        self.l1_cache: MemoryCache | None = None
        self.l2_cache: RedisCache | None = None
        self.l3_cache = None  # Disk cache implementation would go here

        # Services
        self.compression_service = CompressionService(self.config.compression_algorithm)
        self.serialization_service = SerializationService()

        # Metrics
        self.metrics = CacheMetrics()
        self._metrics_lock = asyncio.Lock()

        # Operation tracking
        self._operation_times: list[float] = []
        self._last_metrics_update = time.time()

        # Initialize backends
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize cache backends."""
        if self.config.enable_l1_memory:
            self.l1_cache = MemoryCache(
                max_size_mb=self.config.l1_max_size_mb,
                max_entries=self.config.l1_max_entries,
            )

        if self.config.enable_l2_redis:
            self.l2_cache = RedisCache(self.config)

    async def get(
        self,
        key: str,
        serialization_format: SerializationFormat | None = None,
    ) -> Any | None:
        """Get value from cache using multi-level strategy."""
        start_time = time.time()

        try:
            # Normalize key
            normalized_key = self._normalize_key(key)
            format_type = serialization_format or self.config.default_serialization

            # Try L1 cache first
            if self.l1_cache:
                l1_data = await self.l1_cache.get(normalized_key)
                if l1_data is not None:
                    await self._update_metrics("l1_hits")
                    value = await self._deserialize_data(l1_data, format_type)
                    return value
                else:
                    await self._update_metrics("l1_misses")

            # Try L2 cache
            if self.l2_cache:
                l2_data = await self.l2_cache.get(normalized_key)
                if l2_data is not None:
                    await self._update_metrics("l2_hits")

                    # Promote to L1 cache
                    if self.l1_cache:
                        await self.l1_cache.set(
                            normalized_key, l2_data, self.config.l1_ttl_seconds
                        )

                    value = await self._deserialize_data(l2_data, format_type)
                    return value
                else:
                    await self._update_metrics("l2_misses")

            # Try L3 cache (if implemented)
            # ... L3 implementation would go here

            return None

        finally:
            operation_time = (time.time() - start_time) * 1000
            self._operation_times.append(operation_time)
            if len(self._operation_times) > 1000:
                self._operation_times = self._operation_times[-1000:]

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        serialization_format: SerializationFormat | None = None,
        cache_levels: list[CacheLevel] | None = None,
    ) -> bool:
        """Set value in cache using multi-level strategy."""
        start_time = time.time()

        try:
            # Normalize key and serialize value
            normalized_key = self._normalize_key(key)
            format_type = serialization_format or self._get_serialization_format(value)

            serialized_data = await self._serialize_data(value, format_type)

            # Determine which cache levels to use
            if cache_levels is None:
                cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

            success = True

            # Set in specified cache levels
            for level in cache_levels:
                level_success = False

                if level == CacheLevel.L1_MEMORY and self.l1_cache:
                    level_ttl = ttl_seconds or self.config.l1_ttl_seconds
                    level_success = await self.l1_cache.set(
                        normalized_key, serialized_data, level_ttl
                    )

                elif level == CacheLevel.L2_REDIS and self.l2_cache:
                    level_ttl = ttl_seconds or self.config.l2_ttl_seconds
                    level_success = await self.l2_cache.set(
                        normalized_key, serialized_data, level_ttl
                    )

                # elif level == CacheLevel.L3_DISK and self.l3_cache:
                #     # L3 implementation would go here
                #     pass

                if not level_success:
                    success = False
                    self.logger.warning(
                        f"Failed to set key {key} in cache level {level}"
                    )

            return success

        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False

        finally:
            operation_time = (time.time() - start_time) * 1000
            self._operation_times.append(operation_time)

    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        normalized_key = self._normalize_key(key)
        success = True

        # Delete from all cache levels
        if self.l1_cache:
            if not await self.l1_cache.delete(normalized_key):
                success = False

        if self.l2_cache:
            if not await self.l2_cache.delete(normalized_key):
                success = False

        return success

    async def exists(self, key: str) -> bool:
        """Check if key exists in any cache level."""
        normalized_key = self._normalize_key(key)

        # Check L1 first
        if self.l1_cache and await self.l1_cache.exists(normalized_key):
            return True

        # Check L2
        if self.l2_cache and await self.l2_cache.exists(normalized_key):
            return True

        return False

    async def clear(self, cache_levels: list[CacheLevel] | None = None) -> bool:
        """Clear specified cache levels."""
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

        success = True

        for level in cache_levels:
            if level == CacheLevel.L1_MEMORY and self.l1_cache:
                if not await self.l1_cache.clear():
                    success = False

            elif level == CacheLevel.L2_REDIS and self.l2_cache:
                if not await self.l2_cache.clear():
                    success = False

        return success

    async def get_metrics(self) -> CacheMetrics:
        """Get cache performance metrics."""
        async with self._metrics_lock:
            # Update size metrics
            if self.l1_cache:
                l1_size, l1_count = await self.l1_cache.get_size()
                self.metrics.l1_size_bytes = l1_size
                self.metrics.l1_entry_count = l1_count

            if self.l2_cache:
                l2_size, l2_count = await self.l2_cache.get_size()
                self.metrics.l2_size_bytes = l2_size
                self.metrics.l2_entry_count = l2_count

            # Update performance metrics
            if self._operation_times:
                self.metrics.avg_get_latency_ms = sum(self._operation_times) / len(
                    self._operation_times
                )

            return self.metrics

    async def warm_cache(
        self,
        data_provider: callable,
        keys: list[str],
        batch_size: int = 100,
    ) -> int:
        """Warm cache with data from provider function."""
        if not self.config.enable_cache_warming:
            return 0

        warmed_count = 0

        # Process keys in batches
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i : i + batch_size]

            try:
                # Get data for batch
                batch_data = await data_provider(batch_keys)

                # Set in cache
                for key, value in batch_data.items():
                    if await self.set(key, value):
                        warmed_count += 1

            except Exception as e:
                self.logger.error(f"Error warming cache batch: {e}")

        self.logger.info(f"Cache warming completed: {warmed_count}/{len(keys)} entries")
        return warmed_count

    async def _serialize_data(
        self, value: Any, format_type: SerializationFormat
    ) -> bytes:
        """Serialize and optionally compress data."""
        # Serialize
        serialized = self.serialization_service.serialize(value, format_type)

        # Compress if data is large enough
        if (
            len(serialized) >= self.config.compression_min_size_bytes
            and self.config.compression_algorithm != CompressionAlgorithm.NONE
        ):
            compressed = self.compression_service.compress(serialized)

            # Only use compressed version if it's actually smaller
            if len(compressed) < len(serialized):
                return compressed

        return serialized

    async def _deserialize_data(
        self, data: bytes, format_type: SerializationFormat
    ) -> Any:
        """Decompress and deserialize data."""
        # Try decompression first
        decompressed = self.compression_service.decompress(data)

        # Deserialize
        return self.serialization_service.deserialize(decompressed, format_type)

    def _normalize_key(self, key: str) -> str:
        """Normalize cache key."""
        # Create hash for very long keys
        if len(key) > 250:
            return hashlib.sha256(key.encode()).hexdigest()

        # Sanitize key
        return key.replace(" ", "_").replace(":", "_")

    def _get_serialization_format(self, value: Any) -> SerializationFormat:
        """Determine appropriate serialization format for value."""
        if isinstance(value, np.ndarray):
            return self.config.feature_serialization

        # Check if it's a model-like object
        if hasattr(value, "predict") or hasattr(value, "fit"):
            return self.config.model_serialization

        return self.config.default_serialization

    async def _update_metrics(self, metric_name: str) -> None:
        """Update cache metrics."""
        async with self._metrics_lock:
            current_value = getattr(self.metrics, metric_name, 0)
            setattr(self.metrics, metric_name, current_value + 1)

    async def get_batch(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple keys in a single operation for improved performance."""
        start_time = time.time()
        results = {}

        try:
            # Normalize all keys
            normalized_keys = {key: self._normalize_key(key) for key in keys}

            # Try L1 cache first for all keys
            if self.l1_cache:
                for key, normalized_key in normalized_keys.items():
                    value = await self.l1_cache.get(normalized_key)
                    if value is not None:
                        results[key] = value
                        await self._update_metrics("l1_hits")

            # Find missing keys for L2 lookup
            missing_keys = [key for key in keys if key not in results]

            # Batch lookup in L2 cache (Redis)
            if self.l2_cache and missing_keys:
                # Use Redis pipeline for batch operations
                l2_results = await self._batch_get_l2(
                    [normalized_keys[key] for key in missing_keys]
                )

                for i, key in enumerate(missing_keys):
                    if l2_results[i] is not None:
                        # Deserialize and add to L1 cache
                        deserialized_value = await self._deserialize_data(
                            l2_results[i],
                            self._determine_serialization_format(l2_results[i]),
                        )
                        results[key] = deserialized_value
                        await self._update_metrics("l2_hits")

                        # Promote to L1
                        if self.l1_cache:
                            await self.l1_cache.set(
                                normalized_keys[key],
                                deserialized_value,
                                ttl_seconds=self.config.default_ttl_seconds,
                            )
                    else:
                        await self._update_metrics("cache_misses")

        except Exception as e:
            self.logger.error(f"Batch cache operation failed: {e}")
        finally:
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)

        return results

    async def set_batch(
        self, items: dict[str, Any], ttl_seconds: int | None = None
    ) -> dict[str, bool]:
        """Set multiple key-value pairs in a single operation for improved performance."""
        start_time = time.time()
        results = {}

        try:
            ttl = ttl_seconds or self.config.default_ttl_seconds

            # Process all items
            normalized_items = {}
            serialized_items = {}

            for key, value in items.items():
                normalized_key = self._normalize_key(key)
                normalized_items[key] = normalized_key

                # Determine serialization format
                serialization_format = self._determine_serialization_format(value)
                serialized_value = await self._serialize_data(
                    value, serialization_format
                )
                serialized_items[normalized_key] = serialized_value

            # Batch set to L1 cache
            if self.l1_cache:
                for key, value in items.items():
                    success = await self.l1_cache.set(
                        normalized_items[key], value, ttl_seconds=ttl
                    )
                    results[key] = success

            # Batch set to L2 cache (Redis)
            if self.l2_cache:
                l2_success = await self._batch_set_l2(serialized_items, ttl)
                # If L1 failed but L2 succeeded, update results
                for key in items.keys():
                    if not results.get(key, False) and l2_success:
                        results[key] = True

        except Exception as e:
            self.logger.error(f"Batch cache set operation failed: {e}")
            # Set all results to False on error
            results = dict.fromkeys(items.keys(), False)
        finally:
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)

        return results

    async def _batch_get_l2(self, keys: list[str]) -> list[bytes | None]:
        """Batch get operation for L2 Redis cache."""
        if not self.l2_cache:
            return [None] * len(keys)

        try:
            # Use Redis pipeline for batch operations
            redis_client = await self.l2_cache._get_client()
            async with redis_client.pipeline() as pipe:
                for key in keys:
                    pipe.get(key)
                results = await pipe.execute()
                return results
        except Exception as e:
            self.logger.error(f"L2 batch get failed: {e}")
            return [None] * len(keys)

    async def _batch_set_l2(self, items: dict[str, bytes], ttl_seconds: int) -> bool:
        """Batch set operation for L2 Redis cache."""
        if not self.l2_cache:
            return False

        try:
            redis_client = await self.l2_cache._get_client()
            async with redis_client.pipeline() as pipe:
                for key, value in items.items():
                    pipe.setex(key, ttl_seconds, value)
                await pipe.execute()
                return True
        except Exception as e:
            self.logger.error(f"L2 batch set failed: {e}")
            return False

    async def close(self) -> None:
        """Close cache connections."""
        if self.l2_cache:
            await self.l2_cache.close()


# Specialized cache services for different data types


class ModelCache(AdvancedCacheService):
    """Specialized cache for ML models."""

    def __init__(self, config: CacheConfig | None = None):
        if config is None:
            config = CacheConfig()

        # Optimize for models
        config.default_serialization = SerializationFormat.PICKLE
        config.l1_ttl_seconds = 86400  # Models live longer
        config.l2_ttl_seconds = 604800  # 1 week
        config.compression_algorithm = CompressionAlgorithm.ZSTD

        super().__init__(config)

    async def cache_model(
        self, model_id: str, model: Any, version: str = "latest"
    ) -> bool:
        """Cache a trained model."""
        cache_key = f"model:{model_id}:{version}"
        return await self.set(
            cache_key,
            model,
            ttl_seconds=self.config.l2_ttl_seconds,
            serialization_format=SerializationFormat.PICKLE,
        )

    async def get_model(self, model_id: str, version: str = "latest") -> Any | None:
        """Get cached model."""
        cache_key = f"model:{model_id}:{version}"
        return await self.get(cache_key, SerializationFormat.PICKLE)


class FeatureCache(AdvancedCacheService):
    """Specialized cache for feature data."""

    def __init__(self, config: CacheConfig | None = None):
        if config is None:
            config = CacheConfig()

        # Optimize for features
        config.default_serialization = SerializationFormat.NUMPY
        config.l1_ttl_seconds = 300  # 5 minutes
        config.l2_ttl_seconds = 3600  # 1 hour
        config.compression_algorithm = CompressionAlgorithm.LZ4

        super().__init__(config)

    async def cache_features(
        self, dataset_id: str, features: np.ndarray, feature_hash: str
    ) -> bool:
        """Cache feature data."""
        cache_key = f"features:{dataset_id}:{feature_hash}"
        return await self.set(
            cache_key,
            features,
            ttl_seconds=self.config.l2_ttl_seconds,
            serialization_format=SerializationFormat.NUMPY,
        )

    async def get_features(
        self, dataset_id: str, feature_hash: str
    ) -> np.ndarray | None:
        """Get cached features."""
        cache_key = f"features:{dataset_id}:{feature_hash}"
        return await self.get(cache_key, SerializationFormat.NUMPY)


class PredictionCache(AdvancedCacheService):
    """Specialized cache for prediction results."""

    def __init__(self, config: CacheConfig | None = None):
        if config is None:
            config = CacheConfig()

        # Optimize for predictions
        config.default_serialization = SerializationFormat.JSON
        config.l1_ttl_seconds = 60  # 1 minute
        config.l2_ttl_seconds = 300  # 5 minutes
        config.compression_algorithm = CompressionAlgorithm.LZ4

        super().__init__(config)

    async def cache_prediction(
        self, input_hash: str, model_id: str, prediction: dict[str, Any]
    ) -> bool:
        """Cache prediction result."""
        cache_key = f"prediction:{model_id}:{input_hash}"
        return await self.set(
            cache_key,
            prediction,
            ttl_seconds=self.config.l1_ttl_seconds,
            serialization_format=SerializationFormat.JSON,
        )

    async def get_prediction(
        self, input_hash: str, model_id: str
    ) -> dict[str, Any] | None:
        """Get cached prediction."""
        cache_key = f"prediction:{model_id}:{input_hash}"
        return await self.get(cache_key, SerializationFormat.JSON)
