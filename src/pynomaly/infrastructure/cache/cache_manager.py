"""Advanced cache management system with multiple backend support."""

from __future__ import annotations

import pickle
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:

    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    tags: set[str] = field(default_factory=set)

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() >= self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    def touch(self) -> None:
        """Update access information."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate miss rate."""
        return 1.0 - self.hit_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "current_size": self.current_size,
            "max_size": self.max_size,
            "entry_count": self.entry_count,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
        }


class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    def get(self, key: str) -> Any:
        """Get value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value by key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all entries."""
        pass

    @abstractmethod
    def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching pattern."""
        pass

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        pass


class InMemoryCache(CacheBackend):
    """High-performance in-memory cache with LRU eviction."""

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: int | None = None,
        cleanup_interval: int = 60,
    ):
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default TTL in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_size)

        # Background cleanup
        self._cleanup_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""

        def cleanup_worker():
            while not self._shutdown_event.wait(self.cleanup_interval):
                try:
                    self._cleanup_expired()
                except Exception as e:
                    print(f"Error in cache cleanup: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]

            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1

    def _evict_lru(self):
        """Evict least recently used entries."""
        with self._lock:
            while (
                len(self._cache) >= self.max_size
                or self._current_memory_usage() >= self.max_memory_bytes
            ):
                if not self._cache:
                    break

                # Remove least recently used (first item in OrderedDict)
                key, entry = self._cache.popitem(last=False)
                self._stats.evictions += 1

                if len(self._cache) == 0:
                    break

    def _current_memory_usage(self) -> int:
        """Calculate current memory usage."""
        return sum(entry.size_bytes for entry in self._cache.values())

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback to rough estimation
            return len(str(value).encode("utf-8"))

    def get(self, key: str) -> Any:
        """Get value by key."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            self._stats.hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value with optional TTL."""
        ttl = ttl or self.default_ttl
        expires_at = None
        if ttl:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)

        size_bytes = self._calculate_size(value)

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Evict if necessary
            self._evict_lru()

            # Create new entry
            entry = CacheEntry(
                key=key, value=value, expires_at=expires_at, size_bytes=size_bytes
            )

            self._cache[key] = entry
            self._stats.sets += 1

            # Update stats
            self._update_stats()

            return True

    def delete(self, key: str) -> bool:
        """Delete value by key."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.deletes += 1
                self._update_stats()
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                self._stats.evictions += 1
                return False

            return True

    def clear(self) -> bool:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._update_stats()
            return True

    def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching pattern."""
        import fnmatch

        with self._lock:
            # Clean up expired entries first
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1

            # Return matching keys
            if pattern == "*":
                return list(self._cache.keys())
            else:
                return [
                    key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)
                ]

    def _update_stats(self):
        """Update cache statistics."""
        self._stats.current_size = self._current_memory_usage()
        self._stats.entry_count = len(self._cache)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            self._update_stats()
            return self._stats.to_dict()

    def get_entries_by_tag(self, tag: str) -> list[CacheEntry]:
        """Get entries by tag."""
        with self._lock:
            return [entry for entry in self._cache.values() if tag in entry.tags]

    def delete_by_tag(self, tag: str) -> int:
        """Delete entries by tag."""
        with self._lock:
            keys_to_delete = [
                key for key, entry in self._cache.items() if tag in entry.tags
            ]

            for key in keys_to_delete:
                del self._cache[key]
                self._stats.deletes += 1

            self._update_stats()
            return len(keys_to_delete)

    def shutdown(self):
        """Shutdown cache and cleanup threads."""
        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)


class RedisCache(CacheBackend):
    """Redis-based cache backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        default_ttl: int | None = None,
        key_prefix: str = "pynomaly:",
        connection_pool_size: int = 10,
    ):
        """Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            default_ttl: Default TTL in seconds
            key_prefix: Key prefix for namespacing
            connection_pool_size: Connection pool size
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisCache")

        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

        # Create connection pool
        self._pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=connection_pool_size,
            decode_responses=False,  # We handle encoding ourselves
        )

        self._client = redis.Redis(connection_pool=self._pool)
        self._stats = CacheStats()

        # Test connection
        try:
            self._client.ping()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if data is None:
            return None
        return pickle.loads(data)

    def get(self, key: str) -> Any:
        """Get value by key."""
        try:
            redis_key = self._make_key(key)
            data = self._client.get(redis_key)

            if data is None:
                self._stats.misses += 1
                return None

            self._stats.hits += 1
            return self._deserialize(data)

        except Exception as e:
            print(f"Redis get error for key {key}: {e}")
            self._stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value with optional TTL."""
        try:
            redis_key = self._make_key(key)
            data = self._serialize(value)
            ttl = ttl or self.default_ttl

            if ttl:
                result = self._client.setex(redis_key, ttl, data)
            else:
                result = self._client.set(redis_key, data)

            if result:
                self._stats.sets += 1
                return True
            return False

        except Exception as e:
            print(f"Redis set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value by key."""
        try:
            redis_key = self._make_key(key)
            result = self._client.delete(redis_key)

            if result > 0:
                self._stats.deletes += 1
                return True
            return False

        except Exception as e:
            print(f"Redis delete error for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            redis_key = self._make_key(key)
            return bool(self._client.exists(redis_key))

        except Exception as e:
            print(f"Redis exists error for key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all entries with our prefix."""
        try:
            # Get all keys with our prefix
            pattern = f"{self.key_prefix}*"
            keys = self._client.keys(pattern)

            if keys:
                deleted = self._client.delete(*keys)
                self._stats.deletes += deleted

            return True

        except Exception as e:
            print(f"Redis clear error: {e}")
            return False

    def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching pattern."""
        try:
            redis_pattern = f"{self.key_prefix}{pattern}"
            redis_keys = self._client.keys(redis_pattern)

            # Remove prefix from keys
            prefix_len = len(self.key_prefix)
            return [key.decode("utf-8")[prefix_len:] for key in redis_keys]

        except Exception as e:
            print(f"Redis keys error: {e}")
            return []

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            redis_info = self._client.info()
            redis_stats = {
                "redis_version": redis_info.get("redis_version", "unknown"),
                "used_memory": redis_info.get("used_memory", 0),
                "connected_clients": redis_info.get("connected_clients", 0),
                "keyspace_hits": redis_info.get("keyspace_hits", 0),
                "keyspace_misses": redis_info.get("keyspace_misses", 0),
            }

            stats_dict = self._stats.to_dict()
            stats_dict.update(redis_stats)
            return stats_dict

        except Exception as e:
            print(f"Redis stats error: {e}")
            return self._stats.to_dict()


class CacheManager:
    """Advanced cache manager with multiple backends and strategies."""

    def __init__(
        self,
        primary_backend: CacheBackend,
        fallback_backend: CacheBackend | None = None,
        enable_write_through: bool = False,
        enable_write_behind: bool = False,
        compression_threshold: int = 1024,
        enable_metrics: bool = True,
    ):
        """Initialize cache manager.

        Args:
            primary_backend: Primary cache backend
            fallback_backend: Optional fallback backend
            enable_write_through: Enable write-through caching
            enable_write_behind: Enable write-behind caching
            compression_threshold: Compress values larger than this
            enable_metrics: Enable metrics collection
        """
        self.primary_backend = primary_backend
        self.fallback_backend = fallback_backend
        self.enable_write_through = enable_write_through
        self.enable_write_behind = enable_write_behind
        self.compression_threshold = compression_threshold
        self.enable_metrics = enable_metrics

        self._lock = threading.RLock()
        self._write_behind_queue: list[tuple[str, Any, int | None]] = []
        self._write_behind_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        if enable_write_behind:
            self._start_write_behind_thread()

    def _start_write_behind_thread(self):
        """Start write-behind background thread."""

        def write_behind_worker():
            while not self._shutdown_event.wait(1):  # Check every second
                try:
                    self._process_write_behind_queue()
                except Exception as e:
                    print(f"Error in write-behind processing: {e}")

        self._write_behind_thread = threading.Thread(
            target=write_behind_worker, daemon=True
        )
        self._write_behind_thread.start()

    def _process_write_behind_queue(self):
        """Process write-behind queue."""
        with self._lock:
            if not self._write_behind_queue:
                return

            # Process batch of writes
            batch = self._write_behind_queue[:100]  # Process up to 100 at a time
            self._write_behind_queue = self._write_behind_queue[100:]

        for key, value, ttl in batch:
            try:
                if self.fallback_backend:
                    self.fallback_backend.set(key, value, ttl)
            except Exception as e:
                print(f"Write-behind error for key {key}: {e}")

    def _compress_if_needed(self, value: Any) -> Any:
        """Compress value if it exceeds threshold."""
        try:
            import zlib

            serialized = pickle.dumps(value)

            if len(serialized) > self.compression_threshold:
                compressed = zlib.compress(serialized)
                return {"_compressed": True, "data": compressed}
            else:
                return value
        except Exception:
            return value

    def _decompress_if_needed(self, value: Any) -> Any:
        """Decompress value if it was compressed."""
        try:
            if isinstance(value, dict) and value.get("_compressed"):
                import zlib

                decompressed = zlib.decompress(value["data"])
                return pickle.loads(decompressed)
            else:
                return value
        except Exception:
            return value

    def get(self, key: str) -> Any:
        """Get value by key with fallback."""
        # Try primary backend first
        value = self.primary_backend.get(key)

        if value is not None:
            return self._decompress_if_needed(value)

        # Try fallback backend
        if self.fallback_backend:
            value = self.fallback_backend.get(key)
            if value is not None:
                # Populate primary cache
                self.primary_backend.set(key, value)
                return self._decompress_if_needed(value)

        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value with caching strategies."""
        compressed_value = self._compress_if_needed(value)

        # Always set in primary backend
        primary_result = self.primary_backend.set(key, compressed_value, ttl)

        # Handle secondary backend based on strategy
        if self.fallback_backend:
            if self.enable_write_through:
                # Synchronous write to fallback
                self.fallback_backend.set(key, compressed_value, ttl)
            elif self.enable_write_behind:
                # Asynchronous write to fallback
                with self._lock:
                    self._write_behind_queue.append((key, compressed_value, ttl))

        return primary_result

    def delete(self, key: str) -> bool:
        """Delete value from all backends."""
        primary_result = self.primary_backend.delete(key)

        if self.fallback_backend:
            fallback_result = self.fallback_backend.delete(key)
            return primary_result or fallback_result

        return primary_result

    def exists(self, key: str) -> bool:
        """Check if key exists in any backend."""
        if self.primary_backend.exists(key):
            return True

        if self.fallback_backend and self.fallback_backend.exists(key):
            return True

        return False

    def clear(self) -> bool:
        """Clear all backends."""
        primary_result = self.primary_backend.clear()

        if self.fallback_backend:
            fallback_result = self.fallback_backend.clear()
            return primary_result and fallback_result

        return primary_result

    def keys(self, pattern: str = "*") -> list[str]:
        """Get keys from primary backend."""
        return self.primary_backend.keys(pattern)

    def stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "primary_backend": self.primary_backend.stats(),
            "write_behind_queue_size": (
                len(self._write_behind_queue) if self.enable_write_behind else 0
            ),
            "compression_enabled": self.compression_threshold > 0,
            "write_through_enabled": self.enable_write_through,
            "write_behind_enabled": self.enable_write_behind,
        }

        if self.fallback_backend:
            stats["fallback_backend"] = self.fallback_backend.stats()

        return stats

    def shutdown(self):
        """Shutdown cache manager."""
        self._shutdown_event.set()

        # Process remaining write-behind queue
        if self.enable_write_behind:
            self._process_write_behind_queue()

        # Wait for write-behind thread
        if self._write_behind_thread and self._write_behind_thread.is_alive():
            self._write_behind_thread.join(timeout=5)

        # Shutdown backends
        if hasattr(self.primary_backend, "shutdown"):
            self.primary_backend.shutdown()

        if self.fallback_backend and hasattr(self.fallback_backend, "shutdown"):
            self.fallback_backend.shutdown()


# Factory functions
def create_cache_manager(
    backend_type: str = "memory", redis_url: str | None = None, **kwargs
) -> CacheManager:
    """Factory function to create cache manager.

    Args:
        backend_type: Type of primary backend ('memory', 'redis')
        redis_url: Redis URL for redis backend
        **kwargs: Additional backend-specific arguments

    Returns:
        Configured cache manager
    """
    if backend_type == "memory":
        primary_backend = InMemoryCache(**kwargs)
        return CacheManager(primary_backend)

    elif backend_type == "redis":
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for Redis backend")

        if redis_url:
            # Parse Redis URL
            import redis

            primary_backend = RedisCache(
                **redis.from_url(redis_url).connection_pool.connection_kwargs
            )
        else:
            primary_backend = RedisCache(**kwargs)

        # Use in-memory as fallback
        fallback_backend = InMemoryCache(max_size=1000)

        return CacheManager(
            primary_backend=primary_backend, fallback_backend=fallback_backend, **kwargs
        )

    elif backend_type == "hybrid":
        # Redis primary with in-memory fallback
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for hybrid backend")

        redis_backend = RedisCache(**kwargs)
        memory_backend = InMemoryCache(max_size=1000)

        return CacheManager(
            primary_backend=redis_backend,
            fallback_backend=memory_backend,
            enable_write_through=True,
        )

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


# Global cache manager instance
_global_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager | None:
    """Get global cache manager."""
    return _global_cache_manager


def configure_cache(backend_type: str = "memory", **kwargs) -> CacheManager:
    """Configure global cache manager."""
    global _global_cache_manager
    _global_cache_manager = create_cache_manager(backend_type, **kwargs)
    return _global_cache_manager


def shutdown_cache():
    """Shutdown global cache manager."""
    global _global_cache_manager
    if _global_cache_manager:
        _global_cache_manager.shutdown()
        _global_cache_manager = None
