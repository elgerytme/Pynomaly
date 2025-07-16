"""Core cache implementation classes."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheInterface(ABC):
    """Abstract base class for cache implementations."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class InMemoryCache(CacheInterface):
    """In-memory cache implementation with LRU eviction."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int | None = None,
        eviction_policy: str = "lru",
    ):
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            eviction_policy: Eviction policy ('lru', 'lfu', 'fifo')
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._expiry: dict[str, float] = {}
        self._access_count: dict[str, int] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            # Check if key exists and is not expired
            if key in self._cache:
                if self._is_expired(key):
                    self._remove_key(key)
                    self._stats["misses"] += 1
                    return default

                # Update access patterns
                self._access_count[key] = self._access_count.get(key, 0) + 1

                # Move to end for LRU
                if self.eviction_policy == "lru":
                    self._cache.move_to_end(key)

                self._stats["hits"] += 1
                return self._cache[key]

            self._stats["misses"] += 1
            return default

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove key if it exists
            if key in self._cache:
                self._remove_key(key)

            # Check if we need to evict
            if len(self._cache) >= self.max_size:
                self._evict()

            # Add new entry
            self._cache[key] = value
            self._access_count[key] = 1

            # Set expiry
            ttl = ttl or self.default_ttl
            if ttl:
                self._expiry[key] = time.time() + ttl

            self._stats["sets"] += 1

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                self._stats["deletes"] += 1
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key in self._cache:
                if self._is_expired(key):
                    self._remove_key(key)
                    return False
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            self._access_count.clear()

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                **self._stats,
            }

    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        if key in self._expiry:
            return time.time() > self._expiry[key]
        return False

    def _remove_key(self, key: str) -> None:
        """Remove key from cache and related data structures."""
        self._cache.pop(key, None)
        self._expiry.pop(key, None)
        self._access_count.pop(key, None)

    def _evict(self) -> None:
        """Evict entries based on policy."""
        if not self._cache:
            return

        if self.eviction_policy == "lru":
            # Remove least recently used (first item)
            key = next(iter(self._cache))
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self._access_count, key=self._access_count.get)
        else:  # fifo
            # Remove first in (first item)
            key = next(iter(self._cache))

        self._remove_key(key)
        self._stats["evictions"] += 1


class CacheKeyBuilder:
    """Builder for cache keys."""

    def __init__(
        self,
        separator: str = ":",
        max_key_length: int = 250,
        normalize_keys: bool = True,
    ):
        """Initialize cache key builder.

        Args:
            separator: Key component separator
            max_key_length: Maximum key length
            normalize_keys: Whether to normalize keys
        """
        self.separator = separator
        self.max_key_length = max_key_length
        self.normalize_keys = normalize_keys

    def build_key(
        self,
        key: str,
        namespace: str | None = None,
        version: str | None = None,
    ) -> str:
        """Build cache key.

        Args:
            key: Base key
            namespace: Optional namespace
            version: Optional version

        Returns:
            Built cache key
        """
        components = []

        if namespace:
            components.append(namespace)
        if version:
            components.append(version)

        components.append(key)

        built_key = self.separator.join(components)

        if self.normalize_keys:
            built_key = self._normalize_key(built_key)

        if len(built_key) > self.max_key_length:
            built_key = self._hash_key(built_key)

        return built_key

    def build_key_from_components(self, components: list[str]) -> str:
        """Build key from components.

        Args:
            components: List of key components

        Returns:
            Built cache key
        """
        return self.separator.join(components)

    def build_key_with_params(self, base_key: str, params: dict[str, Any]) -> str:
        """Build key with parameters.

        Args:
            base_key: Base key
            params: Parameters to include

        Returns:
            Built cache key
        """
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{base_key}?{param_str}"

    def build_key_from_object(
        self,
        obj: Any,
        key_func: Callable[[Any], str] | None = None,
    ) -> str:
        """Build key from object.

        Args:
            obj: Object to build key from
            key_func: Optional function to extract key

        Returns:
            Built cache key
        """
        if key_func:
            return key_func(obj)

        if hasattr(obj, "id"):
            return f"{obj.__class__.__name__.lower()}:{obj.id}"

        return f"{obj.__class__.__name__.lower()}:{id(obj)}"

    def _normalize_key(self, key: str) -> str:
        """Normalize cache key."""
        # Remove special characters and convert to lowercase
        import re

        key = re.sub(r"[^\w\-_.]", "_", key.lower())
        return key

    def _hash_key(self, key: str) -> str:
        """Hash long key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"hash:{key_hash}"


class CacheSerializer:
    """Serializer for cache values."""

    def __init__(
        self,
        format: str = "pickle",
        compression: str | None = None,
    ):
        """Initialize cache serializer.

        Args:
            format: Serialization format ('pickle', 'json', 'msgpack')
            compression: Optional compression ('gzip', 'zlib')
        """
        self.format = format
        self.compression = compression

    def serialize(self, value: Any) -> bytes:
        """Serialize value.

        Args:
            value: Value to serialize

        Returns:
            Serialized bytes
        """
        try:
            if self.format == "json":
                data = json.dumps(value).encode()
            elif self.format == "msgpack":
                try:
                    import msgpack

                    data = msgpack.packb(value)
                except ImportError:
                    raise ImportError("msgpack not available")
            else:  # pickle
                data = pickle.dumps(value)

            if self.compression == "gzip":
                import gzip

                data = gzip.compress(data)
            elif self.compression == "zlib":
                import zlib

                data = zlib.compress(data)

            return data
        except Exception as e:
            from pynomaly.domain.exceptions import SerializationError

            raise SerializationError(f"Serialization failed: {e}")

    def deserialize(self, data: bytes) -> Any:
        """Deserialize value.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized value
        """
        try:
            if self.compression == "gzip":
                import gzip

                data = gzip.decompress(data)
            elif self.compression == "zlib":
                import zlib

                data = zlib.decompress(data)

            if self.format == "json":
                return json.loads(data.decode())
            elif self.format == "msgpack":
                try:
                    import msgpack

                    return msgpack.unpackb(data, raw=False)
                except ImportError:
                    raise ImportError("msgpack not available")
            else:  # pickle
                return pickle.loads(data)
        except Exception as e:
            from pynomaly.domain.exceptions import SerializationError

            raise SerializationError(f"Deserialization failed: {e}")


class CacheDecorator:
    """Decorator for caching function results."""

    def __init__(self, cache: CacheInterface):
        """Initialize cache decorator.

        Args:
            cache: Cache instance to use
        """
        self.cache = cache
        self.key_builder = CacheKeyBuilder()
        self._function_cache: WeakKeyDictionary[Callable, dict] = WeakKeyDictionary()

    def cache_result(
        self,
        ttl: int | None = None,
        key_prefix: str | None = None,
        condition: Callable[[Any], bool] | None = None,
        include_self: bool = False,
    ) -> Callable:
        """Decorator to cache function results.

        Args:
            ttl: Time to live in seconds
            key_prefix: Optional key prefix
            condition: Optional condition function
            include_self: Whether to include self in key for methods

        Returns:
            Decorated function
        """

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(
                    func, args, kwargs, key_prefix, include_self
                )

                # Try to get from cache
                result = self.cache.get(cache_key)
                if result is not None:
                    return result

                # Execute function
                result = func(*args, **kwargs)

                # Check condition if provided
                if condition and not condition(result):
                    return result

                # Store in cache
                self.cache.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator

    def cache_result_async(
        self,
        ttl: int | None = None,
        key_prefix: str | None = None,
        condition: Callable[[Any], bool] | None = None,
        include_self: bool = False,
    ) -> Callable:
        """Decorator to cache async function results.

        Args:
            ttl: Time to live in seconds
            key_prefix: Optional key prefix
            condition: Optional condition function
            include_self: Whether to include self in key for methods

        Returns:
            Decorated async function
        """

        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(
                    func, args, kwargs, key_prefix, include_self
                )

                # Try to get from cache
                result = self.cache.get(cache_key)
                if result is not None:
                    return result

                # Execute function
                result = await func(*args, **kwargs)

                # Check condition if provided
                if condition and not condition(result):
                    return result

                # Store in cache
                self.cache.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator

    def invalidate_function_cache(
        self,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        key_prefix: str | None = None,
        include_self: bool = False,
    ) -> None:
        """Invalidate cache for specific function call.

        Args:
            func: Function to invalidate
            args: Function arguments
            kwargs: Function keyword arguments
            key_prefix: Optional key prefix
            include_self: Whether to include self in key for methods
        """
        cache_key = self._generate_cache_key(
            func, args, kwargs, key_prefix, include_self
        )
        self.cache.delete(cache_key)

    def _generate_cache_key(
        self,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        key_prefix: str | None = None,
        include_self: bool = False,
    ) -> str:
        """Generate cache key for function call.

        Args:
            func: Function
            args: Function arguments
            kwargs: Function keyword arguments
            key_prefix: Optional key prefix
            include_self: Whether to include self in key for methods

        Returns:
            Cache key
        """
        # Build base key
        base_key = key_prefix or func.__name__

        # Process arguments
        key_args = args
        if not include_self and args and hasattr(args[0], func.__name__):
            # Skip self for methods
            key_args = args[1:]

        # Create key components
        components = [base_key]

        # Add args
        for arg in key_args:
            components.append(str(hash(str(arg))))

        # Add kwargs
        for k, v in sorted(kwargs.items()):
            components.append(f"{k}={hash(str(v))}")

        return self.key_builder.build_key_from_components(components)


class CacheEvictionPolicy:
    """Cache eviction policy implementation."""

    def __init__(self, policy: str = "lru", max_size: int = 1000):
        """Initialize eviction policy.

        Args:
            policy: Eviction policy ('lru', 'lfu', 'ttl', 'size')
            max_size: Maximum cache size
        """
        self.policy = policy
        self.max_size = max_size
        self._access_order: OrderedDict[str, None] = OrderedDict()
        self._access_count: dict[str, int] = {}
        self._item_sizes: dict[str, int] = {}
        self._ttl_info: dict[str, float] = {}

    def access_item(self, key: str, value: Any | None = None) -> None:
        """Record item access.

        Args:
            key: Item key
            value: Item value (for updates)
        """
        if self.policy == "lru":
            # Move to end for LRU
            self._access_order.pop(key, None)
            self._access_order[key] = None

        elif self.policy == "lfu":
            # Increment access count
            self._access_count[key] = self._access_count.get(key, 0) + 1

    def add_item(self, key: str, value: Any) -> str | None:
        """Add item and return key to evict if needed.

        Args:
            key: Item key
            value: Item value

        Returns:
            Key to evict or None
        """
        if len(self._access_order) >= self.max_size:
            return self._select_eviction_candidate()

        self.access_item(key, value)
        return None

    def should_evict(self, key: str) -> bool:
        """Check if key should be evicted.

        Args:
            key: Item key

        Returns:
            True if should evict
        """
        if self.policy == "ttl":
            return self._is_expired(key)
        return False

    def set_item_ttl(self, key: str, ttl: float) -> None:
        """Set TTL for item.

        Args:
            key: Item key
            ttl: TTL in seconds
        """
        self._ttl_info[key] = time.time() + ttl

    def track_item_size(self, key: str, size: int) -> None:
        """Track item size.

        Args:
            key: Item key
            size: Item size in bytes
        """
        self._item_sizes[key] = size

    def check_memory_limit(self, new_item_size: int) -> bool:
        """Check if adding new item would exceed memory limit.

        Args:
            new_item_size: Size of new item

        Returns:
            True if would exceed limit
        """
        if self.policy == "size":
            current_size = sum(self._item_sizes.values())
            return current_size + new_item_size > self.max_size
        return False

    def get_items_to_evict_for_size(self, needed_size: int) -> list[str]:
        """Get items to evict to free up space.

        Args:
            needed_size: Required size to free

        Returns:
            List of keys to evict
        """
        if self.policy != "size":
            return []

        items_to_evict = []
        freed_size = 0

        # Sort by access order (LRU)
        sorted_keys = list(self._access_order.keys())

        for key in sorted_keys:
            if freed_size >= needed_size:
                break

            items_to_evict.append(key)
            freed_size += self._item_sizes.get(key, 0)

        return items_to_evict

    def get_expired_items(self) -> list[str]:
        """Get expired items.

        Returns:
            List of expired keys
        """
        if self.policy != "ttl":
            return []

        current_time = time.time()
        expired_keys = []

        for key, expire_time in self._ttl_info.items():
            if current_time > expire_time:
                expired_keys.append(key)

        return expired_keys

    def _select_eviction_candidate(self) -> str:
        """Select candidate for eviction.

        Returns:
            Key to evict
        """
        if self.policy == "lru":
            # Return first item (least recently used)
            return next(iter(self._access_order))

        elif self.policy == "lfu":
            # Return least frequently used
            return min(self._access_count, key=self._access_count.get)

        else:
            # Default to first item
            return next(iter(self._access_order))

    def _is_expired(self, key: str) -> bool:
        """Check if key is expired.

        Args:
            key: Item key

        Returns:
            True if expired
        """
        if key in self._ttl_info:
            return time.time() > self._ttl_info[key]
        return False
