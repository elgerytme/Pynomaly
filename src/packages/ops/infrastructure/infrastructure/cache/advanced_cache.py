"""Advanced cache management implementations."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from .cache_core import CacheInterface, InMemoryCache
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)


class CacheManager:
    """Comprehensive cache manager with tiered caching and advanced features."""

    def __init__(
        self,
        primary_cache_type: str = "memory",
        fallback_cache_type: str | None = None,
        redis_config: dict[str, Any] | None = None,
        memory_config: dict[str, Any] | None = None,
        enable_monitoring: bool = False,
        enable_distributed_locking: bool = False,
    ):
        """Initialize cache manager.

        Args:
            primary_cache_type: Primary cache type ('redis', 'memory')
            fallback_cache_type: Fallback cache type
            redis_config: Redis configuration
            memory_config: Memory cache configuration
            enable_monitoring: Enable performance monitoring
            enable_distributed_locking: Enable distributed locking
        """
        self.primary_cache_type = primary_cache_type
        self.fallback_cache_type = fallback_cache_type
        self.enable_monitoring = enable_monitoring
        self.enable_distributed_locking = enable_distributed_locking

        # Initialize caches
        self.primary_cache = self._create_cache(
            primary_cache_type, redis_config, memory_config
        )
        self.fallback_cache = None
        if fallback_cache_type:
            self.fallback_cache = self._create_cache(
                fallback_cache_type, redis_config, memory_config
            )

        # Performance monitoring
        self._performance_metrics = {
            "total_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "failover_count": 0,
            "response_times": [],
        }
        self._lock = threading.Lock()

        # Tag management
        self._tag_mappings: dict[str, set[str]] = {}

    def _create_cache(
        self,
        cache_type: str,
        redis_config: dict[str, Any] | None,
        memory_config: dict[str, Any] | None,
    ) -> CacheInterface:
        """Create cache instance.

        Args:
            cache_type: Cache type
            redis_config: Redis configuration
            memory_config: Memory cache configuration

        Returns:
            Cache instance
        """
        if cache_type == "redis":
            config = redis_config or {}
            return RedisCache(
                host=config.get("host", "localhost"),
                port=config.get("port", 6379),
                db=config.get("db", 0),
                password=config.get("password"),
                enable_failover=True,
            )
        elif cache_type == "memory":
            config = memory_config or {}
            return InMemoryCache(
                max_size=config.get("max_size", 1000),
                default_ttl=config.get("default_ttl"),
                eviction_policy=config.get("eviction_policy", "lru"),
            )
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with fallback.

        Args:
            key: Cache key
            default: Default value

        Returns:
            Cached value or default
        """
        start_time = time.time()

        try:
            # Try primary cache
            result = self.primary_cache.get(key, default)
            if result != default:
                self._record_hit(time.time() - start_time)
                return result

            # Try fallback cache
            if self.fallback_cache:
                result = self.fallback_cache.get(key, default)
                if result != default:
                    # Populate primary cache
                    try:
                        self.primary_cache.set(key, result)
                    except Exception:
                        pass  # Fail silently
                    self._record_hit(time.time() - start_time)
                    return result

            self._record_miss(time.time() - start_time)
            return default

        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            if self.fallback_cache:
                try:
                    result = self.fallback_cache.get(key, default)
                    self._record_failover()
                    return result
                except Exception:
                    pass

            self._record_miss(time.time() - start_time)
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        tags: list[str] | None = None,
        serialization_format: str = "pickle",
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Tags for invalidation
            serialization_format: Serialization format
        """
        start_time = time.time()

        try:
            # Set in primary cache
            self.primary_cache.set(key, value, ttl)

            # Set in fallback cache
            if self.fallback_cache:
                try:
                    self.fallback_cache.set(key, value, ttl)
                except Exception:
                    pass  # Fail silently

            # Record tags
            if tags:
                self._record_tags(key, tags)

            self._record_operation(time.time() - start_time)

        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            if self.fallback_cache:
                try:
                    self.fallback_cache.set(key, value, ttl)
                    self._record_failover()
                except Exception:
                    pass

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        start_time = time.time()

        try:
            # Delete from primary cache
            result = self.primary_cache.delete(key)

            # Delete from fallback cache
            if self.fallback_cache:
                try:
                    self.fallback_cache.delete(key)
                except Exception:
                    pass

            # Remove from tag mappings
            self._remove_from_tags(key)

            self._record_operation(time.time() - start_time)
            return result

        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            if self.fallback_cache:
                try:
                    result = self.fallback_cache.delete(key)
                    self._record_failover()
                    return result
                except Exception:
                    pass
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        try:
            return self.primary_cache.exists(key)
        except Exception:
            if self.fallback_cache:
                try:
                    return self.fallback_cache.exists(key)
                except Exception:
                    pass
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self.primary_cache.clear()
        except Exception:
            pass

        if self.fallback_cache:
            try:
                self.fallback_cache.clear()
            except Exception:
                pass

        self._tag_mappings.clear()

    def get_batch(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key-value pairs
        """
        result = {}

        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value

        return result

    def set_batch(self, data: dict[str, Any], ttl: int | None = None) -> None:
        """Set multiple values in cache.

        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
        """
        for key, value in data.items():
            self.set(key, value, ttl)

    def warm_cache(self, data: dict[str, Any], ttl: int | None = None) -> None:
        """Warm cache with data.

        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
        """
        self.set_batch(data, ttl)

    def invalidate_by_tag(self, tag: str) -> None:
        """Invalidate cache entries by tag.

        Args:
            tag: Tag to invalidate
        """
        if tag in self._tag_mappings:
            keys_to_delete = list(self._tag_mappings[tag])
            for key in keys_to_delete:
                self.delete(key)
            del self._tag_mappings[tag]

    def invalidate_by_pattern(self, pattern: str) -> None:
        """Invalidate cache entries by pattern.

        Args:
            pattern: Pattern to match (supports wildcards)
        """
        # This is a simplified implementation
        # In a real implementation, you'd use Redis pattern matching

        try:
            # For Redis cache, use pattern matching
            if hasattr(self.primary_cache, "get_keys_by_pattern"):
                keys = self.primary_cache.get_keys_by_pattern(pattern)
                for key in keys:
                    self.delete(key)
            else:
                # For in-memory cache, we'd need to iterate through all keys
                # This is not efficient and should be improved
                logger.warning(
                    "Pattern invalidation not efficiently supported for in-memory cache"
                )
        except Exception as e:
            logger.error(f"Pattern invalidation failed: {e}")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics.

        Returns:
            Performance metrics dictionary
        """
        with self._lock:
            total_requests = (
                self._performance_metrics["cache_hits"]
                + self._performance_metrics["cache_misses"]
            )
            hit_rate = (
                self._performance_metrics["cache_hits"] / total_requests
                if total_requests > 0
                else 0
            )

            avg_response_time = (
                sum(self._performance_metrics["response_times"])
                / len(self._performance_metrics["response_times"])
                if self._performance_metrics["response_times"]
                else 0
            )

            return {
                "hit_rate": hit_rate,
                "average_response_time": avg_response_time,
                **self._performance_metrics,
            }

    @contextmanager
    def distributed_lock(
        self, resource: str, timeout: int = 30
    ) -> Generator[None, None, None]:
        """Distributed lock context manager.

        Args:
            resource: Resource to lock
            timeout: Lock timeout in seconds

        Yields:
            None
        """
        if not self.enable_distributed_locking:
            yield
            return

        lock_key = f"lock:{resource}"
        acquired = False

        try:
            # Try to acquire lock
            if hasattr(self.primary_cache, "set"):
                # Use Redis-like locking
                lock_value = f"lock:{time.time()}"

                # Try to set lock (atomic operation)
                if not self.primary_cache.exists(lock_key):
                    self.primary_cache.set(lock_key, lock_value, timeout)
                    acquired = True
                else:
                    raise RuntimeError(
                        f"Could not acquire lock for resource: {resource}"
                    )

            yield

        finally:
            if acquired:
                self.primary_cache.delete(lock_key)

    def _record_hit(self, response_time: float) -> None:
        """Record cache hit."""
        if self.enable_monitoring:
            with self._lock:
                self._performance_metrics["cache_hits"] += 1
                self._performance_metrics["total_operations"] += 1
                self._performance_metrics["response_times"].append(response_time)

    def _record_miss(self, response_time: float) -> None:
        """Record cache miss."""
        if self.enable_monitoring:
            with self._lock:
                self._performance_metrics["cache_misses"] += 1
                self._performance_metrics["total_operations"] += 1
                self._performance_metrics["response_times"].append(response_time)

    def _record_operation(self, response_time: float) -> None:
        """Record cache operation."""
        if self.enable_monitoring:
            with self._lock:
                self._performance_metrics["total_operations"] += 1
                self._performance_metrics["response_times"].append(response_time)

    def _record_failover(self) -> None:
        """Record cache failover."""
        if self.enable_monitoring:
            with self._lock:
                self._performance_metrics["failover_count"] += 1

    def _record_tags(self, key: str, tags: list[str]) -> None:
        """Record key-tag mappings.

        Args:
            key: Cache key
            tags: List of tags
        """
        for tag in tags:
            if tag not in self._tag_mappings:
                self._tag_mappings[tag] = set()
            self._tag_mappings[tag].add(key)

    def _remove_from_tags(self, key: str) -> None:
        """Remove key from tag mappings.

        Args:
            key: Cache key
        """
        tags_to_remove = []
        for tag, keys in self._tag_mappings.items():
            if key in keys:
                keys.remove(key)
                if not keys:
                    tags_to_remove.append(tag)

        for tag in tags_to_remove:
            del self._tag_mappings[tag]


class DistributedCache:
    """Distributed cache implementation."""

    def __init__(
        self,
        nodes: list[str],
        consistency_level: str = "eventual",
        partitioning_strategy: str = "hash",
        replication_factor: int = 1,
        enable_failover: bool = True,
        enable_pipelining: bool = False,
        connection_pooling: bool = False,
        max_connections: int = 10,
    ):
        """Initialize distributed cache.

        Args:
            nodes: List of cache node addresses
            consistency_level: Consistency level ('strong', 'eventual')
            partitioning_strategy: Partitioning strategy ('hash', 'range')
            replication_factor: Number of replicas
            enable_failover: Enable automatic failover
            enable_pipelining: Enable command pipelining
            connection_pooling: Enable connection pooling
            max_connections: Maximum connections per node
        """
        self.nodes = nodes
        self.consistency_level = consistency_level
        self.partitioning_strategy = partitioning_strategy
        self.replication_factor = replication_factor
        self.enable_failover = enable_failover
        self.enable_pipelining = enable_pipelining

        # Initialize node clients
        self.node_clients = {}
        for node in nodes:
            try:
                # Create Redis client for each node
                host, port = node.split(":")
                client = RedisCache(
                    host=host,
                    port=int(port),
                    enable_failover=enable_failover,
                )
                self.node_clients[node] = client
            except Exception as e:
                logger.error(f"Failed to initialize node {node}: {e}")

        self.healthy_nodes = set(self.nodes)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from distributed cache.

        Args:
            key: Cache key
            default: Default value

        Returns:
            Cached value or default
        """
        partition = self.get_partition_for_key(key)
        node = self.nodes[partition]

        try:
            if node in self.node_clients:
                return self.node_clients[node].get(key, default)
        except Exception as e:
            logger.error(f"Failed to get from node {node}: {e}")
            if self.enable_failover:
                # Try other nodes
                for fallback_node in self.healthy_nodes:
                    if fallback_node != node:
                        try:
                            return self.node_clients[fallback_node].get(key, default)
                        except Exception:
                            continue

        return default

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in distributed cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        partition = self.get_partition_for_key(key)
        primary_node = self.nodes[partition]

        # Get replica nodes
        replica_nodes = self._get_replica_nodes(partition)

        if self.consistency_level == "strong":
            # Write to all replicas synchronously
            success_count = 0

            for node in [primary_node] + replica_nodes:
                try:
                    if node in self.node_clients:
                        self.node_clients[node].set(key, value, ttl)
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to set on node {node}: {e}")

            if success_count == 0:
                raise RuntimeError("Failed to write to any replica")

        else:  # eventual consistency
            # Write to primary first, then replicas asynchronously
            try:
                if primary_node in self.node_clients:
                    self.node_clients[primary_node].set(key, value, ttl)

                # Asynchronously replicate to other nodes
                for node in replica_nodes:
                    try:
                        if node in self.node_clients:
                            self.node_clients[node].set(key, value, ttl)
                    except Exception as e:
                        logger.error(f"Failed to replicate to node {node}: {e}")

            except Exception as e:
                logger.error(f"Failed to write to primary node {primary_node}: {e}")
                raise

    def delete(self, key: str) -> bool:
        """Delete value from distributed cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        partition = self.get_partition_for_key(key)
        primary_node = self.nodes[partition]

        try:
            if primary_node in self.node_clients:
                return self.node_clients[primary_node].delete(key)
        except Exception as e:
            logger.error(f"Failed to delete from node {primary_node}: {e}")
            return False

        return False

    def get_partition_for_key(self, key: str) -> int:
        """Get partition for key.

        Args:
            key: Cache key

        Returns:
            Partition index
        """
        if self.partitioning_strategy == "hash":
            return hash(key) % len(self.nodes)
        else:
            # Default to hash partitioning
            return hash(key) % len(self.nodes)

    def set_batch(self, data: dict[str, Any], ttl: int | None = None) -> None:
        """Set multiple values in distributed cache.

        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
        """
        # Group keys by partition
        partitioned_data = {}
        for key, value in data.items():
            partition = self.get_partition_for_key(key)
            if partition not in partitioned_data:
                partitioned_data[partition] = {}
            partitioned_data[partition][key] = value

        # Set data in each partition
        for partition, partition_data in partitioned_data.items():
            node = self.nodes[partition]
            try:
                if node in self.node_clients:
                    client = self.node_clients[node]
                    if hasattr(client, "set_batch"):
                        client.set_batch(partition_data, ttl)
                    else:
                        for key, value in partition_data.items():
                            client.set(key, value, ttl)
            except Exception as e:
                logger.error(f"Failed to batch set on node {node}: {e}")

    def _get_replica_nodes(self, primary_partition: int) -> list[str]:
        """Get replica nodes for a partition.

        Args:
            primary_partition: Primary partition index

        Returns:
            List of replica node addresses
        """
        replica_nodes = []

        for i in range(1, self.replication_factor):
            replica_partition = (primary_partition + i) % len(self.nodes)
            replica_node = self.nodes[replica_partition]
            replica_nodes.append(replica_node)

        return replica_nodes
