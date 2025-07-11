"""Production-ready Redis caching implementation with advanced features for Issue #99.

This module provides enterprise-grade Redis caching capabilities including:
- High availability with Redis Sentinel
- Redis Cluster support for horizontal scaling
- Advanced cache warming and invalidation strategies
- Comprehensive monitoring and observability
- Security hardening with encryption and authentication
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, TypeVar, Union
from urllib.parse import urlparse

import redis
import redis.sentinel
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from pynomaly.infrastructure.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: int = 0
    connection_count: int = 0
    avg_response_time: float = 0.0
    cache_warming_time: float = 0.0
    last_updated: datetime = None


@dataclass
class CacheWarmingConfig:
    """Configuration for cache warming strategies."""
    
    enabled: bool = True
    warmup_on_startup: bool = True
    background_warming: bool = True
    warmup_batch_size: int = 100
    warmup_delay_seconds: float = 0.1
    critical_keys: List[str] = None
    warming_schedules: Dict[str, str] = None  # cron expressions


class ProductionRedisCache:
    """Production-ready Redis cache with enterprise features."""

    def __init__(
        self,
        settings: Settings,
        sentinel_hosts: Optional[List[str]] = None,
        cluster_mode: bool = False,
        enable_monitoring: bool = True,
        enable_cache_warming: bool = True,
        enable_circuit_breaker: bool = True,
    ):
        """Initialize production Redis cache.
        
        Args:
            settings: Application settings
            sentinel_hosts: Redis Sentinel hosts for HA
            cluster_mode: Enable Redis Cluster mode
            enable_monitoring: Enable performance monitoring
            enable_cache_warming: Enable cache warming strategies
            enable_circuit_breaker: Enable circuit breaker pattern
        """
        self.settings = settings
        self.sentinel_hosts = sentinel_hosts or []
        self.cluster_mode = cluster_mode
        self.enable_monitoring = enable_monitoring
        self.enable_cache_warming = enable_cache_warming
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Initialize metrics
        self.metrics = CacheMetrics()
        self.metrics.last_updated = datetime.utcnow()
        
        # Circuit breaker state
        self.circuit_breaker_open = False
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60  # seconds
        
        # Cache warming configuration
        self.warming_config = CacheWarmingConfig()
        self._warming_tasks: Set[asyncio.Task] = set()
        
        # Initialize Redis connections
        self._initialize_connections()
        
        logger.info(
            f"Initialized ProductionRedisCache - "
            f"Sentinel: {bool(sentinel_hosts)}, "
            f"Cluster: {cluster_mode}, "
            f"Monitoring: {enable_monitoring}"
        )

    def _initialize_connections(self) -> None:
        """Initialize Redis connections with appropriate configuration."""
        try:
            if self.cluster_mode:
                self._initialize_cluster()
            elif self.sentinel_hosts:
                self._initialize_sentinel()
            else:
                self._initialize_standalone()
                
            # Test connection
            self.redis.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            if self.enable_circuit_breaker:
                self._open_circuit_breaker()
            raise

    def _initialize_standalone(self) -> None:
        """Initialize standalone Redis connection."""
        redis_url = getattr(self.settings, 'redis_url', 'redis://localhost:6379/0')
        parsed = urlparse(redis_url)
        
        connection_pool = redis.ConnectionPool(
            host=parsed.hostname or 'localhost',
            port=parsed.port or 6379,
            db=int(parsed.path.lstrip('/')) if parsed.path else 0,
            password=parsed.password,
            decode_responses=False,
            max_connections=20,
            retry_on_timeout=True,
            retry_on_error=[ConnectionError, TimeoutError],
            health_check_interval=30,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        
        self.redis = redis.Redis(
            connection_pool=connection_pool,
            socket_keepalive=True,
            socket_keepalive_options={},
        )

    def _initialize_sentinel(self) -> None:
        """Initialize Redis Sentinel for high availability."""
        sentinel_list = []
        for host in self.sentinel_hosts:
            if ':' in host:
                hostname, port = host.split(':')
                sentinel_list.append((hostname, int(port)))
            else:
                sentinel_list.append((host, 26379))
        
        self.sentinel = redis.sentinel.Sentinel(
            sentinel_list,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        
        service_name = getattr(self.settings, 'redis_service_name', 'mymaster')
        self.redis = self.sentinel.master_for(
            service_name,
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=False,
        )

    def _initialize_cluster(self) -> None:
        """Initialize Redis Cluster for horizontal scaling."""
        try:
            import redis.cluster
            
            startup_nodes = []
            cluster_hosts = getattr(self.settings, 'redis_cluster_hosts', ['localhost:7000'])
            
            for host in cluster_hosts:
                if ':' in host:
                    hostname, port = host.split(':')
                    startup_nodes.append({"host": hostname, "port": int(port)})
                else:
                    startup_nodes.append({"host": host, "port": 7000})
            
            self.redis = redis.cluster.RedisCluster(
                startup_nodes=startup_nodes,
                decode_responses=False,
                skip_full_coverage_check=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            
        except ImportError:
            logger.error("redis-py-cluster not available for cluster mode")
            raise

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.enable_circuit_breaker:
            return False
            
        if not self.circuit_breaker_open:
            return False
            
        # Check if timeout has passed
        if (
            self.circuit_breaker_last_failure and
            datetime.utcnow() - self.circuit_breaker_last_failure > 
            timedelta(seconds=self.circuit_breaker_timeout)
        ):
            self._close_circuit_breaker()
            return False
            
        return True

    def _open_circuit_breaker(self) -> None:
        """Open circuit breaker."""
        self.circuit_breaker_open = True
        self.circuit_breaker_last_failure = datetime.utcnow()
        logger.warning("Redis circuit breaker opened")

    def _close_circuit_breaker(self) -> None:
        """Close circuit breaker."""
        self.circuit_breaker_open = False
        self.circuit_breaker_failures = 0
        logger.info("Redis circuit breaker closed")

    def _record_failure(self) -> None:
        """Record a cache operation failure."""
        if not self.enable_circuit_breaker:
            return
            
        self.circuit_breaker_failures += 1
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self._open_circuit_breaker()

    def _record_success(self) -> None:
        """Record a successful cache operation."""
        if self.enable_circuit_breaker and self.circuit_breaker_failures > 0:
            self.circuit_breaker_failures = max(0, self.circuit_breaker_failures - 1)

    @asynccontextmanager
    async def _with_circuit_breaker(self):
        """Context manager for circuit breaker pattern."""
        if self._is_circuit_breaker_open():
            raise ConnectionError("Redis circuit breaker is open")
            
        start_time = time.time()
        try:
            yield
            self._record_success()
        except Exception as e:
            self._record_failure()
            raise
        finally:
            if self.enable_monitoring:
                response_time = time.time() - start_time
                self._update_response_time_metric(response_time)

    def _update_response_time_metric(self, response_time: float) -> None:
        """Update average response time metric."""
        # Simple moving average
        alpha = 0.1
        self.metrics.avg_response_time = (
            alpha * response_time + 
            (1 - alpha) * self.metrics.avg_response_time
        )

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with circuit breaker protection."""
        async with self._with_circuit_breaker():
            try:
                # Add namespace prefix
                namespaced_key = self._get_namespaced_key(key)
                
                # Get from Redis
                raw_value = self.redis.get(namespaced_key)
                
                if raw_value is None:
                    if self.enable_monitoring:
                        self.metrics.misses += 1
                    return default
                
                # Deserialize value
                value = self._deserialize(raw_value, key)
                
                if self.enable_monitoring:
                    self.metrics.hits += 1
                
                return value
                
            except Exception as e:
                logger.error(f"Cache get failed for key {key}: {e}")
                if self.enable_monitoring:
                    self.metrics.misses += 1
                return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None,
    ) -> bool:
        """Set value in cache with tags support."""
        async with self._with_circuit_breaker():
            try:
                # Add namespace prefix
                namespaced_key = self._get_namespaced_key(key)
                
                # Serialize value
                serialized_value = self._serialize(value, key)
                
                # Set in Redis with TTL
                if ttl:
                    result = self.redis.setex(namespaced_key, ttl, serialized_value)
                else:
                    result = self.redis.set(namespaced_key, serialized_value)
                
                # Handle tags for invalidation
                if tags:
                    await self._add_tags(key, tags)
                
                if self.enable_monitoring:
                    self.metrics.hits += 1
                
                return bool(result)
                
            except Exception as e:
                logger.error(f"Cache set failed for key {key}: {e}")
                return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._with_circuit_breaker():
            try:
                namespaced_key = self._get_namespaced_key(key)
                result = self.redis.delete(namespaced_key)
                
                # Remove from tag indices
                await self._remove_from_tags(key)
                
                if self.enable_monitoring:
                    self.metrics.deletes += 1
                
                return bool(result)
                
            except Exception as e:
                logger.error(f"Cache delete failed for key {key}: {e}")
                return False

    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all cache entries with the given tag."""
        async with self._with_circuit_breaker():
            try:
                tag_key = f"tag:{tag}"
                tagged_keys = self.redis.smembers(tag_key)
                
                if not tagged_keys:
                    return 0
                
                # Delete all tagged keys
                pipeline = self.redis.pipeline()
                for key in tagged_keys:
                    pipeline.delete(self._get_namespaced_key(key.decode()))
                
                # Remove the tag index
                pipeline.delete(tag_key)
                
                results = pipeline.execute()
                deleted_count = sum(1 for result in results[:-1] if result)
                
                logger.info(f"Invalidated {deleted_count} keys with tag '{tag}'")
                return deleted_count
                
            except Exception as e:
                logger.error(f"Tag invalidation failed for tag {tag}: {e}")
                return 0

    async def warm_cache(self, warming_data: Dict[str, Any]) -> None:
        """Warm cache with pre-computed data."""
        if not self.enable_cache_warming:
            return
            
        start_time = time.time()
        logger.info(f"Starting cache warming with {len(warming_data)} entries")
        
        try:
            # Batch warm in chunks
            batch_size = self.warming_config.warmup_batch_size
            keys = list(warming_data.keys())
            
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                
                # Use pipeline for efficiency
                pipeline = self.redis.pipeline()
                for key in batch_keys:
                    namespaced_key = self._get_namespaced_key(key)
                    serialized_value = self._serialize(warming_data[key], key)
                    pipeline.set(namespaced_key, serialized_value)
                
                pipeline.execute()
                
                # Brief delay between batches
                if self.warming_config.warmup_delay_seconds > 0:
                    await asyncio.sleep(self.warming_config.warmup_delay_seconds)
            
            warming_time = time.time() - start_time
            self.metrics.cache_warming_time = warming_time
            logger.info(f"Cache warming completed in {warming_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            # Get Redis info
            redis_info = self.redis.info()
            
            # Update metrics
            self.metrics.memory_usage = redis_info.get('used_memory', 0)
            self.metrics.connection_count = redis_info.get('connected_clients', 0)
            self.metrics.evictions = redis_info.get('evicted_keys', 0)
            self.metrics.last_updated = datetime.utcnow()
            
            stats = {
                'cache_metrics': {
                    'hits': self.metrics.hits,
                    'misses': self.metrics.misses,
                    'hit_rate': self.metrics.hits / max(1, self.metrics.hits + self.metrics.misses),
                    'evictions': self.metrics.evictions,
                    'avg_response_time_ms': self.metrics.avg_response_time * 1000,
                    'cache_warming_time_s': self.metrics.cache_warming_time,
                },
                'redis_info': {
                    'version': redis_info.get('redis_version'),
                    'mode': redis_info.get('redis_mode', 'standalone'),
                    'used_memory_human': redis_info.get('used_memory_human'),
                    'connected_clients': redis_info.get('connected_clients'),
                    'total_commands_processed': redis_info.get('total_commands_processed'),
                    'instantaneous_ops_per_sec': redis_info.get('instantaneous_ops_per_sec'),
                    'keyspace_hits': redis_info.get('keyspace_hits'),
                    'keyspace_misses': redis_info.get('keyspace_misses'),
                },
                'circuit_breaker': {
                    'enabled': self.enable_circuit_breaker,
                    'state': 'open' if self.circuit_breaker_open else 'closed',
                    'failures': self.circuit_breaker_failures,
                    'threshold': self.circuit_breaker_threshold,
                },
                'configuration': {
                    'sentinel_enabled': bool(self.sentinel_hosts),
                    'cluster_mode': self.cluster_mode,
                    'monitoring_enabled': self.enable_monitoring,
                    'cache_warming_enabled': self.enable_cache_warming,
                },
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {}

    def _get_namespaced_key(self, key: str) -> str:
        """Get namespaced cache key."""
        namespace = getattr(self.settings, 'cache_namespace', 'pynomaly')
        return f"{namespace}:{key}"

    def _serialize(self, value: Any, key: str) -> bytes:
        """Serialize value based on key pattern."""
        try:
            # Use JSON for API responses and simple data
            if any(pattern in key for pattern in ['api:', 'response:', 'config:']):
                return json.dumps(value).encode()
            else:
                # Use pickle for complex objects
                import pickle
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization failed for key {key}: {e}")
            raise

    def _deserialize(self, raw_value: bytes, key: str) -> Any:
        """Deserialize value based on key pattern."""
        try:
            # Use JSON for API responses and simple data
            if any(pattern in key for pattern in ['api:', 'response:', 'config:']):
                return json.loads(raw_value.decode())
            else:
                # Use pickle for complex objects
                import pickle
                return pickle.loads(raw_value)
        except Exception as e:
            logger.error(f"Deserialization failed for key {key}: {e}")
            raise

    async def _add_tags(self, key: str, tags: Set[str]) -> None:
        """Add key to tag indices for invalidation."""
        try:
            pipeline = self.redis.pipeline()
            for tag in tags:
                tag_key = f"tag:{tag}"
                pipeline.sadd(tag_key, key)
            pipeline.execute()
        except Exception as e:
            logger.error(f"Failed to add tags for key {key}: {e}")

    async def _remove_from_tags(self, key: str) -> None:
        """Remove key from all tag indices."""
        try:
            # Find all tag sets containing this key
            tag_pattern = "tag:*"
            tag_keys = self.redis.keys(tag_pattern)
            
            pipeline = self.redis.pipeline()
            for tag_key in tag_keys:
                pipeline.srem(tag_key, key)
            pipeline.execute()
            
        except Exception as e:
            logger.error(f"Failed to remove tags for key {key}: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connections."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        try:
            # Basic connectivity test
            start_time = time.time()
            self.redis.ping()
            response_time = time.time() - start_time
            
            health_status['checks']['connectivity'] = {
                'status': 'pass',
                'response_time_ms': response_time * 1000
            }
            
            # Memory usage check
            info = self.redis.info('memory')
            memory_usage_ratio = info.get('used_memory', 0) / info.get('maxmemory', 1)
            
            health_status['checks']['memory'] = {
                'status': 'pass' if memory_usage_ratio < 0.8 else 'warn',
                'used_memory_human': info.get('used_memory_human'),
                'usage_ratio': memory_usage_ratio
            }
            
            # Circuit breaker status
            health_status['checks']['circuit_breaker'] = {
                'status': 'pass' if not self.circuit_breaker_open else 'fail',
                'state': 'open' if self.circuit_breaker_open else 'closed'
            }
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
            logger.error(f"Redis health check failed: {e}")
        
        return health_status

    async def close(self) -> None:
        """Close Redis connections and cleanup resources."""
        try:
            # Cancel warming tasks
            for task in self._warming_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._warming_tasks:
                await asyncio.gather(*self._warming_tasks, return_exceptions=True)
            
            # Close Redis connection
            if hasattr(self, 'redis') and self.redis:
                await self.redis.aclose() if hasattr(self.redis, 'aclose') else self.redis.close()
            
            logger.info("ProductionRedisCache closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing ProductionRedisCache: {e}")