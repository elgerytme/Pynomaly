"""Enhanced Redis caching implementation with advanced features for Issue #99."""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID

import redis
import redis.sentinel
from pydantic import BaseModel, Field
from redis.exceptions import ConnectionError, ResponseError, TimeoutError

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategies for different use cases."""
    
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"
    CACHE_ASIDE = "cache_aside"
    READ_THROUGH = "read_through"


class EvictionPolicy(str, Enum):
    """Cache eviction policies."""
    
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"
    TTL = "ttl"


class InvalidationStrategy(str, Enum):
    """Cache invalidation strategies."""
    
    IMMEDIATE = "immediate"
    LAZY = "lazy"
    TTL_BASED = "ttl_based"
    EVENT_DRIVEN = "event_driven"
    DEPENDENCY_BASED = "dependency_based"


@dataclass
class CacheConfiguration:
    """Enhanced cache configuration."""
    
    redis_url: str = "redis://localhost:6379/0"
    redis_sentinel_service: Optional[str] = None
    redis_sentinel_hosts: List[tuple] = field(default_factory=list)
    redis_cluster_nodes: List[str] = field(default_factory=list)
    
    # Cache behavior
    default_ttl: int = 3600
    max_memory_cache_size: int = 10000
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress if data > 1KB
    
    # Advanced features
    enable_monitoring: bool = True
    enable_circuit_breaker: bool = True
    enable_cache_warming: bool = True
    enable_predictive_loading: bool = True
    
    # Performance tuning
    connection_pool_size: int = 50
    connection_timeout: int = 5
    socket_timeout: int = 5
    retry_attempts: int = 3
    retry_delay: float = 0.1
    
    # Eviction and invalidation
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    invalidation_strategy: InvalidationStrategy = InvalidationStrategy.TTL_BASED
    
    # Monitoring
    metrics_collection_interval: int = 60
    health_check_interval: int = 30
    
    # Distributed caching
    enable_distributed_cache: bool = False
    cache_replication_factor: int = 1
    
    # Security
    redis_password: Optional[str] = None
    enable_ssl: bool = False
    ssl_cert_path: Optional[str] = None


@dataclass
class CacheMetrics:
    """Comprehensive cache metrics."""
    
    # Basic metrics
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    
    # Performance metrics
    total_requests: int = 0
    average_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float('inf')
    
    # Memory metrics
    memory_usage: int = 0
    memory_peak: int = 0
    key_count: int = 0
    
    # Error metrics
    connection_errors: int = 0
    timeout_errors: int = 0
    serialization_errors: int = 0
    
    # Invalidation metrics
    invalidations_immediate: int = 0
    invalidations_lazy: int = 0
    invalidations_ttl: int = 0
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.utcnow)
    uptime_start: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total_errors = self.connection_errors + self.timeout_errors + self.serialization_errors
        return total_errors / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        return (datetime.utcnow() - self.uptime_start).total_seconds()


@dataclass
class CacheEvent:
    """Cache event for monitoring and analytics."""
    
    event_type: str
    key: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker for Redis connections."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise ConnectionError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class CacheEventListener(ABC):
    """Abstract base class for cache event listeners."""
    
    @abstractmethod
    async def on_event(self, event: CacheEvent):
        """Handle cache event."""
        pass


class MetricsCollector(CacheEventListener):
    """Collect cache metrics from events."""
    
    def __init__(self):
        self.metrics = CacheMetrics()
        self.response_times = []
        self._lock = asyncio.Lock()
    
    async def on_event(self, event: CacheEvent):
        """Process cache event and update metrics."""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.last_updated = datetime.utcnow()
            
            # Update response times
            if event.duration_ms > 0:
                self.response_times.append(event.duration_ms)
                if len(self.response_times) > 1000:  # Keep last 1000 measurements
                    self.response_times.pop(0)
                
                avg_time = sum(self.response_times) / len(self.response_times)
                self.metrics.average_response_time = avg_time
                self.metrics.max_response_time = max(self.metrics.max_response_time, event.duration_ms)
                self.metrics.min_response_time = min(self.metrics.min_response_time, event.duration_ms)
            
            # Update operation-specific metrics
            if event.operation == "get":
                if event.success and event.metadata.get("hit"):
                    self.metrics.hits += 1
                else:
                    self.metrics.misses += 1
            elif event.operation == "set":
                self.metrics.sets += 1
            elif event.operation == "delete":
                self.metrics.deletes += 1
            elif event.operation == "evict":
                self.metrics.evictions += 1
            
            # Update error metrics
            if not event.success:
                if "connection" in event.error.lower():
                    self.metrics.connection_errors += 1
                elif "timeout" in event.error.lower():
                    self.metrics.timeout_errors += 1
                elif "serialization" in event.error.lower():
                    self.metrics.serialization_errors += 1
    
    def get_metrics(self) -> CacheMetrics:
        """Get current metrics."""
        return self.metrics


class CacheInvalidationManager:
    """Manage cache invalidation strategies."""
    
    def __init__(self, redis_client: redis.Redis, strategy: InvalidationStrategy):
        self.redis_client = redis_client
        self.strategy = strategy
        self.dependency_graph = defaultdict(set)
        self.invalidation_queue = asyncio.Queue()
        self.tag_registry = defaultdict(set)
        
    async def add_dependency(self, parent_key: str, child_key: str):
        """Add dependency between cache keys."""
        self.dependency_graph[parent_key].add(child_key)
    
    async def add_tag(self, key: str, tag: str):
        """Add tag to cache key."""
        self.tag_registry[tag].add(key)
    
    async def invalidate_key(self, key: str):
        """Invalidate a specific cache key."""
        if self.strategy == InvalidationStrategy.IMMEDIATE:
            await self._invalidate_immediate(key)
        elif self.strategy == InvalidationStrategy.LAZY:
            await self._invalidate_lazy(key)
        elif self.strategy == InvalidationStrategy.DEPENDENCY_BASED:
            await self._invalidate_dependency_based(key)
    
    async def invalidate_tag(self, tag: str):
        """Invalidate all keys with a specific tag."""
        keys_to_invalidate = self.tag_registry.get(tag, set())
        for key in keys_to_invalidate:
            await self.invalidate_key(key)
    
    async def _invalidate_immediate(self, key: str):
        """Immediately invalidate cache key."""
        try:
            self.redis_client.delete(key)
            logger.info(f"Immediately invalidated cache key: {key}")
        except Exception as e:
            logger.error(f"Failed to invalidate key {key}: {e}")
    
    async def _invalidate_lazy(self, key: str):
        """Mark key for lazy invalidation."""
        try:
            # Add to invalidation queue for background processing
            await self.invalidation_queue.put(key)
            logger.info(f"Queued for lazy invalidation: {key}")
        except Exception as e:
            logger.error(f"Failed to queue key for invalidation {key}: {e}")
    
    async def _invalidate_dependency_based(self, key: str):
        """Invalidate key and all its dependencies."""
        try:
            # Invalidate the key itself
            await self._invalidate_immediate(key)
            
            # Invalidate all dependent keys
            dependent_keys = self.dependency_graph.get(key, set())
            for dependent_key in dependent_keys:
                await self._invalidate_dependency_based(dependent_key)
                
        except Exception as e:
            logger.error(f"Failed to invalidate dependencies for key {key}: {e}")
    
    async def process_invalidation_queue(self):
        """Background task to process lazy invalidations."""
        while True:
            try:
                key = await self.invalidation_queue.get()
                await self._invalidate_immediate(key)
                self.invalidation_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing invalidation queue: {e}")
                await asyncio.sleep(1)


class CacheWarmer:
    """Intelligent cache warming system."""
    
    def __init__(self, redis_client: redis.Redis, cache_manager):
        self.redis_client = redis_client
        self.cache_manager = cache_manager
        self.warming_patterns = []
        self.access_patterns = defaultdict(int)
        self.warming_queue = asyncio.Queue()
        
    async def add_warming_pattern(self, pattern: str, loader_func, priority: int = 1):
        """Add a cache warming pattern."""
        self.warming_patterns.append({
            'pattern': pattern,
            'loader': loader_func,
            'priority': priority
        })
    
    async def track_access(self, key: str):
        """Track key access for predictive warming."""
        self.access_patterns[key] += 1
        
        # If key is frequently accessed, consider warming related keys
        if self.access_patterns[key] > 10:  # Threshold
            await self._predict_and_warm(key)
    
    async def warm_cache(self, keys: List[str]):
        """Warm cache with specific keys."""
        for key in keys:
            await self.warming_queue.put(key)
    
    async def _predict_and_warm(self, key: str):
        """Predict and warm related cache keys."""
        # Simple prediction: warm related keys based on pattern
        base_key = key.split(':')[0]
        related_patterns = [pattern for pattern in self.warming_patterns 
                          if base_key in pattern['pattern']]
        
        for pattern in related_patterns:
            try:
                # Execute loader function for pattern
                data = await pattern['loader']()
                # Cache the data (simplified)
                await self.cache_manager.set(pattern['pattern'], data)
                logger.info(f"Predictively warmed cache key: {pattern['pattern']}")
            except Exception as e:
                logger.error(f"Failed to warm cache key {pattern['pattern']}: {e}")
    
    async def background_warming(self):
        """Background task for cache warming."""
        while True:
            try:
                key = await self.warming_queue.get()
                # Find matching pattern and warm
                for pattern in self.warming_patterns:
                    if pattern['pattern'] in key:
                        try:
                            data = await pattern['loader']()
                            await self.cache_manager.set(key, data)
                            logger.info(f"Warmed cache key: {key}")
                        except Exception as e:
                            logger.error(f"Failed to warm key {key}: {e}")
                        break
                
                self.warming_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background warming: {e}")
                await asyncio.sleep(1)


class EnhancedRedisCache:
    """Enhanced Redis cache with advanced features."""
    
    def __init__(self, config: CacheConfiguration):
        self.config = config
        self.redis_client = self._create_redis_client()
        self.circuit_breaker = CircuitBreaker()
        self.metrics_collector = MetricsCollector()
        self.invalidation_manager = CacheInvalidationManager(
            self.redis_client, config.invalidation_strategy
        )
        self.cache_warmer = CacheWarmer(self.redis_client, self)
        self.event_listeners: List[CacheEventListener] = [self.metrics_collector]
        
        # Background tasks
        self.background_tasks = []
        self._start_background_tasks()
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client with appropriate configuration."""
        try:
            if self.config.redis_sentinel_service and self.config.redis_sentinel_hosts:
                # Use Redis Sentinel for high availability
                sentinel = redis.sentinel.Sentinel(
                    self.config.redis_sentinel_hosts,
                    socket_timeout=self.config.socket_timeout
                )
                return sentinel.master_for(
                    self.config.redis_sentinel_service,
                    socket_timeout=self.config.socket_timeout,
                    password=self.config.redis_password
                )
            
            elif self.config.redis_cluster_nodes:
                # Use Redis Cluster for horizontal scaling
                try:
                    from rediscluster import RedisCluster
                    return RedisCluster(
                        startup_nodes=self.config.redis_cluster_nodes,
                        decode_responses=False,
                        password=self.config.redis_password
                    )
                except ImportError:
                    logger.warning("rediscluster not available, falling back to single instance")
                    return redis.from_url(
                        self.config.redis_url,
                        decode_responses=False,
                        socket_timeout=self.config.socket_timeout,
                        password=self.config.redis_password,
                        ssl=self.config.enable_ssl,
                        ssl_cert_reqs=None if not self.config.enable_ssl else 'required'
                    )
            
            else:
                # Single Redis instance
                return redis.from_url(
                    self.config.redis_url,
                    decode_responses=False,
                    socket_timeout=self.config.socket_timeout,
                    password=self.config.redis_password,
                    ssl=self.config.enable_ssl,
                    ssl_cert_reqs=None if not self.config.enable_ssl else 'required'
                )
        
        except Exception as e:
            logger.error(f"Failed to create Redis client: {e}")
            raise
    
    def _start_background_tasks(self):
        """Start background tasks for cache maintenance."""
        if self.config.enable_monitoring:
            task = asyncio.create_task(self._monitoring_task())
            self.background_tasks.append(task)
        
        if self.config.invalidation_strategy == InvalidationStrategy.LAZY:
            task = asyncio.create_task(self.invalidation_manager.process_invalidation_queue())
            self.background_tasks.append(task)
        
        if self.config.enable_cache_warming:
            task = asyncio.create_task(self.cache_warmer.background_warming())
            self.background_tasks.append(task)
    
    async def _emit_event(self, event: CacheEvent):
        """Emit cache event to all listeners."""
        for listener in self.event_listeners:
            try:
                await listener.on_event(event)
            except Exception as e:
                logger.error(f"Error in event listener: {e}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with comprehensive error handling."""
        start_time = time.time()
        event = CacheEvent(
            event_type="cache_get",
            key=key,
            operation="get"
        )
        
        try:
            # Track access for predictive warming
            await self.cache_warmer.track_access(key)
            
            # Get from Redis with circuit breaker
            data = self.circuit_breaker.call(self.redis_client.get, key)
            
            if data is None:
                event.success = True
                event.metadata = {"hit": False}
                return default
            
            # Decompress if needed
            if self.config.enable_compression and len(data) > self.config.compression_threshold:
                data = zlib.decompress(data)
            
            # Deserialize
            try:
                value = pickle.loads(data)
                event.success = True
                event.metadata = {"hit": True, "compressed": len(data) > self.config.compression_threshold}
                return value
            except Exception as e:
                event.success = False
                event.error = f"Deserialization error: {str(e)}"
                logger.error(f"Failed to deserialize cached data for key {key}: {e}")
                return default
        
        except Exception as e:
            event.success = False
            event.error = str(e)
            logger.error(f"Cache get error for key {key}: {e}")
            return default
        
        finally:
            event.duration_ms = (time.time() - start_time) * 1000
            await self._emit_event(event)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[Set[str]] = None) -> bool:
        """Set value in cache with advanced features."""
        start_time = time.time()
        event = CacheEvent(
            event_type="cache_set",
            key=key,
            operation="set"
        )
        
        try:
            ttl = ttl or self.config.default_ttl
            
            # Serialize
            try:
                data = pickle.dumps(value)
            except Exception as e:
                event.success = False
                event.error = f"Serialization error: {str(e)}"
                logger.error(f"Failed to serialize value for key {key}: {e}")
                return False
            
            # Compress if needed
            compressed = False
            if self.config.enable_compression and len(data) > self.config.compression_threshold:
                data = zlib.compress(data)
                compressed = True
            
            # Set in Redis with circuit breaker
            if ttl > 0:
                result = self.circuit_breaker.call(self.redis_client.setex, key, ttl, data)
            else:
                result = self.circuit_breaker.call(self.redis_client.set, key, data)
            
            # Add tags if provided
            if tags:
                for tag in tags:
                    await self.invalidation_manager.add_tag(key, tag)
            
            event.success = bool(result)
            event.metadata = {"ttl": ttl, "compressed": compressed, "size": len(data)}
            return bool(result)
        
        except Exception as e:
            event.success = False
            event.error = str(e)
            logger.error(f"Cache set error for key {key}: {e}")
            return False
        
        finally:
            event.duration_ms = (time.time() - start_time) * 1000
            await self._emit_event(event)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        start_time = time.time()
        event = CacheEvent(
            event_type="cache_delete",
            key=key,
            operation="delete"
        )
        
        try:
            result = self.circuit_breaker.call(self.redis_client.delete, key)
            event.success = bool(result)
            return bool(result)
        
        except Exception as e:
            event.success = False
            event.error = str(e)
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
        
        finally:
            event.duration_ms = (time.time() - start_time) * 1000
            await self._emit_event(event)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(self.circuit_breaker.call(self.redis_client.exists, key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> bool:
        """Clear cache entries."""
        try:
            if pattern:
                # Delete keys matching pattern
                keys = self.circuit_breaker.call(self.redis_client.keys, pattern)
                if keys:
                    self.circuit_breaker.call(self.redis_client.delete, *keys)
            else:
                # Clear all keys
                self.circuit_breaker.call(self.redis_client.flushdb)
            
            return True
        
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def invalidate_by_tag(self, tag: str):
        """Invalidate all keys with a specific tag."""
        await self.invalidation_manager.invalidate_tag(tag)
    
    async def invalidate_by_pattern(self, pattern: str):
        """Invalidate keys matching a pattern."""
        try:
            keys = self.circuit_breaker.call(self.redis_client.keys, pattern)
            for key in keys:
                await self.invalidation_manager.invalidate_key(key.decode('utf-8'))
        except Exception as e:
            logger.error(f"Pattern invalidation error: {e}")
    
    async def get_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        try:
            redis_info = self.circuit_breaker.call(self.redis_client.info)
            metrics = self.metrics_collector.get_metrics()
            
            return {
                "redis_info": {
                    "version": redis_info.get("redis_version"),
                    "memory_usage": redis_info.get("used_memory_human"),
                    "connected_clients": redis_info.get("connected_clients"),
                    "total_connections_received": redis_info.get("total_connections_received"),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                    "expired_keys": redis_info.get("expired_keys", 0),
                    "evicted_keys": redis_info.get("evicted_keys", 0),
                },
                "circuit_breaker": {
                    "state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                    "last_failure_time": self.circuit_breaker.last_failure_time,
                },
                "metrics": {
                    "hit_rate": metrics.hit_rate,
                    "miss_rate": metrics.miss_rate,
                    "error_rate": metrics.error_rate,
                    "average_response_time": metrics.average_response_time,
                    "total_requests": metrics.total_requests,
                    "uptime_seconds": metrics.uptime_seconds,
                },
                "configuration": {
                    "default_ttl": self.config.default_ttl,
                    "compression_enabled": self.config.enable_compression,
                    "monitoring_enabled": self.config.enable_monitoring,
                    "invalidation_strategy": self.config.invalidation_strategy,
                    "eviction_policy": self.config.eviction_policy,
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache system."""
        try:
            start_time = time.time()
            
            # Test basic operations
            test_key = "health_check_test"
            test_value = {"timestamp": time.time(), "test": True}
            
            # Test set
            set_success = await self.set(test_key, test_value, ttl=10)
            
            # Test get
            retrieved_value = await self.get(test_key)
            get_success = retrieved_value == test_value
            
            # Test delete
            delete_success = await self.delete(test_key)
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            health_status = {
                "status": "healthy" if all([set_success, get_success, delete_success]) else "unhealthy",
                "response_time_ms": response_time,
                "operations": {
                    "set": set_success,
                    "get": get_success,
                    "delete": delete_success,
                },
                "circuit_breaker_state": self.circuit_breaker.state,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            return health_status
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    async def _monitoring_task(self):
        """Background task for monitoring and maintenance."""
        while True:
            try:
                # Collect metrics
                info = await self.get_info()
                
                # Log important metrics
                if info.get("metrics"):
                    metrics = info["metrics"]
                    logger.info(
                        f"Cache metrics - Hit rate: {metrics['hit_rate']:.2%}, "
                        f"Avg response time: {metrics['average_response_time']:.2f}ms, "
                        f"Total requests: {metrics['total_requests']}"
                    )
                
                # Check health
                health = await self.health_check()
                if health["status"] != "healthy":
                    logger.warning(f"Cache health check failed: {health}")
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
                await asyncio.sleep(self.config.metrics_collection_interval)
    
    async def close(self):
        """Close cache connections and cleanup."""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            try:
                # Redis client close() is not always async
                if hasattr(self.redis_client, 'close'):
                    if asyncio.iscoroutinefunction(self.redis_client.close):
                        await self.redis_client.close()
                    else:
                        self.redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
        
        logger.info("Enhanced Redis cache closed")


# Global cache instance
_cache_instance: Optional[EnhancedRedisCache] = None


def get_cache_instance(config: Optional[CacheConfiguration] = None) -> EnhancedRedisCache:
    """Get or create global cache instance."""
    global _cache_instance
    
    if _cache_instance is None:
        if config is None:
            config = CacheConfiguration()
        _cache_instance = EnhancedRedisCache(config)
    
    return _cache_instance


def cache_decorator(
    key_func: Optional[callable] = None,
    ttl: Optional[int] = None,
    tags: Optional[Set[str]] = None,
    invalidate_on_error: bool = False
):
    """Decorator for caching function results."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache_instance()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await cache.set(cache_key, result, ttl=ttl, tags=tags)
                return result
            
            except Exception as e:
                if invalidate_on_error:
                    await cache.delete(cache_key)
                raise e
        
        return wrapper
    
    return decorator