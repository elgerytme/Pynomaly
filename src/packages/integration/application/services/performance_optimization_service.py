"""
Performance optimization service for integration layer.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import weakref

from integration.domain.entities.integration_config import IntegrationConfig
from integration.domain.value_objects.performance_metrics import PerformanceMetrics
from interfaces.shared.error_handling import handle_exceptions


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with expiration and metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0


@dataclass
class ConnectionPool:
    """Connection pool for managing database connections."""
    name: str
    max_connections: int
    active_connections: int = 0
    available_connections: List[Any] = field(default_factory=list)
    connection_factory: Optional[Callable] = None
    timeout_seconds: int = 30
    idle_timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PerformanceProfile:
    """Performance profile for tracking resource usage."""
    operation_name: str
    execution_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        return sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return (self.success_count / total * 100) if total > 0 else 0.0


class PerformanceOptimizationService:
    """Service for optimizing performance across integrated packages."""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize the performance optimization service."""
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=config.performance.max_concurrent_operations)
        self.memory_limit_mb = config.performance.memory_limit_mb
        self.cache_size_mb = config.performance.cache_size_mb
        
        # Performance monitoring
        self.operation_metrics: Dict[str, List[float]] = defaultdict(list)
        self.resource_usage_history: List[Dict[str, Any]] = []
        
        # Start background tasks
        asyncio.create_task(self._cache_cleanup_task())
        asyncio.create_task(self._connection_pool_maintenance_task())
        asyncio.create_task(self._performance_monitoring_task())
    
    async def _cache_cleanup_task(self) -> None:
        """Background task for cache cleanup and maintenance."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = datetime.utcnow()
                expired_keys = []
                
                # Find expired entries
                for key, entry in self.cache.items():
                    if current_time > entry.expires_at:
                        expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    del self.cache[key]
                
                # Check cache size and evict if necessary
                await self._evict_cache_if_needed()
                
                logger.debug(f"Cache cleanup: removed {len(expired_keys)} expired entries")
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
    
    async def _connection_pool_maintenance_task(self) -> None:
        """Background task for connection pool maintenance."""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                for pool in self.connection_pools.values():
                    # Close idle connections
                    current_time = datetime.utcnow()
                    idle_threshold = current_time - timedelta(seconds=pool.idle_timeout_seconds)
                    
                    # In a real implementation, you'd track connection creation times
                    # and close idle connections
                    
                logger.debug("Connection pool maintenance completed")
                
            except Exception as e:
                logger.error(f"Connection pool maintenance error: {str(e)}")
    
    async def _performance_monitoring_task(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Collect current resource usage
                resource_usage = await self._collect_resource_usage()
                self.resource_usage_history.append(resource_usage)
                
                # Keep only last 24 hours of data
                one_day_ago = datetime.utcnow() - timedelta(hours=24)
                self.resource_usage_history = [
                    usage for usage in self.resource_usage_history
                    if usage.get("timestamp", datetime.utcnow()) > one_day_ago
                ]
                
                # Analyze performance trends
                await self._analyze_performance_trends()
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {str(e)}")
    
    async def _collect_resource_usage(self) -> Dict[str, Any]:
        """Collect current resource usage metrics."""
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get process metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "timestamp": datetime.utcnow(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "process_memory_mb": process_memory,
            "cache_size": len(self.cache),
            "active_connections": sum(pool.active_connections for pool in self.connection_pools.values())
        }
    
    async def _analyze_performance_trends(self) -> None:
        """Analyze performance trends and suggest optimizations."""
        if len(self.resource_usage_history) < 10:
            return
        
        # Calculate averages for recent periods
        recent_usage = self.resource_usage_history[-10:]
        avg_cpu = sum(usage["cpu_percent"] for usage in recent_usage) / len(recent_usage)
        avg_memory = sum(usage["memory_percent"] for usage in recent_usage) / len(recent_usage)
        
        # Generate optimization suggestions
        suggestions = []
        
        if avg_cpu > 80:
            suggestions.append("High CPU usage detected. Consider increasing max_concurrent_operations.")
        
        if avg_memory > 85:
            suggestions.append("High memory usage detected. Consider reducing cache size or memory limits.")
        
        if suggestions:
            logger.warning(f"Performance optimization suggestions: {suggestions}")
    
    @handle_exceptions
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        entry = self.cache.get(key)
        if not entry:
            return None
        
        # Check if expired
        if datetime.utcnow() > entry.expires_at:
            del self.cache[key]
            return None
        
        # Update access statistics
        entry.access_count += 1
        entry.last_accessed = datetime.utcnow()
        
        return entry.value
    
    @handle_exceptions
    async def cache_set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set value in cache."""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        size_bytes = len(str(value))  # Simplified size calculation
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            size_bytes=size_bytes
        )
        
        self.cache[key] = entry
        
        # Check if cache needs eviction
        await self._evict_cache_if_needed()
    
    async def _evict_cache_if_needed(self) -> None:
        """Evict cache entries if cache size exceeds limit."""
        # Calculate current cache size
        current_size_mb = sum(entry.size_bytes for entry in self.cache.values()) / 1024 / 1024
        
        if current_size_mb > self.cache_size_mb:
            # Sort by access frequency and age (LRU-like)
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (x[1].access_count, x[1].last_accessed)
            )
            
            # Remove least recently used entries
            for key, entry in sorted_entries[:len(sorted_entries) // 4]:  # Remove 25%
                del self.cache[key]
            
            logger.info(f"Cache evicted entries to reduce size from {current_size_mb:.2f}MB")
    
    @handle_exceptions
    async def get_connection_pool(self, pool_name: str) -> Optional[ConnectionPool]:
        """Get a connection pool."""
        return self.connection_pools.get(pool_name)
    
    @handle_exceptions
    async def create_connection_pool(self, pool_name: str, max_connections: int,
                                   connection_factory: Callable) -> ConnectionPool:
        """Create a new connection pool."""
        pool = ConnectionPool(
            name=pool_name,
            max_connections=max_connections,
            connection_factory=connection_factory,
            timeout_seconds=self.config.performance.query_timeout_seconds
        )
        
        self.connection_pools[pool_name] = pool
        
        # Pre-populate pool with initial connections
        for _ in range(min(5, max_connections)):
            try:
                connection = connection_factory()
                pool.available_connections.append(connection)
            except Exception as e:
                logger.warning(f"Failed to create initial connection for pool {pool_name}: {str(e)}")
        
        logger.info(f"Created connection pool: {pool_name} with {len(pool.available_connections)} initial connections")
        return pool
    
    @handle_exceptions
    async def get_connection(self, pool_name: str) -> Any:
        """Get a connection from a pool."""
        pool = self.connection_pools.get(pool_name)
        if not pool:
            raise ValueError(f"Connection pool not found: {pool_name}")
        
        # Try to get available connection
        if pool.available_connections:
            connection = pool.available_connections.pop()
            pool.active_connections += 1
            return connection
        
        # Create new connection if under limit
        if pool.active_connections < pool.max_connections:
            try:
                connection = pool.connection_factory()
                pool.active_connections += 1
                return connection
            except Exception as e:
                logger.error(f"Failed to create connection for pool {pool_name}: {str(e)}")
                raise
        
        # Wait for connection to become available
        timeout = pool.timeout_seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if pool.available_connections:
                connection = pool.available_connections.pop()
                pool.active_connections += 1
                return connection
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Timeout waiting for connection from pool: {pool_name}")
    
    @handle_exceptions
    async def return_connection(self, pool_name: str, connection: Any) -> None:
        """Return a connection to a pool."""
        pool = self.connection_pools.get(pool_name)
        if not pool:
            logger.warning(f"Connection pool not found when returning connection: {pool_name}")
            return
        
        pool.available_connections.append(connection)
        pool.active_connections = max(0, pool.active_connections - 1)
    
    @handle_exceptions
    async def optimize_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize database query based on performance patterns."""
        # Generate cache key
        cache_key = f"query:{hash(query)}:{hash(str(sorted(params.items())))}"
        
        # Try to get from cache
        cached_result = await self.cache_get(cache_key)
        if cached_result:
            return {
                "result": cached_result,
                "cache_hit": True,
                "execution_time": 0.0
            }
        
        # Execute query with performance tracking
        start_time = time.time()
        
        try:
            # In a real implementation, you'd execute the actual query
            # For now, simulate query execution
            await asyncio.sleep(0.1)  # Simulate query time
            result = {"data": "simulated_result", "rows": 100}
            
            execution_time = time.time() - start_time
            
            # Cache result if it's not too large
            if len(str(result)) < 1024 * 1024:  # 1MB limit
                await self.cache_set(cache_key, result, ttl_seconds=3600)
            
            # Update performance profile
            await self._update_performance_profile("database_query", execution_time, True)
            
            return {
                "result": result,
                "cache_hit": False,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            await self._update_performance_profile("database_query", execution_time, False)
            raise
    
    async def _update_performance_profile(self, operation_name: str, execution_time: float, 
                                        success: bool) -> None:
        """Update performance profile for an operation."""
        profile = self.performance_profiles.get(operation_name)
        if not profile:
            profile = PerformanceProfile(operation_name=operation_name)
            self.performance_profiles[operation_name] = profile
        
        profile.execution_times.append(execution_time)
        profile.last_updated = datetime.utcnow()
        
        # Keep only recent execution times (last 1000)
        if len(profile.execution_times) > 1000:
            profile.execution_times = profile.execution_times[-1000:]
        
        if success:
            profile.success_count += 1
        else:
            profile.failure_count += 1
    
    @handle_exceptions
    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage across the system."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Get current memory usage
        current_usage = await self._collect_resource_usage()
        process_memory_mb = current_usage["process_memory_mb"]
        
        optimizations = []
        
        # Check if memory usage is high
        if process_memory_mb > self.memory_limit_mb * 0.8:
            # Reduce cache size
            original_cache_size = len(self.cache)
            await self._evict_cache_if_needed()
            new_cache_size = len(self.cache)
            
            if new_cache_size < original_cache_size:
                optimizations.append(f"Reduced cache size from {original_cache_size} to {new_cache_size}")
            
            # Close idle connections
            for pool in self.connection_pools.values():
                if len(pool.available_connections) > 5:
                    connections_to_close = len(pool.available_connections) - 5
                    for _ in range(connections_to_close):
                        if pool.available_connections:
                            pool.available_connections.pop()
                    
                    optimizations.append(f"Closed {connections_to_close} idle connections in pool {pool.name}")
        
        # Final memory check
        final_usage = await self._collect_resource_usage()
        memory_saved = current_usage["process_memory_mb"] - final_usage["process_memory_mb"]
        
        return {
            "initial_memory_mb": process_memory_mb,
            "final_memory_mb": final_usage["process_memory_mb"],
            "memory_saved_mb": memory_saved,
            "optimizations": optimizations
        }
    
    @handle_exceptions
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        current_usage = await self._collect_resource_usage()
        
        # Calculate cache statistics
        cache_stats = {
            "total_entries": len(self.cache),
            "total_size_mb": sum(entry.size_bytes for entry in self.cache.values()) / 1024 / 1024,
            "hit_rate": 0.0  # Would need to track hits/misses
        }
        
        # Calculate connection pool statistics
        pool_stats = {}
        for name, pool in self.connection_pools.items():
            pool_stats[name] = {
                "max_connections": pool.max_connections,
                "active_connections": pool.active_connections,
                "available_connections": len(pool.available_connections),
                "utilization": (pool.active_connections / pool.max_connections) * 100
            }
        
        # Calculate operation performance
        operation_stats = {}
        for name, profile in self.performance_profiles.items():
            operation_stats[name] = {
                "average_execution_time": profile.average_execution_time,
                "success_rate": profile.success_rate,
                "total_operations": profile.success_count + profile.failure_count
            }
        
        return {
            "timestamp": datetime.utcnow(),
            "resource_usage": current_usage,
            "cache_statistics": cache_stats,
            "connection_pool_statistics": pool_stats,
            "operation_statistics": operation_stats,
            "performance_trends": self.resource_usage_history[-10:] if self.resource_usage_history else []
        }
    
    @handle_exceptions
    async def suggest_optimizations(self) -> List[str]:
        """Generate optimization suggestions based on performance data."""
        suggestions = []
        
        # Analyze cache performance
        cache_size_mb = sum(entry.size_bytes for entry in self.cache.values()) / 1024 / 1024
        if cache_size_mb > self.cache_size_mb * 0.9:
            suggestions.append("Cache is near capacity. Consider increasing cache size or implementing better eviction policies.")
        
        # Analyze connection pool utilization
        for name, pool in self.connection_pools.items():
            utilization = (pool.active_connections / pool.max_connections) * 100
            if utilization > 90:
                suggestions.append(f"Connection pool '{name}' is highly utilized ({utilization:.1f}%). Consider increasing pool size.")
        
        # Analyze operation performance
        for name, profile in self.performance_profiles.items():
            if profile.success_rate < 95:
                suggestions.append(f"Operation '{name}' has low success rate ({profile.success_rate:.1f}%). Investigate failure causes.")
            
            if profile.average_execution_time > 5.0:
                suggestions.append(f"Operation '{name}' has high execution time ({profile.average_execution_time:.2f}s). Consider optimization.")
        
        return suggestions
    
    async def shutdown(self) -> None:
        """Shutdown the performance optimization service."""
        logger.info("Shutting down performance optimization service...")
        
        # Close connection pools
        for pool in self.connection_pools.values():
            for connection in pool.available_connections:
                # Close connection (implementation depends on connection type)
                pass
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clear cache
        self.cache.clear()
        
        logger.info("Performance optimization service shutdown complete")