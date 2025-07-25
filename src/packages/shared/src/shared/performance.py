"""Performance optimization utilities for hexagonal architecture packages."""

import asyncio
import time
from typing import Any, Dict, Optional, Callable, TypeVar, ParamSpec
from functools import wraps
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    execution_time: float
    memory_usage: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None

class PerformanceTracker:
    """Tracks performance metrics across operations."""
    
    def __init__(self):
        self._metrics: Dict[str, list[PerformanceMetrics]] = {}
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric."""
        if metric.operation_name not in self._metrics:
            self._metrics[metric.operation_name] = []
        self._metrics[metric.operation_name].append(metric)
    
    def get_metrics(self, operation_name: str) -> list[PerformanceMetrics]:
        """Get metrics for a specific operation."""
        return self._metrics.get(operation_name, [])
    
    def get_average_time(self, operation_name: str) -> float:
        """Get average execution time for an operation."""
        metrics = self._metrics.get(operation_name, [])
        if not metrics:
            return 0.0
        return sum(m.execution_time for m in metrics) / len(metrics)

# Global performance tracker
performance_tracker = PerformanceTracker()

def performance_monitor(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                execution_time = time.time() - start_time
                metric = PerformanceMetrics(
                    operation_name=operation_name,
                    execution_time=execution_time,
                    success=success,
                    error_message=error_message
                )
                performance_tracker.record_metric(metric)
        
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                execution_time = time.time() - start_time
                metric = PerformanceMetrics(
                    operation_name=operation_name,
                    execution_time=execution_time,
                    success=success,
                    error_message=error_message
                )
                performance_tracker.record_metric(metric)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

@asynccontextmanager
async def batch_operation(batch_size: int = 100):
    """Context manager for batch processing optimization."""
    batch = []
    
    async def process_batch():
        if batch:
            # Process items in batch
            tasks = [item() for item in batch if callable(item)]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            batch.clear()
    
    try:
        yield batch
    finally:
        await process_batch()

class ConnectionPool:
    """Generic connection pool for database/external service connections."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._connections = asyncio.Queue(maxsize=max_connections)
        self._created_connections = 0
    
    async def get_connection(self):
        """Get a connection from the pool."""
        try:
            return self._connections.get_nowait()
        except asyncio.QueueEmpty:
            if self._created_connections < self.max_connections:
                self._created_connections += 1
                return self._create_connection()
            else:
                return await self._connections.get()
    
    async def return_connection(self, connection):
        """Return a connection to the pool."""
        try:
            self._connections.put_nowait(connection)
        except asyncio.QueueFull:
            # Connection pool is full, close the connection
            await self._close_connection(connection)
            self._created_connections -= 1
    
    def _create_connection(self):
        """Override in subclasses to create actual connections."""
        return {"connection_id": f"conn_{self._created_connections}"}
    
    async def _close_connection(self, connection):
        """Override in subclasses to properly close connections."""
        pass

class CacheManager:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if time.time() > entry['expires']:
            del self._cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        self._cache[key] = {
            'value': value,
            'expires': time.time() + ttl
        }
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

# Global cache instance
cache_manager = CacheManager()

def cached(ttl: int = 300, key_func: Optional[Callable] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

async def optimize_async_operations(operations: list[Callable], max_concurrent: int = 10):
    """Optimize execution of multiple async operations with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_operation(operation):
        async with semaphore:
            return await operation()
    
    tasks = [limited_operation(op) for op in operations]
    return await asyncio.gather(*tasks, return_exceptions=True)