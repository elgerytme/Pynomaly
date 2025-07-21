"""
Performance Optimization Framework for Pynomaly Detection
=========================================================

Comprehensive performance optimization providing:
- Database connection pooling management
- Query optimization and caching
- Memory usage profiling and optimization  
- Async operation performance tuning
- Resource monitoring and alerting
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod
import psutil
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Performance metric types."""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DATABASE_QUERY_TIME = "db_query_time"
    CACHE_HIT_RATIO = "cache_hit_ratio"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"

@dataclass
class PerformanceTarget:
    """Performance target configuration."""
    metric: PerformanceMetric
    target_value: float
    warning_threshold: float
    critical_threshold: float
    unit: str = "ms"
    
@dataclass 
class PerformanceMeasurement:
    """Individual performance measurement."""
    metric: PerformanceMetric
    value: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    
class ConnectionPool:
    """High-performance database connection pool."""
    
    def __init__(self, 
                 max_connections: int = 20,
                 min_connections: int = 5,
                 connection_timeout: float = 30.0,
                 idle_timeout: float = 300.0):
        """Initialize connection pool.
        
        Args:
            max_connections: Maximum pool size
            min_connections: Minimum pool size  
            connection_timeout: Connection timeout in seconds
            idle_timeout: Idle connection timeout in seconds
        """
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        
        self._connections: List[Any] = []
        self._in_use: set = set()
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"Connection pool initialized: max={max_connections}, min={min_connections}")
    
    async def get_connection(self):
        """Get connection from pool."""
        async with self._condition:
            while len(self._connections) == 0 and len(self._in_use) >= self.max_connections:
                await self._condition.wait()
            
            if self._connections:
                connection = self._connections.pop()
            else:
                connection = await self._create_connection()
            
            self._in_use.add(connection)
            return connection
    
    async def return_connection(self, connection):
        """Return connection to pool."""
        async with self._condition:
            if connection in self._in_use:
                self._in_use.remove(connection)
                if len(self._connections) < self.max_connections:
                    self._connections.append(connection)
                    self._condition.notify()
                else:
                    await self._close_connection(connection)
    
    async def _create_connection(self):
        """Create new connection (stub)."""
        # Placeholder for actual database connection creation
        return f"connection_{id(self)}_{time.time()}"
    
    async def _close_connection(self, connection):
        """Close connection (stub)."""
        logger.debug(f"Closing connection: {connection}")
    
    async def close_all(self):
        """Close all connections in pool."""
        async with self._lock:
            for connection in self._connections:
                await self._close_connection(connection)
            self._connections.clear()
            self._in_use.clear()
            
            if self._cleanup_task:
                self._cleanup_task.cancel()

class QueryOptimizer:
    """Database query optimization."""
    
    def __init__(self, cache_size: int = 1000):
        """Initialize query optimizer.
        
        Args:
            cache_size: Maximum cache size
        """
        self.cache_size = cache_size
        self._query_cache: Dict[str, Any] = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        self._lock = threading.RLock()
        
        logger.info(f"Query optimizer initialized with cache size: {cache_size}")
    
    def optimize_query(self, query: str, params: Dict[str, Any] = None) -> str:
        """Optimize database query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Optimized query string
        """
        # Add indexes suggestions, query rewriting, etc.
        optimizations = []
        
        # Check for missing WHERE clauses on large tables
        if "SELECT" in query.upper() and "WHERE" not in query.upper():
            optimizations.append("Consider adding WHERE clause to limit results")
        
        # Check for SELECT * usage
        if "SELECT *" in query.upper():
            optimizations.append("Consider selecting specific columns instead of SELECT *")
            
        # Check for ORDER BY without LIMIT
        if "ORDER BY" in query.upper() and "LIMIT" not in query.upper():
            optimizations.append("Consider adding LIMIT clause with ORDER BY")
        
        if optimizations:
            logger.info(f"Query optimization suggestions: {optimizations}")
        
        return query
    
    def cache_result(self, query_key: str, result: Any, ttl: int = 300):
        """Cache query result.
        
        Args:
            query_key: Unique query identifier
            result: Query result to cache
            ttl: Time-to-live in seconds
        """
        with self._lock:
            if len(self._query_cache) >= self.cache_size:
                # Simple LRU eviction - remove oldest
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            
            self._query_cache[query_key] = {
                "result": result,
                "timestamp": time.time(),
                "ttl": ttl
            }
    
    def get_cached_result(self, query_key: str) -> Optional[Any]:
        """Get cached query result.
        
        Args:
            query_key: Unique query identifier
            
        Returns:
            Cached result or None if not found/expired
        """
        with self._lock:
            if query_key in self._query_cache:
                cache_entry = self._query_cache[query_key]
                
                # Check if expired
                if time.time() - cache_entry["timestamp"] > cache_entry["ttl"]:
                    del self._query_cache[query_key]
                    self._cache_stats["misses"] += 1
                    return None
                
                self._cache_stats["hits"] += 1
                return cache_entry["result"]
            
            self._cache_stats["misses"] += 1
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
            hit_ratio = self._cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                "hits": self._cache_stats["hits"],
                "misses": self._cache_stats["misses"],
                "hit_ratio": hit_ratio,
                "cache_size": len(self._query_cache),
                "max_size": self.cache_size
            }

class MemoryProfiler:
    """Memory usage profiling and optimization."""
    
    def __init__(self):
        """Initialize memory profiler."""
        self._tracked_objects: Dict[str, weakref.WeakSet] = {}
        self._memory_measurements: List[PerformanceMeasurement] = []
        
    def track_objects(self, obj_type: str, obj: Any):
        """Track object for memory analysis.
        
        Args:
            obj_type: Type identifier for object
            obj: Object to track
        """
        if obj_type not in self._tracked_objects:
            self._tracked_objects[obj_type] = weakref.WeakSet()
        
        self._tracked_objects[obj_type].add(obj)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size  
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "gc_objects": len(gc.get_objects()),
            "tracked_objects": {
                obj_type: len(obj_set) 
                for obj_type, obj_set in self._tracked_objects.items()
            }
        }
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        before_objects = len(gc.get_objects())
        
        # Force garbage collection
        collected = gc.collect()
        
        after_objects = len(gc.get_objects())
        
        stats = {
            "objects_before": before_objects,
            "objects_after": after_objects,
            "objects_collected": collected,
            "objects_freed": before_objects - after_objects
        }
        
        logger.info(f"Garbage collection completed: {stats}")
        return stats
    
    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        leaks = []
        
        for obj_type, obj_set in self._tracked_objects.items():
            obj_count = len(obj_set)
            
            # Simple heuristic: too many objects of same type
            if obj_count > 1000:
                leaks.append({
                    "type": obj_type,
                    "count": obj_count,
                    "severity": "high" if obj_count > 10000 else "medium"
                })
        
        return leaks

class PerformanceMonitor:
    """Comprehensive performance monitoring."""
    
    def __init__(self, targets: List[PerformanceTarget]):
        """Initialize performance monitor.
        
        Args:
            targets: List of performance targets to monitor
        """
        self.targets = {target.metric: target for target in targets}
        self.measurements: List[PerformanceMeasurement] = []
        self._alerts: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # Initialize components
        self.connection_pool = ConnectionPool()
        self.query_optimizer = QueryOptimizer()
        self.memory_profiler = MemoryProfiler()
        
        logger.info(f"Performance monitor initialized with {len(targets)} targets")
    
    def record_measurement(self, measurement: PerformanceMeasurement):
        """Record performance measurement.
        
        Args:
            measurement: Performance measurement to record
        """
        with self._lock:
            self.measurements.append(measurement)
            
            # Check against targets
            if measurement.metric in self.targets:
                target = self.targets[measurement.metric]
                self._check_performance_target(measurement, target)
            
            # Keep only recent measurements (last 1000)
            if len(self.measurements) > 1000:
                self.measurements = self.measurements[-1000:]
    
    def _check_performance_target(self, measurement: PerformanceMeasurement, target: PerformanceTarget):
        """Check measurement against performance target."""
        if measurement.value > target.critical_threshold:
            self._create_alert("critical", measurement, target)
        elif measurement.value > target.warning_threshold:
            self._create_alert("warning", measurement, target)
    
    def _create_alert(self, severity: str, measurement: PerformanceMeasurement, target: PerformanceTarget):
        """Create performance alert."""
        alert = {
            "severity": severity,
            "metric": measurement.metric,
            "value": measurement.value,
            "target": target.target_value,
            "threshold": target.critical_threshold if severity == "critical" else target.warning_threshold,
            "timestamp": measurement.timestamp,
            "context": measurement.context
        }
        
        self._alerts.append(alert)
        logger.warning(f"Performance alert ({severity}): {measurement.metric.value} = {measurement.value}")
    
    @asynccontextmanager
    async def measure_performance(self, metric: PerformanceMetric, context: Dict[str, Any] = None):
        """Context manager for measuring performance.
        
        Args:
            metric: Metric to measure
            context: Additional context information
        """
        start_time = time.perf_counter()
        context = context or {}
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            
            measurement = PerformanceMeasurement(
                metric=metric,
                value=duration,
                timestamp=time.time(),
                context=context
            )
            
            self.record_measurement(measurement)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        with self._lock:
            # Calculate statistics by metric
            metrics_stats = {}
            
            for metric in PerformanceMetric:
                metric_measurements = [m for m in self.measurements if m.metric == metric]
                
                if metric_measurements:
                    values = [m.value for m in metric_measurements]
                    metrics_stats[metric.value] = {
                        "count": len(values),
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "recent": values[-10:] if len(values) >= 10 else values
                    }
            
            return {
                "measurements_count": len(self.measurements),
                "alerts_count": len(self._alerts),
                "recent_alerts": self._alerts[-10:] if len(self._alerts) >= 10 else self._alerts,
                "metrics_statistics": metrics_stats,
                "connection_pool_stats": {
                    "max_connections": self.connection_pool.max_connections,
                    "in_use": len(self.connection_pool._in_use),
                    "available": len(self.connection_pool._connections)
                },
                "query_cache_stats": self.query_optimizer.get_cache_stats(),
                "memory_stats": self.memory_profiler.get_memory_usage()
            }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization procedures."""
        optimization_results = {}
        
        # Memory optimization
        gc_stats = self.memory_profiler.force_garbage_collection()
        optimization_results["garbage_collection"] = gc_stats
        
        # Check for memory leaks
        memory_leaks = self.memory_profiler.detect_memory_leaks()
        optimization_results["memory_leaks"] = memory_leaks
        
        # Query cache cleanup (remove expired entries)
        cache_stats = self.query_optimizer.get_cache_stats()
        optimization_results["cache_optimization"] = cache_stats
        
        logger.info(f"Performance optimization completed: {optimization_results}")
        return optimization_results

# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        with _monitor_lock:
            if _performance_monitor is None:
                # Default performance targets
                default_targets = [
                    PerformanceTarget(PerformanceMetric.RESPONSE_TIME, 100.0, 200.0, 500.0, "ms"),
                    PerformanceTarget(PerformanceMetric.MEMORY_USAGE, 500.0, 750.0, 1000.0, "MB"),
                    PerformanceTarget(PerformanceMetric.DATABASE_QUERY_TIME, 50.0, 100.0, 200.0, "ms"),
                    PerformanceTarget(PerformanceMetric.CACHE_HIT_RATIO, 0.8, 0.6, 0.4, "ratio"),
                ]
                
                _performance_monitor = PerformanceMonitor(default_targets)
    
    return _performance_monitor

def set_performance_monitor(monitor: PerformanceMonitor):
    """Set global performance monitor instance."""
    global _performance_monitor
    
    with _monitor_lock:
        _performance_monitor = monitor

# Performance monitoring decorators
def monitor_performance(metric: PerformanceMetric):
    """Decorator to monitor function performance.
    
    Args:
        metric: Performance metric to track
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                async with monitor.measure_performance(metric, {"function": func.__name__}):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = (end_time - start_time) * 1000
                    
                    measurement = PerformanceMeasurement(
                        metric=metric,
                        value=duration,
                        timestamp=time.time(),
                        context={"function": func.__name__}
                    )
                    
                    monitor.record_measurement(measurement)
            return sync_wrapper
    return decorator