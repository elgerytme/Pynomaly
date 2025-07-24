"""Performance monitoring and profiling for anomaly detection operations."""

from __future__ import annotations

import time
import asyncio
from typing import Dict, Any, List, Optional, Callable, ContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
import threading
import functools

from ..logging import get_logger
from .metrics_collector import get_metrics_collector

logger = get_logger(__name__)


@dataclass
class PerformanceProfile:
    """Performance profile for an operation."""
    
    operation: str
    total_duration_ms: float
    cpu_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    io_operations: int = 0
    network_requests: int = 0
    database_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ResourceUsage:
    """Resource usage snapshot."""
    
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_received_mb: float


class PerformanceContext:
    """Context manager for tracking operation performance."""
    
    def __init__(
        self,
        operation: str,
        monitor: PerformanceMonitor,
        track_memory: bool = True,
        track_io: bool = False,
        custom_metrics: Optional[Dict[str, Callable[[], float]]] = None
    ):
        """Initialize performance context.
        
        Args:
            operation: Name of the operation being monitored
            monitor: Performance monitor instance
            track_memory: Whether to track memory usage
            track_io: Whether to track I/O operations
            custom_metrics: Optional custom metric collectors
        """
        self.operation = operation
        self.monitor = monitor
        self.track_memory = track_memory
        self.track_io = track_io
        self.custom_metrics = custom_metrics or {}
        
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: Optional[float] = None
        self.io_counters_start: Optional[Any] = None
        self.success = True
        self.error_message: Optional[str] = None
        
        # Tracked metrics
        self.io_operations = 0
        self.network_requests = 0
        self.database_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def __enter__(self) -> PerformanceContext:
        """Start performance monitoring."""
        self.start_time = time.perf_counter()
        
        # Track memory if requested
        if self.track_memory:
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                self.start_memory = memory_info.rss / (1024 * 1024)  # MB
                self.peak_memory = self.start_memory
            except ImportError:
                logger.debug("psutil not available, memory tracking disabled")
                self.track_memory = False
        
        # Track I/O if requested
        if self.track_io:
            try:
                import psutil
                process = psutil.Process()
                self.io_counters_start = process.io_counters()
            except (ImportError, AttributeError):
                logger.debug("I/O tracking not available")
                self.track_io = False
        
        logger.debug("Performance monitoring started", operation=self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End performance monitoring and record results."""
        self.end_time = time.perf_counter()
        
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val) if exc_val else str(exc_type)
        
        # Calculate duration
        total_duration_ms = (self.end_time - self.start_time) * 1000 if self.start_time else 0
        
        # Get final memory usage
        memory_usage_mb = None
        if self.track_memory and self.start_memory is not None:
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                current_memory = memory_info.rss / (1024 * 1024)  # MB
                memory_usage_mb = current_memory - self.start_memory
                self.peak_memory = max(self.peak_memory or 0, current_memory)
            except Exception as e:
                logger.debug("Failed to get final memory usage", error=str(e))
        
        # Collect custom metrics
        custom_metrics = {}
        for name, collector in self.custom_metrics.items():
            try:
                custom_metrics[name] = collector()
            except Exception as e:
                logger.warning("Failed to collect custom metric", 
                              metric_name=name, error=str(e))
        
        # Create performance profile
        profile = PerformanceProfile(
            operation=self.operation,
            total_duration_ms=total_duration_ms,
            memory_usage_mb=memory_usage_mb,
            peak_memory_mb=self.peak_memory,
            io_operations=self.io_operations,
            network_requests=self.network_requests,
            database_queries=self.database_queries,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            custom_metrics=custom_metrics,
            success=self.success,
            error_message=self.error_message
        )
        
        # Record profile
        self.monitor.record_profile(profile)
        
        logger.debug("Performance monitoring completed",
                    operation=self.operation,
                    duration_ms=total_duration_ms,
                    success=self.success)
    
    def increment_counter(self, counter_name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        if counter_name == "io_operations":
            self.io_operations += value
        elif counter_name == "network_requests":
            self.network_requests += value
        elif counter_name == "database_queries":
            self.database_queries += value
        elif counter_name == "cache_hits":
            self.cache_hits += value
        elif counter_name == "cache_misses":
            self.cache_misses += value
    
    def update_peak_memory(self) -> None:
        """Update peak memory usage (call periodically during long operations)."""
        if self.track_memory:
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                current_memory = memory_info.rss / (1024 * 1024)  # MB
                self.peak_memory = max(self.peak_memory or 0, current_memory)
            except Exception as e:
                logger.debug("Failed to update peak memory", error=str(e))


class PerformanceMonitor:
    """Performance monitoring and profiling service."""
    
    def __init__(self, buffer_size: int = 5000, enable_resource_tracking: bool = True):
        """Initialize performance monitor.
        
        Args:
            buffer_size: Maximum number of profiles to keep in memory
            enable_resource_tracking: Whether to track system resource usage
        """
        self.buffer_size = buffer_size
        self.enable_resource_tracking = enable_resource_tracking
        
        # Thread-safe collections
        self._lock = threading.RLock()
        self._profiles: deque[PerformanceProfile] = deque(maxlen=buffer_size)
        self._resource_usage: deque[ResourceUsage] = deque(maxlen=1000)
        
        # Aggregated statistics
        self._operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_duration_ms": 0,
            "min_duration_ms": float('inf'),
            "max_duration_ms": 0,
            "success_count": 0,
            "error_count": 0,
            "avg_memory_mb": 0,
            "peak_memory_mb": 0
        })
        
        # Performance thresholds (configurable)
        self.slow_operation_threshold_ms = 5000.0
        self.high_memory_threshold_mb = 500.0
        
        # Get metrics collector
        self.metrics_collector = get_metrics_collector()
        
        # Start resource tracking if enabled
        if self.enable_resource_tracking:
            self._start_resource_tracking()
        
        logger.info("Performance monitor initialized",
                   buffer_size=buffer_size,
                   resource_tracking=enable_resource_tracking)
    
    def _start_resource_tracking(self) -> None:
        """Start background resource usage tracking."""
        def track_resources():
            """Background thread for resource tracking."""
            try:
                import psutil
                while True:
                    try:
                        # Get system resource usage
                        cpu_percent = psutil.cpu_percent(interval=1)
                        memory_info = psutil.virtual_memory()
                        
                        # Get process-specific resource usage
                        process = psutil.Process()
                        process_memory = process.memory_info().rss / (1024 * 1024)  # MB
                        
                        # Get I/O stats if available
                        disk_io = psutil.disk_io_counters()
                        network_io = psutil.net_io_counters()
                        
                        resource_usage = ResourceUsage(
                            timestamp=datetime.utcnow(),
                            cpu_percent=cpu_percent,
                            memory_mb=process_memory,
                            memory_percent=memory_info.percent,
                            disk_io_read_mb=disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                            disk_io_write_mb=disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
                            network_sent_mb=network_io.bytes_sent / (1024 * 1024) if network_io else 0,
                            network_received_mb=network_io.bytes_recv / (1024 * 1024) if network_io else 0
                        )
                        
                        with self._lock:
                            self._resource_usage.append(resource_usage)
                        
                        # Record metrics
                        self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
                        self.metrics_collector.set_gauge("system_memory_percent", memory_info.percent)
                        self.metrics_collector.set_gauge("process_memory_mb", process_memory)
                        
                        time.sleep(30)  # Sample every 30 seconds
                        
                    except Exception as e:
                        logger.error("Resource tracking error", error=str(e))
                        time.sleep(60)  # Wait longer on error
                        
            except ImportError:
                logger.warning("psutil not available, resource tracking disabled")
        
        # Start background thread
        thread = threading.Thread(target=track_resources, daemon=True)
        thread.start()
    
    def create_context(
        self,
        operation: str,
        track_memory: bool = True,
        track_io: bool = False,
        custom_metrics: Optional[Dict[str, Callable[[], float]]] = None
    ) -> PerformanceContext:
        """Create a performance monitoring context.
        
        Args:
            operation: Name of the operation to monitor
            track_memory: Whether to track memory usage
            track_io: Whether to track I/O operations
            custom_metrics: Optional custom metric collectors
            
        Returns:
            Performance context manager
        """
        return PerformanceContext(
            operation=operation,
            monitor=self,
            track_memory=track_memory,
            track_io=track_io,
            custom_metrics=custom_metrics
        )
    
    def record_profile(self, profile: PerformanceProfile) -> None:
        """Record a performance profile.
        
        Args:
            profile: Performance profile to record
        """
        with self._lock:
            self._profiles.append(profile)
            
            # Update operation statistics
            stats = self._operation_stats[profile.operation]
            stats["count"] += 1
            stats["total_duration_ms"] += profile.total_duration_ms
            stats["min_duration_ms"] = min(stats["min_duration_ms"], profile.total_duration_ms)
            stats["max_duration_ms"] = max(stats["max_duration_ms"], profile.total_duration_ms)
            
            if profile.success:
                stats["success_count"] += 1
            else:
                stats["error_count"] += 1
            
            if profile.memory_usage_mb is not None:
                current_avg = stats.get("avg_memory_mb", 0)
                stats["avg_memory_mb"] = (current_avg * (stats["count"] - 1) + profile.memory_usage_mb) / stats["count"]
            
            if profile.peak_memory_mb is not None:
                stats["peak_memory_mb"] = max(stats.get("peak_memory_mb", 0), profile.peak_memory_mb)
        
        # Record metrics
        tags = {
            "operation": profile.operation,
            "success": str(profile.success).lower()
        }
        
        self.metrics_collector.record_timing(
            "operation_duration",
            profile.total_duration_ms,
            tags
        )
        
        if profile.memory_usage_mb is not None:
            self.metrics_collector.record_metric(
                "operation_memory_usage",
                profile.memory_usage_mb,
                tags,
                "megabytes"
            )
        
        # Check for performance issues
        self._check_performance_thresholds(profile)
        
        logger.debug("Performance profile recorded",
                    operation=profile.operation,
                    duration_ms=profile.total_duration_ms,
                    success=profile.success)
    
    def _check_performance_thresholds(self, profile: PerformanceProfile) -> None:
        """Check performance profile against thresholds and log warnings."""
        warnings = []
        
        if profile.total_duration_ms > self.slow_operation_threshold_ms:
            warnings.append(f"slow operation ({profile.total_duration_ms:.1f}ms)")
        
        if profile.peak_memory_mb and profile.peak_memory_mb > self.high_memory_threshold_mb:
            warnings.append(f"high memory usage ({profile.peak_memory_mb:.1f}MB)")
        
        if warnings:
            logger.warning("Performance threshold exceeded",
                          operation=profile.operation,
                          warnings=warnings,
                          duration_ms=profile.total_duration_ms,
                          peak_memory_mb=profile.peak_memory_mb)
    
    def get_operation_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for operations.
        
        Args:
            operation: Specific operation to get stats for, or None for all
            
        Returns:
            Dictionary of operation statistics
        """
        with self._lock:
            if operation:
                if operation in self._operation_stats:
                    stats = self._operation_stats[operation].copy()
                    # Calculate derived metrics
                    if stats["count"] > 0:
                        stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                        stats["success_rate"] = stats["success_count"] / stats["count"]
                        stats["error_rate"] = stats["error_count"] / stats["count"]
                    return {operation: stats}
                else:
                    return {}
            else:
                # Return all operation stats
                all_stats = {}
                for op, stats in self._operation_stats.items():
                    op_stats = stats.copy()
                    if op_stats["count"] > 0:
                        op_stats["avg_duration_ms"] = op_stats["total_duration_ms"] / op_stats["count"]
                        op_stats["success_rate"] = op_stats["success_count"] / op_stats["count"]
                        op_stats["error_rate"] = op_stats["error_count"] / op_stats["count"]
                    all_stats[op] = op_stats
                return all_stats
    
    def get_recent_profiles(
        self,
        operation: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        only_errors: bool = False
    ) -> List[PerformanceProfile]:
        """Get recent performance profiles with filtering.
        
        Args:
            operation: Filter by operation name
            since: Only include profiles after this time
            limit: Maximum number of profiles to return
            only_errors: Only include failed operations
            
        Returns:
            List of performance profiles
        """
        with self._lock:
            profiles = list(self._profiles)
        
        # Apply filters
        if operation:
            profiles = [p for p in profiles if p.operation == operation]
        
        if since:
            profiles = [p for p in profiles if p.timestamp > since]
        
        if only_errors:
            profiles = [p for p in profiles if not p.success]
        
        # Sort by timestamp (newest first)
        profiles.sort(key=lambda p: p.timestamp, reverse=True)
        
        if limit:
            profiles = profiles[:limit]
        
        return profiles
    
    def get_resource_usage(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[ResourceUsage]:
        """Get resource usage history.
        
        Args:
            since: Only include usage after this time
            limit: Maximum number of entries to return
            
        Returns:
            List of resource usage snapshots
        """
        if not self.enable_resource_tracking:
            return []
        
        with self._lock:
            usage_list = list(self._resource_usage)
        
        if since:
            usage_list = [u for u in usage_list if u.timestamp > since]
        
        # Sort by timestamp (newest first)
        usage_list.sort(key=lambda u: u.timestamp, reverse=True)
        
        if limit:
            usage_list = usage_list[:limit]
        
        return usage_list
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary.
        
        Returns:
            Dictionary with performance summary
        """
        with self._lock:
            total_profiles = len(self._profiles)
            
            if total_profiles == 0:
                return {
                    "total_profiles": 0,
                    "operations_monitored": 0,
                    "overall_success_rate": 0.0,
                    "avg_duration_ms": 0.0
                }
            
            # Calculate overall metrics
            total_operations = sum(stats["count"] for stats in self._operation_stats.values())
            total_successes = sum(stats["success_count"] for stats in self._operation_stats.values())
            total_duration = sum(stats["total_duration_ms"] for stats in self._operation_stats.values())
            
            # Recent performance (last 10 minutes)
            recent_cutoff = datetime.utcnow() - timedelta(minutes=10)
            recent_profiles = [p for p in self._profiles if p.timestamp > recent_cutoff]
            
            recent_success_rate = 0.0
            if recent_profiles:
                recent_successes = sum(1 for p in recent_profiles if p.success)
                recent_success_rate = recent_successes / len(recent_profiles)
            
            return {
                "total_profiles": total_profiles,
                "operations_monitored": len(self._operation_stats),
                "total_operations": total_operations,
                "overall_success_rate": total_successes / total_operations if total_operations > 0 else 0.0,
                "avg_duration_ms": total_duration / total_operations if total_operations > 0 else 0.0,
                "recent_profiles": len(recent_profiles),
                "recent_success_rate": recent_success_rate,
                "slowest_operation": max(
                    self._operation_stats.items(),
                    key=lambda x: x[1]["max_duration_ms"],
                    default=("none", {"max_duration_ms": 0})
                )[0] if self._operation_stats else "none"
            }
    
    def clear_profiles(self, operation: Optional[str] = None) -> int:
        """Clear performance profiles.
        
        Args:
            operation: Clear profiles for specific operation, or None for all
            
        Returns:
            Number of profiles cleared
        """
        with self._lock:
            if operation:
                # Clear profiles for specific operation
                original_count = len(self._profiles)
                self._profiles = deque(
                    (p for p in self._profiles if p.operation != operation),
                    maxlen=self.buffer_size
                )
                cleared_count = original_count - len(self._profiles)
                
                # Clear operation stats
                if operation in self._operation_stats:
                    del self._operation_stats[operation]
            else:
                # Clear all profiles
                cleared_count = len(self._profiles)
                self._profiles.clear()
                self._operation_stats.clear()
        
        logger.info("Performance profiles cleared",
                   operation=operation or "all",
                   cleared_count=cleared_count)
        
        return cleared_count


# Decorator for automatic performance monitoring
def monitor_performance(
    operation: Optional[str] = None,
    track_memory: bool = True,
    track_io: bool = False,
    custom_metrics: Optional[Dict[str, Callable[[], float]]] = None
):
    """Decorator for automatic performance monitoring.
    
    Args:
        operation: Operation name (defaults to function name)
        track_memory: Whether to track memory usage
        track_io: Whether to track I/O operations
        custom_metrics: Optional custom metric collectors
    """
    def decorator(func):
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with monitor.create_context(op_name, track_memory, track_io, custom_metrics):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Global performance monitor instance
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_performance_monitor
    
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    
    return _global_performance_monitor