"""Performance profiling and bottleneck identification tools."""

import logging
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, ContextManager
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps
from collections import defaultdict, deque
import sys
import traceback
import gc
import resource
import psutil
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_delta_mb: float
    thread_id: str
    call_stack: List[str]
    custom_metrics: Dict[str, Union[int, float, str]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_ms": self.duration_ms,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "thread_id": self.thread_id,
            "call_stack": self.call_stack,
            "custom_metrics": self.custom_metrics
        }


@dataclass
class BottleneckReport:
    """Report of identified performance bottlenecks."""
    bottleneck_type: str
    severity: str  # low, medium, high, critical
    operation_name: str
    description: str
    impact_score: float
    recommendations: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "bottleneck_type": self.bottleneck_type,
            "severity": self.severity,
            "operation_name": self.operation_name,
            "description": self.description,
            "impact_score": self.impact_score,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SystemSnapshot:
    """System resource snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int
    open_files: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class PerformanceProfiler:
    """Performance profiler for identifying bottlenecks and optimizing performance."""
    
    def __init__(self, 
                 max_metrics_history: int = 10000,
                 snapshot_interval: int = 60,
                 enable_detailed_profiling: bool = False):
        """Initialize performance profiler.
        
        Args:
            max_metrics_history: Maximum number of metrics to keep in memory
            snapshot_interval: Interval in seconds for system snapshots
            enable_detailed_profiling: Enable detailed call stack profiling
        """
        self.max_metrics_history = max_metrics_history
        self.snapshot_interval = snapshot_interval
        self.enable_detailed_profiling = enable_detailed_profiling
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self._metrics_history: deque = deque(maxlen=max_metrics_history)
        self._system_snapshots: deque = deque(maxlen=1440)  # 24 hours at 1min intervals
        self._operation_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Profiling state
        self._active_operations: Dict[str, Dict[str, Any]] = {}
        self._profiling_lock = threading.Lock()
        
        # Initialize system monitoring
        try:
            self.process = psutil.Process()
            self._last_io_counters = self.process.io_counters()
            self._last_net_counters = psutil.net_io_counters()
        except Exception as e:
            self.logger.warning(f"Failed to initialize system monitoring: {e}")
            self.process = None
    
    def start_monitoring(self):
        """Start background system monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background system monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                self._take_system_snapshot()
                time.sleep(self.snapshot_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.snapshot_interval)
    
    def _take_system_snapshot(self):
        """Take a system resource snapshot."""
        if not self.process:
            return
        
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Process-specific metrics
            process_memory = self.process.memory_info()
            
            # I/O metrics
            current_io = self.process.io_counters()
            current_net = psutil.net_io_counters()
            
            # Calculate deltas
            io_read_delta = (current_io.read_bytes - self._last_io_counters.read_bytes) / 1024 / 1024
            io_write_delta = (current_io.write_bytes - self._last_io_counters.write_bytes) / 1024 / 1024
            net_sent_delta = (current_net.bytes_sent - self._last_net_counters.bytes_sent) / 1024 / 1024
            net_recv_delta = (current_net.bytes_recv - self._last_net_counters.bytes_recv) / 1024 / 1024
            
            snapshot = SystemSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_io_read_mb=io_read_delta,
                disk_io_write_mb=io_write_delta,
                network_sent_mb=net_sent_delta,
                network_recv_mb=net_recv_delta,
                active_threads=threading.active_count(),
                open_files=len(self.process.open_files())
            )
            
            self._system_snapshots.append(snapshot)
            
            # Update last counters
            self._last_io_counters = current_io
            self._last_net_counters = current_net
            
        except Exception as e:
            self.logger.error(f"Failed to take system snapshot: {e}")
    
    @contextmanager
    def profile_operation(self, 
                         operation_name: str,
                         custom_metrics: Optional[Dict[str, Any]] = None) -> ContextManager:
        """Context manager for profiling an operation.
        
        Args:
            operation_name: Name of the operation
            custom_metrics: Optional custom metrics
            
        Yields:
            Performance profiling context
        """
        operation_id = f"{operation_name}_{id(threading.current_thread())}_{time.time()}"
        
        # Pre-operation state
        start_time = datetime.now()
        start_memory = self._get_memory_usage()
        start_cpu_time = time.process_time()
        
        # Get call stack if detailed profiling is enabled
        call_stack = []
        if self.enable_detailed_profiling:
            call_stack = [
                f"{frame.filename}:{frame.lineno} in {frame.name}"
                for frame in traceback.extract_stack()[:-1]
            ]
        
        # Store operation state
        with self._profiling_lock:
            self._active_operations[operation_id] = {
                "operation_name": operation_name,
                "start_time": start_time,
                "start_memory": start_memory,
                "start_cpu_time": start_cpu_time,
                "custom_metrics": custom_metrics or {}
            }
        
        try:
            yield operation_id
            
        except Exception as e:
            # Record exception in custom metrics
            if operation_id in self._active_operations:
                self._active_operations[operation_id]["custom_metrics"]["exception"] = str(e)
            raise
            
        finally:
            # Post-operation measurements
            end_time = datetime.now()
            end_memory = self._get_memory_usage()
            end_cpu_time = time.process_time()
            
            with self._profiling_lock:
                if operation_id in self._active_operations:
                    op_data = self._active_operations[operation_id]
                    
                    # Calculate metrics
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                    memory_delta_mb = end_memory - op_data["start_memory"]
                    cpu_time_delta = end_cpu_time - op_data["start_cpu_time"]
                    cpu_usage_percent = (cpu_time_delta / max(duration_ms / 1000, 0.001)) * 100
                    
                    # Create performance metrics
                    metrics = PerformanceMetrics(
                        operation_name=operation_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration_ms,
                        cpu_usage_percent=cpu_usage_percent,
                        memory_usage_mb=end_memory,
                        memory_delta_mb=memory_delta_mb,
                        thread_id=str(threading.current_thread().ident),
                        call_stack=call_stack,
                        custom_metrics=op_data["custom_metrics"]
                    )
                    
                    # Store metrics
                    self._metrics_history.append(metrics)
                    self._update_operation_stats(metrics)
                    
                    # Clean up
                    del self._active_operations[operation_id]
    
    def profile_function(self,
                        operation_name: Optional[str] = None,
                        include_args: bool = False,
                        custom_metrics_func: Optional[Callable] = None):
        """Decorator for profiling function calls.
        
        Args:
            operation_name: Custom operation name
            include_args: Include function arguments in metrics
            custom_metrics_func: Function to generate custom metrics
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                custom_metrics = {}
                
                # Add function arguments if requested
                if include_args:
                    try:
                        custom_metrics["arg_count"] = len(args)
                        custom_metrics["kwarg_count"] = len(kwargs)
                        if args:
                            custom_metrics["first_arg_type"] = type(args[0]).__name__
                    except Exception:
                        pass
                
                # Get custom metrics if function provided
                if custom_metrics_func:
                    try:
                        custom_metrics.update(custom_metrics_func(*args, **kwargs))
                    except Exception as e:
                        custom_metrics["custom_metrics_error"] = str(e)
                
                with self.profile_operation(op_name, custom_metrics):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def profile_async_function(self,
                              operation_name: Optional[str] = None,
                              custom_metrics_func: Optional[Callable] = None):
        """Decorator for profiling async function calls.
        
        Args:
            operation_name: Custom operation name
            custom_metrics_func: Function to generate custom metrics
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                custom_metrics = {}
                
                if custom_metrics_func:
                    try:
                        custom_metrics.update(custom_metrics_func(*args, **kwargs))
                    except Exception as e:
                        custom_metrics["custom_metrics_error"] = str(e)
                
                with self.profile_operation(op_name, custom_metrics):
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            if self.process:
                return self.process.memory_info().rss / 1024 / 1024
            else:
                # Fallback to resource module
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0
    
    def _update_operation_stats(self, metrics: PerformanceMetrics):
        """Update running statistics for an operation.
        
        Args:
            metrics: Performance metrics to incorporate
        """
        op_name = metrics.operation_name
        
        if op_name not in self._operation_stats:
            self._operation_stats[op_name] = {
                "count": 0,
                "total_duration_ms": 0,
                "total_memory_delta_mb": 0,
                "max_duration_ms": 0,
                "min_duration_ms": float('inf'),
                "max_memory_delta_mb": 0,
                "min_memory_delta_mb": float('inf'),
                "error_count": 0,
                "last_execution": None
            }
        
        stats = self._operation_stats[op_name]
        stats["count"] += 1
        stats["total_duration_ms"] += metrics.duration_ms
        stats["total_memory_delta_mb"] += metrics.memory_delta_mb
        stats["max_duration_ms"] = max(stats["max_duration_ms"], metrics.duration_ms)
        stats["min_duration_ms"] = min(stats["min_duration_ms"], metrics.duration_ms)
        stats["max_memory_delta_mb"] = max(stats["max_memory_delta_mb"], metrics.memory_delta_mb)
        stats["min_memory_delta_mb"] = min(stats["min_memory_delta_mb"], metrics.memory_delta_mb)
        stats["last_execution"] = metrics.end_time
        
        # Count errors
        if "exception" in metrics.custom_metrics:
            stats["error_count"] += 1
        
        # Calculate averages
        stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
        stats["avg_memory_delta_mb"] = stats["total_memory_delta_mb"] / stats["count"]
        stats["error_rate"] = stats["error_count"] / stats["count"]
    
    def identify_bottlenecks(self, 
                           time_window_hours: int = 1,
                           min_samples: int = 10) -> List[BottleneckReport]:
        """Identify performance bottlenecks.
        
        Args:
            time_window_hours: Time window to analyze (hours)
            min_samples: Minimum number of samples required for analysis
            
        Returns:
            List of bottleneck reports
        """
        bottlenecks = []
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter recent metrics
        recent_metrics = [
            m for m in self._metrics_history
            if m.start_time >= cutoff_time
        ]
        
        # Group by operation
        operation_metrics = defaultdict(list)
        for metric in recent_metrics:
            operation_metrics[metric.operation_name].append(metric)
        
        # Analyze operations
        for op_name, metrics in operation_metrics.items():
            if len(metrics) < min_samples:
                continue
            
            bottlenecks.extend(self._analyze_operation_bottlenecks(op_name, metrics))
        
        # Analyze system-level bottlenecks
        bottlenecks.extend(self._analyze_system_bottlenecks())
        
        # Sort by impact score
        bottlenecks.sort(key=lambda b: b.impact_score, reverse=True)
        
        return bottlenecks
    
    def _analyze_operation_bottlenecks(self, 
                                     operation_name: str,
                                     metrics: List[PerformanceMetrics]) -> List[BottleneckReport]:
        """Analyze bottlenecks for a specific operation.
        
        Args:
            operation_name: Name of the operation
            metrics: List of performance metrics
            
        Returns:
            List of bottleneck reports
        """
        bottlenecks = []
        
        # Calculate statistics
        durations = [m.duration_ms for m in metrics]
        memory_deltas = [m.memory_delta_mb for m in metrics]
        cpu_usages = [m.cpu_usage_percent for m in metrics]
        
        mean_duration = np.mean(durations)
        p95_duration = np.percentile(durations, 95)
        std_duration = np.std(durations)
        
        mean_memory = np.mean(memory_deltas)
        max_memory = np.max(memory_deltas)
        
        mean_cpu = np.mean(cpu_usages)
        max_cpu = np.max(cpu_usages)
        
        # Check for slow operations
        if mean_duration > 1000:  # > 1 second
            severity = "critical" if mean_duration > 10000 else "high"
            impact_score = min(100, mean_duration / 100)
            
            bottlenecks.append(BottleneckReport(
                bottleneck_type="slow_operation",
                severity=severity,
                operation_name=operation_name,
                description=f"Operation has high average duration: {mean_duration:.1f}ms",
                impact_score=impact_score,
                recommendations=[
                    "Profile the operation to identify slow components",
                    "Consider caching or optimization",
                    "Check for inefficient algorithms or database queries"
                ],
                metrics={
                    "mean_duration_ms": mean_duration,
                    "p95_duration_ms": p95_duration,
                    "std_duration_ms": std_duration,
                    "sample_count": len(metrics)
                },
                timestamp=datetime.now()
            ))
        
        # Check for high variability
        if std_duration > mean_duration * 0.5 and mean_duration > 100:
            bottlenecks.append(BottleneckReport(
                bottleneck_type="high_variability",
                severity="medium",
                operation_name=operation_name,
                description=f"Operation has high duration variability: {std_duration:.1f}ms std",
                impact_score=std_duration / 10,
                recommendations=[
                    "Investigate what causes performance variability",
                    "Check for resource contention",
                    "Consider load balancing or rate limiting"
                ],
                metrics={
                    "mean_duration_ms": mean_duration,
                    "std_duration_ms": std_duration,
                    "coefficient_of_variation": std_duration / mean_duration
                },
                timestamp=datetime.now()
            ))
        
        # Check for memory leaks
        if mean_memory > 50:  # > 50MB average growth
            severity = "critical" if mean_memory > 200 else "high"
            
            bottlenecks.append(BottleneckReport(
                bottleneck_type="memory_leak",
                severity=severity,
                operation_name=operation_name,
                description=f"Operation shows high memory growth: {mean_memory:.1f}MB avg",
                impact_score=min(100, mean_memory / 2),
                recommendations=[
                    "Check for memory leaks in the operation",
                    "Ensure proper cleanup of resources",
                    "Consider implementing object pooling"
                ],
                metrics={
                    "mean_memory_delta_mb": mean_memory,
                    "max_memory_delta_mb": max_memory,
                    "sample_count": len(metrics)
                },
                timestamp=datetime.now()
            ))
        
        # Check for high CPU usage
        if mean_cpu > 80:
            bottlenecks.append(BottleneckReport(
                bottleneck_type="high_cpu_usage",
                severity="high",
                operation_name=operation_name,
                description=f"Operation has high CPU usage: {mean_cpu:.1f}% avg",
                impact_score=mean_cpu,
                recommendations=[
                    "Profile CPU usage to identify hot spots",
                    "Consider algorithmic optimizations",
                    "Implement parallel processing if applicable"
                ],
                metrics={
                    "mean_cpu_percent": mean_cpu,
                    "max_cpu_percent": max_cpu,
                    "sample_count": len(metrics)
                },
                timestamp=datetime.now()
            ))
        
        return bottlenecks
    
    def _analyze_system_bottlenecks(self) -> List[BottleneckReport]:
        """Analyze system-level bottlenecks.
        
        Returns:
            List of system bottleneck reports
        """
        bottlenecks = []
        
        if len(self._system_snapshots) < 10:
            return bottlenecks
        
        # Get recent snapshots
        recent_snapshots = list(self._system_snapshots)[-60:]  # Last hour
        
        # Calculate averages
        avg_cpu = np.mean([s.cpu_percent for s in recent_snapshots])
        avg_memory = np.mean([s.memory_percent for s in recent_snapshots])
        avg_threads = np.mean([s.active_threads for s in recent_snapshots])
        
        # High CPU usage
        if avg_cpu > 80:
            severity = "critical" if avg_cpu > 95 else "high"
            bottlenecks.append(BottleneckReport(
                bottleneck_type="system_high_cpu",
                severity=severity,
                operation_name="system",
                description=f"System CPU usage is high: {avg_cpu:.1f}%",
                impact_score=avg_cpu,
                recommendations=[
                    "Scale up CPU resources",
                    "Optimize CPU-intensive operations",
                    "Implement load balancing"
                ],
                metrics={"avg_cpu_percent": avg_cpu},
                timestamp=datetime.now()
            ))
        
        # High memory usage
        if avg_memory > 85:
            severity = "critical" if avg_memory > 95 else "high"
            bottlenecks.append(BottleneckReport(
                bottleneck_type="system_high_memory",
                severity=severity,
                operation_name="system",
                description=f"System memory usage is high: {avg_memory:.1f}%",
                impact_score=avg_memory,
                recommendations=[
                    "Scale up memory resources",
                    "Optimize memory usage in applications",
                    "Implement memory caching strategies"
                ],
                metrics={"avg_memory_percent": avg_memory},
                timestamp=datetime.now()
            ))
        
        # Too many threads
        if avg_threads > 100:
            bottlenecks.append(BottleneckReport(
                bottleneck_type="high_thread_count",
                severity="medium",
                operation_name="system",
                description=f"High number of active threads: {avg_threads:.0f}",
                impact_score=min(100, avg_threads / 2),
                recommendations=[
                    "Review thread pool configurations",
                    "Consider async programming patterns",
                    "Implement thread pooling"
                ],
                metrics={"avg_active_threads": avg_threads},
                timestamp=datetime.now()
            ))
        
        return bottlenecks
    
    def get_operation_summary(self, 
                            operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for operations.
        
        Args:
            operation_name: Specific operation name (None for all)
            
        Returns:
            Performance summary
        """
        if operation_name:
            if operation_name in self._operation_stats:
                return {operation_name: self._operation_stats[operation_name]}
            else:
                return {}
        
        # Return all operations sorted by total duration
        operations = dict(self._operation_stats)
        
        # Add derived metrics
        for op_name, stats in operations.items():
            if stats["count"] > 0:
                stats["total_time_seconds"] = stats["total_duration_ms"] / 1000
                stats["throughput_per_hour"] = stats["count"] / max(1, 
                    (datetime.now() - stats["last_execution"]).total_seconds() / 3600
                ) if stats["last_execution"] else 0
        
        return operations
    
    def get_slow_operations(self, 
                          limit: int = 10,
                          min_duration_ms: float = 100) -> List[Dict[str, Any]]:
        """Get slowest operations.
        
        Args:
            limit: Maximum number of operations to return
            min_duration_ms: Minimum duration threshold
            
        Returns:
            List of slow operations
        """
        slow_ops = []
        
        for op_name, stats in self._operation_stats.items():
            if stats["avg_duration_ms"] >= min_duration_ms:
                slow_ops.append({
                    "operation_name": op_name,
                    "avg_duration_ms": stats["avg_duration_ms"],
                    "max_duration_ms": stats["max_duration_ms"],
                    "count": stats["count"],
                    "total_time_seconds": stats["total_duration_ms"] / 1000,
                    "error_rate": stats.get("error_rate", 0)
                })
        
        # Sort by average duration
        slow_ops.sort(key=lambda x: x["avg_duration_ms"], reverse=True)
        
        return slow_ops[:limit]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics.
        
        Returns:
            System health information
        """
        if not self._system_snapshots:
            return {"status": "no_data"}
        
        latest = self._system_snapshots[-1]
        
        # Determine health status
        status = "healthy"
        if latest.cpu_percent > 90 or latest.memory_percent > 90:
            status = "critical"
        elif latest.cpu_percent > 75 or latest.memory_percent > 75:
            status = "warning"
        
        return {
            "status": status,
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "memory_available_mb": latest.memory_available_mb,
            "active_threads": latest.active_threads,
            "open_files": latest.open_files,
            "timestamp": latest.timestamp,
            "monitoring_active": self._monitoring_active,
            "metrics_collected": len(self._metrics_history),
            "operations_tracked": len(self._operation_stats)
        }
    
    def generate_performance_report(self, 
                                  time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Args:
            time_window_hours: Time window for the report
            
        Returns:
            Comprehensive performance report
        """
        bottlenecks = self.identify_bottlenecks(time_window_hours)
        operation_summary = self.get_operation_summary()
        slow_operations = self.get_slow_operations()
        system_health = self.get_system_health()
        
        # Calculate report statistics
        total_operations = sum(stats["count"] for stats in operation_summary.values())
        total_errors = sum(stats["error_count"] for stats in operation_summary.values())
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "time_window_hours": time_window_hours,
            "summary": {
                "total_operations": total_operations,
                "total_errors": total_errors,
                "overall_error_rate": total_errors / max(total_operations, 1),
                "operations_tracked": len(operation_summary),
                "bottlenecks_identified": len(bottlenecks)
            },
            "system_health": system_health,
            "bottlenecks": [b.to_dict() for b in bottlenecks],
            "slow_operations": slow_operations,
            "operation_summary": operation_summary
        }
    
    def clear_metrics(self):
        """Clear all stored metrics and statistics."""
        with self._profiling_lock:
            self._metrics_history.clear()
            self._operation_stats.clear()
            self._system_snapshots.clear()
        
        self.logger.info("Performance metrics cleared")


# Global profiler instance
_performance_profiler: Optional[PerformanceProfiler] = None


def initialize_profiler(max_metrics_history: int = 10000,
                       snapshot_interval: int = 60,
                       enable_detailed_profiling: bool = False) -> PerformanceProfiler:
    """Initialize global performance profiler.
    
    Args:
        max_metrics_history: Maximum metrics to keep
        snapshot_interval: System snapshot interval
        enable_detailed_profiling: Enable detailed profiling
        
    Returns:
        Initialized profiler
    """
    global _performance_profiler
    _performance_profiler = PerformanceProfiler(
        max_metrics_history=max_metrics_history,
        snapshot_interval=snapshot_interval,
        enable_detailed_profiling=enable_detailed_profiling
    )
    return _performance_profiler


def get_profiler() -> Optional[PerformanceProfiler]:
    """Get global profiler instance.
    
    Returns:
        Profiler instance or None
    """
    return _performance_profiler


# Convenient decorator functions
def profile_operation(operation_name: str, **kwargs):
    """Convenient decorator for profiling operations."""
    if _performance_profiler:
        return _performance_profiler.profile_function(operation_name, **kwargs)
    else:
        def decorator(func):
            return func
        return decorator


def profile_async_operation(operation_name: str, **kwargs):
    """Convenient decorator for profiling async operations."""
    if _performance_profiler:
        return _performance_profiler.profile_async_function(operation_name, **kwargs)
    else:
        def decorator(func):
            return func
        return decorator