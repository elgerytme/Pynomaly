"""Centralized performance metrics collection and analysis."""

import asyncio
import json
import logging
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
from functools import wraps

import numpy as np


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    operation_name: str
    execution_time: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_usage: float
    system_metrics: Optional[SystemMetrics]
    custom_metrics: Optional[Dict[str, Any]]
    context: Optional[Dict[str, Any]]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.system_metrics:
            data['system_metrics'] = self.system_metrics.to_dict()
        return data


class PerformanceProfiler:
    """Performance profiler for detailed metrics collection."""
    
    def __init__(self, name: str, collect_system_metrics: bool = True):
        """Initialize profiler.
        
        Args:
            name: Name of the operation being profiled
            collect_system_metrics: Whether to collect system metrics
        """
        self.name = name
        self.collect_system_metrics = collect_system_metrics
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: Optional[float] = None
        self.start_system_metrics: Optional[SystemMetrics] = None
        self.custom_metrics: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        
        # Get process for memory monitoring
        self.process = psutil.Process()
        
        # Threading for continuous monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._memory_samples: List[float] = []
        self._cpu_samples: List[float] = []
    
    def __enter__(self):
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.stop()
    
    def start(self):
        """Start profiling."""
        self.start_time = time.time()
        
        # Get initial memory
        memory_info = self.process.memory_info()
        self.start_memory = memory_info.rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
        # Get initial system metrics
        if self.collect_system_metrics:
            self.start_system_metrics = self._collect_system_metrics()
        
        # Start continuous monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop(self) -> PerformanceMetrics:
        """Stop profiling and return metrics."""
        if self.start_time is None:
            raise RuntimeError("Profiler not started")
        
        self.end_time = time.time()
        
        # Stop monitoring
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        # Calculate execution time
        execution_time = self.end_time - self.start_time
        
        # Get final memory
        memory_info = self.process.memory_info()
        final_memory = memory_info.rss / 1024 / 1024  # MB
        memory_delta = final_memory - self.start_memory
        
        # Calculate average CPU usage
        avg_cpu = np.mean(self._cpu_samples) if self._cpu_samples else 0.0
        
        # Get final system metrics
        final_system_metrics = None
        if self.collect_system_metrics and self.start_system_metrics:
            final_system_metrics = self._collect_system_metrics()
        
        return PerformanceMetrics(
            operation_name=self.name,
            execution_time=execution_time,
            memory_peak_mb=self.peak_memory,
            memory_delta_mb=memory_delta,
            cpu_usage=avg_cpu,
            system_metrics=final_system_metrics,
            custom_metrics=self.custom_metrics.copy(),
            context=self.context.copy(),
            timestamp=datetime.now()
        )
    
    def add_metric(self, name: str, value: Any):
        """Add custom metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.custom_metrics[name] = value
    
    def add_context(self, key: str, value: Any):
        """Add context information.
        
        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
    
    def _monitor_resources(self):
        """Monitor resources in background thread."""
        while self._monitoring:
            try:
                # Monitor memory
                memory_info = self.process.memory_info()
                current_memory = memory_info.rss / 1024 / 1024  # MB
                self._memory_samples.append(current_memory)
                
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                
                # Monitor CPU
                cpu_percent = self.process.cpu_percent()
                self._cpu_samples.append(cpu_percent)
                
                time.sleep(0.1)  # Sample every 100ms
                
            except Exception:
                # Process might have ended or other error
                break
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-wide metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / 1024 / 1024 if disk_io else 0
        disk_write_mb = disk_io.write_bytes / 1024 / 1024 if disk_io else 0
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_mb = net_io.bytes_sent / 1024 / 1024 if net_io else 0
        net_recv_mb = net_io.bytes_recv / 1024 / 1024 if net_io else 0
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_io_sent_mb=net_sent_mb,
            network_io_recv_mb=net_recv_mb,
            timestamp=datetime.now()
        )


class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize metrics collector.
        
        Args:
            storage_path: Path to store metrics data
        """
        self.storage_path = storage_path or Path("tests/performance/data/metrics")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._metrics_buffer: List[PerformanceMetrics] = []
        self._buffer_lock = threading.Lock()
        self._auto_flush_enabled = True
        self._flush_interval = 60  # seconds
        self._last_flush = time.time()
    
    def profile_operation(
        self,
        operation_name: str,
        collect_system_metrics: bool = True
    ) -> PerformanceProfiler:
        """Create a performance profiler for an operation.
        
        Args:
            operation_name: Name of the operation
            collect_system_metrics: Whether to collect system metrics
            
        Returns:
            Performance profiler context manager
        """
        return PerformanceProfiler(operation_name, collect_system_metrics)
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics.
        
        Args:
            metrics: Performance metrics to record
        """
        with self._buffer_lock:
            self._metrics_buffer.append(metrics)
        
        # Auto-flush if needed
        if self._auto_flush_enabled and time.time() - self._last_flush > self._flush_interval:
            self.flush_metrics()
    
    def flush_metrics(self):
        """Flush buffered metrics to storage."""
        with self._buffer_lock:
            if not self._metrics_buffer:
                return
            
            metrics_to_save = self._metrics_buffer.copy()
            self._metrics_buffer.clear()
        
        # Save to file
        timestamp = datetime.now()
        filename = f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.storage_path / filename
        
        try:
            metrics_data = {
                "timestamp": timestamp.isoformat(),
                "metrics_count": len(metrics_to_save),
                "metrics": [metrics.to_dict() for metrics in metrics_to_save]
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self._last_flush = time.time()
            self.logger.info(f"Flushed {len(metrics_to_save)} metrics to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error flushing metrics: {e}")
            # Put metrics back in buffer
            with self._buffer_lock:
                self._metrics_buffer.extend(metrics_to_save)
    
    def get_metrics_history(
        self,
        operation_name: Optional[str] = None,
        hours: int = 24
    ) -> List[PerformanceMetrics]:
        """Get metrics history for analysis.
        
        Args:
            operation_name: Filter by operation name
            hours: Number of hours to look back
            
        Returns:
            List of performance metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        metrics_list = []
        
        # Read from all metric files
        for metrics_file in self.storage_path.glob("metrics_*.json"):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                for metric_data in data.get("metrics", []):
                    # Parse timestamp
                    metric_timestamp = datetime.fromisoformat(metric_data["timestamp"])
                    
                    if metric_timestamp >= cutoff_time:
                        # Filter by operation name if specified
                        if operation_name is None or metric_data["operation_name"] == operation_name:
                            # Reconstruct metrics object
                            metrics = PerformanceMetrics(
                                operation_name=metric_data["operation_name"],
                                execution_time=metric_data["execution_time"],
                                memory_peak_mb=metric_data["memory_peak_mb"],
                                memory_delta_mb=metric_data["memory_delta_mb"],
                                cpu_usage=metric_data["cpu_usage"],
                                system_metrics=None,  # Simplified for now
                                custom_metrics=metric_data.get("custom_metrics"),
                                context=metric_data.get("context"),
                                timestamp=metric_timestamp
                            )
                            metrics_list.append(metrics)
                            
            except Exception as e:
                self.logger.error(f"Error reading metrics file {metrics_file}: {e}")
                continue
        
        # Sort by timestamp
        metrics_list.sort(key=lambda x: x.timestamp)
        
        return metrics_list
    
    def get_operation_statistics(
        self,
        operation_name: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with operation statistics
        """
        metrics_list = self.get_metrics_history(operation_name, hours)
        
        if not metrics_list:
            return {"error": "No metrics found for operation"}
        
        # Extract values
        execution_times = [m.execution_time for m in metrics_list]
        memory_peaks = [m.memory_peak_mb for m in metrics_list]
        memory_deltas = [m.memory_delta_mb for m in metrics_list]
        cpu_usages = [m.cpu_usage for m in metrics_list]
        
        return {
            "operation_name": operation_name,
            "sample_count": len(metrics_list),
            "time_range_hours": hours,
            "execution_time": {
                "mean": np.mean(execution_times),
                "median": np.median(execution_times),
                "std": np.std(execution_times),
                "min": np.min(execution_times),
                "max": np.max(execution_times),
                "p95": np.percentile(execution_times, 95),
                "p99": np.percentile(execution_times, 99)
            },
            "memory_peak": {
                "mean": np.mean(memory_peaks),
                "median": np.median(memory_peaks),
                "max": np.max(memory_peaks)
            },
            "memory_delta": {
                "mean": np.mean(memory_deltas),
                "median": np.median(memory_deltas),
                "max": np.max(memory_deltas)
            },
            "cpu_usage": {
                "mean": np.mean(cpu_usages),
                "median": np.median(cpu_usages),
                "max": np.max(cpu_usages)
            }
        }
    
    def cleanup_old_metrics(self, days: int = 30):
        """Clean up old metrics files.
        
        Args:
            days: Remove files older than this many days
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        removed_count = 0
        
        for metrics_file in self.storage_path.glob("metrics_*.json"):
            try:
                # Parse timestamp from filename
                timestamp_str = metrics_file.stem.replace("metrics_", "")
                file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                if file_time < cutoff_time:
                    metrics_file.unlink()
                    removed_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {metrics_file}: {e}")
                continue
        
        self.logger.info(f"Cleaned up {removed_count} old metrics files")


# Decorator for easy performance monitoring
def monitor_performance(
    operation_name: Optional[str] = None,
    collector: Optional[MetricsCollector] = None,
    collect_system_metrics: bool = True
):
    """Decorator to monitor function performance.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        collector: Metrics collector instance
        collect_system_metrics: Whether to collect system metrics
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal operation_name
            if operation_name is None:
                operation_name = f"{func.__module__}.{func.__name__}"
            
            # Use default collector if none provided
            if collector is None:
                _collector = MetricsCollector()
            else:
                _collector = collector
            
            # Profile the function
            with _collector.profile_operation(operation_name, collect_system_metrics) as profiler:
                # Add function context
                profiler.add_context("function", func.__name__)
                profiler.add_context("module", func.__module__)
                profiler.add_context("args_count", len(args))
                profiler.add_context("kwargs_count", len(kwargs))
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Add result context
                if hasattr(result, "__len__"):
                    profiler.add_metric("result_size", len(result))
                
                return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal operation_name
            if operation_name is None:
                operation_name = f"{func.__module__}.{func.__name__}"
            
            # Use default collector if none provided
            if collector is None:
                _collector = MetricsCollector()
            else:
                _collector = collector
            
            # Profile the async function
            with _collector.profile_operation(operation_name, collect_system_metrics) as profiler:
                # Add function context
                profiler.add_context("function", func.__name__)
                profiler.add_context("module", func.__module__)
                profiler.add_context("args_count", len(args))
                profiler.add_context("kwargs_count", len(kwargs))
                profiler.add_context("async", True)
                
                # Execute async function
                result = await func(*args, **kwargs)
                
                # Add result context
                if hasattr(result, "__len__"):
                    profiler.add_metric("result_size", len(result))
                
                return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


# Global metrics collector instance
default_collector = MetricsCollector()


@contextmanager
def performance_context(operation_name: str, **context):
    """Context manager for performance monitoring.
    
    Args:
        operation_name: Name of the operation
        **context: Additional context to add
    """
    with default_collector.profile_operation(operation_name) as profiler:
        for key, value in context.items():
            profiler.add_context(key, value)
        yield profiler