"""Metrics collection and reporting for anomaly detection operations."""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """A single metric measurement."""
    
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass  
class PerformanceMetrics:
    """Performance metrics for operations."""
    
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    error_type: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Metrics for model operations."""
    
    model_id: str
    algorithm: str
    operation: str  # train, predict, save, load
    duration_ms: float
    timestamp: datetime
    success: bool
    samples_processed: Optional[int] = None
    anomalies_detected: Optional[int] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class MetricsCollector:
    """Centralized metrics collection and aggregation."""
    
    def __init__(self, buffer_size: int = 10000, retention_hours: int = 24):
        """Initialize metrics collector.
        
        Args:
            buffer_size: Maximum number of metrics to keep in memory
            retention_hours: How long to retain metrics
        """
        self.buffer_size = buffer_size
        self.retention_hours = retention_hours
        
        # Thread-safe collections
        self._lock = threading.RLock()
        self._metrics: deque[MetricPoint] = deque(maxlen=buffer_size)
        self._performance_metrics: deque[PerformanceMetrics] = deque(maxlen=buffer_size)
        self._model_metrics: deque[ModelMetrics] = deque(maxlen=buffer_size)
        
        # Aggregated counters
        self._counters: Dict[str, int] = defaultdict(int)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        
        # Operation tracking
        self._active_operations: Dict[str, float] = {}  # operation_id -> start_time
        
        logger.info("Metrics collector initialized", 
                   buffer_size=buffer_size, 
                   retention_hours=retention_hours)
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ) -> None:
        """Record a metric point.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for grouping/filtering
            unit: Optional unit of measurement
        """
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            unit=unit
        )
        
        with self._lock:
            self._metrics.append(metric)
            
            # Update aggregated metrics
            self._counters[name] += 1
            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(value)
            
            # Keep timer lists bounded
            if len(self._timers[name]) > 1000:
                self._timers[name] = self._timers[name][-500:]
        
        logger.debug("Metric recorded", 
                    metric_name=name,
                    value=value,
                    tags=tags,
                    unit=unit)
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            counter_key = f"{name}_{tags}" if tags else name
            self._counters[counter_key] += value
        
        self.record_metric(name, value, tags, "count")
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        with self._lock:
            gauge_key = f"{name}_{tags}" if tags else name
            self._gauges[gauge_key] = value
        
        self.record_metric(name, value, tags, "gauge")
    
    def record_timing(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timing metric."""
        self.record_metric(name, duration_ms, tags, "milliseconds")
    
    def start_operation(self, operation_id: str) -> str:
        """Start timing an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            
        Returns:
            Operation ID for use with end_operation
        """
        start_time = time.perf_counter()
        
        with self._lock:
            self._active_operations[operation_id] = start_time
        
        logger.debug("Operation started", operation_id=operation_id)
        return operation_id
    
    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        error_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """End timing an operation and record performance metrics."""
        end_time = time.perf_counter()
        
        with self._lock:
            start_time = self._active_operations.pop(operation_id, end_time)
        
        duration_ms = (end_time - start_time) * 1000
        
        # Record performance metrics
        perf_metric = PerformanceMetrics(
            operation=operation_id,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
            success=success,
            error_type=error_type,
            context=context or {}
        )
        
        with self._lock:
            self._performance_metrics.append(perf_metric)
        
        logger.debug("Operation completed", 
                    operation_id=operation_id,
                    duration_ms=duration_ms,
                    success=success)
        
        return duration_ms
    
    def record_model_metric(
        self,
        model_id: str,
        algorithm: str,
        operation: str,
        duration_ms: float,
        success: bool,
        samples_processed: int | None = None,
        anomalies_detected: int | None = None,
        **performance_metrics: float
    ) -> None:
        """Record model operation metrics."""
        model_metric = ModelMetrics(
            model_id=model_id,
            algorithm=algorithm,
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
            success=success,
            samples_processed=samples_processed,
            anomalies_detected=anomalies_detected,
            accuracy=performance_metrics.get('accuracy'),
            precision=performance_metrics.get('precision'),
            recall=performance_metrics.get('recall'),
            f1_score=performance_metrics.get('f1_score')
        )
        
        with self._lock:
            self._model_metrics.append(model_metric)
        
        logger.debug("Model metric recorded",
                    model_id=model_id,
                    algorithm=algorithm,
                    operation=operation,
                    duration_ms=duration_ms)
    
    def record_model_metrics(
        self,
        model_id: str,
        algorithm: str,
        operation: str,
        duration_ms: float,
        success: bool,
        samples_processed: int | None = None,
        anomalies_detected: int | None = None,
        **performance_metrics: float
    ) -> None:
        """Record model operation metrics (alias for record_model_metric for backward compatibility)."""
        self.record_model_metric(
            model_id=model_id,
            algorithm=algorithm,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            samples_processed=samples_processed,
            anomalies_detected=anomalies_detected,
            **performance_metrics
        )
    
    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for all metrics."""
        with self._lock:
            return {
                "total_metrics": len(self._metrics),
                "total_performance_metrics": len(self._performance_metrics),
                "total_model_metrics": len(self._model_metrics),
                "active_operations": len(self._active_operations),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "buffer_size": self.buffer_size,
                "retention_hours": self.retention_hours
            }
    
    def get_model_metrics(
        self,
        since: datetime | None = None,
        algorithm: str | None = None
    ) -> list[ModelMetrics]:
        """Get model metrics with optional filtering."""
        with self._lock:
            metrics = list(self._model_metrics)
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        if algorithm:
            metrics = [m for m in metrics if m.algorithm == algorithm]
        
        return metrics
    
    def get_performance_metrics(
        self,
        since: datetime | None = None,
        operation: str | None = None
    ) -> list[PerformanceMetrics]:
        """Get performance metrics with optional filtering."""
        with self._lock:
            metrics = list(self._performance_metrics)
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        
        return metrics
    
    def cleanup_old_metrics(self) -> dict[str, int]:
        """Clean up old metrics beyond retention period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            # Clean up metrics
            initial_metrics = len(self._metrics)
            self._metrics = deque(
                (m for m in self._metrics if m.timestamp >= cutoff_time),
                maxlen=self.buffer_size
            )
            metrics_removed = initial_metrics - len(self._metrics)
            
            # Clean up performance metrics
            initial_perf = len(self._performance_metrics)
            self._performance_metrics = deque(
                (m for m in self._performance_metrics if m.timestamp >= cutoff_time),
                maxlen=self.buffer_size
            )
            perf_removed = initial_perf - len(self._performance_metrics)
            
            # Clean up model metrics
            initial_model = len(self._model_metrics)
            self._model_metrics = deque(
                (m for m in self._model_metrics if m.timestamp >= cutoff_time),
                maxlen=self.buffer_size
            )
            model_removed = initial_model - len(self._model_metrics)
            
            # Clean up timer data
            for timer_name, timer_data in self._timers.items():
                if len(timer_data) > 500:  # Keep recent 500
                    self._timers[timer_name] = timer_data[-500:]
        
        cleanup_stats = {
            'metrics_removed': metrics_removed,
            'performance_removed': perf_removed,
            'model_removed': model_removed,
            'total_removed': metrics_removed + perf_removed + model_removed
        }
        
        logger.info("Metrics cleanup completed", **cleanup_stats)
        return cleanup_stats


# Global metrics collector instance
_global_metrics_collector: Optional['MetricsCollector'] = None


def get_metrics_collector() -> 'MetricsCollector':
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    
    return _global_metrics_collector


def cleanup_metrics_collector() -> dict[str, int]:
    """Clean up the global metrics collector."""
    global _global_metrics_collector
    
    if _global_metrics_collector is not None:
        return _global_metrics_collector.cleanup_old_metrics()
    
    return {'total_removed': 0}