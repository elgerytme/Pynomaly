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
        
        logger.debug("Operation completed", 
                    operation_id=operation_id,
                    duration_ms=duration_ms,
                    success=success)
        
        return duration_ms


# Global metrics collector instance
_global_metrics_collector: Optional['MetricsCollector'] = None


def get_metrics_collector() -> 'MetricsCollector':
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    
    return _global_metrics_collector