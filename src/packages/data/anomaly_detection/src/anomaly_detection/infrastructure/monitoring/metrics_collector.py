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
        
        self.record_metric(name, value, tags, "count")\n    \n    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:\n        \"\"\"Set a gauge metric.\"\"\"\n        with self._lock:\n            gauge_key = f\"{name}_{tags}\" if tags else name\n            self._gauges[gauge_key] = value\n        \n        self.record_metric(name, value, tags, \"gauge\")\n    \n    def record_timing(\n        self,\n        name: str,\n        duration_ms: float,\n        tags: Optional[Dict[str, str]] = None\n    ) -> None:\n        \"\"\"Record a timing metric.\"\"\"\n        self.record_metric(name, duration_ms, tags, \"milliseconds\")\n    \n    def start_operation(self, operation_id: str) -> str:\n        \"\"\"Start timing an operation.\n        \n        Args:\n            operation_id: Unique identifier for the operation\n            \n        Returns:\n            Operation ID for use with end_operation\n        \"\"\"\n        start_time = time.perf_counter()\n        \n        with self._lock:\n            self._active_operations[operation_id] = start_time\n        \n        logger.debug(\"Operation started\", operation_id=operation_id)\n        return operation_id\n    \n    def end_operation(\n        self,\n        operation_id: str,\n        success: bool = True,\n        error_type: Optional[str] = None,\n        context: Optional[Dict[str, Any]] = None\n    ) -> float:\n        \"\"\"End timing an operation and record performance metrics.\n        \n        Args:\n            operation_id: Operation identifier from start_operation\n            success: Whether the operation succeeded\n            error_type: Type of error if operation failed\n            context: Additional context about the operation\n            \n        Returns:\n            Duration in milliseconds\n        \"\"\"\n        end_time = time.perf_counter()\n        \n        with self._lock:\n            start_time = self._active_operations.pop(operation_id, end_time)\n        \n        duration_ms = (end_time - start_time) * 1000\n        \n        # Record performance metrics\n        perf_metric = PerformanceMetrics(\n            operation=operation_id,\n            duration_ms=duration_ms,\n            timestamp=datetime.utcnow(),\n            success=success,\n            error_type=error_type,\n            context=context or {}\n        )\n        \n        with self._lock:\n            self._performance_metrics.append(perf_metric)\n        \n        # Record timing metric\n        tags = {\"success\": str(success).lower()}\n        if error_type:\n            tags[\"error_type\"] = error_type\n        \n        self.record_timing(f\"{operation_id}_duration\", duration_ms, tags)\n        \n        logger.debug(\"Operation completed\", \n                    operation_id=operation_id,\n                    duration_ms=duration_ms,\n                    success=success)\n        \n        return duration_ms\n    \n    def record_model_metrics(\n        self,\n        model_id: str,\n        algorithm: str,\n        operation: str,\n        duration_ms: float,\n        success: bool,\n        samples_processed: Optional[int] = None,\n        anomalies_detected: Optional[int] = None,\n        accuracy: Optional[float] = None,\n        precision: Optional[float] = None,\n        recall: Optional[float] = None,\n        f1_score: Optional[float] = None\n    ) -> None:\n        \"\"\"Record model-specific metrics.\"\"\"\n        model_metric = ModelMetrics(\n            model_id=model_id,\n            algorithm=algorithm,\n            operation=operation,\n            duration_ms=duration_ms,\n            timestamp=datetime.utcnow(),\n            success=success,\n            samples_processed=samples_processed,\n            anomalies_detected=anomalies_detected,\n            accuracy=accuracy,\n            precision=precision,\n            recall=recall,\n            f1_score=f1_score\n        )\n        \n        with self._lock:\n            self._model_metrics.append(model_metric)\n        \n        # Record individual metrics\n        tags = {\n            \"model_id\": model_id[:8],  # Truncate for readability\n            \"algorithm\": algorithm,\n            \"operation\": operation,\n            \"success\": str(success).lower()\n        }\n        \n        self.record_timing(\"model_operation_duration\", duration_ms, tags)\n        \n        if samples_processed is not None:\n            self.record_metric(\"model_samples_processed\", samples_processed, tags, \"count\")\n        \n        if anomalies_detected is not None:\n            self.record_metric(\"model_anomalies_detected\", anomalies_detected, tags, \"count\")\n        \n        for metric_name, value in [\n            (\"accuracy\", accuracy),\n            (\"precision\", precision), \n            (\"recall\", recall),\n            (\"f1_score\", f1_score)\n        ]:\n            if value is not None:\n                self.record_metric(f\"model_{metric_name}\", value, tags, \"ratio\")\n        \n        logger.info(\"Model metrics recorded\", \n                   model_id=model_id,\n                   algorithm=algorithm,\n                   operation=operation,\n                   duration_ms=duration_ms,\n                   success=success)\n    \n    def get_summary_stats(self) -> Dict[str, Any]:\n        \"\"\"Get summary statistics for all collected metrics.\"\"\"\n        with self._lock:\n            current_time = datetime.utcnow()\n            \n            # Calculate aggregated statistics\n            summary = {\n                \"collection_time\": current_time.isoformat(),\n                \"total_metrics\": len(self._metrics),\n                \"total_performance_metrics\": len(self._performance_metrics),\n                \"total_model_metrics\": len(self._model_metrics),\n                \"active_operations\": len(self._active_operations),\n                \"counters\": dict(self._counters),\n                \"gauges\": dict(self._gauges),\n                \"timing_stats\": {}\n            }\n            \n            # Calculate timing statistics\n            for name, times in self._timers.items():\n                if times:\n                    summary[\"timing_stats\"][name] = {\n                        \"count\": len(times),\n                        \"min\": min(times),\n                        \"max\": max(times),\n                        \"avg\": sum(times) / len(times),\n                        \"latest\": times[-1]\n                    }\n            \n            # Recent operation success rates\n            recent_cutoff = current_time - timedelta(minutes=10)\n            recent_ops = [p for p in self._performance_metrics if p.timestamp > recent_cutoff]\n            \n            if recent_ops:\n                success_count = sum(1 for p in recent_ops if p.success)\n                summary[\"recent_success_rate\"] = success_count / len(recent_ops)\n                summary[\"recent_operations\"] = len(recent_ops)\n            else:\n                summary[\"recent_success_rate\"] = None\n                summary[\"recent_operations\"] = 0\n        \n        return summary\n    \n    def get_performance_metrics(\n        self,\n        operation: Optional[str] = None,\n        since: Optional[datetime] = None,\n        limit: Optional[int] = None\n    ) -> List[PerformanceMetrics]:\n        \"\"\"Get performance metrics with optional filtering.\"\"\"\n        with self._lock:\n            metrics = list(self._performance_metrics)\n        \n        # Apply filters\n        if operation:\n            metrics = [m for m in metrics if m.operation == operation]\n        \n        if since:\n            metrics = [m for m in metrics if m.timestamp > since]\n        \n        # Sort by timestamp (newest first)\n        metrics.sort(key=lambda m: m.timestamp, reverse=True)\n        \n        if limit:\n            metrics = metrics[:limit]\n        \n        return metrics\n    \n    def get_model_metrics(\n        self,\n        model_id: Optional[str] = None,\n        algorithm: Optional[str] = None,\n        operation: Optional[str] = None,\n        since: Optional[datetime] = None,\n        limit: Optional[int] = None\n    ) -> List[ModelMetrics]:\n        \"\"\"Get model metrics with optional filtering.\"\"\"\n        with self._lock:\n            metrics = list(self._model_metrics)\n        \n        # Apply filters\n        if model_id:\n            metrics = [m for m in metrics if m.model_id == model_id]\n        \n        if algorithm:\n            metrics = [m for m in metrics if m.algorithm == algorithm]\n        \n        if operation:\n            metrics = [m for m in metrics if m.operation == operation]\n        \n        if since:\n            metrics = [m for m in metrics if m.timestamp > since]\n        \n        # Sort by timestamp (newest first)\n        metrics.sort(key=lambda m: m.timestamp, reverse=True)\n        \n        if limit:\n            metrics = metrics[:limit]\n        \n        return metrics\n    \n    def cleanup_old_metrics(self) -> int:\n        \"\"\"Remove metrics older than retention period.\n        \n        Returns:\n            Number of metrics removed\n        \"\"\"\n        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)\n        removed_count = 0\n        \n        with self._lock:\n            # Clean metrics\n            original_len = len(self._metrics)\n            self._metrics = deque(\n                (m for m in self._metrics if m.timestamp > cutoff_time),\n                maxlen=self.buffer_size\n            )\n            removed_count += original_len - len(self._metrics)\n            \n            # Clean performance metrics\n            original_len = len(self._performance_metrics)\n            self._performance_metrics = deque(\n                (m for m in self._performance_metrics if m.timestamp > cutoff_time),\n                maxlen=self.buffer_size\n            )\n            removed_count += original_len - len(self._performance_metrics)\n            \n            # Clean model metrics\n            original_len = len(self._model_metrics)\n            self._model_metrics = deque(\n                (m for m in self._model_metrics if m.timestamp > cutoff_time),\n                maxlen=self.buffer_size\n            )\n            removed_count += original_len - len(self._model_metrics)\n        \n        if removed_count > 0:\n            logger.info(\"Cleaned up old metrics\", removed_count=removed_count)\n        \n        return removed_count\n    \n    def clear_all_metrics(self) -> None:\n        \"\"\"Clear all collected metrics.\"\"\"\n        with self._lock:\n            self._metrics.clear()\n            self._performance_metrics.clear()\n            self._model_metrics.clear()\n            self._counters.clear()\n            self._timers.clear()\n            self._gauges.clear()\n            self._active_operations.clear()\n        \n        logger.info(\"All metrics cleared\")\n\n\n# Global metrics collector instance\n_global_metrics_collector: Optional[MetricsCollector] = None\n\n\ndef get_metrics_collector() -> MetricsCollector:\n    \"\"\"Get the global metrics collector instance.\"\"\"\n    global _global_metrics_collector\n    \n    if _global_metrics_collector is None:\n        _global_metrics_collector = MetricsCollector()\n    \n    return _global_metrics_collector"