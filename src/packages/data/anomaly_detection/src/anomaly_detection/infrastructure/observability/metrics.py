"""Advanced metrics collection with Prometheus for anomaly detection."""

from __future__ import annotations

import logging
import time
import threading
import functools
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass
from collections import defaultdict
import os

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info, Enum,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, multiprocess, values
    )
    from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST as OPENMETRICS_CONTENT_TYPE
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for metrics collection."""
    service_name: str = "anomaly_detection"
    service_version: str = "1.0.0"
    environment: str = "production"
    enable_system_metrics: bool = True
    enable_ml_metrics: bool = True
    enable_business_metrics: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    collection_interval: float = 15.0
    custom_labels: Optional[Dict[str, str]] = None


class MetricsCollector:
    """Advanced metrics collector with Prometheus integration."""
    
    def __init__(self, config: MetricConfig):
        """Initialize metrics collector.
        
        Args:
            config: Metrics configuration
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics disabled")
            self.enabled = False
            return
        
        self.config = config
        self.enabled = True
        self.registry = CollectorRegistry()
        
        # Common labels
        self.common_labels = {
            "service": config.service_name,
            "version": config.service_version,
            "environment": config.environment,
            **(config.custom_labels or {})
        }
        
        # Initialize metrics
        self._init_system_metrics()
        self._init_application_metrics()
        self._init_ml_metrics()
        self._init_business_metrics()
        
        # Background collection
        self._collection_thread = None
        self._stop_collection = threading.Event()
        
        logger.info(f"Metrics collector initialized for {config.service_name}")
    
    def _init_system_metrics(self) -> None:
        """Initialize system-level metrics."""
        if not self.config.enable_system_metrics or not PSUTIL_AVAILABLE:
            return
        
        # CPU metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            labelnames=list(self.common_labels.keys()),
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            labelnames=list(self.common_labels.keys()) + ['type'],
            registry=self.registry
        )
        
        self.memory_usage_percent = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            labelnames=list(self.common_labels.keys()),
            registry=self.registry
        )
        
        # Disk metrics
        self.disk_usage = Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            labelnames=list(self.common_labels.keys()) + ['device', 'type'],
            registry=self.registry
        )
        
        self.disk_io = Counter(
            'system_disk_io_bytes_total',
            'System disk I/O in bytes',
            labelnames=list(self.common_labels.keys()) + ['device', 'direction'],
            registry=self.registry
        )
        
        # Network metrics
        self.network_io = Counter(
            'system_network_io_bytes_total',
            'System network I/O in bytes',
            labelnames=list(self.common_labels.keys()) + ['interface', 'direction'],
            registry=self.registry
        )
        
        # Process metrics
        self.process_count = Gauge(
            'system_process_count',
            'Number of system processes',
            labelnames=list(self.common_labels.keys()),
            registry=self.registry
        )
    
    def _init_application_metrics(self) -> None:
        """Initialize application-level metrics."""
        # HTTP request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            labelnames=list(self.common_labels.keys()) + ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            labelnames=list(self.common_labels.keys()) + ['method', 'endpoint'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        self.http_request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            labelnames=list(self.common_labels.keys()) + ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            labelnames=list(self.common_labels.keys()) + ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Application state metrics
        self.app_info = Info(
            'app_info',
            'Application information',
            labelnames=list(self.common_labels.keys()),
            registry=self.registry
        )
        
        self.app_uptime = Gauge(
            'app_uptime_seconds',
            'Application uptime in seconds',
            labelnames=list(self.common_labels.keys()),
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'app_active_connections',
            'Number of active connections',
            labelnames=list(self.common_labels.keys()) + ['type'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'app_errors_total',
            'Total application errors',
            labelnames=list(self.common_labels.keys()) + ['type', 'severity'],
            registry=self.registry
        )
        
        # Task/job metrics
        self.tasks_total = Counter(
            'app_tasks_total',
            'Total tasks processed',
            labelnames=list(self.common_labels.keys()) + ['type', 'status'],
            registry=self.registry
        )
        
        self.task_duration = Histogram(
            'app_task_duration_seconds',
            'Task processing duration in seconds',
            labelnames=list(self.common_labels.keys()) + ['type'],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0),
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'app_queue_size',
            'Size of processing queues',
            labelnames=list(self.common_labels.keys()) + ['queue'],
            registry=self.registry
        )
    
    def _init_ml_metrics(self) -> None:
        """Initialize ML-specific metrics."""
        if not self.config.enable_ml_metrics:
            return
        
        # Model training metrics
        self.model_training_total = Counter(
            'ml_model_training_total',
            'Total model training sessions',
            labelnames=list(self.common_labels.keys()) + ['algorithm', 'status'],
            registry=self.registry
        )
        
        self.model_training_duration = Histogram(
            'ml_model_training_duration_seconds',
            'Model training duration in seconds',
            labelnames=list(self.common_labels.keys()) + ['algorithm'],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0),
            registry=self.registry
        )
        
        # Model inference metrics
        self.model_predictions_total = Counter(
            'ml_model_predictions_total',
            'Total model predictions',
            labelnames=list(self.common_labels.keys()) + ['model', 'algorithm'],
            registry=self.registry
        )
        
        self.model_inference_duration = Histogram(
            'ml_model_inference_duration_seconds',
            'Model inference duration in seconds',
            labelnames=list(self.common_labels.keys()) + ['model', 'algorithm'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
            registry=self.registry
        )
        
        self.model_batch_size = Histogram(
            'ml_model_batch_size',
            'Model inference batch size',
            labelnames=list(self.common_labels.keys()) + ['model'],
            buckets=(1, 10, 50, 100, 500, 1000, 5000, 10000),
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Model accuracy score',
            labelnames=list(self.common_labels.keys()) + ['model', 'metric'],
            registry=self.registry
        )
        
        self.model_memory_usage = Gauge(
            'ml_model_memory_usage_bytes',
            'Model memory usage in bytes',
            labelnames=list(self.common_labels.keys()) + ['model'],
            registry=self.registry
        )
        
        # Feature metrics
        self.feature_importance = Gauge(
            'ml_feature_importance',
            'Feature importance scores',
            labelnames=list(self.common_labels.keys()) + ['model', 'feature'],
            registry=self.registry
        )
        
        self.data_drift_score = Gauge(
            'ml_data_drift_score',
            'Data drift detection score',
            labelnames=list(self.common_labels.keys()) + ['feature'],
            registry=self.registry
        )
    
    def _init_business_metrics(self) -> None:
        """Initialize business-specific metrics."""
        if not self.config.enable_business_metrics:
            return
        
        # Anomaly detection metrics
        self.anomalies_detected_total = Counter(
            'business_anomalies_detected_total',
            'Total anomalies detected',
            labelnames=list(self.common_labels.keys()) + ['severity', 'type'],
            registry=self.registry
        )
        
        self.anomaly_score = Histogram(
            'business_anomaly_score',
            'Anomaly scores distribution',
            labelnames=list(self.common_labels.keys()) + ['algorithm'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry
        )
        
        self.false_positive_rate = Gauge(
            'business_false_positive_rate',
            'False positive rate',
            labelnames=list(self.common_labels.keys()) + ['model'],
            registry=self.registry
        )
        
        self.true_positive_rate = Gauge(
            'business_true_positive_rate',
            'True positive rate (sensitivity)',
            labelnames=list(self.common_labels.keys()) + ['model'],
            registry=self.registry
        )
        
        # Data quality metrics
        self.data_quality_score = Gauge(
            'business_data_quality_score',
            'Data quality score',
            labelnames=list(self.common_labels.keys()) + ['dataset', 'dimension'],
            registry=self.registry
        )
        
        self.missing_data_ratio = Gauge(
            'business_missing_data_ratio',
            'Ratio of missing data',
            labelnames=list(self.common_labels.keys()) + ['dataset', 'feature'],
            registry=self.registry
        )
        
        # Processing metrics
        self.data_processed_total = Counter(
            'business_data_processed_total',
            'Total data processed',
            labelnames=list(self.common_labels.keys()) + ['source', 'type'],
            registry=self.registry
        )
        
        self.processing_latency = Histogram(
            'business_processing_latency_seconds',
            'End-to-end processing latency',
            labelnames=list(self.common_labels.keys()) + ['pipeline'],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
            registry=self.registry
        )
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None
    ) -> None:
        """Record HTTP request metrics.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
            request_size: Request size in bytes
            response_size: Response size in bytes
        """
        if not self.enabled:
            return
        
        labels = {**self.common_labels, 'method': method, 'endpoint': endpoint}
        
        # Request count
        self.http_requests_total.labels(
            **labels, status=str(status_code)
        ).inc()
        
        # Request duration
        self.http_request_duration.labels(**labels).observe(duration)
        
        # Request/response sizes
        if request_size is not None:
            self.http_request_size.labels(**labels).observe(request_size)
        
        if response_size is not None:
            self.http_response_size.labels(**labels).observe(response_size)
    
    def record_model_training(
        self,
        algorithm: str,
        duration: float,
        status: str = "success",
        **metadata: Any
    ) -> None:
        """Record model training metrics.
        
        Args:
            algorithm: ML algorithm used
            duration: Training duration in seconds
            status: Training status
            **metadata: Additional metadata
        """
        if not self.enabled or not self.config.enable_ml_metrics:
            return
        
        labels = {**self.common_labels, 'algorithm': algorithm}
        
        self.model_training_total.labels(**labels, status=status).inc()
        self.model_training_duration.labels(**labels).observe(duration)
    
    def record_model_prediction(
        self,
        model_name: str,
        algorithm: str,
        duration: float,
        batch_size: int = 1
    ) -> None:
        """Record model prediction metrics.
        
        Args:
            model_name: Name of the model
            algorithm: Algorithm used
            duration: Inference duration in seconds
            batch_size: Batch size
        """
        if not self.enabled or not self.config.enable_ml_metrics:
            return
        
        labels = {**self.common_labels, 'model': model_name, 'algorithm': algorithm}
        model_labels = {**self.common_labels, 'model': model_name}
        
        self.model_predictions_total.labels(**labels).inc(batch_size)
        self.model_inference_duration.labels(**labels).observe(duration)
        self.model_batch_size.labels(**model_labels).observe(batch_size)
    
    def record_anomaly_detection(
        self,
        algorithm: str,
        anomaly_count: int,
        severity: str = "medium",
        anomaly_type: str = "unknown",
        scores: Optional[List[float]] = None
    ) -> None:
        """Record anomaly detection metrics.
        
        Args:
            algorithm: Algorithm used
            anomaly_count: Number of anomalies detected
            severity: Anomaly severity
            anomaly_type: Type of anomaly
            scores: Anomaly scores
        """
        if not self.enabled or not self.config.enable_business_metrics:
            return
        
        # Record anomaly count
        self.anomalies_detected_total.labels(
            **self.common_labels,
            severity=severity,
            type=anomaly_type
        ).inc(anomaly_count)
        
        # Record anomaly scores distribution
        if scores:
            score_labels = {**self.common_labels, 'algorithm': algorithm}
            for score in scores:
                self.anomaly_score.labels(**score_labels).observe(score)
    
    def record_data_quality(
        self,
        dataset: str,
        dimension: str,
        score: float,
        feature_missing_ratios: Optional[Dict[str, float]] = None
    ) -> None:
        """Record data quality metrics.
        
        Args:
            dataset: Dataset name
            dimension: Quality dimension (completeness, accuracy, etc.)
            score: Quality score (0-1)
            feature_missing_ratios: Missing data ratios per feature
        """
        if not self.enabled or not self.config.enable_business_metrics:
            return
        
        # Overall quality score
        self.data_quality_score.labels(
            **self.common_labels,
            dataset=dataset,
            dimension=dimension
        ).set(score)
        
        # Missing data ratios
        if feature_missing_ratios:
            for feature, ratio in feature_missing_ratios.items():
                self.missing_data_ratio.labels(
                    **self.common_labels,
                    dataset=dataset,
                    feature=feature
                ).set(ratio)
    
    def record_model_performance(
        self,
        model_name: str,
        metrics: Dict[str, float],
        memory_usage: Optional[int] = None
    ) -> None:
        """Record model performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics (accuracy, precision, etc.)
            memory_usage: Memory usage in bytes
        """
        if not self.enabled or not self.config.enable_ml_metrics:
            return
        
        # Performance metrics
        for metric_name, value in metrics.items():
            self.model_accuracy.labels(
                **self.common_labels,
                model=model_name,
                metric=metric_name
            ).set(value)
        
        # Memory usage
        if memory_usage is not None:
            self.model_memory_usage.labels(
                **self.common_labels,
                model=model_name
            ).set(memory_usage)
    
    def record_task_execution(
        self,
        task_type: str,
        duration: float,
        status: str = "success"
    ) -> None:
        """Record task execution metrics.
        
        Args:
            task_type: Type of task
            duration: Execution duration in seconds
            status: Task status
        """
        if not self.enabled:
            return
        
        labels = {**self.common_labels, 'type': task_type}
        
        self.tasks_total.labels(**labels, status=status).inc()
        self.task_duration.labels(**labels).observe(duration)
    
    def record_error(
        self,
        error_type: str,
        severity: str = "error"
    ) -> None:
        """Record application error.
        
        Args:
            error_type: Type of error
            severity: Error severity
        """
        if not self.enabled:
            return
        
        self.errors_total.labels(
            **self.common_labels,
            type=error_type,
            severity=severity
        ).inc()
    
    def start_background_collection(self) -> None:
        """Start background metrics collection."""
        if not self.enabled or self._collection_thread:
            return
        
        self._collection_thread = threading.Thread(
            target=self._collect_system_metrics_loop,
            daemon=True
        )
        self._collection_thread.start()
        
        logger.info("Background metrics collection started")
    
    def stop_background_collection(self) -> None:
        """Stop background metrics collection."""
        if not self._collection_thread:
            return
        
        self._stop_collection.set()
        self._collection_thread.join(timeout=5.0)
        self._collection_thread = None
        
        logger.info("Background metrics collection stopped")
    
    def _collect_system_metrics_loop(self) -> None:
        """Background loop for collecting system metrics."""
        start_time = time.time()
        
        while not self._stop_collection.wait(self.config.collection_interval):
            try:
                self._collect_system_metrics()
                
                # Update uptime
                uptime = time.time() - start_time
                self.app_uptime.labels(**self.common_labels).set(uptime)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        if not self.config.enable_system_metrics or not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.labels(**self.common_labels).set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.labels(**self.common_labels, type="used").set(memory.used)
            self.memory_usage.labels(**self.common_labels, type="available").set(memory.available)
            self.memory_usage.labels(**self.common_labels, type="total").set(memory.total)
            self.memory_usage_percent.labels(**self.common_labels).set(memory.percent)
            
            # Process count
            process_count = len(psutil.pids())
            self.process_count.labels(**self.common_labels).set(process_count)
            
        except Exception as e:
            logger.debug(f"Error collecting system metrics: {e}")
    
    def get_metrics(self, openmetrics_format: bool = False) -> bytes:
        """Get current metrics in Prometheus format.
        
        Args:
            openmetrics_format: Whether to use OpenMetrics format
            
        Returns:
            Metrics data as bytes
        """
        if not self.enabled:
            return b""
        
        if multiprocess.MultiProcessCollector:
            # Handle multiprocess mode
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            return generate_latest(registry)
        else:
            return generate_latest(self.registry)
    
    def get_content_type(self, openmetrics_format: bool = False) -> str:
        """Get content type for metrics.
        
        Args:
            openmetrics_format: Whether to use OpenMetrics format
            
        Returns:
            Content type string
        """
        return OPENMETRICS_CONTENT_TYPE if openmetrics_format else CONTENT_TYPE_LATEST
    
    def start_http_server(self, port: Optional[int] = None) -> None:
        """Start HTTP server for metrics endpoint.
        
        Args:
            port: Port to listen on
        """
        if not self.enabled:
            return
        
        port = port or self.config.metrics_port
        
        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Metrics HTTP server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics HTTP server: {e}")


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def initialize_metrics(config: MetricConfig) -> MetricsCollector:
    """Initialize global metrics collection.
    
    Args:
        config: Metrics configuration
        
    Returns:
        Metrics collector instance
    """
    global _global_collector
    _global_collector = MetricsCollector(config)
    return _global_collector


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get global metrics collector.
    
    Returns:
        Global metrics collector or None if not initialized
    """
    return _global_collector


# Convenience decorators
def monitor_duration(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to monitor function execution duration.
    
    Args:
        metric_name: Custom metric name
        labels: Additional labels
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _global_collector or not _global_collector.enabled:
                return func(*args, **kwargs)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record successful execution
                _global_collector.record_task_execution(
                    task_type=metric_name or func.__name__,
                    duration=duration,
                    status="success"
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failed execution
                _global_collector.record_task_execution(
                    task_type=metric_name or func.__name__,
                    duration=duration,
                    status="error"
                )
                
                # Record error
                _global_collector.record_error(
                    error_type=type(e).__name__,
                    severity="error"
                )
                
                raise
        
        return wrapper
    return decorator


def monitor_ml_inference(model_name: str, algorithm: str):
    """Decorator to monitor ML model inference.
    
    Args:
        model_name: Name of the model
        algorithm: Algorithm used
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _global_collector or not _global_collector.enabled:
                return func(*args, **kwargs)
            
            start_time = time.time()
            
            # Try to determine batch size from args
            batch_size = 1
            if args and hasattr(args[0], '__len__'):
                try:
                    batch_size = len(args[0])
                except:
                    pass
            
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            _global_collector.record_model_prediction(
                model_name=model_name,
                algorithm=algorithm,
                duration=duration,
                batch_size=batch_size
            )
            
            return result
        
        return wrapper
    return decorator