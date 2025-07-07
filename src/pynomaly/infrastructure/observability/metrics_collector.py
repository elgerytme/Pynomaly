"""Comprehensive metrics collection for Pynomaly."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import psutil
from prometheus_client import (
    CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram,
    Summary, generate_latest, multiprocess, start_http_server
)


@dataclass
class MetricDefinition:
    """Definition for a custom metric."""
    name: str
    metric_type: str  # counter, gauge, histogram, summary
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


class MetricsCollector:
    """Comprehensive metrics collector for Pynomaly."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector.
        
        Args:
            registry: Prometheus registry to use (default: global registry)
        """
        self.registry = registry or CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._setup_default_metrics()
    
    def _setup_default_metrics(self) -> None:
        """Set up default application metrics."""
        # API Request metrics
        self.request_count = Counter(
            'pynomaly_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'pynomaly_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Detection metrics
        self.detection_count = Counter(
            'pynomaly_detections_total',
            'Total anomaly detections performed',
            ['detector_type', 'dataset_name'],
            registry=self.registry
        )
        
        self.detection_duration = Histogram(
            'pynomaly_detection_duration_seconds',
            'Anomaly detection duration in seconds',
            ['detector_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.anomalies_found = Histogram(
            'pynomaly_anomalies_found',
            'Number of anomalies found per detection',
            ['detector_type'],
            buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000],
            registry=self.registry
        )
        
        # Training metrics
        self.training_count = Counter(
            'pynomaly_model_training_total',
            'Total model training sessions',
            ['detector_type', 'algorithm'],
            registry=self.registry
        )
        
        self.training_duration = Histogram(
            'pynomaly_training_duration_seconds',
            'Model training duration in seconds',
            ['detector_type', 'algorithm'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0],
            registry=self.registry
        )
        
        # System metrics
        self.active_users = Gauge(
            'pynomaly_active_users',
            'Current number of active users',
            registry=self.registry
        )
        
        self.active_detectors = Gauge(
            'pynomaly_active_detectors',
            'Current number of active detectors',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'pynomaly_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'pynomaly_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        # Database metrics
        self.db_connections = Gauge(
            'pynomaly_db_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'pynomaly_db_query_duration_seconds',
            'Database query duration in seconds',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'pynomaly_errors_total',
            'Total errors by type',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'pynomaly_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'pynomaly_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Streaming metrics
        self.stream_messages_processed = Counter(
            'pynomaly_stream_messages_processed_total',
            'Total stream messages processed',
            ['stream_id'],
            registry=self.registry
        )
        
        self.stream_processing_latency = Histogram(
            'pynomaly_stream_processing_latency_seconds',
            'Stream message processing latency',
            ['stream_id'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """Record HTTP request metrics.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
        """
        self.request_count.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_detection(self, detector_type: str, dataset_name: str, duration: float, anomalies_count: int) -> None:
        """Record anomaly detection metrics.
        
        Args:
            detector_type: Type of detector used
            dataset_name: Name of the dataset
            duration: Detection duration in seconds
            anomalies_count: Number of anomalies found
        """
        self.detection_count.labels(detector_type=detector_type, dataset_name=dataset_name).inc()
        self.detection_duration.labels(detector_type=detector_type).observe(duration)
        self.anomalies_found.labels(detector_type=detector_type).observe(anomalies_count)
    
    def record_training(self, detector_type: str, algorithm: str, duration: float) -> None:
        """Record model training metrics.
        
        Args:
            detector_type: Type of detector
            algorithm: Algorithm used
            duration: Training duration in seconds
        """
        self.training_count.labels(detector_type=detector_type, algorithm=algorithm).inc()
        self.training_duration.labels(detector_type=detector_type, algorithm=algorithm).observe(duration)
    
    def record_error(self, error_type: str, component: str) -> None:
        """Record error metrics.
        
        Args:
            error_type: Type of error
            component: Component where error occurred
        """
        self.error_count.labels(error_type=error_type, component=component).inc()
    
    def record_cache_hit(self, cache_type: str) -> None:
        """Record cache hit.
        
        Args:
            cache_type: Type of cache
        """
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str) -> None:
        """Record cache miss.
        
        Args:
            cache_type: Type of cache
        """
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_stream_message(self, stream_id: str, processing_latency: float) -> None:
        """Record stream message processing.
        
        Args:
            stream_id: Stream identifier
            processing_latency: Processing latency in seconds
        """
        self.stream_messages_processed.labels(stream_id=stream_id).inc()
        self.stream_processing_latency.labels(stream_id=stream_id).observe(processing_latency)
    
    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        # Memory usage
        process = psutil.Process()
        self.memory_usage.set(process.memory_info().rss)
        
        # CPU usage
        self.cpu_usage.set(psutil.cpu_percent())
    
    def set_active_users(self, count: int) -> None:
        """Set active users count.
        
        Args:
            count: Number of active users
        """
        self.active_users.set(count)
    
    def set_active_detectors(self, count: int) -> None:
        """Set active detectors count.
        
        Args:
            count: Number of active detectors
        """
        self.active_detectors.set(count)
    
    def set_db_connections(self, count: int) -> None:
        """Set database connections count.
        
        Args:
            count: Number of active database connections
        """
        self.db_connections.set(count)
    
    @contextmanager
    def time_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to time operations.
        
        Args:
            operation_name: Name of the operation being timed
            labels: Optional labels for the metric
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if operation_name.startswith('db_'):
                self.db_query_duration.labels(operation=operation_name).observe(duration)
    
    def create_custom_metric(self, definition: MetricDefinition) -> Any:
        """Create a custom metric.
        
        Args:
            definition: Metric definition
            
        Returns:
            Created metric instance
        """
        if definition.metric_type == 'counter':
            metric = Counter(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == 'gauge':
            metric = Gauge(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == 'histogram':
            metric = Histogram(
                definition.name,
                definition.description,
                definition.labels,
                buckets=definition.buckets,
                registry=self.registry
            )
        elif definition.metric_type == 'summary':
            metric = Summary(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        else:
            raise ValueError(f"Unknown metric type: {definition.metric_type}")
        
        self._metrics[definition.name] = metric
        return metric
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Metric instance or None if not found
        """
        return self._metrics.get(name)


class PrometheusExporter:
    """Prometheus metrics exporter."""
    
    def __init__(self, collector: MetricsCollector, port: int = 8000):
        """Initialize Prometheus exporter.
        
        Args:
            collector: Metrics collector instance
            port: Port to expose metrics on
        """
        self.collector = collector
        self.port = port
        self._server = None
    
    def start_server(self) -> None:
        """Start Prometheus metrics server."""
        self._server = start_http_server(self.port, registry=self.collector.registry)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.collector.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get content type for metrics endpoint.
        
        Returns:
            Content type string
        """
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance.
    
    Returns:
        Global metrics collector
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def setup_metrics_collection(port: int = 8000) -> tuple[MetricsCollector, PrometheusExporter]:
    """Set up metrics collection with Prometheus exporter.
    
    Args:
        port: Port to expose metrics on
        
    Returns:
        Tuple of (metrics collector, prometheus exporter)
    """
    collector = get_metrics_collector()
    exporter = PrometheusExporter(collector, port)
    return collector, exporter