# Monitoring and Observability Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🟢 [Basic Usage](README.md) > 📈 Monitoring

---


This comprehensive guide covers monitoring, observability, and alerting strategies for Pynomaly deployments using OpenTelemetry, Prometheus, structured logging, and modern observability practices.

## Table of Contents

1. [Observability Overview](#observability-overview)
2. [OpenTelemetry Integration](#opentelemetry-integration)
3. [Prometheus Metrics](#prometheus-metrics)
4. [Structured Logging](#structured-logging)
5. [Distributed Tracing](#distributed-tracing)
6. [Alerting and Notifications](#alerting-and-notifications)
7. [Dashboards and Visualization](#dashboards-and-visualization)
8. [Health Checks and SLIs](#health-checks-and-slis)

## Observability Overview

Pynomaly implements comprehensive observability following the three pillars of observability:

- **Metrics**: Quantitative measurements (response times, error rates, throughput)
- **Logs**: Structured event records with context and metadata
- **Traces**: Request flow tracking across distributed components

### Observability Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pynomaly Application                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   Metrics   │ │   Traces    │ │        Logs             │ │
│  │ (Prometheus)│ │(OpenTelemetry)│ │    (Structured)        │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────┬─────────────┬─────────────────┬───────────────┘
              │             │                 │
              ▼             ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Prometheus    │ │   Jaeger        │ │  ELK Stack      │
│   (Metrics)     │ │  (Tracing)      │ │ (Log Analytics) │
└─────────────────┘ └─────────────────┘ └─────────────────┘
              │             │                 │
              └─────────────┼─────────────────┘
                            ▼
              ┌─────────────────────────────────┐
              │           Grafana               │
              │    (Unified Dashboards)         │
              └─────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────────┐
              │         AlertManager           │
              │    (Alerting & Notifications)   │
              └─────────────────────────────────┘
```

## OpenTelemetry Integration

### Comprehensive Telemetry Configuration

```python
# infrastructure/monitoring/telemetry.py
import os
import logging
from typing import Dict, Any, Optional
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class TelemetryManager:
    """Comprehensive telemetry management for Pynomaly."""
    
    def __init__(self, 
                 service_name: str = "pynomaly",
                 service_version: str = "1.0.0",
                 environment: str = "production"):
        
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        
        # Telemetry providers
        self.tracer_provider = None
        self.meter_provider = None
        self.tracer = None
        self.meter = None
        
        # Custom metrics
        self.metrics = {}
        
        # Configuration
        self.config = self._load_config()
        
        # Initialize telemetry
        self._initialize_telemetry()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load telemetry configuration from environment."""
        return {
            "jaeger_endpoint": os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
            "prometheus_endpoint": os.getenv("PROMETHEUS_ENDPOINT", "localhost:8000"),
            "enable_tracing": os.getenv("ENABLE_TRACING", "true").lower() == "true",
            "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
            "trace_sample_rate": float(os.getenv("TRACE_SAMPLE_RATE", "1.0")),
            "export_interval": int(os.getenv("METRIC_EXPORT_INTERVAL", "30")),
        }
    
    def _initialize_telemetry(self):
        """Initialize OpenTelemetry providers and exporters."""
        
        # Create resource with service information
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": self.service_version,
            "deployment.environment": self.environment,
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
        })
        
        # Initialize tracing
        if self.config["enable_tracing"]:
            self._initialize_tracing(resource)
        
        # Initialize metrics
        if self.config["enable_metrics"]:
            self._initialize_metrics(resource)
        
        # Set up automatic instrumentation
        self._setup_auto_instrumentation()
        
        logger.info(f"Telemetry initialized for {self.service_name}")
    
    def _initialize_tracing(self, resource: Resource):
        """Initialize distributed tracing."""
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(
            resource=resource,
            sampler=trace.sampling.TraceIdRatioBasedSampler(
                rate=self.config["trace_sample_rate"]
            )
        )
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            endpoint=self.config["jaeger_endpoint"],
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(
            jaeger_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            export_timeout_millis=30000,
        )
        
        self.tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(
            instrumenting_module_name=__name__,
            instrumenting_library_version=self.service_version
        )
        
        # Set up trace propagation
        set_global_textmap(B3MultiFormat())
        
        logger.info("Distributed tracing initialized")
    
    def _initialize_metrics(self, resource: Resource):
        """Initialize metrics collection."""
        
        # Create Prometheus metric reader
        prometheus_reader = PrometheusMetricReader()
        
        # Create meter provider
        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[prometheus_reader]
        )
        
        # Set global meter provider
        metrics.set_meter_provider(self.meter_provider)
        
        # Get meter
        self.meter = metrics.get_meter(
            name=self.service_name,
            version=self.service_version
        )
        
        # Initialize custom metrics
        self._initialize_custom_metrics()
        
        logger.info("Metrics collection initialized")
    
    def _initialize_custom_metrics(self):
        """Initialize Pynomaly-specific metrics."""
        
        # HTTP request metrics
        self.metrics["http_requests_total"] = self.meter.create_counter(
            name="http_requests_total",
            description="Total number of HTTP requests",
            unit="1"
        )
        
        self.metrics["http_request_duration"] = self.meter.create_histogram(
            name="http_request_duration_seconds",
            description="HTTP request duration in seconds",
            unit="s"
        )
        
        self.metrics["http_request_size"] = self.meter.create_histogram(
            name="http_request_size_bytes",
            description="HTTP request size in bytes",
            unit="By"
        )
        
        # Anomaly detection metrics
        self.metrics["anomaly_detections_total"] = self.meter.create_counter(
            name="anomaly_detections_total",
            description="Total number of anomaly detections performed",
            unit="1"
        )
        
        self.metrics["anomaly_detection_duration"] = self.meter.create_histogram(
            name="anomaly_detection_duration_seconds",
            description="Time taken for anomaly detection",
            unit="s"
        )
        
        self.metrics["anomalies_found_total"] = self.meter.create_counter(
            name="anomalies_found_total",
            description="Total number of anomalies detected",
            unit="1"
        )
        
        self.metrics["detector_training_duration"] = self.meter.create_histogram(
            name="detector_training_duration_seconds",
            description="Time taken for detector training",
            unit="s"
        )
        
        # Dataset metrics
        self.metrics["datasets_processed_total"] = self.meter.create_counter(
            name="datasets_processed_total",
            description="Total number of datasets processed",
            unit="1"
        )
        
        self.metrics["dataset_size"] = self.meter.create_histogram(
            name="dataset_size_samples",
            description="Number of samples in processed datasets",
            unit="1"
        )
        
        # System metrics
        self.metrics["active_detectors"] = self.meter.create_up_down_counter(
            name="active_detectors_count",
            description="Number of currently active detectors",
            unit="1"
        )
        
        self.metrics["memory_usage"] = self.meter.create_gauge(
            name="memory_usage_bytes",
            description="Current memory usage in bytes",
            unit="By"
        )
        
        # Error metrics
        self.metrics["errors_total"] = self.meter.create_counter(
            name="errors_total",
            description="Total number of errors",
            unit="1"
        )
        
        # Cache metrics
        self.metrics["cache_hits_total"] = self.meter.create_counter(
            name="cache_hits_total",
            description="Total number of cache hits",
            unit="1"
        )
        
        self.metrics["cache_misses_total"] = self.meter.create_counter(
            name="cache_misses_total",
            description="Total number of cache misses",
            unit="1"
        )
    
    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation for common libraries."""
        
        # FastAPI instrumentation
        FastAPIInstrumentor().instrument()
        
        # SQLAlchemy instrumentation
        SQLAlchemyInstrumentor().instrument()
        
        # Redis instrumentation
        RedisInstrumentor().instrument()
        
        # Requests instrumentation
        RequestsInstrumentor().instrument()
        
        # Asyncio instrumentation
        AsyncioInstrumentor().instrument()
        
        logger.info("Auto-instrumentation enabled")
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, 
                           duration: float, request_size: int = 0):
        """Record HTTP request metrics."""
        
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code),
            "status_class": f"{status_code // 100}xx"
        }
        
        self.metrics["http_requests_total"].add(1, labels)
        self.metrics["http_request_duration"].record(duration, labels)
        
        if request_size > 0:
            self.metrics["http_request_size"].record(request_size, labels)
    
    def record_anomaly_detection(self, algorithm: str, dataset_size: int, 
                                anomalies_found: int, duration: float, 
                                success: bool = True):
        """Record anomaly detection metrics."""
        
        labels = {
            "algorithm": algorithm,
            "success": str(success).lower()
        }
        
        self.metrics["anomaly_detections_total"].add(1, labels)
        self.metrics["anomaly_detection_duration"].record(duration, labels)
        self.metrics["dataset_size"].record(dataset_size, labels)
        
        if success:
            self.metrics["anomalies_found_total"].add(anomalies_found, labels)
    
    def record_training(self, algorithm: str, dataset_size: int, 
                       duration: float, success: bool = True):
        """Record detector training metrics."""
        
        labels = {
            "algorithm": algorithm,
            "success": str(success).lower()
        }
        
        self.metrics["detector_training_duration"].record(duration, labels)
        
        if success:
            self.metrics["datasets_processed_total"].add(1, labels)
    
    def record_error(self, error_type: str, component: str, severity: str = "error"):
        """Record error metrics."""
        
        labels = {
            "error_type": error_type,
            "component": component,
            "severity": severity
        }
        
        self.metrics["errors_total"].add(1, labels)
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        labels = {"cache_type": cache_type}
        self.metrics["cache_hits_total"].add(1, labels)
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        labels = {"cache_type": cache_type}
        self.metrics["cache_misses_total"].add(1, labels)
    
    def update_active_detectors(self, count: int):
        """Update active detectors count."""
        self.metrics["active_detectors"].add(count)
    
    def update_memory_usage(self, bytes_used: int):
        """Update memory usage metric."""
        self.metrics["memory_usage"].set(bytes_used)
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, **attributes):
        """Context manager for tracing operations."""
        
        if not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
            
            try:
                yield span
            except Exception as e:
                # Record exception in span
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            else:
                span.set_status(trace.Status(trace.StatusCode.OK))
    
    def shutdown(self):
        """Shutdown telemetry providers."""
        
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        
        if self.meter_provider:
            self.meter_provider.shutdown()
        
        logger.info("Telemetry shutdown completed")


# Usage example for monitoring decorator
def monitor_performance(operation_name: str, algorithm: str = None):
    """Decorator to monitor function performance."""
    
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            import time
            
            start_time = time.time()
            
            # Get telemetry manager from container
            try:
                from pynomaly.infrastructure.config import create_container
                container = create_container()
                telemetry = container.telemetry_manager()
            except:
                telemetry = None
            
            if telemetry:
                async with telemetry.trace_operation(
                    operation_name, 
                    function=func.__name__,
                    algorithm=algorithm
                ) as span:
                    try:
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time
                        
                        # Record metrics based on operation type
                        if "detection" in operation_name.lower():
                            dataset_size = kwargs.get('dataset_size', 0)
                            anomalies_found = getattr(result, 'anomaly_count', 0)
                            telemetry.record_anomaly_detection(
                                algorithm or "unknown",
                                dataset_size,
                                anomalies_found,
                                duration,
                                success=True
                            )
                        elif "training" in operation_name.lower():
                            dataset_size = kwargs.get('dataset_size', 0)
                            telemetry.record_training(
                                algorithm or "unknown",
                                dataset_size,
                                duration,
                                success=True
                            )
                        
                        if span:
                            span.set_attribute("duration_seconds", duration)
                            span.set_attribute("success", True)
                        
                        return result
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        
                        if telemetry:
                            telemetry.record_error(
                                error_type=type(e).__name__,
                                component=func.__name__
                            )
                        
                        if span:
                            span.set_attribute("duration_seconds", duration)
                            span.set_attribute("success", False)
                            span.set_attribute("error", str(e))
                        
                        raise
            else:
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            import time
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"{operation_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{operation_name} failed after {duration:.3f}s: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global telemetry instance
telemetry_manager: Optional[TelemetryManager] = None

def get_telemetry_manager() -> Optional[TelemetryManager]:
    """Get global telemetry manager instance."""
    return telemetry_manager

def initialize_telemetry(service_name: str = "pynomaly", 
                        service_version: str = "1.0.0",
                        environment: str = "production") -> TelemetryManager:
    """Initialize global telemetry manager."""
    global telemetry_manager
    
    telemetry_manager = TelemetryManager(
        service_name=service_name,
        service_version=service_version,
        environment=environment
    )
    
    return telemetry_manager
```

## Prometheus Metrics

### Custom Metrics Collection

```python
# infrastructure/monitoring/prometheus_metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
from typing import Dict, Optional
import time
import psutil
import threading
import asyncio

class PrometheusMetrics:
    """Prometheus metrics collection for Pynomaly."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._initialize_metrics()
        self._start_system_metrics_collection()
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # Application metrics
        self.app_info = Info(
            'pynomaly_app_info',
            'Application information',
            registry=self.registry
        )
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'pynomaly_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'pynomaly_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.http_request_size_bytes = Histogram(
            'pynomaly_http_request_size_bytes',
            'HTTP request size',
            ['method', 'endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000],
            registry=self.registry
        )
        
        # Anomaly detection metrics
        self.anomaly_detections_total = Counter(
            'pynomaly_anomaly_detections_total',
            'Total anomaly detections performed',
            ['algorithm', 'status'],
            registry=self.registry
        )
        
        self.anomaly_detection_duration_seconds = Histogram(
            'pynomaly_anomaly_detection_duration_seconds',
            'Anomaly detection duration',
            ['algorithm'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )
        
        self.anomalies_found_total = Counter(
            'pynomaly_anomalies_found_total',
            'Total anomalies detected',
            ['algorithm', 'severity'],
            registry=self.registry
        )
        
        self.detector_accuracy = Gauge(
            'pynomaly_detector_accuracy',
            'Detector accuracy score',
            ['algorithm', 'dataset'],
            registry=self.registry
        )
        
        # Training metrics
        self.training_duration_seconds = Histogram(
            'pynomaly_training_duration_seconds',
            'Model training duration',
            ['algorithm'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0],
            registry=self.registry
        )
        
        self.training_samples_total = Counter(
            'pynomaly_training_samples_total',
            'Total training samples processed',
            ['algorithm'],
            registry=self.registry
        )
        
        # Dataset metrics
        self.datasets_loaded_total = Counter(
            'pynomaly_datasets_loaded_total',
            'Total datasets loaded',
            ['format', 'status'],
            registry=self.registry
        )
        
        self.dataset_size_samples = Histogram(
            'pynomaly_dataset_size_samples',
            'Dataset size in samples',
            ['format'],
            buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
            registry=self.registry
        )
        
        self.dataset_loading_duration_seconds = Histogram(
            'pynomaly_dataset_loading_duration_seconds',
            'Dataset loading duration',
            ['format'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage_bytes = Gauge(
            'pynomaly_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'pynomaly_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.disk_usage_bytes = Gauge(
            'pynomaly_disk_usage_bytes',
            'Disk usage in bytes',
            ['path'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations_total = Counter(
            'pynomaly_cache_operations_total',
            'Total cache operations',
            ['type', 'result'],
            registry=self.registry
        )
        
        self.cache_size_items = Gauge(
            'pynomaly_cache_size_items',
            'Number of items in cache',
            ['cache_type'],
            registry=self.registry
        )
        
        # Database metrics
        self.database_connections_active = Gauge(
            'pynomaly_database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.database_query_duration_seconds = Histogram(
            'pynomaly_database_query_duration_seconds',
            'Database query duration',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'pynomaly_errors_total',
            'Total errors',
            ['component', 'error_type', 'severity'],
            registry=self.registry
        )
        
        # API rate limiting metrics
        self.rate_limit_exceeded_total = Counter(
            'pynomaly_rate_limit_exceeded_total',
            'Total rate limit violations',
            ['client_id', 'endpoint'],
            registry=self.registry
        )
        
        self.active_sessions = Gauge(
            'pynomaly_active_sessions',
            'Number of active user sessions',
            registry=self.registry
        )
    
    def _start_system_metrics_collection(self):
        """Start background thread for system metrics collection."""
        
        def collect_system_metrics():
            while True:
                try:
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.memory_usage_bytes.labels(type='used').set(memory.used)
                    self.memory_usage_bytes.labels(type='available').set(memory.available)
                    self.memory_usage_bytes.labels(type='total').set(memory.total)
                    
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage_percent.set(cpu_percent)
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.disk_usage_bytes.labels(path='/').set(disk.used)
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start background thread
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, 
                           duration: float, request_size: int = 0):
        """Record HTTP request metrics."""
        
        self.http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration_seconds.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        if request_size > 0:
            self.http_request_size_bytes.labels(
                method=method, 
                endpoint=endpoint
            ).observe(request_size)
    
    def record_anomaly_detection(self, algorithm: str, duration: float, 
                                anomalies_found: int, success: bool = True,
                                severity_counts: Dict[str, int] = None):
        """Record anomaly detection metrics."""
        
        status = "success" if success else "failure"
        
        self.anomaly_detections_total.labels(
            algorithm=algorithm, 
            status=status
        ).inc()
        
        if success:
            self.anomaly_detection_duration_seconds.labels(
                algorithm=algorithm
            ).observe(duration)
            
            # Record anomalies by severity
            if severity_counts:
                for severity, count in severity_counts.items():
                    self.anomalies_found_total.labels(
                        algorithm=algorithm,
                        severity=severity
                    ).inc(count)
            else:
                self.anomalies_found_total.labels(
                    algorithm=algorithm,
                    severity="unknown"
                ).inc(anomalies_found)
    
    def record_training(self, algorithm: str, duration: float, 
                       sample_count: int, success: bool = True):
        """Record training metrics."""
        
        if success:
            self.training_duration_seconds.labels(
                algorithm=algorithm
            ).observe(duration)
            
            self.training_samples_total.labels(
                algorithm=algorithm
            ).inc(sample_count)
    
    def record_dataset_loading(self, format_type: str, duration: float, 
                              sample_count: int, success: bool = True):
        """Record dataset loading metrics."""
        
        status = "success" if success else "failure"
        
        self.datasets_loaded_total.labels(
            format=format_type,
            status=status
        ).inc()
        
        if success:
            self.dataset_loading_duration_seconds.labels(
                format=format_type
            ).observe(duration)
            
            self.dataset_size_samples.labels(
                format=format_type
            ).observe(sample_count)
    
    def record_cache_operation(self, operation_type: str, cache_type: str, hit: bool):
        """Record cache operation metrics."""
        
        result = "hit" if hit else "miss"
        
        self.cache_operations_total.labels(
            type=operation_type,
            result=result
        ).inc()
    
    def update_cache_size(self, cache_type: str, size: int):
        """Update cache size metric."""
        self.cache_size_items.labels(cache_type=cache_type).set(size)
    
    def record_database_query(self, operation: str, duration: float):
        """Record database query metrics."""
        self.database_query_duration_seconds.labels(operation=operation).observe(duration)
    
    def update_database_connections(self, active_count: int):
        """Update active database connections."""
        self.database_connections_active.set(active_count)
    
    def record_error(self, component: str, error_type: str, severity: str = "error"):
        """Record error metrics."""
        self.errors_total.labels(
            component=component,
            error_type=error_type,
            severity=severity
        ).inc()
    
    def record_rate_limit_exceeded(self, client_id: str, endpoint: str):
        """Record rate limit violation."""
        self.rate_limit_exceeded_total.labels(
            client_id=client_id,
            endpoint=endpoint
        ).inc()
    
    def update_detector_accuracy(self, algorithm: str, dataset: str, accuracy: float):
        """Update detector accuracy metric."""
        self.detector_accuracy.labels(
            algorithm=algorithm,
            dataset=dataset
        ).set(accuracy)
    
    def update_active_sessions(self, count: int):
        """Update active sessions count."""
        self.active_sessions.set(count)
    
    def set_app_info(self, version: str, environment: str, build_date: str = None):
        """Set application information."""
        info = {
            'version': version,
            'environment': environment
        }
        if build_date:
            info['build_date'] = build_date
        
        self.app_info.info(info)


# FastAPI middleware for automatic metrics collection
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

class PrometheusMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for Prometheus metrics collection."""
    
    def __init__(self, app, metrics: PrometheusMetrics):
        super().__init__(app)
        self.metrics = metrics
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = request.url.path
        
        # Get request size
        request_size = 0
        if hasattr(request, 'body'):
            body = await request.body()
            request_size = len(body) if body else 0
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        self.metrics.record_http_request(
            method=method,
            endpoint=path,
            status_code=response.status_code,
            duration=duration,
            request_size=request_size
        )
        
        return response
```

## Structured Logging

### Comprehensive Logging Configuration

```python
# infrastructure/monitoring/structured_logging.py
import logging
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
import structlog
from structlog.stdlib import LoggerFactory
import asyncio
from contextvars import ContextVar

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredLogger:
    """Structured logging configuration for Pynomaly."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 service_name: str = "pynomaly",
                 environment: str = "production",
                 json_format: bool = True):
        
        self.service_name = service_name
        self.environment = environment
        self.json_format = json_format
        
        # Configure structlog
        self._configure_structlog(log_level)
        
        # Get logger
        self.logger = structlog.get_logger()
    
    def _configure_structlog(self, log_level: str):
        """Configure structlog with proper processors and formatters."""
        
        # Shared processors
        shared_processors = [
            # Add log level
            structlog.stdlib.add_log_level,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Add caller info
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.MODULE,
                ]
            ),
            # Add context
            self._add_context_processor,
            # Add service info
            self._add_service_info_processor,
        ]
        
        if self.json_format:
            # JSON formatting for production
            structlog.configure(
                processors=shared_processors + [
                    # Process exceptions
                    structlog.processors.format_exc_info,
                    # Convert to JSON
                    structlog.processors.JSONRenderer()
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                logger_factory=LoggerFactory(),
                cache_logger_on_first_use=True,
            )
        else:
            # Console formatting for development
            structlog.configure(
                processors=shared_processors + [
                    # Process exceptions
                    structlog.dev.set_exc_info,
                    # Console formatting
                    structlog.dev.ConsoleRenderer(colors=True)
                ],
                wrapper_class=structlog.stdlib.BoundLogger,
                logger_factory=LoggerFactory(),
                cache_logger_on_first_use=True,
            )
        
        # Configure standard logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, log_level.upper())
        )
        
        # Set log levels for third-party libraries
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        logging.getLogger("redis").setLevel(logging.WARNING)
    
    def _add_context_processor(self, logger, method_name, event_dict):
        """Add context variables to log entries."""
        
        # Add request context
        request_id = request_id_var.get()
        if request_id:
            event_dict["request_id"] = request_id
        
        user_id = user_id_var.get()
        if user_id:
            event_dict["user_id"] = user_id
        
        correlation_id = correlation_id_var.get()
        if correlation_id:
            event_dict["correlation_id"] = correlation_id
        
        return event_dict
    
    def _add_service_info_processor(self, logger, method_name, event_dict):
        """Add service information to log entries."""
        
        event_dict["service"] = self.service_name
        event_dict["environment"] = self.environment
        
        return event_dict
    
    def set_request_context(self, request_id: str, user_id: str = None, 
                           correlation_id: str = None):
        """Set request context for logging."""
        
        request_id_var.set(request_id)
        if user_id:
            user_id_var.set(user_id)
        if correlation_id:
            correlation_id_var.set(correlation_id)
    
    def clear_request_context(self):
        """Clear request context."""
        
        request_id_var.set(None)
        user_id_var.set(None)
        correlation_id_var.set(None)
    
    def log_anomaly_detection(self, algorithm: str, dataset_id: str, 
                             anomalies_found: int, duration: float,
                             confidence_scores: list = None):
        """Log anomaly detection event."""
        
        self.logger.info(
            "Anomaly detection completed",
            event_type="anomaly_detection",
            algorithm=algorithm,
            dataset_id=dataset_id,
            anomalies_found=anomalies_found,
            duration_seconds=duration,
            confidence_scores=confidence_scores[:10] if confidence_scores else None  # Log first 10
        )
    
    def log_training_event(self, algorithm: str, dataset_id: str, 
                          sample_count: int, duration: float, 
                          parameters: Dict[str, Any] = None):
        """Log training event."""
        
        self.logger.info(
            "Model training completed",
            event_type="model_training",
            algorithm=algorithm,
            dataset_id=dataset_id,
            sample_count=sample_count,
            duration_seconds=duration,
            parameters=parameters
        )
    
    def log_security_event(self, event_type: str, client_ip: str, 
                          user_id: str = None, details: Dict[str, Any] = None):
        """Log security event."""
        
        self.logger.warning(
            f"Security event: {event_type}",
            event_type="security",
            security_event_type=event_type,
            client_ip=client_ip,
            user_id=user_id,
            details=details
        )
    
    def log_performance_warning(self, operation: str, duration: float, 
                               threshold: float, details: Dict[str, Any] = None):
        """Log performance warning."""
        
        self.logger.warning(
            f"Performance warning: {operation}",
            event_type="performance",
            operation=operation,
            duration_seconds=duration,
            threshold_seconds=threshold,
            details=details
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context."""
        
        self.logger.error(
            f"Error occurred: {str(error)}",
            event_type="error",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            context=context
        )
    
    def log_api_request(self, method: str, path: str, status_code: int, 
                       duration: float, request_size: int = 0, 
                       response_size: int = 0):
        """Log API request."""
        
        self.logger.info(
            f"API request: {method} {path}",
            event_type="api_request",
            method=method,
            path=path,
            status_code=status_code,
            duration_seconds=duration,
            request_size_bytes=request_size,
            response_size_bytes=response_size
        )
    
    def log_database_operation(self, operation: str, table: str, 
                              duration: float, rows_affected: int = 0):
        """Log database operation."""
        
        self.logger.debug(
            f"Database operation: {operation}",
            event_type="database",
            operation=operation,
            table=table,
            duration_seconds=duration,
            rows_affected=rows_affected
        )
    
    def log_cache_operation(self, operation: str, cache_type: str, 
                           key: str, hit: bool, duration: float = None):
        """Log cache operation."""
        
        self.logger.debug(
            f"Cache operation: {operation}",
            event_type="cache",
            operation=operation,
            cache_type=cache_type,
            key=key,
            hit=hit,
            duration_seconds=duration
        )


# FastAPI middleware for request logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import time

class LoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for request logging."""
    
    def __init__(self, app, structured_logger: StructuredLogger):
        super().__init__(app)
        self.logger = structured_logger
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Extract user ID from token if available
        user_id = None
        auth_header = request.headers.get("authorization")
        if auth_header:
            # Extract user ID from JWT token (simplified)
            try:
                # In production, properly decode JWT
                user_id = "extracted_from_jwt"
            except:
                pass
        
        # Set request context
        self.logger.set_request_context(
            request_id=request_id,
            user_id=user_id
        )
        
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Get response size
            response_size = 0
            if hasattr(response, 'body'):
                response_size = len(response.body) if response.body else 0
            
            # Log successful request
            self.logger.log_api_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration,
                response_size=response_size
            )
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            self.logger.log_error(e, {
                "method": request.method,
                "path": request.url.path,
                "duration": duration
            })
            
            raise
        
        finally:
            # Clear request context
            self.logger.clear_request_context()


# Logger configuration for different environments
def configure_logging(environment: str = "production", 
                     log_level: str = "INFO",
                     service_name: str = "pynomaly") -> StructuredLogger:
    """Configure logging based on environment."""
    
    json_format = environment in ["production", "staging"]
    
    return StructuredLogger(
        log_level=log_level,
        service_name=service_name,
        environment=environment,
        json_format=json_format
    )
```

This comprehensive monitoring guide provides production-ready observability implementation with OpenTelemetry, Prometheus metrics, structured logging, and distributed tracing. The configuration enables full visibility into Pynomaly's performance, errors, and usage patterns for effective operational monitoring.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
