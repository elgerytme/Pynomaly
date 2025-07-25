"""
Comprehensive Metrics Collection Framework

Integrates Prometheus, OpenTelemetry, and custom metrics collection
to provide complete observability across the MLOps ecosystem.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricScope(Enum):
    """Scope of metrics collection."""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    ML_MODEL = "ml_model"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected."""
    name: str
    metric_type: MetricType
    description: str
    scope: MetricScope
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: Optional[str] = None
    aggregation_window: Optional[timedelta] = None
    retention_period: Optional[timedelta] = field(default_factory=lambda: timedelta(days=30))


@dataclass
class MetricData:
    """Represents collected metric data."""
    name: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Advanced metrics collection framework that integrates multiple
    observability tools and provides comprehensive monitoring capabilities.
    """
    
    def __init__(
        self,
        service_name: str = "mlops-platform",
        environment: str = "production",
        prometheus_registry: Optional[CollectorRegistry] = None,
        otlp_endpoint: Optional[str] = None,
        enable_prometheus: bool = True,
        enable_opentelemetry: bool = True,
        custom_resource_attributes: Optional[Dict[str, str]] = None
    ):
        self.service_name = service_name
        self.environment = environment
        
        # Initialize Prometheus
        self.prometheus_registry = prometheus_registry or prometheus_client.REGISTRY
        self.prometheus_metrics: Dict[str, Any] = {}
        
        # Initialize OpenTelemetry
        self.meter = None
        self.otel_metrics: Dict[str, Any] = {}
        
        # Metric definitions registry
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Custom collectors and processors
        self.custom_collectors: List[Callable] = []
        self.metric_processors: List[Callable] = []
        
        # Enable/disable specific exporters
        self.enable_prometheus = enable_prometheus
        self.enable_opentelemetry = enable_opentelemetry
        
        if enable_prometheus:
            self._setup_prometheus()
        
        if enable_opentelemetry:
            self._setup_opentelemetry(otlp_endpoint, custom_resource_attributes)
    
    def _setup_prometheus(self) -> None:
        """Setup Prometheus metrics collection."""
        try:
            # Register default system metrics
            self._register_system_metrics()
            logger.info("Prometheus metrics collection initialized")
        except Exception as e:
            logger.error(f"Failed to setup Prometheus: {e}")
    
    def _setup_opentelemetry(
        self, 
        otlp_endpoint: Optional[str] = None,
        custom_attributes: Optional[Dict[str, str]] = None
    ) -> None:
        """Setup OpenTelemetry metrics collection."""
        try:
            # Create resource with service information
            resource_attributes = {
                "service.name": self.service_name,
                "service.version": "1.0.0",
                "deployment.environment": self.environment,
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python"
            }
            
            if custom_attributes:
                resource_attributes.update(custom_attributes)
            
            resource = Resource.create(resource_attributes)
            
            # Setup metric readers
            readers = []
            
            # Prometheus reader for local scraping
            if self.enable_prometheus:
                prometheus_reader = PrometheusMetricReader()
                readers.append(prometheus_reader)
            
            # OTLP reader for remote export
            if otlp_endpoint:
                otlp_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
                otlp_reader = PeriodicExportingMetricReader(
                    exporter=otlp_exporter,
                    export_interval_millis=10000  # 10 seconds
                )
                readers.append(otlp_reader)
            
            # Initialize meter provider
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=readers
            )
            
            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(__name__)
            
            logger.info("OpenTelemetry metrics collection initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry: {e}")
    
    def _register_system_metrics(self) -> None:
        """Register default system-level metrics."""
        system_metrics = [
            MetricDefinition(
                name="system_cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                description="System CPU usage percentage",
                scope=MetricScope.SYSTEM,
                labels=["host", "core"],
                unit="percent"
            ),
            MetricDefinition(
                name="system_memory_usage_bytes",
                metric_type=MetricType.GAUGE,
                description="System memory usage in bytes",
                scope=MetricScope.SYSTEM,
                labels=["host", "type"],
                unit="bytes"
            ),
            MetricDefinition(
                name="system_disk_usage_bytes",
                metric_type=MetricType.GAUGE,
                description="System disk usage in bytes",
                scope=MetricScope.SYSTEM,
                labels=["host", "device", "mountpoint"],
                unit="bytes"
            ),
            MetricDefinition(
                name="system_network_bytes_total",
                metric_type=MetricType.COUNTER,
                description="Total network bytes transferred",
                scope=MetricScope.SYSTEM,
                labels=["host", "interface", "direction"],
                unit="bytes"
            )
        ]
        
        for metric_def in system_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: MetricDefinition) -> None:
        """Register a metric definition with both Prometheus and OpenTelemetry."""
        self.metric_definitions[metric_def.name] = metric_def
        
        # Register with Prometheus
        if self.enable_prometheus:
            self._register_prometheus_metric(metric_def)
        
        # Register with OpenTelemetry
        if self.enable_opentelemetry and self.meter:
            self._register_otel_metric(metric_def)
    
    def _register_prometheus_metric(self, metric_def: MetricDefinition) -> None:
        """Register metric with Prometheus."""
        try:
            if metric_def.metric_type == MetricType.COUNTER:
                metric = Counter(
                    metric_def.name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.prometheus_registry
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    metric_def.name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.prometheus_registry
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                buckets = metric_def.buckets or prometheus_client.DEFAULT_BUCKETS
                metric = Histogram(
                    metric_def.name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    buckets=buckets,
                    registry=self.prometheus_registry
                )
            elif metric_def.metric_type == MetricType.SUMMARY:
                metric = Summary(
                    metric_def.name,
                    metric_def.description,
                    labelnames=metric_def.labels,
                    registry=self.prometheus_registry
                )
            else:
                raise ValueError(f"Unsupported metric type: {metric_def.metric_type}")
            
            self.prometheus_metrics[metric_def.name] = metric
            
        except Exception as e:
            logger.error(f"Failed to register Prometheus metric {metric_def.name}: {e}")
    
    def _register_otel_metric(self, metric_def: MetricDefinition) -> None:
        """Register metric with OpenTelemetry."""
        try:
            if metric_def.metric_type == MetricType.COUNTER:
                metric = self.meter.create_counter(
                    name=metric_def.name,
                    description=metric_def.description,
                    unit=metric_def.unit or "1"
                )
            elif metric_def.metric_type == MetricType.GAUGE:
                metric = self.meter.create_gauge(
                    name=metric_def.name,
                    description=metric_def.description,
                    unit=metric_def.unit or "1"
                )
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                # OpenTelemetry uses different bucket configuration
                metric = self.meter.create_histogram(
                    name=metric_def.name,
                    description=metric_def.description,
                    unit=metric_def.unit or "1"
                )
            else:
                # Summary not directly supported, use histogram
                metric = self.meter.create_histogram(
                    name=metric_def.name,
                    description=metric_def.description,
                    unit=metric_def.unit or "1"
                )
            
            self.otel_metrics[metric_def.name] = metric
            
        except Exception as e:
            logger.error(f"Failed to register OpenTelemetry metric {metric_def.name}: {e}")
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record a metric value with both exporters."""
        labels = labels or {}
        
        # Record with Prometheus
        if self.enable_prometheus and name in self.prometheus_metrics:
            self._record_prometheus_metric(name, value, labels)
        
        # Record with OpenTelemetry
        if self.enable_opentelemetry and name in self.otel_metrics:
            self._record_otel_metric(name, value, labels)
        
        # Process with custom processors
        metric_data = MetricData(
            name=name,
            value=value,
            labels=labels,
            timestamp=timestamp or datetime.utcnow()
        )
        
        for processor in self.metric_processors:
            try:
                processor(metric_data)
            except Exception as e:
                logger.error(f"Error in metric processor: {e}")
    
    def _record_prometheus_metric(
        self,
        name: str,
        value: Union[int, float],
        labels: Dict[str, str]
    ) -> None:
        """Record metric with Prometheus."""
        try:
            metric = self.prometheus_metrics[name]
            metric_def = self.metric_definitions[name]
            
            if metric_def.metric_type == MetricType.COUNTER:
                if labels:
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
            elif metric_def.metric_type == MetricType.GAUGE:
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
            elif metric_def.metric_type == MetricType.HISTOGRAM:
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
            elif metric_def.metric_type == MetricType.SUMMARY:
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
                    
        except Exception as e:
            logger.error(f"Failed to record Prometheus metric {name}: {e}")
    
    def _record_otel_metric(
        self,
        name: str,
        value: Union[int, float],
        labels: Dict[str, str]
    ) -> None:
        """Record metric with OpenTelemetry."""
        try:
            metric = self.otel_metrics[name]
            metric_def = self.metric_definitions[name]
            
            if metric_def.metric_type == MetricType.COUNTER:
                metric.add(value, labels)
            elif metric_def.metric_type == MetricType.GAUGE:
                metric.set(value, labels)
            elif metric_def.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                metric.record(value, labels)
                
        except Exception as e:
            logger.error(f"Failed to record OpenTelemetry metric {name}: {e}")
    
    def increment_counter(
        self,
        name: str,
        amount: Union[int, float] = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Convenience method to increment a counter."""
        self.record_metric(name, amount, labels)
    
    def set_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Convenience method to set a gauge value."""
        self.record_metric(name, value, labels)
    
    def observe_histogram(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Convenience method to observe a histogram value."""
        self.record_metric(name, value, labels)
    
    def time_function(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator to time function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.observe_histogram(metric_name, duration, labels)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    error_labels = (labels or {}).copy()
                    error_labels["status"] = "error"
                    self.observe_histogram(metric_name, duration, error_labels)
                    raise
            return wrapper
        return decorator
    
    async def collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("system_cpu_usage_percent", cpu_percent, {"host": "localhost"})
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_usage_bytes", memory.used, {"host": "localhost", "type": "used"})
            self.set_gauge("system_memory_usage_bytes", memory.available, {"host": "localhost", "type": "available"})
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.set_gauge("system_disk_usage_bytes", disk.used, {"host": "localhost", "device": "/", "mountpoint": "/"})
            
            # Network metrics
            network = psutil.net_io_counters()
            self.increment_counter("system_network_bytes_total", network.bytes_sent, {"host": "localhost", "interface": "total", "direction": "sent"})
            self.increment_counter("system_network_bytes_total", network.bytes_recv, {"host": "localhost", "interface": "total", "direction": "received"})
            
        except ImportError:
            logger.warning("psutil not available, skipping system metrics collection")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def add_custom_collector(self, collector: Callable) -> None:
        """Add a custom metrics collector function."""
        self.custom_collectors.append(collector)
    
    def add_metric_processor(self, processor: Callable[[MetricData], None]) -> None:
        """Add a custom metric processor."""
        self.metric_processors.append(processor)
    
    async def run_collection_loop(self, interval: float = 30.0) -> None:
        """Run the metrics collection loop."""
        logger.info(f"Starting metrics collection loop with {interval}s interval")
        
        while True:
            try:
                # Collect system metrics
                await self.collect_system_metrics()
                
                # Run custom collectors
                for collector in self.custom_collectors:
                    try:
                        if asyncio.iscoroutinefunction(collector):
                            await collector()
                        else:
                            collector()
                    except Exception as e:
                        logger.error(f"Error in custom collector: {e}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(interval)
    
    def get_metric_definitions(self, scope: Optional[MetricScope] = None) -> List[MetricDefinition]:
        """Get all registered metric definitions, optionally filtered by scope."""
        if scope:
            return [metric_def for metric_def in self.metric_definitions.values() if metric_def.scope == scope]
        return list(self.metric_definitions.values())
    
    def export_prometheus_metrics(self) -> str:
        """Export all Prometheus metrics in exposition format."""
        from prometheus_client import generate_latest
        return generate_latest(self.prometheus_registry).decode('utf-8')
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the metrics collection system."""
        return {
            "metrics_collector": {
                "prometheus_enabled": self.enable_prometheus,
                "opentelemetry_enabled": self.enable_opentelemetry,
                "registered_metrics": len(self.metric_definitions),
                "custom_collectors": len(self.custom_collectors),
                "metric_processors": len(self.metric_processors),
                "service_name": self.service_name,
                "environment": self.environment,
                "last_collection": datetime.utcnow().isoformat()
            }
        }


# ML-specific metrics definitions
ML_METRICS_DEFINITIONS = [
    MetricDefinition(
        name="ml_model_prediction_latency_seconds",
        metric_type=MetricType.HISTOGRAM,
        description="Time taken for model prediction",
        scope=MetricScope.ML_MODEL,
        labels=["model_name", "model_version", "deployment_id"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        unit="seconds"
    ),
    MetricDefinition(
        name="ml_model_prediction_total",
        metric_type=MetricType.COUNTER,
        description="Total number of predictions made",
        scope=MetricScope.ML_MODEL,
        labels=["model_name", "model_version", "status"],
        unit="predictions"
    ),
    MetricDefinition(
        name="ml_model_accuracy_score",
        metric_type=MetricType.GAUGE,
        description="Current model accuracy score",
        scope=MetricScope.ML_MODEL,
        labels=["model_name", "model_version"],
        unit="score"
    ),
    MetricDefinition(
        name="ml_model_drift_score",
        metric_type=MetricType.GAUGE,
        description="Data drift detection score",
        scope=MetricScope.ML_MODEL,
        labels=["model_name", "drift_type"],
        unit="score"
    ),
    MetricDefinition(
        name="ml_model_bias_score",
        metric_type=MetricType.GAUGE,
        description="Model bias detection score",
        scope=MetricScope.ML_MODEL,
        labels=["model_name", "protected_attribute"],
        unit="score"
    )
]


# Business metrics definitions
BUSINESS_METRICS_DEFINITIONS = [
    MetricDefinition(
        name="business_api_requests_total",
        metric_type=MetricType.COUNTER,
        description="Total API requests",
        scope=MetricScope.BUSINESS,
        labels=["endpoint", "method", "status_code"],
        unit="requests"
    ),
    MetricDefinition(
        name="business_revenue_total",
        metric_type=MetricType.COUNTER,
        description="Total revenue generated",
        scope=MetricScope.BUSINESS,
        labels=["product", "region"],
        unit="currency"
    ),
    MetricDefinition(
        name="business_active_users",
        metric_type=MetricType.GAUGE,
        description="Number of active users",
        scope=MetricScope.BUSINESS,
        labels=["time_period"],
        unit="users"
    ),
    MetricDefinition(
        name="business_conversion_rate",
        metric_type=MetricType.GAUGE,
        description="Conversion rate percentage",
        scope=MetricScope.BUSINESS,
        labels=["funnel_stage"],
        unit="percent"
    )
]