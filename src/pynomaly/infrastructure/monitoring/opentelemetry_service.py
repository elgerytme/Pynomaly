"""Enterprise OpenTelemetry observability service."""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union
from functools import wraps

try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    metrics = None

from pynomaly.shared.config import Config
from pynomaly.shared.types import PerformanceMetrics


logger = logging.getLogger(__name__)


class OpenTelemetryService:
    """Enterprise OpenTelemetry observability service."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize OpenTelemetry service."""
        self.config = config or Config()
        self.tracer = None
        self.meter = None
        self._initialized = False
        
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, falling back to basic logging")
            return
            
        self._setup_resource()
        self._setup_tracing()
        self._setup_metrics()
        self._setup_propagation()
        self._setup_instrumentation()
        
        self._initialized = True
        logger.info("OpenTelemetry service initialized successfully")
    
    def _setup_resource(self) -> None:
        """Set up OpenTelemetry resource attributes."""
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: "pynomaly",
            ResourceAttributes.SERVICE_VERSION: self.config.get("version", "0.2.0"),
            ResourceAttributes.SERVICE_NAMESPACE: self.config.get("namespace", "production"),
            ResourceAttributes.SERVICE_INSTANCE_ID: os.environ.get("INSTANCE_ID", "local"),
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.get("environment", "production"),
            "pynomaly.tenant_id": self.config.get("tenant_id", "default"),
            "pynomaly.region": self.config.get("region", "us-east-1"),
            "pynomaly.cluster": self.config.get("cluster", "default"),
        })
        self.resource = resource
    
    def _setup_tracing(self) -> None:
        """Set up distributed tracing."""
        if not OPENTELEMETRY_AVAILABLE:
            return
            
        # Configure trace provider
        trace.set_tracer_provider(TracerProvider(resource=self.resource))
        
        # Set up exporters
        otlp_endpoint = self.config.get("telemetry.otlp_endpoint")
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Set up Jaeger exporter if configured
        jaeger_endpoint = self.config.get("telemetry.jaeger_endpoint")
        if jaeger_endpoint:
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                jaeger_exporter = JaegerExporter(
                    agent_host_name=self.config.get("telemetry.jaeger_host", "localhost"),
                    agent_port=self.config.get("telemetry.jaeger_port", 6831),
                )
                jaeger_processor = BatchSpanProcessor(jaeger_exporter)
                trace.get_tracer_provider().add_span_processor(jaeger_processor)
            except ImportError:
                logger.warning("Jaeger exporter not available")
        
        self.tracer = trace.get_tracer(__name__)
    
    def _setup_metrics(self) -> None:
        """Set up metrics collection."""
        if not OPENTELEMETRY_AVAILABLE:
            return
            
        readers = []
        
        # Prometheus metrics reader
        if self.config.get("telemetry.prometheus_enabled", True):
            prometheus_reader = PrometheusMetricReader()
            readers.append(prometheus_reader)
        
        # OTLP metrics exporter
        otlp_endpoint = self.config.get("telemetry.otlp_metrics_endpoint")
        if otlp_endpoint:
            otlp_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
            otlp_reader = PeriodicExportingMetricReader(
                exporter=otlp_exporter,
                export_interval_millis=self.config.get("telemetry.metrics_interval", 30000)
            )
            readers.append(otlp_reader)
        
        # Configure meter provider
        metrics.set_meter_provider(MeterProvider(
            resource=self.resource,
            metric_readers=readers
        ))
        
        self.meter = metrics.get_meter(__name__)
        self._setup_custom_metrics()
    
    def _setup_custom_metrics(self) -> None:
        """Set up custom business metrics."""
        if not self.meter:
            return
            
        # Detection metrics
        self.detection_counter = self.meter.create_counter(
            name="pynomaly_detections_total",
            description="Total number of anomaly detections performed",
            unit="1"
        )
        
        self.detection_duration = self.meter.create_histogram(
            name="pynomaly_detection_duration_seconds",
            description="Duration of anomaly detection operations",
            unit="s"
        )
        
        self.anomaly_score_gauge = self.meter.create_gauge(
            name="pynomaly_anomaly_score",
            description="Current anomaly score",
            unit="1"
        )
        
        # Model metrics
        self.model_training_counter = self.meter.create_counter(
            name="pynomaly_model_training_total",
            description="Total number of model training operations",
            unit="1"
        )
        
        self.model_accuracy_gauge = self.meter.create_gauge(
            name="pynomaly_model_accuracy",
            description="Current model accuracy",
            unit="1"
        )
        
        # System metrics
        self.memory_usage_gauge = self.meter.create_gauge(
            name="pynomaly_memory_usage_bytes",
            description="Current memory usage",
            unit="bytes"
        )
        
        self.cpu_usage_gauge = self.meter.create_gauge(
            name="pynomaly_cpu_usage_percent",
            description="Current CPU usage percentage",
            unit="percent"
        )
        
        # Data processing metrics
        self.data_processing_counter = self.meter.create_counter(
            name="pynomaly_data_processed_total",
            description="Total amount of data processed",
            unit="bytes"
        )
        
        self.data_processing_duration = self.meter.create_histogram(
            name="pynomaly_data_processing_duration_seconds",
            description="Duration of data processing operations",
            unit="s"
        )
    
    def _setup_propagation(self) -> None:
        """Set up trace context propagation."""
        if not OPENTELEMETRY_AVAILABLE:
            return
            
        # Use B3 propagation for distributed tracing
        set_global_textmap(B3MultiFormat())
    
    def _setup_instrumentation(self) -> None:
        """Set up automatic instrumentation."""
        if not OPENTELEMETRY_AVAILABLE:
            return
            
        try:
            # Instrument HTTP requests
            RequestsInstrumentor().instrument()
            
            # Instrument SQLAlchemy
            SQLAlchemyInstrumentor().instrument()
            
            # Note: FastAPI instrumentation would be done at the application level
            logger.info("Automatic instrumentation configured")
        except Exception as e:
            logger.warning(f"Failed to set up some instrumentation: {e}")
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations."""
        if not self._initialized or not self.tracer:
            yield None
            return
            
        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            start_time = time.time()
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("operation.duration", duration)
    
    def record_detection_metrics(
        self, 
        duration: float, 
        anomaly_count: int, 
        algorithm: str,
        tenant_id: Optional[str] = None
    ) -> None:
        """Record anomaly detection metrics."""
        if not self._initialized:
            return
            
        attributes = {
            "algorithm": algorithm,
            "tenant_id": tenant_id or "default"
        }
        
        if self.detection_counter:
            self.detection_counter.add(1, attributes)
        
        if self.detection_duration:
            self.detection_duration.record(duration, attributes)
    
    def record_model_metrics(
        self, 
        training_duration: float, 
        accuracy: float, 
        algorithm: str,
        tenant_id: Optional[str] = None
    ) -> None:
        """Record model training metrics."""
        if not self._initialized:
            return
            
        attributes = {
            "algorithm": algorithm,
            "tenant_id": tenant_id or "default"
        }
        
        if self.model_training_counter:
            self.model_training_counter.add(1, attributes)
        
        if self.model_accuracy_gauge:
            self.model_accuracy_gauge.set(accuracy, attributes)
    
    def record_system_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record system performance metrics."""
        if not self._initialized:
            return
            
        attributes = {"instance": os.environ.get("INSTANCE_ID", "local")}
        
        if self.memory_usage_gauge and hasattr(metrics, 'memory_usage'):
            self.memory_usage_gauge.set(metrics.memory_usage, attributes)
        
        if self.cpu_usage_gauge and hasattr(metrics, 'cpu_usage'):
            self.cpu_usage_gauge.set(metrics.cpu_usage, attributes)
    
    def record_data_processing_metrics(
        self, 
        duration: float, 
        bytes_processed: int,
        operation_type: str,
        tenant_id: Optional[str] = None
    ) -> None:
        """Record data processing metrics."""
        if not self._initialized:
            return
            
        attributes = {
            "operation_type": operation_type,
            "tenant_id": tenant_id or "default"
        }
        
        if self.data_processing_counter:
            self.data_processing_counter.add(bytes_processed, attributes)
        
        if self.data_processing_duration:
            self.data_processing_duration.record(duration, attributes)
    
    def create_custom_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a custom span for manual instrumentation."""
        if not self._initialized or not self.tracer:
            return None
            
        span = self.tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        return span
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID for correlation."""
        if not self._initialized:
            return None
            
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.get_span_context().is_valid:
                return format(current_span.get_span_context().trace_id, '032x')
        except Exception:
            pass
        
        return None
    
    def shutdown(self) -> None:
        """Shutdown the telemetry service."""
        if not self._initialized:
            return
            
        try:
            # Shutdown trace provider
            if trace.get_tracer_provider():
                trace.get_tracer_provider().shutdown()
            
            # Shutdown meter provider
            if metrics.get_meter_provider():
                metrics.get_meter_provider().shutdown()
            
            logger.info("OpenTelemetry service shutdown successfully")
        except Exception as e:
            logger.error(f"Error during telemetry shutdown: {e}")


def trace_anomaly_detection(func):
    """Decorator for tracing anomaly detection operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        telemetry = get_telemetry_service()
        
        # Extract relevant attributes
        attributes = {
            "operation": "anomaly_detection",
            "function": func.__name__
        }
        
        # Try to extract algorithm from args/kwargs
        if args and hasattr(args[0], '__class__'):
            attributes["algorithm"] = args[0].__class__.__name__
        
        with telemetry.trace_operation(f"detection.{func.__name__}", attributes) as span:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Record metrics if we have a result with anomaly information
                if hasattr(result, 'anomaly_count'):
                    duration = time.time() - start_time
                    telemetry.record_detection_metrics(
                        duration=duration,
                        anomaly_count=getattr(result, 'anomaly_count', 0),
                        algorithm=attributes.get("algorithm", "unknown")
                    )
                
                return result
            except Exception as e:
                if span:
                    span.record_exception(e)
                raise
    
    return wrapper


def trace_model_training(func):
    """Decorator for tracing model training operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        telemetry = get_telemetry_service()
        
        attributes = {
            "operation": "model_training",
            "function": func.__name__
        }
        
        if args and hasattr(args[0], '__class__'):
            attributes["algorithm"] = args[0].__class__.__name__
        
        with telemetry.trace_operation(f"training.{func.__name__}", attributes) as span:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Record training metrics
                duration = time.time() - start_time
                accuracy = getattr(result, 'accuracy', 0.0) if hasattr(result, 'accuracy') else 0.0
                
                telemetry.record_model_metrics(
                    training_duration=duration,
                    accuracy=accuracy,
                    algorithm=attributes.get("algorithm", "unknown")
                )
                
                return result
            except Exception as e:
                if span:
                    span.record_exception(e)
                raise
    
    return wrapper


# Global telemetry service instance
_telemetry_service: Optional[OpenTelemetryService] = None


def initialize_telemetry(config: Optional[Config] = None) -> OpenTelemetryService:
    """Initialize the global telemetry service."""
    global _telemetry_service
    _telemetry_service = OpenTelemetryService(config)
    return _telemetry_service


def get_telemetry_service() -> OpenTelemetryService:
    """Get the global telemetry service instance."""
    global _telemetry_service
    if _telemetry_service is None:
        _telemetry_service = OpenTelemetryService()
    return _telemetry_service


def shutdown_telemetry() -> None:
    """Shutdown the global telemetry service."""
    global _telemetry_service
    if _telemetry_service:
        _telemetry_service.shutdown()
        _telemetry_service = None