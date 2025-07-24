"""OpenTelemetry distributed tracing implementation for anomaly detection."""

from __future__ import annotations

import logging
import time
import functools
from typing import Dict, Any, Optional, Callable, Union
from contextlib import contextmanager
import os

try:
    from opentelemetry import trace, baggage, context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.span import Span
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    baggage = None
    context = None

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

logger = logging.getLogger(__name__)


class TracingConfig:
    """Configuration for distributed tracing."""
    
    def __init__(
        self,
        service_name: str = "anomaly-detection",
        service_version: str = "1.0.0",
        environment: str = "production",
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        sampling_rate: float = 1.0,
        enable_console_exporter: bool = False,
        enable_auto_instrumentation: bool = True,
        custom_attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize tracing configuration.
        
        Args:
            service_name: Service name for tracing
            service_version: Service version
            environment: Deployment environment
            jaeger_endpoint: Jaeger collector endpoint
            otlp_endpoint: OTLP collector endpoint
            sampling_rate: Sampling rate (0.0 to 1.0)
            enable_console_exporter: Whether to enable console exporter
            enable_auto_instrumentation: Whether to enable auto-instrumentation
            custom_attributes: Additional resource attributes
        """
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.jaeger_endpoint = jaeger_endpoint or os.getenv("JAEGER_ENDPOINT")
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTLP_ENDPOINT")
        self.sampling_rate = sampling_rate
        self.enable_console_exporter = enable_console_exporter
        self.enable_auto_instrumentation = enable_auto_instrumentation
        self.custom_attributes = custom_attributes or {}


class DistributedTracer:
    """Distributed tracing implementation with OpenTelemetry."""
    
    def __init__(self, config: TracingConfig):
        """Initialize distributed tracer.
        
        Args:
            config: Tracing configuration
        """
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, tracing disabled")
            self.enabled = False
            return
        
        self.config = config
        self.enabled = True
        self.tracer = None
        
        # Initialize tracing
        self._setup_tracing()
        
        logger.info(f"Distributed tracing initialized for service: {config.service_name}")
    
    def _setup_tracing(self) -> None:
        """Set up OpenTelemetry tracing."""
        # Create resource with service information
        resource_attributes = {
            ResourceAttributes.SERVICE_NAME: self.config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            **self.config.custom_attributes
        }
        
        resource = Resource.create(resource_attributes)
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Set up exporters
        self._setup_exporters(tracer_provider)
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version
        )
        
        # Set up auto-instrumentation
        if self.config.enable_auto_instrumentation:
            self._setup_auto_instrumentation()
    
    def _setup_exporters(self, tracer_provider: TracerProvider) -> None:
        """Set up span exporters."""
        exporters = []
        
        # Jaeger exporter
        if self.config.jaeger_endpoint:
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=self.config.jaeger_endpoint.split(':')[0],
                    agent_port=int(self.config.jaeger_endpoint.split(':')[1]),
                )
                exporters.append(jaeger_exporter)
                logger.info(f"Jaeger exporter configured: {self.config.jaeger_endpoint}")
            except Exception as e:
                logger.error(f"Failed to configure Jaeger exporter: {e}")
        
        # OTLP exporter
        if self.config.otlp_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.config.otlp_endpoint,
                    insecure=True  # Use secure=False for HTTPS endpoints
                )
                exporters.append(otlp_exporter)
                logger.info(f"OTLP exporter configured: {self.config.otlp_endpoint}")
            except Exception as e:
                logger.error(f"Failed to configure OTLP exporter: {e}")
        
        # Console exporter for debugging
        if self.config.enable_console_exporter:
            from opentelemetry.exporter.console import ConsoleSpanExporter
            console_exporter = ConsoleSpanExporter()
            exporters.append(console_exporter)
        
        # Add span processors
        for exporter in exporters:
            if self.config.environment == "development":
                # Use simple processor for development
                processor = SimpleSpanProcessor(exporter)
            else:
                # Use batch processor for production
                processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=512,
                    export_timeout_millis=30000,
                    max_export_batch_size=512
                )
            tracer_provider.add_span_processor(processor)
    
    def _setup_auto_instrumentation(self) -> None:
        """Set up automatic instrumentation for common libraries."""
        try:
            # FastAPI instrumentation
            FastAPIInstrumentor().instrument()
            logger.debug("FastAPI auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument FastAPI: {e}")
        
        try:
            # SQLAlchemy instrumentation
            SQLAlchemyInstrumentor().instrument()
            logger.debug("SQLAlchemy auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument SQLAlchemy: {e}")
        
        try:
            # Redis instrumentation
            RedisInstrumentor().instrument()
            logger.debug("Redis auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument Redis: {e}")
        
        try:
            # Requests instrumentation
            RequestsInstrumentor().instrument()
            logger.debug("Requests auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument Requests: {e}")
        
        try:
            # Psycopg2 instrumentation
            Psycopg2Instrumentor().instrument()
            logger.debug("Psycopg2 auto-instrumentation enabled")
        except Exception as e:
            logger.warning(f"Failed to instrument Psycopg2: {e}")
    
    def start_span(
        self,
        name: str,
        kind: Optional[trace.SpanKind] = None,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[Union[Span, trace.SpanContext]] = None
    ) -> Span:
        """Start a new span.
        
        Args:
            name: Span name
            kind: Span kind
            attributes: Span attributes
            parent: Parent span or context
            
        Returns:
            New span
        """
        if not self.enabled or not self.tracer:
            return trace.INVALID_SPAN
        
        span = self.tracer.start_span(
            name,
            kind=kind or trace.SpanKind.INTERNAL,
            attributes=attributes,
            context=trace.set_span_in_context(parent) if parent else None
        )
        
        return span
    
    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True
    ):
        """Context manager for tracing operations.
        
        Args:
            operation_name: Name of the operation
            attributes: Span attributes
            record_exception: Whether to record exceptions
        """
        if not self.enabled:
            yield
            return
        
        span = self.start_span(operation_name, attributes=attributes)
        
        try:
            with trace.use_span(span, end_on_exit=True):
                yield span
        except Exception as e:
            if record_exception and span.is_recording():
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
    
    def trace_function(
        self,
        operation_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True
    ):
        """Decorator for tracing functions.
        
        Args:
            operation_name: Custom operation name
            attributes: Span attributes
            record_exception: Whether to record exceptions
        """
        def decorator(func: Callable) -> Callable:
            if not self.enabled:
                return func
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                
                # Add function metadata as attributes
                func_attributes = {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                    **(attributes or {})
                }
                
                with self.trace_operation(name, func_attributes, record_exception):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def trace_async_function(
        self,
        operation_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True
    ):
        """Decorator for tracing async functions.
        
        Args:
            operation_name: Custom operation name
            attributes: Span attributes
            record_exception: Whether to record exceptions
        """
        def decorator(func: Callable) -> Callable:
            if not self.enabled:
                return func
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                
                func_attributes = {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                    "function.async": True,
                    **(attributes or {})
                }
                
                with self.trace_operation(name, func_attributes, record_exception):
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def add_span_attribute(self, key: str, value: Any) -> None:
        """Add attribute to current span.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.enabled:
            return
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute(key, value)
    
    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to current span.
        
        Args:
            name: Event name
            attributes: Event attributes
        """
        if not self.enabled:
            return
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.add_event(name, attributes or {})
    
    def set_baggage(self, key: str, value: str) -> None:
        """Set baggage item.
        
        Args:
            key: Baggage key
            value: Baggage value
        """
        if not self.enabled or not baggage:
            return
        
        baggage.set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage item.
        
        Args:
            key: Baggage key
            
        Returns:
            Baggage value or None
        """
        if not self.enabled or not baggage:
            return None
        
        return baggage.get_baggage(key)
    
    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID.
        
        Returns:
            Trace ID as hex string or None
        """
        if not self.enabled:
            return None
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            trace_id = current_span.get_span_context().trace_id
            return f"{trace_id:032x}"
        
        return None
    
    def get_span_id(self) -> Optional[str]:
        """Get current span ID.
        
        Returns:
            Span ID as hex string or None
        """
        if not self.enabled:
            return None
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            span_id = current_span.get_span_context().span_id
            return f"{span_id:016x}"
        
        return None
    
    def inject_trace_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into headers.
        
        Args:
            headers: HTTP headers dictionary
            
        Returns:
            Headers with injected trace context
        """
        if not self.enabled:
            return headers
        
        from opentelemetry.propagate import inject
        
        carrier = {}
        inject(carrier)
        headers.update(carrier)
        
        return headers
    
    def extract_trace_context(self, headers: Dict[str, str]) -> Optional[context.Context]:
        """Extract trace context from headers.
        
        Args:
            headers: HTTP headers dictionary
            
        Returns:
            Extracted context or None
        """
        if not self.enabled:
            return None
        
        from opentelemetry.propagate import extract
        
        return extract(headers)


class ModelTracer:
    """Specialized tracer for ML model operations."""
    
    def __init__(self, tracer: DistributedTracer):
        """Initialize model tracer.
        
        Args:
            tracer: Distributed tracer instance
        """
        self.tracer = tracer
    
    def trace_model_training(
        self,
        model_name: str,
        algorithm: str,
        data_shape: tuple,
        parameters: Dict[str, Any]
    ):
        """Decorator for tracing model training.
        
        Args:
            model_name: Name of the model
            algorithm: Algorithm used
            data_shape: Shape of training data
            parameters: Training parameters
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                attributes = {
                    "ml.model.name": model_name,
                    "ml.algorithm": algorithm,
                    "ml.data.samples": data_shape[0] if data_shape else 0,
                    "ml.data.features": data_shape[1] if len(data_shape) > 1 else 0,
                    "ml.operation": "training",
                    **{f"ml.parameter.{k}": str(v) for k, v in parameters.items()}
                }
                
                with self.tracer.trace_operation(
                    f"model.training.{algorithm}",
                    attributes=attributes
                ):
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    training_time = time.time() - start_time
                    
                    self.tracer.add_span_attribute("ml.training.duration", training_time)
                    self.tracer.add_span_event(
                        "training.completed",
                        {"training_time": training_time}
                    )
                    
                    return result
            
            return wrapper
        return decorator
    
    def trace_model_inference(
        self,
        model_name: str,
        batch_size: Optional[int] = None
    ):
        """Decorator for tracing model inference.
        
        Args:
            model_name: Name of the model
            batch_size: Batch size for inference
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                attributes = {
                    "ml.model.name": model_name,
                    "ml.operation": "inference"
                }
                
                if batch_size:
                    attributes["ml.batch.size"] = batch_size
                
                with self.tracer.trace_operation(
                    f"model.inference.{model_name}",
                    attributes=attributes
                ):
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    inference_time = time.time() - start_time
                    
                    self.tracer.add_span_attribute("ml.inference.duration", inference_time)
                    
                    if batch_size and inference_time > 0:
                        throughput = batch_size / inference_time
                        self.tracer.add_span_attribute("ml.inference.throughput", throughput)
                    
                    return result
            
            return wrapper
        return decorator
    
    def trace_data_processing(
        self,
        operation: str,
        data_size: Optional[int] = None
    ):
        """Decorator for tracing data processing operations.
        
        Args:
            operation: Data processing operation name
            data_size: Size of data being processed
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                attributes = {
                    "data.operation": operation,
                    "data.processing": True
                }
                
                if data_size:
                    attributes["data.size"] = data_size
                
                with self.tracer.trace_operation(
                    f"data.processing.{operation}",
                    attributes=attributes
                ):
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    processing_time = time.time() - start_time
                    
                    self.tracer.add_span_attribute("data.processing.duration", processing_time)
                    
                    if data_size and processing_time > 0:
                        throughput = data_size / processing_time
                        self.tracer.add_span_attribute("data.processing.throughput", throughput)
                    
                    return result
            
            return wrapper
        return decorator


# Global tracer instance
_global_tracer: Optional[DistributedTracer] = None


def initialize_tracing(config: TracingConfig) -> DistributedTracer:
    """Initialize global distributed tracing.
    
    Args:
        config: Tracing configuration
        
    Returns:
        Distributed tracer instance
    """
    global _global_tracer
    _global_tracer = DistributedTracer(config)
    return _global_tracer


def get_tracer() -> Optional[DistributedTracer]:
    """Get global tracer instance.
    
    Returns:
        Global tracer or None if not initialized
    """
    return _global_tracer


def get_model_tracer() -> Optional[ModelTracer]:
    """Get model tracer instance.
    
    Returns:
        Model tracer or None if not initialized
    """
    if _global_tracer:
        return ModelTracer(_global_tracer)
    return None


# Convenience decorators using global tracer
def trace(
    operation_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """Convenience decorator for tracing functions.
    
    Args:
        operation_name: Custom operation name
        attributes: Span attributes
    """
    def decorator(func: Callable) -> Callable:
        if _global_tracer:
            return _global_tracer.trace_function(operation_name, attributes)(func)
        return func
    return decorator


def trace_async(
    operation_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """Convenience decorator for tracing async functions.
    
    Args:
        operation_name: Custom operation name
        attributes: Span attributes
    """
    def decorator(func: Callable) -> Callable:
        if _global_tracer:
            return _global_tracer.trace_async_function(operation_name, attributes)(func)
        return func
    return decorator