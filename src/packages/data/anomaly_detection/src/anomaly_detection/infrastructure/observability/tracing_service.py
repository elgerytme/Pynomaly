"""OpenTelemetry distributed tracing service for comprehensive observability."""

import logging
import os
import time
from typing import Dict, List, Any, Optional, Union, Callable, ContextManager
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass
from datetime import datetime

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.propagate import inject, extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace.status import Status, StatusCode


@dataclass
class SpanContext:
    """Context information for a span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    tags: Dict[str, Any]
    

@dataclass
class TracingConfig:
    """Configuration for tracing service."""
    service_name: str
    service_version: str
    environment: str
    jaeger_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None
    sampling_rate: float = 1.0
    max_tag_value_length: int = 1024
    enable_console_export: bool = False
    enable_auto_instrumentation: bool = True
    custom_resource_attributes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.custom_resource_attributes is None:
            self.custom_resource_attributes = {}


class TracingService:
    """Service for distributed tracing with OpenTelemetry."""
    
    def __init__(self, config: TracingConfig):
        """Initialize tracing service.
        
        Args:
            config: Tracing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracer
        self._setup_tracing()
        self.tracer = trace.get_tracer(__name__)
        
        # Span storage for debugging
        self._active_spans: Dict[str, SpanContext] = {}
        
    def _setup_tracing(self):
        """Set up OpenTelemetry tracing configuration."""
        # Create resource with service information
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            **self.config.custom_resource_attributes
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Set up exporters
        self._setup_exporters(tracer_provider)
        
        # Set up auto-instrumentation
        if self.config.enable_auto_instrumentation:
            self._setup_auto_instrumentation()
        
        self.logger.info(f"Tracing initialized for service: {self.config.service_name}")
    
    def _setup_exporters(self, tracer_provider: TracerProvider):
        """Set up trace exporters.
        
        Args:
            tracer_provider: The tracer provider to attach exporters to
        """
        # Jaeger exporter
        if self.config.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_endpoint.split(':')[0],
                agent_port=int(self.config.jaeger_endpoint.split(':')[1]) if ':' in self.config.jaeger_endpoint else 14268,
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            self.logger.info(f"Jaeger exporter configured: {self.config.jaeger_endpoint}")
        
        # OTLP exporter
        if self.config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=True  # Use secure=False for HTTPS in production
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            self.logger.info(f"OTLP exporter configured: {self.config.otlp_endpoint}")
        
        # Console exporter for debugging
        if self.config.enable_console_export:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
            self.logger.info("Console exporter enabled")
    
    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation for common libraries."""
        try:
            # FastAPI instrumentation
            FastAPIInstrumentor().instrument()
            
            # HTTP requests instrumentation
            RequestsInstrumentor().instrument()
            
            # Database instrumentation
            try:
                SQLAlchemyInstrumentor().instrument()
            except Exception as e:
                self.logger.debug(f"SQLAlchemy instrumentation failed: {e}")
            
            try:
                Psycopg2Instrumentor().instrument()
            except Exception as e:
                self.logger.debug(f"Psycopg2 instrumentation failed: {e}")
            
            # Redis instrumentation
            try:
                RedisInstrumentor().instrument()
            except Exception as e:
                self.logger.debug(f"Redis instrumentation failed: {e}")
                
            self.logger.info("Auto-instrumentation enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to set up auto-instrumentation: {e}")
    
    @contextmanager
    def trace_operation(self,
                       operation_name: str,
                       tags: Optional[Dict[str, Any]] = None,
                       parent_context: Optional[Any] = None) -> ContextManager:
        """Context manager for tracing an operation.
        
        Args:
            operation_name: Name of the operation being traced
            tags: Optional tags to add to the span
            parent_context: Optional parent span context
            
        Yields:
            Span context for the operation
        """
        with self.tracer.start_as_current_span(
            operation_name,
            context=parent_context
        ) as span:
            try:
                # Add tags to span
                if tags:
                    for key, value in tags.items():
                        # Truncate long values
                        if isinstance(value, str) and len(value) > self.config.max_tag_value_length:
                            value = value[:self.config.max_tag_value_length] + "..."
                        span.set_attribute(key, value)
                
                # Create span context for tracking
                span_context = SpanContext(
                    trace_id=format(span.get_span_context().trace_id, '032x'),
                    span_id=format(span.get_span_context().span_id, '016x'),
                    parent_span_id=None,  # Could extract if needed
                    operation_name=operation_name,
                    start_time=datetime.now(),
                    tags=tags or {}
                )
                
                self._active_spans[span_context.span_id] = span_context
                
                yield span
                
                # Mark span as successful
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                # Record exception in span
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            
            finally:
                # Clean up span context
                if span_context.span_id in self._active_spans:
                    del self._active_spans[span_context.span_id]
    
    def trace_function(self,
                      operation_name: Optional[str] = None,
                      tags: Optional[Dict[str, Any]] = None,
                      include_args: bool = False,
                      include_result: bool = False):
        """Decorator for tracing function calls.
        
        Args:
            operation_name: Custom operation name (defaults to function name)
            tags: Additional tags to add to span
            include_args: Whether to include function arguments as tags
            include_result: Whether to include function result as tag
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                span_tags = tags.copy() if tags else {}
                
                # Add function arguments as tags
                if include_args:
                    try:
                        # Only include serializable arguments
                        for i, arg in enumerate(args):
                            if isinstance(arg, (str, int, float, bool)):
                                span_tags[f"arg_{i}"] = arg
                        
                        for key, value in kwargs.items():
                            if isinstance(value, (str, int, float, bool)):
                                span_tags[f"kwarg_{key}"] = value
                    except Exception as e:
                        self.logger.debug(f"Failed to serialize function arguments: {e}")
                
                with self.trace_operation(op_name, span_tags) as span:
                    start_time = time.time()
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        # Add execution time
                        execution_time = time.time() - start_time
                        span.set_attribute("execution_time_ms", execution_time * 1000)
                        
                        # Add result as tag if requested
                        if include_result and result is not None:
                            try:
                                if isinstance(result, (str, int, float, bool)):
                                    span.set_attribute("result", result)
                                elif hasattr(result, '__dict__'):
                                    span.set_attribute("result_type", type(result).__name__)
                            except Exception as e:
                                self.logger.debug(f"Failed to serialize function result: {e}")
                        
                        return result
                        
                    except Exception as e:
                        span.set_attribute("error", True)
                        span.set_attribute("error_message", str(e))
                        raise
            
            return wrapper
        return decorator
    
    def trace_async_function(self,
                           operation_name: Optional[str] = None,
                           tags: Optional[Dict[str, Any]] = None,
                           include_args: bool = False):
        """Decorator for tracing async function calls.
        
        Args:
            operation_name: Custom operation name
            tags: Additional tags to add to span
            include_args: Whether to include function arguments as tags
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                span_tags = tags.copy() if tags else {}
                
                if include_args:
                    try:
                        for i, arg in enumerate(args):
                            if isinstance(arg, (str, int, float, bool)):
                                span_tags[f"arg_{i}"] = arg
                        
                        for key, value in kwargs.items():
                            if isinstance(value, (str, int, float, bool)):
                                span_tags[f"kwarg_{key}"] = value
                    except Exception as e:
                        self.logger.debug(f"Failed to serialize async function arguments: {e}")
                
                with self.trace_operation(op_name, span_tags) as span:
                    start_time = time.time()
                    
                    try:
                        result = await func(*args, **kwargs)
                        
                        execution_time = time.time() - start_time
                        span.set_attribute("execution_time_ms", execution_time * 1000)
                        
                        return result
                        
                    except Exception as e:
                        span.set_attribute("error", True)
                        span.set_attribute("error_message", str(e))
                        raise
            
            return wrapper
        return decorator
    
    def add_span_attribute(self, key: str, value: Any):
        """Add attribute to current active span.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute(key, value)
    
    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current active span.
        
        Args:
            name: Event name
            attributes: Optional event attributes
        """
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event(name, attributes or {})
    
    def get_trace_context(self) -> Dict[str, str]:
        """Get current trace context for propagation.
        
        Returns:
            Dictionary containing trace context headers
        """
        headers = {}
        inject(headers)
        return headers
    
    def set_trace_context(self, headers: Dict[str, str]) -> Any:
        """Set trace context from headers.
        
        Args:
            headers: Headers containing trace context
            
        Returns:
            Extracted context
        """
        return extract(headers)
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID.
        
        Returns:
            Current trace ID or None if no active trace
        """
        current_span = trace.get_current_span()
        if current_span.is_recording():
            return format(current_span.get_span_context().trace_id, '032x')
        return None
    
    def get_current_span_id(self) -> Optional[str]:
        """Get current span ID.
        
        Returns:
            Current span ID or None if no active span
        """
        current_span = trace.get_current_span()
        if current_span.is_recording():
            return format(current_span.get_span_context().span_id, '016x')
        return None
    
    def create_child_span(self,
                         operation_name: str,
                         tags: Optional[Dict[str, Any]] = None) -> Any:
        """Create a child span from current context.
        
        Args:
            operation_name: Name of the child operation
            tags: Optional tags for the span
            
        Returns:
            New span
        """
        span = self.tracer.start_span(operation_name)
        
        if tags:
            for key, value in tags.items():
                span.set_attribute(key, value)
        
        return span
    
    def trace_model_inference(self,
                            model_name: str,
                            model_version: str,
                            input_shape: Optional[tuple] = None,
                            algorithm: Optional[str] = None):
        """Decorator specifically for tracing model inference operations.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            input_shape: Shape of input data
            algorithm: Algorithm used
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                tags = {
                    "model.name": model_name,
                    "model.version": model_version,
                    "model.algorithm": algorithm or "unknown"
                }
                
                if input_shape:
                    tags["model.input_shape"] = str(input_shape)
                
                with self.trace_operation(f"model.inference.{model_name}", tags) as span:
                    start_time = time.time()
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        # Add inference metrics
                        inference_time = time.time() - start_time
                        span.set_attribute("inference_time_ms", inference_time * 1000)
                        
                        # Add result metrics if available
                        if hasattr(result, 'get') and isinstance(result, dict):
                            if "anomalies" in result:
                                anomaly_count = sum(result["anomalies"])
                                span.set_attribute("anomalies_detected", anomaly_count)
                                span.set_attribute("total_predictions", len(result["anomalies"]))
                        
                        return result
                        
                    except Exception as e:
                        span.set_attribute("model.error", True)
                        span.set_attribute("model.error_message", str(e))
                        raise
            
            return wrapper
        return decorator
    
    def trace_data_processing(self,
                            operation_type: str,
                            data_source: Optional[str] = None):
        """Decorator for tracing data processing operations.
        
        Args:
            operation_type: Type of data processing (e.g., 'preprocessing', 'validation')
            data_source: Source of the data being processed
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                tags = {
                    "data.operation_type": operation_type,
                    "data.source": data_source or "unknown"
                }
                
                with self.trace_operation(f"data.{operation_type}", tags) as span:
                    start_time = time.time()
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        processing_time = time.time() - start_time
                        span.set_attribute("processing_time_ms", processing_time * 1000)
                        
                        # Add data metrics if available
                        if hasattr(result, '__len__'):
                            span.set_attribute("data.output_size", len(result))
                        
                        return result
                        
                    except Exception as e:
                        span.set_attribute("data.error", True)
                        span.set_attribute("data.error_message", str(e))
                        raise
            
            return wrapper
        return decorator
    
    def get_active_spans_summary(self) -> Dict[str, Any]:
        """Get summary of currently active spans.
        
        Returns:
            Summary of active spans
        """
        return {
            "total_active_spans": len(self._active_spans),
            "active_operations": [span.operation_name for span in self._active_spans.values()],
            "oldest_span_age_seconds": (
                min(
                    (datetime.now() - span.start_time).total_seconds()
                    for span in self._active_spans.values()
                ) if self._active_spans else 0
            )
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for tracing service.
        
        Returns:
            Health status information
        """
        try:
            # Test that tracing is working
            with self.trace_operation("health_check_test"):
                pass
            
            return {
                "status": "healthy",
                "service_name": self.config.service_name,
                "active_spans": len(self._active_spans),
                "exporters_configured": {
                    "jaeger": self.config.jaeger_endpoint is not None,
                    "otlp": self.config.otlp_endpoint is not None,
                    "console": self.config.enable_console_export
                }
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global tracing service instance
_tracing_service: Optional[TracingService] = None


def initialize_tracing(config: TracingConfig) -> TracingService:
    """Initialize global tracing service.
    
    Args:
        config: Tracing configuration
        
    Returns:
        Initialized tracing service
    """
    global _tracing_service
    _tracing_service = TracingService(config)
    return _tracing_service


def get_tracing_service() -> Optional[TracingService]:
    """Get global tracing service instance.
    
    Returns:
        Tracing service instance or None if not initialized
    """
    return _tracing_service


# Convenient decorator functions
def trace_operation(operation_name: str, tags: Optional[Dict[str, Any]] = None):
    """Convenient decorator for tracing operations."""
    if _tracing_service:
        return _tracing_service.trace_function(operation_name, tags)
    else:
        # No-op decorator if tracing not initialized
        def decorator(func):
            return func
        return decorator


def trace_model_inference(model_name: str, model_version: str, **kwargs):
    """Convenient decorator for tracing model inference."""
    if _tracing_service:
        return _tracing_service.trace_model_inference(model_name, model_version, **kwargs)
    else:
        def decorator(func):
            return func
        return decorator


def trace_data_processing(operation_type: str, **kwargs):
    """Convenient decorator for tracing data processing."""
    if _tracing_service:
        return _tracing_service.trace_data_processing(operation_type, **kwargs)
    else:
        def decorator(func):
            return func
        return decorator