"""
Distributed Tracing Management

Provides comprehensive distributed tracing capabilities using OpenTelemetry
with support for Jaeger, Zipkin, and custom exporters.
"""

import asyncio
import logging
import time
import uuid
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.auto_instrumentation import sitecustomize
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.baggage import set_baggage, get_baggage

logger = logging.getLogger(__name__)


class TracingBackend(Enum):
    """Supported tracing backends."""
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    OTLP = "otlp"
    CONSOLE = "console"


class SpanKind(Enum):
    """Types of spans for ML operations."""
    ML_TRAINING = "ml.training"
    ML_INFERENCE = "ml.inference"
    ML_PREPROCESSING = "ml.preprocessing"
    ML_EVALUATION = "ml.evaluation"
    DATA_PROCESSING = "data.processing"
    API_REQUEST = "api.request"
    DATABASE_QUERY = "database.query"
    CACHE_OPERATION = "cache.operation"
    EXTERNAL_SERVICE = "external.service"


@dataclass
class SpanMetadata:
    """Additional metadata for spans."""
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    dataset_id: Optional[str] = None
    experiment_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceContext:
    """Context information for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    sampling_priority: Optional[int] = None


class TracingManager:
    """
    Comprehensive distributed tracing manager that provides end-to-end
    observability across the entire MLOps ecosystem.
    """
    
    def __init__(
        self,
        service_name: str = "mlops-platform",
        service_version: str = "1.0.0",
        environment: str = "production",
        backends: List[TracingBackend] = None,
        sampling_rate: float = 1.0,
        enable_auto_instrumentation: bool = True,
        custom_resource_attributes: Optional[Dict[str, str]] = None
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.sampling_rate = sampling_rate
        
        # Initialize backends
        self.backends = backends or [TracingBackend.JAEGER]
        self.exporters = []
        self.processors = []
        
        # Tracing components
        self.tracer_provider = None
        self.tracer = None
        
        # Active spans tracking
        self.active_spans: Dict[str, trace.Span] = {}
        
        # Custom span processors
        self.custom_processors: List[Callable] = []
        
        # Initialize tracing
        self._setup_tracing(custom_resource_attributes)
        
        # Setup auto-instrumentation
        if enable_auto_instrumentation:
            self._setup_auto_instrumentation()
    
    def _setup_tracing(self, custom_attributes: Optional[Dict[str, str]] = None) -> None:
        """Setup OpenTelemetry tracing."""
        try:
            # Create resource with service information
            resource_attributes = {
                "service.name": self.service_name,
                "service.version": self.service_version,
                "deployment.environment": self.environment,
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python"
            }
            
            if custom_attributes:
                resource_attributes.update(custom_attributes)
            
            resource = Resource.create(resource_attributes)
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            
            # Setup exporters based on configured backends
            self._setup_exporters()
            
            # Setup processors
            for exporter in self.exporters:
                processor = BatchSpanProcessor(exporter)
                self.tracer_provider.add_span_processor(processor)
                self.processors.append(processor)
            
            # Set global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            logger.info(f"Distributed tracing initialized with backends: {[b.value for b in self.backends]}")
            
        except Exception as e:
            logger.error(f"Failed to setup tracing: {e}")
            raise
    
    def _setup_exporters(self) -> None:
        """Setup trace exporters for configured backends."""
        for backend in self.backends:
            try:
                if backend == TracingBackend.JAEGER:
                    exporter = JaegerExporter(
                        agent_host_name="localhost",
                        agent_port=6831,
                        collector_endpoint="http://localhost:14268/api/traces",
                    )
                    self.exporters.append(exporter)
                
                elif backend == TracingBackend.ZIPKIN:
                    exporter = ZipkinExporter(
                        endpoint="http://localhost:9411/api/v2/spans"
                    )
                    self.exporters.append(exporter)
                
                elif backend == TracingBackend.OTLP:
                    exporter = OTLPSpanExporter(
                        endpoint="http://localhost:4317"
                    )
                    self.exporters.append(exporter)
                
                elif backend == TracingBackend.CONSOLE:
                    exporter = ConsoleSpanExporter()
                    self.exporters.append(exporter)
                
                logger.info(f"Configured {backend.value} exporter")
                
            except Exception as e:
                logger.error(f"Failed to setup {backend.value} exporter: {e}")
    
    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for common libraries."""
        try:
            # HTTP requests
            RequestsInstrumentor().instrument()
            
            # Database connections
            SQLAlchemyInstrumentor().instrument()
            AsyncPGInstrumentor().instrument()
            
            # Redis
            RedisInstrumentor().instrument()
            
            logger.info("Auto-instrumentation enabled for common libraries")
            
        except Exception as e:
            logger.error(f"Failed to setup auto-instrumentation: {e}")
    
    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.API_REQUEST,
        metadata: Optional[SpanMetadata] = None,
        parent_context: Optional[trace.Context] = None
    ):
        """Start a new span with context management."""
        span = self.tracer.start_span(
            name=name,
            context=parent_context
        )
        
        try:
            # Set span attributes based on kind and metadata
            self._set_span_attributes(span, kind, metadata)
            
            # Store active span
            span_id = str(span.get_span_context().span_id)
            self.active_spans[span_id] = span
            
            # Set baggage if provided
            if metadata and metadata.custom_attributes:
                for key, value in metadata.custom_attributes.items():
                    set_baggage(key, str(value))
            
            yield span
            
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        
        finally:
            # Remove from active spans
            if span_id in self.active_spans:
                del self.active_spans[span_id]
            
            # End span
            span.end()
    
    @asynccontextmanager
    async def start_async_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.API_REQUEST,
        metadata: Optional[SpanMetadata] = None,
        parent_context: Optional[trace.Context] = None
    ) -> AsyncGenerator[trace.Span, None]:
        """Start a new async span with context management."""
        span = self.tracer.start_span(
            name=name,
            context=parent_context
        )
        
        try:
            # Set span attributes
            self._set_span_attributes(span, kind, metadata)
            
            # Store active span
            span_id = str(span.get_span_context().span_id)
            self.active_spans[span_id] = span
            
            # Set baggage
            if metadata and metadata.custom_attributes:
                for key, value in metadata.custom_attributes.items():
                    set_baggage(key, str(value))
            
            yield span
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        
        finally:
            if span_id in self.active_spans:
                del self.active_spans[span_id]
            span.end()
    
    def _set_span_attributes(
        self,
        span: trace.Span,
        kind: SpanKind,
        metadata: Optional[SpanMetadata] = None
    ) -> None:
        """Set attributes on a span based on kind and metadata."""
        # Set basic attributes
        span.set_attribute("span.kind", kind.value)
        span.set_attribute("service.name", self.service_name)
        span.set_attribute("service.version", self.service_version)
        span.set_attribute("deployment.environment", self.environment)
        
        if not metadata:
            return
        
        # ML-specific attributes
        if metadata.model_name:
            span.set_attribute("ml.model.name", metadata.model_name)
        
        if metadata.model_version:
            span.set_attribute("ml.model.version", metadata.model_version)
        
        if metadata.dataset_id:
            span.set_attribute("ml.dataset.id", metadata.dataset_id)
        
        if metadata.experiment_id:
            span.set_attribute("ml.experiment.id", metadata.experiment_id)
        
        # User and request attributes
        if metadata.user_id:
            span.set_attribute("user.id", metadata.user_id)
        
        if metadata.request_id:
            span.set_attribute("request.id", metadata.request_id)
        
        # Custom attributes
        for key, value in metadata.custom_attributes.items():
            span.set_attribute(f"custom.{key}", str(value))
    
    def trace_ml_training(
        self,
        model_name: str,
        model_version: str,
        dataset_id: str,
        experiment_id: Optional[str] = None
    ):
        """Decorator for tracing ML training operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                metadata = SpanMetadata(
                    model_name=model_name,
                    model_version=model_version,
                    dataset_id=dataset_id,
                    experiment_id=experiment_id
                )
                
                with self.start_span(
                    name=f"ml.training.{func.__name__}",
                    kind=SpanKind.ML_TRAINING,
                    metadata=metadata
                ) as span:
                    # Add training-specific attributes
                    span.set_attribute("ml.operation", "training")
                    span.set_attribute("function.name", func.__name__)
                    
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    span.set_attribute("training.duration_seconds", duration)
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
            return wrapper
        return decorator
    
    def trace_ml_inference(
        self,
        model_name: str,
        model_version: str,
        track_inputs: bool = True,
        track_outputs: bool = True
    ):
        """Decorator for tracing ML inference operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                metadata = SpanMetadata(
                    model_name=model_name,
                    model_version=model_version,
                    request_id=str(uuid.uuid4())
                )
                
                with self.start_span(
                    name=f"ml.inference.{func.__name__}",
                    kind=SpanKind.ML_INFERENCE,
                    metadata=metadata
                ) as span:
                    span.set_attribute("ml.operation", "inference")
                    span.set_attribute("function.name", func.__name__)
                    
                    # Track input data if enabled
                    if track_inputs and args:
                        span.set_attribute("inference.input_size", len(str(args[0])) if args else 0)
                    
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    span.set_attribute("inference.latency_seconds", duration)
                    
                    # Track output data if enabled
                    if track_outputs and result is not None:
                        span.set_attribute("inference.output_size", len(str(result)))
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
            return wrapper
        return decorator
    
    async def trace_async_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        kind: SpanKind = SpanKind.API_REQUEST,
        metadata: Optional[SpanMetadata] = None,
        *args,
        **kwargs
    ) -> Any:
        """Trace an async operation."""
        async with self.start_async_span(
            name=operation_name,
            kind=kind,
            metadata=metadata
        ) as span:
            span.set_attribute("operation.name", operation_name)
            
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            
            duration = time.time() - start_time
            span.set_attribute("operation.duration_seconds", duration)
            span.set_status(Status(StatusCode.OK))
            
            return result
    
    def get_current_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        current_span = trace.get_current_span()
        
        if not current_span or not current_span.is_recording():
            return None
        
        span_context = current_span.get_span_context()
        
        return TraceContext(
            trace_id=format(span_context.trace_id, '032x'),
            span_id=format(span_context.span_id, '016x'),
            baggage={k: get_baggage(k) for k in get_baggage()}
        )
    
    def inject_trace_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into headers for propagation."""
        from opentelemetry.propagate import inject
        
        inject(headers)
        return headers
    
    def extract_trace_context(self, headers: Dict[str, str]) -> trace.Context:
        """Extract trace context from headers."""
        from opentelemetry.propagate import extract
        
        return extract(headers)
    
    def add_span_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add an event to the current span."""
        current_span = trace.get_current_span()
        
        if current_span and current_span.is_recording():
            event_attributes = attributes or {}
            event_timestamp = timestamp or datetime.utcnow()
            
            current_span.add_event(
                name=name,
                attributes=event_attributes,
                timestamp=event_timestamp
            )
    
    def set_span_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current span."""
        current_span = trace.get_current_span()
        
        if current_span and current_span.is_recording():
            current_span.set_attribute(key, str(value))
    
    def record_exception(self, exception: Exception) -> None:
        """Record an exception in the current span."""
        current_span = trace.get_current_span()
        
        if current_span and current_span.is_recording():
            current_span.record_exception(exception)
            current_span.set_status(Status(StatusCode.ERROR, str(exception)))
    
    def get_active_spans(self) -> List[Dict[str, Any]]:
        """Get information about currently active spans."""
        spans_info = []
        
        for span_id, span in self.active_spans.items():
            if span.is_recording():
                span_context = span.get_span_context()
                spans_info.append({
                    "span_id": span_id,
                    "trace_id": format(span_context.trace_id, '032x'),
                    "name": span.name,
                    "start_time": span.start_time,
                    "is_recording": span.is_recording()
                })
        
        return spans_info
    
    def add_custom_processor(self, processor: Callable[[trace.Span], None]) -> None:
        """Add a custom span processor."""
        self.custom_processors.append(processor)
    
    async def flush_spans(self) -> None:
        """Flush all pending spans."""
        for processor in self.processors:
            if hasattr(processor, 'force_flush'):
                processor.force_flush()
    
    def get_tracing_stats(self) -> Dict[str, Any]:
        """Get tracing statistics and health information."""
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "backends": [b.value for b in self.backends],
            "sampling_rate": self.sampling_rate,
            "active_spans": len(self.active_spans),
            "exporters": len(self.exporters),
            "processors": len(self.processors),
            "custom_processors": len(self.custom_processors)
        }
    
    async def shutdown(self) -> None:
        """Shutdown tracing and flush remaining spans."""
        try:
            # Flush all pending spans
            await self.flush_spans()
            
            # Shutdown processors
            for processor in self.processors:
                if hasattr(processor, 'shutdown'):
                    processor.shutdown()
            
            logger.info("Tracing manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during tracing shutdown: {e}")


# ML-specific tracing utilities
class MLTracer:
    """Specialized tracer for ML operations."""
    
    def __init__(self, tracing_manager: TracingManager):
        self.tracing_manager = tracing_manager
    
    @contextmanager
    def trace_data_preprocessing(self, dataset_id: str, operation: str):
        """Trace data preprocessing operations."""
        metadata = SpanMetadata(
            dataset_id=dataset_id,
            custom_attributes={"operation": operation}
        )
        
        with self.tracing_manager.start_span(
            name=f"data.preprocessing.{operation}",
            kind=SpanKind.DATA_PROCESSING,
            metadata=metadata
        ) as span:
            yield span
    
    @contextmanager
    def trace_model_evaluation(self, model_name: str, model_version: str, metrics: Dict[str, float]):
        """Trace model evaluation operations."""
        metadata = SpanMetadata(
            model_name=model_name,
            model_version=model_version,
            custom_attributes=metrics
        )
        
        with self.tracing_manager.start_span(
            name=f"ml.evaluation.{model_name}",
            kind=SpanKind.ML_EVALUATION,
            metadata=metadata
        ) as span:
            # Add evaluation metrics as span attributes
            for metric_name, metric_value in metrics.items():
                span.set_attribute(f"evaluation.{metric_name}", metric_value)
            
            yield span
    
    async def trace_batch_inference(
        self,
        model_name: str,
        model_version: str,
        batch_size: int,
        inference_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Trace batch inference operations."""
        metadata = SpanMetadata(
            model_name=model_name,
            model_version=model_version,
            custom_attributes={"batch_size": batch_size}
        )
        
        async with self.tracing_manager.start_async_span(
            name=f"ml.inference.batch.{model_name}",
            kind=SpanKind.ML_INFERENCE,
            metadata=metadata
        ) as span:
            span.set_attribute("inference.type", "batch")
            span.set_attribute("inference.batch_size", batch_size)
            
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(inference_func):
                result = await inference_func(*args, **kwargs)
            else:
                result = inference_func(*args, **kwargs)
            
            duration = time.time() - start_time
            throughput = batch_size / duration if duration > 0 else 0
            
            span.set_attribute("inference.duration_seconds", duration)
            span.set_attribute("inference.throughput_per_second", throughput)
            
            return result