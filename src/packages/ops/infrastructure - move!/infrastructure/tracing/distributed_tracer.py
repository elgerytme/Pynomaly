#!/usr/bin/env python3
"""
Distributed Tracing System for Pynomaly

This module provides comprehensive distributed tracing capabilities using
OpenTelemetry for microservices observability and performance monitoring.
"""

import asyncio
import functools
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.http import get_excluded_urls


class SpanType(Enum):
    """Types of spans for categorization."""

    HTTP_REQUEST = "http.request"
    HTTP_RESPONSE = "http.response"
    DATABASE_QUERY = "db.query"
    CACHE_OPERATION = "cache.operation"
    MESSAGE_QUEUE = "mq.operation"
    ML_INFERENCE = "ml.inference"
    DATA_PROCESSING = "data.processing"
    AUTHENTICATION = "auth.operation"
    BUSINESS_LOGIC = "business.logic"
    EXTERNAL_API = "external.api"


class TracingBackend(Enum):
    """Supported tracing backends."""

    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    OTLP = "otlp"
    CONSOLE = "console"


@dataclass
class TraceContext:
    """Distributed trace context."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    flags: int = 1
    baggage: dict[str, str] = field(default_factory=dict)

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP headers for propagation."""
        headers = {}

        # W3C Trace Context format
        traceparent = f"00-{self.trace_id}-{self.span_id}-{self.flags:02d}"
        headers["traceparent"] = traceparent

        if self.baggage:
            baggage_str = ",".join(f"{k}={v}" for k, v in self.baggage.items())
            headers["baggage"] = baggage_str

        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> Optional["TraceContext"]:
        """Create trace context from HTTP headers."""
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) != 4 or parts[0] != "00":
                return None

            trace_id = parts[1]
            span_id = parts[2]
            flags = int(parts[3], 16)

            baggage = {}
            baggage_header = headers.get("baggage", "")
            if baggage_header:
                for item in baggage_header.split(","):
                    if "=" in item:
                        key, value = item.split("=", 1)
                        baggage[key.strip()] = value.strip()

            return cls(trace_id=trace_id, span_id=span_id, flags=flags, baggage=baggage)
        except (ValueError, IndexError):
            return None


@dataclass
class SpanMetrics:
    """Metrics collected for spans."""

    duration_ms: float
    error_count: int = 0
    success_count: int = 0
    span_count: int = 0

    def add_span(self, duration_ms: float, success: bool = True):
        """Add span metrics."""
        self.duration_ms += duration_ms
        self.span_count += 1

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    @property
    def average_duration(self) -> float:
        """Calculate average duration."""
        return self.duration_ms / self.span_count if self.span_count > 0 else 0

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total = self.success_count + self.error_count
        return self.error_count / total if total > 0 else 0


class DistributedTracer:
    """Main distributed tracing system."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.service_name = config.get("service_name", "monorepo")
        self.service_version = config.get("service_version", "1.0.0")
        self.environment = config.get("environment", "production")

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Metrics collection
        self.span_metrics: dict[str, SpanMetrics] = {}

        # Setup OpenTelemetry
        self._setup_tracer_provider()
        self._setup_exporters()
        self._setup_instrumentations()

        # Get tracer
        self.tracer = trace.get_tracer(
            instrumenting_module_name=__name__, instrumenting_library_version="1.0.0"
        )

    def _setup_tracer_provider(self):
        """Setup OpenTelemetry tracer provider."""
        resource = Resource.create(
            {
                "service.name": self.service_name,
                "service.version": self.service_version,
                "service.instance.id": str(uuid.uuid4()),
                "deployment.environment": self.environment,
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python",
                "telemetry.sdk.version": "1.0.0",
            }
        )

        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

    def _setup_exporters(self):
        """Setup trace exporters based on configuration."""
        provider = trace.get_tracer_provider()
        backends = self.config.get("backends", ["console"])

        for backend in backends:
            exporter = None

            if backend == TracingBackend.JAEGER.value:
                jaeger_config = self.config.get("jaeger", {})
                exporter = JaegerExporter(
                    agent_host_name=jaeger_config.get("agent_host", "localhost"),
                    agent_port=jaeger_config.get("agent_port", 6831),
                    collector_endpoint=jaeger_config.get("collector_endpoint"),
                )

            elif backend == TracingBackend.ZIPKIN.value:
                zipkin_config = self.config.get("zipkin", {})
                exporter = ZipkinExporter(
                    endpoint=zipkin_config.get(
                        "endpoint", "http://localhost:9411/api/v2/spans"
                    )
                )

            elif backend == TracingBackend.OTLP.value:
                otlp_config = self.config.get("otlp", {})
                exporter = OTLPSpanExporter(
                    endpoint=otlp_config.get("endpoint", "http://localhost:4317"),
                    headers=otlp_config.get("headers", {}),
                )

            elif backend == TracingBackend.CONSOLE.value:
                exporter = ConsoleSpanExporter()

            if exporter:
                processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=self.config.get("max_queue_size", 2048),
                    max_export_batch_size=self.config.get("max_export_batch_size", 512),
                    export_timeout_millis=self.config.get(
                        "export_timeout_millis", 30000
                    ),
                )
                provider.add_span_processor(processor)
                self.logger.info(f"Added {backend} trace exporter")

    def _setup_instrumentations(self):
        """Setup automatic instrumentations."""
        instrumentations = self.config.get("instrumentations", {})

        # HTTP requests instrumentation
        if instrumentations.get("requests", True):
            RequestsInstrumentor().instrument(
                excluded_urls=get_excluded_urls("OTEL_PYTHON_REQUESTS_EXCLUDED_URLS")
            )
            self.logger.info("Enabled requests instrumentation")

        # Database instrumentation
        if instrumentations.get("sqlalchemy", True):
            SQLAlchemyInstrumentor().instrument()
            self.logger.info("Enabled SQLAlchemy instrumentation")

        # Redis instrumentation
        if instrumentations.get("redis", True):
            RedisInstrumentor().instrument()
            self.logger.info("Enabled Redis instrumentation")

        # Celery instrumentation
        if instrumentations.get("celery", True):
            CeleryInstrumentor().instrument()
            self.logger.info("Enabled Celery instrumentation")

    def create_span(
        self,
        name: str,
        span_type: SpanType = SpanType.BUSINESS_LOGIC,
        parent_context: Any | None = None,
        attributes: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> trace.Span:
        """Create a new span."""
        span = self.tracer.start_span(name=name, context=parent_context, kind=kind)

        # Add span type attribute
        span.set_attribute("span.type", span_type.value)

        # Add service attributes
        span.set_attribute("service.name", self.service_name)
        span.set_attribute("service.version", self.service_version)
        span.set_attribute("service.environment", self.environment)

        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        # Add timestamp
        span.set_attribute("span.start_time", datetime.utcnow().isoformat())

        return span

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        span_type: SpanType = SpanType.BUSINESS_LOGIC,
        attributes: dict[str, Any] | None = None,
        record_metrics: bool = True,
    ):
        """Context manager for tracing operations."""
        start_time = time.time()

        with self.tracer.start_as_current_span(operation_name) as span:
            try:
                # Set span type
                span.set_attribute("span.type", span_type.value)

                # Add service attributes
                span.set_attribute("service.name", self.service_name)
                span.set_attribute("service.version", self.service_version)

                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                yield span

                # Mark as successful
                span.set_status(Status(StatusCode.OK))

                # Record metrics
                if record_metrics:
                    duration_ms = (time.time() - start_time) * 1000
                    self._record_span_metrics(span_type.value, duration_ms, True)

            except Exception as e:
                # Record error
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                # Add error attributes
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))

                # Record metrics
                if record_metrics:
                    duration_ms = (time.time() - start_time) * 1000
                    self._record_span_metrics(span_type.value, duration_ms, False)

                raise

            finally:
                # Add duration
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("duration_ms", duration_ms)

    def trace_function(
        self,
        span_name: str | None = None,
        span_type: SpanType = SpanType.BUSINESS_LOGIC,
        attributes: dict[str, Any] | None = None,
        record_metrics: bool = True,
    ):
        """Decorator for tracing functions."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                operation_name = span_name or f"{func.__module__}.{func.__name__}"

                func_attributes = {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                    "function.args_count": len(args),
                    "function.kwargs_count": len(kwargs),
                }

                if attributes:
                    func_attributes.update(attributes)

                with self.trace_operation(
                    operation_name, span_type, func_attributes, record_metrics
                ) as span:
                    result = func(*args, **kwargs)

                    # Add result metadata if available
                    if hasattr(result, "__len__"):
                        try:
                            span.set_attribute("result.length", len(result))
                        except:
                            pass

                    return result

            return wrapper

        return decorator

    def trace_async_function(
        self,
        span_name: str | None = None,
        span_type: SpanType = SpanType.BUSINESS_LOGIC,
        attributes: dict[str, Any] | None = None,
        record_metrics: bool = True,
    ):
        """Decorator for tracing async functions."""

        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                operation_name = span_name or f"{func.__module__}.{func.__name__}"

                func_attributes = {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                    "function.args_count": len(args),
                    "function.kwargs_count": len(kwargs),
                    "function.async": True,
                }

                if attributes:
                    func_attributes.update(attributes)

                with self.trace_operation(
                    operation_name, span_type, func_attributes, record_metrics
                ) as span:
                    result = await func(*args, **kwargs)

                    # Add result metadata if available
                    if hasattr(result, "__len__"):
                        try:
                            span.set_attribute("result.length", len(result))
                        except:
                            pass

                    return result

            return wrapper

        return decorator

    def trace_http_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        status_code: int | None = None,
        response_size: int | None = None,
    ) -> trace.Span:
        """Create span for HTTP request."""
        span_name = f"{method.upper()} {url}"

        attributes = {
            "http.method": method.upper(),
            "http.url": url,
            "http.scheme": "https" if url.startswith("https") else "http",
        }

        if headers:
            # Add select headers (avoid sensitive data)
            safe_headers = ["content-type", "user-agent", "accept"]
            for header in safe_headers:
                if header in headers:
                    attributes[f"http.request.header.{header}"] = headers[header]

        if status_code:
            attributes["http.status_code"] = status_code

        if response_size:
            attributes["http.response.size"] = response_size

        span = self.create_span(
            span_name,
            SpanType.HTTP_REQUEST,
            attributes=attributes,
            kind=SpanKind.CLIENT,
        )

        return span

    def trace_database_query(
        self,
        query: str,
        database: str,
        table: str | None = None,
        operation: str | None = None,
        rows_affected: int | None = None,
    ) -> trace.Span:
        """Create span for database query."""
        span_name = f"db.{operation or 'query'}"

        attributes = {
            "db.system": "postgresql",  # or dynamic based on config
            "db.name": database,
            "db.statement": query[:500],  # Truncate long queries
        }

        if table:
            attributes["db.sql.table"] = table

        if operation:
            attributes["db.operation"] = operation

        if rows_affected is not None:
            attributes["db.rows_affected"] = rows_affected

        span = self.create_span(
            span_name,
            SpanType.DATABASE_QUERY,
            attributes=attributes,
            kind=SpanKind.CLIENT,
        )

        return span

    def trace_ml_inference(
        self,
        model_name: str,
        model_version: str | None = None,
        input_shape: tuple | None = None,
        prediction_type: str | None = None,
        confidence_score: float | None = None,
    ) -> trace.Span:
        """Create span for ML inference."""
        span_name = f"ml.inference.{model_name}"

        attributes = {"ml.model.name": model_name, "ml.task": "inference"}

        if model_version:
            attributes["ml.model.version"] = model_version

        if input_shape:
            attributes["ml.input.shape"] = str(input_shape)

        if prediction_type:
            attributes["ml.prediction.type"] = prediction_type

        if confidence_score is not None:
            attributes["ml.prediction.confidence"] = confidence_score

        span = self.create_span(span_name, SpanType.ML_INFERENCE, attributes=attributes)

        return span

    def inject_trace_context(self, headers: dict[str, str]):
        """Inject trace context into headers for propagation."""
        inject(headers)

    def extract_trace_context(self, headers: dict[str, str]) -> Any:
        """Extract trace context from headers."""
        return extract(headers)

    def get_current_trace_id(self) -> str | None:
        """Get current trace ID."""
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, "032x")
        return None

    def get_current_span_id(self) -> str | None:
        """Get current span ID."""
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().span_id, "016x")
        return None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        """Add event to current span."""
        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes or {})

    def set_baggage(self, key: str, value: str):
        """Set baggage item for trace propagation."""
        # Implementation depends on OpenTelemetry baggage API
        pass

    def get_baggage(self, key: str) -> str | None:
        """Get baggage item from current trace."""
        # Implementation depends on OpenTelemetry baggage API
        pass

    def _record_span_metrics(self, span_type: str, duration_ms: float, success: bool):
        """Record span metrics."""
        if span_type not in self.span_metrics:
            self.span_metrics[span_type] = SpanMetrics(duration_ms=0)

        self.span_metrics[span_type].add_span(duration_ms, success)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get tracing metrics summary."""
        summary = {}

        for span_type, metrics in self.span_metrics.items():
            summary[span_type] = {
                "total_spans": metrics.span_count,
                "success_count": metrics.success_count,
                "error_count": metrics.error_count,
                "error_rate": metrics.error_rate,
                "average_duration_ms": metrics.average_duration,
                "total_duration_ms": metrics.duration_ms,
            }

        return summary

    def health_check(self) -> dict[str, Any]:
        """Perform tracing system health check."""
        health = {
            "status": "healthy",
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "backends": self.config.get("backends", []),
            "instrumentations_enabled": self.config.get("instrumentations", {}),
            "current_trace_id": self.get_current_trace_id(),
            "metrics_summary": self.get_metrics_summary(),
            "issues": [],
        }

        # Check if tracer is working
        try:
            with self.tracer.start_span("health_check") as span:
                span.set_attribute("test", True)
        except Exception as e:
            health["issues"].append(f"Tracer error: {str(e)}")
            health["status"] = "degraded"

        return health


# Global tracer instance
_distributed_tracer = None


def initialize_distributed_tracing(config: dict[str, Any]) -> DistributedTracer:
    """Initialize distributed tracing system."""
    global _distributed_tracer
    _distributed_tracer = DistributedTracer(config)
    return _distributed_tracer


def get_tracer() -> DistributedTracer:
    """Get global distributed tracer."""
    if _distributed_tracer is None:
        raise RuntimeError(
            "Distributed tracing not initialized. Call initialize_distributed_tracing() first."
        )
    return _distributed_tracer


def trace_operation(
    operation_name: str,
    span_type: SpanType = SpanType.BUSINESS_LOGIC,
    attributes: dict[str, Any] | None = None,
):
    """Context manager for tracing operations (convenience function)."""
    tracer = get_tracer()
    return tracer.trace_operation(operation_name, span_type, attributes)


def trace_function(
    span_name: str | None = None,
    span_type: SpanType = SpanType.BUSINESS_LOGIC,
    attributes: dict[str, Any] | None = None,
):
    """Decorator for tracing functions (convenience function)."""
    tracer = get_tracer()
    return tracer.trace_function(span_name, span_type, attributes)


def trace_async_function(
    span_name: str | None = None,
    span_type: SpanType = SpanType.BUSINESS_LOGIC,
    attributes: dict[str, Any] | None = None,
):
    """Decorator for tracing async functions (convenience function)."""
    tracer = get_tracer()
    return tracer.trace_async_function(span_name, span_type, attributes)


async def main():
    """Main function for testing."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Testing distributed tracing system")

    # Configuration
    config = {
        "service_name": "monorepo",
        "service_version": "1.0.0",
        "environment": "test",
        "backends": ["console", "jaeger"],
        "jaeger": {"agent_host": "localhost", "agent_port": 6831},
        "instrumentations": {
            "requests": True,
            "sqlalchemy": True,
            "redis": True,
            "celery": False,
        },
        "max_queue_size": 2048,
        "max_export_batch_size": 512,
        "export_timeout_millis": 30000,
    }

    # Initialize distributed tracing
    tracer = initialize_distributed_tracing(config)

    # Test basic tracing
    with trace_operation(
        "test_operation", SpanType.BUSINESS_LOGIC, {"test": "value"}
    ) as span:
        logger.info("Inside traced operation")
        span.add_event("processing_started")

        # Simulate some work
        await asyncio.sleep(0.1)

        span.add_event("processing_completed")

    # Test function tracing
    @trace_function("test_function", SpanType.DATA_PROCESSING)
    def process_data(data_size: int) -> int:
        """Simulate data processing."""
        time.sleep(0.05)
        return data_size * 2

    result = process_data(100)
    logger.info(f"Processing result: {result}")

    # Test async function tracing
    @trace_async_function(
        "async_ml_inference", SpanType.ML_INFERENCE, {"model": "test_model"}
    )
    async def run_ml_inference(input_data: list[float]) -> float:
        """Simulate ML inference."""
        await asyncio.sleep(0.1)
        return sum(input_data) / len(input_data)

    inference_result = await run_ml_inference([1.0, 2.0, 3.0, 4.0, 5.0])
    logger.info(f"ML inference result: {inference_result}")

    # Test HTTP request tracing
    with tracer.trace_operation("http_request", SpanType.HTTP_REQUEST) as span:
        span.set_attribute("http.method", "GET")
        span.set_attribute("http.url", "https://api.example.com/data")
        span.set_attribute("http.status_code", 200)

        await asyncio.sleep(0.2)  # Simulate network request

    # Test database query tracing
    with tracer.trace_operation("database_query", SpanType.DATABASE_QUERY) as span:
        span.set_attribute("db.system", "postgresql")
        span.set_attribute("db.name", "pynomaly_prod")
        span.set_attribute("db.statement", "SELECT * FROM users WHERE active = true")
        span.set_attribute("db.rows_affected", 42)

        await asyncio.sleep(0.05)  # Simulate database query

    # Test error tracing
    try:
        with trace_operation("error_operation", SpanType.BUSINESS_LOGIC) as span:
            span.add_event("about_to_error")
            raise ValueError("Test error for tracing")
    except ValueError:
        logger.info("Caught expected error")

    # Get current trace info
    trace_id = tracer.get_current_trace_id()
    span_id = tracer.get_current_span_id()
    logger.info(f"Current trace: {trace_id}, span: {span_id}")

    # Get metrics summary
    metrics = tracer.get_metrics_summary()
    logger.info(f"Tracing metrics: {metrics}")

    # Health check
    health = tracer.health_check()
    logger.info(f"Tracing health: {health['status']}")

    logger.info("Distributed tracing testing completed")


if __name__ == "__main__":
    asyncio.run(main())
