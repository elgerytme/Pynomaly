"""Distributed tracing and span management system."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import opentelemetry
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class SpanStatus(Enum):
    """Span status enumeration."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanAttribute:
    """Span attribute with type information."""
    key: str
    value: Any
    type: str = field(init=False)

    def __post_init__(self):
        self.type = type(self.value).__name__


@dataclass
class SpanEvent:
    """Span event with timestamp and attributes."""
    name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes
        }


@dataclass
class TraceContext:
    """Trace context with correlation information."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str | None = None
    flags: int = 1  # Sampled
    baggage: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "flags": self.flags,
            "baggage": self.baggage
        }


@dataclass
class Span:
    """Comprehensive span implementation."""

    # Basic span information
    trace_id: str
    span_id: str
    operation_name: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    parent_span_id: str | None = None

    # Span metadata
    service_name: str = "pynomaly"
    component: str | None = None
    span_kind: str = "internal"  # internal, server, client, producer, consumer

    # Status and errors
    status: SpanStatus = SpanStatus.UNSET
    status_message: str | None = None

    # Data
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)

    # Performance
    duration_ms: float | None = None

    def __post_init__(self):
        """Initialize span with default attributes."""
        self.attributes.update({
            "service.name": self.service_name,
            "span.kind": self.span_kind,
            "component": self.component or "unknown"
        })

    def set_attribute(self, key: str, value: Any) -> Span:
        """Set span attribute."""
        self.attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> Span:
        """Set multiple span attributes."""
        self.attributes.update(attributes)
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        """Add event to span."""
        event = SpanEvent(name=name, attributes=attributes or {})
        self.events.append(event)
        return self

    def add_log(self, level: str, message: str, **kwargs) -> Span:
        """Add log entry to span."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
        return self

    def set_status(self, status: SpanStatus, message: str | None = None) -> Span:
        """Set span status."""
        self.status = status
        self.status_message = message
        return self

    def set_error(self, error: Exception) -> Span:
        """Set span error status with exception details."""
        self.status = SpanStatus.ERROR
        self.status_message = str(error)
        self.attributes.update({
            "error": True,
            "error.type": type(error).__name__,
            "error.message": str(error),
            "error.stack": str(error.__traceback__) if error.__traceback__ else None
        })
        return self

    def finish(self) -> Span:
        """Finish the span."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
            if self.start_time:
                self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "parent_span_id": self.parent_span_id,
            "service_name": self.service_name,
            "component": self.component,
            "span_kind": self.span_kind,
            "status": self.status.value,
            "status_message": self.status_message,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [event.to_dict() for event in self.events],
            "logs": self.logs
        }


class TracingManager:
    """Comprehensive distributed tracing manager."""

    def __init__(
        self,
        service_name: str = "pynomaly",
        jaeger_endpoint: str | None = None,
        sampling_rate: float = 1.0,
        max_spans_in_memory: int = 1000,
        enable_console_export: bool = False,
        storage_path: Path | None = None,
        flush_interval_seconds: int = 30
    ):
        """Initialize tracing manager.

        Args:
            service_name: Name of the service
            jaeger_endpoint: Jaeger collector endpoint
            sampling_rate: Sampling rate (0.0 to 1.0)
            max_spans_in_memory: Maximum spans to keep in memory
            enable_console_export: Whether to export to console
            storage_path: Path to store traces
            flush_interval_seconds: How often to flush traces
        """
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.sampling_rate = sampling_rate
        self.max_spans_in_memory = max_spans_in_memory
        self.enable_console_export = enable_console_export
        self.storage_path = storage_path
        self.flush_interval_seconds = flush_interval_seconds

        # Tracing state
        self._spans: dict[str, Span] = {}
        self._current_span: Span | None = None
        self._trace_context: TraceContext | None = None
        self._lock = threading.RLock()

        # OpenTelemetry integration
        self._tracer_provider: TracerProvider | None = None
        self._tracer = None

        # Performance tracking
        self.stats = {
            "spans_created": 0,
            "spans_finished": 0,
            "spans_exported": 0,
            "export_errors": 0,
            "traces_created": 0
        }

        # Initialize tracing
        self._initialize_tracing()

    def _initialize_tracing(self):
        """Initialize tracing infrastructure."""
        if OPENTELEMETRY_AVAILABLE:
            self._setup_opentelemetry()
        else:
            print("OpenTelemetry not available, using basic tracing")

    def _setup_opentelemetry(self):
        """Set up OpenTelemetry tracing."""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0"
            })

            # Create tracer provider
            self._tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(self._tracer_provider)

            # Add exporters
            if self.enable_console_export:
                console_exporter = ConsoleSpanExporter()
                console_processor = BatchSpanProcessor(console_exporter)
                self._tracer_provider.add_span_processor(console_processor)

            if self.jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=14268,
                    collector_endpoint=self.jaeger_endpoint
                )
                jaeger_processor = BatchSpanProcessor(jaeger_exporter)
                self._tracer_provider.add_span_processor(jaeger_processor)

            # Get tracer
            self._tracer = trace.get_tracer(__name__)

        except Exception as e:
            print(f"Failed to setup OpenTelemetry: {e}")
            self._tracer = None

    def start_trace(self, operation_name: str, trace_context: TraceContext | None = None) -> TraceContext:
        """Start a new trace."""
        if trace_context is None:
            trace_context = TraceContext()

        with self._lock:
            self._trace_context = trace_context
            self.stats["traces_created"] += 1

        # Start root span
        self.start_span(operation_name, trace_context=trace_context)

        return trace_context

    def start_span(
        self,
        operation_name: str,
        parent_span: Span | None = None,
        trace_context: TraceContext | None = None,
        component: str | None = None,
        span_kind: str = "internal",
        attributes: dict[str, Any] | None = None
    ) -> Span:
        """Start a new span."""

        # Determine trace and parent information
        if trace_context:
            trace_id = trace_context.trace_id
            parent_span_id = trace_context.parent_span_id
        elif parent_span:
            trace_id = parent_span.trace_id
            parent_span_id = parent_span.span_id
        elif self._current_span:
            trace_id = self._current_span.trace_id
            parent_span_id = self._current_span.span_id
        else:
            # Create new trace
            trace_context = TraceContext()
            trace_id = trace_context.trace_id
            parent_span_id = None

        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            operation_name=operation_name,
            parent_span_id=parent_span_id,
            service_name=self.service_name,
            component=component,
            span_kind=span_kind
        )

        # Set attributes
        if attributes:
            span.set_attributes(attributes)

        with self._lock:
            self._spans[span.span_id] = span
            self._current_span = span
            self.stats["spans_created"] += 1

        return span

    def finish_span(self, span: Span) -> Span:
        """Finish a span."""
        span.finish()

        with self._lock:
            if span.span_id in self._spans:
                self.stats["spans_finished"] += 1

                # Set current span to parent if this was current
                if self._current_span and self._current_span.span_id == span.span_id:
                    # Find parent span
                    parent_span = None
                    if span.parent_span_id:
                        parent_span = self._spans.get(span.parent_span_id)
                    self._current_span = parent_span

        return span

    def get_current_span(self) -> Span | None:
        """Get the current active span."""
        return self._current_span

    def get_trace_context(self) -> TraceContext | None:
        """Get the current trace context."""
        return self._trace_context

    @contextmanager
    def span(
        self,
        operation_name: str,
        component: str | None = None,
        span_kind: str = "internal",
        attributes: dict[str, Any] | None = None
    ) -> Generator[Span, None, None]:
        """Context manager for automatic span management."""
        span = self.start_span(
            operation_name=operation_name,
            component=component,
            span_kind=span_kind,
            attributes=attributes
        )

        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.finish_span(span)

    def inject_context(self, span: Span) -> dict[str, str]:
        """Inject trace context into headers."""
        return {
            "x-trace-id": span.trace_id,
            "x-span-id": span.span_id,
            "x-parent-span-id": span.parent_span_id or "",
            "x-service-name": span.service_name
        }

    def extract_context(self, headers: dict[str, str]) -> TraceContext | None:
        """Extract trace context from headers."""
        trace_id = headers.get("x-trace-id")
        if not trace_id:
            return None

        return TraceContext(
            trace_id=trace_id,
            span_id=headers.get("x-span-id", str(uuid.uuid4())),
            parent_span_id=headers.get("x-parent-span-id") or None
        )

    def get_all_spans(self) -> list[Span]:
        """Get all spans."""
        with self._lock:
            return list(self._spans.values())

    def get_spans_by_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a specific trace."""
        with self._lock:
            return [span for span in self._spans.values() if span.trace_id == trace_id]

    def get_span_by_id(self, span_id: str) -> Span | None:
        """Get span by ID."""
        with self._lock:
            return self._spans.get(span_id)

    def export_traces(self) -> list[dict[str, Any]]:
        """Export all traces as dictionaries."""
        with self._lock:
            # Group spans by trace ID
            traces = {}
            for span in self._spans.values():
                if span.trace_id not in traces:
                    traces[span.trace_id] = []
                traces[span.trace_id].append(span.to_dict())

            self.stats["spans_exported"] += len(self._spans)

            return [
                {
                    "trace_id": trace_id,
                    "spans": spans,
                    "span_count": len(spans)
                }
                for trace_id, spans in traces.items()
            ]

    def flush_traces(self):
        """Flush traces to storage."""
        if not self.storage_path:
            return

        traces = self.export_traces()
        if not traces:
            return

        try:
            # Ensure storage directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = self.storage_path.parent / f"traces_{timestamp}.json"

            # Write traces to file
            with open(filename, 'w') as f:
                json.dump({
                    "timestamp": datetime.utcnow().isoformat(),
                    "service_name": self.service_name,
                    "trace_count": len(traces),
                    "traces": traces
                }, f, indent=2)

        except Exception as e:
            self.stats["export_errors"] += 1
            print(f"Error flushing traces: {e}")

    def clear_finished_spans(self):
        """Clear finished spans from memory to prevent memory leaks."""
        with self._lock:
            finished_spans = [
                span_id for span_id, span in self._spans.items()
                if span.end_time is not None
            ]

            # Keep only recent spans if we have too many
            if len(finished_spans) > self.max_spans_in_memory:
                # Sort by end time and keep most recent
                sorted_spans = sorted(
                    [(span_id, self._spans[span_id]) for span_id in finished_spans],
                    key=lambda x: x[1].end_time or datetime.min,
                    reverse=True
                )

                # Remove oldest spans
                to_remove = sorted_spans[self.max_spans_in_memory:]
                for span_id, _ in to_remove:
                    del self._spans[span_id]

    def get_stats(self) -> dict[str, Any]:
        """Get tracing statistics."""
        with self._lock:
            return {
                "service_name": self.service_name,
                "spans_in_memory": len(self._spans),
                "current_span": self._current_span.operation_name if self._current_span else None,
                "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
                **self.stats
            }

    def shutdown(self):
        """Shutdown tracing manager."""
        # Flush remaining traces
        try:
            self.flush_traces()
        except Exception:
            pass

        # Shutdown OpenTelemetry
        if self._tracer_provider:
            try:
                self._tracer_provider.shutdown()
            except Exception:
                pass


def trace_decorator(
    operation_name: str | None = None,
    component: str | None = None,
    span_kind: str = "internal",
    tracer: TracingManager | None = None
):
    """Decorator for automatic tracing of functions."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Get tracer (use provided or create default)
            nonlocal tracer
            if tracer is None:
                tracer = get_default_tracer()

            # Determine operation name
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            # Execute with tracing
            with tracer.span(
                operation_name=op_name,
                component=component,
                span_kind=span_kind,
                attributes={
                    "function.name": func.__name__,
                    "function.module": func.__module__
                }
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(SpanStatus.OK)
                    return result
                except Exception as e:
                    span.set_error(e)
                    raise

        return wrapper
    return decorator


# Global tracer instance
_default_tracer: TracingManager | None = None


def get_default_tracer() -> TracingManager:
    """Get or create the default tracer."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = TracingManager()
    return _default_tracer


def configure_tracing(
    service_name: str = "pynomaly",
    jaeger_endpoint: str | None = None,
    sampling_rate: float = 1.0,
    storage_path: Path | None = None,
    enable_console_export: bool = False
) -> TracingManager:
    """Configure the global tracer."""
    global _default_tracer
    _default_tracer = TracingManager(
        service_name=service_name,
        jaeger_endpoint=jaeger_endpoint,
        sampling_rate=sampling_rate,
        storage_path=storage_path,
        enable_console_export=enable_console_export
    )
    return _default_tracer
