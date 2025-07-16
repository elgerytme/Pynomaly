#!/usr/bin/env python3
"""
Simplified Distributed Tracing System for Pynomaly

This module provides basic distributed tracing capabilities without external dependencies
for development and testing environments.
"""

import asyncio
import functools
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SpanKind(Enum):
    """Span kinds for categorization."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Span status enumeration."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TraceContext:
    """Trace context for distributed tracing."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP headers."""
        headers = {"x-trace-id": self.trace_id, "x-span-id": self.span_id}
        if self.parent_span_id:
            headers["x-parent-span-id"] = self.parent_span_id

        if self.baggage:
            baggage_str = ",".join(f"{k}={v}" for k, v in self.baggage.items())
            headers["x-baggage"] = baggage_str

        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> Optional["TraceContext"]:
        """Create from HTTP headers."""
        trace_id = headers.get("x-trace-id")
        span_id = headers.get("x-span-id")

        if not trace_id or not span_id:
            return None

        parent_span_id = headers.get("x-parent-span-id")

        baggage = {}
        baggage_header = headers.get("x-baggage", "")
        if baggage_header:
            for item in baggage_header.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    baggage[key.strip()] = value.strip()

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage,
        )


@dataclass
class Span:
    """Simplified span implementation."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: datetime
    end_time: datetime | None = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        """Add span event."""
        event = {
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        }
        self.events.append(event)

    def set_status(self, status: SpanStatus, description: str | None = None):
        """Set span status."""
        self.status = status
        if description:
            self.attributes["status.description"] = description

    def finish(self):
        """Finish the span."""
        self.end_time = datetime.utcnow()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "kind": self.kind.value,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
        }


class SimpleTracer:
    """Simplified distributed tracer."""

    def __init__(self, service_name: str = "monorepo"):
        self.service_name = service_name
        self.spans: list[Span] = []
        self.current_span: Span | None = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Metrics
        self.metrics = {
            "total_spans": 0,
            "successful_spans": 0,
            "error_spans": 0,
            "total_duration_ms": 0,
        }

    def start_span(
        self,
        operation_name: str,
        parent_context: TraceContext | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Span:
        """Start a new span."""
        trace_id = parent_context.trace_id if parent_context else self._generate_id()
        span_id = self._generate_id()
        parent_span_id = parent_context.span_id if parent_context else None

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            kind=kind,
        )

        # Set service attributes
        span.set_attribute("service.name", self.service_name)
        span.set_attribute("span.kind", kind.value)

        self.spans.append(span)
        self.metrics["total_spans"] += 1

        return span

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ):
        """Context manager for tracing operations."""
        span = self.start_span(operation_name, kind=kind)
        old_current = self.current_span
        self.current_span = span

        try:
            # Add custom attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            yield span

            # Mark as successful
            span.set_status(SpanStatus.OK)
            self.metrics["successful_spans"] += 1

        except Exception as e:
            # Record error
            span.set_status(SpanStatus.ERROR, str(e))
            span.set_attribute("error", True)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))

            self.metrics["error_spans"] += 1

            raise

        finally:
            span.finish()
            self.metrics["total_duration_ms"] += span.duration_ms
            self.current_span = old_current

            # Log span completion
            self.logger.debug(
                f"Span completed: {operation_name} "
                f"({span.duration_ms:.2f}ms, {span.status.value})"
            )

    def trace_function(
        self,
        operation_name: str | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ):
        """Decorator for tracing functions."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span_name = operation_name or f"{func.__module__}.{func.__name__}"

                func_attributes = {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                }

                if attributes:
                    func_attributes.update(attributes)

                with self.trace_operation(span_name, kind, func_attributes) as span:
                    result = func(*args, **kwargs)

                    # Add result metadata
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
        operation_name: str | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ):
        """Decorator for tracing async functions."""

        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                span_name = operation_name or f"{func.__module__}.{func.__name__}"

                func_attributes = {
                    "function.name": func.__name__,
                    "function.module": func.__module__,
                    "function.async": True,
                }

                if attributes:
                    func_attributes.update(attributes)

                with self.trace_operation(span_name, kind, func_attributes) as span:
                    result = await func(*args, **kwargs)

                    # Add result metadata
                    if hasattr(result, "__len__"):
                        try:
                            span.set_attribute("result.length", len(result))
                        except:
                            pass

                    return result

            return wrapper

        return decorator

    def get_current_trace_id(self) -> str | None:
        """Get current trace ID."""
        return self.current_span.trace_id if self.current_span else None

    def get_current_span_id(self) -> str | None:
        """Get current span ID."""
        return self.current_span.span_id if self.current_span else None

    def create_trace_context(self) -> TraceContext | None:
        """Create trace context for propagation."""
        if not self.current_span:
            return None

        return TraceContext(
            trace_id=self.current_span.trace_id, span_id=self.current_span.span_id
        )

    def get_traces(self) -> list[dict[str, Any]]:
        """Get all completed traces."""
        # Group spans by trace_id
        traces = {}

        for span in self.spans:
            if span.trace_id not in traces:
                traces[span.trace_id] = []
            traces[span.trace_id].append(span.to_dict())

        return [
            {
                "trace_id": trace_id,
                "spans": spans,
                "span_count": len(spans),
                "total_duration_ms": sum(s["duration_ms"] for s in spans),
            }
            for trace_id, spans in traces.items()
        ]

    def get_metrics(self) -> dict[str, Any]:
        """Get tracing metrics."""
        avg_duration = (
            self.metrics["total_duration_ms"] / self.metrics["total_spans"]
            if self.metrics["total_spans"] > 0
            else 0
        )

        error_rate = (
            self.metrics["error_spans"] / self.metrics["total_spans"]
            if self.metrics["total_spans"] > 0
            else 0
        )

        return {
            **self.metrics,
            "average_duration_ms": avg_duration,
            "error_rate": error_rate,
            "unique_traces": len(set(s.trace_id for s in self.spans)),
        }

    def export_traces(self, format: str = "json") -> str:
        """Export traces in specified format."""
        traces = self.get_traces()

        if format == "json":
            return json.dumps(traces, indent=2, default=str)
        elif format == "jaeger":
            # Simplified Jaeger format
            jaeger_traces = []
            for trace in traces:
                jaeger_trace = {
                    "traceID": trace["trace_id"],
                    "spans": [
                        {
                            "traceID": span["trace_id"],
                            "spanID": span["span_id"],
                            "parentSpanID": span["parent_span_id"],
                            "operationName": span["operation_name"],
                            "startTime": span["start_time"],
                            "duration": int(span["duration_ms"] * 1000),  # microseconds
                            "tags": [
                                {"key": k, "value": v}
                                for k, v in span["attributes"].items()
                            ],
                            "process": {"serviceName": self.service_name, "tags": []},
                        }
                        for span in trace["spans"]
                    ],
                }
                jaeger_traces.append(jaeger_trace)

            return json.dumps(jaeger_traces, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear_traces(self):
        """Clear all traces and reset metrics."""
        self.spans.clear()
        self.current_span = None
        self.metrics = {
            "total_spans": 0,
            "successful_spans": 0,
            "error_spans": 0,
            "total_duration_ms": 0,
        }

    def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy",
            "service_name": self.service_name,
            "total_spans": len(self.spans),
            "metrics": self.get_metrics(),
            "current_trace_id": self.get_current_trace_id(),
        }

    def _generate_id(self) -> str:
        """Generate a random ID."""
        return uuid.uuid4().hex


# Global tracer instance
_simple_tracer = None


def initialize_simple_tracing(service_name: str = "monorepo") -> SimpleTracer:
    """Initialize simple tracing system."""
    global _simple_tracer
    _simple_tracer = SimpleTracer(service_name)
    return _simple_tracer


def get_tracer() -> SimpleTracer:
    """Get global tracer."""
    if _simple_tracer is None:
        raise RuntimeError(
            "Simple tracing not initialized. Call initialize_simple_tracing() first."
        )
    return _simple_tracer


def trace_operation(
    operation_name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
):
    """Context manager for tracing operations."""
    tracer = get_tracer()
    return tracer.trace_operation(operation_name, kind, attributes)


def trace_function(
    operation_name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
):
    """Decorator for tracing functions."""
    tracer = get_tracer()
    return tracer.trace_function(operation_name, kind, attributes)


def trace_async_function(
    operation_name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
):
    """Decorator for tracing async functions."""
    tracer = get_tracer()
    return tracer.trace_async_function(operation_name, kind, attributes)


async def main():
    """Main function for testing."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Testing simple distributed tracing system")

    # Initialize tracer
    tracer = initialize_simple_tracing("pynomaly-test")

    # Test function tracing
    @trace_function("sync_data_processing", SpanKind.INTERNAL, {"component": "data"})
    def process_data(data_size: int) -> int:
        """Process some data."""
        time.sleep(0.01)  # Simulate processing
        return data_size * 2

    @trace_async_function(
        "async_ml_inference", SpanKind.INTERNAL, {"model": "test_model"}
    )
    async def run_inference(input_data: list[float]) -> float:
        """Run ML inference."""
        await asyncio.sleep(0.02)  # Simulate inference
        return sum(input_data) / len(input_data)

    # Test basic operations
    with trace_operation(
        "main_operation", SpanKind.SERVER, {"user_id": "12345"}
    ) as span:
        span.add_event("operation_started")

        # Nested operation
        result1 = process_data(100)
        span.set_attribute("data_processing_result", result1)

        # Async operation
        result2 = await run_inference([1.0, 2.0, 3.0, 4.0, 5.0])
        span.set_attribute("inference_result", result2)

        span.add_event("operation_completed", {"results_count": 2})

    # Test error handling
    try:
        with trace_operation("error_operation", SpanKind.CLIENT) as span:
            span.set_attribute("will_fail", True)
            raise ValueError("Test error for tracing")
    except ValueError:
        logger.info("Caught expected error")

    # Test multiple concurrent operations
    async def concurrent_operation(op_id: int):
        with trace_operation(f"concurrent_op_{op_id}", SpanKind.INTERNAL) as span:
            span.set_attribute("operation_id", op_id)
            await asyncio.sleep(0.01)
            return op_id * 10

    tasks = [concurrent_operation(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    logger.info(f"Concurrent operations results: {results}")

    # Get and display metrics
    metrics = tracer.get_metrics()
    logger.info(f"Tracing metrics: {metrics}")

    # Get traces
    traces = tracer.get_traces()
    logger.info(
        f"Collected {len(traces)} traces with {sum(t['span_count'] for t in traces)} total spans"
    )

    # Export traces
    json_export = tracer.export_traces("json")
    logger.info(f"JSON export length: {len(json_export)} characters")

    # Health check
    health = tracer.health_check()
    logger.info(f"Tracer health: {health['status']}")

    logger.info("Simple distributed tracing testing completed")


if __name__ == "__main__":
    asyncio.run(main())
