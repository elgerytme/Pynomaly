"""OpenTelemetry monitoring integration for observability.

This module provides comprehensive monitoring capabilities including:
- Distributed tracing
- Metrics collection
- Logging correlation
- Performance monitoring
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Status, StatusCode
from prometheus_client import start_http_server

from pynomaly.infrastructure.config import Settings

logger = logging.getLogger(__name__)


class TelemetryService:
    """Service for managing OpenTelemetry instrumentation."""

    def __init__(self, settings: Settings):
        """Initialize telemetry service with configuration.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.resource = self._create_resource()
        self.tracer_provider: TracerProvider | None = None
        self.meter_provider: MeterProvider | None = None
        self.tracer: trace.Tracer | None = None
        self.meter: metrics.Meter | None = None

        # Metrics collectors
        self._request_counter = None
        self._request_duration = None
        self._active_detectors = None
        self._anomaly_detection_counter = None
        self._model_training_duration = None
        self._cache_hits = None
        self._cache_misses = None

        if settings.monitoring.metrics_enabled or settings.monitoring.tracing_enabled:
            self._setup_telemetry()

    def _create_resource(self) -> Resource:
        """Create OpenTelemetry resource with service information."""
        return Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: "pynomaly",
                ResourceAttributes.SERVICE_VERSION: self.settings.app.version,
                ResourceAttributes.SERVICE_NAMESPACE: self.settings.app.environment,
                ResourceAttributes.HOST_NAME: self.settings.monitoring.host_name,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.settings.app.environment,
            }
        )

    def _setup_telemetry(self) -> None:
        """Set up OpenTelemetry providers and exporters."""
        # Set up tracing
        if self.settings.monitoring.tracing_enabled:
            self._setup_tracing()

        # Set up metrics
        if self.settings.monitoring.metrics_enabled:
            self._setup_metrics()

        # Auto-instrument libraries
        self._setup_auto_instrumentation()

    def _setup_tracing(self) -> None:
        """Configure distributed tracing."""
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=self.resource)

        # Add OTLP exporter if configured
        if self.settings.monitoring.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.settings.monitoring.otlp_endpoint,
                insecure=self.settings.monitoring.otlp_insecure,
            )
            self.tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Set as global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)

        logger.info(
            "Tracing initialized with OTLP endpoint: %s",
            self.settings.monitoring.otlp_endpoint,
        )

    def _setup_metrics(self) -> None:
        """Configure metrics collection."""
        readers = []

        # Add Prometheus exporter
        if self.settings.monitoring.prometheus_enabled:
            prometheus_reader = PrometheusMetricReader()
            readers.append(prometheus_reader)

            # Start Prometheus metrics server
            start_http_server(
                port=self.settings.monitoring.prometheus_port, addr="0.0.0.0"
            )
            logger.info(
                "Prometheus metrics server started on port %d",
                self.settings.monitoring.prometheus_port,
            )

        # Add OTLP exporter if configured
        if self.settings.monitoring.otlp_endpoint:
            otlp_exporter = OTLPMetricExporter(
                endpoint=self.settings.monitoring.otlp_endpoint,
                insecure=self.settings.monitoring.otlp_insecure,
            )
            readers.append(
                PeriodicExportingMetricReader(
                    otlp_exporter, export_interval_millis=60000
                )
            )

        # Create meter provider
        self.meter_provider = MeterProvider(
            resource=self.resource, metric_readers=readers
        )

        # Set as global meter provider
        metrics.set_meter_provider(self.meter_provider)
        self.meter = metrics.get_meter(__name__)

        # Initialize metrics
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize application metrics."""
        # Request metrics
        self._request_counter = self.meter.create_counter(
            name="pynomaly_http_requests_total",
            description="Total number of HTTP requests",
            unit="request",
        )

        self._request_duration = self.meter.create_histogram(
            name="pynomaly_http_request_duration_seconds",
            description="HTTP request duration in seconds",
            unit="s",
        )

        # Detection metrics
        self._anomaly_detection_counter = self.meter.create_counter(
            name="pynomaly_anomaly_detections_total",
            description="Total number of anomaly detection runs",
            unit="detection",
        )

        self._model_training_duration = self.meter.create_histogram(
            name="pynomaly_model_training_duration_seconds",
            description="Model training duration in seconds",
            unit="s",
        )

        # Cache metrics
        self._cache_hits = self.meter.create_counter(
            name="pynomaly_cache_hits_total",
            description="Total number of cache hits",
            unit="hit",
        )

        self._cache_misses = self.meter.create_counter(
            name="pynomaly_cache_misses_total",
            description="Total number of cache misses",
            unit="miss",
        )

        # Gauge metrics with callbacks
        self._active_detectors = self.meter.create_observable_gauge(
            name="pynomaly_active_detectors",
            description="Number of active detectors",
            callbacks=[self._get_active_detectors_count],
        )

    def _get_active_detectors_count(
        self, options: CallbackOptions
    ) -> Iterable[Observation]:
        """Callback to get active detector count."""
        # This would query the actual detector repository
        # For now, return a placeholder
        yield Observation(0, {})

    def _setup_auto_instrumentation(self) -> None:
        """Set up automatic instrumentation for libraries."""
        # FastAPI
        if self.settings.monitoring.instrument_fastapi:
            try:
                FastAPIInstrumentor.instrument()
                logger.info("FastAPI instrumentation enabled")
            except Exception as e:
                logger.warning("Failed to instrument FastAPI: %s", e)

        # Logging
        LoggingInstrumentor().instrument()

        # HTTP requests
        RequestsInstrumentor().instrument()

        # SQLAlchemy (if database is configured)
        if self.settings.monitoring.instrument_sqlalchemy:
            try:
                SQLAlchemyInstrumentor().instrument()
                logger.info("SQLAlchemy instrumentation enabled")
            except Exception as e:
                logger.warning("Failed to instrument SQLAlchemy: %s", e)

    @contextmanager
    def trace_span(self, name: str, attributes: dict[str, Any] | None = None):
        """Create a trace span context manager.

        Args:
            name: Span name
            attributes: Optional span attributes

        Yields:
            Trace span
        """
        if not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                span.set_attributes(attributes)

            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def trace_method(self, name: str | None = None):
        """Decorator to trace method execution.

        Args:
            name: Optional span name (defaults to function name)

        Returns:
            Decorated function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span_name = name or f"{func.__module__}.{func.__name__}"
                with self.trace_span(span_name):
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                span_name = name or f"{func.__module__}.{func.__name__}"
                with self.trace_span(span_name):
                    return func(*args, **kwargs)

            # Return appropriate wrapper based on function type
            import asyncio

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def record_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics.

        Args:
            method: HTTP method
            endpoint: Request endpoint
            status_code: Response status code
            duration: Request duration in seconds
        """
        if self._request_counter:
            self._request_counter.add(
                1,
                {
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": str(status_code),
                },
            )

        if self._request_duration:
            self._request_duration.record(
                duration, {"method": method, "endpoint": endpoint}
            )

    def record_detection(
        self, algorithm: str, dataset_size: int, anomalies_found: int, duration: float
    ):
        """Record anomaly detection metrics.

        Args:
            algorithm: Algorithm used
            dataset_size: Size of dataset
            anomalies_found: Number of anomalies detected
            duration: Detection duration in seconds
        """
        if self._anomaly_detection_counter:
            self._anomaly_detection_counter.add(
                1, {"algorithm": algorithm, "has_anomalies": str(anomalies_found > 0)}
            )

        # Additional metrics can be recorded here
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attributes(
                {
                    "detection.algorithm": algorithm,
                    "detection.dataset_size": dataset_size,
                    "detection.anomalies_found": anomalies_found,
                    "detection.duration_seconds": duration,
                }
            )

    def record_training(self, algorithm: str, dataset_size: int, duration: float):
        """Record model training metrics.

        Args:
            algorithm: Algorithm trained
            dataset_size: Training dataset size
            duration: Training duration in seconds
        """
        if self._model_training_duration:
            self._model_training_duration.record(
                duration,
                {
                    "algorithm": algorithm,
                    "dataset_size_category": self._categorize_size(dataset_size),
                },
            )

    def record_cache_hit(self, cache_type: str):
        """Record cache hit.

        Args:
            cache_type: Type of cache (e.g., "model", "result")
        """
        if self._cache_hits:
            self._cache_hits.add(1, {"cache_type": cache_type})

    def record_cache_miss(self, cache_type: str):
        """Record cache miss.

        Args:
            cache_type: Type of cache
        """
        if self._cache_misses:
            self._cache_misses.add(1, {"cache_type": cache_type})

    def _categorize_size(self, size: int) -> str:
        """Categorize dataset size for metrics.

        Args:
            size: Dataset size

        Returns:
            Size category
        """
        if size < 1000:
            return "small"
        elif size < 10000:
            return "medium"
        elif size < 100000:
            return "large"
        else:
            return "xlarge"

    def shutdown(self):
        """Shutdown telemetry providers."""
        if self.tracer_provider:
            self.tracer_provider.shutdown()

        if self.meter_provider:
            self.meter_provider.shutdown()

        logger.info("Telemetry service shutdown complete")


# Global telemetry instance
_telemetry_service: TelemetryService | None = None


def init_telemetry(settings: Settings) -> TelemetryService:
    """Initialize global telemetry service.

    Args:
        settings: Application settings

    Returns:
        Telemetry service instance
    """
    global _telemetry_service
    _telemetry_service = TelemetryService(settings)
    return _telemetry_service


def get_telemetry() -> TelemetryService | None:
    """Get global telemetry service instance.

    Returns:
        Telemetry service or None if not initialized
    """
    return _telemetry_service


def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Convenience function for creating trace spans.

    Args:
        name: Span name
        attributes: Optional attributes

    Returns:
        Context manager for span
    """
    telemetry = get_telemetry()
    if telemetry:
        return telemetry.trace_span(name, attributes)
    else:
        # Return a no-op context manager
        @contextmanager
        def noop():
            yield None

        return noop()


def trace_method(name: str | None = None):
    """Convenience decorator for method tracing.

    Args:
        name: Optional span name

    Returns:
        Decorator function
    """
    telemetry = get_telemetry()
    if telemetry:
        return telemetry.trace_method(name)
    else:
        # Return identity decorator
        def decorator(func):
            return func

        return decorator
