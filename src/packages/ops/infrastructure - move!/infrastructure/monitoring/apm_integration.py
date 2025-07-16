"""
Application Performance Monitoring (APM) Integration

This module provides comprehensive APM integration with support for multiple
monitoring platforms including New Relic, DataDog, Elastic APM, and custom solutions.
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)


class APMProvider(Enum):
    """Supported APM providers."""

    NEW_RELIC = "new_relic"
    DATADOG = "datadog"
    ELASTIC_APM = "elastic_apm"
    OPENTELEMETRY = "opentelemetry"
    CUSTOM = "custom"


class TraceLevel(Enum):
    """Trace level severity."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class APMMetric:
    """APM metric data structure."""

    name: str
    value: int | float
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, timer


@dataclass
class APMTrace:
    """APM trace data structure."""

    trace_id: str
    span_id: str
    operation_name: str
    start_time: datetime
    duration_ms: float
    status: str = "ok"
    tags: dict[str, str] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    level: TraceLevel = TraceLevel.INFO


@dataclass
class APMError:
    """APM error data structure."""

    error_id: str
    error_type: str
    message: str
    timestamp: datetime
    stack_trace: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    severity: TraceLevel = TraceLevel.ERROR


class APMIntegration:
    """Unified APM integration framework."""

    def __init__(self, provider: APMProvider = APMProvider.OPENTELEMETRY):
        self.provider = provider
        self.enabled = os.getenv("PYNOMALY_APM_ENABLED", "true").lower() == "true"
        self.service_name = os.getenv("PYNOMALY_SERVICE_NAME", "monorepo")
        self.environment = os.getenv("PYNOMALY_ENV", "development")
        self.version = os.getenv("PYNOMALY_VERSION", "1.0.0")

        self.metrics_buffer: list[APMMetric] = []
        self.traces_buffer: list[APMTrace] = []
        self.errors_buffer: list[APMError] = []

        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the selected APM provider."""
        if not self.enabled:
            logger.info("APM monitoring is disabled")
            return

        try:
            if self.provider == APMProvider.NEW_RELIC:
                self._initialize_new_relic()
            elif self.provider == APMProvider.DATADOG:
                self._initialize_datadog()
            elif self.provider == APMProvider.ELASTIC_APM:
                self._initialize_elastic_apm()
            elif self.provider == APMProvider.OPENTELEMETRY:
                self._initialize_opentelemetry()
            else:
                self._initialize_custom()

            logger.info(f"APM provider {self.provider.value} initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize APM provider {self.provider.value}: {e}"
            )
            self.enabled = False

    def _initialize_new_relic(self):
        """Initialize New Relic APM."""
        try:
            import newrelic.agent

            config_file = os.getenv("NEW_RELIC_CONFIG_FILE", "newrelic.ini")
            environment = os.getenv("NEW_RELIC_ENVIRONMENT", self.environment)

            newrelic.agent.initialize(config_file, environment)

            # Set application info
            newrelic.agent.set_application_name(self.service_name)

            self.new_relic = newrelic.agent
            logger.info("New Relic APM initialized")

        except ImportError:
            logger.warning(
                "New Relic agent not installed. Install with: pip install newrelic"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize New Relic: {e}")
            raise

    def _initialize_datadog(self):
        """Initialize DataDog APM."""
        try:
            from ddtrace import patch_all, tracer

            # Auto-patch supported libraries
            patch_all()

            # Configure tracer
            tracer.configure(
                hostname=os.getenv("DD_AGENT_HOST", "localhost"),
                port=int(os.getenv("DD_TRACE_AGENT_PORT", "8126")),
                service_name=self.service_name,
                env=self.environment,
                version=self.version,
            )

            self.dd_tracer = tracer
            logger.info("DataDog APM initialized")

        except ImportError:
            logger.warning(
                "DataDog tracer not installed. Install with: pip install ddtrace"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize DataDog: {e}")
            raise

    def _initialize_elastic_apm(self):
        """Initialize Elastic APM."""
        try:
            import elasticapm

            config = {
                "SERVICE_NAME": self.service_name,
                "SERVICE_VERSION": self.version,
                "ENVIRONMENT": self.environment,
                "SERVER_URL": os.getenv(
                    "ELASTIC_APM_SERVER_URL", "http://localhost:8200"
                ),
                "SECRET_TOKEN": os.getenv("ELASTIC_APM_SECRET_TOKEN"),
            }

            # Remove None values
            config = {k: v for k, v in config.items() if v is not None}

            self.elastic_client = elasticapm.Client(config)
            logger.info("Elastic APM initialized")

        except ImportError:
            logger.warning(
                "Elastic APM not installed. Install with: pip install elastic-apm"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Elastic APM: {e}")
            raise

    def _initialize_opentelemetry(self):
        """Initialize OpenTelemetry."""
        try:
            from opentelemetry import metrics, trace
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Create resource
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                    "service.version": self.version,
                    "service.environment": self.environment,
                }
            )

            # Configure tracing
            trace.set_tracer_provider(TracerProvider(resource=resource))

            # Add Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
                agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
            )

            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)

            # Configure metrics
            metric_reader = PrometheusMetricReader()
            metrics.set_meter_provider(
                MeterProvider(resource=resource, metric_readers=[metric_reader])
            )

            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)

            logger.info("OpenTelemetry initialized")

        except ImportError:
            logger.warning(
                "OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            raise

    def _initialize_custom(self):
        """Initialize custom APM solution."""
        logger.info("Custom APM solution initialized")
        # Custom implementation would go here

    def record_metric(
        self,
        name: str,
        value: int | float,
        tags: dict[str, str] | None = None,
        metric_type: str = "gauge",
    ):
        """Record a custom metric."""
        if not self.enabled:
            return

        try:
            metric = APMMetric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metric_type=metric_type,
            )

            self.metrics_buffer.append(metric)

            # Send metric based on provider
            if self.provider == APMProvider.NEW_RELIC:
                self._send_new_relic_metric(metric)
            elif self.provider == APMProvider.DATADOG:
                self._send_datadog_metric(metric)
            elif self.provider == APMProvider.ELASTIC_APM:
                self._send_elastic_metric(metric)
            elif self.provider == APMProvider.OPENTELEMETRY:
                self._send_opentelemetry_metric(metric)
            else:
                self._send_custom_metric(metric)

        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")

    def _send_new_relic_metric(self, metric: APMMetric):
        """Send metric to New Relic."""
        try:
            if hasattr(self, "new_relic"):
                self.new_relic.record_custom_metric(metric.name, metric.value)
        except Exception as e:
            logger.error(f"Failed to send New Relic metric: {e}")

    def _send_datadog_metric(self, metric: APMMetric):
        """Send metric to DataDog."""
        try:
            if hasattr(self, "dd_tracer"):
                # DataDog metrics would typically be sent via DogStatsD
                logger.debug(f"DataDog metric: {metric.name} = {metric.value}")
        except Exception as e:
            logger.error(f"Failed to send DataDog metric: {e}")

    def _send_elastic_metric(self, metric: APMMetric):
        """Send metric to Elastic APM."""
        try:
            if hasattr(self, "elastic_client"):
                logger.debug(f"Elastic metric: {metric.name} = {metric.value}")
        except Exception as e:
            logger.error(f"Failed to send Elastic metric: {e}")

    def _send_opentelemetry_metric(self, metric: APMMetric):
        """Send metric to OpenTelemetry."""
        try:
            if hasattr(self, "meter"):
                if metric.metric_type == "counter":
                    counter = self.meter.create_counter(metric.name)
                    counter.add(metric.value, metric.tags)
                elif metric.metric_type == "histogram":
                    histogram = self.meter.create_histogram(metric.name)
                    histogram.record(metric.value, metric.tags)
                else:  # gauge
                    gauge = self.meter.create_gauge(metric.name)
                    gauge.set(metric.value, metric.tags)
        except Exception as e:
            logger.error(f"Failed to send OpenTelemetry metric: {e}")

    def _send_custom_metric(self, metric: APMMetric):
        """Send metric to custom solution."""
        logger.debug(
            f"Custom metric: {metric.name} = {metric.value} (tags: {metric.tags})"
        )

    @contextmanager
    def trace_operation(self, operation_name: str, tags: dict[str, str] | None = None):
        """Context manager for tracing operations."""
        if not self.enabled:
            yield None
            return

        trace_id = f"trace_{int(time.time() * 1000)}"
        span_id = f"span_{int(time.time() * 1000000)}"
        start_time = datetime.now()

        try:
            # Start tracing based on provider
            if self.provider == APMProvider.NEW_RELIC:
                span = self._start_new_relic_trace(operation_name, tags)
            elif self.provider == APMProvider.DATADOG:
                span = self._start_datadog_trace(operation_name, tags)
            elif self.provider == APMProvider.ELASTIC_APM:
                span = self._start_elastic_trace(operation_name, tags)
            elif self.provider == APMProvider.OPENTELEMETRY:
                span = self._start_opentelemetry_trace(operation_name, tags)
            else:
                span = self._start_custom_trace(operation_name, tags)

            yield span

            # Successfully completed
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            trace = APMTrace(
                trace_id=trace_id,
                span_id=span_id,
                operation_name=operation_name,
                start_time=start_time,
                duration_ms=duration_ms,
                status="ok",
                tags=tags or {},
                level=TraceLevel.INFO,
            )

            self.traces_buffer.append(trace)

        except Exception as e:
            # Error during operation
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            trace = APMTrace(
                trace_id=trace_id,
                span_id=span_id,
                operation_name=operation_name,
                start_time=start_time,
                duration_ms=duration_ms,
                status="error",
                tags=tags or {},
                level=TraceLevel.ERROR,
            )

            self.traces_buffer.append(trace)

            # Record error
            self.record_error(
                str(e),
                error_type=type(e).__name__,
                context={"operation": operation_name, "trace_id": trace_id},
            )

            raise
        finally:
            # End tracing
            self._end_trace(span if "span" in locals() else None)

    def _start_new_relic_trace(self, operation_name: str, tags: dict[str, str] | None):
        """Start New Relic trace."""
        try:
            if hasattr(self, "new_relic"):
                return self.new_relic.function_trace(name=operation_name)
        except Exception as e:
            logger.error(f"Failed to start New Relic trace: {e}")
        return None

    def _start_datadog_trace(self, operation_name: str, tags: dict[str, str] | None):
        """Start DataDog trace."""
        try:
            if hasattr(self, "dd_tracer"):
                span = self.dd_tracer.trace(operation_name)
                if tags:
                    for key, value in tags.items():
                        span.set_tag(key, value)
                return span
        except Exception as e:
            logger.error(f"Failed to start DataDog trace: {e}")
        return None

    def _start_elastic_trace(self, operation_name: str, tags: dict[str, str] | None):
        """Start Elastic APM trace."""
        try:
            if hasattr(self, "elastic_client"):
                return self.elastic_client.begin_transaction(operation_name)
        except Exception as e:
            logger.error(f"Failed to start Elastic trace: {e}")
        return None

    def _start_opentelemetry_trace(
        self, operation_name: str, tags: dict[str, str] | None
    ):
        """Start OpenTelemetry trace."""
        try:
            if hasattr(self, "tracer"):
                span = self.tracer.start_span(operation_name)
                if tags:
                    for key, value in tags.items():
                        span.set_attribute(key, value)
                return span
        except Exception as e:
            logger.error(f"Failed to start OpenTelemetry trace: {e}")
        return None

    def _start_custom_trace(self, operation_name: str, tags: dict[str, str] | None):
        """Start custom trace."""
        logger.debug(f"Starting custom trace: {operation_name} (tags: {tags})")
        return {"operation": operation_name, "tags": tags, "start_time": time.time()}

    def _end_trace(self, span: Any):
        """End trace span."""
        try:
            if span is None:
                return

            if self.provider == APMProvider.DATADOG and hasattr(span, "finish"):
                span.finish()
            elif self.provider == APMProvider.ELASTIC_APM and hasattr(span, "end"):
                span.end()
            elif self.provider == APMProvider.OPENTELEMETRY and hasattr(span, "end"):
                span.end()
            elif self.provider == APMProvider.CUSTOM:
                duration = time.time() - span.get("start_time", time.time())
                logger.debug(
                    f"Custom trace ended: {span.get('operation')} ({duration:.3f}s)"
                )

        except Exception as e:
            logger.error(f"Failed to end trace: {e}")

    def record_error(
        self,
        message: str,
        error_type: str = "ApplicationError",
        stack_trace: str | None = None,
        context: dict[str, Any] | None = None,
        severity: TraceLevel = TraceLevel.ERROR,
    ):
        """Record an error event."""
        if not self.enabled:
            return

        try:
            error = APMError(
                error_id=f"error_{int(time.time() * 1000)}",
                error_type=error_type,
                message=message,
                timestamp=datetime.now(),
                stack_trace=stack_trace,
                context=context or {},
                severity=severity,
            )

            self.errors_buffer.append(error)

            # Send error based on provider
            if self.provider == APMProvider.NEW_RELIC:
                self._send_new_relic_error(error)
            elif self.provider == APMProvider.DATADOG:
                self._send_datadog_error(error)
            elif self.provider == APMProvider.ELASTIC_APM:
                self._send_elastic_error(error)
            elif self.provider == APMProvider.OPENTELEMETRY:
                self._send_opentelemetry_error(error)
            else:
                self._send_custom_error(error)

        except Exception as e:
            logger.error(f"Failed to record error: {e}")

    def _send_new_relic_error(self, error: APMError):
        """Send error to New Relic."""
        try:
            if hasattr(self, "new_relic"):
                self.new_relic.record_exception()
        except Exception as e:
            logger.error(f"Failed to send New Relic error: {e}")

    def _send_datadog_error(self, error: APMError):
        """Send error to DataDog."""
        try:
            logger.debug(f"DataDog error: {error.error_type} - {error.message}")
        except Exception as e:
            logger.error(f"Failed to send DataDog error: {e}")

    def _send_elastic_error(self, error: APMError):
        """Send error to Elastic APM."""
        try:
            if hasattr(self, "elastic_client"):
                self.elastic_client.capture_exception()
        except Exception as e:
            logger.error(f"Failed to send Elastic error: {e}")

    def _send_opentelemetry_error(self, error: APMError):
        """Send error to OpenTelemetry."""
        try:
            logger.debug(f"OpenTelemetry error: {error.error_type} - {error.message}")
        except Exception as e:
            logger.error(f"Failed to send OpenTelemetry error: {e}")

    def _send_custom_error(self, error: APMError):
        """Send error to custom solution."""
        logger.error(
            f"Custom error: {error.error_type} - {error.message} (context: {error.context})"
        )

    def flush_buffers(self):
        """Flush all buffered metrics, traces, and errors."""
        if not self.enabled:
            return

        try:
            logger.info(
                f"Flushing APM buffers: {len(self.metrics_buffer)} metrics, "
                f"{len(self.traces_buffer)} traces, {len(self.errors_buffer)} errors"
            )

            # Clear buffers
            self.metrics_buffer.clear()
            self.traces_buffer.clear()
            self.errors_buffer.clear()

        except Exception as e:
            logger.error(f"Failed to flush APM buffers: {e}")

    def get_health_status(self) -> dict[str, Any]:
        """Get APM integration health status."""
        return {
            "enabled": self.enabled,
            "provider": self.provider.value,
            "service_name": self.service_name,
            "environment": self.environment,
            "version": self.version,
            "buffer_sizes": {
                "metrics": len(self.metrics_buffer),
                "traces": len(self.traces_buffer),
                "errors": len(self.errors_buffer),
            },
        }


def trace_async_function(
    operation_name: str | None = None, tags: dict[str, str] | None = None
):
    """Decorator for tracing async functions."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            # Get APM instance (would be injected in real application)
            apm = APMIntegration()

            with apm.trace_operation(op_name, tags):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)

                    # Record success metric
                    duration = (time.time() - start_time) * 1000
                    apm.record_metric(
                        f"{op_name}.duration", duration, tags, "histogram"
                    )
                    apm.record_metric(f"{op_name}.success", 1, tags, "counter")

                    return result

                except Exception:
                    # Record error metric
                    apm.record_metric(f"{op_name}.error", 1, tags, "counter")
                    raise

        return wrapper

    return decorator


def trace_function(
    operation_name: str | None = None, tags: dict[str, str] | None = None
):
    """Decorator for tracing synchronous functions."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            # Get APM instance (would be injected in real application)
            apm = APMIntegration()

            with apm.trace_operation(op_name, tags):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)

                    # Record success metric
                    duration = (time.time() - start_time) * 1000
                    apm.record_metric(
                        f"{op_name}.duration", duration, tags, "histogram"
                    )
                    apm.record_metric(f"{op_name}.success", 1, tags, "counter")

                    return result

                except Exception:
                    # Record error metric
                    apm.record_metric(f"{op_name}.error", 1, tags, "counter")
                    raise

        return wrapper

    return decorator


# Global APM instance (would be configured via dependency injection)
apm_instance = None


def get_apm() -> APMIntegration:
    """Get the global APM instance."""
    global apm_instance
    if apm_instance is None:
        provider_name = os.getenv("PYNOMALY_APM_PROVIDER", "opentelemetry")
        try:
            provider = APMProvider(provider_name)
        except ValueError:
            logger.warning(
                f"Unknown APM provider '{provider_name}', defaulting to OpenTelemetry"
            )
            provider = APMProvider.OPENTELEMETRY

        apm_instance = APMIntegration(provider)

    return apm_instance


def shutdown_apm():
    """Shutdown APM integration and flush remaining data."""
    global apm_instance
    if apm_instance:
        apm_instance.flush_buffers()
        apm_instance = None
        logger.info("APM integration shutdown completed")
