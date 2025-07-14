"""Monitoring infrastructure for enterprise applications.

This module provides comprehensive monitoring capabilities including metrics collection,
distributed tracing, health checks, and performance monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
from enterprise_core import (
    ConfigurationError,
    HealthCheck,
    HealthStatus,
    InfrastructureError,
    Metrics,
    TimerContext,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector(Metrics):
    """Base metrics collector with pluggable backends."""

    def __init__(self, backend: Optional[MetricsBackend] = None) -> None:
        self._backend = backend or InMemoryMetricsBackend()
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}

    def counter(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        full_name = self._build_metric_name(name, "counter")
        self._counters[full_name] = self._counters.get(full_name, 0) + value

        metric = MetricPoint(name=full_name, value=value, tags=tags or {})
        self._backend.record_metric(metric)

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        full_name = self._build_metric_name(name, "gauge")
        self._gauges[full_name] = value

        metric = MetricPoint(name=full_name, value=value, tags=tags or {})
        self._backend.record_metric(metric)

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        full_name = self._build_metric_name(name, "histogram")
        metric = MetricPoint(name=full_name, value=value, tags=tags or {})
        self._backend.record_metric(metric)

    def timer(self, name: str) -> TimerContext:
        """Create a timer context for measuring duration."""
        return MetricTimerContext(self, name)

    def _build_metric_name(self, name: str, metric_type: str) -> str:
        """Build a full metric name with type prefix."""
        return f"{metric_type}.{name}"

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "backend_metrics": self._backend.get_metrics(),
        }


class MetricTimerContext:
    """Timer context for measuring execution time."""

    def __init__(self, collector: MetricsCollector, name: str) -> None:
        self.collector = collector
        self.name = name
        self.start_time: Optional[float] = None

    def __enter__(self) -> MetricTimerContext:
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the timer and record the duration."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.histogram(f"{self.name}.duration", duration)


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""

    @abstractmethod
    def record_metric(self, metric: MetricPoint) -> None:
        """Record a metric point."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        pass


class InMemoryMetricsBackend(MetricsBackend):
    """In-memory metrics backend for development and testing."""

    def __init__(self) -> None:
        self._metrics: List[MetricPoint] = []

    def record_metric(self, metric: MetricPoint) -> None:
        """Record a metric point in memory."""
        self._metrics.append(metric)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return {
            "total_metrics": len(self._metrics),
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "tags": m.tags,
                    "timestamp": m.timestamp,
                }
                for m in self._metrics[-100:]  # Last 100 metrics
            ],
        }


class PrometheusMetrics(MetricsBackend):
    """Prometheus metrics backend."""

    def __init__(self, registry=None, namespace: str = "enterprise") -> None:
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
            self._Counter = Counter
            self._Gauge = Gauge
            self._Histogram = Histogram
        except ImportError:
            raise InfrastructureError(
                "Prometheus client not installed. Install with: pip install 'enterprise-infrastructure[monitoring]'",
                error_code="DEPENDENCY_MISSING",
            )

        self._registry = registry or CollectorRegistry()
        self._namespace = namespace
        self._counters: Dict[str, Any] = {}
        self._gauges: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}

    def record_metric(self, metric: MetricPoint) -> None:
        """Record a metric point in Prometheus."""
        metric_name = metric.name.replace(".", "_")
        labels = list(metric.tags.keys())
        label_values = list(metric.tags.values())

        if metric.name.startswith("counter."):
            if metric_name not in self._counters:
                self._counters[metric_name] = self._Counter(
                    metric_name,
                    f"Counter metric: {metric.name}",
                    labels,
                    registry=self._registry,
                )
            self._counters[metric_name].labels(*label_values).inc(metric.value)

        elif metric.name.startswith("gauge."):
            if metric_name not in self._gauges:
                self._gauges[metric_name] = self._Gauge(
                    metric_name,
                    f"Gauge metric: {metric.name}",
                    labels,
                    registry=self._registry,
                )
            self._gauges[metric_name].labels(*label_values).set(metric.value)

        elif metric.name.startswith("histogram."):
            if metric_name not in self._histograms:
                self._histograms[metric_name] = self._Histogram(
                    metric_name,
                    f"Histogram metric: {metric.name}",
                    labels,
                    registry=self._registry,
                )
            self._histograms[metric_name].labels(*label_values).observe(metric.value)

    def get_metrics(self) -> Dict[str, Any]:
        """Get Prometheus metrics summary."""
        return {
            "counters": len(self._counters),
            "gauges": len(self._gauges),
            "histograms": len(self._histograms),
            "registry": str(self._registry),
        }


class OpenTelemetryTracing:
    """OpenTelemetry distributed tracing setup."""

    def __init__(self, service_name: str, jaeger_endpoint: Optional[str] = None) -> None:
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self._tracer = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize OpenTelemetry tracing."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            raise InfrastructureError(
                "OpenTelemetry not installed. Install with: pip install 'enterprise-infrastructure[monitoring]'",
                error_code="DEPENDENCY_MISSING",
            )

        # Set up resource
        resource = Resource.create({"service.name": self.service_name})

        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Set up span processor
        if self.jaeger_endpoint:
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=6831,
                )
                span_processor = BatchSpanProcessor(jaeger_exporter)
                tracer_provider.add_span_processor(span_processor)
            except ImportError:
                logger.warning("Jaeger exporter not available, tracing will be limited")

        # Get tracer
        self._tracer = trace.get_tracer(__name__)
        self._initialized = True

        logger.info(f"OpenTelemetry tracing initialized for service: {self.service_name}")

    @asynccontextmanager
    async def trace_async(self, operation_name: str, **attributes):
        """Async context manager for tracing operations."""
        if not self._initialized:
            self.initialize()

        if not self._tracer:
            yield None
            return

        with self._tracer.start_as_current_span(operation_name) as span:
            # Set attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)

            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise


class HealthCheckManager:
    """Manager for application health checks."""

    def __init__(self) -> None:
        self._checks: Dict[str, HealthCheck] = {}

    def register_check(self, name: str, check: HealthCheck) -> None:
        """Register a health check."""
        self._checks[name] = check
        logger.debug(f"Registered health check: {name}")

    def remove_check(self, name: str) -> None:
        """Remove a health check."""
        self._checks.pop(name, None)
        logger.debug(f"Removed health check: {name}")

    async def check_health(self, check_name: Optional[str] = None) -> Dict[str, HealthStatus]:
        """Perform health checks."""
        if check_name:
            if check_name not in self._checks:
                return {
                    check_name: HealthStatus(
                        status="unknown",
                        message=f"Health check '{check_name}' not found",
                    )
                }

            try:
                result = await self._checks[check_name].check()
                return {check_name: result}
            except Exception as e:
                return {
                    check_name: HealthStatus(
                        status="unhealthy",
                        message=f"Health check failed: {e}",
                        details={"error": str(e)},
                    )
                }

        # Check all registered health checks
        results = {}
        for name, check in self._checks.items():
            try:
                results[name] = await check.check()
            except Exception as e:
                results[name] = HealthStatus(
                    status="unhealthy",
                    message=f"Health check failed: {e}",
                    details={"error": str(e)},
                )

        return results

    async def get_overall_health(self) -> HealthStatus:
        """Get overall application health."""
        results = await self.check_health()

        if not results:
            return HealthStatus(
                status="healthy",
                message="No health checks configured",
            )

        # Determine overall status
        statuses = [result.status for result in results.values()]

        if "unhealthy" in statuses:
            overall_status = "unhealthy"
            message = "One or more health checks failed"
        elif "degraded" in statuses:
            overall_status = "degraded"
            message = "One or more health checks are degraded"
        else:
            overall_status = "healthy"
            message = "All health checks passed"

        return HealthStatus(
            status=overall_status,
            message=message,
            details={
                "total_checks": len(results),
                "healthy": len([s for s in statuses if s == "healthy"]),
                "degraded": len([s for s in statuses if s == "degraded"]),
                "unhealthy": len([s for s in statuses if s == "unhealthy"]),
                "checks": {name: result.status for name, result in results.items()},
            },
        )


class SystemHealthCheck(HealthCheck):
    """System resource health check."""

    def __init__(
        self,
        max_cpu_percent: float = 90.0,
        max_memory_percent: float = 90.0,
        min_disk_free_gb: float = 1.0,
    ) -> None:
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.min_disk_free_gb = min_disk_free_gb

    async def check(self) -> HealthStatus:
        """Check system resource health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_free_gb = disk.free / (1024**3)

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_free_gb": round(disk_free_gb, 2),
                "thresholds": {
                    "max_cpu_percent": self.max_cpu_percent,
                    "max_memory_percent": self.max_memory_percent,
                    "min_disk_free_gb": self.min_disk_free_gb,
                },
            }

            # Check thresholds
            issues = []
            if cpu_percent > self.max_cpu_percent:
                issues.append(f"High CPU usage: {cpu_percent}%")

            if memory_percent > self.max_memory_percent:
                issues.append(f"High memory usage: {memory_percent}%")

            if disk_free_gb < self.min_disk_free_gb:
                issues.append(f"Low disk space: {disk_free_gb:.2f}GB free")

            if issues:
                return HealthStatus(
                    status="degraded" if len(issues) == 1 else "unhealthy",
                    message="; ".join(issues),
                    details=details,
                )

            return HealthStatus(
                status="healthy",
                message="System resources are within normal limits",
                details=details,
            )

        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                message=f"Failed to check system health: {e}",
                details={"error": str(e)},
            )


class PerformanceMonitor:
    """Performance monitoring and profiling."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None) -> None:
        self.metrics = metrics_collector or MetricsCollector()
        self._active_requests = 0
        self._request_times: List[float] = []

    @asynccontextmanager
    async def monitor_request(self, operation: str):
        """Monitor a request or operation."""
        self._active_requests += 1
        self.metrics.gauge("active_requests", self._active_requests)

        start_time = time.time()
        try:
            yield

            # Record success
            self.metrics.counter(f"{operation}.success")

        except Exception as e:
            # Record failure
            self.metrics.counter(f"{operation}.error")
            self.metrics.counter(f"{operation}.error.{type(e).__name__}")
            raise

        finally:
            # Record timing and cleanup
            duration = time.time() - start_time
            self.metrics.histogram(f"{operation}.duration", duration)

            self._active_requests -= 1
            self.metrics.gauge("active_requests", self._active_requests)

            # Keep recent request times for statistics
            self._request_times.append(duration)
            if len(self._request_times) > 1000:
                self._request_times = self._request_times[-1000:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self._request_times:
            return {
                "active_requests": self._active_requests,
                "total_requests": 0,
                "avg_response_time": 0,
                "p95_response_time": 0,
                "p99_response_time": 0,
            }

        sorted_times = sorted(self._request_times)
        n = len(sorted_times)

        return {
            "active_requests": self._active_requests,
            "total_requests": len(self._request_times),
            "avg_response_time": sum(self._request_times) / n,
            "p95_response_time": sorted_times[int(n * 0.95)] if n > 0 else 0,
            "p99_response_time": sorted_times[int(n * 0.99)] if n > 0 else 0,
            "min_response_time": min(self._request_times),
            "max_response_time": max(self._request_times),
        }
