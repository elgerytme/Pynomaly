"""Metrics collector module - alias to observability.metrics for backward compatibility."""

# Import all components from the observability.metrics module
from ..observability.metrics import (
    MetricsCollector,
    MetricConfig,
    MetricPoint,
    get_metrics_collector,
    initialize_metrics,
    PROMETHEUS_AVAILABLE,
    PSUTIL_AVAILABLE
)

__all__ = [
    "MetricsCollector",
    "MetricConfig", 
    "MetricPoint",
    "get_metrics_collector",
    "initialize_metrics",
    "PROMETHEUS_AVAILABLE",
    "PSUTIL_AVAILABLE"
]