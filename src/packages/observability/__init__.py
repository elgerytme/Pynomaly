"""
Advanced Observability Platform for MLOps Ecosystem

This package provides comprehensive observability capabilities including:
- Metrics collection with Prometheus and OpenTelemetry
- Distributed tracing with Jaeger
- Advanced alerting with anomaly detection
- ML-specific monitoring for drift, bias, and performance
- Custom dashboards and visualizations
- Incident management and response
"""

from .core.metrics_collector import MetricsCollector
from .core.tracing_manager import TracingManager
from .core.alerting_engine import AlertingEngine
from .core.dashboard_manager import DashboardManager
from .ml_monitoring.drift_detector import DriftDetector
from .ml_monitoring.model_monitor import ModelMonitor
from .ml_monitoring.bias_detector import BiasDetector
from .incident_management.incident_manager import IncidentManager
from .cost_optimization.resource_optimizer import ResourceOptimizer

__version__ = "1.0.0"
__author__ = "MLOps Platform Team"

__all__ = [
    "MetricsCollector",
    "TracingManager", 
    "AlertingEngine",
    "DashboardManager",
    "DriftDetector",
    "ModelMonitor",
    "BiasDetector",
    "IncidentManager",
    "ResourceOptimizer"
]