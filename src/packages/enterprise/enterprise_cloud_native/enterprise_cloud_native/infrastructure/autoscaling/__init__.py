"""
Auto-scaling infrastructure adapters.
"""

from .kubernetes_autoscaling import KubernetesAutoscalingAdapter
from .predictive_scaling import PredictiveScalingEngine
from .metrics_collectors import MetricsCollector, PrometheusMetricsCollector

__all__ = [
    "KubernetesAutoscalingAdapter",
    "PredictiveScalingEngine",
    "MetricsCollector",
    "PrometheusMetricsCollector"
]