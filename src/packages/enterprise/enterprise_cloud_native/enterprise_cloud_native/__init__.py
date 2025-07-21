"""
Enterprise Cloud-Native Package

Kubernetes operators, service mesh, and auto-scaling for enterprise cloud-native applications.
"""

from .application.services.cloud_native_service import CloudNativeService
from .domain.entities.kubernetes_resource import KubernetesResource, OperatorResource
from .domain.entities.service_mesh import (
    ServiceMeshConfiguration, ServiceMeshService, TrafficPolicy, SecurityPolicy
)
from .domain.entities.autoscaling import (
    HorizontalPodAutoscaler, VerticalPodAutoscaler, ClusterAutoscaler, PredictiveScalingPolicy
)

__version__ = "0.1.0"
__author__ = "Pynomaly Enterprise Team"

__all__ = [
    "CloudNativeService",
    "KubernetesResource",
    "OperatorResource", 
    "ServiceMeshConfiguration",
    "ServiceMeshService",
    "TrafficPolicy",
    "SecurityPolicy",
    "HorizontalPodAutoscaler",
    "VerticalPodAutoscaler",
    "ClusterAutoscaler",
    "PredictiveScalingPolicy"
]