"""Kubernetes domain models for enterprise deployment and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import numpy as np


class DeploymentStatus(Enum):
    """Status of Kubernetes deployments."""
    
    PENDING = "pending"
    RUNNING = "running"
    SCALING = "scaling"
    UPDATING = "updating"
    FAILED = "failed"
    TERMINATED = "terminated"


class ResourceType(Enum):
    """Types of Kubernetes resources."""
    
    DEPLOYMENT = "deployment"
    SERVICE = "service"
    CONFIGMAP = "configmap"
    SECRET = "secret"
    INGRESS = "ingress"
    PERSISTENT_VOLUME = "persistentvolume"
    PERSISTENT_VOLUME_CLAIM = "persistentvolumeclaim"
    HORIZONTAL_POD_AUTOSCALER = "horizontalpodautoscaler"
    CUSTOM_RESOURCE = "customresource"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"


@dataclass
class ResourceRequirements:
    """Container resource requirements."""
    
    cpu_request: str = "100m"  # e.g., "100m", "1", "2"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"  # e.g., "128Mi", "1Gi"
    memory_limit: str = "512Mi"
    
    # Storage requirements
    storage_request: Optional[str] = None
    storage_class: Optional[str] = None
    
    # GPU requirements
    gpu_request: int = 0
    gpu_type: Optional[str] = None  # e.g., "nvidia.com/gpu"
    
    def __post_init__(self):
        if self.gpu_request < 0:
            raise ValueError("GPU request must be non-negative")


@dataclass
class ScalingConfiguration:
    """Auto-scaling configuration."""
    
    policy: ScalingPolicy
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70  # Percentage
    target_memory_utilization: int = 80  # Percentage
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Scaling behavior
    scale_up_stabilization: int = 60  # seconds
    scale_down_stabilization: int = 300  # seconds
    scale_up_policies: List[Dict[str, Any]] = field(default_factory=list)
    scale_down_policies: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.min_replicas < 0:
            raise ValueError("Min replicas must be non-negative")
        if self.max_replicas < self.min_replicas:
            raise ValueError("Max replicas must be >= min replicas")
        if not 0 < self.target_cpu_utilization <= 100:
            raise ValueError("Target CPU utilization must be between 1 and 100")


@dataclass
class HealthCheckConfiguration:
    """Health check configuration for containers."""
    
    # Liveness probe
    liveness_enabled: bool = True
    liveness_path: str = "/health"
    liveness_port: int = 8080
    liveness_initial_delay: int = 30  # seconds
    liveness_period: int = 10  # seconds
    liveness_timeout: int = 5  # seconds
    liveness_failure_threshold: int = 3
    
    # Readiness probe
    readiness_enabled: bool = True
    readiness_path: str = "/ready"
    readiness_port: int = 8080
    readiness_initial_delay: int = 10  # seconds
    readiness_period: int = 5  # seconds
    readiness_timeout: int = 3  # seconds
    readiness_failure_threshold: int = 3
    
    # Startup probe
    startup_enabled: bool = True
    startup_path: str = "/startup"
    startup_port: int = 8080
    startup_initial_delay: int = 10  # seconds
    startup_period: int = 5  # seconds
    startup_timeout: int = 3  # seconds
    startup_failure_threshold: int = 30  # Allow more time for startup


@dataclass
class SecurityConfiguration:
    """Security configuration for deployments."""
    
    # Pod security
    run_as_non_root: bool = True
    run_as_user: Optional[int] = 1000
    run_as_group: Optional[int] = 1000
    fs_group: Optional[int] = 2000
    
    # Container security
    read_only_root_filesystem: bool = True
    allow_privilege_escalation: bool = False
    drop_capabilities: List[str] = field(default_factory=lambda: ["ALL"])
    add_capabilities: List[str] = field(default_factory=list)
    
    # Network security
    network_policy_enabled: bool = True
    allowed_ingress_rules: List[Dict[str, Any]] = field(default_factory=list)
    allowed_egress_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Service mesh
    service_mesh_enabled: bool = False
    mtls_enabled: bool = False
    
    # Secrets management
    secrets_encryption_enabled: bool = True
    external_secrets_enabled: bool = False
    vault_integration: bool = False


@dataclass
class PynomaleDeployment:
    """Pynomaly deployment configuration."""
    
    deployment_id: UUID
    name: str
    namespace: str = "pynomaly"
    
    # Component configuration
    components: Dict[str, bool] = field(default_factory=lambda: {
        "api": True,
        "web_ui": True,
        "workers": True,
        "scheduler": True,
        "monitoring": True,
        "database": True,
        "redis": True,
        "model_registry": True,
    })
    
    # Resource configuration
    api_resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    worker_resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    ui_resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # Scaling configuration
    api_scaling: ScalingConfiguration = field(default_factory=ScalingConfiguration)
    worker_scaling: ScalingConfiguration = field(default_factory=ScalingConfiguration)
    
    # Health checks
    health_config: HealthCheckConfiguration = field(default_factory=HealthCheckConfiguration)
    
    # Security
    security_config: SecurityConfiguration = field(default_factory=SecurityConfiguration)
    
    # Storage configuration
    persistent_storage_enabled: bool = True
    storage_size: str = "10Gi"
    storage_class: str = "fast-ssd"
    
    # Environment configuration
    environment_variables: Dict[str, str] = field(default_factory=dict)
    config_maps: Dict[str, Dict[str, str]] = field(default_factory=dict)
    secrets: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Monitoring and observability
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    jaeger_enabled: bool = True
    elk_stack_enabled: bool = True
    
    # Networking
    ingress_enabled: bool = True
    ingress_class: str = "nginx"
    tls_enabled: bool = True
    cert_manager_enabled: bool = True
    
    # Deployment metadata
    version: str = "latest"
    image_registry: str = "ghcr.io/pynomaly"
    image_pull_policy: str = "Always"
    
    status: DeploymentStatus = DeploymentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Deployment name cannot be empty")
        if not self.namespace:
            raise ValueError("Namespace cannot be empty")


@dataclass
class OperatorEvent:
    """Kubernetes operator event."""
    
    event_id: UUID
    deployment_id: UUID
    event_type: str  # e.g., "DEPLOYMENT_CREATED", "SCALING_TRIGGERED"
    resource_type: ResourceType
    resource_name: str
    
    # Event details
    action: str  # e.g., "CREATE", "UPDATE", "DELETE", "SCALE"
    reason: str
    message: str
    
    # Event metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL
    source_component: str = "pynomaly-operator"
    
    # Kubernetes metadata
    kubernetes_version: Optional[str] = None
    node_name: Optional[str] = None
    namespace: Optional[str] = None
    
    def __post_init__(self):
        if self.severity not in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Severity must be one of: INFO, WARNING, ERROR, CRITICAL")


@dataclass
class OperatorStatus:
    """Kubernetes operator status."""
    
    operator_id: UUID
    operator_name: str = "pynomaly-operator"
    operator_version: str = "1.0.0"
    
    # Operator state
    is_active: bool = True
    is_healthy: bool = True
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    
    # Managed resources
    managed_deployments: Set[UUID] = field(default_factory=set)
    total_pods_managed: int = 0
    total_services_managed: int = 0
    
    # Performance metrics
    reconciliation_count: int = 0
    reconciliation_errors: int = 0
    average_reconciliation_time: float = 0.0
    
    # Resource usage
    operator_cpu_usage: float = 0.0  # CPU cores
    operator_memory_usage: int = 0  # Bytes
    
    # Kubernetes cluster info
    cluster_version: Optional[str] = None
    cluster_nodes: int = 0
    cluster_cpu_capacity: float = 0.0
    cluster_memory_capacity: int = 0
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourceMonitoring:
    """Resource monitoring and metrics."""
    
    monitoring_id: UUID
    deployment_id: UUID
    resource_name: str
    resource_type: ResourceType
    
    # CPU metrics
    cpu_usage_cores: float = 0.0
    cpu_usage_percentage: float = 0.0
    cpu_throttling_time: float = 0.0
    
    # Memory metrics
    memory_usage_bytes: int = 0
    memory_usage_percentage: float = 0.0
    memory_cache_bytes: int = 0
    memory_rss_bytes: int = 0
    
    # Network metrics
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    network_rx_packets: int = 0
    network_tx_packets: int = 0
    
    # Storage metrics
    storage_usage_bytes: int = 0
    storage_available_bytes: int = 0
    storage_iops_read: float = 0.0
    storage_iops_write: float = 0.0
    
    # Application metrics
    request_count: int = 0
    request_rate: float = 0.0
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    error_rate: float = 0.0
    
    # Health status
    pod_ready: bool = True
    pod_status: str = "Running"
    restart_count: int = 0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_health_score(self) -> float:
        """Calculate overall health score."""
        if not self.pod_ready or self.pod_status != "Running":
            return 0.0
        
        # Factor in resource usage and error rates
        cpu_score = max(0, 1 - (self.cpu_usage_percentage / 100))
        memory_score = max(0, 1 - (self.memory_usage_percentage / 100))
        error_score = max(0, 1 - (self.error_rate * 10))  # Assume error_rate is 0-1
        
        return (cpu_score + memory_score + error_score) / 3


@dataclass
class AutoScalingDecision:
    """Auto-scaling decision record."""
    
    decision_id: UUID
    deployment_id: UUID
    current_replicas: int
    target_replicas: int
    
    # Decision rationale
    trigger_metric: str  # e.g., "cpu_usage", "memory_usage", "custom_metric"
    trigger_value: float
    threshold_value: float
    scaling_direction: str  # "up", "down", "none"
    
    # Decision factors
    cpu_utilization: float
    memory_utilization: float
    queue_length: Optional[int] = None
    response_time: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Constraints
    min_replicas_constraint: int
    max_replicas_constraint: int
    cooldown_period: int  # seconds
    last_scaling_time: Optional[datetime] = None
    
    # Decision outcome
    decision_made: bool = False
    scaling_executed: bool = False
    scaling_reason: str = ""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def should_scale(self) -> bool:
        """Determine if scaling should be executed."""
        if not self.decision_made:
            return False
        
        # Check cooldown period
        if self.last_scaling_time:
            time_since_last_scaling = (datetime.utcnow() - self.last_scaling_time).total_seconds()
            if time_since_last_scaling < self.cooldown_period:
                return False
        
        # Check replica constraints
        if self.target_replicas < self.min_replicas_constraint:
            return False
        if self.target_replicas > self.max_replicas_constraint:
            return False
        
        return self.current_replicas != self.target_replicas