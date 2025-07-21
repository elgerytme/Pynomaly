"""
Service Mesh domain entities for cloud-native microservices communication.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ServiceMeshType(str, Enum):
    """Service mesh implementations."""
    ISTIO = "istio"
    LINKERD = "linkerd"
    CONSUL_CONNECT = "consul_connect"
    ENVOY = "envoy"
    TRAEFIK_MESH = "traefik_mesh"
    KUMA = "kuma"
    CUSTOM = "custom"


class TrafficPolicyType(str, Enum):
    """Traffic management policy types."""
    LOAD_BALANCING = "load_balancing"
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY = "retry"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    FAULT_INJECTION = "fault_injection"
    MIRRORING = "mirroring"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"


class SecurityPolicyType(str, Enum):
    """Security policy types."""
    MTLS = "mtls"
    AUTHORIZATION = "authorization"
    AUTHENTICATION = "authentication"
    RBAC = "rbac"
    JWT_VALIDATION = "jwt_validation"
    OAUTH = "oauth"
    REQUEST_AUTHENTICATION = "request_authentication"


class ObservabilityFeature(str, Enum):
    """Observability features."""
    TRACING = "tracing"
    METRICS = "metrics"
    LOGGING = "logging"
    ACCESS_LOGS = "access_logs"
    DISTRIBUTED_TRACING = "distributed_tracing"
    SERVICE_MAP = "service_map"


class ServiceMeshConfiguration(BaseModel):
    """
    Service mesh configuration and management.
    
    Defines the configuration for a service mesh deployment
    including traffic management, security, and observability settings.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Configuration identifier")
    
    # Mesh identification
    name: str = Field(..., description="Service mesh name")
    mesh_type: ServiceMeshType = Field(..., description="Service mesh type")
    version: str = Field(..., description="Service mesh version")
    namespace: str = Field(default="istio-system", description="Control plane namespace")
    
    # Ownership
    tenant_id: UUID = Field(..., description="Owning tenant")
    cluster_id: Optional[UUID] = Field(None, description="Associated cluster ID")
    
    # Control plane configuration
    control_plane_config: Dict[str, Any] = Field(default_factory=dict, description="Control plane settings")
    data_plane_config: Dict[str, Any] = Field(default_factory=dict, description="Data plane settings")
    
    # Traffic management
    traffic_management_enabled: bool = Field(default=True, description="Enable traffic management")
    load_balancing_algorithm: str = Field(default="round_robin", description="Default load balancing")
    circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker")
    retry_policy: Dict[str, Any] = Field(default_factory=dict, description="Default retry policy")
    timeout_policy: Dict[str, Any] = Field(default_factory=dict, description="Default timeout policy")
    
    # Security configuration
    mtls_enabled: bool = Field(default=True, description="Enable mutual TLS")
    mtls_mode: str = Field(default="STRICT", description="mTLS mode (STRICT, PERMISSIVE)")
    authorization_enabled: bool = Field(default=True, description="Enable authorization policies")
    jwt_validation_enabled: bool = Field(default=False, description="Enable JWT validation")
    
    # Observability
    observability_features: List[ObservabilityFeature] = Field(default_factory=list, description="Enabled observability features")
    tracing_sampling_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Tracing sampling rate")
    metrics_retention_days: int = Field(default=15, description="Metrics retention period")
    
    # Ingress and egress
    ingress_gateway_enabled: bool = Field(default=True, description="Enable ingress gateway")
    egress_gateway_enabled: bool = Field(default=False, description="Enable egress gateway")
    external_traffic_policy: str = Field(default="Cluster", description="External traffic policy")
    
    # Deployment settings
    sidecar_injection_enabled: bool = Field(default=True, description="Enable automatic sidecar injection")
    injection_policy: str = Field(default="enabled", description="Sidecar injection policy")
    resource_requirements: Dict[str, Dict[str, str]] = Field(default_factory=dict, description="Resource requirements")
    
    # High availability
    replicas: Dict[str, int] = Field(default_factory=dict, description="Component replica counts")
    anti_affinity_enabled: bool = Field(default=True, description="Enable pod anti-affinity")
    
    # Configuration status
    installation_status: str = Field(default="pending", description="Installation status")
    health_status: str = Field(default="unknown", description="Health status")
    last_health_check: Optional[datetime] = Field(None, description="Last health check")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    installed_at: Optional[datetime] = Field(None, description="Installation timestamp")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_installed(self) -> bool:
        """Check if service mesh is installed."""
        return self.installation_status == "installed"
    
    def is_healthy(self) -> bool:
        """Check if service mesh is healthy."""
        return self.health_status == "healthy"
    
    def get_default_retry_policy(self) -> Dict[str, Any]:
        """Get default retry policy."""
        return self.retry_policy or {
            "attempts": 3,
            "perTryTimeout": "2s",
            "retryOn": "gateway-error,connect-failure,refused-stream"
        }
    
    def get_default_timeout_policy(self) -> Dict[str, Any]:
        """Get default timeout policy."""
        return self.timeout_policy or {
            "timeout": "30s"
        }
    
    def add_observability_feature(self, feature: ObservabilityFeature) -> None:
        """Add observability feature."""
        if feature not in self.observability_features:
            self.observability_features.append(feature)
            self.updated_at = datetime.utcnow()
    
    def update_health_status(self, status: str) -> None:
        """Update health status."""
        self.health_status = status
        self.last_health_check = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class ServiceMeshService(BaseModel):
    """
    Service mesh service registration and configuration.
    
    Represents a service within the service mesh with its
    traffic policies, security settings, and observability configuration.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Service identifier")
    
    # Service identification
    name: str = Field(..., description="Service name")
    namespace: str = Field(default="default", description="Service namespace")
    service_mesh_id: UUID = Field(..., description="Associated service mesh ID")
    
    # Service configuration
    service_type: str = Field(default="ClusterIP", description="Kubernetes service type")
    ports: List[Dict[str, Any]] = Field(..., description="Service ports")
    selector: Dict[str, str] = Field(..., description="Pod selector")
    
    # Traffic management
    traffic_policies: List[UUID] = Field(default_factory=list, description="Applied traffic policies")
    load_balancer_policy: Optional[str] = Field(None, description="Load balancer policy")
    session_affinity: str = Field(default="None", description="Session affinity")
    
    # Security policies
    security_policies: List[UUID] = Field(default_factory=list, description="Applied security policies")
    mtls_required: bool = Field(default=True, description="Require mutual TLS")
    
    # Sidecar configuration
    sidecar_injection: bool = Field(default=True, description="Enable sidecar injection")
    sidecar_config: Dict[str, Any] = Field(default_factory=dict, description="Sidecar configuration")
    
    # Observability
    telemetry_enabled: bool = Field(default=True, description="Enable telemetry")
    access_logging_enabled: bool = Field(default=True, description="Enable access logging")
    distributed_tracing_enabled: bool = Field(default=True, description="Enable distributed tracing")
    
    # Health configuration
    health_check_path: Optional[str] = Field(None, description="Health check path")
    readiness_probe: Optional[Dict[str, Any]] = Field(None, description="Readiness probe")
    liveness_probe: Optional[Dict[str, Any]] = Field(None, description="Liveness probe")
    
    # Status
    status: str = Field(default="pending", description="Service status")
    endpoints_ready: int = Field(default=0, description="Ready endpoints")
    endpoints_total: int = Field(default=0, description="Total endpoints")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self.status == "running" and self.endpoints_ready > 0
    
    def get_endpoint_ratio(self) -> float:
        """Get ready endpoint ratio."""
        if self.endpoints_total == 0:
            return 0.0
        return self.endpoints_ready / self.endpoints_total
    
    def add_traffic_policy(self, policy_id: UUID) -> None:
        """Add traffic policy to service."""
        if policy_id not in self.traffic_policies:
            self.traffic_policies.append(policy_id)
            self.updated_at = datetime.utcnow()
    
    def add_security_policy(self, policy_id: UUID) -> None:
        """Add security policy to service."""
        if policy_id not in self.security_policies:
            self.security_policies.append(policy_id)
            self.updated_at = datetime.utcnow()


class TrafficPolicy(BaseModel):
    """
    Service mesh traffic management policy.
    
    Defines traffic routing, load balancing, fault tolerance,
    and other traffic management behaviors for services.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Policy identifier")
    
    # Policy identification
    name: str = Field(..., description="Policy name")
    policy_type: TrafficPolicyType = Field(..., description="Policy type")
    service_mesh_id: UUID = Field(..., description="Associated service mesh ID")
    
    # Target specification
    target_services: List[str] = Field(..., description="Target services")
    target_namespaces: List[str] = Field(default_factory=list, description="Target namespaces")
    match_labels: Dict[str, str] = Field(default_factory=dict, description="Service match labels")
    
    # Policy configuration
    policy_config: Dict[str, Any] = Field(..., description="Policy-specific configuration")
    
    # Load balancing configuration
    load_balancer_type: Optional[str] = Field(None, description="Load balancer type")
    consistent_hash: Optional[Dict[str, Any]] = Field(None, description="Consistent hash config")
    
    # Circuit breaker configuration
    circuit_breaker_config: Optional[Dict[str, Any]] = Field(None, description="Circuit breaker settings")
    
    # Retry configuration
    retry_config: Optional[Dict[str, Any]] = Field(None, description="Retry policy settings")
    
    # Timeout configuration
    timeout_config: Optional[Dict[str, Any]] = Field(None, description="Timeout settings")
    
    # Rate limiting
    rate_limit_config: Optional[Dict[str, Any]] = Field(None, description="Rate limiting settings")
    
    # Fault injection
    fault_injection_config: Optional[Dict[str, Any]] = Field(None, description="Fault injection settings")
    
    # Traffic splitting
    traffic_split_config: Optional[Dict[str, Any]] = Field(None, description="Traffic splitting config")
    
    # Policy status
    status: str = Field(default="pending", description="Policy status")
    applied_to_services: List[str] = Field(default_factory=list, description="Services with policy applied")
    
    # Priority and ordering
    priority: int = Field(default=100, description="Policy priority")
    execution_order: int = Field(default=100, description="Execution order")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = Field(None, description="Policy application time")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_applied(self) -> bool:
        """Check if policy is applied."""
        return self.status == "applied" and len(self.applied_to_services) > 0
    
    def get_circuit_breaker_config(self) -> Dict[str, Any]:
        """Get circuit breaker configuration."""
        return self.circuit_breaker_config or {
            "consecutive5xxErrors": 5,
            "consecutive4xxErrors": 10,
            "interval": "30s",
            "baseEjectionTime": "30s",
            "maxEjectionPercent": 50,
            "minHealthPercent": 50
        }
    
    def get_retry_config(self) -> Dict[str, Any]:
        """Get retry configuration."""
        return self.retry_config or {
            "attempts": 3,
            "perTryTimeout": "2s",
            "retryOn": "gateway-error,connect-failure,refused-stream"
        }


class SecurityPolicy(BaseModel):
    """
    Service mesh security policy.
    
    Defines authentication, authorization, and encryption
    policies for service-to-service communication.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Security policy identifier")
    
    # Policy identification
    name: str = Field(..., description="Policy name")
    policy_type: SecurityPolicyType = Field(..., description="Security policy type")
    service_mesh_id: UUID = Field(..., description="Associated service mesh ID")
    
    # Target specification
    target_services: List[str] = Field(..., description="Target services")
    target_namespaces: List[str] = Field(default_factory=list, description="Target namespaces")
    
    # Authentication configuration
    authentication_config: Optional[Dict[str, Any]] = Field(None, description="Authentication settings")
    jwt_config: Optional[Dict[str, Any]] = Field(None, description="JWT validation config")
    oauth_config: Optional[Dict[str, Any]] = Field(None, description="OAuth configuration")
    
    # Authorization configuration
    authorization_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Authorization rules")
    rbac_config: Optional[Dict[str, Any]] = Field(None, description="RBAC configuration")
    
    # Mutual TLS configuration
    mtls_config: Optional[Dict[str, Any]] = Field(None, description="Mutual TLS settings")
    tls_mode: str = Field(default="STRICT", description="TLS mode")
    
    # Certificate configuration
    certificate_config: Optional[Dict[str, Any]] = Field(None, description="Certificate settings")
    ca_certificate: Optional[str] = Field(None, description="CA certificate")
    
    # Policy status
    status: str = Field(default="pending", description="Policy status")
    applied_to_services: List[str] = Field(default_factory=list, description="Services with policy applied")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = Field(None, description="Policy application time")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_applied(self) -> bool:
        """Check if security policy is applied."""
        return self.status == "applied" and len(self.applied_to_services) > 0
    
    def requires_mtls(self) -> bool:
        """Check if policy requires mutual TLS."""
        return self.tls_mode == "STRICT"
    
    def add_authorization_rule(self, rule: Dict[str, Any]) -> None:
        """Add authorization rule."""
        self.authorization_rules.append(rule)
        self.updated_at = datetime.utcnow()


class ServiceMeshGateway(BaseModel):
    """
    Service mesh gateway configuration.
    
    Manages ingress and egress gateways for service mesh,
    handling external traffic routing and load balancing.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Gateway identifier")
    
    # Gateway identification
    name: str = Field(..., description="Gateway name")
    gateway_type: str = Field(..., description="Gateway type (ingress/egress)")
    service_mesh_id: UUID = Field(..., description="Associated service mesh ID")
    
    # Gateway configuration
    listeners: List[Dict[str, Any]] = Field(..., description="Gateway listeners")
    routes: List[Dict[str, Any]] = Field(default_factory=list, description="Route configurations")
    
    # TLS configuration
    tls_config: Optional[Dict[str, Any]] = Field(None, description="TLS configuration")
    certificates: List[str] = Field(default_factory=list, description="TLS certificates")
    
    # Load balancing
    load_balancer_config: Optional[Dict[str, Any]] = Field(None, description="Load balancer configuration")
    
    # Health checks
    health_check_config: Optional[Dict[str, Any]] = Field(None, description="Health check configuration")
    
    # Status
    status: str = Field(default="pending", description="Gateway status")
    external_ip: Optional[str] = Field(None, description="External IP address")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_ready(self) -> bool:
        """Check if gateway is ready."""
        return self.status == "ready" and self.external_ip is not None
    
    def add_route(self, route_config: Dict[str, Any]) -> None:
        """Add route configuration."""
        self.routes.append(route_config)
        self.updated_at = datetime.utcnow()
    
    def update_external_ip(self, ip: str) -> None:
        """Update external IP address."""
        self.external_ip = ip
        self.updated_at = datetime.utcnow()