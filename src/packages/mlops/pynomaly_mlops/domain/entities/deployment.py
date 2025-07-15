"""Deployment Entities

Domain entities for model deployment and environment management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4


class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    ROLLBACK = "rollback"
    ARCHIVED = "archived"


class Environment(Enum):
    """Deployment environment enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    BLUE = "blue"
    GREEN = "green"


class DeploymentStrategy(Enum):
    """Deployment strategy enumeration."""
    DIRECT = "direct"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    A_B_TEST = "a_b_test"


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ScalingConfig:
    """Configuration for deployment scaling."""
    
    min_replicas: int = field(default=1)
    max_replicas: int = field(default=10)
    target_cpu_utilization: float = field(default=70.0)
    target_memory_utilization: float = field(default=80.0)
    scale_up_cooldown_seconds: int = field(default=300)
    scale_down_cooldown_seconds: int = field(default=600)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.min_replicas < 1:
            raise ValueError("min_replicas must be at least 1")
        
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")
        
        if not 0 < self.target_cpu_utilization <= 100:
            raise ValueError("target_cpu_utilization must be between 0 and 100")
        
        if not 0 < self.target_memory_utilization <= 100:
            raise ValueError("target_memory_utilization must be between 0 and 100")


@dataclass
class ResourceConfig:
    """Configuration for deployment resources."""
    
    cpu_request: str = field(default="100m")
    cpu_limit: str = field(default="500m")
    memory_request: str = field(default="128Mi")
    memory_limit: str = field(default="512Mi")
    gpu_count: int = field(default=0)
    gpu_type: Optional[str] = field(default=None)
    storage_size: str = field(default="1Gi")
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.gpu_count < 0:
            raise ValueError("gpu_count cannot be negative")
        
        if self.gpu_count > 0 and not self.gpu_type:
            raise ValueError("gpu_type must be specified when gpu_count > 0")


@dataclass
class TrafficConfig:
    """Configuration for traffic routing."""
    
    weight: float = field(default=100.0)
    sticky_sessions: bool = field(default=False)
    session_affinity_timeout: Optional[int] = field(default=None)
    load_balancing_algorithm: str = field(default="round_robin")
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not 0 <= self.weight <= 100:
            raise ValueError("weight must be between 0 and 100")


@dataclass
class HealthCheck:
    """Health check configuration and status."""
    
    # Configuration
    endpoint: str = field(default="/health")
    interval_seconds: int = field(default=30)
    timeout_seconds: int = field(default=5)
    failure_threshold: int = field(default=3)
    success_threshold: int = field(default=1)
    
    # Status
    status: HealthStatus = field(default=HealthStatus.UNKNOWN)
    last_check_at: Optional[datetime] = field(default=None)
    consecutive_failures: int = field(default=0)
    consecutive_successes: int = field(default=0)
    last_error: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
    
    def record_success(self) -> None:
        """Record a successful health check."""
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_check_at = datetime.utcnow()
        self.last_error = None
        
        if self.consecutive_successes >= self.success_threshold:
            self.status = HealthStatus.HEALTHY
    
    def record_failure(self, error: str) -> None:
        """Record a failed health check."""
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_check_at = datetime.utcnow()
        self.last_error = error
        
        if self.consecutive_failures >= self.failure_threshold:
            self.status = HealthStatus.UNHEALTHY


@dataclass
class Deployment:
    """Deployment entity for managing model deployments.
    
    Represents a deployment of a model to a specific environment
    with configuration, scaling, and health monitoring.
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = field()
    model_id: UUID = field()
    
    # Environment and Configuration
    environment: Environment = field()
    strategy: DeploymentStrategy = field(default=DeploymentStrategy.DIRECT)
    namespace: str = field(default="default")
    
    # Status and Lifecycle
    status: DeploymentStatus = field(default=DeploymentStatus.PENDING)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = field(default=None)
    created_by: str = field()
    
    # Configuration
    scaling_config: ScalingConfig = field(default_factory=ScalingConfig)
    resource_config: ResourceConfig = field(default_factory=ResourceConfig)
    traffic_config: TrafficConfig = field(default_factory=TrafficConfig)
    
    # Service Configuration
    service_name: Optional[str] = field(default=None)
    endpoint_url: Optional[str] = field(default=None)
    internal_endpoint: Optional[str] = field(default=None)
    port: int = field(default=8080)
    
    # Container Configuration
    image_uri: Optional[str] = field(default=None)
    image_tag: Optional[str] = field(default=None)
    container_env: Dict[str, str] = field(default_factory=dict)
    container_args: List[str] = field(default_factory=list)
    
    # Health and Monitoring
    health_check: HealthCheck = field(default_factory=HealthCheck)
    current_replicas: int = field(default=0)
    ready_replicas: int = field(default=0)
    
    # Performance Metrics
    request_count: int = field(default=0)
    error_count: int = field(default=0)
    avg_response_time_ms: float = field(default=0.0)
    p99_response_time_ms: float = field(default=0.0)
    
    # Deployment History
    previous_deployment_id: Optional[UUID] = field(default=None)
    rollback_deployment_id: Optional[UUID] = field(default=None)
    deployment_version: int = field(default=1)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Deployment name cannot be empty")
        
        if not self.created_by:
            raise ValueError("Deployment must have a creator")
        
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        
        # Generate service name if not provided
        if not self.service_name:
            self.service_name = f"{self.name}-{self.environment.value}"
        
        # Ensure tags are unique
        self.tags = list(set(self.tags))
    
    def start_deployment(self) -> None:
        """Start the deployment process."""
        self.status = DeploymentStatus.DEPLOYING
        self.updated_at = datetime.utcnow()
    
    def complete_deployment(self, endpoint_url: str) -> None:
        """Mark deployment as successfully completed."""
        self.status = DeploymentStatus.ACTIVE
        self.deployed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.endpoint_url = endpoint_url
    
    def fail_deployment(self, error_message: str) -> None:
        """Mark deployment as failed."""
        self.status = DeploymentStatus.FAILED
        self.updated_at = datetime.utcnow()
        self.annotations["last_error"] = error_message
    
    def deactivate(self) -> None:
        """Deactivate the deployment."""
        self.status = DeploymentStatus.INACTIVE
        self.updated_at = datetime.utcnow()
    
    def start_rollback(self, target_deployment_id: UUID) -> None:
        """Start rollback to a previous deployment."""
        self.status = DeploymentStatus.ROLLBACK
        self.rollback_deployment_id = target_deployment_id
        self.updated_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the deployment."""
        self.status = DeploymentStatus.ARCHIVED
        self.updated_at = datetime.utcnow()
    
    def update_replica_count(self, current: int, ready: int) -> None:
        """Update replica counts."""
        self.current_replicas = current
        self.ready_replicas = ready
        self.updated_at = datetime.utcnow()
    
    def update_performance_metrics(
        self,
        request_count: int,
        error_count: int,
        avg_response_time_ms: float,
        p99_response_time_ms: float
    ) -> None:
        """Update performance metrics."""
        self.request_count = request_count
        self.error_count = error_count
        self.avg_response_time_ms = avg_response_time_ms
        self.p99_response_time_ms = p99_response_time_ms
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the deployment."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the deployment."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def set_label(self, key: str, value: str) -> None:
        """Set a label on the deployment."""
        self.labels[key] = value
    
    def remove_label(self, key: str) -> None:
        """Remove a label from the deployment."""
        if key in self.labels:
            del self.labels[key]
    
    @property
    def is_active(self) -> bool:
        """Check if deployment is active."""
        return self.status == DeploymentStatus.ACTIVE
    
    @property
    def is_healthy(self) -> bool:
        """Check if deployment is healthy."""
        return (
            self.is_active and
            self.health_check.status == HealthStatus.HEALTHY and
            self.ready_replicas > 0
        )
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    @property
    def availability_percentage(self) -> float:
        """Calculate availability percentage based on ready replicas."""
        if self.scaling_config.min_replicas == 0:
            return 100.0
        return (self.ready_replicas / self.scaling_config.min_replicas) * 100
    
    def create_next_version(self, created_by: str) -> "Deployment":
        """Create next version of this deployment.
        
        Args:
            created_by: User creating the new version
            
        Returns:
            New Deployment instance with incremented version
        """
        return Deployment(
            name=self.name,
            model_id=self.model_id,
            environment=self.environment,
            strategy=self.strategy,
            namespace=self.namespace,
            created_by=created_by,
            scaling_config=self.scaling_config,
            resource_config=self.resource_config,
            traffic_config=self.traffic_config,
            port=self.port,
            container_env=self.container_env.copy(),
            container_args=self.container_args.copy(),
            health_check=HealthCheck(
                endpoint=self.health_check.endpoint,
                interval_seconds=self.health_check.interval_seconds,
                timeout_seconds=self.health_check.timeout_seconds,
                failure_threshold=self.health_check.failure_threshold,
                success_threshold=self.health_check.success_threshold,
            ),
            previous_deployment_id=self.id,
            deployment_version=self.deployment_version + 1,
            tags=self.tags.copy(),
            labels=self.labels.copy(),
            description=self.description,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert deployment to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "model_id": str(self.model_id),
            "environment": self.environment.value,
            "strategy": self.strategy.value,
            "namespace": self.namespace,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "created_by": self.created_by,
            "scaling_config": {
                "min_replicas": self.scaling_config.min_replicas,
                "max_replicas": self.scaling_config.max_replicas,
                "target_cpu_utilization": self.scaling_config.target_cpu_utilization,
                "target_memory_utilization": self.scaling_config.target_memory_utilization,
            },
            "resource_config": {
                "cpu_request": self.resource_config.cpu_request,
                "cpu_limit": self.resource_config.cpu_limit,
                "memory_request": self.resource_config.memory_request,
                "memory_limit": self.resource_config.memory_limit,
                "gpu_count": self.resource_config.gpu_count,
                "gpu_type": self.resource_config.gpu_type,
            },
            "service_name": self.service_name,
            "endpoint_url": self.endpoint_url,
            "internal_endpoint": self.internal_endpoint,
            "port": self.port,
            "health_status": self.health_check.status.value,
            "current_replicas": self.current_replicas,
            "ready_replicas": self.ready_replicas,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
            "p99_response_time_ms": self.p99_response_time_ms,
            "availability_percentage": self.availability_percentage,
            "deployment_version": self.deployment_version,
            "tags": self.tags,
            "labels": self.labels,
            "description": self.description,
        }