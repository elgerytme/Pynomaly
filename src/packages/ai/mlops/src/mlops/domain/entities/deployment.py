"""Deployment domain entity for MLOps package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4


class DeploymentStatus(str, Enum):
    """Deployment status enumeration."""
    
    CREATED = "created"
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    TERMINATED = "terminated"
    SCALING = "scaling"
    UPDATING = "updating"


class DeploymentType(str, Enum):
    """Deployment type enumeration."""
    
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    A_B_TEST = "a_b_test"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"


class Environment(str, Enum):
    """Deployment environment enumeration."""
    
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class DeploymentMetrics:
    """Deployment performance metrics."""
    
    requests_per_minute: float = 0.0
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    uptime_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "average_latency_ms": self.average_latency_ms,
            "error_rate": self.error_rate,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "uptime_percentage": self.uptime_percentage,
        }


@dataclass
class ResourceConfig:
    """Resource configuration for deployment."""
    
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "256Mi"
    memory_limit: str = "1Gi"
    replicas: int = 1
    min_replicas: int = 1
    max_replicas: int = 10
    
    def validate(self) -> None:
        """Validate resource configuration."""
        if self.replicas < 1:
            raise ValueError("Replicas must be at least 1")
        
        if self.min_replicas < 1:
            raise ValueError("Minimum replicas must be at least 1")
        
        if self.max_replicas < self.min_replicas:
            raise ValueError("Maximum replicas must be >= minimum replicas")
        
        if self.replicas < self.min_replicas or self.replicas > self.max_replicas:
            raise ValueError("Replicas must be between min and max replicas")


@dataclass
class Deployment:
    """Deployment domain entity.
    
    Represents a model deployment with lifecycle management,
    following Domain-Driven Design principles.
    """
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    status: DeploymentStatus = DeploymentStatus.CREATED
    deployment_type: DeploymentType = DeploymentType.REAL_TIME
    environment: Environment = Environment.DEVELOPMENT
    
    # Model association
    model_id: UUID = field(default_factory=uuid4)
    model_version: str = "latest"
    
    # Deployment configuration
    config: Dict[str, Any] = field(default_factory=dict)
    resource_config: ResourceConfig = field(default_factory=ResourceConfig)
    endpoint_url: Optional[str] = None
    
    # Traffic management
    traffic_percentage: int = 100
    
    # Monitoring
    metrics: DeploymentMetrics = field(default_factory=DeploymentMetrics)
    health_check_url: Optional[str] = None
    health_status: str = "unknown"
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = None
    terminated_at: Optional[datetime] = None
    
    # User tracking
    created_by: str = ""
    team: str = ""
    
    # Tags and labels
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        if not self.name:
            self.name = f"deployment_{self.id.hex[:8]}"
        
        self._validate_deployment()
    
    def _validate_deployment(self) -> None:
        """Validate deployment state and business rules."""
        if not self.name or len(self.name.strip()) == 0:
            raise ValueError("Deployment name cannot be empty")
        
        if len(self.name) > 100:
            raise ValueError("Deployment name cannot exceed 100 characters")
        
        if self.deployment_type not in DeploymentType:
            raise ValueError(f"Invalid deployment type: {self.deployment_type}")
        
        if self.environment not in Environment:
            raise ValueError(f"Invalid environment: {self.environment}")
        
        if not (0 <= self.traffic_percentage <= 100):
            raise ValueError("Traffic percentage must be between 0 and 100")
        
        self.resource_config.validate()
    
    def deploy(self, endpoint_url: Optional[str] = None) -> None:
        """Deploy the model.
        
        Args:
            endpoint_url: Optional endpoint URL for the deployment
            
        Raises:
            ValueError: If deployment is not in valid state for deployment
        """
        if self.status not in [DeploymentStatus.CREATED, DeploymentStatus.FAILED]:
            raise ValueError(f"Cannot deploy from {self.status} status")
        
        self.status = DeploymentStatus.DEPLOYING
        self.deployed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        if endpoint_url:
            self.endpoint_url = endpoint_url
    
    def activate(self, endpoint_url: str) -> None:
        """Activate the deployment.
        
        Args:
            endpoint_url: Endpoint URL for the active deployment
            
        Raises:
            ValueError: If deployment is not in DEPLOYING status
        """
        if self.status != DeploymentStatus.DEPLOYING:
            raise ValueError(f"Cannot activate deployment in {self.status} status")
        
        self.status = DeploymentStatus.ACTIVE
        self.endpoint_url = endpoint_url
        self.health_status = "healthy"
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate the deployment.
        
        Raises:
            ValueError: If deployment is not active
        """
        if self.status != DeploymentStatus.ACTIVE:
            raise ValueError(f"Cannot deactivate deployment in {self.status} status")
        
        self.status = DeploymentStatus.INACTIVE
        self.health_status = "inactive"
        self.updated_at = datetime.utcnow()
    
    def terminate(self) -> None:
        """Terminate the deployment.
        
        Raises:
            ValueError: If deployment is already terminated
        """
        if self.status == DeploymentStatus.TERMINATED:
            raise ValueError("Deployment is already terminated")
        
        self.status = DeploymentStatus.TERMINATED
        self.terminated_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.health_status = "terminated"
    
    def fail(self, error_message: str = "") -> None:
        """Mark deployment as failed.
        
        Args:
            error_message: Optional error message
        """
        self.status = DeploymentStatus.FAILED
        self.updated_at = datetime.utcnow()
        
        if error_message:
            self.config["error_message"] = error_message
    
    def scale(self, replicas: int) -> None:
        """Scale the deployment.
        
        Args:
            replicas: Target number of replicas
            
        Raises:
            ValueError: If replicas is invalid
        """
        if replicas < self.resource_config.min_replicas:
            raise ValueError(f"Replicas cannot be less than minimum: {self.resource_config.min_replicas}")
        
        if replicas > self.resource_config.max_replicas:
            raise ValueError(f"Replicas cannot be more than maximum: {self.resource_config.max_replicas}")
        
        old_replicas = self.resource_config.replicas
        self.resource_config.replicas = replicas
        
        if self.status == DeploymentStatus.ACTIVE:
            self.status = DeploymentStatus.SCALING
        
        self.updated_at = datetime.utcnow()
        
        # Log scaling event
        self.config.setdefault("scaling_history", []).append({
            "timestamp": self.updated_at.isoformat(),
            "from_replicas": old_replicas,
            "to_replicas": replicas,
            "triggered_by": "manual"
        })
    
    def update_traffic(self, percentage: int) -> None:
        """Update traffic percentage for the deployment.
        
        Args:
            percentage: Traffic percentage (0-100)
            
        Raises:
            ValueError: If percentage is invalid
        """
        if not (0 <= percentage <= 100):
            raise ValueError("Traffic percentage must be between 0 and 100")
        
        old_percentage = self.traffic_percentage
        self.traffic_percentage = percentage
        self.updated_at = datetime.utcnow()
        
        # Log traffic update
        self.config.setdefault("traffic_history", []).append({
            "timestamp": self.updated_at.isoformat(),
            "from_percentage": old_percentage,
            "to_percentage": percentage
        })
    
    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """Update deployment metrics.
        
        Args:
            new_metrics: Dictionary of metric updates
        """
        for metric_name, value in new_metrics.items():
            if hasattr(self.metrics, metric_name):
                setattr(self.metrics, metric_name, value)
        
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the deployment.
        
        Args:
            tag: Tag to add
        """
        if not tag or not isinstance(tag, str):
            raise ValueError("Tag must be a non-empty string")
        
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the deployment.
        
        Args:
            tag: Tag to remove
        """
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def set_label(self, key: str, value: str) -> None:
        """Set a label on the deployment.
        
        Args:
            key: Label key
            value: Label value
        """
        if not key or not isinstance(key, str):
            raise ValueError("Label key must be a non-empty string")
        
        if not isinstance(value, str):
            raise ValueError("Label value must be a string")
        
        self.labels[key] = value
        self.updated_at = datetime.utcnow()
    
    def remove_label(self, key: str) -> None:
        """Remove a label from the deployment.
        
        Args:
            key: Label key to remove
        """
        if key in self.labels:
            del self.labels[key]
            self.updated_at = datetime.utcnow()
    
    @property
    def is_active(self) -> bool:
        """Check if deployment is active."""
        return self.status == DeploymentStatus.ACTIVE
    
    @property
    def is_healthy(self) -> bool:
        """Check if deployment is healthy."""
        return self.health_status == "healthy" and self.is_active
    
    @property
    def uptime(self) -> Optional[float]:
        """Get deployment uptime in seconds."""
        if not self.deployed_at:
            return None
        
        end_time = self.terminated_at or datetime.utcnow()
        return (end_time - self.deployed_at).total_seconds()
    
    @property
    def can_receive_traffic(self) -> bool:
        """Check if deployment can receive traffic."""
        return (
            self.is_active and 
            self.traffic_percentage > 0 and 
            self.endpoint_url is not None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert deployment to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "deployment_type": self.deployment_type.value,
            "environment": self.environment.value,
            "model_id": str(self.model_id),
            "model_version": self.model_version,
            "config": self.config,
            "resource_config": {
                "cpu_request": self.resource_config.cpu_request,
                "cpu_limit": self.resource_config.cpu_limit,
                "memory_request": self.resource_config.memory_request,
                "memory_limit": self.resource_config.memory_limit,
                "replicas": self.resource_config.replicas,
                "min_replicas": self.resource_config.min_replicas,
                "max_replicas": self.resource_config.max_replicas,
            },
            "endpoint_url": self.endpoint_url,
            "traffic_percentage": self.traffic_percentage,
            "metrics": self.metrics.to_dict(),
            "health_check_url": self.health_check_url,
            "health_status": self.health_status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "terminated_at": self.terminated_at.isoformat() if self.terminated_at else None,
            "created_by": self.created_by,
            "team": self.team,
            "tags": self.tags,
            "labels": self.labels,
            "is_active": self.is_active,
            "is_healthy": self.is_healthy,
            "uptime": self.uptime,
            "can_receive_traffic": self.can_receive_traffic,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Deployment:
        """Create deployment from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Deployment instance
        """
        # Parse resource config
        resource_config = ResourceConfig()
        if "resource_config" in data:
            rc = data["resource_config"]
            resource_config = ResourceConfig(
                cpu_request=rc.get("cpu_request", "100m"),
                cpu_limit=rc.get("cpu_limit", "500m"),
                memory_request=rc.get("memory_request", "256Mi"),
                memory_limit=rc.get("memory_limit", "1Gi"),
                replicas=rc.get("replicas", 1),
                min_replicas=rc.get("min_replicas", 1),
                max_replicas=rc.get("max_replicas", 10),
            )
        
        # Parse metrics
        metrics = DeploymentMetrics()
        if "metrics" in data:
            m = data["metrics"]
            metrics = DeploymentMetrics(
                requests_per_minute=m.get("requests_per_minute", 0.0),
                average_latency_ms=m.get("average_latency_ms", 0.0),
                error_rate=m.get("error_rate", 0.0),
                cpu_utilization=m.get("cpu_utilization", 0.0),
                memory_utilization=m.get("memory_utilization", 0.0),
                uptime_percentage=m.get("uptime_percentage", 0.0),
            )
        
        deployment = cls(
            id=UUID(data["id"]) if data.get("id") else uuid4(),
            name=data.get("name", ""),
            description=data.get("description", ""),
            status=DeploymentStatus(data.get("status", DeploymentStatus.CREATED)),
            deployment_type=DeploymentType(data.get("deployment_type", DeploymentType.REAL_TIME)),
            environment=Environment(data.get("environment", Environment.DEVELOPMENT)),
            model_id=UUID(data["model_id"]) if data.get("model_id") else uuid4(),
            model_version=data.get("model_version", "latest"),
            config=data.get("config", {}),
            resource_config=resource_config,
            endpoint_url=data.get("endpoint_url"),
            traffic_percentage=data.get("traffic_percentage", 100),
            metrics=metrics,
            health_check_url=data.get("health_check_url"),
            health_status=data.get("health_status", "unknown"),
            created_by=data.get("created_by", ""),
            team=data.get("team", ""),
            tags=data.get("tags", []),
            labels=data.get("labels", {}),
        )
        
        # Handle datetime fields
        if data.get("created_at"):
            deployment.created_at = datetime.fromisoformat(data["created_at"])
        
        if data.get("updated_at"):
            deployment.updated_at = datetime.fromisoformat(data["updated_at"])
        
        if data.get("deployed_at"):
            deployment.deployed_at = datetime.fromisoformat(data["deployed_at"])
        
        if data.get("terminated_at"):
            deployment.terminated_at = datetime.fromisoformat(data["terminated_at"])
        
        return deployment
    
    def __str__(self) -> str:
        """String representation of the deployment."""
        return f"Deployment(id={self.id.hex[:8]}, name='{self.name}', status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the deployment."""
        return (
            f"Deployment("
            f"id={self.id}, "
            f"name='{self.name}', "
            f"status={self.status.value}, "
            f"environment={self.environment.value}, "
            f"replicas={self.resource_config.replicas}"
            f")"
        )