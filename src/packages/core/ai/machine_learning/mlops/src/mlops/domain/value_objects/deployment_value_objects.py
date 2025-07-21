"""Value objects for ML deployments."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from enum import Enum


class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    CREATED = "created"
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    UPDATING = "updating"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    STOPPED = "stopped"
    TERMINATED = "terminated"


class DeploymentType(Enum):
    """Deployment type enumeration."""
    BATCH = "batch"
    REAL_TIME = "real_time"
    STREAMING = "streaming"
    EDGE = "edge"
    SERVERLESS = "serverless"
    CONTAINER = "container"
    KUBERNETES = "kubernetes"


class ScalingType(Enum):
    """Scaling type enumeration."""
    MANUAL = "manual"
    AUTO = "auto"
    SCHEDULED = "scheduled"


@dataclass(frozen=True)
class DeploymentId:
    """Unique identifier for ML deployments."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class DeploymentConfiguration:
    """Configuration for ML deployments."""
    name: str
    deployment_type: DeploymentType
    model_id: str
    model_version: str
    environment: str = "production"
    replicas: int = 1
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    gpu_limit: Optional[str] = None
    auto_scaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    health_check_path: str = "/health"
    readiness_timeout: int = 30
    liveness_timeout: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "deployment_type": self.deployment_type.value,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "environment": self.environment,
            "replicas": self.replicas,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "gpu_limit": self.gpu_limit,
            "auto_scaling": self.auto_scaling,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_cpu_utilization": self.target_cpu_utilization,
            "health_check_path": self.health_check_path,
            "readiness_timeout": self.readiness_timeout,
            "liveness_timeout": self.liveness_timeout,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentConfiguration":
        """Create configuration from dictionary."""
        return cls(
            name=data["name"],
            deployment_type=DeploymentType(data["deployment_type"]),
            model_id=data["model_id"],
            model_version=data["model_version"],
            environment=data.get("environment", "production"),
            replicas=data.get("replicas", 1),
            cpu_limit=data.get("cpu_limit"),
            memory_limit=data.get("memory_limit"),
            gpu_limit=data.get("gpu_limit"),
            auto_scaling=data.get("auto_scaling", False),
            min_replicas=data.get("min_replicas", 1),
            max_replicas=data.get("max_replicas", 10),
            target_cpu_utilization=data.get("target_cpu_utilization", 70),
            health_check_path=data.get("health_check_path", "/health"),
            readiness_timeout=data.get("readiness_timeout", 30),
            liveness_timeout=data.get("liveness_timeout", 30),
        )


@dataclass(frozen=True)
class DeploymentMetrics:
    """Metrics for ML deployments."""
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    model_accuracy: Optional[float] = None
    model_drift: Optional[float] = None
    uptime: float = 0.0
    availability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "requests_per_second": self.requests_per_second,
            "average_response_time": self.average_response_time,
            "error_rate": self.error_rate,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "gpu_utilization": self.gpu_utilization,
            "model_accuracy": self.model_accuracy,
            "model_drift": self.model_drift,
            "uptime": self.uptime,
            "availability": self.availability,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentMetrics":
        """Create metrics from dictionary."""
        return cls(
            requests_per_second=data.get("requests_per_second", 0.0),
            average_response_time=data.get("average_response_time", 0.0),
            error_rate=data.get("error_rate", 0.0),
            cpu_utilization=data.get("cpu_utilization", 0.0),
            memory_utilization=data.get("memory_utilization", 0.0),
            gpu_utilization=data.get("gpu_utilization", 0.0),
            model_accuracy=data.get("model_accuracy"),
            model_drift=data.get("model_drift"),
            uptime=data.get("uptime", 0.0),
            availability=data.get("availability", 0.0),
        )


@dataclass(frozen=True)
class DeploymentHistory:
    """History of deployment changes."""
    version: str
    deployed_at: datetime
    deployed_by: str
    changes: List[str] = field(default_factory=list)
    rollback_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary."""
        return {
            "version": self.version,
            "deployed_at": self.deployed_at.isoformat(),
            "deployed_by": self.deployed_by,
            "changes": self.changes,
            "rollback_info": self.rollback_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentHistory":
        """Create history from dictionary."""
        return cls(
            version=data["version"],
            deployed_at=datetime.fromisoformat(data["deployed_at"]),
            deployed_by=data["deployed_by"],
            changes=data.get("changes", []),
            rollback_info=data.get("rollback_info"),
        )


@dataclass(frozen=True)
class DeploymentMetadata:
    """Metadata for ML deployments."""
    created_by: str
    project_name: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "created_by": self.created_by,
            "project_name": self.project_name,
            "description": self.description,
            "tags": self.tags,
            "endpoint_url": self.endpoint_url,
            "api_key": self.api_key,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentMetadata":
        """Create metadata from dictionary."""
        return cls(
            created_by=data["created_by"],
            project_name=data["project_name"],
            description=data.get("description"),
            tags=data.get("tags", []),
            endpoint_url=data.get("endpoint_url"),
            api_key=data.get("api_key"),
        )