"""Value objects for infrastructure entities."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from enum import Enum


class InfrastructureType(Enum):
    """Infrastructure type enumeration."""
    CLOUD = "cloud"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"
    EDGE = "edge"


@dataclass(frozen=True)
class InfrastructureId:
    """Unique identifier for infrastructure."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ServiceId:
    """Unique identifier for services."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ResourceId:
    """Unique identifier for resources."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class DeploymentId:
    """Unique identifier for deployments."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class EnvironmentId:
    """Unique identifier for environments."""
    value: UUID = field(default_factory=uuid4)
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class InfrastructureConfiguration:
    """Configuration for infrastructure."""
    provider: str
    region: str
    vpc_config: Dict[str, Any] = field(default_factory=dict)
    security_groups: List[str] = field(default_factory=list)
    subnets: List[str] = field(default_factory=list)
    load_balancers: List[str] = field(default_factory=list)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "provider": self.provider,
            "region": self.region,
            "vpc_config": self.vpc_config,
            "security_groups": self.security_groups,
            "subnets": self.subnets,
            "load_balancers": self.load_balancers,
            "monitoring_config": self.monitoring_config,
        }


@dataclass(frozen=True)
class InfrastructureMetrics:
    """Measurements for infrastructure."""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0
    network_throughput: float = 0.0
    availability: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    cost_per_hour: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert measurements to dictionary."""
        return {
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "disk_utilization": self.disk_utilization,
            "network_throughput": self.network_throughput,
            "availability": self.availability,
            "response_time": self.response_time,
            "error_rate": self.error_rate,
            "cost_per_hour": self.cost_per_hour,
        }