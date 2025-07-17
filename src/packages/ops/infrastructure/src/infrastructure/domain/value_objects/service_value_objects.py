"""Value objects for services."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ServiceStatus(Enum):
    """Service status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class ServiceType(Enum):
    """Service type enumeration."""
    WEB_SERVICE = "web_service"
    API_SERVICE = "api_service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    STORAGE = "storage"
    MONITORING = "monitoring"
    LOGGING = "logging"
    SECURITY = "security"


@dataclass(frozen=True)
class ServiceConfiguration:
    """Configuration for services."""
    image: str
    ports: List[int] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    replicas: int = 1
    health_check: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "image": self.image,
            "ports": self.ports,
            "environment_variables": self.environment_variables,
            "volumes": self.volumes,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "replicas": self.replicas,
            "health_check": self.health_check,
        }


@dataclass(frozen=True)
class ServiceMetrics:
    """Metrics for services."""
    requests_per_second: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    uptime: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "requests_per_second": self.requests_per_second,
            "response_time": self.response_time,
            "error_rate": self.error_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "active_connections": self.active_connections,
            "uptime": self.uptime,
        }