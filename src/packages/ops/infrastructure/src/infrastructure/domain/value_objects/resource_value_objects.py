"""Value objects for resources."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ResourceStatus(Enum):
    """Resource status enumeration."""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    TERMINATED = "terminated"


class ResourceType(Enum):
    """Resource type enumeration."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CONTAINER = "container"
    VIRTUAL_MACHINE = "virtual_machine"
    KUBERNETES_CLUSTER = "kubernetes_cluster"


@dataclass(frozen=True)
class ResourceCapacity:
    """Capacity specification for resources."""
    cpu_cores: int = 0
    memory_gb: int = 0
    storage_gb: int = 0
    network_bandwidth_mbps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capacity to dictionary."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "storage_gb": self.storage_gb,
            "network_bandwidth_mbps": self.network_bandwidth_mbps,
        }


@dataclass(frozen=True)
class ResourceAllocation:
    """Allocation information for resources."""
    allocated_cpu: int = 0
    allocated_memory: int = 0
    allocated_storage: int = 0
    allocated_to: Optional[str] = None
    allocation_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert allocation to dictionary."""
        return {
            "allocated_cpu": self.allocated_cpu,
            "allocated_memory": self.allocated_memory,
            "allocated_storage": self.allocated_storage,
            "allocated_to": self.allocated_to,
            "allocation_time": self.allocation_time,
        }


@dataclass(frozen=True)
class ResourceMetrics:
    """Metrics for resources."""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    storage_utilization: float = 0.0
    network_utilization: float = 0.0
    availability: float = 0.0
    latency: float = 0.0
    throughput: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "storage_utilization": self.storage_utilization,
            "network_utilization": self.network_utilization,
            "availability": self.availability,
            "latency": self.latency,
            "throughput": self.throughput,
        }