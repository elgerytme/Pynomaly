"""
Compute Cluster domain entities for distributed computing management.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ClusterType(str, Enum):
    """Types of compute clusters."""
    DASK = "dask"
    RAY = "ray"
    KUBERNETES = "kubernetes"
    SPARK = "spark"
    CUSTOM = "custom"


class ClusterStatus(str, Enum):
    """Cluster status enumeration."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    SCALING = "scaling"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class NodeStatus(str, Enum):
    """Compute node status."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DRAINING = "draining"
    UNREACHABLE = "unreachable"


class ResourceType(str, Enum):
    """Types of compute resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


class ScalingPolicy(str, Enum):
    """Cluster scaling policies."""
    MANUAL = "manual"
    AUTO_CPU = "auto_cpu"
    AUTO_MEMORY = "auto_memory"
    AUTO_WORKLOAD = "auto_workload"
    PREDICTIVE = "predictive"
    CUSTOM = "custom"


class ComputeNode(BaseModel):
    """
    Individual compute node in a distributed cluster.
    
    Represents a single worker node with its resources,
    status, and performance metrics.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Node identifier")
    
    # Node identification
    name: str = Field(..., description="Node name")
    cluster_id: UUID = Field(..., description="Parent cluster ID")
    node_type: str = Field(..., description="Node type/size")
    
    # Infrastructure details
    hostname: str = Field(..., description="Node hostname")
    ip_address: str = Field(..., description="Node IP address")
    port: int = Field(default=8787, description="Node communication port")
    
    # Resource specifications
    cpu_cores: int = Field(..., ge=1, description="Number of CPU cores")
    memory_gb: float = Field(..., ge=0.1, description="Memory in GB")
    gpu_count: int = Field(default=0, ge=0, description="Number of GPUs")
    gpu_type: Optional[str] = Field(None, description="GPU type")
    storage_gb: float = Field(default=100.0, description="Storage in GB")
    
    # Current resource utilization
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_usage_gb: float = Field(default=0.0, ge=0.0)
    gpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    storage_usage_gb: float = Field(default=0.0, ge=0.0)
    network_io_mbps: float = Field(default=0.0, ge=0.0)
    
    # Node status
    status: NodeStatus = Field(default=NodeStatus.PENDING)
    health_score: float = Field(default=100.0, ge=0.0, le=100.0)
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat time")
    
    # Performance metrics
    tasks_running: int = Field(default=0, ge=0, description="Currently running tasks")
    tasks_completed: int = Field(default=0, ge=0, description="Total completed tasks")
    tasks_failed: int = Field(default=0, ge=0, description="Total failed tasks")
    uptime_seconds: float = Field(default=0.0, ge=0.0, description="Node uptime")
    
    # Configuration
    labels: Dict[str, str] = Field(default_factory=dict, description="Node labels")
    taints: List[Dict[str, str]] = Field(default_factory=list, description="Node taints")
    drain_timeout_seconds: int = Field(default=300, description="Drain timeout")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="Node start time")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('memory_usage_gb')
    def validate_memory_usage(cls, v, values):
        """Validate memory usage doesn't exceed capacity."""
        if 'memory_gb' in values and v > values['memory_gb']:
            raise ValueError('Memory usage cannot exceed capacity')
        return v
    
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        if self.status not in [NodeStatus.RUNNING, NodeStatus.DRAINING]:
            return False
        
        if self.last_heartbeat:
            time_since_heartbeat = datetime.utcnow() - self.last_heartbeat
            if time_since_heartbeat > timedelta(minutes=5):
                return False
        
        return self.health_score >= 70.0
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization percentages."""
        return {
            "cpu": self.cpu_usage_percent,
            "memory": (self.memory_usage_gb / self.memory_gb) * 100.0 if self.memory_gb > 0 else 0.0,
            "gpu": self.gpu_usage_percent,
            "storage": (self.storage_usage_gb / self.storage_gb) * 100.0 if self.storage_gb > 0 else 0.0
        }
    
    def has_capacity_for_task(self, required_cpu: float, required_memory: float) -> bool:
        """Check if node has capacity for a task."""
        available_cpu = self.cpu_cores - (self.cpu_cores * self.cpu_usage_percent / 100.0)
        available_memory = self.memory_gb - self.memory_usage_gb
        
        return available_cpu >= required_cpu and available_memory >= required_memory
    
    def update_heartbeat(self) -> None:
        """Update node heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_metrics(
        self,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        gpu_usage: Optional[float] = None,
        storage_usage: Optional[float] = None
    ) -> None:
        """Update node performance metrics."""
        if cpu_usage is not None:
            self.cpu_usage_percent = max(0.0, min(100.0, cpu_usage))
        if memory_usage is not None:
            self.memory_usage_gb = max(0.0, min(self.memory_gb, memory_usage))
        if gpu_usage is not None:
            self.gpu_usage_percent = max(0.0, min(100.0, gpu_usage))
        if storage_usage is not None:
            self.storage_usage_gb = max(0.0, min(self.storage_gb, storage_usage))
        
        self.updated_at = datetime.utcnow()
    
    def drain(self, timeout_seconds: int = 300) -> None:
        """Start draining the node."""
        self.status = NodeStatus.DRAINING
        self.drain_timeout_seconds = timeout_seconds
        self.updated_at = datetime.utcnow()


class ComputeCluster(BaseModel):
    """
    Distributed compute cluster for scalable processing.
    
    Manages a collection of compute nodes with auto-scaling,
    load balancing, and fault tolerance capabilities.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Cluster identifier")
    
    # Cluster identification
    name: str = Field(..., description="Cluster name")
    description: str = Field(default="", description="Cluster description")
    cluster_type: ClusterType = Field(..., description="Type of cluster")
    version: str = Field(..., description="Cluster software version")
    
    # Ownership and access
    tenant_id: UUID = Field(..., description="Owning tenant")
    created_by: UUID = Field(..., description="User who created cluster")
    shared_users: List[UUID] = Field(default_factory=list, description="Users with access")
    
    # Configuration
    scheduler_address: str = Field(..., description="Scheduler/coordinator address")
    dashboard_port: Optional[int] = Field(None, description="Dashboard port")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Cluster config")
    
    # Resource limits
    max_nodes: int = Field(default=100, ge=1, description="Maximum number of nodes")
    min_nodes: int = Field(default=1, ge=0, description="Minimum number of nodes")
    max_cpu_cores: int = Field(default=1000, ge=1, description="Maximum total CPU cores")
    max_memory_gb: float = Field(default=1000.0, ge=0.1, description="Maximum total memory")
    
    # Current state
    status: ClusterStatus = Field(default=ClusterStatus.PENDING)
    current_nodes: int = Field(default=0, ge=0, description="Current node count")
    active_nodes: int = Field(default=0, ge=0, description="Active node count")
    
    # Resource utilization
    total_cpu_cores: int = Field(default=0, ge=0)
    total_memory_gb: float = Field(default=0.0, ge=0.0)
    used_cpu_cores: float = Field(default=0.0, ge=0.0)
    used_memory_gb: float = Field(default=0.0, ge=0.0)
    
    # Scaling configuration
    scaling_policy: ScalingPolicy = Field(default=ScalingPolicy.MANUAL)
    auto_scale_enabled: bool = Field(default=False)
    scale_up_threshold: float = Field(default=80.0, ge=0.0, le=100.0)
    scale_down_threshold: float = Field(default=20.0, ge=0.0, le=100.0)
    scale_up_cooldown_minutes: int = Field(default=5, ge=1)
    scale_down_cooldown_minutes: int = Field(default=10, ge=1)
    
    # Performance metrics
    tasks_submitted: int = Field(default=0, ge=0, description="Total tasks submitted")
    tasks_completed: int = Field(default=0, ge=0, description="Total tasks completed")
    tasks_failed: int = Field(default=0, ge=0, description="Total tasks failed")
    avg_task_duration_seconds: float = Field(default=0.0, ge=0.0)
    
    # Health and monitoring
    health_score: float = Field(default=100.0, ge=0.0, le=100.0)
    last_scaling_action: Optional[datetime] = Field(None, description="Last scaling action")
    error_message: Optional[str] = Field(None, description="Current error message")
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="Cluster start time")
    stopped_at: Optional[datetime] = Field(None, description="Cluster stop time")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('min_nodes')
    def min_nodes_not_greater_than_max(cls, v, values):
        """Ensure min_nodes <= max_nodes."""
        if 'max_nodes' in values and v > values['max_nodes']:
            raise ValueError('min_nodes cannot be greater than max_nodes')
        return v
    
    @validator('scale_down_threshold')
    def scale_thresholds_valid(cls, v, values):
        """Ensure scale down threshold < scale up threshold."""
        if 'scale_up_threshold' in values and v >= values['scale_up_threshold']:
            raise ValueError('scale_down_threshold must be less than scale_up_threshold')
        return v
    
    def is_running(self) -> bool:
        """Check if cluster is running."""
        return self.status == ClusterStatus.RUNNING and self.active_nodes > 0
    
    def is_healthy(self) -> bool:
        """Check if cluster is healthy."""
        if not self.is_running():
            return False
        return self.health_score >= 70.0 and self.active_nodes >= self.min_nodes
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get cluster resource utilization."""
        cpu_util = (self.used_cpu_cores / self.total_cpu_cores) * 100.0 if self.total_cpu_cores > 0 else 0.0
        memory_util = (self.used_memory_gb / self.total_memory_gb) * 100.0 if self.total_memory_gb > 0 else 0.0
        
        return {
            "cpu": cpu_util,
            "memory": memory_util,
            "nodes": (self.current_nodes / self.max_nodes) * 100.0
        }
    
    def should_scale_up(self) -> bool:
        """Check if cluster should scale up."""
        if not self.auto_scale_enabled or self.current_nodes >= self.max_nodes:
            return False
        
        utilization = self.get_resource_utilization()
        return (
            utilization["cpu"] > self.scale_up_threshold or
            utilization["memory"] > self.scale_up_threshold
        )
    
    def should_scale_down(self) -> bool:
        """Check if cluster should scale down."""
        if not self.auto_scale_enabled or self.current_nodes <= self.min_nodes:
            return False
        
        utilization = self.get_resource_utilization()
        return (
            utilization["cpu"] < self.scale_down_threshold and
            utilization["memory"] < self.scale_down_threshold
        )
    
    def can_scale(self, action: str) -> bool:
        """Check if scaling action is allowed based on cooldown."""
        if not self.last_scaling_action:
            return True
        
        time_since_scaling = datetime.utcnow() - self.last_scaling_action
        cooldown_minutes = (
            self.scale_up_cooldown_minutes if action == "up" 
            else self.scale_down_cooldown_minutes
        )
        
        return time_since_scaling >= timedelta(minutes=cooldown_minutes)
    
    def update_resource_totals(self, nodes: List[ComputeNode]) -> None:
        """Update total resource counts from node list."""
        self.current_nodes = len(nodes)
        self.active_nodes = sum(1 for node in nodes if node.status == NodeStatus.RUNNING)
        
        self.total_cpu_cores = sum(node.cpu_cores for node in nodes)
        self.total_memory_gb = sum(node.memory_gb for node in nodes)
        
        self.used_cpu_cores = sum(
            node.cpu_cores * (node.cpu_usage_percent / 100.0) for node in nodes
        )
        self.used_memory_gb = sum(node.memory_usage_gb for node in nodes)
        
        self.updated_at = datetime.utcnow()
    
    def record_scaling_action(self, action: str) -> None:
        """Record that a scaling action occurred."""
        self.last_scaling_action = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def start(self) -> None:
        """Mark cluster as started."""
        self.status = ClusterStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.error_message = None
    
    def stop(self) -> None:
        """Mark cluster as stopped."""
        self.status = ClusterStatus.STOPPED
        self.stopped_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def set_error(self, error_message: str) -> None:
        """Mark cluster as errored."""
        self.status = ClusterStatus.ERROR
        self.error_message = error_message
        self.updated_at = datetime.utcnow()
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get cluster summary information."""
        utilization = self.get_resource_utilization()
        
        return {
            "id": str(self.id),
            "name": self.name,
            "type": self.cluster_type,
            "status": self.status,
            "nodes": {
                "current": self.current_nodes,
                "active": self.active_nodes,
                "max": self.max_nodes,
                "min": self.min_nodes
            },
            "resources": {
                "cpu_cores": self.total_cpu_cores,
                "memory_gb": self.total_memory_gb,
                "utilization": utilization
            },
            "health_score": self.health_score,
            "auto_scaling": self.auto_scale_enabled,
            "tasks": {
                "completed": self.tasks_completed,
                "failed": self.tasks_failed,
                "success_rate": (
                    (self.tasks_completed / (self.tasks_completed + self.tasks_failed)) * 100.0
                    if (self.tasks_completed + self.tasks_failed) > 0 else 0.0
                )
            },
            "uptime_hours": (
                (datetime.utcnow() - self.started_at).total_seconds() / 3600.0
                if self.started_at else 0.0
            )
        }