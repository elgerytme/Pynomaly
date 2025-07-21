"""
Kubernetes Resource domain entities for cloud-native infrastructure management.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ResourceType(str, Enum):
    """Kubernetes resource types."""
    DEPLOYMENT = "deployment"
    STATEFULSET = "statefulset"
    DAEMONSET = "daemonset"
    SERVICE = "service"
    INGRESS = "ingress"
    CONFIGMAP = "configmap"
    SECRET = "secret"
    PVC = "persistentvolumeclaim"
    PV = "persistentvolume"
    NAMESPACE = "namespace"
    POD = "pod"
    JOB = "job"
    CRONJOB = "cronjob"
    HPA = "horizontalpodautoscaler"
    VPA = "verticalpodautoscaler"
    NETWORK_POLICY = "networkpolicy"
    SERVICE_MONITOR = "servicemonitor"
    CUSTOM = "custom"


class ResourceStatus(str, Enum):
    """Kubernetes resource status."""
    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    UPDATING = "updating"
    DELETING = "deleting"
    FAILED = "failed"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"


class OperatorState(str, Enum):
    """Kubernetes operator state."""
    RECONCILING = "reconciling"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    UNKNOWN = "unknown"


class KubernetesResource(BaseModel):
    """
    Kubernetes resource representation and management.
    
    Represents any Kubernetes resource with metadata, specifications,
    status tracking, and lifecycle management capabilities.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Resource identifier")
    
    # Kubernetes metadata
    name: str = Field(..., description="Resource name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    resource_type: ResourceType = Field(..., description="Type of Kubernetes resource")
    api_version: str = Field(..., description="API version")
    kind: str = Field(..., description="Kubernetes kind")
    
    # Ownership and management
    tenant_id: UUID = Field(..., description="Owning tenant")
    managed_by_operator: bool = Field(default=False, description="Managed by operator")
    operator_name: Optional[str] = Field(None, description="Managing operator name")
    
    # Resource specification
    spec: Dict[str, Any] = Field(..., description="Resource specification")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Resource metadata")
    
    # Status and conditions
    status: ResourceStatus = Field(default=ResourceStatus.PENDING)
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Resource conditions")
    phase: Optional[str] = Field(None, description="Resource phase")
    
    # Resource state
    desired_replicas: Optional[int] = Field(None, description="Desired replica count")
    current_replicas: Optional[int] = Field(None, description="Current replica count")
    ready_replicas: Optional[int] = Field(None, description="Ready replica count")
    
    # Resource allocation
    cpu_request: Optional[str] = Field(None, description="CPU request")
    memory_request: Optional[str] = Field(None, description="Memory request")
    cpu_limit: Optional[str] = Field(None, description="CPU limit")
    memory_limit: Optional[str] = Field(None, description="Memory limit")
    
    # Labels and selectors
    labels: Dict[str, str] = Field(default_factory=dict, description="Resource labels")
    selectors: Dict[str, str] = Field(default_factory=dict, description="Resource selectors")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Resource annotations")
    
    # Lifecycle management
    deletion_timestamp: Optional[datetime] = Field(None, description="Deletion timestamp")
    finalizers: List[str] = Field(default_factory=list, description="Resource finalizers")
    owner_references: List[Dict[str, Any]] = Field(default_factory=list, description="Owner references")
    
    # Health and monitoring
    health_check_path: Optional[str] = Field(None, description="Health check endpoint")
    metrics_enabled: bool = Field(default=True, description="Metrics collection enabled")
    last_health_check: Optional[datetime] = Field(None, description="Last health check")
    health_status: Optional[str] = Field(None, description="Health status")
    
    # Deployment configuration
    rolling_update_strategy: Optional[Dict[str, Any]] = Field(None, description="Rolling update strategy")
    restart_policy: str = Field(default="Always", description="Restart policy")
    image_pull_policy: str = Field(default="IfNotPresent", description="Image pull policy")
    
    # Networking
    ports: List[Dict[str, Any]] = Field(default_factory=list, description="Exposed ports")
    service_type: Optional[str] = Field(None, description="Service type")
    ingress_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Ingress rules")
    
    # Storage
    volumes: List[Dict[str, Any]] = Field(default_factory=list, description="Volume definitions")
    volume_mounts: List[Dict[str, Any]] = Field(default_factory=list, description="Volume mounts")
    storage_class: Optional[str] = Field(None, description="Storage class")
    
    # Security
    security_context: Optional[Dict[str, Any]] = Field(None, description="Security context")
    service_account: Optional[str] = Field(None, description="Service account")
    rbac_rules: List[Dict[str, Any]] = Field(default_factory=list, description="RBAC rules")
    
    # Auto-scaling
    hpa_enabled: bool = Field(default=False, description="HPA enabled")
    min_replicas: Optional[int] = Field(None, description="Minimum replicas")
    max_replicas: Optional[int] = Field(None, description="Maximum replicas")
    target_cpu_utilization: Optional[int] = Field(None, description="Target CPU utilization")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_applied_at: Optional[datetime] = Field(None, description="Last applied to cluster")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('name')
    def validate_kubernetes_name(cls, v):
        """Validate Kubernetes resource name."""
        import re
        if not re.match(r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$', v):
            raise ValueError('Invalid Kubernetes resource name format')
        return v
    
    @validator('namespace')
    def validate_kubernetes_namespace(cls, v):
        """Validate Kubernetes namespace name."""
        import re
        if not re.match(r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$', v):
            raise ValueError('Invalid Kubernetes namespace format')
        return v
    
    def is_ready(self) -> bool:
        """Check if resource is ready."""
        if self.resource_type in [ResourceType.DEPLOYMENT, ResourceType.STATEFULSET]:
            return (
                self.status == ResourceStatus.RUNNING and
                self.desired_replicas is not None and
                self.ready_replicas is not None and
                self.ready_replicas >= self.desired_replicas
            )
        elif self.resource_type == ResourceType.POD:
            return self.phase == "Running"
        elif self.resource_type == ResourceType.SERVICE:
            return self.status == ResourceStatus.RUNNING
        else:
            return self.status == ResourceStatus.RUNNING
    
    def is_healthy(self) -> bool:
        """Check if resource is healthy."""
        if not self.is_ready():
            return False
        
        # Check health conditions
        for condition in self.conditions:
            if condition.get("type") == "Ready" and condition.get("status") != "True":
                return False
            if condition.get("type") == "Available" and condition.get("status") != "True":
                return False
        
        return self.health_status in [None, "healthy"]
    
    def get_full_name(self) -> str:
        """Get full resource name with namespace."""
        return f"{self.namespace}/{self.name}"
    
    def get_api_path(self) -> str:
        """Get Kubernetes API path for this resource."""
        api_group = self.api_version.split("/")[0] if "/" in self.api_version else ""
        version = self.api_version.split("/")[1] if "/" in self.api_version else self.api_version
        
        if api_group:
            return f"/apis/{api_group}/{version}/namespaces/{self.namespace}/{self.resource_type.value}s/{self.name}"
        else:
            return f"/api/{version}/namespaces/{self.namespace}/{self.resource_type.value}s/{self.name}"
    
    def add_label(self, key: str, value: str) -> None:
        """Add a label to the resource."""
        self.labels[key] = value
        self.updated_at = datetime.utcnow()
    
    def add_annotation(self, key: str, value: str) -> None:
        """Add an annotation to the resource."""
        self.annotations[key] = value
        self.updated_at = datetime.utcnow()
    
    def add_condition(self, condition_type: str, status: str, reason: str, message: str) -> None:
        """Add a condition to the resource."""
        condition = {
            "type": condition_type,
            "status": status,
            "reason": reason,
            "message": message,
            "lastTransitionTime": datetime.utcnow().isoformat(),
            "lastUpdateTime": datetime.utcnow().isoformat()
        }
        
        # Remove existing condition of same type
        self.conditions = [c for c in self.conditions if c["type"] != condition_type]
        self.conditions.append(condition)
        self.updated_at = datetime.utcnow()
    
    def update_status(self, status: ResourceStatus, phase: Optional[str] = None) -> None:
        """Update resource status."""
        self.status = status
        if phase:
            self.phase = phase
        self.updated_at = datetime.utcnow()
    
    def update_replica_counts(self, desired: Optional[int] = None, current: Optional[int] = None, ready: Optional[int] = None) -> None:
        """Update replica counts."""
        if desired is not None:
            self.desired_replicas = desired
        if current is not None:
            self.current_replicas = current
        if ready is not None:
            self.ready_replicas = ready
        self.updated_at = datetime.utcnow()
    
    def schedule_deletion(self, grace_period_seconds: int = 30) -> None:
        """Schedule resource for deletion."""
        self.deletion_timestamp = datetime.utcnow() + timedelta(seconds=grace_period_seconds)
        self.status = ResourceStatus.DELETING
        self.updated_at = datetime.utcnow()
    
    def to_kubernetes_manifest(self) -> Dict[str, Any]:
        """Convert to Kubernetes manifest format."""
        manifest = {
            "apiVersion": self.api_version,
            "kind": self.kind,
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": dict(self.labels),
                "annotations": dict(self.annotations),
                **self.metadata
            },
            "spec": dict(self.spec)
        }
        
        # Add finalizers if present
        if self.finalizers:
            manifest["metadata"]["finalizers"] = self.finalizers
        
        # Add owner references if present
        if self.owner_references:
            manifest["metadata"]["ownerReferences"] = self.owner_references
        
        return manifest
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource summary information."""
        return {
            "id": str(self.id),
            "name": self.name,
            "namespace": self.namespace,
            "type": self.resource_type,
            "kind": self.kind,
            "status": self.status,
            "phase": self.phase,
            "ready": self.is_ready(),
            "healthy": self.is_healthy(),
            "replicas": {
                "desired": self.desired_replicas,
                "current": self.current_replicas,
                "ready": self.ready_replicas
            } if self.desired_replicas is not None else None,
            "resources": {
                "cpu_request": self.cpu_request,
                "memory_request": self.memory_request,
                "cpu_limit": self.cpu_limit,
                "memory_limit": self.memory_limit
            },
            "labels": dict(self.labels),
            "created_at": self.created_at.isoformat(),
            "managed_by_operator": self.managed_by_operator,
            "operator_name": self.operator_name
        }


class OperatorResource(BaseModel):
    """
    Custom Resource Definition for Kubernetes operators.
    
    Represents custom resources managed by Kubernetes operators
    with custom logic and reconciliation loops.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Operator resource identifier")
    
    # Custom Resource Definition
    crd_name: str = Field(..., description="CRD name")
    crd_version: str = Field(..., description="CRD version")
    crd_group: str = Field(..., description="CRD group")
    crd_kind: str = Field(..., description="CRD kind")
    
    # Resource identification
    name: str = Field(..., description="Resource name")
    namespace: str = Field(default="default", description="Namespace")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Operator configuration
    operator_name: str = Field(..., description="Managing operator")
    reconcile_interval_seconds: int = Field(default=300, description="Reconcile interval")
    
    # Custom resource spec and status
    spec: Dict[str, Any] = Field(..., description="Custom resource specification")
    status: Dict[str, Any] = Field(default_factory=dict, description="Custom resource status")
    
    # Operator state
    operator_state: OperatorState = Field(default=OperatorState.RECONCILING)
    last_reconciled_at: Optional[datetime] = Field(None, description="Last reconciliation time")
    reconcile_generation: int = Field(default=0, description="Reconcile generation")
    
    # Managed resources
    managed_resources: List[UUID] = Field(default_factory=list, description="Managed resource IDs")
    dependencies: List[str] = Field(default_factory=list, description="Resource dependencies")
    
    # Error handling
    error_count: int = Field(default=0, description="Error count")
    last_error: Optional[str] = Field(None, description="Last error message")
    retry_backoff_seconds: int = Field(default=60, description="Retry backoff")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    finalizers: List[str] = Field(default_factory=list)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deletion_timestamp: Optional[datetime] = Field(None)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def get_crd_name(self) -> str:
        """Get full CRD name."""
        return f"{self.crd_name}.{self.crd_group}"
    
    def get_api_version(self) -> str:
        """Get API version."""
        return f"{self.crd_group}/{self.crd_version}"
    
    def is_ready(self) -> bool:
        """Check if operator resource is ready."""
        return self.operator_state == OperatorState.READY
    
    def needs_reconciliation(self) -> bool:
        """Check if resource needs reconciliation."""
        if self.operator_state == OperatorState.ERROR:
            return True
        
        if not self.last_reconciled_at:
            return True
        
        time_since_reconcile = datetime.utcnow() - self.last_reconciled_at
        return time_since_reconcile >= timedelta(seconds=self.reconcile_interval_seconds)
    
    def record_reconciliation_success(self) -> None:
        """Record successful reconciliation."""
        self.operator_state = OperatorState.READY
        self.last_reconciled_at = datetime.utcnow()
        self.reconcile_generation += 1
        self.error_count = 0
        self.last_error = None
        self.updated_at = datetime.utcnow()
    
    def record_reconciliation_error(self, error_message: str) -> None:
        """Record reconciliation error."""
        self.operator_state = OperatorState.ERROR
        self.error_count += 1
        self.last_error = error_message
        self.updated_at = datetime.utcnow()
    
    def add_managed_resource(self, resource_id: UUID) -> None:
        """Add managed resource."""
        if resource_id not in self.managed_resources:
            self.managed_resources.append(resource_id)
            self.updated_at = datetime.utcnow()
    
    def remove_managed_resource(self, resource_id: UUID) -> None:
        """Remove managed resource."""
        if resource_id in self.managed_resources:
            self.managed_resources.remove(resource_id)
            self.updated_at = datetime.utcnow()
    
    def update_status(self, status_update: Dict[str, Any]) -> None:
        """Update custom resource status."""
        self.status.update(status_update)
        self.updated_at = datetime.utcnow()
    
    def to_kubernetes_manifest(self) -> Dict[str, Any]:
        """Convert to Kubernetes custom resource manifest."""
        return {
            "apiVersion": self.get_api_version(),
            "kind": self.crd_kind,
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": dict(self.labels),
                "annotations": dict(self.annotations),
                "finalizers": self.finalizers
            },
            "spec": dict(self.spec),
            "status": dict(self.status)
        }
    
    def get_operator_summary(self) -> Dict[str, Any]:
        """Get operator resource summary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "namespace": self.namespace,
            "crd": self.get_crd_name(),
            "operator": self.operator_name,
            "state": self.operator_state,
            "generation": self.reconcile_generation,
            "managed_resources": len(self.managed_resources),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_reconciled": self.last_reconciled_at.isoformat() if self.last_reconciled_at else None,
            "needs_reconciliation": self.needs_reconciliation()
        }