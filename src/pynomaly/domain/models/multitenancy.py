"""Multi-tenancy domain models for enterprise tenant isolation and management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import numpy as np


class TenantStatus(Enum):
    """Tenant status states."""
    
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING_ACTIVATION = "pending_activation"
    DEACTIVATED = "deactivated"
    ARCHIVED = "archived"


class TenantTier(Enum):
    """Tenant service tiers for resource allocation."""
    
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class IsolationLevel(Enum):
    """Data and resource isolation levels."""
    
    SHARED = "shared"  # Shared infrastructure, logical isolation
    DEDICATED = "dedicated"  # Dedicated resources within shared infrastructure
    ISOLATED = "isolated"  # Completely isolated infrastructure
    HYBRID = "hybrid"  # Mix of shared and dedicated resources


class ResourceType(Enum):
    """Types of resources that can be allocated to tenants."""
    
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    GPU = "gpu"
    NETWORK_BANDWIDTH = "network_bandwidth"
    API_REQUESTS = "api_requests"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    CONCURRENT_USERS = "concurrent_users"


@dataclass
class ResourceQuota:
    """Resource quota configuration for tenants."""
    
    resource_type: ResourceType
    allocated_amount: float
    used_amount: float = 0.0
    
    # Quota configuration
    soft_limit: Optional[float] = None  # Warning threshold
    hard_limit: Optional[float] = None  # Hard enforcement
    burst_limit: Optional[float] = None  # Temporary burst allowance
    
    # Time-based quotas
    time_window: Optional[timedelta] = None  # For rate limiting
    reset_schedule: Optional[str] = None  # "daily", "weekly", "monthly"
    
    # Monitoring
    last_reset: datetime = field(default_factory=datetime.utcnow)
    peak_usage: float = 0.0
    average_usage: float = 0.0
    
    # Units and metadata
    unit: str = ""
    description: str = ""
    
    def __post_init__(self):
        if self.allocated_amount < 0:
            raise ValueError("Allocated amount must be non-negative")
        if self.used_amount < 0:
            raise ValueError("Used amount must be non-negative")
        if self.soft_limit and self.soft_limit > self.allocated_amount:
            raise ValueError("Soft limit cannot exceed allocated amount")
    
    def consume(self, amount: float) -> bool:
        """Consume resources, returns True if successful."""
        if self.hard_limit and (self.used_amount + amount) > self.hard_limit:
            return False
        
        if (self.used_amount + amount) > self.allocated_amount:
            # Check if burst is allowed
            if self.burst_limit and (self.used_amount + amount) <= self.burst_limit:
                self.used_amount += amount
                self.peak_usage = max(self.peak_usage, self.used_amount)
                return True
            return False
        
        self.used_amount += amount
        self.peak_usage = max(self.peak_usage, self.used_amount)
        return True
    
    def release(self, amount: float) -> None:
        """Release consumed resources."""
        self.used_amount = max(0.0, self.used_amount - amount)
    
    def get_utilization_percentage(self) -> float:
        """Get current utilization as percentage."""
        if self.allocated_amount == 0:
            return 0.0
        return (self.used_amount / self.allocated_amount) * 100
    
    def is_soft_limit_exceeded(self) -> bool:
        """Check if soft limit is exceeded."""
        return self.soft_limit is not None and self.used_amount > self.soft_limit
    
    def is_hard_limit_exceeded(self) -> bool:
        """Check if hard limit is exceeded."""
        return self.hard_limit is not None and self.used_amount > self.hard_limit
    
    def reset_usage(self) -> None:
        """Reset usage counters (for time-based quotas)."""
        self.used_amount = 0.0
        self.last_reset = datetime.utcnow()


@dataclass
class TenantConfiguration:
    """Tenant-specific configuration and customization."""
    
    # Feature flags
    enabled_features: Set[str] = field(default_factory=set)
    disabled_features: Set[str] = field(default_factory=set)
    
    # API configuration
    api_version: str = "v1"
    rate_limiting: Dict[str, int] = field(default_factory=dict)  # endpoint -> requests per minute
    webhook_endpoints: List[str] = field(default_factory=list)
    
    # Data configuration
    data_retention_days: int = 365
    backup_enabled: bool = True
    backup_schedule: str = "daily"
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    
    # ML configuration
    max_model_size_mb: int = 1000
    max_concurrent_trainings: int = 5
    max_inference_requests_per_minute: int = 1000
    allowed_algorithms: Set[str] = field(default_factory=set)
    
    # UI/UX configuration
    custom_branding: Dict[str, str] = field(default_factory=dict)
    custom_themes: List[str] = field(default_factory=list)
    custom_dashboards: List[str] = field(default_factory=list)
    
    # Integration configuration
    allowed_integrations: Set[str] = field(default_factory=set)
    sso_configuration: Dict[str, Any] = field(default_factory=dict)
    ldap_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance configuration
    compliance_frameworks: Set[str] = field(default_factory=set)
    audit_level: str = "standard"  # "basic", "standard", "detailed"
    data_residency_requirements: List[str] = field(default_factory=list)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled for this tenant."""
        if feature in self.disabled_features:
            return False
        return feature in self.enabled_features
    
    def enable_feature(self, feature: str) -> None:
        """Enable a feature for this tenant."""
        self.enabled_features.add(feature)
        self.disabled_features.discard(feature)
    
    def disable_feature(self, feature: str) -> None:
        """Disable a feature for this tenant."""
        self.disabled_features.add(feature)
        self.enabled_features.discard(feature)


@dataclass
class TenantMetrics:
    """Tenant usage metrics and analytics."""
    
    tenant_id: UUID
    
    # Usage metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    
    # Resource usage
    cpu_usage_hours: float = 0.0
    memory_usage_gb_hours: float = 0.0
    storage_usage_gb: float = 0.0
    network_bandwidth_gb: float = 0.0
    
    # ML metrics
    models_trained: int = 0
    predictions_made: int = 0
    anomalies_detected: int = 0
    accuracy_scores: List[float] = field(default_factory=list)
    
    # User metrics
    active_users: int = 0
    total_user_sessions: int = 0
    average_session_duration: float = 0.0
    
    # Time period
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    
    # Calculated metrics
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_success_rate(self) -> float:
        """Calculate request success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def calculate_error_rate(self) -> float:
        """Calculate request error rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    def calculate_average_accuracy(self) -> float:
        """Calculate average model accuracy."""
        if not self.accuracy_scores:
            return 0.0
        return sum(self.accuracy_scores) / len(self.accuracy_scores)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "tenant_id": str(self.tenant_id),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
                "duration_hours": (self.period_end - self.period_start).total_seconds() / 3600,
            },
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": self.calculate_success_rate(),
                "error_rate": self.calculate_error_rate(),
                "average_response_time": self.average_response_time,
            },
            "resources": {
                "cpu_usage_hours": self.cpu_usage_hours,
                "memory_usage_gb_hours": self.memory_usage_gb_hours,
                "storage_usage_gb": self.storage_usage_gb,
                "network_bandwidth_gb": self.network_bandwidth_gb,
            },
            "ml_analytics": {
                "models_trained": self.models_trained,
                "predictions_made": self.predictions_made,
                "anomalies_detected": self.anomalies_detected,
                "average_accuracy": self.calculate_average_accuracy(),
            },
            "user_activity": {
                "active_users": self.active_users,
                "total_sessions": self.total_user_sessions,
                "average_session_duration": self.average_session_duration,
            },
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class Tenant:
    """Core tenant entity representing an isolated customer environment."""
    
    tenant_id: UUID
    name: str
    display_name: str
    description: str = ""
    
    # Tenant classification
    tier: TenantTier = TenantTier.STANDARD
    isolation_level: IsolationLevel = IsolationLevel.SHARED
    status: TenantStatus = TenantStatus.PENDING_ACTIVATION
    
    # Contact and billing information
    primary_contact_email: str = ""
    billing_email: str = ""
    organization_name: str = ""
    domain: str = ""
    
    # Resource management
    resource_quotas: Dict[ResourceType, ResourceQuota] = field(default_factory=dict)
    configuration: TenantConfiguration = field(default_factory=TenantConfiguration)
    
    # Infrastructure mapping
    namespace: str = ""  # Kubernetes namespace
    database_schema: str = ""  # Database schema/database name
    storage_bucket: str = ""  # Cloud storage bucket
    
    # Security and access
    allowed_ip_ranges: List[str] = field(default_factory=list)
    security_policies: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle management
    created_at: datetime = field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = None
    suspended_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Parent/child relationships for hierarchical tenancy
    parent_tenant_id: Optional[UUID] = None
    child_tenant_ids: Set[UUID] = field(default_factory=set)
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Tenant name cannot be empty")
        if not self.display_name:
            self.display_name = self.name
        if not self.namespace:
            self.namespace = f"tenant-{self.name.lower().replace('_', '-')}"
        if not self.database_schema:
            self.database_schema = f"tenant_{self.name.lower()}"
        if not self.storage_bucket:
            self.storage_bucket = f"pynomaly-tenant-{self.name.lower()}"
    
    def activate(self) -> None:
        """Activate the tenant."""
        if self.status == TenantStatus.PENDING_ACTIVATION:
            self.status = TenantStatus.ACTIVE
            self.activated_at = datetime.utcnow()
            self.last_activity_at = datetime.utcnow()
    
    def suspend(self, reason: str = "") -> None:
        """Suspend the tenant."""
        if self.status == TenantStatus.ACTIVE:
            self.status = TenantStatus.SUSPENDED
            self.suspended_at = datetime.utcnow()
            if reason:
                self.custom_metadata["suspension_reason"] = reason
    
    def reactivate(self) -> None:
        """Reactivate a suspended tenant."""
        if self.status == TenantStatus.SUSPENDED:
            self.status = TenantStatus.ACTIVE
            self.suspended_at = None
            self.last_activity_at = datetime.utcnow()
            self.custom_metadata.pop("suspension_reason", None)
    
    def deactivate(self) -> None:
        """Deactivate the tenant."""
        self.status = TenantStatus.DEACTIVATED
        self.last_activity_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the tenant (soft delete)."""
        self.status = TenantStatus.ARCHIVED
    
    def add_resource_quota(self, quota: ResourceQuota) -> None:
        """Add or update resource quota."""
        self.resource_quotas[quota.resource_type] = quota
    
    def get_resource_quota(self, resource_type: ResourceType) -> Optional[ResourceQuota]:
        """Get resource quota for specific type."""
        return self.resource_quotas.get(resource_type)
    
    def consume_resource(self, resource_type: ResourceType, amount: float) -> bool:
        """Consume resources, returns True if successful."""
        quota = self.resource_quotas.get(resource_type)
        if not quota:
            return False
        return quota.consume(amount)
    
    def release_resource(self, resource_type: ResourceType, amount: float) -> None:
        """Release consumed resources."""
        quota = self.resource_quotas.get(resource_type)
        if quota:
            quota.release(amount)
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get resource utilization percentages."""
        return {
            resource_type.value: quota.get_utilization_percentage()
            for resource_type, quota in self.resource_quotas.items()
        }
    
    def is_over_quota(self, resource_type: Optional[ResourceType] = None) -> bool:
        """Check if tenant is over quota."""
        if resource_type:
            quota = self.resource_quotas.get(resource_type)
            return quota.is_hard_limit_exceeded() if quota else False
        
        return any(quota.is_hard_limit_exceeded() for quota in self.resource_quotas.values())
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE
    
    def add_child_tenant(self, child_tenant_id: UUID) -> None:
        """Add child tenant (for hierarchical tenancy)."""
        self.child_tenant_ids.add(child_tenant_id)
    
    def remove_child_tenant(self, child_tenant_id: UUID) -> None:
        """Remove child tenant."""
        self.child_tenant_ids.discard(child_tenant_id)
    
    def get_tenant_summary(self) -> Dict[str, Any]:
        """Get comprehensive tenant summary."""
        return {
            "tenant_id": str(self.tenant_id),
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "tier": self.tier.value,
            "isolation_level": self.isolation_level.value,
            "status": self.status.value,
            "organization": {
                "name": self.organization_name,
                "domain": self.domain,
                "primary_contact": self.primary_contact_email,
                "billing_email": self.billing_email,
            },
            "infrastructure": {
                "namespace": self.namespace,
                "database_schema": self.database_schema,
                "storage_bucket": self.storage_bucket,
            },
            "resources": {
                "quotas": {
                    rt.value: {
                        "allocated": quota.allocated_amount,
                        "used": quota.used_amount,
                        "utilization": quota.get_utilization_percentage(),
                        "unit": quota.unit,
                    }
                    for rt, quota in self.resource_quotas.items()
                },
                "over_quota": self.is_over_quota(),
            },
            "security": {
                "allowed_ip_ranges": self.allowed_ip_ranges,
                "security_policies": list(self.security_policies.keys()),
            },
            "lifecycle": {
                "created_at": self.created_at.isoformat(),
                "activated_at": self.activated_at.isoformat() if self.activated_at else None,
                "suspended_at": self.suspended_at.isoformat() if self.suspended_at else None,
                "last_activity_at": self.last_activity_at.isoformat() if self.last_activity_at else None,
            },
            "features": {
                "enabled": list(self.configuration.enabled_features),
                "disabled": list(self.configuration.disabled_features),
            },
            "hierarchy": {
                "parent_tenant_id": str(self.parent_tenant_id) if self.parent_tenant_id else None,
                "child_tenant_ids": [str(cid) for cid in self.child_tenant_ids],
            },
            "tags": self.tags,
        }


@dataclass
class TenantContext:
    """Context information for current tenant in request processing."""
    
    tenant: Tenant
    user_id: Optional[UUID] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Request metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    api_version: str = "v1"
    
    # Permissions and roles
    user_roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    
    # Resource tracking
    consumed_resources: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Request timing
    request_start_time: datetime = field(default_factory=datetime.utcnow)
    
    def consume_resource(self, resource_type: ResourceType, amount: float) -> bool:
        """Consume tenant resource and track in context."""
        success = self.tenant.consume_resource(resource_type, amount)
        if success:
            self.consumed_resources[resource_type] = self.consumed_resources.get(resource_type, 0.0) + amount
        return success
    
    def release_consumed_resources(self) -> None:
        """Release all resources consumed in this context."""
        for resource_type, amount in self.consumed_resources.items():
            self.tenant.release_resource(resource_type, amount)
        self.consumed_resources.clear()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.user_roles
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if feature is enabled for tenant."""
        return self.tenant.configuration.is_feature_enabled(feature)
    
    def get_request_duration(self) -> timedelta:
        """Get current request duration."""
        return datetime.utcnow() - self.request_start_time


@dataclass
class TenantEvent:
    """Events related to tenant lifecycle and activities."""
    
    event_id: UUID
    tenant_id: UUID
    event_type: str
    event_data: Dict[str, Any]
    
    # Event metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "system"
    user_id: Optional[UUID] = None
    
    # Event categorization
    category: str = "general"  # "lifecycle", "resource", "security", "billing"
    severity: str = "info"  # "debug", "info", "warning", "error", "critical"
    
    # Event processing
    processed: bool = False
    processed_at: Optional[datetime] = None
    processing_result: Optional[str] = None
    
    def mark_processed(self, result: str = "success") -> None:
        """Mark event as processed."""
        self.processed = True
        self.processed_at = datetime.utcnow()
        self.processing_result = result
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get event summary."""
        return {
            "event_id": str(self.event_id),
            "tenant_id": str(self.tenant_id),
            "event_type": self.event_type,
            "category": self.category,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "user_id": str(self.user_id) if self.user_id else None,
            "processed": self.processed,
            "data_keys": list(self.event_data.keys()),
        }