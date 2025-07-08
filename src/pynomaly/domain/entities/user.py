"""
User management domain entities for multi-tenant anomaly detection platform.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set

from pynomaly.shared.types import UserId, TenantId, RoleId


class UserRole(str, Enum):
    """User roles within a tenant."""
    SUPER_ADMIN = "super_admin"  # Platform-wide admin
    TENANT_ADMIN = "tenant_admin"  # Tenant administrator
    DATA_SCIENTIST = "data_scientist"  # Can create/manage models
    ANALYST = "analyst"  # Can view results and run detection
    VIEWER = "viewer"  # Read-only access


class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class TenantStatus(str, Enum):
    """Tenant account status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    EXPIRED = "expired"


class TenantPlan(str, Enum):
    """Tenant subscription plans."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass(frozen=True)
class Permission:
    """Individual permission."""
    name: str
    resource: str
    action: str
    description: str = ""


@dataclass
class Role:
    """User role with permissions."""
    id: RoleId
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TenantLimits:
    """Resource limits for a tenant."""
    max_users: int = 10
    max_datasets: int = 100
    max_models: int = 50
    max_detections_per_month: int = 10000
    max_storage_gb: int = 10
    max_api_calls_per_minute: int = 100
    max_concurrent_detections: int = 5


@dataclass
class TenantUsage:
    """Current resource usage for a tenant."""
    users_count: int = 0
    datasets_count: int = 0
    models_count: int = 0
    detections_this_month: int = 0
    storage_used_gb: float = 0.0
    api_calls_this_minute: int = 0
    concurrent_detections: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Tenant:
    """Multi-tenant organization entity."""
    id: TenantId
    name: str
    domain: str
    plan: TenantPlan
    status: TenantStatus
    limits: TenantLimits
    usage: TenantUsage
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    contact_email: str = ""
    billing_email: str = ""
    settings: Dict[str, any] = field(default_factory=dict)

    def is_within_limits(self) -> Dict[str, bool]:
        """Check if tenant is within resource limits."""
        return {
            "users": self.usage.users_count <= self.limits.max_users,
            "datasets": self.usage.datasets_count <= self.limits.max_datasets,
            "models": self.usage.models_count <= self.limits.max_models,
            "detections": self.usage.detections_this_month <= self.limits.max_detections_per_month,
            "storage": self.usage.storage_used_gb <= self.limits.max_storage_gb,
            "api_calls": self.usage.api_calls_this_minute <= self.limits.max_api_calls_per_minute,
            "concurrent": self.usage.concurrent_detections <= self.limits.max_concurrent_detections,
        }

    def get_limit_usage_percentage(self, resource: str) -> float:
        """Get usage percentage for a specific resource."""
        usage_map = {
            "users": (self.usage.users_count, self.limits.max_users),
            "datasets": (self.usage.datasets_count, self.limits.max_datasets),
            "models": (self.usage.models_count, self.limits.max_models),
            "detections": (self.usage.detections_this_month, self.limits.max_detections_per_month),
            "storage": (self.usage.storage_used_gb, self.limits.max_storage_gb),
            "api_calls": (self.usage.api_calls_this_minute, self.limits.max_api_calls_per_minute),
            "concurrent": (self.usage.concurrent_detections, self.limits.max_concurrent_detections),
        }

        if resource in usage_map:
            usage, limit = usage_map[resource]
            return (usage / limit * 100) if limit > 0 else 0.0
        return 0.0


@dataclass
class UserTenantRole:
    """Association between user, tenant, and role."""
    user_id: UserId
    tenant_id: TenantId
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    granted_at: datetime = field(default_factory=datetime.utcnow)
    granted_by: Optional[UserId] = None
    expires_at: Optional[datetime] = None


@dataclass
class User:
    """User entity with multi-tenant support."""
    id: UserId
    email: str
    username: str
    first_name: str
    last_name: str
    status: UserStatus
    tenant_roles: List[UserTenantRole] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None
    email_verified_at: Optional[datetime] = None
    password_hash: str = ""
    settings: Dict[str, any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}".strip()

    def get_tenant_role(self, tenant_id: TenantId) -> Optional[UserTenantRole]:
        """Get user's role for a specific tenant."""
        for tenant_role in self.tenant_roles:
            if tenant_role.tenant_id == tenant_id:
                return tenant_role
        return None

    def has_role_in_tenant(self, tenant_id: TenantId, role: UserRole) -> bool:
        """Check if user has specific role in tenant."""
        tenant_role = self.get_tenant_role(tenant_id)
        return tenant_role is not None and tenant_role.role == role

    def has_permission_in_tenant(self, tenant_id: TenantId, permission: Permission) -> bool:
        """Check if user has specific permission in tenant."""
        tenant_role = self.get_tenant_role(tenant_id)
        if tenant_role is None:
            return False
        return permission in tenant_role.permissions

    def get_tenant_ids(self) -> List[TenantId]:
        """Get list of tenant IDs user has access to."""
        return [tr.tenant_id for tr in self.tenant_roles]

    def is_super_admin(self) -> bool:
        """Check if user is a super admin."""
        return any(tr.role == UserRole.SUPER_ADMIN for tr in self.tenant_roles)


@dataclass
class UserSession:
    """User session for authentication tracking."""
    id: str
    user_id: UserId
    tenant_id: Optional[TenantId]
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow().replace(hour=23, minute=59, second=59))
    ip_address: str = ""
    user_agent: str = ""
    is_active: bool = True
    last_activity: datetime = field(default_factory=datetime.utcnow)


# Default permission sets for each role
DEFAULT_PERMISSIONS = {
    UserRole.SUPER_ADMIN: {
        Permission("platform.manage", "platform", "manage", "Full platform management"),
        Permission("tenant.create", "tenant", "create", "Create new tenants"),
        Permission("tenant.delete", "tenant", "delete", "Delete tenants"),
        Permission("user.manage_all", "user", "manage", "Manage all users"),
    },
    UserRole.TENANT_ADMIN: {
        Permission("tenant.manage", "tenant", "manage", "Manage tenant settings"),
        Permission("user.invite", "user", "invite", "Invite new users"),
        Permission("user.manage", "user", "manage", "Manage tenant users"),
        Permission("dataset.manage", "dataset", "manage", "Manage all datasets"),
        Permission("model.manage", "model", "manage", "Manage all models"),
        Permission("detection.manage", "detection", "manage", "Manage all detections"),
        Permission("billing.view", "billing", "view", "View billing information"),
    },
    UserRole.DATA_SCIENTIST: {
        Permission("dataset.create", "dataset", "create", "Create datasets"),
        Permission("dataset.edit", "dataset", "edit", "Edit own datasets"),
        Permission("dataset.view", "dataset", "view", "View datasets"),
        Permission("model.create", "model", "create", "Create models"),
        Permission("model.edit", "model", "edit", "Edit own models"),
        Permission("model.view", "model", "view", "View models"),
        Permission("detection.run", "detection", "run", "Run detections"),
        Permission("detection.view", "detection", "view", "View detection results"),
    },
    UserRole.ANALYST: {
        Permission("dataset.view", "dataset", "view", "View datasets"),
        Permission("model.view", "model", "view", "View models"),
        Permission("detection.run", "detection", "run", "Run detections"),
        Permission("detection.view", "detection", "view", "View detection results"),
        Permission("report.create", "report", "create", "Create reports"),
    },
    UserRole.VIEWER: {
        Permission("dataset.view", "dataset", "view", "View datasets"),
        Permission("model.view", "model", "view", "View models"),
        Permission("detection.view", "detection", "view", "View detection results"),
        Permission("report.view", "report", "view", "View reports"),
    },
}


def get_default_permissions(role: UserRole) -> Set[Permission]:
    """Get default permissions for a role."""
    return DEFAULT_PERMISSIONS.get(role, set())
