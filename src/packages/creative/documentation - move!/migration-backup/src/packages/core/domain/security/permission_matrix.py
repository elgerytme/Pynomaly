"""Permission matrix for role-based access control in Pynomaly.

This module defines the comprehensive permission matrix that maps roles to
their allowed actions on different resources. This serves as the central
definition of access control rules for the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from monorepo.domain.entities.user import Permission, UserRole


class ResourceType(str, Enum):
    """Types of resources in the system."""

    PLATFORM = "platform"
    TENANT = "tenant"
    USER = "user"
    DATASET = "dataset"
    MODEL = "model"
    DETECTOR = "detector"
    DETECTION = "detection"
    EXPERIMENT = "experiment"
    REPORT = "report"
    METRIC = "metric"
    BILLING = "billing"
    API_KEY = "api_key"
    AUDIT_LOG = "audit_log"


class ActionType(str, Enum):
    """Types of actions that can be performed on resources."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    MANAGE = "manage"
    EXECUTE = "execute"
    INVITE = "invite"
    EXPORT = "export"
    IMPORT = "import"
    SHARE = "share"
    ARCHIVE = "archive"


@dataclass(frozen=True)
class PermissionRule:
    """A single permission rule defining what action is allowed on which resource."""

    resource: ResourceType
    action: ActionType
    description: str
    conditions: list[str] = None  # Additional conditions like "own_only", "tenant_only"

    def to_permission(self) -> Permission:
        """Convert to domain Permission object."""
        return Permission(
            name=f"{self.resource}.{self.action}",
            resource=self.resource.value,
            action=self.action.value,
            description=self.description,
        )


class PermissionMatrix:
    """Central permission matrix defining all role-based access control rules."""

    # Super Admin - Full platform access
    SUPER_ADMIN_PERMISSIONS = [
        PermissionRule(
            ResourceType.PLATFORM, ActionType.MANAGE, "Full platform management"
        ),
        PermissionRule(ResourceType.TENANT, ActionType.CREATE, "Create new tenants"),
        PermissionRule(ResourceType.TENANT, ActionType.READ, "View all tenants"),
        PermissionRule(ResourceType.TENANT, ActionType.UPDATE, "Update any tenant"),
        PermissionRule(ResourceType.TENANT, ActionType.DELETE, "Delete tenants"),
        PermissionRule(
            ResourceType.USER, ActionType.MANAGE, "Manage all users across tenants"
        ),
        PermissionRule(
            ResourceType.AUDIT_LOG, ActionType.READ, "View platform audit logs"
        ),
        PermissionRule(
            ResourceType.BILLING, ActionType.READ, "View all billing information"
        ),
        PermissionRule(ResourceType.METRIC, ActionType.READ, "View platform metrics"),
    ]

    # Tenant Admin - Full access within their tenant
    TENANT_ADMIN_PERMISSIONS = [
        PermissionRule(
            ResourceType.TENANT, ActionType.READ, "View own tenant", ["own_tenant_only"]
        ),
        PermissionRule(
            ResourceType.TENANT,
            ActionType.UPDATE,
            "Update own tenant",
            ["own_tenant_only"],
        ),
        PermissionRule(ResourceType.USER, ActionType.CREATE, "Create users in tenant"),
        PermissionRule(ResourceType.USER, ActionType.READ, "View tenant users"),
        PermissionRule(ResourceType.USER, ActionType.UPDATE, "Update tenant users"),
        PermissionRule(ResourceType.USER, ActionType.DELETE, "Delete tenant users"),
        PermissionRule(ResourceType.USER, ActionType.INVITE, "Invite new users"),
        PermissionRule(
            ResourceType.DATASET, ActionType.MANAGE, "Manage all tenant datasets"
        ),
        PermissionRule(
            ResourceType.MODEL, ActionType.MANAGE, "Manage all tenant models"
        ),
        PermissionRule(
            ResourceType.DETECTOR, ActionType.MANAGE, "Manage all tenant detectors"
        ),
        PermissionRule(
            ResourceType.DETECTION, ActionType.MANAGE, "Manage all tenant detections"
        ),
        PermissionRule(
            ResourceType.EXPERIMENT, ActionType.MANAGE, "Manage all tenant experiments"
        ),
        PermissionRule(
            ResourceType.REPORT, ActionType.MANAGE, "Manage all tenant reports"
        ),
        PermissionRule(ResourceType.BILLING, ActionType.READ, "View tenant billing"),
        PermissionRule(ResourceType.API_KEY, ActionType.MANAGE, "Manage API keys"),
        PermissionRule(
            ResourceType.AUDIT_LOG, ActionType.READ, "View tenant audit logs"
        ),
        PermissionRule(ResourceType.METRIC, ActionType.READ, "View tenant metrics"),
    ]

    # Data Scientist - Can create and manage their own models and datasets
    DATA_SCIENTIST_PERMISSIONS = [
        PermissionRule(ResourceType.DATASET, ActionType.CREATE, "Create datasets"),
        PermissionRule(ResourceType.DATASET, ActionType.READ, "View datasets"),
        PermissionRule(
            ResourceType.DATASET, ActionType.UPDATE, "Update own datasets", ["own_only"]
        ),
        PermissionRule(
            ResourceType.DATASET, ActionType.DELETE, "Delete own datasets", ["own_only"]
        ),
        PermissionRule(ResourceType.DATASET, ActionType.EXPORT, "Export datasets"),
        PermissionRule(ResourceType.DATASET, ActionType.IMPORT, "Import datasets"),
        PermissionRule(ResourceType.MODEL, ActionType.CREATE, "Create models"),
        PermissionRule(ResourceType.MODEL, ActionType.READ, "View models"),
        PermissionRule(
            ResourceType.MODEL, ActionType.UPDATE, "Update own models", ["own_only"]
        ),
        PermissionRule(
            ResourceType.MODEL, ActionType.DELETE, "Delete own models", ["own_only"]
        ),
        PermissionRule(ResourceType.MODEL, ActionType.EXPORT, "Export models"),
        PermissionRule(ResourceType.DETECTOR, ActionType.CREATE, "Create detectors"),
        PermissionRule(ResourceType.DETECTOR, ActionType.READ, "View detectors"),
        PermissionRule(
            ResourceType.DETECTOR,
            ActionType.UPDATE,
            "Update own detectors",
            ["own_only"],
        ),
        PermissionRule(
            ResourceType.DETECTOR,
            ActionType.DELETE,
            "Delete own detectors",
            ["own_only"],
        ),
        PermissionRule(ResourceType.DETECTION, ActionType.EXECUTE, "Run detections"),
        PermissionRule(
            ResourceType.DETECTION, ActionType.READ, "View detection results"
        ),
        PermissionRule(
            ResourceType.EXPERIMENT, ActionType.CREATE, "Create experiments"
        ),
        PermissionRule(ResourceType.EXPERIMENT, ActionType.READ, "View experiments"),
        PermissionRule(
            ResourceType.EXPERIMENT,
            ActionType.UPDATE,
            "Update own experiments",
            ["own_only"],
        ),
        PermissionRule(ResourceType.REPORT, ActionType.CREATE, "Create reports"),
        PermissionRule(ResourceType.REPORT, ActionType.READ, "View reports"),
        PermissionRule(ResourceType.API_KEY, ActionType.CREATE, "Create own API keys"),
        PermissionRule(
            ResourceType.API_KEY, ActionType.READ, "View own API keys", ["own_only"]
        ),
        PermissionRule(ResourceType.METRIC, ActionType.READ, "View metrics"),
    ]

    # Analyst - Can run detections and create reports on existing models
    ANALYST_PERMISSIONS = [
        PermissionRule(ResourceType.DATASET, ActionType.READ, "View datasets"),
        PermissionRule(ResourceType.MODEL, ActionType.READ, "View models"),
        PermissionRule(ResourceType.DETECTOR, ActionType.READ, "View detectors"),
        PermissionRule(ResourceType.DETECTION, ActionType.EXECUTE, "Run detections"),
        PermissionRule(
            ResourceType.DETECTION, ActionType.READ, "View detection results"
        ),
        PermissionRule(ResourceType.EXPERIMENT, ActionType.READ, "View experiments"),
        PermissionRule(ResourceType.REPORT, ActionType.CREATE, "Create reports"),
        PermissionRule(ResourceType.REPORT, ActionType.READ, "View reports"),
        PermissionRule(
            ResourceType.REPORT, ActionType.UPDATE, "Update own reports", ["own_only"]
        ),
        PermissionRule(ResourceType.REPORT, ActionType.EXPORT, "Export reports"),
        PermissionRule(ResourceType.API_KEY, ActionType.CREATE, "Create own API keys"),
        PermissionRule(
            ResourceType.API_KEY, ActionType.READ, "View own API keys", ["own_only"]
        ),
        PermissionRule(ResourceType.METRIC, ActionType.READ, "View metrics"),
    ]

    # Viewer - Read-only access
    VIEWER_PERMISSIONS = [
        PermissionRule(ResourceType.DATASET, ActionType.READ, "View datasets"),
        PermissionRule(ResourceType.MODEL, ActionType.READ, "View models"),
        PermissionRule(ResourceType.DETECTOR, ActionType.READ, "View detectors"),
        PermissionRule(
            ResourceType.DETECTION, ActionType.READ, "View detection results"
        ),
        PermissionRule(ResourceType.EXPERIMENT, ActionType.READ, "View experiments"),
        PermissionRule(ResourceType.REPORT, ActionType.READ, "View reports"),
        PermissionRule(ResourceType.METRIC, ActionType.READ, "View metrics"),
    ]

    @classmethod
    def get_role_permissions(cls, role: UserRole) -> set[Permission]:
        """Get all permissions for a specific role."""
        permission_rules = {
            UserRole.SUPER_ADMIN: cls.SUPER_ADMIN_PERMISSIONS,
            UserRole.TENANT_ADMIN: cls.TENANT_ADMIN_PERMISSIONS,
            UserRole.DATA_SCIENTIST: cls.DATA_SCIENTIST_PERMISSIONS,
            UserRole.ANALYST: cls.ANALYST_PERMISSIONS,
            UserRole.VIEWER: cls.VIEWER_PERMISSIONS,
        }

        rules = permission_rules.get(role, [])
        return {rule.to_permission() for rule in rules}

    @classmethod
    def get_all_permissions(cls) -> set[Permission]:
        """Get all possible permissions in the system."""
        all_permissions = set()
        for role in UserRole:
            all_permissions.update(cls.get_role_permissions(role))
        return all_permissions

    @classmethod
    def get_permission_hierarchy(cls) -> dict[UserRole, int]:
        """Get role hierarchy (higher number = more permissions)."""
        return {
            UserRole.VIEWER: 1,
            UserRole.ANALYST: 2,
            UserRole.DATA_SCIENTIST: 3,
            UserRole.TENANT_ADMIN: 4,
            UserRole.SUPER_ADMIN: 5,
        }

    @classmethod
    def can_role_grant_permission(
        cls, granter_role: UserRole, permission: Permission
    ) -> bool:
        """Check if a role can grant a specific permission to another user."""
        granter_permissions = cls.get_role_permissions(granter_role)

        # Super admins can grant any permission
        if granter_role == UserRole.SUPER_ADMIN:
            return True

        # Tenant admins can grant permissions within their scope
        if granter_role == UserRole.TENANT_ADMIN:
            # Cannot grant super admin permissions
            super_admin_permissions = cls.get_role_permissions(UserRole.SUPER_ADMIN)
            if permission in super_admin_permissions:
                return False
            return True

        # Other roles cannot grant permissions
        return False

    @classmethod
    def get_matrix_summary(cls) -> dict[str, dict[str, list[str]]]:
        """Get a summary of the permission matrix for documentation."""
        summary = {}

        for role in UserRole:
            permissions = cls.get_role_permissions(role)
            role_summary = {}

            # Group permissions by resource
            for permission in permissions:
                resource = permission.resource
                if resource not in role_summary:
                    role_summary[resource] = []
                role_summary[resource].append(permission.action)

            summary[role.value] = role_summary

        return summary


# Permission checking utilities
def has_permission(
    user_permissions: set[Permission], required_permission: Permission
) -> bool:
    """Check if user has a specific permission."""
    return required_permission in user_permissions


def has_resource_access(
    user_permissions: set[Permission], resource: ResourceType, action: ActionType
) -> bool:
    """Check if user can perform an action on a resource."""
    required_permission = Permission(
        name=f"{resource.value}.{action.value}",
        resource=resource.value,
        action=action.value,
        description="",
    )
    return has_permission(user_permissions, required_permission)


def get_user_resource_permissions(
    user_permissions: set[Permission], resource: ResourceType
) -> list[ActionType]:
    """Get all actions a user can perform on a specific resource."""
    actions = []
    for permission in user_permissions:
        if permission.resource == resource.value:
            try:
                action = ActionType(permission.action)
                actions.append(action)
            except ValueError:
                # Skip invalid actions
                pass
    return actions
