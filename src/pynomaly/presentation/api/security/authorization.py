"""
Role-based access control (RBAC) and authorization for Pynomaly API.

This module provides:
- Role-based access control
- Permission management
- Resource-level authorization
- Fine-grained access control
"""

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """System permissions."""

    # Data permissions
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"

    # Model permissions
    READ_MODELS = "read_models"
    CREATE_MODELS = "create_models"
    UPDATE_MODELS = "update_models"
    DELETE_MODELS = "delete_models"
    DEPLOY_MODELS = "deploy_models"

    # Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_PERMISSIONS = "manage_permissions"
    VIEW_AUDIT_LOGS = "view_audit_logs"

    # System permissions
    CONFIGURE_SYSTEM = "configure_system"
    MANAGE_INFRASTRUCTURE = "manage_infrastructure"

    # Governance permissions
    APPROVE_MODELS = "approve_models"
    MANAGE_GOVERNANCE = "manage_governance"


class Role(str, Enum):
    """System roles."""

    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"


class AuthorizationManager:
    """Authorization and permission management."""

    def __init__(self):
        self.role_permissions = self._initialize_role_permissions()
        self.user_roles = {}  # user_id -> set of roles
        self.user_permissions = {}  # user_id -> set of additional permissions
        self.resource_permissions = {}  # resource_id -> permissions required

    def _initialize_role_permissions(self) -> dict[Role, set[Permission]]:
        """Initialize default role permissions."""
        return {
            Role.ADMIN: {
                Permission.READ_DATA,
                Permission.WRITE_DATA,
                Permission.DELETE_DATA,
                Permission.READ_MODELS,
                Permission.CREATE_MODELS,
                Permission.UPDATE_MODELS,
                Permission.DELETE_MODELS,
                Permission.DEPLOY_MODELS,
                Permission.MANAGE_USERS,
                Permission.MANAGE_ROLES,
                Permission.MANAGE_PERMISSIONS,
                Permission.VIEW_AUDIT_LOGS,
                Permission.CONFIGURE_SYSTEM,
                Permission.MANAGE_INFRASTRUCTURE,
                Permission.APPROVE_MODELS,
                Permission.MANAGE_GOVERNANCE,
            },
            Role.DATA_SCIENTIST: {
                Permission.READ_DATA,
                Permission.WRITE_DATA,
                Permission.READ_MODELS,
                Permission.CREATE_MODELS,
                Permission.UPDATE_MODELS,
            },
            Role.ML_ENGINEER: {
                Permission.READ_DATA,
                Permission.WRITE_DATA,
                Permission.READ_MODELS,
                Permission.CREATE_MODELS,
                Permission.UPDATE_MODELS,
                Permission.DELETE_MODELS,
                Permission.DEPLOY_MODELS,
            },
            Role.ANALYST: {
                Permission.READ_DATA,
                Permission.READ_MODELS,
            },
            Role.VIEWER: {
                Permission.READ_DATA,
                Permission.READ_MODELS,
            },
            Role.GUEST: {
                Permission.READ_DATA,
            },
        }

    def assign_role(self, user_id: str, role: Role) -> None:
        """Assign role to user."""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()

        self.user_roles[user_id].add(role)
        logger.info(f"Role {role} assigned to user {user_id}")

    def revoke_role(self, user_id: str, role: Role) -> None:
        """Revoke role from user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role)
            if not self.user_roles[user_id]:
                del self.user_roles[user_id]

        logger.info(f"Role {role} revoked from user {user_id}")

    def grant_permission(self, user_id: str, permission: Permission) -> None:
        """Grant additional permission to user."""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = set()

        self.user_permissions[user_id].add(permission)
        logger.info(f"Permission {permission} granted to user {user_id}")

    def revoke_permission(self, user_id: str, permission: Permission) -> None:
        """Revoke permission from user."""
        if user_id in self.user_permissions:
            self.user_permissions[user_id].discard(permission)
            if not self.user_permissions[user_id]:
                del self.user_permissions[user_id]

        logger.info(f"Permission {permission} revoked from user {user_id}")

    def get_user_permissions(self, user_id: str) -> set[Permission]:
        """Get all permissions for a user."""
        permissions = set()

        # Add permissions from roles
        if user_id in self.user_roles:
            for role in self.user_roles[user_id]:
                permissions.update(self.role_permissions.get(role, set()))

        # Add additional permissions
        if user_id in self.user_permissions:
            permissions.update(self.user_permissions[user_id])

        return permissions

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user_permissions = self.get_user_permissions(user_id)
        return permission in user_permissions

    def has_any_permission(self, user_id: str, permissions: list[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        user_permissions = self.get_user_permissions(user_id)
        return any(perm in user_permissions for perm in permissions)

    def has_all_permissions(self, user_id: str, permissions: list[Permission]) -> bool:
        """Check if user has all specified permissions."""
        user_permissions = self.get_user_permissions(user_id)
        return all(perm in user_permissions for perm in permissions)

    def authorize_request(self, user_id: str, resource: str, action: str) -> bool:
        """Authorize user request for resource and action."""
        required_permissions = self._get_required_permissions(resource, action)

        if not required_permissions:
            # No specific permissions required
            return True

        return self.has_any_permission(user_id, required_permissions)

    def _get_required_permissions(self, resource: str, action: str) -> list[Permission]:
        """Get required permissions for resource and action."""
        permission_map = {
            ("data", "read"): [Permission.READ_DATA],
            ("data", "write"): [Permission.WRITE_DATA],
            ("data", "delete"): [Permission.DELETE_DATA],
            ("models", "read"): [Permission.READ_MODELS],
            ("models", "create"): [Permission.CREATE_MODELS],
            ("models", "update"): [Permission.UPDATE_MODELS],
            ("models", "delete"): [Permission.DELETE_MODELS],
            ("models", "deploy"): [Permission.DEPLOY_MODELS],
            ("users", "manage"): [Permission.MANAGE_USERS],
            ("roles", "manage"): [Permission.MANAGE_ROLES],
            ("permissions", "manage"): [Permission.MANAGE_PERMISSIONS],
            ("audit", "view"): [Permission.VIEW_AUDIT_LOGS],
            ("system", "configure"): [Permission.CONFIGURE_SYSTEM],
            ("infrastructure", "manage"): [Permission.MANAGE_INFRASTRUCTURE],
            ("governance", "approve"): [Permission.APPROVE_MODELS],
            ("governance", "manage"): [Permission.MANAGE_GOVERNANCE],
        }

        return permission_map.get((resource, action), [])


class RoleBasedAccessControl:
    """Role-based access control decorator and middleware."""

    def __init__(self, authorization_manager: AuthorizationManager):
        self.auth_manager = authorization_manager

    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                # Extract user_id from request context
                user_id = self._get_user_from_context()

                if not user_id:
                    raise UnauthorizedError("User not authenticated")

                if not self.auth_manager.has_permission(user_id, permission):
                    raise ForbiddenError(f"Permission {permission} required")

                return func(*args, **kwargs)

            return wrapper

        return decorator

    def require_role(self, role: Role):
        """Decorator to require specific role."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                user_id = self._get_user_from_context()

                if not user_id:
                    raise UnauthorizedError("User not authenticated")

                if role not in self.auth_manager.user_roles.get(user_id, set()):
                    raise ForbiddenError(f"Role {role} required")

                return func(*args, **kwargs)

            return wrapper

        return decorator

    def require_any_permission(self, permissions: list[Permission]):
        """Decorator to require any of the specified permissions."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                user_id = self._get_user_from_context()

                if not user_id:
                    raise UnauthorizedError("User not authenticated")

                if not self.auth_manager.has_any_permission(user_id, permissions):
                    raise ForbiddenError(f"One of {permissions} required")

                return func(*args, **kwargs)

            return wrapper

        return decorator

    def authorize_resource(self, resource: str, action: str):
        """Decorator to authorize resource access."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                user_id = self._get_user_from_context()

                if not user_id:
                    raise UnauthorizedError("User not authenticated")

                if not self.auth_manager.authorize_request(user_id, resource, action):
                    raise ForbiddenError(f"Access denied to {resource}:{action}")

                return func(*args, **kwargs)

            return wrapper

        return decorator

    def _get_user_from_context(self) -> str | None:
        """Get user ID from request context."""
        # This would extract user_id from Flask/FastAPI request context
        # For now, return placeholder
        return "current_user_id"


class ResourceOwnershipManager:
    """Manage resource ownership and access control."""

    def __init__(self):
        self.resource_owners = {}  # resource_id -> owner_user_id
        self.resource_collaborators = {}  # resource_id -> set of user_ids
        self.resource_permissions = {}  # resource_id -> {user_id: permissions}

    def set_resource_owner(self, resource_id: str, owner_id: str) -> None:
        """Set resource owner."""
        self.resource_owners[resource_id] = owner_id
        logger.info(f"Resource {resource_id} owned by {owner_id}")

    def add_collaborator(
        self, resource_id: str, user_id: str, permissions: list[Permission]
    ) -> None:
        """Add collaborator to resource."""
        if resource_id not in self.resource_collaborators:
            self.resource_collaborators[resource_id] = set()

        self.resource_collaborators[resource_id].add(user_id)

        if resource_id not in self.resource_permissions:
            self.resource_permissions[resource_id] = {}

        self.resource_permissions[resource_id][user_id] = set(permissions)
        logger.info(f"User {user_id} added as collaborator to {resource_id}")

    def remove_collaborator(self, resource_id: str, user_id: str) -> None:
        """Remove collaborator from resource."""
        if resource_id in self.resource_collaborators:
            self.resource_collaborators[resource_id].discard(user_id)

        if resource_id in self.resource_permissions:
            self.resource_permissions[resource_id].pop(user_id, None)

        logger.info(f"User {user_id} removed as collaborator from {resource_id}")

    def can_access_resource(
        self, resource_id: str, user_id: str, permission: Permission
    ) -> bool:
        """Check if user can access resource with specific permission."""
        # Check if user is owner
        if self.resource_owners.get(resource_id) == user_id:
            return True

        # Check collaborator permissions
        if resource_id in self.resource_permissions:
            user_permissions = self.resource_permissions[resource_id].get(
                user_id, set()
            )
            return permission in user_permissions

        return False

    def get_user_resources(self, user_id: str) -> dict[str, str]:
        """Get resources accessible by user."""
        resources = {}

        # Add owned resources
        for resource_id, owner_id in self.resource_owners.items():
            if owner_id == user_id:
                resources[resource_id] = "owner"

        # Add collaborated resources
        for resource_id, collaborators in self.resource_collaborators.items():
            if user_id in collaborators:
                resources[resource_id] = "collaborator"

        return resources


class UnauthorizedError(Exception):
    """Raised when user is not authenticated."""

    pass


class ForbiddenError(Exception):
    """Raised when user is authenticated but lacks required permissions."""

    pass


class SecurityPolicy:
    """Security policy enforcement."""

    def __init__(self):
        self.policies = {}
        self.policy_violations = []

    def add_policy(
        self, name: str, policy_func: callable, description: str = ""
    ) -> None:
        """Add security policy."""
        self.policies[name] = {
            "func": policy_func,
            "description": description,
            "violations": [],
        }

    def enforce_policy(self, policy_name: str, context: dict[str, Any]) -> bool:
        """Enforce security policy."""
        if policy_name not in self.policies:
            logger.warning(f"Policy {policy_name} not found")
            return False

        policy = self.policies[policy_name]

        try:
            result = policy["func"](context)
            if not result:
                violation = {
                    "policy": policy_name,
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                policy["violations"].append(violation)
                self.policy_violations.append(violation)
                logger.warning(f"Policy violation: {policy_name}")

            return result
        except Exception as e:
            logger.error(f"Error enforcing policy {policy_name}: {e}")
            return False

    def get_policy_violations(
        self, policy_name: str | None = None
    ) -> list[dict[str, Any]]:
        """Get policy violations."""
        if policy_name:
            return self.policies.get(policy_name, {}).get("violations", [])
        return self.policy_violations
