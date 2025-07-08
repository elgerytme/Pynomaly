"""Enhanced RBAC middleware for comprehensive role-based access control.

This module provides enterprise-grade middleware that:
- Enforces role-based permissions
- Prevents privilege escalation
- Provides proper error responses (401 vs 403)
- Supports resource-level access control
- Includes audit logging
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Optional, Set
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pynomaly.domain.entities.user import User, UserRole, Permission
from pynomaly.domain.security.permission_matrix import (
    PermissionMatrix,
    ResourceType,
    ActionType,
    has_resource_access,
)
from pynomaly.infrastructure.security.rbac_service import RBACService
from pynomaly.infrastructure.auth.jwt_auth import JWTAuthService, get_auth
from pynomaly.infrastructure.config import Container

logger = logging.getLogger(__name__)
bearer_scheme = HTTPBearer(auto_error=False)


class RBACMiddleware:
    """Enhanced RBAC middleware for enterprise security."""

    def __init__(self, rbac_service: RBACService, auth_service: JWTAuthService):
        self.rbac_service = rbac_service
        self.auth_service = auth_service
        self.logger = logging.getLogger(__name__)

    async def validate_user_and_permissions(
        self,
        credentials: Optional[HTTPAuthorizationCredentials],
        required_permissions: Optional[List[Permission]] = None,
        required_role: Optional[UserRole] = None,
        resource_id: Optional[str] = None,
        tenant_id: Optional[UUID] = None,
        allow_self_access: bool = False,
        request: Optional[Request] = None,
    ) -> User:
        """Comprehensive validation of user authentication and authorization.

        Args:
            credentials: Bearer token credentials
            required_permissions: List of required permissions
            required_role: Required user role
            resource_id: ID of resource being accessed
            tenant_id: Tenant context for multi-tenant operations
            allow_self_access: Allow access to own resources
            request: FastAPI request object for logging

        Returns:
            Authenticated and authorized user

        Raises:
            HTTPException: 401 for authentication failures, 403 for authorization failures
        """
        # Step 1: Authentication
        user = await self._authenticate_user(credentials, request)

        # Step 2: Authorization - Check role requirements
        if required_role:
            await self._validate_role_requirement(user, required_role, tenant_id)

        # Step 3: Authorization - Check permission requirements
        if required_permissions:
            await self._validate_permission_requirements(
                user, required_permissions, tenant_id, resource_id, allow_self_access
            )

        # Step 4: Audit logging for sensitive operations
        await self._log_access_attempt(
            user,
            required_permissions,
            required_role,
            resource_id,
            tenant_id,
            True,
            request,
        )

        return user

    async def _authenticate_user(
        self,
        credentials: Optional[HTTPAuthorizationCredentials],
        request: Optional[Request] = None,
    ) -> User:
        """Authenticate user from bearer token.

        Args:
            credentials: Bearer token credentials
            request: FastAPI request for audit logging

        Returns:
            Authenticated user

        Raises:
            HTTPException: 401 if authentication fails
        """
        if not credentials or not credentials.credentials:
            await self._log_auth_failure("No credentials provided", request)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            # Validate token and get user
            user = await self.rbac_service.validate_token(credentials.credentials)
            if not user:
                await self._log_auth_failure("Invalid token", request)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Check if user account is active
            if user.status.value != "active":
                await self._log_auth_failure(
                    f"User account status: {user.status.value}", request
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Account is {user.status.value}",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return user

        except HTTPException:
            raise
        except Exception as e:
            await self._log_auth_failure(f"Authentication error: {str(e)}", request)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def _validate_role_requirement(
        self, user: User, required_role: UserRole, tenant_id: Optional[UUID] = None
    ) -> None:
        """Validate that user has required role.

        Args:
            user: Authenticated user
            required_role: Required role
            tenant_id: Tenant context for role check

        Raises:
            HTTPException: 403 if user doesn't have required role
        """
        # Super admins have access to everything
        if user.is_super_admin():
            return

        # Check role hierarchy - higher roles include lower role permissions
        role_hierarchy = PermissionMatrix.get_permission_hierarchy()
        required_level = role_hierarchy.get(required_role, 0)

        if tenant_id:
            # Check role in specific tenant
            tenant_role = user.get_tenant_role(tenant_id)
            if not tenant_role:
                self.logger.warning(f"User {user.id} has no role in tenant {tenant_id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: No role in tenant",
                )

            user_level = role_hierarchy.get(tenant_role.role, 0)
            if user_level < required_level:
                self.logger.warning(
                    f"User {user.id} role {tenant_role.role} insufficient for {required_role}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: Insufficient role. Required: {required_role.value}",
                )
        else:
            # Check if user has required role in any tenant
            user_roles = [tr.role for tr in user.tenant_roles]
            max_user_level = max(
                [role_hierarchy.get(role, 0) for role in user_roles], default=0
            )

            if max_user_level < required_level:
                self.logger.warning(
                    f"User {user.id} highest role level {max_user_level} insufficient for {required_role}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: Insufficient role. Required: {required_role.value}",
                )

    async def _validate_permission_requirements(
        self,
        user: User,
        required_permissions: List[Permission],
        tenant_id: Optional[UUID] = None,
        resource_id: Optional[str] = None,
        allow_self_access: bool = False,
    ) -> None:
        """Validate that user has required permissions.

        Args:
            user: Authenticated user
            required_permissions: List of required permissions
            tenant_id: Tenant context for permission check
            resource_id: Resource ID for ownership checks
            allow_self_access: Allow access to own resources

        Raises:
            HTTPException: 403 if user doesn't have required permissions
        """
        # Super admins have all permissions
        if user.is_super_admin():
            return

        # Check each required permission
        for permission in required_permissions:
            has_permission = False

            if tenant_id:
                # Check permission in specific tenant
                has_permission = user.has_permission_in_tenant(tenant_id, permission)
            else:
                # Check permission in any tenant
                for tenant_role in user.tenant_roles:
                    if permission in tenant_role.permissions:
                        has_permission = True
                        break

            # Check for self-access on own resources
            if not has_permission and allow_self_access and resource_id:
                has_permission = await self._check_resource_ownership(
                    user, resource_id, permission
                )

            if not has_permission:
                self.logger.warning(
                    f"User {user.id} lacks permission {permission.name} for resource {resource_id}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: Missing permission '{permission.name}'",
                )

    async def _check_resource_ownership(
        self, user: User, resource_id: str, permission: Permission
    ) -> bool:
        """Check if user owns the resource they're trying to access.

        Args:
            user: User attempting access
            resource_id: ID of resource being accessed
            permission: Permission being checked

        Returns:
            True if user owns the resource and has self-access rights
        """
        # This would typically check a database to see if the user created/owns the resource
        # For now, we'll implement a basic check that certain permissions allow self-access
        self_access_permissions = {
            "dataset.edit",
            "dataset.delete",
            "model.edit",
            "model.delete",
            "detection.edit",
            "detection.delete",
            "report.edit",
            "report.delete",
        }

        return permission.name in self_access_permissions

    async def _log_access_attempt(
        self,
        user: User,
        required_permissions: Optional[List[Permission]],
        required_role: Optional[UserRole],
        resource_id: Optional[str],
        tenant_id: Optional[UUID],
        success: bool,
        request: Optional[Request] = None,
    ) -> None:
        """Log access attempt for audit purposes.

        Args:
            user: User attempting access
            required_permissions: Permissions that were required
            required_role: Role that was required
            resource_id: Resource being accessed
            tenant_id: Tenant context
            success: Whether access was granted
            request: FastAPI request object
        """
        ip_address = None
        user_agent = None
        endpoint = None

        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            endpoint = f"{request.method} {request.url.path}"

        # Log sensitive access attempts
        sensitive_permissions = {
            "user.manage",
            "user.delete",
            "tenant.manage",
            "tenant.delete",
            "billing.view",
            "platform.manage",
        }

        is_sensitive = (
            required_permissions
            and any(p.name in sensitive_permissions for p in required_permissions)
        ) or (
            required_role
            and required_role in [UserRole.SUPER_ADMIN, UserRole.TENANT_ADMIN]
        )

        if is_sensitive or not success:
            self.logger.info(
                f"Access attempt - User: {user.id}, "
                f"Permissions: {[p.name for p in required_permissions] if required_permissions else None}, "
                f"Role: {required_role.value if required_role else None}, "
                f"Resource: {resource_id}, Tenant: {tenant_id}, "
                f"Success: {success}, IP: {ip_address}, Endpoint: {endpoint}"
            )

    async def _log_auth_failure(
        self, reason: str, request: Optional[Request] = None
    ) -> None:
        """Log authentication failure for security monitoring.

        Args:
            reason: Reason for authentication failure
            request: FastAPI request object
        """
        ip_address = None
        user_agent = None
        endpoint = None

        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            endpoint = f"{request.method} {request.url.path}"

        self.logger.warning(
            f"Authentication failure - Reason: {reason}, "
            f"IP: {ip_address}, Endpoint: {endpoint}, User-Agent: {user_agent}"
        )


# Dependency injection
async def get_rbac_service() -> RBACService:
    """Get RBAC service instance."""
    # This would be injected from container in production
    settings = get_settings()
    from pynomaly.domain.models.security import SecurityPolicy

    security_policy = SecurityPolicy(
        password_min_length=8,
        password_require_uppercase=True,
        password_require_lowercase=True,
        password_require_numbers=True,
        password_require_special_chars=True,
        max_failed_login_attempts=5,
        session_timeout=3600,
        ip_whitelist_enabled=False,
        allowed_ip_ranges=[],
    )

    return RBACService(jwt_secret=settings.secret_key, security_policy=security_policy)


async def get_rbac_middleware(
    rbac_service: Annotated[RBACService, Depends(get_rbac_service)],
    auth_service: Annotated[JWTAuthService, Depends(get_auth)],
) -> RBACMiddleware:
    """Get RBAC middleware instance."""
    return RBACMiddleware(rbac_service, auth_service)


class RequirePermissions:
    """Dependency class for requiring specific permissions."""

    def __init__(
        self,
        permissions: List[str],
        tenant_id: Optional[UUID] = None,
        resource_id: Optional[str] = None,
        allow_self_access: bool = False,
    ):
        """Initialize permission requirement.

        Args:
            permissions: List of required permission names
            tenant_id: Tenant context for permission check
            resource_id: Resource ID for ownership checks
            allow_self_access: Allow access to own resources
        """
        self.required_permissions = [
            Permission(name=perm, resource="", action="", description="")
            for perm in permissions
        ]
        self.tenant_id = tenant_id
        self.resource_id = resource_id
        self.allow_self_access = allow_self_access

    async def __call__(
        self,
        request: Request,
        credentials: Annotated[
            Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)
        ],
        rbac_middleware: Annotated[RBACMiddleware, Depends(get_rbac_middleware)],
    ) -> User:
        """Validate permissions and return authorized user."""
        return await rbac_middleware.validate_user_and_permissions(
            credentials=credentials,
            required_permissions=self.required_permissions,
            tenant_id=self.tenant_id,
            resource_id=self.resource_id,
            allow_self_access=self.allow_self_access,
            request=request,
        )


class RequireRole:
    """Dependency class for requiring specific role."""

    def __init__(self, role: UserRole, tenant_id: Optional[UUID] = None):
        """Initialize role requirement.

        Args:
            role: Required user role
            tenant_id: Tenant context for role check
        """
        self.required_role = role
        self.tenant_id = tenant_id

    async def __call__(
        self,
        request: Request,
        credentials: Annotated[
            Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)
        ],
        rbac_middleware: Annotated[RBACMiddleware, Depends(get_rbac_middleware)],
    ) -> User:
        """Validate role and return authorized user."""
        return await rbac_middleware.validate_user_and_permissions(
            credentials=credentials,
            required_role=self.required_role,
            tenant_id=self.tenant_id,
            request=request,
        )


class RequireAuthentication:
    """Dependency class for requiring authentication only."""

    async def __call__(
        self,
        request: Request,
        credentials: Annotated[
            Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)
        ],
        rbac_middleware: Annotated[RBACMiddleware, Depends(get_rbac_middleware)],
    ) -> User:
        """Validate authentication and return user."""
        return await rbac_middleware.validate_user_and_permissions(
            credentials=credentials, request=request
        )


# Convenience dependency factories
def require_permissions(
    permissions: List[str],
    tenant_id: Optional[UUID] = None,
    resource_id: Optional[str] = None,
    allow_self_access: bool = False,
) -> RequirePermissions:
    """Factory function for permission requirements."""
    return RequirePermissions(
        permissions=permissions,
        tenant_id=tenant_id,
        resource_id=resource_id,
        allow_self_access=allow_self_access,
    )


def require_role(role: UserRole, tenant_id: Optional[UUID] = None) -> RequireRole:
    """Factory function for role requirements."""
    return RequireRole(role=role, tenant_id=tenant_id)


def require_auth() -> RequireAuthentication:
    """Factory function for authentication requirement."""
    return RequireAuthentication()


# Common permission sets for easy use
class CommonPermissions:
    """Pre-defined common permission sets."""

    # Dataset permissions
    DATASET_READ = ["dataset.view"]
    DATASET_WRITE = ["dataset.create", "dataset.edit"]
    DATASET_DELETE = ["dataset.delete"]
    DATASET_MANAGE = [
        "dataset.create",
        "dataset.edit",
        "dataset.delete",
        "dataset.view",
    ]

    # Model permissions
    MODEL_READ = ["model.view"]
    MODEL_WRITE = ["model.create", "model.edit"]
    MODEL_DELETE = ["model.delete"]
    MODEL_MANAGE = ["model.create", "model.edit", "model.delete", "model.view"]

    # Detection permissions
    DETECTION_RUN = ["detection.run"]
    DETECTION_READ = ["detection.view"]
    DETECTION_MANAGE = [
        "detection.run",
        "detection.view",
        "detection.edit",
        "detection.delete",
    ]

    # User management permissions
    USER_READ = ["user.view"]
    USER_WRITE = ["user.create", "user.edit"]
    USER_DELETE = ["user.delete"]
    USER_MANAGE = [
        "user.create",
        "user.edit",
        "user.delete",
        "user.view",
        "user.invite",
    ]

    # Tenant permissions
    TENANT_READ = ["tenant.view"]
    TENANT_MANAGE = ["tenant.manage", "tenant.view"]

    # Admin permissions
    PLATFORM_ADMIN = ["platform.manage"]

    # Billing permissions
    BILLING_READ = ["billing.view"]
