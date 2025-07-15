"""Privilege escalation prevention and security policy enforcement.

This module provides additional security measures to prevent privilege escalation
and ensure proper access control is maintained throughout the system.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from uuid import UUID

from pynomaly.domain.entities.user import Permission, User, UserRole
from pynomaly.domain.security.permission_matrix import PermissionMatrix
from pynomaly.infrastructure.security.rbac_service import RBACService

logger = logging.getLogger(__name__)


class PrivilegeEscalationPrevention:
    """Security measures to prevent privilege escalation attacks."""

    def __init__(self, rbac_service: RBACService):
        self.rbac_service = rbac_service
        self.logger = logging.getLogger(__name__)

        # Track suspicious activities
        self.failed_privilege_attempts: dict[UUID, list[datetime]] = {}
        self.role_assignment_history: dict[UUID, list[dict]] = {}

    async def validate_role_assignment(
        self,
        granter_id: UUID,
        target_user_id: UUID,
        new_role: UserRole,
        tenant_id: UUID | None = None,
    ) -> bool:
        """Validate that a role assignment is legitimate and not a privilege escalation.

        Args:
            granter_id: User attempting to grant the role
            target_user_id: User receiving the role
            new_role: Role being assigned
            tenant_id: Tenant context for role assignment

        Returns:
            True if role assignment is valid

        Raises:
            SecurityException: If privilege escalation attempt detected
        """
        granter = await self._get_user_safely(granter_id)
        target_user = await self._get_user_safely(target_user_id)

        if not granter or not target_user:
            await self._log_security_event(
                "Invalid user in role assignment attempt",
                granter_id,
                target_user_id,
                new_role,
            )
            return False

        # Rule 1: Cannot grant role to yourself (prevent self-escalation)
        if granter_id == target_user_id:
            await self._log_security_event(
                "Self-role assignment attempt blocked",
                granter_id,
                target_user_id,
                new_role,
            )
            return False

        # Rule 2: Can only grant roles you have permission to grant
        if not await self._can_grant_role(granter, new_role, tenant_id):
            await self._log_security_event(
                "Unauthorized role grant attempt", granter_id, target_user_id, new_role
            )
            return False

        # Rule 3: Cannot grant higher privileges than you have
        if not await self._has_sufficient_privileges(granter, new_role, tenant_id):
            await self._log_security_event(
                "Privilege escalation attempt blocked",
                granter_id,
                target_user_id,
                new_role,
            )
            return False

        # Rule 4: Rate limiting on role assignments
        if not await self._check_role_assignment_rate_limit(granter_id):
            await self._log_security_event(
                "Role assignment rate limit exceeded",
                granter_id,
                target_user_id,
                new_role,
            )
            return False

        # Rule 5: Cannot create super admin unless you are one
        if new_role == UserRole.SUPER_ADMIN and not granter.is_super_admin():
            await self._log_security_event(
                "Unauthorized super admin creation attempt",
                granter_id,
                target_user_id,
                new_role,
            )
            return False

        # Log legitimate role assignment
        await self._record_role_assignment(
            granter_id, target_user_id, new_role, tenant_id
        )

        return True

    async def validate_permission_grant(
        self,
        granter_id: UUID,
        target_user_id: UUID,
        permission: Permission,
        tenant_id: UUID | None = None,
    ) -> bool:
        """Validate that a permission grant is legitimate.

        Args:
            granter_id: User attempting to grant permission
            target_user_id: User receiving permission
            permission: Permission being granted
            tenant_id: Tenant context

        Returns:
            True if permission grant is valid
        """
        granter = await self._get_user_safely(granter_id)

        if not granter:
            return False

        # Cannot grant permissions you don't have
        granter_permissions = await self._get_user_permissions(granter, tenant_id)
        if permission not in granter_permissions:
            await self._log_security_event(
                f"Attempted to grant permission '{permission.name}' without having it",
                granter_id,
                target_user_id,
                permission.name,
            )
            return False

        # Super admins and tenant admins can grant most permissions
        if granter.is_super_admin():
            return True

        if tenant_id and await self._is_tenant_admin(granter, tenant_id):
            # Tenant admins cannot grant platform-level permissions
            platform_permissions = {"platform.manage", "tenant.create", "tenant.delete"}
            if permission.name in platform_permissions:
                await self._log_security_event(
                    f"Tenant admin attempted to grant platform permission '{permission.name}'",
                    granter_id,
                    target_user_id,
                    permission.name,
                )
                return False
            return True

        return False

    async def detect_privilege_escalation_patterns(
        self, user_id: UUID, action: str, target_resource: str | None = None
    ) -> bool:
        """Detect patterns that might indicate privilege escalation attempts.

        Args:
            user_id: User performing the action
            action: Action being performed
            target_resource: Resource being accessed

        Returns:
            True if suspicious pattern detected
        """
        # Pattern 1: Repeated failed privilege attempts
        if action in ["role_assignment", "permission_grant", "admin_action"]:
            recent_failures = self.failed_privilege_attempts.get(user_id, [])
            cutoff = datetime.utcnow() - timedelta(hours=1)
            recent_failures = [ts for ts in recent_failures if ts > cutoff]

            if len(recent_failures) >= 5:  # 5 failed attempts in 1 hour
                await self._log_security_event(
                    "Repeated privilege escalation attempts detected",
                    user_id,
                    None,
                    action,
                )
                return True

        # Pattern 2: Rapid role assignments (potential account takeover)
        if action == "role_assignment":
            assignments = self.role_assignment_history.get(user_id, [])
            recent_assignments = [
                a
                for a in assignments
                if a["timestamp"] > datetime.utcnow() - timedelta(minutes=5)
            ]

            if len(recent_assignments) >= 3:  # 3 assignments in 5 minutes
                await self._log_security_event(
                    "Rapid role assignment pattern detected", user_id, None, action
                )
                return True

        return False

    async def enforce_principle_of_least_privilege(
        self,
        user_id: UUID,
        requested_permissions: list[Permission],
        justification: str | None = None,
    ) -> list[Permission]:
        """Enforce principle of least privilege by filtering excessive permissions.

        Args:
            user_id: User requesting permissions
            requested_permissions: Permissions being requested
            justification: Business justification for permissions

        Returns:
            Filtered list of permissions that follow least privilege principle
        """
        user = await self._get_user_safely(user_id)
        if not user:
            return []

        # Get user's current role-based permissions
        current_permissions = set()
        for tenant_role in user.tenant_roles:
            current_permissions.update(tenant_role.permissions)

        # Filter out permissions that are already granted via roles
        additional_needed = [
            perm for perm in requested_permissions if perm not in current_permissions
        ]

        # Apply least privilege filtering
        filtered_permissions = []

        for permission in additional_needed:
            # High-risk permissions require explicit approval
            high_risk_permissions = {
                "user.delete",
                "tenant.delete",
                "platform.manage",
                "billing.manage",
                "security.manage",
            }

            if permission.name in high_risk_permissions:
                await self._log_security_event(
                    f"High-risk permission '{permission.name}' requested",
                    user_id,
                    None,
                    permission.name,
                    additional_data={"justification": justification},
                )
                # Require manual approval for high-risk permissions
                continue

            # Allow lower-risk permissions
            filtered_permissions.append(permission)

        if len(filtered_permissions) < len(additional_needed):
            await self._log_security_event(
                f"Filtered {len(additional_needed) - len(filtered_permissions)} excessive permissions",
                user_id,
                None,
                "permission_filtering",
            )

        return filtered_permissions

    async def _get_user_safely(self, user_id: UUID) -> User | None:
        """Safely get user with error handling."""
        try:
            # This would typically go through the RBAC service
            return self.rbac_service.users.get(user_id)
        except Exception as e:
            self.logger.error(f"Error getting user {user_id}: {e}")
            return None

    async def _can_grant_role(
        self, granter: User, role: UserRole, tenant_id: UUID | None
    ) -> bool:
        """Check if user can grant a specific role."""
        return PermissionMatrix.can_role_grant_permission(
            granter.get_tenant_role(tenant_id).role
            if tenant_id
            else UserRole.SUPER_ADMIN,
            Permission(
                name=f"role.{role.value}",
                resource="role",
                action="grant",
                description="",
            ),
        )

    async def _has_sufficient_privileges(
        self, granter: User, role: UserRole, tenant_id: UUID | None
    ) -> bool:
        """Check if granter has sufficient privileges to grant role."""
        hierarchy = PermissionMatrix.get_permission_hierarchy()

        if granter.is_super_admin():
            return True

        if tenant_id:
            granter_role = granter.get_tenant_role(tenant_id)
            if not granter_role:
                return False

            granter_level = hierarchy.get(granter_role.role, 0)
            target_level = hierarchy.get(role, 0)

            # Can only grant roles at or below your level
            return granter_level >= target_level

        return False

    async def _check_role_assignment_rate_limit(self, granter_id: UUID) -> bool:
        """Check rate limiting for role assignments."""
        assignments = self.role_assignment_history.get(granter_id, [])
        cutoff = datetime.utcnow() - timedelta(minutes=10)
        recent_assignments = [a for a in assignments if a["timestamp"] > cutoff]

        # Allow max 5 role assignments per 10 minutes
        return len(recent_assignments) < 5

    async def _record_role_assignment(
        self,
        granter_id: UUID,
        target_user_id: UUID,
        role: UserRole,
        tenant_id: UUID | None,
    ) -> None:
        """Record a role assignment for audit and rate limiting."""
        if granter_id not in self.role_assignment_history:
            self.role_assignment_history[granter_id] = []

        self.role_assignment_history[granter_id].append(
            {
                "timestamp": datetime.utcnow(),
                "target_user": target_user_id,
                "role": role.value,
                "tenant_id": tenant_id,
            }
        )

        # Keep only last 50 assignments
        self.role_assignment_history[granter_id] = self.role_assignment_history[
            granter_id
        ][-50:]

    async def _get_user_permissions(
        self, user: User, tenant_id: UUID | None
    ) -> set[Permission]:
        """Get effective permissions for user in tenant context."""
        if user.is_super_admin():
            return PermissionMatrix.get_all_permissions()

        if tenant_id:
            tenant_role = user.get_tenant_role(tenant_id)
            if tenant_role:
                return tenant_role.permissions

        # Return union of all permissions across tenants
        all_permissions = set()
        for tenant_role in user.tenant_roles:
            all_permissions.update(tenant_role.permissions)

        return all_permissions

    async def _is_tenant_admin(self, user: User, tenant_id: UUID) -> bool:
        """Check if user is admin in specific tenant."""
        tenant_role = user.get_tenant_role(tenant_id)
        return tenant_role and tenant_role.role in [
            UserRole.TENANT_ADMIN,
            UserRole.SUPER_ADMIN,
        ]

    async def _log_security_event(
        self,
        event_description: str,
        user_id: UUID,
        target_user_id: UUID | None,
        action_details: str,
        additional_data: dict | None = None,
    ) -> None:
        """Log security events for monitoring and compliance."""
        self.logger.warning(
            f"SECURITY EVENT: {event_description} - "
            f"User: {user_id}, Target: {target_user_id}, "
            f"Action: {action_details}, Data: {additional_data}"
        )

        # Record failed attempts for pattern detection
        if (
            "attempt" in event_description.lower()
            or "blocked" in event_description.lower()
        ):
            if user_id not in self.failed_privilege_attempts:
                self.failed_privilege_attempts[user_id] = []

            self.failed_privilege_attempts[user_id].append(datetime.utcnow())

            # Keep only last 24 hours of attempts
            cutoff = datetime.utcnow() - timedelta(hours=24)
            self.failed_privilege_attempts[user_id] = [
                ts for ts in self.failed_privilege_attempts[user_id] if ts > cutoff
            ]


class SecurityResponseHandler:
    """Handles security responses and error formatting."""

    @staticmethod
    def format_authentication_error(reason: str) -> dict[str, str]:
        """Format authentication error response (401)."""
        return {
            "error": "authentication_required",
            "message": "Valid authentication credentials are required",
            "details": reason,
            "status_code": 401,
        }

    @staticmethod
    def format_authorization_error(
        required_permission: str, user_roles: list[str]
    ) -> dict[str, str]:
        """Format authorization error response (403)."""
        return {
            "error": "insufficient_permissions",
            "message": f"Access denied. Required permission: {required_permission}",
            "user_roles": user_roles,
            "status_code": 403,
        }

    @staticmethod
    def format_privilege_escalation_error() -> dict[str, str]:
        """Format privilege escalation attempt response (403)."""
        return {
            "error": "privilege_escalation_blocked",
            "message": "Privilege escalation attempt detected and blocked",
            "status_code": 403,
        }

    @staticmethod
    def format_rate_limit_error(retry_after: int) -> dict[str, str]:
        """Format rate limit error response (429)."""
        return {
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": retry_after,
            "status_code": 429,
        }
