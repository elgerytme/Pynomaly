"""
User management application service for multi-tenant operations.
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta

from pynomaly.domain.entities.user import (
    Tenant,
    TenantLimits,
    TenantPlan,
    TenantStatus,
    TenantUsage,
    User,
    UserRole,
    UserSession,
    UserStatus,
    UserTenantRole,
    get_default_permissions,
)
from pynomaly.domain.repositories.user_repository import (
    SessionRepositoryProtocol,
    TenantRepositoryProtocol,
    UserRepositoryProtocol,
)
from pynomaly.infrastructure.services.email_service import get_email_service
from pynomaly.shared.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ResourceLimitError,
    TenantNotFoundError,
    UserNotFoundError,
    ValidationError,
)
from pynomaly.shared.types import TenantId, UserId


class UserManagementService:
    """Service for managing users and multi-tenancy."""

    def __init__(
        self,
        user_repository: UserRepositoryProtocol,
        tenant_repository: TenantRepositoryProtocol,
        session_repository: SessionRepositoryProtocol,
    ):
        self._user_repo = user_repository
        self._tenant_repo = tenant_repository
        self._session_repo = session_repository

    # User Management
    async def create_user(
        self,
        email: str,
        username: str,
        first_name: str,
        last_name: str,
        password: str,
        tenant_id: TenantId | None = None,
        role: UserRole = UserRole.VIEWER,
    ) -> User:
        """Create a new user."""
        # Validate email uniqueness
        existing_user = await self._user_repo.get_user_by_email(email)
        if existing_user:
            raise ValidationError(f"User with email {email} already exists")

        # Validate username uniqueness
        existing_user = await self._user_repo.get_user_by_username(username)
        if existing_user:
            raise ValidationError(f"Username {username} already taken")

        # Check tenant limits if adding to tenant
        if tenant_id:
            tenant = await self._tenant_repo.get_tenant_by_id(tenant_id)
            if not tenant:
                raise TenantNotFoundError(f"Tenant {tenant_id} not found")

            if not tenant.is_within_limits()["users"]:
                raise ResourceLimitError("Tenant has reached maximum user limit")

        # Create user
        user = User(
            id=UserId(str(uuid.uuid4())),
            email=email,
            username=username,
            first_name=first_name,
            last_name=last_name,
            status=UserStatus.PENDING_VERIFICATION,
            password_hash=self._hash_password(password),
        )

        # Add to tenant if specified
        if tenant_id:
            tenant_role = UserTenantRole(
                user_id=user.id,
                tenant_id=tenant_id,
                role=role,
                permissions=get_default_permissions(role),
            )
            user.tenant_roles.append(tenant_role)

        created_user = await self._user_repo.create_user(user)

        # Update tenant usage
        if tenant_id:
            await self._tenant_repo.update_tenant_usage(
                tenant_id, {"users_count": "+1"}
            )

        return created_user

    async def authenticate_user(self, email: str, password: str) -> tuple[User, str]:
        """Authenticate user and return user with session token."""
        user = await self._user_repo.get_user_by_email(email)
        if not user:
            raise AuthenticationError("Invalid email or password")

        if user.status != UserStatus.ACTIVE:
            raise AuthenticationError("User account is not active")

        if not self._verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password")

        # Create session
        session = UserSession(
            id=str(uuid.uuid4()),
            user_id=user.id,
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )

        await self._session_repo.create_session(session)

        # Update last login
        user.last_login_at = datetime.utcnow()
        await self._user_repo.update_user(user)

        return user, session.id

    async def get_user_by_session(self, session_id: str) -> User | None:
        """Get user by session ID."""
        session = await self._session_repo.get_session_by_id(session_id)
        if (
            not session
            or not session.is_active
            or session.expires_at < datetime.utcnow()
        ):
            return None

        # Update last activity
        session.last_activity = datetime.utcnow()
        await self._session_repo.update_session(session)

        return await self._user_repo.get_user_by_id(session.user_id)

    async def logout_user(self, session_id: str) -> bool:
        """Logout user by deactivating session."""
        return await self._session_repo.delete_session(session_id)

    async def invite_user_to_tenant(
        self,
        inviter_id: UserId,
        tenant_id: TenantId,
        email: str,
        role: UserRole = UserRole.VIEWER,
    ) -> User:
        """Invite a new user to a tenant."""
        # Check if inviter has permission
        inviter = await self._user_repo.get_user_by_id(inviter_id)
        if not inviter:
            raise UserNotFoundError("Inviter not found")

        if not (
            inviter.is_super_admin()
            or inviter.has_role_in_tenant(tenant_id, UserRole.TENANT_ADMIN)
        ):
            raise AuthorizationError("Insufficient permissions to invite users")

        # Generate temporary password
        temp_password = secrets.token_urlsafe(12)

        # Create user with pending status
        user = await self.create_user(
            email=email,
            username=email.split("@")[0] + str(int(datetime.utcnow().timestamp())),
            first_name="",
            last_name="",
            password=temp_password,
            tenant_id=tenant_id,
            role=role,
        )

        # Send invitation email
        email_service = get_email_service()
        if email_service:
            try:
                # Generate invitation token (could be improved to use JWT tokens)
                invitation_token = secrets.token_urlsafe(32)

                # Store invitation token with user ID for later verification
                # This would typically be stored in a database table
                # For now, we'll just send the email

                await email_service.send_user_invitation_email(
                    email=email,
                    invitation_token=invitation_token,
                    inviter_name=self.current_user.name if self.current_user else "Administrator",
                    organization_name=tenant.name if tenant else "Pynomaly"
                )
            except Exception as e:
                # Log error but don't fail user creation
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to send invitation email to {email}: {e}")
        else:
            # Email service not configured - log warning
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Email service not configured - invitation email not sent to {email}")

        return user

    # Tenant Management
    async def create_tenant(
        self,
        name: str,
        domain: str,
        plan: TenantPlan = TenantPlan.FREE,
        admin_email: str = "",
        admin_password: str = "",
    ) -> tuple[Tenant, User | None]:
        """Create a new tenant with optional admin user."""
        # Validate domain uniqueness
        existing_tenant = await self._tenant_repo.get_tenant_by_domain(domain)
        if existing_tenant:
            raise ValidationError(f"Domain {domain} already exists")

        # Set limits based on plan
        limits = self._get_plan_limits(plan)

        # Create tenant
        tenant = Tenant(
            id=TenantId(str(uuid.uuid4())),
            name=name,
            domain=domain,
            plan=plan,
            status=TenantStatus.ACTIVE
            if plan != TenantPlan.FREE
            else TenantStatus.TRIAL,
            limits=limits,
            usage=TenantUsage(),
        )

        created_tenant = await self._tenant_repo.create_tenant(tenant)

        # Create admin user if provided
        admin_user = None
        if admin_email and admin_password:
            admin_user = await self.create_user(
                email=admin_email,
                username=admin_email.split("@")[0],
                first_name="Admin",
                last_name="User",
                password=admin_password,
                tenant_id=created_tenant.id,
                role=UserRole.TENANT_ADMIN,
            )

            # Set user as active immediately
            admin_user.status = UserStatus.ACTIVE
            admin_user.email_verified_at = datetime.utcnow()
            await self._user_repo.update_user(admin_user)

        return created_tenant, admin_user

    async def get_tenant_usage_report(self, tenant_id: TenantId) -> dict[str, any]:
        """Get comprehensive usage report for tenant."""
        tenant = await self._tenant_repo.get_tenant_by_id(tenant_id)
        if not tenant:
            raise TenantNotFoundError("Tenant not found")

        limits_check = tenant.is_within_limits()

        return {
            "tenant_id": tenant_id,
            "tenant_name": tenant.name,
            "plan": tenant.plan.value,
            "status": tenant.status.value,
            "usage": {
                "users": {
                    "current": tenant.usage.users_count,
                    "limit": tenant.limits.max_users,
                    "percentage": tenant.get_limit_usage_percentage("users"),
                    "within_limit": limits_check["users"],
                },
                "datasets": {
                    "current": tenant.usage.datasets_count,
                    "limit": tenant.limits.max_datasets,
                    "percentage": tenant.get_limit_usage_percentage("datasets"),
                    "within_limit": limits_check["datasets"],
                },
                "models": {
                    "current": tenant.usage.models_count,
                    "limit": tenant.limits.max_models,
                    "percentage": tenant.get_limit_usage_percentage("models"),
                    "within_limit": limits_check["models"],
                },
                "detections_this_month": {
                    "current": tenant.usage.detections_this_month,
                    "limit": tenant.limits.max_detections_per_month,
                    "percentage": tenant.get_limit_usage_percentage("detections"),
                    "within_limit": limits_check["detections"],
                },
                "storage_gb": {
                    "current": tenant.usage.storage_used_gb,
                    "limit": tenant.limits.max_storage_gb,
                    "percentage": tenant.get_limit_usage_percentage("storage"),
                    "within_limit": limits_check["storage"],
                },
            },
            "last_updated": tenant.usage.last_updated.isoformat(),
        }

    async def check_tenant_limits(self, tenant_id: TenantId, resource: str) -> bool:
        """Check if tenant can use more of a specific resource."""
        tenant = await self._tenant_repo.get_tenant_by_id(tenant_id)
        if not tenant:
            raise TenantNotFoundError("Tenant not found")

        limits_check = tenant.is_within_limits()
        return limits_check.get(resource, False)

    async def update_tenant_usage(
        self, tenant_id: TenantId, resource: str, increment: int = 1
    ) -> bool:
        """Update tenant usage for a specific resource."""
        usage_updates = {f"{resource}_count": f"+{increment}"}
        return await self._tenant_repo.update_tenant_usage(tenant_id, usage_updates)

    # Permission Management
    async def update_user(self, user_id: UserId, update_data: dict[str, any]) -> User:
        """Update user information."""
        user = await self._user_repo.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        # Update fields
        if "first_name" in update_data:
            user.first_name = update_data["first_name"]
        if "last_name" in update_data:
            user.last_name = update_data["last_name"]
        if "status" in update_data:
            user.status = update_data["status"]

        user.updated_at = datetime.utcnow()

        return await self._user_repo.update_user(user)

    async def delete_user(self, user_id: UserId) -> bool:
        """Delete a user."""
        user = await self._user_repo.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        return await self._user_repo.delete_user(user_id)

    async def reset_password(self, user_id: UserId, new_password: str) -> bool:
        """Reset user password."""
        user = await self._user_repo.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        # Hash the new password
        user.password_hash = self._hash_password(new_password)
        user.updated_at = datetime.utcnow()

        updated_user = await self._user_repo.update_user(user)
        return updated_user is not None

    async def toggle_user_status(self, user_id: UserId, new_status: UserStatus) -> User:
        """Toggle user status (activate/deactivate)."""
        user = await self._user_repo.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        user.status = new_status
        user.updated_at = datetime.utcnow()

        return await self._user_repo.update_user(user)

    async def list_users(
        self,
        tenant_id: TenantId | None = None,
        status: UserStatus | None = None,
        role: UserRole | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[User]:
        """List users with optional filters."""
        return await self._user_repo.list_users(
            tenant_id=tenant_id, status=status, role=role, limit=limit, offset=offset
        )

    async def check_user_permission(
        self, user_id: UserId, tenant_id: TenantId, resource: str, action: str
    ) -> bool:
        """Check if user has permission for specific action."""
        user = await self._user_repo.get_user_by_id(user_id)
        if not user:
            return False

        # Super admins have all permissions
        if user.is_super_admin():
            return True

        # Check tenant-specific permissions
        tenant_role = user.get_tenant_role(tenant_id)
        if not tenant_role:
            return False

        # Check if user has specific permission
        for permission in tenant_role.permissions:
            if permission.resource == resource and permission.action == action:
                return True

        return False

    # Utility methods
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt, password_hash = stored_hash.split(":")
            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return password_hash == computed_hash
        except ValueError:
            return False

    def _get_plan_limits(self, plan: TenantPlan) -> TenantLimits:
        """Get resource limits for a specific plan."""
        plan_limits = {
            TenantPlan.FREE: TenantLimits(
                max_users=3,
                max_datasets=10,
                max_models=5,
                max_detections_per_month=1000,
                max_storage_gb=1,
                max_api_calls_per_minute=50,
                max_concurrent_detections=1,
            ),
            TenantPlan.STARTER: TenantLimits(
                max_users=10,
                max_datasets=50,
                max_models=20,
                max_detections_per_month=10000,
                max_storage_gb=10,
                max_api_calls_per_minute=200,
                max_concurrent_detections=3,
            ),
            TenantPlan.PROFESSIONAL: TenantLimits(
                max_users=50,
                max_datasets=200,
                max_models=100,
                max_detections_per_month=100000,
                max_storage_gb=100,
                max_api_calls_per_minute=1000,
                max_concurrent_detections=10,
            ),
            TenantPlan.ENTERPRISE: TenantLimits(
                max_users=500,
                max_datasets=1000,
                max_models=500,
                max_detections_per_month=1000000,
                max_storage_gb=1000,
                max_api_calls_per_minute=5000,
                max_concurrent_detections=50,
            ),
        }

        return plan_limits.get(plan, plan_limits[TenantPlan.FREE])
