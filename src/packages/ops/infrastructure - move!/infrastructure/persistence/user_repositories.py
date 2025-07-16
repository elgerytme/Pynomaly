"""Database repository implementations for user management."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

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
from pynomaly.infrastructure.persistence.database_repositories import (
    TenantModel,
    UserModel,
    UserRoleModel,
)
from pynomaly.shared.exceptions import (
    TenantNotFoundError,
    UserNotFoundError,
    ValidationError,
)
from pynomaly.shared.types import TenantId, UserId


class UserSessionModel:
    """SQLAlchemy model for User Session entity."""

    __tablename__ = "user_sessions"

    # Note: This would need to be added to the database_repositories.py Base registry
    # For now, we'll use a simplified in-memory approach for sessions


class DatabaseUserRepository(UserRepositoryProtocol):
    """Database-backed user repository implementation."""

    def __init__(self, session_factory):
        """Initialize with SQLAlchemy session factory."""
        self.session_factory = session_factory

    async def create_user(self, user: User) -> User:
        """Create a new user in the database."""
        with self.session_factory() as session:
            try:
                # Create the user model
                user_model = UserModel(
                    id=user.id,
                    email=user.email,
                    username=user.username,
                    first_name=user.first_name,
                    last_name=user.last_name,
                    status=user.status.value,
                    password_hash=user.password_hash,
                    settings=user.settings,
                    created_at=user.created_at,
                    updated_at=user.updated_at,
                    last_login_at=user.last_login_at,
                    email_verified_at=user.email_verified_at,
                )
                session.add(user_model)

                # Create user-tenant-role associations
                for tenant_role in user.tenant_roles:
                    role_model = UserRoleModel(
                        user_id=user.id,
                        tenant_id=tenant_role.tenant_id,
                        role_id=str(
                            tenant_role.role.value
                        ),  # Store role as string for now
                        permissions={
                            "permissions": [
                                {
                                    "name": p.name,
                                    "resource": p.resource,
                                    "action": p.action,
                                    "description": p.description,
                                }
                                for p in tenant_role.permissions
                            ]
                        },
                        granted_at=tenant_role.granted_at,
                        granted_by=tenant_role.granted_by,
                        expires_at=tenant_role.expires_at,
                    )
                    session.add(role_model)

                session.commit()
                return user

            except IntegrityError as e:
                session.rollback()
                if "email" in str(e).lower():
                    raise ValidationError(
                        f"User with email {user.email} already exists"
                    )
                if "username" in str(e).lower():
                    raise ValidationError(f"Username {user.username} already exists")
                raise ValidationError(f"Failed to create user: {e}")

    async def get_user_by_id(self, user_id: UserId) -> User | None:
        """Get user by ID."""
        with self.session_factory() as session:
            user_model = session.query(UserModel).filter_by(id=user_id).first()
            if not user_model:
                return None

            return await self._model_to_entity(session, user_model)

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email address."""
        with self.session_factory() as session:
            user_model = session.query(UserModel).filter_by(email=email).first()
            if not user_model:
                return None

            return await self._model_to_entity(session, user_model)

    async def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        with self.session_factory() as session:
            user_model = session.query(UserModel).filter_by(username=username).first()
            if not user_model:
                return None

            return await self._model_to_entity(session, user_model)

    async def update_user(self, user: User) -> User:
        """Update existing user."""
        with self.session_factory() as session:
            user_model = session.query(UserModel).filter_by(id=user.id).first()
            if not user_model:
                raise UserNotFoundError(f"User {user.id} not found")

            # Update user fields
            user_model.email = user.email
            user_model.username = user.username
            user_model.first_name = user.first_name
            user_model.last_name = user.last_name
            user_model.status = user.status.value
            user_model.password_hash = user.password_hash
            user_model.settings = user.settings
            user_model.updated_at = user.updated_at
            user_model.last_login_at = user.last_login_at
            user_model.email_verified_at = user.email_verified_at

            # Update tenant roles - remove existing and add new ones
            session.query(UserRoleModel).filter_by(user_id=user.id).delete()

            for tenant_role in user.tenant_roles:
                role_model = UserRoleModel(
                    user_id=user.id,
                    tenant_id=tenant_role.tenant_id,
                    role_id=str(tenant_role.role.value),
                    permissions={
                        "permissions": [
                            {
                                "name": p.name,
                                "resource": p.resource,
                                "action": p.action,
                                "description": p.description,
                            }
                            for p in tenant_role.permissions
                        ]
                    },
                    granted_at=tenant_role.granted_at,
                    granted_by=tenant_role.granted_by,
                    expires_at=tenant_role.expires_at,
                )
                session.add(role_model)

            session.commit()
            return user

    async def delete_user(self, user_id: UserId) -> bool:
        """Delete user and all associated data."""
        with self.session_factory() as session:
            # Delete user roles first (foreign key constraint)
            session.query(UserRoleModel).filter_by(user_id=user_id).delete()

            # Delete user
            deleted_count = session.query(UserModel).filter_by(id=user_id).delete()
            session.commit()

            return deleted_count > 0

    async def get_users_by_tenant(self, tenant_id: TenantId) -> list[User]:
        """Get all users for a specific tenant."""
        with self.session_factory() as session:
            user_ids = (
                session.query(UserRoleModel.user_id)
                .filter_by(tenant_id=tenant_id)
                .distinct()
                .all()
            )

            users = []
            for (user_id,) in user_ids:
                user_model = session.query(UserModel).filter_by(id=user_id).first()
                if user_model:
                    user = await self._model_to_entity(session, user_model)
                    users.append(user)

            return users

    async def add_user_to_tenant(
        self, user_id: UserId, tenant_id: TenantId, role: UserRole
    ) -> UserTenantRole:
        """Add user to tenant with specified role."""
        with self.session_factory() as session:
            # Check if user exists
            user_model = session.query(UserModel).filter_by(id=user_id).first()
            if not user_model:
                raise UserNotFoundError(f"User {user_id} not found")

            # Check if already exists
            existing = (
                session.query(UserRoleModel)
                .filter_by(user_id=user_id, tenant_id=tenant_id)
                .first()
            )
            if existing:
                raise ValidationError(
                    f"User {user_id} already belongs to tenant {tenant_id}"
                )

            tenant_role = UserTenantRole(
                user_id=user_id,
                tenant_id=tenant_id,
                role=role,
                permissions=get_default_permissions(role),
                granted_at=datetime.utcnow(),
            )

            role_model = UserRoleModel(
                user_id=user_id,
                tenant_id=tenant_id,
                role_id=str(role.value),
                permissions={
                    "permissions": [
                        {
                            "name": p.name,
                            "resource": p.resource,
                            "action": p.action,
                            "description": p.description,
                        }
                        for p in tenant_role.permissions
                    ]
                },
                granted_at=tenant_role.granted_at,
                granted_by=tenant_role.granted_by,
                expires_at=tenant_role.expires_at,
            )
            session.add(role_model)
            session.commit()

            return tenant_role

    async def remove_user_from_tenant(
        self, user_id: UserId, tenant_id: TenantId
    ) -> bool:
        """Remove user from tenant."""
        with self.session_factory() as session:
            deleted_count = (
                session.query(UserRoleModel)
                .filter_by(user_id=user_id, tenant_id=tenant_id)
                .delete()
            )
            session.commit()
            return deleted_count > 0

    async def update_user_role_in_tenant(
        self, user_id: UserId, tenant_id: TenantId, role: UserRole
    ) -> UserTenantRole:
        """Update user's role in tenant."""
        with self.session_factory() as session:
            role_model = (
                session.query(UserRoleModel)
                .filter_by(user_id=user_id, tenant_id=tenant_id)
                .first()
            )
            if not role_model:
                raise ValidationError(f"User {user_id} not found in tenant {tenant_id}")

            # Update role and permissions
            role_model.role_id = str(role.value)
            role_model.permissions = {
                "permissions": [
                    {
                        "name": p.name,
                        "resource": p.resource,
                        "action": p.action,
                        "description": p.description,
                    }
                    for p in get_default_permissions(role)
                ]
            }
            session.commit()

            return UserTenantRole(
                user_id=user_id,
                tenant_id=tenant_id,
                role=role,
                permissions=get_default_permissions(role),
                granted_at=role_model.granted_at,
                granted_by=role_model.granted_by,
                expires_at=role_model.expires_at,
            )

    async def list_users(
        self,
        tenant_id: TenantId | None = None,
        status: UserStatus | None = None,
        role: UserRole | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[User]:
        """List users with optional filters."""
        with self.session_factory() as session:
            query = session.query(UserModel)

            # Apply filters
            if status:
                query = query.filter(UserModel.status == status.value)

            if tenant_id or role:
                # Join with user roles table for tenant/role filtering
                query = query.join(UserRoleModel, UserModel.id == UserRoleModel.user_id)

                if tenant_id:
                    query = query.filter(UserRoleModel.tenant_id == tenant_id)

                if role:
                    query = query.filter(UserRoleModel.role_id == str(role.value))

            # Apply pagination
            query = query.offset(offset).limit(limit)
            user_models = query.all()

            users = []
            for user_model in user_models:
                user = await self._model_to_entity(session, user_model)
                users.append(user)

            return users

    async def _model_to_entity(self, session: Session, user_model: UserModel) -> User:
        """Convert database model to domain entity."""
        # Get tenant roles for this user
        role_models = (
            session.query(UserRoleModel).filter_by(user_id=user_model.id).all()
        )

        tenant_roles = []
        for role_model in role_models:
            # Convert stored permissions back to Permission objects
            permissions = set()
            if role_model.permissions and "permissions" in role_model.permissions:
                from pynomaly.domain.entities.user import Permission

                for perm_data in role_model.permissions["permissions"]:
                    permission = Permission(
                        name=perm_data["name"],
                        resource=perm_data["resource"],
                        action=perm_data["action"],
                        description=perm_data.get("description", ""),
                    )
                    permissions.add(permission)

            tenant_role = UserTenantRole(
                user_id=user_model.id,
                tenant_id=role_model.tenant_id,
                role=UserRole(role_model.role_id),
                permissions=permissions,
                granted_at=role_model.granted_at,
                granted_by=role_model.granted_by,
                expires_at=role_model.expires_at,
            )
            tenant_roles.append(tenant_role)

        return User(
            id=user_model.id,
            email=user_model.email,
            username=user_model.username,
            first_name=user_model.first_name,
            last_name=user_model.last_name,
            status=UserStatus(user_model.status),
            tenant_roles=tenant_roles,
            created_at=user_model.created_at,
            updated_at=user_model.updated_at,
            last_login_at=user_model.last_login_at,
            email_verified_at=user_model.email_verified_at,
            password_hash=user_model.password_hash,
            settings=user_model.settings or {},
        )


class DatabaseTenantRepository(TenantRepositoryProtocol):
    """Database-backed tenant repository implementation."""

    def __init__(self, session_factory):
        """Initialize with SQLAlchemy session factory."""
        self.session_factory = session_factory

    async def create_tenant(self, tenant: Tenant) -> Tenant:
        """Create a new tenant in the database."""
        with self.session_factory() as session:
            try:
                tenant_model = TenantModel(
                    id=tenant.id,
                    name=tenant.name,
                    domain=tenant.domain,
                    plan=tenant.plan.value,
                    status=tenant.status.value,
                    limits=self._limits_to_dict(tenant.limits),
                    usage=self._usage_to_dict(tenant.usage),
                    created_at=tenant.created_at,
                    updated_at=tenant.updated_at,
                    expires_at=tenant.expires_at,
                    contact_email=tenant.contact_email,
                    billing_email=tenant.billing_email,
                    settings=tenant.settings,
                )
                session.add(tenant_model)
                session.commit()
                return tenant

            except IntegrityError as e:
                session.rollback()
                if "domain" in str(e).lower():
                    raise ValidationError(
                        f"Tenant with domain {tenant.domain} already exists"
                    )
                raise ValidationError(f"Failed to create tenant: {e}")

    async def get_tenant_by_id(self, tenant_id: TenantId) -> Tenant | None:
        """Get tenant by ID."""
        with self.session_factory() as session:
            tenant_model = session.query(TenantModel).filter_by(id=tenant_id).first()
            if not tenant_model:
                return None

            return self._model_to_entity(tenant_model)

    async def get_tenant_by_domain(self, domain: str) -> Tenant | None:
        """Get tenant by domain."""
        with self.session_factory() as session:
            tenant_model = session.query(TenantModel).filter_by(domain=domain).first()
            if not tenant_model:
                return None

            return self._model_to_entity(tenant_model)

    async def update_tenant(self, tenant: Tenant) -> Tenant:
        """Update existing tenant."""
        with self.session_factory() as session:
            tenant_model = session.query(TenantModel).filter_by(id=tenant.id).first()
            if not tenant_model:
                raise TenantNotFoundError(f"Tenant {tenant.id} not found")

            tenant_model.name = tenant.name
            tenant_model.domain = tenant.domain
            tenant_model.plan = tenant.plan.value
            tenant_model.status = tenant.status.value
            tenant_model.limits = self._limits_to_dict(tenant.limits)
            tenant_model.usage = self._usage_to_dict(tenant.usage)
            tenant_model.updated_at = tenant.updated_at
            tenant_model.expires_at = tenant.expires_at
            tenant_model.contact_email = tenant.contact_email
            tenant_model.billing_email = tenant.billing_email
            tenant_model.settings = tenant.settings

            session.commit()
            return tenant

    async def delete_tenant(self, tenant_id: TenantId) -> bool:
        """Delete tenant and all associated data."""
        with self.session_factory() as session:
            # Delete tenant roles first
            session.query(UserRoleModel).filter_by(tenant_id=tenant_id).delete()

            # Delete tenant
            deleted_count = session.query(TenantModel).filter_by(id=tenant_id).delete()
            session.commit()

            return deleted_count > 0

    async def list_tenants(self, limit: int = 100, offset: int = 0) -> list[Tenant]:
        """List all tenants with pagination."""
        with self.session_factory() as session:
            tenant_models = session.query(TenantModel).offset(offset).limit(limit).all()

            return [self._model_to_entity(model) for model in tenant_models]

    async def update_tenant_usage(
        self, tenant_id: TenantId, usage_updates: dict
    ) -> bool:
        """Update tenant usage statistics."""
        with self.session_factory() as session:
            tenant_model = session.query(TenantModel).filter_by(id=tenant_id).first()
            if not tenant_model:
                return False

            # Parse current usage
            current_usage = tenant_model.usage or {}

            # Apply updates
            for key, value in usage_updates.items():
                if isinstance(value, str) and value.startswith("+"):
                    # Increment operation
                    increment = int(value[1:])
                    current_usage[key] = current_usage.get(key, 0) + increment
                elif isinstance(value, str) and value.startswith("-"):
                    # Decrement operation
                    decrement = int(value[1:])
                    current_usage[key] = max(0, current_usage.get(key, 0) - decrement)
                else:
                    # Direct assignment
                    current_usage[key] = value

            # Update last_updated timestamp
            current_usage["last_updated"] = datetime.utcnow().isoformat()

            tenant_model.usage = current_usage
            session.commit()
            return True

    def _limits_to_dict(self, limits: TenantLimits) -> dict:
        """Convert TenantLimits to dictionary for storage."""
        return {
            "max_users": limits.max_users,
            "max_datasets": limits.max_datasets,
            "max_models": limits.max_models,
            "max_detections_per_month": limits.max_detections_per_month,
            "max_storage_gb": limits.max_storage_gb,
            "max_api_calls_per_minute": limits.max_api_calls_per_minute,
            "max_concurrent_detections": limits.max_concurrent_detections,
        }

    def _usage_to_dict(self, usage: TenantUsage) -> dict:
        """Convert TenantUsage to dictionary for storage."""
        return {
            "users_count": usage.users_count,
            "datasets_count": usage.datasets_count,
            "models_count": usage.models_count,
            "detections_this_month": usage.detections_this_month,
            "storage_used_gb": usage.storage_used_gb,
            "api_calls_this_minute": usage.api_calls_this_minute,
            "concurrent_detections": usage.concurrent_detections,
            "last_updated": usage.last_updated.isoformat(),
        }

    def _model_to_entity(self, tenant_model: TenantModel) -> Tenant:
        """Convert database model to domain entity."""
        # Convert limits dictionary back to TenantLimits
        limits_data = tenant_model.limits or {}
        limits = TenantLimits(
            max_users=limits_data.get("max_users", 10),
            max_datasets=limits_data.get("max_datasets", 100),
            max_models=limits_data.get("max_models", 50),
            max_detections_per_month=limits_data.get("max_detections_per_month", 10000),
            max_storage_gb=limits_data.get("max_storage_gb", 10),
            max_api_calls_per_minute=limits_data.get("max_api_calls_per_minute", 100),
            max_concurrent_detections=limits_data.get("max_concurrent_detections", 5),
        )

        # Convert usage dictionary back to TenantUsage
        usage_data = tenant_model.usage or {}
        usage = TenantUsage(
            users_count=usage_data.get("users_count", 0),
            datasets_count=usage_data.get("datasets_count", 0),
            models_count=usage_data.get("models_count", 0),
            detections_this_month=usage_data.get("detections_this_month", 0),
            storage_used_gb=usage_data.get("storage_used_gb", 0.0),
            api_calls_this_minute=usage_data.get("api_calls_this_minute", 0),
            concurrent_detections=usage_data.get("concurrent_detections", 0),
            last_updated=datetime.fromisoformat(
                usage_data.get("last_updated", datetime.utcnow().isoformat())
            ),
        )

        return Tenant(
            id=tenant_model.id,
            name=tenant_model.name,
            domain=tenant_model.domain,
            plan=TenantPlan(tenant_model.plan),
            status=TenantStatus(tenant_model.status),
            limits=limits,
            usage=usage,
            created_at=tenant_model.created_at,
            updated_at=tenant_model.updated_at,
            expires_at=tenant_model.expires_at,
            contact_email=tenant_model.contact_email or "",
            billing_email=tenant_model.billing_email or "",
            settings=tenant_model.settings or {},
        )


class DatabaseSessionRepository(SessionRepositoryProtocol):
    """Database-backed session repository implementation."""

    def __init__(self, session_factory):
        """Initialize with SQLAlchemy session factory."""
        self.session_factory = session_factory
        # For now, use in-memory storage for sessions
        # In production, you'd want to store sessions in Redis or database
        self._sessions = {}

    async def create_session(self, session: UserSession) -> UserSession:
        """Create a new user session."""
        # Store in memory for now
        self._sessions[session.id] = session
        return session

    async def get_session_by_id(self, session_id: str) -> UserSession | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    async def update_session(self, session: UserSession) -> UserSession:
        """Update existing session."""
        if session.id in self._sessions:
            self._sessions[session.id] = session
            return session
        raise ValueError(f"Session {session.id} not found")

    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    async def get_active_sessions_for_user(self, user_id: UserId) -> list[UserSession]:
        """Get all active sessions for a user."""
        active_sessions = []
        for session in self._sessions.values():
            if (
                session.user_id == user_id
                and session.is_active
                and session.expires_at > datetime.utcnow()
            ):
                active_sessions.append(session)
        return active_sessions

    async def delete_all_sessions_for_user(self, user_id: UserId) -> bool:
        """Delete all sessions for a user."""
        sessions_to_delete = [
            session_id
            for session_id, session in self._sessions.items()
            if session.user_id == user_id
        ]

        for session_id in sessions_to_delete:
            del self._sessions[session_id]

        return len(sessions_to_delete) > 0
