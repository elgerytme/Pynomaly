"""
Repository interface for user management operations.
"""

from abc import ABC, abstractmethod

from monorepo.domain.entities.user import (
    Tenant,
    User,
    UserRole,
    UserSession,
    UserStatus,
    UserTenantRole,
)
from monorepo.shared.types import TenantId, UserId


class UserRepositoryProtocol(ABC):
    """Repository interface for user operations."""

    @abstractmethod
    async def create_user(self, user: User) -> User:
        """Create a new user."""
        pass

    @abstractmethod
    async def get_user_by_id(self, user_id: UserId) -> User | None:
        """Get user by ID."""
        pass

    @abstractmethod
    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        pass

    @abstractmethod
    async def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        pass

    @abstractmethod
    async def update_user(self, user: User) -> User:
        """Update existing user."""
        pass

    @abstractmethod
    async def delete_user(self, user_id: UserId) -> bool:
        """Delete user."""
        pass

    @abstractmethod
    async def get_users_by_tenant(self, tenant_id: TenantId) -> list[User]:
        """Get all users for a tenant."""
        pass

    @abstractmethod
    async def add_user_to_tenant(
        self, user_id: UserId, tenant_id: TenantId, role: UserRole
    ) -> UserTenantRole:
        """Add user to tenant with role."""
        pass

    @abstractmethod
    async def remove_user_from_tenant(
        self, user_id: UserId, tenant_id: TenantId
    ) -> bool:
        """Remove user from tenant."""
        pass

    @abstractmethod
    async def update_user_role_in_tenant(
        self, user_id: UserId, tenant_id: TenantId, role: UserRole
    ) -> UserTenantRole:
        """Update user's role in tenant."""
        pass

    @abstractmethod
    async def list_users(
        self,
        tenant_id: TenantId | None = None,
        status: UserStatus | None = None,
        role: UserRole | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[User]:
        """List users with optional filters."""
        pass


class TenantRepositoryProtocol(ABC):
    """Repository interface for tenant operations."""

    @abstractmethod
    async def create_tenant(self, tenant: Tenant) -> Tenant:
        """Create a new tenant."""
        pass

    @abstractmethod
    async def get_tenant_by_id(self, tenant_id: TenantId) -> Tenant | None:
        """Get tenant by ID."""
        pass

    @abstractmethod
    async def get_tenant_by_domain(self, domain: str) -> Tenant | None:
        """Get tenant by domain."""
        pass

    @abstractmethod
    async def update_tenant(self, tenant: Tenant) -> Tenant:
        """Update existing tenant."""
        pass

    @abstractmethod
    async def delete_tenant(self, tenant_id: TenantId) -> bool:
        """Delete tenant."""
        pass

    @abstractmethod
    async def list_tenants(self, limit: int = 100, offset: int = 0) -> list[Tenant]:
        """List all tenants with pagination."""
        pass

    @abstractmethod
    async def update_tenant_usage(
        self, tenant_id: TenantId, usage_updates: dict
    ) -> bool:
        """Update tenant usage statistics."""
        pass


class SessionRepositoryProtocol(ABC):
    """Repository interface for session operations."""

    @abstractmethod
    async def create_session(self, session: UserSession) -> UserSession:
        """Create a new user session."""
        pass

    @abstractmethod
    async def get_session_by_id(self, session_id: str) -> UserSession | None:
        """Get session by ID."""
        pass

    @abstractmethod
    async def update_session(self, session: UserSession) -> UserSession:
        """Update existing session."""
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        pass

    @abstractmethod
    async def get_active_sessions_for_user(self, user_id: UserId) -> list[UserSession]:
        """Get all active sessions for a user."""
        pass

    @abstractmethod
    async def delete_all_sessions_for_user(self, user_id: UserId) -> bool:
        """Delete all sessions for a user."""
        pass
