"""Repository factory for creating database-backed repositories."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from monorepo.domain.repositories.user_repository import (
    SessionRepositoryProtocol,
    TenantRepositoryProtocol,
    UserRepositoryProtocol,
)
from monorepo.infrastructure.persistence.database_repositories import Base
from monorepo.infrastructure.persistence.user_repositories import (
    DatabaseSessionRepository,
    DatabaseTenantRepository,
    DatabaseUserRepository,
)


def get_database_url() -> str:
    """Get database URL from environment or default to SQLite."""
    return os.getenv("DATABASE_URL", "sqlite:///./monorepo.db")


class RepositoryFactory:
    """Factory for creating database-backed repositories."""

    def __init__(self, database_url: str | None = None):
        """Initialize with database URL."""
        self.database_url = database_url or get_database_url()
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)

    def get_user_repository(self) -> UserRepositoryProtocol:
        """Get user repository instance."""
        return DatabaseUserRepository(self.SessionLocal)

    def get_tenant_repository(self) -> TenantRepositoryProtocol:
        """Get tenant repository instance."""
        return DatabaseTenantRepository(self.SessionLocal)

    def get_session_repository(self) -> SessionRepositoryProtocol:
        """Get session repository instance."""
        return DatabaseSessionRepository(self.SessionLocal)


# Global factory instance
_repository_factory: RepositoryFactory | None = None


def get_repository_factory() -> RepositoryFactory:
    """Get global repository factory instance."""
    global _repository_factory
    if _repository_factory is None:
        _repository_factory = RepositoryFactory()
    return _repository_factory


def get_user_repository() -> UserRepositoryProtocol:
    """Get user repository instance."""
    return get_repository_factory().get_user_repository()


def get_tenant_repository() -> TenantRepositoryProtocol:
    """Get tenant repository instance."""
    return get_repository_factory().get_tenant_repository()


def get_session_repository() -> SessionRepositoryProtocol:
    """Get session repository instance."""
    return get_repository_factory().get_session_repository()
