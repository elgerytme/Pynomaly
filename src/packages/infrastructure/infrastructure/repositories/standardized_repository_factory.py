"""Standardized repository factory for consistent async repository access.

This factory provides a centralized way to create repository instances
that all follow the standardized async RepositoryProtocol pattern.
"""

from __future__ import annotations

from typing import TypeVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pynomaly.infrastructure.persistence.database_repositories import (
    DatabaseDatasetRepository,
    DatabaseDetectionResultRepository,
    DatabaseDetectorRepository,
)
from pynomaly.infrastructure.repositories.in_memory_repositories import (
    InMemoryDatasetRepository,
    InMemoryDetectorRepository,
    InMemoryResultRepository,
)
from pynomaly.shared.protocols.repository_protocol import (
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)

T = TypeVar("T")


class RepositoryType:
    """Enumeration of supported repository types."""

    MEMORY = "memory"
    DATABASE = "database"


class StandardizedRepositoryFactory:
    """Factory for creating standardized async repositories.

    All repositories created by this factory follow the standardized
    async RepositoryProtocol pattern for consistency.
    """

    def __init__(
        self,
        repository_type: str = RepositoryType.MEMORY,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ):
        """Initialize repository factory.

        Args:
            repository_type: Type of repositories to create (memory or database)
            session_factory: Async session factory for database repositories
        """
        self.repository_type = repository_type
        self.session_factory = session_factory

        if repository_type == RepositoryType.DATABASE and not session_factory:
            raise ValueError("session_factory is required for database repository type")

    def create_detector_repository(self) -> DetectorRepositoryProtocol:
        """Create a detector repository.

        Returns:
            Standardized async detector repository
        """
        if self.repository_type == RepositoryType.MEMORY:
            return InMemoryDetectorRepository()
        elif self.repository_type == RepositoryType.DATABASE:
            return DatabaseDetectorRepository(self.session_factory)
        else:
            raise ValueError(f"Unsupported repository type: {self.repository_type}")

    def create_dataset_repository(self) -> DatasetRepositoryProtocol:
        """Create a dataset repository.

        Returns:
            Standardized async dataset repository
        """
        if self.repository_type == RepositoryType.MEMORY:
            return InMemoryDatasetRepository()
        elif self.repository_type == RepositoryType.DATABASE:
            return DatabaseDatasetRepository(self.session_factory)
        else:
            raise ValueError(f"Unsupported repository type: {self.repository_type}")

    def create_detection_result_repository(self) -> DetectionResultRepositoryProtocol:
        """Create a detection result repository.

        Returns:
            Standardized async detection result repository
        """
        if self.repository_type == RepositoryType.MEMORY:
            return InMemoryResultRepository()
        elif self.repository_type == RepositoryType.DATABASE:
            return DatabaseDetectionResultRepository(self.session_factory)
        else:
            raise ValueError(f"Unsupported repository type: {self.repository_type}")

    def create_all_repositories(self) -> dict[str, object]:
        """Create all standard repositories.

        Returns:
            Dictionary mapping repository names to instances
        """
        return {
            "detector": self.create_detector_repository(),
            "dataset": self.create_dataset_repository(),
            "detection_result": self.create_detection_result_repository(),
        }


def get_memory_repositories() -> dict[str, object]:
    """Convenience function to get in-memory repositories.

    Returns:
        Dictionary of standardized in-memory repositories
    """
    factory = StandardizedRepositoryFactory(RepositoryType.MEMORY)
    return factory.create_all_repositories()


def get_database_repositories(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, object]:
    """Convenience function to get database repositories.

    Args:
        session_factory: Async session factory for database access

    Returns:
        Dictionary of standardized database repositories
    """
    factory = StandardizedRepositoryFactory(RepositoryType.DATABASE, session_factory)
    return factory.create_all_repositories()


# Migration helpers for backward compatibility
class RepositoryMigrationHelper:
    """Helper class for migrating from old repository patterns to standardized ones."""

    @staticmethod
    def wrap_legacy_repository(legacy_repo: object, protocol_class: type[T]) -> T:
        """Wrap a legacy repository to follow standardized protocol.

        Args:
            legacy_repo: Legacy repository instance
            protocol_class: Target protocol class

        Returns:
            Wrapped repository following the standardized protocol

        Raises:
            ValueError: If repository cannot be wrapped
        """
        # This would contain logic to wrap legacy repositories
        # For now, we'll raise an error suggesting migration
        raise ValueError(
            f"Legacy repository {type(legacy_repo)} should be migrated to "
            f"implement {protocol_class.__name__} directly. "
            f"Use StandardizedRepositoryFactory to create new repositories."
        )
