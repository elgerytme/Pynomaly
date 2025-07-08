"""Repository factory for intelligent repository selection.

This factory provides intelligent repository selection based on database configuration,
automatically choosing between database-backed and in-memory repositories while
maintaining backward compatibility and ease of use.
"""

from __future__ import annotations

import logging

from pynomaly.infrastructure.config.database_config import (
    DatabaseConfig,
    DatabaseProfile,
    get_database_config,
    get_database_config_manager,
)
from pynomaly.infrastructure.persistence.database import DatabaseManager
from pynomaly.infrastructure.persistence.migrations import DatabaseMigrator
from pynomaly.shared.protocols import (
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)

# In-memory repositories
from .in_memory_repositories import (
    InMemoryDatasetRepository,
    InMemoryDetectorRepository,
    InMemoryResultRepository,
)

# Database repositories (optional import)
try:
    from pynomaly.infrastructure.persistence.database_repositories import (
        DatabaseDatasetRepository,
        DatabaseDetectionResultRepository,
        DatabaseDetectorRepository,
    )

    DATABASE_REPOSITORIES_AVAILABLE = True
except ImportError:
    DatabaseDatasetRepository = None
    DatabaseDetectorRepository = None
    DatabaseDetectionResultRepository = None
    DATABASE_REPOSITORIES_AVAILABLE = False

logger = logging.getLogger(__name__)


class RepositoryFactory:
    """Factory for creating appropriate repository instances based on configuration."""

    def __init__(self, config: DatabaseConfig | None = None):
        """Initialize repository factory.

        Args:
            config: Database configuration (auto-detected if None)
        """
        self.config = config or get_database_config()
        self._database_manager: DatabaseManager | None = None
        self._repositories_cache = {}

    @property
    def database_manager(self) -> DatabaseManager | None:
        """Get database manager instance."""
        if self._database_manager is None and self._should_use_database():
            try:
                self._database_manager = DatabaseManager(
                    database_url=self.config.get_connection_url(), echo=self.config.echo
                )

                # Initialize database if needed
                if self._should_initialize_database():
                    migrator = DatabaseMigrator(self._database_manager)
                    if not migrator.check_tables_exist():
                        logger.info("Database tables don't exist, creating them...")
                        if not migrator.create_all_tables():
                            logger.warning(
                                "Failed to create database tables, falling back to in-memory"
                            )
                            self._database_manager = None
                        else:
                            logger.info("Database tables created successfully")
                    else:
                        logger.info("Database tables already exist")

            except Exception as e:
                logger.warning(f"Failed to initialize database manager: {e}")
                logger.info("Falling back to in-memory repositories")
                self._database_manager = None

        return self._database_manager

    def _should_use_database(self) -> bool:
        """Check if database repositories should be used."""
        # Always use in-memory for memory profile
        if self.config.profile == DatabaseProfile.MEMORY:
            return False

        # Check if database repositories are available
        if not DATABASE_REPOSITORIES_AVAILABLE:
            logger.warning("Database repositories not available, using in-memory")
            return False

        # Check if database configuration is valid
        if not get_database_config_manager().validate_config(self.config):
            logger.warning("Invalid database configuration, using in-memory")
            return False

        return True

    def _should_initialize_database(self) -> bool:
        """Check if database should be auto-initialized."""
        # Always initialize for development and testing
        if self.config.profile in [
            DatabaseProfile.DEVELOPMENT,
            DatabaseProfile.TESTING,
        ]:
            return True

        # Don't auto-initialize for production (should be done manually)
        return False

    def create_detector_repository(self) -> DetectorRepositoryProtocol:
        """Create detector repository.

        Returns:
            Detector repository instance
        """
        cache_key = "detector"

        if cache_key not in self._repositories_cache:
            if self._should_use_database() and self.database_manager:
                logger.info("Creating database detector repository")
                self._repositories_cache[cache_key] = DatabaseDetectorRepository(
                    self.database_manager.get_session
                )
            else:
                logger.info("Creating in-memory detector repository")
                self._repositories_cache[cache_key] = InMemoryDetectorRepository()

        return self._repositories_cache[cache_key]

    def create_dataset_repository(self) -> DatasetRepositoryProtocol:
        """Create dataset repository.

        Returns:
            Dataset repository instance
        """
        cache_key = "dataset"

        if cache_key not in self._repositories_cache:
            if self._should_use_database() and self.database_manager:
                logger.info("Creating database dataset repository")
                self._repositories_cache[cache_key] = DatabaseDatasetRepository(
                    self.database_manager.get_session
                )
            else:
                logger.info("Creating in-memory dataset repository")
                self._repositories_cache[cache_key] = InMemoryDatasetRepository()

        return self._repositories_cache[cache_key]

    def create_result_repository(self) -> DetectionResultRepositoryProtocol:
        """Create detection result repository.

        Returns:
            Detection result repository instance
        """
        cache_key = "result"

        if cache_key not in self._repositories_cache:
            if self._should_use_database() and self.database_manager:
                logger.info("Creating database result repository")
                self._repositories_cache[cache_key] = DatabaseDetectionResultRepository(
                    self.database_manager.get_session
                )
            else:
                logger.info("Creating in-memory result repository")
                self._repositories_cache[cache_key] = InMemoryResultRepository()

        return self._repositories_cache[cache_key]

    def get_repository_info(self) -> dict:
        """Get information about current repository configuration.

        Returns:
            Dictionary with repository information
        """
        using_database = (
            self._should_use_database() and self.database_manager is not None
        )

        info = {
            "profile": self.config.profile.value,
            "db_type": self.config.db_type.value,
            "using_database": using_database,
            "repository_type": "database" if using_database else "memory",
            "connection_url": (
                self.config.get_connection_url() if using_database else None
            ),
            "database_available": DATABASE_REPOSITORIES_AVAILABLE,
            "config_valid": get_database_config_manager().validate_config(self.config),
        }

        if using_database and self.database_manager:
            migrator = DatabaseMigrator(self.database_manager)
            info.update(
                {
                    "database_connected": migrator.check_database_connection(),
                    "tables_exist": migrator.check_tables_exist(),
                    "database_info": migrator.get_database_info(),
                }
            )

        return info

    def reset_cache(self) -> None:
        """Reset repository cache."""
        self._repositories_cache.clear()
        logger.info("Repository cache reset")

    def close(self) -> None:
        """Close database connections and clean up."""
        if self._database_manager:
            self._database_manager.close()
            self._database_manager = None

        self._repositories_cache.clear()
        logger.info("Repository factory closed")


class RepositoryManager:
    """Manages repository lifecycle and provides easy access to repositories."""

    def __init__(self, config: DatabaseConfig | None = None):
        """Initialize repository manager.

        Args:
            config: Database configuration (auto-detected if None)
        """
        self._factory = RepositoryFactory(config)
        self._repositories = {}

    @property
    def detector_repository(self) -> DetectorRepositoryProtocol:
        """Get detector repository."""
        if "detector" not in self._repositories:
            self._repositories["detector"] = self._factory.create_detector_repository()
        return self._repositories["detector"]

    @property
    def dataset_repository(self) -> DatasetRepositoryProtocol:
        """Get dataset repository."""
        if "dataset" not in self._repositories:
            self._repositories["dataset"] = self._factory.create_dataset_repository()
        return self._repositories["dataset"]

    @property
    def result_repository(self) -> DetectionResultRepositoryProtocol:
        """Get detection result repository."""
        if "result" not in self._repositories:
            self._repositories["result"] = self._factory.create_result_repository()
        return self._repositories["result"]

    def get_info(self) -> dict:
        """Get repository manager information."""
        return self._factory.get_repository_info()

    def reset(self) -> None:
        """Reset all repositories."""
        self._repositories.clear()
        self._factory.reset_cache()
        logger.info("Repository manager reset")

    def close(self) -> None:
        """Close repository manager."""
        self._repositories.clear()
        self._factory.close()
        logger.info("Repository manager closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global repository manager instance
_repository_manager: RepositoryManager | None = None


def get_repository_manager(
    config: DatabaseConfig | None = None,
) -> RepositoryManager:
    """Get global repository manager.

    Args:
        config: Database configuration (auto-detected if None)

    Returns:
        Repository manager instance
    """
    global _repository_manager
    if _repository_manager is None:
        _repository_manager = RepositoryManager(config)
    return _repository_manager


def create_detector_repository(
    config: DatabaseConfig | None = None,
) -> DetectorRepositoryProtocol:
    """Create detector repository with intelligent selection.

    Args:
        config: Database configuration (auto-detected if None)

    Returns:
        Detector repository instance
    """
    factory = RepositoryFactory(config)
    return factory.create_detector_repository()


def create_dataset_repository(
    config: DatabaseConfig | None = None,
) -> DatasetRepositoryProtocol:
    """Create dataset repository with intelligent selection.

    Args:
        config: Database configuration (auto-detected if None)

    Returns:
        Dataset repository instance
    """
    factory = RepositoryFactory(config)
    return factory.create_dataset_repository()


def create_result_repository(
    config: DatabaseConfig | None = None,
) -> DetectionResultRepositoryProtocol:
    """Create detection result repository with intelligent selection.

    Args:
        config: Database configuration (auto-detected if None)

    Returns:
        Detection result repository instance
    """
    factory = RepositoryFactory(config)
    return factory.create_result_repository()


def get_repository_info(config: DatabaseConfig | None = None) -> dict:
    """Get repository configuration information.

    Args:
        config: Database configuration (auto-detected if None)

    Returns:
        Repository information dictionary
    """
    factory = RepositoryFactory(config)
    return factory.get_repository_info()


# Convenience functions for testing
def create_testing_repositories() -> (
    tuple[
        DetectorRepositoryProtocol,
        DatasetRepositoryProtocol,
        DetectionResultRepositoryProtocol,
    ]
):
    """Create repositories for testing (always in-memory).

    Returns:
        Tuple of (detector_repo, dataset_repo, result_repo)
    """
    testing_config = get_database_config(DatabaseProfile.TESTING)
    factory = RepositoryFactory(testing_config)

    return (
        factory.create_detector_repository(),
        factory.create_dataset_repository(),
        factory.create_result_repository(),
    )


def create_development_repositories() -> (
    tuple[
        DetectorRepositoryProtocol,
        DatasetRepositoryProtocol,
        DetectionResultRepositoryProtocol,
    ]
):
    """Create repositories for development (SQLite file).

    Returns:
        Tuple of (detector_repo, dataset_repo, result_repo)
    """
    development_config = get_database_config(DatabaseProfile.DEVELOPMENT)
    factory = RepositoryFactory(development_config)

    return (
        factory.create_detector_repository(),
        factory.create_dataset_repository(),
        factory.create_result_repository(),
    )
