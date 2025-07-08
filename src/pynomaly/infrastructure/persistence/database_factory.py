"""Database repository factory for production environments."""

from __future__ import annotations

import os
from typing import Literal

from pynomaly.infrastructure.persistence.database import (
    DatabaseManager,
    POSTGRESQL_LOCAL_URL,
    SQLITE_FILE_URL,
    SQLITE_MEMORY_URL,
)
from pynomaly.infrastructure.persistence.database_repositories import (
    DatabaseDatasetRepository,
    DatabaseDetectionResultRepository,
    DatabaseDetectorRepository,
)
from pynomaly.infrastructure.repositories.repository_service import RepositoryService


class DatabaseRepositoryFactory:
    """Factory for creating database-backed repository implementations."""

    @staticmethod
    def create_production_repository_service(
        database_url: str | None = None,
        echo: bool = False,
        create_tables: bool = True,
    ) -> RepositoryService:
        """Create repository service with production database backends.

        Args:
            database_url: Database connection URL. If None, uses environment variable
            echo: Whether to echo SQL statements
            create_tables: Whether to create database tables if they don't exist

        Returns:
            Repository service with database-backed repositories
        """
        if database_url is None:
            database_url = os.getenv("DATABASE_URL", POSTGRESQL_LOCAL_URL)

        # Initialize database manager
        db_manager = DatabaseManager(database_url, echo=echo)

        # Create tables if requested
        if create_tables:
            db_manager.create_tables()

        # Get session factory
        session_factory = db_manager.session_factory

        # Create database repositories
        detector_repo = DatabaseDetectorRepository(session_factory)
        dataset_repo = DatabaseDatasetRepository(session_factory)
        result_repo = DatabaseDetectionResultRepository(session_factory)

        return RepositoryService(
            detector_repository=detector_repo,
            dataset_repository=dataset_repo,
            result_repository=result_repo,
        )

    @staticmethod
    def create_sqlite_repository_service(
        database_path: str = "./pynomaly.db",
        echo: bool = False,
        create_tables: bool = True,
    ) -> RepositoryService:
        """Create repository service with SQLite backend.

        Args:
            database_path: Path to SQLite database file
            echo: Whether to echo SQL statements
            create_tables: Whether to create database tables if they don't exist

        Returns:
            Repository service with SQLite-backed repositories
        """
        database_url = f"sqlite:///{database_path}"
        return DatabaseRepositoryFactory.create_production_repository_service(
            database_url=database_url,
            echo=echo,
            create_tables=create_tables,
        )

    @staticmethod
    def create_postgresql_repository_service(
        host: str = "localhost",
        port: int = 5432,
        database: str = "pynomaly_production",
        username: str = "pynomaly",
        password: str = "pynomaly",
        echo: bool = False,
        create_tables: bool = True,
    ) -> RepositoryService:
        """Create repository service with PostgreSQL backend.

        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            echo: Whether to echo SQL statements
            create_tables: Whether to create database tables if they don't exist

        Returns:
            Repository service with PostgreSQL-backed repositories
        """
        database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        return DatabaseRepositoryFactory.create_production_repository_service(
            database_url=database_url,
            echo=echo,
            create_tables=create_tables,
        )

    @staticmethod
    def create_from_environment() -> RepositoryService:
        """Create repository service based on environment variables.

        Environment variables:
        - DATABASE_URL: Full database connection URL
        - DB_HOST: Database host (default: localhost)
        - DB_PORT: Database port (default: 5432)
        - DB_NAME: Database name (default: pynomaly_production)
        - DB_USER: Database username (default: pynomaly)
        - DB_PASSWORD: Database password (default: pynomaly)
        - DB_ECHO: Whether to echo SQL statements (default: False)
        - DB_CREATE_TABLES: Whether to create tables (default: True)

        Returns:
            Repository service configured from environment
        """
        database_url = os.getenv("DATABASE_URL")
        
        if database_url is None:
            # Build from individual components
            host = os.getenv("DB_HOST", "localhost")
            port = int(os.getenv("DB_PORT", "5432"))
            database = os.getenv("DB_NAME", "pynomaly_production")
            username = os.getenv("DB_USER", "pynomaly")
            password = os.getenv("DB_PASSWORD", "pynomaly")
            
            database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"

        echo = os.getenv("DB_ECHO", "False").lower() == "true"
        create_tables = os.getenv("DB_CREATE_TABLES", "True").lower() == "true"

        return DatabaseRepositoryFactory.create_production_repository_service(
            database_url=database_url,
            echo=echo,
            create_tables=create_tables,
        )

    @staticmethod
    def create_test_database_service(
        use_memory: bool = True,
        echo: bool = False,
    ) -> RepositoryService:
        """Create repository service for testing.

        Args:
            use_memory: Whether to use in-memory SQLite (faster for tests)
            echo: Whether to echo SQL statements

        Returns:
            Repository service optimized for testing
        """
        database_url = SQLITE_MEMORY_URL if use_memory else SQLITE_FILE_URL
        return DatabaseRepositoryFactory.create_production_repository_service(
            database_url=database_url,
            echo=echo,
            create_tables=True,
        )


def create_production_repository_service(
    database_url: str | None = None,
    echo: bool = False,
    create_tables: bool = True,
) -> RepositoryService:
    """Convenience function to create production repository service.

    Args:
        database_url: Database connection URL
        echo: Whether to echo SQL statements
        create_tables: Whether to create database tables

    Returns:
        Repository service with production database backends
    """
    return DatabaseRepositoryFactory.create_production_repository_service(
        database_url=database_url,
        echo=echo,
        create_tables=create_tables,
    )
