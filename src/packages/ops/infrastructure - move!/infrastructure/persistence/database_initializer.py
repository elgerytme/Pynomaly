"""Database initialization and migration startup handler."""

from __future__ import annotations

import logging
import os

from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.persistence.migration_manager import (
    create_migration_manager,
)

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization and migration on startup."""

    def __init__(self, settings: Settings | None = None):
        """Initialize database initializer.

        Args:
            settings: Application settings. If None, will create new instance.
        """
        self.settings = settings or Settings()
        self.migration_manager = create_migration_manager(self.settings.database_url)

    def initialize_on_startup(self) -> bool:
        """Initialize database on application startup.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing database on startup")

            # Check if auto-migration is enabled
            auto_migrate = os.getenv("PYNOMALY_AUTO_MIGRATE", "true").lower() == "true"

            if auto_migrate:
                logger.info("Auto-migration is enabled")

                # Initialize database if needed
                if not self.migration_manager.initialize_database():
                    logger.error("Failed to initialize database")
                    return False

                # Check if migration is needed
                if self.migration_manager.check_migration_needed():
                    logger.info("Database migration needed, running migrations")

                    # Create backup if in production
                    if self.settings.environment == "production":
                        backup_path = (
                            f"backup_startup_{self.settings.get_timestamp()}.db"
                        )
                        if self.migration_manager.backup_database(backup_path):
                            logger.info(f"Created startup backup: {backup_path}")
                        else:
                            logger.warning("Failed to create startup backup")

                    # Run migrations
                    if not self.migration_manager.run_migrations():
                        logger.error("Failed to run migrations")
                        return False

                    logger.info("Database migrations completed successfully")
                else:
                    logger.info("Database is up to date")
            else:
                logger.info("Auto-migration is disabled")

                # Just check if database is accessible
                if not self.migration_manager.check_migration_needed():
                    logger.info("Database is accessible and up to date")
                else:
                    logger.warning(
                        "Database may need migration but auto-migration is disabled"
                    )

            return True

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False

    def check_database_health(self) -> dict:
        """Check database health status.

        Returns:
            Dictionary with health status information
        """
        try:
            status = self.migration_manager.get_migration_status()

            return {
                "healthy": "error" not in status,
                "current_revision": status.get("current_revision"),
                "head_revision": status.get("head_revision"),
                "migration_needed": status.get("migration_needed", False),
                "database_url": status.get("database_url"),
                "error": status.get("error"),
                "revision_count": len(status.get("revisions", [])),
                "last_check": self.settings.get_timestamp(),
            }

        except Exception as e:
            logger.error(f"Error checking database health: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "last_check": self.settings.get_timestamp(),
            }

    def ensure_database_ready(self) -> bool:
        """Ensure database is ready for use.

        Returns:
            True if database is ready, False otherwise
        """
        try:
            # Check if database is accessible
            health = self.check_database_health()

            if not health["healthy"]:
                logger.error(f"Database is not healthy: {health.get('error')}")
                return False

            # Check if migration is needed
            if health["migration_needed"]:
                logger.warning("Database migration is needed")

                # If auto-migration is enabled, run it
                auto_migrate = (
                    os.getenv("PYNOMALY_AUTO_MIGRATE", "true").lower() == "true"
                )
                if auto_migrate:
                    logger.info("Running required migrations")
                    if not self.migration_manager.run_migrations():
                        logger.error("Failed to run required migrations")
                        return False
                else:
                    logger.error(
                        "Database migration needed but auto-migration is disabled"
                    )
                    return False

            logger.info("Database is ready for use")
            return True

        except Exception as e:
            logger.error(f"Error ensuring database readiness: {e}")
            return False

    def create_development_data(self) -> bool:
        """Create development/test data if in development mode.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.settings.environment == "development":
                logger.info("Creating development data")

                # This could be expanded to create sample data
                # For now, just ensure basic schema is in place
                return True
            else:
                logger.info("Not in development mode, skipping test data creation")
                return True

        except Exception as e:
            logger.error(f"Error creating development data: {e}")
            return False

    def validate_database_schema(self) -> bool:
        """Validate database schema against expected state.

        Returns:
            True if schema is valid, False otherwise
        """
        try:
            # Get current migration status
            status = self.migration_manager.get_migration_status()

            if "error" in status:
                logger.error(f"Schema validation failed: {status['error']}")
                return False

            # Check if we're at the head revision
            if status["migration_needed"]:
                logger.error("Schema is not at the latest revision")
                return False

            # Additional schema validation could be added here
            # For example, checking specific tables exist, etc.

            logger.info("Database schema validation passed")
            return True

        except Exception as e:
            logger.error(f"Error validating database schema: {e}")
            return False


def initialize_database_on_startup(settings: Settings | None = None) -> bool:
    """Initialize database on application startup.

    Args:
        settings: Application settings. If None, will create new instance.

    Returns:
        True if successful, False otherwise
    """
    initializer = DatabaseInitializer(settings)
    return initializer.initialize_on_startup()


def check_database_health(settings: Settings | None = None) -> dict:
    """Check database health status.

    Args:
        settings: Application settings. If None, will create new instance.

    Returns:
        Dictionary with health status information
    """
    initializer = DatabaseInitializer(settings)
    return initializer.check_database_health()


def ensure_database_ready(settings: Settings | None = None) -> bool:
    """Ensure database is ready for use.

    Args:
        settings: Application settings. If None, will create new instance.

    Returns:
        True if database is ready, False otherwise
    """
    initializer = DatabaseInitializer(settings)
    return initializer.ensure_database_ready()
