"""Database migration and initialization utilities."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .database import DatabaseManager
from .database_repositories import Base

logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Database migration and initialization manager."""

    def __init__(self, database_manager: DatabaseManager):
        """Initialize with database manager.

        Args:
            database_manager: Database manager instance
        """
        self.db_manager = database_manager
        self.engine = database_manager.engine

    def create_all_tables(self) -> bool:
        """Create all database tables.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database tables: {e}")
            return False

    def drop_all_tables(self) -> bool:
        """Drop all database tables.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("Dropping all database tables...")
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop database tables: {e}")
            return False

    def check_database_connection(self) -> bool:
        """Check if database connection is working.

        Returns:
            True if connection is working, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def check_tables_exist(self) -> bool:
        """Check if required tables exist.

        Returns:
            True if all tables exist, False otherwise
        """
        try:
            required_tables = {"datasets", "detectors", "detection_results"}
            existing_tables = set(self.engine.table_names())
            missing_tables = required_tables - existing_tables

            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}")
                return False

            logger.info("All required tables exist")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to check table existence: {e}")
            return False

    def initialize_database(self) -> bool:
        """Initialize database with tables and initial data.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Initializing database...")

        # Check connection first
        if not self.check_database_connection():
            return False

        # Create tables if they don't exist
        if not self.check_tables_exist():
            if not self.create_all_tables():
                return False

        # Add initial data (roles and permissions)
        if not self.seed_default_roles_and_permissions():
            logger.warning("Failed to seed default roles and permissions")
            # Don't fail the initialization for this

        logger.info("Database initialization completed successfully")
        return True

    def seed_default_roles_and_permissions(self) -> bool:
        """Seed default roles and permissions into the database.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Seeding default roles and permissions...")

        # Example roles and permissions
        roles_permissions = [
            ("SUPER_ADMIN", ["platform:read", "platform:write", "platform:delete"]),
            ("VIEWER", ["platform:read"]),
            ("DEVELOPER", ["code:read", "code:write"]),
            ("BUSINESS", ["reports:read"]),
        ]

        # Connect to the database
        try:
            with self.engine.connect() as conn:
                for role, permissions in roles_permissions:
                    # Insert role and permissions
                    conn.execute(text(f"INSERT INTO roles (name) VALUES ('{role}') ON CONFLICT DO NOTHING"))
                    for perm in permissions:
                        conn.execute(text(f"INSERT INTO permissions (name) VALUES ('{perm}') ON CONFLICT DO NOTHING"))
                        conn.execute(text(f"INSERT INTO role_permissions (role, permission) VALUES ('{role}', '{perm}') ON CONFLICT DO NOTHING"))
            logger.info("Successfully seeded roles and permissions")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to seed roles and permissions: {e}")
            return False

    def reset_database(self) -> bool:
        """Reset database by dropping and recreating all tables.

        Returns:
            True if successful, False otherwise
        """
        logger.warning("Resetting database...")

        if not self.drop_all_tables():
            return False

        if not self.create_all_tables():
            return False

        logger.info("Database reset completed successfully")
        return True

    def get_database_info(self) -> dict:
        """Get database information and status.

        Returns:
            Dictionary with database information
        """
        info = {
            "database_url": self.db_manager.database_url,
            "connection_working": False,
            "tables_exist": False,
            "table_count": 0,
            "engine_info": {},
        }

        try:
            # Check connection
            info["connection_working"] = self.check_database_connection()

            if info["connection_working"]:
                # Check tables
                info["tables_exist"] = self.check_tables_exist()

                # Get table count
                try:
                    table_names = self.engine.table_names()
                    info["table_count"] = len(table_names)
                    info["tables"] = table_names
                except Exception:
                    pass

                # Get engine info
                info["engine_info"] = {
                    "driver": str(self.engine.driver),
                    "dialect": str(self.engine.dialect),
                    "pool_size": getattr(self.engine.pool, "size", "unknown"),
                    "pool_checked_out": getattr(
                        self.engine.pool, "checkedout", "unknown"
                    ),
                }

        except Exception as e:
            logger.error(f"Failed to get database info: {e}")

        return info


def create_database_migrator(database_url: str, echo: bool = False) -> DatabaseMigrator:
    """Create database migrator with given URL.

    Args:
        database_url: Database connection URL
        echo: Whether to echo SQL statements

    Returns:
        Database migrator instance
    """
    db_manager = DatabaseManager(database_url, echo)
    return DatabaseMigrator(db_manager)


def quick_init_database(database_url: str) -> bool:
    """Quick database initialization.

    Args:
        database_url: Database connection URL

    Returns:
        True if successful, False otherwise
    """
    migrator = create_database_migrator(database_url)
    return migrator.initialize_database()


def quick_reset_database(database_url: str) -> bool:
    """Quick database reset.

    Args:
        database_url: Database connection URL

    Returns:
        True if successful, False otherwise
    """
    migrator = create_database_migrator(database_url)
    return migrator.reset_database()
