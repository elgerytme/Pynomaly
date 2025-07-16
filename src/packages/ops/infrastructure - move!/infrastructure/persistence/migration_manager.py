"""Database migration management system for Pynomaly."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from monorepo.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database migrations using Alembic."""

    def __init__(self, database_url: str | None = None):
        """Initialize migration manager.

        Args:
            database_url: Database URL. If None, will use settings.
        """
        self.database_url = database_url or self._get_database_url()
        self.engine = create_engine(self.database_url)
        self.config = self._get_alembic_config()

    def _get_database_url(self) -> str:
        """Get database URL from environment or settings."""
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            return db_url

        try:
            settings = Settings()
            return settings.database_url
        except Exception:
            return "sqlite:///./storage/monorepo.db"

    def _get_alembic_config(self) -> Config:
        """Get Alembic configuration."""
        # Find the alembic.ini file
        project_root = Path(__file__).parent.parent.parent.parent.parent
        alembic_ini = project_root / "alembic.ini"

        if not alembic_ini.exists():
            raise FileNotFoundError(
                f"Alembic configuration file not found at {alembic_ini}"
            )

        config = Config(str(alembic_ini))
        config.set_main_option("sqlalchemy.url", self.database_url)

        return config

    def check_migration_needed(self) -> bool:
        """Check if database migration is needed."""
        try:
            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_revision = context.get_current_revision()

                script_dir = ScriptDirectory.from_config(self.config)
                head_revision = script_dir.get_current_head()

                if current_revision is None:
                    logger.info("Database has no migration history")
                    return True

                if current_revision != head_revision:
                    logger.info(
                        f"Database revision {current_revision} != head {head_revision}"
                    )
                    return True

                logger.info("Database is up to date")
                return False

        except SQLAlchemyError as e:
            logger.error(f"Error checking migration status: {e}")
            return True

    def get_migration_status(self) -> dict:
        """Get current migration status."""
        try:
            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_revision = context.get_current_revision()

                script_dir = ScriptDirectory.from_config(self.config)
                head_revision = script_dir.get_current_head()

                # Get all revisions
                revisions = []
                for revision in script_dir.walk_revisions():
                    revisions.append(
                        {
                            "revision": revision.revision,
                            "down_revision": revision.down_revision,
                            "doc": revision.doc,
                            "create_date": revision.create_date.isoformat()
                            if revision.create_date
                            else None,
                            "is_current": revision.revision == current_revision,
                            "is_head": revision.revision == head_revision,
                        }
                    )

                return {
                    "current_revision": current_revision,
                    "head_revision": head_revision,
                    "migration_needed": current_revision != head_revision,
                    "revisions": revisions,
                    "database_url": self.database_url,
                }

        except SQLAlchemyError as e:
            logger.error(f"Error getting migration status: {e}")
            return {"error": str(e), "database_url": self.database_url}

    def create_migration(self, message: str, autogenerate: bool = True) -> str:
        """Create a new migration.

        Args:
            message: Migration message
            autogenerate: Whether to auto-generate migration from model changes

        Returns:
            Revision ID of the created migration
        """
        try:
            # Create migration
            if autogenerate:
                command.revision(self.config, message=message, autogenerate=True)
            else:
                command.revision(self.config, message=message)

            # Get the latest revision
            script_dir = ScriptDirectory.from_config(self.config)
            head_revision = script_dir.get_current_head()

            logger.info(f"Created migration: {head_revision} - {message}")
            return head_revision

        except Exception as e:
            logger.error(f"Error creating migration: {e}")
            raise

    def run_migrations(self, target_revision: str = "head") -> bool:
        """Run database migrations.

        Args:
            target_revision: Target revision to migrate to

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Running migrations to {target_revision}")
            command.upgrade(self.config, target_revision)
            logger.info("Migrations completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error running migrations: {e}")
            return False

    def rollback_migration(self, target_revision: str = "-1") -> bool:
        """Rollback database migration.

        Args:
            target_revision: Target revision to rollback to

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Rolling back to revision {target_revision}")
            command.downgrade(self.config, target_revision)
            logger.info("Rollback completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error rolling back migration: {e}")
            return False

    def initialize_database(self) -> bool:
        """Initialize database with base migration."""
        try:
            # Check if alembic_version table exists
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
                    )
                )
                if result.fetchone() is None:
                    # Initialize alembic
                    command.stamp(self.config, "head")
                    logger.info("Database initialized with current schema")
                else:
                    logger.info("Database already initialized")

                return True

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False

    def reset_database(self) -> bool:
        """Reset database to initial state."""
        try:
            logger.warning("Resetting database to initial state")

            # Drop all tables
            from monorepo.infrastructure.repositories.sqlalchemy_user_repository import (
                Base,
            )

            Base.metadata.drop_all(self.engine)

            # Recreate tables
            Base.metadata.create_all(self.engine)

            # Stamp with head revision
            command.stamp(self.config, "head")

            logger.info("Database reset completed")
            return True

        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False

    def validate_migration(self, revision: str) -> bool:
        """Validate a specific migration.

        Args:
            revision: Revision ID to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            script_dir = ScriptDirectory.from_config(self.config)
            script = script_dir.get_revision(revision)

            if script is None:
                logger.error(f"Revision {revision} not found")
                return False

            # Basic validation - check if the script exists and is readable
            if not os.path.exists(script.path):
                logger.error(f"Migration file not found: {script.path}")
                return False

            logger.info(f"Migration {revision} is valid")
            return True

        except Exception as e:
            logger.error(f"Error validating migration {revision}: {e}")
            return False

    def get_migration_history(self) -> list[dict]:
        """Get migration history."""
        try:
            history = []

            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)

                # Get applied migrations
                applied_migrations = []
                try:
                    result = conn.execute(
                        text("SELECT version_num FROM alembic_version")
                    )
                    applied_migrations = [row[0] for row in result]
                except SQLAlchemyError:
                    # Table doesn't exist yet
                    pass

                script_dir = ScriptDirectory.from_config(self.config)

                for revision in script_dir.walk_revisions():
                    history.append(
                        {
                            "revision": revision.revision,
                            "down_revision": revision.down_revision,
                            "message": revision.doc,
                            "create_date": revision.create_date.isoformat()
                            if revision.create_date
                            else None,
                            "applied": revision.revision in applied_migrations,
                        }
                    )

                return sorted(
                    history, key=lambda x: x["create_date"] or "", reverse=True
                )

        except Exception as e:
            logger.error(f"Error getting migration history: {e}")
            return []

    def backup_database(self, backup_path: str) -> bool:
        """Create a database backup before migration.

        Args:
            backup_path: Path to store backup

        Returns:
            True if successful, False otherwise
        """
        try:
            if "sqlite" in self.database_url:
                # For SQLite, just copy the file
                import shutil

                db_path = self.database_url.replace("sqlite:///", "")
                shutil.copy2(db_path, backup_path)
                logger.info(f"Database backed up to {backup_path}")
                return True
            else:
                # For other databases, would need pg_dump, mysqldump, etc.
                logger.warning("Database backup not implemented for this database type")
                return False

        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False

    def cleanup_old_backups(self, backup_dir: str, keep_count: int = 5) -> bool:
        """Clean up old database backups.

        Args:
            backup_dir: Directory containing backups
            keep_count: Number of backups to keep

        Returns:
            True if successful, False otherwise
        """
        try:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                return True

            # Find backup files
            backup_files = list(backup_path.glob("*.db"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove old backups
            for backup_file in backup_files[keep_count:]:
                backup_file.unlink()
                logger.info(f"Removed old backup: {backup_file}")

            return True

        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
            return False


def create_migration_manager(database_url: str | None = None) -> MigrationManager:
    """Create a migration manager instance.

    Args:
        database_url: Database URL. If None, will use settings.

    Returns:
        Migration manager instance
    """
    return MigrationManager(database_url)


def quick_migrate(database_url: str | None = None) -> bool:
    """Quick database migration.

    Args:
        database_url: Database URL. If None, will use settings.

    Returns:
        True if successful, False otherwise
    """
    manager = create_migration_manager(database_url)
    return manager.run_migrations()


def init_and_migrate(database_url: str | None = None) -> bool:
    """Initialize and migrate database.

    Args:
        database_url: Database URL. If None, will use settings.

    Returns:
        True if successful, False otherwise
    """
    manager = create_migration_manager(database_url)

    # Initialize if needed
    if not manager.initialize_database():
        return False

    # Run migrations
    return manager.run_migrations()
