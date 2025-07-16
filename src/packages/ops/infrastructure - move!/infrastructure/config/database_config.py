"""Flexible database configuration system for Pynomaly.

This module provides intelligent database configuration with support for:
- Development/Local: SQLite file-based database (easy, no setup required)
- Testing: In-memory SQLite (fast, ephemeral)
- Production: PostgreSQL, MySQL, and other production databases
- In-Memory: Original in-memory repositories as an option

The configuration system automatically detects the environment and selects
appropriate database settings while maintaining backward compatibility.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class DatabaseType(str, Enum):
    """Supported database types."""

    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MARIADB = "mariadb"
    ORACLE = "oracle"
    MSSQL = "mssql"


class DatabaseProfile(str, Enum):
    """Database configuration profiles."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    MEMORY = "memory"


class DatabaseConfig(BaseModel):
    """Database configuration model."""

    profile: DatabaseProfile = DatabaseProfile.DEVELOPMENT
    db_type: DatabaseType = DatabaseType.SQLITE
    url: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None

    # Connection pooling settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True

    # SQLAlchemy settings
    echo: bool = False
    echo_pool: bool = False

    # File-based database settings (SQLite)
    file_path: Path | None = None
    wal_mode: bool = True  # Write-Ahead Logging for SQLite

    # Connection args for specific databases
    connect_args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("file_path")
    @classmethod
    def ensure_file_path_exists(cls, v: Path | None) -> Path | None:
        """Ensure file path parent directory exists."""
        if v is not None:
            v.parent.mkdir(parents=True, exist_ok=True)
        return v

    def get_connection_url(self) -> str:
        """Generate database connection URL."""
        if self.url:
            return self.url

        if self.db_type == DatabaseType.MEMORY:
            return "sqlite:///:memory:"

        elif self.db_type == DatabaseType.SQLITE:
            if self.file_path:
                return f"sqlite:///{self.file_path}"
            else:
                return "sqlite:///./data/pynomaly.db"

        elif self.db_type == DatabaseType.POSTGRESQL:
            host = self.host or "localhost"
            port = self.port or 5432
            database = self.database or "pynomaly"
            username = self.username or "pynomaly"
            password = self.password or ""

            if password:
                return f"postgresql://{username}:{password}@{host}:{port}/{database}"
            else:
                return f"postgresql://{username}@{host}:{port}/{database}"

        elif self.db_type == DatabaseType.MYSQL:
            host = self.host or "localhost"
            port = self.port or 3306
            database = self.database or "pynomaly"
            username = self.username or "pynomaly"
            password = self.password or ""

            if password:
                return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            else:
                return f"mysql+pymysql://{username}@{host}:{port}/{database}"

        elif self.db_type == DatabaseType.MARIADB:
            host = self.host or "localhost"
            port = self.port or 3306
            database = self.database or "pynomaly"
            username = self.username or "pynomaly"
            password = self.password or ""

            if password:
                return (
                    f"mariadb+pymysql://{username}:{password}@{host}:{port}/{database}"
                )
            else:
                return f"mariadb+pymysql://{username}@{host}:{port}/{database}"

        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def get_engine_kwargs(self) -> dict[str, Any]:
        """Get SQLAlchemy engine configuration."""
        kwargs = {
            "echo": self.echo,
            "pool_pre_ping": self.pool_pre_ping,
            "pool_recycle": self.pool_recycle,
        }

        if self.db_type == DatabaseType.MEMORY:
            kwargs.update(
                {
                    "poolclass": "StaticPool",
                    "connect_args": {"check_same_thread": False},
                }
            )

        elif self.db_type == DatabaseType.SQLITE:
            kwargs.update(
                {
                    "poolclass": "StaticPool",
                    "connect_args": {"check_same_thread": False, **self.connect_args},
                }
            )

            # Enable WAL mode for better concurrency
            if self.wal_mode:
                kwargs["connect_args"]["isolation_level"] = None

        elif self.db_type in [
            DatabaseType.POSTGRESQL,
            DatabaseType.MYSQL,
            DatabaseType.MARIADB,
        ]:
            kwargs.update(
                {
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow,
                    "pool_timeout": self.pool_timeout,
                    "echo_pool": self.echo_pool,
                }
            )

            if self.connect_args:
                kwargs["connect_args"] = self.connect_args

        return kwargs


class DatabaseConfigManager:
    """Manages database configuration with intelligent defaults and environment detection."""

    def __init__(self):
        """Initialize database configuration manager."""
        self._config_cache: dict[str, DatabaseConfig] = {}
        self._default_profiles = self._create_default_profiles()

    def _create_default_profiles(self) -> dict[DatabaseProfile, DatabaseConfig]:
        """Create default database configuration profiles."""
        profiles = {}

        # Development profile - SQLite file for easy local development
        profiles[DatabaseProfile.DEVELOPMENT] = DatabaseConfig(
            profile=DatabaseProfile.DEVELOPMENT,
            db_type=DatabaseType.SQLITE,
            file_path=Path("./data/pynomaly.db"),
            echo=False,
            wal_mode=True,
            pool_size=5,
            max_overflow=10,
        )

        # Testing profile - In-memory SQLite for fast tests
        profiles[DatabaseProfile.TESTING] = DatabaseConfig(
            profile=DatabaseProfile.TESTING,
            db_type=DatabaseType.MEMORY,
            echo=False,
            pool_size=1,
            max_overflow=0,
        )

        # Production profile - PostgreSQL with connection pooling
        profiles[DatabaseProfile.PRODUCTION] = DatabaseConfig(
            profile=DatabaseProfile.PRODUCTION,
            db_type=DatabaseType.POSTGRESQL,
            pool_size=20,
            max_overflow=30,
            pool_timeout=60,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False,
            echo_pool=False,
        )

        # Memory profile - In-memory repositories (original behavior)
        profiles[DatabaseProfile.MEMORY] = DatabaseConfig(
            profile=DatabaseProfile.MEMORY,
            db_type=DatabaseType.MEMORY,
            echo=False,
            pool_size=1,
            max_overflow=0,
        )

        return profiles

    def detect_environment(self) -> DatabaseProfile:
        """Detect current environment and return appropriate database profile."""
        # Check environment variables first
        env_profile = os.environ.get("PYNOMALY_DB_PROFILE", "").lower()
        if env_profile in [p.value for p in DatabaseProfile]:
            logger.info(
                f"Using database profile from PYNOMALY_DB_PROFILE: {env_profile}"
            )
            return DatabaseProfile(env_profile)

        # Check for testing environment
        if (
            os.environ.get("PYTEST_CURRENT_TEST")
            or os.environ.get("TESTING")
            or os.environ.get("CI")
        ):
            logger.info("Testing environment detected, using testing database profile")
            return DatabaseProfile.TESTING

        # Check for production environment
        env = os.environ.get("PYNOMALY_ENVIRONMENT", "").lower()
        if env in ["production", "prod"]:
            logger.info("Production environment detected")
            return DatabaseProfile.PRODUCTION

        # Check for explicit database URL
        if os.environ.get("DATABASE_URL") or os.environ.get("PYNOMALY_DATABASE_URL"):
            logger.info("Database URL detected, using production profile")
            return DatabaseProfile.PRODUCTION

        # Default to development
        logger.info("Using development database profile (default)")
        return DatabaseProfile.DEVELOPMENT

    def get_database_config(
        self, profile: DatabaseProfile | None = None, **overrides
    ) -> DatabaseConfig:
        """Get database configuration for the specified profile.

        Args:
            profile: Database profile to use (auto-detected if None)
            **overrides: Configuration overrides

        Returns:
            Database configuration
        """
        if profile is None:
            profile = self.detect_environment()

        cache_key = f"{profile.value}:{hash(str(sorted(overrides.items())))}"

        if cache_key not in self._config_cache:
            # Get base configuration
            base_config = self._default_profiles[profile].model_copy()

            # Apply environment variable overrides
            self._apply_env_overrides(base_config)

            # Apply explicit overrides
            if overrides:
                for key, value in overrides.items():
                    if hasattr(base_config, key):
                        setattr(base_config, key, value)

            self._config_cache[cache_key] = base_config

        return self._config_cache[cache_key]

    def _apply_env_overrides(self, config: DatabaseConfig) -> None:
        """Apply environment variable overrides to configuration."""
        # Database URL
        db_url = os.environ.get("DATABASE_URL") or os.environ.get(
            "PYNOMALY_DATABASE_URL"
        )
        if db_url:
            config.url = db_url
            # Parse URL to extract database type
            parsed = urlparse(db_url)
            if parsed.scheme:
                if parsed.scheme.startswith("postgresql"):
                    config.db_type = DatabaseType.POSTGRESQL
                elif parsed.scheme.startswith("mysql"):
                    config.db_type = DatabaseType.MYSQL
                elif parsed.scheme.startswith("mariadb"):
                    config.db_type = DatabaseType.MARIADB
                elif parsed.scheme.startswith("sqlite"):
                    config.db_type = DatabaseType.SQLITE

        # Database type
        db_type = os.environ.get("PYNOMALY_DB_TYPE", "").lower()
        if db_type in [t.value for t in DatabaseType]:
            config.db_type = DatabaseType(db_type)

        # Connection details
        if os.environ.get("PYNOMALY_DB_HOST"):
            config.host = os.environ["PYNOMALY_DB_HOST"]

        if os.environ.get("PYNOMALY_DB_PORT"):
            config.port = int(os.environ["PYNOMALY_DB_PORT"])

        if os.environ.get("PYNOMALY_DB_NAME"):
            config.database = os.environ["PYNOMALY_DB_NAME"]

        if os.environ.get("PYNOMALY_DB_USER"):
            config.username = os.environ["PYNOMALY_DB_USER"]

        if os.environ.get("PYNOMALY_DB_PASSWORD"):
            config.password = os.environ["PYNOMALY_DB_PASSWORD"]

        # SQLite file path
        if os.environ.get("PYNOMALY_DB_FILE"):
            config.file_path = Path(os.environ["PYNOMALY_DB_FILE"])

        # Pool settings
        if os.environ.get("PYNOMALY_DB_POOL_SIZE"):
            config.pool_size = int(os.environ["PYNOMALY_DB_POOL_SIZE"])

        if os.environ.get("PYNOMALY_DB_MAX_OVERFLOW"):
            config.max_overflow = int(os.environ["PYNOMALY_DB_MAX_OVERFLOW"])

        # Debug settings
        if os.environ.get("PYNOMALY_DB_ECHO", "").lower() in ["true", "1", "yes"]:
            config.echo = True

        if os.environ.get("PYNOMALY_DB_ECHO_POOL", "").lower() in ["true", "1", "yes"]:
            config.echo_pool = True

    def create_profile(self, name: str, config: DatabaseConfig) -> None:
        """Create a custom database profile.

        Args:
            name: Profile name
            config: Database configuration
        """
        # Create custom profile enum value if needed
        custom_profile = (
            DatabaseProfile(name)
            if name in [p.value for p in DatabaseProfile]
            else name
        )
        self._default_profiles[custom_profile] = config
        logger.info(f"Created custom database profile: {name}")

    def list_profiles(self) -> dict[str, dict[str, Any]]:
        """List all available database profiles.

        Returns:
            Dictionary of profile names to their configurations
        """
        profiles = {}
        for profile_name, config in self._default_profiles.items():
            profiles[profile_name.value] = {
                "db_type": config.db_type.value,
                "url": config.get_connection_url(),
                "pool_size": config.pool_size,
                "max_overflow": config.max_overflow,
                "echo": config.echo,
            }
        return profiles

    def validate_config(self, config: DatabaseConfig) -> bool:
        """Validate database configuration.

        Args:
            config: Database configuration to validate

        Returns:
            True if configuration is valid
        """
        try:
            # Try to generate connection URL
            url = config.get_connection_url()

            # Validate URL format
            parsed = urlparse(url)
            if not parsed.scheme:
                logger.error(f"Invalid database URL: {url}")
                return False

            # Check for required files/directories for SQLite
            if config.db_type == DatabaseType.SQLITE and config.file_path:
                if not config.file_path.parent.exists():
                    logger.error(
                        f"SQLite database directory does not exist: {config.file_path.parent}"
                    )
                    return False

            logger.info(f"Database configuration is valid: {config.db_type.value}")
            return True

        except Exception as e:
            logger.error(f"Database configuration validation failed: {e}")
            return False

    def get_repository_type(self, config: DatabaseConfig) -> str:
        """Get repository type for the given configuration.

        Args:
            config: Database configuration

        Returns:
            Repository type ("database" or "memory")
        """
        if config.profile == DatabaseProfile.MEMORY:
            return "memory"
        else:
            return "database"


# Global configuration manager instance
_config_manager: DatabaseConfigManager | None = None


def get_database_config_manager() -> DatabaseConfigManager:
    """Get global database configuration manager.

    Returns:
        Database configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = DatabaseConfigManager()
    return _config_manager


def get_database_config(
    profile: DatabaseProfile | None = None, **overrides
) -> DatabaseConfig:
    """Get database configuration for the specified profile.

    Args:
        profile: Database profile to use (auto-detected if None)
        **overrides: Configuration overrides

    Returns:
        Database configuration
    """
    return get_database_config_manager().get_database_config(profile, **overrides)


def detect_database_profile() -> DatabaseProfile:
    """Detect appropriate database profile for current environment.

    Returns:
        Database profile
    """
    return get_database_config_manager().detect_environment()


def validate_database_config(config: DatabaseConfig) -> bool:
    """Validate database configuration.

    Args:
        config: Database configuration to validate

    Returns:
        True if configuration is valid
    """
    return get_database_config_manager().validate_config(config)


# Convenience functions for common configurations
def get_development_config(**overrides) -> DatabaseConfig:
    """Get development database configuration."""
    return get_database_config(DatabaseProfile.DEVELOPMENT, **overrides)


def get_testing_config(**overrides) -> DatabaseConfig:
    """Get testing database configuration."""
    return get_database_config(DatabaseProfile.TESTING, **overrides)


def get_production_config(**overrides) -> DatabaseConfig:
    """Get production database configuration."""
    return get_database_config(DatabaseProfile.PRODUCTION, **overrides)


def get_memory_config(**overrides) -> DatabaseConfig:
    """Get in-memory database configuration."""
    return get_database_config(DatabaseProfile.MEMORY, **overrides)
