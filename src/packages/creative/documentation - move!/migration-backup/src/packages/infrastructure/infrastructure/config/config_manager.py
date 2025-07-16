"""Enhanced configuration management system."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from monorepo.domain.exceptions import ConfigurationError

from .settings import Settings

T = TypeVar("T", bound=BaseModel)


class ConfigurationManager:
    """Manages application configuration with validation and environment support."""

    def __init__(
        self,
        config_dir: str | Path = "config",
        environment: str | None = None,
    ) -> None:
        """Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (dev, test, prod, etc.)
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self._settings: Settings | None = None
        self._config_cache: dict[str, Any] = {}

    def load_settings(self) -> Settings:
        """Load and validate application settings."""
        if self._settings is None:
            try:
                self._settings = Settings()
                self._validate_settings(self._settings)
            except ValidationError as e:
                raise ConfigurationError(
                    "Invalid configuration",
                    parameter="settings",
                    actual=str(e),
                ) from e

        return self._settings

    def load_config_file(
        self,
        filename: str,
        config_class: type[T],
        required: bool = True,
    ) -> T | None:
        """Load configuration from file with validation.

        Args:
            filename: Configuration file name
            config_class: Pydantic model class for validation
            required: Whether the file is required

        Returns:
            Validated configuration object or None if not required and missing
        """
        cache_key = f"{filename}:{config_class.__name__}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        config_files = self._get_config_file_paths(filename)

        for config_file in config_files:
            if config_file.exists():
                try:
                    config_data = self._load_file_data(config_file)
                    config_obj = config_class(**config_data)
                    self._config_cache[cache_key] = config_obj
                    return config_obj
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to load configuration from {config_file}",
                        parameter="config_file",
                        expected=str(config_class),
                        actual=str(e),
                    ) from e

        if required:
            raise ConfigurationError(
                f"Required configuration file '{filename}' not found",
                parameter="config_file",
                expected=f"File in {self.config_dir}",
                actual="File not found",
            )

        return None

    def _get_config_file_paths(self, filename: str) -> list[Path]:
        """Get ordered list of configuration file paths to check."""
        base_name = Path(filename).stem
        extension = Path(filename).suffix or ".yaml"

        # Try environment-specific files first, then default
        filenames = [
            f"{base_name}.{self.environment}{extension}",
            f"{base_name}{extension}",
        ]

        return [self.config_dir / fname for fname in filenames]

    def _load_file_data(self, file_path: Path) -> dict[str, Any]:
        """Load data from configuration file."""
        with open(file_path, encoding="utf-8") as f:
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                return yaml.safe_load(f) or {}
            elif file_path.suffix.lower() == ".json":
                return json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {file_path.suffix}",
                    parameter="file_format",
                    expected="yaml, yml, or json",
                    actual=file_path.suffix,
                )

    def _validate_settings(self, settings: Settings) -> None:
        """Validate settings for consistency and requirements."""
        errors = []

        # Validate paths exist or can be created
        for path_name in [
            "storage_path",
            "model_storage_path",
            "experiment_storage_path",
        ]:
            path = getattr(settings, path_name)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create {path_name} '{path}': {e}")

        # Validate database configuration if enabled
        if settings.use_database_repositories and not settings.database_url:
            errors.append(
                "Database repositories enabled but database_url not configured"
            )

        # Validate Redis configuration if caching enabled
        if settings.cache_enabled and not settings.redis_url:
            # This is a warning, not an error - will fall back to memory cache
            pass

        # Validate security settings for production
        if settings.app.environment == "production":
            if settings.security.secret_key == "change-me-in-production":
                errors.append("Secret key must be changed for production environment")

            if not settings.auth_enabled:
                errors.append("Authentication should be enabled in production")

        if errors:
            raise ConfigurationError(
                "Configuration validation failed",
                details={"validation_errors": errors},
            )

    def get_environment_config(self) -> dict[str, Any]:
        """Get environment-specific configuration."""
        return {
            "environment": self.environment,
            "debug": self.environment != "production",
            "testing": self.environment == "test",
            "development": self.environment == "development",
            "production": self.environment == "production",
        }

    def override_setting(self, key: str, value: Any) -> None:
        """Override a setting value (useful for testing)."""
        if self._settings is None:
            self._settings = self.load_settings()

        # Navigate nested attributes
        obj = self._settings
        keys = key.split(".")

        for k in keys[:-1]:
            obj = getattr(obj, k)

        setattr(obj, keys[-1], value)

        # Clear cache to force revalidation
        self._config_cache.clear()

    def export_config(self, include_secrets: bool = False) -> dict[str, Any]:
        """Export current configuration as dictionary."""
        settings = self.load_settings()
        config_dict = settings.model_dump()

        if not include_secrets:
            # Remove sensitive information
            sensitive_keys = ["secret_key", "database_url", "redis_url"]
            for key in sensitive_keys:
                if key in config_dict:
                    config_dict[key] = "***REDACTED***"

        return config_dict

    def validate_config_schema(self, config_data: dict[str, Any]) -> list[str]:
        """Validate configuration against schema."""
        try:
            Settings(**config_data)
            return []
        except ValidationError as e:
            return [f"{error['loc'][0]}: {error['msg']}" for error in e.errors()]


class DatabaseConfigManager:
    """Specialized manager for database configuration."""

    def __init__(self, settings: Settings):
        """Initialize database config manager."""
        self.settings = settings

    def get_database_config(self) -> dict[str, Any]:
        """Get database configuration dictionary."""
        if not self.settings.database_url:
            return {}

        return {
            "url": self.settings.database_url,
            "pool_size": self.settings.database_pool_size,
            "max_overflow": self.settings.database_max_overflow,
            "pool_timeout": self.settings.database_pool_timeout,
            "pool_recycle": self.settings.database_pool_recycle,
            "echo": self.settings.database_echo,
            "echo_pool": self.settings.database_echo_pool,
        }

    def get_alembic_config(self) -> dict[str, Any]:
        """Get Alembic migration configuration."""
        return {
            "sqlalchemy.url": self.settings.database_url,
            "script_location": "migrations",
            "file_template": "%%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s",
        }


class CacheConfigManager:
    """Specialized manager for cache configuration."""

    def __init__(self, settings: Settings):
        """Initialize cache config manager."""
        self.settings = settings

    def get_cache_config(self) -> dict[str, Any]:
        """Get cache configuration."""
        config = {
            "enabled": self.settings.cache_enabled,
            "default_ttl": self.settings.cache_ttl,
        }

        if self.settings.redis_url:
            config["backend"] = "redis"
            config["redis_url"] = self.settings.redis_url
        else:
            config["backend"] = "memory"

        return config


class MonitoringConfigManager:
    """Specialized manager for monitoring configuration."""

    def __init__(self, settings: Settings):
        """Initialize monitoring config manager."""
        self.settings = settings

    def get_prometheus_config(self) -> dict[str, Any]:
        """Get Prometheus configuration."""
        return {
            "enabled": self.settings.monitoring.prometheus_enabled,
            "port": self.settings.monitoring.prometheus_port,
            "metrics_path": "/metrics",
        }

    def get_logging_config(self) -> dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": self.settings.monitoring.log_level,
            "format": self.settings.monitoring.log_format,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.settings.monitoring.log_level,
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": self.settings.log_path / "monorepo.log",
                    "level": self.settings.monitoring.log_level,
                },
            },
        }


def create_config_manager(
    config_dir: str | Path = "config",
    environment: str | None = None,
) -> ConfigurationManager:
    """Create and configure a configuration manager."""
    return ConfigurationManager(config_dir=config_dir, environment=environment)
