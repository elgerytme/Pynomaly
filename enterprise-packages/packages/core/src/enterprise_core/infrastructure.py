"""Infrastructure components for enterprise applications.

This module provides dependency injection, configuration management,
and service registry patterns for building robust enterprise systems.
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from dependency_injector import containers, providers
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .protocols import (
    ConfigurationProvider,
    FeatureFlag,
    HealthCheck,
    HealthStatus,
    Logger,
)

T = TypeVar("T")


class EnterpriseSettings(BaseSettings):
    """Base settings class for enterprise applications."""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
        "case_sensitive": False,
    }

    # Environment configuration
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Logging format")

    # Service configuration
    service_name: str = Field(default="enterprise-service", description="Service name")
    service_version: str = Field(default="1.0.0", description="Service version")

    # Health check configuration
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")

    @classmethod
    def load_from_file(cls, config_file: Union[str, Path]) -> EnterpriseSettings:
        """Load settings from a configuration file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load environment variables from file
        import os
        if config_path.suffix.lower() == ".env":
            from dotenv import load_dotenv
            load_dotenv(config_path)

        return cls()


class ConfigurationManager:
    """Configuration manager with environment-specific settings."""

    def __init__(self, settings: Optional[EnterpriseSettings] = None) -> None:
        self._settings = settings or EnterpriseSettings()
        self._config_cache: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with dot notation support."""
        if key in self._config_cache:
            return self._config_cache[key]

        # Try to get from settings first
        try:
            value = getattr(self._settings, key.replace(".", "_"), None)
            if value is not None:
                self._config_cache[key] = value
                return value
        except AttributeError:
            pass

        # Try environment variable
        env_key = key.upper().replace(".", "_")
        value = os.getenv(env_key, default)
        self._config_cache[key] = value
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a configuration section."""
        section_dict = {}
        prefix = f"{section}."

        # Get from settings
        for field_name, field_info in self._settings.model_fields.items():
            if field_name.startswith(section.replace(".", "_")):
                key = field_name.replace("_", ".")
                if key.startswith(prefix):
                    section_dict[key[len(prefix):]] = getattr(self._settings, field_name)

        # Get from environment variables
        for key, value in os.environ.items():
            env_key = key.lower().replace("_", ".")
            if env_key.startswith(prefix):
                section_dict[env_key[len(prefix):]] = value

        return section_dict

    def has(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return self.get(key) is not None

    @property
    def environment(self) -> str:
        """Get the current environment."""
        return self._settings.environment

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


class FeatureFlagManager:
    """Feature flag manager for controlling feature rollouts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._flags: Dict[str, Dict[str, Any]] = config or {}
        self._config_manager = ConfigurationManager()

    def is_enabled(self, flag: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature flag is enabled."""
        # Check environment variable first
        env_flag = f"FEATURE_{flag.upper()}"
        env_value = os.getenv(env_flag)
        if env_value is not None:
            return env_value.lower() in ("true", "1", "yes", "on")

        # Check configuration
        flag_config = self._flags.get(flag, {})
        if not flag_config:
            return False

        # Simple boolean flag
        if isinstance(flag_config, bool):
            return flag_config

        # Dictionary configuration with conditions
        if isinstance(flag_config, dict):
            enabled = flag_config.get("enabled", False)

            # Check environment-specific flags
            environment = self._config_manager.environment
            env_config = flag_config.get("environments", {})
            if environment in env_config:
                enabled = env_config[environment]

            # Check context-based conditions
            if context and "conditions" in flag_config:
                for condition in flag_config["conditions"]:
                    if self._evaluate_condition(condition, context):
                        enabled = condition.get("enabled", enabled)

            return enabled

        return False

    def get_variant(self, flag: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Get the variant of a feature flag."""
        flag_config = self._flags.get(flag, {})
        if not isinstance(flag_config, dict):
            return "control"

        variants = flag_config.get("variants", {})
        if not variants:
            return "control"

        # Simple case: return default variant
        default_variant = flag_config.get("default_variant", "control")

        # Context-based variant selection
        if context and "variant_conditions" in flag_config:
            for condition in flag_config["variant_conditions"]:
                if self._evaluate_condition(condition, context):
                    return condition.get("variant", default_variant)

        return default_variant

    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a condition against the given context."""
        condition_type = condition.get("type", "equals")
        field = condition.get("field")
        value = condition.get("value")

        if not field or field not in context:
            return False

        context_value = context[field]

        if condition_type == "equals":
            return context_value == value
        elif condition_type == "in":
            return context_value in value
        elif condition_type == "greater_than":
            return context_value > value
        elif condition_type == "less_than":
            return context_value < value
        elif condition_type == "contains":
            return value in str(context_value)

        return False

    def set_flag(self, flag: str, config: Union[bool, Dict[str, Any]]) -> None:
        """Set a feature flag configuration."""
        self._flags[flag] = config

    def remove_flag(self, flag: str) -> None:
        """Remove a feature flag."""
        self._flags.pop(flag, None)


class ServiceRegistry:
    """Service registry for managing service instances and their lifecycles."""

    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
        self._service_configs: Dict[str, Dict[str, Any]] = {}
        self._health_checks: Dict[str, HealthCheck] = {}

    def register(
        self,
        name: str,
        service: Any,
        config: Optional[Dict[str, Any]] = None,
        health_check: Optional[HealthCheck] = None,
    ) -> None:
        """Register a service with the registry."""
        self._services[name] = service
        self._service_configs[name] = config or {}

        if health_check:
            self._health_checks[name] = health_check

    def get(self, name: str) -> Optional[Any]:
        """Get a service from the registry."""
        return self._services.get(name)

    def get_required(self, name: str) -> Any:
        """Get a required service from the registry, raising an error if not found."""
        service = self.get(name)
        if service is None:
            raise ServiceNotFoundError(f"Required service '{name}' not found in registry")
        return service

    def remove(self, name: str) -> None:
        """Remove a service from the registry."""
        self._services.pop(name, None)
        self._service_configs.pop(name, None)
        self._health_checks.pop(name, None)

    def list_services(self) -> List[str]:
        """List all registered service names."""
        return list(self._services.keys())

    def get_config(self, name: str) -> Dict[str, Any]:
        """Get the configuration for a service."""
        return self._service_configs.get(name, {})

    async def health_check(self, name: Optional[str] = None) -> Dict[str, HealthStatus]:
        """Perform health checks on services."""
        results = {}

        if name:
            # Check specific service
            if name in self._health_checks:
                try:
                    results[name] = await self._health_checks[name].check()
                except Exception as e:
                    results[name] = HealthStatus(
                        status="unhealthy",
                        message=f"Health check failed: {e}",
                        details={"error": str(e)},
                    )
            else:
                results[name] = HealthStatus(
                    status="unknown",
                    message="No health check configured",
                )
        else:
            # Check all services
            for service_name, health_check in self._health_checks.items():
                try:
                    results[service_name] = await health_check.check()
                except Exception as e:
                    results[service_name] = HealthStatus(
                        status="unhealthy",
                        message=f"Health check failed: {e}",
                        details={"error": str(e)},
                    )

        return results


class Container(containers.DeclarativeContainer):
    """Dependency injection container for enterprise applications."""

    # Configuration
    config = providers.Configuration()

    # Core services
    configuration_manager = providers.Singleton(
        ConfigurationManager,
        settings=None,
    )

    feature_flag_manager = providers.Singleton(
        FeatureFlagManager,
        config=config.feature_flags,
    )

    service_registry = providers.Singleton(ServiceRegistry)

    @classmethod
    def create_from_config(cls, config_file: Union[str, Path]) -> Container:
        """Create a container from a configuration file."""
        container = cls()

        # Load configuration
        settings = EnterpriseSettings.load_from_file(config_file)
        container.configuration_manager.override(
            providers.Singleton(ConfigurationManager, settings=settings)
        )

        return container


# Decorators for service management
def service(name: str, container: Optional[Container] = None) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a service with the container."""
    def decorator(cls: Type[T]) -> Type[T]:
        if container:
            # Register with specific container
            provider = providers.Singleton(cls)
            setattr(container, name.replace("-", "_"), provider)

        # Add metadata to class
        cls._service_name = name  # type: ignore
        return cls

    return decorator


def inject(container: Container) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to inject dependencies into a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Inject container services into kwargs
            for provider_name in dir(container):
                if not provider_name.startswith("_") and provider_name not in kwargs:
                    provider = getattr(container, provider_name)
                    if hasattr(provider, "provided"):
                        kwargs[provider_name] = provider()

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Exceptions
class ServiceNotFoundError(Exception):
    """Raised when a required service is not found in the registry."""
    pass


class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""
    pass
