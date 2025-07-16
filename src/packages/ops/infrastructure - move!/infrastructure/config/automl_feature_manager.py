"""AutoML Feature Manager for runtime feature toggling and management.

This module provides comprehensive AutoML feature management with runtime
toggling, validation, fallback behavior, and environment-based controls.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar, cast

from .feature_flags import (
    get_automl_configuration,
    is_automl_enabled,
    is_automl_feature_enabled,
    validate_automl_environment,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class AutoMLEnvironment(Enum):
    """AutoML environment types for feature control."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class AutoMLFeatureManager:
    """Manages AutoML features with runtime toggling and environment-based controls."""

    def __init__(self):
        """Initialize the AutoML feature manager."""
        self._current_environment = self._detect_environment()
        self._feature_overrides: dict[str, bool] = {}
        self._fallback_strategies: dict[str, Callable] = {}
        self._feature_usage_stats: dict[str, int] = {}
        self._validation_cache: dict[str, bool] = {}

    def _detect_environment(self) -> AutoMLEnvironment:
        """Detect the current environment from environment variables."""
        env_str = os.getenv("PYNOMALY_ENVIRONMENT", "development").lower()

        environment_mapping = {
            "dev": AutoMLEnvironment.DEVELOPMENT,
            "development": AutoMLEnvironment.DEVELOPMENT,
            "test": AutoMLEnvironment.TESTING,
            "testing": AutoMLEnvironment.TESTING,
            "stage": AutoMLEnvironment.STAGING,
            "staging": AutoMLEnvironment.STAGING,
            "prod": AutoMLEnvironment.PRODUCTION,
            "production": AutoMLEnvironment.PRODUCTION,
        }

        return environment_mapping.get(env_str, AutoMLEnvironment.DEVELOPMENT)

    def get_environment(self) -> AutoMLEnvironment:
        """Get the current environment."""
        return self._current_environment

    def set_environment(self, environment: AutoMLEnvironment) -> None:
        """Set the current environment (for testing purposes)."""
        self._current_environment = environment
        self._validation_cache.clear()  # Clear cache when environment changes

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if an AutoML feature is enabled with environment and override logic."""
        # Check for manual overrides first
        if feature_name in self._feature_overrides:
            return self._feature_overrides[feature_name]

        # Check base AutoML enablement
        if not is_automl_enabled():
            return False

        # Apply environment-specific rules
        if not self._is_feature_allowed_in_environment(feature_name):
            return False

        # Check the actual feature flag
        enabled = is_automl_feature_enabled(feature_name)

        # Track usage statistics
        if enabled:
            self._feature_usage_stats[feature_name] = (
                self._feature_usage_stats.get(feature_name, 0) + 1
            )

        return enabled

    def _is_feature_allowed_in_environment(self, feature_name: str) -> bool:
        """Check if a feature is allowed in the current environment."""
        # Production restrictions
        if self._current_environment == AutoMLEnvironment.PRODUCTION:
            experimental_features = {
                "automl_distributed_search",
                "automl_neural_architecture_search",
                "automl_transfer_learning",
            }

            if feature_name in experimental_features:
                logger.warning(
                    f"Feature '{feature_name}' is experimental and disabled in production"
                )
                return False

        # Testing environment allows all features
        if self._current_environment == AutoMLEnvironment.TESTING:
            return True

        # Development and staging allow most features
        return True

    def enable_feature(self, feature_name: str, temporary: bool = False) -> None:
        """Enable a feature at runtime."""
        if temporary:
            self._feature_overrides[feature_name] = True
            logger.info(f"Temporarily enabled AutoML feature: {feature_name}")
        else:
            # This would require updating environment variables or configuration
            logger.warning(
                f"Permanent feature enabling not implemented. "
                f"Set PYNOMALY_{feature_name.upper()}=true instead."
            )

    def disable_feature(self, feature_name: str, temporary: bool = False) -> None:
        """Disable a feature at runtime."""
        if temporary:
            self._feature_overrides[feature_name] = False
            logger.info(f"Temporarily disabled AutoML feature: {feature_name}")
        else:
            logger.warning(
                f"Permanent feature disabling not implemented. "
                f"Set PYNOMALY_{feature_name.upper()}=false instead."
            )

    def clear_overrides(self) -> None:
        """Clear all feature overrides."""
        self._feature_overrides.clear()
        logger.info("Cleared all AutoML feature overrides")

    def register_fallback(self, feature_name: str, fallback_function: Callable) -> None:
        """Register a fallback function for when a feature is disabled."""
        self._fallback_strategies[feature_name] = fallback_function
        logger.debug(f"Registered fallback for feature: {feature_name}")

    def execute_with_fallback(
        self, feature_name: str, primary_function: Callable, *args, **kwargs
    ) -> Any:
        """Execute a function with fallback if the feature is disabled."""
        if self.is_feature_enabled(feature_name):
            try:
                return primary_function(*args, **kwargs)
            except Exception as e:
                logger.error(f"Primary function failed for {feature_name}: {e}")
                if feature_name in self._fallback_strategies:
                    logger.info(f"Using fallback for {feature_name}")
                    return self._fallback_strategies[feature_name](*args, **kwargs)
                raise
        else:
            if feature_name in self._fallback_strategies:
                logger.info(f"Feature {feature_name} disabled, using fallback")
                return self._fallback_strategies[feature_name](*args, **kwargs)
            else:
                raise RuntimeError(
                    f"Feature '{feature_name}' is disabled and no fallback is available. "
                    f"Enable it by setting PYNOMALY_{feature_name.upper()}=true"
                )

    def validate_feature_environment(self, feature_name: str) -> dict[str, Any]:
        """Validate that a specific feature can run in the current environment."""
        if feature_name in self._validation_cache:
            return {"valid": self._validation_cache[feature_name], "cached": True}

        validation_result = {
            "feature_name": feature_name,
            "environment": self._current_environment.value,
            "enabled": self.is_feature_enabled(feature_name),
            "valid": True,
            "warnings": [],
            "errors": [],
        }

        # Check base requirements
        if not is_automl_enabled():
            validation_result["valid"] = False
            validation_result["errors"].append("Base AutoML features are not enabled")

        # Check environment restrictions
        if not self._is_feature_allowed_in_environment(feature_name):
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Feature not allowed in {self._current_environment.value} environment"
            )

        # Check package dependencies
        env_validation = validate_automl_environment()
        for missing_package in env_validation.get("missing_packages", []):
            if feature_name in missing_package:
                validation_result["warnings"].append(
                    f"Missing package: {missing_package}"
                )

        # Cache the result
        self._validation_cache[feature_name] = validation_result["valid"]

        return validation_result

    def get_feature_status(self) -> dict[str, Any]:
        """Get comprehensive status of all AutoML features."""
        config = get_automl_configuration()

        return {
            "environment": self._current_environment.value,
            "base_automl_enabled": is_automl_enabled(),
            "enabled_features": config["enabled_features"],
            "feature_count": config["feature_count"],
            "overrides": dict(self._feature_overrides),
            "usage_stats": dict(self._feature_usage_stats),
            "fallback_count": len(self._fallback_strategies),
            "environment_validation": config["environment_validation"],
            "warnings": config["warnings"],
            "errors": config["errors"],
        }

    def create_feature_context(self, **feature_overrides) -> AutoMLFeatureContext:
        """Create a context manager for temporary feature overrides."""
        return AutoMLFeatureContext(self, feature_overrides)


class AutoMLFeatureContext:
    """Context manager for temporary AutoML feature overrides."""

    def __init__(self, manager: AutoMLFeatureManager, overrides: dict[str, bool]):
        """Initialize the context with feature overrides."""
        self.manager = manager
        self.overrides = overrides
        self.original_overrides: dict[str, bool] = {}

    def __enter__(self) -> AutoMLFeatureContext:
        """Enter the context and apply overrides."""
        # Save original state
        for feature_name in self.overrides:
            if feature_name in self.manager._feature_overrides:
                self.original_overrides[feature_name] = self.manager._feature_overrides[
                    feature_name
                ]

        # Apply new overrides
        self.manager._feature_overrides.update(self.overrides)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and restore original state."""
        # Remove applied overrides
        for feature_name in self.overrides:
            if feature_name in self.original_overrides:
                self.manager._feature_overrides[feature_name] = self.original_overrides[
                    feature_name
                ]
            else:
                self.manager._feature_overrides.pop(feature_name, None)


# Decorators for AutoML feature control
def require_automl_feature(feature_name: str, fallback: Callable | None = None):
    """Decorator to require a specific AutoML feature."""

    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs):
            if not automl_manager.is_feature_enabled(feature_name):
                if fallback:
                    logger.info(f"Feature {feature_name} disabled, using fallback")
                    return fallback(*args, **kwargs)
                raise RuntimeError(
                    f"AutoML feature '{feature_name}' is not enabled. "
                    f"Enable it by setting PYNOMALY_{feature_name.upper()}=true"
                )
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def automl_feature_gate(feature_name: str, default_value: Any = None):
    """Decorator that returns a default value if the feature is disabled."""

    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs):
            if not automl_manager.is_feature_enabled(feature_name):
                logger.debug(
                    f"Feature {feature_name} disabled, returning default: {default_value}"
                )
                return default_value
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def conditional_automl_execution(
    feature_name: str, enabled_func: Callable, disabled_func: Callable | None = None
):
    """Execute different functions based on feature enablement."""
    if automl_manager.is_feature_enabled(feature_name):
        return enabled_func
    elif disabled_func:
        return disabled_func
    else:

        def noop(*args, **kwargs):
            logger.info(f"Feature {feature_name} disabled, no operation performed")
            return None

        return noop


# Global AutoML feature manager instance
automl_manager = AutoMLFeatureManager()


# Convenience functions
def get_automl_manager() -> AutoMLFeatureManager:
    """Get the global AutoML feature manager instance."""
    return automl_manager


def is_automl_feature_available(feature_name: str) -> bool:
    """Check if an AutoML feature is available (enabled and validated)."""
    return (
        automl_manager.is_feature_enabled(feature_name)
        and automl_manager.validate_feature_environment(feature_name)["valid"]
    )


def get_automl_feature_config() -> dict[str, Any]:
    """Get comprehensive AutoML feature configuration."""
    return {
        "manager_status": automl_manager.get_feature_status(),
        "environment": automl_manager.get_environment().value,
        "global_config": get_automl_configuration(),
    }


def enable_automl_feature_temporarily(feature_name: str) -> AutoMLFeatureContext:
    """Enable an AutoML feature temporarily (context manager)."""
    return automl_manager.create_feature_context(**{feature_name: True})


def disable_automl_feature_temporarily(feature_name: str) -> AutoMLFeatureContext:
    """Disable an AutoML feature temporarily (context manager)."""
    return automl_manager.create_feature_context(**{feature_name: False})
