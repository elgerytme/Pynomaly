"""Feature flag system for controlled feature rollout.

This module provides a centralized feature flag system that allows toggling
of advanced features without breaking core functionality. Essential for
maintaining simplicity while enabling controlled complexity reintroduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseModel, BaseSettings, Field
    ConfigDict = dict


class FeatureStage(Enum):
    """Development stages for feature maturity."""

    EXPERIMENTAL = "experimental"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"


class FeatureCategory(Enum):
    """Categories for organizing features."""

    CORE = "core"
    ML = "ml"
    API = "api"
    UI = "ui"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    OBSERVABILITY = "observability"


@dataclass
class FeatureFlag:
    """Individual feature flag configuration."""

    name: str
    enabled: bool = False
    stage: FeatureStage = FeatureStage.EXPERIMENTAL
    category: FeatureCategory = FeatureCategory.CORE
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    rollout_percentage: float = 0.0  # 0-100
    user_groups: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureFlagConfig(BaseModel):
    """Configuration for feature flags."""

    flags: dict[str, FeatureFlag] = {}
    default_enabled: bool = False
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )


class FeatureFlagManager:
    """Manages feature flags and their state."""

    def __init__(self, config: FeatureFlagConfig | None = None):
        """Initialize feature flag manager."""
        self.config = config or FeatureFlagConfig()
        self._initialize_default_flags()

    def _initialize_default_flags(self) -> None:
        """Initialize default feature flags."""
        default_flags = {
            "advanced_automl": FeatureFlag(
                name="advanced_automl",
                enabled=False,
                stage=FeatureStage.BETA,
                category=FeatureCategory.ML,
                description="Advanced AutoML capabilities",
                rollout_percentage=10.0
            ),
            "real_time_streaming": FeatureFlag(
                name="real_time_streaming",
                enabled=True,
                stage=FeatureStage.STABLE,
                category=FeatureCategory.CORE,
                description="Real-time data streaming"
            ),
            "graphql_api": FeatureFlag(
                name="graphql_api",
                enabled=False,
                stage=FeatureStage.EXPERIMENTAL,
                category=FeatureCategory.API,
                description="GraphQL API endpoints"
            )
        }
        
        for flag_name, flag in default_flags.items():
            if flag_name not in self.config.flags:
                self.config.flags[flag_name] = flag

    def is_enabled(self, flag_name: str, user_id: str = None) -> bool:
        """Check if a feature flag is enabled."""
        if flag_name not in self.config.flags:
            return self.config.default_enabled
        
        flag = self.config.flags[flag_name]
        
        if not flag.enabled:
            return False
        
        # Check rollout percentage
        if flag.rollout_percentage < 100.0:
            # Simple hash-based rollout
            import hashlib
            hash_input = f"{flag_name}{user_id or 'anonymous'}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
            percentage = (hash_value % 100) + 1
            if percentage > flag.rollout_percentage:
                return False
        
        return True

    def get_flag(self, flag_name: str) -> FeatureFlag | None:
        """Get a feature flag by name."""
        return self.config.flags.get(flag_name)

    def set_flag(self, flag_name: str, enabled: bool) -> None:
        """Set a feature flag state."""
        if flag_name in self.config.flags:
            self.config.flags[flag_name].enabled = enabled

    def add_flag(self, flag: FeatureFlag) -> None:
        """Add a new feature flag."""
        self.config.flags[flag.name] = flag

    def list_flags(self, category: FeatureCategory = None) -> list[FeatureFlag]:
        """List all feature flags, optionally filtered by category."""
        flags = list(self.config.flags.values())
        if category:
            flags = [f for f in flags if f.category == category]
        return flags


# Global feature flag manager instance
_feature_manager: FeatureFlagManager | None = None


def get_feature_manager() -> FeatureFlagManager:
    """Get the global feature flag manager."""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureFlagManager()
    return _feature_manager


def is_feature_enabled(flag_name: str, user_id: str = None) -> bool:
    """Check if a feature is enabled."""
    return get_feature_manager().is_enabled(flag_name, user_id)


def feature_flag(flag_name: str):
    """Decorator to conditionally enable functions based on feature flags."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if is_feature_enabled(flag_name):
                return func(*args, **kwargs)
            else:
                raise RuntimeError(f"Feature '{flag_name}' is not enabled")
        return wrapper
    return decorator
