"""Stub implementations for configuration management operations."""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from mlops.domain.interfaces.configuration_management_operations import (
    ConfigurationProviderPort,
    ConfigurationWatcherPort,
    ConfigurationValidatorPort,
    SecretManagementPort,
    EnvironmentConfigurationPort,
    ConfigurationValue,
    ConfigurationChange,
    ConfigurationQuery,
    ConfigurationValidation,
    ConfigurationScope,
    ConfigurationSource
)


class ConfigurationProviderStub(ConfigurationProviderPort):
    """Stub implementation for configuration provider operations."""
    
    async def get_configuration(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        default: Any = None
    ) -> Any:
        """Get a configuration value."""
        return default
    
    async def set_configuration(
        self, 
        key: str, 
        value: Any,
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        tags: Optional[Dict[str, str]] = None,
        encrypt: bool = False
    ) -> bool:
        """Set a configuration value."""
        return True
    
    async def delete_configuration(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> bool:
        """Delete a configuration value."""
        return True
    
    async def list_configurations(
        self, 
        query: ConfigurationQuery
    ) -> List[ConfigurationValue]:
        """List configuration values matching query."""
        return []
    
    async def get_configuration_history(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        limit: int = 50
    ) -> List[ConfigurationChange]:
        """Get configuration change history."""
        return []


class ConfigurationWatcherStub(ConfigurationWatcherPort):
    """Stub implementation for configuration watcher operations."""
    
    async def watch_configuration(
        self, 
        key: str, 
        callback: Callable[[ConfigurationChange], None],
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> str:
        """Watch for configuration changes."""
        return f"watch_{key}"
    
    async def watch_configuration_prefix(
        self, 
        prefix: str, 
        callback: Callable[[List[ConfigurationChange]], None],
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> str:
        """Watch for configuration changes with key prefix."""
        return f"prefix_watch_{prefix}"
    
    async def cancel_watch(self, watch_id: str) -> bool:
        """Cancel a configuration watch."""
        return True


class ConfigurationValidatorStub(ConfigurationValidatorPort):
    """Stub implementation for configuration validator operations."""
    
    async def validate_configuration(
        self, 
        key: str, 
        value: Any, 
        schema: Optional[str] = None
    ) -> ConfigurationValidation:
        """Validate a configuration value."""
        return ConfigurationValidation(
            key=key,
            is_valid=True,
            validated_at=datetime.now()
        )
    
    async def validate_all_configurations(
        self, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> List[ConfigurationValidation]:
        """Validate all configurations in a scope."""
        return []
    
    async def register_validation_schema(
        self, 
        key: str, 
        schema: str, 
        schema_type: str = "json_schema"
    ) -> bool:
        """Register a validation schema for a configuration key."""
        return True


class SecretManagementStub(SecretManagementPort):
    """Stub implementation for secret management operations."""
    
    async def store_secret(
        self, 
        key: str, 
        value: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a secret value."""
        return True
    
    async def get_secret(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> Optional[str]:
        """Get a secret value."""
        return "stub_secret_value"
    
    async def delete_secret(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> bool:
        """Delete a secret value."""
        return True
    
    async def list_secret_keys(
        self, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> List[str]:
        """List secret keys (not values)."""
        return []
    
    async def rotate_secret(
        self, 
        key: str, 
        new_value: str,
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> bool:
        """Rotate a secret value."""
        return True


class EnvironmentConfigurationStub(EnvironmentConfigurationPort):
    """Stub implementation for environment configuration operations."""
    
    async def get_environment_config(
        self, 
        environment: str, 
        service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get configuration for an environment."""
        return {}
    
    async def set_environment_config(
        self, 
        environment: str, 
        config: Dict[str, Any],
        service_name: Optional[str] = None
    ) -> bool:
        """Set configuration for an environment."""
        return True
    
    async def promote_configuration(
        self, 
        from_environment: str, 
        to_environment: str,
        service_name: Optional[str] = None,
        keys: Optional[List[str]] = None
    ) -> bool:
        """Promote configuration from one environment to another."""
        return True
    
    async def list_environments(self) -> List[str]:
        """List available environments."""
        return ["development", "staging", "production"]