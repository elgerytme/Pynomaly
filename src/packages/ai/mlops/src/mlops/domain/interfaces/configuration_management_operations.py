"""Domain interfaces for configuration management operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum


class ConfigurationScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"
    SERVICE = "service"
    ENVIRONMENT = "environment"
    USER = "user"
    TENANT = "tenant"


class ConfigurationSource(Enum):
    """Configuration source types."""
    FILE = "file"
    ENVIRONMENT = "environment"
    CONSUL = "consul"
    ETCD = "etcd"
    KUBERNETES = "kubernetes"
    VAULT = "vault"
    DATABASE = "database"
    REMOTE_API = "remote_api"


@dataclass
class ConfigurationValue:
    """Configuration value with metadata."""
    key: str
    value: Any
    scope: ConfigurationScope
    source: ConfigurationSource
    version: int
    created_at: datetime
    updated_at: datetime
    tags: Optional[Dict[str, str]] = None
    encrypted: bool = False
    validation_schema: Optional[str] = None


@dataclass
class ConfigurationChange:
    """Configuration change event."""
    key: str
    old_value: Any
    new_value: Any
    scope: ConfigurationScope
    source: ConfigurationSource
    changed_at: datetime
    changed_by: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ConfigurationQuery:
    """Query for configuration values."""
    key_pattern: Optional[str] = None
    scope: Optional[ConfigurationScope] = None
    source: Optional[ConfigurationSource] = None
    tags: Optional[Dict[str, str]] = None
    include_encrypted: bool = False


@dataclass
class ConfigurationValidation:
    """Configuration validation result."""
    key: str
    is_valid: bool
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None
    validated_at: datetime = None


class ConfigurationProviderPort(ABC):
    """Port for configuration provider operations."""
    
    @abstractmethod
    async def get_configuration(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        default: Any = None
    ) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            scope: Configuration scope
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        pass
    
    @abstractmethod
    async def set_configuration(
        self, 
        key: str, 
        value: Any,
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        tags: Optional[Dict[str, str]] = None,
        encrypt: bool = False
    ) -> bool:
        """Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            scope: Configuration scope
            tags: Optional tags
            encrypt: Whether to encrypt the value
            
        Returns:
            True if set successfully
        """
        pass
    
    @abstractmethod
    async def delete_configuration(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> bool:
        """Delete a configuration value.
        
        Args:
            key: Configuration key
            scope: Configuration scope
            
        Returns:
            True if deleted successfully
        """
        pass
    
    @abstractmethod
    async def list_configurations(
        self, 
        query: ConfigurationQuery
    ) -> List[ConfigurationValue]:
        """List configuration values matching query.
        
        Args:
            query: Configuration query
            
        Returns:
            List of matching configuration values
        """
        pass
    
    @abstractmethod
    async def get_configuration_history(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        limit: int = 50
    ) -> List[ConfigurationChange]:
        """Get configuration change history.
        
        Args:
            key: Configuration key
            scope: Configuration scope
            limit: Maximum number of changes
            
        Returns:
            List of configuration changes
        """
        pass


class ConfigurationWatcherPort(ABC):
    """Port for configuration watching operations."""
    
    @abstractmethod
    async def watch_configuration(
        self, 
        key: str, 
        callback: Callable[[ConfigurationChange], None],
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> str:
        """Watch for configuration changes.
        
        Args:
            key: Configuration key to watch
            callback: Function to call when configuration changes
            scope: Configuration scope
            
        Returns:
            Watch ID for cancelling the watch
        """
        pass
    
    @abstractmethod
    async def watch_configuration_prefix(
        self, 
        prefix: str, 
        callback: Callable[[List[ConfigurationChange]], None],
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> str:
        """Watch for configuration changes with key prefix.
        
        Args:
            prefix: Key prefix to watch
            callback: Function to call when configurations change
            scope: Configuration scope
            
        Returns:
            Watch ID for cancelling the watch
        """
        pass
    
    @abstractmethod
    async def cancel_watch(self, watch_id: str) -> bool:
        """Cancel a configuration watch.
        
        Args:
            watch_id: ID of the watch to cancel
            
        Returns:
            True if watch cancelled successfully
        """
        pass


class ConfigurationValidatorPort(ABC):
    """Port for configuration validation operations."""
    
    @abstractmethod
    async def validate_configuration(
        self, 
        key: str, 
        value: Any, 
        schema: Optional[str] = None
    ) -> ConfigurationValidation:
        """Validate a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value to validate
            schema: Optional validation schema
            
        Returns:
            Validation result
        """
        pass
    
    @abstractmethod
    async def validate_all_configurations(
        self, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> List[ConfigurationValidation]:
        """Validate all configurations in a scope.
        
        Args:
            scope: Configuration scope
            
        Returns:
            List of validation results
        """
        pass
    
    @abstractmethod
    async def register_validation_schema(
        self, 
        key: str, 
        schema: str, 
        schema_type: str = "json_schema"
    ) -> bool:
        """Register a validation schema for a configuration key.
        
        Args:
            key: Configuration key
            schema: Validation schema
            schema_type: Type of schema (json_schema, regex, etc.)
            
        Returns:
            True if registration successful
        """
        pass


class SecretManagementPort(ABC):
    """Port for secret management operations."""
    
    @abstractmethod
    async def store_secret(
        self, 
        key: str, 
        value: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a secret value.
        
        Args:
            key: Secret key
            value: Secret value
            scope: Configuration scope
            metadata: Optional metadata
            
        Returns:
            True if stored successfully
        """
        pass
    
    @abstractmethod
    async def get_secret(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> Optional[str]:
        """Get a secret value.
        
        Args:
            key: Secret key
            scope: Configuration scope
            
        Returns:
            Secret value or None if not found
        """
        pass
    
    @abstractmethod
    async def delete_secret(
        self, 
        key: str, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> bool:
        """Delete a secret value.
        
        Args:
            key: Secret key
            scope: Configuration scope
            
        Returns:
            True if deleted successfully
        """
        pass
    
    @abstractmethod
    async def list_secret_keys(
        self, 
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> List[str]:
        """List secret keys (not values).
        
        Args:
            scope: Configuration scope
            
        Returns:
            List of secret keys
        """
        pass
    
    @abstractmethod
    async def rotate_secret(
        self, 
        key: str, 
        new_value: str,
        scope: ConfigurationScope = ConfigurationScope.GLOBAL
    ) -> bool:
        """Rotate a secret value.
        
        Args:
            key: Secret key
            new_value: New secret value
            scope: Configuration scope
            
        Returns:
            True if rotation successful
        """
        pass


class EnvironmentConfigurationPort(ABC):
    """Port for environment-specific configuration operations."""
    
    @abstractmethod
    async def get_environment_config(
        self, 
        environment: str, 
        service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get configuration for an environment.
        
        Args:
            environment: Environment name (dev, staging, prod)
            service_name: Optional service name filter
            
        Returns:
            Configuration dictionary
        """
        pass
    
    @abstractmethod
    async def set_environment_config(
        self, 
        environment: str, 
        config: Dict[str, Any],
        service_name: Optional[str] = None
    ) -> bool:
        """Set configuration for an environment.
        
        Args:
            environment: Environment name
            config: Configuration dictionary
            service_name: Optional service name
            
        Returns:
            True if set successfully
        """
        pass
    
    @abstractmethod
    async def promote_configuration(
        self, 
        from_environment: str, 
        to_environment: str,
        service_name: Optional[str] = None,
        keys: Optional[List[str]] = None
    ) -> bool:
        """Promote configuration from one environment to another.
        
        Args:
            from_environment: Source environment
            to_environment: Target environment
            service_name: Optional service name filter
            keys: Optional specific keys to promote
            
        Returns:
            True if promotion successful
        """
        pass
    
    @abstractmethod
    async def list_environments(self) -> List[str]:
        """List available environments.
        
        Returns:
            List of environment names
        """
        pass