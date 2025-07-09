"""Enhanced configuration loader with comprehensive architecture support."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, Field, ValidationError

from pynomaly.domain.abstractions import BaseValueObject, Specification
from pynomaly.domain.exceptions import ConfigurationError

T = TypeVar("T", bound=BaseModel)


class ConfigurationFormat(Enum):
    """Supported configuration formats."""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    ENV = "env"
    INI = "ini"


class ConfigurationSource(Enum):
    """Configuration sources."""
    FILE = "file"
    ENVIRONMENT = "environment"
    REMOTE = "remote"
    DATABASE = "database"
    VAULT = "vault"


class ConfigurationContext(BaseValueObject):
    """Context for configuration loading."""
    
    environment: str = Field(default="development", description="Environment name")
    profile: Optional[str] = Field(None, description="Configuration profile")
    namespace: Optional[str] = Field(None, description="Configuration namespace")
    version: Optional[str] = Field(None, description="Configuration version")
    tenant: Optional[str] = Field(None, description="Tenant identifier")
    
    def get_cache_key(self) -> str:
        """Generate cache key for this context."""
        parts = [self.environment]
        if self.profile:
            parts.append(self.profile)
        if self.namespace:
            parts.append(self.namespace)
        if self.version:
            parts.append(self.version)
        if self.tenant:
            parts.append(self.tenant)
        return ":".join(parts)


class ConfigurationSource(BaseValueObject):
    """Configuration source information."""
    
    source_type: ConfigurationSource
    location: str
    format: ConfigurationFormat
    priority: int = Field(default=100, description="Source priority (lower = higher priority)")
    enabled: bool = Field(default=True, description="Whether source is enabled")
    cache_ttl: Optional[int] = Field(None, description="Cache TTL in seconds")
    
    def __lt__(self, other: "ConfigurationSource") -> bool:
        """Compare sources by priority."""
        return self.priority < other.priority


class ConfigurationLoader(ABC, Generic[T]):
    """Abstract base class for configuration loaders."""
    
    def __init__(self, config_class: Type[T]) -> None:
        """Initialize configuration loader.
        
        Args:
            config_class: Configuration class to load
        """
        self.config_class = config_class
        self._cache: Dict[str, T] = {}
        self._sources: List[ConfigurationSource] = []
    
    @abstractmethod
    def load_from_source(self, source: ConfigurationSource, context: ConfigurationContext) -> Dict[str, Any]:
        """Load configuration from a source.
        
        Args:
            source: Configuration source
            context: Configuration context
            
        Returns:
            Configuration data
        """
        ...
    
    @abstractmethod
    def validate_configuration(self, config_data: Dict[str, Any]) -> None:
        """Validate configuration data.
        
        Args:
            config_data: Configuration data to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        ...
    
    def add_source(self, source: ConfigurationSource) -> None:
        """Add a configuration source.
        
        Args:
            source: Configuration source to add
        """
        self._sources.append(source)
        self._sources.sort()  # Sort by priority
    
    def load_configuration(self, context: ConfigurationContext) -> T:
        """Load configuration from all sources.
        
        Args:
            context: Configuration context
            
        Returns:
            Loaded configuration
        """
        cache_key = context.get_cache_key()
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Merge configuration from all sources
        merged_config = {}
        
        for source in self._sources:
            if not source.enabled:
                continue
                
            try:
                source_config = self.load_from_source(source, context)
                merged_config = self._merge_configurations(merged_config, source_config)
            except Exception as e:
                # Log error but continue with other sources
                print(f"Error loading from source {source.location}: {e}")
        
        # Validate merged configuration
        self.validate_configuration(merged_config)
        
        # Create configuration instance
        try:
            config = self.config_class(**merged_config)
        except ValidationError as e:
            raise ConfigurationError(
                f"Configuration validation failed: {e}",
                parameter="configuration",
                actual=str(e),
            )
        
        # Cache the configuration
        self._cache[cache_key] = config
        
        return config
    
    def _merge_configurations(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configurations(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._cache.clear()
    
    def reload_configuration(self, context: ConfigurationContext) -> T:
        """Reload configuration from sources.
        
        Args:
            context: Configuration context
            
        Returns:
            Reloaded configuration
        """
        cache_key = context.get_cache_key()
        self._cache.pop(cache_key, None)
        return self.load_configuration(context)


class FileConfigurationLoader(ConfigurationLoader[T]):
    """Configuration loader for file-based sources."""
    
    def __init__(self, config_class: Type[T], config_dir: Optional[Path] = None) -> None:
        """Initialize file configuration loader.
        
        Args:
            config_class: Configuration class to load
            config_dir: Configuration directory
        """
        super().__init__(config_class)
        self.config_dir = config_dir or Path("config")
    
    def load_from_source(self, source: ConfigurationSource, context: ConfigurationContext) -> Dict[str, Any]:
        """Load configuration from file source."""
        file_path = self._resolve_file_path(source, context)
        
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if source.format == ConfigurationFormat.YAML:
                    return yaml.safe_load(f) or {}
                elif source.format == ConfigurationFormat.JSON:
                    return json.load(f)
                elif source.format == ConfigurationFormat.TOML:
                    import tomllib
                    return tomllib.load(f)
                else:
                    raise ConfigurationError(f"Unsupported format: {source.format}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {file_path}: {e}")
    
    def _resolve_file_path(self, source: ConfigurationSource, context: ConfigurationContext) -> Path:
        """Resolve file path based on context."""
        base_path = Path(source.location)
        
        if not base_path.is_absolute():
            base_path = self.config_dir / base_path
        
        # Add environment-specific suffixes
        if context.environment and context.environment != "default":
            env_path = base_path.with_suffix(f".{context.environment}{base_path.suffix}")
            if env_path.exists():
                return env_path
        
        return base_path
    
    def validate_configuration(self, config_data: Dict[str, Any]) -> None:
        """Validate configuration data."""
        # Basic validation - can be extended
        if not isinstance(config_data, dict):
            raise ConfigurationError("Configuration must be a dictionary")


class EnvironmentConfigurationLoader(ConfigurationLoader[T]):
    """Configuration loader for environment variables."""
    
    def __init__(self, config_class: Type[T], prefix: str = "PYNOMALY_") -> None:
        """Initialize environment configuration loader.
        
        Args:
            config_class: Configuration class to load
            prefix: Environment variable prefix
        """
        super().__init__(config_class)
        self.prefix = prefix
    
    def load_from_source(self, source: ConfigurationSource, context: ConfigurationContext) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config_data = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                config_key = key[len(self.prefix):].lower()
                config_data[config_key] = self._parse_env_value(value)
        
        return config_data
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try to parse as boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Try to parse as number
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def validate_configuration(self, config_data: Dict[str, Any]) -> None:
        """Validate configuration data."""
        # Basic validation for environment variables
        pass


class RemoteConfigurationLoader(ConfigurationLoader[T]):
    """Configuration loader for remote sources."""
    
    def __init__(self, config_class: Type[T], base_url: str, auth_token: Optional[str] = None) -> None:
        """Initialize remote configuration loader.
        
        Args:
            config_class: Configuration class to load
            base_url: Base URL for remote configuration
            auth_token: Authentication token
        """
        super().__init__(config_class)
        self.base_url = base_url
        self.auth_token = auth_token
    
    def load_from_source(self, source: ConfigurationSource, context: ConfigurationContext) -> Dict[str, Any]:
        """Load configuration from remote source."""
        import requests
        
        url = f"{self.base_url}/{source.location}"
        headers = {}
        
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            if source.format == ConfigurationFormat.JSON:
                return response.json()
            elif source.format == ConfigurationFormat.YAML:
                return yaml.safe_load(response.text) or {}
            else:
                raise ConfigurationError(f"Unsupported format: {source.format}")
        
        except requests.RequestException as e:
            raise ConfigurationError(f"Failed to load remote configuration: {e}")
    
    def validate_configuration(self, config_data: Dict[str, Any]) -> None:
        """Validate configuration data."""
        if not isinstance(config_data, dict):
            raise ConfigurationError("Remote configuration must be a dictionary")


class HierarchicalConfigurationLoader(ConfigurationLoader[T]):
    """Configuration loader that supports hierarchical configuration."""
    
    def __init__(self, config_class: Type[T], config_dir: Optional[Path] = None) -> None:
        """Initialize hierarchical configuration loader.
        
        Args:
            config_class: Configuration class to load
            config_dir: Configuration directory
        """
        super().__init__(config_class)
        self.config_dir = config_dir or Path("config")
        self._setup_default_sources()
    
    def _setup_default_sources(self) -> None:
        """Setup default configuration sources."""
        # Default configuration
        self.add_source(ConfigurationSource(
            source_type=ConfigurationSource.FILE,
            location="default.yaml",
            format=ConfigurationFormat.YAML,
            priority=100
        ))
        
        # Environment-specific configuration
        env = os.getenv("ENVIRONMENT", "development")
        self.add_source(ConfigurationSource(
            source_type=ConfigurationSource.FILE,
            location=f"{env}.yaml",
            format=ConfigurationFormat.YAML,
            priority=50
        ))
        
        # Local overrides
        self.add_source(ConfigurationSource(
            source_type=ConfigurationSource.FILE,
            location="local.yaml",
            format=ConfigurationFormat.YAML,
            priority=10
        ))
        
        # Environment variables
        self.add_source(ConfigurationSource(
            source_type=ConfigurationSource.ENVIRONMENT,
            location="env",
            format=ConfigurationFormat.ENV,
            priority=1
        ))
    
    def load_from_source(self, source: ConfigurationSource, context: ConfigurationContext) -> Dict[str, Any]:
        """Load configuration from source."""
        if source.source_type == ConfigurationSource.FILE:
            loader = FileConfigurationLoader(self.config_class, self.config_dir)
            return loader.load_from_source(source, context)
        elif source.source_type == ConfigurationSource.ENVIRONMENT:
            loader = EnvironmentConfigurationLoader(self.config_class)
            return loader.load_from_source(source, context)
        elif source.source_type == ConfigurationSource.REMOTE:
            # Would need to be configured with remote settings
            raise ConfigurationError("Remote configuration not configured")
        else:
            raise ConfigurationError(f"Unsupported source type: {source.source_type}")
    
    def validate_configuration(self, config_data: Dict[str, Any]) -> None:
        """Validate configuration data."""
        if not isinstance(config_data, dict):
            raise ConfigurationError("Configuration must be a dictionary")


class ConfigurationManager:
    """Central configuration manager."""
    
    def __init__(self, loader: ConfigurationLoader[T]) -> None:
        """Initialize configuration manager.
        
        Args:
            loader: Configuration loader
        """
        self.loader = loader
        self._watchers: List[Any] = []
    
    def get_configuration(self, context: Optional[ConfigurationContext] = None) -> T:
        """Get configuration.
        
        Args:
            context: Configuration context
            
        Returns:
            Configuration instance
        """
        if context is None:
            context = ConfigurationContext()
        
        return self.loader.load_configuration(context)
    
    def reload_configuration(self, context: Optional[ConfigurationContext] = None) -> T:
        """Reload configuration.
        
        Args:
            context: Configuration context
            
        Returns:
            Reloaded configuration
        """
        if context is None:
            context = ConfigurationContext()
        
        return self.loader.reload_configuration(context)
    
    def add_configuration_watcher(self, watcher: Any) -> None:
        """Add configuration watcher.
        
        Args:
            watcher: Configuration watcher
        """
        self._watchers.append(watcher)
    
    def remove_configuration_watcher(self, watcher: Any) -> None:
        """Remove configuration watcher.
        
        Args:
            watcher: Configuration watcher to remove
        """
        if watcher in self._watchers:
            self._watchers.remove(watcher)
    
    def notify_watchers(self, config: T) -> None:
        """Notify watchers of configuration change.
        
        Args:
            config: New configuration
        """
        for watcher in self._watchers:
            try:
                watcher.on_configuration_changed(config)
            except Exception as e:
                print(f"Error notifying watcher: {e}")


class ConfigurationSpecification(Specification[Dict[str, Any]]):
    """Specification for configuration validation."""
    
    def __init__(self, field_name: str, validator: callable) -> None:
        """Initialize configuration specification.
        
        Args:
            field_name: Field name to validate
            validator: Validation function
        """
        self.field_name = field_name
        self.validator = validator
    
    def is_satisfied_by(self, config: Dict[str, Any]) -> bool:
        """Check if configuration satisfies specification."""
        if self.field_name not in config:
            return False
        
        return self.validator(config[self.field_name])


def create_configuration_manager(config_class: Type[T], config_dir: Optional[Path] = None) -> ConfigurationManager[T]:
    """Create a configuration manager with hierarchical loading.
    
    Args:
        config_class: Configuration class
        config_dir: Configuration directory
        
    Returns:
        Configuration manager
    """
    loader = HierarchicalConfigurationLoader(config_class, config_dir)
    return ConfigurationManager(loader)
