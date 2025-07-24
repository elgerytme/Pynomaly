"""Shared configuration framework using Pydantic BaseSettings.

This module provides standardized configuration management across all packages
in the monorepo, implementing the infrastructure standardization recommendations.
"""

from __future__ import annotations

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from abc import ABC, abstractmethod

try:
    from pydantic import BaseSettings, Field, validator
    from pydantic_settings import BaseSettings as PydanticBaseSettings
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for environments without Pydantic
    PYDANTIC_AVAILABLE = False
    from dataclasses import dataclass, field
    
    class BaseSettings:
        """Fallback BaseSettings when Pydantic is not available."""
        pass

T = TypeVar('T', bound='BasePackageSettings')


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


class DatabaseSettings(BaseSettings if PYDANTIC_AVAILABLE else object):
    """Standardized database configuration."""
    
    if PYDANTIC_AVAILABLE:
        host: str = Field(default="localhost", env="DB_HOST")
        port: int = Field(default=5432, env="DB_PORT")
        database: str = Field(default="app_db", env="DB_DATABASE") 
        username: str = Field(default="postgres", env="DB_USERNAME")
        password: str = Field(default="", env="DB_PASSWORD")
        
        # Connection pool settings
        pool_size: int = Field(default=10, env="DB_POOL_SIZE")
        max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
        pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
        pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
        
        # Advanced settings
        async_mode: bool = Field(default=True, env="DB_ASYNC_MODE")
        echo_sql: bool = Field(default=False, env="DB_ECHO_SQL")
        ssl_mode: str = Field(default="prefer", env="DB_SSL_MODE")
        
        class Config:
            env_prefix = ""
            case_sensitive = False
            
        @validator('port')
        def validate_port(cls, v):
            if not 1 <= v <= 65535:
                raise ValueError('Port must be between 1 and 65535')
            return v
            
        @property
        def url(self) -> str:
            """Get database URL."""
            if self.async_mode:
                driver = "postgresql+asyncpg"
            else:
                driver = "postgresql"
                
            if self.password:
                return f"{driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            else:
                return f"{driver}://{self.username}@{self.host}:{self.port}/{self.database}"
                
        @property
        def sync_url(self) -> str:
            """Get synchronous database URL."""
            if self.password:
                return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            else:
                return f"postgresql://{self.username}@{self.host}:{self.port}/{self.database}"
    else:
        def __init__(self):
            self.host = os.getenv("DB_HOST", "localhost")
            self.port = int(os.getenv("DB_PORT", "5432"))
            self.database = os.getenv("DB_DATABASE", "app_db")
            self.username = os.getenv("DB_USERNAME", "postgres")  
            self.password = os.getenv("DB_PASSWORD", "")
            self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
            self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
            self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
            self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
            self.async_mode = os.getenv("DB_ASYNC_MODE", "true").lower() == "true"
            self.echo_sql = os.getenv("DB_ECHO_SQL", "false").lower() == "true"
            self.ssl_mode = os.getenv("DB_SSL_MODE", "prefer")
            
        @property
        def url(self) -> str:
            if self.async_mode:
                driver = "postgresql+asyncpg"
            else:
                driver = "postgresql"
                
            if self.password:
                return f"{driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            else:
                return f"{driver}://{self.username}@{self.host}:{self.port}/{self.database}"


class LoggingSettings(BaseSettings if PYDANTIC_AVAILABLE else object):
    """Standardized logging configuration."""
    
    if PYDANTIC_AVAILABLE:
        level: str = Field(default="INFO", env="LOG_LEVEL")
        format: str = Field(default="json", env="LOG_FORMAT") 
        output: str = Field(default="console", env="LOG_OUTPUT")
        
        # File logging
        file_enabled: bool = Field(default=False, env="LOG_FILE_ENABLED")
        file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
        max_file_size: str = Field(default="10MB", env="LOG_MAX_FILE_SIZE")
        backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
        
        # Structured logging features
        enable_structured_logging: bool = Field(default=True, env="LOG_STRUCTURED")
        enable_request_tracking: bool = Field(default=True, env="LOG_REQUEST_TRACKING")
        enable_performance_logging: bool = Field(default=True, env="LOG_PERFORMANCE")
        enable_error_tracking: bool = Field(default=True, env="LOG_ERROR_TRACKING")
        
        # Performance thresholds
        slow_query_threshold_ms: float = Field(default=1000.0, env="LOG_SLOW_QUERY_THRESHOLD_MS")
        slow_operation_threshold_ms: float = Field(default=5000.0, env="LOG_SLOW_OPERATION_THRESHOLD_MS")
        
        # Error handling
        log_stack_traces: bool = Field(default=True, env="LOG_STACK_TRACES")
        include_request_context: bool = Field(default=True, env="LOG_REQUEST_CONTEXT")
        sanitize_sensitive_data: bool = Field(default=True, env="LOG_SANITIZE_DATA")
        
        class Config:
            env_prefix = ""
            case_sensitive = False
            
        @validator('level')
        def validate_log_level(cls, v):
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if v.upper() not in valid_levels:
                raise ValueError(f'Log level must be one of: {valid_levels}')
            return v.upper()
            
        @validator('format')
        def validate_format(cls, v):
            valid_formats = ['json', 'text', 'structured']
            if v.lower() not in valid_formats:
                raise ValueError(f'Log format must be one of: {valid_formats}')
            return v.lower()
    else:
        def __init__(self):
            self.level = os.getenv("LOG_LEVEL", "INFO").upper()
            self.format = os.getenv("LOG_FORMAT", "json").lower()
            self.output = os.getenv("LOG_OUTPUT", "console").lower()
            self.file_enabled = os.getenv("LOG_FILE_ENABLED", "false").lower() == "true"
            self.file_path = os.getenv("LOG_FILE_PATH")
            self.max_file_size = os.getenv("LOG_MAX_FILE_SIZE", "10MB")
            self.backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
            self.enable_structured_logging = os.getenv("LOG_STRUCTURED", "true").lower() == "true"
            self.enable_request_tracking = os.getenv("LOG_REQUEST_TRACKING", "true").lower() == "true"
            self.enable_performance_logging = os.getenv("LOG_PERFORMANCE", "true").lower() == "true"
            self.enable_error_tracking = os.getenv("LOG_ERROR_TRACKING", "true").lower() == "true"
            self.slow_query_threshold_ms = float(os.getenv("LOG_SLOW_QUERY_THRESHOLD_MS", "1000"))
            self.slow_operation_threshold_ms = float(os.getenv("LOG_SLOW_OPERATION_THRESHOLD_MS", "5000"))
            self.log_stack_traces = os.getenv("LOG_STACK_TRACES", "true").lower() == "true"
            self.include_request_context = os.getenv("LOG_REQUEST_CONTEXT", "true").lower() == "true"
            self.sanitize_sensitive_data = os.getenv("LOG_SANITIZE_DATA", "true").lower() == "true"


class APISettings(BaseSettings if PYDANTIC_AVAILABLE else object):
    """Standardized API configuration."""
    
    if PYDANTIC_AVAILABLE:
        host: str = Field(default="0.0.0.0", env="API_HOST")
        port: int = Field(default=8000, env="API_PORT")
        workers: int = Field(default=1, env="API_WORKERS")
        reload: bool = Field(default=False, env="API_RELOAD")
        debug: bool = Field(default=False, env="API_DEBUG")
        
        # CORS settings
        cors_origins: List[str] = Field(default=["*"], env="API_CORS_ORIGINS")
        cors_credentials: bool = Field(default=True, env="API_CORS_CREDENTIALS")
        cors_methods: List[str] = Field(default=["*"], env="API_CORS_METHODS")
        cors_headers: List[str] = Field(default=["*"], env="API_CORS_HEADERS")
        
        # Rate limiting
        rate_limit_enabled: bool = Field(default=True, env="API_RATE_LIMIT_ENABLED")
        requests_per_minute: int = Field(default=60, env="API_REQUESTS_PER_MINUTE")
        requests_per_hour: int = Field(default=1000, env="API_REQUESTS_PER_HOUR")
        burst_limit: int = Field(default=10, env="API_BURST_LIMIT")
        
        # Security
        trusted_hosts: List[str] = Field(default=["*"], env="API_TRUSTED_HOSTS")
        max_request_size: int = Field(default=16777216, env="API_MAX_REQUEST_SIZE")  # 16MB
        
        class Config:
            env_prefix = ""
            case_sensitive = False
            
        @validator('port')
        def validate_port(cls, v):
            if not 1 <= v <= 65535:
                raise ValueError('Port must be between 1 and 65535')
            return v
            
        @validator('workers')
        def validate_workers(cls, v):
            if v < 1:
                raise ValueError('Workers must be at least 1')
            return v
            
        @validator('cors_origins', pre=True)
        def parse_cors_origins(cls, v):
            if isinstance(v, str):
                return [origin.strip() for origin in v.split(",")]
            return v
    else:
        def __init__(self):
            self.host = os.getenv("API_HOST", "0.0.0.0")
            self.port = int(os.getenv("API_PORT", "8000"))
            self.workers = int(os.getenv("API_WORKERS", "1"))
            self.reload = os.getenv("API_RELOAD", "false").lower() == "true"
            self.debug = os.getenv("API_DEBUG", "false").lower() == "true"
            
            cors_origins_str = os.getenv("API_CORS_ORIGINS", "*")
            self.cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]
            self.cors_credentials = os.getenv("API_CORS_CREDENTIALS", "true").lower() == "true"
            
            cors_methods_str = os.getenv("API_CORS_METHODS", "*")
            self.cors_methods = [method.strip() for method in cors_methods_str.split(",")]
            
            cors_headers_str = os.getenv("API_CORS_HEADERS", "*")
            self.cors_headers = [header.strip() for header in cors_headers_str.split(",")]
            
            self.rate_limit_enabled = os.getenv("API_RATE_LIMIT_ENABLED", "true").lower() == "true"
            self.requests_per_minute = int(os.getenv("API_REQUESTS_PER_MINUTE", "60"))
            self.requests_per_hour = int(os.getenv("API_REQUESTS_PER_HOUR", "1000"))
            self.burst_limit = int(os.getenv("API_BURST_LIMIT", "10"))
            
            trusted_hosts_str = os.getenv("API_TRUSTED_HOSTS", "*")
            self.trusted_hosts = [host.strip() for host in trusted_hosts_str.split(",")]
            self.max_request_size = int(os.getenv("API_MAX_REQUEST_SIZE", "16777216"))


class MonitoringSettings(BaseSettings if PYDANTIC_AVAILABLE else object):
    """Standardized monitoring and observability configuration."""
    
    if PYDANTIC_AVAILABLE:
        enable_metrics: bool = Field(default=True, env="MONITORING_ENABLE_METRICS")
        metrics_port: int = Field(default=9090, env="MONITORING_METRICS_PORT")
        metrics_path: str = Field(default="/metrics", env="MONITORING_METRICS_PATH")
        
        # Health checks
        enable_health_checks: bool = Field(default=True, env="MONITORING_ENABLE_HEALTH_CHECKS")
        health_check_interval: int = Field(default=30, env="MONITORING_HEALTH_CHECK_INTERVAL")
        health_check_timeout: int = Field(default=10, env="MONITORING_HEALTH_CHECK_TIMEOUT")
        
        # Tracing
        enable_tracing: bool = Field(default=False, env="MONITORING_ENABLE_TRACING")
        jaeger_endpoint: Optional[str] = Field(default=None, env="MONITORING_JAEGER_ENDPOINT")
        trace_sample_rate: float = Field(default=0.1, env="MONITORING_TRACE_SAMPLE_RATE")
        
        # Performance monitoring
        enable_performance_monitoring: bool = Field(default=True, env="MONITORING_ENABLE_PERFORMANCE")
        performance_collection_interval: int = Field(default=60, env="MONITORING_PERFORMANCE_INTERVAL")
        
        # Alerting
        enable_alerting: bool = Field(default=True, env="MONITORING_ENABLE_ALERTING")
        alert_webhook_url: Optional[str] = Field(default=None, env="MONITORING_ALERT_WEBHOOK_URL")
        
        class Config:
            env_prefix = ""
            case_sensitive = False
            
        @validator('metrics_port')
        def validate_metrics_port(cls, v):
            if not 1 <= v <= 65535:
                raise ValueError('Metrics port must be between 1 and 65535')
            return v
            
        @validator('trace_sample_rate')
        def validate_sample_rate(cls, v):
            if not 0.0 <= v <= 1.0:
                raise ValueError('Trace sample rate must be between 0.0 and 1.0')
            return v
    else:
        def __init__(self):
            self.enable_metrics = os.getenv("MONITORING_ENABLE_METRICS", "true").lower() == "true"
            self.metrics_port = int(os.getenv("MONITORING_METRICS_PORT", "9090"))
            self.metrics_path = os.getenv("MONITORING_METRICS_PATH", "/metrics")
            self.enable_health_checks = os.getenv("MONITORING_ENABLE_HEALTH_CHECKS", "true").lower() == "true"
            self.health_check_interval = int(os.getenv("MONITORING_HEALTH_CHECK_INTERVAL", "30"))
            self.health_check_timeout = int(os.getenv("MONITORING_HEALTH_CHECK_TIMEOUT", "10"))
            self.enable_tracing = os.getenv("MONITORING_ENABLE_TRACING", "false").lower() == "true"
            self.jaeger_endpoint = os.getenv("MONITORING_JAEGER_ENDPOINT")
            self.trace_sample_rate = float(os.getenv("MONITORING_TRACE_SAMPLE_RATE", "0.1"))
            self.enable_performance_monitoring = os.getenv("MONITORING_ENABLE_PERFORMANCE", "true").lower() == "true"
            self.performance_collection_interval = int(os.getenv("MONITORING_PERFORMANCE_INTERVAL", "60"))
            self.enable_alerting = os.getenv("MONITORING_ENABLE_ALERTING", "true").lower() == "true"
            self.alert_webhook_url = os.getenv("MONITORING_ALERT_WEBHOOK_URL")


class BasePackageSettings(ABC):
    """Abstract base class for package-specific settings.
    
    This class provides the foundation for standardized configuration
    management across all packages in the monorepo.
    """
    
    def __init__(self, package_name: str, env_prefix: str = ""):
        self.package_name = package_name
        self.env_prefix = env_prefix or f"{package_name.upper()}_"
        
        # Core settings common to all packages
        self.environment: str = os.getenv(f"{self.env_prefix}ENVIRONMENT", "development")
        self.debug: bool = os.getenv(f"{self.env_prefix}DEBUG", "false").lower() == "true"
        self.version: str = os.getenv(f"{self.env_prefix}VERSION", "0.1.0")
        
        # Security
        self.secret_key: str = os.getenv(f"{self.env_prefix}SECRET_KEY", "dev-secret-key-change-in-production")
        self.api_key: Optional[str] = os.getenv(f"{self.env_prefix}API_KEY")
        
        # Standard components
        self.database = DatabaseSettings()
        self.logging = LoggingSettings()  
        self.api = APISettings()
        self.monitoring = MonitoringSettings()
    
    @classmethod
    @abstractmethod
    def load(cls: Type[T]) -> T:
        """Load package-specific settings."""
        pass
    
    def load_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ConfigError(f"Unsupported configuration file format: {file_path.suffix}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration from {file_path}: {e}")
    
    def load_from_multiple_sources(self) -> Dict[str, Any]:
        """Load configuration from multiple sources with precedence."""
        config_data = {}
        
        # Configuration file search paths
        config_paths = [
            Path(f"{self.package_name}.yml"),
            Path(f"{self.package_name}.yaml"),
            Path(f"{self.package_name}.json"),
            Path(f"config/{self.package_name}.yml"),
            Path(f"config/{self.package_name}.yaml"),
            Path(f"config/{self.package_name}.json"),
            Path(f".config/{self.package_name}/config.yml"),
            Path(f".config/{self.package_name}/config.yaml"),
            Path(f".config/{self.package_name}/config.json"),
            Path.home() / ".config" / self.package_name / "config.yml",
            Path.home() / ".config" / self.package_name / "config.yaml",
            Path.home() / ".config" / self.package_name / "config.json",
        ]
        
        # Load from first available config file
        for config_path in config_paths:
            if config_path.exists():
                try:
                    config_data = self.load_from_file(config_path)
                    break
                except ConfigError as e:
                    print(f"Warning: {e}")
        
        return config_data
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update settings from dictionary data."""
        # Update core settings
        for key in ["environment", "debug", "version", "secret_key", "api_key"]:
            if key in data:
                setattr(self, key, data[key])
        
        # Update component settings
        components = ["database", "logging", "api", "monitoring"]
        for component_name in components:
            if component_name in data and hasattr(self, component_name):
                component = getattr(self, component_name)
                component_data = data[component_name]
                
                for field_name, field_value in component_data.items():
                    if hasattr(component, field_name):
                        setattr(component, field_name, field_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        result = {
            "package_name": self.package_name,
            "environment": self.environment,
            "debug": self.debug,
            "version": self.version,
        }
        
        # Add component settings
        components = ["database", "logging", "api", "monitoring"]
        for component_name in components:
            if hasattr(self, component_name):
                component = getattr(self, component_name)
                if hasattr(component, '__dict__'):
                    result[component_name] = component.__dict__
        
        return result
    
    def create_example_config(self, output_path: Optional[Path] = None) -> None:
        """Create example configuration files."""
        if output_path is None:
            output_path = Path(f"{self.package_name}.example.yml")
        
        example_config = self.to_dict()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(example_config, f, default_flow_style=False, indent=2)
            print(f"✅ Created example configuration: {output_path}")
        except Exception as e:
            print(f"❌ Error creating example configuration: {e}")


def create_package_settings_class(package_name: str, additional_settings: Optional[Dict[str, Any]] = None):
    """Factory function to create package-specific settings classes."""
    
    class PackageSettings(BasePackageSettings):
        def __init__(self):
            super().__init__(package_name)
            
            # Add any additional package-specific settings
            if additional_settings:
                for key, value in additional_settings.items():
                    setattr(self, key, value)
        
        @classmethod
        def load(cls):
            instance = cls()
            
            # Load from configuration files
            config_data = instance.load_from_multiple_sources()
            if config_data:
                instance.update_from_dict(config_data)
                
            return instance
    
    return PackageSettings


# Utility functions for backward compatibility
def get_database_settings(env_prefix: str = "") -> DatabaseSettings:
    """Get database settings with optional environment prefix."""
    if PYDANTIC_AVAILABLE:
        # Temporarily set environment prefix for Pydantic
        original_env = {}
        if env_prefix:
            for key in ["DB_HOST", "DB_PORT", "DB_DATABASE", "DB_USERNAME", "DB_PASSWORD"]:
                original_key = key
                prefixed_key = f"{env_prefix}{key}"
                if prefixed_key in os.environ:
                    original_env[original_key] = os.environ.get(original_key)
                    os.environ[original_key] = os.environ[prefixed_key]
        
        settings = DatabaseSettings()
        
        # Restore original environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        
        return settings
    else:
        return DatabaseSettings()


def get_logging_settings(env_prefix: str = "") -> LoggingSettings:
    """Get logging settings with optional environment prefix."""
    if PYDANTIC_AVAILABLE:
        return LoggingSettings()
    else:
        return LoggingSettings()


def get_api_settings(env_prefix: str = "") -> APISettings:
    """Get API settings with optional environment prefix.""" 
    if PYDANTIC_AVAILABLE:
        return APISettings()
    else:
        return APISettings()


def get_monitoring_settings(env_prefix: str = "") -> MonitoringSettings:
    """Get monitoring settings with optional environment prefix."""
    if PYDANTIC_AVAILABLE:
        return MonitoringSettings()
    else:
        return MonitoringSettings()


__all__ = [
    "BasePackageSettings",
    "DatabaseSettings", 
    "LoggingSettings",
    "APISettings",
    "MonitoringSettings",
    "ConfigError",
    "create_package_settings_class",
    "get_database_settings",
    "get_logging_settings", 
    "get_api_settings",
    "get_monitoring_settings",
    "PYDANTIC_AVAILABLE"
]