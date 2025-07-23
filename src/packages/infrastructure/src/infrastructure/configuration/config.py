"""Infrastructure configuration management."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings


@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    url: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    
    def __post_init__(self) -> None:
        if not self.url:
            self.url = os.getenv("DATABASE_URL", "sqlite:///app.db")


@dataclass 
class RedisConfig:
    """Redis configuration."""
    
    url: str = ""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    connection_pool_size: int = 10
    
    def __post_init__(self) -> None:
        if not self.url:
            self.url = os.getenv(
                "REDIS_URL", 
                f"redis://{self.host}:{self.port}/{self.db}"
            )


@dataclass
class MessagingConfig:
    """Messaging configuration."""
    
    broker_url: str = ""
    result_backend: str = ""
    task_serializer: str = "json"
    result_serializer: str = "json"
    timezone: str = "UTC"
    enable_utc: bool = True
    
    def __post_init__(self) -> None:
        if not self.broker_url:
            self.broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        if not self.result_backend:
            self.result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")


@dataclass
class SecurityConfig:
    """Security configuration."""
    
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    def __post_init__(self) -> None:
        if not self.secret_key:
            self.secret_key = os.getenv("SECRET_KEY", "change-me-in-production")


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    
    log_level: str = "INFO"
    log_format: str = "json"
    enable_metrics: bool = True
    metrics_port: int = 8000
    enable_tracing: bool = True
    jaeger_endpoint: Optional[str] = None
    
    def __post_init__(self) -> None:
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")


@dataclass
class CacheConfig:
    """Caching configuration."""
    
    redis_url: str = ""
    default_ttl: int = 3600
    max_connections: int = 10
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    def __post_init__(self) -> None:
        if not self.redis_url:
            self.redis_url = os.getenv("CACHE_REDIS_URL", "redis://localhost:6379/1")


@dataclass
class InfrastructureConfig:
    """Main infrastructure configuration."""
    
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    testing: bool = field(default_factory=lambda: os.getenv("TESTING", "false").lower() == "true")
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    messaging: MessagingConfig = field(default_factory=MessagingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Additional settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_environment()
        self._load_custom_settings()
    
    def _validate_environment(self) -> None:
        """Validate environment-specific settings."""
        valid_environments = ["development", "testing", "staging", "production"]
        if self.environment not in valid_environments:
            raise ValueError(f"Invalid environment: {self.environment}")
        
        if self.environment == "production":
            if self.security.secret_key == "change-me-in-production":
                raise ValueError("Secret key must be set in production")
            if self.debug:
                raise ValueError("Debug mode should be disabled in production")
    
    def _load_custom_settings(self) -> None:
        """Load custom settings from environment variables."""
        for key, value in os.environ.items():
            if key.startswith("INFRA_CUSTOM_"):
                setting_name = key[13:].lower()  # Remove "INFRA_CUSTOM_" prefix
                self.custom_settings[setting_name] = value
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == "testing" or self.testing
    
    def get_custom_setting(self, key: str, default: Any = None) -> Any:
        """Get a custom setting value."""
        return self.custom_settings.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "testing": self.testing,
            "database": {
                "url": self.database.url,
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow,
                "pool_timeout": self.database.pool_timeout,
                "pool_recycle": self.database.pool_recycle,
                "echo": self.database.echo,
            },
            "redis": {
                "url": self.redis.url,
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "ssl": self.redis.ssl,
                "connection_pool_size": self.redis.connection_pool_size,
            },
            "messaging": {
                "broker_url": self.messaging.broker_url,
                "result_backend": self.messaging.result_backend,
                "task_serializer": self.messaging.task_serializer,
                "result_serializer": self.messaging.result_serializer,
                "timezone": self.messaging.timezone,
                "enable_utc": self.messaging.enable_utc,
            },
            "security": {
                "jwt_algorithm": self.security.jwt_algorithm,
                "jwt_expiration_hours": self.security.jwt_expiration_hours,
                "password_min_length": self.security.password_min_length,
                "max_login_attempts": self.security.max_login_attempts,
                "lockout_duration_minutes": self.security.lockout_duration_minutes,
            },
            "monitoring": {
                "log_level": self.monitoring.log_level,
                "log_format": self.monitoring.log_format,
                "enable_metrics": self.monitoring.enable_metrics,
                "metrics_port": self.monitoring.metrics_port,
                "enable_tracing": self.monitoring.enable_tracing,
                "jaeger_endpoint": self.monitoring.jaeger_endpoint,
            },
            "cache": {
                "redis_url": self.cache.redis_url,
                "default_ttl": self.cache.default_ttl,
                "max_connections": self.cache.max_connections,
                "retry_on_timeout": self.cache.retry_on_timeout,
                "health_check_interval": self.cache.health_check_interval,
            },
            "custom_settings": self.custom_settings,
        }


# Global configuration instance
_config: Optional[InfrastructureConfig] = None


def get_config() -> InfrastructureConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = InfrastructureConfig()
    return _config


def reload_config() -> InfrastructureConfig:
    """Reload configuration from environment."""
    global _config
    _config = InfrastructureConfig()
    return _config


def set_config(config: InfrastructureConfig) -> None:
    """Set the global configuration instance (mainly for testing)."""
    global _config
    _config = config