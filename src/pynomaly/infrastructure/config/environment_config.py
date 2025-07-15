#!/usr/bin/env python3
"""
Environment Configuration Management for Pynomaly.
Provides centralized configuration management with environment-specific settings.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "pynomaly"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class RedisConfig:
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: str | None = None
    ssl: bool = False
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    max_connections: int = 50


@dataclass
class SecurityConfig:
    """Security configuration."""

    secret_key: str = "your-secret-key-change-this"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_key_header: str = "X-API-Key"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    cors_methods: list[str] = field(default_factory=lambda: ["*"])
    cors_headers: list[str] = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_window_minutes: int = 15


@dataclass
class ModelConfig:
    """Model configuration."""

    registry_path: str = "mlops/models"
    max_model_cache_size: int = 10
    model_timeout_seconds: int = 300
    batch_prediction_size: int = 1000
    auto_scaling_enabled: bool = True
    default_algorithm: str = "isolation_forest"
    feature_selection_threshold: float = 0.95
    cross_validation_folds: int = 5


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    enabled: bool = True
    metrics_retention_days: int = 30
    alert_evaluation_interval_seconds: int = 60
    health_check_interval_seconds: int = 30
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    grafana_enabled: bool = False
    grafana_port: int = 3000


@dataclass
class DeploymentConfig:
    """Deployment configuration."""

    container_registry: str = "localhost:5000"
    kubernetes_namespace: str = "pynomaly"
    default_replicas: int = 2
    max_replicas: int = 10
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "256Mi"
    memory_limit: str = "512Mi"
    health_check_path: str = "/health"
    deployment_timeout_seconds: int = 600


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/pynomaly.log"
    file_max_size_mb: int = 10
    file_backup_count: int = 5
    console_enabled: bool = True
    structured_logging: bool = False


@dataclass
class APIConfig:
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    debug: bool = False
    docs_enabled: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    request_timeout_seconds: int = 30
    max_request_size_mb: int = 10


@dataclass
class PynomlalyConfig:
    """Main Pynomaly configuration."""

    environment: Environment
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    custom: dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Configuration manager with environment-specific settings."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._config: PynomlalyConfig | None = None
        self._environment = self._detect_environment()

        # Load configuration
        self._load_configuration()

    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables."""
        env_name = os.getenv("PYNOMALY_ENV", "development").lower()

        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(
                f"Unknown environment '{env_name}', defaulting to development"
            )
            return Environment.DEVELOPMENT

    def _load_configuration(self):
        """Load configuration from files and environment variables."""
        try:
            # Load base configuration
            base_config = self._load_config_file("base.yaml")

            # Load environment-specific configuration
            env_config = self._load_config_file(f"{self._environment.value}.yaml")

            # Merge configurations
            merged_config = self._merge_configs(base_config, env_config)

            # Override with environment variables
            final_config = self._override_with_env_vars(merged_config)

            # Create configuration object
            self._config = self._create_config_object(final_config)

            logger.info(
                f"Configuration loaded for environment: {self._environment.value}"
            )

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Create default configuration
            self._config = PynomlalyConfig(environment=self._environment)

    def _load_config_file(self, filename: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = self.config_dir / filename

        if not config_path.exists():
            logger.info(f"Configuration file not found: {filename}")
            return {}

        try:
            with open(config_path) as f:
                if filename.endswith(".yaml") or filename.endswith(".yml"):
                    return yaml.safe_load(f) or {}
                elif filename.endswith(".json"):
                    return json.load(f) or {}
                else:
                    logger.warning(f"Unsupported config file format: {filename}")
                    return {}

        except Exception as e:
            logger.error(f"Failed to load config file {filename}: {e}")
            return {}

    def _merge_configs(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _override_with_env_vars(self, config: dict[str, Any]) -> dict[str, Any]:
        """Override configuration with environment variables."""
        # Environment variable mappings
        env_mappings = {
            # Database
            "PYNOMALY_DB_HOST": ["database", "host"],
            "PYNOMALY_DB_PORT": ["database", "port"],
            "PYNOMALY_DB_DATABASE": ["database", "database"],
            "PYNOMALY_DB_USERNAME": ["database", "username"],
            "PYNOMALY_DB_PASSWORD": ["database", "password"],
            # Redis
            "PYNOMALY_REDIS_HOST": ["redis", "host"],
            "PYNOMALY_REDIS_PORT": ["redis", "port"],
            "PYNOMALY_REDIS_PASSWORD": ["redis", "password"],
            # Security
            "PYNOMALY_SECRET_KEY": ["security", "secret_key"],
            "PYNOMALY_JWT_ALGORITHM": ["security", "jwt_algorithm"],
            # API
            "PYNOMALY_API_HOST": ["api", "host"],
            "PYNOMALY_API_PORT": ["api", "port"],
            "PYNOMALY_API_WORKERS": ["api", "workers"],
            "PYNOMALY_API_DEBUG": ["api", "debug"],
            # Logging
            "PYNOMALY_LOG_LEVEL": ["logging", "level"],
            "PYNOMALY_LOG_FILE": ["logging", "file_path"],
            # Monitoring
            "PYNOMALY_MONITORING_ENABLED": ["monitoring", "enabled"],
            "PYNOMALY_PROMETHEUS_PORT": ["monitoring", "prometheus_port"],
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)

                # Set value in config
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = converted_value

        return config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # List conversion (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",")]

        # Return as string
        return value

    def _create_config_object(self, config_dict: dict[str, Any]) -> PynomlalyConfig:
        """Create PynomlalyConfig object from dictionary."""
        # Create sub-configuration objects
        database_config = DatabaseConfig(**config_dict.get("database", {}))
        redis_config = RedisConfig(**config_dict.get("redis", {}))
        security_config = SecurityConfig(**config_dict.get("security", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        monitoring_config = MonitoringConfig(**config_dict.get("monitoring", {}))
        deployment_config = DeploymentConfig(**config_dict.get("deployment", {}))

        # Handle logging config with enum conversion
        logging_dict = config_dict.get("logging", {})
        if "level" in logging_dict and isinstance(logging_dict["level"], str):
            logging_dict["level"] = LogLevel(logging_dict["level"].upper())
        logging_config = LoggingConfig(**logging_dict)

        api_config = APIConfig(**config_dict.get("api", {}))

        return PynomlalyConfig(
            environment=self._environment,
            database=database_config,
            redis=redis_config,
            security=security_config,
            model=model_config,
            monitoring=monitoring_config,
            deployment=deployment_config,
            logging=logging_config,
            api=api_config,
            custom=config_dict.get("custom", {}),
        )

    def get_config(self) -> PynomlalyConfig:
        """Get the current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def get_database_url(self) -> str:
        """Get database connection URL."""
        db = self._config.database
        return f"postgresql://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"

    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        redis = self._config.redis
        password_part = f":{redis.password}@" if redis.password else ""
        ssl_part = "s" if redis.ssl else ""
        return f"redis{ssl_part}://{password_part}{redis.host}:{redis.port}/{redis.database}"

    def save_config(self, filename: str = None):
        """Save current configuration to file."""
        if not filename:
            filename = f"{self._environment.value}.yaml"

        config_path = self.config_dir / filename

        try:
            # Convert config to dictionary
            config_dict = self._config_to_dict(self._config)

            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, indent=2, default_flow_style=False)

            logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def _config_to_dict(self, config: PynomlalyConfig) -> dict[str, Any]:
        """Convert configuration object to dictionary."""
        config_dict = {}

        # Convert dataclass fields
        for field_name in config.__dataclass_fields__:
            if field_name == "environment":
                config_dict[field_name] = config.environment.value
            else:
                field_value = getattr(config, field_name)
                if hasattr(field_value, "__dataclass_fields__"):
                    # Convert nested dataclass
                    nested_dict = {}
                    for nested_field in field_value.__dataclass_fields__:
                        nested_value = getattr(field_value, nested_field)
                        if isinstance(nested_value, Enum):
                            nested_dict[nested_field] = nested_value.value
                        else:
                            nested_dict[nested_field] = nested_value
                    config_dict[field_name] = nested_dict
                else:
                    config_dict[field_name] = field_value

        return config_dict

    def reload_config(self):
        """Reload configuration from files."""
        logger.info("Reloading configuration...")
        self._load_configuration()

    def create_default_configs(self):
        """Create default configuration files."""
        configs = {
            "base.yaml": {
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "pynomaly",
                    "username": "postgres",
                    "pool_size": 10,
                },
                "redis": {"host": "localhost", "port": 6379, "database": 0},
                "security": {
                    "jwt_algorithm": "HS256",
                    "jwt_expiration_hours": 24,
                    "cors_origins": ["*"],
                },
                "model": {
                    "registry_path": "mlops/models",
                    "max_model_cache_size": 10,
                    "default_algorithm": "isolation_forest",
                },
                "monitoring": {
                    "enabled": True,
                    "metrics_retention_days": 30,
                    "alert_evaluation_interval_seconds": 60,
                },
                "api": {"host": "0.0.0.0", "port": 8000, "docs_enabled": True},
                "logging": {
                    "level": "INFO",
                    "file_enabled": True,
                    "console_enabled": True,
                },
            },
            "development.yaml": {
                "api": {"debug": True, "reload": True, "workers": 1},
                "logging": {"level": "DEBUG"},
                "monitoring": {"prometheus_enabled": False, "grafana_enabled": False},
            },
            "production.yaml": {
                "api": {"debug": False, "reload": False, "workers": 4},
                "logging": {"level": "INFO", "structured_logging": True},
                "monitoring": {"prometheus_enabled": True, "grafana_enabled": True},
                "security": {"cors_origins": ["https://yourdomain.com"]},
            },
            "testing.yaml": {
                "database": {"database": "pynomaly_test"},
                "redis": {"database": 1},
                "logging": {"level": "WARNING"},
            },
        }

        for filename, config_data in configs.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                try:
                    with open(config_path, "w") as f:
                        yaml.dump(config_data, f, indent=2, default_flow_style=False)
                    logger.info(f"Created default config: {filename}")
                except Exception as e:
                    logger.error(f"Failed to create config {filename}: {e}")


# Global configuration manager
config_manager = ConfigManager()


# Convenience function to get configuration
def get_config() -> PynomlalyConfig:
    """Get the current configuration."""
    return config_manager.get_config()


# Export for use
__all__ = [
    "ConfigManager",
    "PynomlalyConfig",
    "Environment",
    "DatabaseConfig",
    "RedisConfig",
    "SecurityConfig",
    "ModelConfig",
    "MonitoringConfig",
    "DeploymentConfig",
    "LoggingConfig",
    "APIConfig",
    "config_manager",
    "get_config",
]
