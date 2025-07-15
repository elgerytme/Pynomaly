"""Production configuration management and validation."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pynomaly.infrastructure.config.settings import Settings
from pynomaly.shared.error_handling import (
    ErrorCodes,
    create_infrastructure_error,
)

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class SecurityLevel(Enum):
    """Security configuration levels."""

    BASIC = "basic"
    ENHANCED = "enhanced"
    ENTERPRISE = "enterprise"


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""

    name: Environment
    debug_mode: bool = False
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    allowed_hosts: list[str] = field(default_factory=list)

    @classmethod
    def for_environment(cls, env: Environment) -> EnvironmentConfig:
        """Create environment-specific configuration."""
        if env == Environment.DEVELOPMENT:
            return cls(
                name=env,
                debug_mode=True,
                log_level="DEBUG",
                cors_origins=["*"],
                allowed_hosts=["localhost", "127.0.0.1"],
            )
        elif env == Environment.STAGING:
            return cls(
                name=env,
                debug_mode=False,
                log_level="INFO",
                cors_origins=["https://staging.example.com"],
                allowed_hosts=["staging.example.com"],
            )
        elif env == Environment.PRODUCTION:
            return cls(
                name=env,
                debug_mode=False,
                log_level="WARNING",
                enable_metrics=True,
                enable_tracing=True,
                cors_origins=[],  # Strict CORS in production
                allowed_hosts=["api.example.com", "pynomaly.example.com"],
            )
        else:  # TESTING
            return cls(
                name=env,
                debug_mode=True,
                log_level="DEBUG",
                enable_metrics=False,
                enable_tracing=False,
                cors_origins=["*"],
                allowed_hosts=["*"],
            )


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    level: SecurityLevel = SecurityLevel.BASIC
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_sql_injection_protection: bool = True
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    enable_threat_detection: bool = False
    enable_security_headers: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    password_min_length: int = 8
    require_https: bool = False
    api_key_rotation_days: int = 90

    @classmethod
    def for_environment(cls, env: Environment) -> SecurityConfig:
        """Create environment-specific security configuration."""
        if env == Environment.PRODUCTION:
            return cls(
                level=SecurityLevel.ENTERPRISE,
                enable_threat_detection=True,
                require_https=True,
                session_timeout_minutes=15,
                max_login_attempts=3,
                password_min_length=12,
                api_key_rotation_days=30,
            )
        elif env == Environment.STAGING:
            return cls(
                level=SecurityLevel.ENHANCED,
                enable_threat_detection=True,
                require_https=True,
                session_timeout_minutes=30,
                api_key_rotation_days=60,
            )
        else:  # DEVELOPMENT, TESTING
            return cls(
                level=SecurityLevel.BASIC,
                enable_threat_detection=False,
                require_https=False,
                session_timeout_minutes=60,
                api_key_rotation_days=180,
            )


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""

    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_compression: bool = True
    enable_connection_pooling: bool = True
    max_database_connections: int = 20
    max_redis_connections: int = 10
    request_timeout_seconds: int = 30
    worker_processes: int = 1
    worker_threads: int = 4
    enable_async_processing: bool = True
    batch_size: int = 100
    memory_limit_mb: int = 512
    cpu_limit_percent: float = 80.0

    @classmethod
    def for_environment(cls, env: Environment) -> PerformanceConfig:
        """Create environment-specific performance configuration."""
        if env == Environment.PRODUCTION:
            return cls(
                worker_processes=4,
                worker_threads=8,
                max_database_connections=50,
                max_redis_connections=25,
                request_timeout_seconds=15,
                memory_limit_mb=2048,
                cpu_limit_percent=70.0,
                batch_size=500,
            )
        elif env == Environment.STAGING:
            return cls(
                worker_processes=2,
                worker_threads=4,
                max_database_connections=25,
                max_redis_connections=15,
                request_timeout_seconds=30,
                memory_limit_mb=1024,
                cpu_limit_percent=75.0,
                batch_size=250,
            )
        else:  # DEVELOPMENT, TESTING
            return cls(
                worker_processes=1,
                worker_threads=2,
                max_database_connections=10,
                max_redis_connections=5,
                request_timeout_seconds=60,
                memory_limit_mb=256,
                cpu_limit_percent=90.0,
                batch_size=50,
            )


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""

    enable_health_checks: bool = True
    health_check_interval_seconds: int = 30
    enable_metrics_collection: bool = True
    metrics_export_interval_seconds: int = 60
    enable_distributed_tracing: bool = False
    enable_performance_profiling: bool = False
    enable_error_tracking: bool = True
    enable_alerting: bool = False
    log_retention_days: int = 30
    metrics_retention_days: int = 90
    trace_sampling_rate: float = 0.1

    @classmethod
    def for_environment(cls, env: Environment) -> MonitoringConfig:
        """Create environment-specific monitoring configuration."""
        if env == Environment.PRODUCTION:
            return cls(
                health_check_interval_seconds=15,
                metrics_export_interval_seconds=30,
                enable_distributed_tracing=True,
                enable_performance_profiling=True,
                enable_alerting=True,
                log_retention_days=90,
                metrics_retention_days=365,
                trace_sampling_rate=0.01,  # Lower sampling in production
            )
        elif env == Environment.STAGING:
            return cls(
                health_check_interval_seconds=30,
                enable_distributed_tracing=True,
                enable_performance_profiling=False,
                enable_alerting=True,
                log_retention_days=30,
                trace_sampling_rate=0.05,
            )
        else:  # DEVELOPMENT, TESTING
            return cls(
                health_check_interval_seconds=60,
                enable_distributed_tracing=False,
                enable_performance_profiling=True,
                enable_alerting=False,
                log_retention_days=7,
                trace_sampling_rate=1.0,  # Full sampling in development
            )


@dataclass
class DeploymentConfig:
    """Deployment-specific configuration."""

    container_registry: str = "docker.io"
    image_tag: str = "latest"
    namespace: str = "default"
    replicas: int = 1
    enable_autoscaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    enable_service_mesh: bool = False
    enable_ingress: bool = True
    enable_tls: bool = False

    @classmethod
    def for_environment(cls, env: Environment) -> DeploymentConfig:
        """Create environment-specific deployment configuration."""
        if env == Environment.PRODUCTION:
            return cls(
                image_tag="stable",
                namespace="production",
                replicas=3,
                enable_autoscaling=True,
                min_replicas=3,
                max_replicas=20,
                cpu_request="500m",
                cpu_limit="2000m",
                memory_request="1Gi",
                memory_limit="4Gi",
                enable_service_mesh=True,
                enable_tls=True,
            )
        elif env == Environment.STAGING:
            return cls(
                image_tag="staging",
                namespace="staging",
                replicas=2,
                enable_autoscaling=True,
                min_replicas=1,
                max_replicas=5,
                cpu_request="250m",
                cpu_limit="1000m",
                memory_request="512Mi",
                memory_limit="2Gi",
                enable_tls=True,
            )
        else:  # DEVELOPMENT, TESTING
            return cls(
                image_tag="dev",
                namespace="development",
                replicas=1,
                enable_autoscaling=False,
                cpu_request="100m",
                cpu_limit="500m",
                memory_request="128Mi",
                memory_limit="512Mi",
                enable_tls=False,
            )


@dataclass
class ProductionConfig:
    """Comprehensive production configuration."""

    environment: EnvironmentConfig
    security: SecurityConfig
    performance: PerformanceConfig
    monitoring: MonitoringConfig
    deployment: DeploymentConfig
    version: str = "1.0.0"
    build_time: str | None = None
    git_commit: str | None = None
    custom_settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_environment(cls, env: Environment, **kwargs) -> ProductionConfig:
        """Create production configuration for specific environment."""
        return cls(
            environment=EnvironmentConfig.for_environment(env),
            security=SecurityConfig.for_environment(env),
            performance=PerformanceConfig.for_environment(env),
            monitoring=MonitoringConfig.for_environment(env),
            deployment=DeploymentConfig.for_environment(env),
            **kwargs,
        )

    @classmethod
    def from_settings(cls, settings: Settings, env: Environment) -> ProductionConfig:
        """Create production configuration from application settings."""
        config = cls.for_environment(env)

        # Override with settings values
        if settings.database_url:
            config.custom_settings["database_url"] = settings.database_url
        if settings.redis_url:
            config.custom_settings["redis_url"] = settings.redis_url
        if settings.cache_enabled is not None:
            config.performance.enable_caching = settings.cache_enabled
        if settings.cache_ttl:
            config.performance.cache_ttl_seconds = settings.cache_ttl
        if settings.log_level:
            config.environment.log_level = settings.log_level

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": {
                "name": self.environment.name.value,
                "debug_mode": self.environment.debug_mode,
                "log_level": self.environment.log_level,
                "enable_metrics": self.environment.enable_metrics,
                "enable_tracing": self.environment.enable_tracing,
                "cors_origins": self.environment.cors_origins,
                "allowed_hosts": self.environment.allowed_hosts,
            },
            "security": {
                "level": self.security.level.value,
                "enable_rate_limiting": self.security.enable_rate_limiting,
                "enable_input_validation": self.security.enable_input_validation,
                "enable_sql_injection_protection": self.security.enable_sql_injection_protection,
                "enable_encryption": self.security.enable_encryption,
                "enable_audit_logging": self.security.enable_audit_logging,
                "enable_threat_detection": self.security.enable_threat_detection,
                "enable_security_headers": self.security.enable_security_headers,
                "session_timeout_minutes": self.security.session_timeout_minutes,
                "max_login_attempts": self.security.max_login_attempts,
                "require_https": self.security.require_https,
            },
            "performance": {
                "enable_caching": self.performance.enable_caching,
                "cache_ttl_seconds": self.performance.cache_ttl_seconds,
                "enable_compression": self.performance.enable_compression,
                "enable_connection_pooling": self.performance.enable_connection_pooling,
                "max_database_connections": self.performance.max_database_connections,
                "max_redis_connections": self.performance.max_redis_connections,
                "request_timeout_seconds": self.performance.request_timeout_seconds,
                "worker_processes": self.performance.worker_processes,
                "worker_threads": self.performance.worker_threads,
                "memory_limit_mb": self.performance.memory_limit_mb,
                "cpu_limit_percent": self.performance.cpu_limit_percent,
            },
            "monitoring": {
                "enable_health_checks": self.monitoring.enable_health_checks,
                "health_check_interval_seconds": self.monitoring.health_check_interval_seconds,
                "enable_metrics_collection": self.monitoring.enable_metrics_collection,
                "enable_distributed_tracing": self.monitoring.enable_distributed_tracing,
                "enable_performance_profiling": self.monitoring.enable_performance_profiling,
                "enable_error_tracking": self.monitoring.enable_error_tracking,
                "enable_alerting": self.monitoring.enable_alerting,
                "log_retention_days": self.monitoring.log_retention_days,
                "trace_sampling_rate": self.monitoring.trace_sampling_rate,
            },
            "deployment": {
                "container_registry": self.deployment.container_registry,
                "image_tag": self.deployment.image_tag,
                "namespace": self.deployment.namespace,
                "replicas": self.deployment.replicas,
                "enable_autoscaling": self.deployment.enable_autoscaling,
                "min_replicas": self.deployment.min_replicas,
                "max_replicas": self.deployment.max_replicas,
                "cpu_request": self.deployment.cpu_request,
                "cpu_limit": self.deployment.cpu_limit,
                "memory_request": self.deployment.memory_request,
                "memory_limit": self.deployment.memory_limit,
                "enable_service_mesh": self.deployment.enable_service_mesh,
                "enable_ingress": self.deployment.enable_ingress,
                "enable_tls": self.deployment.enable_tls,
            },
            "metadata": {
                "version": self.version,
                "build_time": self.build_time,
                "git_commit": self.git_commit,
            },
            "custom_settings": self.custom_settings,
        }

    def save_to_file(self, file_path: str | Path) -> None:
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Production configuration saved to {file_path}")


class ProductionConfigValidator:
    """Validates production configuration for completeness and security."""

    def __init__(self):
        self.issues: list[str] = []
        self.warnings: list[str] = []

    def validate(self, config: ProductionConfig) -> bool:
        """Validate production configuration.

        Returns:
            True if configuration is valid for production
        """
        self.issues.clear()
        self.warnings.clear()

        self._validate_environment_config(config.environment)
        self._validate_security_config(config.security)
        self._validate_performance_config(config.performance)
        self._validate_monitoring_config(config.monitoring)
        self._validate_deployment_config(config.deployment)

        return len(self.issues) == 0

    def _validate_environment_config(self, env_config: EnvironmentConfig) -> None:
        """Validate environment configuration."""
        if env_config.name == Environment.PRODUCTION:
            if env_config.debug_mode:
                self.issues.append("Debug mode should be disabled in production")

            if env_config.log_level == "DEBUG":
                self.warnings.append(
                    "Debug logging may impact performance in production"
                )

            if "*" in env_config.cors_origins:
                self.issues.append("Wildcard CORS origins not allowed in production")

            if not env_config.allowed_hosts or "*" in env_config.allowed_hosts:
                self.issues.append(
                    "Specific allowed hosts must be configured in production"
                )

    def _validate_security_config(self, sec_config: SecurityConfig) -> None:
        """Validate security configuration."""
        if not sec_config.enable_rate_limiting:
            self.warnings.append("Rate limiting is disabled")

        if not sec_config.enable_input_validation:
            self.issues.append("Input validation must be enabled")

        if not sec_config.enable_sql_injection_protection:
            self.issues.append("SQL injection protection must be enabled")

        if not sec_config.enable_encryption:
            self.issues.append("Encryption must be enabled")

        if sec_config.session_timeout_minutes > 60:
            self.warnings.append(
                "Session timeout is longer than recommended (60 minutes)"
            )

        if sec_config.password_min_length < 8:
            self.issues.append(
                "Password minimum length should be at least 8 characters"
            )

    def _validate_performance_config(self, perf_config: PerformanceConfig) -> None:
        """Validate performance configuration."""
        if not perf_config.enable_caching:
            self.warnings.append("Caching is disabled, may impact performance")

        if not perf_config.enable_connection_pooling:
            self.warnings.append("Connection pooling is disabled")

        if perf_config.request_timeout_seconds > 60:
            self.warnings.append("Request timeout is very high")

        if perf_config.memory_limit_mb < 128:
            self.warnings.append("Memory limit may be too low for production workloads")

    def _validate_monitoring_config(self, mon_config: MonitoringConfig) -> None:
        """Validate monitoring configuration."""
        if not mon_config.enable_health_checks:
            self.issues.append("Health checks must be enabled in production")

        if not mon_config.enable_metrics_collection:
            self.warnings.append("Metrics collection is disabled")

        if not mon_config.enable_error_tracking:
            self.warnings.append("Error tracking is disabled")

        if mon_config.health_check_interval_seconds > 60:
            self.warnings.append("Health check interval may be too long")

    def _validate_deployment_config(self, dep_config: DeploymentConfig) -> None:
        """Validate deployment configuration."""
        if dep_config.replicas < 2:
            self.warnings.append(
                "Running with less than 2 replicas reduces availability"
            )

        if not dep_config.enable_autoscaling:
            self.warnings.append("Autoscaling is disabled")

        if dep_config.image_tag == "latest":
            self.warnings.append("Using 'latest' tag is not recommended for production")


# Global production configuration
_production_config: ProductionConfig | None = None


def get_production_config(
    environment: Environment | None = None,
    settings: Settings | None = None,
    force_reload: bool = False,
) -> ProductionConfig:
    """Get global production configuration.

    Args:
        environment: Target environment (detected from env var if not provided)
        settings: Application settings to merge
        force_reload: Force reload of configuration

    Returns:
        Production configuration instance
    """
    global _production_config

    # Force reload if environment is explicitly provided and different
    if (
        environment is not None
        and _production_config is not None
        and _production_config.environment.name != environment
    ):
        force_reload = True

    if _production_config is None or force_reload:
        # Detect environment from environment variable
        if environment is None:
            env_name = os.getenv("PYNOMALY_ENV", "development").lower()
            try:
                environment = Environment(env_name)
            except ValueError:
                logger.warning(
                    f"Unknown environment '{env_name}', defaulting to development"
                )
                environment = Environment.DEVELOPMENT

        # Create configuration
        if settings:
            _production_config = ProductionConfig.from_settings(settings, environment)
        else:
            _production_config = ProductionConfig.for_environment(environment)

        logger.info(
            f"Production configuration loaded for environment: {environment.value}"
        )

    return _production_config


def validate_production_config(
    config: ProductionConfig | None = None,
    raise_on_issues: bool = True,
) -> dict[str, Any]:
    """Validate production configuration.

    Args:
        config: Configuration to validate (uses global if not provided)
        raise_on_issues: Raise exception on validation issues

    Returns:
        Validation result dictionary

    Raises:
        InfrastructureError: If validation fails and raise_on_issues is True
    """
    if config is None:
        config = get_production_config()

    validator = ProductionConfigValidator()
    is_valid = validator.validate(config)

    result = {
        "valid": is_valid,
        "issues": validator.issues,
        "warnings": validator.warnings,
        "environment": config.environment.name.value,
        "security_level": config.security.level.value,
    }

    if not is_valid and raise_on_issues:
        raise create_infrastructure_error(
            error_code=ErrorCodes.INF_CONFIG_INVALID,
            message=f"Production configuration validation failed: {'; '.join(validator.issues)}",
            context={"validation_result": result},
        )

    return result
