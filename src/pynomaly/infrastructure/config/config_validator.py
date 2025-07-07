"""Configuration validation and health checking."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from pynomaly.domain.exceptions import ConfigurationError
from .settings import Settings


class ConfigurationValidator:
    """Validates configuration settings and dependencies."""

    def __init__(self, settings: Settings):
        """Initialize validator with settings."""
        self.settings = settings
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def validate_all(self) -> dict[str, Any]:
        """Run comprehensive validation and return results."""
        self.errors.clear()
        self.warnings.clear()
        self.info.clear()

        # Run all validation checks
        self._validate_basic_settings()
        self._validate_paths()
        self._validate_database()
        self._validate_cache()
        self._validate_monitoring()
        self._validate_security()
        self._validate_network()
        self._validate_environment()

        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }

    def _validate_basic_settings(self) -> None:
        """Validate basic application settings."""
        try:
            # Test settings model validation
            Settings.model_validate(self.settings.model_dump())
            self.info.append("Basic settings validation passed")
        except ValidationError as e:
            self.errors.extend([f"Settings validation: {error['msg']}" for error in e.errors()])

    def _validate_paths(self) -> None:
        """Validate filesystem paths."""
        paths_to_check = {
            "storage_path": self.settings.storage_path,
            "model_storage_path": self.settings.model_storage_path,
            "experiment_storage_path": self.settings.experiment_storage_path,
            "temp_path": self.settings.temp_path,
            "log_path": self.settings.log_path,
        }

        for name, path in paths_to_check.items():
            try:
                # Check if path exists or can be created
                path.mkdir(parents=True, exist_ok=True)
                
                # Check write permissions
                test_file = path / ".write_test"
                test_file.touch()
                test_file.unlink()
                
                self.info.append(f"Path {name} is accessible: {path}")
                
            except PermissionError:
                self.errors.append(f"No write permission for {name}: {path}")
            except Exception as e:
                self.errors.append(f"Cannot access {name} '{path}': {e}")

    def _validate_database(self) -> None:
        """Validate database configuration."""
        if not self.settings.database_url:
            if self.settings.use_database_repositories:
                self.errors.append("Database repositories enabled but no database URL configured")
            else:
                self.info.append("Database not configured (using in-memory repositories)")
            return

        try:
            parsed = urlparse(self.settings.database_url)
            
            # Check URL format
            if not parsed.scheme:
                self.errors.append("Database URL missing scheme")
                return
            
            supported_schemes = ["sqlite", "postgresql", "mysql", "oracle"]
            if parsed.scheme not in supported_schemes:
                self.warnings.append(f"Database scheme '{parsed.scheme}' may not be supported")

            # For SQLite, check file permissions
            if parsed.scheme == "sqlite":
                if parsed.path and parsed.path != ":memory:":
                    db_path = Path(parsed.path)
                    db_dir = db_path.parent
                    
                    if not db_dir.exists():
                        try:
                            db_dir.mkdir(parents=True, exist_ok=True)
                            self.info.append(f"Created database directory: {db_dir}")
                        except Exception as e:
                            self.errors.append(f"Cannot create database directory: {e}")
                    
                    # Check write permissions
                    if db_dir.exists():
                        try:
                            test_file = db_dir / ".db_test"
                            test_file.touch()
                            test_file.unlink()
                        except PermissionError:
                            self.errors.append(f"No write permission for database directory: {db_dir}")

            # For network databases, test connectivity
            elif parsed.hostname:
                port = parsed.port or 5432  # Default PostgreSQL port
                if self._test_network_connectivity(parsed.hostname, port):
                    self.info.append(f"Database host {parsed.hostname}:{port} is reachable")
                else:
                    self.warnings.append(f"Cannot reach database host {parsed.hostname}:{port}")

            # Validate pool settings
            if self.settings.database_pool_size <= 0:
                self.errors.append("Database pool size must be positive")
            
            if self.settings.database_max_overflow < 0:
                self.errors.append("Database max overflow cannot be negative")

            self.info.append("Database configuration validated")

        except Exception as e:
            self.errors.append(f"Invalid database URL: {e}")

    def _validate_cache(self) -> None:
        """Validate cache configuration."""
        if not self.settings.cache_enabled:
            self.info.append("Cache disabled")
            return

        if self.settings.redis_url:
            try:
                parsed = urlparse(self.settings.redis_url)
                if parsed.scheme != "redis":
                    self.warnings.append("Redis URL should use 'redis://' scheme")
                
                if parsed.hostname:
                    port = parsed.port or 6379
                    if self._test_network_connectivity(parsed.hostname, port):
                        self.info.append(f"Redis host {parsed.hostname}:{port} is reachable")
                    else:
                        self.warnings.append(f"Cannot reach Redis host {parsed.hostname}:{port}")
                
                self.info.append("Redis cache configuration validated")
                
            except Exception as e:
                self.errors.append(f"Invalid Redis URL: {e}")
        else:
            self.info.append("Using in-memory cache (Redis not configured)")

        if self.settings.cache_ttl <= 0:
            self.errors.append("Cache TTL must be positive")

    def _validate_monitoring(self) -> None:
        """Validate monitoring configuration."""
        monitoring = self.settings.monitoring

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if monitoring.log_level.upper() not in valid_log_levels:
            self.errors.append(f"Invalid log level: {monitoring.log_level}")

        # Validate log format
        valid_log_formats = ["text", "json"]
        if monitoring.log_format not in valid_log_formats:
            self.errors.append(f"Invalid log format: {monitoring.log_format}")

        # Check Prometheus port
        if monitoring.prometheus_enabled:
            if not (1 <= monitoring.prometheus_port <= 65535):
                self.errors.append(f"Invalid Prometheus port: {monitoring.prometheus_port}")
            
            if self._is_port_in_use(monitoring.prometheus_port):
                self.warnings.append(f"Prometheus port {monitoring.prometheus_port} is already in use")

        # Validate OTLP endpoint if tracing enabled
        if monitoring.tracing_enabled and monitoring.otlp_endpoint:
            try:
                parsed = urlparse(monitoring.otlp_endpoint)
                if not parsed.scheme or not parsed.hostname:
                    self.errors.append("Invalid OTLP endpoint URL")
            except Exception:
                self.errors.append("Malformed OTLP endpoint URL")

        self.info.append("Monitoring configuration validated")

    def _validate_security(self) -> None:
        """Validate security configuration."""
        # Check secret key
        if self.settings.secret_key == "change-me-in-production":
            if self.settings.app.environment == "production":
                self.errors.append("Secret key must be changed for production")
            else:
                self.warnings.append("Using default secret key (change for production)")

        # Check secret key strength
        if len(self.settings.secret_key) < 32:
            self.warnings.append("Secret key should be at least 32 characters long")

        # Production security checks
        if self.settings.app.environment == "production":
            if not self.settings.auth_enabled:
                self.warnings.append("Authentication disabled in production")
            
            if self.settings.app.debug:
                self.errors.append("Debug mode should be disabled in production")
            
            if self.settings.docs_enabled:
                self.warnings.append("API documentation enabled in production")

        self.info.append("Security configuration validated")

    def _validate_network(self) -> None:
        """Validate network configuration."""
        # Check API port
        if not (1 <= self.settings.api_port <= 65535):
            self.errors.append(f"Invalid API port: {self.settings.api_port}")
        
        if self._is_port_in_use(self.settings.api_port):
            self.warnings.append(f"API port {self.settings.api_port} is already in use")

        # Check worker count
        if self.settings.api_workers <= 0:
            self.errors.append("API worker count must be positive")
        
        if self.settings.api_workers > 16:
            self.warnings.append("High worker count may cause resource issues")

        # Check rate limit
        if self.settings.api_rate_limit <= 0:
            self.errors.append("API rate limit must be positive")

        self.info.append("Network configuration validated")

    def _validate_environment(self) -> None:
        """Validate environment-specific settings."""
        env = self.settings.app.environment

        # Check required environment variables
        required_vars = []
        if env == "production":
            required_vars.extend([
                "PYNOMALY_SECRET_KEY",
                "PYNOMALY_DATABASE_URL",
            ])

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            self.errors.extend([f"Missing required environment variable: {var}" for var in missing_vars])

        # Environment-specific warnings
        if env == "development" and self.settings.auth_enabled:
            self.info.append("Authentication enabled in development mode")

        if env == "test" and self.settings.cache_enabled:
            self.warnings.append("Cache enabled in test environment")

        self.info.append(f"Environment '{env}' configuration validated")

    def _test_network_connectivity(self, host: str, port: int, timeout: float = 5.0) -> bool:
        """Test network connectivity to host:port."""
        try:
            sock = socket.create_connection((host, port), timeout)
            sock.close()
            return True
        except (socket.error, socket.timeout):
            return False

    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is already in use."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            return result == 0
        except socket.error:
            return False


def validate_configuration(settings: Settings) -> dict[str, Any]:
    """Validate configuration and return results."""
    validator = ConfigurationValidator(settings)
    return validator.validate_all()


def check_configuration_health(settings: Settings) -> bool:
    """Quick health check of configuration."""
    validator = ConfigurationValidator(settings)
    results = validator.validate_all()
    return results["valid"]