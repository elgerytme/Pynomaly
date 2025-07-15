"""Comprehensive tests for infrastructure configuration - Phase 2 Coverage."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from pynomaly.domain.exceptions import ConfigurationError
from pynomaly.infrastructure.config import Settings
from pynomaly.infrastructure.config.container import Container


@pytest.fixture
def sample_config_dict():
    """Create sample configuration dictionary."""
    return {
        "database": {"url": "sqlite:///test.db", "echo": False, "pool_size": 5},
        "redis": {"host": "localhost", "port": 6379, "db": 0, "password": None},
        "monitoring": {
            "enabled": True,
            "endpoint": "http://localhost:8080/metrics",
            "export_interval": 30,
        },
        "algorithms": {
            "default_contamination": 0.1,
            "enable_gpu": False,
            "cache_models": True,
        },
        "api": {"host": "0.0.0.0", "port": 8000, "debug": False, "cors_origins": ["*"]},
    }


@pytest.fixture
def env_variables():
    """Set up environment variables for testing."""
    original_env = dict(os.environ)

    # Set test environment variables
    test_env = {
        "PYNOMALY_DATABASE_URL": "postgresql://test:test@localhost/test",
        "PYNOMALY_REDIS_HOST": "redis.example.com",
        "PYNOMALY_REDIS_PORT": "6380",
        "PYNOMALY_API_DEBUG": "true",
        "PYNOMALY_MONITORING_ENABLED": "false",
    }

    os.environ.update(test_env)

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


class TestSettings:
    """Comprehensive tests for Settings configuration."""

    def test_settings_default_values(self):
        """Test default settings values."""
        settings = Settings()

        # Test database defaults
        assert settings.database_url == "sqlite:///pynomaly.db"
        assert settings.database_echo is False
        assert settings.database_pool_size == 10

        # Test Redis defaults
        assert settings.redis_host == "localhost"
        assert settings.redis_port == 6379
        assert settings.redis_db == 0
        assert settings.redis_password is None

        # Test API defaults
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.api_debug is False

        # Test monitoring defaults
        assert settings.monitoring_enabled is True
        assert settings.monitoring_endpoint == "http://localhost:8080/metrics"

    def test_settings_environment_override(self, env_variables):
        """Test environment variable override of settings."""
        settings = Settings()

        # Test environment overrides
        assert settings.database_url == "postgresql://test:test@localhost/test"
        assert settings.redis_host == "redis.example.com"
        assert settings.redis_port == 6380
        assert settings.api_debug is True
        assert settings.monitoring_enabled is False

    def test_settings_validation(self):
        """Test settings validation."""
        # Test valid settings
        valid_settings = Settings(
            database_url="sqlite:///valid.db", redis_port=6379, api_port=8000
        )
        assert valid_settings.redis_port == 6379

        # Test invalid port (should be clamped or raise error)
        with pytest.raises(ValueError):
            Settings(redis_port=-1)

        with pytest.raises(ValueError):
            Settings(api_port=0)

    def test_settings_nested_configuration(self, sample_config_dict):
        """Test nested configuration structure."""
        settings = Settings(**sample_config_dict)

        # Test nested access
        assert settings.database["url"] == "sqlite:///test.db"
        assert settings.redis["host"] == "localhost"
        assert settings.monitoring["enabled"] is True
        assert settings.algorithms["default_contamination"] == 0.1

    def test_settings_type_conversion(self):
        """Test automatic type conversion in settings."""
        # Test string to bool conversion
        with patch.dict(os.environ, {"PYNOMALY_API_DEBUG": "true"}):
            settings = Settings()
            assert settings.api_debug is True

        with patch.dict(os.environ, {"PYNOMALY_API_DEBUG": "false"}):
            settings = Settings()
            assert settings.api_debug is False

        # Test string to int conversion
        with patch.dict(os.environ, {"PYNOMALY_API_PORT": "9000"}):
            settings = Settings()
            assert settings.api_port == 9000

    def test_settings_file_loading(self, sample_config_dict):
        """Test loading settings from configuration files."""
        # Test YAML file loading
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config_dict, f)
            yaml_path = f.name

        # Test JSON file loading
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_config_dict, f)
            json_path = f.name

        try:
            # Test YAML loading
            yaml_settings = Settings.load_from_file(yaml_path)
            assert yaml_settings.database["url"] == "sqlite:///test.db"

            # Test JSON loading
            json_settings = Settings.load_from_file(json_path)
            assert json_settings.database["url"] == "sqlite:///test.db"

        finally:
            Path(yaml_path).unlink()
            Path(json_path).unlink()

    def test_settings_serialization(self, sample_config_dict):
        """Test settings serialization and deserialization."""
        settings = Settings(**sample_config_dict)

        # Test to dict
        settings_dict = settings.to_dict()
        assert "database" in settings_dict
        assert "redis" in settings_dict

        # Test to JSON
        settings_json = settings.to_json()
        assert isinstance(settings_json, str)

        # Test from JSON
        restored_settings = Settings.from_json(settings_json)
        assert restored_settings.database["url"] == settings.database["url"]

    def test_settings_profile_management(self):
        """Test configuration profile management."""
        # Test development profile
        dev_settings = Settings.load_profile("development")
        assert dev_settings.api_debug is True
        assert dev_settings.database_echo is True

        # Test production profile
        prod_settings = Settings.load_profile("production")
        assert prod_settings.api_debug is False
        assert prod_settings.database_echo is False

        # Test testing profile
        test_settings = Settings.load_profile("testing")
        assert (
            "memory" in test_settings.database_url
            or "test" in test_settings.database_url
        )

    def test_settings_secret_management(self):
        """Test secret and sensitive data management."""
        with patch.dict(
            os.environ,
            {
                "PYNOMALY_DATABASE_PASSWORD": "secret_password",
                "PYNOMALY_JWT_SECRET": "jwt_secret_key",
            },
        ):
            settings = Settings()

            # Test that secrets are loaded
            assert settings.database_password == "secret_password"
            assert settings.jwt_secret == "jwt_secret_key"

            # Test that secrets are masked in serialization
            settings_dict = settings.to_dict(mask_secrets=True)
            assert settings_dict["database_password"] == "***"
            assert settings_dict["jwt_secret"] == "***"


class TestDependencyContainer:
    """Comprehensive tests for dependency injection container."""

    def test_container_service_registration(self):
        """Test service registration in container."""
        container = Container()

        # Test singleton registration
        container.register_singleton("test_service", lambda: "singleton_instance")

        # Test transient registration
        container.register_transient("test_transient", lambda: "transient_instance")

        # Test factory registration
        container.register_factory("test_factory", lambda: Mock())

        # Verify registrations
        assert "test_service" in container._services
        assert "test_transient" in container._services
        assert "test_factory" in container._services

    def test_container_service_resolution(self):
        """Test service resolution from container."""
        container = Container()

        # Register services
        container.register_singleton("singleton", lambda: Mock(name="singleton"))
        container.register_transient("transient", lambda: Mock(name="transient"))

        # Test singleton behavior
        singleton1 = container.resolve("singleton")
        singleton2 = container.resolve("singleton")
        assert singleton1 is singleton2  # Same instance

        # Test transient behavior
        transient1 = container.resolve("transient")
        transient2 = container.resolve("transient")
        assert transient1 is not transient2  # Different instances

    def test_container_dependency_injection(self):
        """Test automatic dependency injection."""
        container = Container()

        # Define services with dependencies
        class ServiceA:
            def __init__(self):
                self.name = "ServiceA"

        class ServiceB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a
                self.name = "ServiceB"

        # Register services
        container.register_singleton("service_a", ServiceA)
        container.register_singleton(
            "service_b", lambda: ServiceB(container.resolve("service_a"))
        )

        # Resolve with dependencies
        service_b = container.resolve("service_b")
        assert service_b.name == "ServiceB"
        assert service_b.service_a.name == "ServiceA"

    def test_container_conditional_registration(self):
        """Test conditional service registration."""
        container = Container()

        # Test conditional registration based on settings
        settings = Settings(monitoring_enabled=True)

        container.register_conditional(
            "monitoring_service",
            lambda: Mock(name="monitoring"),
            condition=lambda: settings.monitoring_enabled,
        )

        container.register_conditional(
            "disabled_service", lambda: Mock(name="disabled"), condition=lambda: False
        )

        # Test conditional resolution
        monitoring = container.resolve("monitoring_service")
        assert monitoring is not None

        with pytest.raises(KeyError):
            container.resolve("disabled_service")

    def test_container_scoped_services(self):
        """Test scoped service lifetimes."""
        container = Container()

        # Register scoped service
        container.register_scoped("scoped_service", lambda: Mock(name="scoped"))

        # Test within scope
        with container.create_scope() as scope:
            service1 = scope.resolve("scoped_service")
            service2 = scope.resolve("scoped_service")
            assert service1 is service2  # Same within scope

        # Test different scope
        with container.create_scope() as scope2:
            service3 = scope2.resolve("scoped_service")
            assert service3 is not service1  # Different across scopes

    def test_container_configuration_integration(self, sample_config_dict):
        """Test container integration with configuration."""
        settings = Settings(**sample_config_dict)
        container = Container(settings)

        # Test configuration-based registration
        container.register_from_config()

        # Test that services are registered based on config
        if settings.database["url"]:
            assert "database_session" in container._services

        if settings.redis["host"]:
            assert "redis_client" in container._services

        if settings.monitoring["enabled"]:
            assert "telemetry_service" in container._services

    def test_container_lazy_initialization(self):
        """Test lazy initialization of services."""
        container = Container()

        initialization_count = [0]

        def expensive_service():
            initialization_count[0] += 1
            return Mock(name=f"expensive_{initialization_count[0]}")

        # Register with lazy initialization
        container.register_lazy("expensive_service", expensive_service)

        # Service should not be initialized yet
        assert initialization_count[0] == 0

        # First resolution should initialize
        service1 = container.resolve("expensive_service")
        assert initialization_count[0] == 1

        # Second resolution should reuse
        service2 = container.resolve("expensive_service")
        assert initialization_count[0] == 1
        assert service1 is service2

    def test_container_service_decoration(self):
        """Test service decoration and middleware."""
        container = Container()

        # Base service
        class BaseService:
            def process(self, data):
                return f"processed: {data}"

        # Decorator
        class LoggingDecorator:
            def __init__(self, service):
                self.service = service
                self.logs = []

            def process(self, data):
                self.logs.append(f"processing: {data}")
                result = self.service.process(data)
                self.logs.append(f"completed: {result}")
                return result

        # Register with decoration
        container.register_singleton("base_service", BaseService)
        container.register_decorator(
            "base_service", lambda service: LoggingDecorator(service)
        )

        # Test decorated service
        service = container.resolve("base_service")
        result = service.process("test")

        assert result == "processed: test"
        assert len(service.logs) == 2
        assert "processing: test" in service.logs

    def test_container_circular_dependency_detection(self):
        """Test circular dependency detection."""
        container = Container()

        # Define circular dependencies
        container.register_singleton(
            "service_a", lambda: container.resolve("service_b")
        )
        container.register_singleton(
            "service_b", lambda: container.resolve("service_a")
        )

        # Should detect and handle circular dependency
        with pytest.raises(ConfigurationError, match="Circular dependency"):
            container.resolve("service_a")

    def test_container_auto_wiring(self):
        """Test automatic service wiring based on type hints."""
        container = Container()

        # Define services with type hints
        class Repository:
            def __init__(self):
                self.data = []

        class Service:
            def __init__(self, repo: Repository):
                self.repo = repo

        class Controller:
            def __init__(self, service: Service):
                self.service = service

        # Register services
        container.register_singleton(Repository, Repository)
        container.register_singleton(Service, Service)
        container.register_singleton(Controller, Controller)

        # Auto-wire and resolve
        controller = container.resolve(Controller)
        assert isinstance(controller.service, Service)
        assert isinstance(controller.service.repo, Repository)


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_configuration_schema_validation(self):
        """Test configuration schema validation."""
        # Test valid configuration
        valid_config = {
            "database": {"url": "sqlite:///test.db", "pool_size": 5},
            "api": {"port": 8000, "debug": False},
        }

        settings = Settings(**valid_config)
        validation_result = settings.validate_schema()
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0

        # Test invalid configuration
        invalid_config = {
            "database": {
                "url": "invalid_url",
                "pool_size": -1,  # Invalid negative pool size
            },
            "api": {
                "port": "not_a_number",  # Invalid port type
                "debug": "not_a_boolean",  # Invalid debug type
            },
        }

        with pytest.raises(ValueError):
            Settings(**invalid_config)

    def test_configuration_environment_validation(self):
        """Test environment-specific configuration validation."""
        # Test production environment requirements
        prod_config = {
            "environment": "production",
            "database": {"url": "sqlite:///test.db"},  # SQLite not allowed in prod
            "api": {"debug": True},  # Debug not allowed in prod
        }

        settings = Settings(**prod_config)
        validation_result = settings.validate_for_environment("production")

        assert validation_result.is_valid is False
        assert any(
            "SQLite not recommended for production" in error
            for error in validation_result.warnings
        )
        assert any(
            "Debug mode should be disabled" in error
            for error in validation_result.errors
        )

    def test_configuration_security_validation(self):
        """Test security-related configuration validation."""
        # Test insecure configuration
        insecure_config = {
            "api": {
                "cors_origins": ["*"],  # Too permissive
                "debug": True,  # Debug enabled
            },
            "database": {"password": None},  # No password
            "jwt": {"secret": "weak"},  # Weak secret
        }

        settings = Settings(**insecure_config)
        security_validation = settings.validate_security()

        assert security_validation.is_valid is False
        assert len(security_validation.warnings) > 0
        assert any(
            "CORS origins too permissive" in warning
            for warning in security_validation.warnings
        )

    def test_configuration_dependency_validation(self):
        """Test dependency configuration validation."""
        # Test missing required dependencies
        config_missing_deps = {
            "redis": {"enabled": True},  # Redis enabled but no connection info
            "monitoring": {"enabled": True},  # Monitoring enabled but no endpoint
        }

        settings = Settings(**config_missing_deps)
        dep_validation = settings.validate_dependencies()

        assert dep_validation.is_valid is False
        assert any(
            "Redis connection information missing" in error
            for error in dep_validation.errors
        )

    def test_configuration_performance_validation(self):
        """Test performance-related configuration validation."""
        # Test performance-impacting configuration
        perf_config = {
            "database": {
                "pool_size": 1,  # Too small
                "echo": True,  # Performance impact
            },
            "algorithms": {
                "cache_models": False,  # Performance impact
                "batch_size": 1,  # Too small
            },
        }

        settings = Settings(**perf_config)
        perf_validation = settings.validate_performance()

        assert len(perf_validation.warnings) > 0
        assert any(
            "Small connection pool" in warning for warning in perf_validation.warnings
        )


class TestConfigurationProfiles:
    """Test configuration profile management."""

    def test_profile_inheritance(self):
        """Test configuration profile inheritance."""
        # Base configuration
        base_config = {
            "database": {"pool_size": 10},
            "api": {"port": 8000},
            "monitoring": {"enabled": True},
        }

        # Development profile (inherits from base)

        # Load with inheritance
        settings = Settings.load_profile("development", base_config=base_config)

        # Test inheritance
        assert settings.database["pool_size"] == 10  # From base
        assert settings.api["port"] == 8000  # From base
        assert settings.api["debug"] is True  # From dev
        assert settings.database["echo"] is True  # From dev
        assert settings.monitoring["enabled"] is True  # From base

    def test_profile_environment_override(self):
        """Test profile with environment variable override."""
        with patch.dict(
            os.environ, {"PYNOMALY_PROFILE": "testing", "PYNOMALY_API_DEBUG": "false"}
        ):
            settings = Settings.load_from_environment()

            # Should load testing profile but override debug
            assert settings.profile == "testing"
            assert settings.api_debug is False  # Overridden by env

    def test_profile_validation(self):
        """Test profile-specific validation."""
        # Test development profile validation
        dev_settings = Settings.load_profile("development")
        dev_validation = dev_settings.validate_profile()

        # Development allows debug mode, in-memory databases, etc.
        assert dev_validation.is_valid is True

        # Test production profile validation
        prod_settings = Settings.load_profile("production")
        prod_validation = prod_settings.validate_profile()

        # Production should have stricter requirements
        if prod_settings.api_debug:
            assert prod_validation.is_valid is False

    def test_custom_profile_creation(self):
        """Test creating custom configuration profiles."""
        custom_profile = {
            "name": "custom",
            "description": "Custom configuration for specific deployment",
            "config": {
                "database": {
                    "url": "postgresql://custom:pass@db:5432/custom",
                    "pool_size": 20,
                },
                "api": {"port": 9000, "workers": 4},
                "algorithms": {"enable_gpu": True, "model_cache_size": "2GB"},
            },
        }

        # Register custom profile
        Settings.register_profile("custom", custom_profile["config"])

        # Load custom profile
        custom_settings = Settings.load_profile("custom")
        assert custom_settings.database["pool_size"] == 20
        assert custom_settings.api["port"] == 9000
        assert custom_settings.algorithms["enable_gpu"] is True


class TestConfigurationMonitoring:
    """Test configuration monitoring and hot reloading."""

    def test_configuration_change_detection(self):
        """Test configuration change detection."""
        settings = Settings()

        # Set up change detection
        change_events = []
        settings.on_change(
            lambda key, old_val, new_val: change_events.append((key, old_val, new_val))
        )

        # Make changes
        settings.api_debug = True
        settings.database_pool_size = 20

        # Verify change detection
        assert len(change_events) == 2
        assert ("api_debug", False, True) in change_events
        assert any("database_pool_size" in event[0] for event in change_events)

    def test_configuration_hot_reload(self, sample_config_dict):
        """Test hot reloading of configuration."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_config_dict, f)
            config_path = f.name

        try:
            # Load settings with hot reload
            settings = Settings.load_with_hot_reload(config_path, check_interval=0.1)

            settings.api["port"]

            # Modify config file
            modified_config = sample_config_dict.copy()
            modified_config["api"]["port"] = 9000

            with open(config_path, "w") as f:
                json.dump(modified_config, f)

            # Wait for hot reload (in real implementation)
            import time

            time.sleep(0.2)

            # Verify reload (mock behavior)
            with patch.object(settings, "_reload_from_file") as mock_reload:
                settings._check_for_changes()
                mock_reload.assert_called_once()

        finally:
            Path(config_path).unlink()

    def test_configuration_health_monitoring(self):
        """Test configuration health monitoring."""
        settings = Settings()

        # Get configuration health status
        health_status = settings.get_health_status()

        assert "status" in health_status
        assert "checks" in health_status
        assert "timestamp" in health_status

        # Verify individual health checks
        checks = health_status["checks"]
        assert "database_connectivity" in checks
        assert "redis_connectivity" in checks
        assert "external_services" in checks

        # Test with unhealthy configuration
        unhealthy_settings = Settings(
            database_url="invalid://invalid", redis_host="nonexistent.host"
        )

        unhealthy_status = unhealthy_settings.get_health_status()
        assert unhealthy_status["status"] == "unhealthy"
        assert any(
            check["status"] == "unhealthy"
            for check in unhealthy_status["checks"].values()
        )


class TestConfigurationExtensions:
    """Test configuration extensions and plugins."""

    def test_configuration_plugin_system(self):
        """Test configuration plugin system."""

        # Define a configuration plugin
        class CustomPlugin:
            def __init__(self, settings):
                self.settings = settings

            def configure(self):
                return {
                    "custom_service": lambda: Mock(name="custom"),
                    "custom_setting": "plugin_value",
                }

        settings = Settings()

        # Register plugin
        settings.register_plugin("custom", CustomPlugin)

        # Load plugin configuration
        plugin_config = settings.load_plugin_config("custom")

        assert "custom_service" in plugin_config
        assert plugin_config["custom_setting"] == "plugin_value"

    def test_configuration_middleware(self):
        """Test configuration middleware."""
        settings = Settings()

        # Add configuration middleware
        def logging_middleware(key, value, next_func):
            print(f"Setting {key} = {value}")
            return next_func(key, value)

        def validation_middleware(key, value, next_func):
            if key == "api_port" and not (1000 <= value <= 65535):
                raise ValueError(f"Invalid port: {value}")
            return next_func(key, value)

        settings.add_middleware(logging_middleware)
        settings.add_middleware(validation_middleware)

        # Test middleware execution
        with patch("builtins.print") as mock_print:
            settings.api_port = 8080
            mock_print.assert_called_with("Setting api_port = 8080")

        # Test validation middleware
        with pytest.raises(ValueError, match="Invalid port"):
            settings.api_port = 99999

    def test_configuration_templating(self):
        """Test configuration templating and variable substitution."""
        template_config = {
            "database": {
                "url": "${DATABASE_URL:sqlite:///default.db}",
                "pool_size": "${DATABASE_POOL_SIZE:10}",
            },
            "api": {"host": "${API_HOST:localhost}", "port": "${API_PORT:8000}"},
        }

        # Test with environment variables
        with patch.dict(
            os.environ,
            {"DATABASE_URL": "postgresql://user:pass@db:5432/prod", "API_PORT": "9000"},
        ):
            settings = Settings.from_template(template_config)

            assert settings.database["url"] == "postgresql://user:pass@db:5432/prod"
            assert settings.database["pool_size"] == 10  # Default value
            assert settings.api["host"] == "localhost"  # Default value
            assert settings.api["port"] == 9000  # From environment

    def test_configuration_encryption(self):
        """Test configuration encryption for sensitive data."""
        sensitive_config = {
            "database": {"password": "encrypted:AES256:base64_encrypted_password"},
            "api": {"secret_key": "encrypted:AES256:base64_encrypted_secret"},
        }

        # Mock encryption key
        with patch("pynomaly.infrastructure.config.get_encryption_key") as mock_key:
            mock_key.return_value = b"0" * 32  # 256-bit key

            settings = Settings.load_encrypted(sensitive_config)

            # Verify decryption occurred
            assert not settings.database["password"].startswith("encrypted:")
            assert not settings.api["secret_key"].startswith("encrypted:")

            # Test encryption on save
            encrypted_config = settings.save_encrypted()
            assert encrypted_config["database"]["password"].startswith("encrypted:")
            assert encrypted_config["api"]["secret_key"].startswith("encrypted:")
