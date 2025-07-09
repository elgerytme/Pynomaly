"""
Comprehensive infrastructure configuration tests.
Tests configuration management, validation, environment handling, and service registry.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import yaml

from pynomaly.domain.exceptions import (
    ConfigurationError,
    ValidationError,
)
from pynomaly.infrastructure.config.config_manager import ConfigManager
from pynomaly.infrastructure.config.config_validator import ConfigValidator
from pynomaly.infrastructure.config.feature_flags import FeatureFlags
from pynomaly.infrastructure.config.service_registry import ServiceRegistry
from pynomaly.infrastructure.config.simplified_container import SimplifiedContainer


class TestConfigManager:
    """Test suite for configuration manager."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration data."""
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "pynomaly_test",
                "username": "test_user",
                "password": "test_password",
            },
            "algorithms": {
                "default": "IsolationForest",
                "available": ["IsolationForest", "LOF", "OneClassSVM"],
                "hyperparameters": {
                    "IsolationForest": {"n_estimators": 100, "contamination": 0.1}
                },
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": ["console", "file"],
            },
            "performance": {
                "max_memory_usage": "1GB",
                "parallel_jobs": 4,
                "cache_size": "100MB",
            },
        }

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create config manager with temporary directory."""
        return ConfigManager(config_dir=str(temp_config_dir))

    def test_config_manager_load_yaml(
        self, config_manager, temp_config_dir, sample_config
    ):
        """Test loading configuration from YAML files."""
        # Create YAML config file
        config_file = temp_config_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Load configuration
        loaded_config = config_manager.load_config("config.yaml")
        assert loaded_config == sample_config

        # Test accessing nested values
        assert config_manager.get("database.host") == "localhost"
        assert config_manager.get("database.port") == 5432
        assert config_manager.get("algorithms.default") == "IsolationForest"

    def test_config_manager_load_json(
        self, config_manager, temp_config_dir, sample_config
    ):
        """Test loading configuration from JSON files."""
        # Create JSON config file
        config_file = temp_config_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(sample_config, f)

        # Load configuration
        loaded_config = config_manager.load_config("config.json")
        assert loaded_config == sample_config

    def test_config_manager_environment_override(
        self, config_manager, temp_config_dir, sample_config
    ):
        """Test environment variable overrides."""
        # Create base config
        config_file = temp_config_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Set environment variables
        env_overrides = {
            "PYNOMALY_DATABASE_HOST": "prod-server",
            "PYNOMALY_DATABASE_PORT": "3306",
            "PYNOMALY_ALGORITHMS_DEFAULT": "LOF",
        }

        with patch.dict(os.environ, env_overrides):
            config_manager.load_config("config.yaml")

            # Test environment overrides
            assert config_manager.get("database.host") == "prod-server"
            assert config_manager.get("database.port") == 3306
            assert config_manager.get("algorithms.default") == "LOF"

    def test_config_manager_profile_specific(
        self, config_manager, temp_config_dir, sample_config
    ):
        """Test profile-specific configurations."""
        # Create base config
        base_config_file = temp_config_dir / "config.yaml"
        with open(base_config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Create development profile config
        dev_config = {
            "database": {"host": "dev-server", "name": "pynomaly_dev"},
            "logging": {"level": "DEBUG"},
        }
        dev_config_file = temp_config_dir / "config-dev.yaml"
        with open(dev_config_file, "w") as f:
            yaml.dump(dev_config, f)

        # Load with development profile
        config_manager.load_config("config.yaml", profile="dev")

        # Test profile overrides
        assert config_manager.get("database.host") == "dev-server"
        assert config_manager.get("database.name") == "pynomaly_dev"
        assert config_manager.get("logging.level") == "DEBUG"
        # Base values should remain
        assert config_manager.get("database.port") == 5432

    def test_config_manager_default_values(self, config_manager):
        """Test default value handling."""
        # Test with no config loaded
        assert config_manager.get("nonexistent.key", "default_value") == "default_value"
        assert config_manager.get("another.missing.key", 42) == 42

        # Test with missing key in loaded config
        config_manager._config = {"existing": {"key": "value"}}
        assert config_manager.get("existing.key") == "value"
        assert config_manager.get("missing.key", "default") == "default"

    def test_config_manager_type_conversion(self, config_manager):
        """Test automatic type conversion."""
        config_manager._config = {
            "strings": {"value": "hello"},
            "integers": {"value": "123"},
            "floats": {"value": "3.14"},
            "booleans": {"true_val": "true", "false_val": "false"},
            "lists": {"value": "item1,item2,item3"},
        }

        # Test type conversions
        assert config_manager.get_int("integers.value") == 123
        assert config_manager.get_float("floats.value") == 3.14
        assert config_manager.get_bool("booleans.true_val") is True
        assert config_manager.get_bool("booleans.false_val") is False
        assert config_manager.get_list("lists.value") == ["item1", "item2", "item3"]

    def test_config_manager_validation(self, config_manager, temp_config_dir):
        """Test configuration validation."""
        # Create invalid config
        invalid_config = {
            "database": {
                "port": "invalid_port",  # Should be integer
                "timeout": -5,  # Should be positive
            }
        }

        config_file = temp_config_dir / "invalid_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        # Define validation schema
        schema = {
            "database": {
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "timeout": {"type": "integer", "minimum": 0},
            }
        }

        with pytest.raises(ConfigurationError):
            config_manager.load_config("invalid_config.yaml", validation_schema=schema)

    def test_config_manager_hot_reload(
        self, config_manager, temp_config_dir, sample_config
    ):
        """Test hot reloading of configuration."""
        config_file = temp_config_dir / "hot_reload_config.yaml"

        # Create initial config
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        config_manager.load_config("hot_reload_config.yaml")
        initial_host = config_manager.get("database.host")
        assert initial_host == "localhost"

        # Modify config file
        modified_config = sample_config.copy()
        modified_config["database"]["host"] = "modified-server"

        with open(config_file, "w") as f:
            yaml.dump(modified_config, f)

        # Trigger reload
        config_manager.reload_config()

        # Verify changes
        assert config_manager.get("database.host") == "modified-server"

    def test_config_manager_encryption(self, config_manager, temp_config_dir):
        """Test encrypted configuration values."""
        # Config with encrypted values
        encrypted_config = {
            "database": {
                "host": "localhost",
                "password": "encrypted:eyJhbGciOiJIUzI1NiJ9...",  # Mock encrypted value
            },
            "api_keys": {"service_key": "encrypted:eyJhbGciOiJIUzI1NiJ9..."},
        }

        config_file = temp_config_dir / "encrypted_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(encrypted_config, f)

        # Mock decryption function
        def mock_decrypt(encrypted_value):
            if encrypted_value.startswith("encrypted:"):
                return "decrypted_password"
            return encrypted_value

        with patch.object(config_manager, "_decrypt_value", side_effect=mock_decrypt):
            config_manager.load_config("encrypted_config.yaml")

            # Test decrypted values
            assert config_manager.get("database.password") == "decrypted_password"
            assert config_manager.get("api_keys.service_key") == "decrypted_password"
            # Non-encrypted values should remain unchanged
            assert config_manager.get("database.host") == "localhost"


class TestConfigValidator:
    """Test suite for configuration validator."""

    @pytest.fixture
    def config_validator(self):
        """Create config validator."""
        return ConfigValidator()

    def test_config_validator_schema_validation(self, config_validator):
        """Test JSON schema validation."""
        schema = {
            "type": "object",
            "properties": {
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                    },
                    "required": ["host", "port"],
                }
            },
            "required": ["database"],
        }

        # Valid config
        valid_config = {"database": {"host": "localhost", "port": 5432}}

        # Should not raise exception
        config_validator.validate(valid_config, schema)

        # Invalid config - missing required field
        invalid_config1 = {
            "database": {
                "host": "localhost"
                # Missing port
            }
        }

        with pytest.raises(ValidationError):
            config_validator.validate(invalid_config1, schema)

        # Invalid config - wrong type
        invalid_config2 = {"database": {"host": "localhost", "port": "not_a_number"}}

        with pytest.raises(ValidationError):
            config_validator.validate(invalid_config2, schema)

    def test_config_validator_custom_rules(self, config_validator):
        """Test custom validation rules."""

        # Define custom validation rule
        def validate_password_strength(password):
            if len(password) < 8:
                raise ValidationError("Password must be at least 8 characters")
            if not any(c.isupper() for c in password):
                raise ValidationError("Password must contain uppercase letter")
            if not any(c.islower() for c in password):
                raise ValidationError("Password must contain lowercase letter")
            if not any(c.isdigit() for c in password):
                raise ValidationError("Password must contain digit")

        config_validator.add_custom_rule(
            "password_strength", validate_password_strength
        )

        # Test valid password
        config_validator.apply_custom_rule("password_strength", "ValidPass123")

        # Test invalid passwords
        with pytest.raises(ValidationError):
            config_validator.apply_custom_rule("password_strength", "weak")

        with pytest.raises(ValidationError):
            config_validator.apply_custom_rule("password_strength", "nouppercase123")

    def test_config_validator_environment_specific(self, config_validator):
        """Test environment-specific validation."""
        production_schema = {
            "type": "object",
            "properties": {
                "logging": {
                    "properties": {
                        "level": {
                            "enum": ["WARNING", "ERROR"]
                        }  # Stricter in production
                    }
                },
                "security": {
                    "properties": {
                        "ssl_enabled": {"const": True}  # Must be enabled in production
                    },
                    "required": ["ssl_enabled"],
                },
            },
        }

        development_schema = {
            "type": "object",
            "properties": {
                "logging": {
                    "properties": {
                        "level": {
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]
                        }  # More lenient
                    }
                }
            },
        }

        config = {"logging": {"level": "DEBUG"}, "security": {"ssl_enabled": False}}

        # Should pass development validation
        config_validator.validate(config, development_schema, environment="development")

        # Should fail production validation
        with pytest.raises(ValidationError):
            config_validator.validate(
                config, production_schema, environment="production"
            )


class TestServiceRegistry:
    """Test suite for service registry."""

    @pytest.fixture
    def service_registry(self):
        """Create service registry."""
        return ServiceRegistry()

    def test_service_registry_registration(self, service_registry):
        """Test service registration and retrieval."""
        # Mock services
        mock_database = Mock()
        mock_cache = Mock()
        mock_logger = Mock()

        # Register services
        service_registry.register("database", mock_database)
        service_registry.register("cache", mock_cache)
        service_registry.register("logger", mock_logger)

        # Test retrieval
        assert service_registry.get("database") is mock_database
        assert service_registry.get("cache") is mock_cache
        assert service_registry.get("logger") is mock_logger
        assert service_registry.get("nonexistent") is None

    def test_service_registry_singleton_behavior(self, service_registry):
        """Test singleton service behavior."""

        class SingletonService:
            def __init__(self):
                self.id = uuid4()

        # Register singleton service
        service_registry.register_singleton("singleton", SingletonService)

        # Get service multiple times
        service1 = service_registry.get("singleton")
        service2 = service_registry.get("singleton")

        # Should be same instance
        assert service1 is service2
        assert service1.id == service2.id

    def test_service_registry_factory_pattern(self, service_registry):
        """Test factory pattern for services."""

        class ConfigurableService:
            def __init__(self, config):
                self.config = config

        # Register factory
        def service_factory(config=None):
            return ConfigurableService(config or {"default": True})

        service_registry.register_factory("configurable", service_factory)

        # Get services with different configs
        service1 = service_registry.get_from_factory("configurable")
        service2 = service_registry.get_from_factory(
            "configurable", config={"custom": True}
        )

        assert service1.config == {"default": True}
        assert service2.config == {"custom": True}
        assert service1 is not service2

    def test_service_registry_dependency_injection(self, service_registry):
        """Test dependency injection."""
        # Register dependencies
        mock_logger = Mock()
        mock_config = Mock()
        service_registry.register("logger", mock_logger)
        service_registry.register("config", mock_config)

        # Service that depends on others
        class DependentService:
            def __init__(self, logger, config):
                self.logger = logger
                self.config = config

        # Register with dependency injection
        service_registry.register_with_deps(
            "dependent", DependentService, dependencies=["logger", "config"]
        )

        # Get service - should have dependencies injected
        dependent_service = service_registry.get("dependent")
        assert dependent_service.logger is mock_logger
        assert dependent_service.config is mock_config

    def test_service_registry_lifecycle_management(self, service_registry):
        """Test service lifecycle management."""

        class ManagedService:
            def __init__(self):
                self.started = False
                self.stopped = False

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        service = ManagedService()
        service_registry.register("managed", service)

        # Test lifecycle methods
        service_registry.start_service("managed")
        assert service.started is True

        service_registry.stop_service("managed")
        assert service.stopped is True

        # Test bulk lifecycle operations
        service_registry.start_all()
        service_registry.stop_all()

    def test_service_registry_health_checks(self, service_registry):
        """Test service health checking."""

        class HealthyService:
            def health_check(self):
                return {"status": "healthy", "details": "All good"}

        class UnhealthyService:
            def health_check(self):
                return {"status": "unhealthy", "details": "Database connection failed"}

        service_registry.register("healthy", HealthyService())
        service_registry.register("unhealthy", UnhealthyService())

        # Test individual health checks
        healthy_status = service_registry.check_service_health("healthy")
        assert healthy_status["status"] == "healthy"

        unhealthy_status = service_registry.check_service_health("unhealthy")
        assert unhealthy_status["status"] == "unhealthy"

        # Test overall health check
        overall_health = service_registry.check_overall_health()
        assert overall_health["overall_status"] == "degraded"  # One unhealthy service
        assert len(overall_health["services"]) == 2


class TestFeatureFlags:
    """Test suite for feature flags."""

    @pytest.fixture
    def feature_flags(self):
        """Create feature flags manager."""
        return FeatureFlags()

    def test_feature_flags_basic_operations(self, feature_flags):
        """Test basic feature flag operations."""
        # Set feature flags
        feature_flags.set("new_algorithm", True)
        feature_flags.set("beta_ui", False)
        feature_flags.set("experimental_feature", True)

        # Test retrieval
        assert feature_flags.is_enabled("new_algorithm") is True
        assert feature_flags.is_enabled("beta_ui") is False
        assert feature_flags.is_enabled("experimental_feature") is True
        assert feature_flags.is_enabled("nonexistent_feature") is False

    def test_feature_flags_percentage_rollout(self, feature_flags):
        """Test percentage-based feature rollout."""
        # Set feature with 50% rollout
        feature_flags.set_percentage("gradual_rollout", 50)

        # Test with different user IDs
        enabled_count = 0
        total_tests = 100

        for i in range(total_tests):
            user_id = f"user_{i}"
            if feature_flags.is_enabled("gradual_rollout", user_id=user_id):
                enabled_count += 1

        # Should be approximately 50% (allow some variance)
        assert 30 <= enabled_count <= 70

    def test_feature_flags_user_targeting(self, feature_flags):
        """Test user-specific feature targeting."""
        # Enable feature for specific users
        feature_flags.set_for_users("vip_feature", ["user_1", "user_2", "user_admin"])

        # Test targeted users
        assert feature_flags.is_enabled("vip_feature", user_id="user_1") is True
        assert feature_flags.is_enabled("vip_feature", user_id="user_2") is True
        assert feature_flags.is_enabled("vip_feature", user_id="user_admin") is True

        # Test non-targeted users
        assert feature_flags.is_enabled("vip_feature", user_id="user_3") is False
        assert feature_flags.is_enabled("vip_feature", user_id="random_user") is False

    def test_feature_flags_environment_based(self, feature_flags):
        """Test environment-based feature flags."""
        # Set environment-specific flags
        feature_flags.set_for_environment("dev_tools", "development", True)
        feature_flags.set_for_environment("dev_tools", "production", False)
        feature_flags.set_for_environment("monitoring", "production", True)

        # Test in different environments
        assert feature_flags.is_enabled("dev_tools", environment="development") is True
        assert feature_flags.is_enabled("dev_tools", environment="production") is False
        assert feature_flags.is_enabled("monitoring", environment="production") is True

    def test_feature_flags_time_based(self, feature_flags):
        """Test time-based feature activation."""
        from datetime import datetime, timedelta

        # Set feature to be enabled in the future
        future_time = datetime.now() + timedelta(hours=1)
        feature_flags.set_time_based("future_feature", start_time=future_time)

        # Should be disabled now
        assert feature_flags.is_enabled("future_feature") is False

        # Set feature with expiration
        past_time = datetime.now() - timedelta(hours=1)
        feature_flags.set_time_based("expired_feature", end_time=past_time)

        # Should be disabled (expired)
        assert feature_flags.is_enabled("expired_feature") is False

    def test_feature_flags_configuration_loading(self, feature_flags, temp_config_dir):
        """Test loading feature flags from configuration."""
        # Create feature flags config
        flags_config = {
            "features": {
                "new_algorithm": True,
                "beta_ui": False,
                "experimental": {
                    "enabled": True,
                    "percentage": 25,
                    "environments": ["development", "staging"],
                },
            }
        }

        config_file = temp_config_dir / "features.yaml"
        with open(config_file, "w") as f:
            yaml.dump(flags_config, f)

        # Load configuration
        feature_flags.load_from_config(str(config_file))

        # Test loaded flags
        assert feature_flags.is_enabled("new_algorithm") is True
        assert feature_flags.is_enabled("beta_ui") is False


class TestSimplifiedContainer:
    """Test suite for simplified dependency injection container."""

    @pytest.fixture
    def container(self):
        """Create simplified container."""
        return SimplifiedContainer()

    def test_container_service_registration(self, container):
        """Test service registration in container."""
        # Register services
        mock_service = Mock()
        container.register("test_service", mock_service)

        # Register with factory
        def factory():
            return Mock(id="factory_created")

        container.register_factory("factory_service", factory)

        # Test retrieval
        assert container.get("test_service") is mock_service

        factory_service = container.get("factory_service")
        assert factory_service.id == "factory_created"

    def test_container_configuration_injection(
        self, container, temp_config_dir, sample_config
    ):
        """Test configuration injection."""
        # Create config file
        config_file = temp_config_dir / "container_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Register config manager
        config_manager = ConfigManager(config_dir=str(temp_config_dir))
        config_manager.load_config("container_config.yaml")
        container.register("config", config_manager)

        # Service that uses configuration
        class ConfigurableService:
            def __init__(self, config):
                self.database_host = config.get("database.host")
                self.algorithm = config.get("algorithms.default")

        # Register service with config injection
        container.register_with_deps(
            "configurable_service", ConfigurableService, dependencies=["config"]
        )

        # Test service creation
        service = container.get("configurable_service")
        assert service.database_host == "localhost"
        assert service.algorithm == "IsolationForest"

    def test_container_circular_dependency_detection(self, container):
        """Test circular dependency detection."""

        class ServiceA:
            def __init__(self, service_b):
                self.service_b = service_b

        class ServiceB:
            def __init__(self, service_a):
                self.service_a = service_a

        # Register services with circular dependencies
        container.register_with_deps("service_a", ServiceA, dependencies=["service_b"])
        container.register_with_deps("service_b", ServiceB, dependencies=["service_a"])

        # Should detect circular dependency
        with pytest.raises(ConfigurationError):
            container.get("service_a")

    def test_container_lazy_loading(self, container):
        """Test lazy loading of services."""
        creation_count = {"count": 0}

        class LazyService:
            def __init__(self):
                creation_count["count"] += 1
                self.id = creation_count["count"]

        # Register as lazy singleton
        container.register_lazy_singleton("lazy_service", LazyService)

        # Service should not be created yet
        assert creation_count["count"] == 0

        # First access should create the service
        service1 = container.get("lazy_service")
        assert creation_count["count"] == 1
        assert service1.id == 1

        # Second access should return same instance
        service2 = container.get("lazy_service")
        assert creation_count["count"] == 1  # Still only created once
        assert service1 is service2

    def test_container_scoped_services(self, container):
        """Test scoped service management."""

        class ScopedService:
            def __init__(self, scope_id):
                self.scope_id = scope_id

        # Register scoped service
        container.register_scoped("scoped_service", ScopedService)

        # Create services in different scopes
        with container.scope("scope1"):
            service1 = container.get("scoped_service")
            assert service1.scope_id == "scope1"

            # Same scope should return same instance
            service1_again = container.get("scoped_service")
            assert service1 is service1_again

        # Different scope should create new instance
        with container.scope("scope2"):
            service2 = container.get("scoped_service")
            assert service2.scope_id == "scope2"
            assert service1 is not service2


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_full_configuration_lifecycle(self, temp_config_dir):
        """Test complete configuration lifecycle."""
        # Create complex configuration
        main_config = {
            "app": {"name": "pynomaly", "version": "1.0.0"},
            "database": {"host": "localhost", "port": 5432},
            "features": {"new_algorithm": True, "beta_ui": False},
        }

        # Save configuration
        config_file = temp_config_dir / "main_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(main_config, f)

        # Initialize components
        config_manager = ConfigManager(config_dir=str(temp_config_dir))
        config_validator = ConfigValidator()
        feature_flags = FeatureFlags()
        service_registry = ServiceRegistry()
        container = SimplifiedContainer()

        # Load and validate configuration
        config_manager.load_config("main_config.yaml")

        # Register components in container
        container.register("config", config_manager)
        container.register("validator", config_validator)
        container.register("features", feature_flags)
        container.register("registry", service_registry)

        # Load feature flags from config
        feature_flags.set("new_algorithm", config_manager.get("features.new_algorithm"))
        feature_flags.set("beta_ui", config_manager.get("features.beta_ui"))

        # Test integrated functionality
        assert config_manager.get("app.name") == "pynomaly"
        assert feature_flags.is_enabled("new_algorithm") is True
        assert feature_flags.is_enabled("beta_ui") is False

        # Test service creation with configuration
        class DatabaseService:
            def __init__(self, config):
                self.host = config.get("database.host")
                self.port = config.get("database.port")

        container.register_with_deps(
            "database", DatabaseService, dependencies=["config"]
        )
        db_service = container.get("database")

        assert db_service.host == "localhost"
        assert db_service.port == 5432

    def test_configuration_error_recovery(self, temp_config_dir):
        """Test configuration error recovery mechanisms."""
        # Create invalid configuration
        invalid_config = {"database": {"port": "invalid_port"}}

        config_file = temp_config_dir / "invalid_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        config_manager = ConfigManager(config_dir=str(temp_config_dir))

        # Should handle invalid config gracefully
        try:
            config_manager.load_config("invalid_config.yaml")
            # If no exception, check that defaults are used
            port = config_manager.get_int("database.port", default=5432)
            assert port == 5432
        except ConfigurationError:
            # If exception is raised, test fallback mechanism
            config_manager.load_defaults()
            assert config_manager.get("database.port", 5432) == 5432
