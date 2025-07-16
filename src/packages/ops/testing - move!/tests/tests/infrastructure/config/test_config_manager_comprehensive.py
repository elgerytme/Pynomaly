"""Comprehensive tests for configuration management infrastructure."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from monorepo.domain.exceptions import ConfigurationError
from monorepo.infrastructure.config.config_manager import (
    CacheConfigManager,
    ConfigurationManager,
    DatabaseConfigManager,
    MonitoringConfigManager,
    create_config_manager,
)
from monorepo.infrastructure.config.settings import Settings


class TestConfigurationManager:
    """Test configuration manager functionality."""

    def test_initialization_defaults(self):
        """Test configuration manager initialization with defaults."""
        manager = ConfigurationManager()

        assert manager.config_dir == Path("config")
        assert manager.environment in ["development", "test", "production"]
        assert manager._settings is None
        assert manager._config_cache == {}

    def test_initialization_custom_values(self):
        """Test configuration manager initialization with custom values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            manager = ConfigurationManager(
                config_dir=config_dir,
                environment="production",
            )

            assert manager.config_dir == config_dir
            assert manager.environment == "production"

    def test_initialization_environment_variable(self):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"ENVIRONMENT": "test"}):
            manager = ConfigurationManager()
            assert manager.environment == "test"

    def test_load_settings_default(self):
        """Test loading default settings."""
        manager = ConfigurationManager()
        settings = manager.load_settings()

        assert isinstance(settings, Settings)
        assert settings.app.name == "Pynomaly"
        assert settings.app.environment == "development"
        assert manager._settings is settings  # Should be cached

    def test_load_settings_cached(self):
        """Test that settings are cached on subsequent loads."""
        manager = ConfigurationManager()

        settings1 = manager.load_settings()
        settings2 = manager.load_settings()

        assert settings1 is settings2  # Same instance

    def test_load_settings_validation_error(self):
        """Test loading settings with validation error."""
        manager = ConfigurationManager()

        # Mock Settings to raise validation error
        with patch(
            "monorepo.infrastructure.config.config_manager.Settings"
        ) as mock_settings:
            from pydantic import ValidationError

            mock_settings.side_effect = ValidationError(errors=[], model=Settings)

            with pytest.raises(ConfigurationError, match="Invalid configuration"):
                manager.load_settings()

    def test_get_config_file_paths_default(self):
        """Test getting configuration file paths with defaults."""
        manager = ConfigurationManager(
            config_dir=Path("/test/config"),
            environment="development",
        )

        paths = manager._get_config_file_paths("app.yaml")

        expected = [
            Path("/test/config/app.development.yaml"),
            Path("/test/config/app.yaml"),
        ]
        assert paths == expected

    def test_get_config_file_paths_with_extension(self):
        """Test getting configuration file paths with existing extension."""
        manager = ConfigurationManager(
            config_dir=Path("/test/config"),
            environment="production",
        )

        paths = manager._get_config_file_paths("database.json")

        expected = [
            Path("/test/config/database.production.json"),
            Path("/test/config/database.json"),
        ]
        assert paths == expected

    def test_get_config_file_paths_no_extension(self):
        """Test getting configuration file paths without extension."""
        manager = ConfigurationManager(
            config_dir=Path("/test/config"),
            environment="test",
        )

        paths = manager._get_config_file_paths("monitoring")

        expected = [
            Path("/test/config/monitoring.test.yaml"),
            Path("/test/config/monitoring.yaml"),
        ]
        assert paths == expected

    def test_load_file_data_yaml(self):
        """Test loading data from YAML file."""
        manager = ConfigurationManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"key": "value", "number": 42}, f)
            temp_path = Path(f.name)

        try:
            data = manager._load_file_data(temp_path)
            assert data == {"key": "value", "number": 42}
        finally:
            temp_path.unlink()

    def test_load_file_data_json(self):
        """Test loading data from JSON file."""
        manager = ConfigurationManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value", "number": 42}, f)
            temp_path = Path(f.name)

        try:
            data = manager._load_file_data(temp_path)
            assert data == {"key": "value", "number": 42}
        finally:
            temp_path.unlink()

    def test_load_file_data_unsupported_format(self):
        """Test loading data from unsupported file format."""
        manager = ConfigurationManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            temp_path = Path(f.name)

        try:
            with pytest.raises(
                ConfigurationError, match="Unsupported configuration file format"
            ):
                manager._load_file_data(temp_path)
        finally:
            temp_path.unlink()

    def test_load_file_data_invalid_yaml(self):
        """Test loading invalid YAML file."""
        manager = ConfigurationManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)

        try:
            with pytest.raises(Exception):  # Should raise some parsing error
                manager._load_file_data(temp_path)
        finally:
            temp_path.unlink()

    def test_load_file_data_empty_yaml(self):
        """Test loading empty YAML file."""
        manager = ConfigurationManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            data = manager._load_file_data(temp_path)
            assert data == {}
        finally:
            temp_path.unlink()

    def test_load_config_file_success(self):
        """Test successful configuration file loading."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            name: str
            value: int

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "test.yaml"

            # Create config file
            with open(config_file, "w") as f:
                yaml.dump({"name": "test", "value": 42}, f)

            manager = ConfigurationManager(config_dir=config_dir)
            config = manager.load_config_file("test.yaml", TestConfig)

            assert config.name == "test"
            assert config.value == 42

    def test_load_config_file_environment_specific(self):
        """Test loading environment-specific configuration file."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            environment: str
            debug: bool

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create default config
            default_config = config_dir / "app.yaml"
            with open(default_config, "w") as f:
                yaml.dump({"environment": "default", "debug": True}, f)

            # Create production-specific config
            prod_config = config_dir / "app.production.yaml"
            with open(prod_config, "w") as f:
                yaml.dump({"environment": "production", "debug": False}, f)

            manager = ConfigurationManager(
                config_dir=config_dir,
                environment="production",
            )

            config = manager.load_config_file("app.yaml", TestConfig)

            # Should load production-specific config
            assert config.environment == "production"
            assert config.debug is False

    def test_load_config_file_fallback_to_default(self):
        """Test fallback to default config when environment-specific doesn't exist."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            name: str

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create only default config
            default_config = config_dir / "app.yaml"
            with open(default_config, "w") as f:
                yaml.dump({"name": "default"}, f)

            manager = ConfigurationManager(
                config_dir=config_dir,
                environment="staging",  # No staging-specific config
            )

            config = manager.load_config_file("app.yaml", TestConfig)

            # Should fall back to default
            assert config.name == "default"

    def test_load_config_file_caching(self):
        """Test configuration file caching."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            name: str

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "test.yaml"

            # Create config file
            with open(config_file, "w") as f:
                yaml.dump({"name": "test"}, f)

            manager = ConfigurationManager(config_dir=config_dir)

            # Load twice
            config1 = manager.load_config_file("test.yaml", TestConfig)
            config2 = manager.load_config_file("test.yaml", TestConfig)

            # Should be the same instance (cached)
            assert config1 is config2

    def test_load_config_file_required_missing(self):
        """Test loading required configuration file that doesn't exist."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            name: str

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            manager = ConfigurationManager(config_dir=config_dir)

            with pytest.raises(ConfigurationError, match="Required configuration file"):
                manager.load_config_file("nonexistent.yaml", TestConfig, required=True)

    def test_load_config_file_optional_missing(self):
        """Test loading optional configuration file that doesn't exist."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            name: str

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            manager = ConfigurationManager(config_dir=config_dir)

            config = manager.load_config_file(
                "nonexistent.yaml", TestConfig, required=False
            )
            assert config is None

    def test_load_config_file_validation_error(self):
        """Test configuration file with validation error."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            name: str
            value: int

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "test.yaml"

            # Create config file with invalid data
            with open(config_file, "w") as f:
                yaml.dump({"name": "test", "value": "not_an_int"}, f)

            manager = ConfigurationManager(config_dir=config_dir)

            with pytest.raises(
                ConfigurationError, match="Failed to load configuration"
            ):
                manager.load_config_file("test.yaml", TestConfig)

    def test_get_environment_config(self):
        """Test getting environment-specific configuration."""
        manager = ConfigurationManager(environment="production")

        env_config = manager.get_environment_config()

        assert env_config["environment"] == "production"
        assert env_config["debug"] is False
        assert env_config["testing"] is False
        assert env_config["development"] is False
        assert env_config["production"] is True

    def test_get_environment_config_development(self):
        """Test getting development environment configuration."""
        manager = ConfigurationManager(environment="development")

        env_config = manager.get_environment_config()

        assert env_config["environment"] == "development"
        assert env_config["debug"] is True
        assert env_config["testing"] is False
        assert env_config["development"] is True
        assert env_config["production"] is False

    def test_override_setting(self):
        """Test overriding a setting value."""
        manager = ConfigurationManager()

        # Load initial settings
        settings = manager.load_settings()
        original_name = settings.app.name

        # Override setting
        manager.override_setting("app.name", "NewName")

        # Check that setting was overridden
        assert settings.app.name == "NewName"
        assert settings.app.name != original_name

    def test_override_setting_nested(self):
        """Test overriding nested setting values."""
        manager = ConfigurationManager()

        # Load initial settings
        settings = manager.load_settings()
        original_env = settings.app.environment

        # Override nested setting
        manager.override_setting("app.environment", "test")

        # Check that nested setting was overridden
        assert settings.app.environment == "test"
        assert settings.app.environment != original_env

    def test_override_setting_clears_cache(self):
        """Test that overriding settings clears the cache."""
        from pydantic import BaseModel

        class TestConfig(BaseModel):
            name: str

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_file = config_dir / "test.yaml"

            # Create config file
            with open(config_file, "w") as f:
                yaml.dump({"name": "test"}, f)

            manager = ConfigurationManager(config_dir=config_dir)

            # Load and cache config
            config1 = manager.load_config_file("test.yaml", TestConfig)
            assert len(manager._config_cache) == 1

            # Override setting (should clear cache)
            manager.override_setting("app.name", "NewName")
            assert len(manager._config_cache) == 0

    def test_export_config_with_secrets(self):
        """Test exporting configuration with secrets included."""
        manager = ConfigurationManager()
        settings = manager.load_settings()

        config_dict = manager.export_config(include_secrets=True)

        assert isinstance(config_dict, dict)
        assert "secret_key" in config_dict
        assert config_dict["secret_key"] == settings.secret_key

    def test_export_config_without_secrets(self):
        """Test exporting configuration with secrets redacted."""
        manager = ConfigurationManager()

        config_dict = manager.export_config(include_secrets=False)

        assert isinstance(config_dict, dict)
        assert "secret_key" in config_dict
        assert config_dict["secret_key"] == "***REDACTED***"

    def test_validate_config_schema_valid(self):
        """Test validating valid configuration schema."""
        manager = ConfigurationManager()

        # Use current settings as valid config
        settings = manager.load_settings()
        config_data = settings.model_dump()

        errors = manager.validate_config_schema(config_data)
        assert errors == []

    def test_validate_config_schema_invalid(self):
        """Test validating invalid configuration schema."""
        manager = ConfigurationManager()

        # Create invalid config data
        invalid_config = {
            "app": {
                "name": 123,  # Should be string
                "version": "1.0.0",
                "environment": "invalid_env",  # Invalid environment
            }
        }

        errors = manager.validate_config_schema(invalid_config)
        assert len(errors) > 0
        assert any("name" in error for error in errors)


class TestConfigurationValidation:
    """Test configuration validation functionality."""

    def test_validate_settings_paths_creation(self):
        """Test that required paths are created during validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a settings object with custom paths
            settings = Settings()
            settings.storage_path = Path(temp_dir) / "storage"
            settings.model_storage_path = Path(temp_dir) / "models"
            settings.experiment_storage_path = Path(temp_dir) / "experiments"

            manager = ConfigurationManager()

            # Validate settings (should create directories)
            manager._validate_settings(settings)

            # Check that directories were created
            assert settings.storage_path.exists()
            assert settings.model_storage_path.exists()
            assert settings.experiment_storage_path.exists()

    def test_validate_settings_path_creation_failure(self):
        """Test validation when path creation fails."""
        settings = Settings()

        # Set path to an invalid location
        settings.storage_path = Path("/root/forbidden")  # Should fail on most systems

        manager = ConfigurationManager()

        # Should raise ConfigurationError
        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            manager._validate_settings(settings)

    def test_validate_settings_database_config(self):
        """Test validation of database configuration."""
        settings = Settings()
        settings.use_database_repositories = True
        settings.database_url = None  # Invalid: enabled but no URL

        manager = ConfigurationManager()

        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            manager._validate_settings(settings)

    def test_validate_settings_production_security(self):
        """Test validation of production security settings."""
        settings = Settings()
        settings.app.environment = "production"
        settings.secret_key = "change-me-in-production"  # Invalid for production
        settings.auth_enabled = False  # Invalid for production

        manager = ConfigurationManager()

        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            manager._validate_settings(settings)

    def test_validate_settings_production_security_valid(self):
        """Test validation of valid production security settings."""
        settings = Settings()
        settings.app.environment = "production"
        settings.secret_key = "secure-production-key"
        settings.auth_enabled = True

        manager = ConfigurationManager()

        # Should not raise an error
        manager._validate_settings(settings)


class TestDatabaseConfigManager:
    """Test database configuration manager."""

    def test_database_config_manager_initialization(self):
        """Test database config manager initialization."""
        settings = Settings()
        manager = DatabaseConfigManager(settings)

        assert manager.settings is settings

    def test_get_database_config_with_url(self):
        """Test getting database configuration with URL."""
        settings = Settings()
        settings.database_url = "postgresql://user:pass@localhost/db"
        settings.database_pool_size = 10
        settings.database_max_overflow = 20

        manager = DatabaseConfigManager(settings)
        config = manager.get_database_config()

        assert config["url"] == "postgresql://user:pass@localhost/db"
        assert config["pool_size"] == 10
        assert config["max_overflow"] == 20

    def test_get_database_config_without_url(self):
        """Test getting database configuration without URL."""
        settings = Settings()
        settings.database_url = None

        manager = DatabaseConfigManager(settings)
        config = manager.get_database_config()

        assert config == {}

    def test_get_alembic_config(self):
        """Test getting Alembic configuration."""
        settings = Settings()
        settings.database_url = "postgresql://user:pass@localhost/db"

        manager = DatabaseConfigManager(settings)
        config = manager.get_alembic_config()

        assert config["sqlalchemy.url"] == "postgresql://user:pass@localhost/db"
        assert config["script_location"] == "migrations"
        assert "file_template" in config


class TestCacheConfigManager:
    """Test cache configuration manager."""

    def test_cache_config_manager_initialization(self):
        """Test cache config manager initialization."""
        settings = Settings()
        manager = CacheConfigManager(settings)

        assert manager.settings is settings

    def test_get_cache_config_with_redis(self):
        """Test getting cache configuration with Redis."""
        settings = Settings()
        settings.cache_enabled = True
        settings.cache_ttl = 3600
        settings.redis_url = "redis://localhost:6379"

        manager = CacheConfigManager(settings)
        config = manager.get_cache_config()

        assert config["enabled"] is True
        assert config["default_ttl"] == 3600
        assert config["backend"] == "redis"
        assert config["redis_url"] == "redis://localhost:6379"

    def test_get_cache_config_without_redis(self):
        """Test getting cache configuration without Redis."""
        settings = Settings()
        settings.cache_enabled = True
        settings.cache_ttl = 1800
        settings.redis_url = None

        manager = CacheConfigManager(settings)
        config = manager.get_cache_config()

        assert config["enabled"] is True
        assert config["default_ttl"] == 1800
        assert config["backend"] == "memory"
        assert "redis_url" not in config

    def test_get_cache_config_disabled(self):
        """Test getting cache configuration when disabled."""
        settings = Settings()
        settings.cache_enabled = False

        manager = CacheConfigManager(settings)
        config = manager.get_cache_config()

        assert config["enabled"] is False


class TestMonitoringConfigManager:
    """Test monitoring configuration manager."""

    def test_monitoring_config_manager_initialization(self):
        """Test monitoring config manager initialization."""
        settings = Settings()
        manager = MonitoringConfigManager(settings)

        assert manager.settings is settings

    def test_get_prometheus_config(self):
        """Test getting Prometheus configuration."""
        settings = Settings()
        settings.monitoring.prometheus_enabled = True
        settings.monitoring.prometheus_port = 9090

        manager = MonitoringConfigManager(settings)
        config = manager.get_prometheus_config()

        assert config["enabled"] is True
        assert config["port"] == 9090
        assert config["metrics_path"] == "/metrics"

    def test_get_logging_config(self):
        """Test getting logging configuration."""
        settings = Settings()
        settings.monitoring.log_level = "INFO"
        settings.monitoring.log_format = "json"

        manager = MonitoringConfigManager(settings)
        config = manager.get_logging_config()

        assert config["level"] == "INFO"
        assert config["format"] == "json"
        assert "handlers" in config
        assert "console" in config["handlers"]
        assert "file" in config["handlers"]


class TestConfigurationIntegration:
    """Test configuration integration scenarios."""

    def test_create_config_manager_function(self):
        """Test create_config_manager function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = create_config_manager(
                config_dir=temp_dir,
                environment="test",
            )

            assert isinstance(manager, ConfigurationManager)
            assert manager.config_dir == Path(temp_dir)
            assert manager.environment == "test"

    def test_end_to_end_config_workflow(self):
        """Test complete end-to-end configuration workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create environment-specific config files
            app_config = config_dir / "app.test.yaml"
            with open(app_config, "w") as f:
                yaml.dump(
                    {
                        "name": "Test App",
                        "version": "1.0.0",
                        "environment": "test",
                        "debug": True,
                    },
                    f,
                )

            db_config = config_dir / "database.yaml"
            with open(db_config, "w") as f:
                yaml.dump(
                    {
                        "host": "localhost",
                        "port": 5432,
                        "database": "testdb",
                    },
                    f,
                )

            # Create configuration manager
            manager = ConfigurationManager(
                config_dir=config_dir,
                environment="test",
            )

            # Load main settings
            settings = manager.load_settings()
            assert isinstance(settings, Settings)

            # Load custom config files
            from pydantic import BaseModel

            class AppConfig(BaseModel):
                name: str
                version: str
                environment: str
                debug: bool

            class DatabaseConfig(BaseModel):
                host: str
                port: int
                database: str

            app_cfg = manager.load_config_file("app.yaml", AppConfig)
            db_cfg = manager.load_config_file("database.yaml", DatabaseConfig)

            # Verify loaded configurations
            assert app_cfg.name == "Test App"
            assert app_cfg.environment == "test"
            assert app_cfg.debug is True

            assert db_cfg.host == "localhost"
            assert db_cfg.port == 5432
            assert db_cfg.database == "testdb"

            # Test environment config
            env_config = manager.get_environment_config()
            assert env_config["environment"] == "test"
            assert env_config["testing"] is True

            # Test configuration export
            exported = manager.export_config(include_secrets=False)
            assert isinstance(exported, dict)
            assert "secret_key" in exported
            assert exported["secret_key"] == "***REDACTED***"

    def test_configuration_error_handling(self):
        """Test configuration error handling scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create invalid YAML file
            invalid_config = config_dir / "invalid.yaml"
            with open(invalid_config, "w") as f:
                f.write("invalid: yaml: [")

            manager = ConfigurationManager(config_dir=config_dir)

            from pydantic import BaseModel

            class TestConfig(BaseModel):
                name: str

            # Should handle invalid YAML gracefully
            with pytest.raises(ConfigurationError):
                manager.load_config_file("invalid.yaml", TestConfig)

    def test_configuration_caching_behavior(self):
        """Test configuration caching behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create config file
            config_file = config_dir / "test.yaml"
            with open(config_file, "w") as f:
                yaml.dump({"name": "original"}, f)

            manager = ConfigurationManager(config_dir=config_dir)

            from pydantic import BaseModel

            class TestConfig(BaseModel):
                name: str

            # Load config (should be cached)
            config1 = manager.load_config_file("test.yaml", TestConfig)
            assert config1.name == "original"

            # Modify file
            with open(config_file, "w") as f:
                yaml.dump({"name": "modified"}, f)

            # Load again (should return cached version)
            config2 = manager.load_config_file("test.yaml", TestConfig)
            assert config2.name == "original"  # Still cached
            assert config1 is config2

            # Clear cache and load again
            manager._config_cache.clear()
            config3 = manager.load_config_file("test.yaml", TestConfig)
            assert config3.name == "modified"  # Now updated

    def test_configuration_with_missing_directories(self):
        """Test configuration with missing directories."""
        # Use a non-existent directory
        non_existent_dir = Path("/non/existent/directory")

        manager = ConfigurationManager(config_dir=non_existent_dir)

        from pydantic import BaseModel

        class TestConfig(BaseModel):
            name: str

        # Should handle missing directory gracefully
        config = manager.load_config_file("test.yaml", TestConfig, required=False)
        assert config is None

    def test_configuration_thread_safety(self):
        """Test configuration manager thread safety."""
        import threading
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create config file
            config_file = config_dir / "test.yaml"
            with open(config_file, "w") as f:
                yaml.dump({"name": "test", "value": 42}, f)

            manager = ConfigurationManager(config_dir=config_dir)

            from pydantic import BaseModel

            class TestConfig(BaseModel):
                name: str
                value: int

            results = []
            errors = []

            def config_worker():
                try:
                    for i in range(10):
                        config = manager.load_config_file("test.yaml", TestConfig)
                        results.append(config.name)
                        time.sleep(0.001)  # Small delay to encourage concurrency
                except Exception as e:
                    errors.append(e)

            # Start multiple threads
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=config_worker)
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Check results
            assert len(errors) == 0
            assert len(results) == 50  # 5 threads * 10 iterations
            assert all(result == "test" for result in results)
