"""Unit tests for infrastructure settings components."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from anomaly_detection.infrastructure.config.settings import (
    DatabaseSettings, LoggingSettings, DetectionSettings, StreamingSettings,
    APISettings, MonitoringSettings, Settings,
    _load_config_file, _update_from_dict,
    get_settings, reload_settings, create_example_config
)


class TestDatabaseSettings:
    """Test cases for DatabaseSettings class."""
    
    def test_default_values(self):
        """Test default database settings values."""
        db_settings = DatabaseSettings()
        
        assert db_settings.host == "localhost"
        assert db_settings.port == 5432
        assert db_settings.database == "anomaly_detection"
        assert db_settings.username == "postgres"
        assert db_settings.password == ""
        assert db_settings.pool_size == 10
    
    def test_url_property_without_password(self):
        """Test database URL generation without password."""
        db_settings = DatabaseSettings(
            host="testhost",
            port=5433,
            database="testdb",
            username="testuser",
            password=""
        )
        
        expected_url = "postgresql://testuser@testhost:5433/testdb"
        assert db_settings.url == expected_url
    
    def test_url_property_with_password(self):
        """Test database URL generation with password."""
        db_settings = DatabaseSettings(
            host="testhost",
            port=5433,
            database="testdb",
            username="testuser",
            password=os.getenv("TEST_DB_PASSWORD", "test_password_placeholder")
        )
        
        expected_url = f"postgresql://testuser:{os.getenv('TEST_DB_PASSWORD', 'test_password_placeholder')}@testhost:5433/testdb"
        assert db_settings.url == expected_url
    
    @patch.dict(os.environ, {
        "DB_HOST": "envhost",
        "DB_PORT": "5434",
        "DB_DATABASE": "envdb",
        "DB_USERNAME": "envuser",
        "DB_PASSWORD": os.getenv("TEST_ENV_DB_PASSWORD", "test_env_password"),
        "DB_POOL_SIZE": "20"
    })
    def test_from_env(self):
        """Test loading database settings from environment variables."""
        db_settings = DatabaseSettings.from_env()
        
        assert db_settings.host == "envhost"
        assert db_settings.port == 5434
        assert db_settings.database == "envdb"
        assert db_settings.username == "envuser"
        assert db_settings.password == os.getenv("TEST_ENV_DB_PASSWORD", "test_env_password")
        assert db_settings.pool_size == 20
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test loading database settings from environment with defaults."""
        db_settings = DatabaseSettings.from_env()
        
        assert db_settings.host == "localhost"
        assert db_settings.port == 5432
        assert db_settings.database == "anomaly_detection"
        assert db_settings.username == "postgres"
        assert db_settings.password == ""
        assert db_settings.pool_size == 10


class TestLoggingSettings:
    """Test cases for LoggingSettings class."""
    
    def test_default_values(self):
        """Test default logging settings values."""
        log_settings = LoggingSettings()
        
        assert log_settings.level == "INFO"
        assert log_settings.format == "json"
        assert log_settings.output == "console"
        assert log_settings.file_path is None
        assert log_settings.max_file_size == "10MB"
        assert log_settings.backup_count == 5
        assert log_settings.file_enabled == True
        
        # Enhanced logging features
        assert log_settings.enable_structured_logging == True
        assert log_settings.enable_request_tracking == True
        assert log_settings.enable_performance_logging == True
        assert log_settings.enable_error_tracking == True
        assert log_settings.enable_metrics_logging == True
        
        # Performance thresholds
        assert log_settings.slow_query_threshold_ms == 1000.0
        assert log_settings.slow_detection_threshold_ms == 5000.0
        assert log_settings.slow_model_training_threshold_ms == 30000.0
        
        # Error handling
        assert log_settings.log_stack_traces == True
        assert log_settings.include_request_context == True
        assert log_settings.sanitize_sensitive_data == True
    
    @patch.dict(os.environ, {
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "text",
        "LOG_OUTPUT": "file",
        "LOG_FILE_PATH": "/var/log/app.log",
        "LOG_MAX_FILE_SIZE": "20MB",
        "LOG_BACKUP_COUNT": "10",
        "LOG_FILE_ENABLED": "false",
        "LOG_STRUCTURED": "false",
        "LOG_REQUEST_TRACKING": "false",
        "LOG_PERFORMANCE": "false",
        "LOG_ERROR_TRACKING": "false",
        "LOG_METRICS": "false",
        "LOG_SLOW_QUERY_THRESHOLD_MS": "2000",
        "LOG_SLOW_DETECTION_THRESHOLD_MS": "10000",
        "LOG_SLOW_TRAINING_THRESHOLD_MS": "60000",
        "LOG_STACK_TRACES": "false",
        "LOG_REQUEST_CONTEXT": "false",
        "LOG_SANITIZE_DATA": "false"
    })
    def test_from_env(self):
        """Test loading logging settings from environment variables."""
        log_settings = LoggingSettings.from_env()
        
        assert log_settings.level == "DEBUG"
        assert log_settings.format == "text"
        assert log_settings.output == "file"
        assert log_settings.file_path == "/var/log/app.log"
        assert log_settings.max_file_size == "20MB"
        assert log_settings.backup_count == 10
        assert log_settings.file_enabled == False
        
        # Enhanced features
        assert log_settings.enable_structured_logging == False
        assert log_settings.enable_request_tracking == False
        assert log_settings.enable_performance_logging == False
        assert log_settings.enable_error_tracking == False
        assert log_settings.enable_metrics_logging == False
        
        # Thresholds
        assert log_settings.slow_query_threshold_ms == 2000.0
        assert log_settings.slow_detection_threshold_ms == 10000.0
        assert log_settings.slow_model_training_threshold_ms == 60000.0
        
        # Error handling
        assert log_settings.log_stack_traces == False
        assert log_settings.include_request_context == False
        assert log_settings.sanitize_sensitive_data == False


class TestDetectionSettings:
    """Test cases for DetectionSettings class."""
    
    def test_default_values(self):
        """Test default detection settings values."""
        detection_settings = DetectionSettings()
        
        assert detection_settings.default_algorithm == "isolation_forest"
        assert detection_settings.default_contamination == 0.1
        assert detection_settings.max_samples == 100000
        assert detection_settings.timeout_seconds == 300
        assert detection_settings.isolation_forest_estimators == 100
        assert detection_settings.lof_neighbors == 20
        assert detection_settings.ocsvm_kernel == "rbf"
    
    @patch.dict(os.environ, {
        "DETECTION_DEFAULT_ALGORITHM": "one_class_svm",
        "DETECTION_DEFAULT_CONTAMINATION": "0.05",
        "DETECTION_MAX_SAMPLES": "50000",
        "DETECTION_TIMEOUT_SECONDS": "600",
        "DETECTION_IF_ESTIMATORS": "200",
        "DETECTION_LOF_NEIGHBORS": "30",
        "DETECTION_OCSVM_KERNEL": "linear"
    })
    def test_from_env(self):
        """Test loading detection settings from environment variables."""
        detection_settings = DetectionSettings.from_env()
        
        assert detection_settings.default_algorithm == "one_class_svm"
        assert detection_settings.default_contamination == 0.05
        assert detection_settings.max_samples == 50000
        assert detection_settings.timeout_seconds == 600
        assert detection_settings.isolation_forest_estimators == 200
        assert detection_settings.lof_neighbors == 30
        assert detection_settings.ocsvm_kernel == "linear"


class TestStreamingSettings:
    """Test cases for StreamingSettings class."""
    
    def test_default_values(self):
        """Test default streaming settings values."""
        streaming_settings = StreamingSettings()
        
        assert streaming_settings.buffer_size == 1000
        assert streaming_settings.update_frequency == 100
        assert streaming_settings.concept_drift_threshold == 0.05
        assert streaming_settings.max_buffer_size == 10000
    
    @patch.dict(os.environ, {
        "STREAMING_BUFFER_SIZE": "2000",
        "STREAMING_UPDATE_FREQUENCY": "200",
        "STREAMING_DRIFT_THRESHOLD": "0.1",
        "STREAMING_MAX_BUFFER_SIZE": "20000"
    })
    def test_from_env(self):
        """Test loading streaming settings from environment variables."""
        streaming_settings = StreamingSettings.from_env()
        
        assert streaming_settings.buffer_size == 2000
        assert streaming_settings.update_frequency == 200
        assert streaming_settings.concept_drift_threshold == 0.1
        assert streaming_settings.max_buffer_size == 20000


class TestAPISettings:
    """Test cases for APISettings class."""
    
    def test_default_values(self):
        """Test default API settings values."""
        api_settings = APISettings()
        
        assert api_settings.host == "0.0.0.0"
        assert api_settings.port == 8000
        assert api_settings.workers == 1
        assert api_settings.reload == False
        assert api_settings.debug == False
        assert api_settings.cors_origins == ["*"]
    
    @patch.dict(os.environ, {
        "API_HOST": "127.0.0.1",
        "API_PORT": "8080",
        "API_WORKERS": "4",
        "API_RELOAD": "true",
        "API_DEBUG": "true",
        "API_CORS_ORIGINS": "http://localhost:3000,http://example.com"
    })
    def test_from_env(self):
        """Test loading API settings from environment variables."""
        api_settings = APISettings.from_env()
        
        assert api_settings.host == "127.0.0.1"
        assert api_settings.port == 8080
        assert api_settings.workers == 4
        assert api_settings.reload == True
        assert api_settings.debug == True
        assert api_settings.cors_origins == ["http://localhost:3000", "http://example.com"]
    
    @patch.dict(os.environ, {"API_CORS_ORIGINS": "*"})
    def test_from_env_single_cors_origin(self):
        """Test loading API settings with single CORS origin."""
        api_settings = APISettings.from_env()
        assert api_settings.cors_origins == ["*"]


class TestMonitoringSettings:
    """Test cases for MonitoringSettings class."""
    
    def test_default_values(self):
        """Test default monitoring settings values."""
        monitoring_settings = MonitoringSettings()
        
        assert monitoring_settings.enable_metrics == True
        assert monitoring_settings.metrics_port == 9090
        assert monitoring_settings.enable_tracing == False
        assert monitoring_settings.jaeger_endpoint is None
    
    @patch.dict(os.environ, {
        "MONITORING_ENABLE_METRICS": "false",
        "MONITORING_METRICS_PORT": "9091",
        "MONITORING_ENABLE_TRACING": "true",
        "MONITORING_JAEGER_ENDPOINT": "http://jaeger:14268"
    })
    def test_from_env(self):
        """Test loading monitoring settings from environment variables."""
        monitoring_settings = MonitoringSettings.from_env()
        
        assert monitoring_settings.enable_metrics == False
        assert monitoring_settings.metrics_port == 9091
        assert monitoring_settings.enable_tracing == True
        assert monitoring_settings.jaeger_endpoint == "http://jaeger:14268"


class TestSettings:
    """Test cases for main Settings class."""
    
    def test_default_values(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.environment == "development"
        assert settings.debug == False
        assert settings.app_name == "Anomaly Detection Service"
        assert settings.version == "0.1.0"
        assert settings.secret_key == "dev-secret-key-change-in-production"
        assert settings.api_key is None
        
        # Component settings should be initialized
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.logging, LoggingSettings)
        assert isinstance(settings.detection, DetectionSettings)
        assert isinstance(settings.streaming, StreamingSettings)
        assert isinstance(settings.api, APISettings)
        assert isinstance(settings.monitoring, MonitoringSettings)
    
    @patch('anomaly_detection.infrastructure.config.settings._load_config_file')
    @patch.dict(os.environ, {
        "ENVIRONMENT": "production",
        "DEBUG": "true",
        "APP_NAME": "Test App",
        "VERSION": "1.0.0",
        "SECRET_KEY": "test-secret",
        "API_KEY": "test-api-key"
    })
    def test_load_from_env(self, mock_load_config):
        """Test loading settings from environment variables."""
        mock_load_config.return_value = None
        
        settings = Settings.load()
        
        assert settings.environment == "production"
        assert settings.debug == True
        assert settings.app_name == "Test App"
        assert settings.version == "1.0.0"
        assert settings.secret_key == "test-secret"
        assert settings.api_key == "test-api-key"
    
    @patch('anomaly_detection.infrastructure.config.settings._load_config_file')
    @patch('anomaly_detection.infrastructure.config.settings._update_from_dict')
    def test_load_with_config_file(self, mock_update_from_dict, mock_load_config):
        """Test loading settings with configuration file."""
        config_data = {"environment": "staging", "debug": True}
        mock_load_config.return_value = config_data
        mock_update_from_dict.return_value = Settings()
        
        settings = Settings.load()
        
        mock_load_config.assert_called_once()
        mock_update_from_dict.assert_called_once()
    
    @patch('anomaly_detection.infrastructure.config.settings._load_config_file')
    def test_load_no_config_file(self, mock_load_config):
        """Test loading settings when no configuration file exists."""
        mock_load_config.return_value = None
        
        settings = Settings.load()
        
        # Should still create settings with environment/defaults
        assert isinstance(settings, Settings)
        mock_load_config.assert_called_once()


class TestConfigFileLoading:
    """Test cases for configuration file loading functions."""
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"environment": "test"}')
    def test_load_config_file_success(self, mock_file, mock_exists):
        """Test successful configuration file loading."""
        mock_exists.return_value = True
        
        result = _load_config_file()
        
        assert result == {"environment": "test"}
        mock_file.assert_called_once()
    
    @patch('pathlib.Path.exists')
    def test_load_config_file_not_found(self, mock_exists):
        """Test configuration file loading when no file exists."""
        mock_exists.return_value = False
        
        result = _load_config_file()
        
        assert result is None
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    @patch('builtins.print')
    def test_load_config_file_invalid_json(self, mock_print, mock_file, mock_exists):
        """Test configuration file loading with invalid JSON."""
        mock_exists.return_value = True
        
        result = _load_config_file()
        
        assert result is None
        mock_print.assert_called()
        assert "Warning: Could not load config" in mock_print.call_args[0][0]
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', side_effect=IOError("File access error"))
    @patch('builtins.print')
    def test_load_config_file_io_error(self, mock_print, mock_file, mock_exists):
        """Test configuration file loading with I/O error."""
        mock_exists.return_value = True
        
        result = _load_config_file()
        
        assert result is None
        mock_print.assert_called()


class TestSettingsUpdate:
    """Test cases for settings update functionality."""
    
    def test_update_from_dict_top_level(self):
        """Test updating top-level settings from dictionary."""
        settings = Settings()
        data = {
            "environment": "testing",
            "debug": True,
            "app_name": "Test Application",
            "version": "2.0.0",
            "secret_key": "new-secret",
            "api_key": "new-api-key"
        }
        
        updated_settings = _update_from_dict(settings, data)
        
        assert updated_settings.environment == "testing"
        assert updated_settings.debug == True
        assert updated_settings.app_name == "Test Application"
        assert updated_settings.version == "2.0.0"
        assert updated_settings.secret_key == "new-secret"
        assert updated_settings.api_key == "new-api-key"
    
    def test_update_from_dict_component_settings(self):
        """Test updating component settings from dictionary."""
        settings = Settings()
        data = {
            "database": {
                "host": "new-host",
                "port": 5433,
                "database": "new-db"
            },
            "logging": {
                "level": "DEBUG",
                "format": "text"
            },
            "detection": {
                "default_algorithm": "lof",
                "max_samples": 50000
            }
        }
        
        updated_settings = _update_from_dict(settings, data)
        
        # Check database settings
        assert updated_settings.database.host == "new-host"
        assert updated_settings.database.port == 5433
        assert updated_settings.database.database == "new-db"
        # Should preserve unspecified values
        assert updated_settings.database.username == "postgres"
        
        # Check logging settings
        assert updated_settings.logging.level == "DEBUG"
        assert updated_settings.logging.format == "text"
        # Should preserve unspecified values
        assert updated_settings.logging.output == "console"
        
        # Check detection settings
        assert updated_settings.detection.default_algorithm == "lof"
        assert updated_settings.detection.max_samples == 50000
        # Should preserve unspecified values
        assert updated_settings.detection.default_contamination == 0.1
    
    def test_update_from_dict_invalid_component_field(self):
        """Test updating with invalid component field (should be ignored)."""
        settings = Settings()
        data = {
            "database": {
                "host": "new-host",
                "invalid_field": "should-be-ignored"
            }
        }
        
        updated_settings = _update_from_dict(settings, data)
        
        assert updated_settings.database.host == "new-host"
        # Invalid field should not cause error and should be ignored
        assert not hasattr(updated_settings.database, "invalid_field")
    
    def test_update_from_dict_unknown_component(self):
        """Test updating with unknown component (should be ignored)."""
        settings = Settings()
        data = {
            "unknown_component": {
                "field": "value"
            }
        }
        
        updated_settings = _update_from_dict(settings, data)
        
        # Should not raise error and settings should remain unchanged
        assert not hasattr(updated_settings, "unknown_component")


class TestGlobalSettingsFunctions:
    """Test cases for global settings functions."""
    
    def test_get_settings(self):
        """Test get_settings function returns settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)
    
    @patch('anomaly_detection.infrastructure.config.settings.Settings.load')
    def test_reload_settings(self, mock_load):
        """Test reload_settings function."""
        mock_settings = Settings()
        mock_load.return_value = mock_settings
        
        reloaded_settings = reload_settings()
        
        mock_load.assert_called_once()
        assert reloaded_settings is mock_settings


class TestExampleConfigCreation:
    """Test cases for example configuration creation."""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_create_example_config(self, mock_print, mock_file):
        """Test creating example configuration files."""
        create_example_config()
        
        # Should open two files
        assert mock_file.call_count == 2
        
        # Check that files are opened with correct names
        file_calls = [call[0][0] for call in mock_file.call_args_list]
        assert "config.example.json" in file_calls
        assert ".env.example" in file_calls
        
        # Check that success message is printed
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Created example configuration files" in call for call in print_calls)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_create_example_config_json_content(self, mock_json_dump, mock_file):
        """Test that example JSON config contains expected structure."""
        create_example_config()
        
        # Check that json.dump was called
        mock_json_dump.assert_called_once()
        
        # Get the data that was passed to json.dump
        dumped_data = mock_json_dump.call_args[0][0]
        
        # Verify structure
        assert "environment" in dumped_data
        assert "database" in dumped_data
        assert "logging" in dumped_data
        assert "detection" in dumped_data
        assert "streaming" in dumped_data
        assert "api" in dumped_data
        assert "monitoring" in dumped_data
        
        # Verify some specific values
        assert dumped_data["environment"] == "development"
        assert dumped_data["database"]["host"] == "localhost"
        assert dumped_data["detection"]["default_algorithm"] == "isolation_forest"


class TestSettingsEnvironmentVariableTypes:
    """Test cases for environment variable type conversion."""
    
    @patch.dict(os.environ, {"DB_PORT": "invalid_int"})
    def test_invalid_integer_environment_variable(self):
        """Test handling of invalid integer environment variable."""
        with pytest.raises(ValueError):
            DatabaseSettings.from_env()
    
    @patch.dict(os.environ, {"DETECTION_DEFAULT_CONTAMINATION": "invalid_float"})
    def test_invalid_float_environment_variable(self):
        """Test handling of invalid float environment variable."""
        with pytest.raises(ValueError):
            DetectionSettings.from_env()
    
    @patch.dict(os.environ, {
        "LOG_FILE_ENABLED": "yes",  # Should be "true" or "false"
        "API_DEBUG": "1"  # Should be "true" or "false"
    })
    def test_boolean_environment_variable_conversion(self):
        """Test boolean environment variable conversion."""
        log_settings = LoggingSettings.from_env()
        api_settings = APISettings.from_env()
        
        # "yes" should be treated as False (not "true")
        assert log_settings.file_enabled == False
        
        # "1" should be treated as False (not "true")
        assert api_settings.debug == False


class TestSettingsIntegration:
    """Integration test cases for settings functionality."""
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"environment": "config_test", "debug": true, "database": {"host": "config-host"}}')
    @patch.dict(os.environ, {
        "ENVIRONMENT": "env_test",
        "DEBUG": "false",
        "DB_HOST": "env-host"
    })
    def test_settings_precedence_config_over_env(self, mock_file, mock_exists):
        """Test that configuration file takes precedence over environment variables."""
        mock_exists.return_value = True
        
        settings = Settings.load()
        
        # Config file values should override environment
        assert settings.environment == "config_test"
        assert settings.debug == True
        assert settings.database.host == "config-host"
    
    def test_settings_singleton_behavior(self):
        """Test that get_settings returns consistent instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should return the same instance (global settings)
        assert settings1 is settings2
    
    @patch('anomaly_detection.infrastructure.config.settings.Settings.load')
    def test_reload_settings_updates_global(self, mock_load):
        """Test that reload_settings updates the global settings instance."""
        # Create a new settings instance
        new_settings = Settings()
        new_settings.environment = "reloaded"
        mock_load.return_value = new_settings
        
        # Reload settings
        reloaded = reload_settings()
        
        # Should return new instance and update global
        assert reloaded.environment == "reloaded"
        assert get_settings().environment == "reloaded"