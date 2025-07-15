"""
Comprehensive test suite for Pynomaly SDK configuration management.

This module provides extensive testing coverage for SDK configuration,
including environment variables, file-based config, and validation.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from pynomaly.presentation.sdk.config import (
    ConfigurationError,
    PynomalyConfig,
    create_default_config,
    load_config_from_env,
    load_config_from_file,
    merge_configs,
    save_config_to_file,
    validate_config,
)


class TestPynomalyConfigInitialization:
    """Test PynomalyConfig initialization and basic functionality."""

    def test_config_default_initialization(self):
        """Test PynomalyConfig initialization with default values."""
        config = PynomalyConfig()

        assert config.base_url == "http://localhost:8000"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.verify_ssl is True
        assert config.debug is False

    def test_config_custom_initialization(self):
        """Test PynomalyConfig initialization with custom values."""
        config = PynomalyConfig(
            base_url="https://api.pynomaly.com",
            api_key="test-api-key",
            timeout=60.0,
            max_retries=5,
            retry_delay=2.0,
            verify_ssl=False,
            debug=True,
        )

        assert config.base_url == "https://api.pynomaly.com"
        assert config.api_key == "test-api-key"
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.verify_ssl is False
        assert config.debug is True

    def test_config_partial_initialization(self):
        """Test PynomalyConfig initialization with partial custom values."""
        config = PynomalyConfig(base_url="https://custom.api.com", api_key="custom-key")

        assert config.base_url == "https://custom.api.com"
        assert config.api_key == "custom-key"
        assert config.timeout == 30.0  # Default value
        assert config.max_retries == 3  # Default value
        assert config.verify_ssl is True  # Default value

    def test_config_validation_invalid_url(self):
        """Test PynomalyConfig validation with invalid URL."""
        with pytest.raises(ConfigurationError) as exc_info:
            PynomalyConfig(base_url="invalid-url")

        assert "Invalid base URL" in str(exc_info.value)

    def test_config_validation_negative_timeout(self):
        """Test PynomalyConfig validation with negative timeout."""
        with pytest.raises(ConfigurationError) as exc_info:
            PynomalyConfig(timeout=-1.0)

        assert "Timeout must be positive" in str(exc_info.value)

    def test_config_validation_negative_retries(self):
        """Test PynomalyConfig validation with negative max_retries."""
        with pytest.raises(ConfigurationError) as exc_info:
            PynomalyConfig(max_retries=-1)

        assert "Max retries must be non-negative" in str(exc_info.value)

    def test_config_validation_negative_retry_delay(self):
        """Test PynomalyConfig validation with negative retry_delay."""
        with pytest.raises(ConfigurationError) as exc_info:
            PynomalyConfig(retry_delay=-0.5)

        assert "Retry delay must be positive" in str(exc_info.value)


class TestConfigFileLoading:
    """Test configuration loading from files."""

    def test_load_config_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "base_url": "https://api.example.com",
            "api_key": "json-api-key",
            "timeout": 45.0,
            "max_retries": 4,
            "verify_ssl": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            json_file_path = f.name

        try:
            config = load_config_from_file(json_file_path)

            assert config.base_url == "https://api.example.com"
            assert config.api_key == "json-api-key"
            assert config.timeout == 45.0
            assert config.max_retries == 4
            assert config.verify_ssl is False
        finally:
            os.unlink(json_file_path)

    def test_load_config_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "base_url": "https://yaml.api.com",
            "api_key": "yaml-api-key",
            "timeout": 50.0,
            "max_retries": 6,
            "retry_delay": 1.5,
            "debug": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            yaml_file_path = f.name

        try:
            config = load_config_from_file(yaml_file_path)

            assert config.base_url == "https://yaml.api.com"
            assert config.api_key == "yaml-api-key"
            assert config.timeout == 50.0
            assert config.max_retries == 6
            assert config.retry_delay == 1.5
            assert config.debug is True
        finally:
            os.unlink(yaml_file_path)

    def test_load_config_from_ini_file(self):
        """Test loading configuration from INI file."""
        ini_content = """
[pynomaly]
base_url = https://ini.api.com
api_key = ini-api-key
timeout = 40.0
max_retries = 3
verify_ssl = false
debug = true
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write(ini_content)
            ini_file_path = f.name

        try:
            config = load_config_from_file(ini_file_path)

            assert config.base_url == "https://ini.api.com"
            assert config.api_key == "ini-api-key"
            assert config.timeout == 40.0
            assert config.max_retries == 3
            assert config.verify_ssl is False
            assert config.debug is True
        finally:
            os.unlink(ini_file_path)

    def test_load_config_nonexistent_file(self):
        """Test loading configuration from nonexistent file."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_config_from_file("/nonexistent/config.json")

        assert "Config file not found" in str(exc_info.value)

    def test_load_config_invalid_json(self):
        """Test loading configuration from invalid JSON file."""
        invalid_json = (
            '{"base_url": "https://api.com", "timeout": 30.0,'  # Missing closing brace
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(invalid_json)
            json_file_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_config_from_file(json_file_path)

            assert "Invalid JSON format" in str(exc_info.value)
        finally:
            os.unlink(json_file_path)

    def test_load_config_unsupported_format(self):
        """Test loading configuration from unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("base_url=https://api.com")
            txt_file_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_config_from_file(txt_file_path)

            assert "Unsupported config file format" in str(exc_info.value)
        finally:
            os.unlink(txt_file_path)


class TestEnvironmentVariableLoading:
    """Test configuration loading from environment variables."""

    def test_load_config_from_env_all_variables(self):
        """Test loading configuration from all environment variables."""
        env_vars = {
            "PYNOMALY_BASE_URL": "https://env.api.com",
            "PYNOMALY_API_KEY": "env-api-key",
            "PYNOMALY_TIMEOUT": "55.0",
            "PYNOMALY_MAX_RETRIES": "7",
            "PYNOMALY_RETRY_DELAY": "2.5",
            "PYNOMALY_VERIFY_SSL": "false",
            "PYNOMALY_DEBUG": "true",
        }

        with patch.dict(os.environ, env_vars):
            config = load_config_from_env()

            assert config.base_url == "https://env.api.com"
            assert config.api_key == "env-api-key"
            assert config.timeout == 55.0
            assert config.max_retries == 7
            assert config.retry_delay == 2.5
            assert config.verify_ssl is False
            assert config.debug is True

    def test_load_config_from_env_partial_variables(self):
        """Test loading configuration from partial environment variables."""
        env_vars = {
            "PYNOMALY_BASE_URL": "https://partial.env.com",
            "PYNOMALY_API_KEY": "partial-env-key",
        }

        with patch.dict(os.environ, env_vars):
            config = load_config_from_env()

            assert config.base_url == "https://partial.env.com"
            assert config.api_key == "partial-env-key"
            assert config.timeout == 30.0  # Default value
            assert config.max_retries == 3  # Default value

    def test_load_config_from_env_no_variables(self):
        """Test loading configuration from environment with no variables."""
        # Clear all Pynomaly-related environment variables
        env_vars_to_clear = [
            "PYNOMALY_BASE_URL",
            "PYNOMALY_API_KEY",
            "PYNOMALY_TIMEOUT",
            "PYNOMALY_MAX_RETRIES",
            "PYNOMALY_RETRY_DELAY",
            "PYNOMALY_VERIFY_SSL",
            "PYNOMALY_DEBUG",
        ]

        clear_env = dict.fromkeys(env_vars_to_clear)

        with patch.dict(os.environ, clear_env, clear=True):
            config = load_config_from_env()

            assert config.base_url == "http://localhost:8000"  # Default
            assert config.api_key is None  # Default
            assert config.timeout == 30.0  # Default

    def test_load_config_from_env_invalid_timeout(self):
        """Test loading configuration from environment with invalid timeout."""
        env_vars = {"PYNOMALY_TIMEOUT": "invalid_timeout"}

        with patch.dict(os.environ, env_vars):
            with pytest.raises(ConfigurationError) as exc_info:
                load_config_from_env()

            assert "Invalid timeout value" in str(exc_info.value)

    def test_load_config_from_env_invalid_max_retries(self):
        """Test loading configuration from environment with invalid max_retries."""
        env_vars = {"PYNOMALY_MAX_RETRIES": "not_a_number"}

        with patch.dict(os.environ, env_vars):
            with pytest.raises(ConfigurationError) as exc_info:
                load_config_from_env()

            assert "Invalid max_retries value" in str(exc_info.value)

    def test_load_config_from_env_invalid_boolean(self):
        """Test loading configuration from environment with invalid boolean."""
        env_vars = {"PYNOMALY_VERIFY_SSL": "maybe"}

        with patch.dict(os.environ, env_vars):
            with pytest.raises(ConfigurationError) as exc_info:
                load_config_from_env()

            assert "Invalid boolean value" in str(exc_info.value)

    def test_load_config_from_env_custom_prefix(self):
        """Test loading configuration from environment with custom prefix."""
        env_vars = {
            "CUSTOM_BASE_URL": "https://custom.prefix.com",
            "CUSTOM_API_KEY": "custom-prefix-key",
        }

        with patch.dict(os.environ, env_vars):
            config = load_config_from_env(prefix="CUSTOM_")

            assert config.base_url == "https://custom.prefix.com"
            assert config.api_key == "custom-prefix-key"


class TestDefaultConfigCreation:
    """Test default configuration creation."""

    def test_create_default_config(self):
        """Test creating default configuration."""
        config = create_default_config()

        assert config.base_url == "http://localhost:8000"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.verify_ssl is True
        assert config.debug is False

    def test_create_default_config_with_overrides(self):
        """Test creating default configuration with overrides."""
        overrides = {"base_url": "https://override.com", "timeout": 60.0, "debug": True}

        config = create_default_config(**overrides)

        assert config.base_url == "https://override.com"  # Overridden
        assert config.timeout == 60.0  # Overridden
        assert config.debug is True  # Overridden
        assert config.max_retries == 3  # Default
        assert config.verify_ssl is True  # Default


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config_dict = {
            "base_url": "https://valid.api.com",
            "api_key": "valid-key",
            "timeout": 30.0,
            "max_retries": 3,
            "retry_delay": 1.0,
            "verify_ssl": True,
            "debug": False,
        }

        # Should not raise any exception
        is_valid = validate_config(config_dict)
        assert is_valid is True

    def test_validate_config_missing_required_field(self):
        """Test configuration validation with missing required field."""
        config_dict = {
            "api_key": "valid-key",
            "timeout": 30.0,
            # Missing base_url
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config_dict)

        assert "Missing required field: base_url" in str(exc_info.value)

    def test_validate_config_invalid_field_type(self):
        """Test configuration validation with invalid field type."""
        config_dict = {
            "base_url": "https://valid.api.com",
            "timeout": "not_a_number",  # Should be float
            "max_retries": 3,
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config_dict)

        assert "Invalid type for field: timeout" in str(exc_info.value)

    def test_validate_config_invalid_url_format(self):
        """Test configuration validation with invalid URL format."""
        config_dict = {"base_url": "not-a-valid-url", "timeout": 30.0}

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config_dict)

        assert "Invalid URL format" in str(exc_info.value)

    def test_validate_config_unknown_field(self):
        """Test configuration validation with unknown field."""
        config_dict = {
            "base_url": "https://valid.api.com",
            "timeout": 30.0,
            "unknown_field": "unknown_value",
        }

        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config_dict)

        assert "Unknown field: unknown_field" in str(exc_info.value)


class TestConfigMerging:
    """Test configuration merging functionality."""

    def test_merge_configs_basic(self):
        """Test basic configuration merging."""
        base_config = {
            "base_url": "https://base.com",
            "timeout": 30.0,
            "max_retries": 3,
            "verify_ssl": True,
        }

        override_config = {
            "base_url": "https://override.com",
            "timeout": 60.0,
            "debug": True,
        }

        merged = merge_configs(base_config, override_config)

        assert merged["base_url"] == "https://override.com"  # Overridden
        assert merged["timeout"] == 60.0  # Overridden
        assert merged["max_retries"] == 3  # From base
        assert merged["verify_ssl"] is True  # From base
        assert merged["debug"] is True  # New field

    def test_merge_configs_multiple(self):
        """Test merging multiple configurations."""
        config1 = {"base_url": "https://config1.com", "timeout": 30.0, "max_retries": 3}

        config2 = {"base_url": "https://config2.com", "retry_delay": 1.5}

        config3 = {"timeout": 60.0, "debug": True}

        merged = merge_configs(config1, config2, config3)

        assert merged["base_url"] == "https://config2.com"  # From config2
        assert merged["timeout"] == 60.0  # From config3 (last wins)
        assert merged["max_retries"] == 3  # From config1
        assert merged["retry_delay"] == 1.5  # From config2
        assert merged["debug"] is True  # From config3

    def test_merge_configs_nested_dictionaries(self):
        """Test merging configurations with nested dictionaries."""
        base_config = {
            "base_url": "https://base.com",
            "advanced": {"connection_pool_size": 10, "keep_alive": True},
        }

        override_config = {
            "timeout": 60.0,
            "advanced": {"connection_pool_size": 20, "compression": True},
        }

        merged = merge_configs(base_config, override_config)

        assert merged["base_url"] == "https://base.com"
        assert merged["timeout"] == 60.0
        assert merged["advanced"]["connection_pool_size"] == 20  # Overridden
        assert merged["advanced"]["keep_alive"] is True  # From base
        assert merged["advanced"]["compression"] is True  # New field

    def test_merge_configs_empty_configs(self):
        """Test merging with empty configurations."""
        base_config = {"base_url": "https://base.com", "timeout": 30.0}

        empty_config = {}

        merged = merge_configs(base_config, empty_config)

        assert merged == base_config

    def test_merge_configs_none_values(self):
        """Test merging with None values."""
        base_config = {
            "base_url": "https://base.com",
            "api_key": "base-key",
            "timeout": 30.0,
        }

        override_config = {
            "api_key": None,  # Explicitly set to None
            "timeout": 60.0,
        }

        merged = merge_configs(base_config, override_config)

        assert merged["base_url"] == "https://base.com"
        assert merged["api_key"] is None  # Overridden to None
        assert merged["timeout"] == 60.0


class TestConfigFileSaving:
    """Test configuration saving to files."""

    def test_save_config_to_json_file(self):
        """Test saving configuration to JSON file."""
        config = PynomalyConfig(
            base_url="https://save.api.com",
            api_key="save-api-key",
            timeout=45.0,
            debug=True,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_file_path = f.name

        try:
            save_config_to_file(config, json_file_path)

            # Load and verify
            with open(json_file_path) as f:
                loaded_data = json.load(f)

            assert loaded_data["base_url"] == "https://save.api.com"
            assert loaded_data["api_key"] == "save-api-key"
            assert loaded_data["timeout"] == 45.0
            assert loaded_data["debug"] is True
        finally:
            os.unlink(json_file_path)

    def test_save_config_to_yaml_file(self):
        """Test saving configuration to YAML file."""
        config = PynomalyConfig(
            base_url="https://yaml.save.com", api_key="yaml-save-key", max_retries=5
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_file_path = f.name

        try:
            save_config_to_file(config, yaml_file_path)

            # Load and verify
            with open(yaml_file_path) as f:
                loaded_data = yaml.safe_load(f)

            assert loaded_data["base_url"] == "https://yaml.save.com"
            assert loaded_data["api_key"] == "yaml-save-key"
            assert loaded_data["max_retries"] == 5
        finally:
            os.unlink(yaml_file_path)

    def test_save_config_invalid_directory(self):
        """Test saving configuration to invalid directory."""
        config = PynomalyConfig()

        with pytest.raises(ConfigurationError) as exc_info:
            save_config_to_file(config, "/nonexistent/directory/config.json")

        assert "Unable to save config file" in str(exc_info.value)


class TestConfigIntegration:
    """Test configuration integration scenarios."""

    def test_config_precedence_env_over_file(self):
        """Test configuration precedence: environment variables over file."""
        # Create config file
        file_config = {
            "base_url": "https://file.api.com",
            "api_key": "file-api-key",
            "timeout": 30.0,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(file_config, f)
            json_file_path = f.name

        # Set environment variables
        env_vars = {
            "PYNOMALY_BASE_URL": "https://env.api.com",
            "PYNOMALY_TIMEOUT": "60.0",
        }

        try:
            with patch.dict(os.environ, env_vars):
                file_config_obj = load_config_from_file(json_file_path)
                env_config_obj = load_config_from_env()

                # Merge with env taking precedence
                merged_dict = merge_configs(
                    file_config_obj.to_dict(), env_config_obj.to_dict()
                )
                final_config = PynomalyConfig(**merged_dict)

                assert final_config.base_url == "https://env.api.com"  # From env
                assert final_config.api_key == "file-api-key"  # From file
                assert final_config.timeout == 60.0  # From env
        finally:
            os.unlink(json_file_path)

    def test_config_home_directory_file(self):
        """Test loading configuration from home directory."""
        home_config = {"base_url": "https://home.api.com", "api_key": "home-api-key"}

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path("/mock/home")

            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.return_value = True

                with patch(
                    "builtins.open", mock_open(read_data=json.dumps(home_config))
                ):
                    config = load_config_from_file("~/.pynomaly/config.json")

                    assert config.base_url == "https://home.api.com"
                    assert config.api_key == "home-api-key"

    def test_config_relative_path_resolution(self):
        """Test configuration file relative path resolution."""
        relative_config = {"base_url": "https://relative.api.com", "timeout": 25.0}

        # Create config in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "relative_config.json"

            with open(config_path, "w") as f:
                json.dump(relative_config, f)

            # Change to temp directory and load relative path
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = load_config_from_file("relative_config.json")

                assert config.base_url == "https://relative.api.com"
                assert config.timeout == 25.0
            finally:
                os.chdir(original_cwd)


# Test fixtures
@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        "base_url": "https://test.api.com",
        "api_key": "test-api-key",
        "timeout": 30.0,
        "max_retries": 3,
        "retry_delay": 1.0,
        "verify_ssl": True,
        "debug": False,
    }


@pytest.fixture
def sample_config_object():
    """Create a sample PynomalyConfig object."""
    return PynomalyConfig(
        base_url="https://fixture.api.com",
        api_key="fixture-api-key",
        timeout=45.0,
        max_retries=4,
        debug=True,
    )


@pytest.fixture
def temporary_config_file():
    """Create a temporary configuration file."""
    config_data = {
        "base_url": "https://temp.api.com",
        "api_key": "temp-api-key",
        "timeout": 40.0,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        temp_file_path = f.name

    yield temp_file_path

    # Cleanup
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)
