"""Comprehensive SDK configuration and settings tests.

This module contains comprehensive tests for SDK configuration management,
settings validation, environment handling, and configuration persistence.
"""

import configparser
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml


class TestSDKConfigurationManagement:
    """Test SDK configuration management capabilities."""

    @pytest.fixture
    def mock_config_sdk(self):
        """Create mock SDK with configuration management."""

        class MockConfigSDK:
            def __init__(self):
                self.configurations = {}
                self.active_config = None
                self.config_history = []
                self.environment_overrides = {}
                self.default_config = {
                    "algorithms": {
                        "default": "IsolationForest",
                        "IsolationForest": {
                            "contamination": 0.1,
                            "n_estimators": 100,
                            "max_samples": "auto",
                            "random_state": None,
                        },
                        "LocalOutlierFactor": {
                            "contamination": 0.1,
                            "n_neighbors": 20,
                            "novelty": True,
                        },
                        "OneClassSVM": {"gamma": "scale", "nu": 0.5},
                    },
                    "data": {
                        "preprocessing": {
                            "normalize": True,
                            "handle_missing": "drop",
                            "feature_selection": False,
                        },
                        "validation": {
                            "check_finite": True,
                            "check_dtype": True,
                            "min_samples": 10,
                        },
                    },
                    "output": {
                        "format": "json",
                        "precision": 6,
                        "include_metadata": True,
                        "save_intermediate": False,
                    },
                    "performance": {
                        "parallel": True,
                        "n_jobs": -1,
                        "memory_limit": "1GB",
                        "timeout": 300,
                    },
                    "logging": {
                        "level": "INFO",
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        "file": None,
                        "console": True,
                    },
                }

            def create_configuration(
                self, config_name: str, config_data: dict[str, Any]
            ) -> str:
                """Create a new configuration."""
                config_id = str(uuid.uuid4())

                # Validate configuration structure
                validation_result = self._validate_configuration(config_data)
                if not validation_result["valid"]:
                    raise ValueError(
                        f"Invalid configuration: {validation_result['errors']}"
                    )

                # Merge with defaults
                merged_config = self._merge_with_defaults(config_data)

                self.configurations[config_id] = {
                    "id": config_id,
                    "name": config_name,
                    "config": merged_config,
                    "created_at": "2023-10-01T10:00:00Z",
                    "version": "1.0",
                }

                return config_id

            def load_configuration_from_file(
                self, file_path: str, config_name: str = None
            ) -> str:
                """Load configuration from file."""
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Configuration file not found: {file_path}"
                    )

                file_extension = Path(file_path).suffix.lower()

                try:
                    if file_extension == ".json":
                        with open(file_path) as f:
                            config_data = json.load(f)
                    elif file_extension in [".yaml", ".yml"]:
                        with open(file_path) as f:
                            config_data = yaml.safe_load(f)
                    elif file_extension in [".ini", ".cfg"]:
                        config_parser = configparser.ConfigParser()
                        config_parser.read(file_path)
                        config_data = self._convert_ini_to_dict(config_parser)
                    else:
                        raise ValueError(
                            f"Unsupported configuration file format: {file_extension}"
                        )

                    config_name = config_name or f"config_{Path(file_path).stem}"
                    return self.create_configuration(config_name, config_data)

                except Exception as e:
                    raise ValueError(
                        f"Failed to load configuration from {file_path}: {str(e)}"
                    )

            def save_configuration_to_file(
                self, config_id: str, file_path: str, format: str = None
            ) -> bool:
                """Save configuration to file."""
                if config_id not in self.configurations:
                    return False

                config_data = self.configurations[config_id]["config"]

                # Determine format from file extension if not specified
                if format is None:
                    format = Path(file_path).suffix.lower()[1:]  # Remove the dot

                try:
                    if format == "json":
                        with open(file_path, "w") as f:
                            json.dump(config_data, f, indent=2)
                    elif format in ["yaml", "yml"]:
                        with open(file_path, "w") as f:
                            yaml.dump(
                                config_data, f, default_flow_style=False, indent=2
                            )
                    elif format in ["ini", "cfg"]:
                        config_parser = self._convert_dict_to_ini(config_data)
                        with open(file_path, "w") as f:
                            config_parser.write(f)
                    else:
                        return False

                    return True

                except Exception:
                    return False

            def set_active_configuration(self, config_id: str) -> bool:
                """Set active configuration."""
                if config_id not in self.configurations:
                    return False

                # Save current config to history
                if self.active_config:
                    self.config_history.append(self.active_config)

                self.active_config = config_id
                return True

            def get_active_configuration(self) -> dict[str, Any] | None:
                """Get active configuration."""
                if (
                    not self.active_config
                    or self.active_config not in self.configurations
                ):
                    return self.default_config

                config = self.configurations[self.active_config]["config"].copy()

                # Apply environment overrides
                self._apply_environment_overrides(config)

                return config

            def update_configuration(
                self, config_id: str, updates: dict[str, Any]
            ) -> bool:
                """Update existing configuration."""
                if config_id not in self.configurations:
                    return False

                try:
                    current_config = self.configurations[config_id]["config"].copy()
                    updated_config = self._deep_merge(current_config, updates)

                    # Validate updated configuration
                    validation_result = self._validate_configuration(updated_config)
                    if not validation_result["valid"]:
                        return False

                    self.configurations[config_id]["config"] = updated_config
                    self.configurations[config_id]["updated_at"] = (
                        "2023-10-01T11:00:00Z"
                    )

                    return True

                except Exception:
                    return False

            def get_configuration_value(
                self, key_path: str, config_id: str = None
            ) -> Any:
                """Get specific configuration value using dot notation."""
                if config_id is None:
                    config = self.get_active_configuration()
                else:
                    if config_id not in self.configurations:
                        return None
                    config = self.configurations[config_id]["config"]

                keys = key_path.split(".")
                current = config

                try:
                    for key in keys:
                        current = current[key]
                    return current
                except (KeyError, TypeError):
                    return None

            def set_configuration_value(
                self, key_path: str, value: Any, config_id: str = None
            ) -> bool:
                """Set specific configuration value using dot notation."""
                if config_id is None:
                    config_id = self.active_config

                if not config_id or config_id not in self.configurations:
                    return False

                config = self.configurations[config_id]["config"]
                keys = key_path.split(".")

                try:
                    current = config
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]

                    current[keys[-1]] = value
                    return True

                except Exception:
                    return False

            def set_environment_override(self, env_var: str, config_path: str):
                """Set environment variable override for configuration path."""
                self.environment_overrides[env_var] = config_path

            def _apply_environment_overrides(self, config: dict[str, Any]):
                """Apply environment variable overrides to configuration."""
                for env_var, config_path in self.environment_overrides.items():
                    env_value = os.environ.get(env_var)
                    if env_value is not None:
                        # Convert string values to appropriate types
                        converted_value = self._convert_env_value(env_value)
                        self._set_nested_value(config, config_path, converted_value)

            def _convert_env_value(self, value: str) -> str | int | float | bool:
                """Convert environment variable string to appropriate type."""
                # Try boolean
                if value.lower() in ["true", "false"]:
                    return value.lower() == "true"

                # Try integer
                try:
                    return int(value)
                except ValueError:
                    pass

                # Try float
                try:
                    return float(value)
                except ValueError:
                    pass

                # Return as string
                return value

            def _set_nested_value(self, config: dict[str, Any], path: str, value: Any):
                """Set nested value in configuration using dot notation."""
                keys = path.split(".")
                current = config

                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

                current[keys[-1]] = value

            def _validate_configuration(self, config: dict[str, Any]) -> dict[str, Any]:
                """Validate configuration structure and values."""
                errors = []

                # Check required sections
                required_sections = ["algorithms", "data", "output"]
                for section in required_sections:
                    if section not in config:
                        errors.append(f"Missing required section: {section}")

                # Validate algorithm parameters
                if "algorithms" in config:
                    for algo_name, params in config["algorithms"].items():
                        if algo_name == "default":
                            continue

                        if isinstance(params, dict):
                            validation_errors = self._validate_algorithm_params(
                                algo_name, params
                            )
                            errors.extend(validation_errors)

                # Validate data processing parameters
                if "data" in config and "validation" in config["data"]:
                    validation_params = config["data"]["validation"]
                    if "min_samples" in validation_params:
                        if (
                            not isinstance(validation_params["min_samples"], int)
                            or validation_params["min_samples"] < 1
                        ):
                            errors.append(
                                "data.validation.min_samples must be a positive integer"
                            )

                return {"valid": len(errors) == 0, "errors": errors}

            def _validate_algorithm_params(
                self, algorithm: str, params: dict[str, Any]
            ) -> list[str]:
                """Validate algorithm-specific parameters."""
                errors = []

                if algorithm == "IsolationForest":
                    if "contamination" in params:
                        contamination = params["contamination"]
                        if not (0.0 < contamination < 0.5):
                            errors.append(
                                f"IsolationForest contamination must be between 0 and 0.5, got {contamination}"
                            )

                    if "n_estimators" in params:
                        n_estimators = params["n_estimators"]
                        if not isinstance(n_estimators, int) or n_estimators < 1:
                            errors.append(
                                f"IsolationForest n_estimators must be positive integer, got {n_estimators}"
                            )

                elif algorithm == "LocalOutlierFactor":
                    if "n_neighbors" in params:
                        n_neighbors = params["n_neighbors"]
                        if not isinstance(n_neighbors, int) or n_neighbors < 1:
                            errors.append(
                                f"LocalOutlierFactor n_neighbors must be positive integer, got {n_neighbors}"
                            )

                elif algorithm == "OneClassSVM":
                    if "nu" in params:
                        nu = params["nu"]
                        if not (0.0 < nu <= 1.0):
                            errors.append(
                                f"OneClassSVM nu must be between 0 and 1, got {nu}"
                            )

                return errors

            def _merge_with_defaults(self, config: dict[str, Any]) -> dict[str, Any]:
                """Merge configuration with defaults."""
                return self._deep_merge(self.default_config.copy(), config)

            def _deep_merge(
                self, base: dict[str, Any], overlay: dict[str, Any]
            ) -> dict[str, Any]:
                """Deep merge two dictionaries."""
                result = base.copy()

                for key, value in overlay.items():
                    if (
                        key in result
                        and isinstance(result[key], dict)
                        and isinstance(value, dict)
                    ):
                        result[key] = self._deep_merge(result[key], value)
                    else:
                        result[key] = value

                return result

            def _convert_ini_to_dict(
                self, config_parser: configparser.ConfigParser
            ) -> dict[str, Any]:
                """Convert ConfigParser to dictionary."""
                result = {}

                for section_name in config_parser.sections():
                    result[section_name] = {}
                    for key, value in config_parser.items(section_name):
                        # Try to convert to appropriate type
                        result[section_name][key] = self._convert_env_value(value)

                return result

            def _convert_dict_to_ini(
                self, config: dict[str, Any]
            ) -> configparser.ConfigParser:
                """Convert dictionary to ConfigParser."""
                config_parser = configparser.ConfigParser()

                for section_name, section_data in config.items():
                    if isinstance(section_data, dict):
                        config_parser.add_section(section_name)
                        for key, value in section_data.items():
                            config_parser.set(section_name, key, str(value))

                return config_parser

            def list_configurations(self) -> list[dict[str, Any]]:
                """List all configurations."""
                return [
                    {
                        "id": config_data["id"],
                        "name": config_data["name"],
                        "version": config_data["version"],
                        "created_at": config_data["created_at"],
                        "is_active": config_data["id"] == self.active_config,
                    }
                    for config_data in self.configurations.values()
                ]

            def delete_configuration(self, config_id: str) -> bool:
                """Delete a configuration."""
                if config_id not in self.configurations:
                    return False

                # Can't delete active configuration
                if config_id == self.active_config:
                    return False

                del self.configurations[config_id]
                return True

        return MockConfigSDK()

    def test_configuration_creation_and_management(self, mock_config_sdk):
        """Test configuration creation and basic management."""
        sdk = mock_config_sdk

        # Create custom configuration
        custom_config = {
            "algorithms": {
                "default": "LocalOutlierFactor",
                "LocalOutlierFactor": {
                    "contamination": 0.15,
                    "n_neighbors": 25,
                    "novelty": True,
                },
            },
            "data": {"preprocessing": {"normalize": False, "handle_missing": "fill"}},
            "output": {"format": "yaml", "precision": 4},
        }

        # Create configuration
        config_id = sdk.create_configuration("Custom Config", custom_config)
        assert config_id is not None

        # Set as active
        assert sdk.set_active_configuration(config_id)

        # Get active configuration
        active_config = sdk.get_active_configuration()
        assert active_config is not None

        # Verify custom values are present
        assert active_config["algorithms"]["default"] == "LocalOutlierFactor"
        assert (
            active_config["algorithms"]["LocalOutlierFactor"]["contamination"] == 0.15
        )
        assert active_config["data"]["preprocessing"]["normalize"] is False
        assert active_config["output"]["format"] == "yaml"

        # Verify defaults are merged
        assert "IsolationForest" in active_config["algorithms"]
        assert "logging" in active_config

        # List configurations
        configs = sdk.list_configurations()
        assert len(configs) == 1
        assert configs[0]["name"] == "Custom Config"
        assert configs[0]["is_active"] is True

    def test_configuration_file_operations(self, mock_config_sdk):
        """Test configuration file loading and saving."""
        sdk = mock_config_sdk

        # Create test configuration
        test_config = {
            "algorithms": {
                "default": "IsolationForest",
                "IsolationForest": {
                    "contamination": 0.2,
                    "n_estimators": 75,
                    "random_state": 42,
                },
            },
            "performance": {"n_jobs": 4, "timeout": 600},
        }

        # Test JSON format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_config, f, indent=2)
            json_path = f.name

        try:
            # Load from JSON
            config_id = sdk.load_configuration_from_file(json_path, "JSON Config")
            assert config_id is not None

            # Verify loaded configuration
            loaded_config = sdk.configurations[config_id]["config"]
            assert (
                loaded_config["algorithms"]["IsolationForest"]["contamination"] == 0.2
            )
            assert loaded_config["performance"]["n_jobs"] == 4

            # Save to different format
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                yaml_path = f.name

            save_success = sdk.save_configuration_to_file(config_id, yaml_path, "yaml")
            assert save_success

            # Verify saved file
            with open(yaml_path) as f:
                saved_config = yaml.safe_load(f)

            assert saved_config["algorithms"]["IsolationForest"]["contamination"] == 0.2
            assert saved_config["performance"]["n_jobs"] == 4

            # Clean up
            Path(yaml_path).unlink()

        finally:
            Path(json_path).unlink()

    def test_configuration_validation(self, mock_config_sdk):
        """Test configuration validation."""
        sdk = mock_config_sdk

        # Test valid configuration
        valid_config = {
            "algorithms": {
                "default": "IsolationForest",
                "IsolationForest": {"contamination": 0.1, "n_estimators": 100},
            },
            "data": {"validation": {"min_samples": 50}},
            "output": {"format": "json"},
        }

        config_id = sdk.create_configuration("Valid Config", valid_config)
        assert config_id is not None

        # Test invalid configuration - missing required section
        invalid_config_1 = {
            "algorithms": {"default": "IsolationForest"}
            # Missing 'data' and 'output' sections
        }

        with pytest.raises(ValueError, match="Invalid configuration"):
            sdk.create_configuration("Invalid Config 1", invalid_config_1)

        # Test invalid configuration - invalid algorithm parameters
        invalid_config_2 = {
            "algorithms": {
                "default": "IsolationForest",
                "IsolationForest": {
                    "contamination": 0.8,  # Too high
                    "n_estimators": -5,  # Negative
                },
            },
            "data": {"validation": {"min_samples": 10}},
            "output": {"format": "json"},
        }

        with pytest.raises(ValueError, match="Invalid configuration"):
            sdk.create_configuration("Invalid Config 2", invalid_config_2)

        # Test invalid configuration - invalid data validation
        invalid_config_3 = {
            "algorithms": {
                "default": "IsolationForest",
                "IsolationForest": {"contamination": 0.1},
            },
            "data": {
                "validation": {
                    "min_samples": -10  # Negative
                }
            },
            "output": {"format": "json"},
        }

        with pytest.raises(ValueError, match="Invalid configuration"):
            sdk.create_configuration("Invalid Config 3", invalid_config_3)

    def test_configuration_value_access(self, mock_config_sdk):
        """Test accessing and setting individual configuration values."""
        sdk = mock_config_sdk

        # Create test configuration
        test_config = {
            "algorithms": {
                "default": "IsolationForest",
                "IsolationForest": {"contamination": 0.1, "n_estimators": 100},
            },
            "data": {"preprocessing": {"normalize": True}},
            "output": {"format": "json"},
        }

        config_id = sdk.create_configuration("Test Config", test_config)
        sdk.set_active_configuration(config_id)

        # Test getting values with dot notation
        assert sdk.get_configuration_value("algorithms.default") == "IsolationForest"
        assert (
            sdk.get_configuration_value("algorithms.IsolationForest.contamination")
            == 0.1
        )
        assert sdk.get_configuration_value("data.preprocessing.normalize") is True
        assert sdk.get_configuration_value("output.format") == "json"

        # Test getting non-existent value
        assert sdk.get_configuration_value("nonexistent.key") is None

        # Test setting values
        assert sdk.set_configuration_value(
            "algorithms.IsolationForest.contamination", 0.15
        )
        assert (
            sdk.get_configuration_value("algorithms.IsolationForest.contamination")
            == 0.15
        )

        assert sdk.set_configuration_value("output.precision", 8)
        assert sdk.get_configuration_value("output.precision") == 8

        # Test setting nested value that doesn't exist
        assert sdk.set_configuration_value("new.nested.value", "test")
        assert sdk.get_configuration_value("new.nested.value") == "test"

    def test_environment_variable_overrides(self, mock_config_sdk):
        """Test environment variable configuration overrides."""
        sdk = mock_config_sdk

        # Set up environment overrides
        sdk.set_environment_override(
            "PYNOMALY_CONTAMINATION", "algorithms.IsolationForest.contamination"
        )
        sdk.set_environment_override(
            "PYNOMALY_N_ESTIMATORS", "algorithms.IsolationForest.n_estimators"
        )
        sdk.set_environment_override(
            "PYNOMALY_NORMALIZE", "data.preprocessing.normalize"
        )
        sdk.set_environment_override("PYNOMALY_OUTPUT_FORMAT", "output.format")

        # Create base configuration
        base_config = {
            "algorithms": {
                "default": "IsolationForest",
                "IsolationForest": {"contamination": 0.1, "n_estimators": 100},
            },
            "data": {"preprocessing": {"normalize": True}},
            "output": {"format": "json"},
        }

        config_id = sdk.create_configuration("Base Config", base_config)
        sdk.set_active_configuration(config_id)

        # Set environment variables
        test_env = {
            "PYNOMALY_CONTAMINATION": "0.2",
            "PYNOMALY_N_ESTIMATORS": "50",
            "PYNOMALY_NORMALIZE": "false",
            "PYNOMALY_OUTPUT_FORMAT": "yaml",
        }

        # Store original environment
        original_env = {}
        for key in test_env:
            if key in os.environ:
                original_env[key] = os.environ[key]

        try:
            # Set test environment variables
            for key, value in test_env.items():
                os.environ[key] = value

            # Get configuration with environment overrides
            active_config = sdk.get_active_configuration()

            # Verify environment overrides are applied
            assert (
                active_config["algorithms"]["IsolationForest"]["contamination"] == 0.2
            )
            assert active_config["algorithms"]["IsolationForest"]["n_estimators"] == 50
            assert active_config["data"]["preprocessing"]["normalize"] is False
            assert active_config["output"]["format"] == "yaml"

        finally:
            # Restore original environment
            for key in test_env:
                if key in original_env:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)

    def test_configuration_updates_and_history(self, mock_config_sdk):
        """Test configuration updates and history tracking."""
        sdk = mock_config_sdk

        # Create initial configuration
        initial_config = {
            "algorithms": {
                "default": "IsolationForest",
                "IsolationForest": {"contamination": 0.1},
            },
            "data": {"preprocessing": {"normalize": True}},
            "output": {"format": "json"},
        }

        config_id = sdk.create_configuration("Initial Config", initial_config)

        # Update configuration
        updates = {
            "algorithms": {
                "IsolationForest": {"contamination": 0.15, "n_estimators": 200}
            },
            "output": {"format": "yaml", "precision": 8},
        }

        update_success = sdk.update_configuration(config_id, updates)
        assert update_success

        # Verify updates were applied
        updated_config = sdk.configurations[config_id]["config"]
        assert updated_config["algorithms"]["IsolationForest"]["contamination"] == 0.15
        assert updated_config["algorithms"]["IsolationForest"]["n_estimators"] == 200
        assert updated_config["output"]["format"] == "yaml"
        assert updated_config["output"]["precision"] == 8

        # Verify original values that weren't updated are preserved
        assert updated_config["data"]["preprocessing"]["normalize"] is True
        assert updated_config["algorithms"]["default"] == "IsolationForest"

        # Test invalid update
        invalid_updates = {
            "algorithms": {
                "IsolationForest": {
                    "contamination": 0.8  # Invalid value
                }
            }
        }

        invalid_update_success = sdk.update_configuration(config_id, invalid_updates)
        assert not invalid_update_success

        # Verify configuration wasn't corrupted by invalid update
        final_config = sdk.configurations[config_id]["config"]
        assert (
            final_config["algorithms"]["IsolationForest"]["contamination"] == 0.15
        )  # Still the valid value

    def test_configuration_deletion_and_management(self, mock_config_sdk):
        """Test configuration deletion and management operations."""
        sdk = mock_config_sdk

        # Create multiple configurations
        config1_data = {
            "algorithms": {"default": "IsolationForest"},
            "data": {"preprocessing": {"normalize": True}},
            "output": {"format": "json"},
        }

        config2_data = {
            "algorithms": {"default": "LocalOutlierFactor"},
            "data": {"preprocessing": {"normalize": False}},
            "output": {"format": "yaml"},
        }

        config_id1 = sdk.create_configuration("Config 1", config1_data)
        config_id2 = sdk.create_configuration("Config 2", config2_data)

        # Verify both configurations exist
        configs = sdk.list_configurations()
        assert len(configs) == 2

        # Set one as active
        sdk.set_active_configuration(config_id1)

        # Try to delete active configuration (should fail)
        delete_success = sdk.delete_configuration(config_id1)
        assert not delete_success

        # Delete inactive configuration (should succeed)
        delete_success = sdk.delete_configuration(config_id2)
        assert delete_success

        # Verify deletion
        configs = sdk.list_configurations()
        assert len(configs) == 1
        assert configs[0]["id"] == config_id1

        # Try to delete non-existent configuration
        delete_success = sdk.delete_configuration("non_existent_id")
        assert not delete_success
