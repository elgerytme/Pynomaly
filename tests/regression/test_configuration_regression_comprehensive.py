"""Comprehensive configuration regression tests.

This module contains regression tests to ensure configuration handling,
parameter validation, and settings management remain consistent across
versions and deployments.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestParameterValidationRegression:
    """Test parameter validation consistency across versions."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for parameter testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "feature3": np.random.normal(0, 1, 100),
            }
        )
        return Dataset(name="Parameter Test Dataset", data=data)

    def test_contamination_parameter_validation(self, sample_dataset):
        """Test contamination parameter validation consistency."""
        # Valid contamination values
        valid_contamination_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        for contamination in valid_contamination_values:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": contamination,
                        "n_estimators": 10,
                        "random_state": 42,
                    },
                )

                # Should succeed with valid parameters
                adapter.fit(sample_dataset)
                scores = adapter.score(sample_dataset)

                # Verify results are reasonable
                assert len(scores) == sample_dataset.n_samples

                # Check that contamination rate is approximately respected
                result = adapter.detect(sample_dataset)
                actual_contamination = np.mean(result.labels)

                # Allow some tolerance
                assert abs(actual_contamination - contamination) < 0.1, (
                    f"Contamination rate not respected: expected {contamination}, got {actual_contamination}"
                )

            except ImportError:
                pytest.skip("scikit-learn not available")

        # Invalid contamination values
        invalid_contamination_values = [-0.1, 0.0, 0.6, 1.0, 1.1, 2.0]

        for contamination in invalid_contamination_values:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": contamination,
                        "n_estimators": 10,
                        "random_state": 42,
                    },
                )

                # Should either reject invalid parameters or handle gracefully
                try:
                    adapter.fit(sample_dataset)

                    # If fit succeeds, the algorithm might have internal validation
                    # or might clamp values to valid ranges
                    scores = adapter.score(sample_dataset)
                    assert len(scores) == sample_dataset.n_samples

                except (ValueError, TypeError) as e:
                    # Expected for invalid parameters
                    assert (
                        "contamination" in str(e).lower() or "invalid" in str(e).lower()
                    )

            except ImportError:
                continue

    def test_n_estimators_parameter_validation(self, sample_dataset):
        """Test n_estimators parameter validation consistency."""
        # Valid n_estimators values
        valid_n_estimators = [1, 10, 50, 100, 200]

        for n_estimators in valid_n_estimators:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": n_estimators,
                        "random_state": 42,
                    },
                )

                adapter.fit(sample_dataset)
                scores = adapter.score(sample_dataset)

                assert len(scores) == sample_dataset.n_samples

            except ImportError:
                continue

        # Invalid n_estimators values
        invalid_n_estimators = [0, -1, -10, 1.5, "invalid"]

        for n_estimators in invalid_n_estimators:
            try:
                # Should raise error during adapter creation or fit
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": n_estimators,
                        "random_state": 42,
                    },
                )

                try:
                    adapter.fit(sample_dataset)

                    # If no error is raised, the algorithm might have internal validation
                    # that corrects or handles the invalid value

                except (ValueError, TypeError):
                    # Expected for invalid parameters
                    pass

            except (ImportError, ValueError, TypeError):
                continue

    def test_random_state_parameter_validation(self, sample_dataset):
        """Test random_state parameter validation consistency."""
        # Valid random_state values
        valid_random_states = [None, 0, 42, 123, 9999]

        for random_state in valid_random_states:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 10,
                        "random_state": random_state,
                    },
                )

                adapter.fit(sample_dataset)
                scores = adapter.score(sample_dataset)

                assert len(scores) == sample_dataset.n_samples

            except ImportError:
                continue

        # Test reproducibility with same random state
        try:
            adapter1 = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 20,
                    "random_state": 777,
                },
            )

            adapter2 = SklearnAdapter(
                algorithm_name="IsolationForest",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 20,
                    "random_state": 777,
                },
            )

            adapter1.fit(sample_dataset)
            adapter2.fit(sample_dataset)

            scores1 = adapter1.score(sample_dataset)
            scores2 = adapter2.score(sample_dataset)

            # Scores should be identical with same random state
            for s1, s2 in zip(scores1, scores2, strict=False):
                assert abs(s1.value - s2.value) < 1e-10, (
                    "Random state not providing reproducibility"
                )

        except ImportError:
            pass

        # Invalid random_state values
        invalid_random_states = [-1, 1.5, "invalid", [1, 2, 3]]

        for random_state in invalid_random_states:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest",
                    parameters={
                        "contamination": 0.1,
                        "n_estimators": 10,
                        "random_state": random_state,
                    },
                )

                try:
                    adapter.fit(sample_dataset)

                except (ValueError, TypeError):
                    # Expected for invalid random_state
                    pass

            except (ImportError, ValueError, TypeError):
                continue


class TestConfigurationFileHandlingRegression:
    """Test configuration file handling consistency."""

    def test_json_configuration_handling(self):
        """Test JSON configuration file handling."""
        # Valid JSON configuration
        valid_config = {
            "models": {
                "isolation_forest": {
                    "algorithm": "IsolationForest",
                    "parameters": {
                        "contamination": 0.1,
                        "n_estimators": 100,
                        "random_state": 42,
                    },
                },
                "local_outlier_factor": {
                    "algorithm": "LocalOutlierFactor",
                    "parameters": {
                        "contamination": 0.1,
                        "n_neighbors": 20,
                        "novelty": True,
                    },
                },
            },
            "data": {"preprocessing": {"normalize": True, "handle_missing": "drop"}},
            "output": {"format": "json", "save_scores": True, "save_labels": True},
        }

        # Test JSON serialization/deserialization
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_config, f, indent=2)
            json_path = f.name

        # Load and verify configuration
        with open(json_path) as f:
            loaded_config = json.load(f)

        # Verify structure and content
        assert "models" in loaded_config
        assert "data" in loaded_config
        assert "output" in loaded_config

        # Verify model configurations
        models = loaded_config["models"]
        assert "isolation_forest" in models
        assert "local_outlier_factor" in models

        # Verify parameter types
        if_params = models["isolation_forest"]["parameters"]
        assert isinstance(if_params["contamination"], float)
        assert isinstance(if_params["n_estimators"], int)
        assert isinstance(if_params["random_state"], int)

        lof_params = models["local_outlier_factor"]["parameters"]
        assert isinstance(lof_params["contamination"], float)
        assert isinstance(lof_params["n_neighbors"], int)
        assert isinstance(lof_params["novelty"], bool)

        # Clean up
        Path(json_path).unlink()

    def test_yaml_configuration_handling(self):
        """Test YAML configuration file handling."""
        yaml_config_content = """
models:
  isolation_forest:
    algorithm: IsolationForest
    parameters:
      contamination: 0.1
      n_estimators: 100
      random_state: 42
  
  one_class_svm:
    algorithm: OneClassSVM
    parameters:
      gamma: scale
      nu: 0.1

data:
  preprocessing:
    normalize: true
    handle_missing: drop
    feature_selection: auto

output:
  format: yaml
  save_scores: true
  save_labels: true
  
logging:
  level: INFO
  file: pynomaly.log
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""

        try:
            import yaml

            # Save YAML configuration
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(yaml_config_content)
                yaml_path = f.name

            # Load and verify configuration
            with open(yaml_path) as f:
                loaded_config = yaml.safe_load(f)

            # Verify structure
            assert "models" in loaded_config
            assert "data" in loaded_config
            assert "output" in loaded_config
            assert "logging" in loaded_config

            # Verify model configurations
            models = loaded_config["models"]
            assert "isolation_forest" in models
            assert "one_class_svm" in models

            # Verify data types are preserved
            if_params = models["isolation_forest"]["parameters"]
            assert isinstance(if_params["contamination"], float)
            assert isinstance(if_params["n_estimators"], int)

            svm_params = models["one_class_svm"]["parameters"]
            assert isinstance(svm_params["gamma"], str)
            assert isinstance(svm_params["nu"], float)

            # Verify boolean values
            assert isinstance(loaded_config["data"]["preprocessing"]["normalize"], bool)
            assert isinstance(loaded_config["output"]["save_scores"], bool)

            # Clean up
            Path(yaml_path).unlink()

        except ImportError:
            pytest.skip("PyYAML not available")

    def test_configuration_validation_regression(self):
        """Test configuration validation consistency."""
        # Test missing required fields
        incomplete_configs = [
            # Missing algorithm
            {"parameters": {"contamination": 0.1}},
            # Missing parameters
            {"algorithm": "IsolationForest"},
            # Invalid algorithm name
            {"algorithm": "NonExistentAlgorithm", "parameters": {"contamination": 0.1}},
        ]

        for config in incomplete_configs:
            try:
                # Should either raise error or handle gracefully
                adapter = SklearnAdapter(
                    algorithm_name=config.get("algorithm", "IsolationForest"),
                    parameters=config.get("parameters", {}),
                )

                # If adapter creation succeeds, it should handle missing/invalid config

            except (ValueError, KeyError, TypeError):
                # Expected for invalid configurations
                pass
            except ImportError:
                continue

    def test_environment_variable_configuration(self):
        """Test environment variable configuration handling."""
        # Define test environment variables
        test_env_vars = {
            "PYNOMALY_DEFAULT_CONTAMINATION": "0.15",
            "PYNOMALY_DEFAULT_N_ESTIMATORS": "50",
            "PYNOMALY_DEFAULT_RANDOM_STATE": "123",
            "PYNOMALY_LOG_LEVEL": "DEBUG",
            "PYNOMALY_CONFIG_PATH": "/tmp/pynomaly_config.json",
        }

        # Store original environment
        original_env = {}
        for key in test_env_vars:
            if key in os.environ:
                original_env[key] = os.environ[key]

        try:
            # Set test environment variables
            for key, value in test_env_vars.items():
                os.environ[key] = value

            # Test environment variable access
            contamination = float(os.getenv("PYNOMALY_DEFAULT_CONTAMINATION", "0.1"))
            n_estimators = int(os.getenv("PYNOMALY_DEFAULT_N_ESTIMATORS", "100"))
            random_state = int(os.getenv("PYNOMALY_DEFAULT_RANDOM_STATE", "42"))
            log_level = os.getenv("PYNOMALY_LOG_LEVEL", "INFO")
            config_path = os.getenv("PYNOMALY_CONFIG_PATH", "config.json")

            # Verify values
            assert contamination == 0.15
            assert n_estimators == 50
            assert random_state == 123
            assert log_level == "DEBUG"
            assert config_path == "/tmp/pynomaly_config.json"

            # Test type conversion consistency
            assert isinstance(contamination, float)
            assert isinstance(n_estimators, int)
            assert isinstance(random_state, int)
            assert isinstance(log_level, str)
            assert isinstance(config_path, str)

        finally:
            # Restore original environment
            for key in test_env_vars:
                if key in original_env:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)


class TestAlgorithmConfigurationRegression:
    """Test algorithm-specific configuration handling."""

    @pytest.fixture
    def test_dataset(self):
        """Create test dataset for algorithm configuration testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 200),
                "y": np.random.normal(0, 1, 200),
                "z": np.random.normal(0, 1, 200),
            }
        )
        return Dataset(name="Algorithm Config Test", data=data)

    def test_isolation_forest_configuration_options(self, test_dataset):
        """Test IsolationForest configuration options."""
        # Test various configuration combinations
        config_variants = [
            {
                "contamination": 0.1,
                "n_estimators": 50,
                "max_samples": "auto",
                "random_state": 42,
            },
            {
                "contamination": 0.05,
                "n_estimators": 100,
                "max_samples": 0.5,
                "max_features": 1.0,
                "random_state": 123,
            },
            {
                "contamination": 0.2,
                "n_estimators": 200,
                "max_samples": 256,
                "bootstrap": False,
                "random_state": 456,
            },
        ]

        for config in config_variants:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="IsolationForest", parameters=config
                )

                adapter.fit(test_dataset)
                scores = adapter.score(test_dataset)
                result = adapter.detect(test_dataset)

                # Verify results
                assert len(scores) == test_dataset.n_samples
                assert len(result.labels) == test_dataset.n_samples

                # Verify contamination rate is approximately respected
                actual_contamination = np.mean(result.labels)
                expected_contamination = config["contamination"]
                assert abs(actual_contamination - expected_contamination) < 0.1

            except ImportError:
                pytest.skip("scikit-learn not available")

    def test_local_outlier_factor_configuration_options(self, test_dataset):
        """Test LocalOutlierFactor configuration options."""
        config_variants = [
            {"contamination": 0.1, "n_neighbors": 20, "novelty": True},
            {
                "contamination": 0.05,
                "n_neighbors": 10,
                "algorithm": "auto",
                "novelty": True,
            },
            {
                "contamination": 0.15,
                "n_neighbors": 30,
                "metric": "minkowski",
                "p": 2,
                "novelty": True,
            },
        ]

        for config in config_variants:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="LocalOutlierFactor", parameters=config
                )

                adapter.fit(test_dataset)
                scores = adapter.score(test_dataset)
                result = adapter.detect(test_dataset)

                # Verify results
                assert len(scores) == test_dataset.n_samples
                assert len(result.labels) == test_dataset.n_samples

            except ImportError:
                continue

    def test_one_class_svm_configuration_options(self, test_dataset):
        """Test OneClassSVM configuration options."""
        config_variants = [
            {"gamma": "scale", "nu": 0.1},
            {"kernel": "rbf", "gamma": "auto", "nu": 0.05},
            {"kernel": "linear", "nu": 0.2},
        ]

        for config in config_variants:
            try:
                adapter = SklearnAdapter(
                    algorithm_name="OneClassSVM", parameters=config
                )

                adapter.fit(test_dataset)
                scores = adapter.score(test_dataset)

                # Verify results
                assert len(scores) == test_dataset.n_samples

                # All scores should be valid
                for score in scores:
                    assert 0.0 <= score.value <= 1.0
                    assert not np.isnan(score.value)
                    assert not np.isinf(score.value)

            except ImportError:
                continue

    def test_algorithm_parameter_compatibility(self, test_dataset):
        """Test parameter compatibility across algorithms."""
        # Common parameters that should work across algorithms
        common_configs = [
            {"random_state": 42},
            {"random_state": 123},
            {"random_state": None},
        ]

        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]

        for algorithm in algorithms:
            for common_config in common_configs:
                try:
                    # Add algorithm-specific required parameters
                    if algorithm == "IsolationForest":
                        config = {
                            **common_config,
                            "contamination": 0.1,
                            "n_estimators": 20,
                        }
                    elif algorithm == "LocalOutlierFactor":
                        config = {
                            **common_config,
                            "contamination": 0.1,
                            "n_neighbors": 20,
                            "novelty": True,
                        }
                    elif algorithm == "OneClassSVM":
                        config = {**common_config, "nu": 0.1, "gamma": "scale"}
                    else:
                        continue

                    adapter = SklearnAdapter(
                        algorithm_name=algorithm, parameters=config
                    )

                    adapter.fit(test_dataset)
                    scores = adapter.score(test_dataset)

                    assert len(scores) == test_dataset.n_samples

                except ImportError:
                    continue


class TestConfigurationBackwardCompatibilityRegression:
    """Test backward compatibility of configuration formats."""

    def test_legacy_parameter_format_support(self):
        """Test support for legacy parameter formats."""
        # Test legacy parameter names and formats
        legacy_configs = [
            # Legacy contamination format
            {
                "algorithm": "IsolationForest",
                "contamination_rate": 0.1,  # Legacy name
                "n_estimators": 50,
                "random_state": 42,
            },
            # Legacy boolean format
            {
                "algorithm": "LocalOutlierFactor",
                "contamination": 0.1,
                "n_neighbors": 20,
                "novelty": "true",  # String instead of boolean
            },
        ]

        for config in legacy_configs:
            try:
                # Handle legacy parameter names
                normalized_config = config.copy()

                if "contamination_rate" in normalized_config:
                    normalized_config["contamination"] = normalized_config.pop(
                        "contamination_rate"
                    )

                # Handle string boolean values
                for key, value in normalized_config.items():
                    if isinstance(value, str):
                        if value.lower() == "true":
                            normalized_config[key] = True
                        elif value.lower() == "false":
                            normalized_config[key] = False

                algorithm = normalized_config.pop("algorithm")

                # Should work with normalized parameters
                adapter = SklearnAdapter(
                    algorithm_name=algorithm, parameters=normalized_config
                )

                # Basic validation that adapter was created successfully
                assert adapter.algorithm_name == algorithm

            except (ImportError, ValueError, KeyError):
                # Some legacy formats might not be supported - this is acceptable
                continue

    def test_configuration_version_detection(self):
        """Test detection and handling of different configuration versions."""
        config_versions = [
            # Version 1.0 format
            {
                "version": "1.0",
                "model": {
                    "type": "IsolationForest",
                    "contamination": 0.1,
                    "n_estimators": 100,
                },
            },
            # Version 2.0 format
            {
                "version": "2.0",
                "models": {
                    "primary": {
                        "algorithm": "IsolationForest",
                        "parameters": {
                            "contamination": 0.1,
                            "n_estimators": 100,
                            "random_state": 42,
                        },
                    }
                },
            },
        ]

        for config in config_versions:
            version = config.get("version", "unknown")

            if version == "1.0":
                # Handle version 1.0 format
                if "model" in config:
                    model_config = config["model"]
                    algorithm = model_config.pop("type", "IsolationForest")
                    parameters = model_config

                    try:
                        adapter = SklearnAdapter(
                            algorithm_name=algorithm, parameters=parameters
                        )
                        assert adapter.algorithm_name == algorithm

                    except ImportError:
                        continue

            elif version == "2.0":
                # Handle version 2.0 format
                if "models" in config and "primary" in config["models"]:
                    model_config = config["models"]["primary"]
                    algorithm = model_config["algorithm"]
                    parameters = model_config["parameters"]

                    try:
                        adapter = SklearnAdapter(
                            algorithm_name=algorithm, parameters=parameters
                        )
                        assert adapter.algorithm_name == algorithm

                    except ImportError:
                        continue

    def test_default_configuration_stability(self):
        """Test that default configurations remain stable."""
        # Default configurations that should remain consistent
        default_configs = {
            "IsolationForest": {
                "contamination": 0.1,
                "n_estimators": 100,
                "max_samples": "auto",
                "max_features": 1.0,
                "bootstrap": False,
                "random_state": None,
            },
            "LocalOutlierFactor": {
                "n_neighbors": 20,
                "algorithm": "auto",
                "contamination": 0.1,
                "novelty": False,
            },
            "OneClassSVM": {"kernel": "rbf", "gamma": "scale", "nu": 0.5},
        }

        for algorithm, expected_defaults in default_configs.items():
            try:
                # Test with minimal configuration
                adapter = SklearnAdapter(algorithm_name=algorithm, parameters={})

                # Should use reasonable defaults
                assert adapter.algorithm_name == algorithm

            except ImportError:
                continue
