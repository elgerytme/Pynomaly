"""
Branch Coverage Enhancement - Conditional Logic Testing
Comprehensive tests targeting conditional branches, error paths, and edge cases to improve branch coverage from 2.4% to 60%+.
"""

import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from monorepo.domain.exceptions import (
    ValidationError,
)
from monorepo.domain.value_objects import ContaminationRate


class TestDomainEntityBranches:
    """Test conditional branches in domain entities."""

    def test_dataset_validation_branches(self):
        """Test Dataset validation conditional branches."""
        from monorepo.domain.entities import Dataset

        # Test valid dataset creation
        valid_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "target": [0, 0, 1, 0, 1],
            }
        )

        dataset = Dataset(name="test_dataset", data=valid_data, target_column="target")

        assert dataset.name == "test_dataset"
        assert dataset.target_column == "target"
        assert dataset.n_samples == 5
        assert dataset.n_features == 3

        # Test empty dataset validation branch
        with pytest.raises(ValidationError, match="Dataset cannot be empty"):
            Dataset(name="empty_dataset", data=pd.DataFrame())

        # Test invalid target column branch
        with pytest.raises(
            ValidationError, match="Target column 'nonexistent' not found"
        ):
            Dataset(name="invalid_target", data=valid_data, target_column="nonexistent")

        # Test None data branch
        with pytest.raises(ValidationError, match="Data cannot be None"):
            Dataset(name="none_data", data=None)

        # Test dataset with no numeric features
        non_numeric_data = pd.DataFrame(
            {"text_col": ["a", "b", "c"], "category_col": ["x", "y", "z"]}
        )

        dataset_no_numeric = Dataset(name="no_numeric", data=non_numeric_data)

        assert len(dataset_no_numeric.get_numeric_features()) == 0

        # Test dataset with mixed types
        mixed_data = pd.DataFrame(
            {
                "numeric": [1, 2, 3],
                "string": ["a", "b", "c"],
                "boolean": [True, False, True],
                "datetime": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            }
        )

        mixed_dataset = Dataset(name="mixed_types", data=mixed_data)

        numeric_features = mixed_dataset.get_numeric_features()
        assert "numeric" in numeric_features
        assert "boolean" in numeric_features  # booleans are numeric
        assert "string" not in numeric_features
        assert "datetime" not in numeric_features

    def test_contamination_rate_validation_branches(self):
        """Test ContaminationRate validation branches."""
        from monorepo.domain.value_objects import ContaminationRate

        # Test valid contamination rates
        valid_rates = [0.01, 0.1, 0.3, 0.49]
        for rate in valid_rates:
            contamination = ContaminationRate(rate)
            assert contamination.value == rate
            assert not contamination.is_auto

        # Test auto contamination rate
        auto_contamination = ContaminationRate.auto()
        assert auto_contamination.is_auto
        assert auto_contamination.value == 0.1  # Default auto value

        # Test invalid contamination rates - too low
        with pytest.raises(
            ValidationError, match="Contamination rate must be between 0 and 0.5"
        ):
            ContaminationRate(0.0)

        with pytest.raises(
            ValidationError, match="Contamination rate must be between 0 and 0.5"
        ):
            ContaminationRate(-0.1)

        # Test invalid contamination rates - too high
        with pytest.raises(
            ValidationError, match="Contamination rate must be between 0 and 0.5"
        ):
            ContaminationRate(0.5)

        with pytest.raises(
            ValidationError, match="Contamination rate must be between 0 and 0.5"
        ):
            ContaminationRate(0.6)

        with pytest.raises(
            ValidationError, match="Contamination rate must be between 0 and 0.5"
        ):
            ContaminationRate(1.0)

    def test_anomaly_score_validation_branches(self):
        """Test AnomalyScore validation branches."""
        from monorepo.domain.value_objects import AnomalyScore

        # Test valid scores
        valid_scores = [0.0, 0.5, 1.0, 0.123, 0.999]
        for score in valid_scores:
            anomaly_score = AnomalyScore(value=score, method="test")
            assert anomaly_score.value == score
            assert anomaly_score.method == "test"

        # Test invalid scores - below range
        with pytest.raises(
            ValidationError, match="Anomaly score must be between 0 and 1"
        ):
            AnomalyScore(value=-0.1, method="test")

        # Test invalid scores - above range
        with pytest.raises(
            ValidationError, match="Anomaly score must be between 0 and 1"
        ):
            AnomalyScore(value=1.1, method="test")

        # Test edge cases
        edge_scores = [float("inf"), float("-inf"), float("nan")]
        for score in edge_scores:
            with pytest.raises(ValidationError):
                AnomalyScore(value=score, method="test")

        # Test confidence interval branch
        score_with_confidence = AnomalyScore(
            value=0.8, method="test", confidence_interval=(0.7, 0.9)
        )
        assert score_with_confidence.confidence_interval == (0.7, 0.9)

        # Test invalid confidence interval
        with pytest.raises(ValidationError, match="Invalid confidence interval"):
            AnomalyScore(
                value=0.8,
                method="test",
                confidence_interval=(0.9, 0.7),  # Lower > upper
            )

    def test_detector_state_branches(self):
        """Test Detector state management branches."""
        from monorepo.domain.entities import Detector

        # Test detector creation with minimal parameters
        detector = Detector(name="test_detector", algorithm_name="test_algorithm")

        assert detector.name == "test_detector"
        assert detector.algorithm_name == "test_algorithm"
        assert not detector.is_fitted
        assert detector.trained_at is None
        assert detector.contamination_rate.is_auto

        # Test detector with full parameters
        contamination = ContaminationRate(0.15)
        full_detector = Detector(
            name="full_detector",
            algorithm_name="full_algorithm",
            contamination_rate=contamination,
            parameters={"param1": "value1"},
            metadata={"meta1": "metavalue1"},
        )

        assert full_detector.contamination_rate.value == 0.15
        assert full_detector.parameters["param1"] == "value1"
        assert full_detector.metadata["meta1"] == "metavalue1"

        # Test parameter updates
        full_detector.update_parameters(param2="value2", param1="updated_value1")
        assert full_detector.parameters["param1"] == "updated_value1"
        assert full_detector.parameters["param2"] == "value2"

        # Test metadata updates
        full_detector.update_metadata("meta2", "metavalue2")
        assert full_detector.metadata["meta2"] == "metavalue2"

        # Test state transitions
        assert not full_detector.is_fitted

        # Simulate fitting
        full_detector.is_fitted = True
        full_detector.trained_at = datetime.now(UTC)

        assert full_detector.is_fitted
        assert full_detector.trained_at is not None

    def test_detection_result_branches(self):
        """Test DetectionResult conditional branches."""
        from monorepo.domain.entities import Anomaly, DetectionResult
        from monorepo.domain.value_objects import AnomalyScore

        # Create test data
        scores = [
            AnomalyScore(0.1, "test"),
            AnomalyScore(0.9, "test"),
            AnomalyScore(0.3, "test"),
        ]
        labels = [0, 1, 0]

        anomaly = Anomaly(
            score=AnomalyScore(0.9, "test"),
            data_point={"feature1": 5.0, "feature2": 10.0},
            detector_name="test_detector",
        )

        # Test with all parameters
        result = DetectionResult(
            detector_id="detector_123",
            dataset_id="dataset_456",
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
            execution_time_ms=100.5,
            metadata={"algorithm": "test"},
        )

        assert result.detector_id == "detector_123"
        assert result.dataset_id == "dataset_456"
        assert len(result.anomalies) == 1
        assert len(result.scores) == 3
        assert result.threshold == 0.5
        assert result.execution_time_ms == 100.5

        # Test with minimal parameters (optional fields)
        minimal_result = DetectionResult(
            detector_id="detector_789",
            dataset_id="dataset_101",
            anomalies=[],
            scores=scores,
            labels=labels,
            threshold=0.7,
        )

        assert minimal_result.execution_time_ms is None
        assert minimal_result.metadata == {}

        # Test result statistics
        assert result.n_anomalies == 1
        assert result.n_samples == 3
        assert minimal_result.n_anomalies == 0


class TestExceptionHandlingBranches:
    """Test exception handling branches across the codebase."""

    def test_detector_not_fitted_branches(self):
        """Test DetectorNotFittedError branches."""
        from monorepo.domain.exceptions import DetectorNotFittedError

        # Test with operation specified
        error_with_operation = DetectorNotFittedError(
            detector_name="test_detector", operation="predict"
        )
        assert "test_detector" in str(error_with_operation)
        assert "predict" in str(error_with_operation)
        assert "not fitted" in str(error_with_operation).lower()

        # Test without operation
        error_without_operation = DetectorNotFittedError(
            detector_name="another_detector"
        )
        assert "another_detector" in str(error_without_operation)
        assert "not fitted" in str(error_without_operation).lower()

    def test_fitting_error_branches(self):
        """Test FittingError branches."""
        from monorepo.domain.exceptions import FittingError

        # Test with all parameters
        full_error = FittingError(
            detector_name="failing_detector",
            reason="Algorithm convergence failed",
            dataset_name="problematic_dataset",
        )
        assert "failing_detector" in str(full_error)
        assert "Algorithm convergence failed" in str(full_error)
        assert "problematic_dataset" in str(full_error)

        # Test with minimal parameters
        minimal_error = FittingError(
            detector_name="minimal_detector", reason="Unknown error"
        )
        assert "minimal_detector" in str(minimal_error)
        assert "Unknown error" in str(minimal_error)

        # Test chaining with original exception
        try:
            raise ValueError("Original error")
        except ValueError as e:
            chained_error = FittingError(
                detector_name="chained_detector", reason="Wrapped error"
            )
            chained_error.__cause__ = e

            assert "chained_detector" in str(chained_error)
            assert "Wrapped error" in str(chained_error)

    def test_invalid_algorithm_error_branches(self):
        """Test InvalidAlgorithmError branches."""
        from monorepo.domain.exceptions import InvalidAlgorithmError

        # Test with available algorithms list
        available_algorithms = ["algo1", "algo2", "algo3"]
        error_with_list = InvalidAlgorithmError(
            algorithm_name="invalid_algo", available_algorithms=available_algorithms
        )
        assert "invalid_algo" in str(error_with_list)
        assert "algo1" in str(error_with_list)
        assert "algo2" in str(error_with_list)
        assert "algo3" in str(error_with_list)

        # Test without available algorithms list
        error_without_list = InvalidAlgorithmError(algorithm_name="another_invalid")
        assert "another_invalid" in str(error_without_list)

    def test_validation_error_branches(self):
        """Test ValidationError branches."""
        from monorepo.domain.exceptions import ValidationError

        # Test simple validation error
        simple_error = ValidationError("Simple validation failed")
        assert "Simple validation failed" in str(simple_error)

        # Test validation error with field information
        field_error = ValidationError(
            message="Invalid value", field="contamination_rate", value=1.5
        )
        assert "Invalid value" in str(field_error)
        assert "contamination_rate" in str(field_error)
        assert "1.5" in str(field_error)

    def test_configuration_error_branches(self):
        """Test ConfigurationError branches."""
        from monorepo.domain.exceptions import ConfigurationError

        # Test configuration error with context
        config_error = ConfigurationError(
            message="Invalid configuration",
            config_section="database",
            config_key="connection_string",
        )
        assert "Invalid configuration" in str(config_error)
        assert "database" in str(config_error)
        assert "connection_string" in str(config_error)

        # Test simple configuration error
        simple_config_error = ConfigurationError("Missing required setting")
        assert "Missing required setting" in str(simple_config_error)


class TestDataProcessingBranches:
    """Test conditional branches in data processing logic."""

    def test_data_validation_branches(self):
        """Test data validation conditional paths."""
        from monorepo.infrastructure.data.validators import DataValidator

        validator = DataValidator()

        # Test valid data
        valid_data = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [1.1, 2.2, 3.3, 4.4, 5.5]}
        )

        validation_result = validator.validate_dataset(valid_data)
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0

        # Test data with missing values
        data_with_na = pd.DataFrame(
            {"feature1": [1, 2, np.nan, 4, 5], "feature2": [1.1, np.nan, 3.3, 4.4, 5.5]}
        )

        na_validation = validator.validate_dataset(data_with_na)
        if not validator.allow_missing_values:
            assert not na_validation.is_valid
            assert any(
                "missing values" in error.lower() for error in na_validation.errors
            )

        # Test data with infinite values
        data_with_inf = pd.DataFrame(
            {
                "feature1": [1, 2, float("inf"), 4, 5],
                "feature2": [1.1, 2.2, 3.3, float("-inf"), 5.5],
            }
        )

        inf_validation = validator.validate_dataset(data_with_inf)
        if not validator.allow_infinite_values:
            assert not inf_validation.is_valid
            assert any("infinite" in error.lower() for error in inf_validation.errors)

        # Test data with duplicate rows
        data_with_duplicates = pd.DataFrame(
            {"feature1": [1, 2, 1, 4, 5], "feature2": [1.1, 2.2, 1.1, 4.4, 5.5]}
        )

        dup_validation = validator.validate_dataset(data_with_duplicates)
        if not validator.allow_duplicates:
            assert not dup_validation.is_valid
            assert any("duplicate" in error.lower() for error in dup_validation.errors)

    def test_feature_selection_branches(self):
        """Test feature selection conditional logic."""
        from monorepo.infrastructure.data.feature_selector import FeatureSelector

        # Create test data with various feature types
        test_data = pd.DataFrame(
            {
                "numeric_int": [1, 2, 3, 4, 5],
                "numeric_float": [1.1, 2.2, 3.3, 4.4, 5.5],
                "categorical": ["A", "B", "C", "A", "B"],
                "boolean": [True, False, True, False, True],
                "datetime": pd.date_range("2023-01-01", periods=5),
                "constant": [1, 1, 1, 1, 1],
                "high_cardinality": ["val1", "val2", "val3", "val4", "val5"],
            }
        )

        selector = FeatureSelector()

        # Test numeric feature selection
        numeric_features = selector.select_numeric_features(test_data)
        assert "numeric_int" in numeric_features
        assert "numeric_float" in numeric_features
        assert "boolean" in numeric_features  # Booleans are numeric
        assert "categorical" not in numeric_features
        assert "datetime" not in numeric_features

        # Test constant feature removal
        if selector.remove_constant_features:
            features_no_constant = selector.remove_constant_features(test_data)
            assert "constant" not in features_no_constant.columns
            assert "numeric_int" in features_no_constant.columns

        # Test high cardinality feature handling
        if hasattr(selector, "max_cardinality"):
            low_cardinality = selector.filter_by_cardinality(
                test_data, max_cardinality=3
            )
            if "high_cardinality" in test_data.columns:
                cardinality = test_data["high_cardinality"].nunique()
                if cardinality > 3:
                    assert "high_cardinality" not in low_cardinality.columns

    def test_data_preprocessing_branches(self):
        """Test data preprocessing conditional paths."""
        from monorepo.infrastructure.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()

        # Test data with various scales
        unscaled_data = pd.DataFrame(
            {
                "small_values": [0.001, 0.002, 0.003],
                "large_values": [1000, 2000, 3000],
                "normal_values": [1, 2, 3],
            }
        )

        # Test scaling branch
        if preprocessor.enable_scaling:
            scaled_data = preprocessor.scale_features(unscaled_data)

            # Check if scaling was applied
            for column in scaled_data.columns:
                col_mean = scaled_data[column].mean()
                col_std = scaled_data[column].std()

                if preprocessor.scaling_method == "standard":
                    assert abs(col_mean) < 1e-10  # Mean should be ~0
                    assert abs(col_std - 1) < 1e-10  # Std should be ~1
                elif preprocessor.scaling_method == "minmax":
                    col_min = scaled_data[column].min()
                    col_max = scaled_data[column].max()
                    assert abs(col_min) < 1e-10  # Min should be ~0
                    assert abs(col_max - 1) < 1e-10  # Max should be ~1

        # Test outlier handling branch
        data_with_outliers = pd.DataFrame(
            {"feature": [1, 2, 3, 4, 5, 100]}  # 100 is an outlier
        )

        if preprocessor.handle_outliers:
            cleaned_data = preprocessor.remove_outliers(data_with_outliers)

            if preprocessor.outlier_method == "iqr":
                Q1 = data_with_outliers["feature"].quantile(0.25)
                Q3 = data_with_outliers["feature"].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Check if outliers were removed
                assert all(cleaned_data["feature"] >= lower_bound)
                assert all(cleaned_data["feature"] <= upper_bound)


class TestAlgorithmSelectionBranches:
    """Test algorithm selection and configuration branches."""

    def test_algorithm_registry_branches(self):
        """Test algorithm registry conditional logic."""
        from monorepo.infrastructure.algorithms.registry import AlgorithmRegistry

        registry = AlgorithmRegistry()

        # Test algorithm registration
        test_algorithm_info = {
            "name": "TestAlgorithm",
            "category": "test",
            "description": "Test algorithm",
            "parameters": {"param1": "default_value"},
            "complexity": {"time": "O(n)", "space": "O(1)"},
        }

        registry.register_algorithm("test_algo", test_algorithm_info)

        # Test algorithm existence check
        assert registry.has_algorithm("test_algo")
        assert not registry.has_algorithm("nonexistent_algo")

        # Test algorithm retrieval
        retrieved_algo = registry.get_algorithm("test_algo")
        assert retrieved_algo["name"] == "TestAlgorithm"

        # Test algorithm retrieval with nonexistent algorithm
        with pytest.raises(KeyError):
            registry.get_algorithm("nonexistent_algo")

        # Test algorithms by category
        test_algorithms = registry.get_algorithms_by_category("test")
        assert len(test_algorithms) >= 1
        assert "test_algo" in [algo["name"] for algo in test_algorithms]

        empty_category = registry.get_algorithms_by_category("empty_category")
        assert len(empty_category) == 0

        # Test algorithm search
        search_results = registry.search_algorithms("Test")
        assert len(search_results) >= 1

        no_results = registry.search_algorithms("NonexistentKeyword")
        assert len(no_results) == 0

    def test_parameter_validation_branches(self):
        """Test parameter validation conditional paths."""
        from monorepo.infrastructure.algorithms.parameter_validator import (
            ParameterValidator,
        )

        validator = ParameterValidator()

        # Define parameter schema
        parameter_schema = {
            "n_estimators": {"type": "int", "min": 1, "max": 1000, "default": 100},
            "contamination": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.1},
            "random_state": {
                "type": "int",
                "min": 0,
                "default": None,
                "optional": True,
            },
        }

        # Test valid parameters
        valid_params = {"n_estimators": 50, "contamination": 0.15, "random_state": 42}

        validation_result = validator.validate_parameters(
            valid_params, parameter_schema
        )
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0

        # Test invalid parameter type
        invalid_type_params = {
            "n_estimators": "fifty",  # String instead of int
            "contamination": 0.15,
        }

        type_validation = validator.validate_parameters(
            invalid_type_params, parameter_schema
        )
        assert not type_validation.is_valid
        assert any("type" in error.lower() for error in type_validation.errors)

        # Test parameter out of range
        out_of_range_params = {
            "n_estimators": 1500,  # Above max
            "contamination": 0.6,  # Above max
        }

        range_validation = validator.validate_parameters(
            out_of_range_params, parameter_schema
        )
        assert not range_validation.is_valid
        assert any(
            "range" in error.lower() or "bound" in error.lower()
            for error in range_validation.errors
        )

        # Test missing required parameter
        missing_params = {
            "contamination": 0.1
            # Missing n_estimators (required)
        }

        missing_validation = validator.validate_parameters(
            missing_params, parameter_schema
        )
        if not parameter_schema["n_estimators"].get("optional", False):
            assert not missing_validation.is_valid
            assert any(
                "required" in error.lower() or "missing" in error.lower()
                for error in missing_validation.errors
            )

        # Test extra parameters
        extra_params = {
            "n_estimators": 100,
            "contamination": 0.1,
            "extra_param": "value",  # Not in schema
        }

        extra_validation = validator.validate_parameters(extra_params, parameter_schema)
        if not validator.allow_extra_parameters:
            assert not extra_validation.is_valid
            assert any(
                "unknown" in error.lower() or "extra" in error.lower()
                for error in extra_validation.errors
            )


class TestConfigurationBranches:
    """Test configuration loading and validation branches."""

    def test_config_loading_branches(self):
        """Test configuration loading conditional paths."""
        from monorepo.infrastructure.config.loader import ConfigLoader

        loader = ConfigLoader()

        # Test loading from dictionary
        config_dict = {
            "database": {"host": "localhost", "port": 5432, "name": "testdb"},
            "algorithms": {"default_contamination": 0.1, "enable_gpu": False},
        }

        config = loader.load_from_dict(config_dict)
        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.algorithms.default_contamination == 0.1

        # Test loading from JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            json_file_path = f.name

        try:
            json_config = loader.load_from_file(json_file_path)
            assert json_config.database.host == "localhost"
        finally:
            os.unlink(json_file_path)

        # Test loading from non-existent file
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("/nonexistent/config.json")

        # Test loading invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            invalid_json_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                loader.load_from_file(invalid_json_path)
        finally:
            os.unlink(invalid_json_path)

        # Test environment variable override
        with patch.dict(os.environ, {"PYNOMALY_DATABASE_HOST": "override_host"}):
            env_config = loader.load_with_env_override(config_dict)
            if loader.enable_env_override:
                assert env_config.database.host == "override_host"

    def test_config_validation_branches(self):
        """Test configuration validation conditional paths."""
        from monorepo.infrastructure.config.validator import ConfigValidator

        validator = ConfigValidator()

        # Test valid configuration
        valid_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "user",
                "password": "pass",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

        validation_result = validator.validate_config(valid_config)
        assert validation_result.is_valid

        # Test missing required section
        missing_section_config = {
            "logging": {"level": "INFO"}
            # Missing database section
        }

        missing_validation = validator.validate_config(missing_section_config)
        if "database" in validator.required_sections:
            assert not missing_validation.is_valid
            assert any(
                "database" in error.lower() for error in missing_validation.errors
            )

        # Test invalid port number
        invalid_port_config = {
            "database": {
                "host": "localhost",
                "port": 99999,  # Invalid port
                "username": "user",
                "password": "pass",
            }
        }

        port_validation = validator.validate_config(invalid_port_config)
        if validator.validate_port_range:
            assert not port_validation.is_valid
            assert any("port" in error.lower() for error in port_validation.errors)

        # Test invalid log level
        invalid_log_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "user",
                "password": "pass",
            },
            "logging": {"level": "INVALID_LEVEL"},
        }

        log_validation = validator.validate_config(invalid_log_config)
        if validator.validate_log_levels:
            assert not log_validation.is_valid
            assert any("level" in error.lower() for error in log_validation.errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
