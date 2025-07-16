"""Comprehensive tests for FeatureValidator domain service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.exceptions import DataTypeError, FeatureMismatchError
from pynomaly.domain.services.feature_validator import FeatureValidator


class TestFeatureValidatorCompatibility:
    """Test feature compatibility validation between datasets."""

    def test_validate_compatibility_identical_datasets(self):
        """Test validation with identical datasets."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        ref_dataset = Dataset(data=data, target_column="target")
        target_dataset = Dataset(data=data, target_column="target")

        # Should not raise any exception
        FeatureValidator.validate_compatibility(ref_dataset, target_dataset)

    def test_validate_compatibility_strict_mode_extra_features(self):
        """Test strict mode validation with extra features in target."""
        ref_data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3], "target": [0, 1, 0]}
        )

        target_data = pd.DataFrame(
            {
                "feature1": [4, 5, 6],
                "feature2": [0.4, 0.5, 0.6],
                "feature3": [1.0, 2.0, 3.0],  # Extra feature
                "target": [1, 0, 1],
            }
        )

        ref_dataset = Dataset(data=ref_data, target_column="target")
        target_dataset = Dataset(data=target_data, target_column="target")

        with pytest.raises(FeatureMismatchError) as exc_info:
            FeatureValidator.validate_compatibility(
                ref_dataset, target_dataset, strict=True
            )

        assert "Feature mismatch between datasets" in str(exc_info.value)
        assert hasattr(exc_info.value, "details")
        assert "extra_features" in exc_info.value.details
        assert "feature3" in exc_info.value.details["extra_features"]

    def test_validate_compatibility_strict_mode_missing_features(self):
        """Test strict mode validation with missing features in target."""
        ref_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [0.1, 0.2, 0.3],
                "feature3": [1.0, 2.0, 3.0],
                "target": [0, 1, 0],
            }
        )

        target_data = pd.DataFrame(
            {
                "feature1": [4, 5, 6],
                "feature2": [0.4, 0.5, 0.6],  # Missing feature3
                "target": [1, 0, 1],
            }
        )

        ref_dataset = Dataset(data=ref_data, target_column="target")
        target_dataset = Dataset(data=target_data, target_column="target")

        with pytest.raises(FeatureMismatchError) as exc_info:
            FeatureValidator.validate_compatibility(
                ref_dataset, target_dataset, strict=True
            )

        assert "Feature mismatch between datasets" in str(exc_info.value)
        assert "missing_features" in exc_info.value.details
        assert "feature3" in exc_info.value.details["missing_features"]

    def test_validate_compatibility_non_strict_mode_extra_features(self):
        """Test non-strict mode allows extra features."""
        ref_data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3], "target": [0, 1, 0]}
        )

        target_data = pd.DataFrame(
            {
                "feature1": [4, 5, 6],
                "feature2": [0.4, 0.5, 0.6],
                "feature3": [1.0, 2.0, 3.0],  # Extra feature (should be allowed)
                "target": [1, 0, 1],
            }
        )

        ref_dataset = Dataset(data=ref_data, target_column="target")
        target_dataset = Dataset(data=target_data, target_column="target")

        # Should not raise exception in non-strict mode
        FeatureValidator.validate_compatibility(
            ref_dataset, target_dataset, strict=False
        )

    def test_validate_compatibility_non_strict_mode_missing_features(self):
        """Test non-strict mode still fails on missing features."""
        ref_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [0.1, 0.2, 0.3],
                "feature3": [1.0, 2.0, 3.0],
                "target": [0, 1, 0],
            }
        )

        target_data = pd.DataFrame(
            {
                "feature1": [4, 5, 6],
                "feature2": [0.4, 0.5, 0.6],  # Missing feature3
                "target": [1, 0, 1],
            }
        )

        ref_dataset = Dataset(data=ref_data, target_column="target")
        target_dataset = Dataset(data=target_data, target_column="target")

        with pytest.raises(FeatureMismatchError) as exc_info:
            FeatureValidator.validate_compatibility(
                ref_dataset, target_dataset, strict=False
            )

        assert "missing required features" in str(exc_info.value)
        assert "missing_features" in exc_info.value.details

    def test_validate_compatibility_no_target_columns(self):
        """Test validation without target columns."""
        ref_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})

        target_data = pd.DataFrame({"feature1": [4, 5, 6], "feature2": [0.4, 0.5, 0.6]})

        ref_dataset = Dataset(data=ref_data)
        target_dataset = Dataset(data=target_data)

        # Should not raise exception
        FeatureValidator.validate_compatibility(ref_dataset, target_dataset)

    def test_validate_compatibility_different_target_columns(self):
        """Test validation with different target column names."""
        ref_data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3], "label": [0, 1, 0]}
        )

        target_data = pd.DataFrame(
            {"feature1": [4, 5, 6], "feature2": [0.4, 0.5, 0.6], "target": [1, 0, 1]}
        )

        ref_dataset = Dataset(data=ref_data, target_column="label")
        target_dataset = Dataset(data=target_data, target_column="target")

        # Should not raise exception (target columns excluded from feature comparison)
        FeatureValidator.validate_compatibility(ref_dataset, target_dataset)

    def test_validate_compatibility_incompatible_dtypes(self):
        """Test validation with incompatible data types."""
        ref_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3],  # int
                "feature2": [0.1, 0.2, 0.3],  # float
            }
        )

        target_data = pd.DataFrame(
            {
                "feature1": ["a", "b", "c"],  # string (incompatible)
                "feature2": [0.4, 0.5, 0.6],  # float (compatible)
            }
        )

        ref_dataset = Dataset(data=ref_data)
        target_dataset = Dataset(data=target_data)

        with pytest.raises(DataTypeError) as exc_info:
            FeatureValidator.validate_compatibility(ref_dataset, target_dataset)

        assert "Incompatible data types for feature 'feature1'" in str(exc_info.value)
        assert exc_info.value.details["feature"] == "feature1"

    def test_validate_compatibility_compatible_numeric_dtypes(self):
        """Test validation with compatible numeric data types."""
        ref_data = pd.DataFrame(
            {
                "feature1": np.array([1, 2, 3], dtype=np.int32),
                "feature2": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            }
        )

        target_data = pd.DataFrame(
            {
                "feature1": np.array(
                    [4, 5, 6], dtype=np.int64
                ),  # Different int type (compatible)
                "feature2": np.array(
                    [0.4, 0.5, 0.6], dtype=np.float64
                ),  # Different float type (compatible)
            }
        )

        ref_dataset = Dataset(data=ref_data)
        target_dataset = Dataset(data=target_data)

        # Should not raise exception (numeric types are compatible)
        FeatureValidator.validate_compatibility(ref_dataset, target_dataset)

    def test_validate_compatibility_empty_datasets(self):
        """Test validation with empty datasets."""
        ref_data = pd.DataFrame()
        target_data = pd.DataFrame()

        ref_dataset = Dataset(data=ref_data)
        target_dataset = Dataset(data=target_data)

        # Should not raise exception
        FeatureValidator.validate_compatibility(ref_dataset, target_dataset)

    def test_validate_compatibility_none_feature_names(self):
        """Test validation when datasets have None feature names."""
        # Create mock datasets with None feature_names
        ref_dataset = MagicMock(spec=Dataset)
        ref_dataset.feature_names = None
        ref_dataset.target_column = None

        target_dataset = MagicMock(spec=Dataset)
        target_dataset.feature_names = None
        target_dataset.target_column = None

        # Should not raise exception
        FeatureValidator.validate_compatibility(ref_dataset, target_dataset)


class TestFeatureValidatorDataTypes:
    """Test data type compatibility validation."""

    def test_are_dtypes_compatible_numeric_types(self):
        """Test numeric dtype compatibility."""
        # All numeric types should be compatible with each other
        numeric_combinations = [
            (np.dtype("int16"), np.dtype("int32")),
            (np.dtype("int32"), np.dtype("int64")),
            (np.dtype("float16"), np.dtype("float32")),
            (np.dtype("float32"), np.dtype("float64")),
            (np.dtype("int32"), np.dtype("float64")),
        ]

        for dtype1, dtype2 in numeric_combinations:
            assert FeatureValidator._are_dtypes_compatible(dtype1, dtype2)
            assert FeatureValidator._are_dtypes_compatible(dtype2, dtype1)

    def test_are_dtypes_compatible_identical_types(self):
        """Test identical dtype compatibility."""
        dtypes = [
            np.dtype("int32"),
            np.dtype("float64"),
            np.dtype("object"),
            np.dtype("bool"),
        ]

        for dtype in dtypes:
            assert FeatureValidator._are_dtypes_compatible(dtype, dtype)

    def test_are_dtypes_compatible_incompatible_types(self):
        """Test incompatible dtype combinations."""
        incompatible_combinations = [
            (np.dtype("int32"), np.dtype("object")),
            (np.dtype("float64"), np.dtype("bool")),
            (np.dtype("object"), np.dtype("bool")),
        ]

        for dtype1, dtype2 in incompatible_combinations:
            assert not FeatureValidator._are_dtypes_compatible(dtype1, dtype2)
            assert not FeatureValidator._are_dtypes_compatible(dtype2, dtype1)

    def test_validate_dtypes_private_method(self):
        """Test private _validate_dtypes method."""
        ref_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})

        compatible_data = pd.DataFrame(
            {"feature1": [4, 5, 6], "feature2": [0.4, 0.5, 0.6]}
        )

        incompatible_data = pd.DataFrame(
            {
                "feature1": ["a", "b", "c"],  # Incompatible type
                "feature2": [0.4, 0.5, 0.6],
            }
        )

        ref_dataset = Dataset(data=ref_data)
        compatible_dataset = Dataset(data=compatible_data)
        incompatible_dataset = Dataset(data=incompatible_data)

        features = {"feature1", "feature2"}

        # Should not raise for compatible types
        FeatureValidator._validate_dtypes(ref_dataset, compatible_dataset, features)

        # Should raise for incompatible types
        with pytest.raises(DataTypeError):
            FeatureValidator._validate_dtypes(
                ref_dataset, incompatible_dataset, features
            )


class TestFeatureValidatorNumericValidation:
    """Test numeric feature validation."""

    def test_validate_numeric_features_all_numeric(self):
        """Test validation with all numeric features."""
        data = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        numeric_features = FeatureValidator.validate_numeric_features(
            dataset, features=["int_col", "float_col"]
        )

        assert numeric_features == ["int_col", "float_col"]

    def test_validate_numeric_features_none_features(self):
        """Test validation with None features (should get all numeric)."""
        data = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "string_col": ["a", "b", "c"],
                "target": [0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        # Mock get_numeric_features to return expected numeric columns
        with patch.object(
            dataset, "get_numeric_features", return_value=["int_col", "float_col"]
        ):
            numeric_features = FeatureValidator.validate_numeric_features(dataset)
            assert numeric_features == ["int_col", "float_col"]

    def test_validate_numeric_features_non_numeric_feature(self):
        """Test validation with non-numeric feature."""
        data = pd.DataFrame(
            {
                "numeric_col": [1, 2, 3],
                "string_col": ["a", "b", "c"],
                "target": [0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        with pytest.raises(DataTypeError) as exc_info:
            FeatureValidator.validate_numeric_features(
                dataset, features=["numeric_col", "string_col"]
            )

        assert "Feature 'string_col' is not numeric" in str(exc_info.value)
        assert exc_info.value.details["feature"] == "string_col"

    def test_validate_numeric_features_missing_feature(self):
        """Test validation with missing feature."""
        data = pd.DataFrame({"existing_col": [1, 2, 3], "target": [0, 1, 0]})

        dataset = Dataset(data=data, target_column="target")

        with pytest.raises(FeatureMismatchError) as exc_info:
            FeatureValidator.validate_numeric_features(
                dataset, features=["existing_col", "missing_col"]
            )

        assert "Feature 'missing_col' not found in dataset" in str(exc_info.value)
        assert exc_info.value.details["missing_features"] == ["missing_col"]

    def test_validate_numeric_features_empty_list(self):
        """Test validation with empty feature list."""
        data = pd.DataFrame({"numeric_col": [1, 2, 3], "target": [0, 1, 0]})

        dataset = Dataset(data=data, target_column="target")

        numeric_features = FeatureValidator.validate_numeric_features(
            dataset, features=[]
        )
        assert numeric_features == []

    def test_validate_numeric_features_boolean_column(self):
        """Test validation with boolean column."""
        data = pd.DataFrame({"bool_col": [True, False, True], "target": [0, 1, 0]})

        dataset = Dataset(data=data, target_column="target")

        # Boolean columns should be considered numeric by pandas
        numeric_features = FeatureValidator.validate_numeric_features(
            dataset, features=["bool_col"]
        )
        assert numeric_features == ["bool_col"]

    def test_validate_numeric_features_mixed_types(self):
        """Test validation with mixed numeric and non-numeric features."""
        data = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "string_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "target": [0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        # Test with valid numeric features
        numeric_features = FeatureValidator.validate_numeric_features(
            dataset, features=["int_col", "float_col", "bool_col"]
        )
        assert set(numeric_features) == {"int_col", "float_col", "bool_col"}

        # Test with mixed valid and invalid features
        with pytest.raises(DataTypeError):
            FeatureValidator.validate_numeric_features(
                dataset, features=["int_col", "string_col"]
            )


class TestFeatureValidatorDataQuality:
    """Test data quality checking methods."""

    def test_check_data_quality_perfect_data(self):
        """Test quality check with perfect data."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        # Mock get_numeric_features
        with patch.object(
            dataset, "get_numeric_features", return_value=["feature1", "feature2"]
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        assert quality_report["n_samples"] == 5
        assert quality_report["n_features"] == 3
        assert quality_report["missing_values"] == {}
        assert quality_report["constant_features"] == []
        assert quality_report["low_variance_features"] == []
        assert quality_report["infinite_values"] == {}
        assert quality_report["duplicate_rows"] == 0
        assert quality_report["quality_score"] == 1.0

    def test_check_data_quality_missing_values(self):
        """Test quality check with missing values."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],
                "feature2": [1.1, np.nan, 3.3, np.nan, 5.5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        with patch.object(
            dataset, "get_numeric_features", return_value=["feature1", "feature2"]
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        assert "feature1" in quality_report["missing_values"]
        assert "feature2" in quality_report["missing_values"]
        assert quality_report["missing_values"]["feature1"]["count"] == 1
        assert quality_report["missing_values"]["feature2"]["count"] == 2
        assert quality_report["missing_values"]["feature1"]["ratio"] == 0.2
        assert quality_report["missing_values"]["feature2"]["ratio"] == 0.4
        assert quality_report["quality_score"] < 1.0

    def test_check_data_quality_constant_features(self):
        """Test quality check with constant features."""
        data = pd.DataFrame(
            {
                "constant_feature": [5, 5, 5, 5, 5],  # Constant
                "normal_feature": [1, 2, 3, 4, 5],  # Normal variation
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        with patch.object(
            dataset,
            "get_numeric_features",
            return_value=["constant_feature", "normal_feature"],
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        assert "constant_feature" in quality_report["constant_features"]
        assert "normal_feature" not in quality_report["constant_features"]
        assert quality_report["quality_score"] < 1.0

    def test_check_data_quality_low_variance_features(self):
        """Test quality check with low variance features."""
        data = pd.DataFrame(
            {
                "low_variance": [1, 1, 1, 1, 2],  # Low variance (80% same value)
                "normal_feature": [1, 2, 3, 4, 5],  # Normal variation
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        with patch.object(
            dataset,
            "get_numeric_features",
            return_value=["low_variance", "normal_feature"],
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        assert "low_variance" in quality_report["low_variance_features"]
        assert "normal_feature" not in quality_report["low_variance_features"]
        assert quality_report["quality_score"] < 1.0

    def test_check_data_quality_infinite_values(self):
        """Test quality check with infinite values."""
        data = pd.DataFrame(
            {
                "feature_with_inf": [1, 2, np.inf, 4, 5],
                "normal_feature": [1.1, 2.2, 3.3, 4.4, 5.5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        with patch.object(
            dataset,
            "get_numeric_features",
            return_value=["feature_with_inf", "normal_feature"],
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        assert "feature_with_inf" in quality_report["infinite_values"]
        assert quality_report["infinite_values"]["feature_with_inf"] == 1
        assert "normal_feature" not in quality_report["infinite_values"]
        assert quality_report["quality_score"] < 1.0

    def test_check_data_quality_duplicate_rows(self):
        """Test quality check with duplicate rows."""
        data = pd.DataFrame(
            {
                "feature1": [
                    1,
                    2,
                    1,
                    4,
                    5,
                ],  # Row 0 and 2 are duplicates (except target)
                "feature2": [1.1, 2.2, 1.1, 4.4, 5.5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        with patch.object(
            dataset, "get_numeric_features", return_value=["feature1", "feature2"]
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        assert quality_report["duplicate_rows"] > 0
        assert quality_report["quality_score"] < 1.0

    def test_check_data_quality_custom_thresholds(self):
        """Test quality check with custom thresholds."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],  # 20% missing
                "feature2": [1, 1, 1, 1, 1],  # Constant
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        with patch.object(
            dataset, "get_numeric_features", return_value=["feature1", "feature2"]
        ):
            # Test with stricter thresholds
            quality_report = FeatureValidator.check_data_quality(
                dataset, max_missing_ratio=0.1, max_constant_ratio=0.9
            )

        # With stricter thresholds, quality score should be lower
        assert quality_report["quality_score"] < 0.8

    def test_check_data_quality_multiple_issues(self):
        """Test quality check with multiple data quality issues."""
        data = pd.DataFrame(
            {
                "missing_feature": [1, np.nan, 3, np.nan, 5],  # Missing values
                "constant_feature": [7, 7, 7, 7, 7],  # Constant
                "inf_feature": [1, 2, np.inf, 4, -np.inf],  # Infinite values
                "duplicate1": [1, 2, 1, 4, 5],  # For duplicates
                "duplicate2": [1, 2, 1, 4, 5],  # For duplicates
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        with patch.object(
            dataset,
            "get_numeric_features",
            return_value=[
                "missing_feature",
                "constant_feature",
                "inf_feature",
                "duplicate1",
                "duplicate2",
            ],
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        # Should detect multiple issues
        assert len(quality_report["missing_values"]) > 0
        assert len(quality_report["constant_features"]) > 0
        assert len(quality_report["infinite_values"]) > 0
        assert quality_report["duplicate_rows"] > 0
        assert quality_report["quality_score"] < 0.5  # Very low quality

    def test_check_data_quality_edge_case_empty_dataset(self):
        """Test quality check with empty dataset."""
        data = pd.DataFrame()
        dataset = Dataset(data=data)

        with patch.object(dataset, "get_numeric_features", return_value=[]):
            quality_report = FeatureValidator.check_data_quality(dataset)

        assert quality_report["n_samples"] == 0
        assert quality_report["n_features"] == 0
        assert quality_report["quality_score"] == 1.0

    def test_check_data_quality_single_row(self):
        """Test quality check with single row dataset."""
        data = pd.DataFrame({"feature1": [1], "feature2": [2.0], "target": [0]})

        dataset = Dataset(data=data, target_column="target")

        with patch.object(
            dataset, "get_numeric_features", return_value=["feature1", "feature2"]
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        assert quality_report["n_samples"] == 1
        # Single row means all features are "constant"
        assert len(quality_report["constant_features"]) > 0

    def test_check_data_quality_score_bounds(self):
        """Test that quality score is properly bounded."""
        data = pd.DataFrame(
            {
                "bad_feature": [np.nan] * 10,  # 100% missing
                "target": list(range(10)),
            }
        )

        dataset = Dataset(data=data, target_column="target")

        with patch.object(
            dataset, "get_numeric_features", return_value=["bad_feature"]
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        # Quality score should never go below 0
        assert quality_report["quality_score"] >= 0.0
        assert quality_report["quality_score"] <= 1.0


class TestFeatureValidatorPreprocessingSuggestions:
    """Test preprocessing suggestion methods."""

    def test_suggest_preprocessing_perfect_data(self):
        """Test suggestions for perfect data."""
        quality_report = {
            "missing_values": {},
            "constant_features": [],
            "low_variance_features": [],
            "infinite_values": {},
            "duplicate_rows": 0,
            "quality_score": 1.0,
        }

        suggestions = FeatureValidator.suggest_preprocessing(quality_report)
        assert suggestions == []

    def test_suggest_preprocessing_missing_values(self):
        """Test suggestions for missing values."""
        quality_report = {
            "missing_values": {"feature1": {"count": 5, "ratio": 0.1}},
            "constant_features": [],
            "low_variance_features": [],
            "infinite_values": {},
            "duplicate_rows": 0,
            "quality_score": 0.9,
        }

        suggestions = FeatureValidator.suggest_preprocessing(quality_report)
        assert any("missing values" in suggestion for suggestion in suggestions)

    def test_suggest_preprocessing_constant_features(self):
        """Test suggestions for constant features."""
        quality_report = {
            "missing_values": {},
            "constant_features": ["feature1", "feature2"],
            "low_variance_features": [],
            "infinite_values": {},
            "duplicate_rows": 0,
            "quality_score": 0.8,
        }

        suggestions = FeatureValidator.suggest_preprocessing(quality_report)
        constant_suggestion = next(s for s in suggestions if "constant features" in s)
        assert "feature1" in constant_suggestion
        assert "feature2" in constant_suggestion

    def test_suggest_preprocessing_low_variance_features(self):
        """Test suggestions for low variance features."""
        quality_report = {
            "missing_values": {},
            "constant_features": [],
            "low_variance_features": ["feature1"],
            "infinite_values": {},
            "duplicate_rows": 0,
            "quality_score": 0.9,
        }

        suggestions = FeatureValidator.suggest_preprocessing(quality_report)
        assert any("low variance" in suggestion for suggestion in suggestions)

    def test_suggest_preprocessing_infinite_values(self):
        """Test suggestions for infinite values."""
        quality_report = {
            "missing_values": {},
            "constant_features": [],
            "low_variance_features": [],
            "infinite_values": {"feature1": 3},
            "duplicate_rows": 0,
            "quality_score": 0.8,
        }

        suggestions = FeatureValidator.suggest_preprocessing(quality_report)
        assert any("infinite values" in suggestion for suggestion in suggestions)

    def test_suggest_preprocessing_duplicate_rows(self):
        """Test suggestions for duplicate rows."""
        quality_report = {
            "missing_values": {},
            "constant_features": [],
            "low_variance_features": [],
            "infinite_values": {},
            "duplicate_rows": 5,
            "quality_score": 0.9,
        }

        suggestions = FeatureValidator.suggest_preprocessing(quality_report)
        duplicate_suggestion = next(s for s in suggestions if "duplicate" in s)
        assert "5" in duplicate_suggestion

    def test_suggest_preprocessing_low_quality_score(self):
        """Test suggestions for low quality score."""
        quality_report = {
            "missing_values": {},
            "constant_features": [],
            "low_variance_features": [],
            "infinite_values": {},
            "duplicate_rows": 0,
            "quality_score": 0.7,  # Low quality
        }

        suggestions = FeatureValidator.suggest_preprocessing(quality_report)
        assert any("Data quality is low" in suggestion for suggestion in suggestions)

    def test_suggest_preprocessing_multiple_issues(self):
        """Test suggestions for multiple issues."""
        quality_report = {
            "missing_values": {"feature1": {"count": 10, "ratio": 0.2}},
            "constant_features": ["feature2"],
            "low_variance_features": ["feature3"],
            "infinite_values": {"feature4": 2},
            "duplicate_rows": 3,
            "quality_score": 0.6,
        }

        suggestions = FeatureValidator.suggest_preprocessing(quality_report)

        # Should have suggestions for all issues
        assert len(suggestions) == 6  # One for each issue type + low quality warning
        assert any("missing values" in suggestion for suggestion in suggestions)
        assert any("constant features" in suggestion for suggestion in suggestions)
        assert any("low variance" in suggestion for suggestion in suggestions)
        assert any("infinite values" in suggestion for suggestion in suggestions)
        assert any("duplicate" in suggestion for suggestion in suggestions)
        assert any("Data quality is low" in suggestion for suggestion in suggestions)

    def test_suggest_preprocessing_empty_quality_report(self):
        """Test suggestions with minimal quality report."""
        quality_report = {
            "missing_values": {},
            "constant_features": [],
            "low_variance_features": [],
            "infinite_values": {},
            "duplicate_rows": 0,
            "quality_score": 0.9,
        }

        suggestions = FeatureValidator.suggest_preprocessing(quality_report)
        assert suggestions == []


class TestFeatureValidatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_compatibility_validation_with_none_data(self):
        """Test compatibility validation edge cases."""
        # This tests the robustness of the validator with edge data
        data = pd.DataFrame({"col1": [None, None, None]})
        dataset = Dataset(data=data)

        with patch.object(dataset, "feature_names", ["col1"]):
            # Should handle None values gracefully
            FeatureValidator.validate_compatibility(dataset, dataset)

    def test_numeric_validation_with_special_pandas_dtypes(self):
        """Test numeric validation with special pandas dtypes."""
        data = pd.DataFrame(
            {
                "category_col": pd.Categorical(["A", "B", "C"]),
                "datetime_col": pd.to_datetime(
                    ["2021-01-01", "2021-01-02", "2021-01-03"]
                ),
                "numeric_col": [1, 2, 3],
            }
        )

        dataset = Dataset(data=data)

        # Category and datetime should not be considered numeric
        with pytest.raises(DataTypeError):
            FeatureValidator.validate_numeric_features(
                dataset, features=["category_col"]
            )

        with pytest.raises(DataTypeError):
            FeatureValidator.validate_numeric_features(
                dataset, features=["datetime_col"]
            )

        # Numeric should pass
        result = FeatureValidator.validate_numeric_features(
            dataset, features=["numeric_col"]
        )
        assert result == ["numeric_col"]

    def test_quality_check_with_all_nan_column(self):
        """Test quality check with column that's entirely NaN."""
        data = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan], "normal": [1, 2, 3]})

        dataset = Dataset(data=data)

        with patch.object(
            dataset, "get_numeric_features", return_value=["all_nan", "normal"]
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        assert quality_report["missing_values"]["all_nan"]["ratio"] == 1.0
        assert quality_report["quality_score"] == 0.0  # Should be very low

    def test_quality_check_with_mixed_inf_types(self):
        """Test quality check with both positive and negative infinity."""
        data = pd.DataFrame(
            {"mixed_inf": [1, np.inf, -np.inf, 4, 5], "normal": [1, 2, 3, 4, 5]}
        )

        dataset = Dataset(data=data)

        with patch.object(
            dataset, "get_numeric_features", return_value=["mixed_inf", "normal"]
        ):
            quality_report = FeatureValidator.check_data_quality(dataset)

        assert quality_report["infinite_values"]["mixed_inf"] == 2  # Both +inf and -inf

    def test_dtype_compatibility_edge_cases(self):
        """Test dtype compatibility with edge cases."""
        # Test with object dtype containing numbers vs actual numeric dtype
        mixed_data = pd.DataFrame(
            {
                "object_numbers": pd.Series(["1", "2", "3"], dtype="object"),
                "real_numbers": [1, 2, 3],
            }
        )

        ref_dataset = Dataset(data=pd.DataFrame({"col": [1, 2, 3]}))
        target_dataset = Dataset(
            data=pd.DataFrame({"col": pd.Series(["1", "2", "3"], dtype="object")})
        )

        # Should be incompatible even though values look similar
        with pytest.raises(DataTypeError):
            FeatureValidator.validate_compatibility(ref_dataset, target_dataset)

    def test_performance_with_large_dataset(self):
        """Test performance with large datasets."""
        # Create large dataset
        np.random.seed(42)
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.random(1000) for i in range(50)}
        )

        dataset = Dataset(data=large_data)

        with patch.object(
            dataset, "get_numeric_features", return_value=list(large_data.columns)
        ):
            # Should handle large datasets efficiently
            quality_report = FeatureValidator.check_data_quality(dataset)

            assert quality_report["n_samples"] == 1000
            assert quality_report["n_features"] == 50
            assert isinstance(quality_report["quality_score"], float)

    def test_error_propagation_and_details(self):
        """Test that errors contain proper details for debugging."""
        ref_data = pd.DataFrame({"feature1": [1, 2, 3]})
        target_data = pd.DataFrame({"feature2": [4, 5, 6]})  # Different feature name

        ref_dataset = Dataset(data=ref_data)
        target_dataset = Dataset(data=target_data)

        with pytest.raises(FeatureMismatchError) as exc_info:
            FeatureValidator.validate_compatibility(
                ref_dataset, target_dataset, strict=True
            )

        # Check that error details are properly populated
        assert hasattr(exc_info.value, "details")
        assert "expected_features" in exc_info.value.details
        assert "actual_features" in exc_info.value.details
        assert "missing_features" in exc_info.value.details
        assert "extra_features" in exc_info.value.details


class TestFeatureValidatorIntegration:
    """Test integration scenarios combining multiple validation methods."""

    def test_full_validation_pipeline(self):
        """Test complete validation pipeline."""
        # Create reference dataset
        ref_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        # Create target dataset with some issues
        target_data = pd.DataFrame(
            {
                "feature1": [6, 7, 8, 9, 10],
                "feature2": [6.6, 7.7, 8.8, 9.9, 10.1],
                "target": [1, 0, 1, 0, 1],
            }
        )

        ref_dataset = Dataset(data=ref_data, target_column="target")
        target_dataset = Dataset(data=target_data, target_column="target")

        # Step 1: Validate compatibility
        FeatureValidator.validate_compatibility(ref_dataset, target_dataset)

        # Step 2: Validate numeric features
        with patch.object(
            target_dataset,
            "get_numeric_features",
            return_value=["feature1", "feature2"],
        ):
            numeric_features = FeatureValidator.validate_numeric_features(
                target_dataset
            )
            assert set(numeric_features) == {"feature1", "feature2"}

        # Step 3: Check data quality
        with patch.object(
            target_dataset,
            "get_numeric_features",
            return_value=["feature1", "feature2"],
        ):
            quality_report = FeatureValidator.check_data_quality(target_dataset)
            assert quality_report["quality_score"] > 0.9

        # Step 4: Get preprocessing suggestions
        suggestions = FeatureValidator.suggest_preprocessing(quality_report)
        assert len(suggestions) == 0  # Should be minimal for good data

    def test_validation_with_preprocessing_workflow(self):
        """Test validation integrated with preprocessing workflow."""
        # Create dataset with quality issues
        problematic_data = pd.DataFrame(
            {
                "good_feature": [1, 2, 3, 4, 5],
                "missing_feature": [1, np.nan, 3, np.nan, 5],
                "constant_feature": [7, 7, 7, 7, 7],
                "inf_feature": [1, 2, np.inf, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=problematic_data, target_column="target")

        # Get numeric features that should be validated
        with patch.object(
            dataset,
            "get_numeric_features",
            return_value=[
                "good_feature",
                "missing_feature",
                "constant_feature",
                "inf_feature",
            ],
        ):
            # Validate only the good feature should pass strict validation
            good_features = FeatureValidator.validate_numeric_features(
                dataset, features=["good_feature"]
            )
            assert good_features == ["good_feature"]

            # Check overall data quality
            quality_report = FeatureValidator.check_data_quality(dataset)
            assert quality_report["quality_score"] < 0.8

            # Get actionable suggestions
            suggestions = FeatureValidator.suggest_preprocessing(quality_report)
            assert len(suggestions) > 0

            # Verify suggestions address the specific issues
            suggestion_text = " ".join(suggestions)
            assert "missing values" in suggestion_text
            assert "constant features" in suggestion_text
            assert "infinite values" in suggestion_text

    def test_cross_dataset_validation_workflow(self):
        """Test validation workflow across multiple datasets."""
        # Training dataset
        train_data = pd.DataFrame(
            {
                "feature1": range(100),
                "feature2": np.random.random(100),
                "target": np.random.choice([0, 1], 100),
            }
        )

        # Validation dataset (compatible)
        val_data = pd.DataFrame(
            {
                "feature1": range(100, 150),
                "feature2": np.random.random(50),
                "target": np.random.choice([0, 1], 50),
            }
        )

        # Test dataset (with issues)
        test_data = pd.DataFrame(
            {
                "feature1": ["a", "b", "c"],  # Wrong type
                "feature2": [1.1, 2.2, 3.3],
                "target": [0, 1, 0],
            }
        )

        train_dataset = Dataset(data=train_data, target_column="target")
        val_dataset = Dataset(data=val_data, target_column="target")
        test_dataset = Dataset(data=test_data, target_column="target")

        # Validation dataset should be compatible
        FeatureValidator.validate_compatibility(train_dataset, val_dataset)

        # Test dataset should be incompatible
        with pytest.raises(DataTypeError):
            FeatureValidator.validate_compatibility(train_dataset, test_dataset)

        # Check quality across datasets
        with patch.object(
            train_dataset, "get_numeric_features", return_value=["feature1", "feature2"]
        ):
            train_quality = FeatureValidator.check_data_quality(train_dataset)

        with patch.object(
            val_dataset, "get_numeric_features", return_value=["feature1", "feature2"]
        ):
            val_quality = FeatureValidator.check_data_quality(val_dataset)

        # Both should have good quality
        assert train_quality["quality_score"] > 0.9
        assert val_quality["quality_score"] > 0.9

    def test_comprehensive_feature_validation_report(self):
        """Test generating comprehensive validation report."""
        data = pd.DataFrame(
            {
                "numeric1": [1, 2, 3, 4, 5],
                "numeric2": [1.1, 2.2, np.nan, 4.4, 5.5],
                "categorical": ["A", "B", "A", "B", "A"],
                "constant": [1, 1, 1, 1, 1],
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(data=data, target_column="target")

        # Comprehensive validation
        validation_report = {}

        # 1. Identify numeric features
        with patch.object(
            dataset,
            "get_numeric_features",
            return_value=["numeric1", "numeric2", "constant"],
        ):
            validation_report["numeric_features"] = (
                FeatureValidator.validate_numeric_features(dataset)
            )

        # 2. Check data quality
        with patch.object(
            dataset,
            "get_numeric_features",
            return_value=["numeric1", "numeric2", "constant"],
        ):
            validation_report["quality_report"] = FeatureValidator.check_data_quality(
                dataset
            )

        # 3. Generate suggestions
        validation_report["suggestions"] = FeatureValidator.suggest_preprocessing(
            validation_report["quality_report"]
        )

        # Verify comprehensive report
        assert "numeric_features" in validation_report
        assert "quality_report" in validation_report
        assert "suggestions" in validation_report

        assert len(validation_report["numeric_features"]) == 3
        assert validation_report["quality_report"]["n_samples"] == 5
        assert len(validation_report["suggestions"]) > 0
