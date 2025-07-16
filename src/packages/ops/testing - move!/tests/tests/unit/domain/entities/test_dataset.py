"""Comprehensive tests for Dataset domain entity."""

from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from monorepo.domain.entities.dataset import Dataset
from monorepo.domain.exceptions import InvalidDataError


class TestDatasetInitialization:
    """Test dataset initialization and validation."""

    def test_dataset_initialization_with_dataframe(self):
        """Test dataset initialization with pandas DataFrame."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "feature3": ["a", "b", "c", "d", "e"],
            }
        )

        dataset = Dataset(name="Test Dataset", data=data)

        assert dataset.name == "Test Dataset"
        assert dataset.data.equals(data)
        assert dataset.feature_names == ["feature1", "feature2", "feature3"]
        assert isinstance(dataset.id, type(uuid4()))
        assert isinstance(dataset.created_at, datetime)
        assert dataset.metadata == {}
        assert dataset.description is None
        assert dataset.target_column is None

    def test_dataset_initialization_with_numpy_array(self):
        """Test dataset initialization with numpy array."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        dataset = Dataset(name="Numpy Dataset", data=data)

        assert dataset.name == "Numpy Dataset"
        assert dataset.data.shape == (3, 3)
        assert dataset.feature_names == ["feature_0", "feature_1", "feature_2"]
        assert list(dataset.data.columns) == ["feature_0", "feature_1", "feature_2"]

    def test_dataset_initialization_with_1d_numpy_array(self):
        """Test dataset initialization with 1D numpy array."""
        data = np.array([1, 2, 3, 4, 5])

        dataset = Dataset(name="1D Dataset", data=data)

        assert dataset.data.shape == (5, 1)
        assert dataset.feature_names == ["feature_0"]
        assert list(dataset.data.columns) == ["feature_0"]

    def test_dataset_initialization_with_feature_names(self):
        """Test dataset initialization with custom feature names."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        feature_names = ["x", "y"]

        dataset = Dataset(
            name="Custom Features", data=data, feature_names=feature_names
        )

        assert dataset.feature_names == ["x", "y"]
        assert list(dataset.data.columns) == ["x", "y"]

    def test_dataset_initialization_with_target_column(self):
        """Test dataset initialization with target column."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(name="Target Dataset", data=data, target_column="target")

        assert dataset.target_column == "target"
        assert dataset.has_target is True
        assert dataset.n_features == 2  # Excludes target column

    def test_dataset_initialization_with_all_parameters(self):
        """Test dataset initialization with all parameters."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "labels": [0, 1, 0]}
        )
        metadata = {"source": "test", "version": "1.0"}

        dataset = Dataset(
            name="Complete Dataset",
            data=data,
            feature_names=["feature1", "feature2"],
            metadata=metadata,
            description="Test dataset for validation",
            target_column="labels",
        )

        assert dataset.name == "Complete Dataset"
        assert dataset.feature_names == ["feature1", "feature2"]
        assert dataset.metadata == metadata
        assert dataset.description == "Test dataset for validation"
        assert dataset.target_column == "labels"

    def test_dataset_validation_empty_name(self):
        """Test dataset validation with empty name."""
        data = pd.DataFrame({"feature1": [1, 2, 3]})

        with pytest.raises(ValueError, match="Dataset name cannot be empty"):
            Dataset(name="", data=data)

    def test_dataset_validation_empty_dataframe(self):
        """Test dataset validation with empty DataFrame."""
        data = pd.DataFrame()

        with pytest.raises(InvalidDataError, match="Dataset cannot be empty"):
            Dataset(name="Empty Dataset", data=data)

    def test_dataset_validation_invalid_data_type(self):
        """Test dataset validation with invalid data type."""
        with pytest.raises(
            TypeError, match="Data must be pandas DataFrame or numpy array"
        ):
            Dataset(name="Invalid Dataset", data=[1, 2, 3])

    def test_dataset_validation_mismatched_feature_names(self):
        """Test dataset validation with mismatched feature names."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        feature_names = ["x", "y"]  # Only 2 names for 3 features

        with pytest.raises(
            ValueError, match="Number of feature names .* doesn't match data dimensions"
        ):
            Dataset(name="Mismatched Dataset", data=data, feature_names=feature_names)

    def test_dataset_validation_invalid_target_column(self):
        """Test dataset validation with invalid target column."""
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        with pytest.raises(
            ValueError, match="Target column 'nonexistent' not found in dataset"
        ):
            Dataset(name="Invalid Target", data=data, target_column="nonexistent")


class TestDatasetProperties:
    """Test dataset properties and getters."""

    def test_shape_property(self):
        """Test shape property."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]}
        )
        dataset = Dataset(name="Test Dataset", data=data)

        assert dataset.shape == (5, 2)

    def test_n_samples_property(self):
        """Test n_samples property."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]}
        )
        dataset = Dataset(name="Test Dataset", data=data)

        assert dataset.n_samples == 5

    def test_n_features_property_without_target(self):
        """Test n_features property without target column."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )
        dataset = Dataset(name="Test Dataset", data=data)

        assert dataset.n_features == 3

    def test_n_features_property_with_target(self):
        """Test n_features property with target column."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )
        dataset = Dataset(name="Test Dataset", data=data, target_column="target")

        assert dataset.n_features == 2  # Excludes target

    def test_features_property_without_target(self):
        """Test features property without target column."""
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        dataset = Dataset(name="Test Dataset", data=data)

        features = dataset.features
        assert features.equals(data)

    def test_features_property_with_target(self):
        """Test features property with target column."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )
        dataset = Dataset(name="Test Dataset", data=data, target_column="target")

        features = dataset.features
        expected_features = data.drop(columns=["target"])
        assert features.equals(expected_features)

    def test_target_property_without_target(self):
        """Test target property without target column."""
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        dataset = Dataset(name="Test Dataset", data=data)

        assert dataset.target is None

    def test_target_property_with_target(self):
        """Test target property with target column."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )
        dataset = Dataset(name="Test Dataset", data=data, target_column="target")

        target = dataset.target
        expected_target = data["target"]
        assert target.equals(expected_target)

    def test_has_target_property(self):
        """Test has_target property."""
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        dataset_without_target = Dataset(name="No Target", data=data)
        assert dataset_without_target.has_target is False

        dataset_with_target = Dataset(
            name="With Target",
            data=data.assign(target=[0, 1, 0]),
            target_column="target",
        )
        assert dataset_with_target.has_target is True

    def test_memory_usage_property(self):
        """Test memory_usage property."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]}
        )
        dataset = Dataset(name="Test Dataset", data=data)

        memory_usage = dataset.memory_usage
        assert isinstance(memory_usage, int)
        assert memory_usage > 0

    def test_dtypes_property(self):
        """Test dtypes property."""
        data = pd.DataFrame(
            {
                "int_feature": [1, 2, 3],
                "float_feature": [1.1, 2.2, 3.3],
                "string_feature": ["a", "b", "c"],
            }
        )
        dataset = Dataset(name="Test Dataset", data=data)

        dtypes = dataset.dtypes
        assert dtypes["int_feature"] == "int64"
        assert dtypes["float_feature"] == "float64"
        assert dtypes["string_feature"] == "object"


class TestDatasetFeatureTypes:
    """Test dataset feature type identification."""

    def test_get_numeric_features(self):
        """Test getting numeric features."""
        data = pd.DataFrame(
            {
                "int_feature": [1, 2, 3],
                "float_feature": [1.1, 2.2, 3.3],
                "string_feature": ["a", "b", "c"],
                "target": [0, 1, 0],
            }
        )
        dataset = Dataset(name="Test Dataset", data=data, target_column="target")

        numeric_features = dataset.get_numeric_features()
        assert set(numeric_features) == {"int_feature", "float_feature"}

    def test_get_categorical_features(self):
        """Test getting categorical features."""
        data = pd.DataFrame(
            {
                "int_feature": [1, 2, 3],
                "float_feature": [1.1, 2.2, 3.3],
                "string_feature": ["a", "b", "c"],
                "target": [0, 1, 0],
            }
        )
        dataset = Dataset(name="Test Dataset", data=data, target_column="target")

        categorical_features = dataset.get_categorical_features()
        assert categorical_features == ["string_feature"]

    def test_get_numeric_features_excludes_target(self):
        """Test that get_numeric_features excludes target column."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "numeric_target": [0, 1, 0]}
        )
        dataset = Dataset(
            name="Test Dataset", data=data, target_column="numeric_target"
        )

        numeric_features = dataset.get_numeric_features()
        assert set(numeric_features) == {"feature1", "feature2"}
        assert "numeric_target" not in numeric_features

    def test_get_categorical_features_excludes_target(self):
        """Test that get_categorical_features excludes target column."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "categorical_feature": ["a", "b", "c"],
                "categorical_target": ["x", "y", "z"],
            }
        )
        dataset = Dataset(
            name="Test Dataset", data=data, target_column="categorical_target"
        )

        categorical_features = dataset.get_categorical_features()
        assert categorical_features == ["categorical_feature"]
        assert "categorical_target" not in categorical_features

    def test_feature_types_with_mixed_dtypes(self):
        """Test feature type identification with mixed data types."""
        data = pd.DataFrame(
            {
                "int16_feature": pd.array([1, 2, 3], dtype="int16"),
                "int32_feature": pd.array([1, 2, 3], dtype="int32"),
                "int64_feature": pd.array([1, 2, 3], dtype="int64"),
                "float16_feature": pd.array([1.1, 2.2, 3.3], dtype="float16"),
                "float32_feature": pd.array([1.1, 2.2, 3.3], dtype="float32"),
                "float64_feature": pd.array([1.1, 2.2, 3.3], dtype="float64"),
                "object_feature": ["a", "b", "c"],
                "bool_feature": [True, False, True],
            }
        )
        dataset = Dataset(name="Mixed Types", data=data)

        numeric_features = dataset.get_numeric_features()
        categorical_features = dataset.get_categorical_features()

        expected_numeric = {
            "int16_feature",
            "int32_feature",
            "int64_feature",
            "float16_feature",
            "float32_feature",
            "float64_feature",
        }
        assert set(numeric_features) == expected_numeric
        assert categorical_features == ["object_feature"]


class TestDatasetMethods:
    """Test dataset methods."""

    def test_sample_method(self):
        """Test dataset sampling method."""
        data = pd.DataFrame({"feature1": range(10), "feature2": range(10, 20)})
        dataset = Dataset(name="Original Dataset", data=data)

        sampled = dataset.sample(n=5, random_state=42)

        assert sampled.n_samples == 5
        assert sampled.n_features == 2
        assert sampled.name == "Original Dataset_sample_5"
        assert sampled.metadata["parent_dataset_id"] == str(dataset.id)
        assert "Sample of 5 rows" in sampled.description

    def test_sample_method_with_target(self):
        """Test dataset sampling with target column."""
        data = pd.DataFrame(
            {"feature1": range(10), "feature2": range(10, 20), "target": [0, 1] * 5}
        )
        dataset = Dataset(name="Target Dataset", data=data, target_column="target")

        sampled = dataset.sample(n=6, random_state=42)

        assert sampled.n_samples == 6
        assert sampled.n_features == 2  # Excludes target
        assert sampled.has_target is True
        assert sampled.target_column == "target"

    def test_sample_method_invalid_size(self):
        """Test dataset sampling with invalid size."""
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        dataset = Dataset(name="Small Dataset", data=data)

        with pytest.raises(
            ValueError, match="Cannot sample 5 rows from dataset with 3 rows"
        ):
            dataset.sample(n=5)

    def test_split_method(self):
        """Test dataset splitting method."""
        data = pd.DataFrame({"feature1": range(10), "feature2": range(10, 20)})
        dataset = Dataset(name="Original Dataset", data=data)

        train, test = dataset.split(test_size=0.2, random_state=42)

        assert train.n_samples == 8
        assert test.n_samples == 2
        assert train.n_features == 2
        assert test.n_features == 2
        assert train.name == "Original Dataset_train"
        assert test.name == "Original Dataset_test"
        assert train.metadata["split"] == "train"
        assert test.metadata["split"] == "test"
        assert train.metadata["parent_dataset_id"] == str(dataset.id)
        assert test.metadata["parent_dataset_id"] == str(dataset.id)

    def test_split_method_with_target(self):
        """Test dataset splitting with target column."""
        data = pd.DataFrame(
            {"feature1": range(10), "feature2": range(10, 20), "target": [0, 1] * 5}
        )
        dataset = Dataset(name="Target Dataset", data=data, target_column="target")

        train, test = dataset.split(test_size=0.3, random_state=42)

        assert train.has_target is True
        assert test.has_target is True
        assert train.target_column == "target"
        assert test.target_column == "target"
        assert train.n_samples == 7
        assert test.n_samples == 3

    def test_split_method_invalid_test_size(self):
        """Test dataset splitting with invalid test size."""
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        dataset = Dataset(name="Test Dataset", data=data)

        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            dataset.split(test_size=1.5)

        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            dataset.split(test_size=0.0)

    def test_add_metadata_method(self):
        """Test adding metadata to dataset."""
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        dataset = Dataset(name="Test Dataset", data=data)

        dataset.add_metadata("version", "1.0")
        dataset.add_metadata("source", "test")

        assert dataset.metadata["version"] == "1.0"
        assert dataset.metadata["source"] == "test"

    def test_add_metadata_overwrite(self):
        """Test overwriting existing metadata."""
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        dataset = Dataset(name="Test Dataset", data=data, metadata={"version": "1.0"})

        dataset.add_metadata("version", "2.0")
        assert dataset.metadata["version"] == "2.0"

    def test_summary_method(self):
        """Test dataset summary method."""
        data = pd.DataFrame(
            {
                "int_feature": [1, 2, 3, 4, 5],
                "float_feature": [1.1, 2.2, 3.3, 4.4, 5.5],
                "string_feature": ["a", "b", "c", "d", "e"],
                "target": [0, 1, 0, 1, 0],
            }
        )
        dataset = Dataset(
            name="Summary Dataset",
            data=data,
            description="Test dataset for summary",
            target_column="target",
        )

        summary = dataset.summary()

        assert summary["name"] == "Summary Dataset"
        assert summary["shape"] == (5, 4)
        assert summary["n_samples"] == 5
        assert summary["n_features"] == 3  # Excludes target
        assert summary["has_target"] is True
        assert summary["numeric_features"] == 2
        assert summary["categorical_features"] == 1
        assert summary["description"] == "Test dataset for summary"
        assert "id" in summary
        assert "memory_usage_mb" in summary
        assert "created_at" in summary

    def test_repr_method(self):
        """Test dataset string representation."""
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )
        dataset = Dataset(name="Test Dataset", data=data, target_column="target")

        repr_str = repr(dataset)
        assert "Dataset(" in repr_str
        assert "name='Test Dataset'" in repr_str
        assert "shape=(3, 3)" in repr_str
        assert "has_target=True" in repr_str


class TestDatasetEdgeCases:
    """Test dataset edge cases and error conditions."""

    def test_dataset_with_single_row(self):
        """Test dataset with single row."""
        data = pd.DataFrame({"feature1": [1], "feature2": [2]})
        dataset = Dataset(name="Single Row", data=data)

        assert dataset.n_samples == 1
        assert dataset.n_features == 2
        assert dataset.shape == (1, 2)

    def test_dataset_with_single_column(self):
        """Test dataset with single column."""
        data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        dataset = Dataset(name="Single Column", data=data)

        assert dataset.n_samples == 5
        assert dataset.n_features == 1
        assert dataset.shape == (5, 1)

    def test_dataset_with_missing_values(self):
        """Test dataset with missing values."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4, 5],
                "feature2": [1.1, np.nan, 3.3, 4.4, 5.5],
                "feature3": ["a", "b", None, "d", "e"],
            }
        )
        dataset = Dataset(name="Missing Values", data=data)

        assert dataset.n_samples == 5
        assert dataset.n_features == 3
        assert pd.isna(dataset.data.iloc[2, 0])
        assert pd.isna(dataset.data.iloc[1, 1])
        assert pd.isna(dataset.data.iloc[2, 2])

    def test_dataset_with_duplicate_feature_names(self):
        """Test dataset with duplicate feature names in DataFrame."""
        # Create DataFrame with duplicate column names
        data = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        data.columns = ["feature1", "feature1", "feature2"]

        dataset = Dataset(name="Duplicate Names", data=data)

        # Should handle duplicate names (pandas behavior)
        assert dataset.n_samples == 2
        assert len(dataset.feature_names) == 3

    def test_dataset_with_large_data(self):
        """Test dataset with large data."""
        # Create large dataset
        np.random.seed(42)
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.randn(1000) for i in range(10)}
        )
        dataset = Dataset(name="Large Dataset", data=large_data)

        assert dataset.n_samples == 1000
        assert dataset.n_features == 10
        assert dataset.memory_usage > 0

    def test_dataset_with_extreme_values(self):
        """Test dataset with extreme values."""
        data = pd.DataFrame(
            {
                "large_values": [1e10, 1e20, 1e30],
                "small_values": [1e-10, 1e-20, 1e-30],
                "infinity": [float("inf"), float("-inf"), np.nan],
                "normal_values": [1, 2, 3],
            }
        )
        dataset = Dataset(name="Extreme Values", data=data)

        assert dataset.n_samples == 3
        assert dataset.n_features == 4
        assert np.isinf(dataset.data.iloc[0, 2])
        assert np.isinf(dataset.data.iloc[1, 2])
        assert np.isnan(dataset.data.iloc[2, 2])

    def test_dataset_splitting_edge_cases(self):
        """Test dataset splitting edge cases."""
        data = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        dataset = Dataset(name="Small Dataset", data=data)

        # Test with very small test size
        train, test = dataset.split(test_size=0.01, random_state=42)
        assert train.n_samples + test.n_samples == 2

        # Test with very large test size
        train, test = dataset.split(test_size=0.99, random_state=42)
        assert train.n_samples + test.n_samples == 2

    def test_dataset_sampling_edge_cases(self):
        """Test dataset sampling edge cases."""
        data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        dataset = Dataset(name="Small Dataset", data=data)

        # Sample all rows
        sampled = dataset.sample(n=3, random_state=42)
        assert sampled.n_samples == 3

        # Sample single row
        sampled = dataset.sample(n=1, random_state=42)
        assert sampled.n_samples == 1

    def test_dataset_with_different_numpy_dtypes(self):
        """Test dataset with different numpy dtypes."""
        data = np.array([[1, 2.5, 3], [4, 5.5, 6], [7, 8.5, 9]], dtype=np.float32)

        dataset = Dataset(name="Float32 Dataset", data=data)

        assert dataset.data.dtypes.iloc[0] == np.float32
        assert dataset.n_samples == 3
        assert dataset.n_features == 3

    def test_dataset_metadata_preservation(self):
        """Test that metadata is preserved across operations."""
        data = pd.DataFrame({"feature1": range(10), "feature2": range(10, 20)})
        original_metadata = {"version": "1.0", "source": "test"}
        dataset = Dataset(name="Original", data=data, metadata=original_metadata)

        # Test sampling preserves metadata
        sampled = dataset.sample(n=5, random_state=42)
        assert "version" in sampled.metadata
        assert "source" in sampled.metadata
        assert sampled.metadata["parent_dataset_id"] == str(dataset.id)

        # Test splitting preserves metadata
        train, test = dataset.split(test_size=0.2, random_state=42)
        assert "version" in train.metadata
        assert "source" in train.metadata
        assert "version" in test.metadata
        assert "source" in test.metadata
