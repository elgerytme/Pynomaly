"""Comprehensive tests for data preprocessing infrastructure - Phase 2 Coverage Enhancement."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from monorepo.domain.entities import Dataset
from monorepo.infrastructure.preprocessing.data_cleaner import (
    DataCleaner,
    MissingValueStrategy,
    OutlierStrategy,
)
from monorepo.infrastructure.preprocessing.data_transformer import (
    DataTransformer,
    EncodingStrategy,
    FeatureSelectionStrategy,
    ScalingStrategy,
)
from monorepo.infrastructure.preprocessing.preprocessing_pipeline import (
    PipelineConfig,
    PreprocessingPipeline,
    PreprocessingStep,
)


class TestDataCleaner:
    """Comprehensive tests for DataCleaner functionality."""

    @pytest.fixture
    def data_cleaner(self):
        """Create DataCleaner instance."""
        return DataCleaner()

    @pytest.fixture
    def sample_dataset_with_missing(self):
        """Create dataset with missing values for testing."""
        data = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, np.nan, 4.0, 5.0],
                "feature_2": [10.0, np.nan, 30.0, np.nan, 50.0],
                "feature_3": [100.0, 200.0, 300.0, 400.0, 500.0],
                "target": [0, 1, 0, 1, 0],
            }
        )
        return Dataset(name="test_missing", data=data, target_column="target")

    @pytest.fixture
    def sample_dataset_with_outliers(self):
        """Create dataset with outliers for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 95)
        outlier_data = np.array([10, -10, 15, -15, 20])  # Clear outliers
        feature_data = np.concatenate([normal_data, outlier_data])

        data = pd.DataFrame(
            {
                "feature_1": feature_data,
                "feature_2": np.random.normal(5, 2, 100),
                "target": np.random.choice([0, 1], 100),
            }
        )
        return Dataset(name="test_outliers", data=data, target_column="target")

    def test_handle_missing_values_drop_rows(
        self, data_cleaner, sample_dataset_with_missing
    ):
        """Test dropping rows with missing values."""
        cleaned = data_cleaner.handle_missing_values(
            sample_dataset_with_missing, strategy=MissingValueStrategy.DROP_ROWS
        )

        # Should remove rows with any NaN values
        assert len(cleaned.data) == 2  # Only rows 0 and 4 have no NaN
        assert cleaned.data.isnull().sum().sum() == 0
        assert cleaned.target_column == "target"

    def test_handle_missing_values_drop_columns(
        self, data_cleaner, sample_dataset_with_missing
    ):
        """Test dropping columns with high missing ratio."""
        cleaned = data_cleaner.handle_missing_values(
            sample_dataset_with_missing,
            strategy=MissingValueStrategy.DROP_COLUMNS,
            threshold=0.3,  # Drop columns with >30% missing
        )

        # feature_2 has 40% missing (2/5), should be dropped
        assert "feature_2" not in cleaned.data.columns
        assert "feature_1" in cleaned.data.columns  # 20% missing, kept
        assert "feature_3" in cleaned.data.columns  # No missing, kept
        assert cleaned.target_column == "target"

    def test_handle_missing_values_fill_mean(
        self, data_cleaner, sample_dataset_with_missing
    ):
        """Test filling missing values with mean."""
        cleaned = data_cleaner.handle_missing_values(
            sample_dataset_with_missing, strategy=MissingValueStrategy.FILL_MEAN
        )

        # Check that NaN values are filled
        assert cleaned.data.isnull().sum().sum() == 0

        # Check mean filling for feature_1 (1,2,4,5 -> mean = 3.0)
        expected_mean = (1.0 + 2.0 + 4.0 + 5.0) / 4
        assert cleaned.data.loc[2, "feature_1"] == expected_mean

    def test_handle_missing_values_fill_median(
        self, data_cleaner, sample_dataset_with_missing
    ):
        """Test filling missing values with median."""
        cleaned = data_cleaner.handle_missing_values(
            sample_dataset_with_missing, strategy=MissingValueStrategy.FILL_MEDIAN
        )

        assert cleaned.data.isnull().sum().sum() == 0

        # Check median filling for feature_1 (1,2,4,5 -> median = 3.0)
        expected_median = np.median([1.0, 2.0, 4.0, 5.0])
        assert cleaned.data.loc[2, "feature_1"] == expected_median

    def test_handle_missing_values_fill_constant(
        self, data_cleaner, sample_dataset_with_missing
    ):
        """Test filling missing values with constant."""
        cleaned = data_cleaner.handle_missing_values(
            sample_dataset_with_missing,
            strategy=MissingValueStrategy.FILL_CONSTANT,
            fill_value=-999,
        )

        assert cleaned.data.isnull().sum().sum() == 0
        assert cleaned.data.loc[2, "feature_1"] == -999
        assert cleaned.data.loc[1, "feature_2"] == -999
        assert cleaned.data.loc[3, "feature_2"] == -999

    def test_handle_missing_values_knn_impute(
        self, data_cleaner, sample_dataset_with_missing
    ):
        """Test KNN imputation for missing values."""
        with patch("sklearn.impute.KNNImputer") as mock_knn:
            mock_imputer = Mock()
            mock_knn.return_value = mock_imputer
            mock_imputer.fit_transform.return_value = np.array(
                [
                    [1.0, 10.0],
                    [2.0, 25.0],  # Imputed values
                    [3.5, 30.0],  # Imputed values
                    [4.0, 35.0],  # Imputed values
                    [5.0, 50.0],
                ]
            )

            cleaned = data_cleaner.handle_missing_values(
                sample_dataset_with_missing, strategy=MissingValueStrategy.KNN_IMPUTE
            )

            assert cleaned.data.isnull().sum().sum() == 0
            mock_knn.assert_called_once()
            mock_imputer.fit_transform.assert_called_once()

    def test_handle_missing_values_specific_columns(
        self, data_cleaner, sample_dataset_with_missing
    ):
        """Test handling missing values for specific columns only."""
        cleaned = data_cleaner.handle_missing_values(
            sample_dataset_with_missing,
            strategy=MissingValueStrategy.FILL_MEAN,
            columns=["feature_1"],  # Only process feature_1
        )

        # feature_1 should be cleaned, feature_2 should still have NaN
        assert cleaned.data["feature_1"].isnull().sum() == 0
        assert cleaned.data["feature_2"].isnull().sum() == 2  # Original NaN count

    def test_detect_outliers_iqr(self, data_cleaner, sample_dataset_with_outliers):
        """Test outlier detection using IQR method."""
        outliers = data_cleaner.detect_outliers(
            sample_dataset_with_outliers, method="iqr", threshold=1.5
        )

        assert isinstance(outliers, dict)
        assert "feature_1" in outliers

        # Should detect the extreme outliers we added
        outlier_indices = outliers["feature_1"]
        assert len(outlier_indices) > 0

        # Check that extreme values are detected
        extreme_values = sample_dataset_with_outliers.data.loc[
            outlier_indices, "feature_1"
        ]
        assert any(abs(val) > 5 for val in extreme_values)

    def test_detect_outliers_zscore(self, data_cleaner, sample_dataset_with_outliers):
        """Test outlier detection using Z-score method."""
        outliers = data_cleaner.detect_outliers(
            sample_dataset_with_outliers, method="zscore", threshold=2.0
        )

        assert isinstance(outliers, dict)
        assert "feature_1" in outliers

        outlier_indices = outliers["feature_1"]
        assert len(outlier_indices) > 0

    def test_handle_outliers_remove(self, data_cleaner, sample_dataset_with_outliers):
        """Test removing outliers from dataset."""
        cleaned = data_cleaner.handle_outliers(
            sample_dataset_with_outliers, strategy=OutlierStrategy.REMOVE, method="iqr"
        )

        # Should have fewer rows after removing outliers
        assert len(cleaned.data) < len(sample_dataset_with_outliers.data)

        # Check that extreme values are removed
        feature_values = cleaned.data["feature_1"]
        assert all(abs(val) < 10 for val in feature_values)  # No extreme outliers

    def test_handle_outliers_clip(self, data_cleaner, sample_dataset_with_outliers):
        """Test clipping outliers to reasonable bounds."""
        cleaned = data_cleaner.handle_outliers(
            sample_dataset_with_outliers, strategy=OutlierStrategy.CLIP, method="iqr"
        )

        # Should have same number of rows
        assert len(cleaned.data) == len(sample_dataset_with_outliers.data)

        # Extreme values should be clipped
        feature_values = cleaned.data["feature_1"]
        assert max(feature_values) < max(sample_dataset_with_outliers.data["feature_1"])
        assert min(feature_values) > min(sample_dataset_with_outliers.data["feature_1"])

    def test_handle_duplicates(self, data_cleaner):
        """Test duplicate row removal."""
        data = pd.DataFrame(
            {
                "feature_1": [1, 2, 1, 3, 2],  # Duplicates: rows 0&2, 1&4
                "feature_2": [10, 20, 10, 30, 20],
                "target": [0, 1, 0, 1, 1],
            }
        )
        dataset = Dataset(name="test_duplicates", data=data, target_column="target")

        cleaned = data_cleaner.handle_duplicates(dataset)

        # Should remove 2 duplicate rows
        assert len(cleaned.data) == 3
        assert cleaned.data.duplicated().sum() == 0

    def test_handle_zero_values_remove(self, data_cleaner):
        """Test removing zero values."""
        data = pd.DataFrame(
            {
                "feature_1": [0, 1, 2, 0, 3],
                "feature_2": [10, 0, 30, 40, 0],
                "target": [0, 1, 0, 1, 0],
            }
        )
        dataset = Dataset(name="test_zeros", data=data, target_column="target")

        cleaned = data_cleaner.handle_zero_values(
            dataset, strategy="remove", columns=["feature_1", "feature_2"]
        )

        # Should remove rows with zeros in specified columns
        assert len(cleaned.data) < len(dataset.data)
        assert (cleaned.data[["feature_1", "feature_2"]] == 0).sum().sum() == 0

    def test_handle_infinite_values(self, data_cleaner):
        """Test handling infinite values."""
        data = pd.DataFrame(
            {
                "feature_1": [1.0, np.inf, 3.0, -np.inf, 5.0],
                "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "target": [0, 1, 0, 1, 0],
            }
        )
        dataset = Dataset(name="test_infinite", data=data, target_column="target")

        cleaned = data_cleaner.handle_infinite_values(dataset, strategy="remove")

        # Should remove rows with infinite values
        assert len(cleaned.data) == 3
        assert not np.isinf(cleaned.data.values).any()

    def test_comprehensive_cleaning(self, data_cleaner):
        """Test comprehensive cleaning with multiple issues."""
        # Create dataset with multiple issues
        data = pd.DataFrame(
            {
                "feature_1": [
                    1.0,
                    np.nan,
                    3.0,
                    1.0,
                    100.0,
                ],  # Missing, duplicate, outlier
                "feature_2": [10.0, 20.0, np.inf, 10.0, 40.0],  # Infinite, duplicate
                "feature_3": [0, 1, 2, 0, 3],  # Zeros
                "target": [0, 1, 0, 0, 1],
            }
        )
        dataset = Dataset(name="test_comprehensive", data=data, target_column="target")

        cleaned = data_cleaner.comprehensive_clean(
            dataset,
            handle_missing=True,
            handle_outliers=True,
            handle_duplicates=True,
            handle_zeros=False,  # Keep zeros
            handle_infinite=True,
        )

        # Should address all specified issues
        assert cleaned.data.isnull().sum().sum() == 0  # No missing values
        assert not np.isinf(cleaned.data.values).any()  # No infinite values
        assert cleaned.data.duplicated().sum() == 0  # No duplicates
        assert len(cleaned.data) > 0  # Some data should remain


class TestDataTransformer:
    """Comprehensive tests for DataTransformer functionality."""

    @pytest.fixture
    def data_transformer(self):
        """Create DataTransformer instance."""
        return DataTransformer()

    @pytest.fixture
    def numeric_dataset(self):
        """Create numeric dataset for transformation testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature_1": np.random.normal(100, 15, 50),
                "feature_2": np.random.exponential(2, 50),
                "feature_3": np.random.uniform(0, 10, 50),
                "target": np.random.choice([0, 1], 50),
            }
        )
        return Dataset(name="test_numeric", data=data, target_column="target")

    @pytest.fixture
    def categorical_dataset(self):
        """Create dataset with categorical features."""
        data = pd.DataFrame(
            {
                "category_1": ["A", "B", "C", "A", "B"] * 10,
                "category_2": ["X", "Y", "X", "Z", "Y"] * 10,
                "numeric_1": np.random.normal(0, 1, 50),
                "target": np.random.choice([0, 1], 50),
            }
        )
        return Dataset(name="test_categorical", data=data, target_column="target")

    def test_scale_features_standard(self, data_transformer, numeric_dataset):
        """Test standard scaling of features."""
        scaled = data_transformer.scale_features(
            numeric_dataset,
            strategy=ScalingStrategy.STANDARD,
            columns=["feature_1", "feature_2"],
        )

        # Check that specified columns are scaled
        for col in ["feature_1", "feature_2"]:
            feature_values = scaled.data[col]
            assert abs(feature_values.mean()) < 1e-10  # Mean ~ 0
            assert abs(feature_values.std() - 1.0) < 1e-10  # Std ~ 1

        # feature_3 should not be scaled
        assert scaled.data["feature_3"].equals(numeric_dataset.data["feature_3"])

    def test_scale_features_minmax(self, data_transformer, numeric_dataset):
        """Test min-max scaling of features."""
        scaled = data_transformer.scale_features(
            numeric_dataset, strategy=ScalingStrategy.MINMAX, feature_range=(0, 1)
        )

        # All numeric features should be scaled to [0, 1]
        for col in ["feature_1", "feature_2", "feature_3"]:
            feature_values = scaled.data[col]
            assert feature_values.min() >= 0
            assert feature_values.max() <= 1

    def test_scale_features_robust(self, data_transformer, numeric_dataset):
        """Test robust scaling of features."""
        scaled = data_transformer.scale_features(
            numeric_dataset, strategy=ScalingStrategy.ROBUST
        )

        # Check that scaling was applied (values should be different)
        for col in ["feature_1", "feature_2", "feature_3"]:
            assert not scaled.data[col].equals(numeric_dataset.data[col])

    def test_encode_categorical_label(self, data_transformer, categorical_dataset):
        """Test label encoding of categorical features."""
        encoded = data_transformer.encode_categorical(
            categorical_dataset,
            strategy=EncodingStrategy.LABEL,
            columns=["category_1", "category_2"],
        )

        # Categorical columns should be converted to numeric
        assert encoded.data["category_1"].dtype in [np.int64, np.int32]
        assert encoded.data["category_2"].dtype in [np.int64, np.int32]

        # Values should be integers starting from 0
        assert encoded.data["category_1"].min() >= 0
        assert encoded.data["category_2"].min() >= 0

    def test_encode_categorical_onehot(self, data_transformer, categorical_dataset):
        """Test one-hot encoding of categorical features."""
        encoded = data_transformer.encode_categorical(
            categorical_dataset,
            strategy=EncodingStrategy.ONEHOT,
            columns=["category_1"],
        )

        # Should create new columns for each category
        category_1_cols = [
            col for col in encoded.data.columns if col.startswith("category_1_")
        ]
        assert len(category_1_cols) == 3  # A, B, C

        # Original categorical column should be removed
        assert "category_1" not in encoded.data.columns

        # Each row should have exactly one 1 in the one-hot columns
        for _, row in encoded.data[category_1_cols].iterrows():
            assert row.sum() == 1

    def test_select_features_variance(self, data_transformer, numeric_dataset):
        """Test variance-based feature selection."""
        # Add a constant feature
        numeric_dataset.data["constant_feature"] = 1.0

        selected = data_transformer.select_features(
            numeric_dataset,
            strategy=FeatureSelectionStrategy.VARIANCE_THRESHOLD,
            threshold=0.01,
        )

        # Constant feature should be removed
        assert "constant_feature" not in selected.data.columns

        # Other features should remain
        for col in ["feature_1", "feature_2", "feature_3"]:
            assert col in selected.data.columns

    def test_select_features_correlation(self, data_transformer, numeric_dataset):
        """Test correlation-based feature selection."""
        # Add highly correlated feature
        numeric_dataset.data["correlated_feature"] = (
            numeric_dataset.data["feature_1"] * 1.01
        )

        selected = data_transformer.select_features(
            numeric_dataset,
            strategy=FeatureSelectionStrategy.CORRELATION_THRESHOLD,
            threshold=0.95,
        )

        # One of the correlated features should be removed
        feature_cols = [
            col
            for col in selected.data.columns
            if col.startswith("feature") or col.startswith("correlated")
        ]
        original_cols = [
            col
            for col in numeric_dataset.data.columns
            if col.startswith("feature") or col.startswith("correlated")
        ]
        assert len(feature_cols) < len(original_cols)

    def test_generate_polynomial_features(self, data_transformer, numeric_dataset):
        """Test polynomial feature generation."""
        poly_dataset = data_transformer.generate_polynomial_features(
            numeric_dataset, degree=2, columns=["feature_1", "feature_2"]
        )

        # Should have original features plus polynomial combinations
        original_count = len(numeric_dataset.data.columns)
        poly_count = len(poly_dataset.data.columns)
        assert poly_count > original_count

        # Check that interaction features exist
        interaction_cols = [col for col in poly_dataset.data.columns if " " in col]
        assert len(interaction_cols) > 0

    def test_convert_data_types_optimize(self, data_transformer):
        """Test data type optimization."""
        data = pd.DataFrame(
            {
                "int_feature": [1, 2, 3, 4, 5],  # Can be int8
                "float_feature": [1.1, 2.2, 3.3, 4.4, 5.5],  # Can be float32
                "categorical": ["A", "B", "A", "B", "A"],  # Can be category
                "target": [0, 1, 0, 1, 0],
            }
        )
        dataset = Dataset(name="test_types", data=data, target_column="target")

        optimized = data_transformer.convert_data_types(dataset, optimize_memory=True)

        # Check that memory usage is reduced
        original_memory = dataset.data.memory_usage(deep=True).sum()
        optimized_memory = optimized.data.memory_usage(deep=True).sum()
        assert optimized_memory <= original_memory

    def test_transform_non_numeric_to_numeric(self, data_transformer):
        """Test conversion of non-numeric columns to numeric."""
        data = pd.DataFrame(
            {
                "string_numbers": ["1", "2", "3", "4", "5"],
                "mixed_types": [1, "2", 3.0, "4", 5],
                "pure_strings": ["A", "B", "C", "D", "E"],
                "target": [0, 1, 0, 1, 0],
            }
        )
        dataset = Dataset(name="test_conversion", data=data, target_column="target")

        converted = data_transformer.transform_non_numeric_to_numeric(dataset)

        # string_numbers should be converted to numeric
        assert pd.api.types.is_numeric_dtype(converted.data["string_numbers"])

        # mixed_types should be converted to numeric
        assert pd.api.types.is_numeric_dtype(converted.data["mixed_types"])

        # pure_strings should be label encoded to numeric
        assert pd.api.types.is_numeric_dtype(converted.data["pure_strings"])


class TestPreprocessingPipeline:
    """Comprehensive tests for PreprocessingPipeline functionality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for pipeline testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.exponential(1, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
                "target": np.random.choice([0, 1], 100),
            }
        )
        # Add some missing values
        data.loc[5:10, "feature_1"] = np.nan

        return Dataset(name="test_pipeline", data=data, target_column="target")

    def test_pipeline_creation_empty(self):
        """Test creating empty preprocessing pipeline."""
        pipeline = PreprocessingPipeline()

        assert len(pipeline.steps) == 0
        assert pipeline.fitted is False
        assert pipeline.metadata == {}

    def test_pipeline_add_step(self):
        """Test adding steps to pipeline."""
        pipeline = PreprocessingPipeline()

        # Add cleaning step
        pipeline.add_step(
            name="clean_missing",
            operation="handle_missing_values",
            parameters={"strategy": "fill_mean"},
            enabled=True,
        )

        # Add transformation step
        pipeline.add_step(
            name="scale_features",
            operation="scale_features",
            parameters={"strategy": "standard"},
            enabled=True,
        )

        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].name == "clean_missing"
        assert pipeline.steps[1].name == "scale_features"

    def test_pipeline_fit_and_transform(self, sample_dataset):
        """Test fitting and transforming data through pipeline."""
        pipeline = PreprocessingPipeline()

        # Add steps
        pipeline.add_step(
            "clean_missing", "handle_missing_values", {"strategy": "fill_mean"}
        )
        pipeline.add_step(
            "encode_categorical",
            "encode_categorical",
            {"strategy": "label", "columns": ["category"]},
        )
        pipeline.add_step("scale_features", "scale_features", {"strategy": "standard"})

        # Fit pipeline
        fitted_pipeline = pipeline.fit(sample_dataset)
        assert fitted_pipeline.fitted is True

        # Transform data
        transformed = fitted_pipeline.transform(sample_dataset)

        # Check transformations applied
        assert transformed.data.isnull().sum().sum() == 0  # No missing values
        assert pd.api.types.is_numeric_dtype(transformed.data["category"])  # Encoded

        # Check scaling (features should have mean ~0, std ~1)
        numeric_cols = ["feature_1", "feature_2"]
        for col in numeric_cols:
            assert abs(transformed.data[col].mean()) < 0.1
            assert abs(transformed.data[col].std() - 1.0) < 0.1

    def test_pipeline_fit_transform_convenience(self, sample_dataset):
        """Test fit_transform convenience method."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step(
            "clean_missing", "handle_missing_values", {"strategy": "fill_mean"}
        )

        transformed = pipeline.fit_transform(sample_dataset)

        assert pipeline.fitted is True
        assert transformed.data.isnull().sum().sum() == 0

    def test_pipeline_step_management(self):
        """Test pipeline step management operations."""
        pipeline = PreprocessingPipeline()

        # Add steps
        pipeline.add_step("step1", "operation1", {})
        pipeline.add_step("step2", "operation2", {})
        pipeline.add_step("step3", "operation3", {})

        # Remove step
        pipeline.remove_step("step2")
        assert len(pipeline.steps) == 2
        assert "step2" not in [step.name for step in pipeline.steps]

        # Disable step
        pipeline.disable_step("step3")
        assert pipeline.steps[1].enabled is False

        # Enable step
        pipeline.enable_step("step3")
        assert pipeline.steps[1].enabled is True

    def test_pipeline_configuration_save_load(self):
        """Test saving and loading pipeline configuration."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step("clean", "handle_missing_values", {"strategy": "fill_mean"})
        pipeline.add_step("scale", "scale_features", {"strategy": "standard"})

        # Save configuration
        config = pipeline.get_configuration()

        assert isinstance(config, PipelineConfig)
        assert len(config.steps) == 2
        assert config.steps[0].name == "clean"

        # Create new pipeline from configuration
        new_pipeline = PreprocessingPipeline.from_configuration(config)

        assert len(new_pipeline.steps) == 2
        assert new_pipeline.steps[0].name == "clean"
        assert new_pipeline.steps[1].name == "scale"

    def test_pipeline_configuration_file_operations(self):
        """Test saving and loading pipeline configuration from files."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step("clean", "handle_missing_values", {"strategy": "fill_mean"})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = Path(f.name)

        try:
            # Save to file
            pipeline.save_configuration(config_path)
            assert config_path.exists()

            # Load from file
            loaded_pipeline = PreprocessingPipeline.load_configuration(config_path)

            assert len(loaded_pipeline.steps) == 1
            assert loaded_pipeline.steps[0].name == "clean"

        finally:
            if config_path.exists():
                config_path.unlink()

    def test_pipeline_preset_basic_cleaning(self, sample_dataset):
        """Test basic cleaning preset pipeline."""
        pipeline = PreprocessingPipeline.create_basic_cleaning_pipeline()

        # Should have standard cleaning steps
        step_names = [step.name for step in pipeline.steps]
        assert "handle_missing" in step_names
        assert "handle_duplicates" in step_names
        assert "handle_outliers" in step_names

        # Should work on sample data
        transformed = pipeline.fit_transform(sample_dataset)
        assert transformed.data.isnull().sum().sum() == 0

    def test_pipeline_preset_anomaly_detection(self, sample_dataset):
        """Test anomaly detection optimized preset pipeline."""
        pipeline = PreprocessingPipeline.create_anomaly_detection_pipeline()

        # Should have anomaly detection specific steps
        step_names = [step.name for step in pipeline.steps]
        assert "handle_missing" in step_names
        assert "scale_features" in step_names
        assert "encode_categorical" in step_names

        # Should work on sample data
        transformed = pipeline.fit_transform(sample_dataset)
        assert transformed.data.isnull().sum().sum() == 0
        assert pd.api.types.is_numeric_dtype(transformed.data["category"])

    def test_pipeline_metadata_tracking(self, sample_dataset):
        """Test metadata tracking throughout pipeline."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step("clean", "handle_missing_values", {"strategy": "fill_mean"})

        pipeline.fit_transform(sample_dataset)

        # Should have metadata about transformations
        assert len(pipeline.metadata) > 0
        assert "transformations_applied" in pipeline.metadata
        assert "original_shape" in pipeline.metadata
        assert "final_shape" in pipeline.metadata

    def test_pipeline_error_handling(self, sample_dataset):
        """Test pipeline error handling with invalid operations."""
        pipeline = PreprocessingPipeline()

        # Add invalid step
        pipeline.add_step("invalid", "nonexistent_operation", {})

        # Should handle gracefully
        with pytest.raises((AttributeError, ValueError)):
            pipeline.fit_transform(sample_dataset)

    def test_pipeline_step_validation(self):
        """Test validation of pipeline steps."""
        pipeline = PreprocessingPipeline()

        # Test adding step with missing parameters
        with pytest.raises(ValueError):
            pipeline.add_step("", "operation", {})  # Empty name

        with pytest.raises(ValueError):
            pipeline.add_step("name", "", {})  # Empty operation

    def test_preprocessing_step_model(self):
        """Test PreprocessingStep model validation."""
        step = PreprocessingStep(
            name="test_step",
            operation="test_operation",
            parameters={"param1": "value1"},
            enabled=True,
        )

        assert step.name == "test_step"
        assert step.operation == "test_operation"
        assert step.parameters == {"param1": "value1"}
        assert step.enabled is True

    def test_pipeline_config_model(self):
        """Test PipelineConfig model validation."""
        steps = [
            PreprocessingStep(name="step1", operation="op1", parameters={}),
            PreprocessingStep(name="step2", operation="op2", parameters={}),
        ]

        config = PipelineConfig(
            name="test_config",
            description="Test configuration",
            steps=steps,
            metadata={"version": "1.0"},
        )

        assert config.name == "test_config"
        assert config.description == "Test configuration"
        assert len(config.steps) == 2
        assert config.metadata["version"] == "1.0"
