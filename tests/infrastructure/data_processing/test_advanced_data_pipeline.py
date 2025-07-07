"""Test cases for advanced data processing pipeline."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.infrastructure.data_processing.advanced_data_pipeline import (
    AdvancedDataPipeline,
    EncodingMethod,
    ImputationStrategy,
    PreprocessingStep,
    ProcessingConfig,
    ProcessingReport,
    ScalingMethod,
    ValidationRule,
)


class TestProcessingConfig:
    """Test cases for ProcessingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProcessingConfig()

        assert config.remove_duplicates is True
        assert config.handle_missing is True
        assert config.imputation_strategy == ImputationStrategy.MEDIAN
        assert config.missing_threshold == 0.5
        assert config.apply_scaling is True
        assert config.scaling_method == ScalingMethod.ROBUST
        assert config.encode_categoricals is True
        assert config.encoding_method == EncodingMethod.ONEHOT
        assert config.max_categories == 10
        assert config.remove_low_variance is True
        assert config.variance_threshold == 0.01
        assert config.validate_data is True
        assert config.parallel_processing is True
        assert config.max_workers == 4
        assert config.memory_efficient is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProcessingConfig(
            remove_duplicates=False,
            imputation_strategy=ImputationStrategy.MEAN,
            scaling_method=ScalingMethod.STANDARD,
            encoding_method=EncodingMethod.LABEL,
            max_categories=20,
            variance_threshold=0.05,
            strict_validation=True,
            max_workers=8,
        )

        assert config.remove_duplicates is False
        assert config.imputation_strategy == ImputationStrategy.MEAN
        assert config.scaling_method == ScalingMethod.STANDARD
        assert config.encoding_method == EncodingMethod.LABEL
        assert config.max_categories == 20
        assert config.variance_threshold == 0.05
        assert config.strict_validation is True
        assert config.max_workers == 8


class TestValidationRule:
    """Test cases for ValidationRule."""

    def test_validation_rule_creation(self):
        """Test validation rule creation."""
        rule = ValidationRule(
            column="age",
            rule_type="range",
            parameters={"min": 0, "max": 120},
            severity="error",
            message="Age must be between 0 and 120",
        )

        assert rule.column == "age"
        assert rule.rule_type == "range"
        assert rule.parameters == {"min": 0, "max": 120}
        assert rule.severity == "error"
        assert rule.message == "Age must be between 0 and 120"

    def test_validation_rule_defaults(self):
        """Test validation rule default values."""
        rule = ValidationRule(column="test", rule_type="type")

        assert rule.parameters == {}
        assert rule.severity == "error"
        assert rule.message is None


class TestProcessingReport:
    """Test cases for ProcessingReport."""

    def test_processing_report_creation(self):
        """Test processing report creation."""
        report = ProcessingReport(
            original_shape=(1000, 10),
            final_shape=(950, 8),
            processing_time=2.5,
            steps_performed=["remove_duplicates", "handle_missing", "scale_features"],
        )

        assert report.original_shape == (1000, 10)
        assert report.final_shape == (950, 8)
        assert report.processing_time == 2.5
        assert len(report.steps_performed) == 3
        assert "remove_duplicates" in report.steps_performed

    def test_report_properties(self):
        """Test report calculated properties."""
        report = ProcessingReport(
            original_shape=(1000, 10),
            final_shape=(950, 8),
            processing_time=2.5,
            steps_performed=["test"],
        )

        assert report.rows_removed == 50
        assert report.features_removed == 2
        assert report.success is True

        # Test with errors
        report.errors = ["Test error"]
        assert report.success is False


class TestAdvancedDataPipeline:
    """Test cases for AdvancedDataPipeline."""

    def test_init_default(self):
        """Test default pipeline initialization."""
        pipeline = AdvancedDataPipeline()

        assert isinstance(pipeline.config, ProcessingConfig)
        assert pipeline._fitted_transformers == {}
        assert pipeline._feature_names is None
        assert pipeline._original_columns is None
        assert pipeline._processing_history == []

    def test_init_custom_config(self):
        """Test pipeline initialization with custom config."""
        config = ProcessingConfig(apply_scaling=False)
        pipeline = AdvancedDataPipeline(config=config)

        assert pipeline.config.apply_scaling is False

    def test_process_dataset_simple(self):
        """Test processing a simple dataset."""
        # Create test dataset
        df = pd.DataFrame(
            {
                "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "numeric2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "categorical": ["A", "B", "A", "C", "B"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        dataset = Dataset(
            name="test_dataset", data=df, target_column="target", metadata={}
        )

        # Simple config to avoid complex processing
        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=False,
            apply_scaling=False,
            encode_categoricals=False,
            remove_low_variance=False,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)

        result, report = pipeline.process_dataset(dataset, return_report=True)

        assert isinstance(result, Dataset)
        assert isinstance(report, ProcessingReport)
        assert result.name == "test_dataset_processed"
        assert report.success is True
        assert len(pipeline._processing_history) == 1

    def test_process_dataset_with_duplicates(self):
        """Test processing dataset with duplicate removal."""
        df = pd.DataFrame(
            {"col1": [1, 2, 2, 3, 3], "col2": [10, 20, 20, 30, 30]}  # Duplicates
        )

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=True,
            handle_missing=False,
            apply_scaling=False,
            encode_categoricals=False,
            remove_low_variance=False,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, report = pipeline.process_dataset(dataset, return_report=True)

        # Should remove 2 duplicate rows
        assert len(result.data) == 3
        assert report.rows_removed == 2
        assert "remove_duplicates" in report.steps_performed

    def test_process_dataset_with_missing_values(self):
        """Test processing dataset with missing value handling."""
        df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, np.nan, 4.0, 5.0],
                "categorical": ["A", "B", None, "C", "B"],
            }
        )

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=True,
            imputation_strategy=ImputationStrategy.MEDIAN,
            apply_scaling=False,
            encode_categoricals=False,
            remove_low_variance=False,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, _ = pipeline.process_dataset(dataset, return_report=False)

        # Should impute missing numeric values and fill categorical
        assert not result.data["numeric"].isnull().any()
        assert not result.data["categorical"].isnull().any()
        assert "Unknown" in result.data["categorical"].values

    def test_process_dataset_with_scaling(self):
        """Test processing dataset with feature scaling."""
        df = pd.DataFrame(
            {"feature1": [1.0, 100.0, 1000.0], "feature2": [0.1, 0.2, 0.3]}
        )

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=False,
            apply_scaling=True,
            scaling_method=ScalingMethod.STANDARD,
            encode_categoricals=False,
            remove_low_variance=False,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, _ = pipeline.process_dataset(dataset, return_report=False)

        # Features should be standardized (mean ~0, std ~1)
        assert abs(result.data["feature1"].mean()) < 0.1
        assert abs(result.data["feature1"].std() - 1.0) < 0.1
        assert "scaler" in pipeline._fitted_transformers

    def test_process_dataset_with_categorical_encoding(self):
        """Test processing dataset with categorical encoding."""
        df = pd.DataFrame({"numeric": [1, 2, 3, 4], "category": ["A", "B", "A", "C"]})

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=False,
            apply_scaling=False,
            encode_categoricals=True,
            encoding_method=EncodingMethod.ONEHOT,
            remove_low_variance=False,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, _ = pipeline.process_dataset(dataset, return_report=False)

        # Should have one-hot encoded columns
        expected_columns = ["numeric", "category_A", "category_B", "category_C"]
        assert all(col in result.data.columns for col in expected_columns)
        assert "category" not in result.data.columns  # Original should be removed

    def test_process_dataset_with_low_variance_removal(self):
        """Test processing dataset with low variance feature removal."""
        df = pd.DataFrame(
            {
                "good_feature": [1, 2, 3, 4, 5],
                "low_variance": [1.0, 1.0, 1.0, 1.0, 1.0],  # No variance
                "some_variance": [1.0, 1.001, 1.0, 1.001, 1.0],  # Very low variance
            }
        )

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=False,
            apply_scaling=False,
            encode_categoricals=False,
            remove_low_variance=True,
            variance_threshold=0.01,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, report = pipeline.process_dataset(dataset, return_report=True)

        # Should remove low variance features
        assert "good_feature" in result.data.columns
        assert "low_variance" not in result.data.columns
        assert report.features_removed > 0

    def test_transform_new_data(self):
        """Test transforming new data with fitted transformers."""
        # Train on original data
        train_df = pd.DataFrame(
            {"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [10.0, 20.0, 30.0, 40.0]}
        )

        train_dataset = Dataset(name="train", data=train_df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=True,
            apply_scaling=True,
            scaling_method=ScalingMethod.STANDARD,
            encode_categoricals=False,
            remove_low_variance=False,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        pipeline.process_dataset(train_dataset, fit_transformers=True)

        # Transform new data
        new_df = pd.DataFrame({"feature1": [5.0, 6.0], "feature2": [50.0, 60.0]})

        result = pipeline.transform_new_data(new_df)

        assert len(result) == 2
        assert list(result.columns) == ["feature1", "feature2"]
        # Should use same scaling as training data
        assert "scaler" in pipeline._fitted_transformers

    def test_transform_new_data_no_transformers(self):
        """Test error when transforming without fitted transformers."""
        pipeline = AdvancedDataPipeline()

        df = pd.DataFrame({"col1": [1, 2, 3]})

        with pytest.raises(ValueError, match="No fitted transformers available"):
            pipeline.transform_new_data(df)

    def test_validate_data_success(self):
        """Test successful data validation."""
        df = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [10.0, 20.0, 30.0, 40.0, 50.0]}
        )

        pipeline = AdvancedDataPipeline()

        is_valid, warnings, errors = pipeline.validate_data(df)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_data_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()

        pipeline = AdvancedDataPipeline()

        is_valid, warnings, errors = pipeline.validate_data(df)

        assert is_valid is False
        assert "Dataset is empty" in errors

    def test_validate_data_with_rules(self):
        """Test validation with custom rules."""
        df = pd.DataFrame(
            {
                "age": [25, 30, 150, 35],  # One invalid age
                "score": [0.5, 0.8, 0.3, 1.2],  # One invalid score
            }
        )

        rules = [
            ValidationRule(
                column="age",
                rule_type="range",
                parameters={"min": 0, "max": 120},
                severity="error",
            ),
            ValidationRule(
                column="score",
                rule_type="range",
                parameters={"min": 0, "max": 1},
                severity="warning",
            ),
        ]

        pipeline = AdvancedDataPipeline()

        is_valid, warnings, errors = pipeline.validate_data(df, rules)

        assert is_valid is False
        assert len(errors) >= 1  # Age validation error
        assert len(warnings) >= 1  # Score validation warning

    def test_validate_data_null_percentage_rule(self):
        """Test validation with null percentage rule."""
        df = pd.DataFrame(
            {
                "mostly_null": [1, None, None, None, None],  # 80% null
                "some_null": [1, 2, None, 4, 5],  # 20% null
            }
        )

        rules = [
            ValidationRule(
                column="mostly_null",
                rule_type="null_percentage",
                parameters={"max_percentage": 0.5},
                severity="error",
            )
        ]

        pipeline = AdvancedDataPipeline()

        is_valid, warnings, errors = pipeline.validate_data(df, rules)

        assert is_valid is False
        assert any("null values" in error for error in errors)

    def test_validate_data_type_rule(self):
        """Test validation with data type rule."""
        df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.0, 2.0, 3.0]})

        rules = [
            ValidationRule(
                column="int_col",
                rule_type="type",
                parameters={"dtype": "float64"},  # Expecting float but got int
                severity="warning",
            )
        ]

        pipeline = AdvancedDataPipeline()

        is_valid, warnings, errors = pipeline.validate_data(df, rules)

        assert is_valid is True  # Only warning, not error
        assert len(warnings) >= 1
        assert any("expected float64" in warning for warning in warnings)

    def test_validate_data_missing_column(self):
        """Test validation with missing column."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        rules = [
            ValidationRule(
                column="missing_col",
                rule_type="range",
                parameters={"min": 0, "max": 10},
                severity="error",
            )
        ]

        pipeline = AdvancedDataPipeline()

        is_valid, warnings, errors = pipeline.validate_data(df, rules)

        assert is_valid is False
        assert any("not found" in error for error in errors)

    def test_strict_validation_failure(self):
        """Test strict validation failure."""
        df = pd.DataFrame()  # Empty DataFrame

        config = ProcessingConfig(validate_data=True, strict_validation=True)

        dataset = Dataset(name="test", data=df, metadata={})
        pipeline = AdvancedDataPipeline(config=config)

        with pytest.raises(DataValidationError, match="Data validation failed"):
            pipeline.process_dataset(dataset)

    def test_get_processing_report(self):
        """Test getting processing report."""
        pipeline = AdvancedDataPipeline()

        # No reports initially
        assert pipeline.get_processing_report() is None

        # Process a dataset
        df = pd.DataFrame({"col1": [1, 2, 3]})
        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(validate_data=False)
        pipeline.config = config

        pipeline.process_dataset(dataset)

        report = pipeline.get_processing_report()
        assert isinstance(report, ProcessingReport)

    def test_get_feature_info(self):
        """Test getting feature information."""
        pipeline = AdvancedDataPipeline()

        # No features initially
        assert pipeline.get_feature_info() == {}

        # Process a dataset
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=False,
            apply_scaling=False,
            encode_categoricals=False,
            remove_low_variance=False,
            validate_data=False,
        )
        pipeline.config = config

        pipeline.process_dataset(dataset)

        info = pipeline.get_feature_info()
        assert info["n_original"] == 2
        assert info["n_processed"] == 2
        assert "original_features" in info
        assert "processed_features" in info

    def test_reset_transformers(self):
        """Test resetting fitted transformers."""
        pipeline = AdvancedDataPipeline()

        # Add some mock transformers
        pipeline._fitted_transformers = {"scaler": Mock(), "imputer": Mock()}
        pipeline._feature_names = ["feature1", "feature2"]
        pipeline._original_columns = ["original1", "original2"]

        pipeline.reset_transformers()

        assert pipeline._fitted_transformers == {}
        assert pipeline._feature_names is None
        assert pipeline._original_columns is None

    def test_memory_efficient_dtype_optimization(self):
        """Test memory efficient data type optimization."""
        df = pd.DataFrame(
            {
                "small_int": np.array([1, 2, 3, 4, 5], dtype=np.int64),
                "large_int": np.array([1000000, 2000000, 3000000], dtype=np.int64),
                "float_col": np.array([1.1, 2.2, 3.3], dtype=np.float64),
                "category_col": ["A", "B", "A", "B", "A"],
            }
        )

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=False,
            apply_scaling=False,
            encode_categoricals=False,
            remove_low_variance=False,
            validate_data=False,
            memory_efficient=True,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, _ = pipeline.process_dataset(dataset, return_report=False)

        # Small integers should be optimized to smaller types
        # Note: Exact dtype depends on the values and optimization logic
        assert result.data["small_int"].dtype != np.int64  # Should be optimized
        assert "optimize_dtypes" in pipeline.get_processing_report().steps_performed

    def test_knn_imputation(self):
        """Test KNN imputation strategy."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, np.nan, 50.0],
                "feature3": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=True,
            imputation_strategy=ImputationStrategy.KNN,
            apply_scaling=False,
            encode_categoricals=False,
            remove_low_variance=False,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, _ = pipeline.process_dataset(dataset, return_report=False)

        # Should impute missing values using KNN
        assert not result.data.isnull().any().any()
        assert isinstance(pipeline._fitted_transformers["imputer"], KNNImputer)

    def test_label_encoding(self):
        """Test label encoding for categorical variables."""
        df = pd.DataFrame({"category": ["A", "B", "C", "A", "B"]})

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=False,
            apply_scaling=False,
            encode_categoricals=True,
            encoding_method=EncodingMethod.LABEL,
            remove_low_variance=False,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, _ = pipeline.process_dataset(dataset, return_report=False)

        # Should have numeric encoded values
        assert result.data["category"].dtype in [np.int64, np.int32, np.float64]
        assert "encoders" in pipeline._fitted_transformers

    def test_high_cardinality_categorical_skip(self):
        """Test skipping high cardinality categorical variables."""
        # Create categorical with many unique values
        categories = [f"cat_{i}" for i in range(20)]
        df = pd.DataFrame(
            {"high_cardinality": categories, "normal_cat": ["A", "B"] * 10}
        )

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=False,
            apply_scaling=False,
            encode_categoricals=True,
            encoding_method=EncodingMethod.ONEHOT,
            max_categories=10,  # Lower than unique values in high_cardinality
            remove_low_variance=False,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, _ = pipeline.process_dataset(dataset, return_report=False)

        # High cardinality column should remain unchanged
        assert "high_cardinality" in result.data.columns
        # Normal categorical should be encoded
        assert (
            "normal_cat_A" in result.data.columns
            or "normal_cat" not in result.data.columns
        )

    def test_feature_selection_with_target(self):
        """Test feature selection when target is available."""
        df = pd.DataFrame(
            {
                "good_feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "noise1": np.random.random(10),
                "noise2": np.random.random(10),
                "noise3": np.random.random(10),
                "noise4": np.random.random(10),
            }
        )

        # Create target correlated with good_feature
        target = (df["good_feature"] > 5).astype(int)

        dataset = Dataset(name="test", data=df, target_column="target", metadata={})
        dataset._target = target  # Set target directly for testing

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=False,
            apply_scaling=False,
            encode_categoricals=False,
            remove_low_variance=False,
            apply_feature_selection=True,
            max_features=2,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, report = pipeline.process_dataset(dataset, return_report=True)

        # Should select top 2 features
        assert len(result.data.columns) == 2
        assert "feature_selection" in report.steps_performed
        assert "feature_selector" in pipeline._fitted_transformers

    def test_parallel_processing_config(self):
        """Test that parallel processing config is handled."""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]})

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            parallel_processing=True, max_workers=2, validate_data=False
        )

        pipeline = AdvancedDataPipeline(config=config)

        # Should process without error (parallel processing is mainly for future use)
        result, _ = pipeline.process_dataset(dataset, return_report=False)
        assert len(result.data) == 5

    def test_processing_error_handling(self):
        """Test error handling during processing."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        dataset = Dataset(name="test", data=df, metadata={})

        pipeline = AdvancedDataPipeline()

        # Mock a processing step to raise an error
        with patch.object(
            pipeline, "_step_validate_data", side_effect=Exception("Test error")
        ):
            with pytest.raises(DataValidationError, match="Data processing failed"):
                pipeline.process_dataset(dataset)

    def test_ordinal_encoding(self):
        """Test ordinal encoding for categorical variables."""
        df = pd.DataFrame({"ordinal_cat": ["low", "medium", "high", "low", "high"]})

        dataset = Dataset(name="test", data=df, metadata={})

        config = ProcessingConfig(
            remove_duplicates=False,
            handle_missing=False,
            apply_scaling=False,
            encode_categoricals=True,
            encoding_method=EncodingMethod.ORDINAL,
            remove_low_variance=False,
            validate_data=False,
        )

        pipeline = AdvancedDataPipeline(config=config)
        result, _ = pipeline.process_dataset(dataset, return_report=False)

        # Should have numeric encoded values
        assert result.data["ordinal_cat"].dtype in [np.int64, np.int32, np.float64]
        assert "encoders" in pipeline._fitted_transformers
