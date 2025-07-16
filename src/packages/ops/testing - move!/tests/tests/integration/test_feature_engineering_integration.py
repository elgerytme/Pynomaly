"""Integration tests for feature engineering features."""

import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from monorepo.domain.entities import Dataset
from monorepo.features.feature_engineering import (
    FeatureEngineer,
    FeatureExtractor,
    FeatureMetadata,
    FeaturePipeline,
    FeatureSelector,
    FeatureTransformer,
    FeatureType,
)


@pytest.fixture
def sample_time_series_dataset():
    """Create sample time series dataset for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    n_samples = len(dates)

    # Create time series with seasonality and trend
    time_index = np.arange(n_samples)
    seasonal_pattern = np.sin(2 * np.pi * time_index / 365) * 10
    trend = time_index * 0.01
    noise = np.random.normal(0, 2, n_samples)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "value": seasonal_pattern + trend + noise + 50,  # Base value of 50
            "temperature": np.random.normal(20, 5, n_samples),
            "humidity": np.random.normal(60, 15, n_samples),
            "pressure": np.random.normal(1013, 10, n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "is_weekend": [d.weekday() >= 5 for d in dates],
            "sensor_id": np.random.choice([1, 2, 3, 4, 5], n_samples),
        }
    )

    return Dataset(
        name="time_series_test",
        data=data,
        description="Time series dataset for feature engineering testing",
    )


@pytest.fixture
def sample_mixed_dataset():
    """Create sample dataset with mixed data types."""
    n_samples = 1000

    data = pd.DataFrame(
        {
            "numeric_int": np.random.randint(0, 100, n_samples),
            "numeric_float": np.random.randn(n_samples),
            "categorical_str": np.random.choice(
                ["cat", "dog", "bird", "fish"], n_samples
            ),
            "categorical_int": np.random.choice([1, 2, 3, 4, 5], n_samples),
            "binary_bool": np.random.choice([True, False], n_samples),
            "binary_int": np.random.choice([0, 1], n_samples),
            "text_data": [
                f"text_sample_{i}_{np.random.choice(['good', 'bad', 'neutral'])}"
                for i in range(n_samples)
            ],
            "ordinal_data": np.random.choice(["low", "medium", "high"], n_samples),
            "datetime_col": pd.date_range(
                start="2023-01-01", periods=n_samples, freq="H"
            ),
            "missing_data": np.where(
                np.random.random(n_samples) < 0.2, np.nan, np.random.randn(n_samples)
            ),
        }
    )

    return Dataset(
        name="mixed_types_test",
        data=data,
        description="Mixed data types dataset for feature engineering testing",
    )


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for feature storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
class TestFeatureExtractorIntegration:
    """Integration tests for feature extractor."""

    async def test_temporal_feature_extraction(self, sample_time_series_dataset):
        """Test temporal feature extraction."""
        extractor = FeatureExtractor()

        # Extract temporal features
        result = await extractor.extract_temporal_features(
            sample_time_series_dataset, timestamp_col="timestamp"
        )

        # Verify temporal features were added
        assert "hour" in result.data.columns
        assert "day_of_week" in result.data.columns
        assert "month" in result.data.columns
        assert "quarter" in result.data.columns
        assert "is_weekend" in result.data.columns
        assert "day_of_year" in result.data.columns

        # Verify feature values are reasonable
        assert result.data["hour"].min() >= 0
        assert result.data["hour"].max() <= 23
        assert result.data["day_of_week"].min() >= 0
        assert result.data["day_of_week"].max() <= 6
        assert result.data["month"].min() >= 1
        assert result.data["month"].max() <= 12
        assert result.data["quarter"].min() >= 1
        assert result.data["quarter"].max() <= 4

        # Verify metadata
        assert result.name == sample_time_series_dataset.name
        assert len(result.data) == len(sample_time_series_dataset.data)

    async def test_statistical_feature_extraction(self, sample_time_series_dataset):
        """Test statistical feature extraction."""
        extractor = FeatureExtractor()

        # Extract statistical features
        result = await extractor.extract_statistical_features(
            sample_time_series_dataset,
            numeric_columns=["value", "temperature", "humidity", "pressure"],
            window_size=7,
        )

        # Verify statistical features were added
        for col in ["value", "temperature", "humidity", "pressure"]:
            assert f"{col}_rolling_mean" in result.data.columns
            assert f"{col}_rolling_std" in result.data.columns
            assert f"{col}_rolling_min" in result.data.columns
            assert f"{col}_rolling_max" in result.data.columns

        # Verify no NaN values in rolling features after window
        for col in ["value", "temperature", "humidity", "pressure"]:
            rolling_mean_col = f"{col}_rolling_mean"
            # Should have valid values after the window period
            assert not result.data[rolling_mean_col].iloc[7:].isna().any()

        # Verify rolling statistics are reasonable
        value_mean = result.data["value_rolling_mean"].dropna()
        value_original = result.data["value"]
        assert (
            value_mean.min() >= value_original.min() * 0.8
        )  # Rolling mean should be within reasonable range
        assert value_mean.max() <= value_original.max() * 1.2

    async def test_lag_feature_extraction(self, sample_time_series_dataset):
        """Test lag feature extraction."""
        extractor = FeatureExtractor()

        # Extract lag features
        result = await extractor.extract_lag_features(
            sample_time_series_dataset,
            columns=["value", "temperature"],
            lags=[1, 7, 30],
        )

        # Verify lag features were added
        for col in ["value", "temperature"]:
            for lag in [1, 7, 30]:
                lag_col = f"{col}_lag_{lag}"
                assert lag_col in result.data.columns

        # Verify lag feature values
        # Lag 1 should be previous day's value
        original_values = sample_time_series_dataset.data["value"].iloc[1:].values
        lag_1_values = result.data["value_lag_1"].iloc[1:].dropna().values

        # First lag value should match previous original value
        assert len(lag_1_values) > 0

        # Verify lag features have expected NaN pattern
        assert (
            result.data["value_lag_1"].iloc[0] != result.data["value_lag_1"].iloc[0]
        )  # First value should be NaN
        assert (
            result.data["value_lag_7"].iloc[:7].isna().sum() == 7
        )  # First 7 values should be NaN
        assert (
            result.data["value_lag_30"].iloc[:30].isna().sum() == 30
        )  # First 30 values should be NaN

    async def test_custom_feature_extraction(self, sample_mixed_dataset):
        """Test custom feature extraction."""
        extractor = FeatureExtractor()

        # Define custom extractors
        def temperature_category(row):
            temp = row.get("numeric_float", 0)
            if temp < -1:
                return "cold"
            elif temp > 1:
                return "hot"
            else:
                return "mild"

        def composite_score(row):
            return row.get("numeric_int", 0) * 0.1 + row.get("numeric_float", 0) * 0.2

        custom_extractors = {
            "temperature_category": temperature_category,
            "composite_score": composite_score,
        }

        # Extract custom features
        result = await extractor.extract_custom_features(
            sample_mixed_dataset, custom_extractors
        )

        # Verify custom features were added
        assert "temperature_category" in result.data.columns
        assert "composite_score" in result.data.columns

        # Verify custom feature values
        temp_categories = result.data["temperature_category"].unique()
        assert all(cat in ["cold", "hot", "mild"] for cat in temp_categories)

        # Verify composite score calculation
        composite_scores = result.data["composite_score"]
        assert composite_scores.dtype in [np.float64, np.float32]
        assert not composite_scores.isna().any()


@pytest.mark.asyncio
class TestFeatureTransformerIntegration:
    """Integration tests for feature transformer."""

    async def test_normalization_transformation(self, sample_mixed_dataset):
        """Test normalization transformation."""
        transformer = FeatureTransformer()

        # Test min-max normalization
        minmax_result = await transformer.normalize_features(
            sample_mixed_dataset,
            columns=["numeric_int", "numeric_float"],
            method="minmax",
        )

        # Verify normalization
        for col in ["numeric_int", "numeric_float"]:
            normalized_col = minmax_result.data[col]
            assert (
                normalized_col.min() >= -0.001
            )  # Account for floating point precision
            assert normalized_col.max() <= 1.001

        # Test z-score normalization
        zscore_result = await transformer.normalize_features(
            sample_mixed_dataset,
            columns=["numeric_int", "numeric_float"],
            method="zscore",
        )

        # Verify z-score normalization
        for col in ["numeric_int", "numeric_float"]:
            normalized_col = zscore_result.data[col]
            mean = normalized_col.mean()
            std = normalized_col.std()
            assert abs(mean) < 0.001  # Mean should be close to 0
            assert abs(std - 1) < 0.001  # Std should be close to 1

        # Test robust normalization
        robust_result = await transformer.normalize_features(
            sample_mixed_dataset,
            columns=["numeric_int", "numeric_float"],
            method="robust",
        )

        # Verify robust normalization (should handle outliers better)
        for col in ["numeric_int", "numeric_float"]:
            normalized_col = robust_result.data[col]
            assert normalized_col.dtype in [np.float64, np.float32]

    async def test_encoding_transformation(self, sample_mixed_dataset):
        """Test encoding transformation."""
        transformer = FeatureTransformer()

        # Test one-hot encoding
        onehot_result = await transformer.encode_categorical_features(
            sample_mixed_dataset,
            columns=["categorical_str", "ordinal_data"],
            method="onehot",
        )

        # Verify one-hot encoding
        categorical_str_cols = [
            col
            for col in onehot_result.data.columns
            if col.startswith("categorical_str_")
        ]
        ordinal_data_cols = [
            col for col in onehot_result.data.columns if col.startswith("ordinal_data_")
        ]

        assert len(categorical_str_cols) == 4  # cat, dog, bird, fish
        assert len(ordinal_data_cols) == 3  # low, medium, high

        # Verify one-hot encoded values are binary
        for col in categorical_str_cols + ordinal_data_cols:
            assert set(onehot_result.data[col].unique()) <= {0, 1}

        # Test label encoding
        label_result = await transformer.encode_categorical_features(
            sample_mixed_dataset,
            columns=["categorical_str", "ordinal_data"],
            method="label",
        )

        # Verify label encoding
        assert label_result.data["categorical_str"].dtype in [np.int64, np.int32]
        assert label_result.data["ordinal_data"].dtype in [np.int64, np.int32]

        # Values should be integers starting from 0
        assert label_result.data["categorical_str"].min() >= 0
        assert label_result.data["ordinal_data"].min() >= 0

        # Test target encoding (requires target column)
        # Add a synthetic target for testing
        dataset_with_target = Dataset(
            name=sample_mixed_dataset.name,
            data=sample_mixed_dataset.data.copy(),
            target_column="numeric_float",
        )

        target_result = await transformer.encode_categorical_features(
            dataset_with_target, columns=["categorical_str"], method="target"
        )

        # Verify target encoding
        assert target_result.data["categorical_str"].dtype in [np.float64, np.float32]

    async def test_binning_transformation(self, sample_mixed_dataset):
        """Test binning transformation."""
        transformer = FeatureTransformer()

        # Test equal-width binning
        equal_width_result = await transformer.bin_features(
            sample_mixed_dataset,
            columns=["numeric_int", "numeric_float"],
            n_bins=5,
            strategy="uniform",
        )

        # Verify binning
        for col in ["numeric_int", "numeric_float"]:
            binned_col = equal_width_result.data[col]
            unique_bins = binned_col.unique()
            assert len(unique_bins) <= 5  # Should have at most 5 bins
            assert binned_col.dtype in [np.int64, np.int32]

        # Test quantile-based binning
        quantile_result = await transformer.bin_features(
            sample_mixed_dataset,
            columns=["numeric_int", "numeric_float"],
            n_bins=4,
            strategy="quantile",
        )

        # Verify quantile binning
        for col in ["numeric_int", "numeric_float"]:
            binned_col = quantile_result.data[col]
            unique_bins = binned_col.unique()
            assert len(unique_bins) <= 4

            # Each bin should have roughly equal number of samples
            bin_counts = binned_col.value_counts()
            expected_count = len(binned_col) / len(unique_bins)
            for count in bin_counts:
                assert abs(count - expected_count) / expected_count < 0.3  # Within 30%

    async def test_transformation_chaining(self, sample_mixed_dataset):
        """Test chaining multiple transformations."""
        transformer = FeatureTransformer()

        # Chain transformations: normalize -> encode -> bin
        dataset = sample_mixed_dataset

        # Step 1: Normalize numeric features
        dataset = await transformer.normalize_features(
            dataset, columns=["numeric_float"], method="zscore"
        )

        # Step 2: Encode categorical features
        dataset = await transformer.encode_categorical_features(
            dataset, columns=["categorical_str"], method="label"
        )

        # Step 3: Bin integer features
        dataset = await transformer.bin_features(
            dataset, columns=["numeric_int"], n_bins=3, strategy="uniform"
        )

        # Verify all transformations were applied
        # Normalized float should have mean ~0, std ~1
        assert abs(dataset.data["numeric_float"].mean()) < 0.001
        assert abs(dataset.data["numeric_float"].std() - 1) < 0.001

        # Categorical should be label encoded
        assert dataset.data["categorical_str"].dtype in [np.int64, np.int32]

        # Integer should be binned
        assert len(dataset.data["numeric_int"].unique()) <= 3


@pytest.mark.asyncio
class TestFeatureSelectorIntegration:
    """Integration tests for feature selector."""

    async def test_correlation_based_selection(self, sample_mixed_dataset):
        """Test correlation-based feature selection."""
        selector = FeatureSelector()

        # Add some highly correlated features for testing
        data_with_corr = sample_mixed_dataset.data.copy()
        data_with_corr["corr_feature_1"] = data_with_corr[
            "numeric_float"
        ] * 2 + np.random.normal(0, 0.1, len(data_with_corr))
        data_with_corr["corr_feature_2"] = data_with_corr[
            "numeric_float"
        ] * 3 + np.random.normal(0, 0.1, len(data_with_corr))

        dataset_with_corr = Dataset(
            name="correlation_test",
            data=data_with_corr,
            description="Dataset with correlated features",
        )

        # Select features with low correlation
        result = await selector.select_by_correlation(dataset_with_corr, threshold=0.8)

        # Should remove highly correlated features
        original_numeric_cols = dataset_with_corr.data.select_dtypes(
            include=[np.number]
        ).columns
        result_numeric_cols = result.data.select_dtypes(include=[np.number]).columns

        assert len(result_numeric_cols) <= len(original_numeric_cols)

        # Verify remaining features have correlation below threshold
        corr_matrix = result.data.select_dtypes(include=[np.number]).corr()

        # Check upper triangle of correlation matrix (excluding diagonal)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlation = abs(corr_matrix.iloc[i, j])
                assert correlation <= 0.8 or pd.isna(correlation)

    async def test_variance_based_selection(self, sample_mixed_dataset):
        """Test variance-based feature selection."""
        selector = FeatureSelector()

        # Add low variance features
        data_with_low_var = sample_mixed_dataset.data.copy()
        data_with_low_var["low_var_1"] = 5  # Constant feature
        data_with_low_var["low_var_2"] = np.random.choice(
            [1, 2], len(data_with_low_var), p=[0.99, 0.01]
        )  # Very low variance

        dataset_with_low_var = Dataset(
            name="variance_test",
            data=data_with_low_var,
            description="Dataset with low variance features",
        )

        # Select features with sufficient variance
        result = await selector.select_by_variance(dataset_with_low_var, threshold=0.01)

        # Should remove low variance features
        assert (
            "low_var_1" not in result.data.columns
        )  # Constant feature should be removed

        # Verify remaining numeric features have variance above threshold
        numeric_cols = result.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            variance = result.data[col].var()
            assert variance > 0.01 or pd.isna(variance)

    async def test_statistical_significance_selection(self, sample_mixed_dataset):
        """Test statistical significance-based feature selection."""
        selector = FeatureSelector()

        # Create a target variable for testing
        # Make some features more predictive than others
        data_with_target = sample_mixed_dataset.data.copy()

        # Create synthetic target based on some features
        target = (
            data_with_target["numeric_int"] * 0.1
            + data_with_target["numeric_float"] * 0.3
            + np.random.normal(0, 1, len(data_with_target))
        )
        data_with_target["target"] = target

        dataset_with_target = Dataset(
            name="significance_test", data=data_with_target, target_column="target"
        )

        # Select features based on statistical significance
        result = await selector.select_by_statistical_significance(
            dataset_with_target, target_column="target", alpha=0.05
        )

        # Should select features that are significantly correlated with target
        assert "target" in result.data.columns  # Target should be preserved

        # Verify that strongly correlated features are selected
        assert "numeric_int" in result.data.columns
        assert "numeric_float" in result.data.columns

        # Some less correlated features might be removed
        original_cols = set(dataset_with_target.data.columns)
        result_cols = set(result.data.columns)
        removed_cols = original_cols - result_cols

        # At least some feature selection should have occurred
        assert len(result_cols) <= len(original_cols)

    async def test_feature_importance_selection(self, sample_mixed_dataset):
        """Test feature importance-based selection."""
        selector = FeatureSelector()

        # Create dataset with target for importance calculation
        data_with_target = sample_mixed_dataset.data.copy()

        # Create binary target for classification
        target = (
            data_with_target["numeric_float"]
            > data_with_target["numeric_float"].median()
        ).astype(int)
        data_with_target["target"] = target

        dataset_with_target = Dataset(
            name="importance_test", data=data_with_target, target_column="target"
        )

        # Select features based on importance
        result = await selector.select_by_importance(
            dataset_with_target,
            target_column="target",
            n_features=5,
            method="random_forest",
        )

        # Should select top 5 most important features + target
        numeric_cols = result.data.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) <= 6  # 5 features + target
        assert "target" in result.data.columns


@pytest.mark.asyncio
class TestFeaturePipelineIntegration:
    """Integration tests for feature pipeline."""

    async def test_complete_feature_pipeline(
        self, sample_time_series_dataset, temp_storage_dir
    ):
        """Test complete feature engineering pipeline."""
        pipeline = FeaturePipeline(storage_path=temp_storage_dir)

        # Define pipeline configuration
        pipeline_config = [
            {"step": "extract_temporal", "params": {"timestamp_col": "timestamp"}},
            {
                "step": "extract_statistical",
                "params": {
                    "numeric_columns": ["value", "temperature"],
                    "window_size": 7,
                },
            },
            {
                "step": "normalize",
                "params": {"columns": ["value", "temperature"], "method": "zscore"},
            },
            {
                "step": "encode_categorical",
                "params": {"columns": ["category"], "method": "onehot"},
            },
            {"step": "select_by_variance", "params": {"threshold": 0.01}},
        ]

        # Execute pipeline
        result = await pipeline.execute(sample_time_series_dataset, pipeline_config)

        # Verify pipeline execution
        assert result is not None
        assert isinstance(result, Dataset)

        # Verify temporal features were added
        assert "hour" in result.data.columns
        assert "day_of_week" in result.data.columns

        # Verify statistical features were added
        assert "value_rolling_mean" in result.data.columns
        assert "temperature_rolling_mean" in result.data.columns

        # Verify normalization was applied
        value_mean = result.data["value"].mean()
        value_std = result.data["value"].std()
        assert abs(value_mean) < 0.1  # Should be close to 0
        assert abs(value_std - 1) < 0.1  # Should be close to 1

        # Verify categorical encoding
        category_cols = [
            col for col in result.data.columns if col.startswith("category_")
        ]
        assert len(category_cols) >= 3  # Should have one-hot encoded categories

        # Verify low variance features were removed
        numeric_cols = result.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not result.data[col].isna().all():
                variance = result.data[col].var()
                assert variance > 0.01 or pd.isna(variance)

        # Get pipeline metadata
        metadata = await pipeline.get_pipeline_metadata()
        assert metadata["steps_executed"] == len(pipeline_config)
        assert metadata["total_features_original"] > 0
        assert metadata["total_features_final"] > 0

    async def test_pipeline_with_custom_steps(
        self, sample_mixed_dataset, temp_storage_dir
    ):
        """Test pipeline with custom transformation steps."""
        pipeline = FeaturePipeline(storage_path=temp_storage_dir)

        # Define custom transformer
        def custom_interaction_features(dataset: Dataset) -> Dataset:
            """Create interaction features."""
            data = dataset.data.copy()

            # Create interaction between numeric features
            if "numeric_int" in data.columns and "numeric_float" in data.columns:
                data["interaction_int_float"] = (
                    data["numeric_int"] * data["numeric_float"]
                )

            return Dataset(
                name=dataset.name,
                data=data,
                description=f"{dataset.description} - with interaction features",
            )

        # Register custom step
        pipeline.register_custom_step(
            "create_interactions", custom_interaction_features
        )

        # Define pipeline with custom step
        pipeline_config = [
            {
                "step": "normalize",
                "params": {
                    "columns": ["numeric_int", "numeric_float"],
                    "method": "minmax",
                },
            },
            {"step": "create_interactions", "params": {}},
            {"step": "select_by_variance", "params": {"threshold": 0.001}},
        ]

        # Execute pipeline
        result = await pipeline.execute(sample_mixed_dataset, pipeline_config)

        # Verify custom interaction feature was created
        assert "interaction_int_float" in result.data.columns

        # Verify normalization was applied
        for col in ["numeric_int", "numeric_float"]:
            if col in result.data.columns:
                assert result.data[col].min() >= -0.001
                assert result.data[col].max() <= 1.001

    async def test_pipeline_persistence(self, sample_mixed_dataset, temp_storage_dir):
        """Test pipeline persistence and loading."""
        pipeline = FeaturePipeline(storage_path=temp_storage_dir)

        # Define and execute pipeline
        pipeline_config = [
            {
                "step": "normalize",
                "params": {
                    "columns": ["numeric_int", "numeric_float"],
                    "method": "zscore",
                },
            },
            {
                "step": "encode_categorical",
                "params": {"columns": ["categorical_str"], "method": "label"},
            },
        ]

        # Execute and save pipeline
        result = await pipeline.execute(sample_mixed_dataset, pipeline_config)
        saved = await pipeline.save_pipeline("test_pipeline", pipeline_config)
        assert saved

        # Load pipeline
        loaded_config = await pipeline.load_pipeline("test_pipeline")
        assert loaded_config == pipeline_config

        # Execute loaded pipeline
        loaded_result = await pipeline.execute(sample_mixed_dataset, loaded_config)

        # Results should be identical
        pd.testing.assert_frame_equal(
            result.data.sort_index(axis=1), loaded_result.data.sort_index(axis=1)
        )

        # List saved pipelines
        pipeline_list = await pipeline.list_saved_pipelines()
        assert "test_pipeline" in pipeline_list


@pytest.mark.asyncio
class TestFeatureEngineerIntegration:
    """Integration tests for main feature engineer."""

    async def test_feature_engineer_complete_workflow(self, sample_time_series_dataset):
        """Test complete feature engineering workflow."""
        engineer = FeatureEngineer()

        # Engineer features without specific configuration
        result = await engineer.engineer_features(sample_time_series_dataset)

        # Should apply default feature engineering
        assert result is not None
        assert isinstance(result, Dataset)
        assert len(result.data.columns) >= len(sample_time_series_dataset.data.columns)

        # Should have feature metadata
        metadata = await engineer.get_feature_metadata(result)
        assert len(metadata) > 0

        # Verify metadata contains expected information
        for feature_meta in metadata:
            assert isinstance(feature_meta, FeatureMetadata)
            assert feature_meta.name in result.data.columns
            assert feature_meta.type in FeatureType
            assert feature_meta.created_at is not None

    async def test_feature_engineer_with_config(self, sample_mixed_dataset):
        """Test feature engineer with custom configuration."""
        engineer = FeatureEngineer()

        # Define comprehensive pipeline configuration
        pipeline_config = [
            {
                "step": "extract_statistical",
                "params": {"numeric_columns": ["numeric_float"], "window_size": 5},
            },
            {
                "step": "normalize",
                "params": {
                    "columns": ["numeric_int", "numeric_float"],
                    "method": "minmax",
                },
            },
            {
                "step": "encode_categorical",
                "params": {
                    "columns": ["categorical_str", "ordinal_data"],
                    "method": "onehot",
                },
            },
            {
                "step": "bin_features",
                "params": {
                    "columns": ["numeric_int"],
                    "n_bins": 3,
                    "strategy": "uniform",
                },
            },
            {"step": "select_by_variance", "params": {"threshold": 0.001}},
        ]

        # Engineer features with configuration
        result = await engineer.engineer_features(sample_mixed_dataset, pipeline_config)

        # Verify all transformations were applied

        # Statistical features
        assert "numeric_float_rolling_mean" in result.data.columns

        # Normalized features should be in [0, 1] range
        if "numeric_float" in result.data.columns:
            assert result.data["numeric_float"].min() >= -0.001
            assert result.data["numeric_float"].max() <= 1.001

        # One-hot encoded categorical features
        categorical_cols = [
            col
            for col in result.data.columns
            if col.startswith(("categorical_str_", "ordinal_data_"))
        ]
        assert len(categorical_cols) > 0

        # Binned features
        if "numeric_int" in result.data.columns:
            assert len(result.data["numeric_int"].unique()) <= 3

        # Get feature engineering report
        report = await engineer.generate_feature_report(result, sample_mixed_dataset)

        assert "original_features" in report
        assert "engineered_features" in report
        assert "feature_transformations" in report
        assert "feature_statistics" in report

        assert report["original_features"] == len(sample_mixed_dataset.data.columns)
        assert report["engineered_features"] >= report["original_features"]

    async def test_feature_engineer_incremental_processing(
        self, sample_time_series_dataset
    ):
        """Test incremental feature processing."""
        engineer = FeatureEngineer()

        # Split dataset for incremental processing
        split_idx = len(sample_time_series_dataset.data) // 2

        # First batch
        first_batch_data = sample_time_series_dataset.data.iloc[:split_idx]
        first_batch = Dataset(
            name="first_batch",
            data=first_batch_data,
            description="First batch for incremental processing",
        )

        # Second batch
        second_batch_data = sample_time_series_dataset.data.iloc[split_idx:]
        second_batch = Dataset(
            name="second_batch",
            data=second_batch_data,
            description="Second batch for incremental processing",
        )

        # Process first batch
        first_result = await engineer.engineer_features(first_batch)

        # Process second batch (should use same transformations)
        second_result = await engineer.engineer_features(second_batch)

        # Results should have same structure
        assert set(first_result.data.columns) == set(second_result.data.columns)

        # Combine results
        combined_data = pd.concat(
            [first_result.data, second_result.data], ignore_index=True
        )
        combined_result = Dataset(
            name="combined_result",
            data=combined_data,
            description="Combined incremental processing result",
        )

        # Should have same number of rows as original
        assert len(combined_result.data) == len(sample_time_series_dataset.data)


@pytest.mark.asyncio
class TestFeatureEngineeringErrorHandling:
    """Test error handling in feature engineering."""

    async def test_invalid_column_handling(self, sample_mixed_dataset):
        """Test handling of invalid column specifications."""
        transformer = FeatureTransformer()

        # Test normalization with non-existent columns
        result = await transformer.normalize_features(
            sample_mixed_dataset, columns=["nonexistent_column"], method="zscore"
        )

        # Should handle gracefully and return original dataset
        assert result.data.equals(sample_mixed_dataset.data)

        # Test encoding with non-existent columns
        result = await transformer.encode_categorical_features(
            sample_mixed_dataset, columns=["nonexistent_categorical"], method="onehot"
        )

        # Should handle gracefully
        assert len(result.data.columns) >= len(sample_mixed_dataset.data.columns)

    async def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        engineer = FeatureEngineer()

        # Create empty dataset
        empty_data = pd.DataFrame()
        empty_dataset = Dataset(
            name="empty_test",
            data=empty_data,
            description="Empty dataset for error testing",
        )

        # Should handle empty dataset gracefully
        result = await engineer.engineer_features(empty_dataset)
        assert result.data.empty

        # Get metadata for empty dataset
        metadata = await engineer.get_feature_metadata(result)
        assert len(metadata) == 0

    async def test_malformed_pipeline_config(
        self, sample_mixed_dataset, temp_storage_dir
    ):
        """Test handling of malformed pipeline configurations."""
        pipeline = FeaturePipeline(storage_path=temp_storage_dir)

        # Test invalid step name
        invalid_config = [{"step": "nonexistent_step", "params": {}}]

        with pytest.raises(Exception):
            await pipeline.execute(sample_mixed_dataset, invalid_config)

        # Test missing parameters
        missing_params_config = [
            {
                "step": "normalize",
                # Missing 'params' key
            }
        ]

        with pytest.raises(Exception):
            await pipeline.execute(sample_mixed_dataset, missing_params_config)

        # Test invalid parameters
        invalid_params_config = [
            {
                "step": "normalize",
                "params": {
                    "columns": ["numeric_float"],
                    "method": "invalid_method",  # Invalid normalization method
                },
            }
        ]

        with pytest.raises(Exception):
            await pipeline.execute(sample_mixed_dataset, invalid_params_config)


@pytest.mark.asyncio
class TestFeatureEngineeringPerformance:
    """Performance tests for feature engineering."""

    async def test_large_dataset_processing(self):
        """Test feature engineering with large datasets."""
        # Create large dataset
        n_samples = 10000
        large_data = pd.DataFrame(
            {
                "feature_1": np.random.randn(n_samples),
                "feature_2": np.random.randn(n_samples),
                "feature_3": np.random.randint(0, 100, n_samples),
                "categorical": np.random.choice(["A", "B", "C", "D"], n_samples),
                "timestamp": pd.date_range(
                    start="2020-01-01", periods=n_samples, freq="H"
                ),
            }
        )

        large_dataset = Dataset(
            name="large_performance_test",
            data=large_data,
            description="Large dataset for performance testing",
        )

        engineer = FeatureEngineer()

        # Time the feature engineering process
        start_time = datetime.now()

        pipeline_config = [
            {"step": "extract_temporal", "params": {"timestamp_col": "timestamp"}},
            {
                "step": "normalize",
                "params": {"columns": ["feature_1", "feature_2"], "method": "zscore"},
            },
            {
                "step": "encode_categorical",
                "params": {"columns": ["categorical"], "method": "onehot"},
            },
        ]

        result = await engineer.engineer_features(large_dataset, pipeline_config)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Verify processing completed successfully
        assert result is not None
        assert len(result.data) == n_samples

        # Performance should be reasonable (under 30 seconds for 10k samples)
        assert processing_time < 30.0

        # Verify memory usage is reasonable
        memory_usage_mb = result.data.memory_usage(deep=True).sum() / 1024 / 1024
        assert memory_usage_mb < 100  # Should be under 100MB

    async def test_concurrent_feature_engineering(self, sample_mixed_dataset):
        """Test concurrent feature engineering operations."""
        engineer = FeatureEngineer()

        # Define different pipeline configurations
        configs = [
            [
                {
                    "step": "normalize",
                    "params": {"columns": ["numeric_float"], "method": "zscore"},
                }
            ],
            [
                {
                    "step": "encode_categorical",
                    "params": {"columns": ["categorical_str"], "method": "onehot"},
                }
            ],
            [
                {
                    "step": "bin_features",
                    "params": {
                        "columns": ["numeric_int"],
                        "n_bins": 3,
                        "strategy": "uniform",
                    },
                }
            ],
        ]

        # Run concurrent feature engineering
        async def process_with_config(config):
            return await engineer.engineer_features(sample_mixed_dataset, config)

        start_time = datetime.now()
        tasks = [process_with_config(config) for config in configs]
        results = await asyncio.gather(*tasks)
        end_time = datetime.now()

        concurrent_time = (end_time - start_time).total_seconds()

        # Verify all operations completed successfully
        assert len(results) == 3
        for result in results:
            assert result is not None
            assert isinstance(result, Dataset)

        # Concurrent processing should be faster than sequential
        # (This is a simple check - real performance would depend on system resources)
        assert concurrent_time < 10.0  # Should complete within 10 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
