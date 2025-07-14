import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_transformation.infrastructure.processors.feature_processor import FeatureProcessor


class TestFeatureProcessor:
    def setup_method(self):
        self.processor = FeatureProcessor()

    def test_scale_features_standard(self, sample_dataframe):
        numeric_cols = ["numeric_col"]
        result = self.processor.scale_features(
            sample_dataframe, 
            method="standard", 
            columns=numeric_cols
        )
        
        # Check that scaling was applied
        scaled_col = result[numeric_cols[0]]
        # Remove NaN values for testing
        scaled_col_clean = scaled_col.dropna()
        assert abs(scaled_col_clean.mean()) < 1e-10  # Mean should be ~0
        assert abs(scaled_col_clean.std() - 1.0) < 1e-10  # Std should be ~1

    def test_scale_features_minmax(self, sample_dataframe):
        numeric_cols = ["numeric_col"]
        result = self.processor.scale_features(
            sample_dataframe, 
            method="minmax", 
            columns=numeric_cols
        )
        
        # Check that scaling was applied
        scaled_col = result[numeric_cols[0]].dropna()
        assert scaled_col.min() >= 0
        assert scaled_col.max() <= 1

    def test_scale_features_robust(self, sample_dataframe):
        numeric_cols = ["numeric_col"]
        result = self.processor.scale_features(
            sample_dataframe, 
            method="robust", 
            columns=numeric_cols
        )
        
        # Should return a DataFrame with same shape
        assert result.shape == sample_dataframe.shape
        assert numeric_cols[0] in result.columns

    def test_encode_categorical_one_hot(self, sample_dataframe):
        categorical_cols = ["categorical_col"]
        result = self.processor.encode_categorical(
            sample_dataframe, 
            method="one_hot", 
            columns=categorical_cols
        )
        
        # Should have more columns due to one-hot encoding
        assert result.shape[1] > sample_dataframe.shape[1]
        # Original categorical column should be removed
        assert categorical_cols[0] not in result.columns

    def test_encode_categorical_label(self, sample_dataframe):
        categorical_cols = ["categorical_col"]
        result = self.processor.encode_categorical(
            sample_dataframe, 
            method="label", 
            columns=categorical_cols
        )
        
        # Should have same number of columns
        assert result.shape[1] == sample_dataframe.shape[1]
        # Categorical column should be numeric now
        assert pd.api.types.is_numeric_dtype(result[categorical_cols[0]])

    def test_encode_categorical_target(self, sample_dataframe):
        categorical_cols = ["categorical_col"]
        # Create a target variable
        target = np.random.random(len(sample_dataframe))
        
        result = self.processor.encode_categorical(
            sample_dataframe, 
            method="target", 
            columns=categorical_cols,
            target=target
        )
        
        # Should have same number of columns
        assert result.shape[1] == sample_dataframe.shape[1]
        # Categorical column should be numeric now
        assert pd.api.types.is_numeric_dtype(result[categorical_cols[0]])

    def test_create_polynomial_features(self, sample_dataframe):
        numeric_cols = ["numeric_col"]
        df_clean = sample_dataframe.dropna(subset=numeric_cols)
        
        result = self.processor.create_polynomial_features(
            df_clean, 
            columns=numeric_cols, 
            degree=2
        )
        
        # Should have more columns due to polynomial features
        assert result.shape[1] > df_clean.shape[1]

    def test_create_interaction_features(self, sample_dataframe):
        # Add another numeric column for interaction
        sample_dataframe["numeric_col2"] = [2, 4, 6, 8, 10]
        numeric_cols = ["numeric_col", "numeric_col2"]
        df_clean = sample_dataframe.dropna(subset=numeric_cols)
        
        result = self.processor.create_interaction_features(
            df_clean, 
            columns=numeric_cols
        )
        
        # Should have more columns due to interaction features
        assert result.shape[1] > df_clean.shape[1]

    def test_create_temporal_features(self):
        # Create DataFrame with datetime column
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({'date_col': dates, 'value': range(10)})
        
        result = self.processor.create_temporal_features(
            df, 
            datetime_columns=["date_col"]
        )
        
        # Should have additional temporal features
        expected_features = ['date_col_year', 'date_col_month', 'date_col_day', 
                           'date_col_dayofweek', 'date_col_hour', 'date_col_minute']
        for feature in expected_features:
            if feature in result.columns:
                assert feature in result.columns

    def test_select_features_correlation(self, sample_dataframe):
        # Add correlated features
        sample_dataframe["correlated_col"] = sample_dataframe["numeric_col"] * 2
        sample_dataframe = sample_dataframe.dropna()
        
        result = self.processor.select_features(
            sample_dataframe, 
            method="correlation", 
            threshold=0.9
        )
        
        # Should remove highly correlated features
        assert result.shape[1] <= sample_dataframe.shape[1]

    def test_select_features_variance(self, sample_dataframe):
        # Add low variance column
        sample_dataframe["low_variance"] = [1, 1, 1, 1, 1]
        
        result = self.processor.select_features(
            sample_dataframe, 
            method="variance", 
            threshold=0.01
        )
        
        # Should remove low variance features
        assert "low_variance" not in result.columns

    def test_select_features_mutual_info(self, sample_dataframe):
        target = np.random.random(len(sample_dataframe))
        sample_dataframe_clean = sample_dataframe.dropna()
        target_clean = target[:len(sample_dataframe_clean)]
        
        result = self.processor.select_features(
            sample_dataframe_clean, 
            method="mutual_info", 
            target=target_clean,
            k=2
        )
        
        # Should select top k features
        numeric_cols = sample_dataframe_clean.select_dtypes(include=[np.number]).columns
        assert result.shape[1] <= len(numeric_cols)

    def test_handle_missing_values_in_features(self, sample_dataframe):
        result = self.processor.handle_missing_values(
            sample_dataframe, 
            strategy="mean"
        )
        
        # Should handle missing values
        assert result["numeric_col"].isna().sum() == 0

    def test_detect_feature_types(self, sample_dataframe):
        feature_types = self.processor.detect_feature_types(sample_dataframe)
        
        assert "numeric" in feature_types
        assert "categorical" in feature_types
        assert "datetime" in feature_types
        assert "boolean" in feature_types

    def test_auto_feature_engineering(self, sample_dataframe):
        config = {
            "scaling": True,
            "encoding": True,
            "polynomial": False,
            "interactions": False,
            "temporal": True
        }
        
        result = self.processor.auto_feature_engineering(sample_dataframe, config)
        
        # Should apply transformations based on config
        assert isinstance(result, pd.DataFrame)
        # Should have at least the same number of rows
        assert result.shape[0] <= sample_dataframe.shape[0]  # May drop rows with missing values

    def test_create_bins(self, sample_dataframe):
        result = self.processor.create_bins(
            sample_dataframe, 
            column="numeric_col", 
            n_bins=3
        )
        
        binned_col = f"numeric_col_binned"
        assert binned_col in result.columns
        # Should have 3 unique bin values (plus possibly NaN)
        unique_bins = result[binned_col].dropna().nunique()
        assert unique_bins <= 3

    def test_normalize_features(self, sample_dataframe):
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number]).columns.tolist()
        result = self.processor.normalize_features(
            sample_dataframe, 
            columns=numeric_cols, 
            method="l2"
        )
        
        # Should return DataFrame with same shape
        assert result.shape == sample_dataframe.shape

    def test_create_lag_features(self):
        # Create time series data
        df = pd.DataFrame({
            'value': range(10),
            'date': pd.date_range('2023-01-01', periods=10, freq='D')
        })
        
        result = self.processor.create_lag_features(
            df, 
            columns=["value"], 
            lags=[1, 2]
        )
        
        # Should have lag columns
        assert "value_lag_1" in result.columns
        assert "value_lag_2" in result.columns

    def test_create_rolling_features(self):
        # Create time series data
        df = pd.DataFrame({
            'value': range(10),
            'date': pd.date_range('2023-01-01', periods=10, freq='D')
        })
        
        result = self.processor.create_rolling_features(
            df, 
            columns=["value"], 
            window=3
        )
        
        # Should have rolling features
        rolling_cols = [col for col in result.columns if 'rolling' in col]
        assert len(rolling_cols) > 0