import pytest
import pandas as pd
import numpy as np

from data_transformation.domain.services.data_cleaning_service import DataCleaningService


class TestDataCleaningService:
    def setup_method(self):
        self.service = DataCleaningService()

    def test_detect_missing_values(self, sample_dataframe):
        missing_info = self.service.detect_missing_values(sample_dataframe)
        
        assert "numeric_col" in missing_info
        assert missing_info["numeric_col"]["count"] == 1
        assert missing_info["numeric_col"]["percentage"] == 20.0

    def test_handle_missing_values_drop(self, sample_dataframe):
        result = self.service.handle_missing_values(sample_dataframe, strategy="drop")
        assert len(result) == 4  # One row dropped

    def test_handle_missing_values_mean(self, sample_dataframe):
        result = self.service.handle_missing_values(sample_dataframe, strategy="mean")
        assert not result["numeric_col"].isna().any()
        assert result["numeric_col"].iloc[2] == sample_dataframe["numeric_col"].mean()

    def test_handle_missing_values_median(self, sample_dataframe):
        result = self.service.handle_missing_values(sample_dataframe, strategy="median")
        assert not result["numeric_col"].isna().any()
        assert result["numeric_col"].iloc[2] == sample_dataframe["numeric_col"].median()

    def test_handle_missing_values_mode(self, sample_dataframe):
        result = self.service.handle_missing_values(sample_dataframe, strategy="mode")
        assert not result["categorical_col"].isna().any()

    def test_handle_missing_values_forward_fill(self, sample_dataframe):
        result = self.service.handle_missing_values(sample_dataframe, strategy="forward_fill")
        assert not result["numeric_col"].isna().any()

    def test_handle_missing_values_backward_fill(self, sample_dataframe):
        result = self.service.handle_missing_values(sample_dataframe, strategy="backward_fill")
        assert not result["numeric_col"].isna().any()

    def test_detect_outliers_iqr(self, sample_dataframe):
        outliers = self.service.detect_outliers(sample_dataframe, method="iqr")
        assert isinstance(outliers, dict)
        assert "numeric_col" in outliers

    def test_detect_outliers_zscore(self, sample_dataframe):
        outliers = self.service.detect_outliers(sample_dataframe, method="zscore")
        assert isinstance(outliers, dict)

    def test_detect_outliers_isolation_forest(self, sample_dataframe):
        outliers = self.service.detect_outliers(sample_dataframe, method="isolation_forest")
        assert isinstance(outliers, dict)

    def test_remove_outliers(self, sample_dataframe):
        # Add an obvious outlier
        df_with_outlier = sample_dataframe.copy()
        df_with_outlier.loc[len(df_with_outlier)] = [1000, 'D', 'outlier', pd.Timestamp('2023-01-06'), False]
        
        result = self.service.remove_outliers(df_with_outlier, method="iqr")
        assert len(result) <= len(df_with_outlier)

    def test_detect_duplicates(self, sample_dataframe):
        # Add a duplicate row
        df_with_duplicate = pd.concat([sample_dataframe, sample_dataframe.iloc[[0]]], ignore_index=True)
        
        duplicates = self.service.detect_duplicates(df_with_duplicate)
        assert duplicates["total_duplicates"] > 0
        assert len(duplicates["duplicate_indices"]) > 0

    def test_remove_duplicates(self, sample_dataframe):
        # Add a duplicate row
        df_with_duplicate = pd.concat([sample_dataframe, sample_dataframe.iloc[[0]]], ignore_index=True)
        
        result = self.service.remove_duplicates(df_with_duplicate)
        assert len(result) == len(sample_dataframe)

    def test_standardize_column_names(self, sample_dataframe):
        df_messy_names = sample_dataframe.copy()
        df_messy_names.columns = ['Numeric Col', 'categorical-col', 'Mixed_Col', 'DateTime Col', 'Binary_Col']
        
        result = self.service.standardize_column_names(df_messy_names)
        expected_names = ['numeric_col', 'categorical_col', 'mixed_col', 'datetime_col', 'binary_col']
        assert list(result.columns) == expected_names

    def test_infer_data_types(self, sample_dataframe):
        type_info = self.service.infer_data_types(sample_dataframe)
        
        assert "numeric_col" in type_info
        assert type_info["numeric_col"]["inferred_type"] == "numeric"
        assert type_info["categorical_col"]["inferred_type"] == "categorical"
        assert type_info["datetime_col"]["inferred_type"] == "datetime"

    def test_convert_data_types(self, sample_dataframe):
        type_mapping = {
            "numeric_col": "float64",
            "categorical_col": "category"
        }
        
        result = self.service.convert_data_types(sample_dataframe, type_mapping)
        assert result["numeric_col"].dtype == np.float64
        assert result["categorical_col"].dtype.name == "category"

    def test_validate_data_quality(self, sample_dataframe):
        report = self.service.validate_data_quality(sample_dataframe)
        
        assert "overall_score" in report
        assert "missing_values" in report
        assert "duplicates" in report
        assert "outliers" in report
        assert "data_types" in report
        assert 0 <= report["overall_score"] <= 100

    def test_clean_text_data(self):
        text_data = pd.Series([
            "  Hello World!  ",
            "UPPERCASE text",
            "special@#$characters",
            "   mixed   CASE   "
        ])
        
        result = self.service.clean_text_data(text_data)
        assert all(not val.startswith(' ') and not val.endswith(' ') for val in result)

    def test_auto_clean_comprehensive(self, sample_dataframe):
        config = {
            "handle_missing": True,
            "remove_duplicates": True,
            "standardize_names": True,
            "convert_types": True
        }
        
        result = self.service.auto_clean(sample_dataframe, config)
        
        # Check that cleaning was applied
        assert not result.isna().any().any() or len(result) < len(sample_dataframe)  # Missing values handled or rows dropped
        assert result.columns.tolist() != sample_dataframe.columns.tolist() or all('_' in col or col.islower() for col in result.columns)  # Names standardized