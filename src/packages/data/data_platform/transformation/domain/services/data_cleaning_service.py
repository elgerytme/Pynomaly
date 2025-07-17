"""Data cleaning domain service."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from ..value_objects.pipeline_config import CleaningStrategy

logger = logging.getLogger(__name__)


class DataCleaningService:
    """
    Domain service for data cleaning operations.
    
    Implements various data cleaning strategies including missing value handling,
    outlier detection and treatment, duplicate removal, and data validation.
    """
    
    def __init__(self) -> None:
        """Initialize the data cleaning service."""
        self._scalers: Dict[str, Any] = {}
    
    def clean_data(
        self,
        data: pd.DataFrame,
        strategy: CleaningStrategy = CleaningStrategy.AUTO,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply comprehensive data cleaning based on strategy.
        
        Args:
            data: Input dataframe to clean
            strategy: Cleaning strategy to apply
            **kwargs: Additional cleaning parameters
            
        Returns:
            Tuple of (cleaned_data, cleaning_report)
        """
        logger.info(f"Starting data cleaning with strategy: {strategy}")
        
        # Initialize cleaning report
        report = {
            "strategy": strategy,
            "original_shape": data.shape,
            "operations": [],
            "warnings": [],
            "statistics": {}
        }
        
        cleaned_data = data.copy()
        
        # Apply strategy-specific cleaning
        if strategy == CleaningStrategy.CONSERVATIVE:
            cleaned_data, ops = self._apply_conservative_cleaning(cleaned_data, **kwargs)
        elif strategy == CleaningStrategy.AGGRESSIVE:
            cleaned_data, ops = self._apply_aggressive_cleaning(cleaned_data, **kwargs)
        elif strategy == CleaningStrategy.AUTO:
            cleaned_data, ops = self._apply_auto_cleaning(cleaned_data, **kwargs)
        elif strategy == CleaningStrategy.CUSTOM:
            cleaned_data, ops = self._apply_custom_cleaning(cleaned_data, **kwargs)
        else:  # NONE
            ops = []
        
        report["operations"].extend(ops)
        report["final_shape"] = cleaned_data.shape
        report["records_removed"] = data.shape[0] - cleaned_data.shape[0]
        report["features_removed"] = data.shape[1] - cleaned_data.shape[1]
        
        logger.info(f"Data cleaning completed. Shape: {data.shape} -> {cleaned_data.shape}")
        
        return cleaned_data, report
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        strategy: str = "auto",
        threshold: float = 0.5
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input dataframe
            strategy: Missing value strategy ('auto', 'drop', 'impute')
            threshold: Threshold for dropping columns/rows
            
        Returns:
            Tuple of (processed_data, missing_value_report)
        """
        report = {
            "missing_counts": data.isnull().sum().to_dict(),
            "missing_percentages": (data.isnull().sum() / len(data)).to_dict(),
            "strategy": strategy,
            "actions_taken": []
        }
        
        result_data = data.copy()
        
        # Drop columns with too many missing values
        missing_pct = data.isnull().sum() / len(data)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            result_data = result_data.drop(columns=cols_to_drop)
            report["actions_taken"].append(f"Dropped columns: {cols_to_drop}")
        
        # Handle remaining missing values
        if strategy == "auto":
            result_data = self._auto_impute_missing(result_data, report)
        elif strategy == "drop":
            result_data = result_data.dropna()
            report["actions_taken"].append("Dropped rows with missing values")
        elif strategy == "impute":
            result_data = self._impute_missing_values(result_data, report)
        
        return result_data, report
    
    def detect_outliers(
        self,
        data: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect outliers in numerical columns.
        
        Args:
            data: Input dataframe
            method: Outlier detection method ('iqr', 'zscore', 'isolation')
            threshold: Threshold for outlier detection
            columns: Specific columns to check (None for all numerical)
            
        Returns:
            Tuple of (outlier_mask, outlier_report)
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
        report = {"method": method, "threshold": threshold, "outliers_by_column": {}}
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if method == "iqr":
                mask = self._detect_outliers_iqr(data[col], threshold)
            elif method == "zscore":
                mask = self._detect_outliers_zscore(data[col], threshold)
            elif method == "isolation":
                mask = self._detect_outliers_isolation(data[col])
            else:
                mask = pd.Series(False, index=data.index)
            
            outlier_mask[col] = mask
            report["outliers_by_column"][col] = mask.sum()
        
        return outlier_mask, report
    
    def remove_duplicates(
        self,
        data: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove duplicate records from the dataset.
        
        Args:
            data: Input dataframe
            subset: Columns to consider for duplicate detection
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            Tuple of (deduplicated_data, duplicate_report)
        """
        original_count = len(data)
        
        # Identify duplicates
        duplicate_mask = data.duplicated(subset=subset, keep=False)
        duplicate_count = duplicate_mask.sum()
        
        # Remove duplicates
        cleaned_data = data.drop_duplicates(subset=subset, keep=keep)
        
        report = {
            "original_records": original_count,
            "duplicate_records": duplicate_count,
            "final_records": len(cleaned_data),
            "duplicates_removed": original_count - len(cleaned_data),
            "subset_columns": subset,
            "keep_strategy": keep
        }
        
        return cleaned_data, report
    
    def validate_data_types(
        self,
        data: pd.DataFrame,
        expected_types: Optional[Dict[str, str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate and correct data types.
        
        Args:
            data: Input dataframe
            expected_types: Expected data types for columns
            
        Returns:
            Tuple of (type_corrected_data, validation_report)
        """
        result_data = data.copy()
        report = {
            "original_types": data.dtypes.to_dict(),
            "type_corrections": {},
            "conversion_errors": {}
        }
        
        # Auto-detect and correct common type issues
        for col in data.columns:
            original_dtype = data[col].dtype
            
            # Try to convert object columns to more specific types
            if original_dtype == "object":
                converted_data, conversion_info = self._auto_convert_column(data[col])
                if conversion_info["success"]:
                    result_data[col] = converted_data
                    report["type_corrections"][col] = conversion_info
        
        # Apply expected types if provided
        if expected_types:
            for col, expected_type in expected_types.items():
                if col in result_data.columns:
                    try:
                        result_data[col] = result_data[col].astype(expected_type)
                        report["type_corrections"][col] = {
                            "from": str(data[col].dtype),
                            "to": expected_type,
                            "success": True
                        }
                    except Exception as e:
                        report["conversion_errors"][col] = str(e)
        
        report["final_types"] = result_data.dtypes.to_dict()
        return result_data, report
    
    def _apply_conservative_cleaning(
        self,
        data: pd.DataFrame,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Apply conservative cleaning strategy."""
        operations = []
        result_data = data.copy()
        
        # Only remove obvious duplicates
        result_data, dup_report = self.remove_duplicates(result_data)
        operations.append({"operation": "remove_duplicates", "report": dup_report})
        
        # Handle missing values conservatively
        result_data, missing_report = self.handle_missing_values(
            result_data, strategy="impute", threshold=0.8
        )
        operations.append({"operation": "handle_missing_values", "report": missing_report})
        
        return result_data, operations
    
    def _apply_aggressive_cleaning(
        self,
        data: pd.DataFrame,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Apply aggressive cleaning strategy."""
        operations = []
        result_data = data.copy()
        
        # Remove duplicates
        result_data, dup_report = self.remove_duplicates(result_data)
        operations.append({"operation": "remove_duplicates", "report": dup_report})
        
        # Aggressive missing value handling
        result_data, missing_report = self.handle_missing_values(
            result_data, strategy="drop", threshold=0.3
        )
        operations.append({"operation": "handle_missing_values", "report": missing_report})
        
        # Remove outliers
        outlier_mask, outlier_report = self.detect_outliers(result_data, method="iqr")
        outlier_rows = outlier_mask.any(axis=1)
        result_data = result_data[~outlier_rows]
        operations.append({"operation": "remove_outliers", "report": outlier_report})
        
        return result_data, operations
    
    def _apply_auto_cleaning(
        self,
        data: pd.DataFrame,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Apply automatic cleaning strategy based on data characteristics."""
        operations = []
        result_data = data.copy()
        
        # Analyze data characteristics
        missing_pct = data.isnull().sum() / len(data)
        duplicate_pct = data.duplicated().sum() / len(data)
        
        # Remove duplicates if significant
        if duplicate_pct > 0.01:  # More than 1% duplicates
            result_data, dup_report = self.remove_duplicates(result_data)
            operations.append({"operation": "remove_duplicates", "report": dup_report})
        
        # Handle missing values based on amount
        threshold = 0.7 if missing_pct.max() > 0.5 else 0.5
        result_data, missing_report = self.handle_missing_values(
            result_data, strategy="auto", threshold=threshold
        )
        operations.append({"operation": "handle_missing_values", "report": missing_report})
        
        # Validate and correct data types
        result_data, type_report = self.validate_data_types(result_data)
        operations.append({"operation": "validate_data_types", "report": type_report})
        
        return result_data, operations
    
    def _apply_custom_cleaning(
        self,
        data: pd.DataFrame,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Apply custom cleaning operations based on provided parameters."""
        operations = []
        result_data = data.copy()
        
        # Custom operations based on kwargs
        if kwargs.get("remove_duplicates", True):
            result_data, dup_report = self.remove_duplicates(result_data)
            operations.append({"operation": "remove_duplicates", "report": dup_report})
        
        if "missing_strategy" in kwargs:
            result_data, missing_report = self.handle_missing_values(
                result_data,
                strategy=kwargs["missing_strategy"],
                threshold=kwargs.get("missing_threshold", 0.5)
            )
            operations.append({"operation": "handle_missing_values", "report": missing_report})
        
        return result_data, operations
    
    def _auto_impute_missing(
        self,
        data: pd.DataFrame,
        report: Dict[str, Any]
    ) -> pd.DataFrame:
        """Auto-impute missing values based on column types."""
        result_data = data.copy()
        
        for col in data.columns:
            if data[col].isnull().any():
                if data[col].dtype in ["int64", "float64"]:
                    # Use median for numerical columns
                    fill_value = data[col].median()
                    result_data[col] = data[col].fillna(fill_value)
                    report["actions_taken"].append(f"Imputed {col} with median: {fill_value}")
                else:
                    # Use mode for categorical columns
                    fill_value = data[col].mode().iloc[0] if not data[col].mode().empty else "Unknown"
                    result_data[col] = data[col].fillna(fill_value)
                    report["actions_taken"].append(f"Imputed {col} with mode: {fill_value}")
        
        return result_data
    
    def _impute_missing_values(
        self,
        data: pd.DataFrame,
        report: Dict[str, Any]
    ) -> pd.DataFrame:
        """Impute missing values using various strategies."""
        return self._auto_impute_missing(data, report)
    
    def _detect_outliers_iqr(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    def _detect_outliers_isolation(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Isolation Forest (placeholder)."""
        # For now, return False for all values
        # TODO: Implement isolation forest when sklearn is available
        return pd.Series(False, index=series.index)
    
    def _auto_convert_column(self, series: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
        """Auto-convert column to appropriate data type."""
        conversion_info = {"success": False, "from": str(series.dtype), "to": None}
        
        try:
            # Try to convert to numeric
            converted = pd.to_numeric(series, errors="coerce")
            if not converted.isnull().all():
                conversion_info.update({"success": True, "to": "numeric"})
                return converted, conversion_info
        except Exception:
            pass
        
        try:
            # Try to convert to datetime
            converted = pd.to_datetime(series, errors="coerce")
            if not converted.isnull().all():
                conversion_info.update({"success": True, "to": "datetime"})
                return converted, conversion_info
        except Exception:
            pass
        
        # Return original if no conversion worked
        return series, conversion_info