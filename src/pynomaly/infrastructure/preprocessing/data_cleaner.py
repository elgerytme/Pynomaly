"""Data cleaning functionality for preprocessing."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from pynomaly.domain.entities import Dataset


class MissingValueStrategy(Enum):
    """Strategy for handling missing values."""

    DROP_ROWS = "drop_rows"
    DROP_COLUMNS = "drop_columns"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_CONSTANT = "fill_constant"
    FILL_FORWARD = "fill_forward"
    FILL_BACKWARD = "fill_backward"
    INTERPOLATE = "interpolate"
    KNN_IMPUTE = "knn_impute"


class OutlierStrategy(Enum):
    """Strategy for handling outliers."""

    REMOVE = "remove"
    CLIP = "clip"
    TRANSFORM_LOG = "transform_log"
    TRANSFORM_SQRT = "transform_sqrt"
    WINSORIZE = "winsorize"


class DataCleaner:
    """Service for cleaning data issues like missing values, outliers, and duplicates."""

    def __init__(self):
        """Initialize the data cleaner."""
        self._imputers = {}
        self._scalers = {}

    def handle_missing_values(
        self,
        dataset: Dataset,
        strategy: MissingValueStrategy | str = MissingValueStrategy.DROP_ROWS,
        threshold: float = 0.5,
        fill_value: Any = None,
        columns: list[str] | None = None,
    ) -> Dataset:
        """Handle missing values in the dataset.

        Args:
            dataset: Input dataset
            strategy: Strategy for handling missing values
            threshold: Threshold for dropping columns (0.0 to 1.0)
            fill_value: Value to use for constant filling
            columns: Specific columns to process (None = all columns)

        Returns:
            Cleaned dataset
        """
        if isinstance(strategy, str):
            strategy = MissingValueStrategy(strategy)

        df = dataset.data.copy()

        if columns is None:
            columns = list(df.columns)
            if dataset.target_column:
                columns = [col for col in columns if col != dataset.target_column]

        # Calculate missing ratios
        missing_info = {}
        for col in columns:
            missing_count = df[col].isnull().sum()
            missing_ratio = missing_count / len(df)
            missing_info[col] = {"count": missing_count, "ratio": missing_ratio}

        cleaned_data = df.copy()

        if strategy == MissingValueStrategy.DROP_ROWS:
            # Drop rows with any missing values in specified columns
            cleaned_data = cleaned_data.dropna(subset=columns)

        elif strategy == MissingValueStrategy.DROP_COLUMNS:
            # Drop columns that exceed missing threshold
            cols_to_drop = [
                col for col, info in missing_info.items() if info["ratio"] > threshold
            ]
            if cols_to_drop:
                cleaned_data = cleaned_data.drop(columns=cols_to_drop)

        elif strategy == MissingValueStrategy.FILL_MEAN:
            # Fill with mean for numeric columns
            for col in columns:
                if pd.api.types.is_numeric_dtype(cleaned_data[col]):
                    cleaned_data[col] = cleaned_data[col].fillna(
                        cleaned_data[col].mean()
                    )

        elif strategy == MissingValueStrategy.FILL_MEDIAN:
            # Fill with median for numeric columns
            for col in columns:
                if pd.api.types.is_numeric_dtype(cleaned_data[col]):
                    cleaned_data[col] = cleaned_data[col].fillna(
                        cleaned_data[col].median()
                    )

        elif strategy == MissingValueStrategy.FILL_MODE:
            # Fill with mode (most frequent value)
            for col in columns:
                mode_value = cleaned_data[col].mode()
                if not mode_value.empty:
                    cleaned_data[col] = cleaned_data[col].fillna(mode_value.iloc[0])

        elif strategy == MissingValueStrategy.FILL_CONSTANT:
            # Fill with constant value
            if fill_value is not None:
                for col in columns:
                    cleaned_data[col] = cleaned_data[col].fillna(fill_value)

        elif strategy == MissingValueStrategy.FILL_FORWARD:
            # Forward fill
            cleaned_data[columns] = cleaned_data[columns].fillna(method="ffill")

        elif strategy == MissingValueStrategy.FILL_BACKWARD:
            # Backward fill
            cleaned_data[columns] = cleaned_data[columns].fillna(method="bfill")

        elif strategy == MissingValueStrategy.INTERPOLATE:
            # Interpolate for numeric columns
            for col in columns:
                if pd.api.types.is_numeric_dtype(cleaned_data[col]):
                    cleaned_data[col] = cleaned_data[col].interpolate()

        elif strategy == MissingValueStrategy.KNN_IMPUTE:
            # KNN imputation for numeric columns
            numeric_cols = [
                col
                for col in columns
                if pd.api.types.is_numeric_dtype(cleaned_data[col])
            ]
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=5)
                cleaned_data[numeric_cols] = imputer.fit_transform(
                    cleaned_data[numeric_cols]
                )

        # Create new dataset with cleaned data
        return Dataset(
            name=f"{dataset.name}_cleaned",
            data=cleaned_data,
            target_column=dataset.target_column,
            metadata={
                **dataset.metadata,
                "cleaning_applied": True,
                "missing_strategy": strategy.value,
                "original_shape": dataset.shape,
                "cleaned_shape": cleaned_data.shape,
                "missing_info": missing_info,
            },
        )

    def remove_duplicates(
        self, dataset: Dataset, subset: list[str] | None = None, keep: str = "first"
    ) -> Dataset:
        """Remove duplicate rows from the dataset.

        Args:
            dataset: Input dataset
            subset: Columns to consider for duplication (None = all columns)
            keep: Which duplicates to keep ('first', 'last', False)

        Returns:
            Dataset without duplicates
        """
        df = dataset.data.copy()
        original_count = len(df)

        # Remove duplicates
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        duplicates_removed = original_count - len(df_cleaned)

        return Dataset(
            name=f"{dataset.name}_no_duplicates",
            data=df_cleaned,
            target_column=dataset.target_column,
            metadata={
                **dataset.metadata,
                "duplicates_removed": duplicates_removed,
                "original_count": original_count,
                "final_count": len(df_cleaned),
            },
        )

    def handle_outliers(
        self,
        dataset: Dataset,
        strategy: OutlierStrategy | str = OutlierStrategy.CLIP,
        method: str = "iqr",
        threshold: float = 1.5,
        columns: list[str] | None = None,
    ) -> Dataset:
        """Handle outliers in numeric columns.

        Args:
            dataset: Input dataset
            strategy: Strategy for handling outliers
            method: Method for detecting outliers ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            columns: Specific columns to process (None = all numeric columns)

        Returns:
            Dataset with outliers handled
        """
        if isinstance(strategy, str):
            strategy = OutlierStrategy(strategy)

        df = dataset.data.copy()

        if columns is None:
            columns = dataset.get_numeric_features()

        outlier_info = {}

        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Detect outliers
            outlier_mask = self._detect_outliers(df[col], method, threshold)
            outlier_count = outlier_mask.sum()
            outlier_info[col] = outlier_count

            if outlier_count == 0:
                continue

            if strategy == OutlierStrategy.REMOVE:
                # Remove outlier rows
                df = df[~outlier_mask]

            elif strategy == OutlierStrategy.CLIP:
                # Clip to percentiles
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                df[col] = df[col].clip(lower_bound, upper_bound)

            elif strategy == OutlierStrategy.TRANSFORM_LOG:
                # Log transformation (handle negative values)
                if (df[col] > 0).all():
                    df[col] = np.log1p(df[col])
                else:
                    warnings.warn(
                        f"Cannot apply log transform to {col} with non-positive values"
                    )

            elif strategy == OutlierStrategy.TRANSFORM_SQRT:
                # Square root transformation (handle negative values)
                if (df[col] >= 0).all():
                    df[col] = np.sqrt(df[col])
                else:
                    warnings.warn(
                        f"Cannot apply sqrt transform to {col} with negative values"
                    )

            elif strategy == OutlierStrategy.WINSORIZE:
                # Winsorize (replace with percentile values)
                lower_percentile = df[col].quantile(0.05)
                upper_percentile = df[col].quantile(0.95)
                df.loc[df[col] < lower_percentile, col] = lower_percentile
                df.loc[df[col] > upper_percentile, col] = upper_percentile

        return Dataset(
            name=f"{dataset.name}_outliers_handled",
            data=df,
            target_column=dataset.target_column,
            metadata={
                **dataset.metadata,
                "outlier_strategy": strategy.value,
                "outlier_method": method,
                "outliers_detected": outlier_info,
                "original_shape": dataset.shape,
                "final_shape": df.shape,
            },
        )

    def _detect_outliers(
        self, series: pd.Series, method: str, threshold: float
    ) -> pd.Series:
        """Detect outliers in a series.

        Args:
            series: Data series
            method: Detection method
            threshold: Threshold value

        Returns:
            Boolean mask of outliers
        """
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (series < lower_bound) | (series > upper_bound)

        elif method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold

        elif method == "modified_zscore":
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def handle_zero_values(
        self,
        dataset: Dataset,
        strategy: str = "keep",
        replacement_value: float | None = None,
        columns: list[str] | None = None,
    ) -> Dataset:
        """Handle zero values in numeric columns.

        Args:
            dataset: Input dataset
            strategy: Strategy for handling zeros ('keep', 'remove', 'replace', 'log_transform')
            replacement_value: Value to replace zeros with
            columns: Specific columns to process (None = all numeric columns)

        Returns:
            Dataset with zero values handled
        """
        df = dataset.data.copy()

        if columns is None:
            columns = dataset.get_numeric_features()

        zero_info = {}

        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            zero_mask = df[col] == 0
            zero_count = zero_mask.sum()
            zero_info[col] = zero_count

            if zero_count == 0:
                continue

            if strategy == "keep":
                # Do nothing
                continue

            elif strategy == "remove":
                # Remove rows with zeros
                df = df[~zero_mask]

            elif strategy == "replace":
                # Replace with specified value or mean
                if replacement_value is not None:
                    df.loc[zero_mask, col] = replacement_value
                else:
                    # Replace with mean of non-zero values
                    non_zero_mean = df.loc[~zero_mask, col].mean()
                    df.loc[zero_mask, col] = non_zero_mean

            elif strategy == "log_transform":
                # Add small constant before log transform
                df[col] = np.log1p(df[col])

        return Dataset(
            name=f"{dataset.name}_zeros_handled",
            data=df,
            target_column=dataset.target_column,
            metadata={
                **dataset.metadata,
                "zero_strategy": strategy,
                "zeros_detected": zero_info,
                "original_shape": dataset.shape,
                "final_shape": df.shape,
            },
        )

    def handle_infinite_values(
        self,
        dataset: Dataset,
        strategy: str = "replace",
        replacement_value: float | None = None,
        columns: list[str] | None = None,
    ) -> Dataset:
        """Handle infinite values in numeric columns.

        Args:
            dataset: Input dataset
            strategy: Strategy for handling infinites ('remove', 'replace', 'clip')
            replacement_value: Value to replace infinites with
            columns: Specific columns to process (None = all numeric columns)

        Returns:
            Dataset with infinite values handled
        """
        df = dataset.data.copy()

        if columns is None:
            columns = dataset.get_numeric_features()

        inf_info = {}

        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            inf_mask = np.isinf(df[col])
            inf_count = inf_mask.sum()
            inf_info[col] = inf_count

            if inf_count == 0:
                continue

            if strategy == "remove":
                # Remove rows with infinites
                df = df[~inf_mask]

            elif strategy == "replace":
                # Replace with specified value or large finite value
                if replacement_value is not None:
                    df.loc[inf_mask, col] = replacement_value
                else:
                    # Replace with max finite value
                    finite_values = df.loc[~inf_mask, col]
                    if not finite_values.empty:
                        max_finite = finite_values.max()
                        df.loc[df[col] == np.inf, col] = max_finite * 10
                        df.loc[df[col] == -np.inf, col] = finite_values.min() * 10

            elif strategy == "clip":
                # Clip to finite range
                finite_values = df.loc[~inf_mask, col]
                if not finite_values.empty:
                    df[col] = df[col].clip(finite_values.min(), finite_values.max())

        return Dataset(
            name=f"{dataset.name}_infs_handled",
            data=df,
            target_column=dataset.target_column,
            metadata={
                **dataset.metadata,
                "infinite_strategy": strategy,
                "infinites_detected": inf_info,
                "original_shape": dataset.shape,
                "final_shape": df.shape,
            },
        )

    def comprehensive_clean(
        self,
        dataset: Dataset,
        missing_strategy: MissingValueStrategy | str = MissingValueStrategy.FILL_MEDIAN,
        outlier_strategy: OutlierStrategy | str = OutlierStrategy.CLIP,
        handle_duplicates: bool = True,
        handle_zeros: bool = True,
        handle_infinites: bool = True,
    ) -> Dataset:
        """Perform comprehensive data cleaning.

        Args:
            dataset: Input dataset
            missing_strategy: Strategy for missing values
            outlier_strategy: Strategy for outliers
            handle_duplicates: Whether to remove duplicates
            handle_zeros: Whether to handle zero values
            handle_infinites: Whether to handle infinite values

        Returns:
            Comprehensively cleaned dataset
        """
        cleaned_dataset = dataset

        # Handle missing values
        cleaned_dataset = self.handle_missing_values(
            cleaned_dataset, strategy=missing_strategy
        )

        # Handle duplicates
        if handle_duplicates:
            cleaned_dataset = self.remove_duplicates(cleaned_dataset)

        # Handle infinite values
        if handle_infinites:
            cleaned_dataset = self.handle_infinite_values(cleaned_dataset)

        # Handle zero values
        if handle_zeros:
            cleaned_dataset = self.handle_zero_values(cleaned_dataset)

        # Handle outliers
        cleaned_dataset = self.handle_outliers(
            cleaned_dataset, strategy=outlier_strategy
        )

        # Update metadata
        cleaned_dataset.metadata.update(
            {
                "comprehensive_cleaning": True,
                "original_dataset_id": str(dataset.id),
                "cleaning_steps": [
                    "missing_values",
                    "duplicates" if handle_duplicates else None,
                    "infinite_values" if handle_infinites else None,
                    "zero_values" if handle_zeros else None,
                    "outliers",
                ],
            }
        )

        return cleaned_dataset
