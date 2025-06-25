"""Data transformation functionality for preprocessing."""

from __future__ import annotations

import warnings
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError


class ScalingStrategy(Enum):
    """Strategy for scaling numeric features."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE_UNIFORM = "quantile_uniform"
    QUANTILE_NORMAL = "quantile_normal"
    POWER_YEO_JOHNSON = "power_yeo_johnson"
    POWER_BOX_COX = "power_box_cox"


class EncodingStrategy(Enum):
    """Strategy for encoding categorical features."""

    LABEL = "label"
    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    TARGET = "target"
    BINARY = "binary"
    FREQUENCY = "frequency"


class FeatureSelectionStrategy(Enum):
    """Strategy for feature selection."""

    NONE = "none"
    UNIVARIATE_F = "univariate_f"
    UNIVARIATE_MI = "univariate_mi"
    VARIANCE_THRESHOLD = "variance_threshold"
    CORRELATION_THRESHOLD = "correlation_threshold"


class DataTransformer:
    """Service for transforming data including scaling, encoding, and feature engineering."""

    def __init__(self):
        """Initialize the data transformer."""
        self._scalers = {}
        self._encoders = {}
        self._feature_selectors = {}
        self._fitted_transformers = {}

    def scale_features(
        self,
        dataset: Dataset,
        strategy: ScalingStrategy | str = ScalingStrategy.STANDARD,
        columns: list[str] | None = None,
        fit: bool = True,
    ) -> Dataset:
        """Scale numeric features in the dataset.

        Args:
            dataset: Input dataset
            strategy: Scaling strategy to use
            columns: Specific columns to scale (None = all numeric columns)
            fit: Whether to fit the scaler or use previously fitted one

        Returns:
            Dataset with scaled features
        """
        if isinstance(strategy, str):
            strategy = ScalingStrategy(strategy)

        df = dataset.data.copy()

        if columns is None:
            columns = dataset.get_numeric_features()

        # Remove non-numeric columns from the list
        numeric_columns = [
            col for col in columns if pd.api.types.is_numeric_dtype(df[col])
        ]

        if not numeric_columns:
            return dataset

        # Get or create scaler
        scaler_key = f"{dataset.name}_{strategy.value}"

        if fit or scaler_key not in self._scalers:
            if strategy == ScalingStrategy.STANDARD:
                scaler = StandardScaler()
            elif strategy == ScalingStrategy.MINMAX:
                scaler = MinMaxScaler()
            elif strategy == ScalingStrategy.ROBUST:
                scaler = RobustScaler()
            elif strategy == ScalingStrategy.QUANTILE_UNIFORM:
                scaler = QuantileTransformer(output_distribution="uniform")
            elif strategy == ScalingStrategy.QUANTILE_NORMAL:
                scaler = QuantileTransformer(output_distribution="normal")
            elif strategy == ScalingStrategy.POWER_YEO_JOHNSON:
                scaler = PowerTransformer(method="yeo-johnson")
            elif strategy == ScalingStrategy.POWER_BOX_COX:
                scaler = PowerTransformer(method="box-cox")
            else:
                raise ValueError(f"Unknown scaling strategy: {strategy}")

            # Fit the scaler
            try:
                if strategy == ScalingStrategy.POWER_BOX_COX:
                    # Box-Cox requires positive values
                    for col in numeric_columns:
                        if (df[col] <= 0).any():
                            raise ValueError(
                                f"Box-Cox transformation requires positive values in {col}"
                            )

                scaler.fit(df[numeric_columns])
                self._scalers[scaler_key] = scaler

            except Exception as e:
                raise DataValidationError(
                    f"Failed to fit scaler {strategy.value}: {e}",
                    feature_columns=numeric_columns,
                )
        else:
            scaler = self._scalers[scaler_key]

        # Transform the data
        try:
            scaled_data = scaler.transform(df[numeric_columns])
            df[numeric_columns] = scaled_data

        except Exception as e:
            raise DataValidationError(
                f"Failed to transform data with {strategy.value}: {e}",
                feature_columns=numeric_columns,
            )

        return Dataset(
            name=f"{dataset.name}_scaled",
            data=df,
            target_column=dataset.target_column,
            metadata={
                **dataset.metadata,
                "scaling_applied": True,
                "scaling_strategy": strategy.value,
                "scaled_columns": numeric_columns,
                "scaler_key": scaler_key,
            },
        )

    def encode_categorical_features(
        self,
        dataset: Dataset,
        strategy: EncodingStrategy | str = EncodingStrategy.ONEHOT,
        columns: list[str] | None = None,
        fit: bool = True,
        drop_first: bool = True,
        handle_unknown: str = "ignore",
    ) -> Dataset:
        """Encode categorical features to numeric representations.

        Args:
            dataset: Input dataset
            strategy: Encoding strategy to use
            columns: Specific columns to encode (None = all categorical columns)
            fit: Whether to fit the encoder or use previously fitted one
            drop_first: Whether to drop first category in one-hot encoding
            handle_unknown: How to handle unknown categories

        Returns:
            Dataset with encoded categorical features
        """
        if isinstance(strategy, str):
            strategy = EncodingStrategy(strategy)

        df = dataset.data.copy()

        if columns is None:
            columns = dataset.get_categorical_features()

        # Remove non-categorical columns from the list
        categorical_columns = [
            col
            for col in columns
            if df[col].dtype == "object" or df[col].dtype.name == "category"
        ]

        if not categorical_columns:
            return dataset

        encoder_key = f"{dataset.name}_{strategy.value}"
        encoded_columns = []

        for col in categorical_columns:
            try:
                if strategy == EncodingStrategy.LABEL:
                    # Label encoding
                    if fit or f"{encoder_key}_{col}" not in self._encoders:
                        encoder = LabelEncoder()
                        df[col] = encoder.fit_transform(df[col].astype(str))
                        self._encoders[f"{encoder_key}_{col}"] = encoder
                    else:
                        encoder = self._encoders[f"{encoder_key}_{col}"]
                        df[col] = encoder.transform(df[col].astype(str))

                    encoded_columns.append(col)

                elif strategy == EncodingStrategy.ONEHOT:
                    # One-hot encoding
                    if fit or f"{encoder_key}_{col}" not in self._encoders:
                        encoder = OneHotEncoder(
                            drop="first" if drop_first else None,
                            handle_unknown=handle_unknown,
                            sparse_output=False,
                        )
                        encoded_data = encoder.fit_transform(df[[col]])
                        self._encoders[f"{encoder_key}_{col}"] = encoder
                    else:
                        encoder = self._encoders[f"{encoder_key}_{col}"]
                        encoded_data = encoder.transform(df[[col]])

                    # Create new column names
                    feature_names = encoder.get_feature_names_out([col])

                    # Add encoded columns to dataframe
                    for i, name in enumerate(feature_names):
                        df[name] = encoded_data[:, i]
                        encoded_columns.append(name)

                    # Drop original column
                    df = df.drop(columns=[col])

                elif strategy == EncodingStrategy.ORDINAL:
                    # Ordinal encoding
                    if fit or f"{encoder_key}_{col}" not in self._encoders:
                        encoder = OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        )
                        df[col] = encoder.fit_transform(df[[col]]).flatten()
                        self._encoders[f"{encoder_key}_{col}"] = encoder
                    else:
                        encoder = self._encoders[f"{encoder_key}_{col}"]
                        df[col] = encoder.transform(df[[col]]).flatten()

                    encoded_columns.append(col)

                elif strategy == EncodingStrategy.FREQUENCY:
                    # Frequency encoding
                    value_counts = df[col].value_counts()
                    df[col] = df[col].map(value_counts)
                    encoded_columns.append(col)

                elif strategy == EncodingStrategy.TARGET:
                    # Target encoding (requires target column)
                    if not dataset.has_target:
                        warnings.warn(
                            f"Target encoding skipped for {col} - no target column available", stacklevel=2
                        )
                        continue

                    target_means = df.groupby(col)[dataset.target_column].mean()
                    df[col] = df[col].map(target_means)
                    encoded_columns.append(col)

                elif strategy == EncodingStrategy.BINARY:
                    # Binary encoding (custom implementation)
                    unique_values = df[col].unique()
                    n_bits = int(np.ceil(np.log2(len(unique_values))))

                    # Create label encoder first
                    label_encoder = LabelEncoder()
                    encoded_labels = label_encoder.fit_transform(df[col])

                    # Convert to binary
                    for bit in range(n_bits):
                        col_name = f"{col}_binary_{bit}"
                        df[col_name] = (encoded_labels >> bit) & 1
                        encoded_columns.append(col_name)

                    # Drop original column
                    df = df.drop(columns=[col])

            except Exception as e:
                warnings.warn(
                    f"Failed to encode column {col} with {strategy.value}: {e}", stacklevel=2
                )
                continue

        return Dataset(
            name=f"{dataset.name}_encoded",
            data=df,
            target_column=dataset.target_column,
            metadata={
                **dataset.metadata,
                "encoding_applied": True,
                "encoding_strategy": strategy.value,
                "encoded_columns": encoded_columns,
                "original_categorical_columns": categorical_columns,
            },
        )

    def create_polynomial_features(
        self,
        dataset: Dataset,
        degree: int = 2,
        columns: list[str] | None = None,
        interaction_only: bool = False,
        include_bias: bool = False,
    ) -> Dataset:
        """Create polynomial features from numeric columns.

        Args:
            dataset: Input dataset
            degree: Degree of polynomial features
            columns: Specific columns to use (None = all numeric columns)
            interaction_only: Only create interaction features
            include_bias: Include bias column

        Returns:
            Dataset with polynomial features
        """
        try:
            from sklearn.preprocessing import PolynomialFeatures
        except ImportError:
            raise ImportError("sklearn is required for polynomial features")

        df = dataset.data.copy()

        if columns is None:
            columns = dataset.get_numeric_features()[
                :5
            ]  # Limit to 5 features to avoid explosion

        numeric_columns = [
            col for col in columns if pd.api.types.is_numeric_dtype(df[col])
        ]

        if not numeric_columns:
            return dataset

        # Create polynomial features
        poly = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )

        poly_features = poly.fit_transform(df[numeric_columns])
        feature_names = poly.get_feature_names_out(numeric_columns)

        # Create new dataframe with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

        # Remove original columns and add polynomial features
        df_new = df.drop(columns=numeric_columns)
        df_new = pd.concat([df_new, poly_df], axis=1)

        return Dataset(
            name=f"{dataset.name}_poly",
            data=df_new,
            target_column=dataset.target_column,
            metadata={
                **dataset.metadata,
                "polynomial_features": True,
                "polynomial_degree": degree,
                "original_columns": numeric_columns,
                "polynomial_columns": list(feature_names),
            },
        )

    def select_features(
        self,
        dataset: Dataset,
        strategy: FeatureSelectionStrategy | str = FeatureSelectionStrategy.VARIANCE_THRESHOLD,
        k: int = 10,
        threshold: float = 0.01,
        columns: list[str] | None = None,
    ) -> Dataset:
        """Select the most relevant features.

        Args:
            dataset: Input dataset
            strategy: Feature selection strategy
            k: Number of features to select (for univariate methods)
            threshold: Threshold for variance/correlation methods
            columns: Specific columns to consider (None = all features)

        Returns:
            Dataset with selected features
        """
        if isinstance(strategy, str):
            strategy = FeatureSelectionStrategy(strategy)

        if strategy == FeatureSelectionStrategy.NONE:
            return dataset

        df = dataset.data.copy()

        if columns is None:
            columns = [col for col in df.columns if col != dataset.target_column]

        selected_features = []

        if strategy == FeatureSelectionStrategy.VARIANCE_THRESHOLD:
            # Remove low variance features
            numeric_cols = [
                col for col in columns if pd.api.types.is_numeric_dtype(df[col])
            ]

            for col in numeric_cols:
                if df[col].var() > threshold:
                    selected_features.append(col)

            # Add non-numeric columns
            selected_features.extend(
                [col for col in columns if col not in numeric_cols]
            )

        elif strategy == FeatureSelectionStrategy.CORRELATION_THRESHOLD:
            # Remove highly correlated features
            numeric_cols = [
                col for col in columns if pd.api.types.is_numeric_dtype(df[col])
            ]

            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().abs()

                # Find pairs of highly correlated features
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )

                # Select features to drop
                to_drop = [
                    column
                    for column in upper_tri.columns
                    if any(upper_tri[column] > threshold)
                ]

                selected_features = [col for col in numeric_cols if col not in to_drop]
            else:
                selected_features = numeric_cols

            # Add non-numeric columns
            selected_features.extend(
                [col for col in columns if col not in numeric_cols]
            )

        elif strategy in [
            FeatureSelectionStrategy.UNIVARIATE_F,
            FeatureSelectionStrategy.UNIVARIATE_MI,
        ]:
            # Univariate feature selection (requires target)
            if not dataset.has_target:
                warnings.warn("Univariate feature selection requires target column", stacklevel=2)
                return dataset

            numeric_cols = [
                col for col in columns if pd.api.types.is_numeric_dtype(df[col])
            ]

            if not numeric_cols:
                return dataset

            X = df[numeric_cols]
            y = df[dataset.target_column]

            # Choose scoring function
            if strategy == FeatureSelectionStrategy.UNIVARIATE_F:
                score_func = (
                    f_classif if pd.api.types.is_integer_dtype(y) else f_classif
                )
            else:  # UNIVARIATE_MI
                score_func = (
                    mutual_info_classif
                    if pd.api.types.is_integer_dtype(y)
                    else mutual_info_classif
                )

            # Select features
            selector = SelectKBest(score_func=score_func, k=min(k, len(numeric_cols)))
            selector.fit(X, y)

            # Get selected feature names
            selected_mask = selector.get_support()
            selected_features = [
                numeric_cols[i] for i, selected in enumerate(selected_mask) if selected
            ]

            # Add non-numeric columns
            selected_features.extend(
                [col for col in columns if col not in numeric_cols]
            )

        # Create new dataset with selected features
        if dataset.target_column and dataset.target_column not in selected_features:
            selected_features.append(dataset.target_column)

        df_selected = df[selected_features]

        return Dataset(
            name=f"{dataset.name}_selected",
            data=df_selected,
            target_column=dataset.target_column,
            metadata={
                **dataset.metadata,
                "feature_selection": True,
                "selection_strategy": strategy.value,
                "original_features": len(columns),
                "selected_features": len(selected_features)
                - (1 if dataset.target_column else 0),
                "selected_feature_names": [
                    col for col in selected_features if col != dataset.target_column
                ],
            },
        )

    def convert_data_types(
        self,
        dataset: Dataset,
        type_mapping: dict[str, str] | None = None,
        infer_types: bool = True,
        optimize_memory: bool = True,
    ) -> Dataset:
        """Convert data types for better memory usage and compatibility.

        Args:
            dataset: Input dataset
            type_mapping: Explicit type mapping {column: dtype}
            infer_types: Whether to automatically infer optimal types
            optimize_memory: Whether to optimize for memory usage

        Returns:
            Dataset with optimized data types
        """
        df = dataset.data.copy()
        conversion_info = {}

        # Apply explicit type mapping
        if type_mapping:
            for col, dtype in type_mapping.items():
                if col in df.columns:
                    try:
                        original_dtype = str(df[col].dtype)
                        df[col] = df[col].astype(dtype)
                        conversion_info[col] = {
                            "original": original_dtype,
                            "converted": dtype,
                            "method": "explicit",
                        }
                    except Exception as e:
                        warnings.warn(f"Failed to convert {col} to {dtype}: {e}", stacklevel=2)

        # Infer optimal types
        if infer_types:
            for col in df.columns:
                if col == dataset.target_column:
                    continue

                original_dtype = str(df[col].dtype)

                # Try to convert object columns to numeric
                if df[col].dtype == "object":
                    # Try numeric conversion
                    try:
                        numeric_series = pd.to_numeric(df[col], errors="coerce")
                        if not numeric_series.isnull().all():
                            df[col] = numeric_series
                            conversion_info[col] = {
                                "original": original_dtype,
                                "converted": str(df[col].dtype),
                                "method": "inferred_numeric",
                            }
                            continue
                    except:
                        pass

                    # Try datetime conversion
                    try:
                        datetime_series = pd.to_datetime(df[col], errors="coerce")
                        if not datetime_series.isnull().all():
                            df[col] = datetime_series
                            conversion_info[col] = {
                                "original": original_dtype,
                                "converted": str(df[col].dtype),
                                "method": "inferred_datetime",
                            }
                            continue
                    except:
                        pass

                    # Convert to category if low cardinality
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        df[col] = df[col].astype("category")
                        conversion_info[col] = {
                            "original": original_dtype,
                            "converted": "category",
                            "method": "inferred_category",
                        }

        # Optimize memory usage
        if optimize_memory:
            for col in df.columns:
                if col == dataset.target_column:
                    continue

                original_dtype = str(df[col].dtype)

                # Optimize integer columns
                if pd.api.types.is_integer_dtype(df[col]):
                    min_val = df[col].min()
                    max_val = df[col].max()

                    if min_val >= 0:  # Unsigned integers
                        if max_val < 255:
                            df[col] = df[col].astype("uint8")
                        elif max_val < 65535:
                            df[col] = df[col].astype("uint16")
                        elif max_val < 4294967295:
                            df[col] = df[col].astype("uint32")
                    else:  # Signed integers
                        if min_val > -128 and max_val < 127:
                            df[col] = df[col].astype("int8")
                        elif min_val > -32768 and max_val < 32767:
                            df[col] = df[col].astype("int16")
                        elif min_val > -2147483648 and max_val < 2147483647:
                            df[col] = df[col].astype("int32")

                    if str(df[col].dtype) != original_dtype:
                        conversion_info[col] = {
                            "original": original_dtype,
                            "converted": str(df[col].dtype),
                            "method": "memory_optimized",
                        }

                # Optimize float columns
                elif pd.api.types.is_float_dtype(df[col]):
                    if df[col].dtype == "float64":
                        # Check if float32 is sufficient
                        if (df[col] == df[col].astype("float32")).all():
                            df[col] = df[col].astype("float32")
                            conversion_info[col] = {
                                "original": original_dtype,
                                "converted": "float32",
                                "method": "memory_optimized",
                            }

        return Dataset(
            name=f"{dataset.name}_optimized",
            data=df,
            target_column=dataset.target_column,
            metadata={
                **dataset.metadata,
                "type_conversion": True,
                "conversions": conversion_info,
                "memory_usage_before": dataset.memory_usage,
                "memory_usage_after": int(df.memory_usage(deep=True).sum()),
            },
        )
