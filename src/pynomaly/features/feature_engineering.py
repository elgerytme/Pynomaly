"""Feature engineering and preprocessing for Pynomaly."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset
from pynomaly.shared.error_handling import (
    ErrorCodes,
    create_infrastructure_error,
)

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    BINARY = "binary"
    ORDINAL = "ordinal"


class TransformationType(Enum):
    """Types of feature transformations."""

    NORMALIZATION = "normalization"
    STANDARDIZATION = "standardization"
    ENCODING = "encoding"
    BINNING = "binning"
    AGGREGATION = "aggregation"
    EXTRACTION = "extraction"
    SELECTION = "selection"


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""

    name: str
    type: FeatureType
    description: str = ""
    nullable: bool = True
    unique_values: int | None = None
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    std_value: float | None = None
    missing_count: int = 0
    missing_percentage: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    transformations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "nullable": self.nullable,
            "unique_values": self.unique_values,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "missing_count": self.missing_count,
            "missing_percentage": self.missing_percentage,
            "created_at": self.created_at.isoformat(),
            "transformations": self.transformations,
        }


@dataclass
class TransformationResult:
    """Result of a feature transformation."""

    transformation_type: TransformationType
    feature_name: str
    original_data: pd.Series
    transformed_data: pd.Series
    parameters: dict[str, Any]
    execution_time_ms: float
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "transformation_type": self.transformation_type.value,
            "feature_name": self.feature_name,
            "parameters": self.parameters,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "error_message": self.error_message,
            "original_shape": self.original_data.shape,
            "transformed_shape": self.transformed_data.shape,
        }


class FeatureExtractor:
    """Extract features from raw data."""

    def __init__(self):
        """Initialize feature extractor."""
        self.extractors: dict[str, Callable] = {}
        self.extraction_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "extraction_times": [],
        }

    def register_extractor(
        self, name: str, extractor: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> None:
        """Register a custom feature extractor.

        Args:
            name: Extractor name
            extractor: Extractor function
        """
        self.extractors[name] = extractor
        logger.info(f"Registered feature extractor: {name}")

    async def extract_temporal_features(
        self, data: pd.DataFrame, datetime_column: str
    ) -> pd.DataFrame:
        """Extract temporal features from datetime column.

        Args:
            data: Input dataframe
            datetime_column: Name of datetime column

        Returns:
            Dataframe with temporal features
        """
        try:
            start_time = datetime.now()

            if datetime_column not in data.columns:
                raise ValueError(f"Column {datetime_column} not found in data")

            # Convert to datetime if not already
            datetime_series = pd.to_datetime(data[datetime_column])

            # Extract temporal features
            temporal_features = pd.DataFrame(
                {
                    f"{datetime_column}_year": datetime_series.dt.year,
                    f"{datetime_column}_month": datetime_series.dt.month,
                    f"{datetime_column}_day": datetime_series.dt.day,
                    f"{datetime_column}_hour": datetime_series.dt.hour,
                    f"{datetime_column}_minute": datetime_series.dt.minute,
                    f"{datetime_column}_dayofweek": datetime_series.dt.dayofweek,
                    f"{datetime_column}_dayofyear": datetime_series.dt.dayofyear,
                    f"{datetime_column}_quarter": datetime_series.dt.quarter,
                    f"{datetime_column}_is_weekend": datetime_series.dt.dayofweek.isin(
                        [5, 6]
                    ).astype(int),
                    f"{datetime_column}_is_month_end": datetime_series.dt.is_month_end.astype(
                        int
                    ),
                    f"{datetime_column}_is_month_start": datetime_series.dt.is_month_start.astype(
                        int
                    ),
                }
            )

            # Update stats
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.extraction_stats["total_extractions"] += 1
            self.extraction_stats["successful_extractions"] += 1
            self.extraction_stats["extraction_times"].append(execution_time)

            logger.info(
                f"Extracted {len(temporal_features.columns)} temporal features from {datetime_column}"
            )
            return temporal_features

        except Exception as e:
            self.extraction_stats["total_extractions"] += 1
            self.extraction_stats["failed_extractions"] += 1

            logger.error(f"Temporal feature extraction failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Temporal feature extraction failed: {str(e)}",
                cause=e,
            )

    async def extract_statistical_features(
        self,
        data: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        window_size: int = 5,
    ) -> pd.DataFrame:
        """Extract statistical features from numeric columns.

        Args:
            data: Input dataframe
            numeric_columns: List of numeric columns (all numeric if None)
            window_size: Window size for rolling statistics

        Returns:
            Dataframe with statistical features
        """
        try:
            start_time = datetime.now()

            if numeric_columns is None:
                numeric_columns = data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()

            statistical_features = pd.DataFrame(index=data.index)

            for col in numeric_columns:
                if col not in data.columns:
                    logger.warning(f"Column {col} not found in data")
                    continue

                series = data[col]

                # Basic statistics
                statistical_features[f"{col}_mean"] = series.rolling(
                    window=window_size
                ).mean()
                statistical_features[f"{col}_std"] = series.rolling(
                    window=window_size
                ).std()
                statistical_features[f"{col}_min"] = series.rolling(
                    window=window_size
                ).min()
                statistical_features[f"{col}_max"] = series.rolling(
                    window=window_size
                ).max()
                statistical_features[f"{col}_median"] = series.rolling(
                    window=window_size
                ).median()

                # Advanced statistics
                statistical_features[f"{col}_skew"] = series.rolling(
                    window=window_size
                ).skew()
                statistical_features[f"{col}_kurt"] = series.rolling(
                    window=window_size
                ).kurt()
                statistical_features[f"{col}_var"] = series.rolling(
                    window=window_size
                ).var()

                # Difference features
                statistical_features[f"{col}_diff"] = series.diff()
                statistical_features[f"{col}_pct_change"] = series.pct_change()

                # Lag features
                statistical_features[f"{col}_lag_1"] = series.shift(1)
                statistical_features[f"{col}_lag_2"] = series.shift(2)
                statistical_features[f"{col}_lag_3"] = series.shift(3)

            # Update stats
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.extraction_stats["total_extractions"] += 1
            self.extraction_stats["successful_extractions"] += 1
            self.extraction_stats["extraction_times"].append(execution_time)

            logger.info(
                f"Extracted {len(statistical_features.columns)} statistical features"
            )
            return statistical_features

        except Exception as e:
            self.extraction_stats["total_extractions"] += 1
            self.extraction_stats["failed_extractions"] += 1

            logger.error(f"Statistical feature extraction failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Statistical feature extraction failed: {str(e)}",
                cause=e,
            )

    async def extract_custom_features(
        self, data: pd.DataFrame, extractor_name: str, **kwargs
    ) -> pd.DataFrame:
        """Extract features using custom extractor.

        Args:
            data: Input dataframe
            extractor_name: Name of registered extractor
            **kwargs: Additional arguments for extractor

        Returns:
            Dataframe with extracted features
        """
        try:
            start_time = datetime.now()

            if extractor_name not in self.extractors:
                raise ValueError(f"Extractor {extractor_name} not found")

            extractor = self.extractors[extractor_name]
            extracted_features = extractor(data, **kwargs)

            # Update stats
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.extraction_stats["total_extractions"] += 1
            self.extraction_stats["successful_extractions"] += 1
            self.extraction_stats["extraction_times"].append(execution_time)

            logger.info(
                f"Extracted {len(extracted_features.columns)} features using {extractor_name}"
            )
            return extracted_features

        except Exception as e:
            self.extraction_stats["total_extractions"] += 1
            self.extraction_stats["failed_extractions"] += 1

            logger.error(f"Custom feature extraction failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Custom feature extraction failed: {str(e)}",
                cause=e,
            )

    async def get_extraction_stats(self) -> dict[str, Any]:
        """Get feature extraction statistics."""
        extraction_times = self.extraction_stats["extraction_times"]

        return {
            **self.extraction_stats,
            "success_rate": (
                self.extraction_stats["successful_extractions"]
                / self.extraction_stats["total_extractions"]
                if self.extraction_stats["total_extractions"] > 0
                else 0
            ),
            "average_extraction_time_ms": (
                sum(extraction_times) / len(extraction_times) if extraction_times else 0
            ),
            "registered_extractors": list(self.extractors.keys()),
        }


class FeatureTransformer:
    """Transform features for better anomaly detection."""

    def __init__(self):
        """Initialize feature transformer."""
        self.transformations: dict[str, Any] = {}
        self.transformation_stats = {
            "total_transformations": 0,
            "successful_transformations": 0,
            "failed_transformations": 0,
            "transformation_times": [],
        }

    async def normalize_features(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        method: str = "min_max",
    ) -> TransformationResult:
        """Normalize features to [0, 1] range.

        Args:
            data: Input dataframe
            columns: Columns to normalize (all numeric if None)
            method: Normalization method ('min_max' or 'z_score')

        Returns:
            Transformation result
        """
        start_time = datetime.now()

        try:
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()

            transformed_data = data.copy()
            parameters = {"method": method, "columns": columns}

            for col in columns:
                if col not in data.columns:
                    logger.warning(f"Column {col} not found in data")
                    continue

                series = data[col]

                if method == "min_max":
                    min_val = series.min()
                    max_val = series.max()
                    if max_val != min_val:
                        transformed_data[col] = (series - min_val) / (max_val - min_val)
                    else:
                        transformed_data[col] = 0.0

                    parameters[f"{col}_min"] = min_val
                    parameters[f"{col}_max"] = max_val

                elif method == "z_score":
                    mean_val = series.mean()
                    std_val = series.std()
                    if std_val != 0:
                        transformed_data[col] = (series - mean_val) / std_val
                    else:
                        transformed_data[col] = 0.0

                    parameters[f"{col}_mean"] = mean_val
                    parameters[f"{col}_std"] = std_val

                # Store transformation parameters
                self.transformations[f"normalize_{col}"] = parameters

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update stats
            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["successful_transformations"] += 1
            self.transformation_stats["transformation_times"].append(execution_time)

            logger.info(f"Normalized {len(columns)} features using {method} method")

            return TransformationResult(
                transformation_type=TransformationType.NORMALIZATION,
                feature_name=f"normalize_{method}",
                original_data=data[columns[0]] if columns else pd.Series(),
                transformed_data=transformed_data[columns[0]]
                if columns
                else pd.Series(),
                parameters=parameters,
                execution_time_ms=execution_time,
                success=True,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["failed_transformations"] += 1

            logger.error(f"Feature normalization failed: {e}")

            return TransformationResult(
                transformation_type=TransformationType.NORMALIZATION,
                feature_name=f"normalize_{method}",
                original_data=pd.Series(),
                transformed_data=pd.Series(),
                parameters={"method": method, "columns": columns or []},
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e),
            )

    async def encode_categorical_features(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        method: str = "one_hot",
    ) -> TransformationResult:
        """Encode categorical features.

        Args:
            data: Input dataframe
            columns: Columns to encode (all categorical if None)
            method: Encoding method ('one_hot', 'label', 'target')

        Returns:
            Transformation result
        """
        start_time = datetime.now()

        try:
            if columns is None:
                columns = data.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

            transformed_data = data.copy()
            parameters = {"method": method, "columns": columns}

            for col in columns:
                if col not in data.columns:
                    logger.warning(f"Column {col} not found in data")
                    continue

                series = data[col]

                if method == "one_hot":
                    # One-hot encoding
                    dummies = pd.get_dummies(series, prefix=col)
                    transformed_data = pd.concat(
                        [transformed_data.drop(col, axis=1), dummies], axis=1
                    )
                    parameters[f"{col}_categories"] = dummies.columns.tolist()

                elif method == "label":
                    # Label encoding
                    unique_values = series.unique()
                    label_map = {val: idx for idx, val in enumerate(unique_values)}
                    transformed_data[col] = series.map(label_map)
                    parameters[f"{col}_label_map"] = label_map

                elif method == "target":
                    # Target encoding (simplified - would need target variable)
                    unique_values = series.unique()
                    target_map = {val: np.random.random() for val in unique_values}
                    transformed_data[col] = series.map(target_map)
                    parameters[f"{col}_target_map"] = target_map

                # Store transformation parameters
                self.transformations[f"encode_{col}"] = parameters

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update stats
            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["successful_transformations"] += 1
            self.transformation_stats["transformation_times"].append(execution_time)

            logger.info(
                f"Encoded {len(columns)} categorical features using {method} method"
            )

            return TransformationResult(
                transformation_type=TransformationType.ENCODING,
                feature_name=f"encode_{method}",
                original_data=data[columns[0]] if columns else pd.Series(),
                transformed_data=transformed_data[columns[0]]
                if columns
                else pd.Series(),
                parameters=parameters,
                execution_time_ms=execution_time,
                success=True,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["failed_transformations"] += 1

            logger.error(f"Categorical encoding failed: {e}")

            return TransformationResult(
                transformation_type=TransformationType.ENCODING,
                feature_name=f"encode_{method}",
                original_data=pd.Series(),
                transformed_data=pd.Series(),
                parameters={"method": method, "columns": columns or []},
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e),
            )

    async def bin_features(
        self,
        data: pd.DataFrame,
        columns: list[str] | None = None,
        n_bins: int = 5,
        strategy: str = "uniform",
    ) -> TransformationResult:
        """Bin continuous features.

        Args:
            data: Input dataframe
            columns: Columns to bin (all numeric if None)
            n_bins: Number of bins
            strategy: Binning strategy ('uniform' or 'quantile')

        Returns:
            Transformation result
        """
        start_time = datetime.now()

        try:
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()

            transformed_data = data.copy()
            parameters = {"n_bins": n_bins, "strategy": strategy, "columns": columns}

            for col in columns:
                if col not in data.columns:
                    logger.warning(f"Column {col} not found in data")
                    continue

                series = data[col]

                if strategy == "uniform":
                    # Uniform binning
                    transformed_data[f"{col}_binned"] = pd.cut(
                        series, bins=n_bins, labels=False
                    )
                    bin_edges = pd.cut(series, bins=n_bins, retbins=True)[1]
                    parameters[f"{col}_bin_edges"] = bin_edges.tolist()

                elif strategy == "quantile":
                    # Quantile binning
                    transformed_data[f"{col}_binned"] = pd.qcut(
                        series, q=n_bins, labels=False, duplicates="drop"
                    )
                    quantiles = series.quantile(np.linspace(0, 1, n_bins + 1))
                    parameters[f"{col}_quantiles"] = quantiles.tolist()

                # Store transformation parameters
                self.transformations[f"bin_{col}"] = parameters

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update stats
            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["successful_transformations"] += 1
            self.transformation_stats["transformation_times"].append(execution_time)

            logger.info(f"Binned {len(columns)} features using {strategy} strategy")

            return TransformationResult(
                transformation_type=TransformationType.BINNING,
                feature_name=f"bin_{strategy}",
                original_data=data[columns[0]] if columns else pd.Series(),
                transformed_data=transformed_data[f"{columns[0]}_binned"]
                if columns
                else pd.Series(),
                parameters=parameters,
                execution_time_ms=execution_time,
                success=True,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["failed_transformations"] += 1

            logger.error(f"Feature binning failed: {e}")

            return TransformationResult(
                transformation_type=TransformationType.BINNING,
                feature_name=f"bin_{strategy}",
                original_data=pd.Series(),
                transformed_data=pd.Series(),
                parameters={
                    "n_bins": n_bins,
                    "strategy": strategy,
                    "columns": columns or [],
                },
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e),
            )

    async def get_transformation_stats(self) -> dict[str, Any]:
        """Get transformation statistics."""
        transformation_times = self.transformation_stats["transformation_times"]

        return {
            **self.transformation_stats,
            "success_rate": (
                self.transformation_stats["successful_transformations"]
                / self.transformation_stats["total_transformations"]
                if self.transformation_stats["total_transformations"] > 0
                else 0
            ),
            "average_transformation_time_ms": (
                sum(transformation_times) / len(transformation_times)
                if transformation_times
                else 0
            ),
            "stored_transformations": len(self.transformations),
        }


class FeatureSelector:
    """Select most relevant features for anomaly detection."""

    def __init__(self):
        """Initialize feature selector."""
        self.selection_stats = {
            "total_selections": 0,
            "features_selected": 0,
            "features_removed": 0,
            "selection_times": [],
        }

    async def select_by_variance(
        self, data: pd.DataFrame, threshold: float = 0.01
    ) -> list[str]:
        """Select features based on variance threshold.

        Args:
            data: Input dataframe
            threshold: Minimum variance threshold

        Returns:
            List of selected feature names
        """
        try:
            start_time = datetime.now()

            numeric_columns = data.select_dtypes(include=[np.number]).columns
            selected_features = []

            for col in numeric_columns:
                variance = data[col].var()
                if variance > threshold:
                    selected_features.append(col)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update stats
            self.selection_stats["total_selections"] += 1
            self.selection_stats["features_selected"] += len(selected_features)
            self.selection_stats["features_removed"] += len(numeric_columns) - len(
                selected_features
            )
            self.selection_stats["selection_times"].append(execution_time)

            logger.info(
                f"Selected {len(selected_features)} features based on variance threshold {threshold}"
            )
            return selected_features

        except Exception as e:
            logger.error(f"Variance-based feature selection failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Variance-based feature selection failed: {str(e)}",
                cause=e,
            )

    async def select_by_correlation(
        self, data: pd.DataFrame, threshold: float = 0.95
    ) -> list[str]:
        """Select features by removing highly correlated ones.

        Args:
            data: Input dataframe
            threshold: Correlation threshold for removal

        Returns:
            List of selected feature names
        """
        try:
            start_time = datetime.now()

            numeric_columns = data.select_dtypes(include=[np.number]).columns
            correlation_matrix = data[numeric_columns].corr().abs()

            # Find pairs of highly correlated features
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > threshold:
                        high_corr_pairs.append(
                            (
                                correlation_matrix.columns[i],
                                correlation_matrix.columns[j],
                            )
                        )

            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for feature1, feature2 in high_corr_pairs:
                # Remove the feature with higher mean correlation with other features
                corr1 = correlation_matrix[feature1].mean()
                corr2 = correlation_matrix[feature2].mean()

                if corr1 > corr2:
                    features_to_remove.add(feature1)
                else:
                    features_to_remove.add(feature2)

            selected_features = [
                col for col in numeric_columns if col not in features_to_remove
            ]

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update stats
            self.selection_stats["total_selections"] += 1
            self.selection_stats["features_selected"] += len(selected_features)
            self.selection_stats["features_removed"] += len(features_to_remove)
            self.selection_stats["selection_times"].append(execution_time)

            logger.info(
                f"Selected {len(selected_features)} features by removing {len(features_to_remove)} highly correlated features"
            )
            return selected_features

        except Exception as e:
            logger.error(f"Correlation-based feature selection failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Correlation-based feature selection failed: {str(e)}",
                cause=e,
            )

    async def select_top_k(
        self, data: pd.DataFrame, k: int = 10, method: str = "variance"
    ) -> list[str]:
        """Select top k features based on criteria.

        Args:
            data: Input dataframe
            k: Number of features to select
            method: Selection method ('variance', 'mean', 'std')

        Returns:
            List of selected feature names
        """
        try:
            start_time = datetime.now()

            numeric_columns = data.select_dtypes(include=[np.number]).columns

            if method == "variance":
                feature_scores = (
                    data[numeric_columns].var().sort_values(ascending=False)
                )
            elif method == "mean":
                feature_scores = (
                    data[numeric_columns].mean().abs().sort_values(ascending=False)
                )
            elif method == "std":
                feature_scores = (
                    data[numeric_columns].std().sort_values(ascending=False)
                )
            else:
                raise ValueError(f"Unknown selection method: {method}")

            selected_features = feature_scores.head(k).index.tolist()

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update stats
            self.selection_stats["total_selections"] += 1
            self.selection_stats["features_selected"] += len(selected_features)
            self.selection_stats["features_removed"] += len(numeric_columns) - len(
                selected_features
            )
            self.selection_stats["selection_times"].append(execution_time)

            logger.info(f"Selected top {k} features using {method} method")
            return selected_features

        except Exception as e:
            logger.error(f"Top-k feature selection failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Top-k feature selection failed: {str(e)}",
                cause=e,
            )

    async def get_selection_stats(self) -> dict[str, Any]:
        """Get feature selection statistics."""
        selection_times = self.selection_stats["selection_times"]

        return {
            **self.selection_stats,
            "average_selection_time_ms": (
                sum(selection_times) / len(selection_times) if selection_times else 0
            ),
            "selection_ratio": (
                self.selection_stats["features_selected"]
                / (
                    self.selection_stats["features_selected"]
                    + self.selection_stats["features_removed"]
                )
                if (
                    self.selection_stats["features_selected"]
                    + self.selection_stats["features_removed"]
                )
                > 0
                else 0
            ),
        }


class FeaturePipeline:
    """Complete feature engineering pipeline."""

    def __init__(self):
        """Initialize feature pipeline."""
        self.extractor = FeatureExtractor()
        self.transformer = FeatureTransformer()
        self.selector = FeatureSelector()
        self.pipeline_stats = {
            "total_pipelines": 0,
            "successful_pipelines": 0,
            "failed_pipelines": 0,
            "pipeline_times": [],
        }

    async def run_pipeline(
        self, data: pd.DataFrame, steps: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Run complete feature engineering pipeline.

        Args:
            data: Input dataframe
            steps: List of pipeline steps

        Returns:
            Processed dataframe
        """
        try:
            start_time = datetime.now()

            processed_data = data.copy()

            for step in steps:
                step_type = step.get("type")
                step_params = step.get("params", {})

                if step_type == "extract_temporal":
                    datetime_column = step_params.get("datetime_column")
                    if datetime_column:
                        temporal_features = (
                            await self.extractor.extract_temporal_features(
                                processed_data, datetime_column
                            )
                        )
                        processed_data = pd.concat(
                            [processed_data, temporal_features], axis=1
                        )

                elif step_type == "extract_statistical":
                    statistical_features = (
                        await self.extractor.extract_statistical_features(
                            processed_data, **step_params
                        )
                    )
                    processed_data = pd.concat(
                        [processed_data, statistical_features], axis=1
                    )

                elif step_type == "normalize":
                    result = await self.transformer.normalize_features(
                        processed_data, **step_params
                    )
                    if result.success:
                        processed_data = pd.DataFrame(result.transformed_data)

                elif step_type == "encode_categorical":
                    result = await self.transformer.encode_categorical_features(
                        processed_data, **step_params
                    )
                    if result.success:
                        processed_data = pd.DataFrame(result.transformed_data)

                elif step_type == "bin_features":
                    result = await self.transformer.bin_features(
                        processed_data, **step_params
                    )
                    if result.success:
                        processed_data = pd.DataFrame(result.transformed_data)

                elif step_type == "select_by_variance":
                    selected_features = await self.selector.select_by_variance(
                        processed_data, **step_params
                    )
                    processed_data = processed_data[selected_features]

                elif step_type == "select_by_correlation":
                    selected_features = await self.selector.select_by_correlation(
                        processed_data, **step_params
                    )
                    processed_data = processed_data[selected_features]

                elif step_type == "select_top_k":
                    selected_features = await self.selector.select_top_k(
                        processed_data, **step_params
                    )
                    processed_data = processed_data[selected_features]

                else:
                    logger.warning(f"Unknown pipeline step type: {step_type}")

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update stats
            self.pipeline_stats["total_pipelines"] += 1
            self.pipeline_stats["successful_pipelines"] += 1
            self.pipeline_stats["pipeline_times"].append(execution_time)

            logger.info(f"Pipeline completed successfully in {execution_time:.2f}ms")
            return processed_data

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.pipeline_stats["total_pipelines"] += 1
            self.pipeline_stats["failed_pipelines"] += 1
            self.pipeline_stats["pipeline_times"].append(execution_time)

            logger.error(f"Feature pipeline failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Feature pipeline failed: {str(e)}",
                cause=e,
            )

    async def get_pipeline_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        pipeline_times = self.pipeline_stats["pipeline_times"]

        return {
            **self.pipeline_stats,
            "success_rate": (
                self.pipeline_stats["successful_pipelines"]
                / self.pipeline_stats["total_pipelines"]
                if self.pipeline_stats["total_pipelines"] > 0
                else 0
            ),
            "average_pipeline_time_ms": (
                sum(pipeline_times) / len(pipeline_times) if pipeline_times else 0
            ),
            "extractor_stats": await self.extractor.get_extraction_stats(),
            "transformer_stats": await self.transformer.get_transformation_stats(),
            "selector_stats": await self.selector.get_selection_stats(),
        }


class FeatureEngineer:
    """Main feature engineering facade."""

    def __init__(self):
        """Initialize feature engineer."""
        self.pipeline = FeaturePipeline()
        self.feature_metadata: dict[str, FeatureMetadata] = {}

    async def analyze_features(self, data: pd.DataFrame) -> dict[str, FeatureMetadata]:
        """Analyze features and create metadata.

        Args:
            data: Input dataframe

        Returns:
            Dictionary of feature metadata
        """
        try:
            feature_metadata = {}

            for col in data.columns:
                series = data[col]

                # Determine feature type
                if series.dtype in ["int64", "float64"]:
                    feature_type = FeatureType.NUMERICAL
                elif series.dtype in ["object", "category"]:
                    feature_type = FeatureType.CATEGORICAL
                elif pd.api.types.is_datetime64_any_dtype(series):
                    feature_type = FeatureType.TEMPORAL
                else:
                    feature_type = FeatureType.CATEGORICAL

                # Calculate statistics
                missing_count = series.isnull().sum()
                missing_percentage = (missing_count / len(series)) * 100

                metadata = FeatureMetadata(
                    name=col,
                    type=feature_type,
                    unique_values=series.nunique(),
                    missing_count=missing_count,
                    missing_percentage=missing_percentage,
                )

                # Add numeric statistics
                if feature_type == FeatureType.NUMERICAL:
                    metadata.min_value = float(series.min())
                    metadata.max_value = float(series.max())
                    metadata.mean_value = float(series.mean())
                    metadata.std_value = float(series.std())

                feature_metadata[col] = metadata

            self.feature_metadata = feature_metadata
            logger.info(f"Analyzed {len(feature_metadata)} features")

            return feature_metadata

        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Feature analysis failed: {str(e)}",
                cause=e,
            )

    async def engineer_features(
        self, dataset: Dataset, pipeline_config: list[dict[str, Any]] | None = None
    ) -> Dataset:
        """Engineer features for anomaly detection.

        Args:
            dataset: Input dataset
            pipeline_config: Pipeline configuration

        Returns:
            Dataset with engineered features
        """
        try:
            # Default pipeline configuration
            if pipeline_config is None:
                pipeline_config = [
                    {"type": "extract_statistical", "params": {"window_size": 5}},
                    {"type": "normalize", "params": {"method": "min_max"}},
                    {"type": "encode_categorical", "params": {"method": "one_hot"}},
                    {"type": "select_by_variance", "params": {"threshold": 0.01}},
                ]

            # Run pipeline
            processed_data = await self.pipeline.run_pipeline(
                dataset.data, pipeline_config
            )

            # Create new dataset with engineered features
            engineered_dataset = Dataset(
                name=f"{dataset.name}_engineered",
                data=processed_data,
                description=f"Engineered features for {dataset.name}",
                target_column=dataset.target_column,
                metadata={
                    **dataset.metadata,
                    "feature_engineering": {
                        "pipeline_config": pipeline_config,
                        "original_features": len(dataset.data.columns),
                        "engineered_features": len(processed_data.columns),
                        "timestamp": datetime.now().isoformat(),
                    },
                },
            )

            # Update feature metadata
            await self.analyze_features(processed_data)

            logger.info(
                f"Feature engineering completed: {len(dataset.data.columns)} -> {len(processed_data.columns)} features"
            )
            return engineered_dataset

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Feature engineering failed: {str(e)}",
                cause=e,
            )

    async def get_feature_summary(self) -> dict[str, Any]:
        """Get summary of features and engineering process.

        Returns:
            Feature summary
        """
        if not self.feature_metadata:
            return {"error": "No feature metadata available"}

        # Calculate summary statistics
        total_features = len(self.feature_metadata)
        feature_types = {}
        total_missing = 0

        for metadata in self.feature_metadata.values():
            feature_type = metadata.type.value
            feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
            total_missing += metadata.missing_count

        pipeline_stats = await self.pipeline.get_pipeline_stats()

        return {
            "total_features": total_features,
            "feature_types": feature_types,
            "total_missing_values": total_missing,
            "average_missing_percentage": (
                sum(m.missing_percentage for m in self.feature_metadata.values())
                / total_features
                if total_features > 0
                else 0
            ),
            "pipeline_stats": pipeline_stats,
            "feature_metadata": {
                name: metadata.to_dict()
                for name, metadata in self.feature_metadata.items()
            },
        }


# Global feature engineer
_feature_engineer: FeatureEngineer | None = None


def get_feature_engineer() -> FeatureEngineer:
    """Get global feature engineer.

    Returns:
        Feature engineer instance
    """
    global _feature_engineer

    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer()

    return _feature_engineer
