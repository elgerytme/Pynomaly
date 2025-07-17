"""Data validation service following single responsibility principle."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .interfaces.pipeline_services import DataValidationResult, IDataValidationService

logger = logging.getLogger(__name__)


class DataValidationService(IDataValidationService):
    """Service for validating data quality and generating quality measurements."""

    def __init__(self, min_quality_threshold: float = 0.7):
        """Initialize data validation service.

        Args:
            min_quality_threshold: Minimum quality score for validation to pass
        """
        self.min_quality_threshold = min_quality_threshold

    async def validate_data(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> DataValidationResult:
        """Validate input data and assess quality.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            Validation result with quality measurements
        """
        logger.info("🔍 Validating data quality")

        issues = []
        recommendations = []
        statistics = {}

        try:
            # Basic shape validation
            n_samples, n_features = X.shape
            logger.info(f"Data shape: {n_samples} samples, {n_features} features")

            # Check for empty data_collection
            if n_samples == 0 or n_features == 0:
                issues.append("Empty data_collection: no samples or features found")
                recommendations.append("Ensure data_collection contains data before validation")
                return DataValidationResult(
                    is_valid=False,
                    statistics={"n_samples": n_samples, "n_features": n_features},
                    quality_score=0.0,
                    issues=issues,
                    recommendations=recommendations,
                )

            # Check for minimum data requirements
            if n_samples < 10:
                issues.append("Insufficient data: less than 10 samples")
                recommendations.append("Collect more data samples")
            elif n_samples < 100:
                issues.append("Small data_collection: less than 100 samples")
                recommendations.append(
                    "Consider collecting more data for better processor performance"
                )

            if n_features < 1:
                issues.append("No features found in data_collection")
                recommendations.append("Ensure data_collection contains feature columns")

            # Missing value analysis
            missing_stats = self._analyze_missing_values(X)
            statistics["missing_values"] = missing_stats

            if missing_stats["total_missing_ratio"] > 0.5:
                issues.append("More than 50% of data is missing")
                recommendations.append("Review data collection process")
            elif missing_stats["total_missing_ratio"] > 0.2:
                issues.append("More than 20% of data is missing")
                recommendations.append("Consider advanced imputation techniques")

            # Data type analysis
            dtype_stats = self._analyze_data_types(X)
            statistics["data_types"] = dtype_stats

            # Duplicate analysis
            duplicate_stats = self._analyze_duplicates(X)
            statistics["duplicates"] = duplicate_stats

            if duplicate_stats["duplicate_ratio"] > 0.1:
                issues.append("More than 10% of data is duplicated")
                recommendations.append("Remove duplicate records")

            # Constant features analysis
            constant_stats = self._analyze_constant_features(X)
            statistics["constant_features"] = constant_stats

            if constant_stats["count"] > 0:
                issues.append(f"{constant_stats['count']} constant features found")
                recommendations.append("Remove constant features")

            # Statistical analysis
            numeric_stats = self._analyze_numeric_features(X)
            statistics["numeric"] = numeric_stats

            # Target analysis if available
            if y is not None:
                target_stats = self._analyze_target(y)
                statistics["target"] = target_stats

                if target_stats["class_imbalance_ratio"] > 10:
                    issues.append("Severe class imbalance detected")
                    recommendations.append("Consider rebalancing techniques")
                elif target_stats["class_imbalance_ratio"] > 3:
                    issues.append("Class imbalance detected")
                    recommendations.append("Monitor processor performance across classes")

            # Memory usage analysis
            memory_stats = self._analyze_memory_usage(X)
            statistics["memory"] = memory_stats

            if memory_stats["memory_usage_mb"] > 1000:
                issues.append("Large data_collection detected (>1GB)")
                recommendations.append("Consider data sampling or chunked processing")

            # Calculate overall quality score
            quality_score = self._calculate_quality_score(statistics)

            # Determine if validation passes
            is_valid = quality_score >= self.min_quality_threshold and len(issues) == 0

            # Add final recommendations based on quality score
            if quality_score < 0.3:
                recommendations.append(
                    "Data quality is poor - major issues need attention"
                )
            elif quality_score < 0.7:
                recommendations.append(
                    "Data quality is moderate - some improvements needed"
                )
            else:
                recommendations.append("Data quality is good")

            statistics["quality_score"] = quality_score
            statistics["n_samples"] = n_samples
            statistics["n_features"] = n_features

            logger.info(
                f"Data validation completed. Quality score: {quality_score:.3f}"
            )

            return DataValidationResult(
                is_valid=is_valid,
                statistics=statistics,
                quality_score=quality_score,
                issues=issues,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return DataValidationResult(
                is_valid=False,
                statistics={"error": str(e)},
                quality_score=0.0,
                issues=[f"Validation failed: {str(e)}"],
                recommendations=["Check data format and fix any data loading issues"],
            )

    def _analyze_missing_values(self, X: pd.DataFrame) -> dict[str, Any]:
        """Analyze missing values in the data_collection."""
        missing_per_column = X.isnull().sum()
        missing_ratios = missing_per_column / len(X)

        return {
            "total_missing_count": int(missing_per_column.sum()),
            "total_missing_ratio": float(
                missing_per_column.sum() / (len(X) * len(X.columns))
            ),
            "columns_with_missing": missing_per_column[
                missing_per_column > 0
            ].to_dict(),
            "missing_ratios": missing_ratios[missing_ratios > 0].to_dict(),
            "columns_fully_missing": list(missing_ratios[missing_ratios == 1.0].index),
        }

    def _analyze_data_types(self, X: pd.DataFrame) -> dict[str, Any]:
        """Analyze data types in the data_collection."""
        dtype_counts = X.dtypes.value_counts().to_dict()

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_cols = X.select_dtypes(include=["datetime"]).columns.tolist()

        return {
            "dtype_distribution": {str(k): int(v) for k, v in dtype_counts.items()},
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "numeric_count": len(numeric_cols),
            "categorical_count": len(categorical_cols),
            "datetime_count": len(datetime_cols),
        }

    def _analyze_duplicates(self, X: pd.DataFrame) -> dict[str, Any]:
        """Analyze duplicate records in the data_collection."""
        duplicate_count = X.duplicated().sum()

        return {
            "duplicate_count": int(duplicate_count),
            "duplicate_ratio": float(duplicate_count / len(X)),
            "unique_count": int(len(X) - duplicate_count),
        }

    def _analyze_constant_features(self, X: pd.DataFrame) -> dict[str, Any]:
        """Analyze constant features in the data_collection."""
        constant_features = []

        for col in X.columns:
            if X[col].nunique(dropna=False) <= 1:
                constant_features.append(col)

        return {
            "constant_features": constant_features,
            "count": len(constant_features),
        }

    def _analyze_numeric_features(self, X: pd.DataFrame) -> dict[str, Any]:
        """Analyze numeric features in the data_collection."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return {
                "count": 0,
                "statistics": {},
                "outlier_analysis": {},
                "correlation_analysis": {},
            }

        numeric_data = X[numeric_cols]

        # Basic statistics
        stats = numeric_data.describe().to_dict()

        # Outlier analysis using IQR method
        outlier_analysis = {}
        for col in numeric_cols:
            q1 = numeric_data[col].quantile(0.25)
            q3 = numeric_data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = (
                (numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)
            ).sum()
            outlier_analysis[col] = {
                "outlier_count": int(outliers),
                "outlier_ratio": float(outliers / len(numeric_data)),
            }

        # Correlation analysis
        correlation_analysis = {}
        if len(numeric_cols) > 1:
            corr_matrix = numeric_data.corr()

            # Find highly correlated pairs (>0.9)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.9:
                        high_corr_pairs.append(
                            {
                                "feature1": corr_matrix.columns[i],
                                "feature2": corr_matrix.columns[j],
                                "correlation": float(corr_val),
                            }
                        )

            correlation_analysis = {
                "high_correlation_pairs": high_corr_pairs,
                "max_correlation": float(
                    corr_matrix.abs()
                    .values[np.triu_indices_from(corr_matrix.values, k=1)]
                    .max()
                ),
            }

        return {
            "count": len(numeric_cols),
            "statistics": stats,
            "outlier_analysis": outlier_analysis,
            "correlation_analysis": correlation_analysis,
        }

    def _analyze_target(self, y: pd.Series) -> dict[str, Any]:
        """Analyze target variable."""
        value_counts = y.value_counts()

        analysis = {
            "unique_values": int(y.nunique()),
            "value_distribution": value_counts.to_dict(),
            "missing_count": int(y.isnull().sum()),
            "missing_ratio": float(y.isnull().sum() / len(y)),
        }

        # Class imbalance analysis for classification
        if y.nunique() <= 20:  # Assume classification if few unique values
            class_counts = value_counts.values
            if len(class_counts) > 1:
                max_class = class_counts.max()
                min_class = class_counts.min()
                analysis["class_imbalance_ratio"] = float(max_class / min_class)
                analysis["is_balanced"] = analysis["class_imbalance_ratio"] <= 2.0
            else:
                analysis["class_imbalance_ratio"] = 1.0
                analysis["is_balanced"] = True

        return analysis

    def _analyze_memory_usage(self, X: pd.DataFrame) -> dict[str, Any]:
        """Analyze memory usage of the data_collection."""
        memory_usage = X.memory_usage(deep=True)
        total_memory_bytes = memory_usage.sum()

        return {
            "memory_usage_bytes": int(total_memory_bytes),
            "memory_usage_mb": float(total_memory_bytes / (1024 * 1024)),
            "memory_per_column": memory_usage.to_dict(),
        }

    def _calculate_quality_score(self, statistics: dict[str, Any]) -> float:
        """Calculate overall data quality score (0-1)."""
        score_factors = []

        # Missing values factor (30%)
        missing_ratio = statistics.get("missing_values", {}).get(
            "total_missing_ratio", 0
        )
        missing_factor = 1.0 - missing_ratio
        score_factors.append(missing_factor * 0.3)

        # Duplicate factor (20%)
        duplicate_ratio = statistics.get("duplicates", {}).get("duplicate_ratio", 0)
        duplicate_factor = 1.0 - duplicate_ratio
        score_factors.append(duplicate_factor * 0.2)

        # Constant features factor (15%)
        n_features = statistics.get("n_features", 1)
        constant_count = statistics.get("constant_features", {}).get("count", 0)
        constant_factor = 1.0 - (constant_count / n_features)
        score_factors.append(constant_factor * 0.15)

        # Data size factor (15%)
        n_samples = statistics.get("n_samples", 0)
        if n_samples >= 1000:
            size_factor = 1.0
        elif n_samples >= 100:
            size_factor = 0.8
        elif n_samples >= 10:
            size_factor = 0.5
        else:
            size_factor = 0.1
        score_factors.append(size_factor * 0.15)

        # Target balance factor (20% if target exists, otherwise redistribute)
        target_stats = statistics.get("target")
        if target_stats:
            if target_stats.get("is_balanced", True):
                balance_factor = 1.0
            else:
                imbalance_ratio = target_stats.get("class_imbalance_ratio", 1.0)
                balance_factor = max(0.1, 1.0 - (imbalance_ratio - 1.0) / 10.0)
            score_factors.append(balance_factor * 0.2)
        else:
            # Redistribute weight to other factors
            redistribution = 0.2 / 4
            score_factors = [f + redistribution for f in score_factors]

        return min(sum(score_factors), 1.0)
