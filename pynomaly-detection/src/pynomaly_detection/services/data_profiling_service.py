"""Data profiling service following single responsibility principle."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .interfaces.pipeline_services import DataProfile, IDataProfilingService

logger = logging.getLogger(__name__)


class DataProfilingService(IDataProfilingService):
    """Service for profiling datasets to understand their characteristics."""

    def __init__(self, include_advanced_analysis: bool = True):
        """Initialize data profiling service.

        Args:
            include_advanced_analysis: Whether to include expensive analysis
        """
        self.include_advanced_analysis = include_advanced_analysis

    async def profile_data(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> DataProfile:
        """Profile dataset to understand its characteristics.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            Data profile with characteristics
        """
        logger.info("📊 Profiling dataset characteristics")

        try:
            # Basic statistics
            basic_stats = self._calculate_basic_stats(X)

            # Feature analysis
            feature_analysis = self._analyze_features(X)

            # Data quality analysis
            data_quality = self._analyze_data_quality(X)

            # Advanced analysis if enabled
            sparsity_ratio = self._calculate_sparsity_ratio(X)
            missing_values_ratio = self._calculate_missing_values_ratio(X)
            complexity_score = self._calculate_complexity_score(
                X, sparsity_ratio, missing_values_ratio
            )

            # Include target analysis if available
            if y is not None:
                target_analysis = self._analyze_target_variable(y)
                basic_stats["target_analysis"] = target_analysis

            logger.info(
                f"Dataset profiling completed. Complexity score: {complexity_score:.3f}"
            )

            return DataProfile(
                basic_stats=basic_stats,
                feature_analysis=feature_analysis,
                data_quality=data_quality,
                sparsity_ratio=sparsity_ratio,
                missing_values_ratio=missing_values_ratio,
                complexity_score=complexity_score,
            )

        except Exception as e:
            logger.error(f"Data profiling failed: {e}")
            # Return minimal profile on error
            return DataProfile(
                basic_stats={"error": str(e)},
                feature_analysis={},
                data_quality={},
                sparsity_ratio=0.0,
                missing_values_ratio=0.0,
                complexity_score=0.0,
            )

    def _calculate_basic_stats(self, X: pd.DataFrame) -> dict[str, Any]:
        """Calculate basic dataset statistics."""
        n_samples, n_features = X.shape
        memory_usage_mb = X.memory_usage(deep=True).sum() / (1024 * 1024)

        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "memory_usage_mb": float(memory_usage_mb),
            "data_shape": f"{n_samples}x{n_features}",
            "memory_per_sample_kb": float(memory_usage_mb * 1024 / n_samples)
            if n_samples > 0
            else 0.0,
        }

    def _analyze_features(self, X: pd.DataFrame) -> dict[str, Any]:
        """Analyze feature types and characteristics."""
        # Categorize features by type
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_features = X.select_dtypes(include=["datetime"]).columns.tolist()
        boolean_features = X.select_dtypes(include=["bool"]).columns.tolist()

        # Feature type mapping
        feature_types = {}
        for col in X.columns:
            if col in numeric_features:
                feature_types[col] = "numeric"
            elif col in categorical_features:
                feature_types[col] = "categorical"
            elif col in datetime_features:
                feature_types[col] = "datetime"
            elif col in boolean_features:
                feature_types[col] = "boolean"
            else:
                feature_types[col] = "other"

        # Feature statistics for numeric columns
        numeric_stats = {}
        if numeric_features:
            numeric_data = X[numeric_features]
            numeric_stats = {
                "mean_values": numeric_data.mean().to_dict(),
                "std_values": numeric_data.std().to_dict(),
                "min_values": numeric_data.min().to_dict(),
                "max_values": numeric_data.max().to_dict(),
                "zero_counts": (numeric_data == 0).sum().to_dict(),
                "infinite_counts": np.isinf(numeric_data).sum().to_dict(),
            }

        # Categorical feature statistics
        categorical_stats = {}
        if categorical_features:
            for col in categorical_features:
                categorical_stats[col] = {
                    "unique_count": int(X[col].nunique()),
                    "mode": X[col].mode().iloc[0] if not X[col].mode().empty else None,
                    "most_frequent_count": int(X[col].value_counts().iloc[0])
                    if not X[col].value_counts().empty
                    else 0,
                }

        return {
            "feature_types": feature_types,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "datetime_features": datetime_features,
            "boolean_features": boolean_features,
            "numeric_count": len(numeric_features),
            "categorical_count": len(categorical_features),
            "datetime_count": len(datetime_features),
            "boolean_count": len(boolean_features),
            "numeric_statistics": numeric_stats,
            "categorical_statistics": categorical_stats,
        }

    def _analyze_data_quality(self, X: pd.DataFrame) -> dict[str, Any]:
        """Analyze data quality metrics."""
        # Missing values analysis
        missing_per_column = X.isnull().sum()
        missing_ratios = missing_per_column / len(X)

        # Duplicate analysis
        duplicate_count = X.duplicated().sum()
        duplicate_ratio = duplicate_count / len(X)

        # Constant features
        constant_features = []
        near_constant_features = []
        for col in X.columns:
            unique_count = X[col].nunique(dropna=False)
            if unique_count <= 1:
                constant_features.append(col)
            elif unique_count <= 2:
                near_constant_features.append(col)

        # Data consistency checks
        consistency_issues = []

        # Check for mixed data types in object columns
        for col in X.select_dtypes(include=["object"]).columns:
            sample_types = {type(val).__name__ for val in X[col].dropna().iloc[:100]}
            if len(sample_types) > 1:
                consistency_issues.append(
                    f"Mixed types in column '{col}': {sample_types}"
                )

        # Check for extreme outliers in numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        extreme_outliers = {}
        for col in numeric_cols:
            if X[col].std() > 0:  # Avoid division by zero
                z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
                extreme_outlier_count = (z_scores > 5).sum()
                if extreme_outlier_count > 0:
                    extreme_outliers[col] = int(extreme_outlier_count)

        return {
            "missing_values": {
                "total_missing": int(missing_per_column.sum()),
                "missing_per_column": missing_per_column.to_dict(),
                "missing_ratios": missing_ratios.to_dict(),
                "columns_with_missing": missing_per_column[
                    missing_per_column > 0
                ].index.tolist(),
                "fully_missing_columns": missing_ratios[
                    missing_ratios == 1.0
                ].index.tolist(),
            },
            "duplicates": {
                "duplicate_count": int(duplicate_count),
                "duplicate_ratio": float(duplicate_ratio),
                "unique_count": int(len(X) - duplicate_count),
            },
            "feature_variance": {
                "constant_features": constant_features,
                "near_constant_features": near_constant_features,
                "constant_count": len(constant_features),
                "near_constant_count": len(near_constant_features),
            },
            "consistency": {
                "issues": consistency_issues,
                "extreme_outliers": extreme_outliers,
                "issue_count": len(consistency_issues) + len(extreme_outliers),
            },
        }

    def _calculate_sparsity_ratio(self, X: pd.DataFrame) -> float:
        """Calculate sparsity ratio for numeric features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.0

        numeric_data = X[numeric_cols]
        zero_count = (numeric_data == 0).sum().sum()
        total_elements = len(X) * len(numeric_cols)

        return float(zero_count / total_elements) if total_elements > 0 else 0.0

    def _calculate_missing_values_ratio(self, X: pd.DataFrame) -> float:
        """Calculate overall missing values ratio."""
        total_missing = X.isnull().sum().sum()
        total_elements = len(X) * len(X.columns)

        return float(total_missing / total_elements) if total_elements > 0 else 0.0

    def _calculate_complexity_score(
        self, X: pd.DataFrame, sparsity_ratio: float, missing_values_ratio: float
    ) -> float:
        """Calculate dataset complexity score (0-1)."""
        complexity_factors = []

        # Size factor (30%)
        size_factor = min(len(X) / 10000, 1.0)
        complexity_factors.append(size_factor * 0.3)

        # Dimensionality factor (25%)
        dim_factor = min(len(X.columns) / 1000, 1.0)
        complexity_factors.append(dim_factor * 0.25)

        # Sparsity factor (20%)
        complexity_factors.append(sparsity_ratio * 0.2)

        # Missing data factor (15%)
        complexity_factors.append(missing_values_ratio * 0.15)

        # Feature type diversity factor (10%)
        numeric_ratio = len(X.select_dtypes(include=[np.number]).columns) / len(
            X.columns
        )
        categorical_ratio = len(
            X.select_dtypes(include=["object", "category"]).columns
        ) / len(X.columns)
        diversity_factor = 1.0 - abs(0.5 - numeric_ratio) - abs(0.5 - categorical_ratio)
        complexity_factors.append(max(diversity_factor, 0.0) * 0.1)

        return min(sum(complexity_factors), 1.0)

    def _analyze_target_variable(self, y: pd.Series) -> dict[str, Any]:
        """Analyze target variable characteristics."""
        unique_values = y.nunique()
        value_counts = y.value_counts()
        missing_count = y.isnull().sum()

        analysis = {
            "unique_values": int(unique_values),
            "missing_count": int(missing_count),
            "missing_ratio": float(missing_count / len(y)),
            "data_type": str(y.dtype),
        }

        # Classification vs regression detection
        if unique_values <= 20:  # Likely classification
            analysis["task_type"] = "classification"
            analysis["class_distribution"] = value_counts.to_dict()

            # Class imbalance analysis
            if len(value_counts) > 1:
                max_class_count = value_counts.max()
                min_class_count = value_counts.min()
                analysis["class_imbalance_ratio"] = float(
                    max_class_count / min_class_count
                )
                analysis["is_balanced"] = analysis["class_imbalance_ratio"] <= 2.0
            else:
                analysis["class_imbalance_ratio"] = 1.0
                analysis["is_balanced"] = True

        else:  # Likely regression
            analysis["task_type"] = "regression"
            if pd.api.types.is_numeric_dtype(y):
                analysis["target_statistics"] = {
                    "mean": float(y.mean()),
                    "std": float(y.std()),
                    "min": float(y.min()),
                    "max": float(y.max()),
                    "median": float(y.median()),
                }

        return analysis

    def get_profiling_summary(self, profile: DataProfile) -> dict[str, Any]:
        """Get a human-readable summary of the profiling results."""
        n_samples = profile.basic_stats["n_samples"]
        n_features = profile.basic_stats["n_features"]
        memory_mb = profile.basic_stats["memory_usage_mb"]

        summary = {
            "dataset_size": f"{n_samples:,} samples × {n_features} features",
            "memory_usage": f"{memory_mb:.1f} MB",
            "feature_breakdown": {
                "numeric": profile.feature_analysis["numeric_count"],
                "categorical": profile.feature_analysis["categorical_count"],
                "datetime": profile.feature_analysis["datetime_count"],
                "boolean": profile.feature_analysis["boolean_count"],
            },
            "data_quality_score": self._calculate_quality_score(profile),
            "key_insights": self._generate_insights(profile),
            "recommendations": self._generate_recommendations(profile),
        }

        return summary

    def _calculate_quality_score(self, profile: DataProfile) -> float:
        """Calculate overall data quality score from profile."""
        score_factors = []

        # Missing values factor (40%)
        missing_penalty = profile.missing_values_ratio
        score_factors.append((1.0 - missing_penalty) * 0.4)

        # Duplicate factor (20%)
        duplicate_ratio = profile.data_quality.get("duplicates", {}).get(
            "duplicate_ratio", 0
        )
        score_factors.append((1.0 - duplicate_ratio) * 0.2)

        # Constant features factor (20%)
        total_features = profile.basic_stats.get("n_features", 1)
        constant_count = profile.data_quality.get("feature_variance", {}).get(
            "constant_count", 0
        )
        constant_ratio = constant_count / total_features if total_features > 0 else 0
        score_factors.append((1.0 - constant_ratio) * 0.2)

        # Consistency factor (20%)
        consistency_issues = profile.data_quality.get("consistency", {}).get(
            "issue_count", 0
        )
        max_expected_issues = max(
            total_features * 0.1, 1
        )  # Allow up to 10% of features to have issues
        consistency_penalty = min(consistency_issues / max_expected_issues, 1.0)
        score_factors.append((1.0 - consistency_penalty) * 0.2)

        return min(sum(score_factors), 1.0)

    def _generate_insights(self, profile: DataProfile) -> list[str]:
        """Generate key insights from the profile."""
        insights = []

        # Dataset size insights
        n_samples = profile.basic_stats["n_samples"]
        n_features = profile.basic_stats["n_features"]

        if n_samples < 100:
            insights.append(
                "Very small dataset - may need more data for reliable models"
            )
        elif n_samples > 100000:
            insights.append("Large dataset - consider sampling for faster prototyping")

        if n_features > n_samples:
            insights.append(
                "High-dimensional dataset - feature selection may be beneficial"
            )

        # Feature type insights
        numeric_ratio = profile.feature_analysis["numeric_count"] / n_features
        if numeric_ratio < 0.3:
            insights.append(
                "Mostly categorical features - may need extensive preprocessing"
            )
        elif numeric_ratio > 0.9:
            insights.append("Mostly numeric features - good for most ML algorithms")

        # Data quality insights
        if profile.missing_values_ratio > 0.3:
            insights.append("High missing data ratio - imputation strategy needed")

        constant_count = profile.data_quality.get("feature_variance", {}).get(
            "constant_count", 0
        )
        if constant_count > 0:
            insights.append(f"{constant_count} constant features can be removed")

        # Complexity insights
        if profile.complexity_score > 0.7:
            insights.append("Complex dataset - may require advanced preprocessing")
        elif profile.complexity_score < 0.3:
            insights.append("Simple dataset - basic algorithms should work well")

        return insights

    def _generate_recommendations(self, profile: DataProfile) -> list[str]:
        """Generate actionable recommendations from the profile."""
        recommendations = []

        # Missing values recommendations
        if profile.missing_values_ratio > 0.1:
            if profile.missing_values_ratio > 0.5:
                recommendations.append("Consider collecting more complete data")
            else:
                recommendations.append("Implement missing value imputation strategy")

        # Feature engineering recommendations
        numeric_count = profile.feature_analysis["numeric_count"]
        categorical_count = profile.feature_analysis["categorical_count"]

        if categorical_count > 0:
            recommendations.append("Apply encoding techniques for categorical features")

        if numeric_count > 5:
            recommendations.append("Consider feature scaling/normalization")

        # Data cleaning recommendations
        duplicate_ratio = profile.data_quality.get("duplicates", {}).get(
            "duplicate_ratio", 0
        )
        if duplicate_ratio > 0.05:
            recommendations.append("Remove duplicate records before training")

        constant_count = profile.data_quality.get("feature_variance", {}).get(
            "constant_count", 0
        )
        if constant_count > 0:
            recommendations.append("Remove constant features to reduce dimensionality")

        # Performance recommendations
        if profile.basic_stats["memory_usage_mb"] > 1000:
            recommendations.append(
                "Consider data sampling or chunked processing for large dataset"
            )

        consistency_issues = profile.data_quality.get("consistency", {}).get(
            "issue_count", 0
        )
        if consistency_issues > 0:
            recommendations.append("Address data consistency issues before modeling")

        return recommendations
