"""Data profiling service for autonomous detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pynomaly.application.services.autonomous_preprocessing import (
    AutonomousPreprocessingOrchestrator,
    DataQualityReport,
)
from pynomaly.domain.entities import Dataset


@dataclass
class DataProfile:
    """Data profiling results."""

    n_samples: int
    n_features: int
    numeric_features: int
    categorical_features: int
    temporal_features: int
    missing_values_ratio: float
    data_types: dict[str, str]
    correlation_score: float
    sparsity_ratio: float
    outlier_ratio_estimate: float
    seasonality_detected: bool
    trend_detected: bool
    recommended_contamination: float
    complexity_score: float

    # Preprocessing-related fields
    quality_score: float = 1.0
    quality_report: DataQualityReport | None = None
    preprocessing_recommended: bool = False
    preprocessing_applied: bool = False
    preprocessing_metadata: dict[str, Any] | None = None


class DataProfilingService:
    """Service responsible for data profiling and quality assessment."""

    def __init__(self, preprocessing_orchestrator: AutonomousPreprocessingOrchestrator | None = None):
        """Initialize data profiling service.

        Args:
            preprocessing_orchestrator: Orchestrator for preprocessing operations
        """
        self.preprocessing_orchestrator = preprocessing_orchestrator or AutonomousPreprocessingOrchestrator()
        self.logger = logging.getLogger(__name__)

    async def profile_dataset(
        self,
        dataset: Dataset,
        max_samples: int = 10000,
        verbose: bool = False,
        existing_profile: DataProfile | None = None,
    ) -> DataProfile:
        """Profile dataset and analyze characteristics.

        Args:
            dataset: Dataset to profile
            max_samples: Maximum samples to analyze
            verbose: Enable verbose logging
            existing_profile: Existing profile to update (if any)

        Returns:
            Complete data profile
        """
        if verbose:
            self.logger.info(f"Profiling dataset: {dataset.name}")

        df = dataset.data

        # Sample data if too large
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            if verbose:
                self.logger.info(f"Sampled {max_samples} rows for profiling")

        # Basic statistics
        basic_stats = self._calculate_basic_statistics(df)

        # Data types analysis
        type_analysis = self._analyze_data_types(df)

        # Missing values analysis
        missing_analysis = self._analyze_missing_values(df)

        # Correlation analysis
        correlation_analysis = self._analyze_correlations(df)

        # Outlier analysis
        outlier_analysis = self._analyze_outliers(df)

        # Temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(df)

        # Complexity analysis
        complexity_analysis = self._calculate_complexity_score(df)

        # Contamination recommendation
        contamination_recommendation = self._recommend_contamination_rate(df, outlier_analysis)

        # Create profile
        profile = DataProfile(
            n_samples=len(dataset.data),
            n_features=len(df.columns),
            numeric_features=type_analysis["numeric_count"],
            categorical_features=type_analysis["categorical_count"],
            temporal_features=type_analysis["temporal_count"],
            missing_values_ratio=missing_analysis["missing_ratio"],
            data_types=type_analysis["data_types"],
            correlation_score=correlation_analysis["avg_correlation"],
            sparsity_ratio=missing_analysis["sparsity_ratio"],
            outlier_ratio_estimate=outlier_analysis["outlier_ratio"],
            seasonality_detected=temporal_analysis["seasonality_detected"],
            trend_detected=temporal_analysis["trend_detected"],
            recommended_contamination=contamination_recommendation,
            complexity_score=complexity_analysis["complexity_score"],
        )

        # Update existing profile if provided
        if existing_profile:
            profile.quality_score = existing_profile.quality_score
            profile.quality_report = existing_profile.quality_report
            profile.preprocessing_recommended = existing_profile.preprocessing_recommended
            profile.preprocessing_applied = existing_profile.preprocessing_applied
            profile.preprocessing_metadata = existing_profile.preprocessing_metadata

        if verbose:
            self.logger.info(f"Profiling completed: {profile.n_samples} samples, {profile.n_features} features")

        return profile

    def _calculate_basic_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate basic dataset statistics.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of basic statistics
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            return {
                "n_samples": len(df),
                "n_features": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "numeric_features": len(numeric_cols),
                "categorical_features": len(df.columns) - len(numeric_cols),
                "mean_values": df[numeric_cols].mean().to_dict() if len(numeric_cols) > 0 else {},
                "std_values": df[numeric_cols].std().to_dict() if len(numeric_cols) > 0 else {},
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate basic statistics: {e}")
            return {"n_samples": len(df), "n_features": len(df.columns)}

    def _analyze_data_types(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze data types and categorize features.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of data type analysis
        """
        try:
            data_types = {}
            numeric_count = 0
            categorical_count = 0
            temporal_count = 0

            for col in df.columns:
                dtype = str(df[col].dtype)

                if pd.api.types.is_numeric_dtype(df[col]):
                    data_types[col] = "numeric"
                    numeric_count += 1
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    data_types[col] = "temporal"
                    temporal_count += 1
                elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == "object":
                    # Check if it's actually temporal
                    if self._is_temporal_column(df[col]):
                        data_types[col] = "temporal"
                        temporal_count += 1
                    else:
                        data_types[col] = "categorical"
                        categorical_count += 1
                else:
                    data_types[col] = "other"
                    categorical_count += 1

            return {
                "data_types": data_types,
                "numeric_count": numeric_count,
                "categorical_count": categorical_count,
                "temporal_count": temporal_count,
            }
        except Exception as e:
            self.logger.warning(f"Failed to analyze data types: {e}")
            return {"data_types": {}, "numeric_count": 0, "categorical_count": 0, "temporal_count": 0}

    def _is_temporal_column(self, series: pd.Series) -> bool:
        """Check if a series contains temporal data.

        Args:
            series: Series to check

        Returns:
            True if series contains temporal data
        """
        try:
            # Try to parse as datetime
            sample_size = min(100, len(series))
            sample = series.dropna().head(sample_size)

            if len(sample) == 0:
                return False

            # Check for common datetime patterns
            datetime_patterns = [
                r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
                r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
                r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
                r"\d{2}-\d{2}-\d{4}",  # MM-DD-YYYY
            ]

            import re
            for pattern in datetime_patterns:
                if any(re.search(pattern, str(val)) for val in sample):
                    return True

            # Try pandas datetime conversion
            try:
                pd.to_datetime(sample)
                return True
            except:
                return False

        except Exception:
            return False

    def _analyze_missing_values(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze missing values in the dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of missing values analysis
        """
        try:
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            missing_ratio = missing_cells / total_cells if total_cells > 0 else 0

            missing_by_column = df.isnull().sum().to_dict()
            missing_patterns = {}

            # Analyze sparsity (ratio of zero/empty values)
            sparse_cells = 0
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    sparse_cells += (df[col] == 0).sum()
                elif df[col].dtype == "object":
                    sparse_cells += (df[col] == "").sum()

            sparsity_ratio = sparse_cells / total_cells if total_cells > 0 else 0

            return {
                "missing_ratio": missing_ratio,
                "missing_cells": missing_cells,
                "missing_by_column": missing_by_column,
                "missing_patterns": missing_patterns,
                "sparsity_ratio": sparsity_ratio,
            }
        except Exception as e:
            self.logger.warning(f"Failed to analyze missing values: {e}")
            return {"missing_ratio": 0, "sparsity_ratio": 0}

    def _analyze_correlations(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze feature correlations.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of correlation analysis
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) < 2:
                return {"avg_correlation": 0.0, "max_correlation": 0.0, "correlation_matrix": {}}

            corr_matrix = df[numeric_cols].corr()

            # Get upper triangle (excluding diagonal)
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            correlations = upper_triangle.stack().abs()

            return {
                "avg_correlation": correlations.mean(),
                "max_correlation": correlations.max(),
                "correlation_matrix": corr_matrix.to_dict(),
                "highly_correlated_pairs": self._find_highly_correlated_pairs(corr_matrix),
            }
        except Exception as e:
            self.logger.warning(f"Failed to analyze correlations: {e}")
            return {"avg_correlation": 0.0, "max_correlation": 0.0}

    def _find_highly_correlated_pairs(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> list[tuple]:
        """Find highly correlated feature pairs.

        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold

        Returns:
            List of highly correlated pairs
        """
        try:
            pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) >= threshold:
                        pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
            return pairs
        except Exception as e:
            self.logger.warning(f"Failed to find correlated pairs: {e}")
            return []

    def _analyze_outliers(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze outliers in the dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of outlier analysis
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                return {"outlier_ratio": 0.0, "outliers_by_column": {}}

            total_outliers = 0
            outliers_by_column = {}

            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue

                # Use IQR method to detect outliers
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                outliers_by_column[col] = outliers
                total_outliers += outliers

            outlier_ratio = total_outliers / (len(df) * len(numeric_cols)) if len(numeric_cols) > 0 else 0

            return {
                "outlier_ratio": outlier_ratio,
                "outliers_by_column": outliers_by_column,
                "total_outliers": total_outliers,
            }
        except Exception as e:
            self.logger.warning(f"Failed to analyze outliers: {e}")
            return {"outlier_ratio": 0.0}

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze temporal patterns in the dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of temporal analysis
        """
        try:
            temporal_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or self._is_temporal_column(df[col]):
                    temporal_cols.append(col)

            if not temporal_cols:
                return {"seasonality_detected": False, "trend_detected": False}

            seasonality_detected = False
            trend_detected = False

            # Analyze first temporal column
            if temporal_cols:
                temporal_series = df[temporal_cols[0]].dropna()
                if len(temporal_series) > 10:
                    # Simple trend detection using linear regression
                    try:
                        if pd.api.types.is_datetime64_any_dtype(temporal_series):
                            numeric_time = pd.to_numeric(temporal_series)
                            slope, _, _, p_value, _ = stats.linregress(
                                range(len(numeric_time)), numeric_time
                            )
                            trend_detected = p_value < 0.05 and abs(slope) > 0

                        # Basic seasonality detection (simplified)
                        if len(temporal_series) > 24:
                            # Check for repeating patterns
                            seasonality_detected = self._detect_seasonality(temporal_series)
                    except Exception:
                        pass

            return {
                "seasonality_detected": seasonality_detected,
                "trend_detected": trend_detected,
                "temporal_columns": temporal_cols,
            }
        except Exception as e:
            self.logger.warning(f"Failed to analyze temporal patterns: {e}")
            return {"seasonality_detected": False, "trend_detected": False}

    def _detect_seasonality(self, series: pd.Series) -> bool:
        """Detect seasonality in a time series.

        Args:
            series: Time series to analyze

        Returns:
            True if seasonality is detected
        """
        try:
            # Simple autocorrelation check
            if len(series) < 50:
                return False

            # Convert to numeric if needed
            if pd.api.types.is_datetime64_any_dtype(series):
                numeric_series = pd.to_numeric(series)
            else:
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()

            if len(numeric_series) < 50:
                return False

            # Check autocorrelation at various lags
            autocorr_values = []
            for lag in [7, 12, 24, 30, 365]:  # Common seasonal periods
                if lag < len(numeric_series):
                    autocorr = numeric_series.autocorr(lag=lag)
                    if not np.isnan(autocorr):
                        autocorr_values.append(abs(autocorr))

            # If any autocorrelation is high, consider it seasonal
            return any(autocorr > 0.3 for autocorr in autocorr_values)
        except Exception:
            return False

    def _calculate_complexity_score(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate dataset complexity score.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of complexity analysis
        """
        try:
            # Factors contributing to complexity
            factors = {}

            # Size factor
            size_factor = min(len(df) / 100000, 1.0)  # Normalize to 0-1
            factors["size"] = size_factor

            # Dimensionality factor
            dim_factor = min(len(df.columns) / 1000, 1.0)  # Normalize to 0-1
            factors["dimensionality"] = dim_factor

            # Missing values factor
            missing_factor = df.isnull().sum().sum() / (len(df) * len(df.columns))
            factors["missing_values"] = missing_factor

            # Data type diversity factor
            unique_types = len(set(str(dtype) for dtype in df.dtypes))
            type_factor = min(unique_types / 10, 1.0)  # Normalize to 0-1
            factors["type_diversity"] = type_factor

            # Categorical complexity factor
            categorical_complexity = 0
            for col in df.columns:
                if df[col].dtype == "object":
                    unique_values = df[col].nunique()
                    categorical_complexity += min(unique_values / 100, 1.0)

            if len(df.columns) > 0:
                categorical_complexity /= len(df.columns)
            factors["categorical_complexity"] = categorical_complexity

            # Overall complexity score (weighted average)
            complexity_score = (
                0.2 * size_factor +
                0.2 * dim_factor +
                0.2 * missing_factor +
                0.2 * type_factor +
                0.2 * categorical_complexity
            )

            return {
                "complexity_score": complexity_score,
                "factors": factors,
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate complexity score: {e}")
            return {"complexity_score": 0.5}

    def _recommend_contamination_rate(self, df: pd.DataFrame, outlier_analysis: dict[str, Any]) -> float:
        """Recommend contamination rate based on data analysis.

        Args:
            df: DataFrame to analyze
            outlier_analysis: Results from outlier analysis

        Returns:
            Recommended contamination rate
        """
        try:
            # Base contamination rate
            base_rate = 0.1  # Default 10%

            # Adjust based on outlier analysis
            outlier_ratio = outlier_analysis.get("outlier_ratio", 0.1)

            # Use outlier ratio as a guide, but constrain it
            if outlier_ratio > 0:
                # Use outlier ratio but clamp between 0.01 and 0.5
                recommended_rate = max(0.01, min(0.5, outlier_ratio))
            else:
                recommended_rate = base_rate

            # Adjust based on dataset size
            if len(df) < 1000:
                # For small datasets, use slightly higher contamination
                recommended_rate = min(recommended_rate * 1.5, 0.2)
            elif len(df) > 100000:
                # For large datasets, use slightly lower contamination
                recommended_rate = max(recommended_rate * 0.8, 0.01)

            return recommended_rate
        except Exception as e:
            self.logger.warning(f"Failed to recommend contamination rate: {e}")
            return 0.1

    def create_summary_report(self, profile: DataProfile) -> str:
        """Create a human-readable summary report.

        Args:
            profile: Data profile to summarize

        Returns:
            Human-readable summary report
        """
        try:
            report_lines = [
                "=== DATA PROFILE SUMMARY ===",
                f"Dataset Size: {profile.n_samples:,} samples × {profile.n_features} features",
                f"Feature Types: {profile.numeric_features} numeric, {profile.categorical_features} categorical, {profile.temporal_features} temporal",
                f"Data Quality: {profile.missing_values_ratio:.1%} missing values, {profile.sparsity_ratio:.1%} sparse values",
                f"Correlations: Average correlation {profile.correlation_score:.3f}",
                f"Outliers: {profile.outlier_ratio_estimate:.1%} estimated outlier ratio",
                f"Complexity: {profile.complexity_score:.3f} complexity score",
                f"Recommended Contamination: {profile.recommended_contamination:.1%}",
            ]

            if profile.temporal_features > 0:
                report_lines.append(f"Temporal Patterns: Seasonality {'detected' if profile.seasonality_detected else 'not detected'}, Trend {'detected' if profile.trend_detected else 'not detected'}")

            if profile.preprocessing_applied:
                report_lines.append(f"Preprocessing: Applied (Quality Score: {profile.quality_score:.3f})")

            return "\n".join(report_lines)
        except Exception as e:
            self.logger.warning(f"Failed to create summary report: {e}")
            return "Failed to generate summary report"
