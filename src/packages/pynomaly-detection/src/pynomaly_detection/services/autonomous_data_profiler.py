"""Autonomous data profiling and analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pynomaly.application.services.autonomous_detection_config import (
    AutonomousConfig,
    DataProfile,
)
from pynomaly.domain.entities import Dataset


class AutonomousDataProfiler:
    """Service for autonomous data profiling and analysis."""

    async def profile_data(
        self,
        dataset: Dataset,
        config: AutonomousConfig,
        initial_profile: DataProfile | None = None,
    ) -> DataProfile:
        """Profile dataset to understand its characteristics.

        Args:
            dataset: Dataset to profile
            config: Configuration options
            initial_profile: Initial profile with preprocessing info

        Returns:
            Complete data profile
        """
        df = dataset.data
        n_samples, n_features = df.shape

        # Sample data if too large
        if n_samples > config.max_samples_analysis:
            sample_df = df.sample(n=config.max_samples_analysis, random_state=42)
        else:
            sample_df = df

        # Basic statistics
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
        categorical_cols = sample_df.select_dtypes(
            include=["object", "category"]
        ).columns

        numeric_features = len(numeric_cols)
        categorical_features = len(categorical_cols)

        # Detect temporal features
        temporal_features = self._detect_temporal_features(sample_df)

        # Missing values
        missing_ratio = sample_df.isnull().sum().sum() / (
            sample_df.shape[0] * sample_df.shape[1]
        )

        # Data types
        data_types = {col: str(dtype) for col, dtype in sample_df.dtypes.items()}

        # Correlation analysis (numeric features only)
        correlation_score = self._calculate_correlation_score(sample_df, numeric_cols)

        # Sparsity analysis
        sparsity_ratio = self._calculate_sparsity_ratio(sample_df, numeric_cols)

        # Rough outlier estimation using IQR
        outlier_ratio_estimate = self._estimate_outlier_ratio(sample_df, numeric_cols)

        # Time series analysis
        seasonality_detected, trend_detected = self._analyze_time_series(
            sample_df, temporal_features, numeric_cols
        )

        # Recommended contamination rate
        recommended_contamination = min(0.1, max(0.01, outlier_ratio_estimate * 1.5))

        # Complexity score (0-1, higher = more complex)
        complexity_score = self._calculate_complexity_score(
            n_features,
            categorical_features,
            correlation_score,
            missing_ratio,
            sparsity_ratio,
        )

        # Create full profile with preprocessing information if available
        profile = DataProfile(
            n_samples=n_samples,
            n_features=n_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            temporal_features=temporal_features,
            missing_values_ratio=missing_ratio,
            data_types=data_types,
            correlation_score=correlation_score,
            sparsity_ratio=sparsity_ratio,
            outlier_ratio_estimate=outlier_ratio_estimate,
            seasonality_detected=seasonality_detected,
            trend_detected=trend_detected,
            recommended_contamination=recommended_contamination,
            complexity_score=complexity_score,
        )

        # Copy preprocessing information from initial profile if available
        if initial_profile:
            profile.quality_score = initial_profile.quality_score
            profile.quality_report = initial_profile.quality_report
            profile.preprocessing_recommended = (
                initial_profile.preprocessing_recommended
            )
            profile.preprocessing_applied = initial_profile.preprocessing_applied
            profile.preprocessing_metadata = initial_profile.preprocessing_metadata

        return profile

    def _detect_temporal_features(self, df: pd.DataFrame) -> int:
        """Detect temporal features in the dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Number of temporal features detected
        """
        temporal_features = 0
        for col in df.columns:
            if (
                df[col].dtype == "datetime64[ns]"
                or "date" in col.lower()
                or "time" in col.lower()
            ):
                temporal_features += 1
        return temporal_features

    def _calculate_correlation_score(
        self, df: pd.DataFrame, numeric_cols: pd.Index
    ) -> float:
        """Calculate correlation score for numeric features.

        Args:
            df: DataFrame to analyze
            numeric_cols: Numeric column names

        Returns:
            Average correlation score
        """
        if len(numeric_cols) <= 1:
            return 0.0

        corr_matrix = df[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        return upper_triangle.stack().mean()

    def _calculate_sparsity_ratio(
        self, df: pd.DataFrame, numeric_cols: pd.Index
    ) -> float:
        """Calculate sparsity ratio for numeric features.

        Args:
            df: DataFrame to analyze
            numeric_cols: Numeric column names

        Returns:
            Sparsity ratio (proportion of zeros)
        """
        if len(numeric_cols) == 0:
            return 0.0

        numeric_data = df[numeric_cols].values
        return np.count_nonzero(numeric_data == 0) / numeric_data.size

    def _estimate_outlier_ratio(
        self, df: pd.DataFrame, numeric_cols: pd.Index
    ) -> float:
        """Estimate outlier ratio using IQR method.

        Args:
            df: DataFrame to analyze
            numeric_cols: Numeric column names

        Returns:
            Estimated outlier ratio
        """
        if len(numeric_cols) == 0:
            return 0.0

        outlier_counts = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts.append(outliers)

        return max(outlier_counts) / len(df) if outlier_counts else 0.0

    def _analyze_time_series(
        self, df: pd.DataFrame, temporal_features: int, numeric_cols: pd.Index
    ) -> tuple[bool, bool]:
        """Analyze time series patterns.

        Args:
            df: DataFrame to analyze
            temporal_features: Number of temporal features
            numeric_cols: Numeric column names

        Returns:
            Tuple of (seasonality_detected, trend_detected)
        """
        seasonality_detected = False
        trend_detected = False

        if temporal_features > 0 and len(numeric_cols) > 0:
            # Simple trend detection
            for col in numeric_cols:
                values = df[col].dropna().values
                if len(values) > 10:
                    trend_coef = np.corrcoef(np.arange(len(values)), values)[0, 1]
                    if abs(trend_coef) > 0.3:
                        trend_detected = True
                        break

        return seasonality_detected, trend_detected

    def _calculate_complexity_score(
        self,
        n_features: int,
        categorical_features: int,
        correlation_score: float,
        missing_ratio: float,
        sparsity_ratio: float,
    ) -> float:
        """Calculate dataset complexity score.

        Args:
            n_features: Number of features
            categorical_features: Number of categorical features
            correlation_score: Average correlation score
            missing_ratio: Missing values ratio
            sparsity_ratio: Sparsity ratio

        Returns:
            Complexity score (0-1, higher = more complex)
        """
        complexity_factors = [
            n_features / 100,  # Feature count
            categorical_features / max(1, n_features),  # Categorical ratio
            correlation_score,  # Correlation complexity
            missing_ratio,  # Missing data complexity
            sparsity_ratio,  # Sparsity complexity
        ]
        return min(1.0, np.mean(complexity_factors))
