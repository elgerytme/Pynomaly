#!/usr/bin/env python3
"""
Data Processing Service - Handles data validation, profiling, and feature engineering
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from pynomaly_detection.application.services.automl_service import DatasetProfile
from pynomaly_detection.domain.models.pipeline_models import PipelineConfig

logger = logging.getLogger(__name__)


class DataProcessingService:
    """Service responsible for data processing operations"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def process_data(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """
        Process input data through validation, profiling, and feature engineering

        Returns:
            Dictionary containing processed data and metadata
        """

        # Validate data
        validation_result = await self.validate_data(X, y)
        if not validation_result["valid"]:
            raise ValueError(f"Data validation failed: {validation_result['issues']}")

        # Profile data
        profile = await self.profile_data(X, y)

        # Engineer features if enabled
        if self.config.enable_feature_engineering:
            X_processed = await self.engineer_features(X, y)
        else:
            X_processed = X.copy()

        return {
            "X": X_processed,
            "y": y,
            "validation_result": validation_result,
            "profile": profile,
        }

    async def validate_data(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Validate input data quality and characteristics"""

        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        # Basic validation
        if X.empty:
            validation_results["valid"] = False
            validation_results["issues"].append("Dataset is empty")
            return validation_results

        # Check for missing values
        missing_ratio = X.isnull().sum().sum() / (len(X) * len(X.columns))
        validation_results["statistics"]["missing_ratio"] = missing_ratio

        if missing_ratio > 0.5:
            validation_results["issues"].append(
                f"High missing value ratio: {missing_ratio:.2%}"
            )
        elif missing_ratio > 0.2:
            validation_results["warnings"].append(
                f"Moderate missing values: {missing_ratio:.2%}"
            )

        # Check data types
        numeric_ratio = len(X.select_dtypes(include=[np.number]).columns) / len(
            X.columns
        )
        validation_results["statistics"]["numeric_ratio"] = numeric_ratio

        if numeric_ratio < 0.5:
            validation_results["warnings"].append(
                "Low numeric feature ratio, may need encoding"
            )

        # Check sample size
        if len(X) < 100:
            validation_results["warnings"].append(
                "Small dataset size, consider collecting more data"
            )

        # Target validation
        if y is not None:
            target_stats = {
                "unique_values": len(y.unique()),
                "missing_ratio": y.isnull().sum() / len(y),
            }

            if target_stats["missing_ratio"] > 0:
                validation_results["issues"].append(
                    f"Missing target values: {target_stats['missing_ratio']:.2%}"
                )

            if target_stats["unique_values"] == 1:
                validation_results["issues"].append("Target has only one unique value")

            validation_results["statistics"]["target"] = target_stats

        # Memory usage check
        memory_usage = X.memory_usage(deep=True).sum() / (1024**3)  # GB
        validation_results["statistics"]["memory_usage_gb"] = memory_usage

        if memory_usage > self.config.max_memory_usage_gb:
            validation_results["warnings"].append(
                f"High memory usage: {memory_usage:.2f}GB"
            )

        return validation_results

    async def profile_data(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> DatasetProfile:
        """Profile the dataset to understand its characteristics"""

        # Feature analysis
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Calculate sparsity for numeric features
        sparsity_ratio = 0.0
        if numeric_features:
            sparsity_ratio = (X[numeric_features] == 0).sum().sum() / (
                len(X) * len(numeric_features)
            )

        # Calculate missing data ratio
        missing_ratio = X.isnull().sum().sum() / (len(X) * len(X.columns))

        # Create DatasetProfile
        profile = DatasetProfile(
            n_samples=len(X),
            n_features=len(X.columns),
            contamination_estimate=0.1,  # Default estimate
            feature_types={
                col: "numeric" if col in numeric_features else "categorical"
                for col in X.columns
            },
            missing_values_ratio=missing_ratio,
            categorical_features=categorical_features,
            numerical_features=numeric_features,
            time_series_features=[],  # Could be detected in future
            sparsity_ratio=sparsity_ratio,
            dimensionality_ratio=len(X.columns) / len(X),
            dataset_size_mb=X.memory_usage(deep=True).sum() / (1024**2),
        )

        return profile

    async def engineer_features(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> pd.DataFrame:
        """Perform automated feature engineering"""

        logger.info("🔨 Performing feature engineering")

        X_engineered = X.copy()
        original_features = list(X.columns)

        try:
            # Handle missing values first
            if X_engineered.isnull().any().any():
                X_engineered = self._handle_missing_values(X_engineered)

            # Get feature types
            numeric_cols = X_engineered.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            categorical_cols = X_engineered.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Engineer numeric features
            if numeric_cols:
                X_engineered = self._engineer_numeric_features(
                    X_engineered, numeric_cols
                )

            # Engineer categorical features
            if categorical_cols:
                X_engineered = self._engineer_categorical_features(
                    X_engineered, categorical_cols
                )

            # Create interaction features
            if len(numeric_cols) > 1 and len(numeric_cols) <= 10:
                X_engineered = self._create_interaction_features(
                    X_engineered, numeric_cols
                )

            # Feature selection
            X_engineered = self._select_features(X_engineered, y)

            logger.info(
                f"Feature engineering: {len(X.columns)} -> {len(X_engineered.columns)} features"
            )

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            X_engineered = X.copy()

        return X_engineered

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        X_filled = X.copy()

        for col in X_filled.columns:
            if X_filled[col].isnull().any():
                if X_filled[col].dtype in ["int64", "float64"]:
                    X_filled[col] = X_filled[col].fillna(X_filled[col].median())
                else:
                    mode_value = X_filled[col].mode()
                    if len(mode_value) > 0:
                        X_filled[col] = X_filled[col].fillna(mode_value[0])
                    else:
                        X_filled[col] = X_filled[col].fillna("unknown")

        return X_filled

    def _engineer_numeric_features(
        self, X: pd.DataFrame, numeric_cols: list[str]
    ) -> pd.DataFrame:
        """Engineer numeric features"""
        X_enhanced = X.copy()

        if len(numeric_cols) > 1:
            # Statistical aggregation features
            X_enhanced["numeric_mean"] = X[numeric_cols].mean(axis=1)
            X_enhanced["numeric_std"] = X[numeric_cols].std(axis=1)
            X_enhanced["numeric_max"] = X[numeric_cols].max(axis=1)
            X_enhanced["numeric_min"] = X[numeric_cols].min(axis=1)
            X_enhanced["numeric_range"] = (
                X_enhanced["numeric_max"] - X_enhanced["numeric_min"]
            )

        # Individual feature transformations (limited to avoid explosion)
        for col in numeric_cols[:5]:
            if X[col].std() > 0:
                # Log transform for positive values
                if (X[col] > 0).all():
                    X_enhanced[f"{col}_log"] = np.log1p(X[col])

                # Square transform
                X_enhanced[f"{col}_squared"] = X[col] ** 2

        return X_enhanced

    def _engineer_categorical_features(
        self, X: pd.DataFrame, categorical_cols: list[str]
    ) -> pd.DataFrame:
        """Engineer categorical features"""
        X_enhanced = X.copy()

        for col in categorical_cols:
            unique_count = X[col].nunique()

            if unique_count > 1 and unique_count <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(X[col], prefix=f"{col}_")
                X_enhanced = pd.concat([X_enhanced, dummies], axis=1)
            elif unique_count > 10:
                # Frequency encoding for high cardinality
                freq_encoding = X[col].value_counts().to_dict()
                X_enhanced[f"{col}_freq"] = X[col].map(freq_encoding)

        return X_enhanced

    def _create_interaction_features(
        self, X: pd.DataFrame, numeric_cols: list[str]
    ) -> pd.DataFrame:
        """Create interaction features between numeric columns"""
        X_enhanced = X.copy()

        # Limit interactions to prevent feature explosion
        max_interactions = min(self.config.max_feature_combinations, 20)
        interaction_count = 0

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                if interaction_count >= max_interactions:
                    break

                # Multiplication
                X_enhanced[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                interaction_count += 1

                # Addition
                if interaction_count < max_interactions:
                    X_enhanced[f"{col1}_plus_{col2}"] = X[col1] + X[col2]
                    interaction_count += 1

            if interaction_count >= max_interactions:
                break

        return X_enhanced

    def _select_features(self, X: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
        """Select features based on variance and correlation"""
        X_selected = X.copy()

        # Remove constant features
        constant_features = [
            col for col in X_selected.columns if X_selected[col].nunique() <= 1
        ]
        if constant_features:
            X_selected = X_selected.drop(columns=constant_features)
            logger.info(f"Removed {len(constant_features)} constant features")

        # Remove low variance features
        numeric_cols = X_selected.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            try:
                from sklearn.feature_selection import VarianceThreshold

                selector = VarianceThreshold(threshold=0.01)
                numeric_data = X_selected[numeric_cols]
                selected_numeric_mask = selector.fit(numeric_data).get_support()
                selected_numeric_cols = numeric_cols[selected_numeric_mask]

                # Keep selected numeric + all non-numeric
                non_numeric_cols = X_selected.select_dtypes(exclude=[np.number]).columns
                selected_features = list(selected_numeric_cols) + list(non_numeric_cols)
                X_selected = X_selected[selected_features]

            except Exception as e:
                logger.warning(f"Feature selection failed: {e}")

        return X_selected
