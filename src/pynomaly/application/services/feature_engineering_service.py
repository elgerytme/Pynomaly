"""Feature engineering service following single responsibility principle."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .interfaces.pipeline_services import (
    FeatureEngineeringResult,
    IFeatureEngineeringService,
)

logger = logging.getLogger(__name__)


class FeatureEngineeringService(IFeatureEngineeringService):
    """Service for automated feature engineering and feature selection."""

    def __init__(
        self,
        variance_threshold: float = 0.01,
        max_interaction_features: int = 50,
        enable_statistical_features: bool = True,
        enable_interaction_features: bool = True,
    ):
        """Initialize feature engineering service.

        Args:
            variance_threshold: Minimum variance for feature selection
            max_interaction_features: Maximum number of interaction features to create
            enable_statistical_features: Whether to create statistical features
            enable_interaction_features: Whether to create interaction features
        """
        self.variance_threshold = variance_threshold
        self.max_interaction_features = max_interaction_features
        self.enable_statistical_features = enable_statistical_features
        self.enable_interaction_features = enable_interaction_features

    async def engineer_features(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> FeatureEngineeringResult:
        """Engineer features for improved model performance.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            Feature engineering result with transformed data
        """
        logger.info("🔧 Engineering features for improved model performance")

        try:
            # Start with original data
            X_engineered = X.copy()
            original_features = list(X.columns)
            feature_metadata = {}

            # Statistical feature engineering
            if self.enable_statistical_features:
                X_engineered, stats_metadata = self._create_statistical_features(
                    X_engineered
                )
                feature_metadata.update(stats_metadata)

            # Interaction feature engineering
            if self.enable_interaction_features:
                X_engineered, interaction_metadata = self._create_interaction_features(
                    X_engineered, original_features
                )
                feature_metadata.update(interaction_metadata)

            # Feature selection
            X_selected, selected_features = self._select_features(X_engineered)

            # Identify engineered features
            engineered_features = [
                col for col in X_selected.columns if col not in original_features
            ]

            # Create detailed metadata
            feature_metadata.update(
                {
                    "original_feature_count": len(original_features),
                    "engineered_feature_count": len(engineered_features),
                    "total_feature_count": len(X_selected.columns),
                    "selection_method": "variance_threshold",
                    "variance_threshold": self.variance_threshold,
                }
            )

            logger.info(
                f"Feature engineering completed: {len(X.columns)} -> "
                f"{len(X_selected.columns)} features "
                f"({len(engineered_features)} engineered)"
            )

            return FeatureEngineeringResult(
                engineered_data=X_selected,
                selected_features=selected_features,
                engineered_features=engineered_features,
                feature_metadata=feature_metadata,
            )

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            # Return original data on error
            return FeatureEngineeringResult(
                engineered_data=X,
                selected_features=list(X.columns),
                engineered_features=[],
                feature_metadata={"error": str(e)},
            )

    def _create_statistical_features(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Create statistical features from numeric columns."""
        metadata = {"statistical_features": []}

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            return X, metadata

        # Row-wise statistical features
        if len(numeric_cols) > 1:
            numeric_data = X[numeric_cols]

            # Mean across features
            feature_name = "stats_row_mean"
            X[feature_name] = numeric_data.mean(axis=1)
            metadata["statistical_features"].append(feature_name)

            # Standard deviation across features
            feature_name = "stats_row_std"
            X[feature_name] = numeric_data.std(axis=1)
            metadata["statistical_features"].append(feature_name)

            # Maximum value across features
            feature_name = "stats_row_max"
            X[feature_name] = numeric_data.max(axis=1)
            metadata["statistical_features"].append(feature_name)

            # Minimum value across features
            feature_name = "stats_row_min"
            X[feature_name] = numeric_data.min(axis=1)
            metadata["statistical_features"].append(feature_name)

            # Range (max - min)
            feature_name = "stats_row_range"
            X[feature_name] = X["stats_row_max"] - X["stats_row_min"]
            metadata["statistical_features"].append(feature_name)

            # Skewness across features
            feature_name = "stats_row_skew"
            X[feature_name] = numeric_data.skew(axis=1)
            metadata["statistical_features"].append(feature_name)

            # Kurtosis across features
            feature_name = "stats_row_kurtosis"
            X[feature_name] = numeric_data.kurtosis(axis=1)
            metadata["statistical_features"].append(feature_name)

        # Column-wise transformations
        for col in numeric_cols:
            if X[col].std() > 0:  # Only transform if there's variance
                # Log transformation (for positive values)
                if (X[col] > 0).all():
                    feature_name = f"{col}_log"
                    X[feature_name] = np.log1p(X[col])
                    metadata["statistical_features"].append(feature_name)

                # Square root transformation (for non-negative values)
                if (X[col] >= 0).all():
                    feature_name = f"{col}_sqrt"
                    X[feature_name] = np.sqrt(X[col])
                    metadata["statistical_features"].append(feature_name)

                # Square transformation
                feature_name = f"{col}_squared"
                X[feature_name] = X[col] ** 2
                metadata["statistical_features"].append(feature_name)

        return X, metadata

    def _create_interaction_features(
        self, X: pd.DataFrame, original_features: list[str]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Create interaction features between numeric columns."""
        metadata = {"interaction_features": []}

        numeric_cols = [
            col
            for col in original_features
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col])
        ]

        if len(numeric_cols) < 2:
            return X, metadata

        interaction_count = 0

        # Create pairwise interactions
        for i, col1 in enumerate(numeric_cols):
            if interaction_count >= self.max_interaction_features:
                break

            for col2 in numeric_cols[i + 1 :]:
                if interaction_count >= self.max_interaction_features:
                    break

                # Multiplication interaction
                feature_name = f"{col1}_x_{col2}"
                X[feature_name] = X[col1] * X[col2]
                metadata["interaction_features"].append(feature_name)
                interaction_count += 1

                # Division interaction (avoid division by zero)
                if (X[col2] != 0).all():
                    feature_name = f"{col1}_div_{col2}"
                    X[feature_name] = X[col1] / X[col2]
                    metadata["interaction_features"].append(feature_name)
                    interaction_count += 1

                # Addition interaction
                feature_name = f"{col1}_plus_{col2}"
                X[feature_name] = X[col1] + X[col2]
                metadata["interaction_features"].append(feature_name)
                interaction_count += 1

                # Subtraction interaction
                feature_name = f"{col1}_minus_{col2}"
                X[feature_name] = X[col1] - X[col2]
                metadata["interaction_features"].append(feature_name)
                interaction_count += 1

        return X, metadata

    def _select_features(self, X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Select features based on variance threshold."""
        # Separate numeric and non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        selected_features = []

        # Apply variance threshold to numeric features
        if numeric_cols:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            numeric_data = X[numeric_cols]

            try:
                # Fit the selector
                selector.fit(numeric_data)
                selected_mask = selector.get_support()
                selected_numeric = [
                    col
                    for col, selected in zip(numeric_cols, selected_mask, strict=True)
                    if selected
                ]
                selected_features.extend(selected_numeric)

                logger.info(
                    f"Variance selection: {len(numeric_cols)} -> "
                    f"{len(selected_numeric)} numeric features"
                )

            except Exception as e:
                logger.warning(
                    f"Variance selection failed: {e}, keeping all numeric features"
                )
                selected_features.extend(numeric_cols)

        # Keep all non-numeric features
        selected_features.extend(non_numeric_cols)

        # Return selected data
        X_selected = X[selected_features]

        return X_selected, selected_features

    def _encode_categorical_features(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Encode categorical features using one-hot encoding."""
        metadata = {"categorical_encoding": {}}

        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if not categorical_cols:
            return X, metadata

        X_encoded = X.copy()

        for col in categorical_cols:
            # Only encode if reasonable number of unique values
            unique_values = X[col].nunique()

            if unique_values <= 20:  # Reasonable for one-hot encoding
                # Create dummy variables
                dummies = pd.get_dummies(X[col], prefix=col, dummy_na=True)

                # Add to dataframe
                X_encoded = pd.concat([X_encoded, dummies], axis=1)

                # Remove original column
                X_encoded = X_encoded.drop(columns=[col])

                metadata["categorical_encoding"][col] = {
                    "method": "one_hot",
                    "unique_values": unique_values,
                    "dummy_columns": list(dummies.columns),
                }

            else:
                # Too many unique values, use label encoding
                X_encoded[f"{col}_encoded"] = X[col].astype("category").cat.codes
                metadata["categorical_encoding"][col] = {
                    "method": "label_encoding",
                    "unique_values": unique_values,
                    "encoded_column": f"{col}_encoded",
                }

        return X_encoded, metadata

    def _handle_missing_values(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Handle missing values with appropriate imputation strategies."""
        metadata = {"missing_value_imputation": {}}

        X_imputed = X.copy()

        for col in X.columns:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Use median for numeric columns
                    fill_value = X[col].median()
                    imputation_method = "median"
                else:
                    # Use mode for categorical columns
                    fill_value = (
                        X[col].mode().iloc[0] if not X[col].mode().empty else "unknown"
                    )
                    imputation_method = "mode"

                X_imputed[col] = X[col].fillna(fill_value)

                metadata["missing_value_imputation"][col] = {
                    "missing_count": int(missing_count),
                    "missing_ratio": float(missing_count / len(X)),
                    "imputation_method": imputation_method,
                    "fill_value": fill_value,
                }

        return X_imputed, metadata

    def get_feature_importance_scores(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> dict[str, float]:
        """Calculate feature importance scores using various methods."""
        importance_scores = {}

        if y is None:
            # Use variance as importance for unsupervised case
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                importance_scores[col] = float(X[col].var())
        else:
            # Use correlation with target for supervised case
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    correlation = abs(X[col].corr(y))
                    importance_scores[col] = (
                        correlation if not np.isnan(correlation) else 0.0
                    )
                except Exception:
                    importance_scores[col] = 0.0

        return importance_scores

    def create_feature_summary(
        self, result: FeatureEngineeringResult
    ) -> dict[str, Any]:
        """Create a summary of the feature engineering process."""
        metadata = result.feature_metadata

        summary = {
            "transformation_summary": {
                "original_features": metadata.get("original_feature_count", 0),
                "engineered_features": metadata.get("engineered_feature_count", 0),
                "final_features": metadata.get("total_feature_count", 0),
                "feature_reduction_ratio": self._calculate_reduction_ratio(metadata),
            },
            "feature_types": {
                "statistical_features": len(metadata.get("statistical_features", [])),
                "interaction_features": len(metadata.get("interaction_features", [])),
            },
            "selection_info": {
                "method": metadata.get("selection_method", "none"),
                "variance_threshold": metadata.get("variance_threshold", 0),
            },
            "recommendations": self._generate_feature_recommendations(result),
        }

        return summary

    def _calculate_reduction_ratio(self, metadata: dict[str, Any]) -> float:
        """Calculate the feature reduction ratio."""
        original_count = metadata.get("original_feature_count", 1)
        final_count = metadata.get("total_feature_count", 1)

        if original_count == 0:
            return 0.0

        return (original_count - final_count) / original_count

    def _generate_feature_recommendations(
        self, result: FeatureEngineeringResult
    ) -> list[str]:
        """Generate recommendations based on feature engineering results."""
        recommendations = []
        metadata = result.feature_metadata

        # Feature count recommendations
        original_count = metadata.get("original_feature_count", 0)
        final_count = metadata.get("total_feature_count", 0)

        if final_count > original_count * 2:
            recommendations.append(
                "Consider more aggressive feature selection to reduce dimensionality"
            )

        if final_count < original_count * 0.5:
            recommendations.append(
                "Many features were removed - verify important information isn't lost"
            )

        # Statistical features recommendations
        stats_count = len(metadata.get("statistical_features", []))
        if stats_count > 0:
            recommendations.append(
                f"Created {stats_count} statistical features - monitor for overfitting"
            )

        # Interaction features recommendations
        interaction_count = len(metadata.get("interaction_features", []))
        if interaction_count > 0:
            recommendations.append(
                f"Created {interaction_count} interaction features - "
                "validate business meaning"
            )

        # Data quality recommendations
        if "error" in metadata:
            recommendations.append(
                "Feature engineering encountered errors - review data quality"
            )

        return recommendations
