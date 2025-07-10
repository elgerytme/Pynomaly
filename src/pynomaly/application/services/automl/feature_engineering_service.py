#!/usr/bin/env python3
"""
AutoML Feature Engineering Service
Handles automated feature engineering, selection, and preprocessing
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """Service for automated feature engineering and selection."""

    def __init__(self, max_feature_combinations: int = 20):
        """Initialize feature engineering service.
        
        Args:
            max_feature_combinations: Maximum number of interaction features to create
        """
        self.max_feature_combinations = max_feature_combinations

    async def engineer_features(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """Perform comprehensive automated feature engineering.
        
        Args:
            X: Input dataframe
            y: Optional target series
            
        Returns:
            Engineered dataframe
        """
        logger.info("🔨 Performing feature engineering")
        
        X_engineered = X.copy()
        original_features = list(X.columns)
        engineered_features = []

        try:
            # Handle missing values first
            if X_engineered.isnull().any().any():
                logger.info("Handling missing values")
                X_engineered = self._handle_missing_values(X_engineered)

            # Get numeric and categorical columns
            numeric_cols = X_engineered.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            categorical_cols = X_engineered.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Numeric feature engineering
            if len(numeric_cols) > 0:
                logger.info(
                    f"Engineering features for {len(numeric_cols)} numeric columns"
                )
                X_engineered, new_numeric_features = self._engineer_numeric_features(
                    X_engineered, numeric_cols
                )
                engineered_features.extend(new_numeric_features)

            # Categorical feature engineering
            if len(categorical_cols) > 0:
                logger.info(
                    f"Engineering features for {len(categorical_cols)} categorical columns"
                )
                X_engineered, new_categorical_features = (
                    self._engineer_categorical_features(X_engineered, categorical_cols)
                )
                engineered_features.extend(new_categorical_features)

            # Interaction features (limited to avoid explosion)
            if len(numeric_cols) > 1 and len(numeric_cols) <= 10:
                logger.info("Creating interaction features")
                X_engineered, interaction_features = self._create_interaction_features(
                    X_engineered, numeric_cols
                )
                engineered_features.extend(interaction_features)

            # Feature selection
            X_engineered = self._select_features(X_engineered, y)

            # Final engineered features count
            final_engineered_features = [
                col for col in X_engineered.columns if col not in original_features
            ]

            logger.info(
                f"Feature engineering: {len(X.columns)} -> {len(X_engineered.columns)} features"
            )
            logger.info(f"Added {len(final_engineered_features)} engineered features")
            logger.info(
                f"Removed {len(original_features) + len(engineered_features) - len(X_engineered.columns)} low-variance features"
            )

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            logger.info("Returning original features")
            X_engineered = X.copy()

        return X_engineered

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        X_filled = X.copy()

        for col in X_filled.columns:
            if X_filled[col].isnull().any():
                if X_filled[col].dtype in ["int64", "float64"]:
                    # Numeric: fill with median
                    X_filled[col] = X_filled[col].fillna(X_filled[col].median())
                else:
                    # Categorical: fill with mode
                    mode_value = X_filled[col].mode()
                    if len(mode_value) > 0:
                        X_filled[col] = X_filled[col].fillna(mode_value[0])
                    else:
                        X_filled[col] = X_filled[col].fillna("unknown")

        return X_filled

    def _engineer_numeric_features(
        self, X: pd.DataFrame, numeric_cols: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """Engineer numeric features."""
        X_enhanced = X.copy()
        new_features = []

        if len(numeric_cols) > 1:
            # Statistical aggregation features
            X_enhanced["numeric_mean"] = X[numeric_cols].mean(axis=1)
            X_enhanced["numeric_std"] = X[numeric_cols].std(axis=1)
            X_enhanced["numeric_max"] = X[numeric_cols].max(axis=1)
            X_enhanced["numeric_min"] = X[numeric_cols].min(axis=1)
            X_enhanced["numeric_range"] = (
                X_enhanced["numeric_max"] - X_enhanced["numeric_min"]
            )
            X_enhanced["numeric_skew"] = X[numeric_cols].skew(axis=1)

            new_features.extend(
                [
                    "numeric_mean",
                    "numeric_std",
                    "numeric_max",
                    "numeric_min",
                    "numeric_range",
                    "numeric_skew",
                ]
            )

        # Individual feature transformations
        for col in numeric_cols[:5]:  # Limit to first 5 features to avoid explosion
            if X[col].std() > 0:  # Only if feature has variance
                # Log transform (for positive values)
                if (X[col] > 0).all():
                    X_enhanced[f"{col}_log"] = np.log1p(X[col])
                    new_features.append(f"{col}_log")

                # Square root transform (for non-negative values)
                if (X[col] >= 0).all():
                    X_enhanced[f"{col}_sqrt"] = np.sqrt(X[col])
                    new_features.append(f"{col}_sqrt")

                # Squared transform
                X_enhanced[f"{col}_squared"] = X[col] ** 2
                new_features.append(f"{col}_squared")

                # Binning (discretization)
                try:
                    X_enhanced[f"{col}_binned"] = pd.cut(X[col], bins=5, labels=False)
                    new_features.append(f"{col}_binned")
                except Exception:
                    pass

        return X_enhanced, new_features

    def _engineer_categorical_features(
        self, X: pd.DataFrame, categorical_cols: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """Engineer categorical features."""
        X_enhanced = X.copy()
        new_features = []

        for col in categorical_cols:
            # Count unique values
            unique_count = X[col].nunique()

            if (
                unique_count > 1 and unique_count <= 10
            ):  # One-hot encode low cardinality
                # One-hot encoding
                dummies = pd.get_dummies(X[col], prefix=f"{col}_")
                X_enhanced = pd.concat([X_enhanced, dummies], axis=1)
                new_features.extend(dummies.columns.tolist())
            elif unique_count > 10:  # Target encoding for high cardinality
                # Frequency encoding
                freq_encoding = X[col].value_counts().to_dict()
                X_enhanced[f"{col}_freq"] = X[col].map(freq_encoding)
                new_features.append(f"{col}_freq")

                # Label encoding
                try:
                    from sklearn.preprocessing import LabelEncoder

                    le = LabelEncoder()
                    X_enhanced[f"{col}_encoded"] = le.fit_transform(X[col].astype(str))
                    new_features.append(f"{col}_encoded")
                except Exception:
                    pass

        return X_enhanced, new_features

    def _create_interaction_features(
        self, X: pd.DataFrame, numeric_cols: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """Create interaction features between numeric columns."""
        X_enhanced = X.copy()
        new_features = []

        # Limit interactions to prevent feature explosion
        max_interactions = min(self.max_feature_combinations, 20)
        interaction_count = 0

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                if interaction_count >= max_interactions:
                    break

                # Multiplication
                X_enhanced[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                new_features.append(f"{col1}_x_{col2}")
                interaction_count += 1

                # Addition
                if interaction_count < max_interactions:
                    X_enhanced[f"{col1}_plus_{col2}"] = X[col1] + X[col2]
                    new_features.append(f"{col1}_plus_{col2}")
                    interaction_count += 1

                # Ratio (avoid division by zero)
                if interaction_count < max_interactions and (X[col2] != 0).all():
                    X_enhanced[f"{col1}_div_{col2}"] = X[col1] / X[col2]
                    new_features.append(f"{col1}_div_{col2}")
                    interaction_count += 1

            if interaction_count >= max_interactions:
                break

        return X_enhanced, new_features

    def _select_features(self, X: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
        """Select features based on variance and correlation."""
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

                # Apply to numeric features
                numeric_data = X_selected[numeric_cols]
                selected_numeric_mask = selector.fit(numeric_data).get_support()
                selected_numeric_cols = numeric_cols[selected_numeric_mask]

                # Keep selected numeric + all non-numeric
                non_numeric_cols = X_selected.select_dtypes(exclude=[np.number]).columns
                selected_features = list(selected_numeric_cols) + list(non_numeric_cols)
                X_selected = X_selected[selected_features]

                removed_count = len(numeric_cols) - len(selected_numeric_cols)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} low-variance features")

            except Exception as e:
                logger.warning(f"Feature selection failed: {e}")

        # Remove highly correlated features
        if len(numeric_cols) > 1:
            try:
                corr_matrix = X_selected[numeric_cols].corr().abs()
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )

                # Find features with correlation > threshold
                high_corr_features = [
                    column
                    for column in upper_tri.columns
                    if any(upper_tri[column] > 0.95)
                ]

                if high_corr_features:
                    X_selected = X_selected.drop(columns=high_corr_features)
                    logger.info(
                        f"Removed {len(high_corr_features)} highly correlated features"
                    )

            except Exception as e:
                logger.warning(f"Correlation-based feature removal failed: {e}")

        return X_selected

    def get_feature_importance(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> dict[str, float]:
        """Calculate feature importance using various methods.
        
        Args:
            X: Input features
            y: Optional target
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance_scores = {}

        try:
            # Statistical importance for numeric features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                # Variance-based importance
                variances = X[numeric_cols].var()
                max_var = variances.max() if len(variances) > 0 else 1.0
                
                for col in numeric_cols:
                    importance_scores[col] = variances[col] / max_var if max_var > 0 else 0.0

            # Categorical feature importance (based on cardinality and distribution)
            categorical_cols = X.select_dtypes(include=["object", "category"]).columns
            
            for col in categorical_cols:
                unique_ratio = X[col].nunique() / len(X)
                # Balanced cardinality is more important
                importance_scores[col] = 1.0 - abs(unique_ratio - 0.5)

            # Correlation with target if supervised
            if y is not None and len(numeric_cols) > 0:
                try:
                    correlations = X[numeric_cols].corrwith(y).abs()
                    for col in numeric_cols:
                        if col in correlations and not pd.isna(correlations[col]):
                            # Combine variance and correlation importance
                            variance_imp = importance_scores.get(col, 0.0)
                            corr_imp = correlations[col]
                            importance_scores[col] = (variance_imp + corr_imp) / 2
                except Exception as e:
                    logger.warning(f"Failed to calculate correlations: {e}")

        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")

        return importance_scores

    def create_feature_summary(self, X_original: pd.DataFrame, X_engineered: pd.DataFrame) -> dict[str, Any]:
        """Create a summary of feature engineering results.
        
        Args:
            X_original: Original dataframe
            X_engineered: Engineered dataframe
            
        Returns:
            Summary dictionary
        """
        original_features = set(X_original.columns)
        engineered_features = set(X_engineered.columns)
        
        added_features = engineered_features - original_features
        removed_features = original_features - engineered_features
        
        summary = {
            "original_feature_count": len(X_original.columns),
            "final_feature_count": len(X_engineered.columns),
            "features_added": len(added_features),
            "features_removed": len(removed_features),
            "added_feature_names": list(added_features),
            "removed_feature_names": list(removed_features),
            "feature_types": {
                "numeric": len(X_engineered.select_dtypes(include=[np.number]).columns),
                "categorical": len(X_engineered.select_dtypes(include=["object", "category"]).columns),
                "datetime": len(X_engineered.select_dtypes(include=["datetime64"]).columns),
            },
            "missing_values": X_engineered.isnull().sum().sum(),
            "memory_usage_mb": X_engineered.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        
        return summary