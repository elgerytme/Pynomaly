"""Feature processing and engineering implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, LabelEncoder, OneHotEncoder
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, chi2, 
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    Processor for feature engineering and transformation operations.
    
    Provides comprehensive feature engineering capabilities including
    scaling, encoding, feature generation, and feature selection.
    """
    
    def __init__(self) -> None:
        """Initialize the feature processor."""
        self._scalers: Dict[str, Any] = {}
        self._encoders: Dict[str, Any] = {}
        self._feature_selectors: Dict[str, Any] = {}
    
    def scale_features(
        self,
        data: pd.DataFrame,
        method: str = "standard",
        columns: Optional[List[str]] = None,
        fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features using various scaling methods.
        
        Args:
            data: Input dataframe
            method: Scaling method ('standard', 'minmax', 'robust', 'quantile', 'power')
            columns: Specific columns to scale (None for all numerical)
            fit_scaler: Whether to fit a new scaler or use existing
            
        Returns:
            DataFrame with scaled features
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out non-existent columns
        columns = [col for col in columns if col in data.columns]
        
        if not columns:
            logger.warning("No numerical columns found for scaling")
            return data.copy()
        
        logger.info(f"Scaling features using {method} method: {columns}")
        
        result_data = data.copy()
        
        # Get or create scaler
        scaler_key = f"{method}_scaler"
        
        if fit_scaler or scaler_key not in self._scalers:
            scaler = self._create_scaler(method)
            # Fit scaler on the data
            scaler.fit(data[columns])
            self._scalers[scaler_key] = scaler
        else:
            scaler = self._scalers[scaler_key]
        
        # Transform the data
        scaled_data = scaler.transform(data[columns])
        result_data[columns] = scaled_data
        
        logger.info(f"Successfully scaled {len(columns)} features")
        return result_data
    
    def encode_categorical_features(
        self,
        data: pd.DataFrame,
        strategy: str = "onehot",
        columns: Optional[List[str]] = None,
        max_categories: int = 20
    ) -> pd.DataFrame:
        """
        Encode categorical features using various encoding strategies.
        
        Args:
            data: Input dataframe
            strategy: Encoding strategy ('onehot', 'label', 'target', 'frequency')
            columns: Specific columns to encode (None for all categorical)
            max_categories: Maximum number of categories for one-hot encoding
            
        Returns:
            DataFrame with encoded features
        """
        if columns is None:
            columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Filter out non-existent columns
        columns = [col for col in columns if col in data.columns]
        
        if not columns:
            logger.warning("No categorical columns found for encoding")
            return data.copy()
        
        logger.info(f"Encoding categorical features using {strategy}: {columns}")
        
        result_data = data.copy()
        
        for col in columns:
            if strategy == "onehot":
                result_data = self._encode_onehot(result_data, col, max_categories)
            elif strategy == "label":
                result_data = self._encode_label(result_data, col)
            elif strategy == "frequency":
                result_data = self._encode_frequency(result_data, col)
            else:
                logger.warning(f"Unknown encoding strategy: {strategy}")
        
        logger.info(f"Successfully encoded {len(columns)} categorical features")
        return result_data
    
    def create_polynomial_features(
        self,
        data: pd.DataFrame,
        degree: int = 2,
        columns: Optional[List[str]] = None,
        interaction_only: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial features from numerical columns.
        
        Args:
            data: Input dataframe
            degree: Degree of polynomial features
            columns: Specific columns to use (None for all numerical)
            interaction_only: Only create interaction terms
            
        Returns:
            DataFrame with additional polynomial features
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter columns and limit to reasonable number
        columns = [col for col in columns if col in data.columns][:10]  # Limit to 10 columns
        
        if not columns:
            logger.warning("No numerical columns found for polynomial features")
            return data.copy()
        
        logger.info(f"Creating polynomial features (degree={degree}) for: {columns}")
        
        result_data = data.copy()
        
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            poly = PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=False
            )
            
            # Transform selected columns
            poly_features = poly.fit_transform(data[columns])
            
            # Get feature names
            feature_names = poly.get_feature_names_out(columns)
            
            # Create dataframe with polynomial features
            poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)
            
            # Remove original features from poly_df to avoid duplication
            original_features = set(columns)
            new_features = [col for col in poly_df.columns if col not in original_features]
            
            # Add only new polynomial features
            for feature in new_features:
                result_data[f"poly_{feature}"] = poly_df[feature]
            
            logger.info(f"Created {len(new_features)} polynomial features")
            
        except ImportError:
            logger.warning("sklearn not available for polynomial features")
        except Exception as e:
            logger.error(f"Error creating polynomial features: {str(e)}")
        
        return result_data
    
    def create_interaction_features(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        max_interactions: int = 20
    ) -> pd.DataFrame:
        """
        Create interaction features between numerical columns.
        
        Args:
            data: Input dataframe
            columns: Specific columns to use (None for all numerical)
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            DataFrame with additional interaction features
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to reasonable number of columns
        columns = [col for col in columns if col in data.columns][:5]  # Limit to 5 columns
        
        if len(columns) < 2:
            logger.warning("Need at least 2 numerical columns for interactions")
            return data.copy()
        
        logger.info(f"Creating interaction features for: {columns}")
        
        result_data = data.copy()
        interactions_created = 0
        
        # Create pairwise interactions
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                if interactions_created >= max_interactions:
                    break
                
                # Multiplicative interaction
                interaction_name = f"interact_{col1}_x_{col2}"
                result_data[interaction_name] = data[col1] * data[col2]
                interactions_created += 1
                
                # Additive interaction (if room for more)
                if interactions_created < max_interactions:
                    add_interaction_name = f"interact_{col1}_plus_{col2}"
                    result_data[add_interaction_name] = data[col1] + data[col2]
                    interactions_created += 1
            
            if interactions_created >= max_interactions:
                break
        
        logger.info(f"Created {interactions_created} interaction features")
        return result_data
    
    def create_temporal_features(
        self,
        data: pd.DataFrame,
        datetime_columns: Optional[List[str]] = None,
        extract_components: bool = True
    ) -> pd.DataFrame:
        """
        Create temporal features from datetime columns.
        
        Args:
            data: Input dataframe
            datetime_columns: Specific datetime columns (None for auto-detection)
            extract_components: Whether to extract time components
            
        Returns:
            DataFrame with additional temporal features
        """
        if datetime_columns is None:
            datetime_columns = data.select_dtypes(include=["datetime64", "datetime"]).columns.tolist()
        
        # Also check for columns that might be datetime strings
        for col in data.columns:
            if col not in datetime_columns and data[col].dtype == "object":
                # Try to parse a sample to see if it's datetime
                sample = data[col].dropna().iloc[:5] if not data[col].dropna().empty else []
                if len(sample) > 0:
                    try:
                        pd.to_datetime(sample.iloc[0])
                        datetime_columns.append(col)
                    except Exception:
                        pass
        
        if not datetime_columns:
            logger.warning("No datetime columns found for temporal features")
            return data.copy()
        
        logger.info(f"Creating temporal features for: {datetime_columns}")
        
        result_data = data.copy()
        
        for col in datetime_columns:
            try:
                # Convert to datetime if needed
                if result_data[col].dtype == "object":
                    result_data[col] = pd.to_datetime(result_data[col], errors="coerce")
                
                if extract_components:
                    # Extract time components
                    result_data[f"{col}_year"] = result_data[col].dt.year
                    result_data[f"{col}_month"] = result_data[col].dt.month
                    result_data[f"{col}_day"] = result_data[col].dt.day
                    result_data[f"{col}_dayofweek"] = result_data[col].dt.dayofweek
                    result_data[f"{col}_hour"] = result_data[col].dt.hour
                    result_data[f"{col}_quarter"] = result_data[col].dt.quarter
                    
                    # Create cyclical features
                    result_data[f"{col}_month_sin"] = np.sin(2 * np.pi * result_data[col].dt.month / 12)
                    result_data[f"{col}_month_cos"] = np.cos(2 * np.pi * result_data[col].dt.month / 12)
                    result_data[f"{col}_day_sin"] = np.sin(2 * np.pi * result_data[col].dt.day / 31)
                    result_data[f"{col}_day_cos"] = np.cos(2 * np.pi * result_data[col].dt.day / 31)
                
                logger.info(f"Created temporal features for column: {col}")
                
            except Exception as e:
                logger.error(f"Error creating temporal features for {col}: {str(e)}")
        
        return result_data
    
    def select_features(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        method: str = "variance",
        k_features: Optional[int] = None,
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using various feature selection methods.
        
        Args:
            data: Input dataframe
            target: Target variable (required for supervised methods)
            method: Selection method ('variance', 'univariate', 'recursive', 'l1')
            k_features: Number of features to select (None for threshold-based)
            threshold: Threshold for feature selection
            
        Returns:
            Tuple of (selected_data, selected_feature_names)
        """
        logger.info(f"Performing feature selection using {method} method")
        
        # Get numerical columns for feature selection
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_columns:
            logger.warning("No numerical columns found for feature selection")
            return data.copy(), data.columns.tolist()
        
        X = data[numerical_columns]
        
        try:
            if method == "variance":
                selector = VarianceThreshold(threshold=threshold)
                selected_data = selector.fit_transform(X)
                selected_features = X.columns[selector.get_support()].tolist()
                
            elif method == "univariate" and target is not None:
                if k_features is None:
                    k_features = min(10, len(numerical_columns))
                
                # Determine score function based on target type
                if target.dtype in ["int64", "int32", "bool"]:
                    score_func = chi2 if (X >= 0).all().all() else f_classif
                else:
                    score_func = f_classif
                
                selector = SelectKBest(score_func=score_func, k=k_features)
                selected_data = selector.fit_transform(X, target)
                selected_features = X.columns[selector.get_support()].tolist()
                
            else:
                # Default to variance threshold if method not supported or target missing
                logger.warning(f"Method {method} not supported or target missing, using variance threshold")
                selector = VarianceThreshold(threshold=threshold)
                selected_data = selector.fit_transform(X)
                selected_features = X.columns[selector.get_support()].tolist()
            
            # Create result dataframe
            result_data = data.copy()
            
            # Keep non-numerical columns and selected numerical columns
            non_numerical_columns = [col for col in data.columns if col not in numerical_columns]
            all_selected_features = non_numerical_columns + selected_features
            
            result_data = result_data[all_selected_features]
            
            logger.info(f"Selected {len(selected_features)} features out of {len(numerical_columns)}")
            
            return result_data, all_selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return data.copy(), data.columns.tolist()
    
    def _create_scaler(self, method: str) -> Any:
        """Create a scaler based on the method."""
        if method == "standard":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        elif method == "robust":
            return RobustScaler()
        elif method == "quantile":
            return QuantileTransformer()
        elif method == "power":
            return PowerTransformer()
        else:
            logger.warning(f"Unknown scaling method: {method}, using standard")
            return StandardScaler()
    
    def _encode_onehot(
        self,
        data: pd.DataFrame,
        column: str,
        max_categories: int
    ) -> pd.DataFrame:
        """Apply one-hot encoding to a categorical column."""
        try:
            # Check number of unique values
            unique_values = data[column].nunique()
            
            if unique_values > max_categories:
                logger.warning(f"Column {column} has {unique_values} categories, skipping one-hot encoding")
                return data
            
            # Create dummy variables
            dummies = pd.get_dummies(data[column], prefix=column, dummy_na=True)
            
            # Remove original column and add dummy columns
            result_data = data.drop(columns=[column])
            result_data = pd.concat([result_data, dummies], axis=1)
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error in one-hot encoding for {column}: {str(e)}")
            return data
    
    def _encode_label(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply label encoding to a categorical column."""
        try:
            encoder_key = f"label_encoder_{column}"
            
            if encoder_key not in self._encoders:
                encoder = LabelEncoder()
                # Fit encoder on non-null values
                non_null_values = data[column].dropna()
                if not non_null_values.empty:
                    encoder.fit(non_null_values)
                    self._encoders[encoder_key] = encoder
                else:
                    logger.warning(f"No non-null values found for label encoding: {column}")
                    return data
            else:
                encoder = self._encoders[encoder_key]
            
            # Transform the data
            result_data = data.copy()
            non_null_mask = data[column].notna()
            
            if non_null_mask.any():
                result_data.loc[non_null_mask, column] = encoder.transform(data.loc[non_null_mask, column])
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error in label encoding for {column}: {str(e)}")
            return data
    
    def _encode_frequency(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply frequency encoding to a categorical column."""
        try:
            # Calculate frequency mapping
            frequency_map = data[column].value_counts().to_dict()
            
            # Apply frequency encoding
            result_data = data.copy()
            result_data[column] = data[column].map(frequency_map).fillna(0)
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error in frequency encoding for {column}: {str(e)}")
            return data
    
    def get_feature_importance(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        method: str = "random_forest"
    ) -> pd.DataFrame:
        """
        Calculate feature importance scores.
        
        Args:
            data: Input features
            target: Target variable
            method: Method for calculating importance
            
        Returns:
            DataFrame with feature names and importance scores
        """
        try:
            numerical_data = data.select_dtypes(include=[np.number])
            
            if numerical_data.empty:
                logger.warning("No numerical features for importance calculation")
                return pd.DataFrame(columns=["feature", "importance"])
            
            if method == "random_forest":
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(numerical_data.fillna(0), target)
                importance_scores = rf.feature_importances_
            else:
                # Default to correlation-based importance
                correlations = numerical_data.corrwith(target).abs()
                importance_scores = correlations.fillna(0).values
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                "feature": numerical_data.columns,
                "importance": importance_scores
            }).sort_values("importance", ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return pd.DataFrame(columns=["feature", "importance"])