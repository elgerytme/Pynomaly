"""Data preprocessing utilities for anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ScalingMethod(Enum):
    """Available scaling methods."""
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


class ImputationStrategy(Enum):
    """Available imputation strategies."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    CONSTANT = "constant"
    DROP = "drop"


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Scaling configuration
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    scale_features: Optional[List[str]] = None
    
    # Missing value handling
    imputation_strategy: ImputationStrategy = ImputationStrategy.MEDIAN
    imputation_constant: Optional[float] = None
    missing_threshold: float = 0.5  # Drop columns with more than 50% missing
    
    # Categorical encoding
    encode_categoricals: bool = True
    categorical_columns: Optional[List[str]] = None
    max_categories: int = 50  # Maximum unique categories before dropping
    
    # Outlier handling
    remove_outliers: bool = False
    outlier_method: str = "iqr"  # "iqr" or "zscore"
    outlier_threshold: float = 3.0
    
    # Feature engineering
    create_interaction_features: bool = False
    polynomial_features: bool = False
    polynomial_degree: int = 2
    
    # Data validation
    min_samples: int = 10
    min_features: int = 1
    max_features: Optional[int] = None


class DataPreprocessor:
    """Comprehensive data preprocessing for anomaly detection."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.is_fitted = False
        self.feature_names_: Optional[List[str]] = None
        self.preprocessor_: Optional[Pipeline] = None
        self.original_shape_: Optional[Tuple[int, int]] = None
        self.dropped_columns_: List[str] = []
        self.categorical_encoders_: Dict[str, LabelEncoder] = {}
        
    def fit(self, X: Union[pd.DataFrame, npt.NDArray], y: Optional[npt.NDArray] = None) -> DataPreprocessor:
        """Fit the preprocessor on training data.
        
        Args:
            X: Training data
            y: Optional target values (not used but kept for sklearn compatibility)
            
        Returns:
            Self for method chaining
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        X = X.copy()
        self.original_shape_ = X.shape
        
        # Validate input
        self._validate_input(X)
        
        # Handle missing values and feature selection
        X_processed = self._fit_missing_value_handling(X)
        
        # Handle categorical variables
        X_processed = self._fit_categorical_encoding(X_processed)
        
        # Create preprocessing pipeline
        self._fit_preprocessing_pipeline(X_processed)
        
        # Store feature names
        self.feature_names_ = list(X_processed.columns)
        self.is_fitted = True
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, npt.NDArray]) -> npt.NDArray[np.floating]:
        """Transform data using fitted preprocessor.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data as numpy array
            
        Raises:
            ValueError: If preprocessor is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if self.feature_names_ and len(self.feature_names_) == X.shape[1]:
                X = pd.DataFrame(X, columns=self.feature_names_[:X.shape[1]])
            else:
                X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        X = X.copy()
        
        # Apply same transformations as in fit
        X_processed = self._transform_missing_values(X)
        X_processed = self._transform_categorical_encoding(X_processed)
        
        # Apply preprocessing pipeline
        X_transformed = self.preprocessor_.transform(X_processed)
        
        return X_transformed.astype(np.float64)
    
    def fit_transform(self, X: Union[pd.DataFrame, npt.NDArray], 
                     y: Optional[npt.NDArray] = None) -> npt.NDArray[np.floating]:
        """Fit preprocessor and transform data in one step.
        
        Args:
            X: Data to fit and transform
            y: Optional target values
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data."""
        if X.empty:
            raise ValueError("Input data is empty")
        
        if X.shape[0] < self.config.min_samples:
            raise ValueError(f"Insufficient samples: {X.shape[0]} < {self.config.min_samples}")
        
        if X.shape[1] < self.config.min_features:
            raise ValueError(f"Insufficient features: {X.shape[1]} < {self.config.min_features}")
        
        if self.config.max_features and X.shape[1] > self.config.max_features:
            raise ValueError(f"Too many features: {X.shape[1]} > {self.config.max_features}")
    
    def _fit_missing_value_handling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit missing value handling and return processed data."""
        # Calculate missing percentages
        missing_percentages = X.isnull().mean()
        
        # Drop columns with too many missing values
        columns_to_drop = missing_percentages[missing_percentages > self.config.missing_threshold].index.tolist()
        self.dropped_columns_.extend(columns_to_drop)
        
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)
        
        return X
    
    def _transform_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform missing values using fitted strategy."""
        # Drop columns that were dropped during fit
        columns_to_drop = [col for col in self.dropped_columns_ if col in X.columns]
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)
        
        return X
    
    def _fit_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit categorical encoding and return processed data."""
        if not self.config.encode_categoricals:
            return X
        
        # Identify categorical columns
        if self.config.categorical_columns:
            categorical_cols = [col for col in self.config.categorical_columns if col in X.columns]
        else:
            # Auto-detect categorical columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Encode categorical variables
        for col in categorical_cols:
            if X[col].nunique() > self.config.max_categories:
                # Drop columns with too many categories
                self.dropped_columns_.append(col)
                X = X.drop(columns=[col])
                continue
            
            # Fit label encoder
            encoder = LabelEncoder()
            # Handle missing values by treating them as a separate category
            X_col = X[col].fillna('__MISSING__')
            encoder.fit(X_col)
            self.categorical_encoders_[col] = encoder
            
            # Transform the column
            X[col] = encoder.transform(X_col)
        
        return X
    
    def _transform_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables using fitted encoders."""
        if not self.config.encode_categoricals:
            return X
        
        # Apply encoders
        for col, encoder in self.categorical_encoders_.items():
            if col in X.columns:
                X_col = X[col].fillna('__MISSING__')
                # Handle unseen categories
                X_col = X_col.map(lambda x: x if x in encoder.classes_ else '__MISSING__')
                X[col] = encoder.transform(X_col)
        
        return X
    
    def _fit_preprocessing_pipeline(self, X: pd.DataFrame) -> None:
        """Fit the main preprocessing pipeline."""
        # Separate numerical and remaining categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        transformers = []
        
        # Numerical pipeline
        if numerical_cols:
            numerical_steps = []
            
            # Imputation
            if self.config.imputation_strategy != ImputationStrategy.DROP:
                if self.config.imputation_strategy == ImputationStrategy.CONSTANT:
                    imputer = SimpleImputer(
                        strategy='constant',
                        fill_value=self.config.imputation_constant or 0
                    )
                else:
                    imputer = SimpleImputer(strategy=self.config.imputation_strategy.value)
                numerical_steps.append(('imputer', imputer))
            
            # Scaling
            if self.config.scaling_method != ScalingMethod.NONE:
                if self.config.scaling_method == ScalingMethod.STANDARD:
                    scaler = StandardScaler()
                elif self.config.scaling_method == ScalingMethod.ROBUST:
                    scaler = RobustScaler()
                elif self.config.scaling_method == ScalingMethod.MINMAX:
                    scaler = MinMaxScaler()
                
                numerical_steps.append(('scaler', scaler))
            
            if numerical_steps:
                numerical_pipeline = Pipeline(numerical_steps)
                transformers.append(('numerical', numerical_pipeline, numerical_cols))
        
        # Create column transformer
        if transformers:
            self.preprocessor_ = ColumnTransformer(
                transformers=transformers,
                remainder='drop'  # Drop any remaining columns
            )
        else:
            # If no transformers, create identity transformer
            from sklearn.preprocessing import FunctionTransformer
            self.preprocessor_ = FunctionTransformer(lambda x: x)
        
        # Fit the pipeline
        self.preprocessor_.fit(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        if hasattr(self.preprocessor_, 'get_feature_names_out'):
            try:
                return list(self.preprocessor_.get_feature_names_out())
            except:
                pass
        
        # Fallback: generate generic feature names
        if hasattr(self.preprocessor_, 'transform'):
            # Try to get the number of output features
            dummy_input = pd.DataFrame([[0] * len(self.feature_names_)], columns=self.feature_names_)
            try:
                output = self.preprocessor_.transform(dummy_input)
                n_features = output.shape[1]
                return [f"feature_{i}" for i in range(n_features)]
            except:
                pass
        
        return self.feature_names_ or []
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations applied."""
        if not self.is_fitted:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "original_shape": self.original_shape_,
            "final_features": len(self.get_feature_names_out()),
            "dropped_columns": self.dropped_columns_,
            "categorical_encoders": list(self.categorical_encoders_.keys()),
            "scaling_method": self.config.scaling_method.value,
            "imputation_strategy": self.config.imputation_strategy.value,
            "config": {
                "missing_threshold": self.config.missing_threshold,
                "max_categories": self.config.max_categories,
                "encode_categoricals": self.config.encode_categoricals
            }
        }
    
    def detect_data_issues(self, X: Union[pd.DataFrame, npt.NDArray]) -> Dict[str, Any]:
        """Detect potential data quality issues."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        issues = {
            "missing_values": {},
            "constant_columns": [],
            "high_cardinality_columns": {},
            "potential_outliers": {},
            "duplicate_rows": 0,
            "data_types": {}
        }
        
        # Missing values
        missing_counts = X.isnull().sum()
        issues["missing_values"] = {col: int(count) for col, count in missing_counts.items() if count > 0}
        
        # Constant columns
        for col in X.columns:
            if X[col].nunique() <= 1:
                issues["constant_columns"].append(col)
        
        # High cardinality columns
        for col in X.columns:
            unique_count = X[col].nunique()
            if unique_count > len(X) * 0.9:  # More than 90% unique values
                issues["high_cardinality_columns"][col] = unique_count
        
        # Duplicate rows
        issues["duplicate_rows"] = X.duplicated().sum()
        
        # Data types
        issues["data_types"] = {col: str(dtype) for col, dtype in X.dtypes.items()}
        
        # Potential outliers (for numerical columns)
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X[col].nunique() > 1:  # Skip constant columns
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
                if outliers > 0:
                    issues["potential_outliers"][col] = int(outliers)
        
        return issues


def create_anomaly_preprocessing_pipeline(
    scaling_method: str = "robust",
    handle_missing: bool = True,
    encode_categoricals: bool = True
) -> DataPreprocessor:
    """Create a preprocessing pipeline optimized for anomaly detection.
    
    Args:
        scaling_method: Scaling method to use ("standard", "robust", "minmax", "none")
        handle_missing: Whether to handle missing values
        encode_categoricals: Whether to encode categorical variables
        
    Returns:
        Configured DataPreprocessor
    """
    config = PreprocessingConfig(
        scaling_method=ScalingMethod(scaling_method),
        imputation_strategy=ImputationStrategy.MEDIAN if handle_missing else ImputationStrategy.DROP,
        encode_categoricals=encode_categoricals,
        missing_threshold=0.8,  # More lenient for anomaly detection
        remove_outliers=False,  # Keep outliers as they might be anomalies
        max_categories=100  # More categories allowed
    )
    
    return DataPreprocessor(config)