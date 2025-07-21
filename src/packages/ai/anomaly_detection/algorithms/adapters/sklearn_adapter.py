"""Scikit-learn algorithm adapter for anomaly detection."""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Try to import sklearn components
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    IsolationForest = None
    LocalOutlierFactor = None
    OneClassSVM = None
    PCA = None
    StandardScaler = None
    NearestNeighbors = None
    SKLEARN_AVAILABLE = False


class SklearnAdapter:
    """Adapter for scikit-learn anomaly detection algorithms.
    
    Provides a unified interface for multiple sklearn-based anomaly detection
    algorithms including Isolation Forest, Local Outlier Factor, One-Class SVM,
    and PCA-based detection.
    """
    
    def __init__(self, algorithm: str = "iforest", **kwargs):
        """Initialize adapter.
        
        Args:
            algorithm: Algorithm name ('iforest', 'lof', 'ocsvm', 'pca')
            **kwargs: Algorithm-specific parameters
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for SklearnAdapter")
            
        self.algorithm = algorithm
        self.parameters = kwargs
        self.model = None
        self._scaler = None
        self._fitted = False
        
        # Set default parameters
        self._set_defaults()
        
    def _set_defaults(self) -> None:
        """Set default parameters for each algorithm."""
        defaults = {
            "iforest": {
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42,
                "max_samples": "auto"
            },
            "lof": {
                "n_neighbors": 20,
                "contamination": 0.1,
                "algorithm": "auto"
            },
            "ocsvm": {
                "nu": 0.1,  # Equivalent to contamination
                "kernel": "rbf",
                "gamma": "scale"
            },
            "pca": {
                "contamination": 0.1,
                "n_components": None,  # Will be set based on data
                "threshold_percentile": 95
            }
        }
        
        if self.algorithm in defaults:
            for key, value in defaults[self.algorithm].items():
                if key not in self.parameters:
                    self.parameters[key] = value
    
    def fit(self, data: npt.NDArray[np.floating]) -> SklearnAdapter:
        """Fit the algorithm on training data.
        
        Args:
            data: Training data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
            
        # Handle preprocessing
        processed_data = self._preprocess_data(data, fit=True)
        
        # Initialize and fit model
        self.model = self._create_model()
        
        if self.algorithm == "pca":
            # Special handling for PCA-based detection
            self._fit_pca(processed_data)
        else:
            self.model.fit(processed_data)
            
        self._fitted = True
        return self
        
    def predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Predict anomalies in data.
        
        Args:
            data: Data to predict on of shape (n_samples, n_features)
            
        Returns:
            Binary predictions where 1 indicates anomaly, 0 indicates normal
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
            
        processed_data = self._preprocess_data(data, fit=False)
        
        if self.algorithm == "pca":
            return self._predict_pca(processed_data)
        else:
            predictions = self.model.predict(processed_data)
            # Convert sklearn format (-1, 1) to (1, 0) for anomalies
            return (predictions == -1).astype(int)
    
    def fit_predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Fit and predict in one step.
        
        Args:
            data: Data to fit and predict on
            
        Returns:
            Binary predictions where 1 indicates anomaly, 0 indicates normal
        """
        self.fit(data)
        return self.predict(data)
    
    def decision_function(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get anomaly scores for data.
        
        Args:
            data: Data to score
            
        Returns:
            Anomaly scores (higher values indicate more anomalous)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before scoring")
            
        processed_data = self._preprocess_data(data, fit=False)
        
        if self.algorithm == "pca":
            return self._score_pca(processed_data)
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(processed_data)
            # Convert to positive scores (higher = more anomalous)
            return -scores
        elif hasattr(self.model, 'negative_outlier_factor_'):
            # LOF case
            return -self.model.negative_outlier_factor_
        else:
            # Fallback - use predict probabilities or uniform scores
            return np.ones(len(data)) * 0.5
    
    def _create_model(self) -> Any:
        """Create the sklearn model based on algorithm."""
        if self.algorithm == "iforest":
            return IsolationForest(**self.parameters)
        elif self.algorithm == "lof":
            return LocalOutlierFactor(**self.parameters)
        elif self.algorithm == "ocsvm":
            return OneClassSVM(**self.parameters)
        elif self.algorithm == "pca":
            # PCA parameters are handled separately
            n_components = self.parameters.get("n_components")
            return PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _preprocess_data(
        self, 
        data: npt.NDArray[np.floating], 
        fit: bool = False
    ) -> npt.NDArray[np.floating]:
        """Preprocess data (scaling, etc.)."""
        # Apply scaling for algorithms that benefit from it
        if self.algorithm in ["ocsvm", "pca"]:
            if fit:
                self._scaler = StandardScaler()
                return self._scaler.fit_transform(data)
            elif self._scaler is not None:
                return self._scaler.transform(data)
                
        return data
    
    def _fit_pca(self, data: npt.NDArray[np.floating]) -> None:
        """Fit PCA-based anomaly detection."""
        # Determine number of components if not specified
        if self.parameters.get("n_components") is None:
            # Use 95% variance retention
            pca_temp = PCA()
            pca_temp.fit(data)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= 0.95) + 1
            self.parameters["n_components"] = min(n_components, data.shape[1] - 1)
            self.model = PCA(n_components=self.parameters["n_components"])
        
        self.model.fit(data)
        
        # Calculate reconstruction errors for threshold
        reconstructed = self.model.inverse_transform(self.model.transform(data))
        reconstruction_errors = np.sum((data - reconstructed) ** 2, axis=1)
        
        # Set threshold based on percentile
        threshold_percentile = self.parameters.get("threshold_percentile", 95)
        self._pca_threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    def _predict_pca(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Predict using PCA reconstruction error."""
        scores = self._score_pca(data)
        return (scores > self._pca_threshold).astype(int)
    
    def _score_pca(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Score using PCA reconstruction error."""
        reconstructed = self.model.inverse_transform(self.model.transform(data))
        reconstruction_errors = np.sum((data - reconstructed) ** 2, axis=1)
        return reconstruction_errors
    
    def get_feature_importances(self) -> npt.NDArray[np.floating] | None:
        """Get feature importances if available."""
        if not self._fitted:
            return None
            
        if self.algorithm == "pca" and self.model is not None:
            # Return PCA component weights (absolute values)
            components = self.model.components_
            return np.mean(np.abs(components), axis=0)
        elif self.algorithm == "iforest" and hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current algorithm parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, **params: Any) -> SklearnAdapter:
        """Set algorithm parameters.
        
        Args:
            **params: Parameters to update
            
        Returns:
            Self for method chaining
        """
        self.parameters.update(params)
        # Reset fitted state since parameters changed
        self._fitted = False
        self.model = None
        return self
    
    @staticmethod
    def get_available_algorithms() -> list[str]:
        """Get list of available algorithms."""
        if not SKLEARN_AVAILABLE:
            return []
        return ["iforest", "lof", "ocsvm", "pca"]
    
    @staticmethod
    def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
        """Get information about a specific algorithm."""
        info = {
            "iforest": {
                "name": "Isolation Forest",
                "description": "Ensemble method for anomaly detection based on random forests",
                "parameters": {
                    "n_estimators": "Number of trees (default: 100)",
                    "contamination": "Expected proportion of outliers (default: 0.1)",
                    "max_samples": "Number of samples per tree (default: 'auto')"
                }
            },
            "lof": {
                "name": "Local Outlier Factor", 
                "description": "Local density-based anomaly detection",
                "parameters": {
                    "n_neighbors": "Number of neighbors (default: 20)",
                    "contamination": "Expected proportion of outliers (default: 0.1)",
                    "algorithm": "Neighbor search algorithm (default: 'auto')"
                }
            },
            "ocsvm": {
                "name": "One-Class SVM",
                "description": "Support Vector Machine for novelty detection",
                "parameters": {
                    "nu": "Upper bound on fraction of outliers (default: 0.1)",
                    "kernel": "Kernel type (default: 'rbf')",
                    "gamma": "Kernel coefficient (default: 'scale')"
                }
            },
            "pca": {
                "name": "PCA-based Detection",
                "description": "Anomaly detection using PCA reconstruction error",
                "parameters": {
                    "n_components": "Number of components (default: auto)",
                    "contamination": "Expected proportion of outliers (default: 0.1)",
                    "threshold_percentile": "Percentile for threshold (default: 95)"
                }
            }
        }
        
        return info.get(algorithm, {"name": "Unknown", "description": "Unknown algorithm"})
    
    def __str__(self) -> str:
        """String representation."""
        return f"SklearnAdapter(algorithm='{self.algorithm}', fitted={self._fitted})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"SklearnAdapter(algorithm='{self.algorithm}', parameters={self.parameters})"