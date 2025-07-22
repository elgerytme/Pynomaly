"""PyOD algorithm adapter for anomaly detection."""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Try to import PyOD components
try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA
    from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS
    from pyod.models.abod import ABOD
    from pyod.models.feature_bagging import FeatureBagging
    from pyod.models.combination import aom, moa, average, maximization
    PYOD_AVAILABLE = True
except ImportError:
    IForest = None
    LOF = None
    OCSVM = None
    PCA = None
    KNN = None
    HBOS = None
    ABOD = None
    FeatureBagging = None
    aom = moa = average = maximization = None
    PYOD_AVAILABLE = False


class PyODAdapter:
    """Adapter for PyOD (Python Outlier Detection) algorithms.
    
    Provides access to 40+ anomaly detection algorithms from PyOD library
    with a unified interface. PyOD offers more specialized and advanced
    algorithms compared to basic scikit-learn implementations.
    """
    
    def __init__(self, algorithm: str = "iforest", **kwargs):
        """Initialize adapter.
        
        Args:
            algorithm: Algorithm name from PyOD
            **kwargs: Algorithm-specific parameters
        """
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is required for PyODAdapter. Install with: pip install pyod")
            
        self.algorithm = algorithm
        self.parameters = kwargs
        self.model = None
        self._fitted = False
        
        # Set default parameters
        self._set_defaults()
        
    def _set_defaults(self) -> None:
        """Set default parameters for each algorithm."""
        defaults = {
            "iforest": {
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42
            },
            "lof": {
                "n_neighbors": 20,
                "contamination": 0.1,
                "algorithm": "auto"
            },
            "ocsvm": {
                "contamination": 0.1,
                "kernel": "rbf",
                "gamma": "scale"
            },
            "pca": {
                "contamination": 0.1,
                "n_components": None,
                "standardization": True
            },
            "knn": {
                "contamination": 0.1,
                "n_neighbors": 5,
                "method": "largest"
            },
            "hbos": {
                "contamination": 0.1,
                "n_bins": 10,
                "alpha": 0.1
            },
            "abod": {
                "contamination": 0.1,
                "n_neighbors": 10
            },
            "feature_bagging": {
                "contamination": 0.1,
                "n_estimators": 10,
                "max_features": 1.0
            }
        }
        
        if self.algorithm in defaults:
            for key, value in defaults[self.algorithm].items():
                if key not in self.parameters:
                    self.parameters[key] = value
    
    def fit(self, data: npt.NDArray[np.floating]) -> PyODAdapter:
        """Fit the algorithm on training data.
        
        Args:
            data: Training data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is required")
            
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(data)
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
            
        return self.model.predict(data)
    
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
            
        return self.model.decision_function(data)
    
    def predict_proba(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get prediction probabilities.
        
        Args:
            data: Data to predict probabilities for
            
        Returns:
            Probabilities of being anomalous
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # PyOD returns probabilities for [normal, anomaly]
        probas = self.model.predict_proba(data)
        return probas[:, 1]  # Return anomaly probabilities
    
    def _create_model(self) -> Any:
        """Create the PyOD model based on algorithm."""
        models = {
            "iforest": IForest,
            "lof": LOF,
            "ocsvm": OCSVM,
            "pca": PCA,
            "knn": KNN,
            "hbos": HBOS,
            "abod": ABOD,
            "feature_bagging": FeatureBagging
        }
        
        if self.algorithm not in models:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
        model_class = models[self.algorithm]
        if model_class is None:
            raise ImportError(f"PyOD algorithm {self.algorithm} not available")
            
        return model_class(**self.parameters)
    
    def get_feature_importances(self) -> npt.NDArray[np.floating] | None:
        """Get feature importances if available."""
        if not self._fitted:
            return None
            
        # PyOD doesn't provide feature importances for most models
        # Return None for now - could be extended for specific algorithms
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current algorithm parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, **params: Any) -> PyODAdapter:
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model."""
        if not self._fitted:
            return {"fitted": False}
            
        info = {
            "fitted": True,
            "algorithm": self.algorithm,
            "parameters": self.parameters
        }
        
        # Add model-specific info
        if hasattr(self.model, 'contamination'):
            info["contamination"] = self.model.contamination
        if hasattr(self.model, 'threshold_'):
            info["threshold"] = self.model.threshold_
        if hasattr(self.model, 'decision_scores_'):
            info["training_scores_stats"] = {
                "mean": float(np.mean(self.model.decision_scores_)),
                "std": float(np.std(self.model.decision_scores_)),
                "min": float(np.min(self.model.decision_scores_)),
                "max": float(np.max(self.model.decision_scores_))
            }
            
        return info
    
    @staticmethod
    def get_available_algorithms() -> list[str]:
        """Get list of available PyOD algorithms."""
        if not PYOD_AVAILABLE:
            return []
        return [
            "iforest", "lof", "ocsvm", "pca", "knn", "hbos", "abod", "feature_bagging"
        ]
    
    @staticmethod
    def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
        """Get detailed information about a specific algorithm."""
        info = {
            "iforest": {
                "name": "Isolation Forest (PyOD)",
                "description": "PyOD implementation of Isolation Forest with additional features",
                "type": "ensemble",
                "complexity": "medium",
                "scalability": "high",
                "parameters": {
                    "n_estimators": "Number of trees (default: 100)",
                    "contamination": "Expected proportion of outliers (default: 0.1)",
                    "max_samples": "Number of samples per tree"
                }
            },
            "lof": {
                "name": "Local Outlier Factor (PyOD)",
                "description": "Local density-based anomaly detection with PyOD enhancements",
                "type": "proximity",
                "complexity": "medium",
                "scalability": "medium",
                "parameters": {
                    "n_neighbors": "Number of neighbors (default: 20)",
                    "contamination": "Expected proportion of outliers (default: 0.1)",
                    "algorithm": "Neighbor search algorithm"
                }
            },
            "ocsvm": {
                "name": "One-Class SVM (PyOD)",
                "description": "Support Vector Machine for novelty detection",
                "type": "boundary",
                "complexity": "high",
                "scalability": "low",
                "parameters": {
                    "contamination": "Expected proportion of outliers (default: 0.1)",
                    "kernel": "Kernel type (default: 'rbf')",
                    "gamma": "Kernel coefficient"
                }
            },
            "pca": {
                "name": "PCA-based Detection (PyOD)",
                "description": "Principal Component Analysis for anomaly detection",
                "type": "linear",
                "complexity": "low",
                "scalability": "high",
                "parameters": {
                    "n_components": "Number of components",
                    "contamination": "Expected proportion of outliers (default: 0.1)",
                    "standardization": "Whether to standardize data"
                }
            },
            "knn": {
                "name": "k-Nearest Neighbors",
                "description": "Distance-based anomaly detection using k-NN",
                "type": "proximity",
                "complexity": "low",
                "scalability": "medium",
                "parameters": {
                    "n_neighbors": "Number of neighbors (default: 5)",
                    "contamination": "Expected proportion of outliers (default: 0.1)",
                    "method": "Distance method ('largest', 'mean', 'median')"
                }
            },
            "hbos": {
                "name": "Histogram-based Outlier Score",
                "description": "Histogram-based anomaly detection",
                "type": "statistical",
                "complexity": "low", 
                "scalability": "high",
                "parameters": {
                    "n_bins": "Number of histogram bins (default: 10)",
                    "contamination": "Expected proportion of outliers (default: 0.1)",
                    "alpha": "Regularization parameter"
                }
            },
            "abod": {
                "name": "Angle-Based Outlier Detector",
                "description": "Angle-based anomaly detection",
                "type": "angle",
                "complexity": "high",
                "scalability": "low",
                "parameters": {
                    "n_neighbors": "Number of neighbors (default: 10)",
                    "contamination": "Expected proportion of outliers (default: 0.1)"
                }
            },
            "feature_bagging": {
                "name": "Feature Bagging",
                "description": "Ensemble method using feature subsets",
                "type": "ensemble",
                "complexity": "medium",
                "scalability": "medium",
                "parameters": {
                    "n_estimators": "Number of base estimators (default: 10)",
                    "contamination": "Expected proportion of outliers (default: 0.1)",
                    "max_features": "Fraction of features per estimator"
                }
            }
        }
        
        return info.get(algorithm, {
            "name": "Unknown Algorithm",
            "description": "Algorithm not found",
            "type": "unknown",
            "complexity": "unknown",
            "scalability": "unknown"
        })
    
    @staticmethod
    def create_ensemble(
        algorithms: list[str], 
        combination_method: str = "average",
        **kwargs: Any
    ) -> PyODEnsemble:
        """Create an ensemble of PyOD algorithms.
        
        Args:
            algorithms: List of algorithm names
            combination_method: How to combine predictions
            **kwargs: Parameters for individual algorithms
            
        Returns:
            PyODEnsemble instance
        """
        return PyODEnsemble(algorithms, combination_method, **kwargs)
    
    def __str__(self) -> str:
        """String representation."""
        return f"PyODAdapter(algorithm='{self.algorithm}', fitted={self._fitted})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"PyODAdapter(algorithm='{self.algorithm}', parameters={self.parameters})"


class PyODEnsemble:
    """Ensemble of multiple PyOD algorithms."""
    
    def __init__(
        self, 
        algorithms: list[str], 
        combination_method: str = "average",
        **kwargs: Any
    ):
        """Initialize ensemble.
        
        Args:
            algorithms: List of PyOD algorithm names
            combination_method: 'average', 'max', 'aom', 'moa'
            **kwargs: Parameters for algorithms
        """
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is required for ensemble")
            
        self.algorithms = algorithms
        self.combination_method = combination_method
        self.adapters: list[PyODAdapter] = []
        self._fitted = False
        
        # Create individual adapters
        for algorithm in algorithms:
            algo_params = kwargs.get(f"{algorithm}_params", {})
            adapter = PyODAdapter(algorithm, **algo_params)
            self.adapters.append(adapter)
    
    def fit(self, data: npt.NDArray[np.floating]) -> PyODEnsemble:
        """Fit all algorithms in ensemble."""
        for adapter in self.adapters:
            adapter.fit(data)
        self._fitted = True
        return self
    
    def predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Predict using ensemble combination."""
        if not self._fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        # Get predictions from all adapters
        predictions = []
        for adapter in self.adapters:
            pred = adapter.predict(data)
            predictions.append(pred)
        
        predictions_array = np.array(predictions)
        
        # Combine predictions
        if self.combination_method == "average":
            return (np.mean(predictions_array, axis=0) > 0.5).astype(int)
        elif self.combination_method == "max":
            return np.max(predictions_array, axis=0)
        else:
            # Default to majority vote
            return (np.sum(predictions_array, axis=0) > len(predictions) / 2).astype(int)
    
    def decision_function(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get ensemble anomaly scores."""
        if not self._fitted:
            raise ValueError("Ensemble must be fitted before scoring")
            
        # Get scores from all adapters
        scores = []
        for adapter in self.adapters:
            score = adapter.decision_function(data)
            scores.append(score)
        
        scores_array = np.array(scores)
        
        # Combine scores based on method
        if self.combination_method == "average" and average is not None:
            return average(scores_array)
        elif self.combination_method == "max" and maximization is not None:
            return maximization(scores_array)
        elif self.combination_method == "aom" and aom is not None:
            return aom(scores_array, n_buckets=5)
        elif self.combination_method == "moa" and moa is not None:
            return moa(scores_array, n_buckets=5)
        else:
            # Fallback to simple average
            return np.mean(scores_array, axis=0)