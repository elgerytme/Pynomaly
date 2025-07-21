"""Simplified PyOD adapter for anomaly detection algorithms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import numpy.typing as npt


class SimplePyODAdapter:
    """Simplified adapter for PyOD algorithms without domain dependencies.
    
    This adapter provides a clean interface to PyOD's 80+ anomaly detection
    algorithms without requiring complex domain architecture.
    """

    # Available PyOD algorithms mapping
    ALGORITHM_MAPPING: Dict[str, str] = {
        # Linear models
        "pca": "pyod.models.pca.PCA",
        "mcd": "pyod.models.mcd.MCD", 
        "ocsvm": "pyod.models.ocsvm.OCSVM",
        "lmdd": "pyod.models.lmdd.LMDD",
        
        # Proximity-based
        "lof": "pyod.models.lof.LOF",
        "cof": "pyod.models.cof.COF",
        "cblof": "pyod.models.cblof.CBLOF",
        "knn": "pyod.models.knn.KNN",
        "hbos": "pyod.models.hbos.HBOS",
        
        # Probabilistic
        "abod": "pyod.models.abod.ABOD",
        "copod": "pyod.models.copod.COPOD",
        "ecod": "pyod.models.ecod.ECOD",
        
        # Outlier ensembles
        "iforest": "pyod.models.iforest.IForest",
        "feature_bagging": "pyod.models.feature_bagging.FeatureBagging",
        "lscp": "pyod.models.lscp.LSCP",
        
        # Neural networks
        "auto_encoder": "pyod.models.auto_encoder.AutoEncoder",
        "vae": "pyod.models.vae.VAE",
        "beta_vae": "pyod.models.beta_vae.BetaVAE",
        "so_gaal": "pyod.models.so_gaal.SO_GAAL",
        "mo_gaal": "pyod.models.mo_gaal.MO_GAAL",
        
        # Statistical
        "sos": "pyod.models.sos.SOS",
        "loda": "pyod.models.loda.LODA",
        "inne": "pyod.models.inne.INNE",
        "cd": "pyod.models.cd.CD",
        "gm": "pyod.models.gmm.GMM",
        
        # Advanced methods
        "deep_svdd": "pyod.models.deep_svdd.DeepSVDD",
        "alad": "pyod.models.alad.ALAD",
        "anogan": "pyod.models.anogan.AnoGAN",
        "lunar": "pyod.models.lunar.LUNAR",
    }

    def __init__(
        self,
        algorithm: str = "iforest",
        contamination: float = 0.1,
        **kwargs: Any
    ):
        """Initialize PyOD adapter with specified algorithm.
        
        Args:
            algorithm: Name of PyOD algorithm to use
            contamination: Expected proportion of outliers (0.0 to 0.5)
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.contamination = contamination
        self.kwargs = kwargs
        self._model: Optional[Any] = None
        self._fitted = False

    def _get_algorithm_class(self, algorithm: str) -> Any:
        """Get PyOD algorithm class by name."""
        if algorithm not in self.ALGORITHM_MAPPING:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(self.ALGORITHM_MAPPING.keys())}")
        
        module_path = self.ALGORITHM_MAPPING[algorithm]
        module_name, class_name = module_path.rsplit('.', 1)
        
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(f"Failed to import {module_path}: {e}")

    def fit(self, X: npt.NDArray[np.floating]) -> SimplePyODAdapter:
        """Fit the PyOD model to the data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            
        Returns:
            self: Fitted adapter instance
        """
        algorithm_class = self._get_algorithm_class(self.algorithm)
        
        # Prepare parameters
        params = {"contamination": self.contamination}
        params.update(self.kwargs)
        
        # Initialize and fit model
        self._model = algorithm_class(**params)
        self._model.fit(X)
        self._fitted = True
        
        return self

    def predict(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Predict anomalies in the data.
        
        Args:
            X: Data to predict on of shape (n_samples, n_features)
            
        Returns:
            Binary predictions where 1 indicates anomaly, 0 indicates normal
            
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self._fitted or self._model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # PyOD returns 0 for normal, 1 for anomaly (opposite of sklearn)
        predictions = self._model.predict(X)
        return predictions.astype(np.integer)

    def fit_predict(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Fit model and predict anomalies in one step.
        
        Args:
            X: Data to fit and predict on of shape (n_samples, n_features)
            
        Returns:
            Binary predictions where 1 indicates anomaly, 0 indicates normal
        """
        return self.fit(X).predict(X)

    def decision_function(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get anomaly scores for the data.
        
        Args:
            X: Data to score of shape (n_samples, n_features)
            
        Returns:
            Anomaly scores (higher values indicate more anomalous)
            
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self._fitted or self._model is None:
            raise ValueError("Model must be fitted before scoring")
        
        return self._model.decision_function(X)

    @classmethod
    def list_algorithms(cls) -> List[str]:
        """Get list of available PyOD algorithms.
        
        Returns:
            List of algorithm names
        """
        return list(cls.ALGORITHM_MAPPING.keys())

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about current algorithm configuration.
        
        Returns:
            Dictionary with algorithm details
        """
        return {
            "algorithm": self.algorithm,
            "contamination": self.contamination,
            "parameters": self.kwargs,
            "fitted": self._fitted,
            "available_algorithms": self.list_algorithms()
        }


# Convenience function for quick usage
def create_pyod_detector(algorithm: str = "iforest", **kwargs) -> SimplePyODAdapter:
    """Create a PyOD detector with specified algorithm.
    
    Args:
        algorithm: PyOD algorithm name
        **kwargs: Algorithm parameters
        
    Returns:
        Configured SimplePyODAdapter instance
    """
    return SimplePyODAdapter(algorithm=algorithm, **kwargs)