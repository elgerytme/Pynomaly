"""Comprehensive PyOD adapter with full algorithm support for anomaly detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import numpy.typing as npt
from enum import Enum
from dataclasses import dataclass

from ..logging import get_logger

logger = get_logger(__name__)

# Try to import all PyOD components
try:
    # Basic algorithms
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA
    from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS
    from pyod.models.abod import ABOD
    from pyod.models.cblof import CBLOF
    from pyod.models.feature_bagging import FeatureBagging
    from pyod.models.copod import COPOD
    
    # Advanced algorithms
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE
    from pyod.models.so_gaal import SO_GAAL
    from pyod.models.mo_gaal import MO_GAAL
    from pyod.models.deepsvdd import DeepSVDD
    from pyod.models.anogan import AnoGAN
    
    # Statistical algorithms
    from pyod.models.ecod import ECOD
    from pyod.models.cd import CD
    from pyod.models.mcd import MCD
    from pyod.models.mad import MAD
    from pyod.models.sod import SOD
    from pyod.models.rod import ROD
    
    # Linear models
    from pyod.models.suod import SUOD
    from pyod.models.lscp import LSCP
    from pyod.models.xgbod import XGBOD
    from pyod.models.inne import INNE
    from pyod.models.dif import DIF
    
    # Proximity-based
    from pyod.models.lunar import LUNAR
    from pyod.models.rgraph import RGraph
    
    # Combination methods
    from pyod.models.combination import aom, moa, average, maximization
    
    PYOD_AVAILABLE = True
    logger.info("PyOD library successfully imported with full algorithm support")
    
except ImportError as e:
    # Set all classes to None if PyOD is not available
    IForest = LOF = OCSVM = PCA = KNN = HBOS = ABOD = CBLOF = None
    FeatureBagging = COPOD = AutoEncoder = VAE = SO_GAAL = MO_GAAL = None
    DeepSVDD = AnoGAN = ECOD = CD = MCD = MAD = SOD = ROD = None
    SUOD = LSCP = XGBOD = INNE = DIF = LUNAR = RGraph = None
    aom = moa = average = maximization = None
    PYOD_AVAILABLE = False
    logger.warning("PyOD library not available", error=str(e))


class AlgorithmCategory(Enum):
    """Categories of PyOD algorithms."""
    PROXIMITY_BASED = "proximity_based"
    LINEAR_MODELS = "linear_models"
    PROBABILISTIC = "probabilistic"
    NEURAL_NETWORKS = "neural_networks"
    ENSEMBLE = "ensemble"
    COMBINATION = "combination"


@dataclass
class AlgorithmInfo:
    """Information about a PyOD algorithm."""
    name: str
    display_name: str
    category: AlgorithmCategory
    model_class: Any
    description: str
    parameters: Dict[str, Any]
    requires_scaling: bool = True
    supports_streaming: bool = False
    computational_complexity: str = "medium"
    memory_usage: str = "medium"


class ComprehensivePyODAdapter:
    """Comprehensive adapter for PyOD (Python Outlier Detection) algorithms.
    
    Provides access to 40+ anomaly detection algorithms from PyOD library
    with a unified interface, advanced parameter management, and comprehensive
    algorithm information.
    """
    
    def __init__(self, algorithm: str = "iforest", **kwargs):
        """Initialize comprehensive PyOD adapter.
        
        Args:
            algorithm: Algorithm name from PyOD
            **kwargs: Algorithm-specific parameters
        """
        if not PYOD_AVAILABLE:
            raise ImportError(
                "PyOD is required for ComprehensivePyODAdapter. "
                "Install with: pip install pyod"
            )
            
        self.algorithm = algorithm
        self.parameters = kwargs
        self.model = None
        self._fitted = False
        self._training_data = None
        self._algorithm_info = None
        
        # Initialize algorithm registry
        self._initialize_algorithm_registry()
        
        # Validate algorithm
        if algorithm not in self.available_algorithms:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Available algorithms: {list(self.available_algorithms.keys())}"
            )
        
        self._algorithm_info = self.available_algorithms[algorithm]
        
        # Set default parameters
        self._set_defaults()
        
        logger.info("Comprehensive PyOD adapter initialized",
                   algorithm=algorithm,
                   category=self._algorithm_info.category.value)
    
    def _initialize_algorithm_registry(self) -> None:
        """Initialize the registry of all available algorithms."""
        self.available_algorithms = {}
        
        # Proximity-based algorithms
        if IForest is not None:
            self.available_algorithms["iforest"] = AlgorithmInfo(
                name="iforest",
                display_name="Isolation Forest",
                category=AlgorithmCategory.PROXIMITY_BASED,
                model_class=IForest,
                description="Tree-based anomaly detection using isolation paths",
                parameters={
                    "n_estimators": 100,
                    "contamination": 0.1,
                    "random_state": 42,
                    "behaviour": "new"
                },
                requires_scaling=False,
                supports_streaming=True,
                computational_complexity="low",
                memory_usage="medium"
            )
        
        if LOF is not None:
            self.available_algorithms["lof"] = AlgorithmInfo(
                name="lof",
                display_name="Local Outlier Factor",
                category=AlgorithmCategory.PROXIMITY_BASED,
                model_class=LOF,
                description="Density-based anomaly detection using local outlier factor",
                parameters={
                    "n_neighbors": 20,
                    "contamination": 0.1,
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "metric": "minkowski"
                },
                computational_complexity="medium",
                memory_usage="medium"
            )
        
        if KNN is not None:
            self.available_algorithms["knn"] = AlgorithmInfo(
                name="knn",
                display_name="K-Nearest Neighbors",
                category=AlgorithmCategory.PROXIMITY_BASED,
                model_class=KNN,
                description="Distance-based anomaly detection using k-nearest neighbors",
                parameters={
                    "contamination": 0.1,
                    "n_neighbors": 5,
                    "method": "largest",
                    "radius": 1.0,
                    "algorithm": "auto"
                },
                computational_complexity="medium",
                memory_usage="high"
            )
        
        if ABOD is not None:
            self.available_algorithms["abod"] = AlgorithmInfo(
                name="abod",
                display_name="Angle-Based Outlier Detection",
                category=AlgorithmCategory.PROXIMITY_BASED,
                model_class=ABOD,
                description="Angle-based anomaly detection for high-dimensional data",
                parameters={
                    "contamination": 0.1,
                    "n_neighbors": 10
                },
                computational_complexity="high",
                memory_usage="high"
            )
        
        if CBLOF is not None:
            self.available_algorithms["cblof"] = AlgorithmInfo(
                name="cblof",
                display_name="Cluster-Based Local Outlier Factor",
                category=AlgorithmCategory.PROXIMITY_BASED,
                model_class=CBLOF,
                description="Cluster-based local outlier factor for anomaly detection",
                parameters={
                    "contamination": 0.1,
                    "n_clusters": 8,
                    "alpha": 0.9,
                    "beta": 5,
                    "use_weights": False
                },
                computational_complexity="medium",
                memory_usage="medium"
            )
        
        # Linear models
        if PCA is not None:
            self.available_algorithms["pca"] = AlgorithmInfo(
                name="pca",
                display_name="Principal Component Analysis",
                category=AlgorithmCategory.LINEAR_MODELS,
                model_class=PCA,
                description="Linear dimensionality reduction for anomaly detection",
                parameters={
                    "contamination": 0.1,
                    "n_components": None,
                    "n_selected_components": None,
                    "copy": True,
                    "whiten": False,
                    "svd_solver": "auto",
                    "tol": 0.0,
                    "iterated_power": "auto",
                    "random_state": None,
                    "weighted": True,
                    "standardization": True
                },
                computational_complexity="low",
                memory_usage="low"
            )
        
        if MCD is not None:
            self.available_algorithms["mcd"] = AlgorithmInfo(
                name="mcd",
                display_name="Minimum Covariance Determinant",
                category=AlgorithmCategory.LINEAR_MODELS,
                model_class=MCD,
                description="Robust covariance estimation for anomaly detection",
                parameters={
                    "contamination": 0.1,
                    "store_precision": True,
                    "assume_centered": False,
                    "support_fraction": None,
                    "random_state": None
                },
                computational_complexity="medium",
                memory_usage="medium"
            )
        
        if OCSVM is not None:
            self.available_algorithms["ocsvm"] = AlgorithmInfo(
                name="ocsvm",
                display_name="One-Class Support Vector Machine",
                category=AlgorithmCategory.LINEAR_MODELS,
                model_class=OCSVM,
                description="One-class SVM for novelty detection",
                parameters={
                    "contamination": 0.1,
                    "kernel": "rbf",
                    "degree": 3,
                    "gamma": "scale",
                    "coef0": 0.0,
                    "tol": 1e-3,
                    "nu": 0.5,
                    "shrinking": True,
                    "cache_size": 200,
                    "verbose": False,
                    "max_iter": -1
                },
                computational_complexity="medium",
                memory_usage="medium"
            )
        
        # Probabilistic algorithms
        if HBOS is not None:
            self.available_algorithms["hbos"] = AlgorithmInfo(
                name="hbos",
                display_name="Histogram-Based Outlier Score",
                category=AlgorithmCategory.PROBABILISTIC,
                model_class=HBOS,
                description="Histogram-based anomaly detection",
                parameters={
                    "contamination": 0.1,
                    "n_bins": 10,
                    "alpha": 0.1,
                    "tol": 0.5
                },
                requires_scaling=False,
                computational_complexity="low",
                memory_usage="low"
            )
        
        if COPOD is not None:
            self.available_algorithms["copod"] = AlgorithmInfo(
                name="copod",
                display_name="Copula-Based Outlier Detection",
                category=AlgorithmCategory.PROBABILISTIC,
                model_class=COPOD,
                description="Copula-based anomaly detection",
                parameters={
                    "contamination": 0.1
                },
                computational_complexity="low",
                memory_usage="low"
            )
        
        if ECOD is not None:
            self.available_algorithms["ecod"] = AlgorithmInfo(
                name="ecod",
                display_name="Empirical Cumulative Distribution",
                category=AlgorithmCategory.PROBABILISTIC,
                model_class=ECOD,
                description="Empirical cumulative distribution-based anomaly detection",
                parameters={
                    "contamination": 0.1
                },
                computational_complexity="low",
                memory_usage="low"
            )
        
        if MAD is not None:
            self.available_algorithms["mad"] = AlgorithmInfo(
                name="mad",
                display_name="Median Absolute Deviation",
                category=AlgorithmCategory.PROBABILISTIC,
                model_class=MAD,
                description="Median absolute deviation-based anomaly detection",
                parameters={
                    "contamination": 0.1,
                    "threshold": None
                },
                requires_scaling=False,
                computational_complexity="low",
                memory_usage="low"
            )
        
        # Neural network algorithms
        if AutoEncoder is not None:
            self.available_algorithms["autoencoder"] = AlgorithmInfo(
                name="autoencoder",
                display_name="Auto Encoder",
                category=AlgorithmCategory.NEURAL_NETWORKS,
                model_class=AutoEncoder,
                description="Deep learning autoencoder for anomaly detection",
                parameters={
                    "contamination": 0.1,
                    "hidden_neurons": [64, 32, 32, 64],
                    "hidden_activation": "relu",
                    "output_activation": "sigmoid",
                    "optimizer": "adam",
                    "epochs": 100,
                    "batch_size": 32,
                    "dropout_rate": 0.2,
                    "l2_regularizer": 0.1,
                    "validation_size": 0.1,
                    "preprocessing": True,
                    "verbose": 1,
                    "random_state": None,
                    "contamination": 0.1
                },
                computational_complexity="high",
                memory_usage="high"
            )
        
        if VAE is not None:
            self.available_algorithms["vae"] = AlgorithmInfo(
                name="vae",
                display_name="Variational Autoencoder",
                category=AlgorithmCategory.NEURAL_NETWORKS,
                model_class=VAE,
                description="Variational autoencoder for anomaly detection",
                parameters={
                    "contamination": 0.1,
                    "encoder_neurons": [128, 64, 32],
                    "decoder_neurons": [32, 64, 128],
                    "hidden_activation": "relu",
                    "output_activation": "sigmoid",
                    "optimizer": "adam",
                    "epochs": 100,
                    "batch_size": 32,
                    "dropout_rate": 0.2,
                    "l2_regularizer": 0.1,
                    "validation_size": 0.1,
                    "preprocessing": True,
                    "verbose": 1,
                    "beta": 1.0,
                    "capacity": 0.0,
                    "random_state": None
                },
                computational_complexity="high",
                memory_usage="high"
            )
        
        if SO_GAAL is not None:
            self.available_algorithms["so_gaal"] = AlgorithmInfo(
                name="so_gaal",
                display_name="Single-Objective Generative Adversarial Active Learning",
                category=AlgorithmCategory.NEURAL_NETWORKS,
                model_class=SO_GAAL,
                description="GAN-based anomaly detection with active learning",
                parameters={
                    "contamination": 0.1,
                    "stop_epochs": 20,
                    "lr_d": 0.01,
                    "lr_g": 0.0001,
                    "decay": 1e-6,
                    "momentum": 0.9
                },
                computational_complexity="very_high",
                memory_usage="very_high"
            )
        
        if DeepSVDD is not None:
            self.available_algorithms["deepsvdd"] = AlgorithmInfo(
                name="deepsvdd",
                display_name="Deep Support Vector Data Description",
                category=AlgorithmCategory.NEURAL_NETWORKS,
                model_class=DeepSVDD,
                description="Deep learning extension of SVDD for anomaly detection",
                parameters={
                    "contamination": 0.1,
                    "c": None,
                    "eta": 1.0,
                    "optimizer_name": "adam",
                    "lr": 0.001,
                    "n_epochs": 50,
                    "lr_milestones": [50],
                    "batch_size": 128,
                    "weight_decay": 1e-6,
                    "preprocessing": True,
                    "verbose": 1,
                    "random_state": None
                },
                computational_complexity="high",
                memory_usage="high"
            )
        
        # Ensemble algorithms
        if FeatureBagging is not None:
            self.available_algorithms["feature_bagging"] = AlgorithmInfo(
                name="feature_bagging",
                display_name="Feature Bagging",
                category=AlgorithmCategory.ENSEMBLE,
                model_class=FeatureBagging,
                description="Feature bagging ensemble for anomaly detection",
                parameters={
                    "contamination": 0.1,
                    "n_estimators": 10,
                    "max_features": 1.0,
                    "bootstrap_features": False,
                    "check_detector": True,
                    "check_estimator": False,
                    "n_jobs": None,
                    "random_state": None,
                    "combination": "average",
                    "verbose": 0,
                    "estimator_params": {}
                },
                supports_streaming=True,
                computational_complexity="medium",
                memory_usage="medium"
            )
        
        if SUOD is not None:
            self.available_algorithms["suod"] = AlgorithmInfo(
                name="suod",
                display_name="Scalable Unsupervised Outlier Detection",
                category=AlgorithmCategory.ENSEMBLE,
                model_class=SUOD,
                description="Scalable ensemble for large-scale anomaly detection",
                parameters={
                    "contamination": 0.1,
                    "base_estimators": None,
                    "n_jobs": None,
                    "rp_flag_global": True,
                    "rp_method": "basic",
                    "rp_clf_list": None,
                    "rp_ng_clf_list": None,
                    "rp_percent_list": None,
                    "approx_flag_global": False,
                    "approx_method": "average",
                    "approx_clf_list": None,
                    "approx_ng_clf_list": None,
                    "approx_percent_list": None,
                    "cost_forecast_loc_fit": None,
                    "cost_forecast_loc_pred": None,
                    "verbose": False,
                    "random_state": None
                },
                supports_streaming=True,
                computational_complexity="medium",
                memory_usage="low"
            )
        
        if LSCP is not None:
            self.available_algorithms["lscp"] = AlgorithmInfo(
                name="lscp",
                display_name="Locally Selective Combination in Parallel Outlier Ensembles",
                category=AlgorithmCategory.ENSEMBLE,
                model_class=LSCP,
                description="Locally selective combination of parallel outlier ensembles",
                parameters={
                    "contamination": 0.1,
                    "detector_list": None,
                    "local_region_size": 30,
                    "local_max_features": 1.0,
                    "local_region_iterations": 100,
                    "local_region_training_proportion": 0.8,
                    "use_weights": False,
                    "random_state": None
                },
                computational_complexity="high",
                memory_usage="high"
            )
        
        # Advanced algorithms
        if XGBOD is not None:
            self.available_algorithms["xgbod"] = AlgorithmInfo(
                name="xgbod",
                display_name="Extreme Gradient Boosting Outlier Detection",
                category=AlgorithmCategory.ENSEMBLE,
                model_class=XGBOD,
                description="XGBoost-based anomaly detection with unsupervised representation learning",
                parameters={
                    "contamination": 0.1,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "silent": True,
                    "objective": "binary:logistic",
                    "booster": "gbtree",
                    "n_jobs": 1,
                    "nthread": None,
                    "gamma": 0,
                    "min_child_weight": 1,
                    "max_delta_step": 0,
                    "subsample": 1,
                    "colsample_bytree": 1,
                    "colsample_bylevel": 1,
                    "reg_alpha": 0,
                    "reg_lambda": 1,
                    "scale_pos_weight": 1,
                    "base_score": 0.5,
                    "random_state": None,
                    "seed": None,
                    "missing": None
                },
                computational_complexity="medium",
                memory_usage="medium"
            )
        
        logger.info("Algorithm registry initialized",
                   total_algorithms=len(self.available_algorithms),
                   categories=list(set(info.category.value for info in self.available_algorithms.values())))
    
    def _set_defaults(self) -> None:
        """Set default parameters for the selected algorithm."""
        if self._algorithm_info:
            for key, value in self._algorithm_info.parameters.items():
                if key not in self.parameters:
                    self.parameters[key] = value
    
    def fit(self, data: npt.NDArray[np.floating]) -> 'ComprehensivePyODAdapter':
        """Fit the algorithm on training data.
        
        Args:
            data: Training data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is required")
        
        # Store training data for potential reuse
        self._training_data = data.copy()
        
        # Create and fit model
        self.model = self._create_model()
        
        try:
            self.model.fit(data)
            self._fitted = True
            
            logger.info("Model fitted successfully",
                       algorithm=self.algorithm,
                       data_shape=data.shape,
                       contamination=self.parameters.get('contamination', 'unknown'))
        
        except Exception as e:
            logger.error("Error fitting model",
                        algorithm=self.algorithm,
                        error=str(e))
            raise
        
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
        
        try:
            predictions = self.model.predict(data)
            
            logger.debug("Predictions completed",
                        algorithm=self.algorithm,
                        data_shape=data.shape,
                        anomalies_detected=np.sum(predictions))
            
            return predictions
        
        except Exception as e:
            logger.error("Error making predictions",
                        algorithm=self.algorithm,
                        error=str(e))
            raise
    
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
        
        try:
            scores = self.model.decision_function(data)
            
            logger.debug("Decision function completed",
                        algorithm=self.algorithm,
                        data_shape=data.shape,
                        score_range=(np.min(scores), np.max(scores)))
            
            return scores
        
        except Exception as e:
            logger.error("Error computing decision function",
                        algorithm=self.algorithm,
                        error=str(e))
            raise
    
    def predict_proba(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get prediction probabilities.
        
        Args:
            data: Data to predict probabilities for
            
        Returns:
            Probabilities of being anomalous
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # PyOD returns probabilities for [normal, anomaly]
            probas = self.model.predict_proba(data)
            anomaly_probas = probas[:, 1]  # Return anomaly probabilities
            
            logger.debug("Predict proba completed",
                        algorithm=self.algorithm,
                        data_shape=data.shape,
                        prob_range=(np.min(anomaly_probas), np.max(anomaly_probas)))
            
            return anomaly_probas
        
        except Exception as e:
            logger.error("Error computing prediction probabilities",
                        algorithm=self.algorithm,
                        error=str(e))
            raise
    
    def predict_confidence(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get prediction confidence scores.
        
        Args:
            data: Data to get confidence for
            
        Returns:
            Confidence scores for predictions
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict_confidence'):
            return self.model.predict_confidence(data)
        else:
            # Fallback to decision function as confidence proxy
            return np.abs(self.decision_function(data))
    
    def _create_model(self) -> Any:
        """Create the PyOD model based on algorithm."""
        if self._algorithm_info is None:
            raise ValueError(f"Algorithm info not found for: {self.algorithm}")
        
        model_class = self._algorithm_info.model_class
        if model_class is None:
            raise ImportError(f"PyOD algorithm {self.algorithm} not available")
        
        try:
            model = model_class(**self.parameters)
            
            logger.debug("Model created successfully",
                        algorithm=self.algorithm,
                        model_class=model_class.__name__,
                        parameters=self.parameters)
            
            return model
        
        except Exception as e:
            logger.error("Error creating model",
                        algorithm=self.algorithm,
                        model_class=model_class.__name__,
                        error=str(e))
            raise
    
    def get_algorithm_info(self) -> AlgorithmInfo:
        """Get detailed information about the current algorithm."""
        if self._algorithm_info is None:
            raise ValueError(f"Algorithm info not found for: {self.algorithm}")
        return self._algorithm_info
    
    def get_feature_importances(self) -> Optional[npt.NDArray[np.floating]]:
        """Get feature importances if available."""
        if not self._fitted:
            return None
        
        # Check if model has feature importances
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        else:
            return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current algorithm parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, **params: Any) -> 'ComprehensivePyODAdapter':
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
        
        logger.info("Parameters updated",
                   algorithm=self.algorithm,
                   updated_params=list(params.keys()))
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the fitted model."""
        info = {
            "fitted": self._fitted,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "algorithm_info": {
                "display_name": self._algorithm_info.display_name,
                "category": self._algorithm_info.category.value,
                "description": self._algorithm_info.description,
                "computational_complexity": self._algorithm_info.computational_complexity,
                "memory_usage": self._algorithm_info.memory_usage,
                "requires_scaling": self._algorithm_info.requires_scaling,
                "supports_streaming": self._algorithm_info.supports_streaming
            } if self._algorithm_info else None
        }
        
        if self._fitted and self.model:
            # Add model-specific info
            if hasattr(self.model, 'contamination'):
                info["contamination"] = self.model.contamination
            if hasattr(self.model, 'threshold_'):
                info["threshold"] = float(self.model.threshold_)
            if hasattr(self.model, 'decision_scores_'):
                info["training_scores_stats"] = {
                    "mean": float(np.mean(self.model.decision_scores_)),
                    "std": float(np.std(self.model.decision_scores_)),
                    "min": float(np.min(self.model.decision_scores_)),
                    "max": float(np.max(self.model.decision_scores_))
                }
        
        return info
    
    @classmethod
    def list_available_algorithms(cls) -> Dict[str, Dict[str, Any]]:
        """List all available algorithms with their information."""
        if not PYOD_AVAILABLE:
            return {"error": "PyOD library not available"}
        
        # Create temporary instance to access algorithm registry
        temp_adapter = cls.__new__(cls)
        temp_adapter._initialize_algorithm_registry()
        
        algorithms = {}
        for name, info in temp_adapter.available_algorithms.items():
            algorithms[name] = {
                "display_name": info.display_name,
                "category": info.category.value,
                "description": info.description,
                "computational_complexity": info.computational_complexity,
                "memory_usage": info.memory_usage,
                "requires_scaling": info.requires_scaling,
                "supports_streaming": info.supports_streaming,
                "default_parameters": info.parameters
            }
        
        return algorithms
    
    @classmethod
    def get_algorithms_by_category(cls, category: AlgorithmCategory) -> List[str]:
        """Get algorithms filtered by category."""
        if not PYOD_AVAILABLE:
            return []
        
        temp_adapter = cls.__new__(cls)
        temp_adapter._initialize_algorithm_registry()
        
        return [
            name for name, info in temp_adapter.available_algorithms.items()
            if info.category == category
        ]
    
    @classmethod
    def get_recommended_algorithms(
        cls,
        data_size: str = "medium",  # "small", "medium", "large"
        complexity_preference: str = "medium",  # "low", "medium", "high"
        interpretability_required: bool = False
    ) -> List[str]:
        """Get recommended algorithms based on requirements."""
        if not PYOD_AVAILABLE:
            return []
        
        temp_adapter = cls.__new__(cls)
        temp_adapter._initialize_algorithm_registry()
        
        recommendations = []
        
        for name, info in temp_adapter.available_algorithms.items():
            # Filter by computational complexity
            if complexity_preference == "low" and info.computational_complexity in ["high", "very_high"]:
                continue
            elif complexity_preference == "medium" and info.computational_complexity == "very_high":
                continue
            
            # Filter by data size (memory usage proxy)
            if data_size == "large" and info.memory_usage in ["high", "very_high"]:
                continue
            elif data_size == "small" and info.memory_usage == "very_high":
                continue
            
            # Filter by interpretability
            if interpretability_required and info.category == AlgorithmCategory.NEURAL_NETWORKS:
                continue
            
            recommendations.append(name)
        
        return recommendations
    
    def evaluate_algorithm_suitability(
        self,
        data_shape: Tuple[int, int],
        has_labels: bool = False,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """Evaluate how suitable the current algorithm is for the given data."""
        if self._algorithm_info is None:
            return {"error": "Algorithm info not available"}
        
        n_samples, n_features = data_shape
        suitability_score = 100  # Start with perfect score
        warnings = []
        recommendations = []
        
        # Check data size compatibility
        if n_samples > 100000 and self._algorithm_info.computational_complexity in ["high", "very_high"]:
            suitability_score -= 30
            warnings.append(f"Large dataset ({n_samples} samples) with high complexity algorithm")
            recommendations.append("Consider using algorithms with lower computational complexity")
        
        if n_samples > 50000 and self._algorithm_info.memory_usage in ["high", "very_high"]:
            suitability_score -= 20
            warnings.append(f"Large dataset with high memory usage algorithm")
            recommendations.append("Consider algorithms with lower memory requirements")
        
        # Check dimensionality
        if n_features > 1000 and self.algorithm in ["abod", "lof"]:
            suitability_score -= 25
            warnings.append(f"High-dimensional data ({n_features} features) may not be suitable for {self.algorithm}")
            recommendations.append("Consider dimensionality reduction or algorithms designed for high-dimensional data")
        
        # Check streaming compatibility
        if streaming and not self._algorithm_info.supports_streaming:
            suitability_score -= 40
            warnings.append("Algorithm does not support streaming")
            recommendations.append("Consider algorithms that support streaming for real-time applications")
        
        # Check scaling requirements
        if self._algorithm_info.requires_scaling:
            recommendations.append("Data scaling/normalization is recommended for this algorithm")
        
        return {
            "suitability_score": max(0, suitability_score),
            "warnings": warnings,
            "recommendations": recommendations,
            "algorithm_properties": {
                "computational_complexity": self._algorithm_info.computational_complexity,
                "memory_usage": self._algorithm_info.memory_usage,
                "requires_scaling": self._algorithm_info.requires_scaling,
                "supports_streaming": self._algorithm_info.supports_streaming
            }
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model to disk."""
        if not self._fitted:
            raise ValueError("Model must be fitted before saving")
        
        import pickle
        
        model_data = {
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "model": self.model,
            "fitted": self._fitted,
            "training_data_shape": self._training_data.shape if self._training_data is not None else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model saved successfully",
                   algorithm=self.algorithm,
                   filepath=filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ComprehensivePyODAdapter':
        """Load a fitted model from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create adapter instance
        adapter = cls(
            algorithm=model_data["algorithm"],
            **model_data["parameters"]
        )
        
        # Restore model state
        adapter.model = model_data["model"]
        adapter._fitted = model_data["fitted"]
        
        logger.info("Model loaded successfully",
                   algorithm=adapter.algorithm,
                   filepath=filepath)
        
        return adapter