"""
PyOD Algorithm Adapter Implementation

Concrete implementation of the algorithm adapter using PyOD library.
"""

import asyncio
import time
from typing import Any, Dict, List
import numpy as np
from sklearn.preprocessing import StandardScaler

from .algorithm_adapter import AlgorithmAdapter, AlgorithmResult, AlgorithmExecutionError
from ...domain.value_objects.algorithm_config import AlgorithmConfig, AlgorithmType


class PyODAlgorithmAdapter(AlgorithmAdapter):
    """
    Concrete implementation of AlgorithmAdapter using PyOD library.
    
    This adapter provides integration with the PyOD (Python Outlier Detection)
    library for executing various anomaly detection algorithms.
    """
    
    def __init__(self, preprocessing: bool = True):
        """
        Initialize the PyOD algorithm adapter.
        
        Args:
            preprocessing: Whether to apply standard scaling to input data.
        """
        self._preprocessing = preprocessing
        self._scaler = StandardScaler() if preprocessing else None
        self._supported_algorithms = {
            AlgorithmType.ISOLATION_FOREST: self._run_isolation_forest,
            AlgorithmType.LOCAL_OUTLIER_FACTOR: self._run_local_outlier_factor,
            AlgorithmType.ONE_CLASS_SVM: self._run_one_class_svm,
            AlgorithmType.ELLIPTIC_ENVELOPE: self._run_elliptic_envelope,
            AlgorithmType.AUTOENCODER: self._run_autoencoder
        }
    
    async def detect_anomalies(
        self, 
        data: List[float], 
        algorithm_config: AlgorithmConfig
    ) -> AlgorithmResult:
        """
        Execute anomaly detection using PyOD algorithms.
        
        Args:
            data: Input data for anomaly detection.
            algorithm_config: Configuration for the algorithm.
            
        Returns:
            AlgorithmResult: Results of the anomaly detection.
            
        Raises:
            AlgorithmExecutionError: If algorithm execution fails.
        """
        if not await self.validate_algorithm_support(algorithm_config):
            raise AlgorithmExecutionError(
                algorithm_config.algorithm_type.value,
                "Algorithm not supported by this adapter"
            )
        
        try:
            # Prepare data
            data_array = np.array(data).reshape(-1, 1)
            
            if self._preprocessing and self._scaler:
                data_array = self._scaler.fit_transform(data_array)
            
            # Execute algorithm
            start_time = time.time()
            
            algorithm_func = self._supported_algorithms[algorithm_config.algorithm_type]
            anomalies, scores = await algorithm_func(data_array, algorithm_config)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return AlgorithmResult(
                anomalies=anomalies.tolist(),
                scores=scores.tolist(),
                algorithm_type=algorithm_config.algorithm_type.value,
                execution_time_ms=execution_time,
                metadata={
                    "contamination": algorithm_config.contamination,
                    "preprocessing_applied": self._preprocessing,
                    "data_size": len(data)
                }
            )
            
        except Exception as e:
            raise AlgorithmExecutionError(
                algorithm_config.algorithm_type.value,
                str(e),
                e
            )
    
    async def validate_algorithm_support(self, algorithm_config: AlgorithmConfig) -> bool:
        """
        Check if the adapter supports the specified algorithm.
        
        Args:
            algorithm_config: Configuration to validate.
            
        Returns:
            bool: True if the algorithm is supported.
        """
        return algorithm_config.algorithm_type in self._supported_algorithms
    
    async def get_supported_algorithms(self) -> List[str]:
        """
        Get list of supported algorithm types.
        
        Returns:
            List[str]: List of supported algorithm type names.
        """
        return [algo_type.value for algo_type in self._supported_algorithms.keys()]
    
    async def estimate_execution_time(
        self, 
        data_size: int, 
        algorithm_config: AlgorithmConfig
    ) -> int:
        """
        Estimate execution time based on data size and algorithm.
        
        Args:
            data_size: Size of the input data.
            algorithm_config: Algorithm configuration.
            
        Returns:
            int: Estimated execution time in milliseconds.
        """
        # Simple heuristic-based estimation
        base_times = {
            AlgorithmType.ISOLATION_FOREST: 10,
            AlgorithmType.LOCAL_OUTLIER_FACTOR: 50,
            AlgorithmType.ONE_CLASS_SVM: 100,
            AlgorithmType.ELLIPTIC_ENVELOPE: 20,
            AlgorithmType.AUTOENCODER: 200
        }
        
        base_time = base_times.get(algorithm_config.algorithm_type, 50)
        
        # Scale with data size (non-linear for some algorithms)
        if algorithm_config.algorithm_type == AlgorithmType.LOCAL_OUTLIER_FACTOR:
            # LOF has quadratic complexity
            scaling_factor = (data_size / 1000) ** 1.5
        elif algorithm_config.algorithm_type == AlgorithmType.ONE_CLASS_SVM:
            # SVM has higher complexity
            scaling_factor = (data_size / 1000) ** 2
        else:
            # Linear scaling for other algorithms
            scaling_factor = data_size / 1000
        
        return int(base_time * max(1, scaling_factor))
    
    async def _run_isolation_forest(
        self, 
        data: np.ndarray, 
        config: AlgorithmConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Isolation Forest algorithm."""
        try:
            # Import here to avoid dependency issues if PyOD is not installed
            from pyod.models.iforest import IForest
            
            # Extract parameters
            params = config.parameters.copy()
            params["contamination"] = config.contamination
            if config.random_state is not None:
                params["random_state"] = config.random_state
            
            # Run algorithm in thread to avoid blocking
            def _execute():
                model = IForest(**params)
                model.fit(data)
                scores = model.decision_scores_
                predictions = model.labels_
                return predictions, scores
            
            loop = asyncio.get_event_loop()
            predictions, scores = await loop.run_in_executor(None, _execute)
            
            return predictions, scores
            
        except ImportError:
            raise AlgorithmExecutionError(
                "isolation_forest",
                "PyOD library not installed. Please install with: pip install pyod"
            )
    
    async def _run_local_outlier_factor(
        self, 
        data: np.ndarray, 
        config: AlgorithmConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Local Outlier Factor algorithm."""
        try:
            from pyod.models.lof import LOF
            
            params = config.parameters.copy()
            params["contamination"] = config.contamination
            
            def _execute():
                model = LOF(**params)
                model.fit(data)
                scores = model.decision_scores_
                predictions = model.labels_
                return predictions, scores
            
            loop = asyncio.get_event_loop()
            predictions, scores = await loop.run_in_executor(None, _execute)
            
            return predictions, scores
            
        except ImportError:
            raise AlgorithmExecutionError(
                "local_outlier_factor",
                "PyOD library not installed. Please install with: pip install pyod"
            )
    
    async def _run_one_class_svm(
        self, 
        data: np.ndarray, 
        config: AlgorithmConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run One-Class SVM algorithm."""
        try:
            from pyod.models.ocsvm import OCSVM
            
            params = config.parameters.copy()
            params["contamination"] = config.contamination
            
            def _execute():
                model = OCSVM(**params)
                model.fit(data)
                scores = model.decision_scores_
                predictions = model.labels_
                return predictions, scores
            
            loop = asyncio.get_event_loop()
            predictions, scores = await loop.run_in_executor(None, _execute)
            
            return predictions, scores
            
        except ImportError:
            raise AlgorithmExecutionError(
                "one_class_svm",
                "PyOD library not installed. Please install with: pip install pyod"
            )
    
    async def _run_elliptic_envelope(
        self, 
        data: np.ndarray, 
        config: AlgorithmConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Elliptic Envelope algorithm."""
        try:
            from sklearn.covariance import EllipticEnvelope
            
            params = config.parameters.copy()
            params["contamination"] = config.contamination
            if config.random_state is not None:
                params["random_state"] = config.random_state
            
            def _execute():
                model = EllipticEnvelope(**params)
                model.fit(data)
                scores = model.decision_function(data)
                predictions = model.predict(data)
                # Convert sklearn predictions (-1, 1) to (1, 0)
                predictions = np.where(predictions == -1, 1, 0)
                return predictions, scores
            
            loop = asyncio.get_event_loop()
            predictions, scores = await loop.run_in_executor(None, _execute)
            
            return predictions, scores
            
        except ImportError:
            raise AlgorithmExecutionError(
                "elliptic_envelope",
                "scikit-learn not installed. Please install with: pip install scikit-learn"
            )
    
    async def _run_autoencoder(
        self, 
        data: np.ndarray, 
        config: AlgorithmConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Autoencoder algorithm."""
        try:
            from pyod.models.auto_encoder import AutoEncoder
            
            params = config.parameters.copy()
            params["contamination"] = config.contamination
            if config.random_state is not None:
                params["random_state"] = config.random_state
            
            def _execute():
                model = AutoEncoder(**params)
                model.fit(data)
                scores = model.decision_scores_
                predictions = model.labels_
                return predictions, scores
            
            loop = asyncio.get_event_loop()
            predictions, scores = await loop.run_in_executor(None, _execute)
            
            return predictions, scores
            
        except ImportError:
            raise AlgorithmExecutionError(
                "autoencoder",
                "PyOD with deep learning dependencies not installed. Please install with: pip install pyod[dl]"
            )