"""Integration module for connecting comprehensive PyOD adapter with detection services."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import numpy.typing as npt

from ..logging import get_logger
from .comprehensive_pyod_adapter import ComprehensivePyODAdapter, PYOD_AVAILABLE, AlgorithmCategory
from ...domain.entities.detection_result import DetectionResult
from ...domain.services.detection_service import DetectionService

logger = get_logger(__name__)


class PyODIntegration:
    """Integration class for connecting comprehensive PyOD adapter with detection services."""
    
    def __init__(self, detection_service: Optional[DetectionService] = None):
        """Initialize PyOD integration.
        
        Args:
            detection_service: Optional detection service to register adapters with
        """
        self.detection_service = detection_service
        self._registered_algorithms: Dict[str, ComprehensivePyODAdapter] = {}
        
        if not PYOD_AVAILABLE:
            logger.warning("PyOD library not available - PyOD integration will have limited functionality")
            return
        
        # Auto-register common algorithms if service provided
        if self.detection_service:
            self._register_common_algorithms()
    
    def _register_common_algorithms(self) -> None:
        """Register common PyOD algorithms with the detection service."""
        
        common_algorithms = [
            "iforest", "lof", "ocsvm", "pca", "knn", "hbos", "copod", "ecod"
        ]
        
        for algorithm in common_algorithms:
            try:
                self.register_algorithm(algorithm)
                logger.debug(f"Registered PyOD algorithm: {algorithm}")
            except Exception as e:
                logger.warning(f"Failed to register {algorithm}: {e}")
    
    def register_algorithm(
        self, 
        algorithm: str, 
        **default_params: Any
    ) -> None:
        """Register a PyOD algorithm with the detection service.
        
        Args:
            algorithm: PyOD algorithm name
            **default_params: Default parameters for the algorithm
        """
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD library not available")
        
        if not self.detection_service:
            raise ValueError("No detection service provided")
        
        # Create adapter wrapper
        adapter = PyODAdapterWrapper(algorithm, **default_params)
        
        # Register with detection service using a prefixed name to avoid conflicts
        adapter_name = f"pyod_{algorithm}"
        self.detection_service.register_adapter(adapter_name, adapter)
        self._registered_algorithms[adapter_name] = adapter
        
        logger.info(f"Registered PyOD algorithm adapter", 
                   algorithm=algorithm, 
                   adapter_name=adapter_name)
    
    def get_algorithm_recommendations(
        self,
        data_shape: tuple[int, int],
        performance_preference: str = "balanced",  # "speed", "accuracy", "balanced"
        interpretability_required: bool = False,
        streaming_support: bool = False
    ) -> List[str]:
        """Get algorithm recommendations based on data characteristics and requirements.
        
        Args:
            data_shape: Shape of the data (n_samples, n_features)
            performance_preference: Performance preference
            interpretability_required: Whether interpretability is required
            streaming_support: Whether streaming support is needed
            
        Returns:
            List of recommended algorithm names
        """
        if not PYOD_AVAILABLE:
            return []
        
        n_samples, n_features = data_shape
        
        # Map performance preference to complexity preference
        complexity_mapping = {
            "speed": "low",
            "balanced": "medium", 
            "accuracy": "high"
        }
        complexity_preference = complexity_mapping.get(performance_preference, "medium")
        
        # Determine data size category
        if n_samples < 1000:
            data_size = "small"
        elif n_samples < 10000:
            data_size = "medium"
        else:
            data_size = "large"
        
        # Get recommendations from comprehensive adapter
        recommendations = ComprehensivePyODAdapter.get_recommended_algorithms(
            data_size=data_size,
            complexity_preference=complexity_preference,
            interpretability_required=interpretability_required
        )
        
        # Filter by streaming support if required
        if streaming_support and recommendations:
            adapter = ComprehensivePyODAdapter()
            streaming_algorithms = []
            
            for algo in recommendations:
                if algo in adapter.available_algorithms:
                    algo_info = adapter.available_algorithms[algo]
                    if algo_info.supports_streaming:
                        streaming_algorithms.append(algo)
            
            recommendations = streaming_algorithms
        
        # Add prefix for registered names
        prefixed_recommendations = [f"pyod_{algo}" for algo in recommendations]
        
        logger.info(f"Generated algorithm recommendations",
                   data_shape=data_shape,
                   performance_preference=performance_preference,
                   recommendations=recommendations)
        
        return prefixed_recommendations
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """Get detailed information about a PyOD algorithm.
        
        Args:
            algorithm: Algorithm name (with or without pyod_ prefix)
            
        Returns:
            Dictionary with algorithm information
        """
        if not PYOD_AVAILABLE:
            return {"error": "PyOD library not available"}
        
        # Remove prefix if present
        clean_name = algorithm.replace("pyod_", "")
        
        try:
            adapter = ComprehensivePyODAdapter(algorithm=clean_name)
            algo_info = adapter.get_algorithm_info()
            
            return {
                "name": algo_info.name,
                "display_name": algo_info.display_name,
                "category": algo_info.category.value,
                "description": algo_info.description,
                "computational_complexity": algo_info.computational_complexity,
                "memory_usage": algo_info.memory_usage,
                "requires_scaling": algo_info.requires_scaling,
                "supports_streaming": algo_info.supports_streaming,
                "default_parameters": algo_info.parameters,
                "registered_name": f"pyod_{clean_name}"
            }
        
        except Exception as e:
            logger.error(f"Failed to get algorithm info for {algorithm}: {e}")
            return {"error": f"Algorithm information not available: {str(e)}"}
    
    def list_available_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """List all available PyOD algorithms with their information.
        
        Returns:
            Dictionary mapping algorithm names to their information
        """
        if not PYOD_AVAILABLE:
            return {"error": "PyOD library not available"}
        
        try:
            algorithms = ComprehensivePyODAdapter.list_available_algorithms()
            
            # Add prefix and additional metadata
            prefixed_algorithms = {}
            for name, info in algorithms.items():
                prefixed_name = f"pyod_{name}"
                prefixed_algorithms[prefixed_name] = {
                    **info,
                    "original_name": name,
                    "registered_name": prefixed_name,
                    "integration_type": "pyod_comprehensive"
                }
            
            return prefixed_algorithms
        
        except Exception as e:
            logger.error(f"Failed to list algorithms: {e}")
            return {"error": f"Failed to list algorithms: {str(e)}"}
    
    def get_algorithms_by_category(self, category: AlgorithmCategory) -> List[str]:
        """Get algorithms filtered by category.
        
        Args:
            category: Algorithm category
            
        Returns:
            List of algorithm names (with pyod_ prefix)
        """
        if not PYOD_AVAILABLE:
            return []
        
        try:
            algorithms = ComprehensivePyODAdapter.get_algorithms_by_category(category)
            return [f"pyod_{algo}" for algo in algorithms]
        
        except Exception as e:
            logger.error(f"Failed to get algorithms by category {category}: {e}")
            return []
    
    def evaluate_algorithm_suitability(
        self,
        algorithm: str,
        data_shape: tuple[int, int],
        has_labels: bool = False,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """Evaluate algorithm suitability for given data characteristics.
        
        Args:
            algorithm: Algorithm name (with or without pyod_ prefix)
            data_shape: Shape of the data
            has_labels: Whether labels are available
            streaming: Whether streaming capability is needed
            
        Returns:
            Suitability evaluation results
        """
        if not PYOD_AVAILABLE:
            return {"error": "PyOD library not available"}
        
        # Remove prefix if present
        clean_name = algorithm.replace("pyod_", "")
        
        try:
            adapter = ComprehensivePyODAdapter(algorithm=clean_name)
            evaluation = adapter.evaluate_algorithm_suitability(
                data_shape=data_shape,
                has_labels=has_labels,
                streaming=streaming
            )
            
            # Add integration-specific recommendations
            evaluation["integration_recommendations"] = []
            
            if evaluation["suitability_score"] < 50:
                evaluation["integration_recommendations"].append(
                    "Consider using algorithm recommendations for better suitability"
                )
            
            if streaming and not evaluation["algorithm_properties"]["supports_streaming"]:
                alternatives = self.get_algorithms_by_category(AlgorithmCategory.PROXIMITY_BASED)
                if alternatives:
                    evaluation["integration_recommendations"].append(
                        f"For streaming support, consider: {', '.join(alternatives[:3])}"
                    )
            
            return evaluation
        
        except Exception as e:
            logger.error(f"Failed to evaluate algorithm suitability: {e}")
            return {"error": f"Evaluation failed: {str(e)}"}
    
    def get_registered_algorithms(self) -> List[str]:
        """Get list of registered algorithm names.
        
        Returns:
            List of registered algorithm adapter names
        """
        return list(self._registered_algorithms.keys())
    
    def unregister_algorithm(self, algorithm: str) -> bool:
        """Unregister a PyOD algorithm adapter.
        
        Args:
            algorithm: Algorithm name (with or without pyod_ prefix)
            
        Returns:
            True if successfully unregistered, False otherwise
        """
        adapter_name = algorithm if algorithm.startswith("pyod_") else f"pyod_{algorithm}"
        
        if adapter_name in self._registered_algorithms:
            del self._registered_algorithms[adapter_name]
            logger.info(f"Unregistered PyOD algorithm adapter: {adapter_name}")
            return True
        
        return False


class PyODAdapterWrapper:
    """Wrapper to make ComprehensivePyODAdapter compatible with DetectionService protocol."""
    
    def __init__(self, algorithm: str, **default_params: Any):
        """Initialize wrapper.
        
        Args:
            algorithm: PyOD algorithm name
            **default_params: Default parameters for the algorithm
        """
        self.algorithm = algorithm
        self.default_params = default_params
        self._adapter: Optional[ComprehensivePyODAdapter] = None
        self._fitted = False
    
    def fit(self, data: npt.NDArray[np.floating]) -> None:
        """Fit the algorithm on data."""
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD library not available")
        
        # Create adapter with parameters
        self._adapter = ComprehensivePyODAdapter(
            algorithm=self.algorithm,
            **self.default_params
        )
        
        # Fit the adapter
        self._adapter.fit(data)
        self._fitted = True
        
        logger.debug(f"PyOD adapter fitted", 
                    algorithm=self.algorithm,
                    data_shape=data.shape)
    
    def predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Predict anomalies in data."""
        if not self._fitted or self._adapter is None:
            raise ValueError("Adapter must be fitted before prediction")
        
        predictions = self._adapter.predict(data)
        
        # Convert from PyOD format (0/1) to sklearn format (-1/1)
        # PyOD: 0 = normal, 1 = anomaly
        # sklearn: 1 = normal, -1 = anomaly
        sklearn_predictions = np.where(predictions == 0, 1, -1)
        
        logger.debug(f"PyOD adapter prediction completed",
                    algorithm=self.algorithm,
                    data_shape=data.shape,
                    anomalies_detected=np.sum(predictions == 1))
        
        return sklearn_predictions.astype(np.integer)
    
    def fit_predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Fit and predict in one step."""
        self.fit(data)
        return self.predict(data)
    
    def decision_function(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get anomaly scores."""
        if not self._fitted or self._adapter is None:
            raise ValueError("Adapter must be fitted before scoring")
        
        return self._adapter.decision_function(data)
    
    def predict_proba(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get prediction probabilities."""
        if not self._fitted or self._adapter is None:
            raise ValueError("Adapter must be fitted before prediction")
        
        return self._adapter.predict_proba(data)
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information."""
        if self._adapter is None:
            # Create temporary adapter for info
            temp_adapter = ComprehensivePyODAdapter(algorithm=self.algorithm)
            info = temp_adapter.get_algorithm_info()
        else:
            info = self._adapter.get_algorithm_info()
        
        return {
            "name": info.name,
            "display_name": info.display_name,
            "category": info.category.value,
            "description": info.description,
            "parameters": self.default_params
        }


def create_pyod_integration(detection_service: Optional[DetectionService] = None) -> PyODIntegration:
    """Create and configure PyOD integration.
    
    Args:
        detection_service: Optional detection service to integrate with
        
    Returns:
        Configured PyOD integration instance
    """
    integration = PyODIntegration(detection_service)
    
    logger.info("PyOD integration created",
               available=PYOD_AVAILABLE,
               registered_algorithms=len(integration._registered_algorithms))
    
    return integration


def register_recommended_algorithms(
    integration: PyODIntegration,
    data_shape: tuple[int, int],
    max_algorithms: int = 5,
    **preferences: Any
) -> List[str]:
    """Register recommended algorithms for given data characteristics.
    
    Args:
        integration: PyOD integration instance
        data_shape: Shape of the data
        max_algorithms: Maximum number of algorithms to register
        **preferences: Additional preferences for recommendations
        
    Returns:
        List of registered algorithm names
    """
    if not integration.detection_service:
        raise ValueError("Integration must have a detection service")
    
    # Get recommendations
    recommendations = integration.get_algorithm_recommendations(
        data_shape=data_shape,
        **preferences
    )
    
    # Register up to max_algorithms
    registered = []
    for i, algorithm in enumerate(recommendations[:max_algorithms]):
        try:
            # Remove prefix for registration
            clean_name = algorithm.replace("pyod_", "")
            integration.register_algorithm(clean_name)
            registered.append(algorithm)
        except Exception as e:
            logger.warning(f"Failed to register recommended algorithm {algorithm}: {e}")
    
    logger.info(f"Registered recommended algorithms",
               data_shape=data_shape,
               registered_count=len(registered),
               algorithms=registered)
    
    return registered