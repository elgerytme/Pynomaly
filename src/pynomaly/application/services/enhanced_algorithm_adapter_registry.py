"""Enhanced Algorithm Adapter Registry with MLOps persistence."""

from __future__ import annotations

import logging
from typing import Any

from pynomaly.application.services.algorithm_adapter_registry import AlgorithmAdapterRegistry
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import FittingError
from pynomaly.domain.value_objects import AnomalyScore

# Import MLOps infrastructure
try:
    from pynomaly.mlops.model_registry import ModelRegistry, ModelType
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedAlgorithmAdapterRegistry:
    """Enhanced adapter registry with async MLOps model persistence."""

    def __init__(self, base_registry: AlgorithmAdapterRegistry = None):
        """Initialize enhanced registry.
        
        Args:
            base_registry: Base algorithm adapter registry
        """
        self.base_registry = base_registry or AlgorithmAdapterRegistry()
        self.model_registry = None
        
        if MLOPS_AVAILABLE:
            try:
                self.model_registry = ModelRegistry()
                logger.info("Enhanced registry initialized with MLOps persistence")
            except Exception as e:
                logger.warning(f"Failed to initialize model registry: {e}")

    def get_adapter_for_algorithm(self, algorithm_name: str):
        """Get adapter for algorithm."""
        return self.base_registry.get_adapter_for_algorithm(algorithm_name)

    def get_supported_algorithms(self) -> list[str]:
        """Get supported algorithms."""
        return self.base_registry.get_supported_algorithms()

    async def fit_detector(self, detector: Detector, dataset: Dataset) -> None:
        """Fit detector with MLOps persistence."""
        # First, fit using base registry
        self.base_registry.fit_detector(detector, dataset)
        
        # Then save to MLOps registry if available
        if self.model_registry is not None and detector.metadata.get("_needs_registry_save"):
            try:
                fitted_algorithm = detector.metadata.get("_fitted_algorithm")
                model_type = detector.metadata.get("_model_type")
                
                if fitted_algorithm and model_type:
                    model_id = await self.model_registry.register_model(
                        model=fitted_algorithm,
                        name=f"{detector.name}_{detector.algorithm_name}",
                        version="1.0.0",
                        model_type=model_type,
                        author="pynomaly_system",
                        description=f"Anomaly detector: {detector.name}",
                        hyperparameters=detector.parameters,
                        training_data_info={
                            "dataset_name": dataset.name,
                            "n_samples": len(dataset.data),
                            "n_features": len(dataset.data.columns),
                        }
                    )
                    
                    # Update detector metadata
                    detector.metadata["model_registry_id"] = model_id
                    detector.metadata.pop("_fitted_algorithm", None)
                    detector.metadata.pop("_needs_registry_save", None)
                    detector.metadata.pop("_model_type", None)
                    
                    logger.info(f"Model saved to registry: {model_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to save model to registry: {e}")

    async def predict_with_detector(self, detector: Detector, dataset: Dataset) -> list[int]:
        """Predict with detector, loading from registry if needed."""
        try:
            return self.base_registry.predict_with_detector(detector, dataset)
        except FittingError:
            # Try to load from registry
            if await self._load_model_from_registry(detector):
                return self.base_registry.predict_with_detector(detector, dataset)
            raise

    async def score_with_detector(self, detector: Detector, dataset: Dataset) -> list[AnomalyScore]:
        """Score with detector, loading from registry if needed."""
        try:
            return self.base_registry.score_with_detector(detector, dataset)
        except FittingError:
            # Try to load from registry
            if await self._load_model_from_registry(detector):
                return self.base_registry.score_with_detector(detector, dataset)
            raise

    async def _load_model_from_registry(self, detector: Detector) -> bool:
        """Load model from MLOps registry into base registry cache."""
        if self.model_registry is None:
            return False
            
        try:
            model_registry_id = detector.metadata.get("model_registry_id")
            if not model_registry_id:
                logger.warning(f"No model registry ID found for detector {detector.name}")
                return False
            
            # Load model from registry
            model, metadata = await self.model_registry.get_model(model_registry_id)
            
            if model is not None:
                # Get the appropriate adapter
                adapter = self.base_registry.get_adapter_for_algorithm(detector.algorithm_name)
                if adapter and hasattr(adapter, '_fitted_models'):
                    # Store in adapter's fitted models cache
                    adapter._fitted_models[str(detector.id)] = model
                    logger.info(f"Model loaded from registry: {model_registry_id}")
                    return True
                    
        except Exception as e:
            logger.warning(f"Failed to load model from registry: {e}")
            
        return False