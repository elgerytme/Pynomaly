"""Model Validation domain service interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..entities.data_science_model import DataScienceModel
from ..value_objects.model_performance_metrics import ModelPerformanceMetrics


class IModelValidationService(ABC):
    """Domain service for model validation operations."""
    
    @abstractmethod
    async def validate_model_performance(self, model: DataScienceModel,
                                       validation_data: Any,
                                       metrics: Optional[List[str]] = None) -> ModelPerformanceMetrics:
        """Validate model performance on validation data."""
        pass
    
    @abstractmethod
    async def cross_validate_model(self, model: DataScienceModel,
                                 dataset: Any,
                                 cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on model."""
        pass
    
    @abstractmethod
    async def validate_model_fairness(self, model: DataScienceModel,
                                    test_data: Any,
                                    protected_attributes: List[str]) -> Dict[str, Any]:
        """Validate model fairness across protected groups."""
        pass