"""Feature Engineering domain service interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from ..entities.feature_store import FeatureStore
from ..value_objects.feature_importance import FeatureImportance


class IFeatureEngineeringService(ABC):
    """Domain service for feature engineering operations."""
    
    @abstractmethod
    async def generate_automated_features(self, dataset: Any,
                                        target_column: Optional[str] = None,
                                        feature_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate features automatically using various techniques."""
        pass
    
    @abstractmethod
    async def select_optimal_features(self, dataset: Any,
                                     target_column: str,
                                     selection_methods: Optional[List[str]] = None,
                                     max_features: Optional[int] = None) -> FeatureImportance:
        """Select optimal features using multiple selection methods."""
        pass
    
    @abstractmethod
    async def transform_features(self, dataset: Any,
                               transformations: Dict[str, str],
                               fit_params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Apply feature transformations to dataset."""
        pass