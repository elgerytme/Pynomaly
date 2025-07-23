"""Repository interface for ML models."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.model import Model
from ..value_objects.model_value_objects import ModelId, ModelStatus, ModelType


class ModelRepository(ABC):
    """Abstract repository for ML models."""
    
    @abstractmethod
    async def save(self, model: Model) -> Model:
        """Save a model."""
        pass
    
    @abstractmethod
    async def get_by_id(self, model_id: ModelId) -> Optional[Model]:
        """Get model by ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> List[Model]:
        """Get models by name."""
        pass
    
    @abstractmethod
    async def get_by_type(self, model_type: ModelType) -> List[Model]:
        """Get models by type."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: ModelStatus) -> List[Model]:
        """Get models by status."""
        pass
    
    @abstractmethod
    async def get_by_experiment(self, experiment_id: str) -> List[Model]:
        """Get models by experiment ID."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Model]:
        """Get all models."""
        pass
    
    @abstractmethod
    async def delete(self, model_id: ModelId) -> bool:
        """Delete a model."""
        pass
    
    @abstractmethod
    async def update(self, model: Model) -> Model:
        """Update a model."""
        pass