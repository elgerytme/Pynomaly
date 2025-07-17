"""In-memory repository implementation for ML models."""

from typing import Dict, List, Optional

from ...domain.entities.model import Model
from ...domain.repositories.model_repository import ModelRepository
from ...domain.value_objects.model_value_objects import ModelId, ModelStatus, ModelType


class InMemoryModelRepository(ModelRepository):
    """In-memory implementation of model repository."""
    
    def __init__(self):
        self._models: Dict[str, Model] = {}
    
    async def save(self, model: Model) -> Model:
        """Save a model."""
        model_id_str = str(model.model_id)
        self._models[model_id_str] = model
        return model
    
    async def get_by_id(self, model_id: ModelId) -> Optional[Model]:
        """Get model by ID."""
        return self._models.get(str(model_id))
    
    async def get_by_name(self, name: str) -> List[Model]:
        """Get models by name."""
        return [
            model for model in self._models.values()
            if model.name == name
        ]
    
    async def get_by_type(self, model_type: ModelType) -> List[Model]:
        """Get models by type."""
        return [
            model for model in self._models.values()
            if model.model_type == model_type
        ]
    
    async def get_by_status(self, status: ModelStatus) -> List[Model]:
        """Get models by status."""
        return [
            model for model in self._models.values()
            if model.status == status
        ]
    
    async def get_by_experiment(self, experiment_id: str) -> List[Model]:
        """Get models by experiment ID."""
        return [
            model for model in self._models.values()
            if model.experiment_id == experiment_id
        ]
    
    async def get_all(self) -> List[Model]:
        """Get all models."""
        return list(self._models.values())
    
    async def delete(self, model_id: ModelId) -> bool:
        """Delete a model."""
        model_id_str = str(model_id)
        if model_id_str in self._models:
            del self._models[model_id_str]
            return True
        return False
    
    async def update(self, model: Model) -> Model:
        """Update a model."""
        model_id_str = str(model.model_id)
        if model_id_str in self._models:
            self._models[model_id_str] = model
            return model
        else:
            raise ValueError(f"Model with ID {model_id_str} not found")
    
    def clear(self) -> None:
        """Clear all models (for testing)."""
        self._models.clear()
    
    def count(self) -> int:
        """Get count of models."""
        return len(self._models)