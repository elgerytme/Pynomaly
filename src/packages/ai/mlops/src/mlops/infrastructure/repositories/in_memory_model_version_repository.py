"""In-memory repository implementation for model versions."""

from typing import Dict, List, Optional
from uuid import UUID

from ...domain.entities.model_version import ModelVersion


class ModelVersionRepositoryProtocol:
    """Protocol for model version repository."""
    
    async def save(self, model_version: ModelVersion) -> ModelVersion:
        """Save a model version."""
        pass
    
    async def find_by_id(self, version_id: UUID) -> Optional[ModelVersion]:
        """Find model version by ID."""
        pass
    
    async def find_by_model_id(self, model_id: UUID) -> List[ModelVersion]:
        """Find model versions by model ID."""
        pass
    
    async def find_latest_by_model_id(self, model_id: UUID) -> Optional[ModelVersion]:
        """Find latest model version by model ID."""
        pass
    
    async def delete(self, version_id: UUID) -> None:
        """Delete a model version."""
        pass


class InMemoryModelVersionRepository(ModelVersionRepositoryProtocol):
    """In-memory implementation of model version repository."""
    
    def __init__(self):
        self._versions: Dict[str, ModelVersion] = {}
    
    async def save(self, model_version: ModelVersion) -> ModelVersion:
        """Save a model version."""
        version_id_str = str(model_version.id)
        self._versions[version_id_str] = model_version
        return model_version
    
    async def find_by_id(self, version_id: UUID) -> Optional[ModelVersion]:
        """Find model version by ID."""
        return self._versions.get(str(version_id))
    
    async def find_by_model_id(self, model_id: UUID) -> List[ModelVersion]:
        """Find model versions by model ID."""
        return [
            version for version in self._versions.values()
            if version.model_id == model_id
        ]
    
    async def find_latest_by_model_id(self, model_id: UUID) -> Optional[ModelVersion]:
        """Find latest model version by model ID."""
        versions = await self.find_by_model_id(model_id)
        if not versions:
            return None
        
        # Sort by created_at descending and return first
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions[0]
    
    async def delete(self, version_id: UUID) -> None:
        """Delete a model version."""
        version_id_str = str(version_id)
        if version_id_str in self._versions:
            del self._versions[version_id_str]
    
    async def find_all(self) -> List[ModelVersion]:
        """Find all model versions."""
        return list(self._versions.values())