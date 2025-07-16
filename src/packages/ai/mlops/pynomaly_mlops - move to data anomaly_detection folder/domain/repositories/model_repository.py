"""Model Repository Contract

Abstract repository interface for Model entity persistence and querying.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from pynomaly_mlops.domain.entities.model import Model, ModelStatus, ModelType
from pynomaly_mlops.domain.value_objects.semantic_version import SemanticVersion


class ModelRepository(ABC):
    """Abstract repository for Model entities."""
    
    @abstractmethod
    async def save(self, model: Model) -> Model:
        """Save or update a model.
        
        Args:
            model: Model to save
            
        Returns:
            Saved model with updated metadata
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, model_id: UUID) -> Optional[Model]:
        """Get model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_name_and_version(
        self, 
        name: str, 
        version: SemanticVersion
    ) -> Optional[Model]:
        """Get model by name and version.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Model if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def list_by_name(self, name: str) -> List[Model]:
        """List all versions of a model by name.
        
        Args:
            name: Model name
            
        Returns:
            List of models with the given name
        """
        pass
    
    @abstractmethod
    async def list_by_status(self, status: ModelStatus) -> List[Model]:
        """List models by status.
        
        Args:
            status: Model status to filter by
            
        Returns:
            List of models with the given status
        """
        pass
    
    @abstractmethod
    async def list_by_type(self, model_type: ModelType) -> List[Model]:
        """List models by type.
        
        Args:
            model_type: Model type to filter by
            
        Returns:
            List of models with the given type
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        name_pattern: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        model_type: Optional[ModelType] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Model]:
        """Search models with filters.
        
        Args:
            name_pattern: Pattern to match model names
            status: Model status filter
            model_type: Model type filter
            tags: Tags to match (all must be present)
            created_by: Creator filter
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching models
        """
        pass
    
    @abstractmethod
    async def get_latest_by_name(self, name: str) -> Optional[Model]:
        """Get the latest version of a model by name.
        
        Args:
            name: Model name
            
        Returns:
            Latest version of the model, None if not found
        """
        pass
    
    @abstractmethod
    async def get_production_models(self) -> List[Model]:
        """Get all models currently in production.
        
        Returns:
            List of production models
        """
        pass
    
    @abstractmethod
    async def delete(self, model_id: UUID) -> bool:
        """Delete a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def exists(self, name: str, version: SemanticVersion) -> bool:
        """Check if a model with given name and version exists.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if model exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count models matching filters.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Number of matching models
        """
        pass
    
    @abstractmethod
    async def get_model_lineage(self, model_id: UUID) -> List[Model]:
        """Get the lineage (ancestors and descendants) of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of models in the lineage chain
        """
        pass