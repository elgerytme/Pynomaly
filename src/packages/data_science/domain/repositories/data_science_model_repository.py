"""Data Science Model repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from ..entities.data_science_model import DataScienceModel, ModelId, UserId


class IDataScienceModelRepository(ABC):
    """Repository interface for data science model persistence."""
    
    @abstractmethod
    async def save(self, model: DataScienceModel) -> None:
        """Save a data science model."""
        pass
    
    @abstractmethod
    async def get_by_id(self, model_id: ModelId) -> Optional[DataScienceModel]:
        """Get model by ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str, version: Optional[str] = None) -> Optional[DataScienceModel]:
        """Get model by name and optional version."""
        pass
    
    @abstractmethod
    async def get_by_user_id(self, user_id: UserId) -> List[DataScienceModel]:
        """Get all models created by a user."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str) -> List[DataScienceModel]:
        """Get models by status."""
        pass
    
    @abstractmethod
    async def get_by_model_type(self, model_type: str) -> List[DataScienceModel]:
        """Get models by type."""
        pass
    
    @abstractmethod
    async def get_deployed_models(self) -> List[DataScienceModel]:
        """Get all deployed models."""
        pass
    
    @abstractmethod
    async def get_models_by_performance_threshold(self, metric: str, threshold: float) -> List[DataScienceModel]:
        """Get models above performance threshold."""
        pass
    
    @abstractmethod
    async def search_models(self, query: str, tags: Optional[List[str]] = None) -> List[DataScienceModel]:
        """Search models by query and optional tags."""
        pass
    
    @abstractmethod
    async def get_model_versions(self, model_name: str) -> List[DataScienceModel]:
        """Get all versions of a model."""
        pass
    
    @abstractmethod
    async def get_latest_version(self, model_name: str) -> Optional[DataScienceModel]:
        """Get latest version of a model."""
        pass
    
    @abstractmethod
    async def update_deployment_status(self, model_id: ModelId, deployment_status: str, 
                                     deployment_config: Optional[Dict[str, Any]] = None) -> None:
        """Update model deployment status."""
        pass
    
    @abstractmethod
    async def update_performance_metrics(self, model_id: ModelId, metrics: Dict[str, Any]) -> None:
        """Update model performance metrics."""
        pass
    
    @abstractmethod
    async def delete(self, model_id: ModelId) -> None:
        """Delete a model."""
        pass
    
    @abstractmethod
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[DataScienceModel]:
        """List all models with pagination."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count total number of models."""
        pass
    
    @abstractmethod
    async def get_model_lineage(self, model_id: ModelId) -> Dict[str, Any]:
        """Get model lineage information."""
        pass
    
    @abstractmethod
    async def archive_model(self, model_id: ModelId) -> None:
        """Archive a model."""
        pass
    
    @abstractmethod
    async def restore_model(self, model_id: ModelId) -> None:
        """Restore an archived model."""
        pass