"""Dataset repository interface for MLOps domain."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID


class DatasetRepository(ABC):
    """Abstract repository for datasets."""
    
    @abstractmethod
    async def save(self, dataset_id: UUID, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Save dataset metadata."""
        pass
    
    @abstractmethod
    async def find_by_id(self, dataset_id: UUID) -> Optional[Dict[str, Any]]:
        """Find dataset by ID."""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find dataset by name."""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[Dict[str, Any]]:
        """Find all datasets."""
        pass
    
    @abstractmethod
    async def delete(self, dataset_id: UUID) -> None:
        """Delete a dataset."""
        pass