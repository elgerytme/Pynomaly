"""Pipeline repository interface for MLOps domain."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.pipeline import Pipeline


class PipelineRepository(ABC):
    """Abstract repository for pipelines."""
    
    @abstractmethod
    async def save(self, pipeline: Pipeline) -> Pipeline:
        """Save a pipeline."""
        pass
    
    @abstractmethod
    async def find_by_id(self, pipeline_id: UUID) -> Optional[Pipeline]:
        """Find pipeline by ID."""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Pipeline]:
        """Find pipeline by name."""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[Pipeline]:
        """Find all pipelines."""
        pass
    
    @abstractmethod
    async def find_by_type(self, pipeline_type: str) -> List[Pipeline]:
        """Find pipelines by type."""
        pass
    
    @abstractmethod
    async def find_scheduled(self) -> List[Pipeline]:
        """Find scheduled pipelines."""
        pass
    
    @abstractmethod
    async def delete(self, pipeline_id: UUID) -> None:
        """Delete a pipeline."""
        pass