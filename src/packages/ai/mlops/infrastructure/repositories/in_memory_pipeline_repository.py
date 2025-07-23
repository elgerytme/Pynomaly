"""In-memory repository implementation for pipelines."""

from typing import Dict, List, Optional
from uuid import UUID

from ...domain.entities.pipeline import Pipeline
from ...domain.repositories.pipeline_repository import PipelineRepository


class InMemoryPipelineRepository(PipelineRepository):
    """In-memory implementation of pipeline repository."""
    
    def __init__(self):
        self._pipelines: Dict[str, Pipeline] = {}
    
    async def save(self, pipeline: Pipeline) -> Pipeline:
        """Save a pipeline."""
        pipeline_id_str = str(pipeline.id)
        self._pipelines[pipeline_id_str] = pipeline
        return pipeline
    
    async def find_by_id(self, pipeline_id: UUID) -> Optional[Pipeline]:
        """Find pipeline by ID."""
        return self._pipelines.get(str(pipeline_id))
    
    async def find_by_name(self, name: str) -> Optional[Pipeline]:
        """Find pipeline by name."""
        for pipeline in self._pipelines.values():
            if pipeline.name == name:
                return pipeline
        return None
    
    async def find_all(self) -> List[Pipeline]:
        """Find all pipelines."""
        return list(self._pipelines.values())
    
    async def find_by_type(self, pipeline_type: str) -> List[Pipeline]:
        """Find pipelines by type."""
        return [
            pipeline for pipeline in self._pipelines.values()
            if pipeline.pipeline_type.value == pipeline_type
        ]
    
    async def find_scheduled(self) -> List[Pipeline]:
        """Find scheduled pipelines."""
        return [
            pipeline for pipeline in self._pipelines.values()
            if pipeline.is_scheduled
        ]
    
    async def delete(self, pipeline_id: UUID) -> None:
        """Delete a pipeline."""
        pipeline_id_str = str(pipeline_id)
        if pipeline_id_str in self._pipelines:
            del self._pipelines[pipeline_id_str]