"""Pipeline Repository Contract

Repository interface for pipeline persistence and retrieval operations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from pynomaly_mlops.domain.entities.pipeline import Pipeline, PipelineRun, PipelineStatus


class PipelineRepository(ABC):
    """Abstract repository for pipeline persistence operations."""
    
    @abstractmethod
    async def save(self, pipeline: Pipeline) -> Pipeline:
        """Save a pipeline to the repository.
        
        Args:
            pipeline: Pipeline to save
            
        Returns:
            The saved pipeline with updated metadata
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, pipeline_id: UUID) -> Optional[Pipeline]:
        """Find a pipeline by its ID.
        
        Args:
            pipeline_id: Pipeline ID to search for
            
        Returns:
            Pipeline if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Pipeline]:
        """Find a pipeline by its name.
        
        Args:
            name: Pipeline name to search for
            
        Returns:
            Pipeline if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_all(self) -> List[Pipeline]:
        """Find all pipelines.
        
        Returns:
            List of all pipelines
        """
        pass
    
    @abstractmethod
    async def find_by_status(self, status: PipelineStatus) -> List[Pipeline]:
        """Find pipelines by status.
        
        Args:
            status: Pipeline status to filter by
            
        Returns:
            List of pipelines with the specified status
        """
        pass
    
    @abstractmethod
    async def find_by_tags(self, tags: List[str]) -> List[Pipeline]:
        """Find pipelines by tags.
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of pipelines matching any of the tags
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        name_pattern: Optional[str] = None,
        status: Optional[PipelineStatus] = None,
        tags: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Pipeline]:
        """Search pipelines with multiple criteria.
        
        Args:
            name_pattern: Pattern to match against pipeline names
            status: Pipeline status to filter by
            tags: List of tags to filter by
            created_after: Find pipelines created after this date
            created_before: Find pipelines created before this date
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of pipelines matching the search criteria
        """
        pass
    
    @abstractmethod
    async def delete(self, pipeline_id: UUID) -> bool:
        """Delete a pipeline from the repository.
        
        Args:
            pipeline_id: Pipeline ID to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_pipeline_history(
        self, 
        pipeline_id: UUID, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get execution history for a pipeline.
        
        Args:
            pipeline_id: Pipeline ID to get history for
            limit: Maximum number of history entries to return
            
        Returns:
            List of pipeline execution history entries
        """
        pass


class PipelineRunRepository(ABC):
    """Abstract repository for pipeline run persistence operations."""
    
    @abstractmethod
    async def save(self, pipeline_run: PipelineRun) -> PipelineRun:
        """Save a pipeline run to the repository.
        
        Args:
            pipeline_run: Pipeline run to save
            
        Returns:
            The saved pipeline run with updated metadata
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, run_id: UUID) -> Optional[PipelineRun]:
        """Find a pipeline run by its ID.
        
        Args:
            run_id: Pipeline run ID to search for
            
        Returns:
            Pipeline run if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_by_pipeline_id(
        self, 
        pipeline_id: UUID,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[PipelineRun]:
        """Find pipeline runs by pipeline ID.
        
        Args:
            pipeline_id: Pipeline ID to search for
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of pipeline runs for the specified pipeline
        """
        pass
    
    @abstractmethod
    async def find_by_status(
        self, 
        status: PipelineStatus,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[PipelineRun]:
        """Find pipeline runs by status.
        
        Args:
            status: Pipeline run status to filter by
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of pipeline runs with the specified status
        """
        pass
    
    @abstractmethod
    async def find_latest_for_pipeline(self, pipeline_id: UUID) -> Optional[PipelineRun]:
        """Find the latest run for a specific pipeline.
        
        Args:
            pipeline_id: Pipeline ID to search for
            
        Returns:
            Latest pipeline run if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_running_runs(self) -> List[PipelineRun]:
        """Find all currently running pipeline runs.
        
        Returns:
            List of running pipeline runs
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        pipeline_id: Optional[UUID] = None,
        status: Optional[PipelineStatus] = None,
        triggered_by: Optional[str] = None,
        trigger_type: Optional[str] = None,
        started_after: Optional[datetime] = None,
        started_before: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[PipelineRun]:
        """Search pipeline runs with multiple criteria.
        
        Args:
            pipeline_id: Pipeline ID to filter by
            status: Pipeline run status to filter by
            triggered_by: User who triggered the run
            trigger_type: Type of trigger (manual, scheduled, etc.)
            started_after: Find runs started after this date
            started_before: Find runs started before this date
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of pipeline runs matching the search criteria
        """
        pass
    
    @abstractmethod
    async def delete(self, run_id: UUID) -> bool:
        """Delete a pipeline run from the repository.
        
        Args:
            run_id: Pipeline run ID to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_run_statistics(
        self, 
        pipeline_id: Optional[UUID] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get pipeline run statistics.
        
        Args:
            pipeline_id: Optional pipeline ID to filter by
            days: Number of days to include in statistics
            
        Returns:
            Dictionary containing run statistics
        """
        pass