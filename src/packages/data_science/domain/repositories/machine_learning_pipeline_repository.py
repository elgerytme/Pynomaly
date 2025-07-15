"""Machine Learning Pipeline repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from ..entities.machine_learning_pipeline import MachineLearningPipeline


class IMachineLearningPipelineRepository(ABC):
    """Repository interface for machine learning pipeline persistence."""
    
    @abstractmethod
    async def save(self, pipeline: MachineLearningPipeline) -> None:
        """Save a machine learning pipeline."""
        pass
    
    @abstractmethod
    async def get_by_id(self, pipeline_id: UUID) -> Optional[MachineLearningPipeline]:
        """Get pipeline by ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str, version: Optional[str] = None) -> Optional[MachineLearningPipeline]:
        """Get pipeline by name and optional version."""
        pass
    
    @abstractmethod
    async def get_by_user_id(self, user_id: UUID) -> List[MachineLearningPipeline]:
        """Get all pipelines created by a user."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str) -> List[MachineLearningPipeline]:
        """Get pipelines by execution status."""
        pass
    
    @abstractmethod
    async def get_by_pipeline_type(self, pipeline_type: str) -> List[MachineLearningPipeline]:
        """Get pipelines by type."""
        pass
    
    @abstractmethod
    async def get_active_pipelines(self) -> List[MachineLearningPipeline]:
        """Get all active/running pipelines."""
        pass
    
    @abstractmethod
    async def get_scheduled_pipelines(self) -> List[MachineLearningPipeline]:
        """Get scheduled pipelines."""
        pass
    
    @abstractmethod
    async def get_pipelines_by_model_id(self, model_id: UUID) -> List[MachineLearningPipeline]:
        """Get pipelines associated with a model."""
        pass
    
    @abstractmethod
    async def get_pipelines_by_dataset_id(self, dataset_id: UUID) -> List[MachineLearningPipeline]:
        """Get pipelines using a dataset."""
        pass
    
    @abstractmethod
    async def get_pipeline_versions(self, pipeline_name: str) -> List[MachineLearningPipeline]:
        """Get all versions of a pipeline."""
        pass
    
    @abstractmethod
    async def get_latest_version(self, pipeline_name: str) -> Optional[MachineLearningPipeline]:
        """Get latest version of a pipeline."""
        pass
    
    @abstractmethod
    async def get_pipelines_by_date_range(self, start_date: datetime, end_date: datetime) -> List[MachineLearningPipeline]:
        """Get pipelines created within date range."""
        pass
    
    @abstractmethod
    async def get_pipelines_by_execution_time(self, min_duration: float, max_duration: Optional[float] = None) -> List[MachineLearningPipeline]:
        """Get pipelines by execution duration."""
        pass
    
    @abstractmethod
    async def get_failed_pipelines(self, since: Optional[datetime] = None) -> List[MachineLearningPipeline]:
        """Get failed pipelines, optionally since a date."""
        pass
    
    @abstractmethod
    async def get_pipelines_with_dependencies(self) -> List[MachineLearningPipeline]:
        """Get pipelines that have dependencies."""
        pass
    
    @abstractmethod
    async def get_dependent_pipelines(self, pipeline_id: UUID) -> List[MachineLearningPipeline]:
        """Get pipelines that depend on the given pipeline."""
        pass
    
    @abstractmethod
    async def search_pipelines(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[MachineLearningPipeline]:
        """Search pipelines by query and filters."""
        pass
    
    @abstractmethod
    async def get_pipeline_execution_history(self, pipeline_id: UUID) -> List[Dict[str, Any]]:
        """Get pipeline execution history."""
        pass
    
    @abstractmethod
    async def get_pipeline_lineage(self, pipeline_id: UUID) -> Dict[str, Any]:
        """Get pipeline lineage information."""
        pass
    
    @abstractmethod
    async def get_pipeline_performance_metrics(self, pipeline_id: UUID) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        pass
    
    @abstractmethod
    async def update_execution_status(self, pipeline_id: UUID, status: str, 
                                    progress: Optional[float] = None,
                                    current_step: Optional[str] = None) -> None:
        """Update pipeline execution status."""
        pass
    
    @abstractmethod
    async def update_step_status(self, pipeline_id: UUID, step_name: str, 
                               status: str, output: Optional[Dict[str, Any]] = None) -> None:
        """Update individual step status."""
        pass
    
    @abstractmethod
    async def add_execution_log(self, pipeline_id: UUID, log_entry: str, 
                              step_name: Optional[str] = None, level: str = "INFO") -> None:
        """Add execution log entry."""
        pass
    
    @abstractmethod
    async def get_execution_logs(self, pipeline_id: UUID, step_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pipeline execution logs."""
        pass
    
    @abstractmethod
    async def start_execution(self, pipeline_id: UUID, execution_config: Optional[Dict[str, Any]] = None) -> str:
        """Start pipeline execution and return execution ID."""
        pass
    
    @abstractmethod
    async def stop_execution(self, pipeline_id: UUID, reason: Optional[str] = None) -> None:
        """Stop pipeline execution."""
        pass
    
    @abstractmethod
    async def pause_execution(self, pipeline_id: UUID) -> None:
        """Pause pipeline execution."""
        pass
    
    @abstractmethod
    async def resume_execution(self, pipeline_id: UUID) -> None:
        """Resume paused pipeline execution."""
        pass
    
    @abstractmethod
    async def retry_failed_step(self, pipeline_id: UUID, step_name: str) -> None:
        """Retry a failed pipeline step."""
        pass
    
    @abstractmethod
    async def clone_pipeline(self, pipeline_id: UUID, new_name: str, user_id: UUID) -> UUID:
        """Clone an existing pipeline."""
        pass
    
    @abstractmethod
    async def update_schedule(self, pipeline_id: UUID, schedule_config: Dict[str, Any]) -> None:
        """Update pipeline schedule."""
        pass
    
    @abstractmethod
    async def enable_pipeline(self, pipeline_id: UUID) -> None:
        """Enable pipeline for execution."""
        pass
    
    @abstractmethod
    async def disable_pipeline(self, pipeline_id: UUID, reason: Optional[str] = None) -> None:
        """Disable pipeline execution."""
        pass
    
    @abstractmethod
    async def archive_pipeline(self, pipeline_id: UUID) -> None:
        """Archive a pipeline."""
        pass
    
    @abstractmethod
    async def restore_pipeline(self, pipeline_id: UUID) -> None:
        """Restore an archived pipeline."""
        pass
    
    @abstractmethod
    async def delete(self, pipeline_id: UUID) -> None:
        """Delete a pipeline."""
        pass
    
    @abstractmethod
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[MachineLearningPipeline]:
        """List all pipelines with pagination."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count total number of pipelines."""
        pass
    
    @abstractmethod
    async def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        pass
    
    @abstractmethod
    async def export_pipeline(self, pipeline_id: UUID, format_type: str = "yaml") -> bytes:
        """Export pipeline definition."""
        pass
    
    @abstractmethod
    async def import_pipeline(self, pipeline_data: bytes, format_type: str = "yaml", user_id: UUID) -> UUID:
        """Import pipeline definition."""
        pass
    
    @abstractmethod
    async def validate_pipeline(self, pipeline_id: UUID) -> Dict[str, Any]:
        """Validate pipeline configuration."""
        pass
    
    @abstractmethod
    async def cleanup_old_executions(self, days_old: int) -> int:
        """Clean up old execution logs and artifacts."""
        pass