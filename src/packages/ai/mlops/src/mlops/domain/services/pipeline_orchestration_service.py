"""Domain service for managing ML pipelines and orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from datetime import datetime

from ..entities.pipeline import Pipeline, PipelineType, PipelineStatus


class InvalidPipelineStateError(Exception):
    """Raised when pipeline state is invalid for operation."""
    pass


class PipelineNotFoundError(Exception):
    """Raised when pipeline is not found."""
    pass


# Simplified data structures for pipeline orchestration
class PipelineExecution:
    """Represents a pipeline execution/run."""
    
    def __init__(self, pipeline_id: UUID, triggered_by: str = "manual"):
        self.id = uuid4()
        self.pipeline_id = pipeline_id
        self.status = "running"
        self.triggered_by = triggered_by
        self.started_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.outputs: Dict[str, Any] = {}
    
    def complete(self, outputs: Optional[Dict[str, Any]] = None):
        """Mark execution as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        if outputs:
            self.outputs.update(outputs)
    
    def fail(self, error_message: str):
        """Mark execution as failed."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        self.error_message = error_message


class PipelineOrchestrationService:
    """Service for managing ML pipelines and their orchestration."""
    
    def __init__(self):
        """Initialize the pipeline orchestration service."""
        self._pipelines: Dict[UUID, Pipeline] = {}
        self._executions: Dict[UUID, PipelineExecution] = {}
    
    def register_pipeline(self, pipeline: Pipeline) -> UUID:
        """Register a new pipeline for orchestration."""
        self._pipelines[pipeline.id] = pipeline
        return pipeline.id
    
    def execute_pipeline(self, pipeline_id: UUID, triggered_by: str = "manual") -> UUID:
        """Execute a pipeline and return execution ID."""
        if pipeline_id not in self._pipelines:
            raise PipelineNotFoundError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self._pipelines[pipeline_id]
        
        if not pipeline.stages:
            raise InvalidPipelineStateError("Pipeline has no stages to execute")
        
        # Create execution
        execution = PipelineExecution(pipeline_id, triggered_by)
        self._executions[execution.id] = execution
        
        # Start pipeline
        pipeline.start()
        
        return execution.id
    
    def get_execution_status(self, execution_id: UUID) -> Dict[str, Any]:
        """Get the status of a pipeline execution."""
        if execution_id not in self._executions:
            raise Exception(f"Execution {execution_id} not found")
        
        execution = self._executions[execution_id]
        pipeline = self._pipelines[execution.pipeline_id]
        
        return {
            "execution_id": str(execution.id),
            "pipeline_id": str(execution.pipeline_id),
            "pipeline_name": pipeline.name,
            "status": execution.status,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "triggered_by": execution.triggered_by,
            "error_message": execution.error_message,
            "stage_count": len(pipeline.stages),
            "outputs": execution.outputs,
        }
    
    def complete_execution(self, execution_id: UUID, outputs: Optional[Dict[str, Any]] = None):
        """Mark an execution as completed."""
        if execution_id not in self._executions:
            raise Exception(f"Execution {execution_id} not found")
        
        execution = self._executions[execution_id]
        pipeline = self._pipelines[execution.pipeline_id]
        
        execution.complete(outputs)
        pipeline.complete()
    
    def fail_execution(self, execution_id: UUID, error_message: str):
        """Mark an execution as failed."""
        if execution_id not in self._executions:
            raise Exception(f"Execution {execution_id} not found")
        
        execution = self._executions[execution_id]
        pipeline = self._pipelines[execution.pipeline_id]
        
        execution.fail(error_message)
        pipeline.fail(error_message)
    
    def get_pipeline_history(self, pipeline_id: UUID, limit: int = 10) -> List[Dict[str, Any]]:
        """Get execution history for a pipeline."""
        if pipeline_id not in self._pipelines:
            raise PipelineNotFoundError(f"Pipeline {pipeline_id} not found")
        
        # Find all executions for this pipeline
        executions = [
            exec for exec in self._executions.values() 
            if exec.pipeline_id == pipeline_id
        ]
        
        # Sort by start time, most recent first
        executions.sort(key=lambda x: x.started_at, reverse=True)
        
        return [
            {
                "execution_id": str(exec.id),
                "status": exec.status,
                "started_at": exec.started_at.isoformat(),
                "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                "triggered_by": exec.triggered_by,
                "error_message": exec.error_message,
            }
            for exec in executions[:limit]
        ]
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get all currently active pipeline executions."""
        active_executions = [
            exec for exec in self._executions.values()
            if exec.status == "running"
        ]
        
        return [
            {
                "execution_id": str(exec.id),
                "pipeline_id": str(exec.pipeline_id),
                "pipeline_name": self._pipelines[exec.pipeline_id].name,
                "started_at": exec.started_at.isoformat(),
                "triggered_by": exec.triggered_by,
            }
            for exec in active_executions
        ]
