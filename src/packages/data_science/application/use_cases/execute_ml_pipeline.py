"""Use case for executing machine learning pipelines."""

import asyncio
from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime

from ..dto.ml_pipeline_dto import (
    ExecutePipelineRequestDTO,
    ExecutePipelineResponseDTO,
    PipelineStatusResponseDTO
)
from ...domain.entities.machine_learning_pipeline import MachineLearningPipeline, PipelineStatus
from ...domain.repositories.machine_learning_pipeline_repository import IMachineLearningPipelineRepository


class ExecuteMLPipelineUseCase:
    """Use case for executing machine learning pipelines."""
    
    def __init__(self, pipeline_repository: IMachineLearningPipelineRepository):
        self._repository = pipeline_repository
    
    async def execute(self, request: ExecutePipelineRequestDTO) -> ExecutePipelineResponseDTO:
        """Execute pipeline execution use case.
        
        Args:
            request: Pipeline execution request parameters
            
        Returns:
            Pipeline execution response with execution details
            
        Raises:
            PipelineExecutionError: If pipeline execution fails to start
        """
        try:
            # Get pipeline
            pipeline = await self._repository.get_by_id(request.pipeline_id)
            if not pipeline:
                raise ValueError(f"Pipeline not found: {request.pipeline_id}")
            
            # Validate pipeline can be executed
            self._validate_pipeline_for_execution(pipeline)
            
            # Start execution
            execution_id = await self._repository.start_execution(
                request.pipeline_id,
                request.execution_config
            )
            
            # Update pipeline status
            await self._repository.update_execution_status(
                request.pipeline_id,
                "running",
                progress=0.0,
                current_step=pipeline.steps[0]["name"] if pipeline.steps else None
            )
            
            # Schedule asynchronous pipeline execution
            asyncio.create_task(self._execute_pipeline_async(
                pipeline, 
                execution_id,
                request.execution_config or {},
                request.input_data
            ))
            
            return ExecutePipelineResponseDTO(
                execution_id=execution_id,
                pipeline_id=request.pipeline_id,
                status="running",
                started_at=datetime.utcnow(),
                progress=0.0,
                current_step=pipeline.steps[0]["name"] if pipeline.steps else None
            )
            
        except Exception as e:
            raise ValueError(f"Failed to start pipeline execution: {str(e)}")
    
    async def get_pipeline_status(self, pipeline_id: Any) -> PipelineStatusResponseDTO:
        """Get current pipeline execution status."""
        try:
            pipeline = await self._repository.get_by_id(pipeline_id)
            if not pipeline:
                raise ValueError(f"Pipeline not found: {pipeline_id}")
            
            # Get execution logs
            logs = await self._repository.get_execution_logs(pipeline_id)
            
            # Get performance metrics if available
            metrics = await self._repository.get_pipeline_performance_metrics(pipeline_id)
            
            return PipelineStatusResponseDTO(
                pipeline_id=pipeline_id,
                execution_id=pipeline.current_execution_id,
                status=pipeline.status.value,
                progress=pipeline.execution_progress,
                current_step=pipeline.current_step,
                started_at=pipeline.last_execution_started,
                completed_at=pipeline.last_execution_completed,
                execution_time_seconds=pipeline.last_execution_duration_seconds,
                step_statuses=self._get_step_statuses(pipeline),
                metrics=metrics,
                logs=[log["message"] for log in logs[-10:]] if logs else None  # Last 10 logs
            )
            
        except Exception as e:
            raise ValueError(f"Failed to get pipeline status: {str(e)}")
    
    def _validate_pipeline_for_execution(self, pipeline: MachineLearningPipeline) -> None:
        """Validate that pipeline can be executed."""
        if pipeline.status == PipelineStatus.ARCHIVED:
            raise ValueError("Cannot execute archived pipeline")
        
        if not pipeline.steps:
            raise ValueError("Pipeline has no steps to execute")
        
        if not pipeline.is_valid():
            raise ValueError("Pipeline configuration is invalid")
    
    async def _execute_pipeline_async(
        self, 
        pipeline: MachineLearningPipeline,
        execution_id: str,
        execution_config: Dict[str, Any],
        input_data: Optional[Any]
    ) -> None:
        """Execute pipeline asynchronously."""
        try:
            total_steps = len(pipeline.steps)
            
            for i, step in enumerate(pipeline.steps):
                # Update current step
                await self._repository.update_execution_status(
                    pipeline.pipeline_id,
                    "running",
                    progress=(i / total_steps) * 100,
                    current_step=step["name"]
                )
                
                # Log step start
                await self._repository.add_execution_log(
                    pipeline.pipeline_id,
                    f"Starting step: {step['name']}",
                    step_name=step["name"],
                    level="INFO"
                )
                
                # Execute step
                step_result = await self._execute_step(step, execution_config, input_data)
                
                # Update step status
                await self._repository.update_step_status(
                    pipeline.pipeline_id,
                    step["name"],
                    "completed",
                    output=step_result
                )
                
                # Log step completion
                await self._repository.add_execution_log(
                    pipeline.pipeline_id,
                    f"Completed step: {step['name']}",
                    step_name=step["name"],
                    level="INFO"
                )
                
                # Use step output as input for next step
                input_data = step_result
            
            # Mark pipeline as completed
            await self._repository.update_execution_status(
                pipeline.pipeline_id,
                "completed",
                progress=100.0
            )
            
            await self._repository.add_execution_log(
                pipeline.pipeline_id,
                "Pipeline execution completed successfully",
                level="INFO"
            )
            
        except Exception as e:
            # Mark pipeline as failed
            await self._repository.update_execution_status(
                pipeline.pipeline_id,
                "failed",
                progress=None
            )
            
            await self._repository.add_execution_log(
                pipeline.pipeline_id,
                f"Pipeline execution failed: {str(e)}",
                level="ERROR"
            )
    
    async def _execute_step(
        self, 
        step: Dict[str, Any], 
        execution_config: Dict[str, Any],
        input_data: Any
    ) -> Any:
        """Execute a single pipeline step."""
        step_type = step["type"]
        step_config = step["config"]
        
        # Mock step execution - in real implementation, this would:
        # 1. Load the appropriate step executor
        # 2. Execute the step with the given configuration
        # 3. Return the step output
        
        if step_type == "data_preprocessing":
            return await self._execute_data_preprocessing_step(step_config, input_data)
        elif step_type == "feature_engineering":
            return await self._execute_feature_engineering_step(step_config, input_data)
        elif step_type == "model_training":
            return await self._execute_model_training_step(step_config, input_data)
        elif step_type == "model_validation":
            return await self._execute_model_validation_step(step_config, input_data)
        elif step_type == "model_deployment":
            return await self._execute_model_deployment_step(step_config, input_data)
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    async def _execute_data_preprocessing_step(self, config: Dict[str, Any], input_data: Any) -> Any:
        """Execute data preprocessing step."""
        # Mock implementation
        await asyncio.sleep(2)  # Simulate processing time
        return {"processed_data": "mock_processed_data", "preprocessing_stats": {"rows": 1000}}
    
    async def _execute_feature_engineering_step(self, config: Dict[str, Any], input_data: Any) -> Any:
        """Execute feature engineering step."""
        # Mock implementation
        await asyncio.sleep(3)  # Simulate processing time
        return {"features": "mock_features", "feature_stats": {"num_features": 50}}
    
    async def _execute_model_training_step(self, config: Dict[str, Any], input_data: Any) -> Any:
        """Execute model training step."""
        # Mock implementation
        await asyncio.sleep(5)  # Simulate training time
        return {"model": "mock_trained_model", "training_metrics": {"accuracy": 0.85}}
    
    async def _execute_model_validation_step(self, config: Dict[str, Any], input_data: Any) -> Any:
        """Execute model validation step."""
        # Mock implementation
        await asyncio.sleep(2)  # Simulate validation time
        return {"validation_results": {"accuracy": 0.83, "precision": 0.84, "recall": 0.82}}
    
    async def _execute_model_deployment_step(self, config: Dict[str, Any], input_data: Any) -> Any:
        """Execute model deployment step."""
        # Mock implementation
        await asyncio.sleep(1)  # Simulate deployment time
        return {"deployment_url": "https://api.example.com/model/predict", "deployment_id": "deploy_123"}
    
    def _get_step_statuses(self, pipeline: MachineLearningPipeline) -> List[Dict[str, Any]]:
        """Get status of all pipeline steps."""
        step_statuses = []
        
        for step in pipeline.steps:
            # In real implementation, this would query step execution status
            step_statuses.append({
                "name": step["name"],
                "type": step["type"],
                "status": "completed",  # Mock status
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "duration_seconds": 30
            })
        
        return step_statuses