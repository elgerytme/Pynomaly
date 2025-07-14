"""Pipeline Execution Engine

Advanced DAG-based pipeline execution engine with parallel processing,
resource management, monitoring, and failure recovery capabilities.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable
from uuid import UUID, uuid4

from pynomaly_mlops.domain.entities.pipeline import (
    Pipeline, PipelineStep, StepStatus, PipelineStatus, PipelineRun
)


class ExecutionContext:
    """Execution context for pipeline runs."""
    
    def __init__(self, pipeline: Pipeline, run_id: UUID):
        self.pipeline = pipeline
        self.run_id = run_id
        self.started_at = datetime.now(timezone.utc)
        self.running_steps: Set[UUID] = set()
        self.completed_steps: Set[UUID] = set()
        self.failed_steps: Set[UUID] = set()
        self.step_outputs: Dict[UUID, Dict[str, Any]] = {}
        self.artifacts: Dict[str, str] = {}
        self.metrics: Dict[str, float] = {}
        self.cancelled = False


class StepExecutor:
    """Executor for individual pipeline steps."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    async def execute_step(
        self, 
        step: PipelineStep, 
        context: ExecutionContext,
        step_executor_fn: Optional[Callable[[PipelineStep, ExecutionContext], Awaitable[Dict[str, Any]]]] = None
    ) -> bool:
        """Execute a single pipeline step.
        
        Args:
            step: The step to execute
            context: Execution context
            step_executor_fn: Optional custom step executor function
            
        Returns:
            True if step completed successfully, False otherwise
        """
        self.logger.info(f"Starting execution of step {step.name} (ID: {step.id})")
        
        try:
            step.start_execution()
            
            # Execute the step
            if step_executor_fn:
                # Use custom executor
                outputs = await step_executor_fn(step, context)
            else:
                # Default execution (placeholder)
                outputs = await self._default_execute_step(step, context)
            
            # Mark step as completed
            step.complete_execution(exit_code=0, stdout="Step completed successfully")
            context.step_outputs[step.id] = outputs
            context.completed_steps.add(step.id)
            
            self.logger.info(f"Step {step.name} completed successfully")
            return True
            
        except Exception as e:
            # Mark step as failed
            error_msg = f"Step execution failed: {str(e)}"
            step.fail_execution(exit_code=1, stderr=error_msg)
            context.failed_steps.add(step.id)
            
            self.logger.error(f"Step {step.name} failed: {error_msg}")
            return False
            
        finally:
            context.running_steps.discard(step.id)
    
    async def _default_execute_step(
        self, 
        step: PipelineStep, 
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Default step execution implementation."""
        # Simulate step execution
        await asyncio.sleep(0.1)  # Minimal delay for testing
        
        return {
            "step_id": str(step.id),
            "step_name": step.name,
            "execution_time": time.time(),
            "status": "completed"
        }


class PipelineExecutor:
    """Advanced pipeline execution engine with DAG orchestration."""
    
    def __init__(
        self, 
        step_executor_fn: Optional[Callable[[PipelineStep, ExecutionContext], Awaitable[Dict[str, Any]]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.step_executor_fn = step_executor_fn
        self.logger = logger or logging.getLogger(__name__)
        self.step_executor = StepExecutor(logger)
        
        # Execution state
        self.active_executions: Dict[UUID, ExecutionContext] = {}
    
    async def execute_pipeline(
        self, 
        pipeline: Pipeline,
        run_id: Optional[UUID] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> PipelineRun:
        """Execute a complete pipeline with DAG orchestration.
        
        Args:
            pipeline: The pipeline to execute
            run_id: Optional run ID (generated if not provided)
            parameters: Optional runtime parameters
            
        Returns:
            PipelineRun with execution results
        """
        if not run_id:
            run_id = uuid4()
        
        self.logger.info(f"Starting pipeline execution: {pipeline.name} (Run ID: {run_id})")
        
        # Validate pipeline DAG
        validation_errors = pipeline.validate_dag()
        if validation_errors:
            raise ValueError(f"Pipeline validation failed: {validation_errors}")
        
        # Create execution context
        context = ExecutionContext(pipeline, run_id)
        self.active_executions[run_id] = context
        
        # Create pipeline run
        pipeline_run = PipelineRun(
            id=run_id,
            pipeline_id=pipeline.id,
            pipeline_version=pipeline.version,
            parameters=parameters or {},
            status=PipelineStatus.RUNNING
        )
        
        try:
            # Start pipeline execution
            pipeline.start_pipeline(run_id)
            
            # Execute pipeline steps according to DAG
            success = await self._execute_dag(pipeline, context)
            
            if success and not context.cancelled:
                # Mark pipeline as completed
                pipeline.complete_pipeline()
                pipeline_run.complete_run()
                self.logger.info(f"Pipeline {pipeline.name} completed successfully")
            else:
                # Mark pipeline as failed
                pipeline.fail_pipeline()
                pipeline_run.fail_run()
                self.logger.error(f"Pipeline {pipeline.name} failed or was cancelled")
            
            # Set run artifacts and metrics
            pipeline_run.artifacts = context.artifacts
            pipeline_run.metrics = context.metrics
            
            return pipeline_run
            
        except Exception as e:
            self.logger.error(f"Pipeline execution error: {str(e)}")
            pipeline.fail_pipeline()
            pipeline_run.fail_run()
            raise
            
        finally:
            # Clean up execution context
            self.active_executions.pop(run_id, None)
    
    async def _execute_dag(self, pipeline: Pipeline, context: ExecutionContext) -> bool:
        """Execute pipeline steps according to DAG dependencies.
        
        Args:
            pipeline: The pipeline to execute
            context: Execution context
            
        Returns:
            True if all steps completed successfully, False otherwise
        """
        execution_levels = pipeline.get_execution_order()
        
        for level in execution_levels:
            if context.cancelled:
                self.logger.info("Pipeline execution cancelled")
                return False
            
            # Execute steps in parallel for this level
            level_success = await self._execute_level(pipeline, level, context)
            
            if not level_success:
                self.logger.error("Level execution failed, stopping pipeline")
                return False
        
        return True
    
    async def _execute_level(
        self, 
        pipeline: Pipeline, 
        step_ids: List[UUID], 
        context: ExecutionContext
    ) -> bool:
        """Execute a level of steps in parallel.
        
        Args:
            pipeline: The pipeline being executed
            step_ids: List of step IDs to execute in parallel
            context: Execution context
            
        Returns:
            True if all steps in level completed successfully, False otherwise
        """
        if not step_ids:
            return True
        
        self.logger.info(f"Executing level with {len(step_ids)} steps")
        
        # Filter steps that can actually be started
        ready_steps = []
        for step_id in step_ids:
            if pipeline.can_start_step(step_id):
                step = pipeline.steps[step_id]
                ready_steps.append(step)
                context.running_steps.add(step_id)
            else:
                self.logger.warning(f"Step {step_id} not ready to execute, skipping")
        
        if not ready_steps:
            return True
        
        # Execute steps in parallel with concurrency limit
        max_parallel = min(len(ready_steps), pipeline.max_parallel_steps)
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_with_semaphore(step: PipelineStep) -> bool:
            async with semaphore:
                return await self.step_executor.execute_step(
                    step, context, self.step_executor_fn
                )
        
        # Execute all steps in parallel
        tasks = [execute_with_semaphore(step) for step in ready_steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        all_successful = True
        for i, result in enumerate(results):
            step = ready_steps[i]
            
            if isinstance(result, Exception):
                self.logger.error(f"Step {step.name} failed with exception: {result}")
                context.failed_steps.add(step.id)
                all_successful = False
            elif not result:
                self.logger.error(f"Step {step.name} failed")
                all_successful = False
        
        return all_successful
    
    async def cancel_pipeline(self, run_id: UUID) -> bool:
        """Cancel a running pipeline execution.
        
        Args:
            run_id: The run ID to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        context = self.active_executions.get(run_id)
        if not context:
            self.logger.warning(f"No active execution found for run ID: {run_id}")
            return False
        
        self.logger.info(f"Cancelling pipeline execution: {run_id}")
        context.cancelled = True
        context.pipeline.cancel_pipeline()
        
        return True
    
    def get_execution_status(self, run_id: UUID) -> Optional[Dict[str, Any]]:
        """Get the status of a running pipeline execution.
        
        Args:
            run_id: The run ID to check
            
        Returns:
            Execution status information or None if not found
        """
        context = self.active_executions.get(run_id)
        if not context:
            return None
        
        return {
            "run_id": str(run_id),
            "pipeline_id": str(context.pipeline.id),
            "pipeline_name": context.pipeline.name,
            "started_at": context.started_at.isoformat(),
            "running_steps": len(context.running_steps),
            "completed_steps": len(context.completed_steps),
            "failed_steps": len(context.failed_steps),
            "total_steps": len(context.pipeline.steps),
            "progress": context.pipeline.get_progress(),
            "cancelled": context.cancelled
        }
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active pipeline executions.
        
        Returns:
            List of active execution information
        """
        return [
            self.get_execution_status(run_id) 
            for run_id in self.active_executions.keys()
        ]


class PipelineScheduler:
    """Scheduler for pipeline execution with cron support."""
    
    def __init__(self, executor: PipelineExecutor, logger: Optional[logging.Logger] = None):
        self.executor = executor
        self.logger = logger or logging.getLogger(__name__)
        self.scheduled_pipelines: Dict[UUID, Pipeline] = {}
        self.running = False
    
    def add_scheduled_pipeline(self, pipeline: Pipeline) -> None:
        """Add a pipeline to the scheduler.
        
        Args:
            pipeline: Pipeline with schedule configuration
        """
        if pipeline.schedule and pipeline.schedule.enabled:
            self.scheduled_pipelines[pipeline.id] = pipeline
            self.logger.info(f"Added pipeline {pipeline.name} to scheduler")
    
    def remove_scheduled_pipeline(self, pipeline_id: UUID) -> None:
        """Remove a pipeline from the scheduler.
        
        Args:
            pipeline_id: Pipeline ID to remove
        """
        if pipeline_id in self.scheduled_pipelines:
            del self.scheduled_pipelines[pipeline_id]
            self.logger.info(f"Removed pipeline {pipeline_id} from scheduler")
    
    async def start_scheduler(self) -> None:
        """Start the pipeline scheduler."""
        self.running = True
        self.logger.info("Pipeline scheduler started")
        
        while self.running:
            try:
                await self._check_scheduled_pipelines()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler error: {str(e)}")
                await asyncio.sleep(60)  # Continue after error
    
    def stop_scheduler(self) -> None:
        """Stop the pipeline scheduler."""
        self.running = False
        self.logger.info("Pipeline scheduler stopped")
    
    async def _check_scheduled_pipelines(self) -> None:
        """Check for pipelines that need to be executed."""
        current_time = datetime.now(timezone.utc)
        
        for pipeline in self.scheduled_pipelines.values():
            if await self._should_execute_pipeline(pipeline, current_time):
                try:
                    await self.executor.execute_pipeline(
                        pipeline,
                        parameters={"trigger_type": "scheduled"}
                    )
                    self.logger.info(f"Executed scheduled pipeline: {pipeline.name}")
                except Exception as e:
                    self.logger.error(f"Failed to execute scheduled pipeline {pipeline.name}: {str(e)}")
    
    async def _should_execute_pipeline(self, pipeline: Pipeline, current_time: datetime) -> bool:
        """Check if a pipeline should be executed now.
        
        Args:
            pipeline: Pipeline to check
            current_time: Current timestamp
            
        Returns:
            True if pipeline should be executed, False otherwise
        """
        if not pipeline.schedule or not pipeline.schedule.enabled:
            return False
        
        # Simple cron check (basic implementation)
        if pipeline.schedule.cron_expression:
            # This is a simplified check - in production, use a proper cron library
            return await self._check_cron_expression(pipeline.schedule.cron_expression, current_time)
        
        return False
    
    async def _check_cron_expression(self, cron_expr: str, current_time: datetime) -> bool:
        """Check if current time matches cron expression.
        
        Args:
            cron_expr: Cron expression to check
            current_time: Current timestamp
            
        Returns:
            True if time matches, False otherwise
        """
        # Simplified cron implementation - in production use croniter or similar
        # This is just a placeholder for demonstration
        return False