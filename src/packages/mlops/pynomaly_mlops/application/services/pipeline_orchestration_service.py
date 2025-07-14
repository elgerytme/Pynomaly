"""Pipeline Orchestration Service

High-level service for managing pipeline lifecycle, execution, and monitoring
with comprehensive artifact management and lineage tracking.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Awaitable
from uuid import UUID, uuid4

from pynomaly_mlops.domain.entities.pipeline import (
    Pipeline, PipelineStep, PipelineRun, PipelineStatus, StepType
)
from pynomaly_mlops.domain.repositories.pipeline_repository import (
    PipelineRepository, PipelineRunRepository
)
from pynomaly_mlops.infrastructure.execution.pipeline_executor import (
    PipelineExecutor, PipelineScheduler, ExecutionContext
)


class PipelineOrchestrationService:
    """Comprehensive pipeline orchestration and management service."""
    
    def __init__(
        self,
        pipeline_repository: PipelineRepository,
        pipeline_run_repository: PipelineRunRepository,
        step_executor_fn: Optional[Callable[[PipelineStep, ExecutionContext], Awaitable[Dict[str, Any]]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.pipeline_repository = pipeline_repository
        self.pipeline_run_repository = pipeline_run_repository
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize execution engine
        self.executor = PipelineExecutor(step_executor_fn, logger)
        self.scheduler = PipelineScheduler(self.executor, logger)
        
        # Pipeline templates for common MLOps workflows
        self._pipeline_templates = {}
        self._initialize_templates()
    
    async def create_pipeline(
        self,
        name: str,
        description: Optional[str] = None,
        version: str = "1.0.0",
        created_by: Optional[str] = None
    ) -> Pipeline:
        """Create a new pipeline.
        
        Args:
            name: Pipeline name
            description: Optional description
            version: Pipeline version
            created_by: Creator identifier
            
        Returns:
            Created pipeline
        """
        pipeline = Pipeline(
            name=name,
            description=description,
            version=version,
            created_by=created_by
        )
        
        saved_pipeline = await self.pipeline_repository.save(pipeline)
        self.logger.info(f"Created pipeline: {name} (ID: {saved_pipeline.id})")
        
        return saved_pipeline
    
    async def create_pipeline_from_template(
        self,
        template_name: str,
        pipeline_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> Pipeline:
        """Create a pipeline from a predefined template.
        
        Args:
            template_name: Name of the template to use
            pipeline_name: Name for the new pipeline
            parameters: Template parameters
            created_by: Creator identifier
            
        Returns:
            Created pipeline from template
        """
        if template_name not in self._pipeline_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template_fn = self._pipeline_templates[template_name]
        pipeline = await template_fn(pipeline_name, parameters or {}, created_by)
        
        saved_pipeline = await self.pipeline_repository.save(pipeline)
        self.logger.info(f"Created pipeline from template '{template_name}': {pipeline_name}")
        
        return saved_pipeline
    
    async def add_step_to_pipeline(
        self,
        pipeline_id: UUID,
        step_name: str,
        step_type: StepType,
        command: str,
        depends_on: Optional[List[UUID]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        environment_variables: Optional[Dict[str, str]] = None
    ) -> Pipeline:
        """Add a step to an existing pipeline.
        
        Args:
            pipeline_id: Pipeline ID
            step_name: Name of the step
            step_type: Type of the step
            command: Command to execute
            depends_on: List of step IDs this step depends on
            parameters: Step parameters
            environment_variables: Environment variables
            
        Returns:
            Updated pipeline
        """
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        
        step = PipelineStep(
            name=step_name,
            step_type=step_type,
            command=command,
            depends_on=set(depends_on or []),
            parameters=parameters or {},
            environment_variables=environment_variables or {}
        )
        
        pipeline.add_step(step)
        
        # Validate DAG after adding step
        validation_errors = pipeline.validate_dag()
        if validation_errors:
            raise ValueError(f"Adding step would create invalid DAG: {validation_errors}")
        
        saved_pipeline = await self.pipeline_repository.save(pipeline)
        self.logger.info(f"Added step '{step_name}' to pipeline {pipeline.name}")
        
        return saved_pipeline
    
    async def execute_pipeline(
        self,
        pipeline_id: UUID,
        parameters: Optional[Dict[str, Any]] = None,
        triggered_by: Optional[str] = None
    ) -> PipelineRun:
        """Execute a pipeline.
        
        Args:
            pipeline_id: Pipeline ID to execute
            parameters: Runtime parameters
            triggered_by: User who triggered the execution
            
        Returns:
            Pipeline run with execution results
        """
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        
        self.logger.info(f"Starting execution of pipeline: {pipeline.name}")
        
        # Execute pipeline
        pipeline_run = await self.executor.execute_pipeline(
            pipeline,
            parameters=parameters
        )
        
        # Update run metadata
        if triggered_by:
            pipeline_run.triggered_by = triggered_by
        
        # Save run to database
        saved_run = await self.pipeline_run_repository.save(pipeline_run)
        
        # Update pipeline with run information
        await self.pipeline_repository.save(pipeline)
        
        return saved_run
    
    async def get_pipeline_execution_status(self, run_id: UUID) -> Optional[Dict[str, Any]]:
        """Get the status of a pipeline execution.
        
        Args:
            run_id: Pipeline run ID
            
        Returns:
            Execution status information
        """
        # Check if run is active in executor
        active_status = self.executor.get_execution_status(run_id)
        if active_status:
            return active_status
        
        # Check database for completed runs
        pipeline_run = await self.pipeline_run_repository.find_by_id(run_id)
        if pipeline_run:
            return {
                "run_id": str(run_id),
                "pipeline_id": str(pipeline_run.pipeline_id),
                "status": pipeline_run.status,
                "started_at": pipeline_run.started_at.isoformat(),
                "completed_at": pipeline_run.completed_at.isoformat() if pipeline_run.completed_at else None,
                "execution_duration": pipeline_run.execution_duration,
                "artifacts": pipeline_run.artifacts,
                "metrics": pipeline_run.metrics
            }
        
        return None
    
    async def cancel_pipeline_execution(self, run_id: UUID) -> bool:
        """Cancel a running pipeline execution.
        
        Args:
            run_id: Pipeline run ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        success = await self.executor.cancel_pipeline(run_id)
        
        if success:
            # Update run status in database
            pipeline_run = await self.pipeline_run_repository.find_by_id(run_id)
            if pipeline_run:
                pipeline_run.status = PipelineStatus.CANCELLED
                pipeline_run.completed_at = datetime.now(timezone.utc)
                await self.pipeline_run_repository.save(pipeline_run)
            
            self.logger.info(f"Cancelled pipeline execution: {run_id}")
        
        return success
    
    async def get_pipeline_history(
        self,
        pipeline_id: UUID,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get execution history for a pipeline.
        
        Args:
            pipeline_id: Pipeline ID
            limit: Maximum number of history entries
            
        Returns:
            List of execution history entries
        """
        runs = await self.pipeline_run_repository.find_by_pipeline_id(
            pipeline_id, limit=limit
        )
        
        history = []
        for run in runs:
            history.append({
                "run_id": str(run.id),
                "status": run.status,
                "started_at": run.started_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "execution_duration": run.execution_duration,
                "triggered_by": run.triggered_by,
                "trigger_type": run.trigger_type,
                "artifacts_count": len(run.artifacts),
                "metrics_count": len(run.metrics)
            })
        
        return history
    
    async def get_pipeline_statistics(
        self,
        pipeline_id: Optional[UUID] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get pipeline execution statistics.
        
        Args:
            pipeline_id: Optional pipeline ID to filter by
            days: Number of days to include in statistics
            
        Returns:
            Pipeline statistics
        """
        stats = await self.pipeline_run_repository.get_run_statistics(
            pipeline_id=pipeline_id,
            days=days
        )
        
        return stats
    
    async def start_scheduler(self) -> None:
        """Start the pipeline scheduler for automated executions."""
        # Add scheduled pipelines to scheduler
        pipelines = await self.pipeline_repository.find_all()
        for pipeline in pipelines:
            if pipeline.schedule and pipeline.schedule.enabled:
                self.scheduler.add_scheduled_pipeline(pipeline)
        
        # Start scheduler
        await self.scheduler.start_scheduler()
    
    def stop_scheduler(self) -> None:
        """Stop the pipeline scheduler."""
        self.scheduler.stop_scheduler()
    
    async def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active pipeline executions.
        
        Returns:
            List of active execution information
        """
        return self.executor.list_active_executions()
    
    async def search_pipelines(
        self,
        name_pattern: Optional[str] = None,
        status: Optional[PipelineStatus] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Pipeline]:
        """Search pipelines with multiple criteria.
        
        Args:
            name_pattern: Pattern to match against pipeline names
            status: Pipeline status to filter by
            tags: List of tags to filter by
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching pipelines
        """
        return await self.pipeline_repository.search(
            name_pattern=name_pattern,
            status=status,
            tags=tags,
            limit=limit,
            offset=offset
        )
    
    async def get_pipeline_lineage(self, pipeline_id: UUID) -> Dict[str, Any]:
        """Get lineage information for a pipeline.
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            Pipeline lineage information
        """
        # This would be implemented with lineage tracking
        # For now, return basic information
        pipeline = await self.pipeline_repository.find_by_id(pipeline_id)
        if not pipeline:
            return {}
        
        runs = await self.pipeline_run_repository.find_by_pipeline_id(
            pipeline_id, limit=10
        )
        
        return {
            "pipeline_id": str(pipeline_id),
            "pipeline_name": pipeline.name,
            "total_runs": len(runs),
            "recent_runs": [
                {
                    "run_id": str(run.id),
                    "status": run.status,
                    "started_at": run.started_at.isoformat(),
                    "artifacts": list(run.artifacts.keys()),
                    "metrics": list(run.metrics.keys())
                }
                for run in runs[:5]
            ]
        }
    
    def _initialize_templates(self) -> None:
        """Initialize pipeline templates for common MLOps workflows."""
        self._pipeline_templates = {
            "ml_training": self._create_ml_training_template,
            "model_deployment": self._create_model_deployment_template,
            "data_pipeline": self._create_data_pipeline_template,
            "model_validation": self._create_model_validation_template
        }
    
    async def _create_ml_training_template(
        self,
        name: str,
        parameters: Dict[str, Any],
        created_by: Optional[str]
    ) -> Pipeline:
        """Create ML training pipeline template."""
        pipeline = Pipeline(
            name=name,
            description="ML model training pipeline",
            created_by=created_by
        )
        
        # Data ingestion step
        data_step = PipelineStep(
            name="data_ingestion",
            step_type=StepType.DATA_INGESTION,
            command=parameters.get("data_command", "python scripts/ingest_data.py"),
            parameters={"data_source": parameters.get("data_source", "default")}
        )
        pipeline.add_step(data_step)
        
        # Data validation step
        validation_step = PipelineStep(
            name="data_validation",
            step_type=StepType.DATA_VALIDATION,
            command=parameters.get("validation_command", "python scripts/validate_data.py"),
            depends_on={data_step.id}
        )
        pipeline.add_step(validation_step)
        
        # Feature engineering step
        feature_step = PipelineStep(
            name="feature_engineering",
            step_type=StepType.FEATURE_ENGINEERING,
            command=parameters.get("feature_command", "python scripts/feature_engineering.py"),
            depends_on={validation_step.id}
        )
        pipeline.add_step(feature_step)
        
        # Model training step
        training_step = PipelineStep(
            name="model_training",
            step_type=StepType.MODEL_TRAINING,
            command=parameters.get("training_command", "python scripts/train_model.py"),
            depends_on={feature_step.id},
            parameters={
                "algorithm": parameters.get("algorithm", "random_forest"),
                "hyperparameters": parameters.get("hyperparameters", {})
            }
        )
        pipeline.add_step(training_step)
        
        # Model evaluation step
        evaluation_step = PipelineStep(
            name="model_evaluation",
            step_type=StepType.MODEL_EVALUATION,
            command=parameters.get("evaluation_command", "python scripts/evaluate_model.py"),
            depends_on={training_step.id}
        )
        pipeline.add_step(evaluation_step)
        
        return pipeline
    
    async def _create_model_deployment_template(
        self,
        name: str,
        parameters: Dict[str, Any],
        created_by: Optional[str]
    ) -> Pipeline:
        """Create model deployment pipeline template."""
        pipeline = Pipeline(
            name=name,
            description="Model deployment pipeline",
            created_by=created_by
        )
        
        # Model validation step
        validation_step = PipelineStep(
            name="model_validation",
            step_type=StepType.MODEL_VALIDATION,
            command=parameters.get("validation_command", "python scripts/validate_model.py"),
            parameters={"model_id": parameters.get("model_id")}
        )
        pipeline.add_step(validation_step)
        
        # Model deployment step
        deployment_step = PipelineStep(
            name="model_deployment",
            step_type=StepType.MODEL_DEPLOYMENT,
            command=parameters.get("deployment_command", "python scripts/deploy_model.py"),
            depends_on={validation_step.id},
            parameters={
                "environment": parameters.get("environment", "staging"),
                "scaling_config": parameters.get("scaling_config", {})
            }
        )
        pipeline.add_step(deployment_step)
        
        # Monitoring setup step
        monitoring_step = PipelineStep(
            name="setup_monitoring",
            step_type=StepType.MONITORING,
            command=parameters.get("monitoring_command", "python scripts/setup_monitoring.py"),
            depends_on={deployment_step.id}
        )
        pipeline.add_step(monitoring_step)
        
        return pipeline
    
    async def _create_data_pipeline_template(
        self,
        name: str,
        parameters: Dict[str, Any],
        created_by: Optional[str]
    ) -> Pipeline:
        """Create data processing pipeline template."""
        pipeline = Pipeline(
            name=name,
            description="Data processing pipeline",
            created_by=created_by
        )
        
        # Data ingestion
        ingestion_step = PipelineStep(
            name="data_ingestion",
            step_type=StepType.DATA_INGESTION,
            command=parameters.get("ingestion_command", "python scripts/ingest_data.py"),
            parameters={"sources": parameters.get("data_sources", [])}
        )
        pipeline.add_step(ingestion_step)
        
        # Data preprocessing
        preprocessing_step = PipelineStep(
            name="data_preprocessing",
            step_type=StepType.DATA_PREPROCESSING,
            command=parameters.get("preprocessing_command", "python scripts/preprocess_data.py"),
            depends_on={ingestion_step.id}
        )
        pipeline.add_step(preprocessing_step)
        
        # Data validation
        validation_step = PipelineStep(
            name="data_validation",
            step_type=StepType.DATA_VALIDATION,
            command=parameters.get("validation_command", "python scripts/validate_data.py"),
            depends_on={preprocessing_step.id}
        )
        pipeline.add_step(validation_step)
        
        return pipeline
    
    async def _create_model_validation_template(
        self,
        name: str,
        parameters: Dict[str, Any],
        created_by: Optional[str]
    ) -> Pipeline:
        """Create model validation pipeline template."""
        pipeline = Pipeline(
            name=name,
            description="Model validation pipeline",
            created_by=created_by
        )
        
        # Model loading
        loading_step = PipelineStep(
            name="load_model",
            step_type=StepType.CUSTOM,
            command=parameters.get("loading_command", "python scripts/load_model.py"),
            parameters={"model_id": parameters.get("model_id")}
        )
        pipeline.add_step(loading_step)
        
        # Performance evaluation
        performance_step = PipelineStep(
            name="performance_evaluation",
            step_type=StepType.MODEL_EVALUATION,
            command=parameters.get("performance_command", "python scripts/evaluate_performance.py"),
            depends_on={loading_step.id}
        )
        pipeline.add_step(performance_step)
        
        # Bias testing
        bias_step = PipelineStep(
            name="bias_testing",
            step_type=StepType.MODEL_VALIDATION,
            command=parameters.get("bias_command", "python scripts/test_bias.py"),
            depends_on={loading_step.id}
        )
        pipeline.add_step(bias_step)
        
        # Security validation
        security_step = PipelineStep(
            name="security_validation",
            step_type=StepType.MODEL_VALIDATION,
            command=parameters.get("security_command", "python scripts/validate_security.py"),
            depends_on={loading_step.id}
        )
        pipeline.add_step(security_step)
        
        # Final validation report
        report_step = PipelineStep(
            name="validation_report",
            step_type=StepType.CUSTOM,
            command=parameters.get("report_command", "python scripts/generate_report.py"),
            depends_on={performance_step.id, bias_step.id, security_step.id}
        )
        pipeline.add_step(report_step)
        
        return pipeline