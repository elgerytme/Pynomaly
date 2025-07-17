"""
Workflow orchestration service for managing end-to-end data science workflows.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

from data_platform.integration.domain.entities.workflow import Workflow, WorkflowStep, WorkflowStatus, WorkflowStepType
from data_platform.integration.domain.entities.integration_config import IntegrationConfig
from data_platform.integration.application.services.unified_api_service import UnifiedApiService
from software.interfaces.shared.error_handling import handle_exceptions


logger = logging.getLogger(__name__)


@dataclass
class WorkflowTemplate:
    """Template for creating workflows."""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    default_config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class WorkflowOrchestrationService:
    """Service for orchestrating end-to-end data science workflows."""
    
    def __init__(self, config: IntegrationConfig, unified_api: UnifiedApiService):
        """Initialize the workflow orchestration service."""
        self.config = config
        self.unified_api = unified_api
        self.active_workflows: Dict[str, Workflow] = {}
        self.workflow_templates: Dict[str, WorkflowTemplate] = {}
        self.scheduler_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize default workflow templates
        self._initialize_default_templates()
    
    def _initialize_default_templates(self) -> None:
        """Initialize default workflow templates."""
        # Complete data science workflow
        complete_workflow = WorkflowTemplate(
            name="complete_data_science_workflow",
            description="Complete data science workflow: profiling → quality → ML pipeline",
            steps=[
                {
                    "name": "data_profiling",
                    "type": WorkflowStepType.DATA_PROFILING,
                    "package": "data_profiling",
                    "operation": "profile_dataset",
                    "dependencies": [],
                    "config": {
                        "include_schema_analysis": True,
                        "include_statistical_profiling": True,
                        "include_pattern_discovery": True
                    }
                },
                {
                    "name": "quality_assessment",
                    "type": WorkflowStepType.DATA_QUALITY,
                    "package": "data_quality",
                    "operation": "assess_quality",
                    "dependencies": ["data_profiling"],
                    "config": {
                        "use_ml_detection": True,
                        "track_lineage": True
                    }
                },
                {
                    "name": "feature_engineering",
                    "type": WorkflowStepType.FEATURE_ENGINEERING,
                    "package": "data_science",
                    "operation": "engineer_features",
                    "dependencies": ["quality_assessment"],
                    "config": {
                        "auto_feature_selection": True,
                        "apply_scaling": True
                    }
                },
                {
                    "name": "model_training",
                    "type": WorkflowStepType.MODEL_TRAINING,
                    "package": "data_science",
                    "operation": "train_model",
                    "dependencies": ["feature_engineering"],
                    "config": {
                        "auto_hyperparameter_tuning": True,
                        "cross_validation": True
                    }
                },
                {
                    "name": "model_validation",
                    "type": WorkflowStepType.MODEL_VALIDATION,
                    "package": "data_science",
                    "operation": "validate_model",
                    "dependencies": ["model_training"],
                    "config": {
                        "validation_metrics": ["accuracy", "precision", "recall", "f1"],
                        "generate_report": True
                    }
                }
            ],
            default_config={
                "timeout_seconds": 3600,
                "retry_count": 3,
                "parallel_execution": True
            },
            tags=["complete", "ml", "production"]
        )
        
        # Real-time streaming workflow
        streaming_workflow = WorkflowTemplate(
            name="streaming_data_workflow",
            description="Real-time streaming data processing workflow",
            steps=[
                {
                    "name": "stream_profiling",
                    "type": WorkflowStepType.DATA_PROFILING,
                    "package": "data_profiling",
                    "operation": "profile_stream",
                    "dependencies": [],
                    "config": {
                        "streaming_mode": True,
                        "window_size": 1000,
                        "update_frequency": 60
                    }
                },
                {
                    "name": "real_time_quality",
                    "type": WorkflowStepType.DATA_QUALITY,
                    "package": "data_quality",
                    "operation": "monitor_quality",
                    "dependencies": ["stream_profiling"],
                    "config": {
                        "real_time_alerts": True,
                        "anomaly_detection": True
                    }
                },
                {
                    "name": "stream_monitoring",
                    "type": WorkflowStepType.MONITORING,
                    "package": "data_observability",
                    "operation": "monitor_stream",
                    "dependencies": ["real_time_quality"],
                    "config": {
                        "dashboard_updates": True,
                        "alert_thresholds": {"error_rate": 0.01, "latency": 1000}
                    }
                }
            ],
            default_config={
                "timeout_seconds": 300,
                "retry_count": 1,
                "parallel_execution": True
            },
            tags=["streaming", "realtime", "monitoring"]
        )
        
        # Quality-focused workflow
        quality_workflow = WorkflowTemplate(
            name="data_quality_workflow",
            description="Data quality assessment and improvement workflow",
            steps=[
                {
                    "name": "initial_profiling",
                    "type": WorkflowStepType.DATA_PROFILING,
                    "package": "data_profiling",
                    "operation": "profile_dataset",
                    "dependencies": [],
                    "config": {
                        "focus_on_quality": True,
                        "detect_anomalies": True
                    }
                },
                {
                    "name": "quality_rules_discovery",
                    "type": WorkflowStepType.DATA_QUALITY,
                    "package": "data_quality",
                    "operation": "discover_rules",
                    "dependencies": ["initial_profiling"],
                    "config": {
                        "use_ml_discovery": True,
                        "confidence_threshold": 0.8
                    }
                },
                {
                    "name": "quality_validation",
                    "type": WorkflowStepType.DATA_QUALITY,
                    "package": "data_quality",
                    "operation": "validate_quality",
                    "dependencies": ["quality_rules_discovery"],
                    "config": {
                        "generate_report": True,
                        "auto_fix_issues": True
                    }
                }
            ],
            default_config={
                "timeout_seconds": 1800,
                "retry_count": 2,
                "parallel_execution": False
            },
            tags=["quality", "validation", "governance"]
        )
        
        self.workflow_templates = {
            "complete_data_science_workflow": complete_workflow,
            "streaming_data_workflow": streaming_workflow,
            "data_quality_workflow": quality_workflow
        }
        
        logger.info(f"Initialized {len(self.workflow_templates)} workflow templates")
    
    def add_workflow_template(self, template: WorkflowTemplate) -> None:
        """Add a new workflow template."""
        self.workflow_templates[template.name] = template
        logger.info(f"Added workflow template: {template.name}")
    
    def get_workflow_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a workflow template by name."""
        return self.workflow_templates.get(name)
    
    def list_workflow_templates(self) -> List[str]:
        """List all available workflow templates."""
        return list(self.workflow_templates.keys())
    
    @handle_exceptions
    async def create_workflow_from_template(self, template_name: str, 
                                          workflow_name: str,
                                          config_overrides: Optional[Dict[str, Any]] = None) -> Workflow:
        """Create a workflow from a template."""
        template = self.get_workflow_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Create workflow
        workflow = Workflow(
            name=workflow_name,
            description=template.description,
            config=template.default_config.copy(),
            tags=template.tags.copy()
        )
        
        # Apply config overrides
        if config_overrides:
            workflow.config.update(config_overrides)
        
        # Create workflow steps
        for step_config in template.steps:
            step = WorkflowStep(
                id=step_config["name"],
                name=step_config["name"],
                step_type=step_config["type"],
                config=step_config.get("config", {}),
                dependencies=step_config.get("dependencies", []),
                timeout_seconds=step_config.get("timeout_seconds", 3600),
                retry_count=step_config.get("retry_count", 3)
            )
            
            workflow.add_step(step)
        
        return workflow
    
    @handle_exceptions
    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute a workflow."""
        workflow_id = str(workflow.id)
        self.active_workflows[workflow_id] = workflow
        
        logger.info(f"Starting workflow execution: {workflow.name} ({workflow_id})")
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()
        
        try:
            results = {}
            
            # Execute steps in dependency order
            while True:
                ready_steps = workflow.get_ready_steps()
                
                if not ready_steps:
                    # Check if we're done
                    if workflow.is_completed():
                        workflow.status = WorkflowStatus.COMPLETED
                        workflow.completed_at = datetime.utcnow()
                        break
                    elif workflow.has_failed():
                        workflow.status = WorkflowStatus.FAILED
                        break
                    else:
                        # Wait for running steps to complete
                        await asyncio.sleep(1)
                        continue
                
                # Execute ready steps
                tasks = []
                for step in ready_steps:
                    task = asyncio.create_task(
                        self._execute_workflow_step(workflow, step, results)
                    )
                    tasks.append(task)
                
                # Wait for all tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)
            
            execution_results = {
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "status": workflow.status.value,
                "execution_time": workflow.get_execution_time(),
                "step_results": results,
                "metrics": workflow.get_step_metrics()
            }
            
            logger.info(f"Workflow execution completed: {workflow.name} ({workflow_id})")
            return execution_results
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            logger.error(f"Workflow execution failed: {workflow.name} ({workflow_id}): {str(e)}")
            raise
        finally:
            # Clean up
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
    
    async def _execute_workflow_step(self, workflow: Workflow, step: WorkflowStep, 
                                   results: Dict[str, Any]) -> Any:
        """Execute a single workflow step."""
        logger.info(f"Executing step: {step.name} in workflow: {workflow.name}")
        
        workflow.update_step_status(step.id, WorkflowStatus.RUNNING)
        
        try:
            # Prepare step parameters
            step_params = step.config.copy()
            
            # Add results from dependent steps
            for dep_id in step.dependencies:
                if dep_id in results:
                    step_params[f"{dep_id}_result"] = results[dep_id]
            
            # Execute step based on type
            if step.step_type == WorkflowStepType.DATA_PROFILING:
                result = await self._execute_profiling_step(step, step_params)
            elif step.step_type == WorkflowStepType.DATA_QUALITY:
                result = await self._execute_quality_step(step, step_params)
            elif step.step_type == WorkflowStepType.FEATURE_ENGINEERING:
                result = await self._execute_feature_engineering_step(step, step_params)
            elif step.step_type == WorkflowStepType.MODEL_TRAINING:
                result = await self._execute_model_training_step(step, step_params)
            elif step.step_type == WorkflowStepType.MODEL_VALIDATION:
                result = await self._execute_model_validation_step(step, step_params)
            elif step.step_type == WorkflowStepType.MONITORING:
                result = await self._execute_monitoring_step(step, step_params)
            else:
                raise ValueError(f"Unknown step type: {step.step_type}")
            
            # Store result
            results[step.id] = result
            
            # Update step metrics
            step.metrics = {
                "execution_time": (datetime.utcnow() - step.start_time).total_seconds(),
                "success": True,
                "result_size": len(str(result)) if result else 0
            }
            
            workflow.update_step_status(step.id, WorkflowStatus.COMPLETED)
            
            logger.info(f"Step completed successfully: {step.name}")
            return result
            
        except Exception as e:
            logger.error(f"Step failed: {step.name}: {str(e)}")
            workflow.update_step_status(step.id, WorkflowStatus.FAILED, str(e))
            raise
    
    async def _execute_profiling_step(self, step: WorkflowStep, params: Dict[str, Any]) -> Any:
        """Execute a data profiling step."""
        # Determine which profiling operation to use
        if step.config.get("include_schema_analysis", True):
            return await self.unified_api.execute_operation(
                "data_profiling", "analyze_schema", **params
            )
        elif step.config.get("include_statistical_profiling", True):
            return await self.unified_api.execute_operation(
                "data_profiling", "profile_dataset", **params
            )
        else:
            return await self.unified_api.execute_operation(
                "data_profiling", "discover_patterns", **params
            )
    
    async def _execute_quality_step(self, step: WorkflowStep, params: Dict[str, Any]) -> Any:
        """Execute a data quality step."""
        if step.config.get("use_ml_detection", True):
            return await self.unified_api.execute_operation(
                "data_quality", "detect_quality_issues", **params
            )
        else:
            return await self.unified_api.execute_operation(
                "data_quality", "discover_rules", **params
            )
    
    async def _execute_feature_engineering_step(self, step: WorkflowStep, params: Dict[str, Any]) -> Any:
        """Execute a feature engineering step."""
        return await self.unified_api.execute_operation(
            "data_science", "engineer_features", **params
        )
    
    async def _execute_model_training_step(self, step: WorkflowStep, params: Dict[str, Any]) -> Any:
        """Execute a model training step."""
        return await self.unified_api.execute_operation(
            "data_science", "train_model", **params
        )
    
    async def _execute_model_validation_step(self, step: WorkflowStep, params: Dict[str, Any]) -> Any:
        """Execute a model validation step."""
        return await self.unified_api.execute_operation(
            "data_science", "validate_model", **params
        )
    
    async def _execute_monitoring_step(self, step: WorkflowStep, params: Dict[str, Any]) -> Any:
        """Execute a monitoring step."""
        return await self.unified_api.execute_operation(
            "data_observability", "monitor_stream", **params
        )
    
    @handle_exceptions
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a workflow."""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at,
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at,
            "execution_time": workflow.get_execution_time(),
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "type": step.step_type.value,
                    "status": step.status.value,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "error_message": step.error_message,
                    "metrics": step.metrics
                }
                for step in workflow.steps
            ],
            "metrics": workflow.get_step_metrics()
        }
    
    @handle_exceptions
    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel a running workflow."""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow.status = WorkflowStatus.CANCELLED
        
        # Cancel running steps
        for step in workflow.steps:
            if step.status == WorkflowStatus.RUNNING:
                workflow.update_step_status(step.id, WorkflowStatus.CANCELLED)
        
        logger.info(f"Workflow cancelled: {workflow.name} ({workflow_id})")
    
    @handle_exceptions
    async def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows."""
        return [
            {
                "workflow_id": workflow_id,
                "name": workflow.name,
                "status": workflow.status.value,
                "created_at": workflow.created_at,
                "started_at": workflow.started_at
            }
            for workflow_id, workflow in self.active_workflows.items()
        ]