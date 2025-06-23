"""Coordination service for managing distributed anomaly detection workflows."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

from .task_queue import TaskQueue, Task, TaskStatus, TaskPriority
from .worker_pool import WorkerPool
from .manager import DistributedProcessingManager
from .coordinator import DetectionCoordinator
from pynomaly.domain.entities import Detector, Dataset, DetectionResult
from pynomaly.domain.exceptions import ProcessingError


logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class WorkflowStep:
    """Represents a step in a distributed workflow."""
    id: str
    type: str
    name: str
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    task_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def can_execute(self, completed_steps: Set[str]) -> bool:
        """Check if step can be executed based on dependencies."""
        return all(dep in completed_steps for dep in self.dependencies)


@dataclass
class Workflow:
    """Represents a distributed workflow."""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: int = TaskPriority.NORMAL.value
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get workflow execution duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def progress(self) -> float:
        """Get workflow completion progress (0.0 to 1.0)."""
        if not self.steps:
            return 0.0
        
        completed = len([s for s in self.steps if s.status == TaskStatus.COMPLETED])
        return completed / len(self.steps)
    
    def get_ready_steps(self) -> List[WorkflowStep]:
        """Get steps that are ready to execute."""
        completed_steps = {s.id for s in self.steps if s.status == TaskStatus.COMPLETED}
        
        return [
            step for step in self.steps
            if step.status == TaskStatus.PENDING and step.can_execute(completed_steps)
        ]


class CoordinationService:
    """Service for coordinating complex distributed workflows."""
    
    def __init__(self,
                 task_queue: TaskQueue,
                 worker_pool: WorkerPool,
                 processing_manager: DistributedProcessingManager,
                 detection_coordinator: DetectionCoordinator):
        self.task_queue = task_queue
        self.worker_pool = worker_pool
        self.processing_manager = processing_manager
        self.detection_coordinator = detection_coordinator
        
        # Workflow management
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Set[str] = set()
        self.completed_workflows: Dict[str, Workflow] = {}
        
        # Workflow templates
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._running = False
        self._workflow_executor: Optional[asyncio.Task] = None
        self._workflow_monitor: Optional[asyncio.Task] = None
        
        # Event handlers
        self._workflow_callbacks: List[Callable] = []
        
        # Built-in workflow templates
        self._initialize_templates()
        
        logger.info("Coordination service initialized")
    
    async def start(self) -> None:
        """Start the coordination service."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._workflow_executor = asyncio.create_task(self._execute_workflows())
        self._workflow_monitor = asyncio.create_task(self._monitor_workflows())
        
        logger.info("Coordination service started")
    
    async def stop(self) -> None:
        """Stop the coordination service."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        for task in [self._workflow_executor, self._workflow_monitor]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Complete or cancel active workflows
        for workflow_id in self.active_workflows.copy():
            await self.cancel_workflow(workflow_id)
        
        logger.info("Coordination service stopped")
    
    async def create_workflow(self,
                            name: str,
                            description: str,
                            steps: List[Dict[str, Any]],
                            priority: int = TaskPriority.NORMAL.value,
                            metadata: Dict[str, Any] = None) -> str:
        """Create a new workflow."""
        workflow_id = str(uuid.uuid4())
        
        # Convert step dictionaries to WorkflowStep objects
        workflow_steps = []
        for i, step_config in enumerate(steps):
            step = WorkflowStep(
                id=step_config.get('id', f"step_{i}"),
                type=step_config['type'],
                name=step_config['name'],
                dependencies=step_config.get('dependencies', []),
                parameters=step_config.get('parameters', {}),
                timeout=step_config.get('timeout'),
                max_retries=step_config.get('max_retries', 3)
            )
            workflow_steps.append(step)
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            steps=workflow_steps,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow {workflow_id}: {name}")
        return workflow_id
    
    async def create_workflow_from_template(self,
                                          template_name: str,
                                          parameters: Dict[str, Any],
                                          priority: int = TaskPriority.NORMAL.value) -> str:
        """Create a workflow from a predefined template."""
        if template_name not in self.workflow_templates:
            raise ProcessingError(f"Workflow template '{template_name}' not found")
        
        template = self.workflow_templates[template_name]
        
        # Substitute parameters in template
        workflow_config = await self._substitute_template_parameters(template, parameters)
        
        return await self.create_workflow(
            name=workflow_config['name'],
            description=workflow_config['description'],
            steps=workflow_config['steps'],
            priority=priority,
            metadata={'template': template_name, 'parameters': parameters}
        )
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """Start workflow execution."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status != WorkflowStatus.PENDING:
            return False
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        self.active_workflows.add(workflow_id)
        
        await self._notify_workflow_event("workflow_started", workflow)
        
        logger.info(f"Started workflow {workflow_id}")
        return True
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel workflow execution."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        
        # Cancel running tasks
        for step in workflow.steps:
            if step.task_id and step.status in [TaskStatus.QUEUED, TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
                await self.task_queue.cancel_task(step.task_id)
        
        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.now()
        
        # Move to completed workflows
        if workflow_id in self.active_workflows:
            self.active_workflows.remove(workflow_id)
        self.completed_workflows[workflow_id] = workflow
        
        await self._notify_workflow_event("workflow_cancelled", workflow)
        
        logger.info(f"Cancelled workflow {workflow_id}")
        return True
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause workflow execution."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status != WorkflowStatus.RUNNING:
            return False
        
        workflow.status = WorkflowStatus.PAUSED
        
        # Pause pending tasks (running tasks will continue)
        for step in workflow.steps:
            if step.task_id and step.status == TaskStatus.QUEUED:
                await self.task_queue.cancel_task(step.task_id)
                step.status = TaskStatus.PENDING
                step.task_id = None
        
        await self._notify_workflow_event("workflow_paused", workflow)
        
        logger.info(f"Paused workflow {workflow_id}")
        return True
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume workflow execution."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status != WorkflowStatus.PAUSED:
            return False
        
        workflow.status = WorkflowStatus.RUNNING
        
        await self._notify_workflow_event("workflow_resumed", workflow)
        
        logger.info(f"Resumed workflow {workflow_id}")
        return True
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed workflow status."""
        # Check active workflows
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
        # Check completed workflows
        elif workflow_id in self.completed_workflows:
            workflow = self.completed_workflows[workflow_id]
        else:
            return None
        
        return {
            'id': workflow.id,
            'name': workflow.name,
            'description': workflow.description,
            'status': workflow.status.value,
            'progress': workflow.progress,
            'created_at': workflow.created_at.isoformat(),
            'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
            'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
            'duration': workflow.duration,
            'priority': workflow.priority,
            'metadata': workflow.metadata,
            'steps': [
                {
                    'id': step.id,
                    'name': step.name,
                    'type': step.type,
                    'status': step.status.value,
                    'dependencies': step.dependencies,
                    'task_id': step.task_id,
                    'retry_count': step.retry_count,
                    'error': step.error
                }
                for step in workflow.steps
            ]
        }
    
    async def list_workflows(self,
                           status: Optional[WorkflowStatus] = None,
                           limit: int = 100,
                           offset: int = 0) -> List[Dict[str, Any]]:
        """List workflows with optional filtering."""
        all_workflows = list(self.workflows.values()) + list(self.completed_workflows.values())
        
        # Apply status filter
        if status:
            all_workflows = [w for w in all_workflows if w.status == status]
        
        # Sort by creation time (newest first)
        all_workflows.sort(key=lambda w: w.created_at, reverse=True)
        
        # Apply pagination
        workflows = all_workflows[offset:offset + limit]
        
        return [
            {
                'id': w.id,
                'name': w.name,
                'status': w.status.value,
                'progress': w.progress,
                'created_at': w.created_at.isoformat(),
                'duration': w.duration
            }
            for w in workflows
        ]
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        return {
            'workflows': {
                'active': len(self.active_workflows),
                'completed': len(self.completed_workflows),
                'total': len(self.workflows) + len(self.completed_workflows)
            },
            'task_queue': await self.task_queue.get_queue_stats(),
            'worker_pool': await self.worker_pool.get_worker_stats(),
            'processing_manager': await self.processing_manager.get_system_metrics(),
            'detection_coordinator': await self.detection_coordinator.get_system_metrics()
        }
    
    def register_workflow_template(self, name: str, template: Dict[str, Any]) -> None:
        """Register a workflow template."""
        self.workflow_templates[name] = template
        logger.info(f"Registered workflow template: {name}")
    
    def add_workflow_callback(self, callback: Callable) -> None:
        """Add a callback for workflow events."""
        self._workflow_callbacks.append(callback)
    
    async def _execute_workflows(self) -> None:
        """Main workflow execution loop."""
        while self._running:
            try:
                # Process each active workflow
                for workflow_id in self.active_workflows.copy():
                    workflow = self.workflows.get(workflow_id)
                    if not workflow or workflow.status != WorkflowStatus.RUNNING:
                        continue
                    
                    await self._process_workflow(workflow)
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Workflow execution error: {e}")
                await asyncio.sleep(1)
    
    async def _process_workflow(self, workflow: Workflow) -> None:
        """Process a single workflow."""
        # Get ready steps
        ready_steps = workflow.get_ready_steps()
        
        # Submit ready steps as tasks
        for step in ready_steps:
            try:
                task_id = await self._submit_workflow_step(workflow, step)
                step.task_id = task_id
                step.status = TaskStatus.QUEUED
                
            except Exception as e:
                logger.error(f"Failed to submit step {step.id}: {e}")
                step.status = TaskStatus.FAILED
                step.error = str(e)
        
        # Check if workflow is complete
        if all(s.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] for s in workflow.steps):
            await self._complete_workflow(workflow)
    
    async def _submit_workflow_step(self, workflow: Workflow, step: WorkflowStep) -> str:
        """Submit a workflow step as a task."""
        # Prepare task payload
        payload = {
            'workflow_id': workflow.id,
            'step_id': step.id,
            'step_type': step.type,
            'parameters': step.parameters
        }
        
        # Determine required capabilities based on step type
        required_capabilities = []
        if step.type in ['pyod_detection', 'sklearn_detection']:
            required_capabilities = ['sklearn', 'pyod']
        elif step.type == 'pytorch_detection':
            required_capabilities = ['pytorch']
        elif step.type == 'tensorflow_detection':
            required_capabilities = ['tensorflow']
        
        # Submit task
        task_id = await self.task_queue.submit_task(
            task_type=f"workflow_step_{step.type}",
            payload=payload,
            priority=workflow.priority,
            timeout=step.timeout,
            max_retries=step.max_retries,
            metadata={
                'workflow_id': workflow.id,
                'step_id': step.id,
                'required_capabilities': required_capabilities
            }
        )
        
        return task_id
    
    async def _monitor_workflows(self) -> None:
        """Monitor workflow execution and update step statuses."""
        while self._running:
            try:
                # Check task statuses for all active workflows
                for workflow_id in self.active_workflows.copy():
                    workflow = self.workflows.get(workflow_id)
                    if not workflow:
                        continue
                    
                    await self._update_workflow_progress(workflow)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Workflow monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _update_workflow_progress(self, workflow: Workflow) -> None:
        """Update workflow progress based on task statuses."""
        for step in workflow.steps:
            if not step.task_id or step.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                continue
            
            # Check task status
            task = await self.task_queue.get_task(step.task_id)
            if not task:
                continue
            
            # Update step status based on task status
            if task.status == TaskStatus.COMPLETED:
                step.status = TaskStatus.COMPLETED
                step.result = task.result
                
            elif task.status == TaskStatus.FAILED:
                step.status = TaskStatus.FAILED
                step.error = task.error
                step.retry_count = task.retry_count
    
    async def _complete_workflow(self, workflow: Workflow) -> None:
        """Mark workflow as completed."""
        # Determine final status
        failed_steps = [s for s in workflow.steps if s.status == TaskStatus.FAILED]
        
        if failed_steps:
            workflow.status = WorkflowStatus.FAILED
        else:
            workflow.status = WorkflowStatus.COMPLETED
        
        workflow.completed_at = datetime.now()
        
        # Move to completed workflows
        self.active_workflows.remove(workflow.id)
        self.completed_workflows[workflow.id] = workflow
        del self.workflows[workflow.id]
        
        await self._notify_workflow_event("workflow_completed", workflow)
        
        logger.info(f"Workflow {workflow.id} completed with status {workflow.status.value}")
    
    async def _substitute_template_parameters(self,
                                            template: Dict[str, Any],
                                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute parameters in a workflow template."""
        template_str = json.dumps(template)
        
        # Simple parameter substitution
        for key, value in parameters.items():
            placeholder = f"${{{key}}}"
            template_str = template_str.replace(placeholder, str(value))
        
        return json.loads(template_str)
    
    async def _notify_workflow_event(self, event_type: str, workflow: Workflow) -> None:
        """Notify callbacks of workflow events."""
        for callback in self._workflow_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, workflow)
                else:
                    callback(event_type, workflow)
            except Exception as e:
                logger.error(f"Workflow event callback failed: {e}")
    
    def _initialize_templates(self) -> None:
        """Initialize built-in workflow templates."""
        # Basic anomaly detection workflow
        self.workflow_templates['basic_detection'] = {
            'name': 'Basic Anomaly Detection',
            'description': 'Single detector anomaly detection workflow',
            'steps': [
                {
                    'id': 'load_data',
                    'type': 'data_loading',
                    'name': 'Load Dataset',
                    'parameters': {
                        'dataset_id': '${dataset_id}'
                    }
                },
                {
                    'id': 'detect_anomalies',
                    'type': 'anomaly_detection',
                    'name': 'Detect Anomalies',
                    'dependencies': ['load_data'],
                    'parameters': {
                        'detector_id': '${detector_id}',
                        'algorithm': '${algorithm}'
                    }
                },
                {
                    'id': 'save_results',
                    'type': 'result_storage',
                    'name': 'Save Results',
                    'dependencies': ['detect_anomalies'],
                    'parameters': {
                        'output_format': '${output_format}'
                    }
                }
            ]
        }
        
        # Ensemble detection workflow
        self.workflow_templates['ensemble_detection'] = {
            'name': 'Ensemble Anomaly Detection',
            'description': 'Multi-detector ensemble workflow',
            'steps': [
                {
                    'id': 'load_data',
                    'type': 'data_loading',
                    'name': 'Load Dataset',
                    'parameters': {
                        'dataset_id': '${dataset_id}'
                    }
                },
                {
                    'id': 'detect_pyod',
                    'type': 'pyod_detection',
                    'name': 'PyOD Detection',
                    'dependencies': ['load_data'],
                    'parameters': {
                        'algorithm': '${pyod_algorithm}'
                    }
                },
                {
                    'id': 'detect_sklearn',
                    'type': 'sklearn_detection',
                    'name': 'Sklearn Detection',
                    'dependencies': ['load_data'],
                    'parameters': {
                        'algorithm': '${sklearn_algorithm}'
                    }
                },
                {
                    'id': 'ensemble_results',
                    'type': 'ensemble_aggregation',
                    'name': 'Combine Results',
                    'dependencies': ['detect_pyod', 'detect_sklearn'],
                    'parameters': {
                        'voting_strategy': '${voting_strategy}'
                    }
                },
                {
                    'id': 'save_results',
                    'type': 'result_storage',
                    'name': 'Save Results',
                    'dependencies': ['ensemble_results'],
                    'parameters': {
                        'output_format': '${output_format}'
                    }
                }
            ]
        }
        
        logger.info("Initialized built-in workflow templates")