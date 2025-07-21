"""
Data Pipeline Orchestrator for Pynomaly Detection
==================================================

Comprehensive pipeline orchestration and management providing:
- Multi-stage data processing pipelines
- Dependency management and scheduling
- Data validation and quality checks
- Pipeline monitoring and error handling
- Integration with external systems
"""

import logging
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import numpy as np
import pandas as pd

try:
    import sqlalchemy
    from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, JSON, Float, Integer
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import celery
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

from ...simplified_services.core_detection_service import CoreDetectionService

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

@dataclass
class TaskConfig:
    """Configuration for pipeline task."""
    task_id: str
    task_type: str  # data_ingestion, preprocessing, detection, validation, output
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    retry_delay: float = 60.0
    timeout: float = 3600.0
    resources: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PipelineConfig:
    """Configuration for data pipeline."""
    pipeline_id: str
    pipeline_name: str
    description: str = ""
    schedule_interval: Optional[str] = None  # cron expression
    max_active_runs: int = 1
    enable_catchup: bool = False
    enable_monitoring: bool = True
    enable_data_validation: bool = True
    retry_delay: float = 300.0
    email_on_failure: bool = False
    email_on_retry: bool = False
    tags: List[str] = field(default_factory=list)

@dataclass
class TaskExecution:
    """Task execution record."""
    task_id: str
    execution_id: str
    pipeline_id: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    output_data: Optional[Any] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineExecution:
    """Pipeline execution record."""
    execution_id: str
    pipeline_id: str
    status: PipelineStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    task_executions: Dict[str, TaskExecution] = field(default_factory=dict)
    trigger_type: str = "manual"
    trigger_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class DataPipelineOrchestrator:
    """Comprehensive data pipeline orchestration system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config or {}
        self.core_service = CoreDetectionService()
        
        # Pipeline management
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.pipeline_tasks: Dict[str, List[TaskConfig]] = {}
        self.pipeline_graphs: Dict[str, Dict[str, List[str]]] = {}
        
        # Execution tracking
        self.active_executions: Dict[str, PipelineExecution] = {}
        self.execution_history: deque = deque(maxlen=10000)
        
        # Scheduling and workers
        self.scheduler_thread = None
        self.worker_pool: Dict[str, threading.Thread] = {}
        self.task_queue: deque = deque()
        
        # State management
        self.is_running = False
        self.lock = threading.RLock()
        
        # Data storage
        self.database_engine = None
        self.redis_client = None
        self.celery_app = None
        
        # Callbacks and hooks
        self.pipeline_callbacks: Dict[str, List[Callable]] = {
            'on_start': [],
            'on_success': [],
            'on_failure': [],
            'on_retry': []
        }
        self.task_callbacks: Dict[str, List[Callable]] = {
            'on_start': [],
            'on_success': [],
            'on_failure': [],
            'on_retry': []
        }
        
        # Monitoring and metrics
        self.metrics = {
            'pipelines_registered': 0,
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'active_pipelines': 0,
            'avg_execution_time': 0,
            'queue_size': 0
        }
        
        # Initialize storage backends
        self._initialize_storage()
        
        logger.info("Data Pipeline Orchestrator initialized")
    
    def register_pipeline(self, pipeline_config: PipelineConfig, tasks: List[TaskConfig]) -> bool:
        """Register a new data pipeline.
        
        Args:
            pipeline_config: Pipeline configuration
            tasks: List of pipeline tasks
            
        Returns:
            Registration success status
        """
        try:
            pipeline_id = pipeline_config.pipeline_id
            
            # Validate pipeline configuration
            if not self._validate_pipeline_config(pipeline_config, tasks):
                return False
            
            # Build dependency graph
            task_graph = self._build_task_graph(tasks)
            if not self._validate_task_graph(task_graph):
                logger.error(f"Invalid task graph for pipeline {pipeline_id}")
                return False
            
            # Store pipeline
            self.pipelines[pipeline_id] = pipeline_config
            self.pipeline_tasks[pipeline_id] = tasks
            self.pipeline_graphs[pipeline_id] = task_graph
            
            # Update metrics
            self.metrics['pipelines_registered'] += 1
            
            logger.info(f"Pipeline registered: {pipeline_id} with {len(tasks)} tasks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register pipeline {pipeline_config.pipeline_id}: {e}")
            return False
    
    def unregister_pipeline(self, pipeline_id: str) -> bool:
        """Unregister a data pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            Unregistration success status
        """
        try:
            if pipeline_id not in self.pipelines:
                logger.warning(f"Pipeline not found: {pipeline_id}")
                return False
            
            # Cancel active executions
            active_executions = [
                exec_id for exec_id, execution in self.active_executions.items()
                if execution.pipeline_id == pipeline_id
            ]
            
            for exec_id in active_executions:
                self.cancel_execution(exec_id)
            
            # Remove pipeline
            del self.pipelines[pipeline_id]
            del self.pipeline_tasks[pipeline_id]
            del self.pipeline_graphs[pipeline_id]
            
            logger.info(f"Pipeline unregistered: {pipeline_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister pipeline {pipeline_id}: {e}")
            return False
    
    def start_orchestrator(self):
        """Start the pipeline orchestrator."""
        if self.is_running:
            logger.warning("Pipeline orchestrator already running")
            return
        
        try:
            self.is_running = True
            
            # Start scheduler thread
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            
            # Start worker threads
            num_workers = self.config.get('num_workers', 4)
            for i in range(num_workers):
                worker_thread = threading.Thread(target=self._worker_loop, args=(f"worker-{i}",), daemon=True)
                worker_thread.start()
                self.worker_pool[f"worker-{i}"] = worker_thread
            
            logger.info(f"Pipeline orchestrator started with {num_workers} workers")
            
        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            self.is_running = False
            raise
    
    def stop_orchestrator(self):
        """Stop the pipeline orchestrator."""
        self.is_running = False
        
        # Cancel all active executions
        for exec_id in list(self.active_executions.keys()):
            self.cancel_execution(exec_id)
        
        # Wait for threads
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        for worker_thread in self.worker_pool.values():
            if worker_thread.is_alive():
                worker_thread.join(timeout=5.0)
        
        logger.info("Pipeline orchestrator stopped")
    
    def execute_pipeline(self, pipeline_id: str, trigger_data: Optional[Dict[str, Any]] = None) -> str:
        """Execute a pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            trigger_data: Optional trigger data
            
        Returns:
            Execution ID
        """
        try:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline not found: {pipeline_id}")
            
            pipeline_config = self.pipelines[pipeline_id]
            
            # Check max active runs
            active_count = sum(
                1 for execution in self.active_executions.values()
                if execution.pipeline_id == pipeline_id and execution.status == PipelineStatus.RUNNING
            )
            
            if active_count >= pipeline_config.max_active_runs:
                raise RuntimeError(f"Max active runs ({pipeline_config.max_active_runs}) exceeded for pipeline {pipeline_id}")
            
            # Create execution record
            execution_id = f"{pipeline_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            execution = PipelineExecution(
                execution_id=execution_id,
                pipeline_id=pipeline_id,
                status=PipelineStatus.PENDING,
                trigger_type="manual",
                trigger_data=trigger_data
            )
            
            # Store execution
            self.active_executions[execution_id] = execution
            
            # Queue pipeline for execution
            self.task_queue.append({
                'type': 'pipeline_execution',
                'execution_id': execution_id
            })
            
            logger.info(f"Pipeline queued for execution: {pipeline_id} ({execution_id})")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to execute pipeline {pipeline_id}: {e}")
            raise
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a pipeline execution.
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            Cancellation success status
        """
        try:
            if execution_id not in self.active_executions:
                logger.warning(f"Execution not found: {execution_id}")
                return False
            
            execution = self.active_executions[execution_id]
            execution.status = PipelineStatus.CANCELLED
            execution.end_time = datetime.now()
            
            # Calculate duration
            if execution.start_time:
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution_id]
            
            logger.info(f"Execution cancelled: {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel execution {execution_id}: {e}")
            return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status.
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            Execution status information
        """
        # Check active executions
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            return self._serialize_execution(execution)
        
        # Check history
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return self._serialize_execution(execution)
        
        return None
    
    def get_pipeline_executions(self, pipeline_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history for a pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            limit: Maximum number of executions to return
            
        Returns:
            List of execution records
        """
        executions = []
        
        # Add active executions
        for execution in self.active_executions.values():
            if execution.pipeline_id == pipeline_id:
                executions.append(self._serialize_execution(execution))
        
        # Add historical executions
        for execution in self.execution_history:
            if execution.pipeline_id == pipeline_id:
                executions.append(self._serialize_execution(execution))
        
        # Sort by start time (most recent first)
        executions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return executions[:limit]
    
    def add_pipeline_callback(self, event: str, callback: Callable[[PipelineExecution], None]):
        """Add pipeline event callback.
        
        Args:
            event: Event type (on_start, on_success, on_failure, on_retry)
            callback: Callback function
        """
        if event in self.pipeline_callbacks:
            self.pipeline_callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown pipeline event: {event}")
    
    def add_task_callback(self, event: str, callback: Callable[[TaskExecution], None]):
        """Add task event callback.
        
        Args:
            event: Event type (on_start, on_success, on_failure, on_retry)
            callback: Callback function
        """
        if event in self.task_callbacks:
            self.task_callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown task event: {event}")
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        with self.lock:
            metrics = self.metrics.copy()
            metrics.update({
                'active_executions': len(self.active_executions),
                'queue_size': len(self.task_queue),
                'worker_count': len(self.worker_pool),
                'is_running': self.is_running
            })
            return metrics
    
    def _initialize_storage(self):
        """Initialize storage backends."""
        try:
            # Initialize database if configured
            database_url = self.config.get('database_url')
            if database_url and SQLALCHEMY_AVAILABLE:
                self.database_engine = create_engine(database_url)
                self._create_database_tables()
            
            # Initialize Redis if configured
            redis_url = self.config.get('redis_url')
            if redis_url and REDIS_AVAILABLE:
                self.redis_client = redis.from_url(redis_url)
            
            # Initialize Celery if configured
            celery_broker = self.config.get('celery_broker')
            if celery_broker and CELERY_AVAILABLE:
                self.celery_app = Celery('pynomaly_orchestrator', broker=celery_broker)
            
        except Exception as e:
            logger.warning(f"Storage initialization failed: {e}")
    
    def _create_database_tables(self):
        """Create database tables for execution tracking."""
        try:
            metadata = MetaData()
            
            # Pipeline executions table
            executions_table = Table(
                'pipeline_executions',
                metadata,
                Column('execution_id', String(255), primary_key=True),
                Column('pipeline_id', String(255)),
                Column('status', String(50)),
                Column('start_time', DateTime),
                Column('end_time', DateTime),
                Column('duration', Float),
                Column('trigger_type', String(50)),
                Column('trigger_data', JSON),
                Column('error_message', String(1000))
            )
            
            # Task executions table
            task_executions_table = Table(
                'task_executions',
                metadata,
                Column('task_id', String(255)),
                Column('execution_id', String(255)),
                Column('pipeline_id', String(255)),
                Column('status', String(50)),
                Column('start_time', DateTime),
                Column('end_time', DateTime),
                Column('duration', Float),
                Column('retry_count', Integer),
                Column('error_message', String(1000)),
                Column('metrics', JSON)
            )
            
            metadata.create_all(self.database_engine)
            
        except Exception as e:
            logger.error(f"Database table creation failed: {e}")
    
    def _validate_pipeline_config(self, config: PipelineConfig, tasks: List[TaskConfig]) -> bool:
        """Validate pipeline configuration."""
        if not config.pipeline_id or not config.pipeline_name:
            logger.error("Pipeline ID and name are required")
            return False
        
        if not tasks:
            logger.error("Pipeline must have at least one task")
            return False
        
        # Check for duplicate task IDs
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            logger.error("Duplicate task IDs found")
            return False
        
        return True
    
    def _build_task_graph(self, tasks: List[TaskConfig]) -> Dict[str, List[str]]:
        """Build task dependency graph.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Dependency graph
        """
        graph = {}
        task_ids = {task.task_id for task in tasks}
        
        for task in tasks:
            # Validate dependencies exist
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise ValueError(f"Unknown dependency '{dep}' for task '{task.task_id}'")
            
            graph[task.task_id] = task.dependencies.copy()
        
        return graph
    
    def _validate_task_graph(self, graph: Dict[str, List[str]]) -> bool:
        """Validate task dependency graph for cycles."""
        def has_cycle():
            # DFS cycle detection
            WHITE, GRAY, BLACK = 0, 1, 2
            color = {node: WHITE for node in graph}
            
            def dfs(node):
                if color[node] == GRAY:
                    return True  # Back edge found, cycle detected
                if color[node] == BLACK:
                    return False
                
                color[node] = GRAY
                for neighbor in graph[node]:
                    if dfs(neighbor):
                        return True
                color[node] = BLACK
                return False
            
            for node in graph:
                if color[node] == WHITE:
                    if dfs(node):
                        return True
            return False
        
        if has_cycle():
            logger.error("Cyclic dependency detected in task graph")
            return False
        
        return True
    
    def _scheduler_loop(self):
        """Scheduler loop for pipeline execution."""
        while self.is_running:
            try:
                # Process scheduled pipelines (cron-based scheduling would go here)
                
                # Process queued tasks
                if self.task_queue:
                    with self.lock:
                        if self.task_queue:
                            task = self.task_queue.popleft()
                            self._process_scheduler_task(task)
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5.0)
    
    def _worker_loop(self, worker_id: str):
        """Worker loop for task execution."""
        while self.is_running:
            try:
                # Check for available tasks to execute
                task_to_execute = self._get_next_executable_task()
                
                if task_to_execute:
                    self._execute_task(task_to_execute)
                else:
                    time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                time.sleep(1.0)
    
    def _process_scheduler_task(self, task: Dict[str, Any]):
        """Process scheduler task."""
        if task['type'] == 'pipeline_execution':
            execution_id = task['execution_id']
            self._start_pipeline_execution(execution_id)
    
    def _start_pipeline_execution(self, execution_id: str):
        """Start pipeline execution."""
        try:
            execution = self.active_executions[execution_id]
            execution.status = PipelineStatus.RUNNING
            execution.start_time = datetime.now()
            
            # Initialize task executions
            pipeline_id = execution.pipeline_id
            tasks = self.pipeline_tasks[pipeline_id]
            
            for task in tasks:
                task_execution = TaskExecution(
                    task_id=task.task_id,
                    execution_id=execution_id,
                    pipeline_id=pipeline_id,
                    status=TaskStatus.PENDING
                )
                execution.task_executions[task.task_id] = task_execution
            
            # Trigger pipeline start callbacks
            for callback in self.pipeline_callbacks['on_start']:
                try:
                    callback(execution)
                except Exception as e:
                    logger.error(f"Pipeline start callback failed: {e}")
            
            logger.info(f"Pipeline execution started: {execution_id}")
            
        except Exception as e:
            logger.error(f"Failed to start pipeline execution {execution_id}: {e}")
    
    def _get_next_executable_task(self) -> Optional[Dict[str, Any]]:
        """Get next executable task."""
        with self.lock:
            for execution in self.active_executions.values():
                if execution.status != PipelineStatus.RUNNING:
                    continue
                
                pipeline_id = execution.pipeline_id
                task_graph = self.pipeline_graphs[pipeline_id]
                
                # Find tasks ready for execution
                for task_id, dependencies in task_graph.items():
                    task_execution = execution.task_executions[task_id]
                    
                    if task_execution.status != TaskStatus.PENDING:
                        continue
                    
                    # Check if all dependencies are completed
                    dependencies_met = all(
                        execution.task_executions[dep_id].status == TaskStatus.SUCCESS
                        for dep_id in dependencies
                    )
                    
                    if dependencies_met:
                        task_execution.status = TaskStatus.RUNNING
                        task_execution.start_time = datetime.now()
                        
                        return {
                            'execution_id': execution.execution_id,
                            'task_id': task_id,
                            'task_execution': task_execution
                        }
            
            return None
    
    def _execute_task(self, task_info: Dict[str, Any]):
        """Execute a pipeline task."""
        execution_id = task_info['execution_id']
        task_id = task_info['task_id']
        task_execution = task_info['task_execution']
        
        try:
            # Get task configuration
            execution = self.active_executions[execution_id]
            pipeline_id = execution.pipeline_id
            task_config = next(
                (task for task in self.pipeline_tasks[pipeline_id] if task.task_id == task_id),
                None
            )
            
            if not task_config:
                raise ValueError(f"Task configuration not found: {task_id}")
            
            # Trigger task start callbacks
            for callback in self.task_callbacks['on_start']:
                try:
                    callback(task_execution)
                except Exception as e:
                    logger.error(f"Task start callback failed: {e}")
            
            # Execute task function
            logger.info(f"Executing task {task_id} in execution {execution_id}")
            
            # Prepare task context
            task_context = {
                'execution_id': execution_id,
                'pipeline_id': pipeline_id,
                'task_id': task_id,
                'parameters': task_config.parameters,
                'core_service': self.core_service
            }
            
            # Execute with timeout
            start_time = time.time()
            result = task_config.function(task_context)
            execution_time = time.time() - start_time
            
            # Update task execution
            task_execution.status = TaskStatus.SUCCESS
            task_execution.end_time = datetime.now()
            task_execution.duration = execution_time
            task_execution.output_data = result
            
            # Trigger success callbacks
            for callback in self.task_callbacks['on_success']:
                try:
                    callback(task_execution)
                except Exception as e:
                    logger.error(f"Task success callback failed: {e}")
            
            logger.info(f"Task completed successfully: {task_id}")
            
            # Check if pipeline is complete
            self._check_pipeline_completion(execution_id)
            
        except Exception as e:
            # Handle task failure
            task_execution.status = TaskStatus.FAILED
            task_execution.end_time = datetime.now()
            task_execution.error_message = str(e)
            
            # Trigger failure callbacks
            for callback in self.task_callbacks['on_failure']:
                try:
                    callback(task_execution)
                except Exception as e:
                    logger.error(f"Task failure callback failed: {e}")
            
            logger.error(f"Task failed: {task_id} - {e}")
            
            # Fail the entire pipeline
            self._fail_pipeline_execution(execution_id, f"Task {task_id} failed: {e}")
    
    def _check_pipeline_completion(self, execution_id: str):
        """Check if pipeline execution is complete."""
        try:
            execution = self.active_executions[execution_id]
            
            # Check if all tasks are completed
            all_completed = all(
                task_exec.status in [TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.SKIPPED]
                for task_exec in execution.task_executions.values()
            )
            
            if all_completed:
                # Check if any tasks failed
                any_failed = any(
                    task_exec.status == TaskStatus.FAILED
                    for task_exec in execution.task_executions.values()
                )
                
                if any_failed:
                    self._fail_pipeline_execution(execution_id, "One or more tasks failed")
                else:
                    self._complete_pipeline_execution(execution_id)
            
        except Exception as e:
            logger.error(f"Error checking pipeline completion: {e}")
    
    def _complete_pipeline_execution(self, execution_id: str):
        """Complete pipeline execution successfully."""
        try:
            execution = self.active_executions[execution_id]
            execution.status = PipelineStatus.SUCCESS
            execution.end_time = datetime.now()
            
            if execution.start_time:
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Update metrics
            self.metrics['successful_executions'] += 1
            self.metrics['total_executions'] += 1
            
            # Trigger success callbacks
            for callback in self.pipeline_callbacks['on_success']:
                try:
                    callback(execution)
                except Exception as e:
                    logger.error(f"Pipeline success callback failed: {e}")
            
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution_id]
            
            logger.info(f"Pipeline execution completed successfully: {execution_id}")
            
        except Exception as e:
            logger.error(f"Error completing pipeline execution: {e}")
    
    def _fail_pipeline_execution(self, execution_id: str, error_message: str):
        """Fail pipeline execution."""
        try:
            execution = self.active_executions[execution_id]
            execution.status = PipelineStatus.FAILED
            execution.end_time = datetime.now()
            execution.error_message = error_message
            
            if execution.start_time:
                execution.duration = (execution.end_time - execution.start_time).total_seconds()
            
            # Update metrics
            self.metrics['failed_executions'] += 1
            self.metrics['total_executions'] += 1
            
            # Trigger failure callbacks
            for callback in self.pipeline_callbacks['on_failure']:
                try:
                    callback(execution)
                except Exception as e:
                    logger.error(f"Pipeline failure callback failed: {e}")
            
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution_id]
            
            logger.error(f"Pipeline execution failed: {execution_id} - {error_message}")
            
        except Exception as e:
            logger.error(f"Error failing pipeline execution: {e}")
    
    def _serialize_execution(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Serialize execution for API response."""
        return {
            'execution_id': execution.execution_id,
            'pipeline_id': execution.pipeline_id,
            'status': execution.status.value,
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'duration': execution.duration,
            'trigger_type': execution.trigger_type,
            'trigger_data': execution.trigger_data,
            'error_message': execution.error_message,
            'task_executions': {
                task_id: {
                    'task_id': task_exec.task_id,
                    'status': task_exec.status.value,
                    'start_time': task_exec.start_time.isoformat() if task_exec.start_time else None,
                    'end_time': task_exec.end_time.isoformat() if task_exec.end_time else None,
                    'duration': task_exec.duration,
                    'retry_count': task_exec.retry_count,
                    'error_message': task_exec.error_message,
                    'metrics': task_exec.metrics
                }
                for task_id, task_exec in execution.task_executions.items()
            }
        }