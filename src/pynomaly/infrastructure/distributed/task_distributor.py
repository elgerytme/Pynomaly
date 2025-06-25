"""Task distribution and management for distributed processing."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from concurrent.futures import Future

from pydantic import BaseModel, Field

from .distributed_config import get_distributed_config_manager, DistributedConfig

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    """Types of tasks that can be distributed."""
    ANOMALY_DETECTION = "anomaly_detection"
    MODEL_TRAINING = "model_training"
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    MODEL_EVALUATION = "model_evaluation"
    BATCH_PREDICTION = "batch_prediction"
    ENSEMBLE_AGGREGATION = "ensemble_aggregation"


@dataclass
class TaskMetadata:
    """Metadata for distributed tasks."""
    
    # Basic information
    task_id: str
    task_type: TaskType
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Timing information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Resource requirements
    estimated_memory_mb: int = 512
    estimated_cpu_cores: int = 1
    estimated_duration_seconds: float = 60.0
    requires_gpu: bool = False
    
    # Dependencies and constraints
    dependencies: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Execution information
    assigned_worker: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Progress tracking
    progress_percentage: float = 0.0
    current_step: str = ""
    total_steps: int = 1


class DistributedTask(BaseModel):
    """Represents a task for distributed execution."""
    
    # Core task information
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Task priority")
    
    # Task payload
    function_name: str = Field(..., description="Function to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Function arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Function keyword arguments")
    
    # Execution context
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    # Resource requirements
    resource_requirements: Dict[str, Union[int, float, bool]] = Field(
        default_factory=lambda: {
            "memory_mb": 512,
            "cpu_cores": 1,
            "duration_seconds": 60.0,
            "requires_gpu": False
        },
        description="Resource requirements"
    )
    
    # Metadata
    metadata: Optional[TaskMetadata] = Field(default=None, description="Task metadata")
    
    def __post_init__(self):
        """Initialize metadata after creation."""
        if self.metadata is None:
            self.metadata = TaskMetadata(
                task_id=self.task_id,
                task_type=self.task_type,
                priority=self.priority,
                estimated_memory_mb=self.resource_requirements.get("memory_mb", 512),
                estimated_cpu_cores=self.resource_requirements.get("cpu_cores", 1),
                estimated_duration_seconds=self.resource_requirements.get("duration_seconds", 60.0),
                requires_gpu=self.resource_requirements.get("requires_gpu", False)
            )


class TaskResult(BaseModel):
    """Result of a distributed task execution."""
    
    # Task identification
    task_id: str = Field(..., description="Task identifier")
    worker_id: str = Field(..., description="Worker that executed the task")
    
    # Execution status
    status: TaskStatus = Field(..., description="Task execution status")
    
    # Result data
    result: Optional[Any] = Field(default=None, description="Task result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    traceback: Optional[str] = Field(default=None, description="Error traceback")
    
    # Timing information
    started_at: datetime = Field(..., description="Task start time")
    completed_at: datetime = Field(..., description="Task completion time")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    
    # Resource usage
    memory_used_mb: float = Field(default=0.0, description="Memory used in MB")
    cpu_time_seconds: float = Field(default=0.0, description="CPU time used")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @property
    def is_successful(self) -> bool:
        """Check if task was successful."""
        return self.status == TaskStatus.COMPLETED and self.error is None


class TaskQueue:
    """Thread-safe task queue with priority support."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize task queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queues = {
            TaskPriority.CRITICAL: deque(),
            TaskPriority.HIGH: deque(),
            TaskPriority.NORMAL: deque(),
            TaskPriority.LOW: deque()
        }
        self._task_index = {}  # task_id -> task mapping
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)
        self._size = 0
    
    async def put(self, task: DistributedTask) -> bool:
        """Add task to queue.
        
        Args:
            task: Task to add
            
        Returns:
            True if task was added, False if queue is full
        """
        async with self._not_full:
            if self._size >= self.max_size:
                return False
            
            self._queues[task.priority].append(task)
            self._task_index[task.task_id] = task
            self._size += 1
            
            self._not_empty.notify()
            return True
    
    async def get(self) -> Optional[DistributedTask]:
        """Get next task from queue.
        
        Returns:
            Next task or None if queue is empty
        """
        async with self._not_empty:
            # Try to get task from highest priority queue first
            for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                           TaskPriority.NORMAL, TaskPriority.LOW]:
                if self._queues[priority]:
                    task = self._queues[priority].popleft()
                    del self._task_index[task.task_id]
                    self._size -= 1
                    self._not_full.notify()
                    return task
            
            return None
    
    async def wait_for_task(self, timeout: Optional[float] = None) -> Optional[DistributedTask]:
        """Wait for a task to become available.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            Next available task or None if timeout
        """
        async with self._not_empty:
            # Check if task is immediately available
            task = await self.get()
            if task:
                return task
            
            # Wait for task with timeout
            try:
                await asyncio.wait_for(self._not_empty.wait(), timeout=timeout)
                return await self.get()
            except asyncio.TimeoutError:
                return None
    
    async def remove(self, task_id: str) -> bool:
        """Remove task from queue.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was removed
        """
        async with self._lock:
            if task_id not in self._task_index:
                return False
            
            task = self._task_index[task_id]
            try:
                self._queues[task.priority].remove(task)
                del self._task_index[task_id]
                self._size -= 1
                self._not_full.notify()
                return True
            except ValueError:
                return False
    
    def size(self) -> int:
        """Get current queue size."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._size == 0
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self._size >= self.max_size
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics.
        
        Returns:
            Queue statistics by priority
        """
        return {
            "total": self._size,
            "critical": len(self._queues[TaskPriority.CRITICAL]),
            "high": len(self._queues[TaskPriority.HIGH]),
            "normal": len(self._queues[TaskPriority.NORMAL]),
            "low": len(self._queues[TaskPriority.LOW])
        }


class TaskDistributor:
    """Manages task distribution across worker nodes."""
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        """Initialize task distributor.
        
        Args:
            config: Distributed configuration
        """
        self.config = config or get_distributed_config_manager().get_effective_config()
        
        # Task management
        self.pending_queue = TaskQueue(max_size=self.config.worker.max_concurrent_tasks * 100)
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.failed_tasks: Dict[str, TaskResult] = {}
        
        # Worker tracking
        self.available_workers: Set[str] = set()
        self.worker_loads: Dict[str, float] = defaultdict(float)
        self.worker_capabilities: Dict[str, Set[str]] = defaultdict(set)
        
        # Callbacks and handlers
        self.task_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.completion_handlers: List[Callable[[TaskResult], None]] = []
        
        # Metrics
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        
        # Background tasks
        self._distribution_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the task distributor."""
        if self._running:
            return
        
        self._running = True
        self._distribution_task = asyncio.create_task(self._distribution_loop())
        logger.info("Task distributor started")
    
    async def stop(self) -> None:
        """Stop the task distributor."""
        if not self._running:
            return
        
        self._running = False
        
        if self._distribution_task:
            self._distribution_task.cancel()
            try:
                await self._distribution_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Task distributor stopped")
    
    async def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed execution.
        
        Args:
            task: Task to submit
            
        Returns:
            Task ID
            
        Raises:
            RuntimeError: If task queue is full
        """
        # Initialize metadata if not set
        if task.metadata is None:
            task.metadata = TaskMetadata(
                task_id=task.task_id,
                task_type=task.task_type,
                priority=task.priority
            )
        
        # Add to pending queue
        if not await self.pending_queue.put(task):
            raise RuntimeError("Task queue is full")
        
        self.total_tasks_submitted += 1
        logger.info(f"Task {task.task_id} submitted with priority {task.priority}")
        
        return task.task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled
        """
        # Try to remove from pending queue first
        if await self.pending_queue.remove(task_id):
            logger.info(f"Task {task_id} cancelled (was pending)")
            return True
        
        # Check if task is running
        if task_id in self.running_tasks:
            # TODO: Implement cancellation for running tasks
            logger.warning(f"Task {task_id} cancellation not yet implemented for running tasks")
            return False
        
        return False
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status or None if not found
        """
        # Check pending queue
        if task_id in self.pending_queue._task_index:
            return TaskStatus.QUEUED
        
        # Check running tasks
        if task_id in self.running_tasks:
            return TaskStatus.RUNNING
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return TaskStatus.COMPLETED
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            return TaskStatus.FAILED
        
        return None
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result of a completed task.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for result
            
        Returns:
            Task result or None if not available
        """
        start_time = time.time()
        
        while True:
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                return self.failed_tasks[task_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    def register_worker(self, worker_id: str, capabilities: Set[str]) -> None:
        """Register a worker node.
        
        Args:
            worker_id: Worker identifier
            capabilities: Worker capabilities
        """
        self.available_workers.add(worker_id)
        self.worker_capabilities[worker_id] = capabilities
        self.worker_loads[worker_id] = 0.0
        
        logger.info(f"Worker {worker_id} registered with capabilities: {capabilities}")
    
    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker node.
        
        Args:
            worker_id: Worker identifier
        """
        self.available_workers.discard(worker_id)
        self.worker_capabilities.pop(worker_id, None)
        self.worker_loads.pop(worker_id, None)
        
        # Move any running tasks back to pending
        tasks_to_reassign = []
        for task_id, task in self.running_tasks.items():
            if task.metadata and task.metadata.assigned_worker == worker_id:
                tasks_to_reassign.append(task)
        
        for task in tasks_to_reassign:
            del self.running_tasks[task.task_id]
            task.metadata.assigned_worker = None
            task.metadata.retry_count += 1
            # Re-queue if retries available
            if task.metadata.retry_count <= task.metadata.max_retries:
                asyncio.create_task(self.pending_queue.put(task))
        
        logger.info(f"Worker {worker_id} unregistered")
    
    def update_worker_load(self, worker_id: str, load: float) -> None:
        """Update worker load information.
        
        Args:
            worker_id: Worker identifier
            load: Current load (0.0 to 1.0)
        """
        if worker_id in self.available_workers:
            self.worker_loads[worker_id] = load
    
    async def _distribution_loop(self) -> None:
        """Main task distribution loop."""
        while self._running:
            try:
                # Get next task from queue
                task = await self.pending_queue.wait_for_task(timeout=1.0)
                if not task:
                    continue
                
                # Find suitable worker
                worker_id = self._find_best_worker(task)
                if not worker_id:
                    # No suitable worker available, put task back
                    await self.pending_queue.put(task)
                    await asyncio.sleep(1.0)
                    continue
                
                # Assign task to worker
                await self._assign_task_to_worker(task, worker_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in distribution loop: {e}")
                await asyncio.sleep(1.0)
    
    def _find_best_worker(self, task: DistributedTask) -> Optional[str]:
        """Find the best worker for a task.
        
        Args:
            task: Task to assign
            
        Returns:
            Best worker ID or None if no suitable worker
        """
        if not self.available_workers:
            return None
        
        # Filter workers by capabilities
        suitable_workers = []
        for worker_id in self.available_workers:
            worker_caps = self.worker_capabilities[worker_id]
            
            # Check if worker supports the task type
            if task.task_type.value in worker_caps or "all" in worker_caps:
                suitable_workers.append(worker_id)
        
        if not suitable_workers:
            return None
        
        # Use load balancing strategy
        strategy = self.config.cluster.load_balancing_strategy
        
        if strategy == "round_robin":
            # Simple round-robin (using hash of task ID)
            index = hash(task.task_id) % len(suitable_workers)
            return suitable_workers[index]
        
        elif strategy == "least_loaded":
            # Select worker with lowest load
            return min(suitable_workers, key=lambda w: self.worker_loads[w])
        
        elif strategy == "weighted":
            # TODO: Implement weighted selection
            return min(suitable_workers, key=lambda w: self.worker_loads[w])
        
        else:
            # Default to least loaded
            return min(suitable_workers, key=lambda w: self.worker_loads[w])
    
    async def _assign_task_to_worker(self, task: DistributedTask, worker_id: str) -> None:
        """Assign a task to a specific worker.
        
        Args:
            task: Task to assign
            worker_id: Target worker ID
        """
        # Update task metadata
        if task.metadata:
            task.metadata.assigned_worker = worker_id
            task.metadata.started_at = datetime.now(timezone.utc)
        
        # Add to running tasks
        self.running_tasks[task.task_id] = task
        
        # TODO: Actually send task to worker
        # For now, simulate task execution
        asyncio.create_task(self._simulate_task_execution(task, worker_id))
        
        logger.info(f"Task {task.task_id} assigned to worker {worker_id}")
    
    async def _simulate_task_execution(self, task: DistributedTask, worker_id: str) -> None:
        """Simulate task execution (placeholder for actual worker communication).
        
        Args:
            task: Task to execute
            worker_id: Worker executing the task
        """
        try:
            # Simulate execution time
            execution_time = task.resource_requirements.get("duration_seconds", 1.0)
            await asyncio.sleep(min(execution_time, 5.0))  # Cap simulation time
            
            # Create successful result
            result = TaskResult(
                task_id=task.task_id,
                worker_id=worker_id,
                status=TaskStatus.COMPLETED,
                result={"simulated": True, "task_type": task.task_type},
                started_at=task.metadata.started_at if task.metadata else datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                execution_time_seconds=execution_time,
                memory_used_mb=task.resource_requirements.get("memory_mb", 512),
                cpu_time_seconds=execution_time * 0.8
            )
            
            await self._handle_task_completion(result)
            
        except Exception as e:
            # Create failed result
            result = TaskResult(
                task_id=task.task_id,
                worker_id=worker_id,
                status=TaskStatus.FAILED,
                error=str(e),
                started_at=task.metadata.started_at if task.metadata else datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                execution_time_seconds=0.0
            )
            
            await self._handle_task_completion(result)
    
    async def _handle_task_completion(self, result: TaskResult) -> None:
        """Handle task completion.
        
        Args:
            result: Task execution result
        """
        # Remove from running tasks
        self.running_tasks.pop(result.task_id, None)
        
        # Store result
        if result.is_successful:
            self.completed_tasks[result.task_id] = result
            self.total_tasks_completed += 1
        else:
            self.failed_tasks[result.task_id] = result
            self.total_tasks_failed += 1
        
        # Call completion handlers
        for handler in self.completion_handlers:
            try:
                handler(result)
            except Exception as e:
                logger.error(f"Error in completion handler: {e}")
        
        logger.info(f"Task {result.task_id} completed with status {result.status}")
    
    def add_completion_handler(self, handler: Callable[[TaskResult], None]) -> None:
        """Add a task completion handler.
        
        Args:
            handler: Completion handler function
        """
        self.completion_handlers.append(handler)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get distribution statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_submitted": self.total_tasks_submitted,
            "total_completed": self.total_tasks_completed,
            "total_failed": self.total_tasks_failed,
            "pending_count": self.pending_queue.size(),
            "running_count": len(self.running_tasks),
            "queue_stats": self.pending_queue.get_stats(),
            "available_workers": len(self.available_workers),
            "worker_loads": dict(self.worker_loads)
        }