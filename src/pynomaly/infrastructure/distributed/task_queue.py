"""Task queue management for distributed processing."""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import heapq
import time

from pynomaly.domain.exceptions import ProcessingError


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10


@dataclass
class Task:
    """Represents a task in the distributed system."""
    id: str
    type: str
    payload: Dict[str, Any]
    priority: int = TaskPriority.NORMAL.value
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    assigned_worker: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.scheduled_at is None:
            self.scheduled_at = self.created_at
    
    def __lt__(self, other: 'Task') -> bool:
        """Compare tasks for priority queue (higher priority first)."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to be executed."""
        now = datetime.now()
        return (
            self.status == TaskStatus.QUEUED and
            (self.scheduled_at is None or self.scheduled_at <= now) and
            (self.deadline is None or self.deadline > now)
        )
    
    @property
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if self.deadline is None:
            return False
        return datetime.now() > self.deadline
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get task execution time if completed."""
        if self.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return None
        
        start_time = self.metadata.get('start_time')
        end_time = self.metadata.get('end_time')
        
        if start_time and end_time:
            return end_time - start_time
        return None
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return (
            self.status == TaskStatus.FAILED and
            self.retry_count < self.max_retries and
            not self.is_expired
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'id': self.id,
            'type': self.type,
            'payload': self.payload,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'assigned_worker': self.assigned_worker,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'metadata': self.metadata,
            'dependencies': self.dependencies,
            'execution_time': self.execution_time
        }


class TaskQueue:
    """Priority queue for managing distributed tasks."""
    
    def __init__(self, 
                 max_size: int = 10000,
                 cleanup_interval: int = 300,
                 task_timeout: int = 3600):
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.task_timeout = task_timeout
        
        # Task storage
        self._pending_heap: List[Task] = []
        self._tasks: Dict[str, Task] = {}
        self._task_lock = asyncio.Lock()
        
        # Task tracking
        self._task_dependencies: Dict[str, List[str]] = {}  # task_id -> dependent_task_ids
        self._completed_tasks: Dict[str, Task] = {}
        self._failed_tasks: Dict[str, Task] = {}
        
        # Background tasks
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._dependency_checker: Optional[asyncio.Task] = None
        
        # Event handlers
        self._task_handlers: Dict[str, Callable] = {}
        self._status_change_callbacks: List[Callable] = []
        
        logger.info("Task queue initialized")
    
    async def start(self) -> None:
        """Start the task queue."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._dependency_checker = asyncio.create_task(self._dependency_check_loop())
        
        logger.info("Task queue started")
    
    async def stop(self) -> None:
        """Stop the task queue."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._dependency_checker:
            self._dependency_checker.cancel()
            try:
                await self._dependency_checker
            except asyncio.CancelledError:
                pass
        
        logger.info("Task queue stopped")
    
    async def submit_task(self, 
                         task_type: str,
                         payload: Dict[str, Any],
                         priority: int = TaskPriority.NORMAL.value,
                         scheduled_at: Optional[datetime] = None,
                         deadline: Optional[datetime] = None,
                         timeout: Optional[int] = None,
                         max_retries: int = 3,
                         dependencies: List[str] = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Submit a new task to the queue."""
        if len(self._tasks) >= self.max_size:
            raise ProcessingError("Task queue is full")
        
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            type=task_type,
            payload=payload,
            priority=priority,
            scheduled_at=scheduled_at,
            deadline=deadline,
            timeout=timeout or self.task_timeout,
            max_retries=max_retries,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        async with self._task_lock:
            # Check dependencies exist
            if task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id not in self._tasks and dep_id not in self._completed_tasks:
                        raise ProcessingError(f"Dependency task {dep_id} not found")
                
                # Track reverse dependencies
                for dep_id in task.dependencies:
                    if dep_id not in self._task_dependencies:
                        self._task_dependencies[dep_id] = []
                    self._task_dependencies[dep_id].append(task_id)
            
            # Add task
            self._tasks[task_id] = task
            
            # Queue if dependencies are satisfied
            if await self._check_dependencies(task):
                task.status = TaskStatus.QUEUED
                heapq.heappush(self._pending_heap, task)
        
        # Notify callbacks
        await self._notify_status_change(task)
        
        logger.info(f"Task {task_id} submitted with priority {priority}")
        return task_id
    
    async def get_next_task(self, worker_capabilities: List[str] = None) -> Optional[Task]:
        """Get the next available task for processing."""
        async with self._task_lock:
            # Find suitable task
            available_tasks = []
            temp_heap = []
            
            while self._pending_heap:
                task = heapq.heappop(self._pending_heap)
                
                if task.id not in self._tasks:
                    continue  # Task was removed
                
                if not task.is_ready:
                    temp_heap.append(task)
                    continue
                
                if task.is_expired:
                    task.status = TaskStatus.TIMEOUT
                    await self._move_to_failed(task)
                    continue
                
                # Check worker capabilities
                required_capabilities = task.metadata.get('required_capabilities', [])
                if required_capabilities and worker_capabilities:
                    if not all(cap in worker_capabilities for cap in required_capabilities):
                        temp_heap.append(task)
                        continue
                
                available_tasks.append(task)
            
            # Restore remaining tasks to heap
            for task in temp_heap:
                heapq.heappush(self._pending_heap, task)
            
            # Return highest priority available task
            if available_tasks:
                task = min(available_tasks)  # Highest priority
                task.status = TaskStatus.ASSIGNED
                
                # Put other tasks back
                for other_task in available_tasks[1:]:
                    heapq.heappush(self._pending_heap, other_task)
                
                await self._notify_status_change(task)
                return task
        
        return None
    
    async def assign_task(self, task_id: str, worker_id: str) -> bool:
        """Assign a task to a worker."""
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.ASSIGNED:
                return False
            
            task.assigned_worker = worker_id
            task.status = TaskStatus.RUNNING
            task.metadata['start_time'] = time.time()
            task.metadata['worker_id'] = worker_id
        
        await self._notify_status_change(task)
        logger.info(f"Task {task_id} assigned to worker {worker_id}")
        return True
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed."""
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.RUNNING:
                return False
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.metadata['end_time'] = time.time()
            
            # Move to completed tasks
            self._completed_tasks[task_id] = task
            del self._tasks[task_id]
            
            # Check dependent tasks
            await self._check_dependent_tasks(task_id)
        
        await self._notify_status_change(task)
        logger.info(f"Task {task_id} completed successfully")
        return True
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed."""
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            task.error = error
            task.metadata['end_time'] = time.time()
            
            if task.can_retry():
                # Retry task
                task.retry_count += 1
                task.status = TaskStatus.QUEUED
                task.assigned_worker = None
                task.error = None
                
                # Reset timing
                task.metadata.pop('start_time', None)
                task.metadata.pop('end_time', None)
                
                # Re-queue
                heapq.heappush(self._pending_heap, task)
                
                logger.info(f"Task {task_id} queued for retry {task.retry_count}")
            else:
                # Move to failed tasks
                task.status = TaskStatus.FAILED
                await self._move_to_failed(task)
                
                logger.error(f"Task {task_id} failed permanently: {error}")
        
        await self._notify_status_change(task)
        return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if not task or task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False
            
            task.status = TaskStatus.CANCELLED
            task.metadata['end_time'] = time.time()
            
            # Remove from pending heap if queued
            if task in self._pending_heap:
                self._pending_heap.remove(task)
                heapq.heapify(self._pending_heap)
            
            # Move to failed tasks
            await self._move_to_failed(task)
        
        await self._notify_status_change(task)
        logger.info(f"Task {task_id} cancelled")
        return True
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        # Check active tasks
        if task_id in self._tasks:
            return self._tasks[task_id]
        
        # Check completed tasks
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id]
        
        # Check failed tasks
        if task_id in self._failed_tasks:
            return self._failed_tasks[task_id]
        
        return None
    
    async def list_tasks(self, 
                        status: Optional[TaskStatus] = None,
                        task_type: Optional[str] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Task]:
        """List tasks with optional filtering."""
        all_tasks = []
        
        # Collect all tasks
        all_tasks.extend(self._tasks.values())
        all_tasks.extend(self._completed_tasks.values())
        all_tasks.extend(self._failed_tasks.values())
        
        # Apply filters
        if status:
            all_tasks = [t for t in all_tasks if t.status == status]
        
        if task_type:
            all_tasks = [t for t in all_tasks if t.type == task_type]
        
        # Sort by creation time (newest first)
        all_tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        # Apply pagination
        return all_tasks[offset:offset + limit]
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = {
            'pending': len([t for t in self._tasks.values() if t.status == TaskStatus.QUEUED]),
            'assigned': len([t for t in self._tasks.values() if t.status == TaskStatus.ASSIGNED]),
            'running': len([t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]),
            'completed': len(self._completed_tasks),
            'failed': len(self._failed_tasks),
            'total_active': len(self._tasks),
            'queue_utilization': len(self._tasks) / self.max_size,
            'dependency_chains': len(self._task_dependencies)
        }
        
        # Task type distribution
        type_counts = {}
        for task in self._tasks.values():
            type_counts[task.type] = type_counts.get(task.type, 0) + 1
        stats['task_types'] = type_counts
        
        return stats
    
    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for a specific task type."""
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def add_status_change_callback(self, callback: Callable) -> None:
        """Add a callback for task status changes."""
        self._status_change_callbacks.append(callback)
    
    async def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id in self._tasks:
                return False  # Dependency still pending
            if dep_id not in self._completed_tasks:
                return False  # Dependency failed or doesn't exist
        
        return True
    
    async def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """Check and queue tasks that depend on a completed task."""
        if completed_task_id not in self._task_dependencies:
            return
        
        dependent_task_ids = self._task_dependencies[completed_task_id]
        
        for task_id in dependent_task_ids:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status == TaskStatus.PENDING and await self._check_dependencies(task):
                    task.status = TaskStatus.QUEUED
                    heapq.heappush(self._pending_heap, task)
                    await self._notify_status_change(task)
        
        # Clean up dependency tracking
        del self._task_dependencies[completed_task_id]
    
    async def _move_to_failed(self, task: Task) -> None:
        """Move task to failed tasks storage."""
        self._failed_tasks[task.id] = task
        if task.id in self._tasks:
            del self._tasks[task.id]
        
        # Remove from heap if present
        if task in self._pending_heap:
            self._pending_heap.remove(task)
            heapq.heapify(self._pending_heap)
        
        # Fail dependent tasks
        if task.id in self._task_dependencies:
            dependent_task_ids = self._task_dependencies[task.id]
            for dep_task_id in dependent_task_ids:
                if dep_task_id in self._tasks:
                    await self.fail_task(dep_task_id, f"Dependency {task.id} failed")
    
    async def _notify_status_change(self, task: Task) -> None:
        """Notify callbacks of task status changes."""
        for callback in self._status_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                logger.error(f"Status change callback failed: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old tasks."""
        while self._running:
            try:
                await self._cleanup_old_tasks()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    async def _cleanup_old_tasks(self) -> None:
        """Clean up old completed and failed tasks."""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep tasks for 24 hours
        
        # Clean completed tasks
        old_completed = [
            task_id for task_id, task in self._completed_tasks.items()
            if task.created_at < cutoff_time
        ]
        
        for task_id in old_completed:
            del self._completed_tasks[task_id]
        
        # Clean failed tasks
        old_failed = [
            task_id for task_id, task in self._failed_tasks.items()
            if task.created_at < cutoff_time
        ]
        
        for task_id in old_failed:
            del self._failed_tasks[task_id]
        
        if old_completed or old_failed:
            logger.info(f"Cleaned up {len(old_completed)} completed and {len(old_failed)} failed tasks")
    
    async def _dependency_check_loop(self) -> None:
        """Periodic check for tasks with satisfied dependencies."""
        while self._running:
            try:
                async with self._task_lock:
                    pending_tasks = [
                        task for task in self._tasks.values()
                        if task.status == TaskStatus.PENDING
                    ]
                    
                    for task in pending_tasks:
                        if await self._check_dependencies(task):
                            task.status = TaskStatus.QUEUED
                            heapq.heappush(self._pending_heap, task)
                            await self._notify_status_change(task)
                
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dependency check loop error: {e}")
                await asyncio.sleep(10)