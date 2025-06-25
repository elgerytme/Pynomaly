"""Task tracking service for long-running operations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of tasks that can be tracked."""

    AUTOML_OPTIMIZATION = "automl_optimization"
    ENSEMBLE_CREATION = "ensemble_creation"
    DETECTOR_TRAINING = "detector_training"
    BATCH_DETECTION = "batch_detection"
    DATA_PROCESSING = "data_processing"


@dataclass
class TaskProgress:
    """Task progress information."""

    current: int = 0
    total: int = 100
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total == 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100.0)


@dataclass
class TaskInfo:
    """Complete task information."""

    task_id: str
    task_type: TaskType
    status: TaskStatus
    name: str
    description: str
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    result: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "name": self.name,
            "description": self.description,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "progress": {
                "current": self.progress.current,
                "total": self.progress.total,
                "percentage": self.progress.percentage,
                "message": self.progress.message,
                "details": self.progress.details,
            },
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class TaskTrackingService:
    """Service for tracking long-running tasks with real-time updates."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tasks: dict[str, TaskInfo] = {}
        self.subscribers: dict[str, set[Callable]] = {}
        self.background_tasks: set[asyncio.Task] = set()

    def create_task(
        self,
        task_type: TaskType,
        name: str,
        description: str,
        total_steps: int = 100,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new tracked task."""
        task_id = str(uuid4())

        task_info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            name=name,
            description=description,
            started_at=datetime.now(),
            updated_at=datetime.now(),
            progress=TaskProgress(total=total_steps),
            metadata=metadata or {},
        )

        self.tasks[task_id] = task_info
        self.subscribers[task_id] = set()

        self.logger.info(f"Created task {task_id}: {name}")
        return task_id

    def start_task(self, task_id: str) -> bool:
        """Mark task as started."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.updated_at = datetime.now()

        self._notify_subscribers(task_id)
        self.logger.info(f"Started task {task_id}")
        return True

    def update_progress(
        self,
        task_id: str,
        current: int | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Update task progress."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        if current is not None:
            task.progress.current = current

        if message is not None:
            task.progress.message = message

        if details is not None:
            task.progress.details.update(details)

        task.updated_at = datetime.now()

        self._notify_subscribers(task_id)
        return True

    def complete_task(self, task_id: str, result: Any | None = None) -> bool:
        """Mark task as completed."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.progress.current = task.progress.total
        task.progress.message = "Task completed successfully"
        task.completed_at = datetime.now()
        task.updated_at = datetime.now()
        task.result = result

        self._notify_subscribers(task_id)
        self.logger.info(f"Completed task {task_id}")
        return True

    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = datetime.now()
        task.updated_at = datetime.now()

        self._notify_subscribers(task_id)
        self.logger.error(f"Failed task {task_id}: {error}")
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        task.updated_at = datetime.now()

        self._notify_subscribers(task_id)
        self.logger.info(f"Cancelled task {task_id}")
        return True

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task information."""
        return self.tasks.get(task_id)

    def get_tasks_by_type(self, task_type: TaskType) -> list[TaskInfo]:
        """Get all tasks of a specific type."""
        return [task for task in self.tasks.values() if task.task_type == task_type]

    def get_active_tasks(self) -> list[TaskInfo]:
        """Get all active (running or pending) tasks."""
        return [
            task
            for task in self.tasks.values()
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        ]

    def get_recent_tasks(self, limit: int = 20) -> list[TaskInfo]:
        """Get recent tasks ordered by update time."""
        sorted_tasks = sorted(
            self.tasks.values(), key=lambda t: t.updated_at, reverse=True
        )
        return sorted_tasks[:limit]

    def subscribe_to_task(
        self, task_id: str, callback: Callable[[TaskInfo], None]
    ) -> bool:
        """Subscribe to task updates."""
        if task_id not in self.tasks:
            return False

        self.subscribers[task_id].add(callback)
        return True

    def unsubscribe_from_task(
        self, task_id: str, callback: Callable[[TaskInfo], None]
    ) -> bool:
        """Unsubscribe from task updates."""
        if task_id not in self.subscribers:
            return False

        self.subscribers[task_id].discard(callback)
        return True

    def _notify_subscribers(self, task_id: str):
        """Notify all subscribers of task updates."""
        if task_id not in self.subscribers:
            return

        task = self.tasks[task_id]
        for callback in self.subscribers[task_id]:
            try:
                callback(task)
            except Exception as e:
                self.logger.error(f"Error in task subscriber callback: {e}")

    async def run_task_with_tracking(
        self, task_id: str, async_func: Callable, *args, **kwargs
    ) -> Any:
        """Run an async function with automatic task tracking."""
        try:
            self.start_task(task_id)
            result = await async_func(*args, **kwargs)
            self.complete_task(task_id, result)
            return result
        except Exception as e:
            self.fail_task(task_id, str(e))
            raise

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old completed/failed tasks."""
        current_time = datetime.now()
        cutoff_time = current_time.timestamp() - (max_age_hours * 3600)

        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (
                task.status
                in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                and task.updated_at.timestamp() < cutoff_time
            ):
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            if task_id in self.subscribers:
                del self.subscribers[task_id]

        if tasks_to_remove:
            self.logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")


# Example progress tracking decorators
def track_progress(
    task_service: TaskTrackingService, task_type: TaskType, name: str, description: str
):
    """Decorator for automatic task tracking."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            task_id = task_service.create_task(task_type, name, description)
            try:
                task_service.start_task(task_id)
                result = await func(task_service, task_id, *args, **kwargs)
                task_service.complete_task(task_id, result)
                return result
            except Exception as e:
                task_service.fail_task(task_id, str(e))
                raise

        return wrapper

    return decorator


# Example usage context manager
class TaskContext:
    """Context manager for task tracking."""

    def __init__(
        self,
        task_service: TaskTrackingService,
        task_type: TaskType,
        name: str,
        description: str,
        total_steps: int = 100,
    ):
        self.task_service = task_service
        self.task_id = task_service.create_task(
            task_type, name, description, total_steps
        )

    def __enter__(self):
        self.task_service.start_task(self.task_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.task_service.complete_task(self.task_id)
        else:
            self.task_service.fail_task(self.task_id, str(exc_val))

    def update(
        self, current: int | None = None, message: str | None = None, **details
    ):
        """Update task progress."""
        self.task_service.update_progress(self.task_id, current, message, details)
