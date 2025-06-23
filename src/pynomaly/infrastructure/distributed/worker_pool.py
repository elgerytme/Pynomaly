"""Worker pool management for distributed processing."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import uuid

from .worker import DistributedWorker
from .task_queue import TaskQueue, Task, TaskStatus
from pynomaly.domain.exceptions import ProcessingError


logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker status enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    FAILED = "failed"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class Worker:
    """Worker information."""
    id: str
    host: str
    port: int
    capacity: int
    current_load: int = 0
    status: WorkerStatus = WorkerStatus.IDLE
    capabilities: List[str] = None
    last_heartbeat: Optional[datetime] = None
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_task_time: float = 0.0
    worker_instance: Optional[DistributedWorker] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
    
    @property
    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        return (
            self.status == WorkerStatus.IDLE and
            self.current_load < self.capacity and
            self.last_heartbeat and
            datetime.now() - self.last_heartbeat < timedelta(minutes=2)
        )
    
    @property
    def load_percentage(self) -> float:
        """Get current load as percentage."""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 100
    
    @property
    def success_rate(self) -> float:
        """Get task success rate."""
        total = self.total_tasks_completed + self.total_tasks_failed
        return (self.total_tasks_completed / total) * 100 if total > 0 else 100
    
    def update_task_stats(self, success: bool, execution_time: float) -> None:
        """Update worker task statistics."""
        if success:
            self.total_tasks_completed += 1
        else:
            self.total_tasks_failed += 1
        
        # Update average execution time (exponential moving average)
        if self.total_tasks_completed > 0:
            self.average_task_time = (
                0.9 * self.average_task_time + 0.1 * execution_time
            )


class WorkerPool:
    """Manages a pool of distributed workers."""
    
    def __init__(self,
                 task_queue: TaskQueue,
                 max_workers: int = 10,
                 heartbeat_timeout: int = 120,
                 auto_scale: bool = True,
                 min_workers: int = 2,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3):
        self.task_queue = task_queue
        self.max_workers = max_workers
        self.heartbeat_timeout = heartbeat_timeout
        self.auto_scale = auto_scale
        self.min_workers = min_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        # Worker management
        self.workers: Dict[str, Worker] = {}
        self.worker_locks: Dict[str, asyncio.Lock] = {}
        
        # Task assignment
        self.assigned_tasks: Dict[str, str] = {}  # task_id -> worker_id
        self.worker_tasks: Dict[str, Set[str]] = {}  # worker_id -> task_ids
        
        # Background tasks
        self._running = False
        self._task_dispatcher: Optional[asyncio.Task] = None
        self._health_monitor: Optional[asyncio.Task] = None
        self._auto_scaler: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        self.start_time = datetime.now()
        
        # Event handlers
        self._worker_event_callbacks: List[Callable] = []
        
        logger.info("Worker pool initialized")
    
    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._task_dispatcher = asyncio.create_task(self._dispatch_tasks())
        self._health_monitor = asyncio.create_task(self._monitor_workers())
        
        if self.auto_scale:
            self._auto_scaler = asyncio.create_task(self._auto_scale_workers())
        
        logger.info("Worker pool started")
    
    async def stop(self) -> None:
        """Stop the worker pool."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        for task in [self._task_dispatcher, self._health_monitor, self._auto_scaler]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop all workers
        for worker in self.workers.values():
            if worker.worker_instance:
                await worker.worker_instance.stop()
        
        logger.info("Worker pool stopped")
    
    async def add_worker(self,
                        worker_id: str,
                        host: str,
                        port: int,
                        capacity: int,
                        capabilities: List[str] = None) -> bool:
        """Add a worker to the pool."""
        if worker_id in self.workers:
            logger.warning(f"Worker {worker_id} already exists")
            return False
        
        if len(self.workers) >= self.max_workers:
            logger.warning(f"Maximum number of workers ({self.max_workers}) reached")
            return False
        
        worker = Worker(
            id=worker_id,
            host=host,
            port=port,
            capacity=capacity,
            capabilities=capabilities or [],
            last_heartbeat=datetime.now()
        )
        
        self.workers[worker_id] = worker
        self.worker_locks[worker_id] = asyncio.Lock()
        self.worker_tasks[worker_id] = set()
        
        await self._notify_worker_event("worker_added", worker)
        
        logger.info(f"Worker {worker_id} added to pool")
        return True
    
    async def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker from the pool."""
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        
        # Reassign active tasks
        active_tasks = self.worker_tasks.get(worker_id, set()).copy()
        for task_id in active_tasks:
            await self._reassign_task(task_id)
        
        # Stop worker instance if managed
        if worker.worker_instance:
            await worker.worker_instance.stop()
        
        # Remove worker
        del self.workers[worker_id]
        del self.worker_locks[worker_id]
        del self.worker_tasks[worker_id]
        
        await self._notify_worker_event("worker_removed", worker)
        
        logger.info(f"Worker {worker_id} removed from pool")
        return True
    
    async def update_worker_heartbeat(self,
                                    worker_id: str,
                                    status: str = "idle",
                                    current_load: int = 0,
                                    metrics: Dict[str, Any] = None) -> bool:
        """Update worker heartbeat and status."""
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        worker.last_heartbeat = datetime.now()
        worker.current_load = current_load
        
        # Update status
        try:
            worker.status = WorkerStatus(status)
        except ValueError:
            logger.warning(f"Invalid worker status: {status}")
        
        # Update metrics if provided
        if metrics:
            if 'average_task_time' in metrics:
                worker.average_task_time = metrics['average_task_time']
        
        return True
    
    async def get_worker_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker pool statistics."""
        total_capacity = sum(w.capacity for w in self.workers.values())
        total_load = sum(w.current_load for w in self.workers.values())
        available_workers = [w for w in self.workers.values() if w.is_available]
        
        # Calculate average metrics
        if self.workers:
            avg_success_rate = sum(w.success_rate for w in self.workers.values()) / len(self.workers)
            avg_task_time = sum(w.average_task_time for w in self.workers.values()) / len(self.workers)
        else:
            avg_success_rate = 0
            avg_task_time = 0
        
        # Worker status distribution
        status_counts = {}
        for worker in self.workers.values():
            status_counts[worker.status.value] = status_counts.get(worker.status.value, 0) + 1
        
        return {
            'total_workers': len(self.workers),
            'available_workers': len(available_workers),
            'total_capacity': total_capacity,
            'current_load': total_load,
            'utilization_percentage': (total_load / total_capacity) * 100 if total_capacity > 0 else 0,
            'average_success_rate': avg_success_rate,
            'average_task_time': avg_task_time,
            'total_tasks_processed': self.total_tasks_processed,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'status_distribution': status_counts,
            'workers': {
                worker_id: {
                    'host': worker.host,
                    'port': worker.port,
                    'status': worker.status.value,
                    'capacity': worker.capacity,
                    'current_load': worker.current_load,
                    'load_percentage': worker.load_percentage,
                    'is_available': worker.is_available,
                    'capabilities': worker.capabilities,
                    'total_tasks_completed': worker.total_tasks_completed,
                    'total_tasks_failed': worker.total_tasks_failed,
                    'success_rate': worker.success_rate,
                    'average_task_time': worker.average_task_time,
                    'last_heartbeat': worker.last_heartbeat.isoformat() if worker.last_heartbeat else None
                }
                for worker_id, worker in self.workers.items()
            }
        }
    
    def add_worker_event_callback(self, callback: Callable) -> None:
        """Add a callback for worker events."""
        self._worker_event_callbacks.append(callback)
    
    async def _dispatch_tasks(self) -> None:
        """Main task dispatcher loop."""
        while self._running:
            try:
                # Get next task from queue
                task = await self.task_queue.get_next_task()
                if not task:
                    await asyncio.sleep(1)
                    continue
                
                # Find suitable worker
                worker = await self._find_suitable_worker(task)
                if not worker:
                    # No suitable worker available, put task back
                    await self.task_queue.fail_task(
                        task.id, 
                        "No suitable worker available"
                    )
                    await asyncio.sleep(1)
                    continue
                
                # Assign task to worker
                await self._assign_task_to_worker(task, worker)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task dispatcher error: {e}")
                await asyncio.sleep(1)
    
    async def _find_suitable_worker(self, task: Task) -> Optional[Worker]:
        """Find the most suitable worker for a task."""
        available_workers = [w for w in self.workers.values() if w.is_available]
        
        if not available_workers:
            return None
        
        # Filter by capabilities if specified
        required_capabilities = task.metadata.get('required_capabilities', [])
        if required_capabilities:
            suitable_workers = []
            for worker in available_workers:
                if all(cap in worker.capabilities for cap in required_capabilities):
                    suitable_workers.append(worker)
            available_workers = suitable_workers
        
        if not available_workers:
            return None
        
        # Score workers based on load, success rate, and task time
        scored_workers = []
        for worker in available_workers:
            # Lower scores are better
            load_score = worker.load_percentage / 100
            failure_score = (100 - worker.success_rate) / 100
            time_score = worker.average_task_time / 100 if worker.average_task_time > 0 else 0
            
            total_score = load_score + failure_score + time_score
            scored_workers.append((total_score, worker))
        
        # Return worker with lowest score
        scored_workers.sort(key=lambda x: x[0])
        return scored_workers[0][1]
    
    async def _assign_task_to_worker(self, task: Task, worker: Worker) -> None:
        """Assign a task to a specific worker."""
        async with self.worker_locks[worker.id]:
            # Update task and worker state
            await self.task_queue.assign_task(task.id, worker.id)
            
            # Track assignment
            self.assigned_tasks[task.id] = worker.id
            self.worker_tasks[worker.id].add(task.id)
            
            # Update worker load
            worker.current_load += 1
            if worker.current_load >= worker.capacity:
                worker.status = WorkerStatus.BUSY
            
            # Start task execution
            asyncio.create_task(self._execute_task(task, worker))
            
            logger.info(f"Task {task.id} assigned to worker {worker.id}")
    
    async def _execute_task(self, task: Task, worker: Worker) -> None:
        """Execute a task on a worker."""
        start_time = datetime.now()
        success = False
        
        try:
            # TODO: Implement actual task execution
            # For now, simulate execution
            await asyncio.sleep(2)
            
            # Simulate success/failure
            import random
            success = random.random() > 0.1  # 90% success rate
            
            if success:
                result = {
                    'task_id': task.id,
                    'worker_id': worker.id,
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'result': 'simulated_result'
                }
                await self.task_queue.complete_task(task.id, result)
            else:
                await self.task_queue.fail_task(task.id, "Simulated failure")
        
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            await self.task_queue.fail_task(task.id, str(e))
        
        finally:
            # Update worker statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            worker.update_task_stats(success, execution_time)
            
            # Update pool statistics
            self.total_tasks_processed += 1
            self.total_processing_time += execution_time
            
            # Clean up task assignment
            async with self.worker_locks[worker.id]:
                if task.id in self.assigned_tasks:
                    del self.assigned_tasks[task.id]
                
                if task.id in self.worker_tasks[worker.id]:
                    self.worker_tasks[worker.id].remove(task.id)
                
                # Update worker load and status
                worker.current_load = max(0, worker.current_load - 1)
                if worker.current_load < worker.capacity:
                    worker.status = WorkerStatus.IDLE
    
    async def _reassign_task(self, task_id: str) -> None:
        """Reassign a task to a different worker."""
        if task_id not in self.assigned_tasks:
            return
        
        old_worker_id = self.assigned_tasks[task_id]
        
        # Clean up old assignment
        if old_worker_id in self.worker_tasks:
            self.worker_tasks[old_worker_id].discard(task_id)
        
        if task_id in self.assigned_tasks:
            del self.assigned_tasks[task_id]
        
        # Mark task for retry
        await self.task_queue.fail_task(task_id, f"Worker {old_worker_id} failed")
        
        logger.info(f"Task {task_id} reassigned from failed worker {old_worker_id}")
    
    async def _monitor_workers(self) -> None:
        """Monitor worker health and handle failures."""
        while self._running:
            try:
                current_time = datetime.now()
                failed_workers = []
                
                for worker_id, worker in self.workers.items():
                    # Check heartbeat timeout
                    if (worker.last_heartbeat and 
                        current_time - worker.last_heartbeat > timedelta(seconds=self.heartbeat_timeout)):
                        failed_workers.append(worker_id)
                        worker.status = WorkerStatus.OFFLINE
                
                # Handle failed workers
                for worker_id in failed_workers:
                    logger.warning(f"Worker {worker_id} appears offline")
                    await self._handle_failed_worker(worker_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _handle_failed_worker(self, worker_id: str) -> None:
        """Handle a failed worker."""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        worker.status = WorkerStatus.FAILED
        
        # Reassign all tasks from this worker
        active_tasks = self.worker_tasks.get(worker_id, set()).copy()
        for task_id in active_tasks:
            await self._reassign_task(task_id)
        
        await self._notify_worker_event("worker_failed", worker)
        
        logger.error(f"Worker {worker_id} failed, reassigned {len(active_tasks)} tasks")
    
    async def _auto_scale_workers(self) -> None:
        """Automatically scale worker pool based on demand."""
        while self._running:
            try:
                stats = await self.get_worker_stats()
                utilization = stats['utilization_percentage']
                
                # Scale up if utilization is high
                if (utilization > self.scale_up_threshold * 100 and
                    len(self.workers) < self.max_workers):
                    
                    # TODO: Implement actual worker scaling
                    logger.info(f"Auto-scaling: High utilization ({utilization:.1f}%), need more workers")
                
                # Scale down if utilization is low
                elif (utilization < self.scale_down_threshold * 100 and
                      len(self.workers) > self.min_workers):
                    
                    # TODO: Implement actual worker scaling
                    logger.info(f"Auto-scaling: Low utilization ({utilization:.1f}%), can reduce workers")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(60)
    
    async def _notify_worker_event(self, event_type: str, worker: Worker) -> None:
        """Notify callbacks of worker events."""
        for callback in self._worker_event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, worker)
                else:
                    callback(event_type, worker)
            except Exception as e:
                logger.error(f"Worker event callback failed: {e}")