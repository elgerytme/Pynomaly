"""Distributed processing manager for coordinating anomaly detection across multiple nodes."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import uuid

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import ProcessingError


logger = logging.getLogger(__name__)


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    id: str
    host: str
    port: int
    capacity: int
    current_load: int = 0
    status: str = "idle"  # idle, busy, offline
    last_heartbeat: Optional[datetime] = None
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
    
    @property
    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        return (
            self.status == "idle" and 
            self.current_load < self.capacity and
            self.last_heartbeat and
            datetime.now() - self.last_heartbeat < timedelta(minutes=2)
        )
    
    @property
    def load_percentage(self) -> float:
        """Get current load as percentage."""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 100


@dataclass
class DetectionTask:
    """Represents a detection task to be distributed."""
    id: str
    detector_id: str
    dataset_chunk: Dataset
    priority: int = 5  # 1-10, 10 being highest
    created_at: datetime = None
    assigned_worker: Optional[str] = None
    status: str = "pending"  # pending, assigned, running, completed, failed
    result: Optional[DetectionResult] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class DistributedProcessingManager:
    """Manager for distributed anomaly detection processing."""
    
    def __init__(self, 
                 max_workers: int = 10,
                 task_timeout: int = 300,
                 heartbeat_interval: int = 30):
        self.max_workers = max_workers
        self.task_timeout = task_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_locks: Dict[str, asyncio.Lock] = {}
        
        # Task management
        self.pending_tasks: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, DetectionTask] = {}
        self.completed_tasks: Dict[str, DetectionTask] = {}
        
        # Internal state
        self._running = False
        self._task_dispatcher: Optional[asyncio.Task] = None
        self._heartbeat_monitor: Optional[asyncio.Task] = None
        
        logger.info("Distributed processing manager initialized")
    
    async def start(self) -> None:
        """Start the distributed processing manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._task_dispatcher = asyncio.create_task(self._dispatch_tasks())
        self._heartbeat_monitor = asyncio.create_task(self._monitor_heartbeats())
        
        logger.info("Distributed processing manager started")
    
    async def stop(self) -> None:
        """Stop the distributed processing manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._task_dispatcher:
            self._task_dispatcher.cancel()
            try:
                await self._task_dispatcher
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_monitor:
            self._heartbeat_monitor.cancel()
            try:
                await self._heartbeat_monitor
            except asyncio.CancelledError:
                pass
        
        # Wait for active tasks to complete (with timeout)
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
            try:
                await asyncio.wait_for(self._wait_for_completion(), timeout=60)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks to complete")
        
        logger.info("Distributed processing manager stopped")
    
    async def register_worker(self, 
                            worker_id: str,
                            host: str,
                            port: int,
                            capacity: int,
                            capabilities: List[str] = None) -> bool:
        """Register a new worker node."""
        if worker_id in self.workers:
            logger.warning(f"Worker {worker_id} already registered")
            return False
        
        if len(self.workers) >= self.max_workers:
            logger.warning(f"Maximum number of workers ({self.max_workers}) reached")
            return False
        
        worker = WorkerNode(
            id=worker_id,
            host=host,
            port=port,
            capacity=capacity,
            capabilities=capabilities or [],
            last_heartbeat=datetime.now()
        )
        
        self.workers[worker_id] = worker
        self.worker_locks[worker_id] = asyncio.Lock()
        
        logger.info(f"Worker {worker_id} registered at {host}:{port} with capacity {capacity}")
        return True
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker node."""
        if worker_id not in self.workers:
            return False
        
        # Reassign active tasks from this worker
        tasks_to_reassign = [
            task for task in self.active_tasks.values()
            if task.assigned_worker == worker_id
        ]
        
        for task in tasks_to_reassign:
            await self._reassign_task(task)
        
        # Remove worker
        del self.workers[worker_id]
        del self.worker_locks[worker_id]
        
        logger.info(f"Worker {worker_id} unregistered")
        return True
    
    async def heartbeat(self, worker_id: str, status: str = "idle", current_load: int = 0) -> bool:
        """Process heartbeat from worker."""
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        worker.last_heartbeat = datetime.now()
        worker.status = status
        worker.current_load = current_load
        
        return True
    
    async def submit_detection_task(self,
                                  detector: Detector,
                                  dataset: Dataset,
                                  priority: int = 5) -> str:
        """Submit a detection task for distributed processing."""
        # Split dataset into chunks for parallel processing
        chunks = await self._split_dataset(dataset)
        
        task_ids = []
        for i, chunk in enumerate(chunks):
            task_id = f"{detector.id}_{dataset.id}_{i}_{uuid.uuid4().hex[:8]}"
            
            task = DetectionTask(
                id=task_id,
                detector_id=detector.id,
                dataset_chunk=chunk,
                priority=priority
            )
            
            await self.pending_tasks.put(task)
            task_ids.append(task_id)
        
        logger.info(f"Submitted {len(chunks)} detection tasks for dataset {dataset.id}")
        return task_ids[0]  # Return main task ID
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "id": task.id,
                "status": task.status,
                "assigned_worker": task.assigned_worker,
                "progress": "running",
                "created_at": task.created_at.isoformat(),
                "retry_count": task.retry_count
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                "id": task.id,
                "status": task.status,
                "result": task.result.id if task.result else None,
                "error": task.error,
                "created_at": task.created_at.isoformat(),
                "retry_count": task.retry_count
            }
        
        return None
    
    async def get_worker_status(self) -> Dict[str, Any]:
        """Get status of all workers."""
        workers_status = {}
        
        for worker_id, worker in self.workers.items():
            workers_status[worker_id] = {
                "host": worker.host,
                "port": worker.port,
                "status": worker.status,
                "capacity": worker.capacity,
                "current_load": worker.current_load,
                "load_percentage": worker.load_percentage,
                "is_available": worker.is_available,
                "last_heartbeat": worker.last_heartbeat.isoformat() if worker.last_heartbeat else None,
                "capabilities": worker.capabilities
            }
        
        return {
            "total_workers": len(self.workers),
            "available_workers": len([w for w in self.workers.values() if w.is_available]),
            "workers": workers_status
        }
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        return {
            "workers": await self.get_worker_status(),
            "tasks": {
                "pending": self.pending_tasks.qsize(),
                "active": len(self.active_tasks),
                "completed": len(self.completed_tasks)
            },
            "system": {
                "running": self._running,
                "uptime": "N/A"  # Could track uptime
            }
        }
    
    async def _dispatch_tasks(self) -> None:
        """Background task dispatcher."""
        while self._running:
            try:
                # Get next task (with timeout to allow periodic checks)
                try:
                    task = await asyncio.wait_for(self.pending_tasks.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Find available worker
                worker = await self._find_available_worker(task)
                if not worker:
                    # No available worker, put task back
                    await self.pending_tasks.put(task)
                    await asyncio.sleep(1)  # Wait before retrying
                    continue
                
                # Assign task to worker
                await self._assign_task(task, worker)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task dispatcher: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_heartbeats(self) -> None:
        """Monitor worker heartbeats."""
        while self._running:
            try:
                current_time = datetime.now()
                stale_workers = []
                
                for worker_id, worker in self.workers.items():
                    if (worker.last_heartbeat and 
                        current_time - worker.last_heartbeat > timedelta(minutes=2)):
                        stale_workers.append(worker_id)
                        worker.status = "offline"
                
                # Handle stale workers
                for worker_id in stale_workers:
                    logger.warning(f"Worker {worker_id} appears offline")
                    await self._handle_offline_worker(worker_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _find_available_worker(self, task: DetectionTask) -> Optional[WorkerNode]:
        """Find the best available worker for a task."""
        available_workers = [
            worker for worker in self.workers.values()
            if worker.is_available
        ]
        
        if not available_workers:
            return None
        
        # Sort by load percentage (least loaded first)
        available_workers.sort(key=lambda w: w.load_percentage)
        
        # TODO: Add capability matching
        return available_workers[0]
    
    async def _assign_task(self, task: DetectionTask, worker: WorkerNode) -> None:
        """Assign a task to a worker."""
        async with self.worker_locks[worker.id]:
            task.assigned_worker = worker.id
            task.status = "assigned"
            
            self.active_tasks[task.id] = task
            worker.current_load += 1
            
            # TODO: Send task to worker via API call
            logger.info(f"Task {task.id} assigned to worker {worker.id}")
            
            # Simulate task completion (replace with actual worker communication)
            asyncio.create_task(self._simulate_task_completion(task))
    
    async def _reassign_task(self, task: DetectionTask) -> None:
        """Reassign a task to a different worker."""
        if task.retry_count >= task.max_retries:
            task.status = "failed"
            task.error = "Maximum retries exceeded"
            await self._complete_task(task)
            return
        
        task.retry_count += 1
        task.assigned_worker = None
        task.status = "pending"
        
        # Remove from active tasks
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        # Put back in queue
        await self.pending_tasks.put(task)
        logger.info(f"Task {task.id} reassigned (retry {task.retry_count})")
    
    async def _complete_task(self, task: DetectionTask) -> None:
        """Mark task as completed."""
        # Remove from active tasks
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        # Add to completed tasks
        self.completed_tasks[task.id] = task
        
        # Update worker load
        if task.assigned_worker and task.assigned_worker in self.workers:
            worker = self.workers[task.assigned_worker]
            worker.current_load = max(0, worker.current_load - 1)
        
        logger.info(f"Task {task.id} completed with status {task.status}")
    
    async def _handle_offline_worker(self, worker_id: str) -> None:
        """Handle a worker going offline."""
        # Reassign all active tasks from this worker
        tasks_to_reassign = [
            task for task in self.active_tasks.values()
            if task.assigned_worker == worker_id
        ]
        
        for task in tasks_to_reassign:
            await self._reassign_task(task)
        
        logger.warning(f"Reassigned {len(tasks_to_reassign)} tasks from offline worker {worker_id}")
    
    async def _split_dataset(self, dataset: Dataset) -> List[Dataset]:
        """Split dataset into chunks for parallel processing."""
        # For now, return the original dataset
        # TODO: Implement intelligent dataset splitting
        return [dataset]
    
    async def _simulate_task_completion(self, task: DetectionTask) -> None:
        """Simulate task completion (for testing)."""
        # Wait for simulated processing time
        await asyncio.sleep(2)
        
        # Simulate successful completion
        task.status = "completed"
        # task.result = ... # Would be actual DetectionResult
        
        await self._complete_task(task)
    
    async def _wait_for_completion(self) -> None:
        """Wait for all active tasks to complete."""
        while self.active_tasks:
            await asyncio.sleep(0.1)