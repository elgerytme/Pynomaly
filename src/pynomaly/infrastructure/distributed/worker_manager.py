"""Worker node management for distributed processing."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, Future

from pydantic import BaseModel, Field

from .distributed_config import get_distributed_config_manager, WorkerConfig
from .task_distributor import DistributedTask, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


class WorkerStatus(str, Enum):
    """Worker node status."""
    STARTING = "starting"
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    ERROR = "error"
    OFFLINE = "offline"
    SHUTTING_DOWN = "shutting_down"


class HealthStatus(str, Enum):
    """Worker health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class WorkerCapabilities:
    """Worker node capabilities."""
    
    # Supported algorithms
    supported_algorithms: Set[str] = field(default_factory=set)
    
    # Hardware capabilities
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    
    # Software capabilities
    python_version: str = ""
    installed_packages: Set[str] = field(default_factory=set)
    supported_data_formats: Set[str] = field(default_factory=set)
    
    # Performance characteristics
    max_concurrent_tasks: int = 1
    preferred_batch_size: int = 1000
    network_bandwidth_mbps: float = 100.0
    
    # Specialized features
    supports_streaming: bool = False
    supports_incremental_learning: bool = False
    supports_distributed_training: bool = False


@dataclass
class WorkerMetrics:
    """Worker performance and health metrics."""
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_used_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    
    # Network metrics
    network_in_mbps: float = 0.0
    network_out_mbps: float = 0.0
    
    # Task metrics
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_duration: float = 0.0
    
    # Health metrics
    uptime_seconds: float = 0.0
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_count: int = 0
    
    # Performance metrics
    throughput_tasks_per_minute: float = 0.0
    latency_ms: float = 0.0
    
    @property
    def health_status(self) -> HealthStatus:
        """Calculate overall health status."""
        if self.error_count > 10:
            return HealthStatus.CRITICAL
        elif self.cpu_usage_percent > 90 or self.memory_usage_percent > 90:
            return HealthStatus.CRITICAL
        elif self.cpu_usage_percent > 75 or self.memory_usage_percent > 75:
            return HealthStatus.WARNING
        elif (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds() > 300:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY


class WorkerNode(BaseModel):
    """Represents a worker node in the distributed system."""
    
    # Basic information
    worker_id: str = Field(..., description="Unique worker identifier")
    host: str = Field(..., description="Worker host address")
    port: int = Field(..., description="Worker port")
    
    # Status
    status: WorkerStatus = Field(default=WorkerStatus.STARTING, description="Current worker status")
    
    # Capabilities and configuration
    capabilities: WorkerCapabilities = Field(default_factory=WorkerCapabilities, description="Worker capabilities")
    config: WorkerConfig = Field(default_factory=WorkerConfig, description="Worker configuration")
    
    # Runtime information
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Start time")
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last heartbeat")
    
    # Metrics
    metrics: WorkerMetrics = Field(default_factory=WorkerMetrics, description="Worker metrics")
    
    # Current workload
    active_tasks: Dict[str, DistributedTask] = Field(default_factory=dict, description="Currently executing tasks")
    task_queue: List[str] = Field(default_factory=list, description="Queued task IDs")
    
    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        return self.metrics.health_status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
    
    @property
    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        return (self.status in [WorkerStatus.IDLE, WorkerStatus.BUSY] and 
                self.is_healthy and 
                len(self.active_tasks) < self.capabilities.max_concurrent_tasks)
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0)."""
        if self.capabilities.max_concurrent_tasks == 0:
            return 1.0
        
        task_load = len(self.active_tasks) / self.capabilities.max_concurrent_tasks
        cpu_load = self.metrics.cpu_usage_percent / 100.0
        memory_load = self.metrics.memory_usage_percent / 100.0
        
        # Return weighted average of different load factors
        return (0.5 * task_load + 0.3 * cpu_load + 0.2 * memory_load)


class WorkerTaskExecutor:
    """Executes tasks on a worker node."""
    
    def __init__(self, worker_id: str, max_workers: int = 4):
        """Initialize task executor.
        
        Args:
            worker_id: Worker identifier
            max_workers: Maximum concurrent threads
        """
        self.worker_id = worker_id
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_futures: Dict[str, Future] = {}
        self.task_functions: Dict[str, Callable] = {}
        
        # Register built-in task functions
        self._register_builtin_functions()
    
    def _register_builtin_functions(self) -> None:
        """Register built-in task functions."""
        self.task_functions.update({
            "anomaly_detection": self._execute_anomaly_detection,
            "model_training": self._execute_model_training,
            "data_preprocessing": self._execute_data_preprocessing,
            "feature_extraction": self._execute_feature_extraction,
            "hyperparameter_tuning": self._execute_hyperparameter_tuning,
            "model_evaluation": self._execute_model_evaluation,
            "batch_prediction": self._execute_batch_prediction,
            "ensemble_aggregation": self._execute_ensemble_aggregation
        })
    
    async def execute_task(self, task: DistributedTask) -> TaskResult:
        """Execute a distributed task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get task function
            func = self.task_functions.get(task.function_name)
            if not func:
                raise ValueError(f"Unknown function: {task.function_name}")
            
            # Execute task in thread pool
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                self.executor,
                func,
                task.arguments,
                task.kwargs,
                task.context
            )
            
            self.active_futures[task.task_id] = future
            result_data = await future
            
            # Create successful result
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()
            
            return TaskResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status=TaskStatus.COMPLETED,
                result=result_data,
                started_at=start_time,
                completed_at=end_time,
                execution_time_seconds=execution_time,
                memory_used_mb=self._get_memory_usage(),
                cpu_time_seconds=execution_time * 0.8  # Estimate
            )
            
        except Exception as e:
            # Create failed result
            end_time = datetime.now(timezone.utc)
            
            return TaskResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status=TaskStatus.FAILED,
                error=str(e),
                traceback=str(e.__traceback__) if hasattr(e, '__traceback__') else None,
                started_at=start_time,
                completed_at=end_time,
                execution_time_seconds=(end_time - start_time).total_seconds()
            )
        
        finally:
            self.active_futures.pop(task.task_id, None)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled
        """
        future = self.active_futures.get(task_id)
        if future and not future.done():
            return future.cancel()
        return False
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    # Task execution functions (placeholders for actual implementations)
    
    def _execute_anomaly_detection(self, args: Dict[str, Any], kwargs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute anomaly detection task."""
        # Placeholder implementation
        time.sleep(1.0)  # Simulate processing
        return {
            "anomalies_detected": 5,
            "anomaly_scores": [0.8, 0.7, 0.9, 0.6, 0.85],
            "processing_time": 1.0
        }
    
    def _execute_model_training(self, args: Dict[str, Any], kwargs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training task."""
        # Placeholder implementation
        time.sleep(2.0)  # Simulate training
        return {
            "model_trained": True,
            "training_accuracy": 0.95,
            "training_time": 2.0
        }
    
    def _execute_data_preprocessing(self, args: Dict[str, Any], kwargs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data preprocessing task."""
        # Placeholder implementation
        time.sleep(0.5)  # Simulate preprocessing
        return {
            "rows_processed": 10000,
            "features_extracted": 50,
            "processing_time": 0.5
        }
    
    def _execute_feature_extraction(self, args: Dict[str, Any], kwargs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature extraction task."""
        # Placeholder implementation
        time.sleep(1.5)  # Simulate feature extraction
        return {
            "features_extracted": 25,
            "feature_importance": [0.1, 0.2, 0.15, 0.3, 0.25],
            "processing_time": 1.5
        }
    
    def _execute_hyperparameter_tuning(self, args: Dict[str, Any], kwargs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hyperparameter tuning task."""
        # Placeholder implementation
        time.sleep(3.0)  # Simulate tuning
        return {
            "best_params": {"C": 1.0, "gamma": 0.1},
            "best_score": 0.92,
            "trials_completed": 50
        }
    
    def _execute_model_evaluation(self, args: Dict[str, Any], kwargs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model evaluation task."""
        # Placeholder implementation
        time.sleep(1.0)  # Simulate evaluation
        return {
            "accuracy": 0.94,
            "precision": 0.91,
            "recall": 0.89,
            "f1_score": 0.90
        }
    
    def _execute_batch_prediction(self, args: Dict[str, Any], kwargs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute batch prediction task."""
        # Placeholder implementation
        time.sleep(0.8)  # Simulate prediction
        return {
            "predictions": [0, 1, 0, 1, 0] * 1000,  # 5000 predictions
            "prediction_probabilities": [0.1, 0.9, 0.2, 0.8, 0.15] * 1000,
            "samples_processed": 5000
        }
    
    def _execute_ensemble_aggregation(self, args: Dict[str, Any], kwargs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ensemble aggregation task."""
        # Placeholder implementation
        time.sleep(0.3)  # Simulate aggregation
        return {
            "ensemble_predictions": [0, 1, 0, 1, 0] * 500,
            "confidence_scores": [0.85, 0.92, 0.78, 0.89, 0.81] * 500,
            "models_aggregated": 5
        }


class WorkerManager:
    """Manages worker nodes in the distributed system."""
    
    def __init__(self, config: Optional[WorkerConfig] = None):
        """Initialize worker manager.
        
        Args:
            config: Worker configuration
        """
        self.config = config or get_distributed_config_manager().get_effective_config().worker
        
        # Worker registry
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_executors: Dict[str, WorkerTaskExecutor] = {}
        
        # Monitoring
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the worker manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Worker manager started")
    
    async def stop(self) -> None:
        """Stop the worker manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        for task in [self._monitoring_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown all workers
        for worker_id in list(self.workers.keys()):
            await self.remove_worker(worker_id)
        
        logger.info("Worker manager stopped")
    
    async def register_worker(self, worker: WorkerNode) -> bool:
        """Register a new worker node.
        
        Args:
            worker: Worker node to register
            
        Returns:
            True if worker was registered successfully
        """
        try:
            # Initialize worker capabilities if not set
            if not worker.capabilities.cpu_cores:
                worker.capabilities = await self._detect_worker_capabilities()
            
            # Create task executor
            executor = WorkerTaskExecutor(
                worker.worker_id,
                max_workers=worker.capabilities.max_concurrent_tasks
            )
            
            # Register worker
            self.workers[worker.worker_id] = worker
            self.worker_executors[worker.worker_id] = executor
            
            # Update worker status
            worker.status = WorkerStatus.IDLE
            worker.started_at = datetime.now(timezone.utc)
            worker.last_seen = datetime.now(timezone.utc)
            
            logger.info(f"Worker {worker.worker_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register worker {worker.worker_id}: {e}")
            return False
    
    async def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker node.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            True if worker was removed
        """
        if worker_id not in self.workers:
            return False
        
        try:
            worker = self.workers[worker_id]
            worker.status = WorkerStatus.SHUTTING_DOWN
            
            # Cancel all active tasks
            executor = self.worker_executors.get(worker_id)
            if executor:
                for task_id in list(worker.active_tasks.keys()):
                    executor.cancel_task(task_id)
            
            # Remove from registry
            del self.workers[worker_id]
            if worker_id in self.worker_executors:
                del self.worker_executors[worker_id]
            
            logger.info(f"Worker {worker_id} removed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove worker {worker_id}: {e}")
            return False
    
    async def assign_task(self, worker_id: str, task: DistributedTask) -> bool:
        """Assign a task to a specific worker.
        
        Args:
            worker_id: Target worker ID
            task: Task to assign
            
        Returns:
            True if task was assigned successfully
        """
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        
        # Check if worker can accept the task
        if not worker.is_available:
            return False
        
        try:
            # Add task to worker
            worker.active_tasks[task.task_id] = task
            worker.status = WorkerStatus.BUSY
            
            # Execute task
            executor = self.worker_executors[worker_id]
            result = await executor.execute_task(task)
            
            # Remove task from worker
            worker.active_tasks.pop(task.task_id, None)
            
            # Update worker status
            if not worker.active_tasks:
                worker.status = WorkerStatus.IDLE
            
            # Update metrics
            worker.metrics.completed_tasks += 1
            if not result.is_successful:
                worker.metrics.failed_tasks += 1
            
            logger.info(f"Task {task.task_id} completed on worker {worker_id}")
            return True
            
        except Exception as e:
            # Remove task and update error metrics
            worker.active_tasks.pop(task.task_id, None)
            worker.metrics.failed_tasks += 1
            worker.metrics.error_count += 1
            
            if not worker.active_tasks:
                worker.status = WorkerStatus.ERROR
            
            logger.error(f"Task {task.task_id} failed on worker {worker_id}: {e}")
            return False
    
    def get_available_workers(self) -> List[WorkerNode]:
        """Get list of available workers.
        
        Returns:
            List of available workers
        """
        return [worker for worker in self.workers.values() if worker.is_available]
    
    def get_worker_by_capability(self, required_capabilities: Set[str]) -> List[WorkerNode]:
        """Get workers that have required capabilities.
        
        Args:
            required_capabilities: Required capabilities
            
        Returns:
            List of suitable workers
        """
        suitable_workers = []
        
        for worker in self.workers.values():
            if worker.is_available:
                worker_caps = worker.capabilities.supported_algorithms
                if required_capabilities.issubset(worker_caps) or "all" in worker_caps:
                    suitable_workers.append(worker)
        
        return suitable_workers
    
    def get_worker_statistics(self) -> Dict[str, Any]:
        """Get worker statistics.
        
        Returns:
            Worker statistics
        """
        total_workers = len(self.workers)
        if total_workers == 0:
            return {
                "total_workers": 0,
                "available_workers": 0,
                "busy_workers": 0,
                "offline_workers": 0,
                "average_load": 0.0,
                "total_tasks": 0,
                "total_completed": 0,
                "total_failed": 0
            }
        
        status_counts = defaultdict(int)
        total_load = 0.0
        total_tasks = 0
        total_completed = 0
        total_failed = 0
        
        for worker in self.workers.values():
            status_counts[worker.status] += 1
            total_load += worker.load_factor
            total_tasks += len(worker.active_tasks)
            total_completed += worker.metrics.completed_tasks
            total_failed += worker.metrics.failed_tasks
        
        return {
            "total_workers": total_workers,
            "available_workers": status_counts[WorkerStatus.IDLE],
            "busy_workers": status_counts[WorkerStatus.BUSY],
            "offline_workers": status_counts[WorkerStatus.OFFLINE],
            "error_workers": status_counts[WorkerStatus.ERROR],
            "average_load": total_load / total_workers,
            "total_active_tasks": total_tasks,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "status_distribution": dict(status_counts)
        }
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self._update_worker_metrics()
                await asyncio.sleep(self.config.metrics_reporting_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                await self._cleanup_offline_workers()
                await asyncio.sleep(60.0)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _update_worker_metrics(self) -> None:
        """Update metrics for all workers."""
        current_time = datetime.now(timezone.utc)
        
        for worker in self.workers.values():
            try:
                # Update basic metrics
                worker.metrics.active_tasks = len(worker.active_tasks)
                worker.metrics.uptime_seconds = (current_time - worker.started_at).total_seconds()
                
                # Get system metrics if this is the local worker
                if worker.host in ["localhost", "127.0.0.1"]:
                    await self._update_local_worker_metrics(worker)
                
                # Store metrics history
                self.metrics_history[worker.worker_id].append({
                    "timestamp": current_time,
                    "cpu_usage": worker.metrics.cpu_usage_percent,
                    "memory_usage": worker.metrics.memory_usage_percent,
                    "active_tasks": worker.metrics.active_tasks,
                    "load_factor": worker.load_factor
                })
                
            except Exception as e:
                logger.error(f"Error updating metrics for worker {worker.worker_id}: {e}")
                worker.metrics.error_count += 1
    
    async def _update_local_worker_metrics(self, worker: WorkerNode) -> None:
        """Update metrics for local worker using psutil."""
        try:
            # CPU and memory usage
            worker.metrics.cpu_usage_percent = psutil.cpu_percent(interval=None)
            
            memory = psutil.virtual_memory()
            worker.metrics.memory_usage_percent = memory.percent
            worker.metrics.memory_used_mb = memory.used / (1024 * 1024)
            
            # Network usage (simplified)
            net_io = psutil.net_io_counters()
            if hasattr(worker.metrics, '_last_net_io'):
                time_delta = 1.0  # Assume 1 second between updates
                bytes_in_delta = net_io.bytes_recv - worker.metrics._last_net_io.bytes_recv
                bytes_out_delta = net_io.bytes_sent - worker.metrics._last_net_io.bytes_sent
                
                worker.metrics.network_in_mbps = (bytes_in_delta / time_delta) / (1024 * 1024)
                worker.metrics.network_out_mbps = (bytes_out_delta / time_delta) / (1024 * 1024)
            
            worker.metrics._last_net_io = net_io
            
        except Exception as e:
            logger.warning(f"Could not update local metrics: {e}")
    
    async def _cleanup_offline_workers(self) -> None:
        """Remove workers that haven't been seen recently."""
        current_time = datetime.now(timezone.utc)
        offline_threshold = 300.0  # 5 minutes
        
        offline_workers = []
        for worker_id, worker in self.workers.items():
            time_since_seen = (current_time - worker.last_seen).total_seconds()
            if time_since_seen > offline_threshold:
                offline_workers.append(worker_id)
                worker.status = WorkerStatus.OFFLINE
        
        for worker_id in offline_workers:
            logger.warning(f"Worker {worker_id} marked as offline")
            # Optionally remove offline workers
            # await self.remove_worker(worker_id)
    
    async def _detect_worker_capabilities(self) -> WorkerCapabilities:
        """Detect worker capabilities automatically."""
        capabilities = WorkerCapabilities()
        
        try:
            # Hardware detection
            capabilities.cpu_cores = psutil.cpu_count(logical=True)
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            capabilities.memory_gb = round(memory_gb, 2)
            
            # Python version
            capabilities.python_version = platform.python_version()
            
            # Detect installed packages (simplified)
            try:
                import pkg_resources
                installed_packages = {pkg.project_name for pkg in pkg_resources.working_set}
                capabilities.installed_packages = installed_packages
            except:
                pass
            
            # Set reasonable defaults
            capabilities.max_concurrent_tasks = min(capabilities.cpu_cores, 8)
            capabilities.preferred_batch_size = 1000
            capabilities.supported_data_formats = {"csv", "json", "parquet"}
            capabilities.supported_algorithms = {"all"}  # Support all algorithms by default
            
        except Exception as e:
            logger.warning(f"Could not detect all capabilities: {e}")
        
        return capabilities