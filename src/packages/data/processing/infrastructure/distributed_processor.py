"""Distributed processing with Ray for scalable anomaly detection."""

from __future__ import annotations

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import numpy.typing as npt

try:
    import ray
    from ray import ObjectRef
    from ray.util.multiprocessing import Process
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    ObjectRef = None
    Process = None
    RAY_AVAILABLE = False

try:
    import dask
    from dask import delayed, compute
    from dask.distributed import Client, as_completed as dask_as_completed
    DASK_AVAILABLE = True
except ImportError:
    dask = None
    delayed = None
    compute = None
    Client = None
    dask_as_completed = None
    DASK_AVAILABLE = False

from ..adapters.algorithms.adapters.sklearn_adapter import SklearnAdapter
from ..adapters.algorithms.adapters.deeplearning_adapter import DeepLearningAdapter

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Represents a processing task."""
    task_id: str
    data: npt.NDArray[np.floating]
    algorithm: str
    parameters: Dict[str, Any]
    task_type: str  # 'training', 'inference', 'batch_inference'
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingResult:
    """Result of a processing task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    worker_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DistributedProcessor:
    """Distributed processing engine for anomaly detection."""
    
    def __init__(
        self,
        backend: str = "ray",
        num_workers: Optional[int] = None,
        ray_config: Optional[Dict[str, Any]] = None,
        dask_config: Optional[Dict[str, Any]] = None,
        enable_gpu: bool = False,
        batch_size: int = 1000,
        max_concurrent_tasks: int = 10
    ):
        """Initialize distributed processor.
        
        Args:
            backend: Processing backend ('ray', 'dask', 'threading')
            num_workers: Number of worker processes
            ray_config: Ray-specific configuration
            dask_config: Dask-specific configuration
            enable_gpu: Whether to enable GPU processing
            batch_size: Default batch size for processing
            max_concurrent_tasks: Maximum concurrent tasks
        """
        self.backend = backend
        self.num_workers = num_workers or self._get_default_workers()
        self.ray_config = ray_config or {}
        self.dask_config = dask_config or {}
        self.enable_gpu = enable_gpu
        self.batch_size = batch_size
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # State management
        self.is_initialized = False
        self.active_tasks: Dict[str, Any] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        
        # Performance tracking
        self.task_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0
        }
        
        # Initialize backend
        self._initialize_backend()
        
        logger.info(f"Distributed processor initialized with {self.backend} backend, "
                   f"{self.num_workers} workers")
    
    def _get_default_workers(self) -> int:
        """Get default number of workers."""
        import multiprocessing
        return min(multiprocessing.cpu_count(), 8)
    
    def _initialize_backend(self) -> None:
        """Initialize the processing backend."""
        try:
            if self.backend == "ray":
                self._initialize_ray()
            elif self.backend == "dask":
                self._initialize_dask()
            elif self.backend == "threading":
                self._initialize_threading()
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.backend} backend: {e}")
            raise
    
    def _initialize_ray(self) -> None:
        """Initialize Ray backend."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for Ray backend")
        
        if not ray.is_initialized():
            ray_config = {
                "num_cpus": self.num_workers,
                "ignore_reinit_error": True,
                **self.ray_config
            }
            
            if self.enable_gpu:
                ray_config["num_gpus"] = 1
            
            ray.init(**ray_config)
        
        # Register remote functions
        self._register_ray_functions()
        
        logger.info(f"Ray initialized with {ray.cluster_resources()}")
    
    def _initialize_dask(self) -> None:
        """Initialize Dask backend."""
        if not DASK_AVAILABLE:
            raise ImportError("Dask is required for Dask backend")
        
        dask_config = {
            "n_workers": self.num_workers,
            "threads_per_worker": 1,
            **self.dask_config
        }
        
        self.dask_client = Client(**dask_config)
        logger.info(f"Dask client initialized: {self.dask_client}")
    
    def _initialize_threading(self) -> None:
        """Initialize threading backend."""
        self.thread_executor = ThreadPoolExecutor(max_workers=self.num_workers)
        logger.info(f"Thread executor initialized with {self.num_workers} threads")
    
    def _register_ray_functions(self) -> None:
        """Register Ray remote functions."""
        @ray.remote
        def train_model_remote(algorithm: str, parameters: Dict[str, Any], 
                              data: npt.NDArray[np.floating]) -> Dict[str, Any]:
            """Remote function for model training."""
            try:
                start_time = time.time()
                
                # Create adapter
                if algorithm in ["iforest", "lof", "ocsvm", "pca"]:
                    adapter = SklearnAdapter(algorithm, **parameters)
                elif algorithm == "autoencoder":
                    adapter = DeepLearningAdapter(**parameters)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                # Train model
                adapter.fit(data)
                
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "adapter": adapter,
                    "processing_time": processing_time,
                    "data_shape": data.shape
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
        
        @ray.remote
        def inference_remote(adapter: Any, data: npt.NDArray[np.floating]) -> Dict[str, Any]:
            """Remote function for inference."""
            try:
                start_time = time.time()
                
                predictions = adapter.predict(data)
                scores = None
                
                if hasattr(adapter, 'decision_function'):
                    scores = adapter.decision_function(data)
                
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "predictions": predictions,
                    "scores": scores,
                    "processing_time": processing_time,
                    "data_shape": data.shape
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
        
        @ray.remote
        def batch_inference_remote(adapter: Any, data_batches: List[npt.NDArray[np.floating]]) -> Dict[str, Any]:
            """Remote function for batch inference."""
            try:
                start_time = time.time()
                
                all_predictions = []
                all_scores = []
                
                for batch in data_batches:
                    predictions = adapter.predict(batch)
                    all_predictions.append(predictions)
                    
                    if hasattr(adapter, 'decision_function'):
                        scores = adapter.decision_function(batch)
                        all_scores.append(scores)
                
                # Concatenate results
                final_predictions = np.concatenate(all_predictions)
                final_scores = np.concatenate(all_scores) if all_scores else None
                
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "predictions": final_predictions,
                    "scores": final_scores,
                    "processing_time": processing_time,
                    "total_samples": sum(len(batch) for batch in data_batches)
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
        
        # Store remote functions
        self.train_model_remote = train_model_remote
        self.inference_remote = inference_remote
        self.batch_inference_remote = batch_inference_remote
    
    def submit_task(self, task: ProcessingTask) -> str:
        """Submit a processing task.
        
        Args:
            task: Processing task to submit
            
        Returns:
            Task ID for tracking
        """
        if not self.is_initialized:
            raise RuntimeError("Processor not initialized")
        
        self.task_stats["total_tasks"] += 1
        
        if self.backend == "ray":
            future = self._submit_ray_task(task)
        elif self.backend == "dask":
            future = self._submit_dask_task(task)
        elif self.backend == "threading":
            future = self._submit_threading_task(task)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        self.active_tasks[task.task_id] = {
            "task": task,
            "future": future,
            "submitted_at": time.time()
        }
        
        logger.debug(f"Submitted task {task.task_id} ({task.task_type})")
        return task.task_id
    
    def _submit_ray_task(self, task: ProcessingTask) -> ObjectRef:
        """Submit task to Ray."""
        if task.task_type == "training":
            return self.train_model_remote.remote(
                task.algorithm, task.parameters, task.data
            )
        elif task.task_type == "inference":
            # Need to get trained model (this would be from model storage)
            adapter = task.metadata.get("adapter")
            if not adapter:
                raise ValueError("Adapter required for inference task")
            
            return self.inference_remote.remote(adapter, task.data)
        elif task.task_type == "batch_inference":
            # Split data into batches
            batches = self._split_into_batches(task.data, self.batch_size)
            adapter = task.metadata.get("adapter")
            
            return self.batch_inference_remote.remote(adapter, batches)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _submit_dask_task(self, task: ProcessingTask) -> Any:
        """Submit task to Dask."""
        @delayed
        def process_task(task_obj):
            return self._execute_task_sync(task_obj)
        
        return process_task(task)
    
    def _submit_threading_task(self, task: ProcessingTask) -> Any:
        """Submit task to thread executor."""
        return self.thread_executor.submit(self._execute_task_sync, task)
    
    def _execute_task_sync(self, task: ProcessingTask) -> Dict[str, Any]:
        """Execute task synchronously (for Dask/threading backends)."""
        try:
            start_time = time.time()
            
            if task.task_type == "training":
                result = self._train_model_sync(task.algorithm, task.parameters, task.data)
            elif task.task_type == "inference":
                adapter = task.metadata.get("adapter")
                result = self._inference_sync(adapter, task.data)
            elif task.task_type == "batch_inference":
                adapter = task.metadata.get("adapter")
                result = self._batch_inference_sync(adapter, task.data)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["success"] = True
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _train_model_sync(self, algorithm: str, parameters: Dict[str, Any], 
                         data: npt.NDArray[np.floating]) -> Dict[str, Any]:
        """Train model synchronously."""
        if algorithm in ["iforest", "lof", "ocsvm", "pca"]:
            adapter = SklearnAdapter(algorithm, **parameters)
        elif algorithm == "autoencoder":
            adapter = DeepLearningAdapter(**parameters)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        adapter.fit(data)
        
        return {
            "adapter": adapter,
            "data_shape": data.shape
        }
    
    def _inference_sync(self, adapter: Any, data: npt.NDArray[np.floating]) -> Dict[str, Any]:
        """Run inference synchronously."""
        predictions = adapter.predict(data)
        scores = None
        
        if hasattr(adapter, 'decision_function'):
            scores = adapter.decision_function(data)
        
        return {
            "predictions": predictions,
            "scores": scores,
            "data_shape": data.shape
        }
    
    def _batch_inference_sync(self, adapter: Any, data: npt.NDArray[np.floating]) -> Dict[str, Any]:
        """Run batch inference synchronously."""
        batches = self._split_into_batches(data, self.batch_size)
        
        all_predictions = []
        all_scores = []
        
        for batch in batches:
            predictions = adapter.predict(batch)
            all_predictions.append(predictions)
            
            if hasattr(adapter, 'decision_function'):
                scores = adapter.decision_function(batch)
                all_scores.append(scores)
        
        final_predictions = np.concatenate(all_predictions)
        final_scores = np.concatenate(all_scores) if all_scores else None
        
        return {
            "predictions": final_predictions,
            "scores": final_scores,
            "total_samples": len(data)
        }
    
    def _split_into_batches(self, data: npt.NDArray[np.floating], 
                           batch_size: int) -> List[npt.NDArray[np.floating]]:
        """Split data into batches."""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> ProcessingResult:
        """Get result of a submitted task.
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds
            
        Returns:
            Processing result
        """
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_info = self.active_tasks[task_id]
        future = task_info["future"]
        
        try:
            start_time = time.time()
            
            if self.backend == "ray":
                if timeout:
                    result_data = ray.get(future, timeout=timeout)
                else:
                    result_data = ray.get(future)
            elif self.backend == "dask":
                if timeout:
                    result_data = future.result(timeout=timeout)
                else:
                    result_data = future.compute()
            elif self.backend == "threading":
                result_data = future.result(timeout=timeout)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            # Create result object
            result = ProcessingResult(
                task_id=task_id,
                success=result_data.get("success", False),
                result=result_data,
                processing_time=result_data.get("processing_time", 0.0),
                worker_id=result_data.get("worker_id")
            )
            
            if not result.success:
                result.error = result_data.get("error", "Unknown error")
                self.task_stats["failed_tasks"] += 1
            else:
                self.task_stats["completed_tasks"] += 1
            
            # Update stats
            self.task_stats["total_processing_time"] += result.processing_time
            if self.task_stats["completed_tasks"] > 0:
                self.task_stats["avg_processing_time"] = (
                    self.task_stats["total_processing_time"] / 
                    self.task_stats["completed_tasks"]
                )
            
            # Move to completed tasks
            self.completed_tasks[task_id] = result
            del self.active_tasks[task_id]
            
            logger.debug(f"Task {task_id} completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_result = ProcessingResult(
                task_id=task_id,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
            
            self.task_stats["failed_tasks"] += 1
            self.completed_tasks[task_id] = error_result
            del self.active_tasks[task_id]
            
            logger.error(f"Task {task_id} failed: {e}")
            return error_result
    
    def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> List[ProcessingResult]:
        """Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task IDs
            timeout: Timeout in seconds
            
        Returns:
            List of processing results
        """
        results = []
        start_time = time.time()
        
        for task_id in task_ids:
            remaining_timeout = None
            if timeout:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
                
                if remaining_timeout <= 0:
                    break
            
            result = self.get_result(task_id, remaining_timeout)
            results.append(result)
        
        return results
    
    def submit_parallel_training(
        self,
        algorithms: List[str],
        parameter_sets: List[Dict[str, Any]],
        data: npt.NDArray[np.floating]
    ) -> List[str]:
        """Submit multiple training tasks in parallel.
        
        Args:
            algorithms: List of algorithms to train
            parameter_sets: Parameter sets for each algorithm
            data: Training data
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        for i, (algorithm, parameters) in enumerate(zip(algorithms, parameter_sets)):
            task = ProcessingTask(
                task_id=f"parallel_train_{i}_{int(time.time())}",
                data=data,
                algorithm=algorithm,
                parameters=parameters,
                task_type="training"
            )
            
            task_id = self.submit_task(task)
            task_ids.append(task_id)
        
        logger.info(f"Submitted {len(task_ids)} parallel training tasks")
        return task_ids
    
    def submit_distributed_inference(
        self,
        adapter: Any,
        data: npt.NDArray[np.floating],
        num_partitions: Optional[int] = None
    ) -> List[str]:
        """Submit distributed inference tasks.
        
        Args:
            adapter: Trained model adapter
            data: Data for inference
            num_partitions: Number of partitions to split data into
            
        Returns:
            List of task IDs
        """
        if num_partitions is None:
            num_partitions = min(self.num_workers, len(data) // self.batch_size + 1)
        
        # Split data into partitions
        partition_size = len(data) // num_partitions
        partitions = []
        
        for i in range(num_partitions):
            start_idx = i * partition_size
            if i == num_partitions - 1:
                end_idx = len(data)  # Last partition gets remaining data
            else:
                end_idx = start_idx + partition_size
            
            partitions.append(data[start_idx:end_idx])
        
        # Submit tasks
        task_ids = []
        for i, partition in enumerate(partitions):
            task = ProcessingTask(
                task_id=f"distributed_inference_{i}_{int(time.time())}",
                data=partition,
                algorithm="",  # Not needed for inference
                parameters={},
                task_type="inference",
                metadata={"adapter": adapter}
            )
            
            task_id = self.submit_task(task)
            task_ids.append(task_id)
        
        logger.info(f"Submitted {len(task_ids)} distributed inference tasks")
        return task_ids
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status information."""
        status = {
            "backend": self.backend,
            "is_initialized": self.is_initialized,
            "num_workers": self.num_workers,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "task_stats": self.task_stats.copy()
        }
        
        if self.backend == "ray" and ray.is_initialized():
            status["ray_cluster_resources"] = ray.cluster_resources()
            status["ray_available_resources"] = ray.available_resources()
        elif self.backend == "dask" and hasattr(self, 'dask_client'):
            status["dask_scheduler_info"] = self.dask_client.scheduler_info()
        
        return status
    
    def scale_cluster(self, num_workers: int) -> bool:
        """Scale the cluster to a different number of workers.
        
        Args:
            num_workers: New number of workers
            
        Returns:
            True if scaling succeeded
        """
        try:
            if self.backend == "ray":
                # Ray auto-scaling would need to be configured separately
                logger.warning("Ray auto-scaling not implemented in this basic version")
                return False
            elif self.backend == "dask" and hasattr(self, 'dask_client'):
                self.dask_client.restart()
                # Would need to reconfigure cluster
                return True
            elif self.backend == "threading":
                # Recreate thread executor
                self.thread_executor.shutdown(wait=True)
                self.thread_executor = ThreadPoolExecutor(max_workers=num_workers)
                self.num_workers = num_workers
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to scale cluster: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the distributed processor."""
        try:
            # Wait for active tasks to complete
            if self.active_tasks:
                active_task_ids = list(self.active_tasks.keys())
                logger.info(f"Waiting for {len(active_task_ids)} active tasks to complete...")
                self.wait_for_tasks(active_task_ids, timeout=30.0)
            
            # Shutdown backend
            if self.backend == "ray" and ray.is_initialized():
                ray.shutdown()
            elif self.backend == "dask" and hasattr(self, 'dask_client'):
                self.dask_client.close()
            elif self.backend == "threading" and hasattr(self, 'thread_executor'):
                self.thread_executor.shutdown(wait=True)
            
            self.is_initialized = False
            logger.info("Distributed processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()