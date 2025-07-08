"""Async processing optimizations for improved performance."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
from uuid import UUID

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class TaskMetrics:
    """Metrics for async task execution."""
    
    task_id: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None
    memory_usage: Optional[float] = None
    
    def complete(self, error: Optional[Exception] = None):
        """Mark task as completed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if error:
            self.status = "failed"
            self.error = str(error)
        else:
            self.status = "completed"


class AsyncTaskPool:
    """Pool for managing async tasks with concurrency control."""
    
    def __init__(self, max_concurrent_tasks: int = 10, max_workers: int = 4):
        """Initialize async task pool."""
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, TaskMetrics] = {}
        self._results: Dict[str, Any] = {}
    
    async def submit_async_task(
        self, 
        task_id: str, 
        coro: Awaitable[T]
    ) -> T:
        """Submit async coroutine task."""
        async with self._semaphore:
            metrics = TaskMetrics(task_id=task_id, start_time=time.time())
            self._tasks[task_id] = metrics
            metrics.status = "running"
            
            try:
                result = await coro
                metrics.complete()
                self._results[task_id] = result
                return result
            except Exception as e:
                metrics.complete(e)
                logger.error(f"Async task {task_id} failed: {e}")
                raise
    
    async def submit_cpu_task(
        self, 
        task_id: str, 
        func: Callable[..., T], 
        *args, **kwargs
    ) -> T:
        """Submit CPU-bound task to thread pool."""
        async with self._semaphore:
            metrics = TaskMetrics(task_id=task_id, start_time=time.time())
            self._tasks[task_id] = metrics
            metrics.status = "running"
            
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor, func, *args, **kwargs
                )
                metrics.complete()
                self._results[task_id] = result
                return result
            except Exception as e:
                metrics.complete(e)
                logger.error(f"CPU task {task_id} failed: {e}")
                raise
    
    async def submit_batch_tasks(
        self, 
        tasks: List[tuple[str, Awaitable[T]]]
    ) -> Dict[str, T]:
        """Submit multiple tasks and wait for all to complete."""
        async_tasks = [
            self.submit_async_task(task_id, coro) 
            for task_id, coro in tasks
        ]
        
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        return {
            tasks[i][0]: result 
            for i, result in enumerate(results)
            if not isinstance(result, Exception)
        }
    
    def get_task_metrics(self, task_id: str) -> Optional[TaskMetrics]:
        """Get metrics for a specific task."""
        return self._tasks.get(task_id)
    
    def get_all_metrics(self) -> Dict[str, TaskMetrics]:
        """Get metrics for all tasks."""
        return self._tasks.copy()
    
    def get_result(self, task_id: str) -> Optional[Any]:
        """Get result for a completed task."""
        return self._results.get(task_id)
    
    def clear_completed_tasks(self):
        """Clear metrics and results for completed tasks."""
        completed_tasks = [
            task_id for task_id, metrics in self._tasks.items()
            if metrics.status in ("completed", "failed")
        ]
        
        for task_id in completed_tasks:
            self._tasks.pop(task_id, None)
            self._results.pop(task_id, None)
    
    async def shutdown(self):
        """Shutdown the task pool."""
        self._executor.shutdown(wait=True)


class DataProcessor:
    """Optimized async data processing utilities."""
    
    def __init__(self, task_pool: AsyncTaskPool):
        """Initialize with task pool."""
        self.task_pool = task_pool
    
    async def process_dataframe_parallel(
        self, 
        df: pd.DataFrame, 
        processor_func: Callable[[pd.DataFrame], Any],
        chunk_size: int = 1000,
        max_parallel: int = 4
    ) -> List[Any]:
        """Process DataFrame in parallel chunks."""
        # Split DataFrame into chunks
        chunks = [
            df.iloc[i:i + chunk_size] 
            for i in range(0, len(df), chunk_size)
        ]
        
        # Create async tasks for chunks
        tasks = []
        for i, chunk in enumerate(chunks[:max_parallel]):
            task_id = f"chunk_{i}"
            task = self.task_pool.submit_cpu_task(
                task_id, processor_func, chunk
            )
            tasks.append((task_id, task))
        
        # Process remaining chunks in batches
        all_results = []
        for batch_start in range(0, len(chunks), max_parallel):
            batch_chunks = chunks[batch_start:batch_start + max_parallel]
            batch_tasks = [
                (f"batch_{batch_start}_{i}", self.task_pool.submit_cpu_task(
                    f"batch_{batch_start}_{i}", processor_func, chunk
                ))
                for i, chunk in enumerate(batch_chunks)
            ]
            
            batch_results = await self.task_pool.submit_batch_tasks(batch_tasks)
            all_results.extend(batch_results.values())
        
        return all_results
    
    async def process_multiple_datasets(
        self, 
        datasets: Dict[str, pd.DataFrame],
        processor_func: Callable[[pd.DataFrame], Any]
    ) -> Dict[str, Any]:
        """Process multiple datasets in parallel."""
        tasks = [
            (dataset_id, self.task_pool.submit_cpu_task(
                f"dataset_{dataset_id}", processor_func, df
            ))
            for dataset_id, df in datasets.items()
        ]
        
        return await self.task_pool.submit_batch_tasks(tasks)
    
    async def map_reduce_async(
        self, 
        data: List[T],
        map_func: Callable[[T], R],
        reduce_func: Callable[[List[R]], Any],
        chunk_size: int = 100
    ) -> Any:
        """Async map-reduce operation."""
        # Map phase - process in chunks
        chunks = [
            data[i:i + chunk_size] 
            for i in range(0, len(data), chunk_size)
        ]
        
        map_tasks = [
            (f"map_{i}", self.task_pool.submit_cpu_task(
                f"map_{i}", lambda chunk: [map_func(item) for item in chunk], chunk
            ))
            for i, chunk in enumerate(chunks)
        ]
        
        map_results = await self.task_pool.submit_batch_tasks(map_tasks)
        
        # Flatten map results
        all_mapped = []
        for result in map_results.values():
            all_mapped.extend(result)
        
        # Reduce phase
        return await self.task_pool.submit_cpu_task(
            "reduce", reduce_func, all_mapped
        )


class DetectionOptimizer:
    """Optimized async anomaly detection processing."""
    
    def __init__(self, task_pool: AsyncTaskPool):
        """Initialize with task pool."""
        self.task_pool = task_pool
    
    async def detect_anomalies_batch(
        self, 
        detector_configs: List[Dict[str, Any]],
        dataset: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run multiple detectors on dataset in parallel."""
        tasks = []
        
        for i, config in enumerate(detector_configs):
            task_id = f"detector_{config.get('algorithm', i)}"
            task = self.task_pool.submit_cpu_task(
                task_id, self._run_single_detector, config, dataset
            )
            tasks.append((task_id, task))
        
        return await self.task_pool.submit_batch_tasks(tasks)
    
    async def detect_anomalies_ensemble(
        self, 
        detectors: List[Any],
        dataset: pd.DataFrame,
        voting_strategy: str = "majority"
    ) -> Dict[str, Any]:
        """Run ensemble detection in parallel."""
        # Run all detectors in parallel
        detection_tasks = [
            (f"detector_{i}", self.task_pool.submit_cpu_task(
                f"detector_{i}", self._run_detector_predict, detector, dataset
            ))
            for i, detector in enumerate(detectors)
        ]
        
        detection_results = await self.task_pool.submit_batch_tasks(detection_tasks)
        
        # Combine results using voting strategy
        return await self.task_pool.submit_cpu_task(
            "ensemble_voting", self._combine_predictions, 
            list(detection_results.values()), voting_strategy
        )
    
    async def cross_validate_async(
        self, 
        detector_configs: List[Dict[str, Any]],
        dataset: pd.DataFrame,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Perform async cross-validation for multiple detectors."""
        from sklearn.model_selection import KFold
        
        # Create CV folds
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        folds = list(kf.split(dataset))
        
        # Run CV for each detector in parallel
        cv_tasks = []
        for i, config in enumerate(detector_configs):
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                task_id = f"cv_{config.get('algorithm', i)}_fold_{fold_idx}"
                task = self.task_pool.submit_cpu_task(
                    task_id, self._run_cv_fold, 
                    config, dataset.iloc[train_idx], dataset.iloc[test_idx]
                )
                cv_tasks.append((task_id, task))
        
        cv_results = await self.task_pool.submit_batch_tasks(cv_tasks)
        
        # Aggregate CV results
        return await self.task_pool.submit_cpu_task(
            "cv_aggregate", self._aggregate_cv_results, cv_results, detector_configs
        )
    
    def _run_single_detector(self, config: Dict[str, Any], dataset: pd.DataFrame) -> Dict[str, Any]:
        """Run single detector (CPU-bound operation)."""
        # Placeholder implementation - replace with actual detection logic
        algorithm = config.get('algorithm', 'IsolationForest')
        parameters = config.get('parameters', {})
        
        # Simulate detection processing
        time.sleep(0.1)  # Simulate computation time
        
        # Return mock results
        n_samples = len(dataset)
        anomaly_scores = np.random.random(n_samples)
        predictions = (anomaly_scores > 0.5).astype(int)
        
        return {
            'algorithm': algorithm,
            'parameters': parameters,
            'scores': anomaly_scores.tolist(),
            'predictions': predictions.tolist(),
            'n_anomalies': int(predictions.sum()),
            'anomaly_rate': float(predictions.mean())
        }
    
    def _run_detector_predict(self, detector: Any, dataset: pd.DataFrame) -> np.ndarray:
        """Run detector prediction (CPU-bound operation)."""
        # Placeholder - replace with actual detector prediction
        return np.random.random(len(dataset))
    
    def _combine_predictions(
        self, 
        predictions: List[np.ndarray], 
        voting_strategy: str
    ) -> Dict[str, Any]:
        """Combine predictions from multiple detectors."""
        if not predictions:
            return {}
        
        stacked_predictions = np.stack(predictions)
        
        if voting_strategy == "majority":
            combined = (stacked_predictions > 0.5).sum(axis=0) > len(predictions) // 2
        elif voting_strategy == "average":
            combined = stacked_predictions.mean(axis=0)
        elif voting_strategy == "max":
            combined = stacked_predictions.max(axis=0)
        else:
            combined = stacked_predictions.mean(axis=0)
        
        return {
            'ensemble_scores': combined.tolist(),
            'individual_scores': [pred.tolist() for pred in predictions],
            'voting_strategy': voting_strategy,
            'n_detectors': len(predictions)
        }
    
    def _run_cv_fold(
        self, 
        config: Dict[str, Any], 
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run single CV fold (CPU-bound operation)."""
        # Placeholder implementation
        time.sleep(0.05)  # Simulate training time
        
        # Mock CV results
        return {
            'algorithm': config.get('algorithm'),
            'train_size': len(train_data),
            'test_size': len(test_data),
            'accuracy': np.random.random(),
            'precision': np.random.random(),
            'recall': np.random.random(),
            'f1_score': np.random.random()
        }
    
    def _aggregate_cv_results(
        self, 
        cv_results: Dict[str, Any], 
        detector_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        # Group results by algorithm
        algorithm_results = {}
        
        for result in cv_results.values():
            algorithm = result['algorithm']
            if algorithm not in algorithm_results:
                algorithm_results[algorithm] = []
            algorithm_results[algorithm].append(result)
        
        # Calculate mean and std for each algorithm
        aggregated = {}
        for algorithm, results in algorithm_results.items():
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            aggregated[algorithm] = {
                metric: {
                    'mean': np.mean([r[metric] for r in results]),
                    'std': np.std([r[metric] for r in results]),
                    'values': [r[metric] for r in results]
                }
                for metric in metrics
            }
        
        return aggregated


class AsyncStreamProcessor:
    """Process streaming data with async optimizations."""
    
    def __init__(self, buffer_size: int = 1000):
        """Initialize stream processor."""
        self.buffer_size = buffer_size
        self._buffer = []
        self._lock = asyncio.Lock()
        self._processing = False
    
    async def add_data(self, data: Any):
        """Add data to processing buffer."""
        async with self._lock:
            self._buffer.append(data)
            
            if len(self._buffer) >= self.buffer_size:
                await self._process_buffer()
    
    async def process_remaining(self):
        """Process any remaining data in buffer."""
        async with self._lock:
            if self._buffer:
                await self._process_buffer()
    
    async def _process_buffer(self):
        """Process current buffer contents."""
        if self._processing:
            return
        
        self._processing = True
        
        try:
            # Process buffer contents
            buffer_copy = self._buffer.copy()
            self._buffer.clear()
            
            # Simulate async processing
            await asyncio.sleep(0.01)
            logger.debug(f"Processed batch of {len(buffer_copy)} items")
            
        finally:
            self._processing = False


@asynccontextmanager
async def async_performance_monitor(operation_name: str):
    """Context manager for monitoring async operation performance."""
    start_time = time.time()
    start_memory = 0  # Could integrate with memory monitor
    
    logger.info(f"Starting async operation: {operation_name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Completed async operation: {operation_name}")
        logger.info(f"Duration: {duration:.2f} seconds")


class AsyncOptimizationManager:
    """Central manager for async processing optimizations."""
    
    def __init__(
        self, 
        max_concurrent_tasks: int = 10,
        max_workers: int = 4,
        buffer_size: int = 1000
    ):
        """Initialize async optimization manager."""
        self.task_pool = AsyncTaskPool(max_concurrent_tasks, max_workers)
        self.data_processor = DataProcessor(self.task_pool)
        self.detection_optimizer = DetectionOptimizer(self.task_pool)
        self.stream_processor = AsyncStreamProcessor(buffer_size)
    
    async def optimize_batch_detection(
        self, 
        detectors: List[Dict[str, Any]], 
        datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Optimize batch anomaly detection across multiple datasets."""
        # Process each dataset with all detectors in parallel
        dataset_tasks = []
        
        for dataset_id, dataset in datasets.items():
            task = self.detection_optimizer.detect_anomalies_batch(
                detectors, dataset
            )
            dataset_tasks.append((dataset_id, task))
        
        return await self.task_pool.submit_batch_tasks(dataset_tasks)
    
    async def optimize_data_pipeline(
        self, 
        pipeline_stages: List[Callable],
        data: Any
    ) -> Any:
        """Optimize data processing pipeline with async stages."""
        result = data
        
        for i, stage in enumerate(pipeline_stages):
            task_id = f"pipeline_stage_{i}"
            result = await self.task_pool.submit_cpu_task(
                task_id, stage, result
            )
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        all_metrics = self.task_pool.get_all_metrics()
        
        # Calculate statistics
        completed_tasks = [m for m in all_metrics.values() if m.status == "completed"]
        failed_tasks = [m for m in all_metrics.values() if m.status == "failed"]
        
        if completed_tasks:
            avg_duration = np.mean([m.duration for m in completed_tasks if m.duration])
            total_duration = sum(m.duration for m in completed_tasks if m.duration)
        else:
            avg_duration = 0
            total_duration = 0
        
        return {
            'total_tasks': len(all_metrics),
            'completed_tasks': len(completed_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(completed_tasks) / len(all_metrics) if all_metrics else 0,
            'average_duration': avg_duration,
            'total_duration': total_duration,
            'task_details': {
                task_id: {
                    'status': metrics.status,
                    'duration': metrics.duration,
                    'error': metrics.error
                }
                for task_id, metrics in all_metrics.items()
            }
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.stream_processor.process_remaining()
        await self.task_pool.shutdown()
        self.task_pool.clear_completed_tasks()