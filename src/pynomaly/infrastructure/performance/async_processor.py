"""Advanced async processing utilities and optimizations."""

from __future__ import annotations

import asyncio
import logging
import time
from asyncio import Queue, Semaphore
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class TaskResult:
    """Result of an async task execution."""
    
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class ProcessingStats:
    """Statistics for async processing operations."""
    
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    throughput_per_second: float = 0.0


class AsyncTaskQueue:
    """High-performance async task queue with priority and batching."""

    def __init__(self, max_size: int = 1000, max_workers: int = 10):
        """Initialize async task queue."""
        self.max_size = max_size
        self.max_workers = max_workers
        self._queue: Queue = Queue(maxsize=max_size)
        self._semaphore = Semaphore(max_workers)
        self._workers: List[asyncio.Task] = []
        self._stats = ProcessingStats()
        self._running = False

    async def put(self, task: Callable, *args, priority: int = 0, **kwargs) -> str:
        """Add task to queue with optional priority."""
        task_id = str(uuid4())
        task_item = {
            'id': task_id,
            'func': task,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'created_at': time.time()
        }
        
        await self._queue.put(task_item)
        self._stats.total_tasks += 1
        
        logger.debug(f"Added task {task_id} to queue (priority: {priority})")
        return task_id

    async def start_workers(self) -> None:
        """Start worker tasks to process the queue."""
        if self._running:
            return
            
        self._running = True
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Started {self.max_workers} async workers")

    async def stop_workers(self) -> None:
        """Stop all worker tasks."""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Stopped all async workers")

    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine that processes tasks from the queue."""
        logger.debug(f"Started worker: {worker_name}")
        
        while self._running:
            try:
                # Wait for task with timeout
                task_item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                
                async with self._semaphore:
                    await self._execute_task(task_item, worker_name)
                    
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")

    async def _execute_task(self, task_item: Dict[str, Any], worker_name: str) -> TaskResult:
        """Execute a single task."""
        task_id = task_item['id']
        func = task_item['func']
        args = task_item['args']
        kwargs = task_item['kwargs']
        
        start_time = time.time()
        
        try:
            # Execute task
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            self._stats.completed_tasks += 1
            self._stats.total_execution_time += execution_time
            self._update_stats()
            
            logger.debug(f"Task {task_id} completed by {worker_name} in {execution_time:.3f}s")
            
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self._stats.failed_tasks += 1
            self._stats.total_execution_time += execution_time
            self._update_stats()
            
            logger.error(f"Task {task_id} failed in {worker_name}: {e}")
            
            return TaskResult(
                task_id=task_id,
                success=False,
                error=e,
                execution_time=execution_time
            )

    def _update_stats(self) -> None:
        """Update processing statistics."""
        if self._stats.completed_tasks + self._stats.failed_tasks > 0:
            total_processed = self._stats.completed_tasks + self._stats.failed_tasks
            self._stats.average_execution_time = self._stats.total_execution_time / total_processed
            self._stats.throughput_per_second = total_processed / self._stats.total_execution_time if self._stats.total_execution_time > 0 else 0

    def get_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self._stats

    async def wait_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for queue to be empty."""
        start_time = time.time()
        
        while not self._queue.empty():
            if timeout and (time.time() - start_time) > timeout:
                return False
            await asyncio.sleep(0.1)
        
        return True


class AsyncBatchProcessor:
    """Process items in async batches for improved throughput."""

    def __init__(self, batch_size: int = 100, max_concurrent_batches: int = 5,
                 batch_timeout: float = 1.0):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.batch_timeout = batch_timeout
        self._semaphore = Semaphore(max_concurrent_batches)

    async def process_items(self, items: List[T], 
                          processor: Callable[[List[T]], Awaitable[List[R]]]) -> List[R]:
        """Process items in concurrent batches."""
        if not items:
            return []

        # Split items into batches
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]

        logger.info(f"Processing {len(items)} items in {len(batches)} batches")

        # Process batches concurrently
        batch_tasks = [
            self._process_batch(batch, processor, i) 
            for i, batch in enumerate(batches)
        ]

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing error: {batch_result}")
                continue
            if isinstance(batch_result, list):
                results.extend(batch_result)

        logger.info(f"Completed processing {len(results)} items")
        return results

    async def _process_batch(self, batch: List[T], 
                           processor: Callable[[List[T]], Awaitable[List[R]]],
                           batch_id: int) -> List[R]:
        """Process a single batch."""
        async with self._semaphore:
            start_time = time.time()
            
            try:
                result = await asyncio.wait_for(
                    processor(batch), 
                    timeout=self.batch_timeout
                )
                
                execution_time = time.time() - start_time
                logger.debug(f"Batch {batch_id} processed {len(batch)} items in {execution_time:.3f}s")
                
                return result
                
            except asyncio.TimeoutError:
                logger.error(f"Batch {batch_id} timed out after {self.batch_timeout}s")
                raise
            except Exception as e:
                logger.error(f"Batch {batch_id} processing error: {e}")
                raise


class AsyncDataFrameProcessor:
    """Async processing utilities specifically for pandas DataFrames."""

    def __init__(self, chunk_size: int = 10000, max_workers: int = 4):
        """Initialize DataFrame processor."""
        self.chunk_size = chunk_size
        self.max_workers = max_workers

    async def process_dataframe(self, df: pd.DataFrame, 
                              processor: Callable[[pd.DataFrame], pd.DataFrame],
                              combine_func: Optional[Callable[[List[pd.DataFrame]], pd.DataFrame]] = None) -> pd.DataFrame:
        """Process DataFrame in async chunks."""
        if df.empty:
            return df

        # Split DataFrame into chunks
        chunks = [df.iloc[i:i + self.chunk_size].copy() 
                 for i in range(0, len(df), self.chunk_size)]

        logger.info(f"Processing DataFrame ({len(df)} rows) in {len(chunks)} chunks")

        # Process chunks concurrently
        semaphore = Semaphore(self.max_workers)
        
        async def process_chunk(chunk, chunk_id):
            async with semaphore:
                loop = asyncio.get_event_loop()
                # Run CPU-bound processing in thread pool
                return await loop.run_in_executor(None, processor, chunk)

        chunk_tasks = [
            process_chunk(chunk, i) 
            for i, chunk in enumerate(chunks)
        ]

        processed_chunks = await asyncio.gather(*chunk_tasks)

        # Combine results
        if combine_func:
            result = combine_func(processed_chunks)
        else:
            result = pd.concat(processed_chunks, ignore_index=True)

        logger.info(f"DataFrame processing completed: {len(result)} rows")
        return result

    async def apply_async(self, df: pd.DataFrame, func: Callable, 
                         axis: int = 0, **kwargs) -> pd.Series:
        """Async version of DataFrame.apply()."""
        if axis == 0:
            # Apply to columns
            column_tasks = []
            for col in df.columns:
                task = asyncio.create_task(
                    self._apply_to_series(df[col], func, **kwargs)
                )
                column_tasks.append(task)
            
            results = await asyncio.gather(*column_tasks)
            return pd.Series(results, index=df.columns)
        else:
            # Apply to rows - process in chunks
            return await self.process_dataframe(
                df, 
                lambda chunk: chunk.apply(func, axis=1, **kwargs)
            )

    async def _apply_to_series(self, series: pd.Series, func: Callable, **kwargs) -> Any:
        """Apply function to series in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(series, **kwargs))


class AsyncModelTrainer:
    """Async utilities for training machine learning models."""

    def __init__(self, max_concurrent_models: int = 3):
        """Initialize async model trainer."""
        self.max_concurrent_models = max_concurrent_models
        self._semaphore = Semaphore(max_concurrent_models)

    async def train_models_parallel(self, model_configs: List[Dict[str, Any]],
                                  train_data: Union[pd.DataFrame, np.ndarray],
                                  trainer_func: Callable) -> List[TaskResult]:
        """Train multiple models in parallel."""
        logger.info(f"Training {len(model_configs)} models in parallel")

        training_tasks = [
            self._train_single_model(config, train_data, trainer_func, i)
            for i, config in enumerate(model_configs)
        ]

        results = await asyncio.gather(*training_tasks, return_exceptions=True)

        # Convert exceptions to TaskResult objects
        task_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_results.append(TaskResult(
                    task_id=f"model_{i}",
                    success=False,
                    error=result
                ))
            else:
                task_results.append(result)

        successful = sum(1 for r in task_results if r.success)
        logger.info(f"Model training completed: {successful}/{len(model_configs)} successful")

        return task_results

    async def _train_single_model(self, config: Dict[str, Any], 
                                train_data: Union[pd.DataFrame, np.ndarray],
                                trainer_func: Callable, model_id: int) -> TaskResult:
        """Train a single model."""
        async with self._semaphore:
            start_time = time.time()
            task_id = f"model_{model_id}"

            try:
                logger.debug(f"Starting training for {task_id}")

                # Run training in process pool for CPU-intensive work
                loop = asyncio.get_event_loop()
                with ProcessPoolExecutor() as executor:
                    model = await loop.run_in_executor(
                        executor, trainer_func, config, train_data
                    )

                execution_time = time.time() - start_time
                logger.info(f"Model {task_id} trained successfully in {execution_time:.2f}s")

                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result=model,
                    execution_time=execution_time,
                    metadata=config
                )

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Model {task_id} training failed: {e}")

                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=e,
                    execution_time=execution_time,
                    metadata=config
                )


# Async decorators and utilities

def async_retry(max_attempts: int = 3, delay: float = 1.0, 
               backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Decorator for async functions with retry logic."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            raise last_exception

        return wrapper
    return decorator


def async_timeout(timeout_seconds: float):
    """Decorator to add timeout to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
                raise

        return wrapper
    return decorator


def async_rate_limit(calls_per_second: float):
    """Decorator to rate limit async function calls."""
    min_interval = 1.0 / calls_per_second
    last_called = {}

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            key = id(func)

            if key in last_called:
                elapsed = now - last_called[key]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    await asyncio.sleep(sleep_time)

            last_called[key] = time.time()
            return await func(*args, **kwargs)

        return wrapper
    return decorator


@asynccontextmanager
async def async_profiler(operation_name: str = "operation"):
    """Async context manager for profiling operations."""
    start_time = time.time()
    start_memory = None
    
    try:
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        pass

    logger.info(f"Starting {operation_name}")

    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time

        memory_info = ""
        if start_memory:
            try:
                end_memory = process.memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
                memory_info = f", Δ{memory_delta:+.2f}MB"
            except:
                pass

        logger.info(f"Completed {operation_name}: {execution_time:.3f}s{memory_info}")


class AsyncContextManager:
    """Utility for managing async context and resources."""

    def __init__(self):
        """Initialize async context manager."""
        self._resources: Dict[str, Any] = {}
        self._cleanup_funcs: Dict[str, Callable] = {}

    async def add_resource(self, name: str, resource: Any, 
                          cleanup_func: Optional[Callable] = None) -> None:
        """Add a resource to be managed."""
        self._resources[name] = resource
        if cleanup_func:
            self._cleanup_funcs[name] = cleanup_func

        logger.debug(f"Added resource: {name}")

    async def get_resource(self, name: str) -> Any:
        """Get a managed resource."""
        if name not in self._resources:
            raise KeyError(f"Resource not found: {name}")
        return self._resources[name]

    async def cleanup_all(self) -> None:
        """Clean up all managed resources."""
        logger.info("Cleaning up async resources")

        for name, cleanup_func in self._cleanup_funcs.items():
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func(self._resources[name])
                else:
                    cleanup_func(self._resources[name])
                logger.debug(f"Cleaned up resource: {name}")
            except Exception as e:
                logger.error(f"Error cleaning up resource {name}: {e}")

        self._resources.clear()
        self._cleanup_funcs.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup_all()


# Global instances
_global_task_queue: Optional[AsyncTaskQueue] = None
_global_batch_processor: Optional[AsyncBatchProcessor] = None


def get_task_queue(max_workers: int = 10) -> AsyncTaskQueue:
    """Get global async task queue."""
    global _global_task_queue
    if _global_task_queue is None:
        _global_task_queue = AsyncTaskQueue(max_workers=max_workers)
    return _global_task_queue


def get_batch_processor(batch_size: int = 100) -> AsyncBatchProcessor:
    """Get global batch processor."""
    global _global_batch_processor
    if _global_batch_processor is None:
        _global_batch_processor = AsyncBatchProcessor(batch_size=batch_size)
    return _global_batch_processor