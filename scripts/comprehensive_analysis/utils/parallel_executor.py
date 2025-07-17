"""Parallel execution utilities for comprehensive static analysis."""

import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional, TypeVar, Awaitable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import time
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ParallelExecutor:
    """Manages parallel execution of analysis tasks."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(os.cpu_count() or 1, 8)
        self.use_processes = use_processes
        self.semaphore = asyncio.Semaphore(self.max_workers)
        self.active_tasks = []
        
        logger.info(f"Initialized ParallelExecutor with {self.max_workers} workers")
    
    async def execute_tasks(self, tasks: List[Callable[[], Awaitable[T]]]) -> List[T]:
        """Execute multiple async tasks in parallel."""
        if not tasks:
            return []
        
        logger.info(f"Executing {len(tasks)} tasks in parallel")
        start_time = time.time()
        
        # Create semaphore-controlled tasks
        semaphore_tasks = [self._execute_with_semaphore(task) for task in tasks]
        
        # Execute all tasks
        results = await asyncio.gather(*semaphore_tasks, return_exceptions=True)
        
        # Separate successful results from exceptions
        successful_results = []
        exceptions = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                exceptions.append((i, result))
                logger.error(f"Task {i} failed with exception: {result}")
            else:
                successful_results.append(result)
        
        execution_time = time.time() - start_time
        logger.info(f"Completed {len(successful_results)}/{len(tasks)} tasks in {execution_time:.2f}s")
        
        if exceptions:
            logger.warning(f"{len(exceptions)} tasks failed with exceptions")
        
        return successful_results
    
    async def _execute_with_semaphore(self, task: Callable[[], Awaitable[T]]) -> T:
        """Execute a task with semaphore control."""
        async with self.semaphore:
            return await task()
    
    async def execute_tool_analysis(self, tool_tasks: List[Dict[str, Any]]) -> List[Any]:
        """Execute analysis tools in parallel."""
        if not tool_tasks:
            return []
        
        logger.info(f"Executing {len(tool_tasks)} analysis tools in parallel")
        
        # Create async tasks for each tool
        async_tasks = []
        for tool_task in tool_tasks:
            tool = tool_task["tool"]
            files = tool_task["files"]
            
            # Create async task callable
            async_task = self._create_tool_task(tool, files)
            async_tasks.append(async_task)
        
        # Execute all tool tasks
        results = await self.execute_tasks(async_tasks)
        
        return results
    
    def _create_tool_task(self, tool, files: List[Path]) -> Callable[[], Awaitable[Any]]:
        """Create an async task for a tool."""
        async def tool_task():
            try:
                return await tool.analyze(files)
            except Exception as e:
                logger.error(f"Tool {tool.name} failed: {e}")
                raise
        
        return tool_task
    
    def execute_cpu_bound_tasks(self, tasks: List[Callable[[], T]]) -> List[T]:
        """Execute CPU-bound tasks using process pool."""
        if not tasks:
            return []
        
        logger.info(f"Executing {len(tasks)} CPU-bound tasks")
        start_time = time.time()
        
        if self.use_processes:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            results = []
            
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"CPU-bound task {i} failed: {e}")
                    results.append(None)
        
        execution_time = time.time() - start_time
        logger.info(f"Completed CPU-bound tasks in {execution_time:.2f}s")
        
        return [r for r in results if r is not None]
    
    async def execute_with_timeout(self, task: Callable[[], Awaitable[T]], 
                                 timeout: float = 300.0) -> Optional[T]:
        """Execute a task with timeout."""
        try:
            return await asyncio.wait_for(task(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Task timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Task failed with exception: {e}")
            return None
    
    async def execute_with_retry(self, task: Callable[[], Awaitable[T]], 
                               max_retries: int = 3, delay: float = 1.0) -> Optional[T]:
        """Execute a task with retry logic."""
        for attempt in range(max_retries + 1):
            try:
                return await task()
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Task failed after {max_retries} retries: {e}")
                    return None
                else:
                    logger.warning(f"Task failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
        
        return None
    
    def batch_files(self, files: List[Path], batch_size: int) -> List[List[Path]]:
        """Batch files into groups for parallel processing."""
        if batch_size <= 0:
            return [files]
        
        batches = []
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batches.append(batch)
        
        logger.debug(f"Created {len(batches)} batches from {len(files)} files")
        return batches
    
    def estimate_batch_size(self, files: List[Path], target_batches: int) -> int:
        """Estimate optimal batch size based on number of files and target batches."""
        if not files or target_batches <= 0:
            return len(files)
        
        batch_size = max(1, len(files) // target_batches)
        logger.debug(f"Estimated batch size: {batch_size} for {len(files)} files")
        return batch_size
    
    async def execute_batched_analysis(self, files: List[Path], 
                                     tool_analyzers: List[Any],
                                     batch_size: Optional[int] = None) -> List[Any]:
        """Execute analysis on batched files."""
        if not files or not tool_analyzers:
            return []
        
        # Determine batch size
        if batch_size is None:
            batch_size = self.estimate_batch_size(files, self.max_workers * 2)
        
        # Create batches
        file_batches = self.batch_files(files, batch_size)
        
        # Create tasks for each tool and batch combination
        all_tasks = []
        for tool in tool_analyzers:
            for batch in file_batches:
                task_info = {
                    "tool": tool,
                    "files": batch,
                    "batch_id": len(all_tasks)
                }
                all_tasks.append(task_info)
        
        # Execute all tasks
        results = await self.execute_tool_analysis(all_tasks)
        
        return results
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "max_workers": self.max_workers,
            "use_processes": self.use_processes,
            "active_tasks": len(self.active_tasks),
            "semaphore_count": self.semaphore._value if hasattr(self.semaphore, '_value') else 0,
        }
    
    async def shutdown(self):
        """Shutdown the executor and clean up resources."""
        logger.info("Shutting down ParallelExecutor")
        
        # Cancel active tasks
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete or be cancelled
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        
        self.active_tasks.clear()
        logger.info("ParallelExecutor shutdown complete")


class TaskManager:
    """Manages task execution with progress tracking."""
    
    def __init__(self, executor: ParallelExecutor):
        self.executor = executor
        self.progress_callback = None
        self.total_tasks = 0
        self.completed_tasks = 0
    
    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback
    
    async def execute_with_progress(self, tasks: List[Callable[[], Awaitable[T]]]) -> List[T]:
        """Execute tasks with progress tracking."""
        self.total_tasks = len(tasks)
        self.completed_tasks = 0
        
        if self.progress_callback:
            self.progress_callback(self.completed_tasks, self.total_tasks)
        
        # Wrap tasks with progress tracking
        wrapped_tasks = [self._wrap_task_with_progress(task) for task in tasks]
        
        # Execute tasks
        results = await self.executor.execute_tasks(wrapped_tasks)
        
        return results
    
    async def _wrap_task_with_progress(self, task: Callable[[], Awaitable[T]]) -> Callable[[], Awaitable[T]]:
        """Wrap a task with progress tracking."""
        async def wrapped_task():
            try:
                result = await task()
                self.completed_tasks += 1
                
                if self.progress_callback:
                    self.progress_callback(self.completed_tasks, self.total_tasks)
                
                return result
            except Exception as e:
                self.completed_tasks += 1
                
                if self.progress_callback:
                    self.progress_callback(self.completed_tasks, self.total_tasks)
                
                raise
        
        return wrapped_task