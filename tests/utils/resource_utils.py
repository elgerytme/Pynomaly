"""Resource management utilities for test cleanup and monitoring."""

import asyncio
import threading
import weakref
import psutil
import os
import tempfile
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import List, Dict, Any, Optional, Set, Union
from pathlib import Path
import gc

logger = logging.getLogger(__name__)


class ResourceTracker:
    """Track and manage test resources to prevent leaks."""
    
    def __init__(self):
        self.active_files: Set[int] = set()  # File descriptor IDs
        self.active_threads: Set[threading.Thread] = set()
        self.active_tasks: Set[asyncio.Task] = set()
        self.temp_files: List[Path] = []
        self.temp_dirs: List[Path] = []
        self.database_connections: Set[Any] = set()
        self.network_connections: Set[Any] = set()
        self._initial_metrics = None
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource usage metrics."""
        process = psutil.Process()
        return {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "open_files": len(process.open_files()),
            "connections": len(process.connections()),
            "threads": process.num_threads(),
            "cpu_percent": process.cpu_percent(),
        }
    
    def start_tracking(self) -> None:
        """Start tracking resources at test beginning."""
        self._initial_metrics = self.get_current_metrics()
        logger.debug(f"Started resource tracking: {self._initial_metrics}")
    
    def register_file(self, file_obj: Any) -> None:
        """Register a file object for tracking."""
        if hasattr(file_obj, 'fileno'):
            try:
                self.active_files.add(file_obj.fileno())
            except (OSError, ValueError):
                pass  # File already closed or doesn't have fileno
    
    def register_thread(self, thread: threading.Thread) -> None:
        """Register a thread for tracking."""
        self.active_threads.add(thread)
    
    def register_task(self, task: asyncio.Task) -> None:
        """Register an async task for tracking."""
        self.active_tasks.add(task)
    
    def register_temp_file(self, temp_file: Union[str, Path]) -> None:
        """Register a temporary file for cleanup."""
        path = Path(temp_file)
        self.temp_files.append(path)
    
    def register_temp_dir(self, temp_dir: Union[str, Path]) -> None:
        """Register a temporary directory for cleanup."""
        path = Path(temp_dir)
        self.temp_dirs.append(path)
    
    def register_db_connection(self, connection: Any) -> None:
        """Register a database connection for tracking."""
        self.database_connections.add(connection)
    
    def register_network_connection(self, connection: Any) -> None:
        """Register a network connection for tracking."""
        self.network_connections.add(connection)
    
    async def cleanup_all(self, force: bool = False) -> Dict[str, Any]:
        """Clean up all tracked resources."""
        cleanup_report = {
            "files_cleaned": 0,
            "threads_stopped": 0,
            "tasks_cancelled": 0,
            "temp_files_removed": 0,
            "temp_dirs_removed": 0,
            "db_connections_closed": 0,
            "network_connections_closed": 0,
            "errors": [],
        }
        
        # Clean up async tasks
        for task in list(self.active_tasks):
            try:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                cleanup_report["tasks_cancelled"] += 1
            except Exception as e:
                cleanup_report["errors"].append(f"Task cleanup error: {e}")
        self.active_tasks.clear()
        
        # Clean up threads
        for thread in list(self.active_threads):
            try:
                if thread.is_alive():
                    if force and hasattr(thread, '_stop'):
                        thread._stop()
                    else:
                        thread.join(timeout=2.0)
                    if thread.is_alive():
                        cleanup_report["errors"].append(f"Thread {thread.name} did not stop")
                    else:
                        cleanup_report["threads_stopped"] += 1
            except Exception as e:
                cleanup_report["errors"].append(f"Thread cleanup error: {e}")
        self.active_threads.clear()
        
        # Clean up database connections
        for conn in list(self.database_connections):
            try:
                if hasattr(conn, 'close'):
                    if asyncio.iscoroutinefunction(conn.close):
                        await conn.close()
                    else:
                        conn.close()
                elif hasattr(conn, 'disconnect'):
                    if asyncio.iscoroutinefunction(conn.disconnect):
                        await conn.disconnect()
                    else:
                        conn.disconnect()
                cleanup_report["db_connections_closed"] += 1
            except Exception as e:
                cleanup_report["errors"].append(f"DB connection cleanup error: {e}")
        self.database_connections.clear()
        
        # Clean up network connections
        for conn in list(self.network_connections):
            try:
                if hasattr(conn, 'close'):
                    if asyncio.iscoroutinefunction(conn.close):
                        await conn.close()
                    else:
                        conn.close()
                cleanup_report["network_connections_closed"] += 1
            except Exception as e:
                cleanup_report["errors"].append(f"Network connection cleanup error: {e}")
        self.network_connections.clear()
        
        # Clean up temporary files
        for temp_file in list(self.temp_files):
            try:
                if temp_file.exists():
                    temp_file.unlink()
                cleanup_report["temp_files_removed"] += 1
            except Exception as e:
                cleanup_report["errors"].append(f"Temp file cleanup error: {e}")
        self.temp_files.clear()
        
        # Clean up temporary directories
        for temp_dir in list(self.temp_dirs):
            try:
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
                cleanup_report["temp_dirs_removed"] += 1
            except Exception as e:
                cleanup_report["errors"].append(f"Temp dir cleanup error: {e}")
        self.temp_dirs.clear()
        
        # Force garbage collection
        gc.collect()
        
        return cleanup_report
    
    def get_resource_diff(self) -> Dict[str, Any]:
        """Get resource usage difference from start of tracking."""
        if not self._initial_metrics:
            return {}
        
        current = self.get_current_metrics()
        return {
            "memory_diff_mb": current["memory_mb"] - self._initial_metrics["memory_mb"],
            "files_diff": current["open_files"] - self._initial_metrics["open_files"],
            "connections_diff": current["connections"] - self._initial_metrics["connections"],
            "threads_diff": current["threads"] - self._initial_metrics["threads"],
            "current_metrics": current,
            "initial_metrics": self._initial_metrics,
        }


class SafeResourceManager:
    """Context manager for safe resource handling."""
    
    def __init__(self, tracker: Optional[ResourceTracker] = None):
        self.tracker = tracker or ResourceTracker()
        self.cleanup_report = None
    
    async def __aenter__(self):
        self.tracker.start_tracking()
        return self.tracker
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_report = await self.tracker.cleanup_all(force=True)
        
        if self.cleanup_report["errors"]:
            logger.warning(f"Resource cleanup errors: {self.cleanup_report['errors']}")
        
        # Check for resource leaks
        diff = self.tracker.get_resource_diff()
        if diff.get("memory_diff_mb", 0) > 10:  # More than 10MB increase
            logger.warning(f"Potential memory leak detected: {diff['memory_diff_mb']:.1f}MB")
        
        if diff.get("files_diff", 0) > 5:  # More than 5 files left open
            logger.warning(f"Potential file leak detected: {diff['files_diff']} files")
        
        logger.debug(f"Resource cleanup completed: {self.cleanup_report}")


@contextmanager
def managed_temp_file(suffix=None, prefix=None, dir=None, tracker: Optional[ResourceTracker] = None):
    """Context manager for temporary files with guaranteed cleanup."""
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, prefix=prefix, dir=dir, delete=False)
        if tracker:
            tracker.register_temp_file(temp_file.name)
            tracker.register_file(temp_file)
        yield temp_file
    finally:
        if temp_file:
            try:
                temp_file.close()
                if tracker:
                    # Remove from tracker since we're cleaning up manually
                    temp_path = Path(temp_file.name)
                    if temp_path in tracker.temp_files:
                        tracker.temp_files.remove(temp_path)
                # Always try to remove the file
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file.name}: {e}")


@contextmanager
def managed_temp_dir(suffix=None, prefix=None, dir=None, tracker: Optional[ResourceTracker] = None):
    """Context manager for temporary directories with guaranteed cleanup."""
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        if tracker:
            tracker.register_temp_dir(temp_dir)
        yield Path(temp_dir)
    finally:
        if temp_dir:
            try:
                if tracker:
                    # Remove from tracker since we're cleaning up manually
                    temp_path = Path(temp_dir)
                    if temp_path in tracker.temp_dirs:
                        tracker.temp_dirs.remove(temp_path)
                # Always try to remove the directory
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")


@asynccontextmanager
async def managed_task(coro, tracker: Optional[ResourceTracker] = None):
    """Context manager for async tasks with guaranteed cleanup."""
    task = None
    try:
        task = asyncio.create_task(coro)
        if tracker:
            tracker.register_task(task)
        yield task
    finally:
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass


class ThreadManager:
    """Enhanced thread management with proper cleanup."""
    
    def __init__(self, tracker: Optional[ResourceTracker] = None):
        self.tracker = tracker
        self.threads: List[threading.Thread] = []
    
    def start_thread(self, target, args=(), kwargs=None, **thread_kwargs) -> threading.Thread:
        """Start a thread with tracking."""
        if kwargs is None:
            kwargs = {}
        
        thread = threading.Thread(target=target, args=args, kwargs=kwargs, **thread_kwargs)
        thread.start()
        
        self.threads.append(thread)
        if self.tracker:
            self.tracker.register_thread(thread)
        
        return thread
    
    def join_all(self, timeout: Optional[float] = None) -> List[threading.Thread]:
        """Join all threads, return list of threads that didn't finish."""
        still_running = []
        
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    still_running.append(thread)
        
        return still_running
    
    def stop_all(self, timeout: Optional[float] = 2.0) -> Dict[str, Any]:
        """Stop all threads forcefully if needed."""
        result = {"stopped": 0, "forced": 0, "failed": []}
        
        still_running = self.join_all(timeout)
        
        for thread in still_running:
            try:
                # Try graceful stop first
                if hasattr(thread, 'stop'):
                    thread.stop()
                    thread.join(timeout=1.0)
                    
                if thread.is_alive():
                    # Force stop if available
                    if hasattr(thread, '_stop'):
                        thread._stop()
                        result["forced"] += 1
                    else:
                        result["failed"].append(thread.name)
                else:
                    result["stopped"] += 1
                    
            except Exception as e:
                result["failed"].append(f"{thread.name}: {e}")
        
        return result


# Global resource tracker for convenience
global_tracker = ResourceTracker()


def get_global_tracker() -> ResourceTracker:
    """Get the global resource tracker."""
    return global_tracker