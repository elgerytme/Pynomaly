"""Distributed worker node for processing anomaly detection tasks."""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp
import psutil

from pynomaly.domain.entities import Detector, Dataset, DetectionResult
from pynomaly.domain.exceptions import ProcessingError


logger = logging.getLogger(__name__)


class DistributedWorker:
    """Worker node for distributed anomaly detection processing."""
    
    def __init__(self,
                 worker_id: str,
                 manager_host: str,
                 manager_port: int,
                 host: str = "localhost",
                 port: int = 8001,
                 capacity: int = 4,
                 capabilities: List[str] = None):
        self.worker_id = worker_id
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.host = host
        self.port = port
        self.capacity = capacity
        self.capabilities = capabilities or ["pytorch", "sklearn", "pyod"]
        
        # Internal state
        self.current_load = 0
        self.status = "idle"  # idle, busy, stopping
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._task_processor: Optional[asyncio.Task] = None
        
        # HTTP client for communication
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Worker {worker_id} initialized with capacity {capacity}")
    
    async def start(self) -> None:
        """Start the worker node."""
        if self._running:
            return
        
        self._running = True
        self._session = aiohttp.ClientSession()
        
        # Register with manager
        registered = await self._register_with_manager()
        if not registered:
            logger.error("Failed to register with manager")
            await self.stop()
            return
        
        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._task_processor = asyncio.create_task(self._process_tasks())
        
        logger.info(f"Worker {self.worker_id} started and registered")
    
    async def stop(self) -> None:
        """Stop the worker node."""
        if not self._running:
            return
        
        self._running = False
        self.status = "stopping"
        
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._task_processor:
            self._task_processor.cancel()
            try:
                await self._task_processor
            except asyncio.CancelledError:
                pass
        
        # Wait for active tasks to complete
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
            while self.active_tasks and time.time() < time.time() + 30:  # 30 second timeout
                await asyncio.sleep(0.5)
        
        # Unregister from manager
        await self._unregister_from_manager()
        
        # Close HTTP session
        if self._session:
            await self._session.close()
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    async def process_detection_task(self,
                                   task_id: str,
                                   detector_config: Dict[str, Any],
                                   dataset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a detection task."""
        if self.current_load >= self.capacity:
            raise ProcessingError("Worker at capacity")
        
        start_time = time.time()
        
        try:
            # Update status
            self.current_load += 1
            self.status = "busy" if self.current_load == self.capacity else "idle"
            
            # Track task
            self.active_tasks[task_id] = {
                "start_time": start_time,
                "detector_id": detector_config.get("id"),
                "status": "processing"
            }
            
            # TODO: Reconstruct detector and dataset from configs
            # For now, simulate processing
            await asyncio.sleep(1)  # Simulate processing time
            
            # Create mock result
            result = {
                "id": f"result_{task_id}",
                "detector_id": detector_config.get("id"),
                "dataset_id": dataset_data.get("id"),
                "n_anomalies": 5,
                "anomaly_rate": 0.1,
                "execution_time": time.time() - start_time,
                "worker_id": self.worker_id
            }
            
            self.active_tasks[task_id]["status"] = "completed"
            logger.info(f"Task {task_id} completed in {result['execution_time']:.2f}s")
            
            return result
            
        except Exception as e:
            self.active_tasks[task_id]["status"] = "failed"
            logger.error(f"Task {task_id} failed: {e}")
            raise ProcessingError(f"Task processing failed: {e}")
        
        finally:
            # Update status
            self.current_load = max(0, self.current_load - 1)
            self.status = "busy" if self.current_load >= self.capacity else "idle"
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        return {
            "worker_id": self.worker_id,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "capacity": self.capacity,
            "current_load": self.current_load,
            "load_percentage": (self.current_load / self.capacity) * 100,
            "capabilities": self.capabilities,
            "active_tasks": len(self.active_tasks),
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available // 1024 // 1024
            },
            "uptime": time.time()  # Could track actual uptime
        }
    
    async def _register_with_manager(self) -> bool:
        """Register this worker with the manager."""
        try:
            url = f"http://{self.manager_host}:{self.manager_port}/workers/register"
            data = {
                "worker_id": self.worker_id,
                "host": self.host,
                "port": self.port,
                "capacity": self.capacity,
                "capabilities": self.capabilities
            }
            
            async with self._session.post(url, json=data) as response:
                if response.status == 200:
                    logger.info("Successfully registered with manager")
                    return True
                else:
                    logger.error(f"Registration failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to register with manager: {e}")
            return False
    
    async def _unregister_from_manager(self) -> None:
        """Unregister this worker from the manager."""
        try:
            url = f"http://{self.manager_host}:{self.manager_port}/workers/{self.worker_id}/unregister"
            
            async with self._session.delete(url) as response:
                if response.status == 200:
                    logger.info("Successfully unregistered from manager")
                else:
                    logger.warning(f"Unregistration failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to unregister from manager: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to manager."""
        while self._running:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                await asyncio.sleep(30)
    
    async def _send_heartbeat(self) -> None:
        """Send heartbeat to manager."""
        try:
            url = f"http://{self.manager_host}:{self.manager_port}/workers/{self.worker_id}/heartbeat"
            data = {
                "status": self.status,
                "current_load": self.current_load,
                "timestamp": datetime.now().isoformat()
            }
            
            async with self._session.post(url, json=data) as response:
                if response.status != 200:
                    logger.warning(f"Heartbeat failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    async def _process_tasks(self) -> None:
        """Process incoming tasks from manager."""
        while self._running:
            try:
                # Check for new tasks from manager
                await self._check_for_tasks()
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task processing loop failed: {e}")
                await asyncio.sleep(5)
    
    async def _check_for_tasks(self) -> None:
        """Check for new tasks from manager."""
        try:
            url = f"http://{self.manager_host}:{self.manager_port}/workers/{self.worker_id}/tasks"
            
            async with self._session.get(url) as response:
                if response.status == 200:
                    tasks = await response.json()
                    
                    for task_data in tasks:
                        asyncio.create_task(self._handle_task(task_data))
                
        except Exception as e:
            logger.error(f"Failed to check for tasks: {e}")
    
    async def _handle_task(self, task_data: Dict[str, Any]) -> None:
        """Handle a specific task."""
        task_id = task_data.get("id")
        
        try:
            result = await self.process_detection_task(
                task_id=task_id,
                detector_config=task_data.get("detector_config", {}),
                dataset_data=task_data.get("dataset_data", {})
            )
            
            # Send result back to manager
            await self._send_task_result(task_id, result)
            
        except Exception as e:
            logger.error(f"Task {task_id} handling failed: {e}")
            # Send error back to manager
            await self._send_task_error(task_id, str(e))
    
    async def _send_task_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """Send task result to manager."""
        try:
            url = f"http://{self.manager_host}:{self.manager_port}/tasks/{task_id}/result"
            
            async with self._session.post(url, json=result) as response:
                if response.status == 200:
                    logger.info(f"Task {task_id} result sent successfully")
                else:
                    logger.warning(f"Failed to send task result: {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to send task result: {e}")
    
    async def _send_task_error(self, task_id: str, error: str) -> None:
        """Send task error to manager."""
        try:
            url = f"http://{self.manager_host}:{self.manager_port}/tasks/{task_id}/error"
            data = {"error": error, "worker_id": self.worker_id}
            
            async with self._session.post(url, json=data) as response:
                if response.status == 200:
                    logger.info(f"Task {task_id} error sent successfully")
                else:
                    logger.warning(f"Failed to send task error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to send task error: {e}")


async def main():
    """Main function for running worker as standalone service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pynomaly Distributed Worker")
    parser.add_argument("--worker-id", required=True, help="Worker ID")
    parser.add_argument("--manager-host", default="localhost", help="Manager host")
    parser.add_argument("--manager-port", type=int, default=8000, help="Manager port")
    parser.add_argument("--host", default="localhost", help="Worker host")
    parser.add_argument("--port", type=int, default=8001, help="Worker port")
    parser.add_argument("--capacity", type=int, default=4, help="Worker capacity")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and start worker
    worker = DistributedWorker(
        worker_id=args.worker_id,
        manager_host=args.manager_host,
        manager_port=args.manager_port,
        host=args.host,
        port=args.port,
        capacity=args.capacity
    )
    
    try:
        await worker.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())