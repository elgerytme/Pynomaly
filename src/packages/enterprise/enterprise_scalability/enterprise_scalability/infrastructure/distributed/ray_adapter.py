"""
Ray distributed computing adapter for enterprise scalability.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from uuid import UUID

from structlog import get_logger

try:
    import ray
    from ray.util.queue import Queue as RayQueue
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from ...domain.entities.compute_cluster import ComputeCluster, ComputeNode, ClusterStatus
from ...domain.entities.distributed_task import DistributedTask, TaskResult, TaskStatus

logger = get_logger(__name__)


class RayClusterAdapter:
    """
    Ray cluster management adapter.
    
    Provides integration with Ray distributed computing framework
    for AI/ML workloads, distributed training, and high-performance computing.
    """
    
    def __init__(self):
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not available. Install with: pip install 'ray[default]'")
        
        self.clusters: Dict[UUID, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, Any] = {}
        
        logger.info("RayClusterAdapter initialized")
    
    async def create_cluster(
        self,
        cluster: ComputeCluster,
        node_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create and start a Ray cluster."""
        logger.info("Creating Ray cluster", cluster_id=cluster.id, name=cluster.name)
        
        try:
            # Configure Ray cluster
            ray_config = self._build_ray_config(cluster, node_config)
            
            # Initialize Ray cluster
            if cluster.configuration.get("deployment_type") == "kubernetes":
                cluster_info = await self._create_kubernetes_cluster(cluster, ray_config)
            else:
                cluster_info = await self._create_local_cluster(cluster, ray_config)
            
            # Store cluster reference
            self.clusters[cluster.id] = cluster_info
            
            # Wait for cluster to be ready
            await self._wait_for_cluster_ready(cluster.id, cluster.min_nodes)
            
            logger.info("Ray cluster created successfully", cluster_id=cluster.id)
            return True
            
        except Exception as e:
            logger.error("Failed to create Ray cluster", error=str(e), cluster_id=cluster.id)
            return False
    
    async def scale_cluster(
        self,
        cluster_id: UUID,
        target_workers: int
    ) -> bool:
        """Scale Ray cluster to target number of workers."""
        logger.info("Scaling Ray cluster", cluster_id=cluster_id, target_workers=target_workers)
        
        try:
            cluster_info = self.clusters.get(cluster_id)
            if not cluster_info:
                raise ValueError(f"Cluster {cluster_id} not found")
            
            # Get current cluster state
            cluster_state = ray.cluster_resources()
            current_workers = self._count_workers(cluster_state)
            
            if target_workers == current_workers:
                logger.info("Cluster already at target size", current=current_workers)
                return True
            
            # Perform scaling
            if cluster_info["deployment_type"] == "kubernetes":
                success = await self._scale_kubernetes_cluster(cluster_id, target_workers)
            else:
                success = await self._scale_local_cluster(cluster_id, target_workers)
            
            if success:
                # Wait for scaling to complete
                await self._wait_for_scaling_complete(cluster_id, target_workers)
                logger.info("Ray cluster scaled successfully", cluster_id=cluster_id, workers=target_workers)
            
            return success
            
        except Exception as e:
            logger.error("Failed to scale Ray cluster", error=str(e), cluster_id=cluster_id)
            return False
    
    async def submit_task(
        self,
        cluster_id: UUID,
        task: DistributedTask
    ) -> Optional[str]:
        """Submit a task to Ray cluster."""
        logger.debug("Submitting task to Ray", task_id=task.id, cluster_id=cluster_id)
        
        try:
            cluster_info = self.clusters.get(cluster_id)
            if not cluster_info:
                raise ValueError(f"Cluster {cluster_id} not found")
            
            # Create Ray remote function
            remote_func = self._create_remote_function(task)
            
            # Submit task with resource constraints
            future = remote_func.options(
                num_cpus=task.resources.cpu_cores,
                memory=int(task.resources.memory_gb * 1024 * 1024 * 1024),  # Convert to bytes
                num_gpus=task.resources.gpu_count if task.resources.gpu_count > 0 else None
            ).remote(*task.function_args, **task.function_kwargs)
            
            # Store task reference
            task_id = str(task.id)
            self.active_tasks[task_id] = {
                "future": future,
                "cluster_id": cluster_id,
                "submitted_at": datetime.utcnow(),
                "task": task
            }
            
            return task_id
            
        except Exception as e:
            logger.error("Failed to submit task to Ray", error=str(e), task_id=task.id)
            return None
    
    async def get_task_result(
        self,
        cluster_id: UUID,
        task_key: str
    ) -> Optional[TaskResult]:
        """Get task execution result from Ray."""
        try:
            task_info = self.active_tasks.get(task_key)
            if not task_info:
                return None
            
            future = task_info["future"]
            start_time = task_info["submitted_at"]
            
            try:
                # Check if task is ready (non-blocking)
                ready_futures, remaining_futures = ray.wait([future], timeout=0)
                
                if not ready_futures:
                    # Task not ready yet
                    return None
                
                # Get result
                result = ray.get(ready_futures[0])
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Clean up
                del self.active_tasks[task_key]
                
                return TaskResult(
                    success=True,
                    return_value=result,
                    execution_time_seconds=execution_time,
                    computed_at=datetime.utcnow(),
                    computed_by=f"ray-worker"
                )
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Clean up
                del self.active_tasks[task_key]
                
                return TaskResult(
                    success=False,
                    error_message=str(e),
                    execution_time_seconds=execution_time,
                    computed_at=datetime.utcnow(),
                    computed_by=f"ray-worker"
                )
                
        except Exception as e:
            logger.error("Failed to get task result from Ray", error=str(e), task_key=task_key)
            return None
    
    async def cancel_task(
        self,
        cluster_id: UUID,
        task_key: str
    ) -> bool:
        """Cancel a running task in Ray."""
        try:
            task_info = self.active_tasks.get(task_key)
            if not task_info:
                return False
            
            # Cancel Ray task
            future = task_info["future"]
            ray.cancel(future)
            
            # Clean up
            del self.active_tasks[task_key]
            
            return True
            
        except Exception as e:
            logger.error("Failed to cancel Ray task", error=str(e), task_key=task_key)
            return False
    
    async def get_cluster_info(self, cluster_id: UUID) -> Optional[Dict[str, Any]]:
        """Get Ray cluster information."""
        try:
            cluster_info = self.clusters.get(cluster_id)
            if not cluster_info:
                return None
            
            # Get cluster resources and nodes
            cluster_resources = ray.cluster_resources()
            nodes = ray.nodes()
            
            # Process node information
            workers = []
            total_cpus = 0
            total_memory = 0
            used_cpus = 0
            used_memory = 0
            
            for node in nodes:
                if node["Alive"]:
                    node_resources = node.get("Resources", {})
                    node_cpus = node_resources.get("CPU", 0)
                    node_memory = node_resources.get("memory", 0)
                    
                    # Estimate usage (Ray doesn't provide detailed usage stats by default)
                    node_used_cpus = node_cpus * 0.1  # Placeholder estimation
                    node_used_memory = node_memory * 0.1  # Placeholder estimation
                    
                    workers.append({
                        "node_id": node["NodeID"],
                        "address": node.get("NodeManagerAddress", ""),
                        "status": "alive" if node["Alive"] else "dead",
                        "cpus": node_cpus,
                        "memory_gb": node_memory / (1024**3) if node_memory else 0,
                        "used_cpus": node_used_cpus,
                        "used_memory_gb": node_used_memory / (1024**3) if node_used_memory else 0,
                        "gpus": node_resources.get("GPU", 0),
                        "object_store_memory": node_resources.get("object_store_memory", 0) / (1024**3)
                    })
                    
                    total_cpus += node_cpus
                    total_memory += node_memory
                    used_cpus += node_used_cpus
                    used_memory += node_used_memory
            
            # Get task statistics
            active_tasks = len(self.active_tasks)
            
            return {
                "cluster_id": str(cluster_id),
                "head_node_address": cluster_info.get("address", ""),
                "worker_count": len(workers),
                "workers": workers,
                "total_cpus": total_cpus,
                "total_memory_gb": total_memory / (1024**3) if total_memory else 0,
                "used_cpus": used_cpus,
                "used_memory_gb": used_memory / (1024**3) if used_memory else 0,
                "tasks": {
                    "active": active_tasks,
                    "total_submitted": cluster_info.get("total_tasks_submitted", 0)
                },
                "ray_version": ray.__version__
            }
            
        except Exception as e:
            logger.error("Failed to get Ray cluster info", error=str(e), cluster_id=cluster_id)
            return None
    
    async def shutdown_cluster(self, cluster_id: UUID) -> bool:
        """Shutdown Ray cluster."""
        logger.info("Shutting down Ray cluster", cluster_id=cluster_id)
        
        try:
            cluster_info = self.clusters.get(cluster_id)
            if not cluster_info:
                logger.warning("Cluster not found for shutdown", cluster_id=cluster_id)
                return True
            
            # Cancel all active tasks for this cluster
            tasks_to_cancel = [
                task_id for task_id, task_info in self.active_tasks.items()
                if task_info["cluster_id"] == cluster_id
            ]
            
            for task_id in tasks_to_cancel:
                await self.cancel_task(cluster_id, task_id)
            
            # Shutdown Ray cluster
            if cluster_info["deployment_type"] == "local":
                ray.shutdown()
            else:
                # For Kubernetes, we would trigger cluster deletion
                await self._shutdown_kubernetes_cluster(cluster_id)
            
            # Remove from tracking
            del self.clusters[cluster_id]
            
            logger.info("Ray cluster shutdown complete", cluster_id=cluster_id)
            return True
            
        except Exception as e:
            logger.error("Failed to shutdown Ray cluster", error=str(e), cluster_id=cluster_id)
            return False
    
    # Private helper methods
    
    def _build_ray_config(
        self,
        cluster: ComputeCluster,
        node_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build Ray cluster configuration."""
        config = {
            "num_cpus": node_config.get("cpu_cores", 2) if node_config else 2,
            "num_gpus": node_config.get("gpu_count", 0) if node_config else 0,
            "memory": int((node_config.get("memory_gb", 4) if node_config else 4) * 1024 * 1024 * 1024),
            "object_store_memory": int((node_config.get("object_store_gb", 1) if node_config else 1) * 1024 * 1024 * 1024)
        }
        
        if cluster.dashboard_port:
            config["dashboard_port"] = cluster.dashboard_port
        
        return config
    
    async def _create_local_cluster(
        self,
        cluster: ComputeCluster,
        ray_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create local Ray cluster."""
        # Initialize Ray
        ray.init(
            num_cpus=ray_config["num_cpus"],
            num_gpus=ray_config.get("num_gpus", 0),
            memory=ray_config.get("memory"),
            object_store_memory=ray_config.get("object_store_memory"),
            dashboard_port=ray_config.get("dashboard_port"),
            ignore_reinit_error=True
        )
        
        return {
            "deployment_type": "local",
            "address": ray.get_runtime_context().get_dashboard_url(),
            "config": ray_config,
            "total_tasks_submitted": 0
        }
    
    async def _create_kubernetes_cluster(
        self,
        cluster: ComputeCluster,
        ray_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create Kubernetes-based Ray cluster."""
        # This would integrate with Ray on Kubernetes
        # For now, we'll create a local cluster as a placeholder
        logger.warning("Kubernetes Ray cluster creation not fully implemented, using local cluster")
        return await self._create_local_cluster(cluster, ray_config)
    
    async def _wait_for_cluster_ready(
        self,
        cluster_id: UUID,
        min_workers: int,
        timeout_seconds: int = 300
    ) -> None:
        """Wait for Ray cluster to be ready."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            try:
                cluster_resources = ray.cluster_resources()
                current_workers = self._count_workers(cluster_resources)
                
                if current_workers >= min_workers:
                    logger.info("Ray cluster ready", cluster_id=cluster_id, workers=current_workers)
                    return
                
                await asyncio.sleep(5)
                
            except Exception:
                await asyncio.sleep(5)
                continue
        
        raise TimeoutError(f"Cluster did not become ready within {timeout_seconds} seconds")
    
    async def _wait_for_scaling_complete(
        self,
        cluster_id: UUID,
        target_workers: int,
        timeout_seconds: int = 300
    ) -> None:
        """Wait for scaling operation to complete."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            try:
                cluster_resources = ray.cluster_resources()
                current_workers = self._count_workers(cluster_resources)
                
                if current_workers == target_workers:
                    logger.info("Ray scaling complete", current_workers=current_workers)
                    return
                
                await asyncio.sleep(10)
                
            except Exception:
                await asyncio.sleep(10)
                continue
        
        raise TimeoutError(f"Scaling did not complete within {timeout_seconds} seconds")
    
    def _count_workers(self, cluster_resources: Dict[str, float]) -> int:
        """Count active workers from cluster resources."""
        # Ray doesn't have a direct way to count workers
        # We estimate based on CPU resources
        total_cpus = cluster_resources.get("CPU", 0)
        return max(1, int(total_cpus // 2))  # Assume 2 CPUs per worker
    
    def _create_remote_function(self, task: DistributedTask) -> Any:
        """Create Ray remote function for task."""
        # Import the function
        module = __import__(task.module_name, fromlist=[task.function_name])
        func = getattr(module, task.function_name)
        
        # Convert to Ray remote function
        remote_func = ray.remote(func)
        
        return remote_func
    
    async def _scale_kubernetes_cluster(
        self,
        cluster_id: UUID,
        target_workers: int
    ) -> bool:
        """Scale Kubernetes-based Ray cluster."""
        # Placeholder for Kubernetes scaling implementation
        logger.warning("Kubernetes Ray cluster scaling not implemented")
        return True
    
    async def _scale_local_cluster(
        self,
        cluster_id: UUID,
        target_workers: int
    ) -> bool:
        """Scale local Ray cluster."""
        # Local clusters can't be dynamically scaled in Ray
        # This would require restarting with new configuration
        logger.info("Local Ray cluster scaling requires restart")
        return True
    
    async def _shutdown_kubernetes_cluster(self, cluster_id: UUID) -> None:
        """Shutdown Kubernetes Ray cluster."""
        # Placeholder for Kubernetes cluster shutdown
        logger.warning("Kubernetes Ray cluster shutdown not implemented")