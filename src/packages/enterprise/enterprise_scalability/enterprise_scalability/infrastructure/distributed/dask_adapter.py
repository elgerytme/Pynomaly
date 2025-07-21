"""
Dask distributed computing adapter for enterprise scalability.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from uuid import UUID

from structlog import get_logger

try:
    import dask
    from dask.distributed import Client, as_completed, Future
    from dask_kubernetes import KubeCluster, make_pod_spec
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from ...domain.entities.compute_cluster import ComputeCluster, ComputeNode, ClusterStatus
from ...domain.entities.distributed_task import DistributedTask, TaskResult, TaskStatus

logger = get_logger(__name__)


class DaskClusterAdapter:
    """
    Dask cluster management adapter.
    
    Provides integration with Dask distributed computing framework
    for scalable task execution and cluster management.
    """
    
    def __init__(self):
        if not DASK_AVAILABLE:
            raise ImportError("Dask is not available. Install with: pip install 'dask[distributed]'")
        
        self.clusters: Dict[UUID, Client] = {}
        self.cluster_configs: Dict[UUID, Dict[str, Any]] = {}
        
        logger.info("DaskClusterAdapter initialized")
    
    async def create_cluster(
        self,
        cluster: ComputeCluster,
        node_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create and start a Dask cluster."""
        logger.info("Creating Dask cluster", cluster_id=cluster.id, name=cluster.name)
        
        try:
            # Configure cluster based on deployment type
            if cluster.configuration.get("deployment_type") == "kubernetes":
                client = await self._create_kubernetes_cluster(cluster, node_config)
            else:
                client = await self._create_local_cluster(cluster, node_config)
            
            # Store cluster reference
            self.clusters[cluster.id] = client
            self.cluster_configs[cluster.id] = node_config or {}
            
            # Wait for cluster to be ready
            await self._wait_for_cluster_ready(client, cluster)
            
            logger.info("Dask cluster created successfully", cluster_id=cluster.id)
            return True
            
        except Exception as e:
            logger.error("Failed to create Dask cluster", error=str(e), cluster_id=cluster.id)
            return False
    
    async def scale_cluster(
        self,
        cluster_id: UUID,
        target_workers: int
    ) -> bool:
        """Scale Dask cluster to target number of workers."""
        logger.info("Scaling Dask cluster", cluster_id=cluster_id, target_workers=target_workers)
        
        try:
            client = self.clusters.get(cluster_id)
            if not client:
                raise ValueError(f"Cluster {cluster_id} not found")
            
            # Get current worker count
            current_workers = len(client.scheduler_info()["workers"])
            
            if target_workers == current_workers:
                logger.info("Cluster already at target size", current=current_workers)
                return True
            
            # Scale cluster
            if hasattr(client.cluster, 'scale'):
                # For adaptive clusters
                client.cluster.scale(target_workers)
            else:
                # For manual scaling
                if target_workers > current_workers:
                    # Scale up
                    for _ in range(target_workers - current_workers):
                        client.cluster.scale_up(1)
                else:
                    # Scale down
                    for _ in range(current_workers - target_workers):
                        client.cluster.scale_down(1)
            
            # Wait for scaling to complete
            await self._wait_for_scaling_complete(client, target_workers)
            
            logger.info("Dask cluster scaled successfully", cluster_id=cluster_id, workers=target_workers)
            return True
            
        except Exception as e:
            logger.error("Failed to scale Dask cluster", error=str(e), cluster_id=cluster_id)
            return False
    
    async def submit_task(
        self,
        cluster_id: UUID,
        task: DistributedTask
    ) -> Optional[str]:
        """Submit a task to Dask cluster."""
        logger.debug("Submitting task to Dask", task_id=task.id, cluster_id=cluster_id)
        
        try:
            client = self.clusters.get(cluster_id)
            if not client:
                raise ValueError(f"Cluster {cluster_id} not found")
            
            # Import function dynamically
            module = __import__(task.module_name, fromlist=[task.function_name])
            func = getattr(module, task.function_name)
            
            # Submit task with resource constraints
            future = client.submit(
                func,
                *task.function_args,
                **task.function_kwargs,
                resources={
                    'CPU': task.resources.cpu_cores,
                    'memory': f"{task.resources.memory_gb}GB"
                },
                key=str(task.id),
                priority=self._get_dask_priority(task.priority)
            )
            
            # Store future for tracking
            return str(future.key)
            
        except Exception as e:
            logger.error("Failed to submit task to Dask", error=str(e), task_id=task.id)
            return None
    
    async def get_task_result(
        self,
        cluster_id: UUID,
        future_key: str
    ) -> Optional[TaskResult]:
        """Get task execution result from Dask."""
        try:
            client = self.clusters.get(cluster_id)
            if not client:
                raise ValueError(f"Cluster {cluster_id} not found")
            
            # Get future by key
            future = client.futures.get(future_key)
            if not future:
                return None
            
            start_time = datetime.utcnow()
            
            try:
                # Get result (blocks until complete)
                result = await asyncio.to_thread(future.result, timeout=1)  # 1 second timeout
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                return TaskResult(
                    success=True,
                    return_value=result,
                    execution_time_seconds=execution_time,
                    computed_at=datetime.utcnow(),
                    computed_by=str(future.key)
                )
                
            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                return TaskResult(
                    success=False,
                    error_message=str(e),
                    execution_time_seconds=execution_time,
                    computed_at=datetime.utcnow(),
                    computed_by=str(future.key)
                )
                
        except Exception as e:
            logger.error("Failed to get task result from Dask", error=str(e), future_key=future_key)
            return None
    
    async def cancel_task(
        self,
        cluster_id: UUID,
        future_key: str
    ) -> bool:
        """Cancel a running task in Dask."""
        try:
            client = self.clusters.get(cluster_id)
            if not client:
                raise ValueError(f"Cluster {cluster_id} not found")
            
            # Get and cancel future
            future = client.futures.get(future_key)
            if future:
                future.cancel()
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to cancel Dask task", error=str(e), future_key=future_key)
            return False
    
    async def get_cluster_info(self, cluster_id: UUID) -> Optional[Dict[str, Any]]:
        """Get Dask cluster information."""
        try:
            client = self.clusters.get(cluster_id)
            if not client:
                return None
            
            info = client.scheduler_info()
            
            # Extract worker information
            workers = []
            total_cores = 0
            total_memory = 0
            used_cores = 0
            used_memory = 0
            
            for worker_id, worker_info in info["workers"].items():
                worker_cores = worker_info.get("nthreads", 1)
                worker_memory = worker_info.get("memory_limit", 0)
                worker_used_memory = worker_info.get("metrics", {}).get("memory", 0)
                
                workers.append({
                    "id": worker_id,
                    "address": worker_info.get("address", ""),
                    "status": worker_info.get("status", "unknown"),
                    "cores": worker_cores,
                    "memory_gb": worker_memory / (1024**3) if worker_memory else 0,
                    "used_memory_gb": worker_used_memory / (1024**3) if worker_used_memory else 0,
                    "tasks": len(worker_info.get("processing", {}))
                })
                
                total_cores += worker_cores
                total_memory += worker_memory
                used_memory += worker_used_memory
            
            return {
                "scheduler_address": info.get("address", ""),
                "worker_count": len(workers),
                "workers": workers,
                "total_cores": total_cores,
                "total_memory_gb": total_memory / (1024**3) if total_memory else 0,
                "used_memory_gb": used_memory / (1024**3) if used_memory else 0,
                "tasks": {
                    "total": sum(len(w.get("processing", {})) for w in info["workers"].values()),
                    "queued": len(info.get("tasks", {}))
                }
            }
            
        except Exception as e:
            logger.error("Failed to get Dask cluster info", error=str(e), cluster_id=cluster_id)
            return None
    
    async def shutdown_cluster(self, cluster_id: UUID) -> bool:
        """Shutdown Dask cluster."""
        logger.info("Shutting down Dask cluster", cluster_id=cluster_id)
        
        try:
            client = self.clusters.get(cluster_id)
            if not client:
                logger.warning("Cluster not found for shutdown", cluster_id=cluster_id)
                return True
            
            # Close client and cluster
            await asyncio.to_thread(client.close)
            
            # Remove from tracking
            del self.clusters[cluster_id]
            if cluster_id in self.cluster_configs:
                del self.cluster_configs[cluster_id]
            
            logger.info("Dask cluster shutdown complete", cluster_id=cluster_id)
            return True
            
        except Exception as e:
            logger.error("Failed to shutdown Dask cluster", error=str(e), cluster_id=cluster_id)
            return False
    
    # Private helper methods
    
    async def _create_local_cluster(
        self,
        cluster: ComputeCluster,
        node_config: Optional[Dict[str, Any]]
    ) -> Client:
        """Create local Dask cluster."""
        from dask.distributed import LocalCluster
        
        # Configure cluster
        cluster_kwargs = {
            'n_workers': cluster.min_nodes,
            'threads_per_worker': node_config.get('cpu_cores', 2) if node_config else 2,
            'memory_limit': f"{node_config.get('memory_gb', 4)}GB" if node_config else "4GB",
            'dashboard_address': f":{cluster.dashboard_port}" if cluster.dashboard_port else None
        }
        
        # Create local cluster
        local_cluster = LocalCluster(**cluster_kwargs)
        client = Client(local_cluster)
        
        return client
    
    async def _create_kubernetes_cluster(
        self,
        cluster: ComputeCluster,
        node_config: Optional[Dict[str, Any]]
    ) -> Client:
        """Create Kubernetes-based Dask cluster."""
        if not node_config:
            node_config = {}
        
        # Create pod spec
        pod_spec = make_pod_spec(
            image=node_config.get('image', 'daskdev/dask:latest'),
            memory_limit=f"{node_config.get('memory_gb', 4)}Gi",
            memory_request=f"{node_config.get('memory_gb', 4)}Gi",
            cpu_limit=str(node_config.get('cpu_cores', 2)),
            cpu_request=str(node_config.get('cpu_cores', 2))
        )
        
        # Create Kubernetes cluster
        kube_cluster = KubeCluster(
            pod_template=pod_spec,
            name=cluster.name.lower().replace('_', '-'),
            namespace=node_config.get('namespace', 'default')
        )
        
        # Scale to minimum nodes
        kube_cluster.scale(cluster.min_nodes)
        
        # Create client
        client = Client(kube_cluster)
        
        return client
    
    async def _wait_for_cluster_ready(
        self,
        client: Client,
        cluster: ComputeCluster,
        timeout_seconds: int = 300
    ) -> None:
        """Wait for cluster to be ready."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            try:
                info = client.scheduler_info()
                if len(info["workers"]) >= cluster.min_nodes:
                    logger.info("Dask cluster ready", cluster_id=cluster.id, workers=len(info["workers"]))
                    return
                
                await asyncio.sleep(5)
                
            except Exception:
                await asyncio.sleep(5)
                continue
        
        raise TimeoutError(f"Cluster did not become ready within {timeout_seconds} seconds")
    
    async def _wait_for_scaling_complete(
        self,
        client: Client,
        target_workers: int,
        timeout_seconds: int = 300
    ) -> None:
        """Wait for scaling operation to complete."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            try:
                info = client.scheduler_info()
                current_workers = len(info["workers"])
                
                if current_workers == target_workers:
                    logger.info("Scaling complete", current_workers=current_workers)
                    return
                
                await asyncio.sleep(10)
                
            except Exception:
                await asyncio.sleep(10)
                continue
        
        raise TimeoutError(f"Scaling did not complete within {timeout_seconds} seconds")
    
    def _get_dask_priority(self, priority: str) -> int:
        """Convert task priority to Dask priority."""
        priority_map = {
            "low": -10,
            "normal": 0,
            "high": 10,
            "critical": 20
        }
        return priority_map.get(priority, 0)