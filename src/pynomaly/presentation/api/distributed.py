"""API endpoints for distributed processing management."""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel, Field
from dependency_injector.wiring import inject, Provide

from pynomaly.infrastructure.config.container import Container
from pynomaly.infrastructure.distributed import (
    DistributedProcessingManager,
    DetectionCoordinator,
    LoadBalancer,
    LoadBalancingStrategy
)
from pynomaly.domain.entities import Detector, Dataset
from pynomaly.application.dto import DetectorResponseDTO, DatasetResponseDTO


router = APIRouter(prefix="/distributed", tags=["distributed"])


# Request/Response Models
class WorkerRegistrationRequest(BaseModel):
    """Request model for worker registration."""
    worker_id: str = Field(..., description="Unique worker identifier")
    host: str = Field(..., description="Worker host address")
    port: int = Field(..., description="Worker port number")
    capacity: int = Field(4, description="Worker processing capacity")
    capabilities: List[str] = Field(default_factory=list, description="Worker capabilities")


class WorkerHeartbeatRequest(BaseModel):
    """Request model for worker heartbeat."""
    status: str = Field(..., description="Worker status (idle, busy, offline)")
    current_load: int = Field(0, description="Current load on worker")


class DistributedDetectionRequest(BaseModel):
    """Request model for distributed detection."""
    detector_id: str = Field(..., description="Detector ID to use")
    dataset_id: str = Field(..., description="Dataset ID to process")
    chunk_size: Optional[int] = Field(None, description="Optional chunk size for splitting")
    priority: int = Field(5, description="Task priority (1-10)")


class ServerInstanceRequest(BaseModel):
    """Request model for adding server instance."""
    server_id: str = Field(..., description="Unique server identifier")
    host: str = Field(..., description="Server host address")
    port: int = Field(..., description="Server port number")
    weight: int = Field(1, description="Server weight for load balancing")
    max_connections: int = Field(100, description="Maximum concurrent connections")


class LoadBalancerConfigRequest(BaseModel):
    """Request model for load balancer configuration."""
    strategy: str = Field(..., description="Load balancing strategy")
    health_check_interval: int = Field(30, description="Health check interval in seconds")
    health_check_timeout: int = Field(10, description="Health check timeout in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts")


# Distributed Processing Manager Endpoints

@router.post("/workers/register")
@inject
async def register_worker(
    request: WorkerRegistrationRequest,
    background_tasks: BackgroundTasks,
    manager: DistributedProcessingManager = Depends(Provide[Container.distributed_processing_manager])
) -> Dict[str, Any]:
    """Register a new worker node."""
    success = await manager.register_worker(
        worker_id=request.worker_id,
        host=request.host,
        port=request.port,
        capacity=request.capacity,
        capabilities=request.capabilities
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to register worker"
        )
    
    return {
        "message": f"Worker {request.worker_id} registered successfully",
        "worker_id": request.worker_id
    }


@router.delete("/workers/{worker_id}")
@inject
async def unregister_worker(
    worker_id: str,
    manager: DistributedProcessingManager = Depends(Provide[Container.distributed_processing_manager])
) -> Dict[str, Any]:
    """Unregister a worker node."""
    success = await manager.unregister_worker(worker_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Worker not found"
        )
    
    return {"message": f"Worker {worker_id} unregistered successfully"}


@router.post("/workers/{worker_id}/heartbeat")
@inject
async def worker_heartbeat(
    worker_id: str,
    request: WorkerHeartbeatRequest,
    manager: DistributedProcessingManager = Depends(Provide[Container.distributed_processing_manager])
) -> Dict[str, Any]:
    """Process worker heartbeat."""
    success = await manager.heartbeat(
        worker_id=worker_id,
        status=request.status,
        current_load=request.current_load
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Worker not found"
        )
    
    return {"message": "Heartbeat processed successfully"}


@router.get("/workers/status")
@inject
async def get_workers_status(
    manager: DistributedProcessingManager = Depends(Provide[Container.distributed_processing_manager])
) -> Dict[str, Any]:
    """Get status of all workers."""
    return await manager.get_worker_status()


@router.get("/tasks/{task_id}/status")
@inject
async def get_task_status(
    task_id: str,
    manager: DistributedProcessingManager = Depends(Provide[Container.distributed_processing_manager])
) -> Dict[str, Any]:
    """Get status of a specific task."""
    status = await manager.get_task_status(task_id)
    
    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    return status


@router.get("/system/metrics")
@inject
async def get_system_metrics(
    manager: DistributedProcessingManager = Depends(Provide[Container.distributed_processing_manager])
) -> Dict[str, Any]:
    """Get comprehensive system metrics."""
    return await manager.get_system_metrics()


# Detection Coordinator Endpoints

@router.post("/detection/submit")
@inject
async def submit_distributed_detection(
    request: DistributedDetectionRequest,
    coordinator: DetectionCoordinator = Depends(Provide[Container.detection_coordinator])
) -> Dict[str, Any]:
    """Submit a distributed detection job."""
    # TODO: Get detector and dataset from repositories
    # For now, create mock objects
    from pynomaly.domain.entities import Detector, Dataset
    from pynomaly.domain.value_objects import ContaminationRate
    
    detector = Detector(
        id=request.detector_id,
        name="Mock Detector",
        algorithm_name="IsolationForest",
        contamination_rate=ContaminationRate(0.1)
    )
    
    dataset = Dataset(
        id=request.dataset_id,
        name="Mock Dataset",
        data=None  # Would be loaded from repository
    )
    
    job_id = await coordinator.submit_distributed_detection(
        detector=detector,
        dataset=dataset,
        chunk_size=request.chunk_size,
        priority=request.priority
    )
    
    return {
        "job_id": job_id,
        "message": "Distributed detection job submitted successfully"
    }


@router.get("/detection/jobs/{job_id}/status")
@inject
async def get_job_status(
    job_id: str,
    coordinator: DetectionCoordinator = Depends(Provide[Container.detection_coordinator])
) -> Dict[str, Any]:
    """Get status of a distributed detection job."""
    status = await coordinator.get_job_status(job_id)
    
    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return status


@router.get("/detection/jobs/{job_id}/result")
@inject
async def get_job_result(
    job_id: str,
    coordinator: DetectionCoordinator = Depends(Provide[Container.detection_coordinator])
) -> Dict[str, Any]:
    """Get result of a completed distributed detection job."""
    result = await coordinator.get_job_result(job_id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job result not found or job not completed"
        )
    
    return {
        "job_id": job_id,
        "result": {
            "id": result.id,
            "detector_id": result.detector_id,
            "dataset_id": result.dataset_id,
            "n_anomalies": result.n_anomalies,
            "anomaly_rate": result.anomaly_rate,
            "execution_time": result.execution_time
        }
    }


@router.delete("/detection/jobs/{job_id}")
@inject
async def cancel_job(
    job_id: str,
    coordinator: DetectionCoordinator = Depends(Provide[Container.detection_coordinator])
) -> Dict[str, Any]:
    """Cancel a distributed detection job."""
    success = await coordinator.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return {"message": f"Job {job_id} cancelled successfully"}


@router.get("/detection/jobs")
@inject
async def list_jobs(
    status_filter: Optional[str] = None,
    coordinator: DetectionCoordinator = Depends(Provide[Container.detection_coordinator])
) -> Dict[str, Any]:
    """List all distributed detection jobs."""
    jobs = await coordinator.list_jobs(status=status_filter)
    
    return {
        "jobs": jobs,
        "total": len(jobs)
    }


# Load Balancer Endpoints

@router.post("/loadbalancer/servers")
@inject
async def add_server(
    request: ServerInstanceRequest,
    load_balancer: LoadBalancer = Depends(Provide[Container.load_balancer])
) -> Dict[str, Any]:
    """Add a server to the load balancer."""
    success = await load_balancer.add_server(
        server_id=request.server_id,
        host=request.host,
        port=request.port,
        weight=request.weight,
        max_connections=request.max_connections
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add server"
        )
    
    return {
        "message": f"Server {request.server_id} added successfully",
        "server_id": request.server_id
    }


@router.delete("/loadbalancer/servers/{server_id}")
@inject
async def remove_server(
    server_id: str,
    load_balancer: LoadBalancer = Depends(Provide[Container.load_balancer])
) -> Dict[str, Any]:
    """Remove a server from the load balancer."""
    success = await load_balancer.remove_server(server_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Server not found"
        )
    
    return {"message": f"Server {server_id} removed successfully"}


@router.get("/loadbalancer/servers/status")
@inject
async def get_servers_status(
    load_balancer: LoadBalancer = Depends(Provide[Container.load_balancer])
) -> Dict[str, Any]:
    """Get status of all servers in the load balancer."""
    return await load_balancer.get_server_status()


@router.post("/loadbalancer/forward")
@inject
async def forward_request(
    method: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, str]] = None,
    session_id: Optional[str] = None,
    load_balancer: LoadBalancer = Depends(Provide[Container.load_balancer])
) -> Dict[str, Any]:
    """Forward a request through the load balancer."""
    try:
        result = await load_balancer.forward_request(
            method=method,
            path=path,
            headers=headers,
            json_data=json_data,
            params=params,
            session_id=session_id
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Request forwarding failed: {str(e)}"
        )


# System Management Endpoints

@router.post("/system/start")
@inject
async def start_distributed_system(
    manager: DistributedProcessingManager = Depends(Provide[Container.distributed_processing_manager]),
    coordinator: DetectionCoordinator = Depends(Provide[Container.detection_coordinator]),
    load_balancer: LoadBalancer = Depends(Provide[Container.load_balancer])
) -> Dict[str, Any]:
    """Start the distributed processing system."""
    await manager.start()
    await coordinator.start()
    await load_balancer.start()
    
    return {"message": "Distributed processing system started successfully"}


@router.post("/system/stop")
@inject
async def stop_distributed_system(
    manager: DistributedProcessingManager = Depends(Provide[Container.distributed_processing_manager]),
    coordinator: DetectionCoordinator = Depends(Provide[Container.detection_coordinator]),
    load_balancer: LoadBalancer = Depends(Provide[Container.load_balancer])
) -> Dict[str, Any]:
    """Stop the distributed processing system."""
    await coordinator.stop()
    await manager.stop()
    await load_balancer.stop()
    
    return {"message": "Distributed processing system stopped successfully"}


@router.get("/system/status")
@inject
async def get_distributed_system_status(
    coordinator: DetectionCoordinator = Depends(Provide[Container.detection_coordinator])
) -> Dict[str, Any]:
    """Get comprehensive distributed system status."""
    return await coordinator.get_system_metrics()