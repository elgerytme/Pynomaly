"""FastAPI application for MLOps service."""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from mlops.infrastructure.container.container import MLOpsContainer
from mlops.domain.entities.pipeline_config import PipelineConfig
from mlops.domain.entities.pipeline_execution import PipelineExecution
from mlops.domain.interfaces.pipeline_operations import (
    PipelineManagementPort, PipelineExecutionPort, PipelineMonitoringPort
)
from mlops.domain.interfaces.configuration_operations import (
    ConfigurationProviderPort, ServiceDiscoveryPort
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Service",
    description="Hexagonal Architecture MLOps API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize container
container = MLOpsContainer()

# Dependency injection
def get_pipeline_management_service() -> PipelineManagementPort:
    return container.get(PipelineManagementPort)

def get_pipeline_execution_service() -> PipelineExecutionPort:
    return container.get(PipelineExecutionPort)

def get_pipeline_monitoring_service() -> PipelineMonitoringPort:
    return container.get(PipelineMonitoringPort)

def get_configuration_service() -> ConfigurationProviderPort:
    return container.get(ConfigurationProviderPort)

def get_service_discovery_service() -> ServiceDiscoveryPort:
    return container.get(ServiceDiscoveryPort)

# Health and readiness endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mlops", "timestamp": datetime.utcnow()}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Perform basic service checks
        pipeline_service = get_pipeline_management_service()
        return {"status": "ready", "service": "mlops", "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Basic metrics - in production, use prometheus_client library
    return {
        "mlops_pipeline_executions_total": 75,
        "mlops_pipeline_executions_failed_total": 3,
        "mlops_pipelines_active": 5,
        "mlops_services_discovered": 4,
        "mlops_configuration_updates_total": 20
    }

# Pipeline Management API
@app.post("/api/v1/pipelines", status_code=status.HTTP_201_CREATED)
async def create_pipeline(
    request: Dict[str, Any],
    management_service: PipelineManagementPort = Depends(get_pipeline_management_service)
):
    """Create a new ML pipeline."""
    try:
        # Validate request
        if "pipeline_name" not in request or "steps" not in request:
            raise HTTPException(status_code=400, detail="pipeline_name and steps are required")
        
        # Create pipeline config
        pipeline_config = PipelineConfig(
            pipeline_id=f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            name=request["pipeline_name"],
            steps=request["steps"],
            schedule=request.get("schedule"),
            environment=request.get("environment", "development")
        )
        
        # Create pipeline
        success = await management_service.create_pipeline(pipeline_config)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create pipeline")
        
        return {
            "status": "success",
            "data": {
                "pipeline_id": pipeline_config.pipeline_id,
                "pipeline_name": pipeline_config.name,
                "steps": pipeline_config.steps,
                "environment": pipeline_config.environment,
                "created_at": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipelines")
async def list_pipelines(
    management_service: PipelineManagementPort = Depends(get_pipeline_management_service)
):
    """List all pipelines."""
    try:
        pipelines = await management_service.list_pipelines()
        
        return {
            "status": "success",
            "data": {
                "pipelines": [
                    {
                        "pipeline_id": pipeline.pipeline_id,
                        "name": pipeline.name,
                        "steps": pipeline.steps,
                        "schedule": pipeline.schedule,
                        "environment": pipeline.environment,
                        "is_active": pipeline.is_active
                    }
                    for pipeline in pipelines
                ],
                "total_count": len(pipelines),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to list pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipelines/{pipeline_id}")
async def get_pipeline(
    pipeline_id: str,
    management_service: PipelineManagementPort = Depends(get_pipeline_management_service)
):
    """Get pipeline details."""
    try:
        pipeline = await management_service.get_pipeline(pipeline_id)
        
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return {
            "status": "success",
            "data": {
                "pipeline_id": pipeline.pipeline_id,
                "name": pipeline.name,
                "steps": pipeline.steps,
                "schedule": pipeline.schedule,
                "environment": pipeline.environment,
                "is_active": pipeline.is_active,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/pipelines/{pipeline_id}")
async def delete_pipeline(
    pipeline_id: str,
    management_service: PipelineManagementPort = Depends(get_pipeline_management_service)
):
    """Delete a pipeline."""
    try:
        success = await management_service.delete_pipeline(pipeline_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return {
            "status": "success",
            "data": {
                "pipeline_id": pipeline_id,
                "deleted": success,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Pipeline Execution API
@app.post("/api/v1/pipelines/{pipeline_id}/execute")
async def execute_pipeline(
    pipeline_id: str,
    request: Dict[str, Any] = None,
    execution_service: PipelineExecutionPort = Depends(get_pipeline_execution_service)
):
    """Execute a pipeline."""
    try:
        # Execute pipeline
        execution_result = await execution_service.execute_pipeline(
            pipeline_id, 
            request.get("parameters", {}) if request else {}
        )
        
        return {
            "status": "success",
            "data": {
                "execution_id": execution_result.execution_id,
                "pipeline_id": execution_result.pipeline_id,
                "status": execution_result.status,
                "started_at": execution_result.started_at,
                "parameters": execution_result.parameters
            }
        }
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipelines/{pipeline_id}/executions")
async def get_pipeline_executions(
    pipeline_id: str,
    monitoring_service: PipelineMonitoringPort = Depends(get_pipeline_monitoring_service)
):
    """Get pipeline execution history."""
    try:
        executions = await monitoring_service.get_pipeline_executions(pipeline_id)
        
        return {
            "status": "success",
            "data": {
                "pipeline_id": pipeline_id,
                "executions": [
                    {
                        "execution_id": execution.execution_id,
                        "status": execution.status,
                        "started_at": execution.started_at,
                        "completed_at": execution.completed_at,
                        "duration_seconds": execution.duration_seconds,
                        "success": execution.success
                    }
                    for execution in executions
                ],
                "total_count": len(executions),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get pipeline executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/executions/{execution_id}")
async def get_execution_details(
    execution_id: str,
    monitoring_service: PipelineMonitoringPort = Depends(get_pipeline_monitoring_service)
):
    """Get execution details."""
    try:
        execution = await monitoring_service.get_execution_details(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return {
            "status": "success",
            "data": {
                "execution_id": execution.execution_id,
                "pipeline_id": execution.pipeline_id,
                "status": execution.status,
                "started_at": execution.started_at,
                "completed_at": execution.completed_at,
                "duration_seconds": execution.duration_seconds,
                "parameters": execution.parameters,
                "results": execution.results,
                "success": execution.success,
                "error_message": execution.error_message
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration API
@app.get("/api/v1/config")
async def get_configuration(
    config_service: ConfigurationProviderPort = Depends(get_configuration_service)
):
    """Get service configuration."""
    try:
        config = await config_service.get_configuration("mlops")
        
        return {
            "status": "success",
            "data": {
                "service": "mlops",
                "configuration": config,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/config")
async def update_configuration(
    request: Dict[str, Any],
    config_service: ConfigurationProviderPort = Depends(get_configuration_service)
):
    """Update service configuration."""
    try:
        success = await config_service.update_configuration("mlops", request)
        
        return {
            "status": "success",
            "data": {
                "updated": success,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Service Discovery API
@app.get("/api/v1/services")
async def discover_services(
    discovery_service: ServiceDiscoveryPort = Depends(get_service_discovery_service)
):
    """Discover available services."""
    try:
        services = await discovery_service.discover_services()
        
        return {
            "status": "success",
            "data": {
                "services": [
                    {
                        "service_name": service.service_name,
                        "service_url": service.service_url,
                        "health_check_url": service.health_check_url,
                        "version": service.version,
                        "is_healthy": service.is_healthy
                    }
                    for service in services
                ],
                "total_count": len(services),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to discover services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/services/{service_name}")
async def get_service_info(
    service_name: str,
    discovery_service: ServiceDiscoveryPort = Depends(get_service_discovery_service)
):
    """Get information about a specific service."""
    try:
        service = await discovery_service.get_service(service_name)
        
        if not service:
            raise HTTPException(status_code=404, detail="Service not found")
        
        return {
            "status": "success",
            "data": {
                "service_name": service.service_name,
                "service_url": service.service_url,
                "health_check_url": service.health_check_url,
                "version": service.version,
                "is_healthy": service.is_healthy,
                "last_health_check": service.last_health_check,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def get_service_status():
    """Get service status and configuration."""
    return {
        "status": "running",
        "service": "mlops",
        "version": "1.0.0",
        "environment": "development",
        "timestamp": datetime.utcnow(),
        "capabilities": [
            "pipeline_management",
            "pipeline_execution",
            "pipeline_monitoring",
            "configuration_management",
            "service_discovery"
        ]
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )