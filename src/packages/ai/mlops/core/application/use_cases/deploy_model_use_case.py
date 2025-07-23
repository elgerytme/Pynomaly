"""Deploy model use case."""

from typing import Dict, Any, Optional
from uuid import UUID
from pydantic import BaseModel, Field

from ...domain.entities.model import Model
from ...domain.entities.deployment import Deployment
from ...domain.services.model_management_service import ModelManagementService
from ...domain.value_objects.deployment_value_objects import DeploymentEnvironment
from ..services.model_deployment_service import ModelDeploymentService, DeploymentStrategy


class DeployModelRequest(BaseModel):
    """Request for deploying a model."""
    model_id: UUID
    environment: DeploymentEnvironment
    deployment_name: Optional[str] = None
    version: str = "1.0.0"
    replicas: int = Field(default=1, ge=1)
    resources: Dict[str, Any] = Field(default_factory=dict)
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    endpoint_url: Optional[str] = None
    auto_scaling: bool = False
    monitoring_enabled: bool = True
    health_check_config: Optional[Dict[str, Any]] = None


class DeployModelResponse(BaseModel):
    """Response for model deployment."""
    deployment_id: UUID
    model_id: UUID
    deployment_name: str
    environment: str
    version: str
    endpoint_url: str
    status: str
    replicas: int
    deployment_time: str
    health_status: Dict[str, Any]


class DeployModelUseCase:
    """Use case for deploying models."""
    
    def __init__(
        self,
        model_service: ModelManagementService,
        deployment_service: ModelDeploymentService,
    ):
        self.model_service = model_service
        self.deployment_service = deployment_service
    
    async def execute(self, request: DeployModelRequest) -> DeployModelResponse:
        """Execute model deployment use case."""
        # Validate model exists and is ready for deployment
        model = await self.model_service.get_model(request.model_id)
        if not model:
            raise ValueError(f"Model {request.model_id} not found")
        
        # Prepare deployment configuration
        deployment_config = {
            "version": request.version,
            "replicas": request.replicas,
            "resources": request.resources,
            "endpoint_url": request.endpoint_url or self._generate_endpoint_url(
                model.name, request.environment
            ),
            "auto_scaling": request.auto_scaling,
            "monitoring_enabled": request.monitoring_enabled,
            "health_check_config": request.health_check_config or self._default_health_check_config(),
        }
        
        # Execute deployment
        deployment = await self.deployment_service.deploy_model(
            model_id=request.model_id,
            environment=request.environment,
            deployment_config=deployment_config,
            strategy=request.strategy,
        )
        
        # Update model deployment status
        await self.model_service.update_model(
            model_id=request.model_id,
            updates={
                "deployment_status": "deployed",
                "last_deployed": deployment.created_at.isoformat(),
                "active_deployments": [str(deployment.id)],
            },
        )
        
        # Get initial health status
        health_status = await self.deployment_service.get_deployment_health(
            deployment.id
        )
        
        return DeployModelResponse(
            deployment_id=deployment.id,
            model_id=request.model_id,
            deployment_name=deployment.name,
            environment=deployment.environment.value,
            version=deployment.version,
            endpoint_url=deployment.endpoint_url,
            status="active" if deployment.is_active else "inactive",
            replicas=deployment.replicas,
            deployment_time=deployment.created_at.isoformat(),
            health_status=health_status,
        )
    
    async def rollback_deployment(
        self,
        deployment_id: UUID,
        target_version: Optional[str] = None,
    ) -> DeployModelResponse:
        """Rollback a deployment to previous version."""
        deployment = await self.deployment_service.rollback_deployment(
            deployment_id=deployment_id,
            target_version=target_version,
        )
        
        health_status = await self.deployment_service.get_deployment_health(
            deployment.id
        )
        
        return DeployModelResponse(
            deployment_id=deployment.id,
            model_id=deployment.model_id,
            deployment_name=deployment.name,
            environment=deployment.environment.value,
            version=deployment.version,
            endpoint_url=deployment.endpoint_url,
            status="active" if deployment.is_active else "inactive",
            replicas=deployment.replicas,
            deployment_time=deployment.created_at.isoformat(),
            health_status=health_status,
        )
    
    async def scale_deployment(
        self,
        deployment_id: UUID,
        target_replicas: int,
    ) -> DeployModelResponse:
        """Scale deployment to target replica count."""
        deployment = await self.deployment_service.scale_deployment(
            deployment_id=deployment_id,
            target_replicas=target_replicas,
        )
        
        health_status = await self.deployment_service.get_deployment_health(
            deployment.id
        )
        
        return DeployModelResponse(
            deployment_id=deployment.id,
            model_id=deployment.model_id,
            deployment_name=deployment.name,
            environment=deployment.environment.value,
            version=deployment.version,
            endpoint_url=deployment.endpoint_url,
            status="active" if deployment.is_active else "inactive",
            replicas=deployment.replicas,
            deployment_time=deployment.created_at.isoformat(),
            health_status=health_status,
        )
    
    def _generate_endpoint_url(self, model_name: str, environment: DeploymentEnvironment) -> str:
        """Generate endpoint URL for deployment."""
        env_prefix = environment.value
        return f"https://{env_prefix}.api.company.com/models/{model_name.lower()}/predict"
    
    def _default_health_check_config(self) -> Dict[str, Any]:
        """Generate default health check configuration."""
        return {
            "path": "/health",
            "interval_seconds": 30,
            "timeout_seconds": 5,
            "retries": 3,
            "port": 8080,
        }