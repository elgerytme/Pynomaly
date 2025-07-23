"""In-memory repository implementation for deployments."""

from typing import Dict, List, Optional
from uuid import UUID

from ...domain.entities.deployment import Deployment
from ...domain.repositories.deployment_repository import DeploymentRepository


class InMemoryDeploymentRepository(DeploymentRepository):
    """In-memory implementation of deployment repository."""
    
    def __init__(self):
        self._deployments: Dict[str, Deployment] = {}
    
    async def save(self, deployment: Deployment) -> Deployment:
        """Save a deployment."""
        deployment_id_str = str(deployment.id)
        self._deployments[deployment_id_str] = deployment
        return deployment
    
    async def find_by_id(self, deployment_id: UUID) -> Optional[Deployment]:
        """Find deployment by ID."""
        return self._deployments.get(str(deployment_id))
    
    async def find_by_name(self, name: str) -> Optional[Deployment]:
        """Find deployment by name."""
        for deployment in self._deployments.values():
            if deployment.name == name:
                return deployment
        return None
    
    async def find_all(self) -> List[Deployment]:
        """Find all deployments."""
        return list(self._deployments.values())
    
    async def find_by_model_id(self, model_id: UUID) -> List[Deployment]:
        """Find deployments by model ID."""
        return [
            deployment for deployment in self._deployments.values()
            if deployment.model_id == model_id
        ]
    
    async def find_active(self) -> List[Deployment]:
        """Find active deployments."""
        return [
            deployment for deployment in self._deployments.values()
            if deployment.is_active
        ]
    
    async def find_by_environment(self, environment: str) -> List[Deployment]:
        """Find deployments by environment."""
        return [
            deployment for deployment in self._deployments.values()
            if deployment.environment.value == environment
        ]
    
    async def delete(self, deployment_id: UUID) -> None:
        """Delete a deployment."""
        deployment_id_str = str(deployment_id)
        if deployment_id_str in self._deployments:
            del self._deployments[deployment_id_str]