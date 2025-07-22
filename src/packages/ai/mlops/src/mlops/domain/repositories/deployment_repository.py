"""Deployment repository interface for MLOps domain."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.deployment import Deployment


class DeploymentRepository(ABC):
    """Abstract repository for deployments."""
    
    @abstractmethod
    async def save(self, deployment: Deployment) -> Deployment:
        """Save a deployment."""
        pass
    
    @abstractmethod
    async def find_by_id(self, deployment_id: UUID) -> Optional[Deployment]:
        """Find deployment by ID."""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Deployment]:
        """Find deployment by name."""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[Deployment]:
        """Find all deployments."""
        pass
    
    @abstractmethod
    async def find_by_model_id(self, model_id: UUID) -> List[Deployment]:
        """Find deployments by model ID."""
        pass
    
    @abstractmethod
    async def find_active(self) -> List[Deployment]:
        """Find active deployments."""
        pass
    
    @abstractmethod
    async def find_by_environment(self, environment: str) -> List[Deployment]:
        """Find deployments by environment."""
        pass
    
    @abstractmethod
    async def delete(self, deployment_id: UUID) -> None:
        """Delete a deployment."""
        pass