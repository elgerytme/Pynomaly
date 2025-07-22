"""Use case for creating a new ML model."""

from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel

from ...domain.services.model_management_service import ModelManagementService


class CreateModelRequest(BaseModel):
    """Request for creating a new model."""
    model_config = {"protected_namespaces": ()}
    
    name: str
    description: str
    model_type: str
    algorithm_family: str
    created_by: str
    team: str = ""
    use_cases: list[str] = []


class CreateModelResponse(BaseModel):
    """Response for model creation."""
    model_config = {"protected_namespaces": ()}
    
    model_id: str
    name: str
    status: str
    created_at: str


class CreateModelUseCase:
    """Use case for creating a new ML model."""
    
    def __init__(self, model_service: ModelManagementService):
        """Initialize the use case.
        
        Args:
            model_service: Domain service for model management
        """
        self.model_service = model_service
    
    async def execute(self, request: CreateModelRequest) -> CreateModelResponse:
        """Execute the create model use case.
        
        Args:
            request: Model creation request
            
        Returns:
            Model creation response
        """
        model = await self.model_service.create_model(
            name=request.name,
            description=request.description,
            model_type=request.model_type,
            algorithm_family=request.algorithm_family,
            created_by=request.created_by,
            team=request.team,
            use_cases=request.use_cases,
        )
        
        return CreateModelResponse(
            model_id=str(model.id),
            name=model.name,
            status=str(model.status),
            created_at=model.created_at.isoformat(),
        )