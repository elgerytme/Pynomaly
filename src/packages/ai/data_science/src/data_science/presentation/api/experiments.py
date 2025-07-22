"""Experiments API endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel

from ...application.services.integrated_data_science_service import IntegratedDataScienceService
from ...application.services.workflow_orchestration_engine import WorkflowOrchestrationEngine

router = APIRouter()


class ExperimentCreateRequest(BaseModel):
    """Request model for creating experiments."""
    name: str
    description: Optional[str] = None
    config: dict = {}


class ExperimentResponse(BaseModel):
    """Response model for experiments."""
    id: str
    name: str
    description: Optional[str]
    status: str
    created_at: str
    updated_at: str


@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreateRequest,
    # service: IntegratedDataScienceService = Depends()
) -> ExperimentResponse:
    """Create a new experiment."""
    # Implementation would use the integrated data science service
    return ExperimentResponse(
        id="exp_123",
        name=request.name,
        description=request.description,
        status="created",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z"
    )


@router.get("/", response_model=List[ExperimentResponse])
async def list_experiments() -> List[ExperimentResponse]:
    """List all experiments."""
    # Implementation would query experiments from repository
    return [
        ExperimentResponse(
            id="exp_123",
            name="Sample Experiment",
            description="A sample experiment",
            status="completed", 
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z"
        )
    ]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str) -> ExperimentResponse:
    """Get experiment by ID."""
    if experiment_id != "exp_123":
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return ExperimentResponse(
        id=experiment_id,
        name="Sample Experiment",
        description="A sample experiment",
        status="completed",
        created_at="2024-01-01T00:00:00Z", 
        updated_at="2024-01-01T00:00:00Z"
    )


@router.post("/{experiment_id}/run")
async def run_experiment(experiment_id: str) -> dict:
    """Run an experiment."""
    if experiment_id != "exp_123":
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Implementation would use workflow orchestration engine
    return {"message": f"Experiment {experiment_id} started", "status": "running"}


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str) -> dict:
    """Delete an experiment."""
    if experiment_id != "exp_123":
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return {"message": f"Experiment {experiment_id} deleted"}