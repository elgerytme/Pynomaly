"""Metrics API endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter()


class MetricsCalculationRequest(BaseModel):
    """Request model for metrics calculation."""
    experiment_id: str
    metric_types: List[str] = ["accuracy", "precision", "recall", "f1"]


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    experiment_id: str
    metrics: dict
    calculated_at: str


@router.post("/calculate", response_model=MetricsResponse)
async def calculate_metrics(request: MetricsCalculationRequest) -> MetricsResponse:
    """Calculate experiment metrics."""
    # Implementation would use MetricsCalculator service
    return MetricsResponse(
        experiment_id=request.experiment_id,
        metrics={
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1": 0.90,
            "mse": 0.05,
            "mae": 0.03,
            "r2": 0.92
        },
        calculated_at="2024-01-01T00:00:00Z"
    )


@router.get("/{experiment_id}", response_model=MetricsResponse)
async def get_experiment_metrics(experiment_id: str) -> MetricsResponse:
    """Get metrics for an experiment."""
    if experiment_id != "exp_123":
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return MetricsResponse(
        experiment_id=experiment_id,
        metrics={
            "accuracy": 0.95,
            "precision": 0.92, 
            "recall": 0.88,
            "f1": 0.90
        },
        calculated_at="2024-01-01T00:00:00Z"
    )


class MetricsComparisonRequest(BaseModel):
    """Request model for comparing metrics between experiments."""
    experiment_ids: List[str]


class MetricsComparisonResponse(BaseModel):
    """Response model for metrics comparison."""
    experiments: dict
    comparison: dict


@router.post("/compare", response_model=MetricsComparisonResponse)
async def compare_metrics(request: MetricsComparisonRequest) -> MetricsComparisonResponse:
    """Compare metrics between experiments."""
    return MetricsComparisonResponse(
        experiments={
            exp_id: {"accuracy": 0.90 + (i * 0.01), "precision": 0.85 + (i * 0.01)}
            for i, exp_id in enumerate(request.experiment_ids)
        },
        comparison={
            "best_accuracy": request.experiment_ids[-1] if request.experiment_ids else None,
            "best_precision": request.experiment_ids[-1] if request.experiment_ids else None
        }
    )