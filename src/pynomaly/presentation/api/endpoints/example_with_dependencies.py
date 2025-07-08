"""
Example router showing how to use the forward-reference-free dependency system.

This demonstrates the pattern for declaring dependencies without type hints
in router files, avoiding circular import issues.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

# Import the dependency wrapper system
from pynomaly.infrastructure.dependencies import (
    DependencyWrapper,
    auth_service,
    detection_service,
    model_service,
    database_service,
)

router = APIRouter()

# Response models
class DetectionResponse(BaseModel):
    """Response model for detection results."""
    success: bool
    message: str
    anomaly_count: Optional[int] = None
    model_id: Optional[str] = None


class TrainingResponse(BaseModel):
    """Response model for training results."""
    success: bool
    message: str
    model_id: str
    accuracy: Optional[float] = None


# Declare dependencies without type hints - this avoids circular imports
get_auth_service = auth_service()
get_detection_service = detection_service()
get_model_service = model_service()
get_database_service = database_service()

# Alternative explicit declaration
get_custom_service = DependencyWrapper("custom_service", optional=True)


@router.post("/detect", response_model=DetectionResponse)
async def detect_anomalies(
    dataset_id: str,
    # Dependencies declared without type hints
    auth_svc = get_auth_service(),
    detection_svc = get_detection_service(),
    db_svc = get_database_service(),
) -> DetectionResponse:
    """
    Detect anomalies in a dataset.
    
    This endpoint demonstrates the use of the dependency wrapper system:
    - No type hints on dependencies (avoids circular imports)
    - Services are injected at runtime after full app initialization
    - Graceful handling of optional services
    
    Args:
        dataset_id: ID of the dataset to analyze
        auth_svc: Authentication service (optional)
        detection_svc: Detection service (required)
        db_svc: Database service (optional)
        
    Returns:
        Detection results
    """
    try:
        # Use the authentication service if available
        if auth_svc:
            # Authenticate request (implementation depends on your auth system)
            pass
        
        # Use detection service (this is required)
        if not detection_svc:
            return DetectionResponse(
                success=False,
                message="Detection service not available"
            )
        
        # Simulate detection process
        # In real implementation, this would call detection_svc.detect()
        anomaly_count = 42  # Mock result
        
        # Use database service if available
        if db_svc:
            # Store results in database
            pass
        
        return DetectionResponse(
            success=True,
            message="Anomaly detection completed successfully",
            anomaly_count=anomaly_count
        )
        
    except Exception as e:
        return DetectionResponse(
            success=False,
            message=f"Detection failed: {str(e)}"
        )


@router.post("/train", response_model=TrainingResponse)
async def train_model(
    dataset_id: str,
    algorithm: str,
    # Dependencies without type hints
    auth_svc = get_auth_service(),
    detection_svc = get_detection_service(),
    model_svc = get_model_service(),
) -> TrainingResponse:
    """
    Train an anomaly detection model.
    
    Args:
        dataset_id: ID of the dataset to train on
        algorithm: Algorithm to use for training
        auth_svc: Authentication service (optional)
        detection_svc: Detection service (required)
        model_svc: Model service (required)
        
    Returns:
        Training results
    """
    try:
        # Use authentication service if available
        if auth_svc:
            # Authenticate request
            pass
        
        # Check required services
        if not detection_svc:
            return TrainingResponse(
                success=False,
                message="Detection service not available",
                model_id=""
            )
        
        if not model_svc:
            return TrainingResponse(
                success=False,
                message="Model service not available",
                model_id=""
            )
        
        # Simulate training process
        # In real implementation, this would call detection_svc.train()
        model_id = f"model_{dataset_id}_{algorithm}"
        accuracy = 0.95  # Mock accuracy
        
        return TrainingResponse(
            success=True,
            message="Model training completed successfully",
            model_id=model_id,
            accuracy=accuracy
        )
        
    except Exception as e:
        return TrainingResponse(
            success=False,
            message=f"Training failed: {str(e)}",
            model_id=""
        )


@router.get("/health")
async def health_check(
    # Optional service dependency
    custom_svc = get_custom_service(),
) -> dict:
    """
    Health check endpoint showing optional dependency usage.
    
    Args:
        custom_svc: Custom service (optional)
        
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "custom_service_available": custom_svc is not None,
        "message": "Example endpoint is working"
    }


# Example of how to add the router to your FastAPI app:
# 
# from fastapi import FastAPI
# from .example_with_dependencies import router as example_router
# 
# app = FastAPI()
# app.include_router(example_router, prefix="/api/v1/example", tags=["example"])
