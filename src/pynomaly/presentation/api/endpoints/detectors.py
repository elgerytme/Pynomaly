"""Detector management endpoints."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from pynomaly.application.dto import CreateDetectorDTO, DetectorDTO, UpdateDetectorDTO
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters import PyODAdapter, SklearnAdapter
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.deps import get_container, get_current_user
from pynomaly.infrastructure.auth import require_read, require_write, require_admin, PermissionChecker


router = APIRouter()


@router.get("/", response_model=List[DetectorDTO])
async def list_detectors(
    algorithm: Optional[str] = Query(None, description="Filter by algorithm"),
    is_fitted: Optional[bool] = Query(None, description="Filter by fitted status"),
    limit: int = Query(100, ge=1, le=1000),
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user),
    _permissions: str = Depends(require_read)
) -> List[DetectorDTO]:
    """List all detectors."""
    detector_repo = container.detector_repository()
    
    # Get all detectors
    detectors = detector_repo.find_all()
    
    # Apply filters
    if algorithm:
        detectors = [d for d in detectors if d.algorithm_name == algorithm]
    
    if is_fitted is not None:
        detectors = [d for d in detectors if d.is_fitted == is_fitted]
    
    # Limit results
    detectors = detectors[:limit]
    
    # Convert to DTOs
    return [
        DetectorDTO(
            id=d.id,
            name=d.name,
            algorithm_name=d.algorithm_name,
            contamination_rate=d.contamination_rate.value,
            is_fitted=d.is_fitted,
            created_at=d.created_at,
            trained_at=d.trained_at,
            parameters=d.parameters,
            metadata=d.metadata,
            requires_fitting=d.requires_fitting,
            supports_streaming=d.supports_streaming,
            supports_multivariate=d.supports_multivariate,
            time_complexity=d.time_complexity,
            space_complexity=d.space_complexity
        )
        for d in detectors
    ]


@router.get("/algorithms")
async def list_algorithms() -> dict:
    """List available algorithms."""
    return {
        "pyod": list(PyODAdapter.ALGORITHM_MAPPING.keys()),
        "sklearn": list(SklearnAdapter.ALGORITHM_MAPPING.keys())
    }


@router.get("/{detector_id}", response_model=DetectorDTO)
async def get_detector(
    detector_id: UUID,
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user),
    _permissions: str = Depends(require_read)
) -> DetectorDTO:
    """Get a specific detector."""
    detector_repo = container.detector_repository()
    
    detector = detector_repo.find_by_id(detector_id)
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")
    
    return DetectorDTO(
        id=detector.id,
        name=detector.name,
        algorithm_name=detector.algorithm_name,
        contamination_rate=detector.contamination_rate.value,
        is_fitted=detector.is_fitted,
        created_at=detector.created_at,
        trained_at=detector.trained_at,
        parameters=detector.parameters,
        metadata=detector.metadata,
        requires_fitting=detector.requires_fitting,
        supports_streaming=detector.supports_streaming,
        supports_multivariate=detector.supports_multivariate,
        time_complexity=detector.time_complexity,
        space_complexity=detector.space_complexity
    )


@router.post("/", response_model=DetectorDTO)
async def create_detector(
    detector_data: CreateDetectorDTO,
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user),
    _permissions: str = Depends(require_write)
) -> DetectorDTO:
    """Create a new detector."""
    detector_repo = container.detector_repository()
    
    # Check if algorithm is supported
    algorithm_name = detector_data.algorithm_name
    
    # Create appropriate adapter
    try:
        if algorithm_name in PyODAdapter.ALGORITHM_MAPPING:
            detector = PyODAdapter(
                algorithm_name=algorithm_name,
                name=detector_data.name,
                contamination_rate=ContaminationRate(detector_data.contamination_rate),
                **detector_data.parameters
            )
        elif algorithm_name in SklearnAdapter.ALGORITHM_MAPPING:
            detector = SklearnAdapter(
                algorithm_name=algorithm_name,
                name=detector_data.name,
                contamination_rate=ContaminationRate(detector_data.contamination_rate),
                **detector_data.parameters
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {algorithm_name}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to create detector: {str(e)}"
        )
    
    # Add metadata
    for key, value in detector_data.metadata.items():
        detector.update_metadata(key, value)
    
    # Save to repository
    detector_repo.save(detector)
    
    return DetectorDTO(
        id=detector.id,
        name=detector.name,
        algorithm_name=detector.algorithm_name,
        contamination_rate=detector.contamination_rate.value,
        is_fitted=detector.is_fitted,
        created_at=detector.created_at,
        trained_at=detector.trained_at,
        parameters=detector.parameters,
        metadata=detector.metadata,
        requires_fitting=detector.requires_fitting,
        supports_streaming=detector.supports_streaming,
        supports_multivariate=detector.supports_multivariate,
        time_complexity=detector.time_complexity,
        space_complexity=detector.space_complexity
    )


@router.patch("/{detector_id}", response_model=DetectorDTO)
async def update_detector(
    detector_id: UUID,
    update_data: UpdateDetectorDTO,
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user),
    _permissions: str = Depends(require_write)
) -> DetectorDTO:
    """Update detector parameters."""
    detector_repo = container.detector_repository()
    
    detector = detector_repo.find_by_id(detector_id)
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")
    
    # Update fields
    if update_data.name is not None:
        detector.name = update_data.name
    
    if update_data.contamination_rate is not None:
        detector.contamination_rate = ContaminationRate(update_data.contamination_rate)
    
    if update_data.parameters is not None:
        detector.update_parameters(**update_data.parameters)
    
    if update_data.metadata is not None:
        for key, value in update_data.metadata.items():
            detector.update_metadata(key, value)
    
    # Save changes
    detector_repo.save(detector)
    
    return DetectorDTO(
        id=detector.id,
        name=detector.name,
        algorithm_name=detector.algorithm_name,
        contamination_rate=detector.contamination_rate.value,
        is_fitted=detector.is_fitted,
        created_at=detector.created_at,
        trained_at=detector.trained_at,
        parameters=detector.parameters,
        metadata=detector.metadata,
        requires_fitting=detector.requires_fitting,
        supports_streaming=detector.supports_streaming,
        supports_multivariate=detector.supports_multivariate,
        time_complexity=detector.time_complexity,
        space_complexity=detector.space_complexity
    )


@router.delete("/{detector_id}")
async def delete_detector(
    detector_id: UUID,
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user),
    _permissions: str = Depends(PermissionChecker(["detectors:delete"]))
) -> dict:
    """Delete a detector."""
    detector_repo = container.detector_repository()
    
    if not detector_repo.exists(detector_id):
        raise HTTPException(status_code=404, detail="Detector not found")
    
    success = detector_repo.delete(detector_id)
    
    return {"success": success, "message": "Detector deleted"}