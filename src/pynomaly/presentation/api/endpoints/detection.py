"""Detection endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from pynomaly.application.dto import DetectionResultDTO
from pynomaly.application.use_cases import (
    DetectAnomaliesRequest,
    EvaluateModelRequest,
    TrainDetectorRequest,
)
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.deps import get_container, get_current_user

router = APIRouter()


class TrainRequest(BaseModel):
    """Request to train a detector."""

    detector_id: UUID
    dataset_id: UUID
    validate_data: bool = True
    save_model: bool = True


class DetectRequest(BaseModel):
    """Request to detect anomalies."""

    detector_id: UUID
    dataset_id: UUID
    validate_features: bool = True
    save_results: bool = True


class BatchDetectRequest(BaseModel):
    """Request for batch detection."""

    detector_ids: list[UUID]
    dataset_id: UUID
    save_results: bool = True


class EvaluateRequest(BaseModel):
    """Request to evaluate a detector."""

    detector_id: UUID
    dataset_id: UUID
    cross_validate: bool = False
    n_folds: int = 5
    metrics: list[str] | None = None


@router.post("/train")
async def train_detector(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> dict:
    """Train a detector on a dataset."""
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()
    train_use_case = container.train_detector_use_case()

    # Validate detector exists
    detector = detector_repo.find_by_id(request.detector_id)
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")

    # Validate dataset exists
    dataset = dataset_repo.find_by_id(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create use case request
    train_request = TrainDetectorRequest(
        detector_id=request.detector_id,
        dataset=dataset,
        validate_data=request.validate_data,
        save_model=request.save_model,
    )

    try:
        # Execute training
        response = await train_use_case.execute(train_request)

        return {
            "success": True,
            "detector_id": str(response.detector_id),
            "training_time_ms": response.training_time_ms,
            "dataset_summary": response.dataset_summary,
            "parameters_used": response.parameters_used,
            "validation_results": response.validation_results,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {str(e)}")


@router.post("/detect", response_model=DetectionResultDTO)
async def detect_anomalies(
    request: DetectRequest,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> DetectionResultDTO:
    """Detect anomalies in a dataset."""
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()
    detect_use_case = container.detect_anomalies_use_case()

    # Validate detector exists
    detector = detector_repo.find_by_id(request.detector_id)
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")

    if not detector.is_fitted:
        raise HTTPException(
            status_code=400, detail="Detector must be trained before detection"
        )

    # Validate dataset exists
    dataset = dataset_repo.find_by_id(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create use case request
    detect_request = DetectAnomaliesRequest(
        detector_id=request.detector_id,
        dataset=dataset,
        validate_features=request.validate_features,
        save_results=request.save_results,
    )

    try:
        # Execute detection
        response = await detect_use_case.execute(detect_request)
        result = response.result

        return DetectionResultDTO(
            id=result.id,
            detector_id=result.detector_id,
            dataset_id=result.dataset_id,
            timestamp=result.timestamp,
            n_samples=result.n_samples,
            n_anomalies=result.n_anomalies,
            anomaly_rate=result.anomaly_rate,
            threshold=result.threshold,
            execution_time_ms=result.execution_time_ms,
            score_statistics=result.score_statistics,
            metadata=result.metadata,
            has_confidence_intervals=result.has_confidence_intervals,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Detection failed: {str(e)}")


@router.post("/detect/batch")
async def batch_detect_anomalies(
    request: BatchDetectRequest,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> dict:
    """Run detection with multiple detectors."""
    dataset_repo = container.dataset_repository()
    detection_service = container.detection_service()

    # Validate dataset exists
    dataset = dataset_repo.find_by_id(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        # Run batch detection
        results = await detection_service.detect_with_multiple_detectors(
            detector_ids=request.detector_ids,
            dataset=dataset,
            save_results=request.save_results,
        )

        # Convert results to response
        response_data = {}
        for detector_id, result in results.items():
            response_data[str(detector_id)] = {
                "n_anomalies": result.n_anomalies,
                "anomaly_rate": result.anomaly_rate,
                "threshold": result.threshold,
                "execution_time_ms": result.execution_time_ms,
            }

        return {"success": True, "n_detectors": len(results), "results": response_data}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch detection failed: {str(e)}")


@router.post("/evaluate")
async def evaluate_detector(
    request: EvaluateRequest,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> dict:
    """Evaluate detector performance."""
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()
    evaluate_use_case = container.evaluate_model_use_case()

    # Validate detector exists
    detector = detector_repo.find_by_id(request.detector_id)
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")

    # Validate dataset exists and has labels
    dataset = dataset_repo.find_by_id(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not dataset.has_target:
        raise HTTPException(
            status_code=400, detail="Dataset must have target labels for evaluation"
        )

    # Create use case request
    eval_request = EvaluateModelRequest(
        detector_id=request.detector_id,
        test_dataset=dataset,
        cross_validate=request.cross_validate,
        n_folds=request.n_folds,
        metrics=request.metrics,
    )

    try:
        # Execute evaluation
        response = await evaluate_use_case.execute(eval_request)

        result = {
            "detector_id": str(response.detector_id),
            "metrics": response.metrics,
        }

        if response.confusion_matrix is not None:
            result["confusion_matrix"] = response.confusion_matrix.tolist()

        if response.cross_validation_scores:
            result["cross_validation_scores"] = response.cross_validation_scores

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {str(e)}")


@router.get("/results")
async def list_detection_results(
    detector_id: UUID | None = Query(None),
    dataset_id: UUID | None = Query(None),
    limit: int = Query(10, ge=1, le=100),
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> list[DetectionResultDTO]:
    """List detection results."""
    result_repo = container.result_repository()

    # Get results based on filters
    if detector_id:
        results = result_repo.find_by_detector(detector_id)
    elif dataset_id:
        results = result_repo.find_by_dataset(dataset_id)
    else:
        results = result_repo.find_recent(limit)

    # Limit and convert to DTOs
    results = results[:limit]

    return [
        DetectionResultDTO(
            id=r.id,
            detector_id=r.detector_id,
            dataset_id=r.dataset_id,
            timestamp=r.timestamp,
            n_samples=r.n_samples,
            n_anomalies=r.n_anomalies,
            anomaly_rate=r.anomaly_rate,
            threshold=r.threshold,
            execution_time_ms=r.execution_time_ms,
            score_statistics=r.score_statistics,
            metadata=r.metadata,
            has_confidence_intervals=r.has_confidence_intervals,
        )
        for r in results
    ]


@router.get("/compare")
async def compare_detectors(
    dataset_id: UUID,
    detector_ids: list[UUID] = Query(...),
    metrics: list[str] | None = Query(None),
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
) -> dict:
    """Compare multiple detectors on a dataset."""
    dataset_repo = container.dataset_repository()
    detection_service = container.detection_service()

    # Validate dataset
    dataset = dataset_repo.find_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not dataset.has_target:
        raise HTTPException(
            status_code=400, detail="Dataset must have labels for comparison"
        )

    try:
        # Run comparison
        comparison = await detection_service.compare_detectors(
            detector_ids=detector_ids, dataset=dataset, metrics=metrics
        )

        return comparison

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comparison failed: {str(e)}")
