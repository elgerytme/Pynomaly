"""Detection endpoints."""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...domain.services.detection_service import DetectionService
from ...domain.services.ensemble_service import EnsembleService
from ...infrastructure.logging import get_logger, async_log_decorator

router = APIRouter()
logger = get_logger(__name__)


class DetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    data: List[List[float]] = Field(..., description="Input data as list of feature vectors")
    algorithm: str = Field(default="isolation_forest", description="Detection algorithm to use")
    contamination: float = Field(default=0.1, ge=0.001, le=0.5, description="Expected contamination rate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific parameters")


class EnsembleRequest(BaseModel):
    """Request model for ensemble detection."""
    data: List[List[float]] = Field(..., description="Input data as list of feature vectors")
    algorithms: List[str] = Field(default=["isolation_forest", "one_class_svm", "lof"], 
                                 description="Algorithms to use in ensemble")
    method: str = Field(default="majority", description="Ensemble combination method")
    contamination: float = Field(default=0.1, ge=0.001, le=0.5, description="Expected contamination rate")
    parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict, 
                                                 description="Algorithm-specific parameters")


class DetectionResponse(BaseModel):
    """Response model for anomaly detection."""
    success: bool = Field(..., description="Whether detection completed successfully")
    anomalies: List[int] = Field(..., description="Indices of detected anomalies")
    scores: Optional[List[float]] = Field(None, description="Anomaly confidence scores")
    algorithm: str = Field(..., description="Algorithm used for detection")
    total_samples: int = Field(..., description="Total number of samples processed")
    anomalies_detected: int = Field(..., description="Number of anomalies detected")
    anomaly_rate: float = Field(..., description="Ratio of anomalies to total samples")
    timestamp: str = Field(..., description="Detection timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# Dependency injection for services
_detection_service: Optional[DetectionService] = None
_ensemble_service: Optional[EnsembleService] = None


def get_detection_service() -> DetectionService:
    """Get detection service instance."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service


def get_ensemble_service() -> EnsembleService:
    """Get ensemble service instance."""
    global _ensemble_service
    if _ensemble_service is None:
        _ensemble_service = EnsembleService()
    return _ensemble_service


@router.post("/detect", response_model=DetectionResponse)
@async_log_decorator(operation="api_detect_anomalies", log_args=False, log_duration=True)
async def detect_anomalies(
    request: DetectionRequest,
    detection_service: DetectionService = Depends(get_detection_service)
) -> DetectionResponse:
    """Detect anomalies in dataset using specified algorithm."""
    start_time = datetime.utcnow()
    
    logger.info("Processing detection request", 
                algorithm=request.algorithm, 
                samples=len(request.data),
                contamination=request.contamination)
    
    try:
        # Validate input data
        if not request.data or not request.data[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data cannot be empty"
            )
        
        # Convert to numpy array
        data_array = np.array(request.data, dtype=np.float64)
        
        # Map algorithm names
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'local_outlier_factor': 'lof',
            'lof': 'lof'
        }
        
        algorithm_name = algorithm_map.get(request.algorithm, request.algorithm)
        
        # Run detection
        result = detection_service.detect_anomalies(
            data=data_array,
            algorithm=algorithm_name,
            contamination=request.contamination,
            **request.parameters
        )
        
        end_time = datetime.utcnow()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return DetectionResponse(
            success=result.success,
            anomalies=result.anomalies,
            scores=result.confidence_scores.tolist() if result.confidence_scores is not None else None,
            algorithm=request.algorithm,
            total_samples=result.total_samples,
            anomalies_detected=result.anomaly_count,
            anomaly_rate=result.anomaly_rate,
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time_ms
        )
        
    except ValueError as e:
        logger.error("Detection validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error("Detection processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@router.post("/ensemble", response_model=DetectionResponse)
async def ensemble_detect(
    request: EnsembleRequest,
    ensemble_service: EnsembleService = Depends(get_ensemble_service)
) -> DetectionResponse:
    """Run ensemble anomaly detection using multiple algorithms."""
    start_time = datetime.utcnow()
    
    logger.info("Processing ensemble detection request",
                algorithms=request.algorithms,
                method=request.method,
                samples=len(request.data))
    
    try:
        # Validate input
        if not request.data or not request.data[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data cannot be empty"
            )
        
        if len(request.algorithms) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ensemble requires at least 2 algorithms"
            )
        
        # Convert to numpy array
        data_array = np.array(request.data, dtype=np.float64)
        
        # Map algorithm names
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'local_outlier_factor': 'lof',
            'lof': 'lof'
        }
        
        mapped_algorithms = [algorithm_map.get(alg, alg) for alg in request.algorithms]
        
        # Get individual results
        individual_results = []
        for algorithm in mapped_algorithms:
            detection_service = DetectionService()
            result = detection_service.detect_anomalies(
                data=data_array,
                algorithm=algorithm,
                contamination=request.contamination,
                **request.parameters.get(algorithm, {})
            )
            individual_results.append(result)
        
        # Combine using ensemble method
        predictions_array = np.array([result.predictions for result in individual_results])
        scores_array = np.array([result.confidence_scores for result in individual_results if result.confidence_scores is not None])
        
        if request.method == 'majority':
            ensemble_predictions = ensemble_service.majority_vote(predictions_array)
            ensemble_scores = None
        elif request.method in ['average', 'weighted_average', 'max'] and len(scores_array) > 0:
            if request.method == 'average':
                ensemble_predictions, ensemble_scores = ensemble_service.average_combination(predictions_array, scores_array)
            elif request.method == 'max':
                ensemble_predictions, ensemble_scores = ensemble_service.max_combination(predictions_array, scores_array)
            else:  # weighted_average
                weights = np.ones(len(request.algorithms)) / len(request.algorithms)
                ensemble_predictions, ensemble_scores = ensemble_service.weighted_combination(
                    predictions_array, scores_array, weights
                )
        else:
            # Fallback to majority vote
            ensemble_predictions = ensemble_service.majority_vote(predictions_array)
            ensemble_scores = None
        
        # Calculate ensemble statistics
        anomaly_count = int(np.sum(ensemble_predictions == -1))
        total_samples = len(ensemble_predictions)
        anomaly_rate = anomaly_count / total_samples if total_samples > 0 else 0.0
        anomaly_indices = np.where(ensemble_predictions == -1)[0].tolist()
        
        end_time = datetime.utcnow()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return DetectionResponse(
            success=True,
            anomalies=anomaly_indices,
            scores=ensemble_scores.tolist() if ensemble_scores is not None else None,
            algorithm=f"ensemble_{request.method}",
            total_samples=total_samples,
            anomalies_detected=anomaly_count,
            anomaly_rate=anomaly_rate,
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time_ms
        )
        
    except ValueError as e:
        logger.error("Ensemble validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error("Ensemble processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ensemble detection failed: {str(e)}"
        )


@router.get("/algorithms")
async def list_algorithms() -> Dict[str, List[str]]:
    """List available detection algorithms and ensemble methods."""
    return {
        "single_algorithms": [
            "isolation_forest",
            "one_class_svm", 
            "local_outlier_factor",
            "lof"
        ],
        "ensemble_methods": [
            "majority",
            "average", 
            "weighted_average",
            "max"
        ],
        "supported_formats": [
            "json",
            "csv"
        ]
    }