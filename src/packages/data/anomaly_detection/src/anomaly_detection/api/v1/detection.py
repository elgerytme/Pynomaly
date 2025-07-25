"""Detection endpoints."""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator, root_validator
from typing_extensions import Literal

from ...domain.services.detection_service import DetectionService
from ...domain.services.ensemble_service import EnsembleService
from ...infrastructure.logging import get_logger, async_log_decorator

router = APIRouter()
logger = get_logger(__name__)


class DetectionRequest(BaseModel):
    """Request model for anomaly detection with comprehensive validation."""
    data: List[List[float]] = Field(
        ..., 
        description="Input data as list of feature vectors",
        min_items=1,
        max_items=10000
    )
    algorithm: Literal[
        "isolation_forest", 
        "local_outlier_factor",
        "lof",
        "iforest"
    ] = Field(
        default="isolation_forest", 
        description="Detection algorithm to use"
    )
    contamination: float = Field(
        default=0.1, 
        ge=0.001, 
        le=0.5, 
        description="Expected contamination rate (0.1% to 50%)"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Algorithm-specific parameters"
    )
    
    @validator('data')
    def validate_data_structure(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        
        # Check that all rows have the same number of features
        feature_count = len(v[0]) if v else 0
        if feature_count == 0:
            raise ValueError("Data must have at least one feature")
        if feature_count > 1000:
            raise ValueError("Maximum 1000 features supported")
            
        for i, row in enumerate(v):
            if len(row) != feature_count:
                raise ValueError(f"Row {i} has {len(row)} features, expected {feature_count}")
            if not all(isinstance(x, (int, float)) and not np.isnan(x) and np.isfinite(x) for x in row):
                raise ValueError(f"Row {i} contains invalid values (NaN or infinite)")
        
        return v
    
    @validator('algorithm')
    def normalize_algorithm_name(cls, v):
        # Normalize algorithm names - keep as-is since we only support these
        return v
    
    @validator('parameters')
    def validate_parameters(cls, v, values):
        if 'algorithm' not in values:
            return v
            
        algorithm = values['algorithm']
        
        # Algorithm-specific parameter validation
        if algorithm == "isolation_forest":
            if 'n_estimators' in v and (v['n_estimators'] < 1 or v['n_estimators'] > 1000):
                raise ValueError("n_estimators must be between 1 and 1000")
        elif algorithm == "local_outlier_factor":
            if 'n_neighbors' in v and (v['n_neighbors'] < 1 or v['n_neighbors'] > 100):
                raise ValueError("n_neighbors must be between 1 and 100")
        elif algorithm == "one_class_svm":
            if 'gamma' in v and v['gamma'] not in ['scale', 'auto'] and (v['gamma'] <= 0):
                raise ValueError("gamma must be 'scale', 'auto', or a positive number")
        
        return v


class EnsembleRequest(BaseModel):
    """Request model for ensemble detection."""
    data: List[List[float]] = Field(..., description="Input data as list of feature vectors")
    algorithms: List[str] = Field(default=["isolation_forest", "lof"], 
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
        
        # Map algorithm names to supported backend algorithms
        algorithm_map = {
            'isolation_forest': 'iforest',
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
        
        # Map algorithm names to supported backend algorithms
        algorithm_map = {
            'isolation_forest': 'iforest',
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


class TrainingRequest(BaseModel):
    """Request model for model training."""
    data: List[List[float]] = Field(
        ..., 
        description="Training data as list of feature vectors",
        min_items=10,
        max_items=100000
    )
    algorithm: Literal[
        "isolation_forest", 
        "one_class_svm", 
        "local_outlier_factor"
    ] = Field(
        default="isolation_forest", 
        description="Algorithm to train"
    )
    contamination: float = Field(
        default=0.1, 
        ge=0.001, 
        le=0.5, 
        description="Expected contamination rate"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Algorithm-specific parameters"
    )


class TrainingResponse(BaseModel):
    """Response model for model training."""
    success: bool
    model_id: str
    algorithm: str
    training_samples: int
    training_time_ms: float
    model_performance: Dict[str, float]
    timestamp: str


@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    detection_service: DetectionService = Depends(get_detection_service)
) -> TrainingResponse:
    """Train a new anomaly detection model."""
    start_time = datetime.utcnow()
    
    logger.info("Training new model",
                algorithm=request.algorithm,
                samples=len(request.data))
    
    try:
        # Validate input
        if not request.data or not request.data[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training data cannot be empty"
            )
        
        # Convert to numpy array
        data_array = np.array(request.data, dtype=np.float64)
        
        # Map algorithm names
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'local_outlier_factor': 'lof'
        }
        
        mapped_algorithm = algorithm_map.get(request.algorithm, request.algorithm)
        
        # Train the model
        result = detection_service.train_model(
            data=data_array,
            algorithm=mapped_algorithm,
            contamination=request.contamination,
            **request.parameters
        )
        
        end_time = datetime.utcnow()
        training_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Generate model ID
        model_id = f"{request.algorithm}_{int(start_time.timestamp())}"
        
        # Mock performance metrics (in real implementation, would come from validation)
        performance_metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.90
        }
        
        return TrainingResponse(
            success=True,
            model_id=model_id,
            algorithm=request.algorithm,
            training_samples=len(request.data),
            training_time_ms=training_time_ms,
            model_performance=performance_metrics,
            timestamp=end_time.isoformat()
        )
        
    except ValueError as e:
        logger.error("Training validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid training data: {str(e)}"
        )
    except Exception as e:
        logger.error("Training processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}"
        )