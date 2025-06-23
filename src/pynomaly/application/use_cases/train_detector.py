"""Use case for training a detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import FittingError, InsufficientDataError
from pynomaly.domain.services import FeatureValidator
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.shared.protocols import DetectorProtocol, DetectorRepositoryProtocol


@dataclass
class TrainDetectorRequest:
    """Request for training a detector."""
    
    detector_id: UUID
    dataset: Dataset
    contamination_rate: Optional[ContaminationRate] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    validate_data: bool = True
    save_model: bool = True
    experiment_name: Optional[str] = None


@dataclass
class TrainDetectorResponse:
    """Response from detector training."""
    
    detector_id: UUID
    training_time_ms: float
    dataset_summary: dict
    parameters_used: dict
    validation_results: Optional[dict] = None


class TrainDetectorUseCase:
    """Use case for training an anomaly detector."""
    
    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        feature_validator: FeatureValidator,
        min_samples: int = 10
    ):
        """Initialize the use case.
        
        Args:
            detector_repository: Repository for detectors
            feature_validator: Service for validating features
            min_samples: Minimum samples required for training
        """
        self.detector_repository = detector_repository
        self.feature_validator = feature_validator
        self.min_samples = min_samples
    
    async def execute(self, request: TrainDetectorRequest) -> TrainDetectorResponse:
        """Execute detector training.
        
        Args:
            request: Training request
            
        Returns:
            Training response with summary
            
        Raises:
            InsufficientDataError: If dataset is too small
            FittingError: If training fails
        """
        start_time = datetime.utcnow()
        
        # Load detector
        detector = self.detector_repository.find_by_id(request.detector_id)
        if detector is None:
            raise ValueError(f"Detector {request.detector_id} not found")
        
        # Check dataset size
        if request.dataset.n_samples < self.min_samples:
            raise InsufficientDataError(
                dataset_name=request.dataset.name,
                n_samples=request.dataset.n_samples,
                min_required=self.min_samples,
                operation="training"
            )
        
        validation_results = None
        
        # Validate data if requested
        if request.validate_data:
            # Check for numeric features
            numeric_features = self.feature_validator.validate_numeric_features(
                request.dataset
            )
            
            if not numeric_features:
                raise FittingError(
                    detector_name=detector.name,
                    reason="No numeric features found in dataset",
                    dataset_name=request.dataset.name
                )
            
            # Check data quality
            quality_report = self.feature_validator.check_data_quality(
                request.dataset
            )
            validation_results = {
                "numeric_features": len(numeric_features),
                "quality_score": quality_report["quality_score"],
                "issues": []
            }
            
            if quality_report["missing_values"]:
                validation_results["issues"].append(
                    f"Missing values in {len(quality_report['missing_values'])} features"
                )
            
            if quality_report["constant_features"]:
                validation_results["issues"].append(
                    f"{len(quality_report['constant_features'])} constant features"
                )
        
        # Update parameters if provided
        if request.parameters:
            detector.update_parameters(**request.parameters)
        
        # Update contamination rate if provided
        if request.contamination_rate:
            detector.contamination_rate = request.contamination_rate
        
        # Train the detector
        try:
            detector.fit(request.dataset)
        except Exception as e:
            raise FittingError(
                detector_name=detector.name,
                reason=str(e),
                dataset_name=request.dataset.name
            ) from e
        
        # Calculate training time
        end_time = datetime.utcnow()
        training_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Update detector metadata
        detector.trained_at = end_time
        detector.is_fitted = True
        detector.update_metadata("last_training_dataset", request.dataset.name)
        detector.update_metadata("last_training_samples", request.dataset.n_samples)
        
        # Save detector and model artifact if requested
        if request.save_model:
            self.detector_repository.save(detector)
            # Model artifact saving would be handled by the adapter
        
        # Create response
        return TrainDetectorResponse(
            detector_id=detector.id,
            training_time_ms=training_time_ms,
            dataset_summary=request.dataset.summary(),
            parameters_used=detector.parameters,
            validation_results=validation_results
        )