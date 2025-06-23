"""Use case for detecting anomalies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.exceptions import DetectorNotFittedError, DatasetError
from pynomaly.domain.services import FeatureValidator
from pynomaly.shared.protocols import DetectorProtocol, DetectorRepositoryProtocol


@dataclass
class DetectAnomaliesRequest:
    """Request for anomaly detection."""
    
    detector_id: UUID
    dataset: Dataset
    validate_features: bool = True
    save_results: bool = True


@dataclass
class DetectAnomaliesResponse:
    """Response from anomaly detection."""
    
    result: DetectionResult
    quality_report: Optional[dict] = None
    warnings: Optional[list[str]] = None


class DetectAnomaliesUseCase:
    """Use case for detecting anomalies in a dataset."""
    
    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        feature_validator: FeatureValidator
    ):
        """Initialize the use case.
        
        Args:
            detector_repository: Repository for loading detectors
            feature_validator: Service for validating features
        """
        self.detector_repository = detector_repository
        self.feature_validator = feature_validator
    
    async def execute(self, request: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
        """Execute anomaly detection.
        
        Args:
            request: Detection request
            
        Returns:
            Detection response with results
            
        Raises:
            DetectorNotFittedError: If detector is not fitted
            DatasetError: If dataset validation fails
        """
        # Load detector
        detector = self.detector_repository.find_by_id(request.detector_id)
        if detector is None:
            raise ValueError(f"Detector {request.detector_id} not found")
        
        # Check if detector is fitted
        if not detector.is_fitted:
            raise DetectorNotFittedError(
                detector_name=detector.name,
                operation="detect"
            )
        
        warnings = []
        quality_report = None
        
        # Validate features if requested
        if request.validate_features:
            # Check data quality
            quality_report = self.feature_validator.check_data_quality(
                request.dataset
            )
            
            # Add warnings for quality issues
            if quality_report["quality_score"] < 0.8:
                warnings.append(
                    f"Data quality score is low: {quality_report['quality_score']:.2f}"
                )
            
            suggestions = self.feature_validator.suggest_preprocessing(
                quality_report
            )
            if suggestions:
                warnings.extend(suggestions)
        
        # Perform detection
        result = detector.detect(request.dataset)
        
        # Save results if requested
        if request.save_results:
            # This would typically use a result repository
            # For now, we'll just add metadata
            result.add_metadata("saved", True)
        
        return DetectAnomaliesResponse(
            result=result,
            quality_report=quality_report,
            warnings=warnings if warnings else None
        )