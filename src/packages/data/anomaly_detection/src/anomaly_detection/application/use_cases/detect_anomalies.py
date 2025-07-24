"""UC-001: Detect Anomalies in Tabular Data use case implementation."""

from typing import Dict, Any
from dataclasses import dataclass

from ...domain.entities.dataset import Dataset
from ...domain.entities.detection_result import DetectionResult
from ...domain.services.detection_service import DetectionService


@dataclass
class DetectAnomaliesRequest:
    """Request for anomaly detection."""
    dataset: Dataset
    algorithm: str = "isolation_forest"
    parameters: Dict[str, Any] = None


@dataclass
class DetectAnomaliesResponse:
    """Response from anomaly detection."""
    result: DetectionResult
    success: bool
    error_message: str = None


class DetectAnomaliesUseCase:
    """Use case for detecting anomalies in tabular data."""
    
    def __init__(self, detection_service: DetectionService):
        self._detection_service = detection_service
    
    def execute(self, request: DetectAnomaliesRequest) -> DetectAnomaliesResponse:
        """Execute anomaly detection.
        
        Args:
            request: Detection request
            
        Returns:
            Detection response
        """
        try:
            # Validate input data
            if not request.dataset.is_valid():
                return DetectAnomaliesResponse(
                    result=None,
                    success=False,
                    error_message="Invalid dataset format or quality"
                )
            
            # Perform detection
            result = self._detection_service.detect(
                request.dataset,
                request.algorithm,
                request.parameters or {}
            )
            
            return DetectAnomaliesResponse(
                result=result,
                success=True
            )
            
        except Exception as e:
            return DetectAnomaliesResponse(
                result=None,
                success=False,
                error_message=str(e)
            )