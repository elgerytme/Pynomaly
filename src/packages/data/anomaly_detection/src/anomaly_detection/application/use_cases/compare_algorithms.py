"""UC-003: Compare Multiple Detection Algorithms use case implementation."""

from typing import Dict, Any, List
from dataclasses import dataclass

from ...domain.entities.dataset import Dataset
from ...domain.services.detection_service import DetectionService


@dataclass
class CompareAlgorithmsRequest:
    """Request for algorithm comparison."""
    dataset: Dataset
    algorithms: List[str]
    evaluation_metrics: List[str] = None
    cross_validation_folds: int = 5


@dataclass
class CompareAlgorithmsResponse:
    """Response from algorithm comparison."""
    comparison_results: Dict[str, Any] = None
    success: bool = False
    error_message: str = None


class CompareAlgorithmsUseCase:
    """Use case for comparing multiple detection algorithms."""
    
    def __init__(self, detection_service: DetectionService):
        self._detection_service = detection_service
    
    def execute(self, request: CompareAlgorithmsRequest) -> CompareAlgorithmsResponse:
        """Execute algorithm comparison.
        
        Args:
            request: Comparison request
            
        Returns:
            Comparison response
        """
        try:
            # Validate input
            if not request.dataset.is_valid():
                return CompareAlgorithmsResponse(
                    success=False,
                    error_message="Invalid dataset format or quality"
                )
            
            if len(request.algorithms) < 2:
                return CompareAlgorithmsResponse(
                    success=False,
                    error_message="Minimum 2 algorithms required for comparison"
                )
            
            if not request.dataset.has_labels():
                return CompareAlgorithmsResponse(
                    success=False,
                    error_message="Dataset must have ground truth labels for comparison"
                )
            
            # Set default metrics if not provided
            evaluation_metrics = request.evaluation_metrics or [
                "precision", "recall", "f1_score", "auc_roc", "auc_pr"
            ]
            
            # Perform comparison
            results = self._detection_service.compare_algorithms(
                request.dataset,
                request.algorithms,
                evaluation_metrics,
                request.cross_validation_folds
            )
            
            return CompareAlgorithmsResponse(
                comparison_results=results,
                success=True
            )
            
        except Exception as e:
            return CompareAlgorithmsResponse(
                success=False,
                error_message=str(e)
            )