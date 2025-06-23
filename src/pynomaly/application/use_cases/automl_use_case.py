"""AutoML use case for automated anomaly detection model selection and optimization."""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from pynomaly.application.dto import AutoMLRequestDTO, AutoMLResponseDTO
from pynomaly.application.services.automl_service import AutoMLService, OptimizationObjective

logger = logging.getLogger(__name__)


@dataclass
class AutoMLRequest:
    """Request for AutoML optimization."""
    dataset_id: str
    objective: str = "auc"
    max_algorithms: int = 3
    max_optimization_time: int = 3600
    enable_ensemble: bool = True
    detector_name: Optional[str] = None


@dataclass
class AutoMLResponse:
    """Response from AutoML optimization."""
    success: bool
    detector_id: Optional[str] = None
    best_algorithm: Optional[str] = None
    best_score: Optional[float] = None
    optimization_time: Optional[float] = None
    algorithm_rankings: Optional[list] = None
    ensemble_created: bool = False
    message: Optional[str] = None
    error: Optional[str] = None


class AutoMLUseCase:
    """Use case for automated machine learning in anomaly detection."""
    
    def __init__(self, automl_service: AutoMLService):
        """Initialize AutoML use case.
        
        Args:
            automl_service: AutoML service for optimization
        """
        self.automl_service = automl_service
    
    async def execute(self, request: AutoMLRequest) -> AutoMLResponse:
        """Execute AutoML optimization.
        
        Args:
            request: AutoML request parameters
            
        Returns:
            AutoML response with results
        """
        try:
            logger.info(f"Starting AutoML for dataset {request.dataset_id}")
            
            # Map objective string to enum
            objective_map = {
                "auc": OptimizationObjective.AUC,
                "precision": OptimizationObjective.PRECISION,
                "recall": OptimizationObjective.RECALL,
                "f1": OptimizationObjective.F1_SCORE,
                "balanced_accuracy": OptimizationObjective.BALANCED_ACCURACY,
                "detection_rate": OptimizationObjective.DETECTION_RATE
            }
            
            objective = objective_map.get(request.objective.lower(), OptimizationObjective.AUC)
            
            # Update AutoML service configuration
            self.automl_service.max_optimization_time = request.max_optimization_time
            
            # Run AutoML optimization
            automl_result = await self.automl_service.auto_select_and_optimize(
                dataset_id=request.dataset_id,
                objective=objective,
                max_algorithms=request.max_algorithms,
                enable_ensemble=request.enable_ensemble
            )
            
            # Create optimized detector
            detector_id = await self.automl_service.create_optimized_detector(
                automl_result=automl_result,
                detector_name=request.detector_name
            )
            
            # Prepare response
            response = AutoMLResponse(
                success=True,
                detector_id=detector_id,
                best_algorithm=automl_result.best_algorithm,
                best_score=automl_result.best_score,
                optimization_time=automl_result.optimization_time,
                algorithm_rankings=automl_result.algorithm_rankings,
                ensemble_created=automl_result.ensemble_config is not None,
                message=f"AutoML completed successfully. Best algorithm: {automl_result.best_algorithm}"
            )
            
            logger.info(f"AutoML completed successfully for dataset {request.dataset_id}. "
                       f"Created detector {detector_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"AutoML failed for dataset {request.dataset_id}: {str(e)}")
            
            return AutoMLResponse(
                success=False,
                error=str(e),
                message="AutoML optimization failed"
            )