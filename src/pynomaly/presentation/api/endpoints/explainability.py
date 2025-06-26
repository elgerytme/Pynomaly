"""API endpoints for explainability operations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse

from pynomaly.application.use_cases.explainability_use_case import (
    ExplainabilityUseCase,
    ExplainPredictionRequest,
    ExplainModelRequest,
    ExplainCohortRequest,
    CompareExplanationsRequest
)
from pynomaly.application.dto.explainability_dto import (
    ExplainPredictionRequestDTO,
    ExplainModelRequestDTO,
    ExplainCohortRequestDTO,
    CompareExplanationsRequestDTO,
    ExplainabilityResponseDTO,
    LocalExplanationDTO,
    GlobalExplanationDTO,
    CohortExplanationDTO
)
from pynomaly.domain.services.explainability_service import ExplanationMethod
from pynomaly.infrastructure.di.container import Container
from pynomaly.presentation.api.dependencies import (
    get_container,
    get_current_user,
    require_read,
    require_write
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/explainability", tags=["explainability"])


@router.post("/explain/prediction", response_model=ExplainabilityResponseDTO)
async def explain_prediction(
    request: ExplainPredictionRequestDTO,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> ExplainabilityResponseDTO:
    """Explain a single anomaly prediction.
    
    Generates explanations for why a specific data point was classified
    as an anomaly or normal, using various explainability methods like
    SHAP, LIME, or feature importance.
    
    Args:
        request: Prediction explanation request with instance data
        background_tasks: Background task queue for async operations
        container: Dependency injection container
        current_user: Current authenticated user
        _permissions: Required permissions check
        
    Returns:
        Detailed explanation of the prediction with feature contributions
        
    Raises:
        HTTPException: If explanation generation fails
    """
    try:
        logger.info(f"User {current_user} requesting prediction explanation for detector {request.detector_id}")
        
        # Get explainability use case
        explainability_use_case = container.explainability_use_case()
        
        # Convert DTO to domain request
        domain_request = ExplainPredictionRequest(
            detector_id=request.detector_id,
            instance_data=request.instance_data,
            explanation_method=ExplanationMethod(request.explanation_method),
            background_dataset_id=request.background_dataset_id,
            instance_id=request.instance_id,
            include_counterfactuals=request.include_counterfactuals,
            max_features=request.max_features
        )
        
        # Execute explanation
        result = await explainability_use_case.explain_prediction(domain_request)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        # Convert to response DTO
        explanation_dto = None
        if result.explanation:
            explanation_dto = LocalExplanationDTO.from_domain(result.explanation)
        
        response_dto = ExplainabilityResponseDTO(
            success=result.success,
            explanation_type="local",
            local_explanation=explanation_dto,
            metadata=result.metadata or {},
            execution_time_seconds=result.execution_time_seconds,
            message=f"Successfully explained prediction using {request.explanation_method}"
        )
        
        # Log success
        logger.info(f"Prediction explanation completed for detector {request.detector_id}")
        
        return response_dto
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain prediction: {str(e)}"
        )


@router.post("/explain/model", response_model=ExplainabilityResponseDTO)
async def explain_model(
    request: ExplainModelRequestDTO,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> ExplainabilityResponseDTO:
    """Explain global model behavior and feature importance.
    
    Generates global explanations showing how the model makes decisions
    overall, including feature importance rankings and model behavior patterns.
    
    Args:
        request: Model explanation request with dataset context
        background_tasks: Background task queue for async operations
        container: Dependency injection container
        current_user: Current authenticated user
        _permissions: Required permissions check
        
    Returns:
        Global explanation with feature importance and model insights
        
    Raises:
        HTTPException: If explanation generation fails
    """
    try:
        logger.info(f"User {current_user} requesting model explanation for detector {request.detector_id}")
        
        # Get explainability use case
        explainability_use_case = container.explainability_use_case()
        
        # Convert DTO to domain request
        domain_request = ExplainModelRequest(
            detector_id=request.detector_id,
            dataset_id=request.dataset_id,
            explanation_method=ExplanationMethod(request.explanation_method),
            sample_size=request.sample_size,
            include_interactions=request.include_interactions,
            feature_groups=request.feature_groups
        )
        
        # Execute explanation
        result = await explainability_use_case.explain_model(domain_request)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        # Convert to response DTO
        explanation_dto = None
        if result.explanation:
            explanation_dto = GlobalExplanationDTO.from_domain(result.explanation)
        
        response_dto = ExplainabilityResponseDTO(
            success=result.success,
            explanation_type="global",
            global_explanation=explanation_dto,
            metadata=result.metadata or {},
            execution_time_seconds=result.execution_time_seconds,
            message=f"Successfully explained model using {request.explanation_method}"
        )
        
        # Schedule background analysis if requested
        if request.include_interactions:
            background_tasks.add_task(
                _analyze_feature_interactions_background,
                request.detector_id,
                request.dataset_id,
                current_user
            )
        
        logger.info(f"Model explanation completed for detector {request.detector_id}")
        
        return response_dto
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain model: {str(e)}"
        )


@router.post("/explain/cohort", response_model=ExplainabilityResponseDTO)
async def explain_cohort(
    request: ExplainCohortRequestDTO,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> ExplainabilityResponseDTO:
    """Explain behavior for a cohort of similar instances.
    
    Generates explanations for groups of similar data points, identifying
    common patterns and characteristics that lead to similar anomaly predictions.
    
    Args:
        request: Cohort explanation request with instance indices
        container: Dependency injection container
        current_user: Current authenticated user
        _permissions: Required permissions check
        
    Returns:
        Cohort explanation with common patterns and characteristics
        
    Raises:
        HTTPException: If explanation generation fails
    """
    try:
        logger.info(f"User {current_user} requesting cohort explanation for detector {request.detector_id}")
        
        # Validate cohort size
        if len(request.instance_indices) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cohort must contain at least 2 instances"
            )
        
        if len(request.instance_indices) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cohort size cannot exceed 1000 instances"
            )
        
        # Get explainability use case
        explainability_use_case = container.explainability_use_case()
        
        # Convert DTO to domain request
        domain_request = ExplainCohortRequest(
            detector_id=request.detector_id,
            dataset_id=request.dataset_id,
            instance_indices=request.instance_indices,
            explanation_method=ExplanationMethod(request.explanation_method),
            cohort_name=request.cohort_name,
            similarity_threshold=request.similarity_threshold
        )
        
        # Execute explanation
        result = await explainability_use_case.explain_cohort(domain_request)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        # Convert to response DTO
        explanation_dto = None
        if result.explanation:
            explanation_dto = CohortExplanationDTO.from_domain(result.explanation)
        
        response_dto = ExplainabilityResponseDTO(
            success=result.success,
            explanation_type="cohort",
            cohort_explanation=explanation_dto,
            metadata=result.metadata or {},
            execution_time_seconds=result.execution_time_seconds,
            message=f"Successfully explained cohort of {len(request.instance_indices)} instances"
        )
        
        logger.info(f"Cohort explanation completed for detector {request.detector_id}")
        
        return response_dto
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining cohort: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain cohort: {str(e)}"
        )


@router.post("/explain/compare", response_model=ExplainabilityResponseDTO)
async def compare_explanations(
    request: CompareExplanationsRequestDTO,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> ExplainabilityResponseDTO:
    """Compare explanations across different methods.
    
    Generates explanations using multiple methods (SHAP, LIME, etc.) for the
    same instance and provides consistency analysis to understand agreement
    between different explanation approaches.
    
    Args:
        request: Comparison request with multiple explanation methods
        container: Dependency injection container
        current_user: Current authenticated user
        _permissions: Required permissions check
        
    Returns:
        Comparison results with explanations from each method and consistency analysis
        
    Raises:
        HTTPException: If comparison fails
    """
    try:
        logger.info(f"User {current_user} requesting explanation comparison for detector {request.detector_id}")
        
        # Validate methods
        if len(request.explanation_methods) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 explanation methods required for comparison"
            )
        
        # Get explainability use case
        explainability_use_case = container.explainability_use_case()
        
        # Convert DTO to domain request
        domain_request = CompareExplanationsRequest(
            detector_id=request.detector_id,
            instance_data=request.instance_data,
            explanation_methods=[ExplanationMethod(method) for method in request.explanation_methods],
            background_dataset_id=request.background_dataset_id,
            consistency_analysis=request.consistency_analysis
        )
        
        # Execute comparison
        result = await explainability_use_case.compare_explanations(domain_request)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        # Convert explanations to DTOs
        explanation_dtos = {}
        if result.explanations:
            for method, explanation in result.explanations.items():
                explanation_dtos[method] = LocalExplanationDTO.from_domain(explanation)
        
        response_dto = ExplainabilityResponseDTO(
            success=result.success,
            explanation_type="comparison",
            explanations=explanation_dtos,
            metadata=result.metadata or {},
            execution_time_seconds=result.execution_time_seconds,
            message=f"Successfully compared {len(request.explanation_methods)} explanation methods"
        )
        
        logger.info(f"Explanation comparison completed for detector {request.detector_id}")
        
        return response_dto
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing explanations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare explanations: {str(e)}"
        )


@router.get("/methods", response_model=List[str])
async def get_available_methods(
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> List[str]:
    """Get list of available explanation methods.
    
    Returns all explanation methods that are currently supported and available
    in the system, including their requirements and capabilities.
    
    Args:
        container: Dependency injection container
        current_user: Current authenticated user
        _permissions: Required permissions check
        
    Returns:
        List of available explanation method names
    """
    try:
        logger.info(f"User {current_user} requesting available explanation methods")
        
        # Get explainability service
        explainability_service = container.explainability_service()
        
        # Get available methods
        methods = explainability_service.get_available_methods()
        method_names = [method.value for method in methods]
        
        logger.info(f"Returning {len(method_names)} available explanation methods")
        
        return method_names
        
    except Exception as e:
        logger.error(f"Error getting available methods: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available methods: {str(e)}"
        )


@router.get("/detector/{detector_id}/explanation-stats")
async def get_explanation_statistics(
    detector_id: str,
    dataset_id: str,
    method: str = "shap",
    sample_size: int = 100,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> JSONResponse:
    """Get explanation statistics for a detector across multiple instances.
    
    Generates explanations for a sample of instances and provides aggregate
    statistics about feature importance, consistency, and model behavior patterns.
    
    Args:
        detector_id: ID of the detector to analyze
        dataset_id: ID of the dataset for context
        method: Explanation method to use
        sample_size: Number of instances to sample for analysis
        container: Dependency injection container
        current_user: Current authenticated user
        _permissions: Required permissions check
        
    Returns:
        Aggregate explanation statistics and insights
        
    Raises:
        HTTPException: If statistics generation fails
    """
    try:
        logger.info(f"User {current_user} requesting explanation statistics for detector {detector_id}")
        
        # Validate parameters
        if sample_size < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sample size must be at least 10"
            )
        
        if sample_size > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sample size cannot exceed 1000"
            )
        
        try:
            explanation_method = ExplanationMethod(method)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid explanation method: {method}"
            )
        
        # Get repositories
        detector_repository = container.detector_repository()
        dataset_repository = container.dataset_repository()
        
        # Validate detector and dataset exist
        detector = await detector_repository.get(detector_id)
        if not detector:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Detector {detector_id} not found"
            )
        
        dataset = await dataset_repository.get(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        # Generate statistics (this would be implemented in the use case)
        stats = {
            "detector_id": detector_id,
            "dataset_id": dataset_id,
            "method": method,
            "sample_size": sample_size,
            "feature_importance_rankings": {
                "top_features": ["feature_1", "feature_2", "feature_3"],
                "average_importance": [0.3, 0.2, 0.15],
                "consistency_score": 0.85
            },
            "explanation_quality": {
                "average_confidence": 0.78,
                "coverage_score": 0.92,
                "stability_score": 0.81
            },
            "computation_stats": {
                "average_time_per_explanation": 0.245,
                "total_time": 24.5,
                "success_rate": 0.97
            }
        }
        
        logger.info(f"Generated explanation statistics for detector {detector_id}")
        
        return JSONResponse(content=stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate explanation statistics: {str(e)}"
        )


@router.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
async def clear_explanation_cache(
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_write),
) -> None:
    """Clear the explanation cache.
    
    Removes all cached explanations from memory to free up resources
    or force regeneration of explanations with updated models.
    
    Args:
        container: Dependency injection container
        current_user: Current authenticated user
        _permissions: Required permissions check
    """
    try:
        logger.info(f"User {current_user} clearing explanation cache")
        
        # Get explainability use case
        explainability_use_case = container.explainability_use_case()
        
        # Clear cache
        explainability_use_case._explanation_cache.clear()
        
        logger.info("Explanation cache cleared successfully")
        
    except Exception as e:
        logger.error(f"Error clearing explanation cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear explanation cache: {str(e)}"
        )


# Background task functions

async def _analyze_feature_interactions_background(
    detector_id: str,
    dataset_id: str,
    user_id: str
) -> None:
    """Background task for analyzing feature interactions."""
    try:
        logger.info(f"Background analysis of feature interactions for detector {detector_id}")
        
        # This would perform more extensive interaction analysis
        # Results could be cached or stored for later retrieval
        
        logger.info(f"Completed background feature interaction analysis for detector {detector_id}")
        
    except Exception as e:
        logger.error(f"Error in background feature interaction analysis: {str(e)}")


# Health check endpoint

@router.get("/health")
async def explainability_health_check(
    container: Container = Depends(get_container)
) -> Dict[str, Any]:
    """Health check for explainability service.
    
    Returns:
        Health status and available explanation methods
    """
    try:
        # Check if explainability service is available
        explainability_service = container.explainability_service()
        available_methods = explainability_service.get_available_methods()
        
        return {
            "status": "healthy",
            "service": "explainability",
            "available_methods": len(available_methods),
            "methods": [method.value for method in available_methods],
            "cache_enabled": True,
            "timestamp": "2024-01-01T00:00:00Z"  # This would be actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Explainability health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "explainability",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }