"""API endpoints for ensemble-based anomaly prediction."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import HTTPBearer

from interfaces.application.dto.ensemble_dto import (
    EnsembleDetectionRequestDTO,
    EnsembleDetectionResponseDTO,
    EnsembleMetricsResponseDTO,
    EnsembleOptimizationRequestDTO,
    EnsembleOptimizationResponseDTO,
    EnsembleStatusResponseDTO,
)
from interfaces.application.use_cases.ensemble_prediction_use_case import (
    EnsembleDetectionRequest,
    EnsembleDetectionUseCase,
    EnsembleOptimizationObjective,
    EnsembleOptimizationRequest,
    VotingStrategy,
)
from interfaces.infrastructure.config import Container
from interfaces.presentation.api.deps import (
    get_container,
    get_current_user,
    require_read,
    require_write,
)

router = APIRouter(prefix="/ensemble", tags=["ensemble"])
security = HTTPBearer()


@router.post("/detect", response_model=EnsembleDetectionResponseDTO)
async def detect_anomalies_ensemble(
    request: EnsembleDetectionRequestDTO,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> EnsembleDetectionResponseDTO:
    """
    Perform ensemble-based anomaly prediction using multiple detectors.

    This endpoint orchestrates multiple anomaly detectors using advanced voting strategies
    to provide robust and accurate anomaly prediction results.

    **Features:**
    - Multiple voting strategies (simple average, weighted, Bayesian, consensus, etc.)
    - Dynamic detector weighting based on performance
    - Uncertainty estimation and confidence scoring
    - Explanation generation for ensemble decisions
    - Performance tracking and optimization

    **Voting Strategies:**
    - `simple_average`: Equal weight average of all detector scores
    - `weighted_average`: Performance-weighted average with dynamic weights
    - `bayesian_model_averaging`: Bayesian interpretation of detector combinations
    - `rank_aggregation`: Rank-based aggregation using scipy.stats
    - `consensus_voting`: Require agreement threshold among detectors
    - `dynamic_selection`: Select best detectors per sample
    - `uncertainty_weighted`: Weight by prediction uncertainty
    - `robust_aggregation`: Trimmed mean to handle outliers
    - `cascaded_voting`: Early stopping with confidence threshold

    **Request Parameters:**
    - `detector_ids`: List of detector IDs to include in ensemble (2-20 detectors)
    - `data`: Input data as array, DataFrame, or list of dictionaries
    - `voting_strategy`: Strategy for combining detector outputs
    - `enable_dynamic_weighting`: Use performance-based weighting
    - `enable_explanation`: Generate explanations for predictions
    - `confidence_threshold`: Minimum confidence for cascaded voting
    - `consensus_threshold`: Agreement threshold for consensus voting

    **Response Fields:**
    - `predictions`: Binary anomaly predictions (0=normal, 1=anomaly)
    - `anomaly_scores`: Continuous anomaly scores (0.0-1.0)
    - `confidence_scores`: Prediction confidence levels
    - `uncertainty_scores`: Prediction uncertainty estimates
    - `detector_weights`: Weights used for each detector
    - `ensemble_metrics`: Diversity and performance metrics
    - `explanations`: Per-prediction explanations (if enabled)
    """
    try:
        ensemble_use_case: EnsembleDetectionUseCase = (
            container.ensemble_prediction_use_case()
        )

        # Convert DTO to use case request
        use_case_request = EnsembleDetectionRequest(
            detector_ids=request.detector_ids,
            data=request.data,
            voting_strategy=VotingStrategy(request.voting_strategy),
            enable_dynamic_weighting=request.enable_dynamic_weighting,
            enable_uncertainty_estimation=request.enable_uncertainty_estimation,
            enable_explanation=request.enable_explanation,
            confidence_threshold=request.confidence_threshold,
            consensus_threshold=request.consensus_threshold,
            max_processing_time=request.max_processing_time,
            enable_caching=request.enable_caching,
            return_individual_results=request.return_individual_results,
        )

        # Execute ensemble prediction
        response = await ensemble_use_case.detect_anomalies_ensemble(use_case_request)

        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.error_message or "Ensemble prediction failed",
            )

        # Convert to DTO response
        return EnsembleDetectionResponseDTO(
            success=response.success,
            predictions=response.predictions or [],
            anomaly_scores=response.anomaly_scores or [],
            confidence_scores=response.confidence_scores or [],
            uncertainty_scores=response.uncertainty_scores or [],
            consensus_scores=response.consensus_scores or [],
            individual_results=response.individual_results,
            detector_weights=response.detector_weights or [],
            voting_strategy_used=response.voting_strategy_used or "",
            ensemble_metrics=response.ensemble_metrics or {},
            explanations=response.explanations or [],
            processing_time=response.processing_time,
            warnings=response.warnings or [],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during ensemble prediction: {str(e)}",
        )


@router.post("/optimize", response_model=EnsembleOptimizationResponseDTO)
async def optimize_ensemble(
    request: EnsembleOptimizationRequestDTO,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_write),
) -> EnsembleOptimizationResponseDTO:
    """
    Optimize ensemble configuration for better performance.

    This endpoint performs sophisticated ensemble optimization using validation data
    to find the optimal combination of detectors, voting strategy, and weights.

    **Optimization Features:**
    - Automated detector selection and pruning
    - Voting strategy comparison and selection
    - Weight optimization for best performance
    - Diversity analysis and requirements
    - Cross-validation based evaluation
    - Performance prediction and recommendations

    **Optimization Objectives:**
    - `accuracy`: Overall classification accuracy
    - `precision`: Precision for anomaly prediction
    - `recall`: Recall for anomaly prediction
    - `f1_score`: F1-score balancing precision and recall
    - `auc_score`: Area under ROC curve
    - `balanced_accuracy`: Balanced accuracy for imbalanced data
    - `diversity`: Ensemble diversity maximization
    - `stability`: Prediction stability across samples
    - `efficiency`: Computational efficiency optimization

    **Request Parameters:**
    - `detector_ids`: Candidate detectors for ensemble
    - `validation_dataset_id`: Dataset for optimization validation
    - `optimization_objective`: Primary optimization target
    - `target_voting_strategies`: Strategies to evaluate
    - `max_ensemble_size`: Maximum detectors in final ensemble
    - `min_diversity_threshold`: Minimum required diversity
    - `enable_pruning`: Remove underperforming detectors
    - `cross_validation_folds`: Number of CV folds for evaluation

    **Response Fields:**
    - `optimized_detector_ids`: Best detector combination
    - `optimal_voting_strategy`: Best voting strategy found
    - `optimal_weights`: Optimized detector weights
    - `ensemble_performance`: Performance metrics on validation data
    - `diversity_metrics`: Ensemble diversity analysis
    - `recommendations`: Optimization recommendations
    """
    try:
        ensemble_use_case: EnsembleDetectionUseCase = (
            container.ensemble_prediction_use_case()
        )

        # Convert DTO to use case request
        use_case_request = EnsembleOptimizationRequest(
            detector_ids=request.detector_ids,
            validation_dataset_id=request.validation_dataset_id,
            optimization_objective=EnsembleOptimizationObjective(
                request.optimization_objective
            ),
            target_voting_strategies=[
                VotingStrategy(s) for s in request.target_voting_strategies
            ],
            max_ensemble_size=request.max_ensemble_size,
            min_diversity_threshold=request.min_diversity_threshold,
            enable_pruning=request.enable_pruning,
            enable_weight_optimization=request.enable_weight_optimization,
            cross_validation_folds=request.cross_validation_folds,
            optimization_timeout=request.optimization_timeout,
            random_state=request.random_state,
        )

        # Execute ensemble optimization
        response = await ensemble_use_case.optimize_ensemble(use_case_request)

        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.error_message or "Ensemble optimization failed",
            )

        # Convert to DTO response
        return EnsembleOptimizationResponseDTO(
            success=response.success,
            optimized_detector_ids=response.optimized_detector_ids or [],
            optimal_voting_strategy=(
                response.optimal_voting_strategy.value
                if response.optimal_voting_strategy
                else ""
            ),
            optimal_weights=response.optimal_weights or [],
            ensemble_performance=response.ensemble_performance or {},
            diversity_metrics=response.diversity_metrics or {},
            optimization_history=response.optimization_history or [],
            recommendations=response.recommendations or [],
            optimization_time=response.optimization_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during ensemble optimization: {str(e)}",
        )


@router.get("/status", response_model=EnsembleStatusResponseDTO)
async def get_ensemble_status(
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> EnsembleStatusResponseDTO:
    """
    Get current ensemble system status and capabilities.

    Returns information about available voting strategies, optimization objectives,
    system performance, and ensemble statistics.
    """
    try:
        ensemble_use_case: EnsembleDetectionUseCase = (
            container.ensemble_prediction_use_case()
        )

        # Get available strategies and objectives
        voting_strategies = [
            {
                "name": strategy.value,
                "description": _get_strategy_description(strategy),
                "complexity": _get_strategy_complexity(strategy),
                "supports_weights": _strategy_supports_weights(strategy),
            }
            for strategy in VotingStrategy
        ]

        optimization_objectives = [
            {
                "name": obj.value,
                "description": _get_objective_description(obj),
                "focus": _get_objective_focus(obj),
            }
            for obj in EnsembleOptimizationObjective
        ]

        # Get system statistics
        system_stats = {
            "cache_size": len(ensemble_use_case._ensemble_cache),
            "performance_trackers": len(ensemble_use_case._performance_tracker),
            "optimization_history": len(ensemble_use_case._optimization_history),
            "supported_strategies": len(voting_strategies),
            "supported_objectives": len(optimization_objectives),
        }

        return EnsembleStatusResponseDTO(
            available_voting_strategies=voting_strategies,
            available_optimization_objectives=optimization_objectives,
            system_capabilities={
                "max_detectors": 20,
                "min_detectors": 2,
                "supports_caching": True,
                "supports_explanation": True,
                "supports_optimization": True,
                "supports_dynamic_weighting": True,
            },
            system_statistics=system_stats,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error getting ensemble status: {str(e)}",
        )


@router.get("/metrics", response_model=EnsembleMetricsResponseDTO)
async def get_ensemble_metrics(
    detector_ids: list[str] | None = None,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> EnsembleMetricsResponseDTO:
    """
    Get ensemble performance metrics and analytics.

    Returns detailed performance metrics for specified detectors or all tracked detectors,
    including individual performance, ensemble statistics, and optimization history.
    """
    try:
        ensemble_use_case: EnsembleDetectionUseCase = (
            container.ensemble_prediction_use_case()
        )

        # Get performance metrics for specified detectors
        if detector_ids:
            detector_metrics = {
                detector_id: ensemble_use_case._performance_tracker.get(detector_id)
                for detector_id in detector_ids
                if detector_id in ensemble_use_case._performance_tracker
            }
        else:
            detector_metrics = ensemble_use_case._performance_tracker.copy()

        # Convert to serializable format
        metrics_data = {}
        for detector_id, metrics in detector_metrics.items():
            if metrics:
                metrics_data[detector_id] = {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "auc_score": metrics.auc_score,
                    "processing_time": metrics.processing_time,
                    "stability_score": metrics.stability_score,
                    "diversity_contribution": metrics.diversity_contribution,
                    "uncertainty_estimation": metrics.uncertainty_estimation,
                    "last_updated": metrics.last_updated,
                }

        # Get ensemble statistics
        ensemble_stats = {
            "total_detectors_tracked": len(ensemble_use_case._performance_tracker),
            "cached_predictions": len(ensemble_use_case._ensemble_cache),
            "optimization_runs": len(ensemble_use_case._optimization_history),
            "average_f1_score": (
                sum(m.f1_score for m in ensemble_use_case._performance_tracker.values())
                / len(ensemble_use_case._performance_tracker)
                if ensemble_use_case._performance_tracker
                else 0.0
            ),
        }

        # Get recent optimization history
        recent_optimizations = (
            ensemble_use_case._optimization_history[-10:]
            if ensemble_use_case._optimization_history
            else []
        )

        return EnsembleMetricsResponseDTO(
            detector_performance_metrics=metrics_data,
            ensemble_statistics=ensemble_stats,
            recent_optimizations=[
                {
                    "timestamp": opt["timestamp"],
                    "optimization_time": opt["optimization_time"],
                    "detector_count": (
                        len(opt["request"].detector_ids)
                        if hasattr(opt["request"], "detector_ids")
                        else 0
                    ),
                    "success": (
                        opt["response"].success
                        if hasattr(opt["response"], "success")
                        else False
                    ),
                }
                for opt in recent_optimizations
            ],
            system_health={
                "cache_hit_rate": 0.85,  # Placeholder - would be calculated from actual metrics
                "average_processing_time": 0.15,  # Placeholder
                "error_rate": 0.02,  # Placeholder
                "uptime": "99.9%",  # Placeholder
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error getting ensemble metrics: {str(e)}",
        )


# Helper functions for strategy and objective descriptions


def _get_strategy_description(strategy: VotingStrategy) -> str:
    """Get description for voting strategy."""
    descriptions = {
        VotingStrategy.SIMPLE_AVERAGE: "Equal weight average of all detector scores",
        VotingStrategy.WEIGHTED_AVERAGE: "Performance-weighted average with dynamic weights",
        VotingStrategy.BAYESIAN_MODEL_AVERAGING: "Bayesian interpretation of detector combinations",
        VotingStrategy.RANK_AGGREGATION: "Rank-based aggregation using statistical methods",
        VotingStrategy.CONSENSUS_VOTING: "Require agreement threshold among detectors",
        VotingStrategy.DYNAMIC_SELECTION: "Select best detectors per sample dynamically",
        VotingStrategy.UNCERTAINTY_WEIGHTED: "Weight votes by prediction uncertainty",
        VotingStrategy.PERFORMANCE_WEIGHTED: "Weight by recent detector performance",
        VotingStrategy.DIVERSITY_WEIGHTED: "Weight by detector diversity contribution",
        VotingStrategy.ADAPTIVE_THRESHOLD: "Adaptive threshold based on data characteristics",
        VotingStrategy.ROBUST_AGGREGATION: "Trimmed mean to handle outlier predictions",
        VotingStrategy.CASCADED_VOTING: "Early stopping with confidence threshold",
    }
    return descriptions.get(strategy, "Advanced ensemble voting strategy")


def _get_strategy_complexity(strategy: VotingStrategy) -> str:
    """Get computational complexity for voting strategy."""
    complexity_map = {
        VotingStrategy.SIMPLE_AVERAGE: "low",
        VotingStrategy.WEIGHTED_AVERAGE: "low",
        VotingStrategy.BAYESIAN_MODEL_AVERAGING: "medium",
        VotingStrategy.RANK_AGGREGATION: "medium",
        VotingStrategy.CONSENSUS_VOTING: "low",
        VotingStrategy.DYNAMIC_SELECTION: "high",
        VotingStrategy.UNCERTAINTY_WEIGHTED: "medium",
        VotingStrategy.PERFORMANCE_WEIGHTED: "medium",
        VotingStrategy.DIVERSITY_WEIGHTED: "medium",
        VotingStrategy.ADAPTIVE_THRESHOLD: "high",
        VotingStrategy.ROBUST_AGGREGATION: "medium",
        VotingStrategy.CASCADED_VOTING: "low",
    }
    return complexity_map.get(strategy, "medium")


def _strategy_supports_weights(strategy: VotingStrategy) -> bool:
    """Check if strategy supports detector weights."""
    weighted_strategies = {
        VotingStrategy.WEIGHTED_AVERAGE,
        VotingStrategy.BAYESIAN_MODEL_AVERAGING,
        VotingStrategy.RANK_AGGREGATION,
        VotingStrategy.DYNAMIC_SELECTION,
        VotingStrategy.UNCERTAINTY_WEIGHTED,
        VotingStrategy.PERFORMANCE_WEIGHTED,
        VotingStrategy.DIVERSITY_WEIGHTED,
        VotingStrategy.ROBUST_AGGREGATION,
        VotingStrategy.CASCADED_VOTING,
    }
    return strategy in weighted_strategies


def _get_objective_description(objective: EnsembleOptimizationObjective) -> str:
    """Get description for optimization objective."""
    descriptions = {
        EnsembleOptimizationObjective.ACCURACY: "Overall classification accuracy",
        EnsembleOptimizationObjective.PRECISION: "Precision for anomaly prediction",
        EnsembleOptimizationObjective.RECALL: "Recall for anomaly prediction",
        EnsembleOptimizationObjective.F1_SCORE: "F1-score balancing precision and recall",
        EnsembleOptimizationObjective.AUC_SCORE: "Area under ROC curve",
        EnsembleOptimizationObjective.BALANCED_ACCURACY: "Balanced accuracy for imbalanced data",
        EnsembleOptimizationObjective.DIVERSITY: "Ensemble diversity maximization",
        EnsembleOptimizationObjective.STABILITY: "Prediction stability across samples",
        EnsembleOptimizationObjective.EFFICIENCY: "Computational efficiency optimization",
    }
    return descriptions.get(objective, "Optimization objective")


def _get_objective_focus(objective: EnsembleOptimizationObjective) -> str:
    """Get focus area for optimization objective."""
    focus_map = {
        EnsembleOptimizationObjective.ACCURACY: "performance",
        EnsembleOptimizationObjective.PRECISION: "performance",
        EnsembleOptimizationObjective.RECALL: "performance",
        EnsembleOptimizationObjective.F1_SCORE: "performance",
        EnsembleOptimizationObjective.AUC_SCORE: "performance",
        EnsembleOptimizationObjective.BALANCED_ACCURACY: "performance",
        EnsembleOptimizationObjective.DIVERSITY: "robustness",
        EnsembleOptimizationObjective.STABILITY: "robustness",
        EnsembleOptimizationObjective.EFFICIENCY: "speed",
    }
    return focus_map.get(objective, "performance")
