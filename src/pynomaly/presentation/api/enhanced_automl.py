"""Enhanced AutoML API endpoints with advanced optimization capabilities."""

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from pynomaly.application.services.enhanced_automl_service import (
    EnhancedAutoMLConfig,
    EnhancedAutoMLResult,
    EnhancedAutoMLService,
)
from pynomaly.infrastructure.config.container import Container


# Request/Response Models
class OptimizationObjectiveRequest(BaseModel):
    """Request model for optimization objective."""

    name: str = Field(
        ..., description="Objective name (auc, precision, recall, training_time, etc.)"
    )
    direction: str = Field(
        "maximize", description="Optimization direction (maximize/minimize)"
    )
    weight: float = Field(
        1.0, description="Objective weight for multi-objective optimization"
    )


class AdvancedOptimizationRequest(BaseModel):
    """Request model for advanced hyperparameter optimization."""

    dataset_id: str = Field(..., description="Dataset identifier")
    algorithm: str = Field(..., description="Algorithm name")
    objectives: list[str] = Field(
        default=["auc"], description="Optimization objectives"
    )
    strategy: str = Field(default="bayesian", description="Optimization strategy")
    acquisition_function: str = Field(
        default="expected_improvement", description="Acquisition function"
    )
    n_trials: int = Field(default=100, description="Number of optimization trials")
    timeout: int = Field(default=3600, description="Optimization timeout in seconds")
    enable_meta_learning: bool = Field(default=True, description="Enable meta-learning")
    enable_parallel: bool = Field(
        default=True, description="Enable parallel optimization"
    )
    enable_early_stopping: bool = Field(
        default=True, description="Enable early stopping"
    )


class AutoMLRequest(BaseModel):
    """Request model for automatic algorithm selection and optimization."""

    dataset_id: str = Field(..., description="Dataset identifier")
    objectives: list[str] = Field(
        default=["auc"], description="Optimization objectives"
    )
    max_algorithms: int = Field(default=3, description="Maximum algorithms to try")
    strategy: str = Field(default="bayesian", description="Optimization strategy")
    n_trials: int = Field(default=100, description="Number of trials per algorithm")
    timeout: int = Field(default=3600, description="Total optimization timeout")
    enable_ensemble: bool = Field(default=True, description="Enable ensemble creation")
    enable_meta_learning: bool = Field(default=True, description="Enable meta-learning")


class MultiObjectiveRequest(BaseModel):
    """Request model for multi-objective optimization."""

    dataset_id: str = Field(..., description="Dataset identifier")
    objectives: list[OptimizationObjectiveRequest] = Field(
        ..., description="Multiple objectives"
    )
    algorithms: list[str] | None = Field(
        None, description="Specific algorithms to optimize"
    )
    n_trials: int = Field(default=150, description="Number of optimization trials")
    timeout: int = Field(default=7200, description="Optimization timeout in seconds")


class OptimizationInsightsResponse(BaseModel):
    """Response model for optimization insights."""

    performance_analysis: dict[str, Any]
    optimization_analysis: dict[str, Any]
    efficiency_analysis: dict[str, Any]
    recommendations: list[str]
    next_steps: list[str]
    parameter_sensitivity: dict[str, float]


class EnhancedAutoMLResultResponse(BaseModel):
    """Response model for enhanced AutoML results."""

    best_algorithm: str
    best_params: dict[str, Any]
    best_score: float
    optimization_time: float
    trials_completed: int
    algorithm_rankings: list[tuple]

    # Enhanced fields
    optimization_strategy_used: str
    meta_learning_effectiveness: float | None = None
    exploration_score: float
    exploitation_score: float
    convergence_stability: float
    parameter_sensitivity: dict[str, float]

    # Multi-objective results
    pareto_front: list[dict[str, Any]] | None = None
    objective_trade_offs: dict[str, float] | None = None

    # Performance analysis
    training_time_breakdown: dict[str, float]
    optimization_recommendations: list[str]
    next_steps: list[str]


# Dependency injection
def get_enhanced_automl_service(
    container: Container = Depends(),
) -> EnhancedAutoMLService:
    """Get enhanced AutoML service from container."""
    if hasattr(container, "enhanced_automl_service"):
        return container.enhanced_automl_service()
    else:
        raise HTTPException(
            status_code=503, detail="Enhanced AutoML service not available"
        )


# Router
router = APIRouter(prefix="/api/v1/enhanced-automl", tags=["Enhanced AutoML"])


@router.post("/optimize", response_model=EnhancedAutoMLResultResponse)
async def optimize_hyperparameters(
    request: AdvancedOptimizationRequest,
    background_tasks: BackgroundTasks,
    service: EnhancedAutoMLService = Depends(get_enhanced_automl_service),
):
    """
    Run advanced hyperparameter optimization for a specific algorithm.

    This endpoint provides state-of-the-art hyperparameter optimization with:
    - Bayesian optimization with advanced acquisition functions
    - Meta-learning for warm starts
    - Multi-objective optimization
    - Automated early stopping
    - Parallel optimization support
    """
    try:
        # Configure enhanced AutoML
        from pynomaly.infrastructure.automl import (
            AcquisitionFunction,
            OptimizationStrategy,
        )

        strategy_enum = OptimizationStrategy(request.strategy)
        acquisition_enum = AcquisitionFunction(request.acquisition_function)

        config = EnhancedAutoMLConfig(
            optimization_strategy=strategy_enum,
            acquisition_function=acquisition_enum,
            n_trials=request.n_trials,
            max_optimization_time=request.timeout,
            enable_meta_learning=request.enable_meta_learning,
            enable_multi_objective=len(request.objectives) > 1,
            objectives=request.objectives,
            enable_parallel=request.enable_parallel,
            enable_early_stopping=request.enable_early_stopping,
        )

        # Update service configuration
        service.config = config

        # Run optimization
        result = await service.advanced_optimize_hyperparameters(
            dataset_id=request.dataset_id,
            algorithm=request.algorithm,
            objectives=request.objectives,
        )

        return _convert_result_to_response(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/auto-optimize", response_model=EnhancedAutoMLResultResponse)
async def auto_optimize(
    request: AutoMLRequest,
    background_tasks: BackgroundTasks,
    service: EnhancedAutoMLService = Depends(get_enhanced_automl_service),
):
    """
    Automatically select and optimize the best algorithms for a dataset.

    This endpoint provides comprehensive AutoML capabilities:
    - Automatic algorithm recommendation based on dataset characteristics
    - Advanced hyperparameter optimization for multiple algorithms
    - Ensemble creation from top-performing models
    - Meta-learning for improved efficiency
    """
    try:
        # Configure enhanced AutoML
        from pynomaly.infrastructure.automl import OptimizationStrategy

        strategy_enum = OptimizationStrategy(request.strategy)

        config = EnhancedAutoMLConfig(
            optimization_strategy=strategy_enum,
            n_trials=request.n_trials,
            max_optimization_time=request.timeout,
            enable_meta_learning=request.enable_meta_learning,
            enable_multi_objective=len(request.objectives) > 1,
            objectives=request.objectives,
            enable_ensemble_optimization=request.enable_ensemble,
        )

        # Update service configuration
        service.config = config

        # Run auto-optimization
        result = await service.auto_select_and_optimize_advanced(
            dataset_id=request.dataset_id,
            objectives=request.objectives,
            max_algorithms=request.max_algorithms,
            enable_ensemble=request.enable_ensemble,
        )

        return _convert_result_to_response(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AutoML failed: {str(e)}")


@router.post("/multi-objective", response_model=EnhancedAutoMLResultResponse)
async def multi_objective_optimization(
    request: MultiObjectiveRequest,
    background_tasks: BackgroundTasks,
    service: EnhancedAutoMLService = Depends(get_enhanced_automl_service),
):
    """
    Run multi-objective optimization to find Pareto optimal solutions.

    This endpoint finds trade-offs between multiple objectives such as:
    - Performance vs Speed
    - Accuracy vs Memory Usage
    - Precision vs Recall
    - Detection Rate vs False Positive Rate
    """
    try:
        # Extract objective names and weights
        objective_names = [obj.name for obj in request.objectives]
        objective_weights = {obj.name: obj.weight for obj in request.objectives}

        # Configure for multi-objective optimization
        config = EnhancedAutoMLConfig(
            optimization_strategy="multi_objective",
            n_trials=request.n_trials,
            max_optimization_time=request.timeout,
            enable_multi_objective=True,
            objectives=objective_names,
            objective_weights=objective_weights,
            enable_meta_learning=True,
        )

        # Update service configuration
        service.config = config

        if request.algorithms:
            # Optimize specified algorithms
            results = []
            for algorithm in request.algorithms:
                try:
                    result = await service.advanced_optimize_hyperparameters(
                        dataset_id=request.dataset_id,
                        algorithm=algorithm,
                        objectives=objective_names,
                    )
                    results.append(result)
                except Exception:
                    # Log warning but continue with other algorithms
                    pass

            if not results:
                raise HTTPException(
                    status_code=400,
                    detail="No algorithms could be successfully optimized",
                )

            # Return best result (could be extended to return all results)
            best_result = max(results, key=lambda x: x.best_score)
        else:
            # Auto-select and optimize
            best_result = await service.auto_select_and_optimize_advanced(
                dataset_id=request.dataset_id,
                objectives=objective_names,
                max_algorithms=5,
            )

        return _convert_result_to_response(best_result)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Multi-objective optimization failed: {str(e)}"
        )


@router.get("/insights/{optimization_id}", response_model=OptimizationInsightsResponse)
async def get_optimization_insights(
    optimization_id: str,
    service: EnhancedAutoMLService = Depends(get_enhanced_automl_service),
):
    """
    Get detailed insights and recommendations from optimization results.

    Provides analysis of:
    - Optimization quality metrics (exploration/exploitation balance)
    - Parameter sensitivity analysis
    - Convergence stability assessment
    - Recommendations for improvement
    - Suggested next steps
    """
    try:
        # This would typically load results from storage
        # For now, we'll return a placeholder implementation
        insights = {
            "performance_analysis": {
                "score_category": "good",
                "ensemble_available": False,
            },
            "optimization_analysis": {
                "strategy_used": "bayesian",
                "exploration_score": 0.7,
                "exploitation_score": 0.8,
                "convergence_stability": 0.75,
            },
            "efficiency_analysis": {
                "total_optimization_time": 120.0,
                "trials_completed": 50,
                "time_per_trial": 2.4,
            },
            "recommendations": [
                "Consider increasing n_trials for better exploration",
                "Current configuration shows good convergence",
            ],
            "next_steps": [
                "Try ensemble methods for improved performance",
                "Consider feature engineering for better results",
            ],
            "parameter_sensitivity": {
                "contamination": 0.8,
                "n_estimators": 0.6,
                "max_features": 0.4,
            },
        }

        return OptimizationInsightsResponse(**insights)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")


@router.get("/algorithms/recommendations/{dataset_id}")
async def get_algorithm_recommendations(
    dataset_id: str,
    max_algorithms: int = 5,
    service: EnhancedAutoMLService = Depends(get_enhanced_automl_service),
):
    """
    Get algorithm recommendations based on dataset characteristics.

    Analyzes dataset properties and recommends the most suitable algorithms
    considering factors like:
    - Dataset size and dimensionality
    - Data types (numerical, categorical, temporal)
    - Missing values and sparsity
    - Computational constraints
    """
    try:
        # Profile dataset
        profile = await service.profile_dataset(dataset_id)

        # Get recommendations
        recommendations = service.recommend_algorithms(profile, max_algorithms)

        # Get algorithm details
        algorithm_details = []
        for algorithm in recommendations:
            if algorithm in service.algorithm_configs:
                config = service.algorithm_configs[algorithm]
                details = {
                    "name": algorithm,
                    "family": config.family.value,
                    "complexity_score": config.complexity_score,
                    "training_time_factor": config.training_time_factor,
                    "memory_factor": config.memory_factor,
                    "recommended_min_samples": config.recommended_min_samples,
                    "recommended_max_samples": config.recommended_max_samples,
                    "supports_streaming": config.supports_streaming,
                    "supports_categorical": config.supports_categorical,
                }
                algorithm_details.append(details)

        return {
            "dataset_profile": {
                "n_samples": profile.n_samples,
                "n_features": profile.n_features,
                "contamination_estimate": profile.contamination_estimate,
                "complexity_score": profile.complexity_score,
                "has_temporal_structure": profile.has_temporal_structure,
                "has_graph_structure": profile.has_graph_structure,
            },
            "recommendations": algorithm_details,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get recommendations: {str(e)}"
        )


@router.get("/strategies")
async def get_optimization_strategies():
    """
    Get available optimization strategies and their descriptions.

    Returns information about different optimization approaches:
    - Bayesian optimization variants
    - Hyperband and BOHB
    - Multi-objective optimization
    - Evolutionary algorithms
    """
    strategies = {
        "bayesian": {
            "name": "Bayesian Optimization",
            "description": "Uses Gaussian processes to model the objective function and select promising parameters",
            "best_for": [
                "Small to medium parameter spaces",
                "Expensive evaluations",
                "Continuous parameters",
            ],
            "acquisition_functions": [
                "expected_improvement",
                "probability_improvement",
                "upper_confidence_bound",
            ],
        },
        "hyperband": {
            "name": "Hyperband",
            "description": "Bandit-based approach that efficiently allocates resources across configurations",
            "best_for": [
                "Large parameter spaces",
                "Iterative algorithms",
                "Quick elimination of poor configs",
            ],
            "acquisition_functions": ["random"],
        },
        "bohb": {
            "name": "BOHB (Bayesian Optimization and Hyperband)",
            "description": "Combines Bayesian optimization with Hyperband for efficient search",
            "best_for": [
                "Large parameter spaces",
                "Expensive evaluations",
                "Best of both worlds",
            ],
            "acquisition_functions": ["expected_improvement"],
        },
        "multi_objective": {
            "name": "Multi-Objective Optimization",
            "description": "Optimizes multiple conflicting objectives simultaneously",
            "best_for": [
                "Trade-off analysis",
                "Multiple performance metrics",
                "Pareto optimization",
            ],
            "acquisition_functions": ["nsga2", "hypervolume"],
        },
        "evolutionary": {
            "name": "Evolutionary Algorithm",
            "description": "Population-based search inspired by natural evolution",
            "best_for": [
                "Complex parameter spaces",
                "Non-convex objectives",
                "Parallel evaluation",
            ],
            "acquisition_functions": ["cma_es"],
        },
    }

    return strategies


def _convert_result_to_response(
    result: EnhancedAutoMLResult,
) -> EnhancedAutoMLResultResponse:
    """Convert enhanced AutoML result to API response."""
    return EnhancedAutoMLResultResponse(
        best_algorithm=result.best_algorithm,
        best_params=result.best_params,
        best_score=result.best_score,
        optimization_time=result.optimization_time,
        trials_completed=result.trials_completed,
        algorithm_rankings=result.algorithm_rankings,
        optimization_strategy_used=result.optimization_strategy_used,
        meta_learning_effectiveness=result.meta_learning_effectiveness,
        exploration_score=result.exploration_score,
        exploitation_score=result.exploitation_score,
        convergence_stability=result.convergence_stability,
        parameter_sensitivity=result.parameter_sensitivity,
        pareto_front=result.pareto_front,
        objective_trade_offs=result.objective_trade_offs,
        training_time_breakdown=result.training_time_breakdown,
        optimization_recommendations=result.optimization_recommendations,
        next_steps=result.next_steps,
    )
