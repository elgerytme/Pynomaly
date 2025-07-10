"""AutoML API endpoints."""

import logging
import time
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from pynomaly.application.dto.automl_dto import (
    AlgorithmRecommendationDTO,
    AutoMLProfileRequestDTO,
    AutoMLProfileResponseDTO,
    AutoMLRequestDTO,
    AutoMLResponseDTO,
    AutoMLResultDTO,
    DatasetProfileDTO,
    HyperparameterOptimizationRequestDTO,
    HyperparameterOptimizationResponseDTO,
)
from pynomaly.infrastructure.auth import require_read, require_write
from pynomaly.infrastructure.config import Container
from pynomaly.infrastructure.config.feature_flags import require_feature
from pynomaly.presentation.api.deps import get_container, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/profile", response_model=AutoMLProfileResponseDTO)
@require_feature("advanced_automl")
async def profile_dataset(
    request: AutoMLProfileRequestDTO,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> AutoMLProfileResponseDTO:
    """Profile a dataset and get algorithm recommendations."""
    start_time = time.time()

    try:
        logger.info(f"Profiling dataset {request.dataset_id}")

        # Get AutoML use case
        automl_use_case = container.automl_optimization_use_case()

        # Profile dataset
        profile_response = await automl_use_case.profile_dataset(request.dataset_id)

        # Convert to DTO format
        dataset_profile = DatasetProfileDTO(
            n_samples=profile_response.n_samples,
            n_features=profile_response.n_features,
            contamination_estimate=profile_response.contamination_estimate,
            feature_types=profile_response.feature_types,
            missing_values_ratio=profile_response.profile_metadata.get(
                "missing_values_ratio", 0.0
            ),
            categorical_features=[],  # Will be filled from feature_types
            numerical_features=[],  # Will be filled from feature_types
            time_series_features=[],  # Will be filled from feature_types
            sparsity_ratio=profile_response.profile_metadata.get("sparsity_ratio", 0.0),
            dimensionality_ratio=profile_response.profile_metadata.get(
                "dimensionality_ratio", 0.0
            ),
            dataset_size_mb=profile_response.profile_metadata.get(
                "dataset_size_mb", 0.0
            ),
            has_temporal_structure=profile_response.has_temporal_structure,
            has_graph_structure=False,  # Not in basic profile
            complexity_score=profile_response.complexity_score,
        )

        # Extract feature lists from feature_types
        for feature, feature_type in profile_response.feature_types.items():
            if feature_type == "numerical":
                dataset_profile.numerical_features.append(feature)
            elif feature_type == "categorical":
                dataset_profile.categorical_features.append(feature)
            elif feature_type == "datetime":
                dataset_profile.time_series_features.append(feature)

        # Get algorithm recommendations if requested
        algorithm_recommendations = []
        if request.include_recommendations:
            try:
                rec_response = await automl_use_case.get_algorithm_recommendations(
                    request.dataset_id, max_algorithms=request.max_recommendations
                )

                for algorithm in rec_response.recommended_algorithms:
                    algorithm_recommendations.append(
                        AlgorithmRecommendationDTO(
                            algorithm_name=algorithm,
                            score=rec_response.algorithm_scores.get(algorithm, 0.0),
                            family="unknown",  # Would need algorithm config access
                            complexity_score=0.5,  # Default
                            recommended_params={},  # Basic params
                            reasoning=rec_response.reasoning.get(algorithm, "").split(
                                "; "
                            ),
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to get algorithm recommendations: {e}")

        execution_time = time.time() - start_time

        return AutoMLProfileResponseDTO(
            success=True,
            dataset_profile=dataset_profile,
            algorithm_recommendations=(
                algorithm_recommendations if algorithm_recommendations else None
            ),
            message=f"Dataset profiling completed successfully in {execution_time:.2f}s",
            execution_time=execution_time,
        )

    except Exception as e:
        logger.error(f"Dataset profiling failed: {e}")
        execution_time = time.time() - start_time

        return AutoMLProfileResponseDTO(
            success=False,
            message="Dataset profiling failed",
            error=str(e),
            execution_time=execution_time,
        )


@router.post("/optimize", response_model=AutoMLResponseDTO)
@require_feature("advanced_automl")
async def optimize_automl(
    request: AutoMLRequestDTO,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_write),
) -> AutoMLResponseDTO:
    """Run complete AutoML optimization."""
    start_time = time.time()

    try:
        logger.info(f"Starting AutoML optimization for dataset {request.dataset_id}")

        # Get AutoML use case
        automl_use_case = container.automl_optimization_use_case()

        # Convert DTO to use case request format
        from pynomaly.application.dto.automl_dto import AutoMLOptimizationRequest

        optimization_request = AutoMLOptimizationRequest(
            dataset_id=request.dataset_id,
            optimization_objective=request.objective,
            max_algorithms_to_try=request.max_algorithms,
            max_optimization_time_minutes=request.max_optimization_time // 60,
            enable_ensemble=request.enable_ensemble,
            detector_name=request.detector_name,
            cross_validation_folds=request.cross_validation_folds,
            random_state=request.random_state,
        )

        # Run AutoML optimization
        optimization_response = await automl_use_case.auto_optimize(
            optimization_request
        )

        if optimization_response.success:
            # Convert to API DTO format
            automl_result = AutoMLResultDTO(
                best_algorithm=optimization_response.best_algorithm,
                best_params=optimization_response.best_parameters,
                best_score=optimization_response.best_score,
                optimization_time=optimization_response.optimization_time_seconds,
                trials_completed=optimization_response.trials_completed,
                algorithm_rankings=optimization_response.algorithm_rankings,
                ensemble_config=None,  # Would need conversion from dict
                cross_validation_scores=None,  # Not in basic response
                feature_importance=None,  # Not in basic response
                optimization_history=None,  # Not in basic response
            )

            execution_time = time.time() - start_time

            return AutoMLResponseDTO(
                success=True,
                detector_id=optimization_response.optimized_detector_id,
                automl_result=automl_result,
                optimization_summary=optimization_response.optimization_summary,
                message=f"AutoML optimization completed successfully. Created detector: {optimization_response.optimized_detector_id}",
                execution_time=execution_time,
            )
        else:
            execution_time = time.time() - start_time

            return AutoMLResponseDTO(
                success=False,
                message="AutoML optimization failed",
                error=optimization_response.error_message,
                execution_time=execution_time,
            )

    except Exception as e:
        logger.error(f"AutoML optimization failed: {e}")
        execution_time = time.time() - start_time

        return AutoMLResponseDTO(
            success=False,
            message="AutoML optimization failed",
            error=str(e),
            execution_time=execution_time,
        )


@router.post(
    "/optimize-algorithm", response_model=HyperparameterOptimizationResponseDTO
)
@require_feature("advanced_automl")
async def optimize_single_algorithm(
    request: HyperparameterOptimizationRequestDTO,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_write),
) -> HyperparameterOptimizationResponseDTO:
    """Optimize hyperparameters for a specific algorithm."""
    start_time = time.time()

    try:
        logger.info(f"Optimizing {request.algorithm} for dataset {request.dataset_id}")

        # Get AutoML use case
        automl_use_case = container.automl_optimization_use_case()

        # Run single algorithm optimization
        optimization_response = await automl_use_case.optimize_single_algorithm(
            dataset_id=request.dataset_id,
            algorithm_name=request.algorithm,
            optimization_objective=request.objective,
            max_trials=request.n_trials,
        )

        optimization_time = time.time() - start_time

        if optimization_response.success:
            return HyperparameterOptimizationResponseDTO(
                success=True,
                best_params=optimization_response.best_parameters,
                best_score=optimization_response.best_score,
                optimization_time=optimization_time,
                trials_completed=optimization_response.trials_completed,
                algorithm=request.algorithm,
                objective=request.objective,
                message=f"Hyperparameter optimization completed for {request.algorithm}",
            )
        else:
            return HyperparameterOptimizationResponseDTO(
                success=False,
                optimization_time=optimization_time,
                algorithm=request.algorithm,
                objective=request.objective,
                message="Hyperparameter optimization failed",
                error=optimization_response.error_message,
            )

    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        optimization_time = time.time() - start_time

        return HyperparameterOptimizationResponseDTO(
            success=False,
            optimization_time=optimization_time,
            algorithm=request.algorithm,
            objective=request.objective,
            message="Hyperparameter optimization failed",
            error=str(e),
        )


@router.get("/algorithms", response_model=dict)
@require_feature("advanced_automl")
async def list_supported_algorithms(
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> dict:
    """List all supported algorithms for AutoML."""
    try:
        # Import feature_flags for runtime check
        from pynomaly.infrastructure.config.feature_flags import feature_flags
        
        # Check if AutoML features are enabled
        if not feature_flags.is_enabled("automl") and not feature_flags.is_enabled("advanced_automl"):
            return {
                "error": "AutoML features are not enabled",
                "message": "Enable AutoML by setting PYNOMALY_AUTOML=true or PYNOMALY_ADVANCED_AUTOML=true",
                "total_algorithms": 0,
                "all_algorithms": [],
                "by_family": {},
                "automl_enabled": False,
            }

        # Get algorithm adapter registry
        registry = container.algorithm_adapter_registry()

        # Get supported algorithms
        algorithms = registry.get_supported_algorithms()

        # Categorize algorithms (basic categorization)
        algorithm_families = {
            "statistical": ["ECOD", "COPOD"],
            "distance_based": ["KNN", "LOF", "OneClassSVM"],
            "isolation_based": ["IsolationForest"],
            "neural_networks": ["AutoEncoder", "VAE"],
            "ensemble": [],  # Would be populated if ensemble algorithms exist
        }

        # Filter available algorithms by family
        available_by_family = {}
        for family, family_algorithms in algorithm_families.items():
            available_by_family[family] = [
                alg for alg in family_algorithms if alg in algorithms
            ]

        return {
            "total_algorithms": len(algorithms),
            "all_algorithms": sorted(algorithms),
            "by_family": available_by_family,
            "optimization_objectives": [
                "auc",
                "precision",
                "recall",
                "f1_score",
                "detection_rate",
                "balanced_accuracy",
            ],
            "ensemble_methods": ["weighted_voting", "stacking", "bagging"],
            "automl_enabled": True,
            "feature_flags": {
                "automl": feature_flags.is_enabled("automl"),
                "advanced_automl": feature_flags.is_enabled("advanced_automl"),
                "meta_learning": feature_flags.is_enabled("meta_learning"),
                "ensemble_optimization": feature_flags.is_enabled("ensemble_optimization"),
            },
        }

    except Exception as e:
        logger.error(f"Failed to list algorithms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{optimization_id}")
@require_feature("advanced_automl")
async def get_optimization_status(
    optimization_id: UUID,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> dict:
    """Get status of a running AutoML optimization."""
    try:
        # This would require a background task tracking system
        # For now, return a placeholder response

        return {
            "optimization_id": str(optimization_id),
            "status": "completed",  # "running", "completed", "failed"
            "progress_percentage": 100.0,
            "current_algorithm": None,
            "completed_trials": 0,
            "total_trials": 0,
            "best_score_so_far": None,
            "elapsed_time_seconds": 0.0,
            "estimated_remaining_time_seconds": None,
            "message": "Status tracking not yet implemented",
        }

    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/optimization/{optimization_id}")
@require_feature("advanced_automl")
async def cancel_optimization(
    optimization_id: UUID,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_write),
) -> dict:
    """Cancel a running AutoML optimization."""
    try:
        # This would require a background task management system
        # For now, return a placeholder response

        logger.info(f"Cancellation requested for optimization {optimization_id}")

        return {
            "optimization_id": str(optimization_id),
            "cancelled": True,
            "message": "Optimization cancellation requested (not yet implemented)",
        }

    except Exception as e:
        logger.error(f"Failed to cancel optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{dataset_id}")
@require_automl
async def get_algorithm_recommendations(
    dataset_id: str,
    max_recommendations: int = 5,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_read),
) -> dict:
    """Get algorithm recommendations for a specific dataset."""
    try:
        logger.info(f"Getting algorithm recommendations for dataset {dataset_id}")

        # Get AutoML use case
        automl_use_case = container.automl_optimization_use_case()

        # Get recommendations
        response = await automl_use_case.get_algorithm_recommendations(
            dataset_id, max_algorithms=max_recommendations
        )

        # Format response
        recommendations = []
        for algorithm in response.recommended_algorithms:
            recommendations.append(
                {
                    "algorithm": algorithm,
                    "score": response.algorithm_scores.get(algorithm, 0.0),
                    "reasoning": response.reasoning.get(
                        algorithm, "No specific reasoning provided"
                    ),
                }
            )

        return {
            "dataset_id": dataset_id,
            "recommendations": recommendations,
            "dataset_characteristics": response.dataset_characteristics,
            "total_recommendations": len(recommendations),
        }

    except Exception as e:
        logger.error(f"Failed to get algorithm recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run")
@require_automl
async def run_automl(
    dataset_path: str,
    algorithm_name: str,
    max_trials: int = 100,
    storage_path: str = "./automl_storage",
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_write),
) -> dict:
    """Run AutoML hyperparameter optimization for PyOD algorithms."""
    start_time = time.time()

    try:
        logger.info(
            f"Starting AutoML optimization for {algorithm_name} on {dataset_path}"
        )

        # Validate PyOD algorithm
        supported_algorithms = {
            "KNN": "KNN",
            "LOF": "LOCAL_OUTLIER_FACTOR",
            "IsolationForest": "ISOLATION_FOREST",
            "OneClassSVM": "ONE_CLASS_SVM",
            "AutoEncoder": "AUTO_ENCODER",
        }

        if algorithm_name not in supported_algorithms:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {algorithm_name}. Supported: {list(supported_algorithms.keys())}",
            )

        # Get AutoML use case
        automl_use_case = container.automl_optimization_use_case()

        # Create optimization request
        from pynomaly.application.dto.automl_dto import AutoMLOptimizationRequest

        optimization_request = AutoMLOptimizationRequest(
            dataset_id=dataset_path,  # Using path as ID for simplicity
            optimization_objective="f1_score",
            max_algorithms_to_try=1,  # Single algorithm optimization
            max_optimization_time_minutes=30,  # 30 minute limit
            enable_ensemble=False,
            detector_name=f"{algorithm_name}_optimized",
            cross_validation_folds=5,
            random_state=42,
        )

        # Run optimization
        optimization_response = await automl_use_case.auto_optimize(
            optimization_request
        )

        execution_time = time.time() - start_time

        if optimization_response.success:
            # Persist trial results
            trial_data = {
                "algorithm": algorithm_name,
                "best_score": optimization_response.best_score,
                "best_parameters": optimization_response.best_parameters,
                "optimization_time": execution_time,
                "trials_completed": optimization_response.trials_completed,
                "storage_path": storage_path,
            }

            # Save to storage (simplified)
            import json
            import os

            os.makedirs(storage_path, exist_ok=True)
            storage_file = os.path.join(
                storage_path, f"{algorithm_name}_optimization_{int(time.time())}.json"
            )

            with open(storage_file, "w") as f:
                json.dump(trial_data, f, indent=2, default=str)

            # Check success criteria (F1 ≥ baseline + 15%)
            baseline_score = 0.5  # Assumed baseline
            improvement = (
                optimization_response.best_score - baseline_score
            ) / baseline_score
            meets_criteria = improvement >= 0.15

            return {
                "success": True,
                "algorithm": algorithm_name,
                "best_score": optimization_response.best_score,
                "best_parameters": optimization_response.best_parameters,
                "optimization_time": execution_time,
                "trials_completed": optimization_response.trials_completed,
                "storage_file": storage_file,
                "performance_improvement": f"{improvement:.1%}",
                "meets_success_criteria": meets_criteria,
                "detector_id": optimization_response.optimized_detector_id,
                "message": f"AutoML optimization completed for {algorithm_name}",
            }
        else:
            return {
                "success": False,
                "algorithm": algorithm_name,
                "optimization_time": execution_time,
                "error": optimization_response.error_message,
                "message": "AutoML optimization failed",
            }

    except Exception as e:
        logger.error(f"AutoML run failed: {e}")
        execution_time = time.time() - start_time

        return {
            "success": False,
            "algorithm": algorithm_name,
            "optimization_time": execution_time,
            "error": str(e),
            "message": "AutoML optimization failed",
        }


@router.post("/batch-optimize")
@require_feature("advanced_automl")
async def batch_optimize(
    dataset_ids: list[str],
    optimization_objective: str = "auc",
    max_algorithms_per_dataset: int = 3,
    enable_ensemble: bool = True,
    background_tasks: BackgroundTasks = None,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
    _permissions: str = Depends(require_write),
) -> dict:
    """Run AutoML optimization on multiple datasets."""
    try:
        if len(dataset_ids) == 0:
            raise HTTPException(
                status_code=400, detail="At least one dataset ID must be provided"
            )

        if len(dataset_ids) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 datasets can be processed in a batch",
            )

        logger.info(
            f"Starting batch AutoML optimization for {len(dataset_ids)} datasets"
        )

        # This would require a proper background task system
        # For now, return a placeholder response

        batch_id = f"batch_{int(time.time())}"

        return {
            "batch_id": batch_id,
            "status": "queued",
            "total_datasets": len(dataset_ids),
            "dataset_ids": dataset_ids,
            "estimated_completion_time_minutes": len(dataset_ids)
            * 15,  # Rough estimate
            "message": "Batch optimization queued (not yet implemented)",
        }

    except Exception as e:
        logger.error(f"Batch optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
