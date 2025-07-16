#!/usr/bin/env python3
"""
Model Optimization API Endpoints
REST API endpoints for advanced model optimization and AutoML capabilities
"""

import json
import logging
import tempfile
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from monorepo.application.services.advanced_model_optimization_service import (
    AdvancedModelOptimizationService,
    AdvancedOptimizationConfig,
    EnsembleStrategy,
    ObjectiveFunction,
    OptimizationObjective,
    OptimizationStrategy,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/optimization", tags=["model_optimization"])


# Dependency injection
def get_optimization_service() -> AdvancedModelOptimizationService:
    return AdvancedModelOptimizationService()


# Pydantic models for API
class OptimizationObjectiveRequest(BaseModel):
    """Request model for optimization objective"""

    function: str = Field(..., description="Objective function name")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Objective weight")
    direction: str = Field(
        "maximize", regex="^(maximize|minimize)$", description="Optimization direction"
    )


class OptimizationConfigRequest(BaseModel):
    """Request model for optimization configuration"""

    strategy: str = Field("bayesian_gp", description="Optimization strategy")
    objectives: list[OptimizationObjectiveRequest] = Field(
        default_factory=lambda: [OptimizationObjectiveRequest(function="f1_score")],
        description="List of optimization objectives",
    )
    n_trials: int = Field(
        100, ge=10, le=1000, description="Number of optimization trials"
    )
    timeout_seconds: int = Field(
        3600, ge=60, le=86400, description="Optimization timeout in seconds"
    )
    ensemble_strategy: str = Field("stacking", description="Ensemble strategy")
    ensemble_size: int = Field(
        5, ge=2, le=10, description="Number of models in ensemble"
    )
    cv_folds: int = Field(5, ge=3, le=10, description="Cross-validation folds")
    enable_meta_learning: bool = Field(True, description="Enable meta-learning")
    enable_automated_feature_engineering: bool = Field(
        True, description="Enable automated feature engineering"
    )
    random_state: int = Field(42, description="Random state for reproducibility")


class ModelTypeRequest(BaseModel):
    """Request model for model type selection"""

    model_types: list[str] | None = Field(
        None, description="List of model types to optimize (if None, uses defaults)"
    )
    include_ensemble: bool = Field(True, description="Include ensemble methods")
    exclude_slow_models: bool = Field(
        False, description="Exclude computationally expensive models"
    )


class OptimizationRequest(BaseModel):
    """Request model for model optimization"""

    dataset_name: str = Field(..., description="Name identifier for the dataset")
    config: OptimizationConfigRequest = Field(default_factory=OptimizationConfigRequest)
    model_selection: ModelTypeRequest = Field(default_factory=ModelTypeRequest)
    target_column: str | None = Field(
        None, description="Target column name for supervised learning"
    )


class OptimizationResultResponse(BaseModel):
    """Response model for optimization results"""

    optimization_id: str
    status: str
    best_model_type: str | None = None
    best_params: dict[str, Any] = Field(default_factory=dict)
    best_scores: dict[str, float] = Field(default_factory=dict)

    # Performance metrics
    optimization_time: float = 0.0
    total_trials: int = 0
    successful_trials: int = 0

    # Multi-objective results
    pareto_front_size: int = 0
    performance_trade_offs: dict[str, Any] = Field(default_factory=dict)

    # Ensemble information
    has_ensemble: bool = False
    ensemble_diversity_score: float = 0.0

    # Meta-learning
    recommendations: list[dict[str, Any]] = Field(default_factory=list)

    # Timestamps
    started_at: str
    completed_at: str | None = None


class OptimizationJobStatus(BaseModel):
    """Model for optimization job status"""

    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = 0.0  # 0.0 to 1.0
    current_trial: int = 0
    total_trials: int = 0
    best_score: float | None = None
    estimated_time_remaining: int | None = None  # seconds
    error_message: str | None = None


class MetaLearningRecommendation(BaseModel):
    """Model for meta-learning recommendations"""

    algorithm: str
    params: dict[str, Any]
    expected_performance: float
    similarity: float
    confidence: float
    reason: str


# Global storage for optimization jobs (in production, use Redis or database)
optimization_jobs: dict[str, dict[str, Any]] = {}


@router.post("/start", response_model=OptimizationResultResponse)
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    optimization_service: AdvancedModelOptimizationService = Depends(
        get_optimization_service
    ),
):
    """
    Start advanced model optimization process.

    This endpoint initiates a comprehensive model optimization process that includes:
    - Multi-objective optimization
    - Advanced hyperparameter tuning
    - Ensemble creation
    - Meta-learning recommendations
    """
    try:
        # Generate optimization ID
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.dataset_name) % 10000:04d}"

        # Initialize job status
        optimization_jobs[optimization_id] = {
            "status": "pending",
            "started_at": datetime.now().isoformat(),
            "request": request.dict(),
            "progress": 0.0,
            "current_trial": 0,
            "total_trials": request.config.n_trials,
        }

        # Start optimization in background
        background_tasks.add_task(
            run_optimization_background, optimization_id, request, optimization_service
        )

        return OptimizationResultResponse(
            optimization_id=optimization_id,
            status="pending",
            started_at=optimization_jobs[optimization_id]["started_at"],
        )

    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start optimization: {str(e)}"
        )


@router.post("/upload-and-optimize")
async def upload_and_optimize(
    file: UploadFile = File(..., description="CSV file containing the dataset"),
    config: str = Form(..., description="JSON string of optimization configuration"),
    target_column: str | None = Form(None, description="Target column name"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    optimization_service: AdvancedModelOptimizationService = Depends(
        get_optimization_service
    ),
):
    """
    Upload a dataset and start optimization process.

    Accepts a CSV file upload and immediately starts the optimization process.
    """
    try:
        # Parse configuration
        try:
            config_dict = json.loads(config)
            optimization_config = OptimizationConfigRequest(**config_dict)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail="Invalid JSON in config parameter"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid configuration: {str(e)}"
            )

        # Read uploaded file
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        content = await file.read()

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".csv"
        ) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Load data
            df = pd.read_csv(tmp_file_path)

            if len(df) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            # Validate target column
            X = df.copy()
            y = None

            if target_column:
                if target_column not in df.columns:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Target column '{target_column}' not found in dataset",
                    )
                y = df[target_column]
                X = df.drop(columns=[target_column])

            # Create optimization request
            request = OptimizationRequest(
                dataset_name=file.filename,
                config=optimization_config,
                target_column=target_column,
            )

            # Generate optimization ID
            optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(file.filename) % 10000:04d}"

            # Initialize job status
            optimization_jobs[optimization_id] = {
                "status": "pending",
                "started_at": datetime.now().isoformat(),
                "request": request.dict(),
                "progress": 0.0,
                "current_trial": 0,
                "total_trials": optimization_config.n_trials,
                "data": {"X": X, "y": y},  # Store data temporarily
            }

            # Start optimization in background
            background_tasks.add_task(
                run_optimization_with_data,
                optimization_id,
                X,
                y,
                optimization_config,
                optimization_service,
            )

            return {
                "optimization_id": optimization_id,
                "status": "pending",
                "message": f"Optimization started for dataset {file.filename}",
                "dataset_info": {
                    "n_samples": len(df),
                    "n_features": len(X.columns),
                    "target_column": target_column,
                    "has_missing_values": df.isnull().any().any(),
                },
            }

        finally:
            # Clean up temporary file
            import os

            os.unlink(tmp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload and optimization failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Upload and optimization failed: {str(e)}"
        )


@router.get("/status/{optimization_id}", response_model=OptimizationJobStatus)
async def get_optimization_status(optimization_id: str):
    """
    Get the status of an optimization job.

    Returns current progress, trial information, and any results if completed.
    """
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")

    job = optimization_jobs[optimization_id]

    return OptimizationJobStatus(
        job_id=optimization_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        current_trial=job.get("current_trial", 0),
        total_trials=job.get("total_trials", 0),
        best_score=job.get("best_score"),
        estimated_time_remaining=job.get("estimated_time_remaining"),
        error_message=job.get("error_message"),
    )


@router.get("/results/{optimization_id}", response_model=OptimizationResultResponse)
async def get_optimization_results(optimization_id: str):
    """
    Get the results of a completed optimization job.
    """
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")

    job = optimization_jobs[optimization_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Optimization not completed. Current status: {job['status']}",
        )

    result = job.get("result")
    if not result:
        raise HTTPException(
            status_code=500, detail="Optimization results not available"
        )

    return OptimizationResultResponse(
        optimization_id=optimization_id,
        status="completed",
        best_model_type=result.get("best_params", {}).get("model_type"),
        best_params=result.get("best_params", {}),
        best_scores=result.get("best_scores", {}),
        optimization_time=result.get("optimization_time", 0.0),
        total_trials=result.get("total_trials", 0),
        successful_trials=result.get("successful_trials", 0),
        pareto_front_size=len(result.get("pareto_front", [])),
        performance_trade_offs=result.get("performance_trade_offs", {}),
        has_ensemble=result.get("ensemble_model") is not None,
        ensemble_diversity_score=result.get("ensemble_diversity_score", 0.0),
        recommendations=job.get("recommendations", []),
        started_at=job["started_at"],
        completed_at=job.get("completed_at"),
    )


@router.get("/recommendations/{optimization_id}")
async def get_meta_learning_recommendations(
    optimization_id: str,
    optimization_service: AdvancedModelOptimizationService = Depends(
        get_optimization_service
    ),
):
    """
    Get meta-learning recommendations for a dataset.
    """
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")

    job = optimization_jobs[optimization_id]

    # Get dataset from job
    data = job.get("data")
    if not data:
        raise HTTPException(
            status_code=400, detail="Dataset not available for recommendations"
        )

    try:
        recommendations = await optimization_service.get_meta_learning_recommendations(
            data["X"]
        )

        return {
            "optimization_id": optimization_id,
            "recommendations": [
                MetaLearningRecommendation(
                    algorithm=rec["algorithm"],
                    params=rec["params"],
                    expected_performance=rec["expected_performance"],
                    similarity=rec["similarity"],
                    confidence=rec["confidence"],
                    reason=f"Based on {rec['similarity']:.2f} similarity to previous datasets",
                )
                for rec in recommendations
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get recommendations: {str(e)}"
        )


@router.post("/export/{optimization_id}")
async def export_optimization_results(
    optimization_id: str,
    include_models: bool = Query(True, description="Include trained models in export"),
    optimization_service: AdvancedModelOptimizationService = Depends(
        get_optimization_service
    ),
):
    """
    Export optimization results to downloadable files.

    Creates a package containing optimization results, models, and analysis.
    """
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")

    job = optimization_jobs[optimization_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Optimization not completed. Current status: {job['status']}",
        )

    try:
        # Create temporary export file
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as tmp_file:
            export_path = tmp_file.name

        # Export results
        result = job.get("result_object")  # Full result object
        if result:
            success = await optimization_service.export_optimization_results(
                result, export_path
            )

            if success:
                return FileResponse(
                    export_path,
                    media_type="application/json",
                    filename=f"optimization_results_{optimization_id}.json",
                )

        raise HTTPException(status_code=500, detail="Failed to export results")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/compare")
async def compare_optimizations(
    optimization_ids: list[str] = Query(
        ..., description="List of optimization IDs to compare"
    ),
):
    """
    Compare results from multiple optimization jobs.
    """
    if len(optimization_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 optimization IDs required for comparison",
        )

    comparisons = []

    for opt_id in optimization_ids:
        if opt_id not in optimization_jobs:
            raise HTTPException(
                status_code=404, detail=f"Optimization job {opt_id} not found"
            )

        job = optimization_jobs[opt_id]
        if job["status"] != "completed":
            raise HTTPException(
                status_code=400, detail=f"Optimization {opt_id} not completed"
            )

        result = job.get("result", {})
        comparisons.append(
            {
                "optimization_id": opt_id,
                "best_scores": result.get("best_scores", {}),
                "optimization_time": result.get("optimization_time", 0.0),
                "total_trials": result.get("total_trials", 0),
                "best_model_type": result.get("best_params", {}).get("model_type"),
                "pareto_front_size": len(result.get("pareto_front", [])),
            }
        )

    # Find best performers
    best_by_score = max(
        comparisons,
        key=lambda x: max(x["best_scores"].values()) if x["best_scores"] else 0,
    )
    fastest = min(comparisons, key=lambda x: x["optimization_time"])
    most_efficient = min(
        comparisons,
        key=lambda x: x["optimization_time"] / max(x["best_scores"].values())
        if x["best_scores"]
        else float("inf"),
    )

    return {
        "comparison": comparisons,
        "summary": {
            "best_performance": best_by_score["optimization_id"],
            "fastest_optimization": fastest["optimization_id"],
            "most_efficient": most_efficient["optimization_id"],
        },
        "recommendations": [
            f"Best overall performance: {best_by_score['optimization_id']}",
            f"Fastest optimization: {fastest['optimization_id']} ({fastest['optimization_time']:.2f}s)",
            f"Best time/performance ratio: {most_efficient['optimization_id']}",
        ],
    }


@router.delete("/cleanup/{optimization_id}")
async def cleanup_optimization_job(optimization_id: str):
    """
    Clean up optimization job data and temporary files.
    """
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")

    try:
        # Remove from jobs
        del optimization_jobs[optimization_id]

        return {
            "message": f"Optimization job {optimization_id} cleaned up successfully"
        }

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/strategies")
async def get_available_strategies():
    """
    Get list of available optimization strategies and their descriptions.
    """
    return {
        "optimization_strategies": [
            {
                "name": "bayesian_gp",
                "display_name": "Bayesian Optimization (Gaussian Process)",
                "description": "Uses Gaussian Process for efficient hyperparameter optimization",
                "complexity": "high",
                "recommended_for": "Small to medium parameter spaces",
            },
            {
                "name": "bayesian_rf",
                "display_name": "Bayesian Optimization (Random Forest)",
                "description": "Uses Random Forest surrogate model for optimization",
                "complexity": "medium",
                "recommended_for": "Medium to large parameter spaces",
            },
            {
                "name": "tpe",
                "display_name": "Tree-structured Parzen Estimator",
                "description": "Probabilistic model for hyperparameter optimization",
                "complexity": "medium",
                "recommended_for": "General purpose optimization",
            },
            {
                "name": "cma_es",
                "display_name": "CMA-ES",
                "description": "Covariance Matrix Adaptation Evolution Strategy",
                "complexity": "high",
                "recommended_for": "Continuous parameter spaces",
            },
            {
                "name": "nsga_ii",
                "display_name": "NSGA-II",
                "description": "Multi-objective optimization using genetic algorithm",
                "complexity": "high",
                "recommended_for": "Multi-objective optimization",
            },
        ],
        "ensemble_strategies": [
            {
                "name": "voting_soft",
                "display_name": "Soft Voting",
                "description": "Combines probability predictions from multiple models",
            },
            {
                "name": "voting_hard",
                "display_name": "Hard Voting",
                "description": "Uses majority vote from multiple models",
            },
            {
                "name": "stacking",
                "display_name": "Stacking",
                "description": "Uses meta-model to combine base model predictions",
            },
            {
                "name": "blending",
                "display_name": "Blending",
                "description": "Weighted combination of model predictions",
            },
        ],
        "objective_functions": [
            {"name": "accuracy", "description": "Classification accuracy"},
            {"name": "precision", "description": "Precision score"},
            {"name": "recall", "description": "Recall score"},
            {
                "name": "f1_score",
                "description": "F1 score (harmonic mean of precision and recall)",
            },
            {"name": "roc_auc", "description": "Area under ROC curve"},
            {"name": "training_time", "description": "Model training time (minimize)"},
            {
                "name": "inference_time",
                "description": "Model inference time (minimize)",
            },
            {"name": "model_size", "description": "Model size in memory (minimize)"},
        ],
    }


# Background task functions
async def run_optimization_background(
    optimization_id: str,
    request: OptimizationRequest,
    optimization_service: AdvancedModelOptimizationService,
):
    """Run optimization in background task"""

    try:
        # Update status
        optimization_jobs[optimization_id]["status"] = "running"

        # Create configuration
        config = _create_optimization_config(request.config)

        # For demo purposes, create synthetic data
        # In real implementation, load data from storage
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(1000, 10), columns=[f"feature_{i}" for i in range(10)]
        )
        y = (
            pd.Series(np.random.choice([0, 1], size=1000, p=[0.8, 0.2]))
            if request.target_column
            else None
        )

        # Store data for later use
        optimization_jobs[optimization_id]["data"] = {"X": X, "y": y}

        # Run optimization
        result = await optimization_service.optimize_model_advanced(
            X, y, request.model_selection.model_types
        )

        # Store results
        optimization_jobs[optimization_id].update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "result": {
                    "best_params": result.best_params,
                    "best_scores": result.best_scores,
                    "optimization_time": result.optimization_time,
                    "total_trials": result.total_trials,
                    "successful_trials": result.successful_trials,
                    "pareto_front": result.pareto_front,
                    "performance_trade_offs": result.performance_trade_offs,
                },
                "result_object": result,  # Store full object for export
            }
        )

        # Get meta-learning recommendations
        recommendations = await optimization_service.get_meta_learning_recommendations(
            X
        )
        optimization_jobs[optimization_id]["recommendations"] = recommendations

    except Exception as e:
        logger.error(f"Optimization {optimization_id} failed: {e}")
        optimization_jobs[optimization_id].update(
            {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now().isoformat(),
            }
        )


async def run_optimization_with_data(
    optimization_id: str,
    X: pd.DataFrame,
    y: pd.Series | None,
    config: OptimizationConfigRequest,
    optimization_service: AdvancedModelOptimizationService,
):
    """Run optimization with provided data"""

    try:
        # Update status
        optimization_jobs[optimization_id]["status"] = "running"

        # Create configuration
        opt_config = _create_optimization_config(config)
        service = AdvancedModelOptimizationService(opt_config)

        # Run optimization
        result = await service.optimize_model_advanced(X, y)

        # Store results
        optimization_jobs[optimization_id].update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "result": {
                    "best_params": result.best_params,
                    "best_scores": result.best_scores,
                    "optimization_time": result.optimization_time,
                    "total_trials": result.total_trials,
                    "successful_trials": result.successful_trials,
                    "pareto_front": result.pareto_front,
                    "performance_trade_offs": result.performance_trade_offs,
                },
                "result_object": result,
            }
        )

        # Get meta-learning recommendations
        recommendations = await service.get_meta_learning_recommendations(X)
        optimization_jobs[optimization_id]["recommendations"] = recommendations

    except Exception as e:
        logger.error(f"Optimization {optimization_id} failed: {e}")
        optimization_jobs[optimization_id].update(
            {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now().isoformat(),
            }
        )


def _create_optimization_config(
    config_request: OptimizationConfigRequest,
) -> AdvancedOptimizationConfig:
    """Convert API request to service configuration"""

    objectives = []
    for obj_req in config_request.objectives:
        objectives.append(
            OptimizationObjective(
                function=ObjectiveFunction(obj_req.function),
                weight=obj_req.weight,
                direction=obj_req.direction,
            )
        )

    return AdvancedOptimizationConfig(
        strategy=OptimizationStrategy(config_request.strategy),
        objectives=objectives,
        n_trials=config_request.n_trials,
        timeout_seconds=config_request.timeout_seconds,
        ensemble_strategy=EnsembleStrategy(config_request.ensemble_strategy),
        ensemble_size=config_request.ensemble_size,
        cv_folds=config_request.cv_folds,
        enable_meta_learning=config_request.enable_meta_learning,
        enable_automated_feature_engineering=config_request.enable_automated_feature_engineering,
        random_state=config_request.random_state,
    )


# Health check endpoint
@router.get("/health")
async def optimization_health():
    """Health check for optimization service"""
    return {
        "status": "healthy",
        "service": "model_optimization",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(
            [job for job in optimization_jobs.values() if job["status"] == "running"]
        ),
        "total_jobs": len(optimization_jobs),
    }
