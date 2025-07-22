"""Machine Learning FastAPI server."""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from uvicorn import run as uvicorn_run
from typing import AsyncGenerator, List, Dict, Any, Optional
from pydantic import BaseModel

logger = structlog.get_logger()


class AutoMLRequest(BaseModel):
    """Request model for AutoML training."""
    dataset: str
    target_column: str
    task_type: str = "classification"  # classification, regression
    time_limit: int = 300
    algorithms: Optional[List[str]] = None
    metrics: List[str] = ["accuracy"]


class AutoMLResponse(BaseModel):
    """Response model for AutoML training."""
    job_id: str
    best_model: str
    best_score: float
    models_evaluated: int
    time_taken: int
    leaderboard: List[Dict[str, Any]]


class EnsembleRequest(BaseModel):
    """Request model for ensemble creation."""
    models: List[str]
    method: str = "voting"  # voting, stacking, bagging
    weights: Optional[List[float]] = None


class EnsembleResponse(BaseModel):
    """Response model for ensemble creation."""
    ensemble_id: str
    method: str
    models: List[str]
    performance: Dict[str, float]


class ExplainabilityRequest(BaseModel):
    """Request model for model explanations."""
    model_id: str
    data: List[List[float]]
    method: str = "shap"  # shap, lime, permutation
    instance_level: bool = True


class ExplainabilityResponse(BaseModel):
    """Response model for explanations."""
    method: str
    global_importance: Optional[Dict[str, float]]
    local_explanations: Optional[List[Dict[str, Any]]]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("Starting Machine Learning API server")
    # Initialize model registry, load pre-trained models, etc.
    yield
    logger.info("Shutting down Machine Learning API server")
    # Cleanup resources, save models, etc.


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(
        title="Machine Learning API",
        description="API for AutoML, ensemble methods, explainable AI, and active learning",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    return app


app = create_app()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "machine-learning"}


@app.post("/api/v1/automl", response_model=AutoMLResponse)
async def train_automl(request: AutoMLRequest) -> AutoMLResponse:
    """Start AutoML training job."""
    logger.info("Starting AutoML job", 
                dataset=request.dataset,
                task=request.task_type,
                time_limit=request.time_limit)
    
    # Implementation would use AutoMLService
    # Mock response for now
    return AutoMLResponse(
        job_id="automl_job_123",
        best_model="RandomForestClassifier",
        best_score=0.92,
        models_evaluated=15,
        time_taken=280,
        leaderboard=[
            {"model": "RandomForestClassifier", "score": 0.92},
            {"model": "XGBoostClassifier", "score": 0.91},
            {"model": "LogisticRegression", "score": 0.87}
        ]
    )


@app.get("/api/v1/automl/{job_id}")
async def get_automl_status(job_id: str) -> Dict[str, Any]:
    """Get AutoML job status."""
    if job_id != "automl_job_123":
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "current_model": "RandomForestClassifier",
        "best_score": 0.92
    }


@app.post("/api/v1/ensemble", response_model=EnsembleResponse)
async def create_ensemble(request: EnsembleRequest) -> EnsembleResponse:
    """Create an ensemble of models."""
    logger.info("Creating ensemble", 
                models=request.models,
                method=request.method)
    
    # Implementation would use EnsembleAggregator
    return EnsembleResponse(
        ensemble_id="ensemble_456",
        method=request.method,
        models=request.models,
        performance={
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.94,
            "f1_score": 0.935
        }
    )


@app.post("/api/v1/explain", response_model=ExplainabilityResponse)
async def explain_model(request: ExplainabilityRequest) -> ExplainabilityResponse:
    """Generate model explanations."""
    logger.info("Generating explanations", 
                model=request.model_id,
                method=request.method,
                samples=len(request.data))
    
    # Implementation would use ExplainabilityService
    return ExplainabilityResponse(
        method=request.method,
        global_importance={
            "feature_0": 0.35,
            "feature_1": 0.28,
            "feature_2": 0.20,
            "feature_3": 0.17
        },
        local_explanations=[
            {
                "instance": i,
                "contributions": {
                    "feature_0": 0.3,
                    "feature_1": -0.2,
                    "feature_2": 0.5
                }
            }
            for i in range(min(5, len(request.data)))
        ] if request.instance_level else None
    )


@app.get("/api/v1/models")
async def list_models() -> Dict[str, List[str]]:
    """List available models and algorithms."""
    return {
        "classification": [
            "RandomForestClassifier",
            "XGBoostClassifier", 
            "LogisticRegression",
            "SVM",
            "NeuralNetwork"
        ],
        "regression": [
            "RandomForestRegressor",
            "XGBoostRegressor",
            "LinearRegression",
            "SVR",
            "NeuralNetwork"
        ],
        "ensemble_methods": [
            "voting",
            "stacking",
            "bagging",
            "boosting"
        ]
    }


@app.post("/api/v1/active_learning/sample")
async def active_learning_sample(
    dataset: str,
    model_id: str,
    budget: int = 100,
    strategy: str = "uncertainty"
) -> Dict[str, Any]:
    """Sample data points for active learning."""
    logger.info("Active learning sampling",
                dataset=dataset,
                model=model_id,
                budget=budget,
                strategy=strategy)
    
    # Implementation would use ActiveLearningService
    return {
        "dataset": dataset,
        "model_id": model_id,
        "strategy": strategy,
        "budget": budget,
        "selected_indices": list(range(0, budget, 10)),  # Mock indices
        "expected_improvement": 0.05,
        "confidence_scores": [0.3, 0.25, 0.4, 0.35, 0.28, 0.33, 0.29, 0.36, 0.31, 0.27]
    }


def main() -> None:
    """Run the server."""
    uvicorn_run(
        "machine_learning.server:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()