"""Statistics FastAPI server."""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from uvicorn import run as uvicorn_run
from typing import AsyncGenerator, List, Dict, Any, Optional
from pydantic import BaseModel

logger = structlog.get_logger()


class StatisticalSummaryRequest(BaseModel):
    """Request model for statistical summary."""
    data_source: str
    columns: List[str]
    statistics: List[str] = ["mean", "std", "min", "max", "count"]


class StatisticalSummaryResponse(BaseModel):
    """Response model for statistical summary."""
    analysis_id: str
    summary: Dict[str, Dict[str, float]]
    status: str


class HypothesisTestRequest(BaseModel):
    """Request model for hypothesis testing."""
    data_source: str
    test_type: str = "ttest"
    columns: List[str]
    alpha: float = 0.05
    parameters: Dict[str, Any] = {}


class HypothesisTestResponse(BaseModel):
    """Response model for hypothesis test."""
    test_id: str
    test_type: str
    results: Dict[str, Any]
    significant: bool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("Starting Statistics API server")
    yield
    logger.info("Shutting down Statistics API server")


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(
        title="Statistics API",
        description="API for statistical analysis, hypothesis testing, and modeling",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    return app


app = create_app()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "statistics"}


@app.post("/api/v1/statistics/summary", response_model=StatisticalSummaryResponse)
async def generate_summary(request: StatisticalSummaryRequest) -> StatisticalSummaryResponse:
    """Generate descriptive statistics summary."""
    logger.info("Generating summary", source=request.data_source, columns=request.columns)
    
    analysis_id = f"summary_{hash(request.data_source) % 10000}"
    
    return StatisticalSummaryResponse(
        analysis_id=analysis_id,
        summary={
            col: {
                "mean": 45.2,
                "std": 12.5,
                "min": 18.0,
                "max": 85.0,
                "count": 1000
            } for col in request.columns
        },
        status="completed"
    )


@app.post("/api/v1/statistics/test", response_model=HypothesisTestResponse)
async def run_hypothesis_test(request: HypothesisTestRequest) -> HypothesisTestResponse:
    """Run hypothesis test."""
    logger.info("Running hypothesis test", test=request.test_type, columns=request.columns)
    
    test_id = f"test_{hash(str(request.columns)) % 10000}"
    
    return HypothesisTestResponse(
        test_id=test_id,
        test_type=request.test_type,
        results={
            "statistic": 2.45,
            "p_value": 0.032,
            "degrees_of_freedom": 98,
            "effect_size": 0.23,
            "confidence_interval": [0.15, 0.31]
        },
        significant=True
    )


@app.post("/api/v1/statistics/model")
async def fit_statistical_model(
    data_source: str,
    target: str,
    features: List[str],
    model_type: str = "linear"
) -> Dict[str, Any]:
    """Fit statistical model."""
    model_id = f"model_{hash(data_source + target) % 10000}"
    
    return {
        "model_id": model_id,
        "data_source": data_source,
        "target": target,
        "features": features,
        "model_type": model_type,
        "performance": {
            "r_squared": 0.85,
            "rmse": 2.34,
            "mae": 1.89,
            "aic": 245.6,
            "bic": 267.8
        },
        "coefficients": {
            f"coef_{i}": round(0.5 + i * 0.1, 2) for i in range(len(features))
        },
        "status": "fitted"
    }


def main() -> None:
    """Run the server."""
    uvicorn_run(
        "statistics.server:app",
        host="0.0.0.0",
        port=8013,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()