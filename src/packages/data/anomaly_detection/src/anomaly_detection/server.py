"""Anomaly Detection FastAPI server."""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from uvicorn import run as uvicorn_run
from typing import AsyncGenerator, List, Dict, Any
from pydantic import BaseModel

logger = structlog.get_logger()


class DetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    data: List[List[float]]
    algorithm: str = "isolation_forest"
    contamination: float = 0.1
    parameters: Dict[str, Any] = {}


class DetectionResponse(BaseModel):
    """Response model for anomaly detection."""
    anomalies: List[int]
    scores: List[float]
    algorithm: str
    total_samples: int
    anomalies_detected: int


class EnsembleRequest(BaseModel):
    """Request model for ensemble detection."""
    data: List[List[float]]
    algorithms: List[str] = ["isolation_forest", "one_class_svm", "lof"]
    method: str = "voting"
    parameters: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("Starting Anomaly Detection API server")
    # Initialize ML models, load configurations, etc.
    yield
    logger.info("Shutting down Anomaly Detection API server")
    # Cleanup resources


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(
        title="Anomaly Detection API",
        description="API for ML-based anomaly detection with ensemble methods and explainability",
        version="0.3.0",
        lifespan=lifespan,
    )
    
    return app


app = create_app()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "anomaly-detection"}


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_anomalies(request: DetectionRequest) -> DetectionResponse:
    """Detect anomalies in dataset."""
    logger.info("Processing detection request", 
                algorithm=request.algorithm, 
                samples=len(request.data))
    
    # Implementation would use DetectionService
    # Mock response for now
    anomalies = [1, 5, 10, 15]  # Mock anomaly indices
    scores = [0.8, 0.6, 0.9, 0.7]  # Mock anomaly scores
    
    return DetectionResponse(
        anomalies=anomalies,
        scores=scores,
        algorithm=request.algorithm,
        total_samples=len(request.data),
        anomalies_detected=len(anomalies)
    )


@app.post("/api/v1/ensemble", response_model=DetectionResponse)
async def ensemble_detect(request: EnsembleRequest) -> DetectionResponse:
    """Run ensemble anomaly detection."""
    logger.info("Processing ensemble detection request",
                algorithms=request.algorithms,
                method=request.method,
                samples=len(request.data))
    
    # Implementation would use EnsembleService
    anomalies = [2, 7, 12, 18, 25]
    scores = [0.85, 0.72, 0.91, 0.68, 0.79]
    
    return DetectionResponse(
        anomalies=anomalies,
        scores=scores,
        algorithm=f"ensemble_{request.method}",
        total_samples=len(request.data),
        anomalies_detected=len(anomalies)
    )


@app.get("/api/v1/algorithms")
async def list_algorithms() -> Dict[str, List[str]]:
    """List available detection algorithms."""
    return {
        "single": [
            "isolation_forest",
            "one_class_svm", 
            "local_outlier_factor",
            "autoencoder",
            "gaussian_mixture"
        ],
        "ensemble": [
            "voting",
            "averaging", 
            "stacking"
        ]
    }


@app.post("/api/v1/explain")
async def explain_detections(
    anomaly_indices: List[int],
    method: str = "shap"
) -> Dict[str, Any]:
    """Generate explanations for detected anomalies."""
    logger.info("Generating explanations", 
                anomalies=len(anomaly_indices),
                method=method)
    
    # Implementation would use ExplanationAnalyzers
    return {
        "method": method,
        "explanations": [
            {
                "index": idx,
                "features": [f"feature_{i}" for i in range(5)],
                "contributions": [0.3, -0.2, 0.5, -0.1, 0.4]
            }
            for idx in anomaly_indices[:5]  # Limit to first 5
        ]
    }


def main() -> None:
    """Run the server."""
    uvicorn_run(
        "anomaly_detection.server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()