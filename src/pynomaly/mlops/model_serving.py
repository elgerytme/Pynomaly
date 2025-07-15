#!/usr/bin/env python3
"""
Model Serving Infrastructure for Pynomaly.
Provides REST API endpoints for model inference, health checks, and performance monitoring.
"""

import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator

from pynomaly.mlops.automated_retraining import retraining_pipeline
from pynomaly.mlops.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ServingStatus(Enum):
    """Model serving status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    LOADING = "loading"
    ERROR = "error"


class InferenceMode(Enum):
    """Inference mode for batch vs real-time processing."""

    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class ModelEndpoint:
    """Model serving endpoint configuration."""

    model_id: str
    model_version: str
    endpoint_name: str
    endpoint_url: str
    status: ServingStatus
    created_at: datetime
    last_health_check: datetime
    request_count: int
    error_count: int
    avg_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    inference_mode: InferenceMode
    auto_scaling_enabled: bool
    max_batch_size: int
    timeout_seconds: int


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    data: list[dict[str, Any]] | dict[str, Any] = Field(
        ..., description="Input data for prediction"
    )
    model_id: str | None = Field(None, description="Specific model ID to use")
    model_version: str | None = Field("latest", description="Model version to use")
    return_probabilities: bool = Field(
        False, description="Return prediction probabilities"
    )
    return_explanations: bool = Field(
        False, description="Return prediction explanations"
    )
    inference_mode: InferenceMode = Field(
        InferenceMode.REAL_TIME, description="Inference processing mode"
    )
    timeout_seconds: int | None = Field(30, description="Request timeout in seconds")


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    request_id: str = Field(..., description="Unique request identifier")
    model_id: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version used")
    predictions: list[dict[str, Any]] = Field(..., description="Prediction results")
    probabilities: list[list[float]] | None = Field(
        None, description="Prediction probabilities"
    )
    explanations: list[dict[str, Any]] | None = Field(
        None, description="Prediction explanations"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(..., description="Prediction timestamp")
    status: str = Field("success", description="Request status")


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: ServingStatus
    model_id: str
    model_version: str
    uptime_seconds: float
    request_count: int
    error_rate: float
    avg_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    last_prediction: datetime | None
    checks: dict[str, bool]
    timestamp: datetime


class ModelServingEngine:
    """Core model serving engine with caching and monitoring."""

    def __init__(self):
        self.model_registry = ModelRegistry()
        self.loaded_models: dict[str, tuple[BaseEstimator, dict[str, Any]]] = {}
        self.endpoints: dict[str, ModelEndpoint] = {}
        self.request_metrics: dict[str, list[float]] = {}
        self.start_time = time.time()

        # Performance monitoring
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0

    async def load_model(self, model_id: str, version: str = "latest") -> bool:
        """Load model into serving cache."""
        try:
            cache_key = f"{model_id}:{version}"

            if cache_key in self.loaded_models:
                logger.info(f"Model already loaded: {cache_key}")
                return True

            # Load from registry
            model, metadata = await self.model_registry.get_model(model_id, version)

            if model is None:
                raise ValueError(f"Model not found: {model_id}:{version}")

            # Cache model and metadata
            self.loaded_models[cache_key] = (model, metadata)

            # Create endpoint if not exists
            if model_id not in self.endpoints:
                self.endpoints[model_id] = ModelEndpoint(
                    model_id=model_id,
                    model_version=version,
                    endpoint_name=f"anomaly-detection-{model_id}",
                    endpoint_url=f"/predict/{model_id}",
                    status=ServingStatus.HEALTHY,
                    created_at=datetime.now(),
                    last_health_check=datetime.now(),
                    request_count=0,
                    error_count=0,
                    avg_response_time_ms=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    inference_mode=InferenceMode.REAL_TIME,
                    auto_scaling_enabled=True,
                    max_batch_size=1000,
                    timeout_seconds=30,
                )

            logger.info(f"Model loaded successfully: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_id}:{version} - {e}")
            return False

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate predictions for input data."""
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            # Validate and prepare data
            model_id = request.model_id or "default"
            version = request.model_version or "latest"
            cache_key = f"{model_id}:{version}"

            # Load model if not cached
            if cache_key not in self.loaded_models:
                loaded = await self.load_model(model_id, version)
                if not loaded:
                    raise HTTPException(
                        status_code=404, detail=f"Model not found: {model_id}"
                    )

            model, metadata = self.loaded_models[cache_key]

            # Prepare input data
            if isinstance(request.data, dict):
                input_data = [request.data]
            else:
                input_data = request.data

            # Convert to DataFrame
            df = pd.DataFrame(input_data)

            # Validate input schema
            await self._validate_input_schema(df, metadata)

            # Generate predictions
            predictions = []
            probabilities = []
            explanations = []

            for idx, row in df.iterrows():
                # Prepare single sample
                sample_data = row.to_dict()
                X = np.array(list(sample_data.values())).reshape(1, -1)

                # Predict
                pred = model.predict(X)[0]

                # Convert to anomaly detection format
                is_anomaly = pred == -1  # Isolation Forest convention
                anomaly_score = (
                    model.score_samples(X)[0]
                    if hasattr(model, "score_samples")
                    else float(pred)
                )

                # Create detection result
                detection_result = {
                    "sample_id": f"sample_{idx}",
                    "is_anomaly": bool(is_anomaly),
                    "anomaly_score": float(anomaly_score),
                    "confidence": abs(anomaly_score),
                    "prediction": int(pred),
                    "input_data": sample_data,
                }
                predictions.append(detection_result)

                # Add probabilities if requested
                if request.return_probabilities:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X)[0].tolist()
                    else:
                        # Convert score to probability-like value
                        score_norm = (anomaly_score + 1) / 2  # Normalize to [0, 1]
                        proba = [1 - score_norm, score_norm]
                    probabilities.append(proba)

                # Add explanations if requested
                if request.return_explanations:
                    explanation = await self._generate_explanation(
                        X, model, sample_data
                    )
                    explanations.append(explanation)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Update metrics
            await self._update_metrics(model_id, processing_time, success=True)

            # Create response
            response = PredictionResponse(
                request_id=request_id,
                model_id=model_id,
                model_version=version,
                predictions=predictions,
                probabilities=probabilities if request.return_probabilities else None,
                explanations=explanations if request.return_explanations else None,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                status="success",
            )

            return response

        except Exception as e:
            # Update error metrics
            await self._update_metrics(request.model_id or "unknown", 0, success=False)

            logger.error(f"Prediction failed for request {request_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    async def health_check(self, model_id: str) -> HealthCheckResponse:
        """Perform health check for model endpoint."""
        try:
            endpoint = self.endpoints.get(model_id)
            if not endpoint:
                raise HTTPException(
                    status_code=404, detail=f"Endpoint not found: {model_id}"
                )

            # Perform health checks
            checks = await self._perform_health_checks(model_id)

            # Determine overall status
            if all(checks.values()):
                status = ServingStatus.HEALTHY
            elif any(checks.values()):
                status = ServingStatus.DEGRADED
            else:
                status = ServingStatus.UNHEALTHY

            # Calculate metrics
            uptime = time.time() - self.start_time
            error_rate = self.error_count / max(1, self.request_count) * 100
            avg_response_time = self.total_processing_time / max(1, self.request_count)

            # Update endpoint status
            endpoint.status = status
            endpoint.last_health_check = datetime.now()

            response = HealthCheckResponse(
                status=status,
                model_id=model_id,
                model_version=endpoint.model_version,
                uptime_seconds=uptime,
                request_count=self.request_count,
                error_rate=error_rate,
                avg_response_time_ms=avg_response_time,
                memory_usage_mb=endpoint.memory_usage_mb,
                cpu_usage_percent=endpoint.cpu_usage_percent,
                last_prediction=datetime.now() if self.request_count > 0 else None,
                checks=checks,
                timestamp=datetime.now(),
            )

            return response

        except Exception as e:
            logger.error(f"Health check failed for {model_id}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Health check failed: {str(e)}"
            )

    async def _validate_input_schema(self, df: pd.DataFrame, metadata: dict[str, Any]):
        """Validate input data schema against model requirements."""
        try:
            expected_features = metadata.get("feature_names", [])
            if expected_features:
                missing_features = set(expected_features) - set(df.columns)
                if missing_features:
                    raise ValueError(
                        f"Missing required features: {list(missing_features)}"
                    )

            # Check data types and ranges
            for column in df.columns:
                if column in expected_features:
                    # Validate numeric ranges if available
                    if column in metadata.get("feature_ranges", {}):
                        col_range = metadata["feature_ranges"][column]
                        if (
                            df[column].min() < col_range["min"]
                            or df[column].max() > col_range["max"]
                        ):
                            logger.warning(
                                f"Feature {column} outside expected range {col_range}"
                            )

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Invalid input data: {e}")

    async def _generate_explanation(
        self, X: np.ndarray, model: BaseEstimator, sample_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate explanation for prediction."""
        try:
            explanation = {
                "method": "feature_importance",
                "features": [],
                "summary": "Anomaly detection based on isolation scoring",
            }

            # Simple feature importance (placeholder)
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                feature_names = list(sample_data.keys())

                for i, (name, value, imp) in enumerate(
                    zip(feature_names, X[0], importance, strict=False)
                ):
                    explanation["features"].append(
                        {
                            "name": name,
                            "value": float(value),
                            "importance": float(imp),
                            "contribution": float(imp * value),
                        }
                    )

            return explanation

        except Exception as e:
            logger.warning(f"Could not generate explanation: {e}")
            return {"method": "unavailable", "error": str(e)}

    async def _perform_health_checks(self, model_id: str) -> dict[str, bool]:
        """Perform comprehensive health checks."""
        checks = {}

        try:
            # Model loaded check
            cache_key = f"{model_id}:latest"
            checks["model_loaded"] = cache_key in self.loaded_models

            # Memory check
            import psutil

            memory_percent = psutil.virtual_memory().percent
            checks["memory_ok"] = memory_percent < 90

            # Model prediction check
            if checks["model_loaded"]:
                try:
                    model, metadata = self.loaded_models[cache_key]
                    # Test prediction with dummy data
                    test_data = np.random.random((1, 5))  # Simple test
                    _ = model.predict(test_data)
                    checks["prediction_ok"] = True
                except Exception:
                    checks["prediction_ok"] = False
            else:
                checks["prediction_ok"] = False

            # Response time check
            model_metrics = self.request_metrics.get(model_id, [])
            if model_metrics:
                avg_time = sum(model_metrics[-10:]) / len(
                    model_metrics[-10:]
                )  # Last 10 requests
                checks["response_time_ok"] = avg_time < 5000  # 5 seconds threshold
            else:
                checks["response_time_ok"] = True

            # Error rate check
            endpoint = self.endpoints.get(model_id)
            if endpoint and endpoint.request_count > 0:
                error_rate = endpoint.error_count / endpoint.request_count
                checks["error_rate_ok"] = error_rate < 0.1  # 10% threshold
            else:
                checks["error_rate_ok"] = True

        except Exception as e:
            logger.error(f"Health check error: {e}")
            checks["health_check_error"] = False

        return checks

    async def _update_metrics(
        self, model_id: str, processing_time: float, success: bool
    ):
        """Update performance metrics."""
        try:
            self.request_count += 1

            if success:
                self.total_processing_time += processing_time

                # Update model-specific metrics
                if model_id not in self.request_metrics:
                    self.request_metrics[model_id] = []
                self.request_metrics[model_id].append(processing_time)

                # Keep only last 1000 measurements
                if len(self.request_metrics[model_id]) > 1000:
                    self.request_metrics[model_id] = self.request_metrics[model_id][
                        -1000:
                    ]
            else:
                self.error_count += 1

            # Update endpoint metrics
            if model_id in self.endpoints:
                endpoint = self.endpoints[model_id]
                endpoint.request_count += 1
                if not success:
                    endpoint.error_count += 1

                # Update average response time
                if success and endpoint.request_count > 0:
                    endpoint.avg_response_time_ms = (
                        endpoint.avg_response_time_ms * (endpoint.request_count - 1)
                        + processing_time
                    ) / endpoint.request_count

        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

    def get_serving_stats(self) -> dict[str, Any]:
        """Get comprehensive serving statistics."""
        stats = {
            "uptime_seconds": time.time() - self.start_time,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count) * 100,
            "avg_response_time_ms": self.total_processing_time
            / max(1, self.request_count),
            "loaded_models": len(self.loaded_models),
            "active_endpoints": len(self.endpoints),
            "endpoints": {
                model_id: asdict(endpoint)
                for model_id, endpoint in self.endpoints.items()
            },
            "timestamp": datetime.now().isoformat(),
        }
        return stats

    async def unload_model(self, model_id: str, version: str = "latest"):
        """Unload model from serving cache."""
        try:
            cache_key = f"{model_id}:{version}"
            if cache_key in self.loaded_models:
                del self.loaded_models[cache_key]
                logger.info(f"Model unloaded: {cache_key}")

            # Update endpoint status
            if model_id in self.endpoints:
                self.endpoints[model_id].status = ServingStatus.LOADING

        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")


# Global serving engine instance
serving_engine = ModelServingEngine()

# FastAPI application
app = FastAPI(
    title="Pynomaly Model Serving API",
    description="Production-ready model serving infrastructure for anomaly detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Basic authentication dependency."""
    # Implement proper authentication logic here
    return {"user_id": "system", "permissions": ["predict", "health"]}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """Generate predictions for input data."""
    response = await serving_engine.predict(request)

    # Background task for performance monitoring
    background_tasks.add_task(
        monitor_performance, request.model_id or "default", response.processing_time_ms
    )

    return response


@app.post("/predict/{model_id}", response_model=PredictionResponse)
async def predict_model(
    model_id: str,
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """Generate predictions using specific model."""
    request.model_id = model_id
    return await predict(request, background_tasks, user)


@app.get("/health", response_model=dict[str, str])
async def health():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/health/{model_id}", response_model=HealthCheckResponse)
async def health_check_model(model_id: str, user: dict = Depends(get_current_user)):
    """Detailed health check for specific model."""
    return await serving_engine.health_check(model_id)


@app.post("/models/{model_id}/load")
async def load_model(
    model_id: str, version: str = "latest", user: dict = Depends(get_current_user)
):
    """Load model into serving cache."""
    success = await serving_engine.load_model(model_id, version)
    if success:
        return {"status": "loaded", "model_id": model_id, "version": version}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")


@app.delete("/models/{model_id}/unload")
async def unload_model(
    model_id: str, version: str = "latest", user: dict = Depends(get_current_user)
):
    """Unload model from serving cache."""
    await serving_engine.unload_model(model_id, version)
    return {"status": "unloaded", "model_id": model_id, "version": version}


@app.get("/stats")
async def get_stats(user: dict = Depends(get_current_user)):
    """Get serving statistics."""
    return serving_engine.get_serving_stats()


@app.get("/models")
async def list_loaded_models(user: dict = Depends(get_current_user)):
    """List currently loaded models."""
    return {
        "loaded_models": list(serving_engine.loaded_models.keys()),
        "endpoints": list(serving_engine.endpoints.keys()),
        "timestamp": datetime.now().isoformat(),
    }


async def monitor_performance(model_id: str, processing_time: float):
    """Background task for performance monitoring."""
    try:
        # Check if retraining should be triggered based on performance
        await retraining_pipeline.check_performance_trigger(
            model_id, f"endpoint_{model_id}"
        )

        # Log metrics for external monitoring systems
        logger.info(
            f"Performance metric - Model: {model_id}, Time: {processing_time}ms"
        )

    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
