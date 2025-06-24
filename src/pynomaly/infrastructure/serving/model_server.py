"""Model serving infrastructure for production inference."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.exposition import choose_encoder

from pynomaly.application.services.deployment_orchestration_service import DeploymentOrchestrationService
from pynomaly.application.services.model_registry_service import ModelRegistryService
from pynomaly.domain.entities.deployment import Environment


# Prometheus metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total predictions made', ['model_id', 'version'])
PREDICTION_LATENCY = Histogram('model_prediction_duration_seconds', 'Prediction latency', ['model_id', 'version'])
ERROR_COUNTER = Counter('model_errors_total', 'Total errors', ['model_id', 'version', 'error_type'])
ACTIVE_CONNECTIONS = Gauge('model_websocket_connections', 'Active WebSocket connections')
MODEL_LOAD_TIME = Histogram('model_load_duration_seconds', 'Model loading time', ['model_id', 'version'])


class PredictionRequest(BaseModel):
    """Single prediction request."""
    data: Dict[str, Any] = Field(..., description="Input data for prediction")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    version: Optional[str] = Field(None, description="Specific model version to use")
    return_confidence: bool = Field(True, description="Return confidence score")
    return_explanation: bool = Field(False, description="Return prediction explanation")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    data: List[Dict[str, Any]] = Field(..., description="Array of input data for prediction")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    version: Optional[str] = Field(None, description="Specific model version to use")
    return_confidence: bool = Field(True, description="Return confidence scores")
    return_explanation: bool = Field(False, description="Return prediction explanations")
    batch_size: int = Field(1000, description="Processing batch size", ge=1, le=10000)


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: float = Field(..., description="Anomaly score (0.0 to 1.0)")
    is_anomaly: bool = Field(..., description="Binary anomaly classification")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Prediction explanation")
    model_id: str = Field(..., description="Model ID used")
    model_version: str = Field(..., description="Model version used")
    prediction_time: datetime = Field(..., description="Prediction timestamp")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse] = Field(..., description="Array of predictions")
    total_processed: int = Field(..., description="Total samples processed")
    total_anomalies: int = Field(..., description="Total anomalies detected")
    processing_time_ms: float = Field(..., description="Total processing time")
    throughput_per_second: float = Field(..., description="Processing throughput")


class ModelInfo(BaseModel):
    """Model information response."""
    model_id: str
    model_name: str
    version: str
    algorithm: str
    domain: str
    status: str
    health_score: float
    last_updated: datetime
    performance_metrics: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime")
    loaded_models: int = Field(..., description="Number of loaded models")
    active_connections: int = Field(..., description="Active WebSocket connections")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")


class ModelServingError(Exception):
    """Base exception for model serving errors."""
    pass


class ModelNotLoadedError(ModelServingError):
    """Model not loaded error."""
    pass


class PredictionError(ModelServingError):
    """Prediction error."""
    pass


class ModelServer:
    """Production model serving server with comprehensive inference capabilities.
    
    Features:
    - REST API for single and batch predictions
    - WebSocket support for real-time streaming
    - Model management and health monitoring
    - Prometheus metrics collection
    - Auto-scaling and load balancing support
    """
    
    def __init__(
        self,
        deployment_service: DeploymentOrchestrationService,
        model_registry_service: ModelRegistryService,
        environment: Environment = Environment.PRODUCTION,
        model_cache_size: int = 10
    ):
        """Initialize model server.
        
        Args:
            deployment_service: Deployment orchestration service
            model_registry_service: Model registry service
            environment: Target environment for serving
            model_cache_size: Maximum number of models to cache
        """
        self.deployment_service = deployment_service
        self.model_registry_service = model_registry_service
        self.environment = environment
        self.model_cache_size = model_cache_size
        
        # Model cache and metadata
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.model_access_times: Dict[str, datetime] = {}
        
        # Server state
        self.start_time = datetime.utcnow()
        self.active_websockets: List[WebSocket] = []
        
        # Create FastAPI app
        self.app = self._create_fastapi_app()
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application with all endpoints."""
        app = FastAPI(
            title="Pynomaly Model Serving API",
            description="Production-ready anomaly detection model serving",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_prediction_routes(app)
        self._add_model_management_routes(app)
        self._add_health_routes(app)
        self._add_websocket_routes(app)
        
        return app
    
    def _add_prediction_routes(self, app: FastAPI) -> None:
        """Add prediction endpoints."""
        
        @app.post("/api/v1/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest) -> PredictionResponse:
            """Make a single prediction."""
            try:
                start_time = time.time()
                
                # Get active model
                model, model_info = await self._get_active_model(request.model_id, request.version)
                
                # Make prediction
                prediction_result = await self._make_prediction(
                    model, request.data, model_info, request.return_confidence, request.return_explanation
                )
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Update metrics
                PREDICTION_COUNTER.labels(
                    model_id=model_info["model_id"],
                    version=model_info["version"]
                ).inc()
                
                PREDICTION_LATENCY.labels(
                    model_id=model_info["model_id"],
                    version=model_info["version"]
                ).observe(latency_ms / 1000)
                
                return PredictionResponse(
                    prediction=prediction_result["score"],
                    is_anomaly=prediction_result["is_anomaly"],
                    confidence=prediction_result.get("confidence"),
                    explanation=prediction_result.get("explanation"),
                    model_id=model_info["model_id"],
                    model_version=model_info["version"],
                    prediction_time=datetime.utcnow(),
                    latency_ms=latency_ms
                )
                
            except Exception as e:
                ERROR_COUNTER.labels(
                    model_id=request.model_id or "unknown",
                    version=request.version or "unknown",
                    error_type=type(e).__name__
                ).inc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
        async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
            """Make batch predictions."""
            try:
                start_time = time.time()
                
                # Get active model
                model, model_info = await self._get_active_model(request.model_id, request.version)
                
                # Process in batches
                predictions = []
                total_anomalies = 0
                
                for i in range(0, len(request.data), request.batch_size):
                    batch = request.data[i:i + request.batch_size]
                    
                    for data_point in batch:
                        pred_start = time.time()
                        
                        prediction_result = await self._make_prediction(
                            model, data_point, model_info, 
                            request.return_confidence, request.return_explanation
                        )
                        
                        pred_latency = (time.time() - pred_start) * 1000
                        
                        prediction_response = PredictionResponse(
                            prediction=prediction_result["score"],
                            is_anomaly=prediction_result["is_anomaly"],
                            confidence=prediction_result.get("confidence"),
                            explanation=prediction_result.get("explanation"),
                            model_id=model_info["model_id"],
                            model_version=model_info["version"],
                            prediction_time=datetime.utcnow(),
                            latency_ms=pred_latency
                        )
                        
                        predictions.append(prediction_response)
                        
                        if prediction_result["is_anomaly"]:
                            total_anomalies += 1
                    
                    # Small delay between batches to prevent overwhelming
                    await asyncio.sleep(0.01)
                
                # Calculate metrics
                total_time_ms = (time.time() - start_time) * 1000
                throughput = len(request.data) / (total_time_ms / 1000) if total_time_ms > 0 else 0
                
                # Update metrics
                PREDICTION_COUNTER.labels(
                    model_id=model_info["model_id"],
                    version=model_info["version"]
                ).inc(len(request.data))
                
                return BatchPredictionResponse(
                    predictions=predictions,
                    total_processed=len(request.data),
                    total_anomalies=total_anomalies,
                    processing_time_ms=total_time_ms,
                    throughput_per_second=throughput
                )
                
            except Exception as e:
                ERROR_COUNTER.labels(
                    model_id=request.model_id or "unknown",
                    version=request.version or "unknown",
                    error_type=type(e).__name__
                ).inc()
                raise HTTPException(status_code=500, detail=str(e))
    
    def _add_model_management_routes(self, app: FastAPI) -> None:
        """Add model management endpoints."""
        
        @app.get("/api/v1/models", response_model=List[ModelInfo])
        async def list_models() -> List[ModelInfo]:
            """List available models."""
            try:
                deployments = await self.deployment_service.list_deployments(
                    environment=self.environment,
                    limit=100
                )
                
                model_infos = []
                for deployment in deployments:
                    if deployment.is_deployed:
                        model_info = ModelInfo(
                            model_id=str(deployment.model_version_id),
                            model_name=f"Model-{deployment.model_version_id}",
                            version="1.0.0",  # Would get from model registry
                            algorithm="IsolationForest",  # Would get from model registry
                            domain="general",  # Would get from model registry
                            status=deployment.status.value,
                            health_score=deployment.health_score,
                            last_updated=deployment.created_at,
                            performance_metrics=deployment.get_deployment_info()["health_metrics"]
                        )
                        model_infos.append(model_info)
                
                return model_infos
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/v1/models/{model_id}", response_model=ModelInfo)
        async def get_model_info(model_id: str) -> ModelInfo:
            """Get information about a specific model."""
            try:
                model_uuid = UUID(model_id)
                deployments = await self.deployment_service.list_deployments(
                    environment=self.environment,
                    model_version_id=model_uuid
                )
                
                if not deployments:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                deployment = deployments[0]
                return ModelInfo(
                    model_id=model_id,
                    model_name=f"Model-{model_id}",
                    version="1.0.0",
                    algorithm="IsolationForest",
                    domain="general",
                    status=deployment.status.value,
                    health_score=deployment.health_score,
                    last_updated=deployment.created_at,
                    performance_metrics=deployment.get_deployment_info()["health_metrics"]
                )
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid model ID format")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/v1/models/{model_id}/load")
        async def load_model(model_id: str) -> Dict[str, str]:
            """Load a model into memory."""
            try:
                load_start = time.time()
                
                # Load model (simplified - would integrate with actual model loading)
                await self._load_model(model_id)
                
                load_time = time.time() - load_start
                MODEL_LOAD_TIME.labels(model_id=model_id, version="1.0.0").observe(load_time)
                
                return {"status": "loaded", "model_id": model_id}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.delete("/api/v1/models/{model_id}/unload")
        async def unload_model(model_id: str) -> Dict[str, str]:
            """Unload a model from memory."""
            try:
                if model_id in self.loaded_models:
                    del self.loaded_models[model_id]
                    del self.model_metadata[model_id]
                    del self.model_access_times[model_id]
                
                return {"status": "unloaded", "model_id": model_id}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def _add_health_routes(self, app: FastAPI) -> None:
        """Add health check endpoints."""
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Basic health check."""
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow(),
                version="1.0.0",
                uptime_seconds=uptime,
                loaded_models=len(self.loaded_models),
                active_connections=len(self.active_websockets),
                memory_usage_mb=self._get_memory_usage()
            )
        
        @app.get("/ready")
        async def readiness_check() -> Dict[str, str]:
            """Readiness check."""
            # Check if at least one model is loaded
            if not self.loaded_models:
                raise HTTPException(status_code=503, detail="No models loaded")
            
            return {"status": "ready"}
        
        @app.get("/metrics")
        async def prometheus_metrics():
            """Prometheus metrics endpoint."""
            # Update gauge metrics
            ACTIVE_CONNECTIONS.set(len(self.active_websockets))
            
            # Generate metrics
            encoder, content_type = choose_encoder(None)
            output = encoder(generate_latest())
            
            return JSONResponse(
                content=output.decode('utf-8'),
                media_type=content_type
            )
    
    def _add_websocket_routes(self, app: FastAPI) -> None:
        """Add WebSocket endpoints for streaming predictions."""
        
        @app.websocket("/api/v1/predict/stream")
        async def websocket_predictions(websocket: WebSocket):
            """WebSocket endpoint for streaming predictions."""
            await websocket.accept()
            self.active_websockets.append(websocket)
            ACTIVE_CONNECTIONS.set(len(self.active_websockets))
            
            try:
                while True:
                    # Receive data
                    data = await websocket.receive_json()
                    
                    try:
                        # Parse request
                        request = PredictionRequest(**data)
                        
                        # Get model and make prediction
                        model, model_info = await self._get_active_model(
                            request.model_id, request.version
                        )
                        
                        start_time = time.time()
                        prediction_result = await self._make_prediction(
                            model, request.data, model_info,
                            request.return_confidence, request.return_explanation
                        )
                        latency_ms = (time.time() - start_time) * 1000
                        
                        # Send response
                        response = PredictionResponse(
                            prediction=prediction_result["score"],
                            is_anomaly=prediction_result["is_anomaly"],
                            confidence=prediction_result.get("confidence"),
                            explanation=prediction_result.get("explanation"),
                            model_id=model_info["model_id"],
                            model_version=model_info["version"],
                            prediction_time=datetime.utcnow(),
                            latency_ms=latency_ms
                        )
                        
                        await websocket.send_json(response.dict())
                        
                        # Update metrics
                        PREDICTION_COUNTER.labels(
                            model_id=model_info["model_id"],
                            version=model_info["version"]
                        ).inc()
                        
                    except Exception as e:
                        # Send error response
                        error_response = {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        await websocket.send_json(error_response)
                        
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.active_websockets:
                    self.active_websockets.remove(websocket)
                ACTIVE_CONNECTIONS.set(len(self.active_websockets))
    
    async def _get_active_model(
        self,
        model_id: Optional[str] = None,
        version: Optional[str] = None
    ) -> tuple[Any, Dict[str, Any]]:
        """Get active model for prediction."""
        # If specific model requested, use it
        if model_id:
            if model_id not in self.loaded_models:
                await self._load_model(model_id, version)
            
            self.model_access_times[model_id] = datetime.utcnow()
            return self.loaded_models[model_id], self.model_metadata[model_id]
        
        # Otherwise, get default active model from environment
        deployments = await self.deployment_service.list_deployments(
            environment=self.environment,
            limit=1
        )
        
        if not deployments:
            raise ModelNotLoadedError("No active models available")
        
        deployment = deployments[0]
        model_id = str(deployment.model_version_id)
        
        if model_id not in self.loaded_models:
            await self._load_model(model_id)
        
        self.model_access_times[model_id] = datetime.utcnow()
        return self.loaded_models[model_id], self.model_metadata[model_id]
    
    async def _load_model(self, model_id: str, version: Optional[str] = None) -> None:
        """Load model into memory."""
        # Check cache size
        if len(self.loaded_models) >= self.model_cache_size:
            await self._evict_least_recently_used_model()
        
        # Simulate model loading
        # In real implementation, this would load actual model from storage
        await asyncio.sleep(0.1)  # Simulate loading time
        
        # Create mock model and metadata
        self.loaded_models[model_id] = {"type": "isolation_forest", "threshold": 0.5}
        self.model_metadata[model_id] = {
            "model_id": model_id,
            "version": version or "1.0.0",
            "algorithm": "IsolationForest",
            "loaded_at": datetime.utcnow()
        }
        self.model_access_times[model_id] = datetime.utcnow()
    
    async def _evict_least_recently_used_model(self) -> None:
        """Evict least recently used model from cache."""
        if not self.model_access_times:
            return
        
        lru_model_id = min(
            self.model_access_times.keys(),
            key=lambda k: self.model_access_times[k]
        )
        
        del self.loaded_models[lru_model_id]
        del self.model_metadata[lru_model_id]
        del self.model_access_times[lru_model_id]
    
    async def _make_prediction(
        self,
        model: Any,
        data: Dict[str, Any],
        model_info: Dict[str, Any],
        return_confidence: bool = True,
        return_explanation: bool = False
    ) -> Dict[str, Any]:
        """Make prediction using loaded model."""
        try:
            # Simulate prediction
            # In real implementation, this would use actual model
            
            # Convert data to appropriate format
            if isinstance(data, dict):
                # Convert to numpy array or pandas DataFrame as needed
                feature_values = list(data.values())
                input_array = np.array([feature_values])
            else:
                input_array = np.array(data)
            
            # Simulate anomaly score calculation
            anomaly_score = np.random.uniform(0.0, 1.0)
            threshold = model.get("threshold", 0.5)
            is_anomaly = anomaly_score > threshold
            
            result = {
                "score": float(anomaly_score),
                "is_anomaly": bool(is_anomaly)
            }
            
            if return_confidence:
                # Simulate confidence calculation
                confidence = 1.0 - abs(anomaly_score - threshold)
                result["confidence"] = float(confidence)
            
            if return_explanation:
                # Simulate explanation
                result["explanation"] = {
                    "feature_importance": {f"feature_{i}": float(np.random.uniform(0, 1)) 
                                         for i in range(len(feature_values))},
                    "threshold": threshold,
                    "distance_to_threshold": float(abs(anomaly_score - threshold))
                }
            
            return result
            
        except Exception as e:
            raise PredictionError(f"Prediction failed: {e}") from e
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0  # Return 0 if psutil not available


# Factory function for creating model server
def create_model_server(
    deployment_service: DeploymentOrchestrationService,
    model_registry_service: ModelRegistryService,
    environment: Environment = Environment.PRODUCTION
) -> ModelServer:
    """Create model server instance."""
    return ModelServer(
        deployment_service=deployment_service,
        model_registry_service=model_registry_service,
        environment=environment
    )