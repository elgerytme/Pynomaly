"""
Production Model Serving Infrastructure

This module provides a robust, scalable model serving layer for production ML systems.
Supports dynamic model loading, health checks, performance monitoring, and request batching.
"""

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from machine_learning.domain.entities.model import Model, ModelStatus
from machine_learning.infrastructure.repositories.model_repository import ModelRepository
from machine_learning.application.services.model_loading_service import ModelLoadingService
from machine_learning.application.services.inference_service import InferenceService


# Metrics
REQUEST_COUNT = Counter('ml_serving_requests_total', 'Total requests', ['model_id', 'status'])
REQUEST_DURATION = Histogram('ml_serving_request_duration_seconds', 'Request duration', ['model_id'])
ACTIVE_MODELS = Gauge('ml_serving_active_models', 'Number of active models')
MODEL_MEMORY_USAGE = Gauge('ml_serving_model_memory_mb', 'Model memory usage', ['model_id'])


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    model_id: str = Field(..., description="Model identifier")
    instances: List[List[float]] = Field(..., description="Input data instances")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model parameters")
    explain: bool = Field(default=False, description="Include prediction explanations")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "fraud_detection_v1.2.3",
                "instances": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "parameters": {"threshold": 0.5},
                "explain": False
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    model_id: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Model version")
    predictions: List[Union[float, int, str]] = Field(..., description="Prediction results")
    probabilities: Optional[List[List[float]]] = Field(None, description="Prediction probabilities")
    explanations: Optional[List[Dict[str, Any]]] = Field(None, description="Prediction explanations")
    confidence_scores: Optional[List[float]] = Field(None, description="Confidence scores")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction_id": "pred_123e4567-e89b-12d3-a456-426614174000",
                "model_id": "fraud_detection_v1.2.3", 
                "model_version": "1.2.3",
                "predictions": [0, 1],
                "probabilities": [[0.8, 0.2], [0.3, 0.7]],
                "confidence_scores": [0.8, 0.7],
                "processing_time_ms": 15.5,
                "timestamp": "2023-12-07T10:30:00Z"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    model_id: str = Field(..., description="Model identifier")
    batch_instances: List[List[List[float]]] = Field(..., description="Batch of input instances")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    callback_url: Optional[str] = Field(None, description="Callback URL for async results")


class ModelHealthStatus(BaseModel):
    """Model health status."""
    model_id: str
    status: str
    version: str
    loaded_at: datetime
    last_prediction: Optional[datetime]
    prediction_count: int
    avg_latency_ms: float
    memory_usage_mb: float
    error_rate: float


class ModelServingEngine:
    """Core model serving engine with caching, batching, and monitoring."""
    
    def __init__(self, model_repository: ModelRepository, max_models: int = 10):
        self.model_repository = model_repository
        self.max_models = max_models
        self.loaded_models: Dict[str, Model] = {}
        self.model_stats: Dict[str, Dict] = {}
        self.loading_service = ModelLoadingService(model_repository)
        self.inference_service = InferenceService()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        
        # Request batching
        self.batch_size = 32
        self.batch_timeout = 0.1  # 100ms
        self.pending_requests: List = []
        self.batch_lock = asyncio.Lock()
        
    async def load_model(self, model_id: str, force_reload: bool = False) -> Model:
        """Load or reload a model into memory."""
        if model_id in self.loaded_models and not force_reload:
            return self.loaded_models[model_id]
        
        # Check memory limits
        if len(self.loaded_models) >= self.max_models:
            await self._evict_least_used_model()
        
        try:
            # Load model from repository
            model = await self.loading_service.load_model(model_id)
            
            # Validate model
            await self._validate_model(model)
            
            # Cache model
            self.loaded_models[model_id] = model
            self.model_stats[model_id] = {
                'loaded_at': datetime.utcnow(),
                'prediction_count': 0,
                'total_latency': 0.0,
                'error_count': 0,
                'last_prediction': None
            }
            
            ACTIVE_MODELS.set(len(self.loaded_models))
            self.logger.info(f"Model {model_id} loaded successfully")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make predictions with monitoring and error handling."""
        start_time = time.time()
        prediction_id = str(uuid.uuid4())
        
        try:
            # Load model if not cached
            model = await self.load_model(request.model_id)
            
            # Validate input
            self._validate_input(request.instances)
            
            # Make prediction
            predictions = await self.inference_service.predict(
                model=model,
                instances=request.instances,
                parameters=request.parameters
            )
            
            # Get additional info if requested
            probabilities = None
            explanations = None
            confidence_scores = None
            
            if hasattr(predictions, 'probabilities'):
                probabilities = predictions.probabilities
            
            if request.explain:
                explanations = await self.inference_service.explain(
                    model=model,
                    instances=request.instances,
                    predictions=predictions.values
                )
            
            if hasattr(predictions, 'confidence'):
                confidence_scores = predictions.confidence
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(request.model_id, processing_time, success=True)
            
            response = PredictionResponse(
                prediction_id=prediction_id,
                model_id=request.model_id,
                model_version=model.version,
                predictions=predictions.values,
                probabilities=probabilities,
                explanations=explanations,
                confidence_scores=confidence_scores,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )
            
            REQUEST_COUNT.labels(model_id=request.model_id, status='success').inc()
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(request.model_id, processing_time, success=False)
            REQUEST_COUNT.labels(model_id=request.model_id, status='error').inc()
            
            self.logger.error(f"Prediction failed for {request.model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def batch_predict(self, request: BatchPredictionRequest) -> List[PredictionResponse]:
        """Process batch predictions efficiently."""
        responses = []
        
        for batch in request.batch_instances:
            pred_request = PredictionRequest(
                model_id=request.model_id,
                instances=batch,
                parameters=request.parameters
            )
            response = await self.predict(pred_request)
            responses.append(response)
        
        return responses
    
    async def get_model_health(self, model_id: str) -> ModelHealthStatus:
        """Get comprehensive model health status."""
        if model_id not in self.loaded_models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not loaded")
        
        model = self.loaded_models[model_id]
        stats = self.model_stats[model_id]
        
        avg_latency = (
            stats['total_latency'] / stats['prediction_count'] 
            if stats['prediction_count'] > 0 else 0.0
        )
        
        error_rate = (
            stats['error_count'] / stats['prediction_count'] 
            if stats['prediction_count'] > 0 else 0.0
        )
        
        # Estimate memory usage (simplified)
        memory_usage = await self._estimate_model_memory(model)
        
        return ModelHealthStatus(
            model_id=model_id,
            status=model.status.value,
            version=model.version,
            loaded_at=stats['loaded_at'],
            last_prediction=stats['last_prediction'],
            prediction_count=stats['prediction_count'],
            avg_latency_ms=avg_latency,
            memory_usage_mb=memory_usage,
            error_rate=error_rate
        )
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload model from memory."""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            del self.model_stats[model_id]
            ACTIVE_MODELS.set(len(self.loaded_models))
            self.logger.info(f"Model {model_id} unloaded")
            return True
        return False
    
    def _validate_input(self, instances: List[List[float]]) -> None:
        """Validate input data format."""
        if not instances:
            raise ValueError("No instances provided")
        
        if not all(isinstance(instance, list) for instance in instances):
            raise ValueError("All instances must be lists")
        
        if not all(isinstance(val, (int, float)) for instance in instances for val in instance):
            raise ValueError("All values must be numeric")
        
        # Check for consistent feature count
        feature_count = len(instances[0])
        if not all(len(instance) == feature_count for instance in instances):
            raise ValueError("All instances must have the same number of features")
    
    async def _validate_model(self, model: Model) -> None:
        """Validate loaded model."""
        if model.status != ModelStatus.TRAINED:
            raise ValueError(f"Model {model.id} is not trained")
        
        if not hasattr(model, 'predict'):
            raise ValueError(f"Model {model.id} does not have predict method")
    
    async def _evict_least_used_model(self) -> None:
        """Evict least recently used model."""
        if not self.loaded_models:
            return
        
        # Find model with oldest last_prediction or lowest prediction_count
        least_used = min(
            self.model_stats.items(),
            key=lambda x: (
                x[1]['last_prediction'] or datetime.min,
                x[1]['prediction_count']
            )
        )
        
        model_id = least_used[0]
        await self.unload_model(model_id)
        self.logger.info(f"Evicted model {model_id} due to memory constraints")
    
    def _update_metrics(self, model_id: str, processing_time: float, success: bool) -> None:
        """Update model performance metrics."""
        if model_id in self.model_stats:
            stats = self.model_stats[model_id]
            stats['prediction_count'] += 1
            stats['total_latency'] += processing_time
            stats['last_prediction'] = datetime.utcnow()
            
            if not success:
                stats['error_count'] += 1
            
            REQUEST_DURATION.labels(model_id=model_id).observe(processing_time / 1000)
    
    async def _estimate_model_memory(self, model: Model) -> float:
        """Estimate model memory usage in MB."""
        # Simplified memory estimation
        # In production, use actual memory profiling
        try:
            if hasattr(model, 'get_memory_usage'):
                return model.get_memory_usage()
            else:
                # Rough estimate based on model parameters
                return 50.0  # Default 50MB
        except:
            return 0.0


# Global serving engine instance
serving_engine: Optional[ModelServingEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global serving_engine
    
    # Startup
    model_repository = ModelRepository()  # Initialize with your config
    serving_engine = ModelServingEngine(model_repository)
    
    yield
    
    # Shutdown
    if serving_engine:
        serving_engine.executor.shutdown(wait=True)


# FastAPI application
app = FastAPI(
    title="ML Model Serving API",
    description="Production-ready ML model serving infrastructure",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware)


def get_serving_engine() -> ModelServingEngine:
    """Dependency to get serving engine."""
    if serving_engine is None:
        raise HTTPException(status_code=503, detail="Serving engine not initialized")
    return serving_engine


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    engine: ModelServingEngine = Depends(get_serving_engine)
) -> PredictionResponse:
    """Make predictions using the specified model."""
    return await engine.predict(request)


@app.post("/v1/batch-predict", response_model=List[PredictionResponse])
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    engine: ModelServingEngine = Depends(get_serving_engine)
) -> List[PredictionResponse]:
    """Process batch predictions."""
    return await engine.batch_predict(request)


@app.post("/v1/models/{model_id}/load")
async def load_model(
    model_id: str,
    force_reload: bool = False,
    engine: ModelServingEngine = Depends(get_serving_engine)
):
    """Load a model into memory."""
    await engine.load_model(model_id, force_reload)
    return {"status": "success", "message": f"Model {model_id} loaded"}


@app.delete("/v1/models/{model_id}")
async def unload_model(
    model_id: str,
    engine: ModelServingEngine = Depends(get_serving_engine)
):
    """Unload a model from memory."""
    success = await engine.unload_model(model_id)
    if success:
        return {"status": "success", "message": f"Model {model_id} unloaded"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")


@app.get("/v1/models/{model_id}/health", response_model=ModelHealthStatus)
async def get_model_health(
    model_id: str,
    engine: ModelServingEngine = Depends(get_serving_engine)
) -> ModelHealthStatus:
    """Get model health and performance metrics."""
    return await engine.get_model_health(model_id)


@app.get("/v1/models")
async def list_loaded_models(
    engine: ModelServingEngine = Depends(get_serving_engine)
):
    """List all loaded models."""
    models = []
    for model_id, model in engine.loaded_models.items():
        health = await engine.get_model_health(model_id)
        models.append({
            "model_id": model_id,
            "version": model.version,
            "status": health.status,
            "prediction_count": health.prediction_count,
            "avg_latency_ms": health.avg_latency_ms
        })
    
    return {"models": models, "total_count": len(models)}


@app.get("/health")
async def health_check():
    """General health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "service": "ml-model-serving",
        "version": "1.0.0"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=8080,
        workers=1,  # Single worker for now, scale with load balancer
        reload=False,
        log_level="info"
    )