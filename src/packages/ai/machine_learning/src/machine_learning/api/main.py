"""FastAPI application for Machine Learning service."""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from machine_learning.infrastructure.container.container import MachineLearningContainer
from machine_learning.domain.entities.training_data import TrainingData
from machine_learning.domain.entities.prediction_request import PredictionRequest
from machine_learning.domain.interfaces.data_operations import (
    DataIngestionPort, DataProcessingPort, DataStoragePort
)
from machine_learning.domain.interfaces.model_operations import (
    ModelTrainingPort, ModelPredictionPort, ModelManagementPort
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Machine Learning Service",
    description="Hexagonal Architecture Machine Learning API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize container
container = MachineLearningContainer()

# Dependency injection
def get_data_ingestion_service() -> DataIngestionPort:
    return container.get(DataIngestionPort)

def get_data_processing_service() -> DataProcessingPort:
    return container.get(DataProcessingPort)

def get_data_storage_service() -> DataStoragePort:
    return container.get(DataStoragePort)

def get_model_training_service() -> ModelTrainingPort:
    return container.get(ModelTrainingPort)

def get_model_prediction_service() -> ModelPredictionPort:
    return container.get(ModelPredictionPort)

def get_model_management_service() -> ModelManagementPort:
    return container.get(ModelManagementPort)

# Health and readiness endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "machine-learning", "timestamp": datetime.utcnow()}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Perform basic service checks
        ingestion_service = get_data_ingestion_service()
        return {"status": "ready", "service": "machine-learning", "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Basic metrics - in production, use prometheus_client library
    return {
        "ml_predictions_total": 500,
        "ml_prediction_errors_total": 10,
        "ml_model_accuracy": 0.95,
        "ml_models_trained_total": 15,
        "data_ingestion_operations_total": 200
    }

# Data Operations API
@app.post("/api/v1/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_data(
    request: Dict[str, Any],
    ingestion_service: DataIngestionPort = Depends(get_data_ingestion_service)
):
    """Ingest training data."""
    try:
        # Validate request
        if "source" not in request:
            raise HTTPException(status_code=400, detail="source is required")
        
        # Ingest data
        training_data = await ingestion_service.ingest_data(request["source"])
        
        return {
            "status": "success",
            "data": {
                "source": training_data.source,
                "features_count": len(training_data.features),
                "labels_count": len(training_data.labels),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to ingest data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/process")
async def process_data(
    request: Dict[str, Any],
    processing_service: DataProcessingPort = Depends(get_data_processing_service),
    ingestion_service: DataIngestionPort = Depends(get_data_ingestion_service)
):
    """Process training data."""
    try:
        # Get data first
        if "source" not in request:
            raise HTTPException(status_code=400, detail="source is required")
        
        raw_data = await ingestion_service.ingest_data(request["source"])
        processed_data = await processing_service.process_data(raw_data)
        
        return {
            "status": "success",
            "data": {
                "source": processed_data.source,
                "processed_features_count": len(processed_data.features),
                "processed_labels_count": len(processed_data.labels),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/store")
async def store_data(
    request: Dict[str, Any],
    storage_service: DataStoragePort = Depends(get_data_storage_service),
    ingestion_service: DataIngestionPort = Depends(get_data_ingestion_service)
):
    """Store training data."""
    try:
        # Validate request
        if "source" not in request or "identifier" not in request:
            raise HTTPException(status_code=400, detail="source and identifier are required")
        
        # Get and store data
        data = await ingestion_service.ingest_data(request["source"])
        success = await storage_service.store_data(data, request["identifier"])
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store data")
        
        return {
            "status": "success",
            "data": {
                "identifier": request["identifier"],
                "stored": success,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to store data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/{identifier}")
async def retrieve_data(
    identifier: str,
    storage_service: DataStoragePort = Depends(get_data_storage_service)
):
    """Retrieve stored training data."""
    try:
        data = await storage_service.retrieve_data(identifier)
        
        if not data:
            raise HTTPException(status_code=404, detail="Data not found")
        
        return {
            "status": "success",
            "data": {
                "identifier": identifier,
                "source": data.source,
                "features_count": len(data.features),
                "labels_count": len(data.labels),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model Operations API
@app.post("/api/v1/train", status_code=status.HTTP_201_CREATED)
async def train_model(
    request: Dict[str, Any],
    training_service: ModelTrainingPort = Depends(get_model_training_service),
    ingestion_service: DataIngestionPort = Depends(get_data_ingestion_service)
):
    """Train a machine learning model."""
    try:
        # Validate request
        if "data_source" not in request or "model_id" not in request:
            raise HTTPException(status_code=400, detail="data_source and model_id are required")
        
        # Get training data
        training_data = await ingestion_service.ingest_data(request["data_source"])
        
        # Train model
        model_metadata = await training_service.train_model(training_data, request["model_id"])
        
        return {
            "status": "success",
            "data": {
                "model_id": model_metadata.model_id,
                "version": model_metadata.version,
                "training_accuracy": model_metadata.accuracy,
                "training_timestamp": model_metadata.created_at,
                "features_used": len(training_data.features[0]) if training_data.features else 0
            }
        }
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict")
async def predict(
    request: Dict[str, Any],
    prediction_service: ModelPredictionPort = Depends(get_model_prediction_service)
):
    """Make predictions using a trained model."""
    try:
        # Validate request
        if "model_id" not in request or "features" not in request:
            raise HTTPException(status_code=400, detail="model_id and features are required")
        
        # Create prediction request
        prediction_request = PredictionRequest(
            model_id=request["model_id"],
            features=request["features"],
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Make prediction
        prediction_result = await prediction_service.predict(prediction_request)
        
        return {
            "status": "success",
            "data": {
                "model_id": prediction_result.model_id,
                "prediction": prediction_result.prediction,
                "confidence": prediction_result.confidence,
                "timestamp": prediction_result.timestamp,
                "features_processed": len(request["features"])
            }
        }
    except Exception as e:
        logger.error(f"Failed to make prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models")
async def list_models(
    management_service: ModelManagementPort = Depends(get_model_management_service)
):
    """List all available models."""
    try:
        models = await management_service.list_models()
        
        return {
            "status": "success",
            "data": {
                "models": [
                    {
                        "model_id": model.model_id,
                        "version": model.version,
                        "accuracy": model.accuracy,
                        "created_at": model.created_at,
                        "is_active": model.is_active
                    }
                    for model in models
                ],
                "total_count": len(models),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/{model_id}")
async def get_model(
    model_id: str,
    management_service: ModelManagementPort = Depends(get_model_management_service)
):
    """Get model details."""
    try:
        model = await management_service.get_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "status": "success",
            "data": {
                "model_id": model.model_id,
                "version": model.version,
                "accuracy": model.accuracy,
                "created_at": model.created_at,
                "is_active": model.is_active,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/models/{model_id}")
async def delete_model(
    model_id: str,
    management_service: ModelManagementPort = Depends(get_model_management_service)
):
    """Delete a model."""
    try:
        success = await management_service.delete_model(model_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "status": "success",
            "data": {
                "model_id": model_id,
                "deleted": success,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def get_service_status():
    """Get service status and configuration."""
    return {
        "status": "running",
        "service": "machine-learning",
        "version": "1.0.0",
        "environment": "development",
        "timestamp": datetime.utcnow(),
        "capabilities": [
            "data_ingestion",
            "data_processing",
            "data_storage",
            "model_training",
            "model_prediction",
            "model_management"
        ]
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )