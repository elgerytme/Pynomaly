"""Anomaly Detection FastAPI server."""

import structlog
import numpy as np
import numpy.typing as npt
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from uvicorn import run as uvicorn_run
from typing import AsyncGenerator, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from .domain.services.detection_service import DetectionService
from .domain.services.ensemble_service import EnsembleService
try:
    from data.processing.domain.entities.model import Model
except ImportError:
    from .domain.entities.model import Model
from .infrastructure.repositories.model_repository import ModelRepository
from .infrastructure.config.settings import get_settings
from .infrastructure.logging import get_logger, async_log_decorator
from .infrastructure.logging.error_handler import (
    ErrorHandler, 
    AnomalyDetectionError, 
    InputValidationError, 
    AlgorithmError, 
    ModelOperationError,
    PersistenceError
)
from .infrastructure.monitoring import (
    get_metrics_collector,
    get_health_checker,
    get_performance_monitor
)
from .infrastructure.monitoring import get_monitoring_dashboard
from .infrastructure.middleware.rate_limiting import create_rate_limit_middleware

logger = get_logger(__name__)
settings = get_settings()
error_handler = ErrorHandler(logger._logger)
metrics_collector = get_metrics_collector()
health_checker = get_health_checker()
performance_monitor = get_performance_monitor()
monitoring_dashboard = get_monitoring_dashboard()

# Global service instances - initialized during startup
global_detection_service: Optional[DetectionService] = None
global_ensemble_service: Optional[EnsembleService] = None
global_model_repository: Optional[ModelRepository] = None


class DetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    data: List[List[float]] = Field(..., description="Input data as list of feature vectors")
    algorithm: str = Field(default="isolation_forest", description="Detection algorithm to use")
    contamination: float = Field(default=0.1, ge=0.001, le=0.5, description="Expected contamination rate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific parameters")


class DetectionResponse(BaseModel):
    """Response model for anomaly detection."""
    success: bool = Field(..., description="Whether detection completed successfully")
    anomalies: List[int] = Field(..., description="Indices of detected anomalies")
    scores: Optional[List[float]] = Field(None, description="Anomaly confidence scores")
    algorithm: str = Field(..., description="Algorithm used for detection")
    total_samples: int = Field(..., description="Total number of samples processed")
    anomalies_detected: int = Field(..., description="Number of anomalies detected")
    anomaly_rate: float = Field(..., description="Ratio of anomalies to total samples")
    timestamp: str = Field(..., description="Detection timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class EnsembleRequest(BaseModel):
    """Request model for ensemble detection."""
    data: List[List[float]] = Field(..., description="Input data as list of feature vectors")
    algorithms: List[str] = Field(default=["isolation_forest", "one_class_svm", "lof"], 
                                 description="Algorithms to use in ensemble")
    method: str = Field(default="majority", description="Ensemble combination method")
    contamination: float = Field(default=0.1, ge=0.001, le=0.5, description="Expected contamination rate")
    parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict, 
                                                 description="Algorithm-specific parameters")


class ModelPredictionRequest(BaseModel):
    """Request model for predictions using saved models."""
    data: List[List[float]] = Field(..., description="Input data as list of feature vectors")
    model_id: str = Field(..., description="ID of the model to use for prediction")


class ModelListResponse(BaseModel):
    """Response model for listing models."""
    models: List[Dict[str, Any]] = Field(..., description="List of available models")
    total_count: int = Field(..., description="Total number of models")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Current timestamp")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")


# Global variables for service instances
_detection_service: Optional[DetectionService] = None
_ensemble_service: Optional[EnsembleService] = None
_model_repository: Optional[ModelRepository] = None
_app_start_time: Optional[datetime] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    global _detection_service, _ensemble_service, _model_repository, _app_start_time
    
    logger.info("Starting Anomaly Detection API server")
    _app_start_time = datetime.utcnow()
    
    # Initialize services
    global global_detection_service, global_ensemble_service, global_model_repository
    global_detection_service = DetectionService()
    global_ensemble_service = EnsembleService()
    global_model_repository = ModelRepository()
    
    # Keep local references for backward compatibility
    _detection_service = global_detection_service
    _ensemble_service = global_ensemble_service
    _model_repository = global_model_repository
    
    logger.info("Services initialized successfully")
    
    yield
    
    logger.info("Shutting down Anomaly Detection API server")
    # Cleanup resources if needed


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    app = FastAPI(
        title="🔍 Anomaly Detection Platform API",
        description="""
        ## Enterprise-Grade Anomaly Detection Platform
        
        A comprehensive REST API for machine learning-based anomaly detection with advanced features:
        
        ### 🚀 Key Features
        - **Multiple Algorithms**: Isolation Forest, One-Class SVM, Local Outlier Factor, Ensemble Methods
        - **Real-time Processing**: Streaming detection with WebSocket support
        - **Model Management**: Training, versioning, deployment, and performance monitoring
        - **Health Monitoring**: System health checks and performance metrics
        - **Batch Processing**: Efficient processing of large datasets
        
        ### 🔧 API Capabilities
        - **Async Processing**: High-performance async endpoints
        - **Input Validation**: Comprehensive request validation with Pydantic
        - **Error Handling**: Structured error responses with detailed context
        - **Rate Limiting**: Production-ready request throttling
        - **Monitoring**: Built-in metrics collection and health checks
        
        ### 📊 Supported Algorithms
        - `isolation_forest`: Isolation Forest (default)
        - `one_class_svm`: One-Class Support Vector Machine
        - `local_outlier_factor`: Local Outlier Factor
        - `ensemble`: Ensemble of multiple algorithms
        
        ### 🏗️ Architecture
        Built with Domain-Driven Design principles using FastAPI, Pydantic, and modern Python patterns.
        
        ---
        
        **Version**: 2.0.0 | **Environment**: Production Ready | **License**: MIT
        """,
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Anomaly Detection Platform Team",
            "email": "support@anomaly-detection.io",
            "url": "https://github.com/monorepo/anomaly_detection"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.anomaly-detection.io",
                "description": "Production server"
            }
        ],
        tags_metadata=[
            {
                "name": "detection",
                "description": "Anomaly detection operations. Core functionality for detecting anomalies in datasets.",
                "externalDocs": {
                    "description": "Detection Documentation",
                    "url": "https://docs.anomaly-detection.io/detection"
                }
            },
            {
                "name": "models",
                "description": "Model management operations. Train, deploy, and manage anomaly detection models.",
                "externalDocs": {
                    "description": "Model Management Guide",
                    "url": "https://docs.anomaly-detection.io/models"
                }
            },
            {
                "name": "streaming",
                "description": "Real-time streaming detection. Process data streams with concept drift monitoring.",
                "externalDocs": {
                    "description": "Streaming Guide",
                    "url": "https://docs.anomaly-detection.io/streaming"
                }
            },
            {
                "name": "health",
                "description": "Health checks and system monitoring. Monitor API health and performance metrics."
            },
            {
                "name": "workers",
                "description": "Background job management. Manage and monitor background processing tasks."
            },
            {
                "name": "monitoring",
                "description": "Performance monitoring and metrics. Real-time system and model performance tracking."
            }
        ]
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add rate limiting middleware
    rate_limit_middleware = create_rate_limit_middleware(
        requests_per_minute=60,  # 60 requests per minute per IP
        requests_per_hour=1000,  # 1000 requests per hour per IP  
        burst_limit=10,          # 10 burst requests
        exempt_paths=["/docs", "/redoc", "/openapi.json", "/health", "/"]
    )
    app.middleware("http")(rate_limit_middleware)
    
    # Add error handling middleware
    @app.middleware("http")
    async def error_handling_middleware(request: Request, call_next):
        """Global error handling middleware."""
        request_id = str(uuid.uuid4())
        
        # Set request context for logging
        logger.set_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path
        )
        
        try:
            response = await call_next(request)
            return response
        except AnomalyDetectionError as e:
            logger.error("AnomalyDetectionError caught in middleware",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        request_id=request_id)
            
            error_response = error_handler.create_error_response(e)
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST if e.recoverable else status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response
            )
        except Exception as e:
            logger.error("Unexpected error caught in middleware",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        request_id=request_id)
            
            # Convert to our error format
            ad_error = error_handler.handle_error(
                error=e,
                context={"request_id": request_id},
                operation="api_request",
                reraise=False
            )
            
            error_response = error_handler.create_error_response(ad_error)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response
            )
        finally:
            logger.clear_request_context()
    
    return app


app = create_app()


def get_detection_service() -> DetectionService:
    """Dependency injection for detection service."""
    if _detection_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection service not initialized"
        )
    return _detection_service


def get_ensemble_service() -> EnsembleService:
    """Dependency injection for ensemble service."""
    if _ensemble_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ensemble service not initialized"
        )
    return _ensemble_service


def get_model_repository() -> ModelRepository:
    """Dependency injection for model repository."""
    if _model_repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model repository not initialized"
        )
    return _model_repository


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    current_time = datetime.utcnow()
    uptime = None
    
    if _app_start_time:
        uptime = (current_time - _app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        service="anomaly-detection-api",
        version="0.3.0",
        timestamp=current_time.isoformat(),
        uptime_seconds=uptime
    )


@app.post("/api/v1/detect", response_model=DetectionResponse)
@async_log_decorator(operation="api_detect_anomalies", log_args=False, log_duration=True)
async def detect_anomalies(
    request: DetectionRequest,
    http_request: Request,
    detection_service: DetectionService = Depends(get_detection_service)
) -> DetectionResponse:
    """Detect anomalies in dataset using specified algorithm."""
    start_time = datetime.utcnow()
    
    logger.info("Processing detection request", 
                algorithm=request.algorithm, 
                samples=len(request.data),
                contamination=request.contamination)
    
    try:
        # Validate input data
        if not request.data or not request.data[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data cannot be empty"
            )
        
        # Convert to numpy array
        data_array = np.array(request.data, dtype=np.float64)
        
        # Map algorithm names
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'local_outlier_factor': 'lof',
            'lof': 'lof'
        }
        
        algorithm_name = algorithm_map.get(request.algorithm, request.algorithm)
        
        # Run detection
        result = detection_service.detect_anomalies(
            data=data_array,
            algorithm=algorithm_name,
            contamination=request.contamination,
            **request.parameters
        )
        
        end_time = datetime.utcnow()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return DetectionResponse(
            success=result.success,
            anomalies=result.anomalies,
            scores=result.confidence_scores.tolist() if result.confidence_scores is not None else None,
            algorithm=request.algorithm,
            total_samples=result.total_samples,
            anomalies_detected=result.anomaly_count,
            anomaly_rate=result.anomaly_rate,
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time_ms
        )
        
    except ValueError as e:
        logger.error("Detection validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error("Detection processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/api/v1/ensemble", response_model=DetectionResponse)
async def ensemble_detect(
    request: EnsembleRequest,
    ensemble_service: EnsembleService = Depends(get_ensemble_service)
) -> DetectionResponse:
    """Run ensemble anomaly detection using multiple algorithms."""
    start_time = datetime.utcnow()
    
    logger.info("Processing ensemble detection request",
                algorithms=request.algorithms,
                method=request.method,
                samples=len(request.data))
    
    try:
        # Validate input
        if not request.data or not request.data[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data cannot be empty"
            )
        
        if len(request.algorithms) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ensemble requires at least 2 algorithms"
            )
        
        # Convert to numpy array
        data_array = np.array(request.data, dtype=np.float64)
        
        # Map algorithm names
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'local_outlier_factor': 'lof',
            'lof': 'lof'
        }
        
        mapped_algorithms = [algorithm_map.get(alg, alg) for alg in request.algorithms]
        
        # Get individual results
        individual_results = []
        for algorithm in mapped_algorithms:
            detection_service = DetectionService()
            result = detection_service.detect_anomalies(
                data=data_array,
                algorithm=algorithm,
                contamination=request.contamination,
                **request.parameters.get(algorithm, {})
            )
            individual_results.append(result)
        
        # Combine using ensemble method
        predictions_array = np.array([result.predictions for result in individual_results])
        scores_array = np.array([result.confidence_scores for result in individual_results if result.confidence_scores is not None])
        
        if request.method == 'majority':
            ensemble_predictions = ensemble_service.majority_vote(predictions_array)
            ensemble_scores = None
        elif request.method in ['average', 'weighted_average', 'max'] and len(scores_array) > 0:
            if request.method == 'average':
                ensemble_predictions, ensemble_scores = ensemble_service.average_combination(predictions_array, scores_array)
            elif request.method == 'max':
                ensemble_predictions, ensemble_scores = ensemble_service.max_combination(predictions_array, scores_array)
            else:  # weighted_average
                weights = np.ones(len(request.algorithms)) / len(request.algorithms)
                ensemble_predictions, ensemble_scores = ensemble_service.weighted_combination(
                    predictions_array, scores_array, weights
                )
        else:
            # Fallback to majority vote
            ensemble_predictions = ensemble_service.majority_vote(predictions_array)
            ensemble_scores = None
        
        # Calculate ensemble statistics
        anomaly_count = int(np.sum(ensemble_predictions == -1))
        total_samples = len(ensemble_predictions)
        anomaly_rate = anomaly_count / total_samples if total_samples > 0 else 0.0
        anomaly_indices = np.where(ensemble_predictions == -1)[0].tolist()
        
        end_time = datetime.utcnow()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return DetectionResponse(
            success=True,
            anomalies=anomaly_indices,
            scores=ensemble_scores.tolist() if ensemble_scores is not None else None,
            algorithm=f"ensemble_{request.method}",
            total_samples=total_samples,
            anomalies_detected=anomaly_count,
            anomaly_rate=anomaly_rate,
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time_ms
        )
        
    except ValueError as e:
        logger.error("Ensemble validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error("Ensemble processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ensemble detection failed: {str(e)}"
        )


@app.post("/api/v1/predict", response_model=DetectionResponse)
async def predict_with_model(
    request: ModelPredictionRequest,
    model_repository: ModelRepository = Depends(get_model_repository)
) -> DetectionResponse:
    """Make predictions using a saved model."""
    start_time = datetime.utcnow()
    
    logger.info("Processing prediction request with saved model",
                model_id=request.model_id,
                samples=len(request.data))
    
    try:
        # Validate input
        if not request.data or not request.data[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data cannot be empty"
            )
        
        # Load model
        try:
            model = model_repository.load(request.model_id)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID '{request.model_id}' not found"
            )
        
        # Convert to numpy array
        data_array = np.array(request.data, dtype=np.float64)
        
        # Make predictions
        predictions = model.predict(data_array)
        
        try:
            scores = model.get_anomaly_scores(data_array)
        except:
            scores = None
        
        # Calculate statistics
        anomaly_count = int(np.sum(predictions == -1))
        total_samples = len(predictions)
        anomaly_rate = anomaly_count / total_samples if total_samples > 0 else 0.0
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        
        end_time = datetime.utcnow()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return DetectionResponse(
            success=True,
            anomalies=anomaly_indices,
            scores=scores.tolist() if scores is not None else None,
            algorithm=model.metadata.algorithm,
            total_samples=total_samples,
            anomalies_detected=anomaly_count,
            anomaly_rate=anomaly_rate,
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model prediction error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/api/v1/algorithms")
async def list_algorithms() -> Dict[str, List[str]]:
    """List available detection algorithms and ensemble methods."""
    return {
        "single_algorithms": [
            "isolation_forest",
            "one_class_svm", 
            "local_outlier_factor",
            "lof"
        ],
        "ensemble_methods": [
            "majority",
            "average", 
            "weighted_average",
            "max"
        ],
        "supported_formats": [
            "json",
            "csv"
        ]
    }


@app.get("/api/v1/models", response_model=ModelListResponse)
async def list_models(
    algorithm: Optional[str] = None,
    status: Optional[str] = None,
    model_repository: ModelRepository = Depends(get_model_repository)
) -> ModelListResponse:
    """List available trained models with optional filtering."""
    try:
        try:
            from data.processing.domain.entities.model import ModelStatus
        except ImportError:
            from .domain.entities.model import ModelStatus
        
        status_filter = None
        if status:
            try:
                status_filter = ModelStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status}"
                )
        
        models = model_repository.list_models(
            status=status_filter,
            algorithm=algorithm
        )
        
        return ModelListResponse(
            models=models,
            total_count=len(models)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error listing models", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.get("/api/v1/models/{model_id}")
async def get_model_info(
    model_id: str,
    model_repository: ModelRepository = Depends(get_model_repository)
) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        metadata = model_repository.get_model_metadata(model_id)
        return metadata
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID '{model_id}' not found"
        )
    except Exception as e:
        logger.error("Error getting model info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


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


@app.get("/api/v1/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics and statistics."""
    try:
        return {
            "metrics_summary": metrics_collector.get_summary_stats(),
            "performance_summary": performance_monitor.get_performance_summary(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@app.get("/api/v1/metrics/performance/{operation}")
async def get_operation_performance(operation: str) -> Dict[str, Any]:
    """Get performance metrics for a specific operation."""
    try:
        stats = performance_monitor.get_operation_stats(operation)
        recent_profiles = performance_monitor.get_recent_profiles(
            operation=operation,
            limit=10
        )
        
        return {
            "operation": operation,
            "statistics": stats,
            "recent_profiles": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "duration_ms": p.total_duration_ms,
                    "success": p.success,
                    "memory_mb": p.memory_usage_mb,
                    "peak_memory_mb": p.peak_memory_mb
                }
                for p in recent_profiles
            ]
        }
    except Exception as e:
        logger.error("Failed to get operation performance", 
                    operation=operation, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )


@app.get("/api/v1/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with all components."""
    try:
        # Run all health checks
        await health_checker.run_all_checks(force=True)
        
        # Get detailed health summary
        health_summary = health_checker.get_health_summary()
        
        return {
            **health_summary,
            "service_info": {
                "name": "anomaly-detection-api",
                "version": "0.3.0",
                "environment": settings.environment
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "overall_status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/api/v1/health/readiness")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for Kubernetes/container orchestration."""
    try:
        # Check critical components only
        critical_checks = ["algorithms"]
        
        results = {}
        overall_ready = True
        
        for check_name in critical_checks:
            result = await health_checker.run_check(check_name)
            if result:
                results[check_name] = {
                    "status": result.status.value,
                    "message": result.message
                }
                if result.status.value not in ["healthy"]:
                    overall_ready = False
            else:
                results[check_name] = {
                    "status": "unknown",
                    "message": "Check not found"
                }
                overall_ready = False
        
        status_code = status.HTTP_200_OK if overall_ready else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content={
                "ready": overall_ready,
                "checks": results,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ready": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/api/v1/health/liveness")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check for Kubernetes/container orchestration."""
    try:
        # Basic liveness - just check if the service is responding
        return {
            "alive": True,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - _app_start_time).total_seconds() if _app_start_time else None
        }
    except Exception as e:
        logger.error("Liveness check failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "alive": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/api/v1/monitoring/resources")
async def get_resource_usage() -> Dict[str, Any]:
    """Get system resource usage information."""
    try:
        recent_usage = performance_monitor.get_resource_usage(
            since=datetime.utcnow() - timedelta(hours=1),
            limit=60  # Last hour of data
        )
        
        return {
            "resource_usage": [
                {
                    "timestamp": usage.timestamp.isoformat(),
                    "cpu_percent": usage.cpu_percent,
                    "memory_mb": usage.memory_mb,
                    "memory_percent": usage.memory_percent,
                    "disk_io_read_mb": usage.disk_io_read_mb,
                    "disk_io_write_mb": usage.disk_io_write_mb,
                    "network_sent_mb": usage.network_sent_mb,
                    "network_received_mb": usage.network_received_mb
                }
                for usage in recent_usage
            ],
            "count": len(recent_usage),
            "period_hours": 1
        }
    except Exception as e:
        logger.error("Failed to get resource usage", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve resource usage: {str(e)}"
        )


@app.get("/api/v1/dashboard/summary")
async def get_dashboard_summary() -> Dict[str, Any]:
    """Get comprehensive dashboard summary."""
    try:
        summary = await monitoring_dashboard.get_dashboard_summary()
        
        return {
            "summary": {
                "overall_health_status": summary.overall_health_status,
                "healthy_checks": summary.healthy_checks,
                "degraded_checks": summary.degraded_checks,
                "unhealthy_checks": summary.unhealthy_checks,
                "total_operations": summary.total_operations,
                "operations_last_hour": summary.operations_last_hour,
                "avg_response_time_ms": summary.avg_response_time_ms,
                "success_rate": summary.success_rate,
                "current_memory_mb": summary.current_memory_mb,
                "current_cpu_percent": summary.current_cpu_percent,
                "peak_memory_mb": summary.peak_memory_mb,
                "total_models": summary.total_models,
                "active_detections": summary.active_detections,
                "anomalies_detected_today": summary.anomalies_detected_today,
                "active_alerts": summary.active_alerts,
                "recent_errors": summary.recent_errors,
                "slow_operations": summary.slow_operations,
                "generated_at": summary.generated_at.isoformat()
            }
        }
    except Exception as e:
        logger.error("Failed to get dashboard summary", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dashboard summary: {str(e)}"
        )


@app.get("/api/v1/dashboard/trends")
async def get_performance_trends(
    hours: int = 24
) -> Dict[str, Any]:
    """Get performance trends over time."""
    try:
        if hours < 1 or hours > 168:  # Limit to 1 hour - 1 week
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hours must be between 1 and 168"
            )
        
        trends = monitoring_dashboard.get_performance_trends(hours)
        return trends
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get performance trends", 
                    hours=hours, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance trends: {str(e)}"
        )


@app.get("/api/v1/dashboard/alerts")
async def get_alerts() -> Dict[str, Any]:
    """Get current alerts and issues."""
    try:
        alerts = monitoring_dashboard.get_alert_summary()
        return alerts
    except Exception as e:
        logger.error("Failed to get alerts", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve alerts: {str(e)}"
        )


@app.get("/api/v1/dashboard/operations")
async def get_operation_breakdown() -> Dict[str, Any]:
    """Get breakdown of operations by type and performance."""
    try:
        breakdown = monitoring_dashboard.get_operation_breakdown()
        return breakdown
    except Exception as e:
        logger.error("Failed to get operation breakdown", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve operation breakdown: {str(e)}"
        )


def main() -> None:
    """Run the server."""
    uvicorn_run(
        "anomaly_detection.server:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=1 if settings.debug else settings.api.workers,
        reload=settings.api.reload or settings.debug,
        log_level=settings.logging.level.lower()
    )


if __name__ == "__main__":
    main()