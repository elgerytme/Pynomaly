#!/usr/bin/env python3
"""
Simple FastAPI application for production deployment testing.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import analytics dashboard
try:
    from .analytics_dashboard import router as analytics_router
except ImportError:
    # Handle case where analytics module is not available
    analytics_router = None
    logger.warning("Analytics dashboard not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Request/Response Models
class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    environment: str = "production"
    uptime_seconds: float | None = None


class DetectionRequest(BaseModel):
    """Anomaly detection request model."""

    data: list[dict[str, Any]] = Field(..., description="Data points to analyze")
    model_id: str | None = Field(None, description="Model ID to use")
    threshold: float | None = Field(None, description="Detection threshold")


class DetectionResponse(BaseModel):
    """Anomaly detection response model."""

    detection_id: str = Field(..., description="Unique detection ID")
    anomalies_detected: int = Field(..., description="Number of anomalies found")
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    model_used: str = Field(default="default", description="Model used for detection")


class MetricsResponse(BaseModel):
    """Metrics response model."""

    total_requests: int = 0
    total_detections: int = 0
    average_processing_time_ms: float = 0.0
    system_health: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)


# Global metrics storage
app_metrics = {
    "start_time": datetime.now(),
    "total_requests": 0,
    "total_detections": 0,
    "processing_times": [],
    "errors": 0,
}


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="Pynomaly API",
        description="Production Anomaly Detection API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify actual domains
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include analytics dashboard router
    if analytics_router:
        app.include_router(analytics_router)
        logger.info("Analytics dashboard included")

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Middleware to track request metrics."""
        start_time = datetime.now()

        # Process request
        response = await call_next(request)

        # Update metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        app_metrics["total_requests"] += 1
        app_metrics["processing_times"].append(processing_time)

        # Keep only last 1000 processing times
        if len(app_metrics["processing_times"]) > 1000:
            app_metrics["processing_times"] = app_metrics["processing_times"][-1000:]

        # Add response headers
        response.headers["X-Processing-Time"] = f"{processing_time:.2f}ms"
        response.headers["X-Request-ID"] = str(uuid.uuid4())

        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        app_metrics["errors"] += 1
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        uptime = (datetime.now() - app_metrics["start_time"]).total_seconds()

        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            environment=os.getenv("ENVIRONMENT", "production"),
            uptime_seconds=uptime,
        )

    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with system metrics."""
        uptime = (datetime.now() - app_metrics["start_time"]).total_seconds()
        processing_times = app_metrics["processing_times"]

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "production"),
            "uptime_seconds": uptime,
            "metrics": {
                "total_requests": app_metrics["total_requests"],
                "total_detections": app_metrics["total_detections"],
                "total_errors": app_metrics["errors"],
                "average_processing_time_ms": (
                    sum(processing_times) / len(processing_times)
                    if processing_times
                    else 0
                ),
                "requests_per_second": app_metrics["total_requests"] / uptime
                if uptime > 0
                else 0,
            },
            "system": {
                "cpu_usage_percent": 45.2,  # Mock data
                "memory_usage_percent": 62.1,  # Mock data
                "disk_usage_percent": 34.5,  # Mock data
                "database_status": "connected",
                "redis_status": "connected",
            },
        }

    @app.post("/api/v1/detect", response_model=DetectionResponse)
    async def detect_anomalies(request: DetectionRequest):
        """Detect anomalies in provided data."""
        start_time = datetime.now()

        try:
            # Simulate processing delay
            await asyncio.sleep(0.1)

            # Simple anomaly detection logic (mock)
            anomalies_count = 0
            for data_point in request.data:
                # Simple threshold-based detection
                values = [v for v in data_point.values() if isinstance(v, (int, float))]
                if values:
                    avg_value = sum(values) / len(values)
                    if avg_value > 50:  # Simple threshold
                        anomalies_count += 1

            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            app_metrics["total_detections"] += 1

            return DetectionResponse(
                detection_id=str(uuid.uuid4()),
                anomalies_detected=anomalies_count,
                timestamp=datetime.now(),
                processing_time_ms=processing_time,
                model_used=request.model_id or "default",
            )

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    @app.get("/api/v1/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Get system metrics."""
        processing_times = app_metrics["processing_times"]

        return MetricsResponse(
            total_requests=app_metrics["total_requests"],
            total_detections=app_metrics["total_detections"],
            average_processing_time_ms=(
                sum(processing_times) / len(processing_times) if processing_times else 0
            ),
            system_health="healthy",
            timestamp=datetime.now(),
        )

    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        processing_times = app_metrics["processing_times"]
        avg_time = (
            sum(processing_times) / len(processing_times) if processing_times else 0
        )

        metrics = f"""# HELP pynomaly_requests_total Total number of requests
# TYPE pynomaly_requests_total counter
pynomaly_requests_total {app_metrics["total_requests"]}

# HELP pynomaly_detections_total Total number of detections
# TYPE pynomaly_detections_total counter
pynomaly_detections_total {app_metrics["total_detections"]}

# HELP pynomaly_errors_total Total number of errors
# TYPE pynomaly_errors_total counter
pynomaly_errors_total {app_metrics["errors"]}

# HELP pynomaly_processing_time_ms Average processing time in milliseconds
# TYPE pynomaly_processing_time_ms gauge
pynomaly_processing_time_ms {avg_time:.2f}

# HELP pynomaly_uptime_seconds Application uptime in seconds
# TYPE pynomaly_uptime_seconds gauge
pynomaly_uptime_seconds {(datetime.now() - app_metrics["start_time"]).total_seconds():.2f}
"""

        return JSONResponse(content=metrics, media_type="text/plain")

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Pynomaly API is running",
            "version": "1.0.0",
            "docs_url": "/docs",
            "health_url": "/health",
        }

    return app


# For direct execution
if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
