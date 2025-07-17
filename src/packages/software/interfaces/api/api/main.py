#!/usr/bin/env python3
"""
Main API Application with Comprehensive Documentation.
This module creates the main FastAPI application with all endpoints and documentation.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ...enterprise.enterprise_service import router as enterprise_router
from ...infrastructure.config import get_settings
from ...infrastructure.security.rate_limiting_middleware import RateLimitMiddleware
from ...infrastructure.security.security_headers import SecurityHeadersMiddleware
from ...infrastructure.security.waf_middleware import WAFMiddleware
from ...mlops.mlops_service import router as mlops_router
from .docs import COMMON_RESPONSES, ENDPOINT_METADATA, configure_api_docs
from .endpoints import waf_management

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("🚀 Starting Software API server...")

    # Startup
    try:
        # Initialize services
        logger.info("✅ Initializing services...")

        # Initialize database connections
        logger.info("✅ Database connections initialized")

        # Initialize processor registry
        logger.info("✅ Processor registry initialized")

        # Initialize monitoring
        logger.info("✅ Monitoring systems initialized")

        logger.info("🎉 Software API server started successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        raise

    yield

    # Shutdown
    logger.info("⏹️ Shutting down Software API server...")
    logger.info("✅ Software API server stopped")


# Create FastAPI application
app = FastAPI(
    title="Software API",
    description="Enterprise Anomaly Processing Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure API documentation
configure_api_docs(app)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add security middleware
settings = get_settings()
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(WAFMiddleware, settings=settings)
app.add_middleware(RateLimitMiddleware, settings=settings)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


# Authentication dependency
async def get_current_user_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    """Get current user from JWT token."""
    try:
        # This would integrate with your authentication system
        # For now, return a mock user for demonstration
        return {"user_id": "demo_user", "tenant_id": "demo_tenant", "role": "admin"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )


# Health check endpoint
@app.get(
    "/health",
    summary="System health check",
    description=ENDPOINT_METADATA["health_check"]["description"],
    tags=ENDPOINT_METADATA["health_check"]["tags"],
    responses={
        200: {
            "description": "System is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "version": "1.0.0",
                        "services": {
                            "database": "healthy",
                            "cache": "healthy",
                            "processor_registry": "healthy",
                            "monitoring": "healthy",
                        },
                    }
                }
            },
        },
        **COMMON_RESPONSES,
    },
)
async def health_check():
    """Get system health status."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T12:00:00Z",
        "version": "1.0.0",
        "services": {
            "database": "healthy",
            "cache": "healthy",
            "processor_registry": "healthy",
            "monitoring": "healthy",
        },
    }


# Root endpoint
@app.get(
    "/",
    summary="API information",
    description="Get basic information about the Software API",
    tags=["Information"],
    responses={
        200: {
            "description": "API information",
            "content": {
                "application/json": {
                    "example": {
                        "name": "Software API",
                        "version": "1.0.0",
                        "description": "Enterprise Anomaly Processing Platform",
                        "documentation": "/docs",
                        "health": "/health",
                    }
                }
            },
        }
    },
)
async def root():
    """Get API information."""
    return {
        "name": "Software API",
        "version": "1.0.0",
        "description": "Enterprise Anomaly Processing Platform",
        "documentation": "/docs",
        "health": "/health",
    }


# Anomaly processing endpoints
@app.post(
    "/detect",
    summary="Detect anomalies in data",
    description=ENDPOINT_METADATA["detect_anomalies"]["description"],
    tags=ENDPOINT_METADATA["detect_anomalies"]["tags"],
    responses={
        200: {
            "description": "Anomaly processing results",
            "content": {
                "application/json": {
                    "example": {
                        "anomalies": [3],
                        "scores": [0.1, 0.2, 0.15, 0.95, 0.18, 0.12],
                        "threshold": 0.5,
                        "processor_id": "isolation_forest_20240101_001",
                        "processing_time_ms": 45.6,
                        "metadata": {
                            "algorithm": "isolation_forest",
                            "contamination": 0.1,
                            "n_estimators": 100,
                        },
                    }
                }
            },
        },
        **COMMON_RESPONSES,
    },
)
async def detect_anomalies(
    request: dict[str, Any],
    current_user: dict[str, Any] = Depends(get_current_user_token),
):
    """Detect anomalies in the provided data."""
    try:
        # Mock response for demonstration
        return {
            "anomalies": [3],
            "scores": [0.1, 0.2, 0.15, 0.95, 0.18, 0.12],
            "threshold": 0.5,
            "processor_id": "isolation_forest_20240101_001",
            "processing_time_ms": 45.6,
            "metadata": {
                "algorithm": request.get("algorithm", "isolation_forest"),
                "contamination": request.get("parameters", {}).get(
                    "contamination", 0.1
                ),
                "user_id": current_user["user_id"],
            },
        }
    except Exception as e:
        logger.error(f"Anomaly processing failed: {e}")
        raise HTTPException(status_code=500, detail="Anomaly processing failed")


# Processor management endpoints
@app.post(
    "/models/train",
    summary="Train a new anomaly processing processor",
    description=ENDPOINT_METADATA["train_processor"]["description"],
    tags=ENDPOINT_METADATA["train_processor"]["tags"],
    responses={
        201: {
            "description": "Processor trained successfully",
            "content": {
                "application/json": {
                    "example": {
                        "processor_id": "isolation_forest_20240101_001",
                        "status": "trained",
                        "measurements": {
                            "accuracy": 0.95,
                            "precision": 0.92,
                            "recall": 0.89,
                            "f1_score": 0.90,
                        },
                        "training_time_ms": 5643.2,
                        "created_at": "2024-01-01T12:00:00Z",
                    }
                }
            },
        },
        **COMMON_RESPONSES,
    },
)
async def train_processor(
    request: dict[str, Any],
    current_user: dict[str, Any] = Depends(get_current_user_token),
):
    """Train a new anomaly processing processor."""
    try:
        # Mock response for demonstration
        return {
            "processor_id": "isolation_forest_20240101_001",
            "status": "trained",
            "measurements": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.89,
                "f1_score": 0.90,
            },
            "training_time_ms": 5643.2,
            "created_at": "2024-01-01T12:00:00Z",
        }
    except Exception as e:
        logger.error(f"Processor training failed: {e}")
        raise HTTPException(status_code=500, detail="Processor training failed")


@app.get(
    "/models/{processor_id}",
    summary="Get processor information",
    description=ENDPOINT_METADATA["get_processor_info"]["description"],
    tags=ENDPOINT_METADATA["get_processor_info"]["tags"],
    responses={
        200: {
            "description": "Processor information",
            "content": {
                "application/json": {
                    "example": {
                        "processor_id": "isolation_forest_20240101_001",
                        "name": "Production Anomaly Detector",
                        "version": "1.0.0",
                        "type": "isolation_forest",
                        "status": "active",
                        "author": "data_scientist@company.com",
                        "created_at": "2024-01-01T12:00:00Z",
                        "measurements": {
                            "accuracy": 0.95,
                            "precision": 0.92,
                            "recall": 0.89,
                        },
                        "deployments": {
                            "production": {
                                "status": "active",
                                "deployed_at": "2024-01-01T14:00:00Z",
                                "endpoint": "https://api.example.com/models/isolation_forest_20240101_001/predict",
                            }
                        },
                    }
                }
            },
        },
        **COMMON_RESPONSES,
    },
)
async def get_processor_info(
    processor_id: str, current_user: dict[str, Any] = Depends(get_current_user_token)
):
    """Get detailed information about a specific processor."""
    try:
        # Mock response for demonstration
        return {
            "processor_id": processor_id,
            "name": "Production Anomaly Detector",
            "version": "1.0.0",
            "type": "isolation_forest",
            "status": "active",
            "author": "data_scientist@company.com",
            "created_at": "2024-01-01T12:00:00Z",
            "measurements": {"accuracy": 0.95, "precision": 0.92, "recall": 0.89},
            "deployments": {
                "production": {
                    "status": "active",
                    "deployed_at": "2024-01-01T14:00:00Z",
                    "endpoint": f"https://api.example.com/models/{processor_id}/predict",
                }
            },
        }
    except Exception as e:
        logger.error(f"Failed to get processor info: {e}")
        raise HTTPException(status_code=404, detail="Processor not found")


# Include enterprise router
app.include_router(
    enterprise_router,
    prefix="/enterprise",
    tags=["Enterprise"],
    dependencies=[Depends(get_current_user_token)],
)

# Include MLOps router
app.include_router(
    mlops_router,
    prefix="/mlops",
    tags=["MLOps"],
    dependencies=[Depends(get_current_user_token)],
)

# Include data profiling router
from .endpoints import data_profiling
app.include_router(
    data_profiling.router,
    dependencies=[Depends(get_current_user_token)]
)

# Include data quality router
from .endpoints import data_quality
app.include_router(
    data_quality.router,
    dependencies=[Depends(get_current_user_token)]
)

# Include ML pipelines router
from .endpoints import ml_pipelines
app.include_router(
    ml_pipelines.router,
    dependencies=[Depends(get_current_user_token)]
)

# Include WAF management router
app.include_router(
    waf_management.router, dependencies=[Depends(get_current_user_token)]
)

# Include pattern recognition router
from .endpoints import pattern_recognition
app.include_router(
    pattern_recognition.router,
    dependencies=[Depends(get_current_user_token)]
)

# Include schema discovery router
from .endpoints import schema_discovery
app.include_router(
    schema_discovery.router,
    dependencies=[Depends(get_current_user_token)]
)

# Include quality governance router
from .endpoints import quality_governance
app.include_router(
    quality_governance.router,
    dependencies=[Depends(get_current_user_token)]
)

# Include data visualization router
from .endpoints import data_visualization
app.include_router(
    data_visualization.router,
    dependencies=[Depends(get_current_user_token)]
)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
