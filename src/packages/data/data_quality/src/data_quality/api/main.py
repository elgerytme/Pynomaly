"""FastAPI application for Data Quality service."""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import List, Dict, Any
from datetime import datetime

from data_quality.infrastructure.container.container import DataQualityContainer
from data_quality.domain.entities.data_profiling_request import DataProfilingRequest
from data_quality.domain.entities.data_quality_rule import DataQualityRule
from data_quality.domain.interfaces.data_processing_operations import (
    DataProfilingPort, DataValidationPort, StatisticalAnalysisPort
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Quality Service",
    description="Hexagonal Architecture Data Quality API",
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
container = DataQualityContainer()

# Dependency injection
def get_data_profiling_service() -> DataProfilingPort:
    return container.get(DataProfilingPort)

def get_data_validation_service() -> DataValidationPort:
    return container.get(DataValidationPort)

def get_statistical_analysis_service() -> StatisticalAnalysisPort:
    return container.get(StatisticalAnalysisPort)

# Health and readiness endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "data-quality", "timestamp": datetime.utcnow()}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Perform basic service checks
        profiling_service = get_data_profiling_service()
        return {"status": "ready", "service": "data-quality", "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Basic metrics - in production, use prometheus_client library
    return {
        "data_quality_checks_total": 100,
        "data_quality_checks_failed_total": 5,
        "data_profiles_created_total": 50,
        "statistical_analyses_total": 25
    }

# API Endpoints
@app.post("/api/v1/profiles", status_code=status.HTTP_201_CREATED)
async def create_data_profile(
    request: Dict[str, Any],
    profiling_service: DataProfilingPort = Depends(get_data_profiling_service)
):
    """Create a data profile for the specified data source."""
    try:
        # Validate request
        if "data_source" not in request:
            raise HTTPException(status_code=400, detail="data_source is required")
        
        profiling_request = DataProfilingRequest(
            data_source=request["data_source"],
            timestamp=request.get("timestamp", datetime.utcnow().isoformat())
        )
        
        # Create profile
        profile = await profiling_service.create_data_profile(profiling_request)
        
        return {
            "status": "success",
            "data": {
                "data_source": profile.data_source,
                "total_rows": profile.total_rows,
                "profile_timestamp": profile.profile_timestamp,
                "column_count": len(profile.column_profiles),
                "columns": list(profile.column_profiles.keys())
            }
        }
    except Exception as e:
        logger.error(f"Failed to create data profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/profiles/{data_source}")
async def get_data_profile(
    data_source: str,
    profiling_service: DataProfilingPort = Depends(get_data_profiling_service)
):
    """Get existing data profile for a data source."""
    try:
        # For this example, we'll create a new profile
        # In production, you'd retrieve from storage
        profiling_request = DataProfilingRequest(
            data_source=data_source,
            timestamp=datetime.utcnow().isoformat()
        )
        
        profile = await profiling_service.create_data_profile(profiling_request)
        
        return {
            "status": "success",
            "data": {
                "data_source": profile.data_source,
                "total_rows": profile.total_rows,
                "profile_timestamp": profile.profile_timestamp,
                "column_profiles": {
                    name: {
                        "column_name": col.column_name,
                        "data_type": col.data_type,
                        "statistics": {
                            "mean": col.statistics.mean,
                            "median": col.statistics.median,
                            "std_dev": col.statistics.std_dev,
                            "min_value": col.statistics.min_value,
                            "max_value": col.statistics.max_value,
                            "null_count": col.statistics.null_count,
                            "unique_count": col.statistics.unique_count
                        }
                    }
                    for name, col in profile.column_profiles.items()
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to get data profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/validate")
async def validate_data(
    request: Dict[str, Any],
    validation_service: DataValidationPort = Depends(get_data_validation_service)
):
    """Validate data against quality rules."""
    try:
        # Validate request
        if "data_source" not in request or "rules" not in request:
            raise HTTPException(status_code=400, detail="data_source and rules are required")
        
        # Convert rules to domain objects
        rules = []
        for rule_data in request["rules"]:
            rule = DataQualityRule(
                rule_name=rule_data["rule_name"],
                description=rule_data.get("description", "")
            )
            rules.append(rule)
        
        # Perform validation
        validation_results = await validation_service.validate_data(
            request["data_source"], rules
        )
        
        return {
            "status": "success",
            "data": {
                "data_source": request["data_source"],
                "validation_results": [
                    {
                        "rule_name": result.rule_name,
                        "passed": result.passed,
                        "message": result.message,
                        "details": result.details
                    }
                    for result in validation_results
                ],
                "summary": {
                    "total_rules": len(validation_results),
                    "passed": sum(1 for r in validation_results if r.passed),
                    "failed": sum(1 for r in validation_results if not r.passed)
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to validate data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze")
async def analyze_data(
    request: Dict[str, Any],
    analysis_service: StatisticalAnalysisPort = Depends(get_statistical_analysis_service)
):
    """Perform statistical analysis on data."""
    try:
        # Validate request
        if "data_source" not in request:
            raise HTTPException(status_code=400, detail="data_source is required")
        
        # Perform analysis
        analysis_result = await analysis_service.analyze_data(request["data_source"])
        
        return {
            "status": "success",
            "data": {
                "data_source": request["data_source"],
                "correlations": analysis_result.correlations,
                "distributions": analysis_result.distributions,
                "outliers": analysis_result.outliers,
                "summary_statistics": analysis_result.summary_statistics
            }
        }
    except Exception as e:
        logger.error(f"Failed to analyze data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def get_service_status():
    """Get service status and configuration."""
    return {
        "status": "running",
        "service": "data-quality",
        "version": "1.0.0",
        "environment": "development",
        "timestamp": datetime.utcnow(),
        "capabilities": [
            "data_profiling",
            "data_validation", 
            "statistical_analysis"
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
        port=8000,
        reload=True,
        log_level="info"
    )