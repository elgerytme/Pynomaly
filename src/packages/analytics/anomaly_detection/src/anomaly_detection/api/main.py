"""FastAPI application for Anomaly Detection service."""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from anomaly_detection.infrastructure.container.container import AnomalyDetectionContainer
from anomaly_detection.domain.entities.detection_request import DetectionRequest
from anomaly_detection.domain.entities.anomaly_alert import AnomalyAlert
from anomaly_detection.domain.interfaces.detection_operations import (
    AnomalyDetectionPort, AlertingPort, DataSourcePort
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Anomaly Detection Service",
    description="Hexagonal Architecture Anomaly Detection API",
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
container = AnomalyDetectionContainer()

# Dependency injection
def get_detection_service() -> AnomalyDetectionPort:
    return container.get(AnomalyDetectionPort)

def get_alerting_service() -> AlertingPort:
    return container.get(AlertingPort)

def get_data_source_service() -> DataSourcePort:
    return container.get(DataSourcePort)

# Health and readiness endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "anomaly-detection", "timestamp": datetime.utcnow()}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Perform basic service checks
        detection_service = get_detection_service()
        return {"status": "ready", "service": "anomaly-detection", "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Basic metrics - in production, use prometheus_client library
    return {
        "anomaly_detection_alerts_total": 25,
        "anomaly_detection_operations_total": 1000,
        "anomaly_detection_accuracy": 0.92,
        "data_sources_monitored": 8,
        "alerts_sent_total": 25
    }

# Anomaly Detection API
@app.post("/api/v1/detect", status_code=status.HTTP_200_OK)
async def detect_anomalies(
    request: Dict[str, Any],
    detection_service: AnomalyDetectionPort = Depends(get_detection_service)
):
    """Detect anomalies in the provided data."""
    try:
        # Validate request
        if "data_points" not in request:
            raise HTTPException(status_code=400, detail="data_points is required")
        
        # Create detection request
        detection_request = DetectionRequest(
            data_source="api_request",
            data_points=request["data_points"],
            threshold=request.get("threshold", 0.95),
            algorithm=request.get("algorithm", "statistical"),
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Perform detection
        detection_result = await detection_service.detect_anomalies(detection_request)
        
        return {
            "status": "success",
            "data": {
                "data_source": detection_result.data_source,
                "total_points": len(detection_request.data_points),
                "anomalies_detected": len(detection_result.anomalous_points),
                "anomalous_points": detection_result.anomalous_points,
                "confidence_scores": detection_result.confidence_scores,
                "threshold_used": detection_request.threshold,
                "algorithm_used": detection_request.algorithm,
                "timestamp": detection_result.timestamp
            }
        }
    except Exception as e:
        logger.error(f"Failed to detect anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/detect/batch")
async def detect_anomalies_batch(
    request: Dict[str, Any],
    detection_service: AnomalyDetectionPort = Depends(get_detection_service)
):
    """Detect anomalies in multiple data sources."""
    try:
        # Validate request
        if "data_sources" not in request:
            raise HTTPException(status_code=400, detail="data_sources is required")
        
        results = []
        for source_data in request["data_sources"]:
            detection_request = DetectionRequest(
                data_source=source_data.get("source_name", "unknown"),
                data_points=source_data["data_points"],
                threshold=source_data.get("threshold", 0.95),
                algorithm=source_data.get("algorithm", "statistical"),
                timestamp=datetime.utcnow().isoformat()
            )
            
            detection_result = await detection_service.detect_anomalies(detection_request)
            
            results.append({
                "data_source": detection_result.data_source,
                "total_points": len(detection_request.data_points),
                "anomalies_detected": len(detection_result.anomalous_points),
                "anomalous_points": detection_result.anomalous_points,
                "confidence_scores": detection_result.confidence_scores,
                "threshold_used": detection_request.threshold,
                "algorithm_used": detection_request.algorithm
            })
        
        return {
            "status": "success",
            "data": {
                "batch_results": results,
                "total_sources": len(results),
                "total_anomalies": sum(r["anomalies_detected"] for r in results),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to detect anomalies in batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/detect/stream/{data_source}")
async def detect_anomalies_stream(
    data_source: str,
    threshold: float = 0.95,
    algorithm: str = "statistical",
    detection_service: AnomalyDetectionPort = Depends(get_detection_service),
    data_source_service: DataSourcePort = Depends(get_data_source_service)
):
    """Detect anomalies in streaming data from a data source."""
    try:
        # Get data from source
        data_points = await data_source_service.get_data(data_source)
        
        # Create detection request
        detection_request = DetectionRequest(
            data_source=data_source,
            data_points=data_points,
            threshold=threshold,
            algorithm=algorithm,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Perform detection
        detection_result = await detection_service.detect_anomalies(detection_request)
        
        return {
            "status": "success",
            "data": {
                "data_source": detection_result.data_source,
                "total_points": len(data_points),
                "anomalies_detected": len(detection_result.anomalous_points),
                "anomalous_points": detection_result.anomalous_points,
                "confidence_scores": detection_result.confidence_scores,
                "threshold_used": threshold,
                "algorithm_used": algorithm,
                "timestamp": detection_result.timestamp
            }
        }
    except Exception as e:
        logger.error(f"Failed to detect anomalies in stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alerting API
@app.post("/api/v1/alerts")
async def create_alert(
    request: Dict[str, Any],
    alerting_service: AlertingPort = Depends(get_alerting_service)
):
    """Create an anomaly alert."""
    try:
        # Validate request
        required_fields = ["alert_type", "severity", "message"]
        for field in required_fields:
            if field not in request:
                raise HTTPException(status_code=400, detail=f"{field} is required")
        
        # Create alert
        alert = AnomalyAlert(
            alert_id=f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            alert_type=request["alert_type"],
            severity=request["severity"],
            message=request["message"],
            data_source=request.get("data_source", "unknown"),
            timestamp=datetime.utcnow().isoformat(),
            details=request.get("details", {})
        )
        
        # Send alert
        success = await alerting_service.send_alert(alert)
        
        return {
            "status": "success",
            "data": {
                "alert_id": alert.alert_id,
                "alert_sent": success,
                "timestamp": alert.timestamp
            }
        }
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/alerts")
async def get_alerts(
    limit: int = 50,
    severity: Optional[str] = None,
    alerting_service: AlertingPort = Depends(get_alerting_service)
):
    """Get recent alerts."""
    try:
        alerts = await alerting_service.get_alerts(limit, severity)
        
        return {
            "status": "success",
            "data": {
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "data_source": alert.data_source,
                        "timestamp": alert.timestamp,
                        "details": alert.details
                    }
                    for alert in alerts
                ],
                "total_count": len(alerts),
                "filters": {
                    "limit": limit,
                    "severity": severity
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/alerts/{alert_id}")
async def get_alert(
    alert_id: str,
    alerting_service: AlertingPort = Depends(get_alerting_service)
):
    """Get a specific alert by ID."""
    try:
        alert = await alerting_service.get_alert(alert_id)
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "status": "success",
            "data": {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "data_source": alert.data_source,
                "timestamp": alert.timestamp,
                "details": alert.details
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Source API
@app.get("/api/v1/sources")
async def list_data_sources(
    data_source_service: DataSourcePort = Depends(get_data_source_service)
):
    """List available data sources."""
    try:
        sources = await data_source_service.list_sources()
        
        return {
            "status": "success",
            "data": {
                "data_sources": sources,
                "total_count": len(sources),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to list data sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sources/{source_name}/data")
async def get_source_data(
    source_name: str,
    limit: int = 100,
    data_source_service: DataSourcePort = Depends(get_data_source_service)
):
    """Get data from a specific source."""
    try:
        data = await data_source_service.get_data(source_name, limit)
        
        return {
            "status": "success",
            "data": {
                "source_name": source_name,
                "data_points": data,
                "count": len(data),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get source data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/algorithms")
async def list_algorithms():
    """List available anomaly detection algorithms."""
    algorithms = [
        {
            "name": "statistical",
            "description": "Statistical outlier detection using z-scores",
            "parameters": ["threshold", "window_size"]
        },
        {
            "name": "isolation_forest",
            "description": "Isolation Forest algorithm for anomaly detection",
            "parameters": ["contamination", "n_estimators"]
        },
        {
            "name": "local_outlier_factor",
            "description": "Local Outlier Factor for anomaly detection",
            "parameters": ["n_neighbors", "contamination"]
        }
    ]
    
    return {
        "status": "success",
        "data": {
            "algorithms": algorithms,
            "total_count": len(algorithms),
            "timestamp": datetime.utcnow().isoformat()
        }
    }

@app.get("/api/v1/status")
async def get_service_status():
    """Get service status and configuration."""
    return {
        "status": "running",
        "service": "anomaly-detection",
        "version": "1.0.0",
        "environment": "development",
        "timestamp": datetime.utcnow(),
        "capabilities": [
            "anomaly_detection",
            "batch_detection",
            "stream_detection",
            "alerting",
            "data_source_integration"
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
        port=8003,
        reload=True,
        log_level="info"
    )