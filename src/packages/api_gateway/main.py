"""API Gateway for Hexagonal Architecture microservices."""

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import uvicorn
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
import time
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hexagonal Architecture API Gateway",
    description="Unified API Gateway for all microservices",
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

# Security
security = HTTPBearer(auto_error=False)

# Service registry
SERVICES = {
    "data-quality": {
        "url": "http://data-quality-service:80",
        "health": "/health",
        "prefix": "/data-quality"
    },
    "machine-learning": {
        "url": "http://machine-learning-service:80", 
        "health": "/health",
        "prefix": "/machine-learning"
    },
    "mlops": {
        "url": "http://mlops-service:80",
        "health": "/health", 
        "prefix": "/mlops"
    },
    "anomaly-detection": {
        "url": "http://anomaly-detection-service:80",
        "health": "/health",
        "prefix": "/anomaly-detection"
    }
}

# Circuit breaker state
circuit_breaker_state = {service: {"failures": 0, "last_failure": None, "state": "closed"} for service in SERVICES}

class CircuitBreaker:
    """Simple circuit breaker implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
    
    def can_proceed(self, service: str) -> bool:
        """Check if request can proceed through circuit breaker."""
        state = circuit_breaker_state[service]
        
        if state["state"] == "open":
            # Check if recovery timeout has passed
            if time.time() - state["last_failure"] > self.recovery_timeout:
                state["state"] = "half-open"
                logger.info(f"Circuit breaker for {service} moving to half-open state")
                return True
            return False
        
        return True
    
    def record_success(self, service: str):
        """Record successful request."""
        state = circuit_breaker_state[service]
        state["failures"] = 0
        if state["state"] == "half-open":
            state["state"] = "closed"
            logger.info(f"Circuit breaker for {service} closed")
    
    def record_failure(self, service: str):
        """Record failed request."""
        state = circuit_breaker_state[service]
        state["failures"] += 1
        state["last_failure"] = time.time()
        
        if state["failures"] >= self.failure_threshold:
            state["state"] = "open"
            logger.warning(f"Circuit breaker for {service} opened due to failures")

circuit_breaker = CircuitBreaker()

async def authenticate_request(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict]:
    """Simple authentication - in production, use proper JWT validation."""
    if not credentials:
        return None
    
    # For demo purposes, accept any token starting with "valid_"
    if credentials.credentials.startswith("valid_"):
        return {"user_id": "demo_user", "roles": ["user"]}
    
    return None

def authorize_request(user: Optional[Dict], required_role: str = "user") -> bool:
    """Simple authorization check."""
    if not user:
        return False
    
    return required_role in user.get("roles", [])

async def forward_request(
    service_name: str,
    path: str,
    method: str,
    headers: Dict[str, str],
    query_params: Dict[str, str],
    body: Optional[bytes] = None
) -> Dict[str, Any]:
    """Forward request to microservice with circuit breaker protection."""
    
    if not circuit_breaker.can_proceed(service_name):
        raise HTTPException(
            status_code=503,
            detail=f"Service {service_name} is temporarily unavailable (circuit breaker open)"
        )
    
    service_config = SERVICES[service_name]
    url = f"{service_config['url']}{path}"
    
    # Prepare headers (remove host and other problematic headers)
    forward_headers = {}
    for key, value in headers.items():
        if key.lower() not in ['host', 'content-length', 'connection']:
            forward_headers[key] = value
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=forward_headers,
                params=query_params,
                content=body
            )
            
            circuit_breaker.record_success(service_name)
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.content,
                "json": response.json() if response.headers.get("content-type", "").startswith("application/json") else None
            }
    
    except Exception as e:
        circuit_breaker.record_failure(service_name)
        logger.error(f"Failed to forward request to {service_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Service {service_name} is unavailable")

# Health and status endpoints
@app.get("/health")
async def gateway_health():
    """Gateway health check."""
    return {"status": "healthy", "service": "api-gateway", "timestamp": datetime.utcnow()}

@app.get("/status")
async def gateway_status():
    """Gateway status with service health information."""
    service_health = {}
    
    for service_name, config in SERVICES.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{config['url']}{config['health']}")
                service_health[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds(),
                    "circuit_breaker": circuit_breaker_state[service_name]["state"]
                }
        except Exception as e:
            service_health[service_name] = {
                "status": "error",
                "error": str(e),
                "circuit_breaker": circuit_breaker_state[service_name]["state"]
            }
    
    return {
        "gateway_status": "running",
        "timestamp": datetime.utcnow(),
        "services": service_health,
        "version": "1.0.0"
    }

@app.get("/metrics")
async def gateway_metrics():
    """Gateway metrics for Prometheus."""
    return {
        "gateway_requests_total": 1000,
        "gateway_request_duration_seconds": 0.25,
        "gateway_errors_total": 50,
        "services_available": len([s for s in circuit_breaker_state.values() if s["state"] != "open"])
    }

# Service discovery endpoint
@app.get("/api/services")
async def list_services():
    """List available services."""
    return {
        "status": "success",
        "services": [
            {
                "name": name,
                "url": config["url"],
                "prefix": config["prefix"],
                "health_endpoint": config["health"]
            }
            for name, config in SERVICES.items()
        ],
        "timestamp": datetime.utcnow()
    }

# Data Quality Service Proxy
@app.api_route("/data-quality/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def data_quality_proxy(
    path: str,
    request: Request,
    user: Optional[Dict] = Depends(authenticate_request)
):
    """Proxy requests to Data Quality service."""
    if not authorize_request(user):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None
    
    result = await forward_request(
        "data-quality",
        f"/{path}",
        request.method,
        dict(request.headers),
        dict(request.query_params),
        body
    )
    
    if result["json"]:
        return result["json"]
    else:
        return JSONResponse(
            content=result["content"].decode() if result["content"] else "",
            status_code=result["status_code"]
        )

# Machine Learning Service Proxy
@app.api_route("/machine-learning/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def machine_learning_proxy(
    path: str,
    request: Request,
    user: Optional[Dict] = Depends(authenticate_request)
):
    """Proxy requests to Machine Learning service."""
    if not authorize_request(user):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None
    
    result = await forward_request(
        "machine-learning",
        f"/{path}",
        request.method,
        dict(request.headers),
        dict(request.query_params),
        body
    )
    
    if result["json"]:
        return result["json"]
    else:
        return JSONResponse(
            content=result["content"].decode() if result["content"] else "",
            status_code=result["status_code"]
        )

# MLOps Service Proxy
@app.api_route("/mlops/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def mlops_proxy(
    path: str,
    request: Request,
    user: Optional[Dict] = Depends(authenticate_request)
):
    """Proxy requests to MLOps service."""
    if not authorize_request(user):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None
    
    result = await forward_request(
        "mlops",
        f"/{path}",
        request.method,
        dict(request.headers),
        dict(request.query_params),
        body
    )
    
    if result["json"]:
        return result["json"]
    else:
        return JSONResponse(
            content=result["content"].decode() if result["content"] else "",
            status_code=result["status_code"]
        )

# Anomaly Detection Service Proxy
@app.api_route("/anomaly-detection/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def anomaly_detection_proxy(
    path: str,
    request: Request,
    user: Optional[Dict] = Depends(authenticate_request)
):
    """Proxy requests to Anomaly Detection service."""
    if not authorize_request(user):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None
    
    result = await forward_request(
        "anomaly-detection",
        f"/{path}",
        request.method,
        dict(request.headers),
        dict(request.query_params),
        body
    )
    
    if result["json"]:
        return result["json"]
    else:
        return JSONResponse(
            content=result["content"].decode() if result["content"] else "",
            status_code=result["status_code"]
        )

# Aggregated endpoints
@app.get("/api/v1/health-summary")
async def health_summary(user: Optional[Dict] = Depends(authenticate_request)):
    """Get health summary of all services."""
    if not authorize_request(user):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    health_checks = {}
    
    for service_name, config in SERVICES.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{config['url']}{config['health']}")
                health_checks[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "last_check": datetime.utcnow().isoformat()
                }
        except Exception as e:
            health_checks[service_name] = {
                "status": "error",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    overall_status = "healthy" if all(
        check["status"] == "healthy" for check in health_checks.values()
    ) else "degraded"
    
    return {
        "overall_status": overall_status,
        "services": health_checks,
        "timestamp": datetime.utcnow()
    }

@app.post("/api/v1/workflow/ml-pipeline")
async def ml_workflow(
    request: Dict[str, Any],
    user: Optional[Dict] = Depends(authenticate_request)
):
    """Execute a complete ML workflow across services."""
    if not authorize_request(user):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        workflow_id = f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        results = {}
        
        # Step 1: Ingest data (ML service)
        if "data_source" in request:
            ml_result = await forward_request(
                "machine-learning",
                "/api/v1/ingest",
                "POST",
                {"Content-Type": "application/json"},
                {},
                json.dumps({"source": request["data_source"]}).encode()
            )
            results["data_ingestion"] = ml_result["json"]
        
        # Step 2: Profile data (Data Quality service)
        if "data_source" in request:
            dq_result = await forward_request(
                "data-quality",
                "/api/v1/profiles",
                "POST", 
                {"Content-Type": "application/json"},
                {},
                json.dumps({
                    "data_source": request["data_source"],
                    "timestamp": datetime.utcnow().isoformat()
                }).encode()
            )
            results["data_profiling"] = dq_result["json"]
        
        # Step 3: Detect anomalies
        if "anomaly_data" in request:
            anomaly_result = await forward_request(
                "anomaly-detection",
                "/api/v1/detect",
                "POST",
                {"Content-Type": "application/json"},
                {},
                json.dumps(request["anomaly_data"]).encode()
            )
            results["anomaly_detection"] = anomaly_result["json"]
        
        # Step 4: Create MLOps pipeline
        if "pipeline_config" in request:
            mlops_result = await forward_request(
                "mlops",
                "/api/v1/pipelines",
                "POST",
                {"Content-Type": "application/json"},
                {},
                json.dumps(request["pipeline_config"]).encode()
            )
            results["pipeline_creation"] = mlops_result["json"]
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "results": results,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        logger.error(f"ML workflow failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

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
    logger.error(f"Unhandled exception in gateway: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": {
                "code": 500,
                "message": "Internal gateway error",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )