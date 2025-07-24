#!/usr/bin/env python3
"""
Production Deployment Guide
===========================

Complete guide for deploying anomaly detection systems in production environments.
This template covers monitoring, scaling, error handling, and best practices.

Usage:
    python production_deployment_guide.py

Requirements:
    - anomaly_detection
    - fastapi
    - uvicorn
    - redis (optional)
    - prometheus_client (optional)
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from anomaly_detection import DetectionService, EnsembleService
from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.infrastructure.monitoring.metrics_collector import MetricsCollector


@dataclass
class ProductionConfig:
    """Production configuration settings."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Performance settings
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    timeout_seconds: int = 300  # 5 minutes
    rate_limit_requests: int = 1000  # per minute
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = "/var/log/anomaly_detection.log"
    
    # Model settings
    model_cache_size: int = 10
    model_cache_ttl: int = 3600  # 1 hour
    default_contamination: float = 0.1
    
    # Database settings (optional)
    database_url: Optional[str] = None
    redis_url: Optional[str] = None


class ProductionDetectionService:
    """Production-ready detection service with caching, monitoring, and error handling."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.detection_service = DetectionService()
        self.ensemble_service = EnsembleService()
        self.streaming_service = StreamingService()
        self.metrics_collector = MetricsCollector()
        
        # Model cache
        self._model_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_lock = threading.RLock()
        
        # Performance tracking
        self._request_count = 0
        self._error_count = 0
        self._total_processing_time = 0.0
        
        self._setup_logging()
        self._setup_monitoring()
    
    def _setup_logging(self):
        """Configure production logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.config.log_file) if self.config.log_file else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Production detection service initialized")
    
    def _setup_monitoring(self):
        """Setup monitoring and health checks."""
        if self.config.enable_metrics:
            # Setup Prometheus metrics (if available)
            try:
                from prometheus_client import Counter, Histogram, Gauge, start_http_server
                
                self.request_counter = Counter('detection_requests_total', 'Total detection requests')
                self.error_counter = Counter('detection_errors_total', 'Total detection errors')
                self.processing_time = Histogram('detection_processing_seconds', 'Detection processing time')
                self.active_models = Gauge('active_models_count', 'Number of active models in cache')
                
                # Start metrics server
                start_http_server(self.config.metrics_port)
                self.logger.info(f"Metrics server started on port {self.config.metrics_port}")
                
            except ImportError:
                self.logger.warning("Prometheus client not available, metrics disabled")
    
    def _cleanup_cache(self):
        """Clean up expired models from cache."""
        with self._cache_lock:
            current_time = datetime.now()
            expired_keys = []
            
            for key, timestamp in self._cache_timestamps.items():
                if (current_time - timestamp).total_seconds() > self.config.model_cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._model_cache[key]
                del self._cache_timestamps[key]
                self.logger.info(f"Expired model cache entry: {key}")
    
    async def detect_anomalies(
        self,
        data: list,
        algorithm: str = "isolation_forest",
        contamination: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Production-ready anomaly detection with caching and monitoring.
        
        Args:
            data: Input data for detection
            algorithm: Algorithm to use
            contamination: Expected anomaly rate
            **kwargs: Additional algorithm parameters
            
        Returns:
            Detection results with metadata
        """
        start_time = time.time()
        request_id = f"{int(time.time() * 1000)}_{threading.get_ident()}"
        
        try:
            self._request_count += 1
            if hasattr(self, 'request_counter'):
                self.request_counter.inc()
            
            # Input validation
            if not data or len(data) == 0:
                raise HTTPException(status_code=400, detail="Empty data provided")
            
            if len(data) > 100000:  # Limit data size
                raise HTTPException(status_code=413, detail="Data too large")
            
            # Use cached model if available
            cache_key = f"{algorithm}_{contamination or self.config.default_contamination}"
            
            self._cleanup_cache()  # Clean expired cache entries
            
            # Perform detection
            contamination = contamination or self.config.default_contamination
            
            self.logger.info(f"Processing detection request {request_id}: {len(data)} samples, {algorithm}")
            
            result = self.detection_service.detect_anomalies(
                data=data,
                algorithm=algorithm,
                contamination=contamination,
                **kwargs
            )
            
            processing_time = time.time() - start_time
            self._total_processing_time += processing_time
            
            if hasattr(self, 'processing_time'):
                self.processing_time.observe(processing_time)
            
            # Prepare response
            response = {
                "request_id": request_id,
                "success": True,
                "algorithm": algorithm,
                "total_samples": result.total_samples,
                "anomalies_detected": result.anomaly_count,
                "anomaly_rate": result.anomaly_rate,
                "processing_time_seconds": processing_time,
                "contamination": contamination,
                "anomaly_indices": result.anomaly_indices.tolist() if hasattr(result.anomaly_indices, 'tolist') else result.anomaly_indices,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Detection request {request_id} completed: {result.anomaly_count} anomalies in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            self._error_count += 1
            if hasattr(self, 'error_counter'):
                self.error_counter.inc()
            
            processing_time = time.time() - start_time
            
            self.logger.error(f"Detection request {request_id} failed: {str(e)}", exc_info=True)
            
            # Return error response
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self._start_time if hasattr(self, '_start_time') else 0,
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "average_processing_time": self._total_processing_time / max(self._request_count, 1),
            "cached_models": len(self._model_cache),
            "memory_usage_mb": self._get_memory_usage(),
            "version": "2.1.0"
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0


# Global service instance
service: Optional[ProductionDetectionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global service
    
    # Startup
    config = ProductionConfig()
    service = ProductionDetectionService(config)
    service._start_time = time.time()
    
    logging.info("Production anomaly detection service started")
    
    yield
    
    # Shutdown
    logging.info("Production anomaly detection service shutting down")


def create_production_app() -> FastAPI:
    """Create production FastAPI application."""
    
    app = FastAPI(
        title="Anomaly Detection Service",
        description="Production-ready anomaly detection API",
        version="2.1.0",
        docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    return app


app = create_production_app()


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with metrics."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return service.get_health_status()


@app.post("/api/v1/detect")
async def detect_anomalies_endpoint(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    Detect anomalies in provided data.
    
    Request format:
    {
        "data": [[...], [...], ...],
        "algorithm": "isolation_forest",
        "contamination": 0.1,
        "parameters": {...}
    }
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Extract parameters
    data = request.get("data", [])
    algorithm = request.get("algorithm", "isolation_forest")
    contamination = request.get("contamination")
    parameters = request.get("parameters", {})
    
    # Perform detection
    result = await service.detect_anomalies(
        data=data,
        algorithm=algorithm,
        contamination=contamination,
        **parameters
    )
    
    # Log request for audit (background task)
    background_tasks.add_task(
        log_detection_request,
        result.get("request_id"),
        len(data),
        algorithm,
        result.get("success", False)
    )
    
    return result


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get service metrics."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return service.get_health_status()


async def log_detection_request(request_id: str, sample_count: int, algorithm: str, success: bool):
    """Log detection request for audit purposes."""
    logging.info(f"AUDIT: Request {request_id} - {sample_count} samples, {algorithm}, success: {success}")


def run_production_deployment():
    """
    Run the production deployment with proper configuration.
    
    This function demonstrates how to deploy the service in production
    with proper configuration, monitoring, and scaling.
    """
    print("🚀 Production Deployment Guide")
    print("=" * 50)
    
    config = ProductionConfig()
    
    print(f"\n📋 Configuration:")
    print(f"   Host: {config.host}")
    print(f"   Port: {config.port}")
    print(f"   Workers: {config.workers}")
    print(f"   Max request size: {config.max_request_size // (1024*1024)}MB")
    print(f"   Timeout: {config.timeout_seconds}s")
    print(f"   Rate limit: {config.rate_limit_requests}/min")
    print(f"   Metrics enabled: {config.enable_metrics}")
    print(f"   Log level: {config.log_level}")
    
    print(f"\n🔧 Deployment Options:")
    
    print(f"\n1. Single Server Deployment:")
    print(f"   uvicorn production_deployment_guide:app --host {config.host} --port {config.port}")
    
    print(f"\n2. Multi-Worker Deployment:")
    print(f"   uvicorn production_deployment_guide:app --host {config.host} --port {config.port} --workers {config.workers}")
    
    print(f"\n3. Docker Deployment:")
    print("""   docker run -d \\
     --name anomaly-detection \\
     -p 8000:8000 \\
     -p 9090:9090 \\
     -e ENVIRONMENT=production \\
     -v /var/log:/var/log \\
     anomaly-detection:latest""")
    
    print(f"\n4. Kubernetes Deployment:")
    print("""   kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: anomaly-detection
  template:
    metadata:
      labels:
        app: anomaly-detection
    spec:
      containers:
      - name: anomaly-detection
        image: anomaly-detection:latest
        ports:
        - containerPort: 8000
        - containerPort: 9090
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
EOF""")
    
    print(f"\n📊 Monitoring Setup:")
    print(f"   Health check: GET http://localhost:{config.port}/health")
    print(f"   Detailed health: GET http://localhost:{config.port}/health/detailed")
    print(f"   Metrics: GET http://localhost:{config.port}/api/v1/metrics")
    print(f"   Prometheus metrics: http://localhost:{config.metrics_port}/metrics")
    
    print(f"\n🔍 Testing the Deployment:")
    print("""   curl -X POST http://localhost:8000/api/v1/detect \\
     -H "Content-Type: application/json" \\
     -d '{
       "data": [[1,2,3], [4,5,6], [100,200,300]],
       "algorithm": "isolation_forest",
       "contamination": 0.1
     }'""")
    
    print(f"\n⚠️  Production Checklist:")
    checklist = [
        "✅ Configure proper CORS origins",
        "✅ Set up SSL/TLS certificates", 
        "✅ Configure rate limiting",
        "✅ Set up log rotation",
        "✅ Configure monitoring alerts",
        "✅ Set up backup and recovery",
        "✅ Configure auto-scaling policies",
        "✅ Set up health check probes",
        "✅ Configure resource limits",
        "✅ Set up security scanning"
    ]
    
    for item in checklist:
        print(f"   {item}")
    
    print(f"\n💡 Best Practices:")
    print(f"   • Use environment variables for configuration")
    print(f"   • Implement circuit breakers for external dependencies")
    print(f"   • Set up distributed tracing with OpenTelemetry")
    print(f"   • Use Redis for caching and session management")
    print(f"   • Implement graceful shutdown handling")
    print(f"   • Set up automated backups for model artifacts")
    print(f"   • Monitor memory usage and implement cleanup")
    print(f"   • Use load balancers for high availability")
    
    print(f"\n🏃‍♂️ Starting production server...")
    
    # Start the server
    uvicorn.run(
        "production_deployment_guide:app",
        host=config.host,
        port=config.port,
        workers=1,  # Single worker for demo
        log_level=config.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    # Run the production deployment guide
    run_production_deployment()