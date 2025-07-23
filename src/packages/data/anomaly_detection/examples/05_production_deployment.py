#!/usr/bin/env python3
"""
Production Deployment Examples for Anomaly Detection Package

This example demonstrates production-ready deployment patterns including:
- FastAPI REST API integration
- Docker containerization
- Kubernetes deployment manifests
- Model serving and scaling patterns
- Health checks and monitoring
- Configuration management
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import tempfile
import subprocess
import logging

# FastAPI and web framework imports
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. Install with: pip install fastapi uvicorn")

# Additional production dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available. Install with: pip install redis")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with: pip install psutil")

import numpy as np
import pandas as pd

# Import anomaly detection components
try:
    from anomaly_detection import DetectionService, EnsembleService
    from anomaly_detection.domain.entities.detection_result import DetectionResult
except ImportError:
    print("Please install the anomaly_detection package first:")
    print("pip install -e .")
    exit(1)


# Pydantic models for API
if FASTAPI_AVAILABLE:
    class DetectionRequest(BaseModel):
        """Request model for anomaly detection."""
        data: List[List[float]] = Field(..., description="Input data matrix")
        algorithm: str = Field(default="iforest", description="Detection algorithm")
        contamination: float = Field(default=0.1, ge=0.0, le=0.5, description="Expected contamination rate")
        parameters: Optional[Dict[str, Any]] = Field(default=None, description="Algorithm-specific parameters")
        
        class Config:
            schema_extra = {
                "example": {
                    "data": [[1.2, 2.3, 3.4], [2.1, 3.2, 4.3], [10.0, 20.0, 30.0]],
                    "algorithm": "iforest",
                    "contamination": 0.1,
                    "parameters": {"n_estimators": 100}
                }
            }

    class DetectionResponse(BaseModel):
        """Response model for anomaly detection."""
        predictions: List[int] = Field(..., description="Anomaly predictions (-1 for anomaly, 1 for normal)")
        anomaly_scores: List[float] = Field(..., description="Anomaly scores")
        anomaly_count: int = Field(..., description="Number of detected anomalies")
        anomaly_rate: float = Field(..., description="Rate of anomalies")
        processing_time: float = Field(..., description="Processing time in seconds")
        algorithm_used: str = Field(..., description="Algorithm used for detection")
        model_metadata: Dict[str, Any] = Field(default={}, description="Model metadata")

    class BatchDetectionRequest(BaseModel):
        """Request model for batch anomaly detection."""
        data: List[List[float]] = Field(..., description="Input data matrix")
        algorithms: List[str] = Field(default=["iforest"], description="Detection algorithms to use")
        contamination: float = Field(default=0.1, description="Expected contamination rate")
        ensemble_method: str = Field(default="majority", description="Ensemble method")

    class HealthResponse(BaseModel):
        """Health check response model."""
        status: str = Field(..., description="Service status")
        timestamp: str = Field(..., description="Current timestamp")
        version: str = Field(..., description="Service version")
        uptime: float = Field(..., description="Service uptime in seconds")
        memory_usage: Dict[str, float] = Field(default={}, description="Memory usage statistics")
        model_status: Dict[str, str] = Field(default={}, description="Model loading status")


class ProductionAnomalyService:
    """Production-ready anomaly detection service."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.detection_service = DetectionService()
        self.ensemble_service = EnsembleService()
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Initialize Redis if available
        self.redis_client = None
        if REDIS_AVAILABLE and self.config.get('redis_url'):
            try:
                self.redis_client = redis.from_url(self.config['redis_url'])
                self.redis_client.ping()
                print("Redis connection established")
            except Exception as e:
                print(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Setup logging
        self.setup_logging()
        
        # Preload models if specified
        self.preloaded_models = {}
        if self.config.get('preload_models'):
            self._preload_models()
    
    def setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _preload_models(self):
        """Preload commonly used models."""
        try:
            algorithms = self.config.get('preload_models', ['iforest', 'lof'])
            sample_data = np.random.randn(100, 10)  # Sample data for initialization
            
            for algorithm in algorithms:
                self.logger.info(f"Preloading {algorithm} model...")
                # Fit with sample data
                self.detection_service.fit(sample_data, algorithm=algorithm)
                self.preloaded_models[algorithm] = True
                
        except Exception as e:
            self.logger.error(f"Error preloading models: {e}")
    
    def get_health_info(self) -> Dict[str, Any]:
        """Get service health information."""
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "uptime": time.time() - self.start_time,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1)
        }
        
        # Memory usage
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            health_info["memory_usage"] = {
                "rss": memory_info.rss / 1024 / 1024,  # MB
                "vms": memory_info.vms / 1024 / 1024,  # MB
                "percent": process.memory_percent()
            }
        
        # Model status
        health_info["model_status"] = {
            algorithm: "loaded" if loaded else "not_loaded"
            for algorithm, loaded in self.preloaded_models.items()
        }
        
        return health_info
    
    async def detect_anomalies(self, request: DetectionRequest) -> DetectionResponse:
        """Perform anomaly detection."""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Convert request data to numpy array
            X = np.array(request.data)
            
            # Check cache if Redis is available
            cache_key = None
            if self.redis_client:
                # Create cache key from request hash
                import hashlib
                request_hash = hashlib.sha256(
                    f"{X.tobytes()}{request.algorithm}{request.contamination}".encode()
                ).hexdigest()
                cache_key = f"anomaly_detection:{request_hash}"
                
                # Try to get from cache
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    self.logger.info("Returning cached result")
                    cached_data = json.loads(cached_result)
                    cached_data["processing_time"] = time.time() - start_time
                    return DetectionResponse(**cached_data)
            
            # Perform detection
            parameters = request.parameters or {}
            result = self.detection_service.detect_anomalies(
                data=X,
                algorithm=request.algorithm,
                contamination=request.contamination,
                **parameters
            )
            
            # Create response
            response_data = {
                "predictions": result.predictions.tolist(),
                "anomaly_scores": result.anomaly_scores.tolist(),
                "anomaly_count": result.anomaly_count,
                "anomaly_rate": result.anomaly_rate,
                "processing_time": time.time() - start_time,
                "algorithm_used": request.algorithm,
                "model_metadata": {
                    "contamination": request.contamination,
                    "data_shape": X.shape,
                    "parameters": parameters
                }
            }
            
            # Cache result if Redis is available
            if self.redis_client and cache_key:
                cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour default
                self.redis_client.setex(
                    cache_key,
                    cache_ttl,
                    json.dumps({k: v for k, v in response_data.items() if k != "processing_time"})
                )
            
            self.logger.info(f"Detection completed in {response_data['processing_time']:.3f}s")
            return DetectionResponse(**response_data)
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Detection error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def batch_detect_anomalies(self, request: BatchDetectionRequest) -> Dict[str, Any]:
        """Perform batch anomaly detection with multiple algorithms."""
        start_time = time.time()
        self.request_count += 1
        
        try:
            X = np.array(request.data)
            
            # Use ensemble service for multiple algorithms
            result = self.ensemble_service.detect_with_ensemble(
                data=X,
                algorithms=request.algorithms,
                method=request.ensemble_method,
                contamination=request.contamination
            )
            
            response = {
                "predictions": result.predictions.tolist(),
                "anomaly_scores": result.anomaly_scores.tolist() if hasattr(result, 'anomaly_scores') else [],
                "anomaly_count": result.anomaly_count,
                "anomaly_rate": result.anomaly_rate,
                "processing_time": time.time() - start_time,
                "algorithms_used": request.algorithms,
                "ensemble_method": request.ensemble_method,
                "metadata": {
                    "contamination": request.contamination,
                    "data_shape": X.shape
                }
            }
            
            self.logger.info(f"Batch detection completed in {response['processing_time']:.3f}s")
            return response
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Batch detection error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


def create_fastapi_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    # Initialize service
    service = ProductionAnomalyService(config)
    
    # Create FastAPI app
    app = FastAPI(
        title="Anomaly Detection API",
        description="Production-ready anomaly detection service",
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
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        health_info = service.get_health_info()
        return HealthResponse(**health_info)
    
    @app.post("/detect", response_model=DetectionResponse)
    async def detect_anomalies(request: DetectionRequest):
        """Detect anomalies in the provided data."""
        return await service.detect_anomalies(request)
    
    @app.post("/batch-detect")
    async def batch_detect_anomalies(request: BatchDetectionRequest):
        """Batch anomaly detection with multiple algorithms."""
        return await service.batch_detect_anomalies(request)
    
    @app.get("/algorithms")
    async def list_algorithms():
        """List available algorithms."""
        return {
            "algorithms": [
                "iforest", "lof", "ocsvm", "pca", "knn", "hbos"
            ],
            "ensemble_methods": [
                "majority", "average", "weighted"
            ]
        }
    
    @app.get("/metrics")
    async def get_metrics():
        """Get service metrics."""
        return service.get_health_info()
    
    return app


def create_docker_files(output_dir: Path):
    """Create Docker configuration files."""
    
    # Dockerfile
    dockerfile_content = """
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "05_production_deployment:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    # requirements.txt
    requirements_content = """
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
redis>=4.5.0
psutil>=5.9.0
pydantic>=2.0.0
python-multipart>=0.0.6
"""
    
    # docker-compose.yml
    docker_compose_content = """
version: '3.8'

services:
  anomaly-detection:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - anomaly-detection
    restart: unless-stopped

volumes:
  redis_data:
"""
    
    # nginx.conf
    nginx_conf_content = """
events {
    worker_connections 1024;
}

http {
    upstream anomaly_detection {
        server anomaly-detection:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://anomaly_detection;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://anomaly_detection/health;
            access_log off;
        }
    }
}
"""
    
    # Write files
    (output_dir / "Dockerfile").write_text(dockerfile_content.strip())
    (output_dir / "requirements.txt").write_text(requirements_content.strip())
    (output_dir / "docker-compose.yml").write_text(docker_compose_content.strip())
    (output_dir / "nginx.conf").write_text(nginx_conf_content.strip())
    
    print(f"Docker files created in {output_dir}")


def create_kubernetes_manifests(output_dir: Path):
    """Create Kubernetes deployment manifests."""
    
    # Namespace
    namespace_yaml = """
apiVersion: v1
kind: Namespace
metadata:
  name: anomaly-detection
"""
    
    # ConfigMap
    configmap_yaml = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: anomaly-detection-config
  namespace: anomaly-detection
data:
  REDIS_URL: "redis://redis-service:6379"
  LOG_LEVEL: "INFO"
  CACHE_TTL: "3600"
"""
    
    # Deployment
    deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detection
  namespace: anomaly-detection
  labels:
    app: anomaly-detection
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
        envFrom:
        - configMapRef:
            name: anomaly-detection-config
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
          requests:
            memory: "512Mi"
            cpu: "250m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
"""
    
    # Service
    service_yaml = """
apiVersion: v1
kind: Service
metadata:
  name: anomaly-detection-service
  namespace: anomaly-detection
spec:
  selector:
    app: anomaly-detection
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
"""
    
    # Ingress
    ingress_yaml = """
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: anomaly-detection-ingress
  namespace: anomaly-detection
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  ingressClassName: nginx
  rules:
  - host: anomaly-detection.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: anomaly-detection-service
            port:
              number: 80
"""
    
    # Redis deployment
    redis_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: anomaly-detection
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          limits:
            memory: "256Mi"
            cpu: "250m"
          requests:
            memory: "128Mi"
            cpu: "125m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: anomaly-detection
spec:
  selector:
    app: redis
  ports:
  - protocol: TCP
    port: 6379
    targetPort: 6379
"""
    
    # HPA (Horizontal Pod Autoscaler)
    hpa_yaml = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: anomaly-detection-hpa
  namespace: anomaly-detection
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-detection
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
    
    # Write files
    k8s_dir = output_dir / "k8s"
    k8s_dir.mkdir(exist_ok=True)
    
    (k8s_dir / "namespace.yaml").write_text(namespace_yaml.strip())
    (k8s_dir / "configmap.yaml").write_text(configmap_yaml.strip())
    (k8s_dir / "deployment.yaml").write_text(deployment_yaml.strip())
    (k8s_dir / "service.yaml").write_text(service_yaml.strip())
    (k8s_dir / "ingress.yaml").write_text(ingress_yaml.strip())
    (k8s_dir / "redis.yaml").write_text(redis_yaml.strip())
    (k8s_dir / "hpa.yaml").write_text(hpa_yaml.strip())
    
    print(f"Kubernetes manifests created in {k8s_dir}")


def create_monitoring_config(output_dir: Path):
    """Create monitoring configuration files."""
    
    # Prometheus configuration
    prometheus_yaml = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'anomaly-detection'
    static_configs:
      - targets: ['anomaly-detection:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""
    
    # Alert rules
    alert_rules_yaml = """
groups:
- name: anomaly_detection_alerts
  rules:
  - alert: HighErrorRate
    expr: (anomaly_detection_errors_total / anomaly_detection_requests_total) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High error rate detected
      description: "Error rate is {{ $value | humanizePercentage }}"

  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes > 1000000000
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High memory usage
      description: "Memory usage is {{ $value | humanizeBytes }}"

  - alert: ServiceDown
    expr: up{job="anomaly-detection"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Anomaly detection service is down
      description: "Service has been down for more than 1 minute"
"""
    
    # Grafana dashboard
    grafana_dashboard = """
{
  "dashboard": {
    "title": "Anomaly Detection Service",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(anomaly_detection_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(anomaly_detection_errors_total[5m]) / rate(anomaly_detection_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(anomaly_detection_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes"
          }
        ]
      }
    ]
  }
}
"""
    
    # Write monitoring files
    monitoring_dir = output_dir / "monitoring"
    monitoring_dir.mkdir(exist_ok=True)
    
    (monitoring_dir / "prometheus.yml").write_text(prometheus_yaml.strip())
    (monitoring_dir / "alert_rules.yml").write_text(alert_rules_yaml.strip())
    (monitoring_dir / "grafana_dashboard.json").write_text(grafana_dashboard.strip())
    
    print(f"Monitoring configuration created in {monitoring_dir}")


def example_1_fastapi_integration():
    """Example 1: FastAPI REST API integration."""
    print("\n" + "="*60)
    print("Example 1: FastAPI REST API Integration")
    print("="*60)
    
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return
    
    # Configuration
    config = {
        'redis_url': 'redis://localhost:6379' if REDIS_AVAILABLE else None,
        'cache_ttl': 3600,
        'preload_models': ['iforest', 'lof']
    }
    
    # Create FastAPI app
    app = create_fastapi_app(config)
    
    print("FastAPI application created with the following endpoints:")
    print("- GET /health - Health check")
    print("- POST /detect - Single algorithm detection")
    print("- POST /batch-detect - Multi-algorithm ensemble detection")
    print("- GET /algorithms - List available algorithms")
    print("- GET /metrics - Service metrics")
    print("- GET /docs - API documentation")
    
    # Example client usage
    print("\nExample client usage:")
    client_code = '''
import requests
import numpy as np

# Generate sample data
data = np.random.randn(100, 5).tolist()

# Detection request
response = requests.post("http://localhost:8000/detect", json={
    "data": data,
    "algorithm": "iforest",
    "contamination": 0.1,
    "parameters": {"n_estimators": 100}
})

result = response.json()
print(f"Detected {result['anomaly_count']} anomalies")
print(f"Processing time: {result['processing_time']:.3f} seconds")
'''
    print(client_code)
    
    # Offer to start the server
    print("\nOptions:")
    print("1. Start development server")
    print("2. Show more examples")
    print("3. Skip to next example")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("\nStarting FastAPI development server...")
        print("Server will be available at http://localhost:8000")
        print("API documentation at http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        
        try:
            # Make the app globally available for uvicorn
            globals()['app'] = app
            uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
        except KeyboardInterrupt:
            print("\nServer stopped.")
    
    elif choice == "2":
        print("\nBatch detection example:")
        batch_example = '''
# Batch detection with multiple algorithms
batch_response = requests.post("http://localhost:8000/batch-detect", json={
    "data": data,
    "algorithms": ["iforest", "lof", "ocsvm"],
    "contamination": 0.1,
    "ensemble_method": "majority"
})

batch_result = batch_response.json()
print(f"Ensemble detected {batch_result['anomaly_count']} anomalies")
print(f"Algorithms used: {batch_result['algorithms_used']}")
'''
        print(batch_example)


def example_2_docker_containerization():
    """Example 2: Docker containerization."""
    print("\n" + "="*60)
    print("Example 2: Docker Containerization")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "docker_deployment"
        output_dir.mkdir()
        
        # Create Docker files
        create_docker_files(output_dir)
        
        print("Docker configuration files created:")
        for file in output_dir.iterdir():
            print(f"- {file.name}")
        
        # Show build and run commands
        print("\nDocker commands:")
        print("1. Build the image:")
        print("   docker build -t anomaly-detection .")
        
        print("\n2. Run with docker-compose:")
        print("   docker-compose up -d")
        
        print("\n3. Check status:")
        print("   docker-compose ps")
        
        print("\n4. View logs:")
        print("   docker-compose logs -f anomaly-detection")
        
        print("\n5. Scale the service:")
        print("   docker-compose up -d --scale anomaly-detection=3")
        
        print("\n6. Stop services:")
        print("   docker-compose down")
        
        # Show Dockerfile content
        print("\nGenerated Dockerfile:")
        print("-" * 40)
        print((output_dir / "Dockerfile").read_text())
        
        # Offer to copy files
        copy_choice = input("\nCopy Docker files to current directory? (y/n): ").strip().lower()
        if copy_choice == 'y':
            import shutil
            target_dir = Path.cwd() / "docker_deployment"
            shutil.copytree(output_dir, target_dir, dirs_exist_ok=True)
            print(f"Docker files copied to {target_dir}")


def example_3_kubernetes_deployment():
    """Example 3: Kubernetes deployment."""
    print("\n" + "="*60)
    print("Example 3: Kubernetes Deployment")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "k8s_deployment"
        output_dir.mkdir()
        
        # Create Kubernetes manifests
        create_kubernetes_manifests(output_dir)
        
        print("Kubernetes manifests created:")
        k8s_dir = output_dir / "k8s"
        for file in k8s_dir.iterdir():
            print(f"- {file.name}")
        
        print("\nDeployment commands:")
        print("1. Create namespace:")
        print("   kubectl apply -f k8s/namespace.yaml")
        
        print("\n2. Deploy application:")
        print("   kubectl apply -f k8s/")
        
        print("\n3. Check deployment status:")
        print("   kubectl get pods -n anomaly-detection")
        
        print("\n4. Check services:")
        print("   kubectl get svc -n anomaly-detection")
        
        print("\n5. View logs:")
        print("   kubectl logs -f deployment/anomaly-detection -n anomaly-detection")
        
        print("\n6. Scale deployment:")
        print("   kubectl scale deployment anomaly-detection --replicas=5 -n anomaly-detection")
        
        print("\n7. Port forward for testing:")
        print("   kubectl port-forward svc/anomaly-detection-service 8080:80 -n anomaly-detection")
        
        # Show key manifest content
        print("\nGenerated Deployment manifest:")
        print("-" * 40)
        deployment_content = (k8s_dir / "deployment.yaml").read_text()
        print(deployment_content[:800] + "..." if len(deployment_content) > 800 else deployment_content)
        
        # Offer to copy files
        copy_choice = input("\nCopy Kubernetes files to current directory? (y/n): ").strip().lower()
        if copy_choice == 'y':
            import shutil
            target_dir = Path.cwd() / "k8s_deployment"
            shutil.copytree(output_dir, target_dir, dirs_exist_ok=True)
            print(f"Kubernetes files copied to {target_dir}")


def example_4_monitoring_setup():
    """Example 4: Monitoring and observability setup."""
    print("\n" + "="*60)
    print("Example 4: Monitoring and Observability")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "monitoring_setup"
        output_dir.mkdir()
        
        # Create monitoring configuration
        create_monitoring_config(output_dir)
        
        print("Monitoring configuration created:")
        monitoring_dir = output_dir / "monitoring"
        for file in monitoring_dir.iterdir():
            print(f"- {file.name}")
        
        print("\nMonitoring stack setup:")
        print("1. Prometheus for metrics collection")
        print("2. Grafana for visualization")
        print("3. Alertmanager for alerting")
        
        # Create docker-compose for monitoring stack
        monitoring_compose = """
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  grafana_data:
"""
        
        (monitoring_dir / "docker-compose-monitoring.yml").write_text(monitoring_compose.strip())
        
        print("\nSetup commands:")
        print("1. Start monitoring stack:")
        print("   docker-compose -f monitoring/docker-compose-monitoring.yml up -d")
        
        print("\n2. Access services:")
        print("   - Prometheus: http://localhost:9090")
        print("   - Grafana: http://localhost:3000 (admin/admin)")
        print("   - Alertmanager: http://localhost:9093")
        
        print("\n3. Import Grafana dashboard:")
        print("   - Login to Grafana")
        print("   - Import the generated dashboard JSON")
        
        print("\nKey metrics to monitor:")
        print("- Request rate and latency")
        print("- Error rate")
        print("- Memory and CPU usage")
        print("- Model performance metrics")
        print("- Cache hit rate (if using Redis)")
        
        # Show alert rules
        print("\nGenerated alert rules:")
        print("-" * 40)
        alert_rules = (monitoring_dir / "alert_rules.yml").read_text()
        print(alert_rules[:600] + "..." if len(alert_rules) > 600 else alert_rules)


def example_5_load_testing():
    """Example 5: Load testing and performance validation."""
    print("\n" + "="*60)
    print("Example 5: Load Testing and Performance")
    print("="*60)
    
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available for load testing example.")
        return
    
    # Create a simple load testing script
    load_test_script = '''
import asyncio
import aiohttp
import time
import numpy as np
from statistics import mean, stdev

async def make_request(session, url, data):
    """Make a single request."""
    start_time = time.time()
    try:
        async with session.post(url, json=data) as response:
            result = await response.json()
            elapsed = time.time() - start_time
            return {
                "success": response.status == 200,
                "elapsed": elapsed,
                "status": response.status,
                "anomaly_count": result.get("anomaly_count", 0) if response.status == 200 else 0
            }
    except Exception as e:
        return {
            "success": False,
            "elapsed": time.time() - start_time,
            "status": 0,
            "error": str(e),
            "anomaly_count": 0
        }

async def load_test(url, num_requests=100, concurrency=10):
    """Run load test."""
    print(f"Starting load test: {num_requests} requests with {concurrency} concurrent workers")
    
    # Generate test data
    test_data = {
        "data": np.random.randn(50, 5).tolist(),
        "algorithm": "iforest",
        "contamination": 0.1
    }
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(session):
        async with semaphore:
            return await make_request(session, url, test_data)
    
    # Run requests
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(session) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    if successful:
        response_times = [r["elapsed"] for r in successful]
        anomaly_counts = [r["anomaly_count"] for r in successful]
        
        print(f"\\nLoad Test Results:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Requests per second: {num_requests / total_time:.2f}")
        print(f"Successful requests: {len(successful)}/{num_requests}")
        print(f"Failed requests: {len(failed)}")
        print(f"Average response time: {mean(response_times):.3f} seconds")
        print(f"Response time std dev: {stdev(response_times):.3f} seconds")
        print(f"Min response time: {min(response_times):.3f} seconds")
        print(f"Max response time: {max(response_times):.3f} seconds")
        print(f"Average anomalies detected: {mean(anomaly_counts):.1f}")
    
    if failed:
        print(f"\\nFailure analysis:")
        status_codes = {}
        for result in failed:
            status = result.get("status", "unknown")
            status_codes[status] = status_codes.get(status, 0) + 1
        
        for status, count in status_codes.items():
            print(f"Status {status}: {count} failures")

if __name__ == "__main__":
    # Run load test
    asyncio.run(load_test("http://localhost:8000/detect", num_requests=100, concurrency=10))
'''
    
    print("Load testing capabilities:")
    print("1. Asynchronous HTTP client for concurrent requests")
    print("2. Configurable request rate and concurrency")
    print("3. Response time analysis")
    print("4. Success/failure rate tracking")
    print("5. Performance metrics collection")
    
    # Create the load test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(load_test_script)
        load_test_file = f.name
    
    print(f"\nLoad test script created: {load_test_file}")
    print("\nTo run load test:")
    print("1. Start the anomaly detection service")
    print("2. Install aiohttp: pip install aiohttp")
    print(f"3. Run: python {load_test_file}")
    
    print("\nLoad testing recommendations:")
    print("- Start with low concurrency and gradually increase")
    print("- Monitor server resources during testing")
    print("- Test different data sizes and algorithms")
    print("- Use tools like Apache Bench or wrk for additional testing")
    print("- Set up proper monitoring before running load tests")


def main():
    """Run all production deployment examples."""
    print("\n" + "="*60)
    print("PRODUCTION DEPLOYMENT FOR ANOMALY DETECTION")
    print("="*60)
    
    examples = [
        ("FastAPI REST API Integration", example_1_fastapi_integration),
        ("Docker Containerization", example_2_docker_containerization),
        ("Kubernetes Deployment", example_3_kubernetes_deployment),
        ("Monitoring and Observability", example_4_monitoring_setup),
        ("Load Testing and Performance", example_5_load_testing)
    ]
    
    while True:
        print("\nAvailable Examples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print("0. Exit")
        
        try:
            choice = int(input("\nSelect an example (0-5): "))
            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= len(examples):
                examples[choice-1][1]()
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error running example: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()