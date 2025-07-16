# ML Engineer Onboarding Guide

🍞 **Breadcrumb:** 🏠 [Home](../../../index.md) > 📚 [User Guides](../../README.md) > 🚀 [Onboarding](../README.md) > 🎯 [Role-Specific](README.md) > 🔧 ML Engineer

---

Welcome, ML Engineer! This guide will help you master Pynomaly's production deployment, API integration, monitoring, and MLOps capabilities for building robust anomaly detection systems at scale.

## 🎯 Learning Objectives

By the end of this guide, you'll be able to:

- **Deploy** Pynomaly in production environments with high availability
- **Integrate** anomaly detection into existing ML pipelines and systems  
- **Monitor** model performance and data drift in real-time
- **Scale** detection systems to handle high-volume data streams
- **Implement** MLOps best practices for anomaly detection workflows

## 🚀 Quick Start (5 minutes)

### Production-Ready Installation

```bash
# Install with production dependencies
pip install "pynomaly[production,monitoring,async]"

# Docker deployment
docker pull pynomaly/pynomaly:latest

# Verify installation
python -c "
import pynomaly
from pynomaly.server import HealthCheck
print(f'Pynomaly version: {pynomaly.__version__}')
print('Production components:', HealthCheck().check_dependencies())
"
```

### Quick API Integration

```python
import asyncio
from pynomaly.client import AsyncPynomalyClient
from pynomaly.models import DetectionRequest, BatchDetectionRequest

async def quick_integration_demo():
    """Demonstrate basic API integration."""
    
    # Initialize client
    client = AsyncPynomalyClient(
        base_url="http://localhost:8000",
        api_key="your-api-key",
        timeout=30.0
    )
    
    # Single detection
    request = DetectionRequest(
        data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        algorithm="isolation_forest",
        contamination=0.1
    )
    
    result = await client.detect_anomalies(request)
    print(f"Anomalies detected: {sum(1 for x in result.predictions if x == -1)}")
    
    # Batch processing
    batch_request = BatchDetectionRequest(
        data_sources=["s3://bucket/data1.csv", "s3://bucket/data2.csv"],
        algorithm="isolation_forest",
        output_format="json"
    )
    
    batch_result = await client.batch_detect(batch_request)
    print(f"Batch job ID: {batch_result.job_id}")
    
    await client.close()

# Run the demo
asyncio.run(quick_integration_demo())
```

## 🏗️ Production Architecture

### High-Level System Design

```mermaid
graph TB
    subgraph "Load Balancing"
        LB[Load Balancer<br/>Nginx/HAProxy]
    end
    
    subgraph "API Layer"
        API1[Pynomaly API 1<br/>FastAPI]
        API2[Pynomaly API 2<br/>FastAPI]
        API3[Pynomaly API N<br/>FastAPI]
    end
    
    subgraph "Processing Layer"
        WORKER1[Async Worker 1<br/>Celery/RQ]
        WORKER2[Async Worker 2<br/>Celery/RQ]
        WORKER3[Async Worker N<br/>Celery/RQ]
    end
    
    subgraph "Data Layer"
        REDIS[(Redis Cluster<br/>Cache/Queue)]
        POSTGRES[(PostgreSQL<br/>Metadata)]
        S3[(Object Storage<br/>S3/MinIO)]
    end
    
    subgraph "Monitoring"
        PROMETHEUS[Prometheus<br/>Metrics]
        GRAFANA[Grafana<br/>Dashboards]
        JAEGER[Jaeger<br/>Tracing]
    end
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> REDIS
    API2 --> REDIS
    API3 --> REDIS
    
    API1 --> POSTGRES
    API2 --> POSTGRES
    API3 --> POSTGRES
    
    REDIS --> WORKER1
    REDIS --> WORKER2
    REDIS --> WORKER3
    
    WORKER1 --> S3
    WORKER2 --> S3
    WORKER3 --> S3
    
    API1 --> PROMETHEUS
    API2 --> PROMETHEUS
    API3 --> PROMETHEUS
    
    PROMETHEUS --> GRAFANA
    API1 --> JAEGER
```

### Container Orchestration Setup

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  # Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - pynomaly-api-1
      - pynomaly-api-2
    restart: unless-stopped
    
  # API Services
  pynomaly-api-1:
    image: pynomaly/pynomaly:latest
    environment:
      - REDIS_URL=redis://redis-cluster:6379
      - DATABASE_URL=postgresql://pynomaly:${POSTGRES_PASSWORD}@postgres:5432/pynomaly_prod
      - WORKER_CONCURRENCY=4
      - LOG_LEVEL=INFO
    depends_on:
      - postgres
      - redis-cluster
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
          
  pynomaly-api-2:
    extends:
      service: pynomaly-api-1
      
  # Async Workers
  pynomaly-worker-1:
    image: pynomaly/pynomaly:latest
    command: celery -A pynomaly.worker worker --loglevel=info --concurrency=4
    environment:
      - REDIS_URL=redis://redis-cluster:6379
      - DATABASE_URL=postgresql://pynomaly:${POSTGRES_PASSWORD}@postgres:5432/pynomaly_prod
    depends_on:
      - postgres
      - redis-cluster
    restart: unless-stopped
    
  pynomaly-worker-2:
    extends:
      service: pynomaly-worker-1
      
  # Data Services
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=pynomaly_prod
      - POSTGRES_USER=pynomaly
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    
  redis-cluster:
    image: redis:7-alpine
    command: redis-server --appendonly yes --cluster-enabled yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

## 🔌 API Integration Patterns

### RESTful API Client

```python
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

class PynomalyAPIClient:
    """Production-ready API client with comprehensive error handling."""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "pynomaly-client/1.0"
            }
        )
        
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.aclose()
        
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic and error handling."""
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params
                )
                
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                    self.logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500 or attempt == self.max_retries:
                    self.logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                    raise
                    
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {self.retry_delay}s")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                if attempt == self.max_retries:
                    self.logger.error(f"Timeout error: {e}")
                    raise
                    
                self.logger.warning(f"Timeout on attempt {attempt + 1}, retrying")
                await asyncio.sleep(self.retry_delay)
                
        raise Exception(f"Max retries ({self.max_retries}) exceeded")
    
    async def detect_anomalies(
        self,
        data: List[List[float]],
        algorithm: str = "isolation_forest",
        contamination: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """Detect anomalies in provided data."""
        
        request_data = {
            "data": data,
            "algorithm": algorithm,
            "contamination": contamination,
            **kwargs
        }
        
        return await self._make_request("POST", "/api/v1/detect", data=request_data)
    
    async def batch_detect(
        self,
        data_source: str,
        algorithm: str = "isolation_forest",
        output_format: str = "json",
        **kwargs
    ) -> Dict[str, Any]:
        """Submit batch detection job."""
        
        request_data = {
            "data_source": data_source,
            "algorithm": algorithm,
            "output_format": output_format,
            **kwargs
        }
        
        return await self._make_request("POST", "/api/v1/batch", data=request_data)
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of batch job."""
        return await self._make_request("GET", f"/api/v1/jobs/{job_id}")
    
    async def train_model(
        self,
        training_data: str,
        algorithm: str,
        model_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Train and persist a model."""
        
        request_data = {
            "training_data": training_data,
            "algorithm": algorithm,
            "model_name": model_name,
            **kwargs
        }
        
        return await self._make_request("POST", "/api/v1/models", data=request_data)
    
    async def predict_with_model(
        self,
        model_name: str,
        data: List[List[float]]
    ) -> Dict[str, Any]:
        """Make predictions using a trained model."""
        
        request_data = {
            "data": data
        }
        
        return await self._make_request(
            "POST", 
            f"/api/v1/models/{model_name}/predict", 
            data=request_data
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        return await self._make_request("GET", "/health")
```

### Streaming Integration

```python
import asyncio
import json
from typing import AsyncIterator
from confluent_kafka import Consumer, Producer
from pynomaly.streaming import StreamingDetector

class KafkaAnomalyDetector:
    """Real-time anomaly detection with Kafka integration."""
    
    def __init__(
        self,
        kafka_config: Dict[str, str],
        input_topic: str,
        output_topic: str,
        detector_config: Dict[str, Any]
    ):
        self.kafka_config = kafka_config
        self.input_topic = input_topic
        self.output_topic = output_topic
        
        # Initialize Kafka components
        self.consumer = Consumer({
            **kafka_config,
            'group.id': 'pynomaly-detector',
            'auto.offset.reset': 'latest'
        })
        
        self.producer = Producer(kafka_config)
        
        # Initialize detector
        self.detector = StreamingDetector(**detector_config)
        
        self.logger = logging.getLogger(__name__)
        
    async def start_processing(self):
        """Start processing messages from Kafka."""
        
        self.consumer.subscribe([self.input_topic])
        self.logger.info(f"Started consuming from {self.input_topic}")
        
        try:
            while True:
                # Poll for messages
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                    
                if msg.error():
                    self.logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                # Process message
                await self._process_message(msg)
                
        except KeyboardInterrupt:
            self.logger.info("Stopping consumer...")
        finally:
            self.consumer.close()
            
    async def _process_message(self, msg):
        """Process individual Kafka message."""
        
        try:
            # Parse message
            data = json.loads(msg.value().decode('utf-8'))
            
            # Extract features
            features = self._extract_features(data)
            
            # Detect anomalies
            result = await self.detector.detect_async(features)
            
            # Prepare output
            output = {
                'timestamp': data.get('timestamp'),
                'original_data': data,
                'is_anomaly': result['is_anomaly'],
                'anomaly_score': result['score'],
                'confidence': result['confidence'],
                'metadata': {
                    'algorithm': self.detector.algorithm_name,
                    'processing_time_ms': result['processing_time_ms']
                }
            }
            
            # Send to output topic
            self.producer.produce(
                topic=self.output_topic,
                key=str(data.get('id', '')),
                value=json.dumps(output)
            )
            
            self.producer.poll(0)  # Trigger delivery callbacks
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            
    def _extract_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from raw data."""
        
        features = []
        
        # Extract numerical fields
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, dict):
                # Nested dictionary - extract recursively
                nested_features = self._extract_features(value)
                features.extend(nested_features)
                
        return features

# Usage example
async def main():
    kafka_config = {
        'bootstrap.servers': 'localhost:9092',
        'security.protocol': 'SASL_SSL',
        'sasl.mechanism': 'PLAIN',
        'sasl.username': 'your-username',
        'sasl.password': 'your-password'
    }
    
    detector_config = {
        'algorithm': 'isolation_forest',
        'contamination': 0.1,
        'n_estimators': 100,
        'update_frequency': 1000  # Update model every 1000 samples
    }
    
    detector = KafkaAnomalyDetector(
        kafka_config=kafka_config,
        input_topic='sensor-data',
        output_topic='anomaly-alerts',
        detector_config=detector_config
    )
    
    await detector.start_processing()

if __name__ == "__main__":
    asyncio.run(main())
```

## 📊 Monitoring and Observability

### Comprehensive Monitoring Setup

```python
import time
import functools
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

class MonitoringInstrumentation:
    """Comprehensive monitoring and observability for Pynomaly."""
    
    def __init__(self, service_name: str = "pynomaly-api"):
        self.service_name = service_name
        
        # Prometheus metrics
        self.request_count = Counter(
            'pynomaly_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'pynomaly_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.anomalies_detected = Counter(
            'pynomaly_anomalies_detected_total',
            'Total number of anomalies detected',
            ['algorithm', 'dataset']
        )
        
        self.model_inference_time = Histogram(
            'pynomaly_model_inference_seconds',
            'Model inference time in seconds',
            ['algorithm', 'data_size']
        )
        
        self.active_models = Gauge(
            'pynomaly_active_models',
            'Number of active models in memory'
        )
        
        self.data_quality_score = Gauge(
            'pynomaly_data_quality_score',
            'Data quality score (0-1)',
            ['dataset']
        )
        
        # Initialize structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(service_name)
        
        # Initialize distributed tracing
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=14268,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        self.tracer = tracer
        
    def monitor_endpoint(self, method: str, endpoint: str):
        """Decorator for monitoring API endpoints."""
        
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                
                with self.tracer.start_as_current_span(f"{method} {endpoint}") as span:
                    try:
                        # Add span attributes
                        span.set_attribute("http.method", method)
                        span.set_attribute("http.route", endpoint)
                        span.set_attribute("service.name", self.service_name)
                        
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Record successful execution
                        span.set_attribute("http.status_code", 200)
                        
                        return result
                        
                    except Exception as e:
                        status = "error"
                        span.set_attribute("http.status_code", 500)
                        span.set_attribute("error.message", str(e))
                        span.set_attribute("error.type", type(e).__name__)
                        
                        self.logger.error(
                            "Endpoint error",
                            method=method,
                            endpoint=endpoint,
                            error=str(e),
                            exc_info=True
                        )
                        
                        raise
                        
                    finally:
                        # Record metrics
                        duration = time.time() - start_time
                        
                        self.request_count.labels(
                            method=method,
                            endpoint=endpoint,
                            status=status
                        ).inc()
                        
                        self.request_duration.labels(
                            method=method,
                            endpoint=endpoint
                        ).observe(duration)
                        
                        self.logger.info(
                            "Request completed",
                            method=method,
                            endpoint=endpoint,
                            status=status,
                            duration_ms=duration * 1000
                        )
            
            return wrapper
        return decorator
    
    def monitor_model_inference(self, algorithm: str):
        """Decorator for monitoring model inference."""
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Estimate data size
                data_size = "unknown"
                if args and hasattr(args[0], '__len__'):
                    size = len(args[0])
                    if size < 100:
                        data_size = "small"
                    elif size < 1000:
                        data_size = "medium"
                    else:
                        data_size = "large"
                
                with self.tracer.start_as_current_span(f"model_inference_{algorithm}") as span:
                    try:
                        span.set_attribute("ml.algorithm", algorithm)
                        span.set_attribute("ml.data_size", data_size)
                        
                        result = func(*args, **kwargs)
                        
                        # Count anomalies if result contains predictions
                        if hasattr(result, 'predictions'):
                            anomaly_count = sum(1 for p in result.predictions if p == -1)
                            self.anomalies_detected.labels(
                                algorithm=algorithm,
                                dataset="realtime"
                            ).inc(anomaly_count)
                            
                            span.set_attribute("ml.anomalies_detected", anomaly_count)
                        
                        return result
                        
                    except Exception as e:
                        span.set_attribute("error.message", str(e))
                        self.logger.error(
                            "Model inference error",
                            algorithm=algorithm,
                            error=str(e),
                            exc_info=True
                        )
                        raise
                        
                    finally:
                        duration = time.time() - start_time
                        self.model_inference_time.labels(
                            algorithm=algorithm,
                            data_size=data_size
                        ).observe(duration)
            
            return wrapper
        return decorator
    
    def track_data_quality(self, dataset: str, quality_metrics: Dict[str, float]):
        """Track data quality metrics."""
        
        # Calculate overall quality score
        overall_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        self.data_quality_score.labels(dataset=dataset).set(overall_score)
        
        self.logger.info(
            "Data quality assessment",
            dataset=dataset,
            overall_score=overall_score,
            metrics=quality_metrics
        )
        
        # Create span for data quality tracking
        with self.tracer.start_as_current_span("data_quality_check") as span:
            span.set_attribute("dataset.name", dataset)
            span.set_attribute("data.quality_score", overall_score)
            
            for metric, value in quality_metrics.items():
                span.set_attribute(f"data.{metric}", value)
    
    def update_model_count(self, count: int):
        """Update active model count."""
        self.active_models.set(count)
    
    def start_metrics_server(self, port: int = 8001):
        """Start Prometheus metrics server."""
        start_http_server(port)
        self.logger.info(f"Metrics server started on port {port}")

# Usage example
monitoring = MonitoringInstrumentation()

@monitoring.monitor_endpoint("POST", "/api/v1/detect")
async def detect_anomalies_endpoint(request):
    """API endpoint with monitoring."""
    
    @monitoring.monitor_model_inference("isolation_forest")
    def run_detection(data):
        # Model inference logic here
        pass
    
    result = run_detection(request.data)
    return result

# Start metrics server
monitoring.start_metrics_server(port=8001)
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "Pynomaly Production Dashboard",
    "tags": ["pynomaly", "anomaly-detection", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(pynomaly_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(pynomaly_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(pynomaly_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(pynomaly_request_duration_seconds_bucket[5m]))",
            "legendFormat": "99th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Anomalies Detected",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(pynomaly_anomalies_detected_total[1h])",
            "legendFormat": "Last hour"
          }
        ]
      },
      {
        "id": 4,
        "title": "Model Performance",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(pynomaly_model_inference_seconds_bucket[5m])",
            "legendFormat": "{{algorithm}}"
          }
        ]
      },
      {
        "id": 5,
        "title": "Data Quality Score",
        "type": "gauge",
        "targets": [
          {
            "expr": "pynomaly_data_quality_score",
            "legendFormat": "{{dataset}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 1,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 0.7},
                {"color": "green", "value": 0.9}
              ]
            }
          }
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## ⚡ Performance Optimization

### High-Performance Data Processing

```python
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import joblib
from typing import List, Union
import time

class HighPerformanceDetector:
    """Optimized anomaly detector for high-throughput scenarios."""
    
    def __init__(
        self,
        algorithm: str = "isolation_forest",
        n_jobs: int = -1,
        batch_size: int = 1000,
        use_gpu: bool = False
    ):
        self.algorithm = algorithm
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        
        # Initialize thread pools
        self.thread_executor = ThreadPoolExecutor(max_workers=self.n_jobs)
        self.process_executor = ProcessPoolExecutor(max_workers=self.n_jobs)
        
        # Model cache
        self.model_cache = {}
        
    async def parallel_detection(
        self,
        data_batches: List[np.ndarray],
        algorithm_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process multiple data batches in parallel."""
        
        # Create tasks for parallel processing
        tasks = []
        for i, (batch, config) in enumerate(zip(data_batches, algorithm_configs)):
            task = asyncio.create_task(
                self._process_batch_async(i, batch, config)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'batch_id': i,
                    'error': str(result),
                    'predictions': None
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_batch_async(
        self,
        batch_id: int,
        data: np.ndarray,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process single batch asynchronously."""
        
        loop = asyncio.get_event_loop()
        
        # Offload CPU-intensive work to thread pool
        result = await loop.run_in_executor(
            self.thread_executor,
            self._process_batch_sync,
            batch_id,
            data,
            config
        )
        
        return result
    
    def _process_batch_sync(
        self,
        batch_id: int,
        data: np.ndarray,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synchronous batch processing."""
        
        start_time = time.time()
        
        # Get or create model
        model_key = self._get_model_key(config)
        
        if model_key not in self.model_cache:
            self.model_cache[model_key] = self._create_model(config)
        
        model = self.model_cache[model_key]
        
        # Process data in chunks if large
        if len(data) > self.batch_size:
            chunk_results = []
            for i in range(0, len(data), self.batch_size):
                chunk = data[i:i + self.batch_size]
                chunk_predictions = model.predict(chunk)
                chunk_results.extend(chunk_predictions)
            predictions = np.array(chunk_results)
        else:
            predictions = model.predict(data)
        
        processing_time = time.time() - start_time
        
        return {
            'batch_id': batch_id,
            'predictions': predictions.tolist(),
            'anomaly_count': np.sum(predictions == -1),
            'processing_time_ms': processing_time * 1000,
            'data_size': len(data)
        }
    
    def _get_model_key(self, config: Dict[str, Any]) -> str:
        """Generate cache key for model configuration."""
        
        # Sort config for consistent key generation
        sorted_config = sorted(config.items())
        return f"{self.algorithm}_{hash(str(sorted_config))}"
    
    def _create_model(self, config: Dict[str, Any]):
        """Create model instance based on configuration."""
        
        if self.algorithm == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            return IsolationForest(**config)
        elif self.algorithm == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            return LocalOutlierFactor(**config)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    async def streaming_detection(
        self,
        data_stream: AsyncIterator[np.ndarray],
        model_config: Dict[str, Any],
        update_frequency: int = 1000
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process streaming data with periodic model updates."""
        
        model = self._create_model(model_config)
        sample_count = 0
        buffer = []
        
        async for data_point in data_stream:
            # Add to buffer
            buffer.append(data_point)
            sample_count += 1
            
            # Process when buffer is full
            if len(buffer) >= self.batch_size:
                batch_data = np.array(buffer)
                
                # Predict
                predictions = model.predict(batch_data)
                
                # Yield results
                for i, (point, pred) in enumerate(zip(buffer, predictions)):
                    yield {
                        'data': point.tolist(),
                        'prediction': pred,
                        'sample_id': sample_count - len(buffer) + i,
                        'is_anomaly': pred == -1
                    }
                
                # Update model periodically
                if sample_count % update_frequency == 0:
                    await self._update_model_async(model, batch_data)
                
                # Clear buffer
                buffer = []
    
    async def _update_model_async(self, model, recent_data: np.ndarray):
        """Update model with recent data asynchronously."""
        
        loop = asyncio.get_event_loop()
        
        # Offload model update to thread pool
        await loop.run_in_executor(
            self.thread_executor,
            model.fit,
            recent_data
        )
    
    def cleanup(self):
        """Clean up resources."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

# GPU-Accelerated Detection (using RAPIDS cuML)
class GPUAnomalyDetector:
    """GPU-accelerated anomaly detection using RAPIDS cuML."""
    
    def __init__(self):
        try:
            import cuml
            self.cuml_available = True
            self.cuml = cuml
        except ImportError:
            self.cuml_available = False
            print("Warning: cuML not available. GPU acceleration disabled.")
    
    def detect_gpu(self, data: np.ndarray, algorithm: str = "isolation_forest") -> np.ndarray:
        """Perform GPU-accelerated anomaly detection."""
        
        if not self.cuml_available:
            raise RuntimeError("cuML not available for GPU acceleration")
        
        # Convert to cuDF DataFrame for GPU processing
        import cudf
        gpu_data = cudf.DataFrame(data)
        
        if algorithm == "isolation_forest":
            from cuml.ensemble import IsolationForest
            detector = IsolationForest(n_estimators=100)
            predictions = detector.fit_predict(gpu_data)
        elif algorithm == "dbscan":
            from cuml.cluster import DBSCAN
            detector = DBSCAN(eps=0.5, min_samples=5)
            clusters = detector.fit_predict(gpu_data)
            # Convert clusters to anomaly predictions (outliers = -1)
            predictions = np.where(clusters == -1, -1, 1)
        else:
            raise ValueError(f"GPU algorithm {algorithm} not supported")
        
        # Convert back to CPU numpy array
        return predictions.to_numpy() if hasattr(predictions, 'to_numpy') else predictions

# Example usage
async def performance_demo():
    """Demonstrate high-performance detection capabilities."""
    
    # Generate sample data
    np.random.seed(42)
    data_batches = [
        np.random.randn(1000, 10) for _ in range(10)
    ]
    
    configs = [
        {'contamination': 0.1, 'n_estimators': 100, 'random_state': i}
        for i in range(10)
    ]
    
    # High-performance detection
    detector = HighPerformanceDetector(n_jobs=4, batch_size=500)
    
    start_time = time.time()
    results = await detector.parallel_detection(data_batches, configs)
    processing_time = time.time() - start_time
    
    print(f"Processed {len(data_batches)} batches in {processing_time:.2f}s")
    
    total_anomalies = sum(r.get('anomaly_count', 0) for r in results)
    print(f"Total anomalies detected: {total_anomalies}")
    
    detector.cleanup()

if __name__ == "__main__":
    asyncio.run(performance_demo())
```

## 🔒 Security and Compliance

### Security Implementation

```python
import hashlib
import hmac
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from cryptography.fernet import Fernet

class SecurityManager:
    """Comprehensive security management for Pynomaly."""
    
    def __init__(self, secret_key: str, encryption_key: Optional[str] = None):
        self.secret_key = secret_key
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.security = HTTPBearer()
        
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_api_key(self, user_id: str, scopes: List[str] = None) -> str:
        """Create secure API key."""
        
        # Generate random component
        random_part = secrets.token_urlsafe(32)
        
        # Create payload
        payload = {
            'user_id': user_id,
            'scopes': scopes or ['read'],
            'created_at': datetime.utcnow().isoformat(),
            'key_id': secrets.token_hex(8)
        }
        
        # Sign with HMAC
        message = f"{random_part}.{jwt.encode(payload, self.secret_key)}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"pk_{message}.{signature}"
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify and decode API key."""
        
        if not api_key.startswith('pk_'):
            raise HTTPException(status_code=401, detail="Invalid API key format")
        
        try:
            # Remove prefix
            key_content = api_key[3:]
            
            # Split components
            parts = key_content.rsplit('.', 1)
            if len(parts) != 2:
                raise HTTPException(status_code=401, detail="Invalid API key structure")
            
            message, signature = parts
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                raise HTTPException(status_code=401, detail="Invalid API key signature")
            
            # Decode payload
            random_part, jwt_token = message.split('.', 1)
            payload = jwt.decode(jwt_token, self.secret_key, algorithms=['HS256'])
            
            return payload
            
        except (ValueError, jwt.InvalidTokenError) as e:
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def rate_limit_check(self, key: str, limit: int, window: int) -> bool:
        """Simple in-memory rate limiting."""
        # In production, use Redis for distributed rate limiting
        current_time = datetime.utcnow()
        # Implementation would track requests per key
        return True  # Simplified for example
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> Dict[str, Any]:
        """Extract current user from token."""
        
        token = credentials.credentials
        
        try:
            # Verify API key
            user_info = self.verify_api_key(token)
            
            # Check rate limits
            if not self.rate_limit_check(
                f"user:{user_info['user_id']}", 
                limit=100, 
                window=3600
            ):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return user_info
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=401, detail="Authentication failed")

# Data Privacy and Compliance
class DataPrivacyManager:
    """Handle data privacy and compliance requirements."""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'(\+?\d{1,3})?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect personally identifiable information in text."""
        
        import re
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive data for analysis."""
        
        anonymized = data.copy()
        
        # Remove direct identifiers
        sensitive_fields = ['name', 'email', 'phone', 'address', 'ssn']
        for field in sensitive_fields:
            if field in anonymized:
                anonymized[field] = self._hash_field(str(anonymized[field]))
        
        # Generalize quasi-identifiers
        if 'age' in anonymized:
            age = anonymized['age']
            if isinstance(age, int):
                anonymized['age_group'] = self._get_age_group(age)
                del anonymized['age']
        
        if 'zip_code' in anonymized:
            zip_code = str(anonymized['zip_code'])
            anonymized['zip_code'] = zip_code[:3] + 'XX'  # Generalize to 3-digit
        
        return anonymized
    
    def _hash_field(self, value: str) -> str:
        """Hash sensitive field value."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def _get_age_group(self, age: int) -> str:
        """Convert age to age group."""
        if age < 18:
            return "under_18"
        elif age < 30:
            return "18_29"
        elif age < 50:
            return "30_49"
        elif age < 65:
            return "50_64"
        else:
            return "65_plus"
    
    def audit_log_entry(self, action: str, user_id: str, data_summary: Dict[str, Any]):
        """Create audit log entry for compliance."""
        
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'user_id': user_id,
            'data_summary': data_summary,
            'ip_address': 'masked',  # Would get from request
            'user_agent': 'masked'   # Would get from request
        }
        
        # In production, store in secure audit database
        print(f"AUDIT: {audit_entry}")

# Example usage
security_manager = SecurityManager(secret_key="your-secret-key")
privacy_manager = DataPrivacyManager()

# Create and verify API key
api_key = security_manager.create_api_key("user123", ["read", "write"])
user_info = security_manager.verify_api_key(api_key)
print(f"User info: {user_info}")

# Anonymize data
sensitive_data = {
    "name": "John Doe",
    "email": "john@example.com",
    "age": 32,
    "zip_code": "12345"
}

anonymized = privacy_manager.anonymize_data(sensitive_data)
print(f"Anonymized: {anonymized}")
```

## 📋 Practice Exercises

### Exercise 1: Production Deployment (45 minutes)

```bash
# Set up complete production environment
# 1. Configure Docker Compose with all services
# 2. Set up NGINX load balancing
# 3. Configure SSL certificates
# 4. Set up monitoring with Prometheus/Grafana
# 5. Test high availability and failover

# Your implementation here...
```

### Exercise 2: API Performance Testing (60 minutes)

```python
# Create comprehensive performance test suite
# 1. Load testing with multiple concurrent users
# 2. Stress testing with increasing load
# 3. Memory leak detection
# 4. Database connection pool optimization
# 5. Caching strategy implementation

# Your implementation here...
```

### Exercise 3: MLOps Pipeline (90 minutes)

```python
# Implement complete MLOps pipeline
# 1. Model training automation
# 2. Model versioning and registry
# 3. Automated testing and validation
# 4. Canary deployments
# 5. Model performance monitoring
# 6. Automated rollback on degradation

# Your implementation here...
```

## 🚀 Next Steps

### Advanced Learning Paths

1. **DevOps Track**: Continue with [DevOps Engineer Guide](devops-engineer.md)
2. **Architecture Track**: Learn about [System Architecture](../../../developer-guides/architecture/overview.md)
3. **Scaling Track**: Master [Performance Optimization](../../advanced-features/performance-optimization.md)

### Recommended Tools and Frameworks

- **Container Orchestration**: Kubernetes, Docker Swarm
- **Service Mesh**: Istio, Linkerd
- **CI/CD**: GitLab CI, GitHub Actions, Jenkins
- **Monitoring**: Datadog, New Relic, Elastic APM
- **MLOps**: MLflow, Kubeflow, Weights & Biases

### Community Resources

- **ML Engineering Slack**: `#ml-engineering`
- **Office Hours**: Thursdays 10-11 AM PST
- **GitHub Discussions**: [MLOps Category](https://github.com/your-org/pynomaly/discussions/categories/mlops)

---

**Congratulations!** You've mastered production deployment and MLOps for Pynomaly. You're ready to build and maintain world-class anomaly detection systems.

**Continue your journey:**
- 🔒 **[Security Specialist Track](../../../security/README.md)** - Deep-dive into security
- 📊 **[Business Analyst Guide](business-analyst.md)** - Learn business applications
- 🏆 **[Certification Program](../certification.md)** - Earn your ML Engineering certification