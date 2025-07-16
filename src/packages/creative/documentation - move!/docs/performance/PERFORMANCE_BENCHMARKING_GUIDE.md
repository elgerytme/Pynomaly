# Performance Benchmarking Guide

## Overview

This guide provides comprehensive instructions for performance testing, benchmarking, and optimization of the Pynomaly anomaly detection platform. It covers testing methodologies, benchmarking procedures, optimization techniques, and troubleshooting approaches.

## Performance Testing Framework

### 1. Testing Categories

#### 1.1 Load Testing
- **Purpose**: Evaluate system performance under expected load conditions
- **Metrics**: Response time, throughput, resource utilization
- **Tools**: pytest-benchmark, locust, Apache JMeter
- **Target**: 95th percentile response time < 500ms

#### 1.2 Stress Testing
- **Purpose**: Determine system breaking point and failure behavior
- **Metrics**: Maximum concurrent users, failure rate, recovery time
- **Tools**: Custom stress testing scripts, load generators
- **Target**: Graceful degradation under extreme load

#### 1.3 Volume Testing
- **Purpose**: Test system performance with large datasets
- **Metrics**: Processing time, memory usage, disk I/O
- **Tools**: Custom dataset generators, memory profilers
- **Target**: Linear scaling with dataset size

#### 1.4 Endurance Testing
- **Purpose**: Evaluate system stability over extended periods
- **Metrics**: Memory leaks, resource accumulation, performance degradation
- **Tools**: Long-running test suites, monitoring dashboards
- **Target**: Stable performance over 24+ hours

### 2. Performance Test Environment

#### 2.1 Infrastructure Requirements
```yaml
# docker-compose.performance.yml
version: '3.8'
services:
  pynomaly-api:
    image: pynomaly:latest
    environment:
      - PYNOMALY_ENVIRONMENT=performance
      - PYNOMALY_DATABASE_POOL_SIZE=20
      - PYNOMALY_CACHE_ENABLED=true
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
  
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=pynomaly_perf
      - POSTGRES_USER=pynomaly
      - POSTGRES_PASSWORD=performance_test
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
  
  redis:
    image: redis:7-alpine
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
```

#### 2.2 Test Configuration
```python
# tests/performance/config.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceTestConfig:
    """Configuration for performance tests."""
    
    # Load test parameters
    concurrent_users: int = 100
    test_duration_seconds: int = 300
    ramp_up_time_seconds: int = 30
    
    # Performance thresholds
    max_response_time_ms: int = 500
    max_error_rate_percent: float = 1.0
    min_throughput_rps: int = 50
    
    # Dataset parameters
    small_dataset_size: int = 1000
    medium_dataset_size: int = 10000
    large_dataset_size: int = 100000
    
    # Memory and CPU limits
    max_memory_usage_mb: int = 2048
    max_cpu_usage_percent: float = 80.0
    
    @classmethod
    def from_environment(cls) -> 'PerformanceTestConfig':
        """Load configuration from environment variables."""
        return cls(
            concurrent_users=int(os.getenv('PERF_CONCURRENT_USERS', '100')),
            test_duration_seconds=int(os.getenv('PERF_DURATION_SECONDS', '300')),
            ramp_up_time_seconds=int(os.getenv('PERF_RAMP_UP_SECONDS', '30')),
            max_response_time_ms=int(os.getenv('PERF_MAX_RESPONSE_MS', '500')),
            max_error_rate_percent=float(os.getenv('PERF_MAX_ERROR_RATE', '1.0')),
            min_throughput_rps=int(os.getenv('PERF_MIN_THROUGHPUT_RPS', '50')),
            small_dataset_size=int(os.getenv('PERF_SMALL_DATASET', '1000')),
            medium_dataset_size=int(os.getenv('PERF_MEDIUM_DATASET', '10000')),
            large_dataset_size=int(os.getenv('PERF_LARGE_DATASET', '100000')),
            max_memory_usage_mb=int(os.getenv('PERF_MAX_MEMORY_MB', '2048')),
            max_cpu_usage_percent=float(os.getenv('PERF_MAX_CPU_PERCENT', '80.0')),
        )
```

## Benchmarking Procedures

### 3. Core Component Benchmarks

#### 3.1 Detection Algorithm Benchmarks
```python
# tests/performance/test_detection_performance.py
import pytest
import time
import numpy as np
from typing import Dict, List

from pynomaly.domain.services.detection_service import DetectionService
from pynomaly.domain.entities.detector import Detector
from tests.performance.config import PerformanceTestConfig

class TestDetectionPerformance:
    """Performance tests for detection algorithms."""
    
    @pytest.fixture
    def config(self):
        return PerformanceTestConfig.from_environment()
    
    @pytest.fixture
    def detection_service(self):
        return DetectionService()
    
    @pytest.fixture
    def sample_datasets(self, config):
        """Generate sample datasets of different sizes."""
        return {
            'small': np.random.randn(config.small_dataset_size, 10),
            'medium': np.random.randn(config.medium_dataset_size, 10),
            'large': np.random.randn(config.large_dataset_size, 10)
        }
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("algorithm", [
        "IsolationForest",
        "LocalOutlierFactor",
        "OneClassSVM",
        "EllipticEnvelope"
    ])
    def test_detection_algorithm_performance(
        self, 
        benchmark, 
        algorithm, 
        detection_service,
        sample_datasets,
        config
    ):
        """Benchmark detection algorithm performance."""
        
        detector = Detector(
            name=f"test-{algorithm}",
            algorithm=algorithm,
            parameters=self._get_default_parameters(algorithm)
        )
        
        def run_detection():
            return detection_service.detect_anomalies(
                detector=detector,
                data=sample_datasets['medium']
            )
        
        # Run benchmark
        result = benchmark(run_detection)
        
        # Validate performance
        assert result.execution_time < config.max_response_time_ms / 1000
        assert result.memory_usage < config.max_memory_usage_mb
        
        # Log results
        self._log_performance_metrics(algorithm, result)
    
    def _get_default_parameters(self, algorithm: str) -> Dict:
        """Get default parameters for algorithms."""
        defaults = {
            "IsolationForest": {"contamination": 0.1, "n_estimators": 100},
            "LocalOutlierFactor": {"contamination": 0.1, "n_neighbors": 20},
            "OneClassSVM": {"nu": 0.1, "kernel": "rbf"},
            "EllipticEnvelope": {"contamination": 0.1}
        }
        return defaults.get(algorithm, {})
    
    def _log_performance_metrics(self, algorithm: str, result):
        """Log performance metrics for analysis."""
        metrics = {
            "algorithm": algorithm,
            "execution_time": result.execution_time,
            "memory_usage": result.memory_usage,
            "cpu_usage": result.cpu_usage,
            "timestamp": time.time()
        }
        
        # Log to performance monitoring system
        # This would integrate with your monitoring infrastructure
        print(f"Performance metrics for {algorithm}: {metrics}")
```

#### 3.2 Training Performance Benchmarks
```python
# tests/performance/test_training_performance.py
import pytest
import numpy as np
from typing import Dict, List

from pynomaly.domain.services.training_service import TrainingService
from pynomaly.domain.entities.training_job import TrainingJob
from tests.performance.config import PerformanceTestConfig

class TestTrainingPerformance:
    """Performance tests for model training."""
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("dataset_size", [1000, 10000, 50000])
    def test_training_scalability(
        self, 
        benchmark, 
        dataset_size,
        training_service,
        config
    ):
        """Test training performance with different dataset sizes."""
        
        # Generate dataset
        data = np.random.randn(dataset_size, 20)
        
        def run_training():
            return training_service.train_model(
                algorithm="IsolationForest",
                data=data,
                parameters={"contamination": 0.1}
            )
        
        # Run benchmark
        result = benchmark(run_training)
        
        # Validate scalability
        expected_time = self._calculate_expected_training_time(dataset_size)
        assert result.execution_time < expected_time
        
        # Log scaling metrics
        self._log_scaling_metrics(dataset_size, result)
    
    def _calculate_expected_training_time(self, dataset_size: int) -> float:
        """Calculate expected training time based on dataset size."""
        # Linear scaling assumption: 1ms per 10 samples
        return (dataset_size / 10) * 0.001
    
    def _log_scaling_metrics(self, dataset_size: int, result):
        """Log scaling metrics for analysis."""
        metrics = {
            "dataset_size": dataset_size,
            "training_time": result.execution_time,
            "memory_usage": result.memory_usage,
            "scaling_factor": result.execution_time / (dataset_size / 1000),
            "timestamp": time.time()
        }
        print(f"Scaling metrics: {metrics}")
```

#### 3.3 API Performance Benchmarks
```python
# tests/performance/test_api_performance.py
import pytest
import asyncio
import aiohttp
import time
from typing import Dict, List

from tests.performance.config import PerformanceTestConfig

class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.fixture
    def config(self):
        return PerformanceTestConfig.from_environment()
    
    @pytest.fixture
    def api_client(self):
        return aiohttp.ClientSession()
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_api_endpoint_performance(
        self, 
        benchmark, 
        api_client,
        config
    ):
        """Test API endpoint performance under load."""
        
        endpoints = [
            "/api/v1/detectors",
            "/api/v1/datasets",
            "/api/v1/health",
            "/api/v1/metrics"
        ]
        
        async def run_api_load_test():
            tasks = []
            for endpoint in endpoints:
                for _ in range(config.concurrent_users):
                    task = self._make_api_request(api_client, endpoint)
                    tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return self._analyze_responses(responses)
        
        # Run benchmark
        result = benchmark(run_api_load_test)
        
        # Validate performance
        assert result.average_response_time < config.max_response_time_ms / 1000
        assert result.error_rate < config.max_error_rate_percent / 100
        assert result.throughput > config.min_throughput_rps
    
    async def _make_api_request(self, client, endpoint):
        """Make API request and measure performance."""
        start_time = time.time()
        try:
            async with client.get(f"http://localhost:8000{endpoint}") as response:
                await response.text()
                return {
                    "status": response.status,
                    "response_time": time.time() - start_time,
                    "endpoint": endpoint
                }
        except Exception as e:
            return {
                "status": 500,
                "response_time": time.time() - start_time,
                "endpoint": endpoint,
                "error": str(e)
            }
    
    def _analyze_responses(self, responses):
        """Analyze API responses for performance metrics."""
        successful_responses = [r for r in responses if r.get("status", 500) < 400]
        
        if not successful_responses:
            return {
                "average_response_time": float('inf'),
                "error_rate": 1.0,
                "throughput": 0
            }
        
        total_time = sum(r["response_time"] for r in successful_responses)
        average_response_time = total_time / len(successful_responses)
        error_rate = (len(responses) - len(successful_responses)) / len(responses)
        throughput = len(successful_responses) / total_time
        
        return {
            "average_response_time": average_response_time,
            "error_rate": error_rate,
            "throughput": throughput
        }
```

### 4. Load Testing Framework

#### 4.1 Locust Load Testing
```python
# tests/performance/locust_load_test.py
from locust import HttpUser, task, between
import random
import json

class PynomályUser(HttpUser):
    """Locust user for load testing Pynomaly API."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session."""
        self.login()
    
    def login(self):
        """Authenticate user."""
        response = self.client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "testpassword"
        })
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task(3)
    def get_detectors(self):
        """Get list of detectors."""
        self.client.get("/api/v1/detectors")
    
    @task(2)
    def get_datasets(self):
        """Get list of datasets."""
        self.client.get("/api/v1/datasets")
    
    @task(1)
    def run_detection(self):
        """Run anomaly detection."""
        detector_id = self._get_random_detector_id()
        data = self._generate_sample_data()
        
        self.client.post(f"/api/v1/detectors/{detector_id}/detect", json={
            "data": data
        })
    
    @task(1)
    def get_metrics(self):
        """Get performance metrics."""
        self.client.get("/api/v1/metrics")
    
    def _get_random_detector_id(self):
        """Get random detector ID for testing."""
        return f"detector_{random.randint(1, 10)}"
    
    def _generate_sample_data(self):
        """Generate sample data for detection."""
        return [[random.gauss(0, 1) for _ in range(10)] for _ in range(100)]
```

#### 4.2 Load Testing Execution
```bash
#!/bin/bash
# scripts/run_load_test.sh

# Performance load testing script
set -e

echo "Starting Pynomaly performance load test..."

# Configuration
USERS=${USERS:-100}
SPAWN_RATE=${SPAWN_RATE:-10}
DURATION=${DURATION:-300}
HOST=${HOST:-http://localhost:8000}

# Start services
echo "Starting services..."
docker-compose -f docker-compose.performance.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Run load test
echo "Running load test with $USERS users for $DURATION seconds..."
locust -f tests/performance/locust_load_test.py \
    --users=$USERS \
    --spawn-rate=$SPAWN_RATE \
    --run-time=${DURATION}s \
    --host=$HOST \
    --html=reports/performance/load_test_report.html \
    --csv=reports/performance/load_test_results

# Generate performance report
echo "Generating performance report..."
python scripts/generate_performance_report.py \
    --input=reports/performance/load_test_results_stats.csv \
    --output=reports/performance/performance_summary.json

# Cleanup
echo "Cleaning up..."
docker-compose -f docker-compose.performance.yml down

echo "Load test completed. Results available in reports/performance/"
```

## Performance Monitoring and Observability

### 5. Monitoring Setup

#### 5.1 Prometheus Metrics
```python
# src/pynomaly/infrastructure/monitoring/performance_metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from typing import Dict, Any

class PerformanceMetrics:
    """Prometheus metrics for performance monitoring."""
    
    def __init__(self):
        # Request metrics
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
        
        # Detection metrics
        self.detection_duration = Histogram(
            'pynomaly_detection_duration_seconds',
            'Detection duration in seconds',
            ['algorithm']
        )
        
        self.detection_count = Counter(
            'pynomaly_detections_total',
            'Total number of detections',
            ['algorithm', 'status']
        )
        
        # Training metrics
        self.training_duration = Histogram(
            'pynomaly_training_duration_seconds',
            'Training duration in seconds',
            ['algorithm']
        )
        
        self.training_count = Counter(
            'pynomaly_training_jobs_total',
            'Total number of training jobs',
            ['algorithm', 'status']
        )
        
        # System metrics
        self.active_connections = Gauge(
            'pynomaly_active_connections',
            'Number of active connections'
        )
        
        self.memory_usage = Gauge(
            'pynomaly_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'pynomaly_cpu_usage_percent',
            'CPU usage percentage'
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record request metrics."""
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_detection(self, algorithm: str, status: str, duration: float):
        """Record detection metrics."""
        self.detection_count.labels(algorithm=algorithm, status=status).inc()
        self.detection_duration.labels(algorithm=algorithm).observe(duration)
    
    def record_training(self, algorithm: str, status: str, duration: float):
        """Record training metrics."""
        self.training_count.labels(algorithm=algorithm, status=status).inc()
        self.training_duration.labels(algorithm=algorithm).observe(duration)
    
    def update_system_metrics(self, connections: int, memory_bytes: int, cpu_percent: float):
        """Update system metrics."""
        self.active_connections.set(connections)
        self.memory_usage.set(memory_bytes)
        self.cpu_usage.set(cpu_percent)
    
    def generate_metrics(self) -> str:
        """Generate Prometheus metrics."""
        return generate_latest()
```

#### 5.2 Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Pynomaly Performance Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(pynomaly_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(pynomaly_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(pynomaly_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Detection Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(pynomaly_detection_duration_seconds_bucket[5m]))",
            "legendFormat": "{{algorithm}} - 95th percentile"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "pynomaly_memory_usage_bytes",
            "legendFormat": "Memory Usage"
          },
          {
            "expr": "pynomaly_cpu_usage_percent",
            "legendFormat": "CPU Usage"
          }
        ]
      }
    ]
  }
}
```

## Performance Optimization Strategies

### 6. Optimization Techniques

#### 6.1 Database Optimization
```python
# Database query optimization
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import sessionmaker

class OptimizedDetectorRepository:
    """Optimized repository with performance improvements."""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(
            connection_string,
            pool_size=20,
            max_overflow=0,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Create optimized indexes
        self._create_performance_indexes()
    
    def _create_performance_indexes(self):
        """Create database indexes for performance."""
        indexes = [
            Index('idx_detector_algorithm', 'detectors.algorithm'),
            Index('idx_detector_status', 'detectors.status'),
            Index('idx_detection_timestamp', 'detections.timestamp'),
            Index('idx_detection_detector_id', 'detections.detector_id'),
            Index('idx_user_tenant_id', 'users.tenant_id'),
        ]
        
        for index in indexes:
            index.create(self.engine, checkfirst=True)
    
    async def get_detectors_optimized(self, limit: int = 100, offset: int = 0):
        """Get detectors with optimized query."""
        query = """
        SELECT d.*, u.email as user_email
        FROM detectors d
        JOIN users u ON d.user_id = u.id
        WHERE d.is_active = true
        ORDER BY d.created_at DESC
        LIMIT :limit OFFSET :offset
        """
        
        async with self.engine.connect() as conn:
            result = await conn.execute(text(query), {"limit": limit, "offset": offset})
            return result.fetchall()
```

#### 6.2 Caching Strategy
```python
# Redis caching for performance
import redis
import json
import pickle
from typing import Any, Optional

class PerformanceCache:
    """Redis-based cache for performance optimization."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour
    
    async def get_detector_results(self, detector_id: str, data_hash: str) -> Optional[Any]:
        """Get cached detection results."""
        cache_key = f"detection:{detector_id}:{data_hash}"
        cached_data = self.redis.get(cache_key)
        
        if cached_data:
            return pickle.loads(cached_data)
        return None
    
    async def cache_detector_results(self, detector_id: str, data_hash: str, results: Any):
        """Cache detection results."""
        cache_key = f"detection:{detector_id}:{data_hash}"
        cached_data = pickle.dumps(results)
        self.redis.setex(cache_key, self.default_ttl, cached_data)
    
    async def get_model_cache(self, model_id: str) -> Optional[Any]:
        """Get cached trained model."""
        cache_key = f"model:{model_id}"
        cached_data = self.redis.get(cache_key)
        
        if cached_data:
            return pickle.loads(cached_data)
        return None
    
    async def cache_model(self, model_id: str, model: Any):
        """Cache trained model."""
        cache_key = f"model:{model_id}"
        cached_data = pickle.dumps(model)
        # Models have longer TTL
        self.redis.setex(cache_key, self.default_ttl * 24, cached_data)
```

#### 6.3 Async Processing
```python
# Asynchronous processing for performance
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

class AsyncDetectionService:
    """Asynchronous detection service for better performance."""
    
    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch_detections(self, detection_requests: List[Dict]) -> List[Any]:
        """Process multiple detections concurrently."""
        tasks = []
        
        for request in detection_requests:
            task = asyncio.create_task(
                self._process_single_detection(request)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _process_single_detection(self, request: Dict) -> Any:
        """Process single detection asynchronously."""
        # CPU-intensive work in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._run_detection_sync,
            request
        )
    
    def _run_detection_sync(self, request: Dict) -> Any:
        """Synchronous detection processing."""
        # This would call the actual detection algorithm
        # Implementation depends on your specific detection logic
        pass
```

## Performance Troubleshooting

### 7. Common Performance Issues

#### 7.1 Database Performance Issues
```python
# Database performance diagnostics
class DatabasePerformanceDiagnostics:
    """Tools for diagnosing database performance issues."""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
    
    async def analyze_slow_queries(self) -> List[Dict]:
        """Analyze slow queries."""
        query = """
        SELECT query, mean_exec_time, calls, total_exec_time
        FROM pg_stat_statements
        WHERE mean_exec_time > 1000
        ORDER BY mean_exec_time DESC
        LIMIT 10
        """
        
        async with self.engine.connect() as conn:
            result = await conn.execute(text(query))
            return [dict(row) for row in result.fetchall()]
    
    async def check_index_usage(self) -> List[Dict]:
        """Check index usage statistics."""
        query = """
        SELECT schemaname, tablename, indexname, 
               idx_scan, idx_tup_read, idx_tup_fetch
        FROM pg_stat_user_indexes
        WHERE idx_scan = 0
        ORDER BY schemaname, tablename
        """
        
        async with self.engine.connect() as conn:
            result = await conn.execute(text(query))
            return [dict(row) for row in result.fetchall()]
    
    async def analyze_table_bloat(self) -> List[Dict]:
        """Analyze table bloat."""
        query = """
        SELECT schemaname, tablename, 
               n_tup_ins, n_tup_upd, n_tup_del,
               n_live_tup, n_dead_tup
        FROM pg_stat_user_tables
        WHERE n_dead_tup > n_live_tup * 0.1
        ORDER BY n_dead_tup DESC
        """
        
        async with self.engine.connect() as conn:
            result = await conn.execute(text(query))
            return [dict(row) for row in result.fetchall()]
```

#### 7.2 Memory Performance Issues
```python
# Memory performance monitoring
import psutil
import tracemalloc
from typing import Dict, Any

class MemoryPerformanceMonitor:
    """Monitor memory performance and detect issues."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        return {
            "rss": memory_info.rss,
            "vms": memory_info.vms,
            "percent": memory_percent,
            "available": psutil.virtual_memory().available,
            "total": psutil.virtual_memory().total
        }
    
    def start_memory_tracing(self):
        """Start memory tracing."""
        tracemalloc.start()
    
    def get_memory_trace(self, limit: int = 10) -> List[Dict]:
        """Get memory trace statistics."""
        if not tracemalloc.is_tracing():
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return [
            {
                "filename": stat.traceback.format()[0],
                "size": stat.size,
                "count": stat.count
            }
            for stat in top_stats[:limit]
        ]
```

### 8. Performance Alerts

#### 8.1 Alert Configuration
```yaml
# alerts/performance_alerts.yml
groups:
  - name: pynomaly_performance
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(pynomaly_request_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"
      
      - alert: HighErrorRate
        expr: rate(pynomaly_requests_total{status=~"5.."}[5m]) / rate(pynomaly_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      - alert: LowThroughput
        expr: rate(pynomaly_requests_total[5m]) < 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low throughput detected"
          description: "Request rate is {{ $value }} requests per second"
      
      - alert: HighMemoryUsage
        expr: pynomaly_memory_usage_bytes > 2000000000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value | humanizeBytes }}"
```

## Running Performance Tests

### 9. Test Execution

#### 9.1 Local Performance Testing
```bash
# Run comprehensive performance test suite
./scripts/run_performance_tests.sh

# Run specific performance tests
pytest tests/performance/test_detection_performance.py -v --benchmark-only

# Run load testing
./scripts/run_load_test.sh

# Generate performance report
python scripts/generate_performance_report.py
```

#### 9.2 CI/CD Integration
```yaml
# .github/workflows/performance-testing.yml
name: Performance Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly performance tests

jobs:
  performance-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e ".[test,performance]"
    
    - name: Start services
      run: |
        docker-compose -f docker-compose.performance.yml up -d
        sleep 30
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-json=performance_results.json
    
    - name: Run load tests
      run: |
        locust -f tests/performance/locust_load_test.py \
          --users=50 --spawn-rate=5 --run-time=60s \
          --host=http://localhost:8000 \
          --csv=load_test_results \
          --headless
    
    - name: Generate performance report
      run: |
        python scripts/generate_performance_report.py \
          --benchmark-results=performance_results.json \
          --load-test-results=load_test_results_stats.csv \
          --output=performance_report.html
    
    - name: Upload performance report
      uses: actions/upload-artifact@v3
      with:
        name: performance-report
        path: performance_report.html
    
    - name: Check performance regression
      run: |
        python scripts/check_performance_regression.py \
          --current-results=performance_results.json \
          --baseline-results=baseline_performance.json
```

## Best Practices

### 10. Performance Best Practices

#### 10.1 Code Optimization
- **Use appropriate data structures**: Choose efficient data structures for your use case
- **Minimize database queries**: Use eager loading and query optimization
- **Implement caching**: Cache frequently accessed data and computation results
- **Use connection pooling**: Optimize database connection management
- **Profile regularly**: Use profiling tools to identify bottlenecks

#### 10.2 Infrastructure Optimization
- **Horizontal scaling**: Scale out rather than up when possible
- **Load balancing**: Distribute traffic across multiple instances
- **CDN usage**: Use content delivery networks for static assets
- **Database optimization**: Proper indexing and query optimization
- **Monitoring**: Continuous performance monitoring and alerting

#### 10.3 Testing Strategy
- **Automated testing**: Include performance tests in CI/CD pipeline
- **Baseline tracking**: Maintain performance baselines for comparison
- **Load testing**: Regular load testing to identify capacity limits
- **Stress testing**: Test system behavior under extreme conditions
- **Real-world scenarios**: Test with realistic data and usage patterns

## Conclusion

This performance benchmarking guide provides comprehensive tools and methodologies for optimizing Pynomaly's performance. Regular performance testing, monitoring, and optimization are essential for maintaining a high-quality anomaly detection platform.

For questions or additional guidance, consult the development team or refer to the performance monitoring dashboard.

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Next Review**: Quarterly  
**Owner**: Performance Engineering Team