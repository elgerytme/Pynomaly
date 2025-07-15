# Performance Benchmarking Guide

## Overview
This comprehensive guide covers performance testing, benchmarking procedures, optimization recommendations, troubleshooting, and monitoring setup for the Pynomaly anomaly detection platform.

## Table of Contents
1. [Performance Testing Framework](#performance-testing-framework)
2. [Benchmarking Procedures](#benchmarking-procedures)
3. [Optimization Recommendations](#optimization-recommendations)
4. [Performance Troubleshooting](#performance-troubleshooting)
5. [Monitoring Setup](#monitoring-setup)
6. [Test Suite](#test-suite)
7. [Performance Baselines](#performance-baselines)

## Performance Testing Framework

### 1. Testing Architecture

#### Core Components
- **Load Testing**: Simulating realistic user loads
- **Stress Testing**: Testing system limits and breaking points
- **Volume Testing**: Handling large datasets and high throughput
- **Endurance Testing**: Long-running performance validation
- **Spike Testing**: Sudden load increases and decreases

#### Tools and Infrastructure
```bash
# Primary tools
pytest-benchmark   # Python function benchmarking
locust            # Load testing framework
memory-profiler   # Memory usage analysis
py-spy           # CPU profiling
cProfile         # Python profiling

# Infrastructure monitoring
prometheus       # Metrics collection
grafana         # Visualization
jaeger          # Distributed tracing
```

### 2. Performance Test Categories

#### API Performance Tests
```python
# Example API performance test
import pytest
import httpx
from pytest_benchmark import benchmark

@pytest.mark.performance
async def test_api_detection_performance(benchmark):
    """Test anomaly detection API performance."""
    
    async def detection_request():
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/v1/detection/predict",
                json={"data": sample_data, "detector_id": "test-detector"}
            )
            return response
    
    result = await benchmark(detection_request)
    assert result.status_code == 200
    # Benchmark automatically captures timing metrics
```

#### ML Algorithm Performance Tests
```python
@pytest.mark.performance
def test_isolation_forest_performance(benchmark):
    """Benchmark Isolation Forest training performance."""
    
    def train_isolation_forest():
        from sklearn.ensemble import IsolationForest
        import numpy as np
        
        # Generate test data
        X = np.random.randn(10000, 20)
        
        # Train model
        model = IsolationForest(n_estimators=100)
        model.fit(X)
        
        # Make predictions
        predictions = model.predict(X[:1000])
        return predictions
    
    result = benchmark(train_isolation_forest)
    assert len(result) == 1000
```

#### Memory Performance Tests
```python
@pytest.mark.performance
@pytest.mark.memory
def test_memory_usage_large_dataset():
    """Test memory usage with large datasets."""
    
    from memory_profiler import profile
    import pandas as pd
    
    @profile
    def process_large_dataset():
        # Create large dataset
        df = pd.DataFrame(np.random.randn(1000000, 50))
        
        # Process data
        result = df.describe()
        return result
    
    # Memory profiling will be captured
    result = process_large_dataset()
    assert result is not None
```

## Benchmarking Procedures

### 1. Baseline Establishment

#### System Requirements
```yaml
# Performance testing environment
environment:
  cpu: "8 cores minimum"
  memory: "16GB minimum"
  storage: "SSD recommended"
  network: "1Gbps minimum"
  
baseline_metrics:
  api_response_time: "< 100ms p95"
  throughput: "> 1000 requests/second"
  memory_usage: "< 2GB peak"
  cpu_usage: "< 80% sustained"
```

#### Baseline Data Collection
```bash
# Run baseline benchmarks
pytest tests/performance/ --benchmark-only --benchmark-json=baseline.json

# Collect system metrics
python scripts/collect_system_baseline.py

# Generate baseline report
python scripts/generate_baseline_report.py
```

### 2. Performance Test Execution

#### Automated Test Runs
```bash
# Run full performance suite
make test-performance

# Run specific performance categories
pytest -m "performance and api" --benchmark-compare=baseline.json
pytest -m "performance and ml" --benchmark-compare=baseline.json
pytest -m "performance and memory" --benchmark-compare=baseline.json

# Run with profiling
pytest -m performance --profile --profile-svg
```

#### Load Testing with Locust
```python
# locustfile.py - Load testing configuration
from locust import HttpUser, task, between

class PynormalyUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login user at start."""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        self.token = response.json()["access_token"]
        self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task(3)
    def detect_anomalies(self):
        """Main detection task - weighted higher."""
        self.client.post("/api/v1/detection/predict", json={
            "detector_id": "test-detector",
            "data": [[1, 2, 3, 4, 5]]
        })
    
    @task(1)
    def get_detectors(self):
        """List detectors task."""
        self.client.get("/api/v1/detectors")
    
    @task(1)
    def get_health(self):
        """Health check task."""
        self.client.get("/api/v1/health")
```

```bash
# Run load tests
locust -f locustfile.py --host=http://localhost:8000
locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 10m --html=report.html
```

### 3. Performance Metrics Collection

#### Key Performance Indicators (KPIs)
```yaml
api_metrics:
  response_time:
    p50: "< 50ms"
    p95: "< 100ms"
    p99: "< 200ms"
  throughput: "> 1000 RPS"
  error_rate: "< 0.1%"

ml_metrics:
  training_time:
    small_dataset: "< 30s (1K samples)"
    medium_dataset: "< 5m (100K samples)"
    large_dataset: "< 30m (1M samples)"
  prediction_time:
    single: "< 10ms"
    batch_1k: "< 100ms"
    batch_10k: "< 1s"

resource_metrics:
  memory_usage:
    api_server: "< 1GB"
    ml_training: "< 8GB"
    peak_usage: "< 12GB"
  cpu_usage:
    sustained: "< 70%"
    peak: "< 90%"
  disk_io:
    read_iops: "> 1000"
    write_iops: "> 500"
```

#### Metrics Collection Script
```python
# scripts/collect_performance_metrics.py
import psutil
import time
import json
from datetime import datetime

class PerformanceCollector:
    def __init__(self):
        self.metrics = []
    
    def collect_system_metrics(self):
        """Collect system-level metrics."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "usage": psutil.disk_usage('/').percent,
                "io": psutil.disk_io_counters()._asdict()
            },
            "network": psutil.net_io_counters()._asdict()
        }
    
    def start_collection(self, duration_seconds=300):
        """Start collecting metrics for specified duration."""
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            metrics = self.collect_system_metrics()
            self.metrics.append(metrics)
            time.sleep(5)  # Collect every 5 seconds
    
    def save_metrics(self, filename):
        """Save collected metrics to file."""
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)

# Usage
if __name__ == "__main__":
    collector = PerformanceCollector()
    collector.start_collection(600)  # 10 minutes
    collector.save_metrics("performance_metrics.json")
```

## Optimization Recommendations

### 1. API Performance Optimization

#### Response Time Optimization
```python
# Use async/await for I/O operations
async def optimized_endpoint():
    # Good: Concurrent database queries
    async with database.transaction():
        user_task = database.fetch_user(user_id)
        data_task = database.fetch_data(data_id)
        
        user, data = await asyncio.gather(user_task, data_task)
    
    return {"user": user, "data": data}

# Implement response caching
from functools import lru_cache
from redis import Redis

@lru_cache(maxsize=1000)
def cached_computation(input_data):
    # Expensive computation
    return complex_calculation(input_data)

# Use Redis for distributed caching
redis_client = Redis()

async def cached_api_response(key: str):
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    
    result = await expensive_operation()
    redis_client.setex(key, 300, json.dumps(result))  # 5-minute cache
    return result
```

#### Database Optimization
```python
# Use connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Optimize queries with indexing
# SQL example:
CREATE INDEX idx_anomaly_detector_id ON anomaly_results(detector_id);
CREATE INDEX idx_anomaly_timestamp ON anomaly_results(timestamp);
CREATE INDEX idx_composite ON anomaly_results(detector_id, timestamp);

# Use query optimization
def optimized_query():
    # Good: Select only needed columns
    query = select([
        models.AnomalyResult.id,
        models.AnomalyResult.score,
        models.AnomalyResult.timestamp
    ]).where(
        models.AnomalyResult.detector_id == detector_id
    ).order_by(
        models.AnomalyResult.timestamp.desc()
    ).limit(100)
    
    return query
```

### 2. ML Algorithm Optimization

#### Algorithm Selection
```python
# Choose algorithms based on dataset size and requirements
def select_optimal_algorithm(dataset_size, real_time_requirement):
    if real_time_requirement and dataset_size < 10000:
        return "LocalOutlierFactor"  # Fast for small datasets
    elif dataset_size > 1000000:
        return "IsolationForest"     # Scales well with large data
    elif dataset_size > 100000:
        return "OneClassSVM"         # Good balance
    else:
        return "EllipticEnvelope"    # Good for medium datasets

# Parameter optimization
from sklearn.model_selection import GridSearchCV

def optimize_isolation_forest_params(X):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'contamination': [0.1, 0.15, 0.2],
        'max_features': [0.5, 0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        IsolationForest(),
        param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )
    
    return grid_search.fit(X)
```

#### Memory Optimization
```python
# Use data streaming for large datasets
def stream_large_dataset(file_path, chunk_size=10000):
    """Process large datasets in chunks."""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        processed_chunk = preprocess_data(chunk)
        yield processed_chunk

# Memory-efficient model training
def memory_efficient_training(data_stream, model):
    """Train model using data streaming."""
    for chunk in data_stream:
        # Partial fit for online learning algorithms
        if hasattr(model, 'partial_fit'):
            model.partial_fit(chunk)
        else:
            # Accumulate gradients for batch training
            model.fit(chunk, warm_start=True)
    
    return model
```

### 3. Infrastructure Optimization

#### Containerization
```dockerfile
# Optimized Dockerfile
FROM python:3.11-slim

# Use multi-stage builds
FROM python:3.11 as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local

# Optimize Python settings
ENV PYTHONOPTIMIZE=2
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Use specific versions and minimal installations
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY src/ /app/src/
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "src.pynomaly.presentation.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Load Balancing
```yaml
# nginx.conf optimization
upstream pynomaly_backend {
    least_conn;
    server pynomaly1:8000 max_fails=3 fail_timeout=30s;
    server pynomaly2:8000 max_fails=3 fail_timeout=30s;
    server pynomaly3:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    
    # Connection optimization
    keepalive_timeout 65;
    keepalive_requests 100;
    
    # Compression
    gzip on;
    gzip_comp_level 6;
    gzip_types application/json application/javascript text/css;
    
    # Caching
    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    location /api/ {
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Connection pooling
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
}
```

## Performance Troubleshooting

### 1. Diagnostic Tools

#### Performance Profiling
```bash
# CPU profiling with py-spy
py-spy top --pid $PID                    # Real-time CPU usage
py-spy record -o profile.svg --pid $PID  # Generate flame graph
py-spy dump --pid $PID                   # Stack trace snapshot

# Memory profiling
python -m memory_profiler your_script.py
mprof run your_script.py
mprof plot

# Application profiling with cProfile
python -m cProfile -o profile.prof your_script.py
python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"
```

#### Database Performance Analysis
```sql
-- PostgreSQL query analysis
EXPLAIN ANALYZE SELECT * FROM anomaly_results WHERE detector_id = 'uuid';

-- Check slow queries
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE tablename = 'anomaly_results';
```

### 2. Common Performance Issues

#### Issue: High API Response Times
```python
# Diagnostic checklist
async def diagnose_api_performance():
    """Diagnose API performance issues."""
    
    # 1. Check database connection pool
    pool_stats = await database.get_pool_stats()
    if pool_stats.checked_out / pool_stats.size > 0.8:
        print("WARNING: Database pool nearly exhausted")
    
    # 2. Check cache hit rate
    cache_stats = await redis_client.info('stats')
    hit_rate = cache_stats['keyspace_hits'] / (cache_stats['keyspace_hits'] + cache_stats['keyspace_misses'])
    if hit_rate < 0.8:
        print(f"WARNING: Low cache hit rate: {hit_rate:.2%}")
    
    # 3. Check memory usage
    import psutil
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        print(f"WARNING: High memory usage: {memory.percent:.1f}%")
    
    # 4. Check CPU usage
    cpu = psutil.cpu_percent(interval=1)
    if cpu > 80:
        print(f"WARNING: High CPU usage: {cpu:.1f}%")

# Solutions
def optimize_api_performance():
    """Apply common API optimizations."""
    
    # Add response caching
    @lru_cache(maxsize=1000)
    def cache_expensive_computation(input_hash):
        return expensive_computation(input_hash)
    
    # Use async I/O
    async def optimized_handler():
        tasks = [
            fetch_data_async(),
            process_data_async(),
            validate_data_async()
        ]
        results = await asyncio.gather(*tasks)
        return results
    
    # Implement connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True
    )
```

#### Issue: Memory Leaks
```python
# Memory leak detection
import tracemalloc

def detect_memory_leaks():
    """Detect memory leaks in application."""
    
    # Start tracing
    tracemalloc.start()
    
    # Run your application code
    run_application_code()
    
    # Take snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)

# Memory optimization
def optimize_memory_usage():
    """Apply memory optimizations."""
    
    # Use generators for large datasets
    def process_large_data():
        for chunk in read_data_chunks():
            yield process_chunk(chunk)
    
    # Clear unused variables
    import gc
    del large_object
    gc.collect()
    
    # Use __slots__ for classes
    class OptimizedClass:
        __slots__ = ['attr1', 'attr2']
        
        def __init__(self, attr1, attr2):
            self.attr1 = attr1
            self.attr2 = attr2
```

### 3. Performance Monitoring Alerts

#### Alert Configuration
```yaml
# alerts.yml - Prometheus alert rules
groups:
- name: pynomaly_performance
  rules:
  - alert: HighAPIResponseTime
    expr: histogram_quantile(0.95, http_request_duration_seconds) > 0.5
    for: 5m
    annotations:
      summary: "High API response time detected"
      description: "95th percentile response time is {{ $value }}s"
  
  - alert: HighMemoryUsage
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) > 0.85
    for: 2m
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value | humanizePercentage }}"
  
  - alert: HighCPUUsage
    expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value | humanizePercentage }}"
```

## Monitoring Setup

### 1. Metrics Collection

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'pynomaly-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

#### Application Metrics
```python
# metrics.py - Custom application metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
REQUEST_COUNT = Counter('pynomaly_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('pynomaly_request_duration_seconds', 'Request duration')
ACTIVE_DETECTORS = Gauge('pynomaly_active_detectors', 'Number of active detectors')
TRAINING_JOBS = Gauge('pynomaly_training_jobs', 'Number of training jobs')

# Middleware to collect metrics
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 2. Grafana Dashboards

#### Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Pynomaly Performance Dashboard",
    "panels": [
      {
        "title": "API Response Time",
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
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(pynomaly_requests_total[5m])",
            "legendFormat": "Requests per second"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes",
            "legendFormat": "Memory usage"
          }
        ]
      },
      {
        "title": "Active ML Models",
        "type": "singlestat",
        "targets": [
          {
            "expr": "pynomaly_active_detectors",
            "legendFormat": "Active detectors"
          }
        ]
      }
    ]
  }
}
```

### 3. Distributed Tracing

#### Jaeger Integration
```python
# tracing.py - Distributed tracing setup
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing():
    """Set up distributed tracing with Jaeger."""
    
    # Configure tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    return tracer

# Usage in application
tracer = setup_tracing()

async def traced_function():
    with tracer.start_as_current_span("anomaly_detection") as span:
        span.set_attribute("detector.algorithm", "isolation_forest")
        span.set_attribute("data.size", len(input_data))
        
        result = await detect_anomalies(input_data)
        
        span.set_attribute("result.anomalies_found", len(result.anomalies))
        return result
```

## Test Suite

### 1. Performance Test Organization

#### Test Structure
```
tests/performance/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── test_api_performance.py     # API performance tests
├── test_ml_performance.py      # ML algorithm performance tests
├── test_database_performance.py # Database performance tests
├── test_memory_performance.py  # Memory usage tests
├── test_load_testing.py        # Load testing scenarios
├── benchmarks/                 # Benchmark suites
│   ├── __init__.py
│   ├── api_benchmarks.py
│   ├── ml_benchmarks.py
│   └── integration_benchmarks.py
└── fixtures/                   # Test data and fixtures
    ├── sample_datasets.py
    ├── performance_data.json
    └── baseline_metrics.json
```

#### Test Configuration
```python
# conftest.py - Performance test configuration
import pytest
import asyncio
from pytest_benchmark import benchmark

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def performance_client():
    """HTTP client for performance testing."""
    import httpx
    return httpx.AsyncClient(base_url="http://localhost:8000")

@pytest.fixture
def sample_data():
    """Sample dataset for testing."""
    import numpy as np
    return np.random.randn(1000, 10)

@pytest.fixture
def large_dataset():
    """Large dataset for stress testing."""
    import numpy as np
    return np.random.randn(100000, 50)

# Benchmark configuration
@pytest.fixture
def benchmark_config():
    """Benchmark configuration."""
    return {
        "min_rounds": 5,
        "max_time": 30,
        "warmup": True,
        "disable_gc": True
    }
```

### 2. Continuous Performance Testing

#### CI/CD Integration
```yaml
# .github/workflows/performance.yml
name: Performance Testing

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  performance-tests:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e .[test,performance]
    
    - name: Start application
      run: |
        uvicorn src.pynomaly.presentation.api.app:app --host 0.0.0.0 --port 8000 &
        sleep 10
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ --benchmark-json=benchmark.json
    
    - name: Compare with baseline
      run: |
        python scripts/compare_performance.py benchmark.json baseline.json
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark.json
```

## Performance Baselines

### 1. Baseline Metrics

#### System Performance Baselines
```json
{
  "baseline_version": "1.0.0",
  "environment": "test",
  "timestamp": "2024-01-15T10:00:00Z",
  "api_performance": {
    "health_check": {
      "mean_time": 0.005,
      "p95_time": 0.010,
      "p99_time": 0.015
    },
    "authentication": {
      "mean_time": 0.050,
      "p95_time": 0.100,
      "p99_time": 0.150
    },
    "anomaly_detection": {
      "mean_time": 0.200,
      "p95_time": 0.500,
      "p99_time": 1.000
    }
  },
  "ml_performance": {
    "isolation_forest": {
      "training_time_1k": 0.5,
      "training_time_10k": 5.0,
      "prediction_time_single": 0.001,
      "prediction_time_batch_1k": 0.050
    },
    "local_outlier_factor": {
      "training_time_1k": 0.2,
      "training_time_10k": 2.0,
      "prediction_time_single": 0.005,
      "prediction_time_batch_1k": 0.200
    }
  },
  "resource_usage": {
    "memory_baseline": "512MB",
    "cpu_baseline": "10%",
    "disk_baseline": "100MB"
  }
}
```

### 2. Performance Regression Detection

#### Regression Analysis Script
```python
# scripts/performance_regression_check.py
import json
import sys
from typing import Dict, Any

def load_benchmark_data(file_path: str) -> Dict[str, Any]:
    """Load benchmark data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_performance(current: Dict, baseline: Dict) -> Dict[str, Any]:
    """Compare current performance with baseline."""
    
    results = {
        "regressions": [],
        "improvements": [],
        "summary": {}
    }
    
    # Compare API performance
    for endpoint, metrics in current.get("api_performance", {}).items():
        baseline_metrics = baseline.get("api_performance", {}).get(endpoint, {})
        
        for metric, value in metrics.items():
            baseline_value = baseline_metrics.get(metric)
            if baseline_value:
                change_percent = ((value - baseline_value) / baseline_value) * 100
                
                if change_percent > 20:  # 20% regression threshold
                    results["regressions"].append({
                        "test": f"api.{endpoint}.{metric}",
                        "current": value,
                        "baseline": baseline_value,
                        "change_percent": change_percent
                    })
                elif change_percent < -10:  # 10% improvement threshold
                    results["improvements"].append({
                        "test": f"api.{endpoint}.{metric}",
                        "current": value,
                        "baseline": baseline_value,
                        "change_percent": change_percent
                    })
    
    # Compare ML performance
    for algorithm, metrics in current.get("ml_performance", {}).items():
        baseline_metrics = baseline.get("ml_performance", {}).get(algorithm, {})
        
        for metric, value in metrics.items():
            baseline_value = baseline_metrics.get(metric)
            if baseline_value:
                change_percent = ((value - baseline_value) / baseline_value) * 100
                
                if change_percent > 30:  # 30% regression threshold for ML
                    results["regressions"].append({
                        "test": f"ml.{algorithm}.{metric}",
                        "current": value,
                        "baseline": baseline_value,
                        "change_percent": change_percent
                    })
    
    # Generate summary
    results["summary"] = {
        "total_regressions": len(results["regressions"]),
        "total_improvements": len(results["improvements"]),
        "overall_status": "PASS" if len(results["regressions"]) == 0 else "FAIL"
    }
    
    return results

def main():
    """Main function to run regression analysis."""
    if len(sys.argv) != 3:
        print("Usage: python performance_regression_check.py <current_file> <baseline_file>")
        sys.exit(1)
    
    current_file, baseline_file = sys.argv[1], sys.argv[2]
    
    try:
        current_data = load_benchmark_data(current_file)
        baseline_data = load_benchmark_data(baseline_file)
        
        results = compare_performance(current_data, baseline_data)
        
        # Print results
        print(f"Performance Analysis Results")
        print(f"===========================")
        print(f"Status: {results['summary']['overall_status']}")
        print(f"Regressions: {results['summary']['total_regressions']}")
        print(f"Improvements: {results['summary']['total_improvements']}")
        
        if results["regressions"]:
            print(f"\nPerformance Regressions:")
            for regression in results["regressions"]:
                print(f"  - {regression['test']}: {regression['change_percent']:.1f}% slower")
        
        if results["improvements"]:
            print(f"\nPerformance Improvements:")
            for improvement in results["improvements"]:
                print(f"  + {improvement['test']}: {improvement['change_percent']:.1f}% faster")
        
        # Exit with error code if regressions found
        if results["summary"]["overall_status"] == "FAIL":
            sys.exit(1)
            
    except Exception as e:
        print(f"Error analyzing performance: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Conclusion

This comprehensive performance benchmarking guide provides:

1. **Framework**: Complete testing architecture and tools
2. **Procedures**: Systematic benchmarking and baseline establishment
3. **Optimization**: Actionable recommendations for all system layers
4. **Troubleshooting**: Diagnostic tools and common issue resolution
5. **Monitoring**: Production monitoring and alerting setup
6. **Testing**: Automated test suite and CI/CD integration
7. **Baselines**: Performance standards and regression detection

Regular use of these procedures ensures optimal performance, early detection of regressions, and continuous optimization of the Pynomaly platform.

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Next Review**: Quarterly  
**Owner**: Performance Engineering Team