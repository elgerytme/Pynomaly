# Pynomaly Performance Optimization Guide

## Overview

This guide provides comprehensive performance optimization strategies for Pynomaly in production environments, covering application-level, infrastructure-level, and database optimizations.

## Application Performance Optimization

### 1. FastAPI Configuration

#### Production ASGI Settings
```python
# config/production.py
from pynomaly.core.factory import create_app

app = create_app()

# ASGI server configuration
ASGI_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,  # Number of worker processes
    "worker_class": "uvicorn.workers.UvicornWorker",
    "worker_connections": 1000,
    "max_requests": 1000,
    "max_requests_jitter": 50,
    "timeout": 30,
    "keepalive": 65,
    "preload": True,  # Preload application
}
```

#### Optimized Middleware Stack
```python
# Optimized middleware order for performance
middleware_stack = [
    "SecurityHeadersMiddleware",      # Security (minimal overhead)
    "GZipMiddleware",                 # Compression
    "RateLimitingMiddleware",         # Rate limiting  
    "CacheMiddleware",                # Response caching
    "MetricsMiddleware",              # Monitoring
    "CORSMiddleware",                 # CORS handling
    "SessionMiddleware",              # Session management
    "RequestLoggingMiddleware",       # Logging (last)
]
```

### 2. Memory Management

#### Python Memory Optimization
```bash
# Environment variables for memory optimization
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Garbage collection tuning
export PYTHON_GC_THRESHOLD_0=700
export PYTHON_GC_THRESHOLD_1=10
export PYTHON_GC_THRESHOLD_2=10
```

#### Memory Pool Configuration
```python
# config/memory.py
MEMORY_CONFIG = {
    # SQLAlchemy connection pooling
    "database_pool_size": 20,
    "database_max_overflow": 30,
    "database_pool_timeout": 30,
    "database_pool_recycle": 3600,
    
    # Redis connection pooling
    "redis_connection_pool_max_connections": 50,
    "redis_connection_pool_retry_on_timeout": True,
    
    # Object caching
    "cache_max_size": 10000,
    "cache_ttl": 3600,
}
```

### 3. Caching Strategies

#### Multi-Level Caching
```python
# Application-level caching configuration
CACHE_CONFIG = {
    # L1: In-memory cache (fastest)
    "memory_cache": {
        "enabled": True,
        "max_size": 1000,
        "ttl": 300,  # 5 minutes
    },
    
    # L2: Redis cache (shared)
    "redis_cache": {
        "enabled": True,
        "ttl": 3600,  # 1 hour
        "key_prefix": "pynomaly:",
    },
    
    # L3: Database query cache
    "query_cache": {
        "enabled": True,
        "ttl": 1800,  # 30 minutes
    },
}
```

#### Cache Warming Strategy
```python
# Background cache warming
async def warm_cache():
    """Warm frequently accessed data."""
    # Pre-load popular models
    await cache_popular_detectors()
    
    # Pre-compute common aggregations
    await cache_model_statistics()
    
    # Pre-load configuration data
    await cache_system_configuration()
```

## Infrastructure Performance Optimization

### 1. Kubernetes Resource Optimization

#### Optimized Resource Requests and Limits
```yaml
# deployment/helm/pynomaly/values-production.yaml
resources:
  requests:
    cpu: 750m        # 75% of a CPU core
    memory: 2Gi      # 2GB RAM
  limits:
    cpu: 2000m       # 2 CPU cores max
    memory: 6Gi      # 6GB RAM max

# Node affinity for performance
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/instance-type
          operator: In
          values:
          - c5.2xlarge    # CPU-optimized instances
          - c5.4xlarge
```

#### Horizontal Pod Autoscaling (HPA)
```yaml
# HPA configuration for optimal scaling
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
  
  # Custom metrics scaling
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 15
```

### 2. Database Performance Optimization

#### PostgreSQL Configuration
```yaml
# PostgreSQL production tuning
postgresql:
  primary:
    configuration:
      # Connection settings
      max_connections: 400
      superuser_reserved_connections: 3
      
      # Memory settings
      shared_buffers: 512MB
      effective_cache_size: 2GB
      maintenance_work_mem: 128MB
      work_mem: 8MB
      
      # Checkpoint settings
      checkpoint_completion_target: 0.9
      checkpoint_timeout: 15min
      max_wal_size: 2GB
      min_wal_size: 512MB
      
      # Query optimization
      random_page_cost: 1.1
      effective_io_concurrency: 200
      default_statistics_target: 100
      
      # Autovacuum settings
      autovacuum_max_workers: 4
      autovacuum_vacuum_scale_factor: 0.1
      autovacuum_analyze_scale_factor: 0.05
      
      # Connection pooling
      listen_addresses: '*'
      port: 5432
```

#### Database Connection Pooling
```python
# Advanced connection pooling with PgBouncer
PGBOUNCER_CONFIG = {
    "pool_mode": "transaction",
    "max_client_conn": 1000,
    "default_pool_size": 20,
    "reserve_pool_size": 5,
    "server_lifetime": 3600,
    "server_idle_timeout": 600,
}
```

### 3. Redis Performance Optimization

#### Redis Configuration
```yaml
# Redis production tuning
redis:
  master:
    configuration: |
      # Memory optimization
      maxmemory 2gb
      maxmemory-policy allkeys-lru
      
      # Persistence optimization
      save 900 1
      save 300 10
      save 60 10000
      rdbcompression yes
      rdbchecksum yes
      
      # Network optimization
      tcp-keepalive 300
      timeout 0
      tcp-backlog 511
      
      # Performance tuning
      hash-max-ziplist-entries 512
      hash-max-ziplist-value 64
      list-max-ziplist-size -2
      set-max-intset-entries 512
      zset-max-ziplist-entries 128
      zset-max-ziplist-value 64
```

#### Redis Clustering for Scale
```yaml
# Redis cluster configuration
redis:
  cluster:
    enabled: true
    nodes: 6
    replicas: 1
    
  sentinel:
    enabled: true
    masterName: pynomaly-master
    quorum: 2
```

## Application-Level Optimizations

### 1. Async Programming Optimization

#### Optimized Async Patterns
```python
# Efficient async request handling
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedDetectionService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def detect_anomalies_batch(self, datasets: List[Dataset]) -> List[Result]:
        """Process multiple datasets concurrently."""
        # Use semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(10)
        
        async def process_with_limit(dataset):
            async with semaphore:
                return await self.detect_anomalies(dataset)
        
        # Process in batches to avoid memory issues
        results = []
        for batch in self._batch(datasets, 20):
            batch_results = await asyncio.gather(
                *[process_with_limit(ds) for ds in batch],
                return_exceptions=True
            )
            results.extend(batch_results)
        
        return results
    
    async def cpu_intensive_task(self, data):
        """Offload CPU-intensive work to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._process_data, 
            data
        )
```

### 2. Data Processing Optimization

#### Efficient Data Handling
```python
# Memory-efficient data processing
import pandas as pd
import numpy as np
from typing import Iterator

class OptimizedDataProcessor:
    @staticmethod
    def process_large_dataset(filepath: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Process large datasets in chunks to manage memory."""
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            # Apply optimizations
            chunk = chunk.astype({
                col: 'category' for col in chunk.select_dtypes(['object']).columns
                if chunk[col].nunique() / len(chunk) < 0.5
            })
            
            # Use efficient data types
            for col in chunk.select_dtypes(['int64']).columns:
                if chunk[col].min() >= 0 and chunk[col].max() <= 255:
                    chunk[col] = chunk[col].astype('uint8')
                elif chunk[col].min() >= -128 and chunk[col].max() <= 127:
                    chunk[col] = chunk[col].astype('int8')
            
            yield chunk
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        # Use sparse arrays for data with many zeros/NaNs
        for col in df.columns:
            if df[col].isna().sum() / len(df) > 0.5:
                df[col] = df[col].astype(pd.SparseDtype(df[col].dtype))
        
        return df
```

### 3. Model Loading and Caching

#### Efficient Model Management
```python
# Model caching and loading optimization
from functools import lru_cache
import pickle
import joblib

class ModelCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache = {}
        self._usage_order = []
    
    @lru_cache(maxsize=10)
    def load_model(self, model_id: str):
        """Load model with LRU caching."""
        if model_id in self._cache:
            self._update_usage(model_id)
            return self._cache[model_id]
        
        # Load model efficiently
        model = self._load_from_storage(model_id)
        
        # Manage cache size
        if len(self._cache) >= self.max_size:
            oldest = self._usage_order.pop(0)
            del self._cache[oldest]
        
        self._cache[model_id] = model
        self._usage_order.append(model_id)
        
        return model
    
    def _load_from_storage(self, model_id: str):
        """Optimized model loading."""
        # Use joblib for scikit-learn models (faster than pickle)
        try:
            return joblib.load(f"models/{model_id}.joblib")
        except:
            # Fallback to pickle
            with open(f"models/{model_id}.pkl", 'rb') as f:
                return pickle.load(f)
```

## Monitoring and Performance Measurement

### 1. Application Performance Monitoring (APM)

#### Custom Metrics Collection
```python
# Performance metrics collection
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
REQUEST_COUNT = Counter('pynomaly_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('pynomaly_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('pynomaly_active_connections', 'Active connections')
MEMORY_USAGE = Gauge('pynomaly_memory_usage_bytes', 'Memory usage')

class PerformanceMonitor:
    @staticmethod
    async def track_request_performance(request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
```

### 2. Performance Benchmarking

#### Automated Performance Tests
```python
# Performance testing suite
import asyncio
import aiohttp
import time
from typing import List, Dict

class PerformanceBenchmark:
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def benchmark_endpoint(
        self, 
        endpoint: str, 
        concurrent_requests: int = 100,
        total_requests: int = 1000
    ) -> Dict:
        """Benchmark API endpoint performance."""
        semaphore = asyncio.Semaphore(concurrent_requests)
        start_time = time.time()
        
        async def make_request(session):
            async with semaphore:
                async with session.get(f"{self.base_url}{endpoint}") as response:
                    return {
                        'status': response.status,
                        'response_time': time.time()
                    }
        
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session) for _ in range(total_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Calculate metrics
        successful_requests = [r for r in results if isinstance(r, dict) and r['status'] == 200]
        response_times = [r['response_time'] - start_time for r in successful_requests]
        
        return {
            'total_requests': total_requests,
            'successful_requests': len(successful_requests),
            'success_rate': len(successful_requests) / total_requests,
            'avg_response_time': sum(response_times) / len(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'requests_per_second': len(successful_requests) / (end_time - start_time),
            'total_duration': end_time - start_time
        }
```

## Performance Tuning Checklist

### Application Level
- [ ] Enable response compression (gzip)
- [ ] Implement multi-level caching
- [ ] Optimize database queries
- [ ] Use connection pooling
- [ ] Enable async processing
- [ ] Implement request rate limiting
- [ ] Optimize data serialization
- [ ] Use efficient data structures

### Infrastructure Level
- [ ] Configure resource requests/limits
- [ ] Enable horizontal pod autoscaling
- [ ] Use CPU-optimized instance types
- [ ] Configure node affinity
- [ ] Implement load balancing
- [ ] Enable cluster autoscaling
- [ ] Use fast SSD storage
- [ ] Configure network optimization

### Database Level
- [ ] Tune PostgreSQL configuration
- [ ] Implement connection pooling
- [ ] Create appropriate indexes
- [ ] Configure autovacuum
- [ ] Monitor query performance
- [ ] Implement read replicas
- [ ] Use connection pooling
- [ ] Enable query caching

### Monitoring Level
- [ ] Set up performance metrics
- [ ] Configure alerting thresholds
- [ ] Implement APM tools
- [ ] Monitor resource utilization
- [ ] Track business metrics
- [ ] Set up log aggregation
- [ ] Configure distributed tracing
- [ ] Implement health checks

## Performance Targets

### Response Time Targets
- **95th percentile**: < 500ms
- **99th percentile**: < 1000ms
- **Average**: < 200ms

### Throughput Targets
- **Requests per second**: > 1000 RPS
- **Concurrent users**: > 500
- **Daily requests**: > 1M

### Resource Utilization Targets
- **CPU utilization**: < 70%
- **Memory utilization**: < 80%
- **Database connections**: < 80% of max
- **Cache hit rate**: > 90%

### Availability Targets
- **Uptime**: 99.9%
- **Error rate**: < 0.1%
- **Recovery time**: < 5 minutes