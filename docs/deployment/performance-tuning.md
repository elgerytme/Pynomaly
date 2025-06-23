# Performance Tuning Guide

## Overview

This comprehensive guide covers performance optimization strategies for Pynomaly in production environments. It addresses database optimization, caching strategies, connection pooling, memory management, and GPU acceleration to ensure optimal performance under various workloads.

## Table of Contents

1. [Database Optimization](#database-optimization)
2. [Caching Strategies](#caching-strategies)
3. [Connection Pooling](#connection-pooling)
4. [Memory Management](#memory-management)
5. [Concurrent Processing](#concurrent-processing)
6. [GPU Acceleration](#gpu-acceleration)
7. [Performance Monitoring](#performance-monitoring)
8. [Load Testing](#load-testing)
9. [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

## Database Optimization

### PostgreSQL Configuration

#### Connection Settings

```sql
-- postgresql.conf optimizations
max_connections = 200
shared_buffers = 1GB
effective_cache_size = 3GB
maintenance_work_mem = 256MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
```

#### Memory Configuration

```sql
-- Memory settings for anomaly detection workloads
work_mem = 64MB              -- For sorting operations
shared_preload_libraries = 'pg_stat_statements'
log_min_duration_statement = 1000  -- Log slow queries
```

#### Query Optimization

```sql
-- Essential indexes for Pynomaly tables
CREATE INDEX CONCURRENTLY idx_detectors_algorithm ON detectors(algorithm_name);
CREATE INDEX CONCURRENTLY idx_detectors_created ON detectors(created_at);
CREATE INDEX CONCURRENTLY idx_datasets_size ON datasets(n_samples);
CREATE INDEX CONCURRENTLY idx_detection_results_timestamp ON detection_results(created_at);
CREATE INDEX CONCURRENTLY idx_detection_results_detector ON detection_results(detector_id);

-- Composite indexes for common queries
CREATE INDEX CONCURRENTLY idx_detectors_algo_fitted ON detectors(algorithm_name, is_fitted);
CREATE INDEX CONCURRENTLY idx_datasets_format_size ON datasets(file_format, n_samples);
```

#### Vacuum and Maintenance

```sql
-- Automated vacuum configuration
ALTER TABLE detectors SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE datasets SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE detection_results SET (autovacuum_vacuum_scale_factor = 0.05);

-- Weekly maintenance script
VACUUM ANALYZE;
REINDEX INDEX CONCURRENTLY idx_detectors_algorithm;
```

### Application-Level Database Optimization

```python
# Optimized repository patterns
class OptimizedDetectorRepository:
    def __init__(self, session_factory, cache_manager):
        self.session_factory = session_factory
        self.cache = cache_manager
    
    async def find_by_algorithm_batch(self, algorithms: List[str]) -> List[Detector]:
        """Batch query to reduce database round trips."""
        cache_key = f"detectors:batch:{':'.join(sorted(algorithms))}"
        
        if cached := await self.cache.get(cache_key):
            return cached
        
        async with self.session_factory() as session:
            query = select(DetectorModel).where(
                DetectorModel.algorithm_name.in_(algorithms)
            ).options(
                selectinload(DetectorModel.hyperparameters)  # Eager loading
            )
            result = await session.execute(query)
            detectors = [self._to_domain(model) for model in result.scalars()]
            
            await self.cache.set(cache_key, detectors, ttl=300)
            return detectors
    
    async def bulk_update_status(self, detector_ids: List[str], status: str):
        """Bulk update to improve write performance."""
        async with self.session_factory() as session:
            await session.execute(
                update(DetectorModel)
                .where(DetectorModel.id.in_(detector_ids))
                .values(status=status, updated_at=datetime.utcnow())
            )
            await session.commit()
```

## Caching Strategies

### Redis Configuration

#### Redis Server Optimization

```conf
# redis.conf optimizations for Pynomaly
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000

# Network optimizations
tcp-keepalive 300
tcp-backlog 511
timeout 0

# Memory optimization
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
```

#### Application Cache Strategies

```python
from typing import Optional, Any
import json
import pickle
from datetime import timedelta

class CacheManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def cache_detector_results(
        self, 
        detector_id: str, 
        data_hash: str, 
        results: Any,
        ttl: int = 3600
    ):
        """Cache detection results with intelligent TTL."""
        cache_key = f"detection:results:{detector_id}:{data_hash}"
        
        # Serialize with compression for large results
        if len(str(results)) > 1000:
            serialized = pickle.dumps(results)
            compressed = gzip.compress(serialized)
            await self.redis.setex(cache_key, ttl, compressed)
        else:
            await self.redis.setex(cache_key, ttl, json.dumps(results))
    
    async def cache_model_metadata(self, detector_id: str, metadata: dict):
        """Cache model metadata with longer TTL."""
        cache_key = f"model:metadata:{detector_id}"
        await self.redis.setex(cache_key, 86400, json.dumps(metadata))  # 24 hours
    
    async def invalidate_detector_cache(self, detector_id: str):
        """Invalidate all cache entries for a detector."""
        patterns = [
            f"detection:results:{detector_id}:*",
            f"model:metadata:{detector_id}",
            f"detector:info:{detector_id}"
        ]
        
        for pattern in patterns:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
```

### Multi-Level Caching

```python
class MultiLevelCache:
    def __init__(self, memory_cache, redis_cache):
        self.L1 = memory_cache  # Fast in-memory cache
        self.L2 = redis_cache   # Distributed cache
    
    async def get(self, key: str) -> Optional[Any]:
        # L1 cache lookup
        if value := self.L1.get(key):
            return value
        
        # L2 cache lookup
        if value := await self.L2.get(key):
            self.L1.set(key, value, ttl=300)  # Populate L1
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        # Set in both levels
        self.L1.set(key, value, min(ttl, 300))  # L1 with shorter TTL
        await self.L2.set(key, value, ttl)      # L2 with full TTL
```

## Connection Pooling

### Database Connection Pool Configuration

```python
from sqlalchemy.pool import QueuePool
from sqlalchemy.ext.asyncio import create_async_engine

# Optimized database connection pool
engine = create_async_engine(
    DATABASE_URL,
    
    # Connection pool settings
    poolclass=QueuePool,
    pool_size=20,              # Base pool size
    max_overflow=30,           # Additional connections under load
    pool_timeout=30,           # Connection acquisition timeout
    pool_recycle=3600,         # Recycle connections every hour
    pool_pre_ping=True,        # Validate connections before use
    
    # Query optimization
    echo=False,                # Disable SQL logging in production
    future=True,
    connect_args={
        "command_timeout": 60,
        "server_settings": {
            "application_name": "pynomaly",
            "jit": "off"       # Disable JIT for predictable performance
        }
    }
)
```

### Redis Connection Pool

```python
import aioredis
from aioredis.connection import ConnectionPool

# Redis connection pool
redis_pool = ConnectionPool.from_url(
    REDIS_URL,
    max_connections=20,
    retry_on_timeout=True,
    health_check_interval=30,
    socket_keepalive=True,
    socket_keepalive_options={
        1: 300,  # TCP_KEEPIDLE
        2: 30,   # TCP_KEEPINTVL
        3: 5,    # TCP_KEEPCNT
    }
)

redis_client = aioredis.Redis(connection_pool=redis_pool)
```

### HTTP Connection Pool

```python
import aiohttp
from aiohttp import TCPConnector

# HTTP client with connection pooling
connector = TCPConnector(
    limit=100,                 # Total connection limit
    limit_per_host=30,         # Connections per host
    ttl_dns_cache=300,         # DNS cache TTL
    use_dns_cache=True,
    keepalive_timeout=30,
    enable_cleanup_closed=True
)

http_session = aiohttp.ClientSession(
    connector=connector,
    timeout=aiohttp.ClientTimeout(total=60)
)
```

## Memory Management

### Python Memory Optimization

```python
import gc
import psutil
import asyncio
from typing import Generator
import numpy as np

class MemoryManager:
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        memory_info = self.process.memory_info()
        total_memory = psutil.virtual_memory().total
        return (memory_info.rss / total_memory) * 100
    
    async def check_memory_pressure(self) -> bool:
        """Check if memory usage is high."""
        return self.get_memory_usage() > self.max_memory_percent
    
    async def cleanup_if_needed(self):
        """Trigger cleanup if memory pressure is high."""
        if await self.check_memory_pressure():
            # Force garbage collection
            gc.collect()
            
            # Clear numpy cache
            if hasattr(np, '_NoValue'):
                np._NoValue._clear_cache()
    
    def memory_efficient_batch_processor(
        self, 
        data: np.ndarray, 
        batch_size: int = 1000
    ) -> Generator[np.ndarray, None, None]:
        """Process data in memory-efficient batches."""
        total_size = len(data)
        
        for i in range(0, total_size, batch_size):
            # Check memory before processing each batch
            if self.get_memory_usage() > self.max_memory_percent:
                gc.collect()
            
            batch = data[i:i + batch_size]
            yield batch
            
            # Explicit cleanup after each batch
            del batch
```

### Large Dataset Processing

```python
import dask.dataframe as dd
import pandas as pd
from pathlib import Path

class EfficientDataProcessor:
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
    
    async def process_large_dataset(
        self, 
        file_path: Path, 
        detector,
        output_path: Path
    ):
        """Process large datasets without loading entirely into memory."""
        
        # Use Dask for out-of-core processing
        if file_path.suffix == '.parquet':
            df = dd.read_parquet(file_path)
        else:
            df = dd.read_csv(file_path)
        
        # Process in partitions
        results = []
        for partition in df.to_delayed():
            chunk_df = partition.compute()
            
            # Process chunk
            chunk_results = await self._process_chunk(chunk_df, detector)
            results.append(chunk_results)
            
            # Memory cleanup
            del chunk_df
            gc.collect()
        
        # Combine and save results
        final_results = pd.concat(results)
        final_results.to_parquet(output_path)
        
        return final_results
    
    async def _process_chunk(self, chunk: pd.DataFrame, detector) -> pd.DataFrame:
        """Process a single chunk of data."""
        # Convert to numpy for processing
        data = chunk.select_dtypes(include=[np.number]).values
        
        # Run detection
        scores = await detector.predict(data)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'anomaly_score': scores,
            'is_anomaly': scores > detector.threshold
        })
        
        return results
```

## Concurrent Processing

### Async Processing Optimization

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Callable, Any

class ConcurrentProcessor:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=os.cpu_count())
    
    async def run_concurrent_detections(
        self, 
        detectors: List[Any], 
        data: np.ndarray
    ) -> List[np.ndarray]:
        """Run multiple detectors concurrently."""
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def run_detector(detector):
            async with semaphore:
                # CPU-intensive work in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.thread_executor,
                    detector.predict,
                    data
                )
        
        # Run all detectors concurrently
        tasks = [run_detector(detector) for detector in detectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        return valid_results
    
    async def parallel_hyperparameter_tuning(
        self, 
        algorithm_class: type,
        param_combinations: List[dict],
        train_data: np.ndarray,
        validation_data: np.ndarray
    ) -> dict:
        """Run hyperparameter tuning in parallel."""
        
        def train_and_evaluate(params):
            detector = algorithm_class(**params)
            detector.fit(train_data)
            scores = detector.predict(validation_data)
            return params, np.mean(scores)
        
        # Use process pool for CPU-intensive tuning
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                self.process_executor,
                train_and_evaluate,
                params
            )
            for params in param_combinations
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Find best parameters
        best_params, best_score = max(results, key=lambda x: x[1])
        return best_params
```

### Batch Processing Optimization

```python
class BatchProcessor:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
    
    async def process_streaming_data(
        self, 
        data_stream: AsyncIterator[np.ndarray],
        detector,
        callback: Callable
    ):
        """Process streaming data with optimal batching."""
        
        batch = []
        
        async for data_point in data_stream:
            batch.append(data_point)
            
            if len(batch) >= self.batch_size:
                # Process batch
                batch_data = np.vstack(batch)
                results = await detector.predict(batch_data)
                
                # Send results via callback
                await callback(results)
                
                # Clear batch
                batch.clear()
                
                # Memory cleanup
                del batch_data
                gc.collect()
        
        # Process remaining data
        if batch:
            batch_data = np.vstack(batch)
            results = await detector.predict(batch_data)
            await callback(results)
```

## GPU Acceleration

### CUDA Configuration

```python
import torch
import numpy as np
from typing import Optional

class GPUManager:
    def __init__(self):
        self.device = self._get_optimal_device()
        self.memory_fraction = 0.8  # Reserve 80% of GPU memory
    
    def _get_optimal_device(self) -> torch.device:
        """Select optimal device for computation."""
        if torch.cuda.is_available():
            # Select GPU with most free memory
            max_memory = 0
            best_device = 0
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory = props.total_memory
                if memory > max_memory:
                    max_memory = memory
                    best_device = i
            
            return torch.device(f'cuda:{best_device}')
        
        return torch.device('cpu')
    
    def optimize_memory(self):
        """Optimize GPU memory usage."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(
                self.memory_fraction, 
                device=self.device
            )
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to optimal device."""
        return tensor.to(self.device, non_blocking=True)
    
    async def process_batch_gpu(
        self, 
        model: torch.nn.Module,
        data: np.ndarray,
        batch_size: int = 512
    ) -> np.ndarray:
        """Process data on GPU in optimal batches."""
        
        model = model.to(self.device)
        model.eval()
        
        results = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # Convert to tensor and move to GPU
                batch_tensor = torch.from_numpy(batch).float()
                batch_tensor = self.to_device(batch_tensor)
                
                # Process on GPU
                output = model(batch_tensor)
                
                # Move back to CPU for results
                batch_results = output.cpu().numpy()
                results.append(batch_results)
                
                # Clean up GPU memory
                del batch_tensor, output
                
                if i % (batch_size * 10) == 0:  # Periodic cleanup
                    torch.cuda.empty_cache()
        
        return np.concatenate(results)
```

### TensorFlow GPU Optimization

```python
import tensorflow as tf

def configure_tensorflow_gpu():
    """Configure TensorFlow for optimal GPU usage."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set virtual GPU memory limit
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=4096  # 4GB limit
                )]
            )
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Enable mixed precision for better performance
    tf.config.optimizer.set_experimental_options({
        'auto_mixed_precision': True
    })

class TensorFlowGPUOptimizer:
    def __init__(self):
        configure_tensorflow_gpu()
        self.strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else None
    
    @tf.function
    def batch_predict(self, model, data):
        """Optimized batch prediction with tf.function."""
        return model(data, training=False)
    
    async def process_with_strategy(self, model, data):
        """Process data using distribution strategy if available."""
        if self.strategy:
            with self.strategy.scope():
                distributed_data = self.strategy.experimental_distribute_dataset(
                    tf.data.Dataset.from_tensor_slices(data).batch(512)
                )
                
                results = []
                for batch in distributed_data:
                    result = self.strategy.run(self.batch_predict, args=(model, batch))
                    results.append(self.strategy.experimental_local_results(result))
                
                return tf.concat([tf.concat(r, axis=0) for r in results], axis=0)
        else:
            return self.batch_predict(model, data)
```

## Performance Monitoring

### Application Performance Monitoring

```python
import time
import psutil
import asyncio
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict, deque

@dataclass
class PerformanceMetrics:
    timestamp: float
    cpu_usage: float
    memory_usage: float
    response_time: float
    throughput: float
    error_rate: float

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.request_times = defaultdict(lambda: deque(maxlen=window_size))
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
    
    async def record_request(self, endpoint: str, duration: float, success: bool):
        """Record request metrics."""
        self.request_times[endpoint].append(duration)
        self.request_counts[endpoint] += 1
        
        if not success:
            self.error_counts[endpoint] += 1
    
    async def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Application metrics
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        # Response time metrics
        all_times = []
        for times in self.request_times.values():
            all_times.extend(times)
        
        avg_response_time = sum(all_times) / len(all_times) if all_times else 0
        
        # Throughput (requests per second)
        throughput = total_requests / 60 if total_requests > 0 else 0
        
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            response_time=avg_response_time,
            throughput=throughput,
            error_rate=error_rate
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    async def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)
        
        return {
            "summary": {
                "avg_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                "avg_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                "avg_response_time": sum(m.response_time for m in recent_metrics) / len(recent_metrics),
                "avg_throughput": sum(m.throughput for m in recent_metrics) / len(recent_metrics),
                "avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            },
            "endpoint_breakdown": {
                endpoint: {
                    "avg_response_time": sum(times) / len(times),
                    "request_count": self.request_counts[endpoint],
                    "error_count": self.error_counts[endpoint],
                    "error_rate": (self.error_counts[endpoint] / self.request_counts[endpoint] * 100) 
                                 if self.request_counts[endpoint] > 0 else 0
                }
                for endpoint, times in self.request_times.items()
            }
        }
```

### Database Performance Monitoring

```python
class DatabaseMonitor:
    def __init__(self, engine):
        self.engine = engine
        self.slow_queries = deque(maxlen=100)
    
    async def monitor_query_performance(self):
        """Monitor database query performance."""
        async with self.engine.connect() as conn:
            # Get current connections
            result = await conn.execute(text("""
                SELECT count(*) as active_connections,
                       avg(extract(epoch from (now() - query_start))) as avg_query_duration
                FROM pg_stat_activity 
                WHERE state = 'active' AND query NOT LIKE '%pg_stat_activity%'
            """))
            
            stats = result.fetchone()
            
            # Get slow queries
            slow_queries = await conn.execute(text("""
                SELECT query, mean_time, calls, total_time
                FROM pg_stat_statements 
                WHERE mean_time > 1000 
                ORDER BY mean_time DESC 
                LIMIT 10
            """))
            
            return {
                "active_connections": stats.active_connections,
                "avg_query_duration": stats.avg_query_duration,
                "slow_queries": [dict(row) for row in slow_queries.fetchall()]
            }
    
    async def get_cache_hit_ratio(self):
        """Get database cache hit ratio."""
        async with self.engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT 
                    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) * 100 as cache_hit_ratio
                FROM pg_statio_user_tables
            """))
            
            return result.scalar()
```

## Load Testing

### Automated Load Testing

```python
import aiohttp
import asyncio
import json
from typing import List, Dict
import statistics

class LoadTester:
    def __init__(self, base_url: str, max_concurrent: int = 50):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_detection_load_test(
        self, 
        detector_id: str,
        test_data: List[Dict],
        duration_seconds: int = 60
    ) -> Dict:
        """Run load test for detection endpoint."""
        
        results = []
        start_time = time.time()
        
        async def make_request(session, data):
            async with self.semaphore:
                request_start = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/detection/predict",
                        json={
                            "detector_id": detector_id,
                            "data": data
                        }
                    ) as response:
                        await response.json()
                        success = response.status == 200
                        
                except Exception as e:
                    success = False
                
                request_time = time.time() - request_start
                return {"success": success, "response_time": request_time}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            while time.time() - start_time < duration_seconds:
                # Create batch of requests
                batch_tasks = [
                    make_request(session, data) 
                    for data in test_data[:10]  # Use first 10 samples
                ]
                
                tasks.extend(batch_tasks)
                
                # Execute batch
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)
                
                # Brief pause between batches
                await asyncio.sleep(0.1)
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in successful_requests]
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_requests),
            "success_rate": len(successful_requests) / len(results) * 100,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
            "requests_per_second": len(results) / duration_seconds
        }
    
    async def stress_test_training(
        self,
        training_requests: List[Dict],
        max_concurrent_training: int = 5
    ) -> Dict:
        """Stress test model training endpoints."""
        
        training_semaphore = asyncio.Semaphore(max_concurrent_training)
        results = []
        
        async def train_model(session, request_data):
            async with training_semaphore:
                start_time = time.time()
                try:
                    async with session.post(
                        f"{self.base_url}/detection/train",
                        json=request_data
                    ) as response:
                        result = await response.json()
                        success = response.status == 200
                        training_time = result.get("training_time_ms", 0) if success else 0
                        
                except Exception as e:
                    success = False
                    training_time = 0
                
                total_time = time.time() - start_time
                return {
                    "success": success, 
                    "training_time": training_time,
                    "total_time": total_time
                }
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                train_model(session, request) 
                for request in training_requests
            ]
            
            results = await asyncio.gather(*tasks)
        
        successful_results = [r for r in results if r["success"]]
        
        return {
            "total_training_jobs": len(results),
            "successful_jobs": len(successful_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "avg_training_time": statistics.mean([r["training_time"] for r in successful_results]) if successful_results else 0,
            "avg_total_time": statistics.mean([r["total_time"] for r in successful_results]) if successful_results else 0
        }
```

## Troubleshooting Performance Issues

### Common Performance Problems

#### Memory Leaks

```python
import tracemalloc
import gc
import weakref

class MemoryLeakDetector:
    def __init__(self):
        self.snapshots = []
        self.object_counts = defaultdict(int)
    
    def start_monitoring(self):
        """Start memory leak monitoring."""
        tracemalloc.start()
        self.take_snapshot()
    
    def take_snapshot(self):
        """Take memory snapshot."""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(snapshot)
        
        # Count object types
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            self.object_counts[obj_type] += 1
    
    def detect_leaks(self) -> Dict:
        """Detect potential memory leaks."""
        if len(self.snapshots) < 2:
            return {"error": "Need at least 2 snapshots"}
        
        # Compare latest snapshots
        current = self.snapshots[-1]
        previous = self.snapshots[-2]
        
        top_stats = current.compare_to(previous, 'lineno')
        
        leaks = []
        for stat in top_stats[:10]:
            if stat.size_diff > 1024 * 1024:  # > 1MB difference
                leaks.append({
                    "traceback": stat.traceback.format(),
                    "size_diff_mb": stat.size_diff / (1024 * 1024),
                    "count_diff": stat.count_diff
                })
        
        return {"potential_leaks": leaks}
```

#### Slow Database Queries

```python
class QueryPerformanceAnalyzer:
    def __init__(self, engine):
        self.engine = engine
    
    async def analyze_slow_queries(self) -> List[Dict]:
        """Analyze slow queries and suggest optimizations."""
        async with self.engine.connect() as conn:
            slow_queries = await conn.execute(text("""
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    rows,
                    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                WHERE mean_time > 100
                ORDER BY mean_time DESC 
                LIMIT 20
            """))
            
            analyses = []
            for query_info in slow_queries.fetchall():
                analysis = {
                    "query": query_info.query[:200] + "..." if len(query_info.query) > 200 else query_info.query,
                    "mean_time_ms": query_info.mean_time,
                    "total_calls": query_info.calls,
                    "cache_hit_percent": query_info.hit_percent,
                    "suggestions": self._generate_suggestions(query_info)
                }
                analyses.append(analysis)
            
            return analyses
    
    def _generate_suggestions(self, query_info) -> List[str]:
        """Generate optimization suggestions for slow queries."""
        suggestions = []
        
        if query_info.hit_percent < 95:
            suggestions.append("Consider adding indexes to improve cache hit ratio")
        
        if "ORDER BY" in query_info.query and "LIMIT" in query_info.query:
            suggestions.append("Ensure indexes support ORDER BY columns")
        
        if "JOIN" in query_info.query:
            suggestions.append("Verify JOIN conditions have appropriate indexes")
        
        if query_info.mean_time > 1000:
            suggestions.append("Consider breaking down complex query into simpler parts")
        
        return suggestions
```

#### Connection Pool Issues

```python
class ConnectionPoolMonitor:
    def __init__(self, engine):
        self.engine = engine
    
    async def check_pool_health(self) -> Dict:
        """Check connection pool health."""
        pool = self.engine.pool
        
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "total_connections": pool.size() + pool.overflow(),
            "utilization_percent": (pool.checkedout() / (pool.size() + pool.overflow())) * 100
        }
    
    async def optimize_pool_settings(self) -> Dict:
        """Suggest pool optimization settings."""
        pool_stats = await self.check_pool_health()
        suggestions = []
        
        if pool_stats["utilization_percent"] > 90:
            suggestions.append("Increase pool_size or max_overflow")
        
        if pool_stats["invalid"] > 0:
            suggestions.append("Check for network issues or database restarts")
        
        if pool_stats["overflow"] > pool_stats["pool_size"]:
            suggestions.append("Consider increasing base pool_size")
        
        return {
            "current_stats": pool_stats,
            "suggestions": suggestions
        }
```

### Performance Monitoring Dashboard

```python
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

performance_router = APIRouter()

@performance_router.get("/performance/dashboard", response_class=HTMLResponse)
async def performance_dashboard():
    """Real-time performance monitoring dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pynomaly Performance Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; }
            .metric-card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; }
            .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <h1>Pynomaly Performance Dashboard</h1>
        
        <div class="dashboard">
            <div class="metric-card">
                <h3>Response Time</h3>
                <div class="metric-value" id="response-time">--</div>
                <canvas id="response-time-chart"></canvas>
            </div>
            
            <div class="metric-card">
                <h3>Throughput</h3>
                <div class="metric-value" id="throughput">--</div>
                <canvas id="throughput-chart"></canvas>
            </div>
            
            <div class="metric-card">
                <h3>Memory Usage</h3>
                <div class="metric-value" id="memory-usage">--</div>
                <canvas id="memory-chart"></canvas>
            </div>
            
            <div class="metric-card">
                <h3>Error Rate</h3>
                <div class="metric-value" id="error-rate">--</div>
                <canvas id="error-chart"></canvas>
            </div>
        </div>
        
        <script>
            // Real-time dashboard implementation
            async function updateMetrics() {
                try {
                    const response = await fetch('/api/performance/metrics');
                    const metrics = await response.json();
                    
                    document.getElementById('response-time').textContent = 
                        metrics.response_time.toFixed(2) + 'ms';
                    document.getElementById('throughput').textContent = 
                        metrics.throughput.toFixed(1) + ' req/s';
                    document.getElementById('memory-usage').textContent = 
                        metrics.memory_usage.toFixed(1) + '%';
                    document.getElementById('error-rate').textContent = 
                        metrics.error_rate.toFixed(2) + '%';
                    
                } catch (error) {
                    console.error('Failed to update metrics:', error);
                }
            }
            
            // Update every 5 seconds
            setInterval(updateMetrics, 5000);
            updateMetrics(); // Initial load
        </script>
    </body>
    </html>
    """

@performance_router.get("/performance/report")
async def detailed_performance_report():
    """Get detailed performance analysis report."""
    monitor = PerformanceMonitor()
    db_monitor = DatabaseMonitor(engine)
    
    return {
        "application_metrics": await monitor.get_performance_report(),
        "database_metrics": await db_monitor.monitor_query_performance(),
        "cache_hit_ratio": await db_monitor.get_cache_hit_ratio(),
        "timestamp": time.time()
    }
```

## Best Practices Summary

### 1. Database Optimization
- Use connection pooling with appropriate settings
- Create indexes for common query patterns
- Monitor slow queries and optimize regularly
- Use read replicas for read-heavy workloads

### 2. Caching Strategy
- Implement multi-level caching (memory + Redis)
- Cache expensive computation results
- Use appropriate TTL values based on data volatility
- Implement cache invalidation strategies

### 3. Memory Management
- Monitor memory usage continuously
- Use generators for large data processing
- Implement batch processing for memory efficiency
- Clean up resources explicitly

### 4. Concurrent Processing
- Use async/await for I/O operations
- Implement proper semaphores for resource limiting
- Use thread/process pools for CPU-intensive tasks
- Balance concurrency with resource constraints

### 5. GPU Acceleration
- Use GPU for large-scale neural network inference
- Implement batch processing for GPU efficiency
- Monitor GPU memory usage
- Provide CPU fallbacks for compatibility

### 6. Monitoring and Alerting
- Set up comprehensive performance monitoring
- Create alerts for critical metrics
- Monitor both application and infrastructure metrics
- Regular performance testing and optimization

This comprehensive performance tuning guide provides the foundation for optimizing Pynomaly in production environments, ensuring optimal performance under various workloads and conditions.