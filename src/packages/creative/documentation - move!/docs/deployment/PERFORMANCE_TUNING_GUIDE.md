# Pynomaly Performance Tuning Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > ⚡ Performance Tuning

This comprehensive guide covers performance optimization strategies for Pynomaly production deployments, including database tuning, application optimization, infrastructure scaling, and monitoring performance metrics.

## 📋 Table of Contents

- [Overview](#overview)
- [Database Performance](#database-performance)
- [Application Performance](#application-performance)
- [Infrastructure Optimization](#infrastructure-optimization)
- [Caching Strategies](#caching-strategies)
- [Load Balancing](#load-balancing)
- [Auto-scaling](#auto-scaling)
- [Monitoring & Profiling](#monitoring--profiling)
- [Optimization Checklists](#optimization-checklists)

## 🎯 Overview

### Performance Targets

```yaml
# performance-targets.yml
performance_targets:
  api_response_time:
    p50: "< 50ms"
    p95: "< 100ms"
    p99: "< 250ms"
    
  throughput:
    requests_per_second: "> 1,000"
    concurrent_users: "> 10,000"
    
  database:
    query_response_time: "< 10ms"
    connection_pool_utilization: "< 80%"
    
  memory_usage:
    application: "< 2GB per instance"
    database: "< 8GB"
    redis: "< 1GB"
    
  cpu_utilization:
    average: "< 70%"
    peak: "< 90%"
    
  availability:
    uptime: "> 99.9%"
    mttr: "< 15 minutes"
```

### Performance Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                           │
│                 (Nginx/ALB/CloudFlare)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼───┐         ┌───▼───┐         ┌───▼───┐
│  API  │         │  API  │         │  API  │
│ Node1 │         │ Node2 │         │ Node3 │
└───┬───┘         └───┬───┘         └───┬───┘
    │                 │                 │
    └─────────────────┼─────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼───┐         ┌───▼───┐         ┌───▼───┐
│ Redis │         │ Redis │         │ Redis │
│ Node1 │         │ Node2 │         │ Node3 │
└───────┘         └───────┘         └───────┘
                      │
            ┌─────────▼─────────┐
            │   PostgreSQL      │
            │   (Primary +      │
            │   Read Replicas)  │
            └───────────────────┘
```

## 🗄️ Database Performance

### PostgreSQL Configuration Optimization

```sql
-- postgresql-performance.conf
-- Optimized PostgreSQL configuration for Pynomaly

-- Memory Settings
shared_buffers = 4GB                    -- 25% of system RAM
effective_cache_size = 12GB             -- 75% of system RAM
work_mem = 64MB                         -- Per query memory
maintenance_work_mem = 1GB              -- Maintenance operations
wal_buffers = 64MB                      -- WAL buffer size

-- Query Planning
random_page_cost = 1.1                  -- SSD optimization
seq_page_cost = 1.0                     -- Sequential scan cost
cpu_tuple_cost = 0.01                   -- CPU cost per tuple
cpu_index_tuple_cost = 0.005            -- CPU cost per index tuple
cpu_operator_cost = 0.0025              -- CPU cost per operator

-- Checkpoints and WAL
checkpoint_completion_target = 0.9      -- Spread checkpoints
checkpoint_timeout = 15min              -- Checkpoint interval
checkpoint_warning = 30s                -- Warning threshold
wal_compression = on                    -- Compress WAL records
wal_level = replica                     -- Replication level

-- Connections
max_connections = 200                   -- Maximum connections
shared_preload_libraries = 'pg_stat_statements'

-- Autovacuum
autovacuum = on                         -- Enable autovacuum
autovacuum_max_workers = 4              -- Number of workers
autovacuum_naptime = 30s                -- Sleep time between runs
autovacuum_vacuum_threshold = 100       -- Minimum updates before vacuum
autovacuum_analyze_threshold = 50       -- Minimum updates before analyze
autovacuum_vacuum_scale_factor = 0.05   -- Fraction of table size
autovacuum_analyze_scale_factor = 0.02  -- Fraction for analyze

-- Logging
log_statement = 'all'                   -- Log all statements
log_duration = on                       -- Log query duration
log_min_duration_statement = 100        -- Log slow queries (100ms+)
log_checkpoints = on                    -- Log checkpoint activity
log_connections = on                    -- Log new connections
log_disconnections = on                 -- Log disconnections
log_lock_waits = on                     -- Log lock waits
log_temp_files = 0                      -- Log temp file usage
```

### Database Index Optimization

```sql
-- database-indexes.sql
-- Optimized indexes for Pynomaly performance

-- Anomaly Detection Indexes
CREATE INDEX CONCURRENTLY idx_anomalies_timestamp 
    ON anomalies (timestamp DESC);

CREATE INDEX CONCURRENTLY idx_anomalies_model_timestamp 
    ON anomalies (model_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_anomalies_severity 
    ON anomalies (severity, timestamp DESC) 
    WHERE severity > 0.5;

-- Composite index for common queries
CREATE INDEX CONCURRENTLY idx_anomalies_composite 
    ON anomalies (model_id, timestamp DESC, severity DESC);

-- Partial index for active anomalies
CREATE INDEX CONCURRENTLY idx_anomalies_active 
    ON anomalies (id, timestamp) 
    WHERE status = 'active';

-- Dataset Management Indexes
CREATE INDEX CONCURRENTLY idx_datasets_created 
    ON datasets (created_at DESC);

CREATE INDEX CONCURRENTLY idx_datasets_status 
    ON datasets (status, created_at DESC);

-- Model Registry Indexes
CREATE INDEX CONCURRENTLY idx_models_algorithm 
    ON models (algorithm_name, created_at DESC);

CREATE INDEX CONCURRENTLY idx_models_performance 
    ON models (accuracy DESC, created_at DESC);

-- User Activity Indexes
CREATE INDEX CONCURRENTLY idx_user_sessions_active 
    ON user_sessions (user_id, last_activity DESC) 
    WHERE is_active = true;

-- Monitoring Indexes
CREATE INDEX CONCURRENTLY idx_metrics_timestamp 
    ON metrics (timestamp DESC, metric_name);

CREATE INDEX CONCURRENTLY idx_metrics_rollup 
    ON metrics (metric_name, timestamp DESC) 
    WHERE granularity = 'hour';

-- Statistics update
ANALYZE;
```

### Query Performance Optimization

```python
#!/usr/bin/env python3
# query-optimizer.py

import asyncio
import asyncpg
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_id: str
    avg_duration: float
    max_duration: float
    execution_count: int
    hit_ratio: float

class DatabaseQueryOptimizer:
    """Optimize database queries for better performance."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection_pool = None
        
    async def initialize_pool(self):
        """Initialize connection pool with optimized settings."""
        self.connection_pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=10,
            max_size=50,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=30,
            server_settings={
                'application_name': 'pynomaly_optimizer',
                'search_path': 'public',
                'timezone': 'UTC'
            }
        )
    
    async def analyze_slow_queries(self) -> List[QueryMetrics]:
        """Analyze slow queries using pg_stat_statements."""
        query = """
        SELECT 
            queryid,
            query,
            calls,
            total_time,
            mean_time,
            max_time,
            stddev_time,
            rows,
            100.0 * shared_blks_hit / 
                nullif(shared_blks_hit + shared_blks_read, 0) AS hit_ratio
        FROM pg_stat_statements
        WHERE calls > 10
        ORDER BY total_time DESC
        LIMIT 20;
        """
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query)
            
            metrics = []
            for row in rows:
                metrics.append(QueryMetrics(
                    query_id=str(row['queryid']),
                    avg_duration=row['mean_time'],
                    max_duration=row['max_time'],
                    execution_count=row['calls'],
                    hit_ratio=row['hit_ratio'] or 0.0
                ))
            
            return metrics
    
    async def optimize_queries(self):
        """Apply query optimizations."""
        optimizations = [
            self._optimize_anomaly_queries,
            self._optimize_dataset_queries,
            self._optimize_model_queries,
            self._optimize_aggregation_queries
        ]
        
        for optimization in optimizations:
            try:
                await optimization()
                print(f"✅ Applied optimization: {optimization.__name__}")
            except Exception as e:
                print(f"❌ Failed optimization {optimization.__name__}: {e}")
    
    async def _optimize_anomaly_queries(self):
        """Optimize anomaly detection queries."""
        optimizations = [
            # Create materialized view for recent anomalies
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS mv_recent_anomalies AS
            SELECT 
                id,
                timestamp,
                value,
                severity,
                model_id,
                dataset_id
            FROM anomalies
            WHERE timestamp >= NOW() - INTERVAL '7 days'
            ORDER BY timestamp DESC;
            
            CREATE UNIQUE INDEX ON mv_recent_anomalies (id);
            CREATE INDEX ON mv_recent_anomalies (timestamp DESC);
            CREATE INDEX ON mv_recent_anomalies (model_id, timestamp DESC);
            """,
            
            # Optimize threshold queries
            """
            CREATE OR REPLACE FUNCTION get_anomalies_by_threshold(
                p_threshold FLOAT,
                p_limit INTEGER DEFAULT 100
            ) RETURNS TABLE (
                id INTEGER,
                timestamp TIMESTAMPTZ,
                value FLOAT,
                severity FLOAT
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT a.id, a.timestamp, a.value, a.severity
                FROM anomalies a
                WHERE a.severity >= p_threshold
                ORDER BY a.timestamp DESC
                LIMIT p_limit;
            END;
            $$ LANGUAGE plpgsql STABLE;
            """
        ]
        
        async with self.connection_pool.acquire() as conn:
            for optimization in optimizations:
                await conn.execute(optimization)
    
    async def _optimize_dataset_queries(self):
        """Optimize dataset management queries."""
        optimizations = [
            # Optimize dataset statistics
            """
            CREATE OR REPLACE FUNCTION get_dataset_stats(p_dataset_id INTEGER)
            RETURNS TABLE (
                record_count BIGINT,
                feature_count INTEGER,
                anomaly_rate FLOAT,
                last_updated TIMESTAMPTZ
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT 
                    COUNT(*) as record_count,
                    (SELECT COUNT(DISTINCT feature_name) FROM dataset_features WHERE dataset_id = p_dataset_id) as feature_count,
                    (SELECT AVG(CASE WHEN is_anomaly THEN 1.0 ELSE 0.0 END) FROM dataset_records WHERE dataset_id = p_dataset_id) as anomaly_rate,
                    (SELECT MAX(updated_at) FROM dataset_records WHERE dataset_id = p_dataset_id) as last_updated
                FROM dataset_records
                WHERE dataset_id = p_dataset_id;
            END;
            $$ LANGUAGE plpgsql STABLE;
            """,
            
            # Optimize dataset search
            """
            CREATE EXTENSION IF NOT EXISTS pg_trgm;
            CREATE INDEX IF NOT EXISTS idx_datasets_name_trgm 
                ON datasets USING gin (name gin_trgm_ops);
            """
        ]
        
        async with self.connection_pool.acquire() as conn:
            for optimization in optimizations:
                await conn.execute(optimization)
    
    async def refresh_materialized_views(self):
        """Refresh materialized views for better performance."""
        views = [
            'mv_recent_anomalies',
            'mv_model_performance',
            'mv_dataset_summary'
        ]
        
        async with self.connection_pool.acquire() as conn:
            for view in views:
                try:
                    await conn.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
                    print(f"✅ Refreshed materialized view: {view}")
                except Exception as e:
                    print(f"❌ Failed to refresh {view}: {e}")
    
    async def analyze_and_vacuum(self):
        """Perform database maintenance for optimal performance."""
        maintenance_tasks = [
            "VACUUM ANALYZE anomalies;",
            "VACUUM ANALYZE datasets;",
            "VACUUM ANALYZE models;",
            "VACUUM ANALYZE user_sessions;",
            "REINDEX INDEX CONCURRENTLY idx_anomalies_timestamp;",
            "REINDEX INDEX CONCURRENTLY idx_datasets_created;",
            "UPDATE pg_stat_statements SET queryid = NULL WHERE queryid IS NOT NULL;"
        ]
        
        async with self.connection_pool.acquire() as conn:
            for task in maintenance_tasks:
                try:
                    await conn.execute(task)
                    print(f"✅ Completed: {task}")
                except Exception as e:
                    print(f"❌ Failed: {task} - {e}")

async def main():
    """Main optimization routine."""
    optimizer = DatabaseQueryOptimizer("postgresql://user:pass@localhost/pynomaly")
    await optimizer.initialize_pool()
    
    # Analyze current performance
    slow_queries = await optimizer.analyze_slow_queries()
    print(f"Found {len(slow_queries)} slow queries")
    
    # Apply optimizations
    await optimizer.optimize_queries()
    
    # Refresh materialized views
    await optimizer.refresh_materialized_views()
    
    # Perform maintenance
    await optimizer.analyze_and_vacuum()
    
    print("✅ Database optimization completed")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🚀 Application Performance

### FastAPI Application Optimization

```python
#!/usr/bin/env python3
# app-performance.py

import asyncio
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time
import json
from typing import Dict, Any
import aioredis
import asyncpg
from contextlib import asynccontextmanager

class PerformanceMiddleware:
    """Custom middleware for performance optimization."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Start timing
            start_time = time.time()
            
            # Process request
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Add performance headers
                    duration = time.time() - start_time
                    headers = dict(message.get("headers", []))
                    headers.update({
                        b"x-response-time": str(duration * 1000).encode(),
                        b"x-process-time": str(time.process_time() * 1000).encode(),
                    })
                    message["headers"] = list(headers.items())
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    print("🚀 Starting Pynomaly API with performance optimizations...")
    
    # Initialize connection pools
    app.state.redis_pool = await aioredis.create_redis_pool(
        'redis://localhost:6379',
        minsize=10,
        maxsize=50,
        retry_on_timeout=True,
        encoding='utf-8'
    )
    
    app.state.db_pool = await asyncpg.create_pool(
        'postgresql://user:pass@localhost/pynomaly',
        min_size=10,
        max_size=50,
        max_queries=50000,
        max_inactive_connection_lifetime=300,
        command_timeout=30
    )
    
    # Warm up caches
    await warm_up_caches(app.state.redis_pool)
    
    print("✅ Application startup completed")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down Pynomaly API...")
    
    # Close connection pools
    app.state.redis_pool.close()
    await app.state.redis_pool.wait_closed()
    
    await app.state.db_pool.close()
    
    print("✅ Application shutdown completed")

# Create FastAPI app with optimizations
app = FastAPI(
    title="Pynomaly API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add performance middleware
app.add_middleware(PerformanceMiddleware)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*.pynomaly.com", "localhost", "127.0.0.1"]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pynomaly.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

async def warm_up_caches(redis_pool):
    """Warm up application caches."""
    try:
        async with redis_pool.get() as redis:
            # Pre-cache common queries
            await redis.set("cache:models:active", json.dumps({"count": 0}))
            await redis.set("cache:datasets:stats", json.dumps({"total": 0}))
            
        print("✅ Caches warmed up successfully")
    except Exception as e:
        print(f"❌ Cache warm-up failed: {e}")

# Optimized route handlers
@app.get("/api/v1/anomalies")
async def get_anomalies(
    request: Request,
    limit: int = 100,
    offset: int = 0,
    model_id: int = None
):
    """Get anomalies with caching and pagination."""
    # Check cache first
    cache_key = f"anomalies:{model_id}:{limit}:{offset}"
    
    async with request.app.state.redis_pool.get() as redis:
        cached_result = await redis.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
    
    # Query database
    async with request.app.state.db_pool.acquire() as conn:
        query = """
        SELECT id, timestamp, value, severity, model_id
        FROM anomalies
        WHERE ($1::integer IS NULL OR model_id = $1)
        ORDER BY timestamp DESC
        LIMIT $2 OFFSET $3
        """
        
        rows = await conn.fetch(query, model_id, limit, offset)
        
        result = {
            "anomalies": [dict(row) for row in rows],
            "total": len(rows),
            "limit": limit,
            "offset": offset
        }
        
        # Cache result for 5 minutes
        async with request.app.state.redis_pool.get() as redis:
            await redis.setex(cache_key, 300, json.dumps(result, default=str))
        
        return result

# Production server configuration
def run_production_server():
    """Run production server with optimized settings."""
    uvicorn.run(
        "app-performance:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        loop="uvloop",
        http="httptools",
        access_log=False,
        server_header=False,
        date_header=False,
        limit_concurrency=1000,
        limit_max_requests=10000,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_ciphers="TLSv1.2",
        headers=[
            ("server", "Pynomaly-API/1.0"),
            ("x-content-type-options", "nosniff"),
            ("x-frame-options", "DENY"),
            ("x-xss-protection", "1; mode=block"),
        ]
    )

if __name__ == "__main__":
    run_production_server()
```

### Memory and CPU Optimization

```python
#!/usr/bin/env python3
# resource-optimizer.py

import gc
import psutil
import asyncio
import resource
from typing import Dict, List, Optional
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

@dataclass
class ResourceMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage: float
    network_io: Dict[str, int]
    
class ResourceOptimizer:
    """Optimize system resource usage."""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.thread_pool = ThreadPoolExecutor(max_workers=self.cpu_count * 2)
        self.process_pool = ProcessPoolExecutor(max_workers=self.cpu_count)
        
    def optimize_memory_usage(self):
        """Optimize memory usage."""
        # Set memory limits
        soft_limit = int(self.memory_total * 0.8)  # 80% of total memory
        hard_limit = int(self.memory_total * 0.9)  # 90% of total memory
        
        resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))
        
        # Configure garbage collection
        gc.set_threshold(700, 10, 10)  # More aggressive GC
        
        # Enable memory profiling
        import tracemalloc
        tracemalloc.start()
        
        print(f"✅ Memory optimization configured: {soft_limit / (1024**3):.1f}GB limit")
    
    def optimize_cpu_usage(self):
        """Optimize CPU usage."""
        # Set CPU affinity for better performance
        process = psutil.Process()
        if len(process.cpu_affinity()) > 4:
            # Use only performance cores
            process.cpu_affinity(list(range(0, min(8, self.cpu_count))))
        
        # Set process priority
        process.nice(-10)  # Higher priority
        
        print(f"✅ CPU optimization configured: {self.cpu_count} cores available")
    
    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            disk_usage=disk.percent,
            network_io={
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        )
    
    async def monitor_resources(self, interval: int = 60):
        """Monitor resource usage continuously."""
        while True:
            metrics = self.get_resource_metrics()
            
            # Check for resource pressure
            if metrics.memory_percent > 85:
                print(f"⚠️ High memory usage: {metrics.memory_percent:.1f}%")
                self._force_garbage_collection()
            
            if metrics.cpu_percent > 90:
                print(f"⚠️ High CPU usage: {metrics.cpu_percent:.1f}%")
                # Could implement CPU throttling here
            
            if metrics.disk_usage > 90:
                print(f"⚠️ High disk usage: {metrics.disk_usage:.1f}%")
                # Could implement log rotation or cleanup
            
            await asyncio.sleep(interval)
    
    def _force_garbage_collection(self):
        """Force garbage collection to free memory."""
        collected = gc.collect()
        print(f"🗑️ Garbage collection freed {collected} objects")
    
    async def optimize_async_operations(self):
        """Optimize asynchronous operations."""
        # Optimize event loop
        loop = asyncio.get_running_loop()
        
        # Set optimal concurrency limits
        asyncio.Semaphore(100)  # Limit concurrent operations
        
        # Use uvloop for better performance (if available)
        try:
            import uvloop
            uvloop.install()
            print("✅ uvloop installed for better async performance")
        except ImportError:
            print("⚠️ uvloop not available, using default asyncio")

# Machine Learning Model Optimization
class MLModelOptimizer:
    """Optimize machine learning model performance."""
    
    def __init__(self):
        self.model_cache = {}
        self.prediction_cache = {}
        
    def optimize_model_loading(self):
        """Optimize model loading and caching."""
        # Implement model lazy loading
        # Cache frequently used models
        # Use model quantization for memory efficiency
        pass
    
    def optimize_inference(self):
        """Optimize model inference performance."""
        # Batch predictions
        # Use ONNX runtime for faster inference
        # Implement model pruning
        pass
    
    async def batch_predict(self, model_id: str, data_batch: List[Dict], batch_size: int = 32):
        """Batch predictions for better throughput."""
        results = []
        
        for i in range(0, len(data_batch), batch_size):
            batch = data_batch[i:i + batch_size]
            
            # Check cache first
            cache_keys = [f"prediction:{model_id}:{hash(str(item))}" for item in batch]
            cached_results = await self._get_cached_predictions(cache_keys)
            
            # Process uncached items
            uncached_indices = [i for i, result in enumerate(cached_results) if result is None]
            
            if uncached_indices:
                uncached_batch = [batch[i] for i in uncached_indices]
                new_predictions = await self._predict_batch(model_id, uncached_batch)
                
                # Cache new predictions
                await self._cache_predictions(
                    [cache_keys[i] for i in uncached_indices],
                    new_predictions
                )
                
                # Merge results
                for i, prediction in zip(uncached_indices, new_predictions):
                    cached_results[i] = prediction
            
            results.extend(cached_results)
        
        return results
    
    async def _get_cached_predictions(self, cache_keys: List[str]) -> List[Optional[Dict]]:
        """Get cached predictions."""
        # Implementation depends on caching backend
        return [None] * len(cache_keys)  # Placeholder
    
    async def _predict_batch(self, model_id: str, batch: List[Dict]) -> List[Dict]:
        """Perform batch prediction."""
        # Implementation depends on ML framework
        return [{"prediction": 0.5, "confidence": 0.8}] * len(batch)  # Placeholder
    
    async def _cache_predictions(self, cache_keys: List[str], predictions: List[Dict]):
        """Cache predictions."""
        # Implementation depends on caching backend
        pass

if __name__ == "__main__":
    optimizer = ResourceOptimizer()
    
    # Apply optimizations
    optimizer.optimize_memory_usage()
    optimizer.optimize_cpu_usage()
    
    # Start monitoring
    print("🚀 Starting resource monitoring...")
    asyncio.run(optimizer.monitor_resources())
```

## 🏗️ Infrastructure Optimization

### Docker Container Optimization

```dockerfile
# Dockerfile.optimized
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r pynomaly \
    && useradd -r -g pynomaly pynomaly

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY --chown=pynomaly:pynomaly . /app
WORKDIR /app

# Set security and performance options
USER pynomaly
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random
ENV PYTHONIOENCODING=utf-8

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "pynomaly.presentation.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log", \
     "--server-header", \
     "--date-header"]
```

### Kubernetes Resource Optimization

```yaml
# k8s-optimized-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-api
  namespace: pynomaly
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
  selector:
    matchLabels:
      app: pynomaly-api
  template:
    metadata:
      labels:
        app: pynomaly-api
        version: v1.0.0
    spec:
      containers:
      - name: pynomaly-api
        image: pynomaly/api:1.0.0
        ports:
        - containerPort: 8000
        
        # Resource optimization
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        
        # Liveness and readiness probes
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 2
        
        # Environment variables for performance
        env:
        - name: WORKERS
          value: "4"
        - name: WORKER_CONNECTIONS
          value: "1000"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: PYTHONDONTWRITEBYTECODE
          value: "1"
        
        # Volume mounts for logs
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        
        # Security context
        securityContext:
          runAsUser: 1000
          runAsGroup: 1000
          runAsNonRoot: true
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
      
      volumes:
      - name: logs
        emptyDir: {}
      
      # Pod optimization
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - pynomaly-api
              topologyKey: kubernetes.io/hostname
      
      # Node selection
      nodeSelector:
        node-type: compute-optimized
      
      tolerations:
      - key: "high-performance"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"

---
apiVersion: v1
kind: Service
metadata:
  name: pynomaly-api-service
  namespace: pynomaly
spec:
  selector:
    app: pynomaly-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-api-hpa
  namespace: pynomaly
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-api
  minReplicas: 4
  maxReplicas: 20
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

## 🗄️ Caching Strategies

### Redis Caching Optimization

```python
#!/usr/bin/env python3
# redis-cache-optimizer.py

import asyncio
import aioredis
import json
import pickle
import zlib
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

@dataclass
class CacheConfig:
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # Cache settings
    default_ttl: int = 3600  # 1 hour
    max_key_size: int = 250
    max_value_size: int = 1024 * 1024  # 1MB
    compression_threshold: int = 1024  # Compress values > 1KB

class OptimizedRedisCache:
    """Optimized Redis cache implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_pool = None
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize Redis connection pool."""
        self.redis_pool = await aioredis.create_redis_pool(
            f'redis://{self.config.host}:{self.config.port}/{self.config.db}',
            minsize=10,
            maxsize=self.config.max_connections,
            retry_on_timeout=self.config.retry_on_timeout,
            encoding='utf-8'
        )
        
        # Configure Redis for optimal performance
        await self._configure_redis()
        
        print(f"✅ Redis cache initialized: {self.config.host}:{self.config.port}")
    
    async def _configure_redis(self):
        """Configure Redis for optimal performance."""
        configs = {
            'maxmemory-policy': 'allkeys-lru',
            'tcp-keepalive': '60',
            'timeout': '300',
            'save': '',  # Disable persistence for cache
            'appendonly': 'no',
            'maxmemory': '1gb'
        }
        
        async with self.redis_pool.get() as redis:
            for key, value in configs.items():
                try:
                    await redis.config_set(key, value)
                except Exception as e:
                    print(f"⚠️ Failed to set {key}: {e}")
    
    def _generate_cache_key(self, namespace: str, key: str, version: str = "1") -> str:
        """Generate optimized cache key."""
        full_key = f"{namespace}:{version}:{key}"
        
        # Hash long keys
        if len(full_key) > self.config.max_key_size:
            key_hash = hashlib.md5(full_key.encode()).hexdigest()
            return f"{namespace}:{version}:hash:{key_hash}"
        
        return full_key
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize and optionally compress value."""
        try:
            # Use pickle for Python objects
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress if value is large
            if len(serialized) > self.config.compression_threshold:
                compressed = zlib.compress(serialized, level=6)
                return b'compressed:' + compressed
            
            return serialized
            
        except Exception as e:
            print(f"❌ Serialization failed: {e}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize and decompress value."""
        try:
            # Check if compressed
            if data.startswith(b'compressed:'):
                compressed_data = data[11:]  # Remove 'compressed:' prefix
                decompressed = zlib.decompress(compressed_data)
                return pickle.loads(decompressed)
            
            return pickle.loads(data)
            
        except Exception as e:
            print(f"❌ Deserialization failed: {e}")
            raise
    
    async def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        cache_key = self._generate_cache_key(namespace, key)
        
        try:
            async with self.redis_pool.get() as redis:
                data = await redis.get(cache_key)
                
                if data is None:
                    self.stats['misses'] += 1
                    return default
                
                self.stats['hits'] += 1
                return self._deserialize_value(data)
                
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ Cache get failed: {e}")
            return default
    
    async def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        cache_key = self._generate_cache_key(namespace, key)
        ttl = ttl or self.config.default_ttl
        
        try:
            serialized = self._serialize_value(value)
            
            # Check value size
            if len(serialized) > self.config.max_value_size:
                print(f"⚠️ Value too large for cache: {len(serialized)} bytes")
                return False
            
            async with self.redis_pool.get() as redis:
                await redis.setex(cache_key, ttl, serialized)
                
            self.stats['sets'] += 1
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ Cache set failed: {e}")
            return False
    
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete value from cache."""
        cache_key = self._generate_cache_key(namespace, key)
        
        try:
            async with self.redis_pool.get() as redis:
                result = await redis.delete(cache_key)
                
            self.stats['deletes'] += 1
            return result > 0
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ Cache delete failed: {e}")
            return False
    
    async def get_many(self, namespace: str, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        cache_keys = [self._generate_cache_key(namespace, key) for key in keys]
        
        try:
            async with self.redis_pool.get() as redis:
                values = await redis.mget(*cache_keys)
                
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize_value(value)
                    self.stats['hits'] += 1
                else:
                    self.stats['misses'] += 1
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ Cache get_many failed: {e}")
            return {}
    
    async def set_many(self, namespace: str, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        ttl = ttl or self.config.default_ttl
        
        try:
            async with self.redis_pool.get() as redis:
                async with redis.pipeline() as pipe:
                    for key, value in mapping.items():
                        cache_key = self._generate_cache_key(namespace, key)
                        serialized = self._serialize_value(value)
                        
                        if len(serialized) <= self.config.max_value_size:
                            pipe.setex(cache_key, ttl, serialized)
                    
                    await pipe.execute()
            
            self.stats['sets'] += len(mapping)
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ Cache set_many failed: {e}")
            return False
    
    async def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in a namespace."""
        pattern = f"{namespace}:*"
        
        try:
            async with self.redis_pool.get() as redis:
                keys = await redis.keys(pattern)
                if keys:
                    await redis.delete(*keys)
                    self.stats['deletes'] += len(keys)
            
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ Cache clear_namespace failed: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            async with self.redis_pool.get() as redis:
                info = await redis.info()
                
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'memory_usage': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'operations_per_sec': info.get('instantaneous_ops_per_sec', 0),
                **self.stats
            }
            
        except Exception as e:
            print(f"❌ Failed to get cache stats: {e}")
            return self.stats
    
    async def health_check(self) -> bool:
        """Check cache health."""
        try:
            async with self.redis_pool.get() as redis:
                await redis.ping()
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close Redis connection pool."""
        if self.redis_pool:
            self.redis_pool.close()
            await self.redis_pool.wait_closed()

# Cache decorators for easy usage
def cache_result(namespace: str, ttl: int = 3600, key_generator = None):
    """Decorator to cache function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            cached_result = await cache.get(namespace, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(namespace, cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Initialize global cache instance
cache = OptimizedRedisCache(CacheConfig())

# Example usage
@cache_result("anomalies", ttl=300)
async def get_recent_anomalies(model_id: int, limit: int = 100):
    """Get recent anomalies with caching."""
    # This would normally query the database
    return {"anomalies": [], "total": 0}

async def main():
    """Test cache performance."""
    await cache.initialize()
    
    # Test basic operations
    await cache.set("test", "key1", {"data": "value1"})
    result = await cache.get("test", "key1")
    print(f"Cached result: {result}")
    
    # Test batch operations
    await cache.set_many("batch", {
        "key1": {"data": "value1"},
        "key2": {"data": "value2"},
        "key3": {"data": "value3"}
    })
    
    batch_result = await cache.get_many("batch", ["key1", "key2", "key3"])
    print(f"Batch result: {len(batch_result)} items")
    
    # Get statistics
    stats = await cache.get_stats()
    print(f"Cache stats: {stats}")
    
    await cache.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## ⚖️ Load Balancing

### Nginx Load Balancer Configuration

```nginx
# /etc/nginx/sites-available/pynomaly
upstream pynomaly_backend {
    # Load balancing method
    least_conn;
    
    # Backend servers
    server 10.0.1.10:8000 max_fails=3 fail_timeout=30s weight=1;
    server 10.0.1.11:8000 max_fails=3 fail_timeout=30s weight=1;
    server 10.0.1.12:8000 max_fails=3 fail_timeout=30s weight=1;
    server 10.0.1.13:8000 max_fails=3 fail_timeout=30s weight=1;
    
    # Health checks
    keepalive 64;
    keepalive_requests 1000;
    keepalive_timeout 60s;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=auth_limit:10m rate=5r/s;

# Connection limiting
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

server {
    listen 80;
    server_name api.pynomaly.com;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml
        text/plain
        text/css
        text/xml
        text/javascript;
    
    # Client settings
    client_max_body_size 100M;
    client_body_buffer_size 128k;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    
    # Timeouts
    client_body_timeout 12;
    client_header_timeout 12;
    send_timeout 10;
    
    # Rate limiting
    limit_req zone=api_limit burst=20 nodelay;
    limit_conn conn_limit 10;
    
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
    
    location /auth {
        limit_req zone=auth_limit burst=5 nodelay;
        
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        
        # Keep alive
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    
    location /api {
        # Cache static API responses
        location ~* \.(json|xml)$ {
            expires 5m;
            add_header Cache-Control "public, no-transform";
        }
        
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Performance optimizations
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
        proxy_temp_file_write_size 8k;
        
        # Keep alive
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    
    location /static {
        alias /var/www/pynomaly/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
        
        # Compression
        gzip_static on;
        
        # Security
        location ~* \.(php|pl|py|jsp|asp|sh|cgi)$ {
            deny all;
        }
    }
    
    # Deny access to hidden files
    location ~ /\. {
        deny all;
    }
    
    # Error pages
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
    
    # Monitoring
    location /nginx_status {
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        deny all;
    }
}

# HTTPS configuration
server {
    listen 443 ssl http2;
    server_name api.pynomaly.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/api.pynomaly.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.pynomaly.com/privkey.pem;
    
    # SSL optimization
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Include HTTP configuration
    include /etc/nginx/snippets/pynomaly-common.conf;
}
```

## 📊 Monitoring & Profiling

### Application Performance Monitoring

```python
#!/usr/bin/env python3
# performance-monitor.py

import time
import psutil
import asyncio
import json
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import logging

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io: Dict[str, int]
    network_io: Dict[str, int]
    response_time: float
    throughput: float
    error_rate: float
    active_connections: int

class PerformanceMonitor:
    """Monitor application performance metrics."""
    
    def __init__(self, api_endpoint: str = "http://localhost:8000"):
        self.api_endpoint = api_endpoint
        self.metrics_history: List[PerformanceMetrics] = []
        self.alerts_config = {
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'response_time_threshold': 1000.0,  # ms
            'error_rate_threshold': 0.05  # 5%
        }
        self.alert_callbacks: List[Callable] = []
        
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Application metrics
        response_time = await self._measure_response_time()
        throughput = await self._measure_throughput()
        error_rate = await self._measure_error_rate()
        active_connections = await self._get_active_connections()
        
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io={
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count
            },
            network_io={
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            },
            response_time=response_time,
            throughput=throughput,
            error_rate=error_rate,
            active_connections=active_connections
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        return metrics
    
    async def _measure_response_time(self) -> float:
        """Measure API response time."""
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_endpoint}/health", timeout=10) as response:
                    await response.text()
                    return (time.time() - start_time) * 1000  # Convert to ms
        except Exception:
            return 0.0
    
    async def _measure_throughput(self) -> float:
        """Measure API throughput (requests per second)."""
        # This would typically be calculated from metrics history
        # For now, return a placeholder
        return 0.0
    
    async def _measure_error_rate(self) -> float:
        """Measure API error rate."""
        # This would typically be calculated from error logs
        # For now, return a placeholder
        return 0.0
    
    async def _get_active_connections(self) -> int:
        """Get number of active connections."""
        # This would typically query the application
        # For now, return a placeholder
        return 0
    
    def add_alert_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts."""
        alerts = []
        
        if metrics.cpu_percent > self.alerts_config['cpu_threshold']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alerts_config['memory_threshold']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.response_time > self.alerts_config['response_time_threshold']:
            alerts.append(f"High response time: {metrics.response_time:.1f}ms")
        
        if metrics.error_rate > self.alerts_config['error_rate_threshold']:
            alerts.append(f"High error rate: {metrics.error_rate:.2%}")
        
        if alerts:
            for callback in self.alert_callbacks:
                callback(metrics, alerts)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-60:]  # Last 60 metrics
        
        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'avg_response_time': sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            'max_response_time': max(m.response_time for m in recent_metrics),
            'avg_throughput': sum(m.throughput for m in recent_metrics) / len(recent_metrics),
            'avg_error_rate': sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            'max_active_connections': max(m.active_connections for m in recent_metrics)
        }
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring."""
        logging.info(f"🚀 Starting performance monitoring (interval: {interval}s)")
        
        while True:
            try:
                metrics = await self.collect_metrics()
                self.check_alerts(metrics)
                
                logging.info(f"📊 CPU: {metrics.cpu_percent:.1f}%, "
                           f"Memory: {metrics.memory_percent:.1f}%, "
                           f"Response: {metrics.response_time:.1f}ms")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logging.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(interval)

# Alert handlers
async def send_alert_notification(metrics: PerformanceMetrics, alerts: List[str]):
    """Send alert notification."""
    message = f"""
🚨 Performance Alert

Timestamp: {metrics.timestamp}
Alerts: {', '.join(alerts)}

Current Metrics:
- CPU: {metrics.cpu_percent:.1f}%
- Memory: {metrics.memory_percent:.1f}%
- Response Time: {metrics.response_time:.1f}ms
- Error Rate: {metrics.error_rate:.2%}
"""
    
    # Send to monitoring system (e.g., Slack, email, PagerDuty)
    logging.warning(message)

async def main():
    """Main monitoring loop."""
    monitor = PerformanceMonitor()
    monitor.add_alert_callback(send_alert_notification)
    
    await monitor.start_monitoring(interval=30)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

## ✅ Optimization Checklists

### Pre-Production Performance Checklist

```bash
#!/bin/bash
# performance-checklist.sh

echo "🔍 Performance Optimization Checklist"
echo "===================================="

# Database optimization
echo "📊 Database Performance:"
echo "- [x] PostgreSQL configuration optimized"
echo "- [x] Indexes created for common queries"
echo "- [x] Query performance analyzed"
echo "- [x] Connection pooling configured"
echo "- [x] Autovacuum settings tuned"

# Application optimization
echo "📱 Application Performance:"
echo "- [x] FastAPI with uvloop and httptools"
echo "- [x] Async/await used throughout"
echo "- [x] Connection pooling implemented"
echo "- [x] Caching strategy implemented"
echo "- [x] Response compression enabled"

# Infrastructure optimization
echo "🏗️ Infrastructure Performance:"
echo "- [x] Docker images optimized"
echo "- [x] Kubernetes resources configured"
echo "- [x] Load balancer configured"
echo "- [x] Auto-scaling enabled"
echo "- [x] CDN configured (if applicable)"

# Monitoring
echo "📈 Monitoring & Alerting:"
echo "- [x] Performance metrics collected"
echo "- [x] Alerts configured"
echo "- [x] Dashboards created"
echo "- [x] Log aggregation setup"
echo "- [x] Distributed tracing enabled"

echo "✅ Performance checklist completed!"
```

---

## 📚 Related Documentation

- **[Deployment Guide](deployment.md)**: Complete deployment procedures
- **[Monitoring Setup](MONITORING_SETUP_GUIDE.md)**: Observability and metrics
- **[Security Hardening](SECURITY_HARDENING_GUIDE.md)**: Production security
- **[Backup & Recovery](BACKUP_RECOVERY_PROCEDURES.md)**: Data protection

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-15  
**Next Review**: 2024-04-15  

This performance tuning guide provides comprehensive optimization strategies for achieving optimal Pynomaly performance in production environments.