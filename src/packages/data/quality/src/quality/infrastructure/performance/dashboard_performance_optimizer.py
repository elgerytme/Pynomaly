"""
Dashboard Performance Optimizer
Optimizes dashboard loading performance to achieve sub-2 second load times.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import OrderedDict
import json
import gzip
import hashlib

import numpy as np
import pandas as pd

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import memcache
    MEMCACHE_AVAILABLE = True
except ImportError:
    MEMCACHE_AVAILABLE = False


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    
    # Cache types
    enable_memory_cache: bool = True
    enable_redis_cache: bool = False
    enable_compression: bool = True
    
    # Cache settings
    memory_cache_size: int = 1000  # Number of items
    memory_cache_ttl_seconds: int = 300  # 5 minutes
    redis_ttl_seconds: int = 900  # 15 minutes
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Compression settings
    compression_threshold_bytes: int = 1024  # 1KB
    compression_level: int = 6


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Parallel processing
    max_workers: int = 4
    enable_async_aggregation: bool = True
    batch_size: int = 1000
    
    # Data optimization
    enable_data_sampling: bool = True
    sample_size_threshold: int = 10000
    max_sample_size: int = 5000
    
    # Query optimization
    enable_query_batching: bool = True
    max_batch_size: int = 100
    query_timeout_seconds: int = 30
    
    # Response optimization
    enable_response_compression: bool = True
    response_compression_threshold: int = 1024
    
    # Precomputation
    enable_precomputation: bool = True
    precompute_interval_minutes: int = 5
    precompute_popular_queries: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    
    # Load times
    total_load_time_ms: float = 0.0
    cache_lookup_time_ms: float = 0.0
    data_fetch_time_ms: float = 0.0
    computation_time_ms: float = 0.0
    serialization_time_ms: float = 0.0
    
    # Cache performance
    cache_hit_rate: float = 0.0
    cache_miss_count: int = 0
    cache_hit_count: int = 0
    
    # Data efficiency
    data_reduction_ratio: float = 0.0
    compression_ratio: float = 0.0
    
    # Concurrency
    concurrent_requests: int = 0
    queued_requests: int = 0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class LRUCache:
    """Simple LRU cache implementation."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
        
        self.access_times[key] = datetime.now()
    
    def clear_expired(self, ttl_seconds: int):
        """Clear expired items."""
        now = datetime.now()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if (now - access_time).total_seconds() > ttl_seconds
        ]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            del self.access_times[key]


class DataCompressor:
    """Data compression utility."""
    
    @staticmethod
    def compress(data: str, level: int = 6) -> bytes:
        """Compress string data."""
        return gzip.compress(data.encode('utf-8'), compresslevel=level)
    
    @staticmethod
    def decompress(compressed_data: bytes) -> str:
        """Decompress data."""
        return gzip.decompress(compressed_data).decode('utf-8')
    
    @staticmethod
    def should_compress(data: str, threshold: int = 1024) -> bool:
        """Check if data should be compressed."""
        return len(data.encode('utf-8')) > threshold


class QueryOptimizer:
    """Query optimization utilities."""
    
    @staticmethod
    def optimize_dataframe_query(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Optimize DataFrame query with filters."""
        result = df
        
        # Apply filters efficiently
        for column, filter_value in filters.items():
            if column not in df.columns:
                continue
            
            if isinstance(filter_value, dict):
                # Range filters
                if 'min' in filter_value:
                    result = result[result[column] >= filter_value['min']]
                if 'max' in filter_value:
                    result = result[result[column] <= filter_value['max']]
                if 'in' in filter_value:
                    result = result[result[column].isin(filter_value['in'])]
            else:
                # Exact match
                result = result[result[column] == filter_value]
        
        return result
    
    @staticmethod
    def optimize_aggregation(df: pd.DataFrame, group_by: List[str], aggregations: Dict[str, str]) -> pd.DataFrame:
        """Optimize DataFrame aggregation."""
        if not group_by:
            # Simple aggregation
            result = {}
            for column, agg_func in aggregations.items():
                if column in df.columns:
                    if agg_func == 'mean':
                        result[column] = df[column].mean()
                    elif agg_func == 'sum':
                        result[column] = df[column].sum()
                    elif agg_func == 'count':
                        result[column] = df[column].count()
                    elif agg_func == 'max':
                        result[column] = df[column].max()
                    elif agg_func == 'min':
                        result[column] = df[column].min()
            
            return pd.DataFrame([result])
        else:
            # Group by aggregation
            valid_group_by = [col for col in group_by if col in df.columns]
            if not valid_group_by:
                return df
            
            grouped = df.groupby(valid_group_by)
            
            agg_dict = {}
            for column, agg_func in aggregations.items():
                if column in df.columns:
                    agg_dict[column] = agg_func
            
            if agg_dict:
                return grouped.agg(agg_dict).reset_index()
            else:
                return grouped.size().reset_index(name='count')


class DashboardPerformanceOptimizer:
    """
    Performance optimizer for dashboard loading with sub-2s target.
    """
    
    def __init__(
        self,
        cache_config: CacheConfig = None,
        performance_config: PerformanceConfig = None
    ):
        self.cache_config = cache_config or CacheConfig()
        self.performance_config = performance_config or PerformanceConfig()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize caching layers
        self.memory_cache = LRUCache(self.cache_config.memory_cache_size)
        self.redis_client = None
        self.memcache_client = None
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.query_stats: Dict[str, List[float]] = {}
        self.popular_queries: Dict[str, int] = {}
        
        # Precomputed data storage
        self.precomputed_data: Dict[str, Any] = {}
        self.precomputation_lock = threading.RLock()
        
        # Request queue for load balancing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.active_requests: Dict[str, datetime] = {}
        
        self._initialize_cache_backends()
        
        if self.performance_config.enable_precomputation:
            self._start_precomputation_worker()
    
    def _initialize_cache_backends(self):
        """Initialize cache backends."""
        try:
            if self.cache_config.enable_redis_cache and REDIS_AVAILABLE:
                self.redis_client = redis.Redis(
                    host=self.cache_config.redis_host,
                    port=self.cache_config.redis_port,
                    db=self.cache_config.redis_db,
                    password=self.cache_config.redis_password,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                self.logger.info("Redis cache initialized")
            
        except Exception as e:
            self.logger.warning(f"Redis cache initialization failed: {str(e)}")
            self.redis_client = None
        
        try:
            if MEMCACHE_AVAILABLE:
                self.memcache_client = memcache.Client(['127.0.0.1:11211'])
                self.logger.info("Memcache initialized")
            
        except Exception as e:
            self.logger.warning(f"Memcache initialization failed: {str(e)}")
            self.memcache_client = None
    
    async def optimize_dashboard_query(
        self,
        query_id: str,
        query_function: Callable,
        query_params: Dict[str, Any] = None,
        cache_ttl: int = None
    ) -> Dict[str, Any]:
        """
        Optimize dashboard query execution with caching and performance enhancements.
        
        Args:
            query_id: Unique identifier for the query
            query_function: Function to execute if not cached
            query_params: Parameters for the query
            cache_ttl: Cache TTL override
            
        Returns:
            Optimized query result with performance metrics
        """
        start_time = time.time()
        query_params = query_params or {}
        cache_ttl = cache_ttl or self.cache_config.memory_cache_ttl_seconds
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(query_id, query_params)
            
            # Track request
            self.active_requests[cache_key] = datetime.now()
            self.metrics.concurrent_requests = len(self.active_requests)
            
            # Try cache lookup
            cache_start = time.time()
            cached_result = await self._get_from_cache(cache_key)
            cache_time = (time.time() - cache_start) * 1000
            
            if cached_result is not None:
                self.metrics.cache_hit_count += 1
                self.metrics.cache_lookup_time_ms = cache_time
                self.metrics.total_load_time_ms = (time.time() - start_time) * 1000
                
                self.logger.debug(f"Cache hit for {query_id} in {cache_time:.1f}ms")
                
                return {
                    'data': cached_result,
                    'cache_hit': True,
                    'load_time_ms': self.metrics.total_load_time_ms,
                    'cache_lookup_time_ms': cache_time
                }
            
            # Cache miss - execute query with optimizations
            self.metrics.cache_miss_count += 1
            
            # Execute optimized query
            fetch_start = time.time()
            result = await self._execute_optimized_query(query_function, query_params)
            fetch_time = (time.time() - fetch_start) * 1000
            
            # Post-process result for optimization
            optimized_result = await self._optimize_result_data(result)
            
            # Cache the result
            await self._set_in_cache(cache_key, optimized_result, cache_ttl)
            
            # Update metrics
            total_time = (time.time() - start_time) * 1000
            self.metrics.data_fetch_time_ms = fetch_time
            self.metrics.total_load_time_ms = total_time
            
            # Track query performance
            self._track_query_performance(query_id, total_time)
            
            # Update cache hit rate
            total_requests = self.metrics.cache_hit_count + self.metrics.cache_miss_count
            self.metrics.cache_hit_rate = self.metrics.cache_hit_count / total_requests
            
            self.logger.info(f"Query {query_id} executed in {total_time:.1f}ms (fetch: {fetch_time:.1f}ms)")
            
            return {
                'data': optimized_result,
                'cache_hit': False,
                'load_time_ms': total_time,
                'fetch_time_ms': fetch_time,
                'cache_lookup_time_ms': cache_time
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing query {query_id}: {str(e)}")
            raise
        finally:
            # Clean up request tracking
            if cache_key in self.active_requests:
                del self.active_requests[cache_key]
            self.metrics.concurrent_requests = len(self.active_requests)
    
    async def precompute_dashboard_data(
        self,
        dashboard_queries: Dict[str, Callable],
        priority_queries: List[str] = None
    ):
        """
        Precompute dashboard data for popular queries.
        
        Args:
            dashboard_queries: Dictionary of query functions
            priority_queries: List of high-priority queries to precompute
        """
        if not self.performance_config.enable_precomputation:
            return
        
        try:
            self.logger.info("Starting dashboard data precomputation")
            
            priority_queries = priority_queries or []
            
            # Identify queries to precompute
            queries_to_compute = []
            
            # Add priority queries
            for query_id in priority_queries:
                if query_id in dashboard_queries:
                    queries_to_compute.append((query_id, dashboard_queries[query_id], 'priority'))
            
            # Add popular queries
            if self.performance_config.precompute_popular_queries:
                popular_sorted = sorted(
                    self.popular_queries.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for query_id, _ in popular_sorted[:10]:  # Top 10 popular queries
                    if query_id in dashboard_queries and query_id not in priority_queries:
                        queries_to_compute.append((query_id, dashboard_queries[query_id], 'popular'))
            
            # Execute precomputation in parallel
            if queries_to_compute:
                with ThreadPoolExecutor(max_workers=self.performance_config.max_workers) as executor:
                    futures = []
                    
                    for query_id, query_func, query_type in queries_to_compute:
                        future = executor.submit(self._precompute_single_query, query_id, query_func)
                        futures.append((future, query_id, query_type))
                    
                    # Process results
                    for future, query_id, query_type in futures:
                        try:
                            result = future.result(timeout=self.performance_config.query_timeout_seconds)
                            
                            with self.precomputation_lock:
                                self.precomputed_data[query_id] = {
                                    'data': result,
                                    'computed_at': datetime.now(),
                                    'type': query_type
                                }
                            
                            self.logger.debug(f"Precomputed {query_type} query: {query_id}")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to precompute {query_id}: {str(e)}")
            
            self.logger.info(f"Precomputation completed for {len(queries_to_compute)} queries")
            
        except Exception as e:
            self.logger.error(f"Error in precomputation: {str(e)}")
    
    async def optimize_data_response(
        self,
        data: Any,
        response_format: str = "json",
        enable_compression: bool = None
    ) -> Dict[str, Any]:
        """
        Optimize data response for fast transmission.
        
        Args:
            data: Data to optimize
            response_format: Response format (json, msgpack, etc.)
            enable_compression: Whether to enable compression
            
        Returns:
            Optimized response data
        """
        start_time = time.time()
        enable_compression = enable_compression if enable_compression is not None else self.performance_config.enable_response_compression
        
        try:
            # Serialize data
            if response_format == "json":
                serialized = json.dumps(data, default=str)
            else:
                serialized = str(data)
            
            serialization_time = (time.time() - start_time) * 1000
            original_size = len(serialized.encode('utf-8'))
            
            # Apply compression if beneficial
            compressed = None
            compression_ratio = 1.0
            
            if enable_compression and original_size > self.performance_config.response_compression_threshold:
                compress_start = time.time()
                compressed = DataCompressor.compress(serialized, level=self.cache_config.compression_level)
                compress_time = (time.time() - compress_start) * 1000
                
                compressed_size = len(compressed)
                compression_ratio = original_size / compressed_size
                
                self.logger.debug(f"Compressed response: {original_size} -> {compressed_size} bytes ({compression_ratio:.2f}x)")
            
            # Update metrics
            self.metrics.serialization_time_ms = serialization_time
            self.metrics.compression_ratio = compression_ratio
            
            return {
                'data': compressed if compressed else serialized,
                'format': response_format,
                'compressed': compressed is not None,
                'original_size': original_size,
                'final_size': len(compressed) if compressed else original_size,
                'compression_ratio': compression_ratio,
                'serialization_time_ms': serialization_time
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing response: {str(e)}")
            return {
                'data': data,
                'format': response_format,
                'compressed': False,
                'error': str(e)
            }
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return {
            'hit_rate': self.metrics.cache_hit_rate,
            'hits': self.metrics.cache_hit_count,
            'misses': self.metrics.cache_miss_count,
            'memory_cache_size': len(self.memory_cache.cache),
            'memory_cache_capacity': self.memory_cache.capacity,
            'redis_available': self.redis_client is not None,
            'memcache_available': self.memcache_client is not None
        }
    
    def get_query_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get query performance statistics."""
        stats = {}
        
        for query_id, times in self.query_stats.items():
            if times:
                stats[query_id] = {
                    'avg_time_ms': np.mean(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'std_time_ms': np.std(times),
                    'call_count': len(times),
                    'popularity_score': self.popular_queries.get(query_id, 0)
                }
        
        return stats
    
    # Private methods
    
    def _generate_cache_key(self, query_id: str, params: Dict[str, Any]) -> str:
        """Generate cache key from query ID and parameters."""
        # Create deterministic key from parameters
        param_str = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:8]
        
        return f"dashboard:{query_id}:{param_hash}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache (multi-tier lookup)."""
        
        # Try memory cache first
        if self.cache_config.enable_memory_cache:
            result = self.memory_cache.get(cache_key)
            if result is not None:
                return result
        
        # Try Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    # Check if compressed
                    if cached_data.startswith('GZIP:'):
                        compressed_data = cached_data[5:].encode('latin-1')
                        data = DataCompressor.decompress(compressed_data)
                        result = json.loads(data)
                    else:
                        result = json.loads(cached_data)
                    
                    # Also cache in memory for faster access
                    self.memory_cache.set(cache_key, result, self.cache_config.memory_cache_ttl_seconds)
                    
                    return result
            except Exception as e:
                self.logger.warning(f"Redis cache lookup failed: {str(e)}")
        
        # Try Memcache
        if self.memcache_client:
            try:
                result = self.memcache_client.get(cache_key)
                if result:
                    # Cache in memory and Redis for faster access
                    self.memory_cache.set(cache_key, result, self.cache_config.memory_cache_ttl_seconds)
                    return result
            except Exception as e:
                self.logger.warning(f"Memcache lookup failed: {str(e)}")
        
        return None
    
    async def _set_in_cache(self, cache_key: str, data: Any, ttl: int):
        """Set data in cache (multi-tier storage)."""
        
        # Store in memory cache
        if self.cache_config.enable_memory_cache:
            self.memory_cache.set(cache_key, data, ttl)
        
        # Store in Redis
        if self.redis_client:
            try:
                serialized_data = json.dumps(data, default=str)
                
                # Compress if beneficial
                if (self.cache_config.enable_compression and 
                    len(serialized_data) > self.cache_config.compression_threshold_bytes):
                    compressed = DataCompressor.compress(serialized_data, self.cache_config.compression_level)
                    cache_value = 'GZIP:' + compressed.decode('latin-1')
                else:
                    cache_value = serialized_data
                
                self.redis_client.setex(cache_key, self.cache_config.redis_ttl_seconds, cache_value)
                
            except Exception as e:
                self.logger.warning(f"Redis cache store failed: {str(e)}")
        
        # Store in Memcache
        if self.memcache_client:
            try:
                self.memcache_client.set(cache_key, data, time=ttl)
            except Exception as e:
                self.logger.warning(f"Memcache store failed: {str(e)}")
    
    async def _execute_optimized_query(self, query_function: Callable, params: Dict[str, Any]) -> Any:
        """Execute query with performance optimizations."""
        
        # Apply data sampling if configured
        if self.performance_config.enable_data_sampling:
            sample_params = params.copy()
            if 'limit' not in sample_params and params.get('data_size', 0) > self.performance_config.sample_size_threshold:
                sample_params['limit'] = self.performance_config.max_sample_size
                sample_params['sample'] = True
            params = sample_params
        
        # Execute query
        if asyncio.iscoroutinefunction(query_function):
            result = await query_function(**params)
        else:
            # Run in thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, query_function, **params)
        
        return result
    
    async def _optimize_result_data(self, data: Any) -> Any:
        """Optimize result data for caching and transmission."""
        
        if isinstance(data, pd.DataFrame):
            # Optimize DataFrame
            optimized_df = data.copy()
            
            # Round float columns to reduce precision if appropriate
            float_columns = optimized_df.select_dtypes(include=[np.float64, np.float32]).columns
            for col in float_columns:
                optimized_df[col] = optimized_df[col].round(4)
            
            # Convert to more efficient data types where possible
            for col in optimized_df.select_dtypes(include=['int64']).columns:
                if optimized_df[col].min() >= -2**31 and optimized_df[col].max() < 2**31:
                    optimized_df[col] = optimized_df[col].astype('int32')
            
            # Convert DataFrame to dict for JSON serialization
            self.metrics.data_reduction_ratio = len(data) / max(len(optimized_df), 1)
            
            return optimized_df.to_dict('records')
        
        elif isinstance(data, dict):
            # Optimize dictionary data
            optimized_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    optimized_data[key] = value.tolist()
                elif isinstance(value, pd.DataFrame):
                    optimized_data[key] = await self._optimize_result_data(value)
                else:
                    optimized_data[key] = value
            
            return optimized_data
        
        elif isinstance(data, list):
            # Optimize list data
            if len(data) > self.performance_config.max_sample_size * 2:
                # Sample large lists
                step = len(data) // self.performance_config.max_sample_size
                optimized_data = data[::step]
                self.metrics.data_reduction_ratio = len(data) / len(optimized_data)
                return optimized_data
        
        return data
    
    def _track_query_performance(self, query_id: str, execution_time_ms: float):
        """Track query performance for optimization."""
        
        if query_id not in self.query_stats:
            self.query_stats[query_id] = []
        
        # Keep last 100 measurements
        self.query_stats[query_id].append(execution_time_ms)
        if len(self.query_stats[query_id]) > 100:
            self.query_stats[query_id] = self.query_stats[query_id][-100:]
        
        # Track popularity
        self.popular_queries[query_id] = self.popular_queries.get(query_id, 0) + 1
    
    def _precompute_single_query(self, query_id: str, query_function: Callable) -> Any:
        """Precompute a single query."""
        try:
            # Execute query with default parameters
            if asyncio.iscoroutinefunction(query_function):
                # Run async function in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(query_function())
                finally:
                    loop.close()
            else:
                result = query_function()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error precomputing {query_id}: {str(e)}")
            return None
    
    def _start_precomputation_worker(self):
        """Start background precomputation worker."""
        def precompute_worker():
            while True:
                try:
                    time.sleep(self.performance_config.precompute_interval_minutes * 60)
                    
                    # Clean expired memory cache
                    self.memory_cache.clear_expired(self.cache_config.memory_cache_ttl_seconds)
                    
                    # Update metrics
                    self.metrics.memory_usage_mb = self._estimate_memory_usage()
                    
                except Exception as e:
                    self.logger.error(f"Error in precomputation worker: {str(e)}")
        
        # Start worker thread
        worker_thread = threading.Thread(target=precompute_worker, daemon=True)
        worker_thread.start()
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def clear_cache(self):
        """Clear all caches."""
        # Clear memory cache
        self.memory_cache.cache.clear()
        self.memory_cache.access_times.clear()
        
        # Clear Redis cache
        if self.redis_client:
            try:
                # Delete all dashboard cache keys
                keys = self.redis_client.keys("dashboard:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.warning(f"Failed to clear Redis cache: {str(e)}")
        
        # Clear precomputed data
        with self.precomputation_lock:
            self.precomputed_data.clear()
        
        self.logger.info("All caches cleared")
    
    def warm_up_cache(self, dashboard_queries: Dict[str, Callable]):
        """Warm up cache with common queries."""
        asyncio.create_task(self.precompute_dashboard_data(dashboard_queries))
        self.logger.info("Cache warm-up initiated")