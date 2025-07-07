"""Performance optimization service for anomaly detection operations."""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._memory_threshold_mb = 1000  # Alert if memory usage exceeds 1GB
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def optimize_numpy_arrays(self, data: np.ndarray) -> np.ndarray:
        """Optimize numpy array memory usage."""
        # Use smaller data types when possible
        if data.dtype == np.float64:
            # Check if we can use float32 without significant precision loss
            float32_data = data.astype(np.float32)
            if np.allclose(data, float32_data, rtol=1e-6):
                self.logger.debug("Optimized array from float64 to float32")
                return float32_data
        
        # Ensure array is contiguous for better cache performance
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
            self.logger.debug("Made array contiguous for better cache performance")
        
        return data
    
    def batch_process_large_arrays(
        self, 
        data: np.ndarray, 
        func: callable, 
        batch_size: int = 10000
    ) -> np.ndarray:
        """Process large arrays in batches to reduce memory usage."""
        if len(data) <= batch_size:
            return func(data)
        
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_result = func(batch)
            results.append(batch_result)
            
            # Force garbage collection between batches
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        return np.concatenate(results)
    
    def memory_efficient_distance_matrix(
        self, 
        X: np.ndarray, 
        Y: Optional[np.ndarray] = None,
        metric: str = 'euclidean'
    ) -> np.ndarray:
        """Compute distance matrix with memory optimization."""
        if Y is None:
            Y = X
        
        n, m = len(X), len(Y)
        
        # For large matrices, use chunked computation
        if n * m > 1000000:  # > 1M elements
            chunk_size = max(100, int(np.sqrt(1000000)))
            distances = np.zeros((n, m), dtype=np.float32)
            
            for i in range(0, n, chunk_size):
                i_end = min(i + chunk_size, n)
                for j in range(0, m, chunk_size):
                    j_end = min(j + chunk_size, m)
                    
                    if metric == 'euclidean':
                        chunk_dist = np.sqrt(
                            np.sum((X[i:i_end, None, :] - Y[None, j:j_end, :]) ** 2, axis=2)
                        )
                    else:
                        # Fallback to scipy for other metrics
                        from scipy.spatial.distance import cdist
                        chunk_dist = cdist(X[i:i_end], Y[j:j_end], metric=metric)
                    
                    distances[i:i_end, j:j_end] = chunk_dist
            
            return distances
        else:
            # Small matrices - compute directly
            if metric == 'euclidean':
                return np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))
            else:
                from scipy.spatial.distance import cdist
                return cdist(X, Y, metric=metric)


class ComputationOptimizer:
    """Computation optimization utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
    def optimize_sklearn_estimator(self, estimator) -> None:
        """Optimize sklearn estimator parameters for performance."""
        # Set n_jobs for parallel algorithms
        if hasattr(estimator, 'n_jobs'):
            if estimator.n_jobs is None or estimator.n_jobs == 1:
                estimator.n_jobs = min(4, psutil.cpu_count())
                self.logger.debug(f"Set n_jobs={estimator.n_jobs} for parallel processing")
        
        # Optimize specific algorithms
        estimator_name = estimator.__class__.__name__
        
        if estimator_name == 'IsolationForest':
            # Use smaller max_samples for large datasets
            if hasattr(estimator, 'max_samples') and estimator.max_samples == 'auto':
                estimator.max_samples = min(256, 0.1)  # Limit sample size
                
        elif estimator_name == 'LocalOutlierFactor':
            # Optimize neighbor algorithms
            if hasattr(estimator, 'algorithm') and estimator.algorithm == 'auto':
                estimator.algorithm = 'kd_tree'  # Usually faster for moderate dimensions
                
        elif estimator_name == 'OneClassSVM':
            # Disable probability estimates if not needed
            if hasattr(estimator, 'probability'):
                estimator.probability = False
    
    @staticmethod
    def fast_correlation_matrix(data: np.ndarray) -> np.ndarray:
        """Compute correlation matrix efficiently."""
        # Center the data
        data_centered = data - np.mean(data, axis=0)
        
        # Use numpy's optimized dot product
        cov_matrix = np.dot(data_centered.T, data_centered) / (len(data) - 1)
        
        # Compute standard deviations
        std_devs = np.sqrt(np.diag(cov_matrix))
        
        # Avoid division by zero
        std_devs = np.maximum(std_devs, 1e-10)
        
        # Compute correlation matrix
        correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
        
        # Clip to valid correlation range to handle numerical errors
        correlation_matrix = np.clip(correlation_matrix, -1.0, 1.0)
        
        return correlation_matrix
    
    def parallel_feature_computation(
        self, 
        data: np.ndarray, 
        feature_funcs: List[callable]
    ) -> np.ndarray:
        """Compute features in parallel."""
        async def compute_feature_async(func, data_chunk):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._thread_pool, func, data_chunk)
        
        async def compute_all_features():
            tasks = [compute_feature_async(func, data) for func in feature_funcs]
            return await asyncio.gather(*tasks)
        
        # Run the async computation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            features = loop.run_until_complete(compute_all_features())
            return np.column_stack(features)
        finally:
            loop.close()


class CacheOptimizer:
    """Cache optimization for frequently computed values."""
    
    def __init__(self, max_cache_size: int = 100):
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_cache_key(self, data: np.ndarray, operation: str) -> str:
        """Generate cache key for data and operation."""
        # Use hash of data properties instead of full data
        data_hash = hash((
            data.shape,
            data.dtype,
            tuple(data.flat[:min(100, data.size)]),  # Sample first 100 elements
            operation
        ))
        return str(data_hash)
    
    def get_cached_result(self, cache_key: str, max_age_seconds: float = 300) -> Optional[Any]:
        """Get cached result if available and not expired."""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if time.time() - timestamp <= max_age_seconds:
                return result
            else:
                # Remove expired entry
                del self._cache[cache_key]
        return None
    
    def cache_result(self, cache_key: str, result: Any) -> None:
        """Cache computation result."""
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        self._cache[cache_key] = (result, time.time())
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self.logger.debug("Cleared optimization cache")


@numba.jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
def _numba_distance_computation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Numba-optimized distance computation."""
    n, m = X.shape[0], Y.shape[0]
    distances = np.zeros((n, m), dtype=np.float32)
    
    for i in range(n):
        for j in range(m):
            dist = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - Y[j, k]
                dist += diff * diff
            distances[i, j] = np.sqrt(dist)
    
    return distances


class PerformanceOptimizationService:
    """Main service for performance optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_optimizer = MemoryOptimizer()
        self.computation_optimizer = ComputationOptimizer()
        self.cache_optimizer = CacheOptimizer()
        
        # Performance metrics
        self.metrics = {
            'operations_optimized': 0,
            'memory_saved_mb': 0.0,
            'time_saved_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
    
    def optimize_data_preprocessing(self, data: np.ndarray) -> np.ndarray:
        """Optimize data preprocessing operations."""
        start_time = time.perf_counter()
        start_memory = self.memory_optimizer.get_memory_usage()
        
        # Apply memory optimizations
        optimized_data = self.memory_optimizer.optimize_numpy_arrays(data)
        
        # Record metrics
        end_time = time.perf_counter()
        end_memory = self.memory_optimizer.get_memory_usage()
        
        self.metrics['operations_optimized'] += 1
        self.metrics['memory_saved_mb'] += max(0, start_memory - end_memory)
        self.metrics['time_saved_ms'] += (end_time - start_time) * 1000
        
        return optimized_data
    
    def optimize_anomaly_detection(
        self, 
        estimator, 
        X_train: np.ndarray, 
        X_test: Optional[np.ndarray] = None
    ) -> Tuple[Any, Optional[np.ndarray]]:
        """Optimize anomaly detection training and prediction."""
        start_time = time.perf_counter()
        
        # Optimize the estimator
        self.computation_optimizer.optimize_sklearn_estimator(estimator)
        
        # Optimize training data
        X_train_opt = self.optimize_data_preprocessing(X_train)
        
        # Check cache for pre-trained model
        cache_key = self.cache_optimizer.get_cache_key(X_train_opt, estimator.__class__.__name__)
        cached_model = self.cache_optimizer.get_cached_result(cache_key)
        
        if cached_model is not None:
            self.metrics['cache_hits'] += 1
            self.logger.debug("Using cached trained model")
            trained_estimator = cached_model
        else:
            self.metrics['cache_misses'] += 1
            
            # Train the model
            trained_estimator = estimator.fit(X_train_opt)
            
            # Cache the trained model
            self.cache_optimizer.cache_result(cache_key, trained_estimator)
        
        # Optimize prediction if test data provided
        predictions = None
        if X_test is not None:
            X_test_opt = self.optimize_data_preprocessing(X_test)
            
            # Use batch processing for large datasets
            if len(X_test_opt) > 10000:
                predictions = self.memory_optimizer.batch_process_large_arrays(
                    X_test_opt, 
                    trained_estimator.predict,
                    batch_size=5000
                )
            else:
                predictions = trained_estimator.predict(X_test_opt)
        
        end_time = time.perf_counter()
        self.metrics['time_saved_ms'] += (end_time - start_time) * 1000
        
        return trained_estimator, predictions
    
    def optimize_distance_computation(
        self, 
        X: np.ndarray, 
        Y: Optional[np.ndarray] = None,
        metric: str = 'euclidean'
    ) -> np.ndarray:
        """Optimize distance matrix computation."""
        cache_key = self.cache_optimizer.get_cache_key(
            X if Y is None else np.vstack([X, Y]), 
            f"distance_{metric}"
        )
        
        cached_result = self.cache_optimizer.get_cached_result(cache_key)
        if cached_result is not None:
            self.metrics['cache_hits'] += 1
            return cached_result
        
        self.metrics['cache_misses'] += 1
        
        # Use optimized distance computation
        if NUMBA_AVAILABLE and metric == 'euclidean':
            Y = X if Y is None else Y
            distances = _numba_distance_computation(X, Y)
        else:
            distances = self.memory_optimizer.memory_efficient_distance_matrix(X, Y, metric)
        
        self.cache_optimizer.cache_result(cache_key, distances)
        return distances
    
    def optimize_streaming_detection(
        self, 
        detector, 
        data_stream: List[np.ndarray],
        batch_size: int = 100
    ) -> List[np.ndarray]:
        """Optimize streaming anomaly detection."""
        results = []
        
        # Process in batches to optimize throughput
        for i in range(0, len(data_stream), batch_size):
            batch = data_stream[i:i + batch_size]
            
            # Stack batch for vectorized processing
            if len(batch) > 1:
                batch_array = np.vstack(batch)
                batch_results = detector.decision_function(batch_array)
                
                # Split results back to individual samples
                for j, result in enumerate(batch_results):
                    results.append(np.array([result]))
            else:
                # Single sample
                result = detector.decision_function(batch[0].reshape(1, -1))
                results.append(result)
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        cache_total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        cache_hit_rate = self.metrics['cache_hits'] / cache_total if cache_total > 0 else 0
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'current_memory_mb': self.memory_optimizer.get_memory_usage(),
            'numba_available': NUMBA_AVAILABLE,
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        for key in self.metrics:
            self.metrics[key] = 0.0 if 'mb' in key or 'ms' in key else 0
    
    def cleanup_resources(self) -> None:
        """Clean up optimization resources."""
        self.cache_optimizer.clear_cache()
        gc.collect()
        self.logger.info("Cleaned up optimization resources")


# Singleton instance
_optimization_service = None


def get_optimization_service() -> PerformanceOptimizationService:
    """Get singleton optimization service."""
    global _optimization_service
    if _optimization_service is None:
        _optimization_service = PerformanceOptimizationService()
    return _optimization_service