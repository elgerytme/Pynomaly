"""Optimized anomaly detection service with performance and memory enhancements."""

from __future__ import annotations

import gc
import time
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..entities.detection_result import DetectionResult
from ...infrastructure.logging import get_logger, timing_decorator
from ...infrastructure.logging.error_handler import ErrorHandler, AlgorithmError
from ...infrastructure.monitoring import get_metrics_collector

logger = get_logger(__name__)
error_handler = ErrorHandler(logger._logger)
metrics_collector = get_metrics_collector()


@dataclass
class OptimizationConfig:
    """Configuration for detection optimization."""
    enable_model_caching: bool = True
    enable_batch_processing: bool = True
    enable_parallel_processing: bool = True
    enable_memory_optimization: bool = True
    cache_size_limit: int = 10
    batch_size_threshold: int = 100
    max_workers: int = 4
    memory_limit_mb: float = 500.0
    use_numba_acceleration: bool = False  # Optional JIT compilation
    

class ModelCache:
    """Thread-safe model cache with LRU eviction and memory management."""
    
    def __init__(self, max_size: int = 10, memory_limit_mb: float = 500.0):
        self.max_size = max_size
        self.memory_limit_mb = memory_limit_mb
        self._cache: Dict[str, Tuple[Any, float, int]] = {}  # model, timestamp, access_count
        self._lock = threading.RLock()
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get model from cache."""
        with self._lock:
            if key in self._cache:
                model, timestamp, access_count = self._cache[key]
                self._cache[key] = (model, timestamp, access_count + 1)
                
                # Update access order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                
                return model
            return None
    
    def put(self, key: str, model: Any) -> None:
        """Store model in cache with memory management."""
        with self._lock:
            # Check memory usage before adding
            self._enforce_memory_limit()
            
            # Add new model
            current_time = time.time()
            self._cache[key] = (model, current_time, 1)
            
            if key not in self._access_order:
                self._access_order.append(key)
            
            # Enforce size limit
            while len(self._cache) > self.max_size:
                self._evict_lru()
    
    def _enforce_memory_limit(self) -> None:
        """Enforce memory limit by evicting models."""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        while memory_mb > self.memory_limit_mb and self._cache:
            self._evict_lru()
            memory_mb = process.memory_info().rss / 1024 / 1024
    
    def _evict_lru(self) -> None:
        """Evict least recently used model."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                gc.collect()  # Force garbage collection
    
    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_access = sum(access_count for _, _, access_count in self._cache.values())
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_accesses": total_access,
                "hit_rate": total_access / max(1, len(self._cache))
            }


class OptimizedDetectionService:
    """High-performance anomaly detection service with advanced optimizations."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize optimized detection service."""
        self.config = config or OptimizationConfig()
        
        # Initialize model cache if enabled
        if self.config.enable_model_caching:
            self.model_cache = ModelCache(
                max_size=self.config.cache_size_limit,
                memory_limit_mb=self.config.memory_limit_mb
            )
        else:
            self.model_cache = None
        
        # Thread pool for parallel processing
        if self.config.enable_parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        else:
            self.executor = None
        
        # Performance tracking
        self._algorithm_stats: Dict[str, Dict[str, float]] = {}
        self._optimization_history: List[Dict[str, Any]] = []
        
        # Pre-compile optimized functions if numba is available
        if self.config.use_numba_acceleration:
            self._setup_numba_acceleration()
        
        logger.info("OptimizedDetectionService initialized",
                   config=self.config.__dict__)
    
    def __del__(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=False)
    
    def detect_anomalies(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str = "iforest", 
        contamination: float = 0.1,
        **kwargs: Any
    ) -> DetectionResult:
        """Optimized anomaly detection with performance enhancements."""
        start_time = time.perf_counter()
        
        try:
            # Input validation and preprocessing
            data = self._preprocess_data(data)
            
            # Choose optimization strategy based on data size
            if len(data) >= self.config.batch_size_threshold and self.config.enable_batch_processing:
                predictions = self._detect_with_batch_optimization(
                    data, algorithm, contamination, **kwargs
                )
            else:
                predictions = self._detect_with_model_caching(
                    data, algorithm, contamination, **kwargs
                )
            
            # Get confidence scores
            confidence_scores = self._get_optimized_confidence_scores(
                data, algorithm, contamination, **kwargs
            )
            
            # Create result
            result = DetectionResult(
                predictions=predictions,
                confidence_scores=confidence_scores,
                algorithm=algorithm,
                metadata={
                    "contamination": contamination,
                    "data_shape": data.shape,
                    "optimization_applied": True,
                    "algorithm_params": kwargs
                }
            )
            
            # Update performance statistics
            duration = time.perf_counter() - start_time
            self._update_algorithm_stats(algorithm, duration, len(data))
            
            # Log performance
            samples_per_sec = len(data) / duration if duration > 0 else 0
            logger.info("Optimized detection completed",
                       algorithm=algorithm,
                       samples=len(data),
                       duration_ms=duration * 1000,
                       throughput=samples_per_sec,
                       anomalies=result.anomaly_count)
            
            return result
            
        except Exception as e:
            logger.error("Optimized detection failed",
                        algorithm=algorithm,
                        error=str(e))
            raise AlgorithmError(f"Optimized detection failed: {str(e)}")
    
    def detect_batch(
        self,
        data_batches: List[npt.NDArray[np.floating]],
        algorithm: str = "iforest",
        contamination: float = 0.1,
        **kwargs: Any
    ) -> List[DetectionResult]:
        """Batch detection with parallel processing optimization."""
        if not self.config.enable_parallel_processing or not self.executor:
            # Sequential processing fallback
            return [
                self.detect_anomalies(batch, algorithm, contamination, **kwargs)
                for batch in data_batches
            ]
        
        # Parallel batch processing
        futures = []
        for batch in data_batches:
            future = self.executor.submit(
                self.detect_anomalies, batch, algorithm, contamination, **kwargs
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)  # 60 second timeout
                results.append(result)
            except Exception as e:
                logger.error("Batch detection failed", error=str(e))
                # Create error result
                error_result = DetectionResult(
                    predictions=np.array([]),
                    confidence_scores=None,
                    algorithm=algorithm,
                    metadata={"error": str(e)}
                )
                results.append(error_result)
        
        return results
    
    def _preprocess_data(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Optimized data preprocessing with memory efficiency."""
        # Ensure data is in optimal format
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        # Convert to optimal dtype if needed
        if data.dtype != np.float32 and self.config.enable_memory_optimization:
            # Use float32 for memory efficiency if precision allows
            if np.allclose(data.astype(np.float32), data, rtol=1e-6):
                data = data.astype(np.float32)
        
        # Handle NaN and infinite values efficiently
        if np.any(~np.isfinite(data)):
            # Use vectorized operations for better performance
            finite_mask = np.isfinite(data)
            if not np.all(finite_mask):
                # Replace non-finite values with column means
                col_means = np.nanmean(data, axis=0)
                data = np.where(finite_mask, data, col_means)
        
        return data
    
    def _detect_with_model_caching(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Detection with intelligent model caching."""
        if not self.config.enable_model_caching or not self.model_cache:
            return self._detect_direct(data, algorithm, contamination, **kwargs)
        
        # Create cache key based on algorithm and parameters
        cache_key = self._create_cache_key(algorithm, contamination, **kwargs)
        
        # Try to get cached model
        cached_model = self.model_cache.get(cache_key)
        
        if cached_model is not None:
            # Use cached model for prediction
            try:
                if hasattr(cached_model, 'predict'):
                    predictions = cached_model.predict(data)
                elif hasattr(cached_model, 'fit_predict'):
                    # For models that don't support separate predict
                    predictions = self._detect_direct(data, algorithm, contamination, **kwargs)
                else:
                    predictions = self._detect_direct(data, algorithm, contamination, **kwargs)
                
                logger.debug("Used cached model", algorithm=algorithm, cache_key=cache_key)
                return predictions
                
            except Exception as e:
                logger.warning("Cached model failed, falling back to direct detection",
                             error=str(e))
        
        # Direct detection and cache the model if possible
        predictions = self._detect_direct(data, algorithm, contamination, **kwargs)
        
        # Try to cache the model for future use
        if algorithm == "iforest":
            try:
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(contamination=contamination, random_state=42, **kwargs)
                model.fit(data)
                self.model_cache.put(cache_key, model)
                logger.debug("Cached model", algorithm=algorithm, cache_key=cache_key)
            except Exception as e:
                logger.debug("Failed to cache model", algorithm=algorithm, error=str(e))
        
        return predictions
    
    def _detect_with_batch_optimization(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Detection with batch processing optimization for large datasets."""
        batch_size = min(self.config.batch_size_threshold, len(data) // 2)
        
        if batch_size >= len(data):
            # Data is not large enough for batching
            return self._detect_with_model_caching(data, algorithm, contamination, **kwargs)
        
        # Split data into batches
        batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        
        if self.config.enable_parallel_processing and self.executor:
            # Parallel batch processing
            futures = []
            for batch in batches:
                future = self.executor.submit(
                    self._detect_with_model_caching, batch, algorithm, contamination, **kwargs
                )
                futures.append(future)
            
            # Collect results
            batch_predictions = []
            for future in futures:
                batch_pred = future.result()
                batch_predictions.append(batch_pred)
        else:
            # Sequential batch processing
            batch_predictions = []
            for batch in batches:
                batch_pred = self._detect_with_model_caching(batch, algorithm, contamination, **kwargs)
                batch_predictions.append(batch_pred)
        
        # Combine batch results
        return np.concatenate(batch_predictions)
    
    def _detect_direct(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Direct algorithm execution with optimizations."""
        if algorithm == "iforest":
            return self._optimized_isolation_forest(data, contamination, **kwargs)
        elif algorithm == "lof":
            return self._optimized_local_outlier_factor(data, contamination, **kwargs)
        elif algorithm == "ocsvm":
            return self._optimized_one_class_svm(data, contamination, **kwargs)
        else:
            raise AlgorithmError(f"Unsupported algorithm: {algorithm}")
    
    @timing_decorator(operation="optimized_isolation_forest")
    def _optimized_isolation_forest(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Optimized Isolation Forest with performance enhancements."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Optimize parameters for performance
            optimized_kwargs = self._optimize_iforest_params(data, **kwargs)
            
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1 if self.config.enable_parallel_processing else 1,
                **optimized_kwargs
            )
            
            # Use fit_predict for better memory efficiency
            predictions = model.fit_predict(data)
            
            # Force garbage collection for memory optimization
            if self.config.enable_memory_optimization:
                del model
                gc.collect()
            
            return predictions.astype(np.integer)
            
        except ImportError:
            raise AlgorithmError("scikit-learn required for IsolationForest")
        except Exception as e:
            raise AlgorithmError(f"Optimized Isolation Forest failed: {str(e)}")
    
    @timing_decorator(operation="optimized_lof")
    def _optimized_local_outlier_factor(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Optimized Local Outlier Factor with performance enhancements."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # Optimize parameters for performance
            optimized_kwargs = self._optimize_lof_params(data, **kwargs)
            
            model = LocalOutlierFactor(
                contamination=contamination,
                n_jobs=-1 if self.config.enable_parallel_processing else 1,
                **optimized_kwargs
            )
            
            predictions = model.fit_predict(data)
            
            # Memory cleanup
            if self.config.enable_memory_optimization:
                del model
                gc.collect()
            
            return predictions.astype(np.integer)
            
        except ImportError:
            raise AlgorithmError("scikit-learn required for LocalOutlierFactor")
        except Exception as e:
            raise AlgorithmError(f"Optimized LOF failed: {str(e)}")
    
    @timing_decorator(operation="optimized_ocsvm")
    def _optimized_one_class_svm(
        self,
        data: npt.NDArray[np.floating],
        contamination: float,
        **kwargs: Any
    ) -> npt.NDArray[np.integer]:
        """Optimized One-Class SVM with performance enhancements."""
        try:
            from sklearn.svm import OneClassSVM
            
            # Optimize parameters for performance
            optimized_kwargs = self._optimize_ocsvm_params(data, **kwargs)
            
            # Calculate nu parameter from contamination
            nu = min(contamination, 0.5)
            
            model = OneClassSVM(
                nu=nu,
                **optimized_kwargs
            )
            
            predictions = model.fit_predict(data)
            
            # Memory cleanup
            if self.config.enable_memory_optimization:
                del model
                gc.collect()
            
            return predictions.astype(np.integer)
            
        except ImportError:
            raise AlgorithmError("scikit-learn required for OneClassSVM")
        except Exception as e:
            raise AlgorithmError(f"Optimized One-Class SVM failed: {str(e)}")
    
    def _optimize_iforest_params(self, data: npt.NDArray[np.floating], **kwargs: Any) -> Dict[str, Any]:
        """Optimize Isolation Forest parameters based on data characteristics."""
        optimized = kwargs.copy()
        
        n_samples, n_features = data.shape
        
        # Optimize n_estimators based on data size
        if 'n_estimators' not in optimized:
            if n_samples < 1000:
                optimized['n_estimators'] = 50  # Fewer trees for small datasets
            elif n_samples < 10000:
                optimized['n_estimators'] = 100  # Default
            else:
                optimized['n_estimators'] = 200  # More trees for large datasets
        
        # Optimize max_samples based on data size
        if 'max_samples' not in optimized:
            if n_samples < 1000:
                optimized['max_samples'] = min(256, n_samples)
            else:
                optimized['max_samples'] = 'auto'
        
        # Optimize max_features based on dimensionality
        if 'max_features' not in optimized:
            if n_features <= 10:
                optimized['max_features'] = 1.0  # Use all features for low-dim data
            else:
                optimized['max_features'] = min(1.0, 10.0 / n_features)  # Limit features for high-dim
        
        return optimized
    
    def _optimize_lof_params(self, data: npt.NDArray[np.floating], **kwargs: Any) -> Dict[str, Any]:
        """Optimize LOF parameters based on data characteristics."""
        optimized = kwargs.copy()
        
        n_samples, n_features = data.shape
        
        # Optimize n_neighbors based on data size
        if 'n_neighbors' not in optimized:
            if n_samples < 100:
                optimized['n_neighbors'] = min(20, n_samples - 1)
            elif n_samples < 1000:
                optimized['n_neighbors'] = 20
            else:
                optimized['n_neighbors'] = min(50, int(np.sqrt(n_samples)))
        
        # Use efficient algorithms for different data characteristics
        if 'algorithm' not in optimized:
            if n_features <= 10:
                optimized['algorithm'] = 'kd_tree'  # Efficient for low dimensions
            else:
                optimized['algorithm'] = 'brute'  # More reliable for high dimensions
        
        return optimized
    
    def _optimize_ocsvm_params(self, data: npt.NDArray[np.floating], **kwargs: Any) -> Dict[str, Any]:
        """Optimize One-Class SVM parameters based on data characteristics."""
        optimized = kwargs.copy()
        
        n_samples, n_features = data.shape
        
        # Choose kernel based on data characteristics
        if 'kernel' not in optimized:
            if n_features <= 10 and n_samples < 1000:
                optimized['kernel'] = 'rbf'  # RBF for small, low-dim data
            else:
                optimized['kernel'] = 'linear'  # Linear for efficiency
        
        # Optimize gamma for RBF kernel
        if optimized.get('kernel') == 'rbf' and 'gamma' not in optimized:
            optimized['gamma'] = 'scale'  # Use sklearn's automatic scaling
        
        # Set cache size for memory efficiency
        if 'cache_size' not in optimized:
            if self.config.enable_memory_optimization:
                optimized['cache_size'] = 100  # Smaller cache for memory efficiency
            else:
                optimized['cache_size'] = 200  # Default
        
        return optimized
    
    def _get_optimized_confidence_scores(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination: float,
        **kwargs: Any
    ) -> Optional[npt.NDArray[np.floating]]:
        """Get confidence scores with optimizations."""
        try:
            if algorithm == "iforest":
                # Use decision function for faster score computation
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(contamination=contamination, random_state=42, **kwargs)
                model.fit(data)
                scores = model.decision_function(data)
                
                # Convert to positive scores (higher = more anomalous)
                scores = -scores
                scores = (scores - scores.min()) / (scores.max() - scores.min())
                
                return scores.astype(np.float32 if self.config.enable_memory_optimization else np.float64)
            
            # For other algorithms, return None to avoid unnecessary computation
            return None
            
        except Exception as e:
            logger.debug("Failed to compute confidence scores", error=str(e))
            return None
    
    def _create_cache_key(self, algorithm: str, contamination: float, **kwargs: Any) -> str:
        """Create cache key for model caching."""
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        params_str = "_".join(f"{k}={v}" for k, v in sorted_params)
        return f"{algorithm}_{contamination}_{hash(params_str)}"
    
    def _update_algorithm_stats(self, algorithm: str, duration: float, samples: int) -> None:
        """Update algorithm performance statistics."""
        if algorithm not in self._algorithm_stats:
            self._algorithm_stats[algorithm] = {
                "total_time": 0.0,
                "total_samples": 0,
                "call_count": 0,
                "avg_time": 0.0,
                "avg_throughput": 0.0
            }
        
        stats = self._algorithm_stats[algorithm]
        stats["total_time"] += duration
        stats["total_samples"] += samples
        stats["call_count"] += 1
        stats["avg_time"] = stats["total_time"] / stats["call_count"]
        stats["avg_throughput"] = stats["total_samples"] / stats["total_time"]
    
    def _setup_numba_acceleration(self) -> None:
        """Set up Numba JIT compilation for performance-critical functions."""
        try:
            import numba
            logger.info("Numba acceleration enabled")
            # Could implement JIT-compiled helper functions here
        except ImportError:
            logger.debug("Numba not available, skipping JIT acceleration")
            self.config.use_numba_acceleration = False
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "config": self.config.__dict__,
            "algorithm_performance": self._algorithm_stats.copy()
        }
        
        if self.model_cache:
            stats["cache_stats"] = self.model_cache.get_stats()
        
        if self.executor:
            stats["thread_pool"] = {
                "max_workers": self.config.max_workers,
                "active": True
            }
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all cached models and reset statistics."""
        if self.model_cache:
            self.model_cache.clear()
        
        self._algorithm_stats.clear()
        self._optimization_history.clear()
        
        logger.info("Optimization cache and statistics cleared")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "optimization_config": self.config.__dict__,
            "algorithm_statistics": self._algorithm_stats,
            "cache_statistics": self.model_cache.get_stats() if self.model_cache else None,
            "memory_info": self._get_memory_info(),
            "recommendations": self._generate_optimization_recommendations()
        }
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def _generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        # Analyze algorithm performance
        if self._algorithm_stats:
            slowest_algorithm = max(
                self._algorithm_stats.items(),
                key=lambda x: x[1]["avg_time"]
            )
            
            if slowest_algorithm[1]["avg_time"] > 1.0:  # More than 1 second average
                recommendations.append({
                    "type": "algorithm_optimization",
                    "algorithm": slowest_algorithm[0],
                    "issue": "slow_performance",
                    "suggestion": "Consider using batch processing or parameter optimization",
                    "current_avg_time": slowest_algorithm[1]["avg_time"]
                })
        
        # Check cache performance
        if self.model_cache:
            cache_stats = self.model_cache.get_stats()
            if cache_stats["hit_rate"] < 0.5:  # Low hit rate
                recommendations.append({
                    "type": "cache_optimization",
                    "issue": "low_cache_hit_rate",
                    "suggestion": "Consider increasing cache size or reviewing cache key strategy",
                    "current_hit_rate": cache_stats["hit_rate"]
                })
        
        return recommendations


# Global optimized service instance
_optimized_service: Optional[OptimizedDetectionService] = None


def get_optimized_detection_service(
    config: Optional[OptimizationConfig] = None
) -> OptimizedDetectionService:
    """Get the global optimized detection service instance."""
    global _optimized_service
    
    if _optimized_service is None or (config is not None):
        _optimized_service = OptimizedDetectionService(config)
    
    return _optimized_service


def initialize_optimized_detection_service(
    config: Optional[OptimizationConfig] = None
) -> OptimizedDetectionService:
    """Initialize the global optimized detection service."""
    global _optimized_service
    _optimized_service = OptimizedDetectionService(config)
    return _optimized_service