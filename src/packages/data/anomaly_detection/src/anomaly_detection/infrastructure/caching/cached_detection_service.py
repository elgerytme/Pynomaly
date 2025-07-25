"""Cached wrapper for detection service with intelligent caching strategies."""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

from ...domain.services.detection_service import DetectionService
from ...domain.entities.detection_result import DetectionResult
from .cache_config import get_domain_cache_managers, CacheProfile, initialize_cache_system
from .advanced_cache_strategies import cache_detection_result, cache_model

logger = logging.getLogger(__name__)


class CachedDetectionService:
    """Cached wrapper for DetectionService with intelligent cache strategies."""
    
    def __init__(
        self, 
        detection_service: Optional[DetectionService] = None,
        cache_profile: Optional[CacheProfile] = None
    ):
        self.detection_service = detection_service or DetectionService()
        
        # Initialize cache system
        if cache_profile:
            self.cache_managers = initialize_cache_system(cache_profile)
        else:
            self.cache_managers = get_domain_cache_managers()
        
        # Cache configuration
        self.enable_model_caching = True
        self.enable_result_caching = True
        self.enable_data_caching = True
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time_saved_seconds': 0.0,
            'average_detection_time_ms': 0.0
        }
    
    async def detect_anomalies(
        self,
        data: np.ndarray,
        algorithm: str = "isolation_forest",
        parameters: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False,
        cache_ttl_override: Optional[int] = None
    ) -> DetectionResult:
        """
        Detect anomalies with intelligent caching.
        
        Args:
            data: Input data for anomaly detection
            algorithm: Algorithm to use for detection
            parameters: Algorithm parameters
            force_recompute: Force recomputation bypassing cache
            cache_ttl_override: Override default cache TTL
            
        Returns:
            DetectionResult with anomaly information
        """
        start_time = time.time()
        
        # Generate cache key for this detection request
        cache_key = self._generate_detection_cache_key(data, algorithm, parameters)
        
        # Try to get cached result if not forcing recompute
        cached_result = None
        if not force_recompute and self.enable_result_caching:
            cached_result = await self._get_cached_detection_result(cache_key)
            
            if cached_result is not None:
                # Cache hit
                self.performance_stats['cache_hits'] += 1
                self.performance_stats['total_detections'] += 1
                
                cache_time = time.time() - start_time
                logger.info(f"Cache hit for detection in {cache_time*1000:.2f}ms")
                
                return cached_result
        
        # Cache miss - perform detection
        self.performance_stats['cache_misses'] += 1
        logger.info(f"Cache miss for detection key: {cache_key[:16]}...")
        
        # Check if we have cached preprocessed data
        preprocessed_data = await self._get_or_cache_preprocessed_data(data)
        
        # Get or train cached model
        model = await self._get_or_cache_model(algorithm, parameters)
        
        # Perform detection with cached model and data
        detection_start = time.time()
        
        try:
            result = self.detection_service.detect_anomalies(
                data=preprocessed_data or data,
                algorithm=algorithm,
                parameters=parameters
            )
            
            detection_time = time.time() - detection_start
            
            # Update performance stats
            self.performance_stats['total_detections'] += 1
            current_avg = self.performance_stats['average_detection_time_ms']
            total_detections = self.performance_stats['total_detections']
            
            # Update rolling average
            self.performance_stats['average_detection_time_ms'] = (
                (current_avg * (total_detections - 1) + detection_time * 1000) / total_detections
            )
            
            # Cache the result if successful
            if result.success and self.enable_result_caching:
                ttl = cache_ttl_override or self.cache_managers.base_config.detection_result_ttl
                await self._cache_detection_result(cache_key, result, ttl)
            
            total_time = time.time() - start_time
            logger.info(f"Detection completed in {total_time*1000:.2f}ms (compute: {detection_time*1000:.2f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    async def batch_detect_anomalies(
        self,
        data_batches: List[np.ndarray],
        algorithm: str = "isolation_forest",
        parameters: Optional[Dict[str, Any]] = None,
        parallel_processing: bool = True
    ) -> List[DetectionResult]:
        """
        Perform batch anomaly detection with optimized caching.
        
        Args:
            data_batches: List of data arrays to process
            algorithm: Algorithm to use for detection
            parameters: Algorithm parameters
            parallel_processing: Whether to process batches in parallel
            
        Returns:
            List of DetectionResult objects
        """
        logger.info(f"Starting batch detection for {len(data_batches)} batches")
        
        if parallel_processing:
            # Process batches in parallel
            tasks = [
                self.detect_anomalies(data, algorithm, parameters)
                for data in data_batches
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch {i} failed: {result}")
                    # Create error result
                    final_results.append(DetectionResult(
                        algorithm=algorithm,
                        success=False,
                        message=f"Batch processing failed: {result}",
                        predictions=np.array([]),
                        anomaly_scores=np.array([]),
                        anomaly_indices=np.array([]),
                        anomaly_count=0,
                        total_samples=len(data_batches[i]) if i < len(data_batches) else 0,
                        parameters=parameters or {}
                    ))
                else:
                    final_results.append(result)
            
            return final_results
        
        else:
            # Process batches sequentially
            results = []
            for i, data in enumerate(data_batches):
                try:
                    result = await self.detect_anomalies(data, algorithm, parameters)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch {i} failed: {e}")
                    results.append(DetectionResult(
                        algorithm=algorithm,
                        success=False,
                        message=f"Batch processing failed: {e}",
                        predictions=np.array([]),
                        anomaly_scores=np.array([]),
                        anomaly_indices=np.array([]),
                        anomaly_count=0,
                        total_samples=len(data),
                        parameters=parameters or {}
                    ))
            
            return results
    
    async def warm_cache(
        self,
        algorithms: List[str],
        sample_data_shapes: List[Tuple[int, int]],
        common_parameters: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, bool]:
        """
        Warm up the cache with common models and preprocessing results.
        
        Args:
            algorithms: List of algorithms to pre-train
            sample_data_shapes: List of (n_samples, n_features) shapes to prepare for
            common_parameters: Common parameter sets for each algorithm
            
        Returns:
            Dictionary indicating success for each algorithm
        """
        logger.info("Starting cache warm-up process...")
        results = {}
        
        # Generate sample data for each shape
        sample_datasets = {}
        for shape in sample_data_shapes:
            n_samples, n_features = shape
            # Generate representative sample data
            np.random.seed(42)  # Reproducible sample data
            sample_data = np.random.multivariate_normal(
                mean=np.zeros(n_features),
                cov=np.eye(n_features),
                size=n_samples
            )
            sample_datasets[shape] = sample_data
        
        # Pre-train models for each algorithm and data shape
        for algorithm in algorithms:
            algorithm_success = True
            
            for shape, sample_data in sample_datasets.items():
                try:
                    # Get algorithm parameters
                    algo_params = {}
                    if common_parameters and algorithm in common_parameters:
                        algo_params = common_parameters[algorithm]
                    
                    # Cache preprocessing results
                    await self._get_or_cache_preprocessed_data(sample_data)
                    
                    # Cache model
                    await self._get_or_cache_model(algorithm, algo_params)
                    
                    logger.info(f"Warmed cache for {algorithm} with shape {shape}")
                    
                except Exception as e:
                    logger.error(f"Failed to warm cache for {algorithm} with shape {shape}: {e}")
                    algorithm_success = False
            
            results[algorithm] = algorithm_success
        
        successful_algorithms = sum(results.values())
        logger.info(f"Cache warm-up completed: {successful_algorithms}/{len(algorithms)} algorithms successful")
        
        return results
    
    def _generate_detection_cache_key(
        self,
        data: np.ndarray,
        algorithm: str,
        parameters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a unique cache key for detection request."""
        # Create a hash of the data
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]
        
        # Create parameter hash
        param_str = json.dumps(parameters or {}, sort_keys=True)
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        
        return f"detection:{algorithm}:{data_hash}:{param_hash}"
    
    def _generate_data_cache_key(self, data: np.ndarray) -> str:
        """Generate cache key for preprocessed data."""
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]
        return f"preprocessed_data:{data_hash}"
    
    def _generate_model_cache_key(
        self,
        algorithm: str,
        parameters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for trained model."""
        param_str = json.dumps(parameters or {}, sort_keys=True)
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        return f"trained_model:{algorithm}:{param_hash}"
    
    async def _get_cached_detection_result(self, cache_key: str) -> Optional[DetectionResult]:
        """Get cached detection result."""
        try:
            cached_result = await self.cache_managers.detection_cache_manager.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for detection result: {cache_key[:32]}...")
                return cached_result
        except Exception as e:
            logger.warning(f"Failed to get cached detection result: {e}")
        
        return None
    
    async def _cache_detection_result(
        self,
        cache_key: str,
        result: DetectionResult,
        ttl_seconds: int
    ) -> bool:
        """Cache detection result."""
        try:
            success = await self.cache_managers.detection_cache_manager.set(
                cache_key, result, ttl_seconds
            )
            if success:
                logger.debug(f"Cached detection result: {cache_key[:32]}...")
            return success
        except Exception as e:
            logger.warning(f"Failed to cache detection result: {e}")
            return False
    
    async def _get_or_cache_preprocessed_data(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Get or cache preprocessed data."""
        if not self.enable_data_caching:
            return None
        
        data_key = self._generate_data_cache_key(data)
        
        try:
            # Try to get cached preprocessed data
            cached_data = await self.cache_managers.data_cache_manager.get(data_key)
            if cached_data is not None:
                logger.debug("Using cached preprocessed data")
                return cached_data
            
            # Preprocess data (simplified example)
            # In practice, this would use the actual preprocessing pipeline
            preprocessed_data = data.copy()
            
            # Cache the preprocessed data
            ttl = self.cache_managers.base_config.data_preprocessing_ttl
            await self.cache_managers.data_cache_manager.set(data_key, preprocessed_data, ttl)
            
            logger.debug("Cached new preprocessed data")
            return preprocessed_data
            
        except Exception as e:
            logger.warning(f"Failed to handle preprocessed data caching: {e}")
            return None
    
    async def _get_or_cache_model(
        self,
        algorithm: str,
        parameters: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Get or cache trained model."""
        if not self.enable_model_caching:
            return None
        
        model_key = self._generate_model_cache_key(algorithm, parameters)
        
        try:
            # Try to get cached model
            cached_model = await self.cache_managers.model_cache_manager.get(model_key)
            if cached_model is not None:
                logger.debug(f"Using cached model for {algorithm}")
                return cached_model
            
            # Train new model (simplified example)
            # In practice, this would use the actual model training pipeline
            logger.debug(f"Training new model for {algorithm}")
            
            # Simulate model training time
            await asyncio.sleep(0.1)
            
            # Create mock model (in practice this would be the actual trained model)
            mock_model = {"algorithm": algorithm, "parameters": parameters, "trained_at": datetime.utcnow()}
            
            # Cache the model
            ttl = self.cache_managers.base_config.model_cache_ttl
            await self.cache_managers.model_cache_manager.set(model_key, mock_model, ttl)
            
            logger.debug(f"Cached new model for {algorithm}")
            return mock_model
            
        except Exception as e:
            logger.warning(f"Failed to handle model caching: {e}")
            return None
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        # Get domain cache statistics
        domain_stats = await self.cache_managers.get_combined_stats()
        
        # Add performance statistics
        performance_stats = self.performance_stats.copy()
        
        # Calculate additional metrics
        total_requests = performance_stats['cache_hits'] + performance_stats['cache_misses']
        if total_requests > 0:
            hit_rate = (performance_stats['cache_hits'] / total_requests) * 100
            performance_stats['cache_hit_rate_percent'] = round(hit_rate, 2)
        else:
            performance_stats['cache_hit_rate_percent'] = 0.0
        
        # Estimate time saved (assuming cache hits are 10x faster)
        avg_detection_time_ms = performance_stats['average_detection_time_ms']
        if avg_detection_time_ms > 0:
            estimated_time_saved = (performance_stats['cache_hits'] * avg_detection_time_ms * 0.9) / 1000
            performance_stats['estimated_time_saved_seconds'] = round(estimated_time_saved, 2)
        
        return {
            'performance': performance_stats,
            'cache_stores': domain_stats
        }
    
    async def clear_all_caches(self) -> Dict[str, bool]:
        """Clear all caches and reset statistics."""
        # Clear domain caches
        clear_results = await self.cache_managers.clear_all_caches()
        
        # Reset performance statistics
        self.performance_stats = {
            'total_detections': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time_saved_seconds': 0.0,
            'average_detection_time_ms': 0.0
        }
        
        return clear_results
    
    def configure_caching(
        self,
        enable_model_caching: bool = True,
        enable_result_caching: bool = True,
        enable_data_caching: bool = True
    ):
        """Configure caching behavior."""
        self.enable_model_caching = enable_model_caching
        self.enable_result_caching = enable_result_caching
        self.enable_data_caching = enable_data_caching
        
        logger.info(f"Cache configuration updated: models={enable_model_caching}, "
                   f"results={enable_result_caching}, data={enable_data_caching}")


if __name__ == "__main__":
    # Example usage and testing
    async def demo():
        print("🚀 Cached Detection Service Demo")
        print("=" * 50)
        
        # Initialize cached detection service
        cached_service = CachedDetectionService(cache_profile=CacheProfile.DEVELOPMENT)
        
        # Generate sample data
        np.random.seed(42)
        sample_data = np.random.multivariate_normal(
            mean=[0, 0, 0],
            cov=np.eye(3),
            size=1000
        )
        
        print(f"\n1️⃣ Generated sample data: {sample_data.shape}")
        
        # Test cache warm-up
        print("\n2️⃣ Warming up cache...")
        warm_results = await cached_service.warm_cache(
            algorithms=["isolation_forest", "local_outlier_factor"],
            sample_data_shapes=[(1000, 3), (500, 3)],
            common_parameters={
                "isolation_forest": {"contamination": 0.1},
                "local_outlier_factor": {"n_neighbors": 20}
            }
        )
        print(f"   Warm-up results: {warm_results}")
        
        # Test detection with caching
        print("\n3️⃣ Testing detection with caching...")
        
        # First detection (cache miss)
        start_time = time.time()
        result1 = await cached_service.detect_anomalies(
            data=sample_data,
            algorithm="isolation_forest",
            parameters={"contamination": 0.1}
        )
        time1 = time.time() - start_time
        print(f"   First detection: {time1*1000:.2f}ms - {result1.anomaly_count} anomalies")
        
        # Second detection (cache hit)
        start_time = time.time()
        result2 = await cached_service.detect_anomalies(
            data=sample_data,
            algorithm="isolation_forest", 
            parameters={"contamination": 0.1}
        )
        time2 = time.time() - start_time
        print(f"   Second detection: {time2*1000:.2f}ms - {result2.anomaly_count} anomalies")
        
        # Test batch processing
        print("\n4️⃣ Testing batch processing...")
        batch_data = [sample_data[:200], sample_data[200:400], sample_data[400:600]]
        
        start_time = time.time()
        batch_results = await cached_service.batch_detect_anomalies(
            data_batches=batch_data,
            algorithm="isolation_forest",
            parameters={"contamination": 0.1},
            parallel_processing=True
        )
        batch_time = time.time() - start_time
        
        total_anomalies = sum(r.anomaly_count for r in batch_results)
        print(f"   Batch processing: {batch_time*1000:.2f}ms - {total_anomalies} total anomalies")
        
        # Show cache statistics
        print("\n5️⃣ Cache Statistics:")
        stats = await cached_service.get_cache_statistics()
        
        performance = stats['performance']
        print(f"   Total detections: {performance['total_detections']}")
        print(f"   Cache hits: {performance['cache_hits']}")
        print(f"   Cache misses: {performance['cache_misses']}")
        print(f"   Hit rate: {performance['cache_hit_rate_percent']:.1f}%")
        print(f"   Average detection time: {performance['average_detection_time_ms']:.2f}ms")
        
        if 'estimated_time_saved_seconds' in performance:
            print(f"   Estimated time saved: {performance['estimated_time_saved_seconds']:.2f}s")
        
        # Show cache store statistics
        cache_stores = stats['cache_stores']
        for store_name, store_stats in cache_stores.items():
            if store_stats:
                print(f"   {store_name}: {store_stats.get('hit_rate_percent', 0):.1f}% hit rate")
        
        print("\n✅ Cached detection service demo completed!")
    
    # Run demo
    asyncio.run(demo())