"""Integration of advanced caching with domain services."""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import asyncio

from ...domain.entities.dataset import Dataset
from ...domain.entities.anomaly_result import AnomalyResult
from .cache_config import get_domain_cache_managers, get_cache_config
from .advanced_cache_strategies import cache_detection_result, cache_model


class CachedDetectionService:
    """Detection service with intelligent caching integration."""
    
    def __init__(self, base_detection_service=None):
        self.base_service = base_detection_service
        self.cache_managers = get_domain_cache_managers()
        self.config = get_cache_config()
    
    def _generate_data_hash(self, data: np.ndarray) -> str:
        """Generate consistent hash for data arrays."""
        # Use shape, dtype, and sample of data for hash
        data_info = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data))
        }
        
        # Add sample values for uniqueness
        if data.size > 0:
            flat_data = data.flatten()
            sample_indices = np.linspace(0, len(flat_data)-1, min(10, len(flat_data)), dtype=int)
            data_info['sample'] = [float(flat_data[i]) for i in sample_indices]
        
        data_str = json.dumps(data_info, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _generate_detection_key(self, data_hash: str, algorithm: str, parameters: Dict) -> str:
        """Generate cache key for detection results."""
        key_data = {
            'data_hash': data_hash,
            'algorithm': algorithm,
            'parameters': sorted(parameters.items()) if parameters else []
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    async def detect_anomalies_cached(
        self,
        data: np.ndarray,
        algorithm: str = "isolation_forest",
        parameters: Optional[Dict[str, Any]] = None
    ) -> AnomalyResult:
        """Detect anomalies with caching support."""
        
        if parameters is None:
            parameters = {}
        
        # Generate cache keys
        data_hash = self._generate_data_hash(data)
        detection_key = self._generate_detection_key(data_hash, algorithm, parameters)
        cache_key = f"detection:{detection_key}"
        
        # Try to get from cache
        cached_result = await self.cache_managers.detection_cache_manager.get(cache_key)
        if cached_result is not None:
            print(f"🎯 Cache HIT for detection {detection_key[:8]}...")
            return cached_result
        
        print(f"⚡ Cache MISS for detection {detection_key[:8]}... Computing...")
        
        # Simulate detection computation (replace with actual service call)
        start_time = time.time()
        result = await self._perform_detection(data, algorithm, parameters)
        computation_time = time.time() - start_time
        
        # Add cache metadata
        result.metadata['cache_key'] = cache_key
        result.metadata['computation_time_seconds'] = computation_time
        result.metadata['cached_at'] = datetime.utcnow().isoformat()
        
        # Cache the result
        await self.cache_managers.detection_cache_manager.set(
            cache_key, 
            result, 
            ttl_seconds=self.config.detection_result_ttl
        )
        
        return result
    
    async def _perform_detection(
        self, 
        data: np.ndarray, 
        algorithm: str, 
        parameters: Dict[str, Any]
    ) -> AnomalyResult:
        """Perform actual anomaly detection computation."""
        
        # Simulate computation delay
        await asyncio.sleep(0.1)
        
        # Simple mock detection for demo
        n_samples = len(data)
        contamination = parameters.get('contamination', 0.1)
        n_anomalies = max(1, int(n_samples * contamination))
        
        # Generate mock results
        predictions = np.ones(n_samples)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        predictions[anomaly_indices] = -1
        
        anomaly_scores = np.random.random(n_samples)
        anomaly_scores[anomaly_indices] *= 2  # Higher scores for anomalies
        
        return AnomalyResult(
            predictions=predictions.tolist(),
            anomaly_scores=anomaly_scores.tolist(),
            anomaly_indices=anomaly_indices.tolist(),
            anomaly_count=n_anomalies,
            algorithm=algorithm,
            parameters=parameters,
            success=True,
            message="Detection completed successfully",
            metadata={
                'total_samples': n_samples,
                'contamination_rate': contamination
            }
        )


class CachedModelService:
    """Model training service with caching for trained models."""
    
    def __init__(self):
        self.cache_managers = get_domain_cache_managers()
        self.config = get_cache_config()
    
    def _generate_model_key(self, algorithm: str, parameters: Dict, data_signature: str) -> str:
        """Generate cache key for trained models."""
        key_data = {
            'algorithm': algorithm,
            'parameters': sorted(parameters.items()) if parameters else [],
            'data_signature': data_signature
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    async def train_model_cached(
        self,
        data: np.ndarray,
        algorithm: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train model with caching support."""
        
        if parameters is None:
            parameters = {}
        
        # Generate model signature
        data_signature = self._generate_data_signature(data)
        model_key = self._generate_model_key(algorithm, parameters, data_signature)
        cache_key = f"trained_model:{model_key}"
        
        # Try to get from cache
        cached_model = await self.cache_managers.model_cache_manager.get(cache_key)
        if cached_model is not None:
            print(f"🎯 Cache HIT for model {model_key[:8]}...")
            return cached_model
        
        print(f"⚡ Cache MISS for model {model_key[:8]}... Training...")
        
        # Train model
        start_time = time.time()
        model_info = await self._train_model(data, algorithm, parameters)
        training_time = time.time() - start_time
        
        # Add cache metadata
        model_info['cache_metadata'] = {
            'model_key': model_key,
            'training_time_seconds': training_time,
            'cached_at': datetime.utcnow().isoformat(),
            'algorithm': algorithm,
            'parameters': parameters
        }
        
        # Cache the model
        await self.cache_managers.model_cache_manager.set(
            cache_key,
            model_info,
            ttl_seconds=self.config.model_cache_ttl
        )
        
        return model_info
    
    def _generate_data_signature(self, data: np.ndarray) -> str:
        """Generate signature for training data."""
        signature_data = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'checksum': hashlib.md5(data.tobytes()).hexdigest()[:8]
        }
        return hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()[:16]
    
    async def _train_model(
        self, 
        data: np.ndarray, 
        algorithm: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform actual model training."""
        
        # Simulate training delay
        await asyncio.sleep(0.2)
        
        # Mock model training results
        return {
            'model_id': f"{algorithm}_{int(time.time())}",
            'algorithm': algorithm,
            'parameters': parameters,
            'training_samples': len(data),
            'features': data.shape[1] if len(data.shape) > 1 else 1,
            'training_metrics': {
                'accuracy': 0.85 + np.random.random() * 0.1,
                'precision': 0.80 + np.random.random() * 0.15,
                'recall': 0.75 + np.random.random() * 0.2
            },
            'model_size_bytes': len(data) * 8  # Mock model size
        }


class CachedDataProcessingService:
    """Data processing service with preprocessing result caching."""
    
    def __init__(self):
        self.cache_managers = get_domain_cache_managers()
        self.config = get_cache_config()
    
    def _generate_preprocessing_key(self, data_hash: str, operations: List[str]) -> str:
        """Generate cache key for preprocessing results."""
        key_data = {
            'data_hash': data_hash,
            'operations': sorted(operations)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    async def preprocess_data_cached(
        self,
        data: np.ndarray,
        operations: List[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess data with caching support."""
        
        if operations is None:
            operations = ['normalize', 'remove_outliers']
        
        # Generate cache keys
        data_hash = self._generate_data_hash(data)
        preprocessing_key = self._generate_preprocessing_key(data_hash, operations)
        cache_key = f"preprocessed:{preprocessing_key}"
        
        # Try to get from cache
        cached_result = await self.cache_managers.data_cache_manager.get(cache_key)
        if cached_result is not None:
            print(f"🎯 Cache HIT for preprocessing {preprocessing_key[:8]}...")
            return cached_result['data'], cached_result['metadata']
        
        print(f"⚡ Cache MISS for preprocessing {preprocessing_key[:8]}... Processing...")
        
        # Perform preprocessing
        start_time = time.time()
        processed_data, metadata = await self._preprocess_data(data, operations)
        processing_time = time.time() - start_time
        
        # Prepare cache entry
        cache_entry = {
            'data': processed_data,
            'metadata': {
                **metadata,
                'cache_key': cache_key,
                'processing_time_seconds': processing_time,
                'cached_at': datetime.utcnow().isoformat()
            }
        }
        
        # Cache the result
        await self.cache_managers.data_cache_manager.set(
            cache_key,
            cache_entry,
            ttl_seconds=self.config.data_preprocessing_ttl
        )
        
        return processed_data, cache_entry['metadata']
    
    def _generate_data_hash(self, data: np.ndarray) -> str:
        """Generate hash for data array."""
        return hashlib.md5(data.tobytes()).hexdigest()[:16]
    
    async def _preprocess_data(
        self, 
        data: np.ndarray, 
        operations: List[str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform actual data preprocessing."""
        
        # Simulate processing delay
        await asyncio.sleep(0.05)
        
        processed_data = data.copy()
        metadata = {
            'original_shape': data.shape,
            'operations_applied': operations,
            'statistics': {
                'original_mean': float(np.mean(data)),
                'original_std': float(np.std(data))
            }
        }
        
        # Apply mock operations
        if 'normalize' in operations:
            processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
            metadata['statistics']['normalized'] = True
        
        if 'remove_outliers' in operations:
            # Simple outlier removal (values beyond 3 std devs)
            std_dev = np.std(processed_data)
            mean_val = np.mean(processed_data)
            outlier_mask = np.abs(processed_data - mean_val) < (3 * std_dev)
            
            if len(processed_data.shape) > 1:
                outlier_mask = np.all(outlier_mask, axis=1)
                processed_data = processed_data[outlier_mask]
            else:
                processed_data = processed_data[outlier_mask]
            
            metadata['outliers_removed'] = int(np.sum(~outlier_mask))
            metadata['final_shape'] = processed_data.shape
        
        metadata['statistics']['final_mean'] = float(np.mean(processed_data))
        metadata['statistics']['final_std'] = float(np.std(processed_data))
        
        return processed_data, metadata


class CacheMetricsCollector:
    """Collector for cache performance metrics."""
    
    def __init__(self):
        self.cache_managers = get_domain_cache_managers()
    
    async def collect_cache_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive cache metrics."""
        
        # Get statistics from all domain cache managers
        domain_stats = await self.cache_managers.get_combined_stats()
        
        # Calculate overall metrics
        total_hits = sum(stats.get('hits', 0) for stats in domain_stats.values() if isinstance(stats, dict))
        total_misses = sum(stats.get('misses', 0) for stats in domain_stats.values() if isinstance(stats, dict))
        total_requests = total_hits + total_misses
        
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_metrics': {
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_requests': total_requests,
                'overall_hit_rate_percent': round(overall_hit_rate, 2)
            },
            'domain_specific_metrics': domain_stats,
            'cache_efficiency': {
                'best_performing_domain': self._find_best_performing_domain(domain_stats),
                'worst_performing_domain': self._find_worst_performing_domain(domain_stats)
            }
        }
        
        return metrics
    
    def _find_best_performing_domain(self, domain_stats: Dict[str, Any]) -> str:
        """Find domain with highest cache hit rate."""
        best_domain = None
        best_hit_rate = 0
        
        for domain, stats in domain_stats.items():
            if isinstance(stats, dict) and 'hit_rate_percent' in stats:
                hit_rate = stats['hit_rate_percent']
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_domain = domain
        
        return best_domain or "none"
    
    def _find_worst_performing_domain(self, domain_stats: Dict[str, Any]) -> str:
        """Find domain with lowest cache hit rate."""
        worst_domain = None
        worst_hit_rate = 100
        
        for domain, stats in domain_stats.items():
            if isinstance(stats, dict) and 'hit_rate_percent' in stats:
                hit_rate = stats['hit_rate_percent']
                if hit_rate < worst_hit_rate:
                    worst_hit_rate = hit_rate
                    worst_domain = domain
        
        return worst_domain or "none"
    
    async def log_cache_performance(self) -> None:
        """Log cache performance metrics."""
        metrics = await self.collect_cache_metrics()
        
        print("\n📊 Cache Performance Report")
        print("=" * 40)
        print(f"Overall Hit Rate: {metrics['overall_metrics']['overall_hit_rate_percent']:.1f}%")
        print(f"Total Requests: {metrics['overall_metrics']['total_requests']}")
        print(f"Total Hits: {metrics['overall_metrics']['total_hits']}")
        print(f"Total Misses: {metrics['overall_metrics']['total_misses']}")
        
        print(f"\nBest Domain: {metrics['cache_efficiency']['best_performing_domain']}")
        print(f"Worst Domain: {metrics['cache_efficiency']['worst_performing_domain']}")
        
        print("\n📋 Domain-Specific Metrics:")
        for domain, stats in metrics['domain_specific_metrics'].items():
            if isinstance(stats, dict):
                hit_rate = stats.get('hit_rate_percent', 0)
                requests = stats.get('total_requests', 0)
                print(f"  {domain}: {hit_rate:.1f}% hit rate ({requests} requests)")


# Integration helper functions
async def benchmark_cache_performance(n_operations: int = 100) -> Dict[str, Any]:
    """Benchmark cache performance across different operations."""
    
    print(f"🚀 Starting cache performance benchmark with {n_operations} operations...")
    
    # Initialize services
    detection_service = CachedDetectionService()
    model_service = CachedModelService()
    data_service = CachedDataProcessingService()
    metrics_collector = CacheMetricsCollector()
    
    # Generate test data
    test_data = np.random.normal(0, 1, (1000, 5))
    
    benchmark_results = {
        'detection_times': [],
        'model_training_times': [],
        'preprocessing_times': []
    }
    
    # Benchmark detection caching
    print("⚡ Benchmarking detection caching...")
    for i in range(n_operations):
        start_time = time.time()
        
        # Add some variation to data to test cache hits/misses
        data_variant = test_data + np.random.normal(0, 0.01, test_data.shape) if i % 5 == 0 else test_data
        
        await detection_service.detect_anomalies_cached(
            data_variant,
            algorithm="isolation_forest",
            parameters={'contamination': 0.1}
        )
        
        benchmark_results['detection_times'].append(time.time() - start_time)
    
    # Benchmark model training caching
    print("⚡ Benchmarking model training caching...")
    for i in range(min(20, n_operations)):  # Fewer model training operations
        start_time = time.time()
        
        await model_service.train_model_cached(
            test_data,
            algorithm="isolation_forest",
            parameters={'n_estimators': 100}
        )
        
        benchmark_results['model_training_times'].append(time.time() - start_time)
    
    # Benchmark data preprocessing caching
    print("⚡ Benchmarking data preprocessing caching...")
    for i in range(n_operations):
        start_time = time.time()
        
        data_variant = test_data + np.random.normal(0, 0.001, test_data.shape) if i % 10 == 0 else test_data
        
        await data_service.preprocess_data_cached(
            data_variant,
            operations=['normalize', 'remove_outliers']
        )
        
        benchmark_results['preprocessing_times'].append(time.time() - start_time)
    
    # Collect final metrics
    final_metrics = await metrics_collector.collect_cache_metrics()
    
    # Calculate benchmark statistics
    benchmark_stats = {}
    for operation, times in benchmark_results.items():
        if times:
            benchmark_stats[operation] = {
                'avg_time_ms': np.mean(times) * 1000,
                'min_time_ms': np.min(times) * 1000,
                'max_time_ms': np.max(times) * 1000,
                'total_operations': len(times)
            }
    
    return {
        'benchmark_stats': benchmark_stats,
        'cache_metrics': final_metrics,
        'timestamp': datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    # Demo of cache integration
    async def demo():
        print("🚀 Cache Integration Demo")
        print("=" * 50)
        
        # Run benchmark
        results = await benchmark_cache_performance(50)
        
        print("\n📊 Benchmark Results:")
        for operation, stats in results['benchmark_stats'].items():
            print(f"\n{operation.replace('_', ' ').title()}:")
            print(f"  Average time: {stats['avg_time_ms']:.2f}ms")
            print(f"  Min time: {stats['min_time_ms']:.2f}ms")
            print(f"  Max time: {stats['max_time_ms']:.2f}ms")
            print(f"  Operations: {stats['total_operations']}")
        
        # Show cache metrics
        cache_metrics = results['cache_metrics']
        print(f"\n🎯 Overall Cache Performance:")
        print(f"  Hit Rate: {cache_metrics['overall_metrics']['overall_hit_rate_percent']:.1f}%")
        print(f"  Total Requests: {cache_metrics['overall_metrics']['total_requests']}")
        
        print("\n✅ Cache integration demo completed!")
    
    asyncio.run(demo())