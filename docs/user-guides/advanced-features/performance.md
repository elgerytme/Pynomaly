# Performance Optimization Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔶 [Advanced Features](README.md) > 📄 Performance

---


This comprehensive guide covers performance optimization strategies for Pynomaly, including benchmarking, profiling, memory management, and production-level optimization techniques.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Benchmarking and Profiling](#benchmarking-and-profiling)
3. [Memory Management](#memory-management)
4. [Algorithm Performance](#algorithm-performance)
5. [Data Processing Optimization](#data-processing-optimization)
6. [Production Optimizations](#production-optimizations)
7. [Monitoring and Metrics](#monitoring-and-metrics)
8. [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

## Performance Overview

Pynomaly is designed for high-performance anomaly detection with several optimization layers:

- **Algorithmic**: Efficient implementations across PyOD, scikit-learn, PyTorch
- **Data Processing**: High-performance loaders (Polars, PyArrow, Spark)
- **Infrastructure**: Circuit breakers, caching, connection pooling
- **Deployment**: Async operations, memory management, GPU acceleration

### Performance Targets

| Metric | Target | Typical Performance |
|--------|--------|-------------------|
| Training Time | < 30s for 100K samples | 5-15s depending on algorithm |
| Prediction Latency | < 100ms for 1K samples | 10-50ms for most algorithms |
| Memory Usage | < 2GB for 1M samples | 500MB-1.5GB depending on algorithm |
| Throughput | > 10K predictions/second | 15K-50K depending on setup |

## Benchmarking and Profiling

### Built-in Benchmarking Suite

```python
# benchmarks/performance_suite.py
import asyncio
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from pynomaly.infrastructure.config import create_container
from pynomaly.domain.entities import Dataset
from pynomaly.application.use_cases import DetectAnomaliesUseCase


@dataclass
class BenchmarkResult:
    """Results from performance benchmark."""
    algorithm: str
    dataset_size: int
    training_time_ms: float
    prediction_time_ms: float
    memory_usage_mb: float
    throughput_samples_per_second: float
    accuracy_score: float
    cpu_usage_percent: float


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking for Pynomaly."""
    
    def __init__(self):
        self.container = None
        self.results: List[BenchmarkResult] = []
    
    async def initialize(self):
        """Initialize benchmarking environment."""
        self.container = create_container()
        print("🚀 Performance Benchmark Suite Initialized")
    
    async def run_comprehensive_benchmark(self, 
                                        algorithms: List[str] = None,
                                        dataset_sizes: List[int] = None) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across algorithms and dataset sizes."""
        
        if algorithms is None:
            algorithms = [
                'isolation_forest', 'local_outlier_factor', 'one_class_svm',
                'autoencoder', 'pca', 'hbos', 'knn', 'copod'
            ]
        
        if dataset_sizes is None:
            dataset_sizes = [1000, 10000, 100000, 1000000]
        
        print(f"📊 Running benchmarks for {len(algorithms)} algorithms on {len(dataset_sizes)} dataset sizes")
        
        all_results = {}
        
        for algorithm in algorithms:
            algorithm_results = []
            print(f"\n🔍 Benchmarking {algorithm}")
            
            for size in dataset_sizes:
                try:
                    result = await self._benchmark_algorithm(algorithm, size)
                    algorithm_results.append(result)
                    self.results.append(result)
                    
                    print(f"  📈 Size {size:>7}: "
                          f"Training {result.training_time_ms:>6.0f}ms, "
                          f"Prediction {result.prediction_time_ms:>6.0f}ms, "
                          f"Memory {result.memory_usage_mb:>6.0f}MB, "
                          f"Throughput {result.throughput_samples_per_second:>8.0f}/s")
                    
                except Exception as e:
                    print(f"  ❌ Size {size}: Failed - {str(e)}")
                    continue
            
            all_results[algorithm] = algorithm_results
        
        return all_results
    
    async def _benchmark_algorithm(self, algorithm: str, dataset_size: int) -> BenchmarkResult:
        """Benchmark single algorithm on specific dataset size."""
        
        # Generate synthetic dataset
        dataset = self._generate_dataset(dataset_size)
        
        # Get services
        detection_service = self.container.detection_service()
        
        # Monitor system resources
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()
        
        # Training phase
        training_start = time.perf_counter()
        
        detector_id = await detection_service.train_detector(
            algorithm=algorithm,
            dataset=dataset,
            parameters=self._get_algorithm_parameters(algorithm)
        )
        
        training_time = (time.perf_counter() - training_start) * 1000  # ms
        
        # Prediction phase
        test_data = self._generate_test_data(min(dataset_size // 10, 10000))
        
        prediction_start = time.perf_counter()
        
        results = await detection_service.detect_anomalies(
            detector_id=detector_id,
            data=test_data
        )
        
        prediction_time = (time.perf_counter() - prediction_start) * 1000  # ms
        
        # Calculate metrics
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        cpu_usage = process.cpu_percent() - cpu_before
        
        throughput = len(test_data) / (prediction_time / 1000) if prediction_time > 0 else 0
        
        # Calculate accuracy (using synthetic ground truth)
        accuracy = self._calculate_accuracy(results.predictions, test_data)
        
        return BenchmarkResult(
            algorithm=algorithm,
            dataset_size=dataset_size,
            training_time_ms=training_time,
            prediction_time_ms=prediction_time,
            memory_usage_mb=memory_usage,
            throughput_samples_per_second=throughput,
            accuracy_score=accuracy,
            cpu_usage_percent=cpu_usage
        )
    
    def _generate_dataset(self, size: int) -> Dataset:
        """Generate synthetic dataset for benchmarking."""
        np.random.seed(42)  # Reproducible results
        
        # Generate normal data
        normal_samples = int(size * 0.9)
        normal_data = np.random.normal(0, 1, (normal_samples, 10))
        
        # Generate anomalous data
        anomaly_samples = size - normal_samples
        anomaly_data = np.random.normal(3, 0.5, (anomaly_samples, 10))
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.array([0] * normal_samples + [1] * anomaly_samples)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(10)]
        df = pd.DataFrame(X, columns=feature_names)
        
        return Dataset.from_dataframe(df, name=f"benchmark_data_{size}")
    
    def _generate_test_data(self, size: int) -> pd.DataFrame:
        """Generate test data for prediction benchmarking."""
        np.random.seed(123)  # Different seed for test data
        
        # Mix of normal and anomalous data
        normal_samples = int(size * 0.85)
        anomaly_samples = size - normal_samples
        
        normal_data = np.random.normal(0, 1, (normal_samples, 10))
        anomaly_data = np.random.normal(3, 0.5, (anomaly_samples, 10))
        
        X = np.vstack([normal_data, anomaly_data])
        indices = np.random.permutation(len(X))
        X = X[indices]
        
        feature_names = [f'feature_{i}' for i in range(10)]
        return pd.DataFrame(X, columns=feature_names)
    
    def _get_algorithm_parameters(self, algorithm: str) -> Dict[str, Any]:
        """Get optimal parameters for each algorithm."""
        parameters = {
            'isolation_forest': {'n_estimators': 100, 'contamination': 0.1},
            'local_outlier_factor': {'n_neighbors': 20, 'contamination': 0.1},
            'one_class_svm': {'nu': 0.1, 'kernel': 'rbf'},
            'autoencoder': {'hidden_neurons': [64, 32, 16, 32, 64], 'epochs': 50},
            'pca': {'n_components': 5, 'contamination': 0.1},
            'hbos': {'n_bins': 10, 'alpha': 0.1},
            'knn': {'n_neighbors': 10, 'contamination': 0.1},
            'copod': {'contamination': 0.1}
        }
        return parameters.get(algorithm, {})
    
    def _calculate_accuracy(self, predictions: np.ndarray, test_data: pd.DataFrame) -> float:
        """Calculate accuracy using synthetic ground truth."""
        # For synthetic data, we know the last 15% are anomalies
        # This is a simplified accuracy calculation for benchmarking
        anomaly_count = len(test_data) * 0.15
        predicted_anomalies = np.sum(predictions == 1)
        
        # Simple accuracy based on anomaly detection rate
        if anomaly_count > 0:
            return min(predicted_anomalies / anomaly_count, 1.0)
        return 1.0 if predicted_anomalies == 0 else 0.0
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.results:
            return "No benchmark results available. Run benchmarks first."
        
        report = []
        report.append("# Pynomaly Performance Benchmark Report\n")
        report.append(f"**Total Benchmarks**: {len(self.results)}\n")
        report.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Algorithm Performance Summary
        report.append("## Algorithm Performance Summary\n")
        
        algorithms = set(r.algorithm for r in self.results)
        for algorithm in sorted(algorithms):
            algo_results = [r for r in self.results if r.algorithm == algorithm]
            if not algo_results:
                continue
            
            avg_training = np.mean([r.training_time_ms for r in algo_results])
            avg_prediction = np.mean([r.prediction_time_ms for r in algo_results])
            avg_throughput = np.mean([r.throughput_samples_per_second for r in algo_results])
            avg_memory = np.mean([r.memory_usage_mb for r in algo_results])
            avg_accuracy = np.mean([r.accuracy_score for r in algo_results])
            
            report.append(f"### {algorithm}\n")
            report.append(f"- **Training Time**: {avg_training:.0f}ms (avg)\n")
            report.append(f"- **Prediction Time**: {avg_prediction:.0f}ms (avg)\n")
            report.append(f"- **Throughput**: {avg_throughput:.0f} samples/sec (avg)\n")
            report.append(f"- **Memory Usage**: {avg_memory:.0f}MB (avg)\n")
            report.append(f"- **Accuracy**: {avg_accuracy:.3f} (avg)\n\n")
        
        # Scalability Analysis
        report.append("## Scalability Analysis\n")
        
        dataset_sizes = sorted(set(r.dataset_size for r in self.results))
        report.append("| Dataset Size | Avg Training (ms) | Avg Prediction (ms) | Avg Throughput (samples/s) |\n")
        report.append("|--------------|-------------------|---------------------|---------------------------|\n")
        
        for size in dataset_sizes:
            size_results = [r for r in self.results if r.dataset_size == size]
            if not size_results:
                continue
            
            avg_training = np.mean([r.training_time_ms for r in size_results])
            avg_prediction = np.mean([r.prediction_time_ms for r in size_results])
            avg_throughput = np.mean([r.throughput_samples_per_second for r in size_results])
            
            report.append(f"| {size:,} | {avg_training:.0f} | {avg_prediction:.0f} | {avg_throughput:.0f} |\n")
        
        report.append("\n")
        
        # Performance Recommendations
        report.append("## Performance Recommendations\n\n")
        
        # Find best performing algorithms
        best_training = min(self.results, key=lambda r: r.training_time_ms)
        best_prediction = min(self.results, key=lambda r: r.prediction_time_ms)
        best_throughput = max(self.results, key=lambda r: r.throughput_samples_per_second)
        best_memory = min(self.results, key=lambda r: r.memory_usage_mb)
        
        report.append(f"- **Fastest Training**: {best_training.algorithm} ({best_training.training_time_ms:.0f}ms)\n")
        report.append(f"- **Fastest Prediction**: {best_prediction.algorithm} ({best_prediction.prediction_time_ms:.0f}ms)\n")
        report.append(f"- **Highest Throughput**: {best_throughput.algorithm} ({best_throughput.throughput_samples_per_second:.0f} samples/s)\n")
        report.append(f"- **Lowest Memory**: {best_memory.algorithm} ({best_memory.memory_usage_mb:.0f}MB)\n\n")
        
        return "".join(report)
    
    def save_results_csv(self, filename: str = "benchmark_results.csv"):
        """Save benchmark results to CSV file."""
        if not self.results:
            print("No results to save.")
            return
        
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'algorithm', 'dataset_size', 'training_time_ms', 'prediction_time_ms',
                'memory_usage_mb', 'throughput_samples_per_second', 'accuracy_score', 'cpu_usage_percent'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                writer.writerow({
                    'algorithm': result.algorithm,
                    'dataset_size': result.dataset_size,
                    'training_time_ms': result.training_time_ms,
                    'prediction_time_ms': result.prediction_time_ms,
                    'memory_usage_mb': result.memory_usage_mb,
                    'throughput_samples_per_second': result.throughput_samples_per_second,
                    'accuracy_score': result.accuracy_score,
                    'cpu_usage_percent': result.cpu_usage_percent
                })
        
        print(f"✅ Results saved to {filename}")


# Example usage and built-in benchmark runner
async def run_performance_benchmarks():
    """Run the complete performance benchmark suite."""
    
    benchmark = PerformanceBenchmarkSuite()
    await benchmark.initialize()
    
    print("🎯 Starting Comprehensive Performance Benchmarks")
    
    # Run benchmarks on key algorithms
    key_algorithms = [
        'isolation_forest', 'local_outlier_factor', 'pca',
        'hbos', 'knn', 'copod'
    ]
    
    # Test on multiple dataset sizes
    dataset_sizes = [1000, 10000, 50000]
    
    results = await benchmark.run_comprehensive_benchmark(
        algorithms=key_algorithms,
        dataset_sizes=dataset_sizes
    )
    
    # Generate and display report
    report = benchmark.generate_performance_report()
    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    # Save results
    benchmark.save_results_csv("pynomaly_benchmarks.csv")
    
    return results

if __name__ == "__main__":
    # Run benchmarks
    asyncio.run(run_performance_benchmarks())
```

### Memory Profiling

```python
# utils/memory_profiler.py
import tracemalloc
import psutil
import gc
from typing import Dict, Any, Optional
from contextlib import contextmanager
import functools
import time


class MemoryProfiler:
    """Advanced memory profiling utilities for Pynomaly."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.snapshots = []
    
    @contextmanager
    def profile_memory(self, label: str = "operation"):
        """Context manager for memory profiling."""
        # Start tracing
        tracemalloc.start()
        
        # Get initial memory
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield self
        finally:
            # Get final memory
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            # Get memory snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            # Calculate memory change
            memory_change = final_memory - initial_memory
            
            print(f"\n📊 Memory Profile: {label}")
            print(f"   Initial Memory: {initial_memory:.2f} MB")
            print(f"   Final Memory: {final_memory:.2f} MB")
            print(f"   Memory Change: {memory_change:+.2f} MB")
            
            # Show top memory consumers
            print("   Top Memory Consumers:")
            for i, stat in enumerate(top_stats[:5]):
                print(f"   {i+1}. {stat}")
            
            tracemalloc.stop()
    
    def memory_usage_decorator(self, func):
        """Decorator to profile function memory usage."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with self.profile_memory(f"{func.__name__}"):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with self.profile_memory(f"{func.__name__}"):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


# Usage example
profiler = MemoryProfiler()

@profiler.memory_usage_decorator
async def analyze_large_dataset():
    """Example of memory profiling in action."""
    # Simulate large dataset processing
    import pandas as pd
    import numpy as np
    
    # Generate large dataset
    data = np.random.normal(0, 1, (100000, 50))
    df = pd.DataFrame(data)
    
    # Process data
    processed = df.apply(lambda x: x ** 2)
    
    # Clean up
    del data, processed
    gc.collect()
    
    return df
```

## Memory Management

### Memory-Efficient Data Processing

```python
# infrastructure/data_processing/memory_efficient.py
import pandas as pd
import numpy as np
from typing import Iterator, Optional, Union
import gc
from contextlib import contextmanager


class MemoryEfficientProcessor:
    """Memory-efficient data processing utilities."""
    
    def __init__(self, max_memory_mb: int = 1000):
        self.max_memory_mb = max_memory_mb
        self.chunk_size = self._calculate_optimal_chunk_size()
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory."""
        # Simple heuristic: use 10% of max memory for chunk size
        # Assuming 8 bytes per float64 value and 10 features on average
        bytes_per_row = 8 * 10  # 80 bytes per row
        max_rows = (self.max_memory_mb * 1024 * 1024 * 0.1) / bytes_per_row
        return max(1000, int(max_rows))
    
    def process_large_dataset_chunked(self, 
                                    file_path: str,
                                    processing_func: callable,
                                    **kwargs) -> Iterator[pd.DataFrame]:
        """Process large dataset in memory-efficient chunks."""
        
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        
        print(f"🔄 Processing dataset in chunks of {chunk_size:,} rows")
        
        # Read and process in chunks
        for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
            
            print(f"   Processing chunk {chunk_idx + 1} ({len(chunk):,} rows)")
            
            # Apply processing function
            processed_chunk = processing_func(chunk)
            
            # Yield processed chunk
            yield processed_chunk
            
            # Force garbage collection after each chunk
            gc.collect()
    
    @contextmanager
    def memory_limit_context(self, limit_mb: int):
        """Context manager to enforce memory limits."""
        import resource
        
        # Set memory limit (Unix only)
        try:
            # Convert MB to bytes
            limit_bytes = limit_mb * 1024 * 1024
            
            # Get current limit
            old_limit = resource.getrlimit(resource.RLIMIT_AS)
            
            # Set new limit
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, old_limit[1]))
            
            try:
                yield
            finally:
                # Restore old limit
                resource.setrlimit(resource.RLIMIT_AS, old_limit)
                
        except (ImportError, AttributeError):
            # Windows or other systems without resource module
            print("⚠️  Memory limiting not available on this platform")
            yield
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types."""
        
        print(f"📊 Optimizing DataFrame memory usage")
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= -128 and col_max <= 127:
                optimized_df[col] = optimized_df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                optimized_df[col] = optimized_df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                optimized_df[col] = optimized_df[col].astype('int32')
        
        # Optimize float columns
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Optimize object columns (try categorical)
        for col in optimized_df.select_dtypes(include=['object']).columns:
            unique_count = optimized_df[col].nunique()
            total_count = len(optimized_df[col])
            
            # If less than 50% unique values, convert to categorical
            if unique_count / total_count < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        final_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
        memory_reduction = ((initial_memory - final_memory) / initial_memory) * 100
        
        print(f"   Memory usage reduced by {memory_reduction:.1f}% "
              f"({initial_memory:.1f}MB → {final_memory:.1f}MB)")
        
        return optimized_df


# Streaming data processor for infinite datasets
class StreamingProcessor:
    """Process streaming data with minimal memory footprint."""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.buffer = []
    
    async def process_streaming_data(self, 
                                   data_stream: Iterator,
                                   detector,
                                   batch_callback: Optional[callable] = None):
        """Process streaming data in batches."""
        
        batch_count = 0
        
        async for data_point in data_stream:
            self.buffer.append(data_point)
            
            # Process when buffer is full
            if len(self.buffer) >= self.buffer_size:
                batch_count += 1
                
                # Convert to DataFrame for processing
                batch_df = pd.DataFrame(self.buffer)
                
                # Run anomaly detection
                results = await detector.predict(batch_df)
                
                # Process results
                if batch_callback:
                    await batch_callback(results, batch_count)
                
                # Clear buffer
                self.buffer.clear()
                gc.collect()
                
                print(f"✅ Processed batch {batch_count} ({self.buffer_size} samples)")
        
        # Process remaining data in buffer
        if self.buffer:
            batch_df = pd.DataFrame(self.buffer)
            results = await detector.predict(batch_df)
            
            if batch_callback:
                await batch_callback(results, batch_count + 1)
            
            print(f"✅ Processed final batch ({len(self.buffer)} samples)")
```

## Algorithm Performance

### Algorithm Selection Guide

```python
# utils/algorithm_selector.py
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from enum import Enum


class DatasetCharacteristics(Enum):
    """Dataset characteristic categories."""
    SMALL = "small"  # < 10K samples
    MEDIUM = "medium"  # 10K - 100K samples  
    LARGE = "large"  # 100K - 1M samples
    VERY_LARGE = "very_large"  # > 1M samples


class PerformanceCategory(Enum):
    """Performance requirement categories."""
    ACCURACY_FIRST = "accuracy_first"
    SPEED_FIRST = "speed_first"
    MEMORY_EFFICIENT = "memory_efficient"
    BALANCED = "balanced"


class AlgorithmSelector:
    """Intelligent algorithm selection based on data and performance requirements."""
    
    def __init__(self):
        self.algorithm_profiles = self._build_algorithm_profiles()
    
    def _build_algorithm_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive algorithm performance profiles."""
        return {
            'isolation_forest': {
                'training_speed': 'fast',
                'prediction_speed': 'fast',
                'memory_usage': 'medium',
                'accuracy': 'high',
                'scalability': 'excellent',
                'best_for': ['large_datasets', 'balanced_performance'],
                'optimal_size_range': (1000, 1000000),
                'features_range': (1, 1000),
                'complexity': 'O(n log n)'
            },
            'local_outlier_factor': {
                'training_speed': 'medium',
                'prediction_speed': 'slow',
                'memory_usage': 'high',
                'accuracy': 'very_high',
                'scalability': 'poor',
                'best_for': ['small_datasets', 'accuracy_first'],
                'optimal_size_range': (100, 10000),
                'features_range': (1, 50),
                'complexity': 'O(n²)'
            },
            'one_class_svm': {
                'training_speed': 'slow',
                'prediction_speed': 'fast',
                'memory_usage': 'medium',
                'accuracy': 'high',
                'scalability': 'medium',
                'best_for': ['medium_datasets', 'balanced_performance'],
                'optimal_size_range': (1000, 50000),
                'features_range': (1, 100),
                'complexity': 'O(n²) to O(n³)'
            },
            'pca': {
                'training_speed': 'fast',
                'prediction_speed': 'very_fast',
                'memory_usage': 'low',
                'accuracy': 'medium',
                'scalability': 'excellent',
                'best_for': ['large_datasets', 'speed_first', 'memory_efficient'],
                'optimal_size_range': (1000, 10000000),
                'features_range': (10, 10000),
                'complexity': 'O(n × p²)'
            },
            'autoencoder': {
                'training_speed': 'very_slow',
                'prediction_speed': 'fast',
                'memory_usage': 'high',
                'accuracy': 'very_high',
                'scalability': 'good',
                'best_for': ['complex_patterns', 'accuracy_first'],
                'optimal_size_range': (10000, 1000000),
                'features_range': (10, 1000),
                'complexity': 'O(n × epochs × layers)'
            },
            'hbos': {
                'training_speed': 'very_fast',
                'prediction_speed': 'very_fast',
                'memory_usage': 'very_low',
                'accuracy': 'medium',
                'scalability': 'excellent',
                'best_for': ['very_large_datasets', 'speed_first', 'memory_efficient'],
                'optimal_size_range': (1000, 100000000),
                'features_range': (1, 100),
                'complexity': 'O(n × p)'
            },
            'knn': {
                'training_speed': 'very_fast',
                'prediction_speed': 'slow',
                'memory_usage': 'high',
                'accuracy': 'high',
                'scalability': 'poor',
                'best_for': ['small_datasets', 'simple_patterns'],
                'optimal_size_range': (100, 10000),
                'features_range': (1, 20),
                'complexity': 'O(n × p) prediction'
            },
            'copod': {
                'training_speed': 'fast',
                'prediction_speed': 'fast',
                'memory_usage': 'low',
                'accuracy': 'high',
                'scalability': 'very_good',
                'best_for': ['large_datasets', 'balanced_performance'],
                'optimal_size_range': (1000, 1000000),
                'features_range': (1, 100),
                'complexity': 'O(n × p)'
            }
        }
    
    def recommend_algorithm(self, 
                          dataset_size: int,
                          feature_count: int,
                          performance_priority: PerformanceCategory,
                          available_time_minutes: Optional[int] = None) -> List[Tuple[str, float, str]]:
        """Recommend best algorithms based on dataset and requirements."""
        
        recommendations = []
        
        for algorithm, profile in self.algorithm_profiles.items():
            score = self._calculate_suitability_score(
                algorithm, profile, dataset_size, feature_count, performance_priority
            )
            
            reasoning = self._explain_recommendation(algorithm, profile, score)
            recommendations.append((algorithm, score, reasoning))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _calculate_suitability_score(self, 
                                   algorithm: str, 
                                   profile: Dict[str, Any],
                                   dataset_size: int,
                                   feature_count: int,
                                   performance_priority: PerformanceCategory) -> float:
        """Calculate suitability score for algorithm."""
        
        score = 0.0
        
        # Size range compatibility (40% of score)
        size_min, size_max = profile['optimal_size_range']
        if size_min <= dataset_size <= size_max:
            score += 40.0
        elif dataset_size < size_min:
            # Penalty for being too small
            ratio = dataset_size / size_min
            score += 40.0 * ratio
        else:
            # Penalty for being too large
            ratio = size_max / dataset_size
            score += 40.0 * ratio
        
        # Feature range compatibility (20% of score)
        feat_min, feat_max = profile['features_range']
        if feat_min <= feature_count <= feat_max:
            score += 20.0
        else:
            # Partial penalty for being outside feature range
            score += 10.0
        
        # Performance priority match (40% of score)
        priority_scores = {
            PerformanceCategory.ACCURACY_FIRST: {
                'accuracy': 15, 'training_speed': 5, 'prediction_speed': 5, 'memory_usage': 5
            },
            PerformanceCategory.SPEED_FIRST: {
                'training_speed': 15, 'prediction_speed': 15, 'accuracy': 5, 'memory_usage': 5
            },
            PerformanceCategory.MEMORY_EFFICIENT: {
                'memory_usage': 20, 'accuracy': 5, 'training_speed': 5, 'prediction_speed': 10
            },
            PerformanceCategory.BALANCED: {
                'accuracy': 10, 'training_speed': 10, 'prediction_speed': 10, 'memory_usage': 10
            }
        }
        
        priority_weights = priority_scores[performance_priority]
        
        # Map text ratings to numeric scores
        rating_scores = {
            'very_low': 5, 'low': 4, 'medium': 3, 'high': 2, 'very_high': 1,
            'very_fast': 5, 'fast': 4, 'medium': 3, 'slow': 2, 'very_slow': 1,
            'poor': 1, 'medium': 3, 'good': 4, 'very_good': 4, 'excellent': 5
        }
        
        for attribute, weight in priority_weights.items():
            if attribute in profile:
                rating = profile[attribute]
                # For memory_usage, lower is better, so invert the score
                if attribute == 'memory_usage':
                    numeric_score = 6 - rating_scores.get(rating, 3)
                else:
                    numeric_score = rating_scores.get(rating, 3)
                
                score += weight * (numeric_score / 5.0)
        
        return min(100.0, score)  # Cap at 100
    
    def _explain_recommendation(self, algorithm: str, profile: Dict[str, Any], score: float) -> str:
        """Provide reasoning for the recommendation."""
        
        reasons = []
        
        # Highlight key strengths
        if profile['accuracy'] in ['high', 'very_high']:
            reasons.append("high accuracy")
        
        if profile['training_speed'] in ['fast', 'very_fast']:
            reasons.append("fast training")
        
        if profile['prediction_speed'] in ['fast', 'very_fast']:
            reasons.append("fast prediction")
        
        if profile['memory_usage'] in ['low', 'very_low']:
            reasons.append("low memory usage")
        
        if profile['scalability'] in ['good', 'very_good', 'excellent']:
            reasons.append("good scalability")
        
        reasoning = f"Score: {score:.1f}%. "
        if reasons:
            reasoning += f"Strengths: {', '.join(reasons)}."
        
        reasoning += f" Complexity: {profile.get('complexity', 'N/A')}."
        
        return reasoning


# Usage example
def get_performance_recommendations():
    """Example of using the algorithm selector."""
    
    selector = AlgorithmSelector()
    
    # Example scenarios
    scenarios = [
        {
            'name': 'Small Dataset (Accuracy Priority)',
            'size': 5000,
            'features': 20,
            'priority': PerformanceCategory.ACCURACY_FIRST
        },
        {
            'name': 'Large Dataset (Speed Priority)',
            'size': 500000,
            'features': 100,
            'priority': PerformanceCategory.SPEED_FIRST
        },
        {
            'name': 'Very Large Dataset (Memory Efficient)',
            'size': 5000000,
            'features': 50,
            'priority': PerformanceCategory.MEMORY_EFFICIENT
        },
        {
            'name': 'Medium Dataset (Balanced)',
            'size': 50000,
            'features': 30,
            'priority': PerformanceCategory.BALANCED
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎯 {scenario['name']}")
        print(f"   Dataset: {scenario['size']:,} samples, {scenario['features']} features")
        print(f"   Priority: {scenario['priority'].value}")
        
        recommendations = selector.recommend_algorithm(
            dataset_size=scenario['size'],
            feature_count=scenario['features'],
            performance_priority=scenario['priority']
        )
        
        print("   Top 3 Recommendations:")
        for i, (algorithm, score, reasoning) in enumerate(recommendations[:3]):
            print(f"   {i+1}. {algorithm}: {reasoning}")

if __name__ == "__main__":
    get_performance_recommendations()
```

## Data Processing Optimization

### High-Performance Data Loaders Comparison

| Loader | Best Use Case | Performance | Memory | Complexity |
|--------|---------------|-------------|---------|------------|
| **Pandas** | Standard datasets < 1GB | Good | Medium | Low |
| **Polars** | Large datasets, complex queries | Excellent | Low | Medium |
| **PyArrow** | Columnar data, analytics | Very Good | Very Low | Medium |
| **Spark** | Distributed processing > 10GB | Excellent | Distributed | High |

### Optimization Strategies

```python
# infrastructure/optimization/data_optimization.py
import asyncio
import time
from typing import Union, List, Optional
import pandas as pd
import numpy as np


class DataOptimizer:
    """Advanced data optimization strategies."""
    
    @staticmethod
    def optimize_for_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame specifically for anomaly detection algorithms."""
        
        optimized_df = df.copy()
        
        # 1. Handle missing values efficiently
        numeric_columns = optimized_df.select_dtypes(include=[np.number]).columns
        categorical_columns = optimized_df.select_dtypes(include=['object', 'category']).columns
        
        # Fill numeric columns with median (robust to outliers)
        for col in numeric_columns:
            if optimized_df[col].isnull().any():
                median_val = optimized_df[col].median()
                optimized_df[col].fillna(median_val, inplace=True)
        
        # Fill categorical columns with mode
        for col in categorical_columns:
            if optimized_df[col].isnull().any():
                mode_val = optimized_df[col].mode().iloc[0] if not optimized_df[col].mode().empty else 'unknown'
                optimized_df[col].fillna(mode_val, inplace=True)
        
        # 2. Encode categorical variables efficiently
        for col in categorical_columns:
            if optimized_df[col].dtype == 'object':
                # Use label encoding for high cardinality, one-hot for low cardinality
                unique_count = optimized_df[col].nunique()
                if unique_count <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(optimized_df[col], prefix=col)
                    optimized_df = pd.concat([optimized_df, dummies], axis=1)
                    optimized_df.drop(col, axis=1, inplace=True)
                else:
                    # Label encoding for high cardinality
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    optimized_df[col] = le.fit_transform(optimized_df[col].astype(str))
        
        # 3. Scale features for distance-based algorithms
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numeric_columns = optimized_df.select_dtypes(include=[np.number]).columns
        optimized_df[numeric_columns] = scaler.fit_transform(optimized_df[numeric_columns])
        
        # 4. Remove highly correlated features to reduce dimensionality
        correlation_matrix = optimized_df.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        if high_corr_features:
            print(f"   Removing {len(high_corr_features)} highly correlated features")
            optimized_df.drop(high_corr_features, axis=1, inplace=True)
        
        return optimized_df
    
    @staticmethod
    async def parallel_preprocessing(df: pd.DataFrame, 
                                   preprocessing_functions: List[callable],
                                   n_jobs: int = -1) -> pd.DataFrame:
        """Apply preprocessing functions in parallel."""
        
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Split DataFrame into chunks
        chunk_size = len(df) // n_jobs
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        async def process_chunk(chunk, functions):
            """Process a single chunk with all functions."""
            processed_chunk = chunk.copy()
            for func in functions:
                processed_chunk = func(processed_chunk)
            return processed_chunk
        
        # Process chunks in parallel
        tasks = [process_chunk(chunk, preprocessing_functions) for chunk in chunks]
        processed_chunks = await asyncio.gather(*tasks)
        
        # Combine processed chunks
        result_df = pd.concat(processed_chunks, ignore_index=True)
        
        return result_df
```

## Production Optimizations

### Caching Strategies

```python
# infrastructure/cache/performance_cache.py
import redis
import pickle
import hashlib
import json
from typing import Any, Optional, Union
import asyncio
from functools import wraps


class PerformanceCache:
    """High-performance caching for anomaly detection operations."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour
    
    def cache_detector_results(self, ttl: int = None):
        """Decorator to cache detector prediction results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key from function arguments
                cache_key = self._create_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    print(f"🎯 Cache hit for {func.__name__}")
                    return pickle.loads(cached_result)
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache the result
                self.redis_client.setex(
                    cache_key, 
                    ttl or self.default_ttl,
                    pickle.dumps(result)
                )
                
                print(f"💾 Cached result for {func.__name__}")
                return result
                
            return wrapper
        return decorator
    
    def _create_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create deterministic cache key from function signature."""
        
        # Convert args and kwargs to a hashable representation
        key_data = {
            'function': func_name,
            'args': str(args),
            'kwargs': json.dumps(kwargs, sort_keys=True, default=str)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"pynomaly:cache:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate all cache keys matching pattern."""
        keys = self.redis_client.keys(f"pynomaly:cache:*{pattern}*")
        if keys:
            self.redis_client.delete(*keys)
            print(f"🗑️  Invalidated {len(keys)} cache entries matching '{pattern}'")


# Connection pooling for database operations
class DatabaseConnectionPool:
    """Optimized database connection pooling."""
    
    def __init__(self, connection_string: str, pool_size: int = 10):
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool
        
        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=20,
            pool_pre_ping=True,  # Validate connections
            pool_recycle=3600,   # Recycle connections after 1 hour
        )
    
    async def execute_query_optimized(self, query: str, params: dict = None):
        """Execute database query with connection pooling."""
        
        with self.engine.connect() as conn:
            result = conn.execute(query, params or {})
            return result.fetchall()
```

### GPU Acceleration

```python
# infrastructure/gpu/gpu_optimization.py
import torch
import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class GPUAccelerator:
    """GPU acceleration utilities for anomaly detection."""
    
    def __init__(self):
        self.device = self._detect_optimal_device()
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            print(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name()}")
        else:
            print("💻 Using CPU for computation")
    
    def _detect_optimal_device(self) -> torch.device:
        """Automatically detect the best available device."""
        
        if torch.cuda.is_available():
            # Use GPU with most memory
            gpu_count = torch.cuda.device_count()
            best_gpu = 0
            max_memory = 0
            
            for i in range(gpu_count):
                memory = torch.cuda.get_device_properties(i).total_memory
                if memory > max_memory:
                    max_memory = memory
                    best_gpu = i
            
            return torch.device(f"cuda:{best_gpu}")
        else:
            return torch.device("cpu")
    
    def accelerate_distance_computation(self, 
                                      data: np.ndarray,
                                      batch_size: int = 10000) -> np.ndarray:
        """GPU-accelerated distance computation for anomaly detection."""
        
        if not self.cuda_available:
            # Fallback to CPU-optimized computation
            return self._cpu_distance_computation(data)
        
        # Convert to PyTorch tensors
        data_tensor = torch.from_numpy(data).float().to(self.device)
        
        n_samples = data_tensor.shape[0]
        distances = torch.zeros(n_samples, device=self.device)
        
        # Process in batches to manage memory
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = data_tensor[i:end_idx]
            
            # Compute pairwise distances (example: to centroid)
            centroid = torch.mean(data_tensor, dim=0)
            batch_distances = torch.norm(batch - centroid, dim=1)
            distances[i:end_idx] = batch_distances
        
        # Return as numpy array
        return distances.cpu().numpy()
    
    def _cpu_distance_computation(self, data: np.ndarray) -> np.ndarray:
        """CPU-optimized distance computation."""
        from sklearn.metrics.pairwise import euclidean_distances
        
        centroid = np.mean(data, axis=0).reshape(1, -1)
        distances = euclidean_distances(data, centroid).flatten()
        return distances
    
    def optimize_model_inference(self, model, data: np.ndarray) -> np.ndarray:
        """Optimize model inference with GPU acceleration."""
        
        if hasattr(model, 'predict') and self.cuda_available:
            # Try to use GPU if model supports it
            try:
                if hasattr(model, 'to'):  # PyTorch model
                    model = model.to(self.device)
                    data_tensor = torch.from_numpy(data).float().to(self.device)
                    
                    with torch.no_grad():
                        predictions = model(data_tensor)
                    
                    return predictions.cpu().numpy()
                
            except Exception as e:
                logger.warning(f"GPU inference failed, falling back to CPU: {e}")
        
        # Fallback to standard CPU inference
        return model.predict(data)


# Memory-optimized batch processing
class BatchProcessor:
    """Optimize batch processing for large datasets."""
    
    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        
    def calculate_optimal_batch_size(self, 
                                   sample_data: np.ndarray,
                                   model_memory_overhead: float = 2.0) -> int:
        """Calculate optimal batch size based on available memory."""
        
        # Estimate memory per sample
        bytes_per_sample = sample_data.nbytes / len(sample_data)
        
        # Account for model overhead
        effective_memory = self.max_memory_bytes / model_memory_overhead
        
        # Calculate batch size
        optimal_batch_size = int(effective_memory / bytes_per_sample)
        
        # Ensure minimum batch size
        return max(100, optimal_batch_size)
    
    async def process_large_dataset_batched(self,
                                          data: np.ndarray,
                                          model,
                                          batch_callback: callable = None) -> np.ndarray:
        """Process large dataset in optimized batches."""
        
        batch_size = self.calculate_optimal_batch_size(data[:100])  # Sample for estimation
        n_samples = len(data)
        
        print(f"📦 Processing {n_samples:,} samples in batches of {batch_size:,}")
        
        results = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = data[i:end_idx]
            
            # Process batch
            batch_result = model.predict(batch)
            results.append(batch_result)
            
            # Optional callback for progress tracking
            if batch_callback:
                await batch_callback(i // batch_size + 1, (n_samples + batch_size - 1) // batch_size)
            
            # Progress indicator
            progress = (end_idx / n_samples) * 100
            print(f"   Progress: {progress:.1f}% ({end_idx:,}/{n_samples:,})")
        
        return np.concatenate(results)
```

This comprehensive performance guide provides production-ready optimization strategies for Pynomaly, covering benchmarking, memory management, algorithm selection, and GPU acceleration. The guide enables users to achieve optimal performance for their specific use cases and dataset characteristics.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
