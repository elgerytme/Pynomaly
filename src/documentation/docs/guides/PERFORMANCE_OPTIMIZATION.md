# Performance Optimization Guide

This guide covers performance optimization strategies and tools available in Pynomaly for handling large datasets and production workloads.

## Quick Start

### Enable All Optimizations
```python
from pynomaly.infrastructure.adapters.optimized_pyod_adapter import OptimizedPyODAdapter
from pynomaly.infrastructure.data_loaders.optimized_csv_loader import OptimizedCSVLoader
from pynomaly.infrastructure.performance.memory_manager import AdaptiveMemoryManager

# Use optimized components
adapter = OptimizedPyODAdapter(
    algorithm="IsolationForest",
    enable_feature_selection=True,
    enable_batch_processing=True,
    enable_prediction_cache=True,
    memory_optimization=True
)

loader = OptimizedCSVLoader(
    memory_optimization=True,
    dtype_inference=True,
    chunk_size=50000
)

memory_manager = AdaptiveMemoryManager(
    target_memory_percent=80.0,
    enable_automatic_optimization=True
)
```

### CLI Performance Tools
```bash
# Run performance benchmarks
pynomaly perf benchmark --suite comprehensive

# Monitor real-time performance
pynomaly perf monitor

# Generate performance report
pynomaly perf report --format html
```

## Performance Optimizations

### 1. Cache Optimization

#### Batch Cache Operations
Significantly faster than individual cache operations for multiple keys:

```python
from pynomaly.infrastructure.caching.advanced_cache_service import AdvancedCacheService

cache = AdvancedCacheService()

# Instead of individual operations (slow)
for key, value in items.items():
    await cache.set(key, value)

# Use batch operations (fast)
results = await cache.set_batch(items)
```

**Performance Gain**: 3-10x faster for batch operations

#### Cache Configuration
```python
from pynomaly.infrastructure.caching.advanced_cache_service import CacheConfig

config = CacheConfig(
    # Multi-level caching
    enable_l1_memory=True,
    enable_l2_redis=True,

    # Memory optimization
    l1_max_size_mb=512,
    l1_max_entries=10000,

    # Compression
    compression_algorithm="lz4",  # Fast compression

    # TTL settings
    default_ttl_seconds=3600,  # 1 hour
)
```

### 2. Data Loading Optimization

#### Optimized CSV Loading
Memory-efficient loading with intelligent type inference:

```python
from pynomaly.infrastructure.data_loaders.optimized_csv_loader import OptimizedCSVLoader

loader = OptimizedCSVLoader(
    chunk_size=50000,           # Process in chunks
    memory_optimization=True,    # Optimize data types
    dtype_inference=True,       # Smart type inference
    categorical_threshold=0.5   # Convert to categorical
)

# Load large files efficiently
dataset = await loader.load("large_dataset.csv")
```

**Benefits**:
- 30-70% memory reduction
- 2-5x faster loading for large files
- Automatic dtype optimization

#### Parallel File Loading
```python
from pynomaly.infrastructure.data_loaders.optimized_csv_loader import ParallelCSVLoader

parallel_loader = ParallelCSVLoader(max_workers=4)

# Load multiple files simultaneously
file_paths = ["data1.csv", "data2.csv", "data3.csv"]
datasets = await parallel_loader.load_multiple_files(file_paths)
```

### 3. Algorithm Optimization

#### Feature Selection
Automatic feature selection reduces dimensionality and improves performance:

```python
from pynomaly.infrastructure.adapters.optimized_pyod_adapter import OptimizedPyODAdapter

adapter = OptimizedPyODAdapter(
    algorithm="IsolationForest",
    enable_feature_selection=True,
    feature_importance_threshold=0.01,  # Remove low-variance features
)

# Automatically selects important features during training
detector = await adapter.train(dataset)
```

**Benefits**:
- 20-80% reduction in features
- 2-5x faster training and prediction
- Improved model generalization

#### Batch Processing
For large datasets, use batch processing to manage memory:

```python
adapter = OptimizedPyODAdapter(
    algorithm="IsolationForest",
    enable_batch_processing=True,
    batch_size=10000,  # Process 10k samples at a time
)

# Automatically handles large datasets in batches
result = await adapter.detect(large_dataset)
```

#### Prediction Caching
Cache prediction results for identical inputs:

```python
adapter = OptimizedPyODAdapter(
    algorithm="IsolationForest",
    enable_prediction_cache=True,
)

# Subsequent predictions on same data use cache
result1 = await adapter.detect(dataset)  # Computed
result2 = await adapter.detect(dataset)  # From cache (fast)
```

### 4. Memory Management

#### Automatic Memory Optimization
```python
from pynomaly.infrastructure.performance.memory_manager import AdaptiveMemoryManager

memory_manager = AdaptiveMemoryManager(
    target_memory_percent=80.0,
    warning_threshold_percent=85.0,
    critical_threshold_percent=95.0,
    enable_automatic_optimization=True,
)

# Start background monitoring
await memory_manager.start_monitoring()

# Register objects for optimization
memory_manager.register_object("dataset", large_dataset)
memory_manager.register_object("model_cache", model_cache)

# Manual optimization when needed
results = await memory_manager.optimize_memory_usage()
```

#### Memory Profiling
```python
from pynomaly.infrastructure.performance.memory_manager import MemoryProfiler

profiler = MemoryProfiler()

@profiler.profile_function("data_loading")
async def load_data():
    return await loader.load("large_file.csv")

# View profiling results
summary = profiler.get_profile_summary()
```

### 5. Parallel Algorithm Execution

Run multiple algorithms simultaneously for comparison:

```python
from pynomaly.infrastructure.adapters.optimized_pyod_adapter import AsyncAlgorithmExecutor

executor = AsyncAlgorithmExecutor(max_concurrent=4)

algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
results = await executor.execute_multiple_algorithms(algorithms, dataset)

# Get results for all algorithms
for algo_name, result in results:
    if result:
        print(f"{algo_name}: {len(result.anomalies)} anomalies detected")
```

## Performance Monitoring

### Real-time Monitoring
```python
from pynomaly.infrastructure.performance.memory_manager import AdaptiveMemoryManager

memory_manager = AdaptiveMemoryManager()

# Get current memory usage
usage = memory_manager.get_memory_usage()
print(f"Memory usage: {usage.percent_used:.1f}%")

# Get performance trends
trends = memory_manager.get_memory_trends(hours=1)
print(f"Memory trend: {trends['trend']}")
```

### Performance Metrics
```python
from pynomaly.infrastructure.adapters.optimized_pyod_adapter import OptimizedPyODAdapter

adapter = OptimizedPyODAdapter("IsolationForest")

# Get performance summary
summary = adapter.get_performance_summary()
print(f"Feature reduction: {summary['feature_selection_stats']['reduction_ratio']:.2%}")
print(f"Cache hits: {summary['cache_stats']['cached_predictions']}")
```

### Benchmarking
```bash
# Run comprehensive benchmarks
pynomaly perf benchmark --suite comprehensive --output-dir ./benchmarks

# Compare specific algorithms
pynomaly perf compare --algorithms IsolationForest LocalOutlierFactor OneClassSVM

# Scalability analysis
pynomaly perf scalability --algorithm IsolationForest --max-size 100000
```

## Configuration Guidelines

### Small Datasets (< 10K rows)
```python
# Minimal optimization needed
adapter = OptimizedPyODAdapter(
    algorithm="IsolationForest",
    enable_feature_selection=False,
    enable_batch_processing=False,
    memory_optimization=False,
)
```

### Medium Datasets (10K - 100K rows)
```python
# Moderate optimization
adapter = OptimizedPyODAdapter(
    algorithm="IsolationForest",
    enable_feature_selection=True,
    enable_batch_processing=False,
    enable_prediction_cache=True,
    memory_optimization=True,
)
```

### Large Datasets (100K - 1M rows)
```python
# Full optimization
adapter = OptimizedPyODAdapter(
    algorithm="IsolationForest",
    enable_feature_selection=True,
    enable_batch_processing=True,
    batch_size=10000,
    enable_prediction_cache=True,
    memory_optimization=True,
)
```

### Very Large Datasets (1M+ rows)
```python
# Maximum optimization + streaming
adapter = OptimizedPyODAdapter(
    algorithm="IsolationForest",
    enable_feature_selection=True,
    enable_batch_processing=True,
    batch_size=50000,
    enable_prediction_cache=True,
    memory_optimization=True,
    max_workers=8,
)

# Use chunked loading
loader = OptimizedCSVLoader(
    chunk_size=100000,
    memory_optimization=True,
    dtype_inference=True,
)
```

## Algorithm-Specific Optimizations

### Isolation Forest
```python
# Optimized parameters for large datasets
parameters = {
    "n_estimators": 100,        # Good balance of accuracy/speed
    "max_samples": "auto",      # Automatic sample size
    "contamination": 0.1,       # Adjust based on data
    "n_jobs": -1,              # Use all CPU cores
    "random_state": 42,        # Reproducibility
}
```

### Local Outlier Factor
```python
# Memory-efficient LOF
parameters = {
    "n_neighbors": 20,          # Smaller neighborhood for speed
    "algorithm": "auto",        # Automatic algorithm selection
    "leaf_size": 30,           # Balanced tree structure
    "n_jobs": -1,              # Parallel processing
}
```

### One-Class SVM
```python
# Fast SVM configuration
parameters = {
    "kernel": "rbf",           # Good general-purpose kernel
    "gamma": "scale",          # Automatic gamma scaling
    "nu": 0.1,                # Contamination estimate
    "cache_size": 500,         # Increase cache for large datasets
}
```

## Production Deployment Optimizations

### Environment Configuration
```bash
# Environment variables for optimization
export OMP_NUM_THREADS=8              # OpenMP threads
export MKL_NUM_THREADS=8              # Intel MKL threads
export OPENBLAS_NUM_THREADS=8         # OpenBLAS threads
export NUMEXPR_MAX_THREADS=8          # NumExpr threads
```

### Memory Settings
```python
# Production memory configuration
memory_manager = AdaptiveMemoryManager(
    target_memory_percent=70.0,        # Conservative target
    warning_threshold_percent=80.0,    # Early warning
    critical_threshold_percent=90.0,   # Emergency threshold
    optimization_interval_seconds=60.0, # Regular optimization
    enable_automatic_optimization=True,
)
```

### Cache Configuration
```python
# Production cache configuration
cache_config = CacheConfig(
    enable_l1_memory=True,
    enable_l2_redis=True,              # Use Redis for persistence
    l1_max_size_mb=1024,              # 1GB L1 cache
    l2_redis_url="redis://localhost:6379",
    compression_algorithm="zstd",      # Better compression ratio
    default_ttl_seconds=7200,         # 2 hours
)
```

## Troubleshooting Performance Issues

### High Memory Usage
1. **Enable memory optimization**:
   ```python
   memory_manager = AdaptiveMemoryManager(enable_automatic_optimization=True)
   await memory_manager.start_monitoring()
   ```

2. **Use chunked processing**:
   ```python
   loader = OptimizedCSVLoader(chunk_size=10000)
   ```

3. **Enable feature selection**:
   ```python
   adapter = OptimizedPyODAdapter(enable_feature_selection=True)
   ```

### Slow Training/Prediction
1. **Enable batch processing**:
   ```python
   adapter = OptimizedPyODAdapter(enable_batch_processing=True, batch_size=50000)
   ```

2. **Use prediction caching**:
   ```python
   adapter = OptimizedPyODAdapter(enable_prediction_cache=True)
   ```

3. **Optimize algorithm parameters**:
   ```python
   # Reduce complexity
   parameters = {"n_estimators": 50, "max_samples": 1000}
   ```

### Cache Performance Issues
1. **Use batch operations**:
   ```python
   # Instead of individual sets
   await cache.set_batch(multiple_items)
   ```

2. **Optimize cache size**:
   ```python
   config = CacheConfig(l1_max_size_mb=2048)  # Increase cache size
   ```

3. **Enable compression**:
   ```python
   config = CacheConfig(compression_algorithm="lz4")
   ```

## Performance Metrics

### Key Performance Indicators
- **Detection Speed**: Samples processed per second
- **Memory Efficiency**: Peak memory usage vs dataset size
- **Cache Hit Rate**: Percentage of cache hits
- **Feature Reduction**: Percentage of features selected
- **Throughput**: Requests processed per minute

### Monitoring Dashboard
Access real-time performance metrics at:
- **Web Interface**: `http://localhost:8000/performance`
- **Prometheus Metrics**: `http://localhost:8000/metrics`
- **CLI Monitoring**: `pynomaly perf monitor`

## Best Practices

1. **Start with defaults** and optimize based on actual performance
2. **Profile before optimizing** to identify bottlenecks
3. **Use appropriate optimization level** for your dataset size
4. **Monitor memory usage** in production
5. **Cache frequently accessed data**
6. **Use batch operations** for multiple items
7. **Enable feature selection** for high-dimensional data
8. **Test performance impact** of each optimization

## Support

For performance-related questions:
- **Documentation**: [Performance Section](https://pynomaly.readthedocs.io/performance)
- **GitHub Issues**: Tag with `performance`
- **Community**: [Discussions](https://github.com/pynomaly/pynomaly/discussions)
