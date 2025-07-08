# Performance Tuning Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔶 [Advanced Features](README.md) > ⚡ Performance Tuning

---


This guide provides comprehensive strategies for optimizing Pynomaly performance across different scales and use cases.

## Overview

Performance optimization in Pynomaly involves several key areas:

- **Algorithm Selection**: Choosing the right algorithm for your data and requirements
- **Data Optimization**: Preprocessing and feature engineering for efficiency
- **System Resources**: Memory, CPU, and I/O optimization
- **Scaling Strategies**: Horizontal and vertical scaling approaches
- **Caching**: Intelligent caching for repeated operations

## Quick Performance Assessment

### Benchmark Your Current Setup

```bash
# Run comprehensive performance benchmark
python examples/performance_benchmarking.py

# Quick algorithm comparison
pynomaly experiments create "Performance Test" dataset_123 \
  --algorithm IsolationForest \
  --algorithm LOF \
  --algorithm COPOD \
  --metric training_time \
  --metric prediction_time
```

### System Resource Monitoring

```bash
# Monitor during operations
top -p $(pgrep -f pynomaly)
htop

# Memory usage tracking
python -c "
import psutil
import pynomaly
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## Algorithm Performance Characteristics

### Speed Rankings (Training Time)

Based on benchmarks with 10,000 samples, 10 features:

| Rank | Algorithm | Training Time | Use Case |
|------|-----------|---------------|----------|
| 1 | COPOD | ~50ms | General purpose, fast training |
| 2 | ECOD | ~75ms | High-dimensional data |
| 3 | IsolationForest | ~150ms | Balanced performance |
| 4 | OCSVM | ~300ms | Non-linear patterns |
| 5 | LOF | ~500ms | Local anomalies |
| 6 | AutoEncoder | ~2000ms | Complex patterns |

### Memory Efficiency Rankings

| Rank | Algorithm | Memory Usage | Scalability |
|------|-----------|--------------|-------------|
| 1 | COPOD | Low | Excellent |
| 2 | ECOD | Low | Excellent |
| 3 | IsolationForest | Medium | Good |
| 4 | LOF | High | Limited |
| 5 | OCSVM | High | Limited |
| 6 | AutoEncoder | Very High | GPU Required |

### Accuracy vs Performance Trade-offs

```python
# Performance-optimized configuration
fast_detector = {
    "algorithm": "COPOD",
    "parameters": {
        "contamination": 0.1
    }
}

# Balanced configuration
balanced_detector = {
    "algorithm": "IsolationForest",
    "parameters": {
        "contamination": 0.1,
        "n_estimators": 100,
        "max_samples": 256
    }
}

# Accuracy-optimized configuration
accurate_detector = {
    "algorithm": "LOF",
    "parameters": {
        "contamination": 0.1,
        "n_neighbors": 20,
        "algorithm": "ball_tree"
    }
}
```

## Data Optimization Strategies

### Feature Engineering for Performance

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PCA
from sklearn.feature_selection import SelectKBest, f_classif

class PerformanceOptimizer:
    """Optimize datasets for anomaly detection performance."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=20)
        self.pca = PCA(n_components=0.95)  # Keep 95% variance

    def optimize_features(self, df, target_features=50):
        """Optimize feature set for performance."""

        # 1. Remove constant features
        constant_features = df.columns[df.nunique() <= 1]
        df = df.drop(columns=constant_features)
        print(f"Removed {len(constant_features)} constant features")

        # 2. Remove highly correlated features
        correlation_matrix = df.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        high_corr_features = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > 0.95)
        ]
        df = df.drop(columns=high_corr_features)
        print(f"Removed {len(high_corr_features)} highly correlated features")

        # 3. Select top K features if still too many
        if len(df.columns) > target_features:
            # Use variance-based selection for unsupervised learning
            feature_vars = df.var().sort_values(ascending=False)
            selected_features = feature_vars.head(target_features).index
            df = df[selected_features]
            print(f"Selected top {target_features} features by variance")

        return df

    def optimize_data_types(self, df):
        """Optimize data types for memory efficiency."""

        for col in df.columns:
            if df[col].dtype == 'int64':
                if df[col].min() >= 0:
                    if df[col].max() < 255:
                        df[col] = df[col].astype('uint8')
                    elif df[col].max() < 65535:
                        df[col] = df[col].astype('uint16')
                    else:
                        df[col] = df[col].astype('uint32')
                else:
                    if df[col].min() > -128 and df[col].max() < 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() > -32768 and df[col].max() < 32767:
                        df[col] = df[col].astype('int16')
                    else:
                        df[col] = df[col].astype('int32')

            elif df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')

        return df

# Usage example
optimizer = PerformanceOptimizer()
optimized_df = optimizer.optimize_features(raw_df)
optimized_df = optimizer.optimize_data_types(optimized_df)
```

### Data Preprocessing Pipeline

```python
from pynomaly.infrastructure.data_processing import DataPreprocessor

class HighPerformancePreprocessor(DataPreprocessor):
    """Performance-optimized data preprocessing."""

    def __init__(self):
        super().__init__()
        self.chunk_size = 10000  # Process in chunks

    def preprocess_large_dataset(self, file_path, output_path):
        """Process large datasets in chunks."""

        chunk_list = []

        # Process in chunks to manage memory
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            # Basic cleaning
            chunk = chunk.dropna()
            chunk = chunk.select_dtypes(include=[np.number])

            # Outlier removal (optional, can be slow)
            if chunk.shape[0] < 1000:  # Only for small chunks
                Q1 = chunk.quantile(0.25)
                Q3 = chunk.quantile(0.75)
                IQR = Q3 - Q1
                chunk = chunk[~((chunk < (Q1 - 1.5 * IQR)) |
                              (chunk > (Q3 + 1.5 * IQR))).any(axis=1)]

            chunk_list.append(chunk)

        # Combine chunks
        result = pd.concat(chunk_list, ignore_index=True)
        result.to_parquet(output_path, compression='snappy')

        return result
```

### Efficient Data Loading

```python
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Use Polars for fast data loading
def load_data_efficiently(file_path, sample_size=None):
    """Load data with optimal performance."""

    if file_path.endswith('.parquet'):
        # Parquet is fastest
        if sample_size:
            return pl.read_parquet(file_path).sample(sample_size)
        return pl.read_parquet(file_path)

    elif file_path.endswith('.csv'):
        # Optimized CSV reading
        df = pl.read_csv(
            file_path,
            try_parse_dates=True,
            infer_schema_length=1000  # Fast schema inference
        )

        if sample_size:
            df = df.sample(sample_size)

        return df

    else:
        # Fallback to pandas
        return pd.read_csv(file_path, nrows=sample_size)

# Example usage
df = load_data_efficiently("large_dataset.parquet", sample_size=10000)
```

## Algorithm-Specific Optimizations

### IsolationForest Optimization

```python
# Performance-optimized IsolationForest
optimal_isolation_forest = {
    "contamination": 0.1,
    "n_estimators": 100,      # Balance: more trees = better accuracy, slower training
    "max_samples": 256,       # Key optimization: limit sample size
    "max_features": 1.0,      # Use all features (usually optimal)
    "bootstrap": False,       # Faster without bootstrap
    "n_jobs": -1,            # Use all CPU cores
    "random_state": 42,      # Reproducibility
    "warm_start": False      # Don't use for one-time training
}

# For very large datasets
large_dataset_isolation_forest = {
    "contamination": 0.1,
    "n_estimators": 50,       # Fewer trees for speed
    "max_samples": 128,       # Smaller sample size
    "max_features": 0.8,      # Use subset of features
    "n_jobs": -1
}
```

### LOF Optimization

```python
# Performance-optimized LOF
optimal_lof = {
    "contamination": 0.1,
    "n_neighbors": 20,        # Start with 20, tune based on data
    "algorithm": "ball_tree", # Usually fastest for medium datasets
    "leaf_size": 30,         # Default is usually optimal
    "metric": "minkowski",   # Euclidean distance (p=2)
    "p": 2,
    "n_jobs": -1
}

# For different dataset sizes
small_dataset_lof = {"n_neighbors": 10, "algorithm": "brute"}
medium_dataset_lof = {"n_neighbors": 20, "algorithm": "ball_tree"}
large_dataset_lof = {"n_neighbors": 30, "algorithm": "kd_tree"}
```

### OCSVM Optimization

```python
# Performance-optimized One-Class SVM
optimal_ocsvm = {
    "contamination": 0.1,
    "kernel": "rbf",         # RBF usually best balance
    "gamma": "scale",        # Automatic gamma scaling
    "nu": 0.1,              # Should match contamination
    "degree": 3,            # Only for poly kernel
    "coef0": 0.0,           # Only for poly/sigmoid
    "tol": 1e-3,            # Tolerance for stopping
    "shrinking": True,      # Use shrinking heuristics
    "cache_size": 200,      # MB of cache (increase for large datasets)
    "max_iter": -1          # No limit
}

# For large datasets (use approximation)
large_dataset_ocsvm = {
    "kernel": "linear",     # Faster than RBF
    "nu": 0.1,
    "tol": 1e-2,           # Looser tolerance
    "shrinking": True,
    "cache_size": 500      # More cache for large datasets
}
```

## System-Level Optimizations

### Memory Management

```python
import gc
import psutil
from contextlib import contextmanager

@contextmanager
def memory_management():
    """Context manager for memory optimization."""

    # Pre-execution cleanup
    gc.collect()

    initial_memory = psutil.virtual_memory().percent
    print(f"Initial memory usage: {initial_memory:.1f}%")

    try:
        yield
    finally:
        # Post-execution cleanup
        gc.collect()
        final_memory = psutil.virtual_memory().percent
        print(f"Final memory usage: {final_memory:.1f}%")
        print(f"Memory change: {final_memory - initial_memory:+.1f}%")

# Usage
with memory_management():
    # Your anomaly detection code here
    results = detector.detect(large_dataset)
```

### CPU Optimization

```python
import os
import multiprocessing as mp

# Set optimal CPU usage
def optimize_cpu_usage():
    """Configure optimal CPU settings."""

    # Set number of threads for NumPy operations
    cpu_count = mp.cpu_count()
    optimal_threads = max(1, cpu_count - 1)  # Leave one core free

    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)

    print(f"Configured {optimal_threads} threads for computation")

    return optimal_threads

# Configure at startup
optimize_cpu_usage()
```

### I/O Optimization

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedDataManager:
    """High-performance data operations."""

    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def load_multiple_datasets(self, file_paths):
        """Load multiple datasets concurrently."""

        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(
                self.executor,
                self.load_single_dataset,
                path
            )
            for path in file_paths
        ]

        datasets = await asyncio.gather(*tasks)
        return datasets

    def load_single_dataset(self, file_path):
        """Load single dataset with optimizations."""

        if file_path.endswith('.parquet'):
            # Parquet with specific optimizations
            return pd.read_parquet(
                file_path,
                engine='pyarrow',
                use_threads=True
            )
        elif file_path.endswith('.csv'):
            # CSV with optimizations
            return pd.read_csv(
                file_path,
                engine='c',           # Fast C engine
                low_memory=False,     # Read entire file into memory
                dtype_backend='pyarrow'  # Use Arrow types
            )
        else:
            return pd.read_csv(file_path)

# Usage
manager = OptimizedDataManager()
datasets = await manager.load_multiple_datasets(['data1.csv', 'data2.csv'])
```

## Caching Strategies

### Result Caching

```python
from functools import lru_cache
import hashlib
import pickle
import redis

class SmartCache:
    """Intelligent caching for anomaly detection results."""

    def __init__(self, cache_size=1000, use_redis=False):
        self.cache_size = cache_size
        self.use_redis = use_redis

        if use_redis:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        else:
            self.memory_cache = {}

    def _hash_data(self, data):
        """Create hash for data."""
        if isinstance(data, pd.DataFrame):
            return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()

    def get_cached_result(self, detector_id, data):
        """Retrieve cached result if available."""

        cache_key = f"{detector_id}_{self._hash_data(data)}"

        if self.use_redis:
            cached = self.redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)
        else:
            return self.memory_cache.get(cache_key)

        return None

    def cache_result(self, detector_id, data, result):
        """Cache detection result."""

        cache_key = f"{detector_id}_{self._hash_data(data)}"

        if self.use_redis:
            self.redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                pickle.dumps(result)
            )
        else:
            if len(self.memory_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]

            self.memory_cache[cache_key] = result

# Integration with detector
class CachedDetector:
    def __init__(self, detector, cache=None):
        self.detector = detector
        self.cache = cache or SmartCache()

    def detect(self, data):
        # Check cache first
        cached_result = self.cache.get_cached_result(
            self.detector.id,
            data
        )

        if cached_result:
            print("Cache hit!")
            return cached_result

        # Compute and cache result
        result = self.detector.detect(data)
        self.cache.cache_result(self.detector.id, data, result)

        return result
```

### Model Caching

```python
import joblib
import os
from pathlib import Path

class ModelCache:
    """Cache trained models for reuse."""

    def __init__(self, cache_dir="~/.pynomaly/model_cache"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, detector_config):
        """Generate consistent path for model."""
        config_hash = hashlib.md5(
            str(sorted(detector_config.items())).encode()
        ).hexdigest()
        return self.cache_dir / f"model_{config_hash}.pkl"

    def load_cached_model(self, detector_config):
        """Load cached model if available."""
        model_path = self._get_model_path(detector_config)

        if model_path.exists():
            print(f"Loading cached model: {model_path}")
            return joblib.load(model_path)

        return None

    def cache_model(self, detector_config, model):
        """Cache trained model."""
        model_path = self._get_model_path(detector_config)
        joblib.dump(model, model_path)
        print(f"Cached model: {model_path}")

    def clear_cache(self):
        """Clear all cached models."""
        for model_file in self.cache_dir.glob("model_*.pkl"):
            model_file.unlink()
        print("Model cache cleared")
```

## Scaling Strategies

### Horizontal Scaling

```python
import dask.dataframe as dd
from dask.distributed import Client
from dask import delayed

class DistributedAnomalyDetector:
    """Distributed anomaly detection using Dask."""

    def __init__(self, n_workers=4):
        self.client = Client(n_workers=n_workers, threads_per_worker=2)
        print(f"Dask cluster: {self.client.dashboard_link}")

    def detect_distributed(self, large_dataset, detector_config, chunk_size=10000):
        """Distribute detection across workers."""

        # Convert to Dask DataFrame
        if isinstance(large_dataset, pd.DataFrame):
            ddf = dd.from_pandas(large_dataset, npartitions=10)
        else:
            ddf = dd.read_csv(large_dataset)

        # Define detection function
        @delayed
        def detect_chunk(chunk):
            # Create detector for this chunk
            detector = self._create_detector(detector_config)
            detector.fit(chunk.sample(frac=0.8))  # Train on subset

            # Detect anomalies
            results = detector.predict(chunk)
            scores = detector.decision_function(chunk)

            return pd.DataFrame({
                'is_anomaly': results == -1,
                'anomaly_score': scores
            })

        # Apply to all partitions
        results = []
        for partition in ddf.to_delayed():
            result = detect_chunk(partition)
            results.append(result)

        # Compute results
        final_results = dd.from_delayed(results).compute()
        return pd.concat(final_results, ignore_index=True)

    def _create_detector(self, config):
        """Create detector instance."""
        from sklearn.ensemble import IsolationForest
        return IsolationForest(**config)
```

### Vertical Scaling

```python
import cupy as cp  # GPU acceleration
import cudf      # GPU DataFrames

class GPUAcceleratedDetector:
    """GPU-accelerated anomaly detection."""

    def __init__(self):
        self.use_gpu = self._check_gpu_availability()

    def _check_gpu_availability(self):
        """Check if GPU is available."""
        try:
            cp.cuda.runtime.getDeviceCount()
            return True
        except:
            return False

    def detect_gpu(self, data, algorithm="isolation_forest"):
        """GPU-accelerated detection."""

        if not self.use_gpu:
            print("GPU not available, falling back to CPU")
            return self.detect_cpu(data, algorithm)

        # Convert to GPU DataFrame
        if isinstance(data, pd.DataFrame):
            gpu_data = cudf.from_pandas(data)
        else:
            gpu_data = data

        if algorithm == "isolation_forest":
            # Use cuML IsolationForest
            from cuml.ensemble import IsolationForest

            detector = IsolationForest(
                contamination=0.1,
                n_estimators=100,
                max_samples=256
            )

            detector.fit(gpu_data)
            results = detector.predict(gpu_data)
            scores = detector.decision_function(gpu_data)

            # Convert back to CPU
            return {
                'predictions': cp.asnumpy(results),
                'scores': cp.asnumpy(scores)
            }

        else:
            raise ValueError(f"Algorithm {algorithm} not supported on GPU")
```

## Monitoring and Profiling

### Performance Monitoring

```python
import time
import psutil
from contextlib import contextmanager
import matplotlib.pyplot as plt

class PerformanceMonitor:
    """Monitor performance metrics during detection."""

    def __init__(self):
        self.metrics = {
            'timestamps': [],
            'memory_usage': [],
            'cpu_usage': [],
            'processing_time': []
        }

    @contextmanager
    def monitor_operation(self, operation_name):
        """Monitor a specific operation."""

        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        start_cpu = psutil.cpu_percent()

        print(f"Starting {operation_name}...")

        try:
            yield self
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
            end_cpu = psutil.cpu_percent()

            processing_time = end_time - start_time

            self.metrics['timestamps'].append(start_time)
            self.metrics['memory_usage'].append(end_memory)
            self.metrics['cpu_usage'].append(end_cpu)
            self.metrics['processing_time'].append(processing_time)

            print(f"{operation_name} completed in {processing_time:.2f}s")
            print(f"Memory usage: {end_memory:.1f}% (+{end_memory-start_memory:+.1f}%)")
            print(f"CPU usage: {end_cpu:.1f}%")

    def plot_metrics(self):
        """Plot performance metrics."""

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Processing time
        axes[0, 0].plot(self.metrics['processing_time'])
        axes[0, 0].set_title('Processing Time')
        axes[0, 0].set_ylabel('Seconds')

        # Memory usage
        axes[0, 1].plot(self.metrics['memory_usage'])
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_ylabel('Percent')

        # CPU usage
        axes[1, 0].plot(self.metrics['cpu_usage'])
        axes[1, 0].set_title('CPU Usage')
        axes[1, 0].set_ylabel('Percent')

        # Combined metrics
        axes[1, 1].scatter(
            self.metrics['memory_usage'],
            self.metrics['processing_time']
        )
        axes[1, 1].set_title('Memory vs Processing Time')
        axes[1, 1].set_xlabel('Memory Usage (%)')
        axes[1, 1].set_ylabel('Processing Time (s)')

        plt.tight_layout()
        plt.show()

# Usage example
monitor = PerformanceMonitor()

with monitor.monitor_operation("Data Loading"):
    data = load_data_efficiently("large_dataset.csv")

with monitor.monitor_operation("Model Training"):
    detector.fit(data)

with monitor.monitor_operation("Anomaly Detection"):
    results = detector.predict(data)

monitor.plot_metrics()
```

### Profiling Code

```python
import cProfile
import pstats
from line_profiler import LineProfiler

def profile_detection_pipeline():
    """Profile the entire detection pipeline."""

    # Create profiler
    profiler = cProfile.Profile()

    # Run with profiling
    profiler.enable()

    # Your detection code here
    data = load_data("dataset.csv")
    detector = create_detector("IsolationForest")
    detector.fit(data)
    results = detector.predict(data)

    profiler.disable()

    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    # Save detailed report
    stats.dump_stats('detection_profile.prof')

# Line-by-line profiling
@profile  # Add this decorator for line profiling
def optimized_detection_function(data):
    """Function optimized for performance."""

    # Each line will be profiled
    cleaned_data = data.dropna()
    numeric_data = cleaned_data.select_dtypes(include=[np.number])
    scaled_data = StandardScaler().fit_transform(numeric_data)

    detector = IsolationForest(n_estimators=100, max_samples=256)
    detector.fit(scaled_data)

    results = detector.predict(scaled_data)
    scores = detector.decision_function(scaled_data)

    return results, scores

# Run line profiler
# kernprof -l -v your_script.py
```

## Production Optimization Checklist

### Pre-Production Performance Audit

```bash
# 1. Run comprehensive benchmarks
python examples/performance_benchmarking.py

# 2. Profile memory usage
python -m memory_profiler your_detection_script.py

# 3. Test with realistic data sizes
pynomaly detect run detector_123 --file large_test_dataset.csv --benchmark

# 4. Monitor resource usage
htop & pynomaly server start --workers 4

# 5. Load testing
ab -n 1000 -c 10 http://localhost:8000/api/v1/detectors
```

### Configuration Recommendations

```yaml
# High-performance production configuration
performance:
  # Algorithm selection
  default_algorithm: "COPOD"  # Fastest general-purpose
  fallback_algorithm: "IsolationForest"

  # Resource limits
  max_memory_mb: 4096
  max_processing_time_seconds: 300
  max_dataset_size: 1000000

  # Concurrency
  max_workers: 4
  async_processing: true
  batch_size: 1000

  # Caching
  enable_result_cache: true
  enable_model_cache: true
  cache_ttl_seconds: 3600

  # Database
  connection_pool_size: 20
  max_overflow: 30
  pool_timeout: 30

  # I/O optimization
  use_parquet: true
  compression: "snappy"
  chunk_size: 10000
```

### Monitoring Setup

```python
# Production monitoring configuration
monitoring_config = {
    "metrics": {
        "processing_time_threshold_ms": 5000,
        "memory_usage_threshold_percent": 80,
        "error_rate_threshold_percent": 5
    },

    "alerts": {
        "slack_webhook": "https://hooks.slack.com/...",
        "email_recipients": ["admin@example.com"],
        "pagerduty_key": "your-pagerduty-key"
    },

    "logging": {
        "level": "INFO",
        "format": "json",
        "rotation": "midnight",
        "retention": "30 days"
    }
}
```

## Common Performance Pitfalls

### 1. Algorithm Mismatches

```python
# AVOID: Using slow algorithms for large datasets
slow_config = {
    "algorithm": "LOF",
    "n_neighbors": 100,  # Too many neighbors
    "dataset_size": 100000  # Too large for LOF
}

# PREFER: Fast algorithms for large datasets
fast_config = {
    "algorithm": "COPOD",
    "dataset_size": 100000
}
```

### 2. Memory Inefficiencies

```python
# AVOID: Loading entire dataset into memory
def inefficient_processing(file_path):
    # Bad: loads everything at once
    df = pd.read_csv(file_path)  # Could be 10GB+
    return process_dataframe(df)

# PREFER: Chunk processing
def efficient_processing(file_path):
    results = []
    for chunk in pd.read_csv(file_path, chunksize=10000):
        result = process_dataframe(chunk)
        results.append(result)
    return pd.concat(results)
```

### 3. Unnecessary Retraining

```python
# AVOID: Retraining for similar data
def inefficient_detection(datasets):
    results = []
    for dataset in datasets:
        detector = IsolationForest()
        detector.fit(dataset)  # Retraining every time
        results.append(detector.predict(dataset))
    return results

# PREFER: Train once, predict many
def efficient_detection(datasets):
    # Train on representative sample
    combined_sample = pd.concat([
        df.sample(1000) for df in datasets[:5]
    ])

    detector = IsolationForest()
    detector.fit(combined_sample)

    # Use trained model for all datasets
    results = []
    for dataset in datasets:
        results.append(detector.predict(dataset))
    return results
```

This performance tuning guide provides comprehensive strategies for optimizing Pynomaly across different scales and use cases. Regular monitoring and profiling will help you identify bottlenecks and optimize for your specific requirements.

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
