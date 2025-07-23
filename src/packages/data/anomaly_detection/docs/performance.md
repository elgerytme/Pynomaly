# Performance Optimization and Monitoring Guide

This guide covers performance optimization techniques, monitoring strategies, and scalability considerations for the Anomaly Detection package in production environments.

## Table of Contents

1. [Overview](#overview)
2. [Performance Profiling](#performance-profiling)
3. [Algorithm Optimization](#algorithm-optimization)
4. [Memory Management](#memory-management)
5. [Parallel Processing](#parallel-processing)
6. [Caching Strategies](#caching-strategies)
7. [Database Optimization](#database-optimization)
8. [Real-time Performance](#real-time-performance)
9. [Monitoring and Metrics](#monitoring-and-metrics)
10. [Scalability Patterns](#scalability-patterns)
11. [Hardware Optimization](#hardware-optimization)
12. [Best Practices](#best-practices)

## Overview

Performance optimization is crucial for anomaly detection systems, especially when processing large datasets or operating in real-time environments. This guide provides comprehensive strategies for optimizing every aspect of your anomaly detection pipeline.

### Performance Dimensions

- **Throughput**: Number of instances processed per second
- **Latency**: Time from input to prediction
- **Memory Usage**: RAM consumption during processing
- **CPU Utilization**: Computational resource efficiency
- **I/O Performance**: Data loading and storage efficiency
- **Scalability**: Performance under increasing load

### Performance Targets

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceTargets:
    """Define performance targets for different deployment scenarios."""
    
    # Throughput targets (instances/second)
    batch_throughput: int = 1000
    streaming_throughput: int = 100
    
    # Latency targets (milliseconds)
    p50_latency: float = 50.0
    p95_latency: float = 200.0
    p99_latency: float = 500.0
    
    # Resource limits
    max_memory_gb: float = 4.0
    max_cpu_cores: int = 4
    
    # Quality thresholds
    min_accuracy: float = 0.85
    max_false_positive_rate: float = 0.05

# Example deployment configurations
PERFORMANCE_PROFILES = {
    'development': PerformanceTargets(
        batch_throughput=100,
        streaming_throughput=10,
        p50_latency=100.0,
        max_memory_gb=2.0
    ),
    'staging': PerformanceTargets(
        batch_throughput=500,
        streaming_throughput=50,
        p50_latency=75.0,
        max_memory_gb=4.0
    ),
    'production': PerformanceTargets(
        batch_throughput=2000,
        streaming_throughput=200,
        p50_latency=25.0,
        max_memory_gb=8.0,
        max_cpu_cores=8
    )
}
```

## Performance Profiling

### Comprehensive Profiler

```python
import time
import psutil
import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import Dict, List, Optional
import cProfile
import pstats
import io
from memory_profiler import profile
import tracemalloc

class PerformanceProfiler:
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process()
    
    @contextmanager
    def profile_section(self, section_name: str):
        """Profile a specific section of code."""
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = self.process.cpu_percent()
            
            self.metrics[section_name] = {
                'duration_ms': (end_time - start_time) * 1000,
                'memory_mb': end_memory,
                'memory_delta_mb': end_memory - start_memory,
                'cpu_percent': end_cpu
            }
    
    def profile_function(self, func, *args, **kwargs):
        """Profile a specific function with detailed statistics."""
        # Setup cProfile
        profiler = cProfile.Profile()
        
        # Setup memory tracking
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        # Profile execution
        start_time = time.perf_counter()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            end_time = time.perf_counter()
            
            # Get memory usage
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Extract profile statistics
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            profile_data = {
                'execution_time_ms': (end_time - start_time) * 1000,
                'memory_used_mb': (current_memory - start_memory) / 1024 / 1024,
                'peak_memory_mb': peak_memory / 1024 / 1024,
                'profile_stats': stats_stream.getvalue(),
                'result': result
            }
            
            return profile_data
    
    def benchmark_algorithms(self, algorithms: Dict, X: np.ndarray, 
                           n_runs: int = 5) -> pd.DataFrame:
        """Benchmark multiple algorithms on the same dataset."""
        results = []
        
        for algo_name, algo_instance in algorithms.items():
            algo_metrics = []
            
            for run in range(n_runs):
                with self.profile_section(f"{algo_name}_run_{run}"):
                    # Training time
                    start_train = time.perf_counter()
                    algo_instance.fit(X)
                    train_time = time.perf_counter() - start_train
                    
                    # Prediction time
                    start_pred = time.perf_counter()
                    scores = algo_instance.decision_function(X)
                    pred_time = time.perf_counter() - start_pred
                    
                    # Memory usage
                    memory_usage = self.process.memory_info().rss / 1024 / 1024
                
                run_metrics = {
                    'algorithm': algo_name,
                    'run': run,
                    'train_time_ms': train_time * 1000,
                    'pred_time_ms': pred_time * 1000,
                    'total_time_ms': (train_time + pred_time) * 1000,
                    'memory_mb': memory_usage,
                    'throughput_instances_per_sec': len(X) / (train_time + pred_time),
                    'latency_per_instance_ms': (pred_time * 1000) / len(X)
                }
                
                algo_metrics.append(run_metrics)
            
            results.extend(algo_metrics)
        
        return pd.DataFrame(results)
    
    def memory_profile_algorithm(self, algorithm, X: np.ndarray):
        """Detailed memory profiling of an algorithm."""
        @profile
        def train_and_predict():
            algorithm.fit(X)
            return algorithm.decision_function(X)
        
        # Run with memory profiling
        result = train_and_predict()
        return result
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'memory_available_gb': psutil.virtual_memory().available / 1024**3,
            'disk_usage': {
                path: psutil.disk_usage(path)._asdict() 
                for path in ['/']  # Add relevant paths
            },
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__
        }

# Example usage
profiler = PerformanceProfiler()

# Generate test data
X = np.random.randn(10000, 20)

# Define algorithms to benchmark
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

algorithms = {
    'IsolationForest': IsolationForest(n_estimators=100, random_state=42),
    'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=20),
    'OneClassSVM': OneClassSVM(kernel='rbf', nu=0.1)
}

# Run benchmark
benchmark_results = profiler.benchmark_algorithms(algorithms, X, n_runs=3)

# Analyze results
print("Algorithm Performance Comparison:")
summary = benchmark_results.groupby('algorithm').agg({
    'train_time_ms': ['mean', 'std'],
    'pred_time_ms': ['mean', 'std'],
    'throughput_instances_per_sec': ['mean', 'std'],
    'latency_per_instance_ms': ['mean', 'std'],
    'memory_mb': ['mean', 'std']
}).round(2)

print(summary)

# Profile specific algorithm in detail
algo_profile = profiler.profile_function(
    lambda: IsolationForest(n_estimators=100).fit(X).decision_function(X)
)
print(f"\nDetailed IsolationForest Profile:")
print(f"Execution time: {algo_profile['execution_time_ms']:.2f} ms")
print(f"Memory used: {algo_profile['memory_used_mb']:.2f} MB")
print(f"Peak memory: {algo_profile['peak_memory_mb']:.2f} MB")
```

### Real-time Performance Monitor

```python
import threading
import queue
import json
from datetime import datetime, timedelta
from collections import deque

class RealTimePerformanceMonitor:
    def __init__(self, window_size: int = 1000, alert_thresholds: Dict = None):
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            'latency_ms': 100,
            'error_rate': 0.05,
            'memory_mb': 1000,
            'cpu_percent': 80
        }
        
        # Metrics storage
        self.metrics_queue = queue.Queue()
        self.latency_window = deque(maxlen=window_size)
        self.error_window = deque(maxlen=window_size)
        self.throughput_window = deque(maxlen=window_size)
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        
        # Alert callbacks
        self.alert_callbacks = []
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def record_prediction(self, latency_ms: float, success: bool = True, 
                         metadata: Dict = None):
        """Record a prediction event."""
        self.metrics_queue.put({
            'timestamp': datetime.now(),
            'latency_ms': latency_ms,
            'success': success,
            'metadata': metadata or {}
        })
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        last_throughput_check = datetime.now()
        throughput_count = 0
        
        while self.monitoring:
            try:
                # Process queued metrics
                while not self.metrics_queue.empty():
                    try:
                        metric = self.metrics_queue.get_nowait()
                        self._process_metric(metric)
                        throughput_count += 1
                    except queue.Empty:
                        break
                
                # Calculate throughput every second
                now = datetime.now()
                if (now - last_throughput_check).seconds >= 1:
                    self.throughput_window.append(throughput_count)
                    throughput_count = 0
                    last_throughput_check = now
                
                # Check alerts
                self._check_alerts()
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"Monitor error: {e}")
    
    def _process_metric(self, metric: Dict):
        """Process individual metric."""
        self.latency_window.append(metric['latency_ms'])
        self.error_window.append(0 if metric['success'] else 1)
    
    def _check_alerts(self):
        """Check for alert conditions."""
        if not self.latency_window:
            return
        
        # Latency alert
        avg_latency = np.mean(list(self.latency_window)[-100:])  # Last 100 requests
        if avg_latency > self.alert_thresholds['latency_ms']:
            self._trigger_alert('high_latency', {
                'current_latency': avg_latency,
                'threshold': self.alert_thresholds['latency_ms']
            })
        
        # Error rate alert
        recent_errors = list(self.error_window)[-100:]
        error_rate = np.mean(recent_errors) if recent_errors else 0
        if error_rate > self.alert_thresholds['error_rate']:
            self._trigger_alert('high_error_rate', {
                'current_error_rate': error_rate,
                'threshold': self.alert_thresholds['error_rate']
            })
        
        # System resource alerts
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        if memory_mb > self.alert_thresholds['memory_mb']:
            self._trigger_alert('high_memory', {
                'current_memory': memory_mb,
                'threshold': self.alert_thresholds['memory_mb']
            })
        
        if cpu_percent > self.alert_thresholds['cpu_percent']:
            self._trigger_alert('high_cpu', {
                'current_cpu': cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent']
            })
    
    def _trigger_alert(self, alert_type: str, data: Dict):
        """Trigger alert callbacks."""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'data': data
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics."""
        if not self.latency_window:
            return {}
        
        latencies = list(self.latency_window)
        errors = list(self.error_window)
        throughputs = list(self.throughput_window)
        
        return {
            'latency': {
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'mean': np.mean(latencies),
                'std': np.std(latencies)
            },
            'error_rate': np.mean(errors) if errors else 0,
            'throughput': {
                'current_rps': throughputs[-1] if throughputs else 0,
                'mean_rps': np.mean(throughputs) if throughputs else 0,
                'max_rps': max(throughputs) if throughputs else 0
            },
            'system': {
                'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_percent': psutil.Process().cpu_percent()
            }
        }

# Example usage with alert handling
def alert_handler(alert):
    print(f"ALERT [{alert['timestamp']}] {alert['type']}: {alert['data']}")
    
    # Could send to monitoring system, log to file, etc.
    if alert['type'] == 'high_latency':
        print("Consider scaling up or optimizing model")
    elif alert['type'] == 'high_error_rate':
        print("Check model health and input data quality")

monitor = RealTimePerformanceMonitor(window_size=1000)
monitor.add_alert_callback(alert_handler)
monitor.start_monitoring()

# Simulate predictions with monitoring
import random
for i in range(100):
    latency = random.uniform(10, 200)  # Simulate varying latency
    success = random.random() > 0.02   # 2% error rate
    monitor.record_prediction(latency, success)
    time.sleep(0.1)

# Get current metrics
current_metrics = monitor.get_current_metrics()
print("Current Performance Metrics:")
print(json.dumps(current_metrics, indent=2, default=str))

monitor.stop_monitoring()
```

## Algorithm Optimization

### Optimized Algorithm Implementations

```python
from sklearn.base import BaseEstimator, OutlierMixin
import numpy as np
from numba import jit, prange
from scipy.spatial.distance import cdist
import joblib

class OptimizedIsolationForest(BaseEstimator, OutlierMixin):
    """Optimized Isolation Forest with performance improvements."""
    
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1,
                 max_features=1.0, n_jobs=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """Fit the model with optimized tree construction."""
        X = np.asarray(X, dtype=np.float32)  # Use float32 for memory efficiency
        self.n_features_ = X.shape[1]
        
        # Calculate actual max_samples
        if self.max_samples == 'auto':
            max_samples = min(256, X.shape[0])
        elif isinstance(self.max_samples, int):
            max_samples = self.max_samples
        else:
            max_samples = int(self.max_samples * X.shape[0])
        
        # Calculate max_features
        if isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = int(self.max_features * self.n_features_)
        
        # Build trees in parallel
        self.trees_ = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self._build_tree)(
                X, max_samples, max_features, i
            ) for i in range(self.n_estimators)
        )
        
        # Precompute average path length for normalization
        self.offset_ = self._compute_offset(max_samples)
        
        return self
    
    def _build_tree(self, X, max_samples, max_features, seed):
        """Build a single isolation tree."""
        rng = np.random.RandomState(seed + self.random_state if self.random_state else seed)
        
        # Sample data
        indices = rng.choice(X.shape[0], max_samples, replace=False)
        sample = X[indices]
        
        # Build tree using optimized recursive function
        tree = self._build_tree_recursive(sample, 0, max_features, rng)
        return tree
    
    @staticmethod
    @jit(nopython=True)
    def _build_tree_recursive(data, depth, max_features, rng_state):
        """Optimized recursive tree building with Numba."""
        n_samples, n_features = data.shape
        
        # Stop conditions
        if n_samples <= 1 or depth >= 10:  # Max depth limit
            return {'type': 'leaf', 'size': n_samples}
        
        # Randomly select feature and split point
        feature_idx = rng_state.randint(0, n_features)
        feature_values = data[:, feature_idx]
        
        if np.all(feature_values == feature_values[0]):
            return {'type': 'leaf', 'size': n_samples}
        
        min_val, max_val = np.min(feature_values), np.max(feature_values)
        split_point = rng_state.uniform(min_val, max_val)
        
        # Split data
        left_mask = feature_values < split_point
        
        if np.all(left_mask) or np.all(~left_mask):
            return {'type': 'leaf', 'size': n_samples}
        
        return {
            'type': 'node',
            'feature': feature_idx,
            'split': split_point,
            'left': self._build_tree_recursive(data[left_mask], depth + 1, max_features, rng_state),
            'right': self._build_tree_recursive(data[~left_mask], depth + 1, max_features, rng_state)
        }
    
    def decision_function(self, X):
        """Optimized scoring function."""
        X = np.asarray(X, dtype=np.float32)
        
        # Parallel path length computation
        path_lengths = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self._compute_path_lengths)(X, tree) 
            for tree in self.trees_
        )
        
        # Average path lengths across all trees
        avg_path_lengths = np.mean(path_lengths, axis=0)
        
        # Normalize and return anomaly scores
        scores = 2 ** (-avg_path_lengths / self.offset_)
        return 0.5 - scores  # Convert to decision function format
    
    @staticmethod
    @jit(nopython=True)
    def _compute_path_lengths(X, tree):
        """Compute path lengths for all instances in a tree."""
        n_samples = X.shape[0]
        path_lengths = np.zeros(n_samples)
        
        for i in prange(n_samples):
            path_lengths[i] = _get_path_length(X[i], tree, 0)
        
        return path_lengths
    
    def _compute_offset(self, n):
        """Compute normalization offset."""
        if n <= 1:
            return 1
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

@jit(nopython=True)
def _get_path_length(instance, node, depth):
    """Get path length for single instance."""
    if node['type'] == 'leaf':
        return depth + _average_path_length(node['size'])
    
    if instance[node['feature']] < node['split']:
        return _get_path_length(instance, node['left'], depth + 1)
    else:
        return _get_path_length(instance, node['right'], depth + 1)

@jit(nopython=True)
def _average_path_length(n):
    """Average path length of unsuccessful search in BST."""
    if n <= 1:
        return 1
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

# Optimized Local Outlier Factor
class OptimizedLOF(BaseEstimator, OutlierMixin):
    """Memory and CPU optimized Local Outlier Factor."""
    
    def __init__(self, n_neighbors=20, contamination=0.1, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.n_jobs = n_jobs
    
    def fit(self, X, y=None):
        """Fit the model with optimized neighbor computation."""
        self.X_train_ = np.asarray(X, dtype=np.float32)
        
        # Precompute distance matrix in chunks to manage memory
        self.distances_, self.neighbors_ = self._compute_neighbors_chunked(self.X_train_)
        
        # Compute LRD for training data
        self.lrd_train_ = self._compute_lrd_batch(self.X_train_, self.distances_, self.neighbors_)
        
        return self
    
    def _compute_neighbors_chunked(self, X, chunk_size=1000):
        """Compute neighbors in chunks to manage memory."""
        n_samples = X.shape[0]
        all_distances = []
        all_neighbors = []
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = X[start_idx:end_idx]
            
            # Compute distances for this chunk
            distances = cdist(chunk, X, metric='euclidean')
            
            # Find k+1 nearest neighbors (including self)
            neighbor_indices = np.argpartition(distances, self.n_neighbors + 1, axis=1)
            neighbor_indices = neighbor_indices[:, :self.n_neighbors + 1]
            
            # Get actual distances
            chunk_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
            
            all_distances.append(chunk_distances)
            all_neighbors.append(neighbor_indices)
        
        return np.vstack(all_distances), np.vstack(all_neighbors)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_lrd_batch(X, distances, neighbors):
        """Compute Local Reachability Density in batch."""
        n_samples = X.shape[0]
        lrd = np.zeros(n_samples)
        
        for i in prange(n_samples):
            # Get k-distance (distance to k-th neighbor, excluding self)
            k_distance = distances[i, -1]  # Last distance is k-th neighbor
            
            # Compute reachability distances
            reach_distances = np.maximum(distances[i, 1:], k_distance)  # Exclude self (index 0)
            
            # LRD is inverse of average reachability distance
            avg_reach_dist = np.mean(reach_distances)
            lrd[i] = 1.0 / (avg_reach_dist + 1e-10)
        
        return lrd
    
    def decision_function(self, X):
        """Optimized LOF computation."""
        X = np.asarray(X, dtype=np.float32)
        
        if np.array_equal(X, self.X_train_):
            # Use precomputed values for training data
            return self._compute_lof_scores(self.lrd_train_, self.neighbors_, self.lrd_train_)
        else:
            # Compute for new data
            test_distances, test_neighbors = self._compute_neighbors_for_test(X)
            test_lrd = self._compute_lrd_for_test(X, test_distances, test_neighbors)
            return self._compute_lof_scores(test_lrd, test_neighbors, self.lrd_train_)
    
    def _compute_neighbors_for_test(self, X_test):
        """Compute neighbors for test data."""
        distances = cdist(X_test, self.X_train_, metric='euclidean')
        neighbor_indices = np.argpartition(distances, self.n_neighbors, axis=1)
        neighbor_indices = neighbor_indices[:, :self.n_neighbors]
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        return neighbor_distances, neighbor_indices
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_lrd_for_test(X_test, distances, neighbors):
        """Compute LRD for test data."""
        n_samples = distances.shape[0]
        lrd = np.zeros(n_samples)
        
        for i in prange(n_samples):
            k_distance = distances[i, -1]
            reach_distances = np.maximum(distances[i], k_distance)
            avg_reach_dist = np.mean(reach_distances)
            lrd[i] = 1.0 / (avg_reach_dist + 1e-10)
        
        return lrd
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_lof_scores(lrd_query, neighbors, lrd_train):
        """Compute LOF scores."""
        n_samples = len(lrd_query)
        lof_scores = np.zeros(n_samples)
        
        for i in prange(n_samples):
            neighbor_lrds = lrd_train[neighbors[i]]
            lof_scores[i] = np.mean(neighbor_lrds) / (lrd_query[i] + 1e-10)
        
        # Convert to decision function format (higher values = more normal)
        return 1.0 - (lof_scores - 1.0)

# Example usage and benchmarking
def benchmark_optimized_algorithms():
    """Benchmark optimized vs standard algorithms."""
    # Generate test data
    X = np.random.randn(5000, 10)
    
    # Standard algorithms
    from sklearn.ensemble import IsolationForest
    standard_if = IsolationForest(n_estimators=100, random_state=42)
    
    # Optimized algorithms
    optimized_if = OptimizedIsolationForest(n_estimators=100, random_state=42)
    
    # Benchmark
    profiler = PerformanceProfiler()
    
    algorithms = {
        'Standard_IF': standard_if,
        'Optimized_IF': optimized_if
    }
    
    results = profiler.benchmark_algorithms(algorithms, X, n_runs=3)
    
    print("Optimization Results:")
    summary = results.groupby('algorithm').agg({
        'train_time_ms': 'mean',
        'pred_time_ms': 'mean',
        'throughput_instances_per_sec': 'mean',
        'memory_mb': 'mean'
    }).round(2)
    
    print(summary)
    
    # Calculate improvement
    standard_stats = summary.loc['Standard_IF']
    optimized_stats = summary.loc['Optimized_IF']
    
    speed_improvement = (standard_stats['pred_time_ms'] / optimized_stats['pred_time_ms'])
    memory_improvement = (standard_stats['memory_mb'] / optimized_stats['memory_mb'])
    
    print(f"\nOptimization Results:")
    print(f"Speed improvement: {speed_improvement:.2f}x")
    print(f"Memory improvement: {memory_improvement:.2f}x")

# Run benchmark
benchmark_optimized_algorithms()
```

## Memory Management

### Memory-Efficient Data Processing

```python
import gc
import numpy as np
import pandas as pd
from typing import Iterator, Tuple
import h5py
import zarr

class MemoryEfficientProcessor:
    """Process large datasets with minimal memory footprint."""
    
    def __init__(self, chunk_size: int = 10000, dtype: str = 'float32'):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.temp_files = []
    
    def fit_transform_chunked(self, data_source, model, transform_func=None) -> Iterator[np.ndarray]:
        """Process data in chunks to manage memory."""
        
        if isinstance(data_source, str):
            # File-based processing
            data_iterator = self._read_file_chunks(data_source)
        elif hasattr(data_source, '__iter__'):
            # Iterator-based processing
            data_iterator = self._chunk_iterator(data_source)
        else:
            # Array-based processing
            data_iterator = self._array_chunks(data_source)
        
        for chunk_idx, chunk in enumerate(data_iterator):
            # Apply preprocessing if provided
            if transform_func:
                chunk = transform_func(chunk)
            
            # Ensure correct dtype for memory efficiency
            chunk = chunk.astype(self.dtype)
            
            # Fit model on first chunk, transform on all
            if chunk_idx == 0:
                model.fit(chunk)
            
            # Transform chunk
            scores = model.decision_function(chunk)
            
            # Force garbage collection after each chunk
            del chunk
            gc.collect()
            
            yield scores
    
    def _read_file_chunks(self, filepath: str) -> Iterator[np.ndarray]:
        """Read file in chunks based on format."""
        if filepath.endswith('.csv'):
            chunk_iter = pd.read_csv(filepath, chunksize=self.chunk_size)
            for chunk_df in chunk_iter:
                yield chunk_df.values.astype(self.dtype)
        
        elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
            with h5py.File(filepath, 'r') as f:
                dataset = f['data']  # Assuming dataset named 'data'
                for i in range(0, len(dataset), self.chunk_size):
                    chunk = dataset[i:i + self.chunk_size]
                    yield chunk.astype(self.dtype)
        
        elif filepath.endswith('.parquet'):
            # Use pyarrow for efficient parquet reading
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(filepath)
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                yield batch.to_pandas().values.astype(self.dtype)
    
    def _chunk_iterator(self, iterator) -> Iterator[np.ndarray]:
        """Convert iterator to chunks."""
        chunk = []
        for item in iterator:
            chunk.append(item)
            if len(chunk) >= self.chunk_size:
                yield np.array(chunk, dtype=self.dtype)
                chunk = []
        
        if chunk:
            yield np.array(chunk, dtype=self.dtype)
    
    def _array_chunks(self, array: np.ndarray) -> Iterator[np.ndarray]:
        """Split array into chunks."""
        for i in range(0, len(array), self.chunk_size):
            yield array[i:i + self.chunk_size].astype(self.dtype)
    
    def process_large_dataset(self, data_source, model, output_path: str = None):
        """Process large dataset and optionally save results."""
        all_scores = []
        memory_usage = []
        
        process = psutil.Process()
        
        for chunk_scores in self.fit_transform_chunked(data_source, model):
            all_scores.append(chunk_scores)
            
            # Track memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(current_memory)
            
            print(f"Processed chunk, current memory: {current_memory:.1f} MB")
        
        # Combine results
        final_scores = np.concatenate(all_scores)
        
        if output_path:
            # Save to memory-mapped file for large results
            np.save(output_path, final_scores)
        
        return {
            'scores': final_scores,
            'peak_memory_mb': max(memory_usage),
            'avg_memory_mb': np.mean(memory_usage)
        }

class MemoryOptimizedDataLoader:
    """Optimized data loader with caching and memory management."""
    
    def __init__(self, cache_size_gb: float = 1.0):
        self.cache_size_bytes = int(cache_size_gb * 1024**3)
        self.cache = {}
        self.cache_order = []  # LRU tracking
        self.current_cache_size = 0
    
    def load_data(self, data_path: str, use_cache: bool = True) -> np.ndarray:
        """Load data with caching."""
        cache_key = f"{data_path}_{os.path.getmtime(data_path)}"
        
        if use_cache and cache_key in self.cache:
            # Move to end of LRU list
            self.cache_order.remove(cache_key)
            self.cache_order.append(cache_key)
            return self.cache[cache_key]
        
        # Load data
        if data_path.endswith('.npy'):
            data = np.load(data_path, mmap_mode='r')  # Memory-mapped loading
        elif data_path.endswith('.csv'):
            data = pd.read_csv(data_path).values.astype('float32')
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path).values.astype('float32')
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Add to cache if it fits
        data_size = data.nbytes
        if use_cache and data_size <= self.cache_size_bytes:
            self._add_to_cache(cache_key, data, data_size)
        
        return data
    
    def _add_to_cache(self, key: str, data: np.ndarray, size: int):
        """Add data to cache with LRU eviction."""
        # Evict items if necessary
        while self.current_cache_size + size > self.cache_size_bytes and self.cache_order:
            oldest_key = self.cache_order.pop(0)
            oldest_data = self.cache.pop(oldest_key)
            self.current_cache_size -= oldest_data.nbytes
            del oldest_data
        
        # Add new item
        self.cache[key] = data
        self.cache_order.append(key)
        self.current_cache_size += size
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.cache_order.clear()
        self.current_cache_size = 0
        gc.collect()

# Example usage
def demonstrate_memory_optimization():
    """Demonstrate memory-efficient processing."""
    
    # Create large synthetic dataset
    print("Creating large synthetic dataset...")
    large_data = np.random.randn(100000, 50).astype('float32')
    np.save('large_dataset.npy', large_data)
    
    # Memory-efficient processing
    processor = MemoryEfficientProcessor(chunk_size=5000)
    model = IsolationForest(n_estimators=50, random_state=42)
    
    print("Processing with memory optimization...")
    results = processor.process_large_dataset(
        'large_dataset.npy', 
        model, 
        output_path='anomaly_scores.npy'
    )
    
    print(f"Processing completed:")
    print(f"  Peak memory usage: {results['peak_memory_mb']:.1f} MB")
    print(f"  Average memory usage: {results['avg_memory_mb']:.1f} MB")
    print(f"  Total anomalies detected: {np.sum(results['scores'] < 0)}")
    
    # Compare with naive approach
    print("\nComparing with naive approach...")
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    
    # Load all data at once
    all_data = np.load('large_dataset.npy')
    model_naive = IsolationForest(n_estimators=50, random_state=42)
    scores_naive = model_naive.fit(all_data).decision_function(all_data)
    
    peak_memory = process.memory_info().rss / 1024 / 1024
    
    print(f"Naive approach peak memory: {peak_memory:.1f} MB")
    print(f"Memory savings: {peak_memory / results['peak_memory_mb']:.2f}x")
    
    # Cleanup
    os.remove('large_dataset.npy')
    os.remove('anomaly_scores.npy')

demonstrate_memory_optimization()
```

## Parallel Processing

### Multi-threaded and Multi-process Optimization

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from queue import Queue
import asyncio
from typing import List, Callable, Any

class ParallelAnomalyDetector:
    """Parallel anomaly detection with multiple processing strategies."""
    
    def __init__(self, n_jobs: int = None, backend: str = 'threading'):
        self.n_jobs = n_jobs or mp.cpu_count()
        self.backend = backend  # 'threading', 'multiprocessing', 'asyncio'
    
    def parallel_fit_predict(self, models: List, datasets: List) -> List:
        """Fit multiple models on different datasets in parallel."""
        
        if self.backend == 'threading':
            return self._threading_fit_predict(models, datasets)
        elif self.backend == 'multiprocessing':
            return self._multiprocessing_fit_predict(models, datasets)
        elif self.backend == 'asyncio':
            return asyncio.run(self._asyncio_fit_predict(models, datasets))
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _threading_fit_predict(self, models: List, datasets: List) -> List:
        """Threading-based parallel processing."""
        results = [None] * len(models)
        
        def fit_predict_worker(idx, model, data):
            model.fit(data)
            scores = model.decision_function(data)
            results[idx] = {
                'model_idx': idx,
                'scores': scores,
                'model': model
            }
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(fit_predict_worker, i, model, data)
                for i, (model, data) in enumerate(zip(models, datasets))
            ]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions
        
        return results
    
    def _multiprocessing_fit_predict(self, models: List, datasets: List) -> List:
        """Multiprocessing-based parallel processing."""
        
        # Worker function for multiprocessing
        def worker(args):
            idx, model, data = args
            model.fit(data)
            scores = model.decision_function(data)
            return {
                'model_idx': idx,
                'scores': scores,
                'model': model
            }
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(worker, (i, model, data))
                for i, (model, data) in enumerate(zip(models, datasets))
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        # Sort results by model index
        results.sort(key=lambda x: x['model_idx'])
        return results
    
    async def _asyncio_fit_predict(self, models: List, datasets: List) -> List:
        """AsyncIO-based parallel processing."""
        
        async def fit_predict_async(idx, model, data):
            # Run CPU-bound task in thread pool
            loop = asyncio.get_event_loop()
            
            def cpu_task():
                model.fit(data)
                return model.decision_function(data)
            
            scores = await loop.run_in_executor(None, cpu_task)
            return {
                'model_idx': idx,
                'scores': scores,
                'model': model
            }
        
        tasks = [
            fit_predict_async(i, model, data)
            for i, (model, data) in enumerate(zip(models, datasets))
        ]
        
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda x: x['model_idx'])
    
    def parallel_batch_prediction(self, model, data_batches: List[np.ndarray]) -> np.ndarray:
        """Predict on multiple batches in parallel."""
        
        def predict_batch(batch):
            return model.decision_function(batch)
        
        if self.backend == 'threading':
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(predict_batch, batch) for batch in data_batches]
                results = [future.result() for future in futures]
        
        elif self.backend == 'multiprocessing':
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(predict_batch, data_batches))
        
        return np.concatenate(results)

class StreamingParallelProcessor:
    """Process streaming data with parallel workers."""
    
    def __init__(self, model, n_workers: int = 4, buffer_size: int = 1000):
        self.model = model
        self.n_workers = n_workers
        self.buffer_size = buffer_size
        
        # Threading components
        self.input_queue = Queue(maxsize=buffer_size)
        self.output_queue = Queue(maxsize=buffer_size)
        self.workers = []
        self.running = False
    
    def start_workers(self):
        """Start worker threads."""
        self.running = True
        
        for i in range(self.n_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop_workers(self):
        """Stop worker threads."""
        self.running = False
        
        # Add sentinel values to wake up workers
        for _ in range(self.n_workers):
            self.input_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()
        
        self.workers.clear()
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        while self.running:
            try:
                item = self.input_queue.get(timeout=1.0)
                
                if item is None:  # Sentinel value
                    break
                
                batch_id, data = item
                
                # Process data
                start_time = time.time()
                scores = self.model.decision_function(data)
                processing_time = time.time() - start_time
                
                # Put result in output queue
                result = {
                    'batch_id': batch_id,
                    'scores': scores,
                    'worker_id': worker_id,
                    'processing_time': processing_time
                }
                
                self.output_queue.put(result)
                
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                continue
    
    def process_stream(self, data_stream):
        """Process streaming data."""
        self.start_workers()
        
        try:
            batch_id = 0
            for data_batch in data_stream:
                # Add to input queue
                self.input_queue.put((batch_id, data_batch))
                batch_id += 1
                
                # Yield results as they become available
                while not self.output_queue.empty():
                    yield self.output_queue.get()
            
            # Process remaining results
            while not self.output_queue.empty():
                yield self.output_queue.get()
                
        finally:
            self.stop_workers()

# GPU-Accelerated Processing (if available)
class GPUAcceleratedProcessor:
    """GPU-accelerated anomaly detection using CuPy."""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        
        if self.gpu_available:
            import cupy as cp
            self.cp = cp
        else:
            print("GPU not available, falling back to CPU")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU and CuPy are available."""
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return True
        except:
            return False
    
    def gpu_distance_matrix(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """Compute distance matrix on GPU."""
        if not self.gpu_available:
            from scipy.spatial.distance import cdist
            return cdist(X, Y if Y is not None else X)
        
        # Transfer to GPU
        X_gpu = self.cp.asarray(X)
        Y_gpu = self.cp.asarray(Y if Y is not None else X)
        
        # Compute distances using broadcasting
        X_expanded = X_gpu[:, None, :]  # (n, 1, d)
        Y_expanded = Y_gpu[None, :, :]  # (1, m, d)
        
        distances = self.cp.sqrt(self.cp.sum((X_expanded - Y_expanded) ** 2, axis=2))
        
        # Transfer back to CPU
        return self.cp.asnumpy(distances)
    
    def gpu_isolation_forest_score(self, X: np.ndarray, trees: List) -> np.ndarray:
        """Compute Isolation Forest scores on GPU."""
        if not self.gpu_available:
            # Fallback to CPU implementation
            return self._cpu_isolation_forest_score(X, trees)
        
        X_gpu = self.cp.asarray(X)
        n_samples = X_gpu.shape[0]
        n_trees = len(trees)
        
        # Parallel path length computation on GPU
        path_lengths = self.cp.zeros((n_samples, n_trees))
        
        for tree_idx, tree in enumerate(trees):
            # Vectorized tree traversal (simplified implementation)
            lengths = self._gpu_tree_path_lengths(X_gpu, tree)
            path_lengths[:, tree_idx] = lengths
        
        # Average path lengths
        avg_lengths = self.cp.mean(path_lengths, axis=1)
        
        # Convert to anomaly scores
        scores = 2 ** (-avg_lengths / self._compute_offset(256))  # Assuming max_samples=256
        
        return self.cp.asnumpy(scores)
    
    def _gpu_tree_path_lengths(self, X_gpu, tree):
        """Compute path lengths for all samples in a tree on GPU."""
        n_samples = X_gpu.shape[0]
        path_lengths = self.cp.zeros(n_samples)
        
        # This is a simplified implementation
        # In practice, you'd implement efficient tree traversal on GPU
        for i in range(n_samples):
            path_lengths[i] = self._gpu_single_path_length(X_gpu[i], tree, 0)
        
        return path_lengths
    
    def _gpu_single_path_length(self, instance, node, depth):
        """Get path length for single instance (GPU version)."""
        # This would be implemented as a GPU kernel for better performance
        if node['type'] == 'leaf':
            return depth + self._average_path_length(node['size'])
        
        if instance[node['feature']] < node['split']:
            return self._gpu_single_path_length(instance, node['left'], depth + 1)
        else:
            return self._gpu_single_path_length(instance, node['right'], depth + 1)

# Example usage and benchmarking
def benchmark_parallel_processing():
    """Benchmark different parallel processing approaches."""
    
    # Generate test data
    n_datasets = 4
    datasets = [np.random.randn(5000, 10) for _ in range(n_datasets)]
    models = [IsolationForest(n_estimators=50, random_state=i) for i in range(n_datasets)]
    
    backends = ['threading', 'multiprocessing']
    results = {}
    
    for backend in backends:
        print(f"\nBenchmarking {backend} backend...")
        processor = ParallelAnomalyDetector(n_jobs=4, backend=backend)
        
        start_time = time.time()
        parallel_results = processor.parallel_fit_predict(models, datasets)
        end_time = time.time()
        
        results[backend] = {
            'time': end_time - start_time,
            'results': parallel_results
        }
        
        print(f"{backend} processing time: {results[backend]['time']:.2f} seconds")
    
    # Compare with sequential processing
    print("\nSequential processing...")
    start_time = time.time()
    sequential_results = []
    for i, (model, data) in enumerate(zip(models, datasets)):
        model.fit(data)
        scores = model.decision_function(data)
        sequential_results.append({
            'model_idx': i,
            'scores': scores,
            'model': model
        })
    sequential_time = time.time() - start_time
    
    print(f"Sequential processing time: {sequential_time:.2f} seconds")
    
    # Calculate speedups
    for backend in backends:
        speedup = sequential_time / results[backend]['time']
        print(f"{backend} speedup: {speedup:.2f}x")

# Run benchmark
benchmark_parallel_processing()
```

This comprehensive performance guide provides the tools and techniques needed to optimize anomaly detection systems for production environments. The combination of profiling, algorithm optimization, memory management, and parallel processing ensures optimal performance across different scales and deployment scenarios.
