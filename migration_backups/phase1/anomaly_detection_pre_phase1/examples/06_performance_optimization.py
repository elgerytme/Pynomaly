#!/usr/bin/env python3
"""
Performance Optimization Examples for Anomaly Detection Package

This example demonstrates performance optimization techniques including:
- Benchmarking different algorithms and configurations
- Memory optimization and efficient data handling
- Parallel processing with multiprocessing and asyncio
- GPU acceleration for deep learning models
- Large dataset handling strategies
- Performance profiling and bottleneck identification
"""

import asyncio
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Memory profiling
try:
    from memory_profiler import profile, memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Warning: memory_profiler not available. Install with: pip install memory-profiler")
    
    # Create dummy decorator if not available
    def profile(func):
        return func

# Performance profiling
try:
    import cProfile
    import pstats
    from pstats import SortKey
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    print("Warning: cProfile not available for performance profiling.")

# GPU acceleration
try:
    import torch
    PYTORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"CUDA available with {torch.cuda.device_count()} device(s)")
    else:
        print("CUDA not available - using CPU only")
except ImportError:
    PYTORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    print("Warning: PyTorch not available. GPU acceleration examples will be skipped.")

# Numba for JIT compilation
try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available. Install with: pip install numba")

# PyOD for algorithm comparisons
try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA as PyOD_PCA
    from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    print("Warning: PyOD not available. Install with: pip install pyod")

# Import anomaly detection components
try:
    from anomaly_detection import DetectionService, EnsembleService
    from anomaly_detection.domain.entities.detection_result import DetectionResult
except ImportError:
    print("Please install the anomaly_detection package first:")
    print("pip install -e .")
    exit(1)


class PerformanceBenchmark:
    """Performance benchmarking and optimization utilities."""
    
    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        return {
            'cpu_count': cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': psutil.__version__,
            'cuda_available': CUDA_AVAILABLE,
            'gpu_count': torch.cuda.device_count() if CUDA_AVAILABLE else 0
        }
    
    def time_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Time a function execution."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
    def measure_memory_usage(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Measure memory usage of a function."""
        if not MEMORY_PROFILER_AVAILABLE:
            result = func(*args, **kwargs)
            return result, 0.0
            
        def wrapper():
            return func(*args, **kwargs)
        
        mem_usage = memory_usage(wrapper, interval=0.1, timeout=None)
        result = wrapper()
        peak_memory = max(mem_usage) - min(mem_usage)
        
        return result, peak_memory
    
    @profile
    def profile_function(self, func: Callable, *args, **kwargs):
        """Profile a function using memory_profiler."""
        return func(*args, **kwargs)
    
    def benchmark_algorithm(self, 
                          algorithm_name: str,
                          algorithm_func: Callable,
                          data_sizes: List[int],
                          n_features: int = 10,
                          n_runs: int = 3) -> Dict[str, List[float]]:
        """Benchmark an algorithm across different data sizes."""
        
        results = {
            'data_sizes': [],
            'mean_times': [],
            'std_times': [],
            'memory_usage': [],
            'throughput': []
        }
        
        print(f"\nBenchmarking {algorithm_name}...")
        
        for size in data_sizes:
            print(f"  Testing with {size} samples...")
            
            # Generate test data
            X = np.random.randn(size, n_features)
            
            # Run multiple times for statistical significance
            times = []
            memory_usages = []
            
            for run in range(n_runs):
                # Measure time and memory
                _, exec_time = self.time_function(algorithm_func, X)
                _, memory_usage = self.measure_memory_usage(algorithm_func, X)
                
                times.append(exec_time)
                memory_usages.append(memory_usage)
            
            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)
            mean_memory = np.mean(memory_usages)
            throughput = size / mean_time  # samples per second
            
            results['data_sizes'].append(size)
            results['mean_times'].append(mean_time)
            results['std_times'].append(std_time)
            results['memory_usage'].append(mean_memory)
            results['throughput'].append(throughput)
            
            print(f"    Time: {mean_time:.3f}±{std_time:.3f}s, "
                  f"Memory: {mean_memory:.1f}MB, "
                  f"Throughput: {throughput:.0f} samples/s")
        
        return results
    
    def compare_algorithms(self, 
                          algorithms: Dict[str, Callable],
                          data_sizes: List[int],
                          n_features: int = 10) -> Dict[str, Dict]:
        """Compare multiple algorithms across data sizes."""
        
        comparison_results = {}
        
        for name, algo_func in algorithms.items():
            comparison_results[name] = self.benchmark_algorithm(
                name, algo_func, data_sizes, n_features
            )
        
        return comparison_results
    
    def plot_performance_comparison(self, results: Dict[str, Dict], 
                                   title: str = "Algorithm Performance Comparison"):
        """Plot performance comparison results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Execution time comparison
        ax1 = axes[0, 0]
        for name, result in results.items():
            ax1.plot(result['data_sizes'], result['mean_times'], 
                    'o-', label=name, linewidth=2, markersize=6)
            ax1.fill_between(result['data_sizes'], 
                           np.array(result['mean_times']) - np.array(result['std_times']),
                           np.array(result['mean_times']) + np.array(result['std_times']),
                           alpha=0.2)
        
        ax1.set_xlabel('Data Size (samples)')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Data Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Memory usage comparison
        ax2 = axes[0, 1]
        for name, result in results.items():
            ax2.plot(result['data_sizes'], result['memory_usage'], 
                    's-', label=name, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Data Size (samples)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Data Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Throughput comparison
        ax3 = axes[1, 0]
        for name, result in results.items():
            ax3.plot(result['data_sizes'], result['throughput'], 
                    '^-', label=name, linewidth=2, markersize=6)
        
        ax3.set_xlabel('Data Size (samples)')
        ax3.set_ylabel('Throughput (samples/second)')
        ax3.set_title('Throughput vs Data Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Performance efficiency (throughput/memory)
        ax4 = axes[1, 1]
        for name, result in results.items():
            efficiency = np.array(result['throughput']) / (np.array(result['memory_usage']) + 1)
            ax4.plot(result['data_sizes'], efficiency, 
                    'd-', label=name, linewidth=2, markersize=6)
        
        ax4.set_xlabel('Data Size (samples)')
        ax4.set_ylabel('Efficiency (throughput/MB)')
        ax4.set_title('Performance Efficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


class ParallelProcessingOptimizer:
    """Parallel processing optimization utilities."""
    
    def __init__(self):
        self.n_cores = cpu_count()
        print(f"Available CPU cores: {self.n_cores}")
    
    def parallel_detection_threading(self, 
                                   data_batches: List[np.ndarray],
                                   algorithm: str = 'iforest',
                                   max_workers: Optional[int] = None) -> List[DetectionResult]:
        """Parallel anomaly detection using threading."""
        
        if max_workers is None:
            max_workers = min(len(data_batches), self.n_cores)
        
        def detect_batch(batch):
            service = DetectionService()
            return service.detect_anomalies(batch, algorithm=algorithm)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(detect_batch, data_batches))
        
        return results
    
    def parallel_detection_multiprocessing(self,
                                         data_batches: List[np.ndarray],
                                         algorithm: str = 'iforest',
                                         max_workers: Optional[int] = None) -> List[DetectionResult]:
        """Parallel anomaly detection using multiprocessing."""
        
        if max_workers is None:
            max_workers = min(len(data_batches), self.n_cores)
        
        def detect_batch(batch):
            service = DetectionService()
            return service.detect_anomalies(batch, algorithm=algorithm)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(detect_batch, data_batches))
        
        return results
    
    async def async_detection(self,
                            data_batches: List[np.ndarray],
                            algorithm: str = 'iforest',
                            batch_size: int = 4) -> List[DetectionResult]:
        """Asynchronous anomaly detection."""
        
        async def detect_batch_async(batch):
            # Simulate async operation (in real scenario, this might be I/O bound)
            loop = asyncio.get_event_loop()
            service = DetectionService()
            
            # Run in thread pool to avoid blocking
            result = await loop.run_in_executor(
                None, 
                lambda: service.detect_anomalies(batch, algorithm=algorithm)
            )
            return result
        
        # Process in batches to control concurrency
        results = []
        for i in range(0, len(data_batches), batch_size):
            batch_group = data_batches[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                detect_batch_async(batch) for batch in batch_group
            ])
            results.extend(batch_results)
        
        return results
    
    def benchmark_parallel_methods(self,
                                 data: np.ndarray,
                                 n_batches: int = 8,
                                 algorithm: str = 'iforest') -> Dict[str, float]:
        """Benchmark different parallel processing methods."""
        
        # Split data into batches
        batch_size = len(data) // n_batches
        data_batches = [
            data[i:i + batch_size] 
            for i in range(0, len(data), batch_size)
        ][:n_batches]
        
        print(f"Benchmarking with {n_batches} batches, {batch_size} samples each")
        
        results = {}
        
        # Sequential processing
        print("Testing sequential processing...")
        start_time = time.time()
        service = DetectionService()
        sequential_results = [
            service.detect_anomalies(batch, algorithm=algorithm) 
            for batch in data_batches
        ]
        results['sequential'] = time.time() - start_time
        
        # Threading
        print("Testing threading...")
        start_time = time.time()
        threading_results = self.parallel_detection_threading(data_batches, algorithm)
        results['threading'] = time.time() - start_time
        
        # Multiprocessing
        print("Testing multiprocessing...")
        start_time = time.time()
        multiprocessing_results = self.parallel_detection_multiprocessing(data_batches, algorithm)
        results['multiprocessing'] = time.time() - start_time
        
        # Async processing
        print("Testing async processing...")
        start_time = time.time()
        async_results = asyncio.run(self.async_detection(data_batches, algorithm))
        results['async'] = time.time() - start_time
        
        # Print results
        print("\nParallel Processing Benchmark Results:")
        print("-" * 50)
        for method, exec_time in results.items():
            speedup = results['sequential'] / exec_time
            print(f"{method:15}: {exec_time:.3f}s (speedup: {speedup:.2f}x)")
        
        return results


class GPUAccelerationOptimizer:
    """GPU acceleration optimization utilities."""
    
    def __init__(self):
        self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        self.available = CUDA_AVAILABLE
        
        if self.available:
            print(f"GPU acceleration available on {self.device}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("GPU acceleration not available - using CPU")
    
    def create_gpu_data(self, X: np.ndarray) -> torch.Tensor:
        """Convert numpy array to GPU tensor."""
        return torch.FloatTensor(X).to(self.device)
    
    @profile
    def gpu_distance_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix on GPU."""
        if not self.available:
            return self.cpu_distance_matrix(X.cpu().numpy())
        
        # Efficient GPU computation of pairwise distances
        x_norm = (X ** 2).sum(1).view(-1, 1)
        y_norm = x_norm.view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(X, torch.transpose(X, 0, 1))
        return torch.sqrt(torch.clamp(dist, 0.0, float('inf')))
    
    def cpu_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix on CPU."""
        from sklearn.metrics.pairwise import euclidean_distances
        return euclidean_distances(X)
    
    def benchmark_gpu_vs_cpu(self, data_sizes: List[int]) -> Dict[str, List[float]]:
        """Benchmark GPU vs CPU performance."""
        
        results = {
            'data_sizes': data_sizes,
            'gpu_times': [],
            'cpu_times': [],
            'speedups': []
        }
        
        print("Benchmarking GPU vs CPU performance...")
        
        for size in data_sizes:
            print(f"  Testing with {size} samples...")
            
            # Generate data
            X_cpu = np.random.randn(size, 50)
            
            # CPU benchmark
            start_time = time.time()
            cpu_result = self.cpu_distance_matrix(X_cpu)
            cpu_time = time.time() - start_time
            
            if self.available:
                # GPU benchmark
                X_gpu = self.create_gpu_data(X_cpu)
                
                # Warm up GPU
                _ = self.gpu_distance_matrix(X_gpu)
                torch.cuda.synchronize()
                
                start_time = time.time()
                gpu_result = self.gpu_distance_matrix(X_gpu)
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                
                speedup = cpu_time / gpu_time
            else:
                gpu_time = float('inf')
                speedup = 0
            
            results['gpu_times'].append(gpu_time)
            results['cpu_times'].append(cpu_time)
            results['speedups'].append(speedup)
            
            print(f"    CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, "
                  f"Speedup: {speedup:.2f}x")
        
        return results
    
    def plot_gpu_benchmark(self, results: Dict[str, List[float]]):
        """Plot GPU vs CPU benchmark results."""
        
        if not self.available:
            print("GPU not available - skipping GPU benchmark plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Execution time comparison
        ax1.plot(results['data_sizes'], results['cpu_times'], 
                'o-', label='CPU', linewidth=2, markersize=6)
        ax1.plot(results['data_sizes'], results['gpu_times'], 
                's-', label='GPU', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Data Size (samples)')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('GPU vs CPU Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Speedup
        ax2.plot(results['data_sizes'], results['speedups'], 
                '^-', color='green', linewidth=2, markersize=6)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        
        ax2.set_xlabel('Data Size (samples)')
        ax2.set_ylabel('Speedup (CPU time / GPU time)')
        ax2.set_title('GPU Speedup vs Data Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.show()


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self):
        self.initial_memory = psutil.virtual_memory().used / (1024**3)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return psutil.virtual_memory().used / (1024**3)
    
    def memory_efficient_chunked_processing(self,
                                          data: np.ndarray,
                                          chunk_size: int = 1000,
                                          algorithm: str = 'iforest') -> DetectionResult:
        """Process large datasets in memory-efficient chunks."""
        
        print(f"Processing {len(data)} samples in chunks of {chunk_size}")
        start_memory = self.get_memory_usage()
        
        service = DetectionService()
        all_predictions = []
        all_scores = []
        
        # Process in chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            
            # Process chunk
            result = service.detect_anomalies(chunk, algorithm=algorithm)
            
            all_predictions.extend(result.predictions)
            all_scores.extend(result.anomaly_scores)
            
            # Print progress
            if (i // chunk_size + 1) % 10 == 0:
                current_memory = self.get_memory_usage()
                print(f"  Processed {i + len(chunk)} samples, "
                      f"Memory: {current_memory - start_memory:.2f} GB")
        
        # Combine results
        combined_predictions = np.array(all_predictions)
        combined_scores = np.array(all_scores)
        
        # Create combined result
        anomaly_count = np.sum(combined_predictions == -1)
        anomaly_rate = anomaly_count / len(combined_predictions)
        
        # Create a mock DetectionResult (in real implementation, this would be proper)
        class CombinedResult:
            def __init__(self, predictions, scores, count, rate):
                self.predictions = predictions
                self.anomaly_scores = scores
                self.anomaly_count = count
                self.anomaly_rate = rate
        
        return CombinedResult(combined_predictions, combined_scores, anomaly_count, anomaly_rate)
    
    def compare_memory_strategies(self,
                                data_size: int = 10000,
                                n_features: int = 50) -> Dict[str, Dict[str, float]]:
        """Compare different memory optimization strategies."""
        
        print(f"Comparing memory strategies with {data_size} samples, {n_features} features")
        
        # Generate large dataset
        X = np.random.randn(data_size, n_features)
        
        results = {}
        
        # Strategy 1: Load all data at once
        print("\n1. Loading all data at once...")
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        service = DetectionService()
        result_all = service.detect_anomalies(X, algorithm='iforest')
        
        end_time = time.time()
        peak_memory = self.get_memory_usage()
        
        results['all_at_once'] = {
            'time': end_time - start_time,
            'peak_memory': peak_memory - start_memory,
            'accuracy': np.mean(result_all.predictions == -1)
        }
        
        # Strategy 2: Chunked processing
        print("\n2. Chunked processing...")
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        result_chunked = self.memory_efficient_chunked_processing(
            X, chunk_size=1000, algorithm='iforest'
        )
        
        end_time = time.time()
        peak_memory = self.get_memory_usage()
        
        results['chunked'] = {
            'time': end_time - start_time,
            'peak_memory': peak_memory - start_memory,
            'accuracy': np.mean(result_chunked.predictions == -1)
        }
        
        # Strategy 3: Data streaming (simulated)
        print("\n3. Streaming processing...")
        start_memory = self.get_memory_usage()
        start_time = time.time()
        
        # Simulate streaming by processing very small chunks
        result_streaming = self.memory_efficient_chunked_processing(
            X, chunk_size=100, algorithm='iforest'
        )
        
        end_time = time.time()
        peak_memory = self.get_memory_usage()
        
        results['streaming'] = {
            'time': end_time - start_time,
            'peak_memory': peak_memory - start_memory,
            'accuracy': np.mean(result_streaming.predictions == -1)
        }
        
        # Print comparison
        print("\nMemory Strategy Comparison:")
        print("-" * 60)
        print(f"{'Strategy':<15} {'Time (s)':<10} {'Memory (GB)':<12} {'Accuracy':<10}")
        print("-" * 60)
        
        for strategy, metrics in results.items():
            print(f"{strategy:<15} {metrics['time']:<10.2f} "
                  f"{metrics['peak_memory']:<12.2f} {metrics['accuracy']:<10.3f}")
        
        return results


def example_1_algorithm_benchmarking():
    """Example 1: Comprehensive algorithm benchmarking."""
    print("\n" + "="*60)
    print("Example 1: Algorithm Benchmarking")
    print("="*60)
    
    if not PYOD_AVAILABLE:
        print("PyOD not available. Skipping algorithm benchmarking.")
        return
    
    benchmark = PerformanceBenchmark()
    
    # Define algorithms to benchmark
    def create_iforest():
        def iforest_func(X):
            model = IForest(contamination=0.1, random_state=42)
            model.fit(X)
            return model.predict(X)
        return iforest_func
    
    def create_lof():
        def lof_func(X):
            model = LOF(contamination=0.1)
            model.fit(X)
            return model.predict(X)
        return lof_func
    
    def create_ocsvm():
        def ocsvm_func(X):
            model = OCSVM(contamination=0.1)
            model.fit(X)
            return model.predict(X)
        return ocsvm_func
    
    def create_pca():
        def pca_func(X):
            model = PyOD_PCA(contamination=0.1)
            model.fit(X)
            return model.predict(X)
        return pca_func
    
    algorithms = {
        'Isolation Forest': create_iforest(),
        'LOF': create_lof(),
        'One-Class SVM': create_ocsvm(),
        'PCA': create_pca()
    }
    
    # Data sizes to test
    data_sizes = [100, 500, 1000, 2000, 5000]
    
    # Run benchmarks
    results = benchmark.compare_algorithms(algorithms, data_sizes, n_features=10)
    
    # Plot results
    benchmark.plot_performance_comparison(results, "Anomaly Detection Algorithm Benchmarks")
    
    # Find best performing algorithm for each metric
    print("\nBest performing algorithms:")
    print("-" * 40)
    
    for size in data_sizes:
        size_idx = data_sizes.index(size)
        
        # Find fastest algorithm
        times = {name: result['mean_times'][size_idx] for name, result in results.items()}
        fastest = min(times, key=times.get)
        
        # Find most memory efficient
        memory_usage = {name: result['memory_usage'][size_idx] for name, result in results.items()}
        most_efficient = min(memory_usage, key=memory_usage.get)
        
        # Find highest throughput
        throughput = {name: result['throughput'][size_idx] for name, result in results.items()}
        highest_throughput = max(throughput, key=throughput.get)
        
        print(f"Data size {size}:")
        print(f"  Fastest: {fastest} ({times[fastest]:.3f}s)")
        print(f"  Most memory efficient: {most_efficient} ({memory_usage[most_efficient]:.1f}MB)")
        print(f"  Highest throughput: {highest_throughput} ({throughput[highest_throughput]:.0f} samples/s)")


def example_2_parallel_processing():
    """Example 2: Parallel processing optimization."""
    print("\n" + "="*60)
    print("Example 2: Parallel Processing Optimization")
    print("="*60)
    
    optimizer = ParallelProcessingOptimizer()
    
    # Generate test data
    data_size = 10000
    n_features = 20
    X = np.random.randn(data_size, n_features)
    
    print(f"Testing parallel processing with {data_size} samples, {n_features} features")
    print(f"System has {optimizer.n_cores} CPU cores")
    
    # Benchmark different batch sizes
    batch_sizes = [2, 4, 8, 16]
    
    print("\nTesting different batch sizes:")
    batch_results = {}
    
    for n_batches in batch_sizes:
        print(f"\nTesting with {n_batches} batches:")
        results = optimizer.benchmark_parallel_methods(X, n_batches=n_batches)
        batch_results[n_batches] = results
    
    # Find optimal batch size
    print("\nOptimal batch size analysis:")
    print("-" * 40)
    
    for method in ['threading', 'multiprocessing', 'async']:
        best_batches = min(batch_results.keys(), 
                          key=lambda b: batch_results[b][method])
        best_time = batch_results[best_batches][method]
        
        print(f"{method.capitalize():15}: {best_batches} batches ({best_time:.3f}s)")
    
    # Visualize results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['sequential', 'threading', 'multiprocessing', 'async']
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, method in enumerate(methods):
        times = [batch_results[b][method] for b in batch_sizes]
        ax.plot(batch_sizes, times, 'o-', color=colors[i], 
               label=method.capitalize(), linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Batches')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Parallel Processing Performance vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_3_gpu_acceleration():
    """Example 3: GPU acceleration benchmarking."""
    print("\n" + "="*60)
    print("Example 3: GPU Acceleration")
    print("="*60)
    
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping GPU acceleration examples.")
        return
    
    optimizer = GPUAccelerationOptimizer()
    
    if not optimizer.available:
        print("GPU not available. Showing CPU-only results.")
    
    # Benchmark GPU vs CPU for different data sizes
    data_sizes = [100, 500, 1000, 2000, 5000]
    
    results = optimizer.benchmark_gpu_vs_cpu(data_sizes)
    
    # Plot results
    optimizer.plot_gpu_benchmark(results)
    
    if optimizer.available:
        # Show GPU memory usage
        print("\nGPU Memory Analysis:")
        print("-" * 30)
        
        for size in data_sizes:
            X = np.random.randn(size, 50)
            X_gpu = optimizer.create_gpu_data(X)
            
            memory_used = torch.cuda.memory_allocated() / 1e6  # MB
            memory_cached = torch.cuda.memory_reserved() / 1e6  # MB
            
            print(f"Size {size:4d}: {memory_used:6.1f} MB used, {memory_cached:6.1f} MB cached")
            
            # Clear GPU memory
            del X_gpu
            torch.cuda.empty_cache()
    
    # Recommendations
    print("\nGPU Acceleration Recommendations:")
    print("-" * 40)
    
    if optimizer.available:
        optimal_sizes = [size for size, speedup in zip(data_sizes, results['speedups']) if speedup > 2]
        if optimal_sizes:
            print(f"GPU acceleration beneficial for data sizes >= {min(optimal_sizes)}")
        else:
            print("GPU acceleration not beneficial for tested data sizes")
        print("Consider GPU for:")
        print("- Large matrix operations")
        print("- Deep learning models")
        print("- Parallel distance calculations")
    else:
        print("GPU not available. Consider:")
        print("- Installing CUDA-compatible PyTorch")
        print("- Using cloud GPU instances")
        print("- Optimizing CPU algorithms instead")


def example_4_memory_optimization():
    """Example 4: Memory optimization strategies."""
    print("\n" + "="*60)
    print("Example 4: Memory Optimization")
    print("="*60)
    
    optimizer = MemoryOptimizer()
    
    print(f"Initial system memory usage: {optimizer.initial_memory:.2f} GB")
    
    # Test different data sizes
    data_sizes = [5000, 10000, 20000]
    
    for data_size in data_sizes:
        print(f"\n{'='*50}")
        print(f"Testing with {data_size} samples")
        print(f"{'='*50}")
        
        results = optimizer.compare_memory_strategies(
            data_size=data_size, n_features=50
        )
        
        # Calculate efficiency metrics
        print(f"\nEfficiency Analysis for {data_size} samples:")
        print("-" * 45)
        
        for strategy, metrics in results.items():
            efficiency = metrics['accuracy'] / (metrics['time'] * metrics['peak_memory'])
            print(f"{strategy:15}: Efficiency = {efficiency:.6f}")
    
    # Memory optimization tips
    print("\nMemory Optimization Tips:")
    print("-" * 30)
    print("1. Use chunked processing for large datasets")
    print("2. Process data in streams when possible")
    print("3. Use appropriate data types (float32 vs float64)")
    print("4. Clear intermediate variables")
    print("5. Use memory mapping for very large files")
    print("6. Consider approximate algorithms for massive datasets")
    
    # Demonstrate data type optimization
    print("\nData Type Impact:")
    print("-" * 20)
    
    test_size = 10000
    test_features = 100
    
    # Float64 (default)
    X_float64 = np.random.randn(test_size, test_features).astype(np.float64)
    memory_float64 = X_float64.nbytes / (1024**2)  # MB
    
    # Float32
    X_float32 = X_float64.astype(np.float32)
    memory_float32 = X_float32.nbytes / (1024**2)  # MB
    
    # Int16 (for discrete data)
    X_int16 = (X_float64 * 1000).astype(np.int16)
    memory_int16 = X_int16.nbytes / (1024**2)  # MB
    
    print(f"Float64: {memory_float64:.1f} MB")
    print(f"Float32: {memory_float32:.1f} MB (saves {memory_float64-memory_float32:.1f} MB)")
    print(f"Int16:   {memory_int16:.1f} MB (saves {memory_float64-memory_int16:.1f} MB)")


def example_5_profiling_and_optimization():
    """Example 5: Performance profiling and bottleneck identification."""
    print("\n" + "="*60)
    print("Example 5: Performance Profiling and Optimization")
    print("="*60)
    
    if not PROFILER_AVAILABLE:
        print("cProfile not available. Skipping profiling examples.")
        return
    
    def inefficient_anomaly_detection(X):
        """Intentionally inefficient implementation for demonstration."""
        n_samples, n_features = X.shape
        distances = []
        
        # Inefficient O(n²) distance calculation
        for i in range(n_samples):
            sample_distances = []
            for j in range(n_samples):
                if i != j:
                    # Inefficient distance calculation
                    dist = 0
                    for k in range(n_features):
                        dist += (X[i, k] - X[j, k]) ** 2
                    sample_distances.append(np.sqrt(dist))
            distances.append(np.mean(sample_distances))
        
        # Simple threshold-based anomaly detection
        threshold = np.percentile(distances, 90)
        predictions = np.array(distances) > threshold
        return predictions.astype(int) * 2 - 1  # Convert to -1/1
    
    def optimized_anomaly_detection(X):
        """Optimized implementation."""
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Efficient distance calculation
        distances = euclidean_distances(X)
        mean_distances = np.mean(distances, axis=1)
        
        # Threshold-based detection
        threshold = np.percentile(mean_distances, 90)
        predictions = mean_distances > threshold
        return predictions.astype(int) * 2 - 1
    
    # Generate test data
    X = np.random.randn(500, 10)
    
    print("Profiling inefficient implementation...")
    
    # Profile inefficient version
    pr = cProfile.Profile()
    pr.enable()
    result_inefficient = inefficient_anomaly_detection(X)
    pr.disable()
    
    # Print profiling results
    print("\nInefficient Implementation Profile:")
    print("-" * 40)
    stats = pstats.Stats(pr)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(10)  # Top 10 functions
    
    # Time both implementations
    print("\nPerformance Comparison:")
    print("-" * 25)
    
    # Inefficient version timing
    start_time = time.time()
    for _ in range(3):  # Multiple runs for accuracy
        _ = inefficient_anomaly_detection(X)
    inefficient_time = (time.time() - start_time) / 3
    
    # Optimized version timing
    start_time = time.time()
    for _ in range(3):
        _ = optimized_anomaly_detection(X)
    optimized_time = (time.time() - start_time) / 3
    
    speedup = inefficient_time / optimized_time
    
    print(f"Inefficient: {inefficient_time:.3f} seconds")
    print(f"Optimized:   {optimized_time:.3f} seconds")
    print(f"Speedup:     {speedup:.1f}x")
    
    # Memory profiling if available
    if MEMORY_PROFILER_AVAILABLE:
        print("\nMemory profiling inefficient implementation...")
        
        @profile
        def memory_test():
            return inefficient_anomaly_detection(X)
        
        # This will print line-by-line memory usage
        memory_test()
    
    # Optimization recommendations
    print("\nCommon Performance Bottlenecks and Solutions:")
    print("-" * 50)
    print("1. Nested loops → Vectorized operations")
    print("2. Repeated calculations → Caching/memoization")
    print("3. Memory allocations → Pre-allocate arrays")
    print("4. Python loops → NumPy/Numba operations")
    print("5. Single-threaded → Parallel processing")
    print("6. Full dataset loading → Chunked processing")
    print("7. Generic algorithms → Specialized implementations")


def main():
    """Run all performance optimization examples."""
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION FOR ANOMALY DETECTION")
    print("="*60)
    
    examples = [
        ("Algorithm Benchmarking", example_1_algorithm_benchmarking),
        ("Parallel Processing", example_2_parallel_processing),
        ("GPU Acceleration", example_3_gpu_acceleration),
        ("Memory Optimization", example_4_memory_optimization),
        ("Profiling and Optimization", example_5_profiling_and_optimization)
    ]
    
    while True:
        print("\nAvailable Examples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print("0. Exit")
        
        try:
            choice = int(input("\nSelect an example (0-5): "))
            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= len(examples):
                examples[choice-1][1]()
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error running example: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()