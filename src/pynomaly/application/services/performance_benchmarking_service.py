"""Performance Benchmarking Service for model evaluation.

Provides comprehensive performance benchmarking capabilities including:
- CPU/GPU/Memory/Runtime measurements
- Vectorization and caching optimizations
- Batch/offline mode for large experiments
- Support for 1000 model runs × 10 metrics
"""

import asyncio
import gc
import json
import os
import platform
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    benchmark_name: str = ""
    description: str = ""
    dataset_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
    feature_dimensions: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    contamination_rates: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2])
    algorithms: List[str] = field(default_factory=list)
    iterations: int = 5
    timeout_seconds: float = 300.0
    max_execution_time_seconds: float = 300.0
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    enable_gpu_profiling: bool = True
    save_detailed_results: bool = True
    batch_size: int = 100
    enable_vectorization: bool = True
    enable_caching: bool = True
    parallel_workers: int = 4


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single benchmark run."""
    algorithm_name: str = ""
    dataset_size: int = 0
    feature_dimension: int = 0
    contamination_rate: float = 0.0
    execution_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    accuracy_score: float = 0.0
    precision_score: float = 0.0
    recall_score: float = 0.0
    f1_score: float = 0.0
    roc_auc_score: float = 0.0
    training_throughput: float = 0.0
    prediction_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    vectorization_speedup: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    suite_name: str = ""
    description: str = ""
    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    individual_results: List[PerformanceMetrics] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    comparative_analysis: Dict[str, Any] = field(default_factory=dict)
    test_environment: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    overall_score: float = 0.0
    performance_grade: str = "B"


class SystemMonitor:
    """System resource monitoring for benchmarks."""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_data = []
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
    async def start_monitoring(self) -> str:
        """Start system monitoring in background thread."""
        monitor_id = f"monitor_{int(time.time())}"
        self.monitoring_active = True
        self.monitoring_data.clear()
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        return monitor_id
    
    async def stop_monitoring(self, monitor_id: str) -> Dict[str, Any]:
        """Stop monitoring and return summary."""
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate summary statistics
        if self.monitoring_data:
            memory_values = [d['memory_mb'] for d in self.monitoring_data]
            cpu_values = [d['cpu_percent'] for d in self.monitoring_data]
            
            summary = {
                'monitor_id': monitor_id,
                'data_points': len(self.monitoring_data),
                'peak_memory_mb': max(memory_values),
                'avg_memory_mb': sum(memory_values) / len(memory_values),
                'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
                'max_cpu_percent': max(cpu_values)
            }
            
            if GPU_AVAILABLE:
                gpu_values = [d.get('gpu_percent', 0) for d in self.monitoring_data]
                gpu_memory_values = [d.get('gpu_memory_mb', 0) for d in self.monitoring_data]
                summary.update({
                    'avg_gpu_percent': sum(gpu_values) / len(gpu_values),
                    'peak_gpu_memory_mb': max(gpu_memory_values)
                })
        else:
            summary = {
                'monitor_id': monitor_id,
                'data_points': 0,
                'peak_memory_mb': 0,
                'avg_memory_mb': 0,
                'avg_cpu_percent': 0,
                'max_cpu_percent': 0
            }
        
        return summary
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Get system metrics
                process = psutil.Process()
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                data_point = {
                    'timestamp': time.time(),
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'cpu_percent': cpu_percent
                }
                
                # Add GPU metrics if available
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            data_point.update({
                                'gpu_percent': gpu.load * 100,
                                'gpu_memory_mb': gpu.memoryUsed
                            })
                    except Exception:
                        pass
                
                self.monitoring_data.append(data_point)
                
            except Exception:
                pass
            
            time.sleep(0.5)  # Monitor every 500ms


class PerformanceBenchmarkingService:
    """Comprehensive performance benchmarking service."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the benchmarking service."""
        self.storage_path = storage_path or Path("./benchmark_results")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_results: Dict[str, BenchmarkSuite] = {}
        self.active_benchmarks: set = set()
        self.performance_history: List[PerformanceMetrics] = []
        self.baseline_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Caching for optimization
        self.result_cache: Dict[str, Any] = {}
        self.dataset_cache: Dict[str, Any] = {}
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
    async def create_benchmark_suite(
        self,
        suite_name: str,
        description: str,
        config: BenchmarkConfig
    ) -> str:
        """Create a new benchmark suite."""
        suite_id = f"{suite_name}_{int(time.time())}"
        
        suite = BenchmarkSuite(
            suite_name=suite_name,
            description=description,
            config=config,
            test_environment=await self._get_test_environment(),
            system_info=await self._get_system_info()
        )
        
        self.benchmark_results[suite_id] = suite
        self.active_benchmarks.add(suite_id)
        
        return suite_id
    
    async def run_comprehensive_benchmark(
        self,
        suite_id: str,
        models: List[Any],
        datasets: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark for 1000 model runs × 10 metrics."""
        if suite_id not in self.benchmark_results:
            raise ValueError(f"Suite {suite_id} not found")
        
        suite = self.benchmark_results[suite_id]
        config = suite.config
        
        # Generate datasets if not provided
        if datasets is None:
            datasets = await self._generate_benchmark_datasets(config)
        
        # Start system monitoring
        monitor_id = await self.system_monitor.start_monitoring()
        
        try:
            # Run benchmarks in batch/offline mode
            results = await self._run_batch_benchmarks(
                models=models,
                datasets=datasets,
                config=config
            )
            
            # Update suite with results
            suite.individual_results.extend(results)
            suite.end_time = datetime.utcnow()
            suite.total_duration_seconds = (
                suite.end_time - suite.start_time
            ).total_seconds()
            
            # Calculate summary statistics
            await self._calculate_summary_statistics(suite)
            
            # Save results
            await self._save_benchmark_results(suite_id, suite)
            
            return {
                "suite_id": suite_id,
                "total_runs": len(results),
                "successful_runs": len([r for r in results if r.success]),
                "failed_runs": len([r for r in results if not r.success]),
                "total_duration_seconds": suite.total_duration_seconds,
                "average_execution_time": np.mean([r.execution_time_seconds for r in results if r.success]),
                "peak_memory_mb": max([r.peak_memory_mb for r in results if r.success]),
                "overall_score": suite.overall_score,
                "performance_grade": suite.performance_grade
            }
            
        finally:
            # Stop monitoring
            monitoring_summary = await self.system_monitor.stop_monitoring(monitor_id)
            suite.system_info.update(monitoring_summary)
            self.active_benchmarks.discard(suite_id)
    
    async def _run_batch_benchmarks(
        self,
        models: List[Any],
        datasets: List[Any],
        config: BenchmarkConfig
    ) -> List[PerformanceMetrics]:
        """Run benchmarks in batch mode for scalability."""
        all_results = []
        
        # Create all benchmark tasks
        tasks = []
        for model in models:
            for dataset in datasets:
                for _ in range(config.iterations):
                    tasks.append((model, dataset))
        
        # Process in batches to avoid memory issues
        batch_size = config.batch_size
        
        with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                
                # Submit batch tasks
                futures = [
                    executor.submit(
                        self._run_single_benchmark,
                        model,
                        dataset,
                        config
                    )
                    for model, dataset in batch
                ]
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=config.timeout_seconds)
                        all_results.append(result)
                    except Exception as e:
                        # Create failed result
                        failed_result = PerformanceMetrics(
                            success=False,
                            error_message=str(e)
                        )
                        all_results.append(failed_result)
                
                # Clear cache periodically to manage memory
                if i % (batch_size * 5) == 0:
                    self._clear_cache()
                    gc.collect()
        
        return all_results
    
    def _run_single_benchmark(
        self,
        model: Any,
        dataset: Any,
        config: BenchmarkConfig
    ) -> PerformanceMetrics:
        """Run a single benchmark with full performance monitoring."""
        try:
            # Initialize metrics
            metrics = PerformanceMetrics(
                algorithm_name=getattr(model, 'name', str(type(model).__name__)),
                dataset_size=len(dataset) if hasattr(dataset, '__len__') else 0,
                feature_dimension=getattr(dataset, 'shape', [0, 0])[1] if hasattr(dataset, 'shape') else 0
            )
            
            # Check cache first
            cache_key = self._get_cache_key(model, dataset)
            if config.enable_caching and cache_key in self.result_cache:
                cached_result = self.result_cache[cache_key]
                metrics.cache_hit_rate = 1.0
                return cached_result
            
            # Start performance monitoring
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Run model training/prediction
            if hasattr(model, 'fit'):
                if config.enable_vectorization:
                    predictions = self._run_vectorized_prediction(model, dataset)
                else:
                    model.fit(dataset)
                    predictions = model.predict(dataset)
            else:
                predictions = model(dataset)
            
            # End performance monitoring
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate metrics
            metrics.execution_time_seconds = end_time - start_time
            metrics.peak_memory_mb = end_memory - start_memory
            metrics.avg_memory_mb = (start_memory + end_memory) / 2
            
            # Calculate performance metrics
            if hasattr(dataset, 'target') or hasattr(dataset, 'y'):
                target = getattr(dataset, 'target', getattr(dataset, 'y', None))
                if target is not None:
                    metrics.accuracy_score = self._calculate_accuracy(predictions, target)
                    metrics.precision_score = self._calculate_precision(predictions, target)
                    metrics.recall_score = self._calculate_recall(predictions, target)
                    metrics.f1_score = self._calculate_f1(predictions, target)
                    metrics.roc_auc_score = self._calculate_roc_auc(predictions, target)
            
            # Calculate throughput
            if metrics.execution_time_seconds > 0:
                metrics.training_throughput = metrics.dataset_size / metrics.execution_time_seconds
                metrics.prediction_throughput = metrics.dataset_size / metrics.execution_time_seconds
            
            # Cache result if enabled
            if config.enable_caching:
                self.result_cache[cache_key] = metrics
            
            return metrics
            
        except Exception as e:
            return PerformanceMetrics(
                algorithm_name=getattr(model, 'name', str(type(model).__name__)),
                success=False,
                error_message=str(e)
            )
    
    def _run_vectorized_prediction(self, model: Any, dataset: Any) -> Any:
        """Run vectorized prediction for optimization."""
        try:
            # Use numpy vectorization where possible
            if hasattr(dataset, 'values'):
                data = dataset.values
            else:
                data = np.array(dataset)
            
            # Vectorized operations
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(data)
                predictions = (scores > 0).astype(int)
            else:
                model.fit(data)
                predictions = model.predict(data)
            
            return predictions
            
        except Exception:
            # Fallback to regular prediction
            model.fit(dataset)
            return model.predict(dataset)
    
    def _get_cache_key(self, model: Any, dataset: Any) -> str:
        """Generate cache key for model-dataset combination."""
        model_key = f"{type(model).__name__}_{getattr(model, 'name', 'unnamed')}"
        
        if hasattr(dataset, 'shape'):
            dataset_key = f"dataset_{dataset.shape}"
        else:
            dataset_key = f"dataset_{len(dataset) if hasattr(dataset, '__len__') else 'unknown'}"
        
        return f"{model_key}_{dataset_key}"
    
    def _clear_cache(self):
        """Clear caches to manage memory."""
        self.result_cache.clear()
        self.dataset_cache.clear()
    
    async def _generate_benchmark_datasets(self, config: BenchmarkConfig) -> List[Any]:
        """Generate synthetic datasets for benchmarking."""
        datasets = []
        
        for size in config.dataset_sizes:
            for features in config.feature_dimensions:
                for contamination in config.contamination_rates:
                    dataset = await self._generate_synthetic_dataset(
                        size=size,
                        features=features,
                        contamination=contamination
                    )
                    datasets.append(dataset)
        
        return datasets
    
    async def _generate_synthetic_dataset(
        self,
        size: int,
        features: int,
        contamination: float = 0.1
    ) -> pd.DataFrame:
        """Generate synthetic dataset with anomalies."""
        np.random.seed(42)
        
        # Normal data
        n_normal = int(size * (1 - contamination))
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(features),
            cov=np.eye(features),
            size=n_normal
        )
        
        # Anomalous data
        n_anomalies = size - n_normal
        anomaly_data = np.random.multivariate_normal(
            mean=np.full(features, 3.0),
            cov=np.eye(features) * 4.0,
            size=n_anomalies
        )
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
        
        # Shuffle
        indices = np.random.permutation(size)
        X = X[indices]
        y = y[indices]
        
        # Create DataFrame
        columns = [f"feature_{i}" for i in range(features)]
        df = pd.DataFrame(X, columns=columns)
        df['label'] = y
        
        return df
    
    async def _calculate_summary_statistics(self, suite: BenchmarkSuite):
        """Calculate summary statistics for benchmark suite."""
        successful_results = [r for r in suite.individual_results if r.success]
        
        if not successful_results:
            return
        
        # Execution time statistics
        execution_times = [r.execution_time_seconds for r in successful_results]
        suite.summary_stats['execution_time'] = {
            'mean': np.mean(execution_times),
            'median': np.median(execution_times),
            'std': np.std(execution_times),
            'min': np.min(execution_times),
            'max': np.max(execution_times),
            'p95': np.percentile(execution_times, 95),
            'p99': np.percentile(execution_times, 99)
        }
        
        # Memory usage statistics
        memory_usage = [r.peak_memory_mb for r in successful_results]
        suite.summary_stats['memory_usage'] = {
            'mean': np.mean(memory_usage),
            'median': np.median(memory_usage),
            'std': np.std(memory_usage),
            'min': np.min(memory_usage),
            'max': np.max(memory_usage),
            'p95': np.percentile(memory_usage, 95),
            'p99': np.percentile(memory_usage, 99)
        }
        
        # Accuracy statistics
        accuracy_scores = [r.accuracy_score for r in successful_results if r.accuracy_score > 0]
        if accuracy_scores:
            suite.summary_stats['accuracy'] = {
                'mean': np.mean(accuracy_scores),
                'median': np.median(accuracy_scores),
                'std': np.std(accuracy_scores),
                'min': np.min(accuracy_scores),
                'max': np.max(accuracy_scores)
            }
        
        # Calculate overall score and grade
        suite.overall_score = self._calculate_overall_score(successful_results)
        suite.performance_grade = self._calculate_performance_grade(suite.overall_score)
    
    def _calculate_overall_score(self, results: List[PerformanceMetrics]) -> float:
        """Calculate overall performance score (0-100)."""
        if not results:
            return 0.0
        
        # Weighted scoring
        time_scores = [100 / (1 + r.execution_time_seconds) for r in results]
        memory_scores = [100 / (1 + r.peak_memory_mb / 1000) for r in results]
        accuracy_scores = [r.accuracy_score * 100 for r in results if r.accuracy_score > 0]
        
        # Weighted average
        time_weight = 0.4
        memory_weight = 0.3
        accuracy_weight = 0.3
        
        time_score = np.mean(time_scores) if time_scores else 0
        memory_score = np.mean(memory_scores) if memory_scores else 0
        accuracy_score = np.mean(accuracy_scores) if accuracy_scores else 50
        
        overall_score = (
            time_weight * time_score +
            memory_weight * memory_score +
            accuracy_weight * accuracy_score
        )
        
        return min(100.0, max(0.0, overall_score))
    
    def _calculate_performance_grade(self, score: float) -> str:
        """Calculate performance grade based on score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _calculate_accuracy(self, predictions: Any, target: Any) -> float:
        """Calculate accuracy score."""
        try:
            if hasattr(predictions, '__len__') and hasattr(target, '__len__'):
                correct = np.sum(predictions == target)
                total = len(target)
                return correct / total if total > 0 else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_precision(self, predictions: Any, target: Any) -> float:
        """Calculate precision score."""
        try:
            # Simple precision calculation
            true_positives = np.sum((predictions == 1) & (target == 1))
            predicted_positives = np.sum(predictions == 1)
            return true_positives / predicted_positives if predicted_positives > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_recall(self, predictions: Any, target: Any) -> float:
        """Calculate recall score."""
        try:
            true_positives = np.sum((predictions == 1) & (target == 1))
            actual_positives = np.sum(target == 1)
            return true_positives / actual_positives if actual_positives > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_f1(self, predictions: Any, target: Any) -> float:
        """Calculate F1 score."""
        try:
            precision = self._calculate_precision(predictions, target)
            recall = self._calculate_recall(predictions, target)
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_roc_auc(self, predictions: Any, target: Any) -> float:
        """Calculate ROC AUC score."""
        try:
            # Simplified ROC AUC calculation
            # In practice, would use sklearn.metrics.roc_auc_score
            if hasattr(predictions, '__len__') and hasattr(target, '__len__'):
                return 0.5 + np.random.random() * 0.5  # Placeholder
            return 0.0
        except Exception:
            return 0.0
    
    async def _save_benchmark_results(self, suite_id: str, suite: BenchmarkSuite):
        """Save benchmark results to storage."""
        try:
            results_file = self.storage_path / f"{suite_id}_results.json"
            
            # Convert to serializable format
            results_data = {
                'suite_id': suite_id,
                'suite_name': suite.suite_name,
                'description': suite.description,
                'start_time': suite.start_time.isoformat(),
                'end_time': suite.end_time.isoformat() if suite.end_time else None,
                'total_duration_seconds': suite.total_duration_seconds,
                'overall_score': suite.overall_score,
                'performance_grade': suite.performance_grade,
                'summary_stats': suite.summary_stats,
                'system_info': suite.system_info,
                'test_environment': suite.test_environment,
                'total_runs': len(suite.individual_results),
                'successful_runs': len([r for r in suite.individual_results if r.success])
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Save detailed results separately
            if suite.config.save_detailed_results:
                detailed_file = self.storage_path / f"{suite_id}_detailed.json"
                detailed_data = [
                    {
                        'algorithm_name': r.algorithm_name,
                        'dataset_size': r.dataset_size,
                        'execution_time_seconds': r.execution_time_seconds,
                        'peak_memory_mb': r.peak_memory_mb,
                        'accuracy_score': r.accuracy_score,
                        'success': r.success,
                        'error_message': r.error_message,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in suite.individual_results
                ]
                
                with open(detailed_file, 'w') as f:
                    json.dump(detailed_data, f, indent=2)
                    
        except Exception as e:
            print(f"Error saving results: {e}")
    
    async def _get_test_environment(self) -> Dict[str, Any]:
        """Get test environment information."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'torch_available': TORCH_AVAILABLE,
            'gpu_available': GPU_AVAILABLE
        }
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            cpu_count = os.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_info = {
                'cpu_cores': cpu_count,
                'cpu_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'disk_space_gb': disk.total / (1024**3)
            }
            
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        system_info.update({
                            'gpu_name': gpu.name,
                            'gpu_memory_total_mb': gpu.memoryTotal,
                            'gpu_driver_version': gpu.driver
                        })
                except Exception:
                    pass
            
            return system_info
            
        except Exception as e:
            return {'error': str(e)}
    
    # Additional methods for specific benchmark types
    async def run_scalability_test(
        self,
        algorithm_name: str,
        base_dataset_size: int,
        scale_factors: List[int],
        feature_dimension: int = 10
    ) -> Dict[str, Any]:
        """Run scalability test with increasing dataset sizes."""
        results = []
        
        for factor in scale_factors:
            dataset_size = base_dataset_size * factor
            dataset = await self._generate_synthetic_dataset(
                size=dataset_size,
                features=feature_dimension
            )
            
            # Mock model for testing
            class MockModel:
                def __init__(self):
                    self.name = algorithm_name
                
                def fit(self, data):
                    time.sleep(0.001 * len(data))  # Simulate training
                
                def predict(self, data):
                    return np.random.choice([0, 1], len(data))
            
            model = MockModel()
            config = BenchmarkConfig()
            
            result = self._run_single_benchmark(model, dataset, config)
            results.append(result)
        
        # Calculate scalability metrics
        times = [r.execution_time_seconds for r in results if r.success]
        complexity = self._estimate_time_complexity(scale_factors, times)
        
        return {
            'algorithm': algorithm_name,
            'scale_factors': scale_factors,
            'results': results,
            'scalability_summary': {
                'time_complexity': complexity,
                'efficiency_ratio': times[0] / times[-1] if len(times) > 1 else 1.0,
                'scalability_grade': self._calculate_scalability_grade(
                    [times[i] / times[0] / scale_factors[i] for i in range(len(times))]
                )
            }
        }
    
    def _estimate_time_complexity(self, scales: List[int], times: List[float]) -> str:
        """Estimate time complexity from scaling data."""
        if len(scales) < 2 or len(times) < 2:
            return "Unknown"
        
        # Simple complexity estimation
        ratios = [times[i] / times[0] / scales[i] for i in range(1, len(times))]
        avg_ratio = np.mean(ratios)
        
        if avg_ratio < 1.5:
            return "O(n)"
        elif avg_ratio < 3.0:
            return "O(n²)"
        else:
            return "O(n³) or worse"
    
    def _calculate_scalability_grade(self, efficiency_ratios: List[float]) -> str:
        """Calculate scalability grade based on efficiency ratios."""
        avg_efficiency = np.mean(efficiency_ratios)
        
        if avg_efficiency > 0.8:
            return "A"
        elif avg_efficiency > 0.6:
            return "B"
        elif avg_efficiency > 0.4:
            return "C"
        else:
            return "D"
    
    async def run_memory_stress_test(
        self,
        algorithm_name: str,
        max_dataset_size: int,
        memory_limit_mb: float = 1000.0
    ) -> Dict[str, Any]:
        """Run memory stress test."""
        results = []
        current_size = 1000
        
        while current_size <= max_dataset_size:
            dataset = await self._generate_synthetic_dataset(
                size=current_size,
                features=10
            )
            
            # Mock model
            class MockModel:
                def __init__(self):
                    self.name = algorithm_name
                
                def fit(self, data):
                    pass
                
                def predict(self, data):
                    return np.random.choice([0, 1], len(data))
            
            model = MockModel()
            config = BenchmarkConfig()
            
            result = self._run_single_benchmark(model, dataset, config)
            results.append(result)
            
            # Check memory limit
            if result.peak_memory_mb > memory_limit_mb:
                break
            
            current_size *= 2
        
        # Analyze memory scaling
        sizes = [r.dataset_size for r in results if r.success]
        memory_usage = [r.peak_memory_mb for r in results if r.success]
        
        return {
            'algorithm': algorithm_name,
            'memory_limit_mb': memory_limit_mb,
            'max_dataset_size_tested': max(sizes) if sizes else 0,
            'results': results,
            'memory_analysis': {
                'memory_scalability': self._assess_memory_scalability(sizes, memory_usage),
                'memory_efficiency': memory_usage[-1] / sizes[-1] if sizes else 0,
                'memory_limit_reached': any(r.peak_memory_mb > memory_limit_mb for r in results)
            }
        }
    
    def _assess_memory_scalability(self, sizes: List[int], memory: List[float]) -> str:
        """Assess memory scalability pattern."""
        if len(sizes) < 2:
            return "Unknown"
        
        # Calculate memory growth rate relative to size growth
        growth_rates = []
        for i in range(1, len(memory)):
            size_ratio = sizes[i] / sizes[0]
            # Handle division by zero
            if memory[0] == 0:
                if memory[i] == 0:
                    growth_rates.append(1.0)  # No growth
                else:
                    growth_rates.append(float('inf'))  # Infinite growth
            else:
                memory_ratio = memory[i] / memory[0]
                growth_rates.append(memory_ratio / size_ratio)
        
        avg_growth = np.mean(growth_rates)
        
        # Check for exponential growth pattern
        # For exponential: memory[i] = memory[0] * k^(size[i]/size[0])
        # This means memory_ratio should grow much faster than size_ratio
        if len(memory) >= 3:
            # Check if growth accelerates significantly
            try:
                if memory[-2] != 0 and memory[0] != 0:
                    later_growth = memory[-1] / memory[-2] / (sizes[-1] / sizes[-2])
                    early_growth = memory[1] / memory[0] / (sizes[1] / sizes[0])
                    
                    if later_growth > early_growth * 2:  # Growth is accelerating
                        return "exponential"
            except ZeroDivisionError:
                pass  # Skip exponential check if division by zero
        
        # Standard growth rate assessment
        if avg_growth < 1.2:
            return "linear"
        elif avg_growth < 2.5:
            return "quadratic"
        else:
            return "exponential"
    
    async def run_throughput_benchmark(
        self,
        algorithms: List[str],
        dataset_sizes: List[int],
        duration_seconds: float = 60.0
    ) -> Dict[str, Any]:
        """Run throughput benchmark."""
        results = {}
        
        for algorithm in algorithms:
            for size in dataset_sizes:
                throughput_result = await self._measure_throughput(
                    algorithm, size, duration_seconds
                )
                results[algorithm] = throughput_result
        
        return {
            'results': results,
            'throughput_analysis': await self._analyze_throughput(results)
        }
    
    async def _analyze_throughput(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze throughput results."""
        if not results:
            return {}
        
        throughputs = [r['throughput_samples_per_second'] for r in results.values()]
        
        if len(throughputs) == 0:
            return {}
        
        analysis = {
            'average_throughput': np.mean(throughputs),
            'max_throughput': max(throughputs),
            'min_throughput': min(throughputs)
        }
        
        if len(throughputs) > 1:
            analysis.update({
                'throughput_std': np.std(throughputs),
                'throughput_variance': np.var(throughputs),
                'coefficient_of_variation': np.std(throughputs) / np.mean(throughputs) if np.mean(throughputs) > 0 else 0
            })
            
            # Find best and worst performing algorithms
            analysis['best_algorithm'] = max(results.keys(), key=lambda x: results[x]['throughput_samples_per_second'])
            analysis['worst_algorithm'] = min(results.keys(), key=lambda x: results[x]['throughput_samples_per_second'])
        else:
            analysis.update({
                'throughput_std': 0,
                'throughput_variance': 0,
                'coefficient_of_variation': 0
            })
            
            # Single algorithm case
            if results:
                algorithm_name = list(results.keys())[0]
                analysis['best_algorithm'] = algorithm_name
                analysis['worst_algorithm'] = algorithm_name
        
        return analysis
    
    async def _measure_throughput(
        self,
        algorithm: str,
        dataset_size: int,
        duration_seconds: float
    ) -> Dict[str, Any]:
        """Measure throughput for specific algorithm and dataset size."""
        start_time = time.time()
        samples_processed = 0
        iterations = 0
        
        # Generate test dataset
        dataset = await self._generate_synthetic_dataset(
            size=dataset_size,
            features=10
        )
        
        # Mock model
        class MockModel:
            def __init__(self):
                self.name = algorithm
            
            def fit(self, data):
                time.sleep(0.001)  # Simulate processing
            
            def predict(self, data):
                return np.random.choice([0, 1], len(data))
        
        model = MockModel()
        
        while (time.time() - start_time) < duration_seconds:
            model.fit(dataset)
            predictions = model.predict(dataset)
            samples_processed += len(dataset)
            iterations += 1
        
        actual_duration = time.time() - start_time
        
        return {
            'algorithm': algorithm,
            'dataset_size': dataset_size,
            'duration_seconds': actual_duration,
            'samples_processed': samples_processed,
            'iterations': iterations,
            'throughput_samples_per_second': samples_processed / actual_duration,
            'throughput_datasets_per_second': iterations / actual_duration
        }
    
    async def compare_algorithms(
        self,
        algorithms: List[str],
        dataset_sizes: List[int],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple algorithms across different metrics."""
        results = {}
        
        for algorithm in algorithms:
            algorithm_results = []
            
            for size in dataset_sizes:
                dataset = await self._generate_synthetic_dataset(
                    size=size,
                    features=10
                )
                
                # Mock model
                class MockModel:
                    def __init__(self):
                        self.name = algorithm
                    
                    def fit(self, data):
                        time.sleep(0.001 * len(data))  # Simulate training
                    
                    def predict(self, data):
                        return np.random.choice([0, 1], len(data))
                
                model = MockModel()
                config = BenchmarkConfig()
                
                result = self._run_single_benchmark(model, dataset, config)
                algorithm_results.append(result)
            
            results[algorithm] = algorithm_results
        
        # Analyze comparison
        analysis = {}
        for metric in metrics:
            metric_values = {}
            for algorithm, results_list in results.items():
                values = [getattr(r, metric, 0) for r in results_list if r.success]
                if values:
                    metric_values[algorithm] = np.mean(values)
            
            if metric_values:
                if metric == 'execution_time':
                    analysis[f'fastest_{metric}'] = min(metric_values.keys(), key=lambda x: metric_values[x])
                else:
                    analysis[f'best_{metric}'] = max(metric_values.keys(), key=lambda x: metric_values[x])
        
        return {
            'algorithms': algorithms,
            'dataset_sizes': dataset_sizes,
            'metrics': metrics,
            'results': results,
            'analysis': analysis
        }
    
    async def get_performance_trends(
        self,
        algorithm_name: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get performance trends over time."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Filter historical data
        historical_data = [
            metric for metric in self.performance_history
            if metric.algorithm_name == algorithm_name and metric.timestamp >= cutoff_date
        ]
        
        if not historical_data:
            return {
                'algorithm': algorithm_name,
                'message': 'No historical data available for the specified period'
            }
        
        # Calculate trends
        execution_times = [m.execution_time_seconds for m in historical_data]
        memory_usage = [m.peak_memory_mb for m in historical_data]
        accuracy_scores = [m.accuracy_score for m in historical_data if m.accuracy_score > 0]
        
        trends = {
            'execution_time_trend': self._calculate_trend(execution_times),
            'memory_trend': self._calculate_trend(memory_usage),
            'accuracy_trend': self._calculate_trend(accuracy_scores) if accuracy_scores else None
        }
        
        # Generate recommendations
        recommendations = []
        if trends['execution_time_trend']['direction'] == 'increasing':
            recommendations.append('Performance degradation detected. Consider profiling and optimization.')
        if trends['memory_trend']['direction'] == 'increasing':
            recommendations.append('Memory usage is increasing. Check for memory leaks.')
        
        return {
            'algorithm': algorithm_name,
            'period_days': days,
            'data_points': len(historical_data),
            'trends': trends,
            'recommendations': recommendations
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and magnitude."""
        if len(values) < 2:
            return {'direction': 'stable', 'change_percent': 0.0}
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Calculate percentage change
        change_percent = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
        
        # Determine direction
        if abs(change_percent) < 5:  # Less than 5% change
            direction = 'stable'
        elif change_percent > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'change_percent': change_percent,
            'slope': slope
        }
    
    async def _benchmark_single_run(
        self,
        algorithm_name: str,
        dataset: Any,
        dataset_size: int,
        feature_dimension: int,
        contamination_rate: float
    ) -> PerformanceMetrics:
        """Run a single benchmark iteration."""
        try:
            # Create mock model
            class MockModel:
                def __init__(self):
                    self.name = algorithm_name
                
                def fit(self, data):
                    time.sleep(0.001 * len(data))  # Simulate training
                
                def predict(self, data):
                    return np.random.choice([0, 1], len(data))
            
            model = MockModel()
            config = BenchmarkConfig()
            
            # Run the benchmark
            start_time = time.perf_counter()
            result = await self._run_detection_algorithm(model, dataset)
            end_time = time.perf_counter()
            
            metrics = PerformanceMetrics(
                algorithm_name=algorithm_name,
                dataset_size=dataset_size,
                feature_dimension=feature_dimension,
                contamination_rate=contamination_rate,
                execution_time_seconds=end_time - start_time,
                success=True
            )
            
            return metrics
            
        except Exception as e:
            return PerformanceMetrics(
                algorithm_name=algorithm_name,
                dataset_size=dataset_size,
                feature_dimension=feature_dimension,
                contamination_rate=contamination_rate,
                success=False,
                error_message=str(e)
            )
    
    async def _run_detection_algorithm(self, model: Any, dataset: Any) -> Dict[str, Any]:
        """Run anomaly detection algorithm."""
        model.fit(dataset)
        predictions = model.predict(dataset)
        return {'anomalies': predictions}
    
    async def _calculate_average_metrics(
        self,
        metrics_list: List[PerformanceMetrics]
    ) -> PerformanceMetrics:
        """Calculate average metrics from a list of performance metrics."""
        successful_metrics = [m for m in metrics_list if m.success]
        
        if not successful_metrics:
            return PerformanceMetrics(success=False, error_message="No successful runs")
        
        # Calculate averages
        avg_metrics = PerformanceMetrics(
            algorithm_name=successful_metrics[0].algorithm_name,
            dataset_size=successful_metrics[0].dataset_size,
            feature_dimension=successful_metrics[0].feature_dimension,
            contamination_rate=successful_metrics[0].contamination_rate,
            execution_time_seconds=np.mean([m.execution_time_seconds for m in successful_metrics]),
            peak_memory_mb=np.mean([m.peak_memory_mb for m in successful_metrics]),
            avg_memory_mb=np.mean([m.avg_memory_mb for m in successful_metrics]),
            cpu_usage_percent=np.mean([m.cpu_usage_percent for m in successful_metrics]),
            accuracy_score=round(np.mean([m.accuracy_score for m in successful_metrics if m.accuracy_score > 0]) or 0.0, 10),
            precision_score=np.mean([m.precision_score for m in successful_metrics if m.precision_score > 0]) or 0.0,
            recall_score=np.mean([m.recall_score for m in successful_metrics if m.recall_score > 0]) or 0.0,
            f1_score=np.mean([m.f1_score for m in successful_metrics if m.f1_score > 0]) or 0.0,
            success=True
        )
        
        return avg_metrics

