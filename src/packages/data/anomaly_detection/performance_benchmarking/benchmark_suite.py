"""Comprehensive benchmarking suite for Phase 2 components."""

from __future__ import annotations

import time
import psutil
import gc
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
from pathlib import Path

# Import Phase 2 components
try:
    from simplified_services.core_detection_service import CoreDetectionService
    from simplified_services.automl_service import AutoMLService
    from simplified_services.ensemble_service import EnsembleService
    from performance.batch_processor import BatchProcessor
    from performance.streaming_detector import StreamingDetector
    from specialized_algorithms.time_series_detector import TimeSeriesDetector
    from specialized_algorithms.text_anomaly_detector import TextAnomalyDetector
    from enhanced_features.model_persistence import ModelPersistence
    from enhanced_features.advanced_explainability import AdvancedExplainability
    from enhanced_features.monitoring_alerting import MonitoringAlertingSystem
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    component_name: str
    test_name: str
    data_size: Tuple[int, int]  # (samples, features)
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_samples_per_second: float
    accuracy_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmark execution."""
    data_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (100, 5), (500, 10), (1000, 20), (5000, 50), (10000, 100)
    ])
    algorithms: List[str] = field(default_factory=lambda: ["iforest", "lof", "svm"])
    contamination_rates: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    n_iterations: int = 3
    warmup_iterations: int = 1
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    parallel_testing: bool = False
    output_directory: str = "benchmark_results"


class BenchmarkSuite:
    """Comprehensive benchmarking suite for Phase 2 components.
    
    This suite provides detailed performance analysis for:
    - Simplified Services (CoreDetectionService, AutoMLService, EnsembleService)
    - Performance Features (BatchProcessor, StreamingDetector)
    - Specialized Algorithms (TimeSeriesDetector, TextAnomalyDetector)
    - Enhanced Features (ModelPersistence, AdvancedExplainability, MonitoringAlertingSystem)
    """
    
    def __init__(self, config: Optional[BenchmarkConfiguration] = None):
        """Initialize benchmark suite.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfiguration()
        self.results: List[BenchmarkResult] = []
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize system monitoring
        self.process = psutil.Process()
        
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run comprehensive benchmark of all Phase 2 components.
        
        Returns:
            List of benchmark results
        """
        print("🚀 Starting comprehensive Phase 2 performance benchmark...")
        
        if not PHASE2_AVAILABLE:
            print("❌ Phase 2 components not available for benchmarking")
            return []
        
        # Benchmark each component category
        self._benchmark_simplified_services()
        self._benchmark_performance_features()
        self._benchmark_specialized_algorithms()
        self._benchmark_enhanced_features()
        
        # Generate comprehensive report
        self._generate_benchmark_report()
        
        print(f"✅ Benchmark completed! {len(self.results)} tests executed")
        return self.results
    
    def _benchmark_simplified_services(self) -> None:
        """Benchmark simplified services."""
        print("\n📊 Benchmarking Simplified Services...")
        
        # CoreDetectionService
        self._benchmark_core_detection_service()
        
        # AutoMLService  
        self._benchmark_automl_service()
        
        # EnsembleService
        self._benchmark_ensemble_service()
    
    def _benchmark_performance_features(self) -> None:
        """Benchmark performance features."""
        print("\n⚡ Benchmarking Performance Features...")
        
        # BatchProcessor
        self._benchmark_batch_processor()
        
        # StreamingDetector
        self._benchmark_streaming_detector()
    
    def _benchmark_specialized_algorithms(self) -> None:
        """Benchmark specialized algorithms."""
        print("\n🔍 Benchmarking Specialized Algorithms...")
        
        # TimeSeriesDetector
        self._benchmark_time_series_detector()
        
        # TextAnomalyDetector
        self._benchmark_text_anomaly_detector()
    
    def _benchmark_enhanced_features(self) -> None:
        """Benchmark enhanced features."""
        print("\n📊 Benchmarking Enhanced Features...")
        
        # ModelPersistence
        self._benchmark_model_persistence()
        
        # AdvancedExplainability
        self._benchmark_advanced_explainability()
        
        # MonitoringAlertingSystem
        self._benchmark_monitoring_alerting()
    
    def _benchmark_core_detection_service(self) -> None:
        """Benchmark CoreDetectionService."""
        service = CoreDetectionService()
        
        for data_size in self.config.data_sizes:
            for algorithm in self.config.algorithms:
                for contamination in self.config.contamination_rates:
                    
                    def benchmark_func():
                        data = self._generate_test_data(data_size)
                        return service.detect_anomalies(
                            data, 
                            algorithm=algorithm, 
                            contamination=contamination
                        )
                    
                    result = self._execute_benchmark(
                        func=benchmark_func,
                        component_name="CoreDetectionService",
                        test_name=f"{algorithm}_contamination_{contamination}",
                        data_size=data_size,
                        metadata={
                            "algorithm": algorithm,
                            "contamination": contamination
                        }
                    )
                    
                    if result:
                        self.results.append(result)
    
    def _benchmark_automl_service(self) -> None:
        """Benchmark AutoMLService."""
        service = AutoMLService()
        
        for data_size in self.config.data_sizes[:3]:  # Limit for AutoML (computationally expensive)
            
            def benchmark_func():
                data = self._generate_test_data(data_size)
                return service.auto_detect(data)
            
            result = self._execute_benchmark(
                func=benchmark_func,
                component_name="AutoMLService",
                test_name="auto_detect",
                data_size=data_size,
                metadata={"test_type": "auto_detect"}
            )
            
            if result:
                self.results.append(result)
            
            # Benchmark algorithm recommendation separately
            def recommend_func():
                data = self._generate_test_data(data_size)
                return service.recommend_algorithm(data)
            
            result = self._execute_benchmark(
                func=recommend_func,
                component_name="AutoMLService",
                test_name="recommend_algorithm",
                data_size=data_size,
                metadata={"test_type": "recommendation"}
            )
            
            if result:
                self.results.append(result)
    
    def _benchmark_ensemble_service(self) -> None:
        """Benchmark EnsembleService."""
        service = EnsembleService()
        
        for data_size in self.config.data_sizes:
            algorithms = ["iforest", "lof", "svm"]
            voting_methods = ["majority", "weighted"]
            
            for voting in voting_methods:
                
                def benchmark_func():
                    data = self._generate_test_data(data_size)
                    return service.ensemble_detect(
                        data,
                        algorithms=algorithms,
                        voting=voting
                    )
                
                result = self._execute_benchmark(
                    func=benchmark_func,
                    component_name="EnsembleService",
                    test_name=f"ensemble_{voting}_voting",
                    data_size=data_size,
                    metadata={
                        "algorithms": algorithms,
                        "voting": voting
                    }
                )
                
                if result:
                    self.results.append(result)
    
    def _benchmark_batch_processor(self) -> None:
        """Benchmark BatchProcessor."""
        processor = BatchProcessor()
        
        # Test larger datasets for batch processing
        large_data_sizes = [(50000, 50), (100000, 100), (200000, 50)]
        
        for data_size in large_data_sizes:
            batch_sizes = [1000, 5000, 10000]
            
            for batch_size in batch_sizes:
                
                def benchmark_func():
                    data = self._generate_test_data(data_size)
                    return processor.process_large_dataset(
                        data,
                        algorithm="iforest",
                        batch_size=batch_size
                    )
                
                result = self._execute_benchmark(
                    func=benchmark_func,
                    component_name="BatchProcessor",
                    test_name=f"batch_size_{batch_size}",
                    data_size=data_size,
                    metadata={
                        "batch_size": batch_size,
                        "algorithm": "iforest"
                    }
                )
                
                if result:
                    self.results.append(result)
    
    def _benchmark_streaming_detector(self) -> None:
        """Benchmark StreamingDetector."""
        detector = StreamingDetector(algorithm="lof", window_size=1000)
        
        batch_sizes = [50, 100, 200, 500]
        
        for batch_size in batch_sizes:
            
            def benchmark_func():
                # Simulate streaming by processing multiple batches
                total_processed = 0
                for _ in range(10):  # Process 10 batches
                    batch = self._generate_test_data((batch_size, 10))
                    detector.process_batch(batch)
                    total_processed += batch_size
                return total_processed
            
            result = self._execute_benchmark(
                func=benchmark_func,
                component_name="StreamingDetector",
                test_name=f"streaming_batch_{batch_size}",
                data_size=(batch_size * 10, 10),
                metadata={
                    "batch_size": batch_size,
                    "n_batches": 10,
                    "algorithm": "lof"
                }
            )
            
            if result:
                self.results.append(result)
    
    def _benchmark_time_series_detector(self) -> None:
        """Benchmark TimeSeriesDetector."""
        detector = TimeSeriesDetector()
        
        ts_lengths = [1000, 5000, 10000]
        methods = ["statistical", "pattern"]
        
        for length in ts_lengths:
            for method in methods:
                
                def benchmark_func():
                    # Generate time series data
                    ts_data = self._generate_time_series_data(length)
                    return detector.detect_anomalies(
                        ts_data,
                        method=method,
                        window_size=100
                    )
                
                result = self._execute_benchmark(
                    func=benchmark_func,
                    component_name="TimeSeriesDetector",
                    test_name=f"{method}_method_length_{length}",
                    data_size=(length, 1),
                    metadata={
                        "method": method,
                        "ts_length": length,
                        "window_size": 100
                    }
                )
                
                if result:
                    self.results.append(result)
    
    def _benchmark_text_anomaly_detector(self) -> None:
        """Benchmark TextAnomalyDetector."""
        detector = TextAnomalyDetector()
        
        text_counts = [100, 500, 1000]
        
        for count in text_counts:
            
            def benchmark_func():
                # Generate text data
                texts = self._generate_text_data(count)
                return detector.detect_anomalies(
                    texts,
                    features=["length", "vocabulary", "format"]
                )
            
            result = self._execute_benchmark(
                func=benchmark_func,
                component_name="TextAnomalyDetector",
                test_name=f"text_count_{count}",
                data_size=(count, 3),  # 3 features
                metadata={
                    "text_count": count,
                    "features": ["length", "vocabulary", "format"]
                }
            )
            
            if result:
                self.results.append(result)
    
    def _benchmark_model_persistence(self) -> None:
        """Benchmark ModelPersistence."""
        persistence = ModelPersistence("benchmark_models")
        
        data_sizes = [(1000, 10), (5000, 50), (10000, 100)]
        
        for data_size in data_sizes:
            
            # Benchmark save operation
            def save_benchmark():
                data = self._generate_test_data(data_size)
                model_data = {"algorithm": "iforest", "trained": True}
                return persistence.save_model(
                    model_data=model_data,
                    training_data=data,
                    algorithm="iforest",
                    contamination=0.1
                )
            
            result = self._execute_benchmark(
                func=save_benchmark,
                component_name="ModelPersistence",
                test_name=f"save_model_{data_size[0]}x{data_size[1]}",
                data_size=data_size,
                metadata={"operation": "save", "algorithm": "iforest"}
            )
            
            if result:
                self.results.append(result)
                
                # Benchmark load operation
                model_id = result.metadata.get("result")
                if model_id:
                    
                    def load_benchmark():
                        return persistence.load_model(model_id)
                    
                    load_result = self._execute_benchmark(
                        func=load_benchmark,
                        component_name="ModelPersistence",
                        test_name=f"load_model_{data_size[0]}x{data_size[1]}",
                        data_size=data_size,
                        metadata={"operation": "load", "model_id": model_id}
                    )
                    
                    if load_result:
                        self.results.append(load_result)
    
    def _benchmark_advanced_explainability(self) -> None:
        """Benchmark AdvancedExplainability."""
        explainer = AdvancedExplainability(
            feature_names=[f"feature_{i}" for i in range(10)]
        )
        
        data_sizes = [(500, 10), (1000, 10), (2000, 10)]
        
        for data_size in data_sizes:
            
            def benchmark_func():
                data = self._generate_test_data(data_size)
                
                # Create a mock detection result
                from simplified_services.core_detection_service import DetectionResult
                predictions = np.random.randint(0, 2, size=data_size[0])
                scores = np.random.random(size=data_size[0])
                
                result = DetectionResult(
                    algorithm="iforest",
                    contamination=0.1,
                    n_samples=data_size[0],
                    n_anomalies=int(np.sum(predictions)),
                    predictions=predictions,
                    scores=scores,
                    metadata={}
                )
                
                # Explain a sample
                return explainer.explain_prediction(
                    sample=data[0],
                    sample_index=0,
                    detection_result=result,
                    training_data=data
                )
            
            result = self._execute_benchmark(
                func=benchmark_func,
                component_name="AdvancedExplainability",
                test_name=f"explain_prediction_{data_size[0]}x{data_size[1]}",
                data_size=data_size,
                metadata={"operation": "explain_prediction"}
            )
            
            if result:
                self.results.append(result)
    
    def _benchmark_monitoring_alerting(self) -> None:
        """Benchmark MonitoringAlertingSystem."""
        monitoring = MonitoringAlertingSystem()
        
        n_records = [100, 500, 1000]
        
        for n in n_records:
            
            def benchmark_func():
                # Simulate recording multiple detection results
                for i in range(n):
                    # Create mock detection result
                    from simplified_services.core_detection_service import DetectionResult
                    result = DetectionResult(
                        algorithm="iforest",
                        contamination=0.1,
                        n_samples=100,
                        n_anomalies=10,
                        predictions=np.ones(100),
                        scores=np.random.random(100),
                        metadata={}
                    )
                    
                    monitoring.record_detection_result(
                        result, 
                        processing_time=0.1, 
                        source=f"benchmark_{i}"
                    )
                
                return monitoring.get_current_metrics()
            
            result = self._execute_benchmark(
                func=benchmark_func,
                component_name="MonitoringAlertingSystem",
                test_name=f"record_{n}_results",
                data_size=(n, 1),
                metadata={"operation": "record_results", "n_records": n}
            )
            
            if result:
                self.results.append(result)
    
    def _execute_benchmark(
        self,
        func: Callable,
        component_name: str,
        test_name: str,
        data_size: Tuple[int, int],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[BenchmarkResult]:
        """Execute a single benchmark test.
        
        Args:
            func: Function to benchmark
            component_name: Name of the component being tested
            test_name: Name of the specific test
            data_size: Size of test data (samples, features)
            metadata: Additional metadata
            
        Returns:
            Benchmark result or None if failed
        """
        try:
            # Warmup iterations
            for _ in range(self.config.warmup_iterations):
                try:
                    func()
                except:
                    pass
            
            # Clear garbage collection
            gc.collect()
            
            # Record initial system state
            initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute benchmark
            times = []
            memory_peaks = []
            cpu_usages = []
            
            for _ in range(self.config.n_iterations):
                # Record CPU usage before
                cpu_before = psutil.cpu_percent(interval=None)
                
                # Execute function
                start_time = time.time()
                result = func()
                end_time = time.time()
                
                # Record metrics
                execution_time = end_time - start_time
                times.append(execution_time)
                
                if self.config.enable_memory_profiling:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_peaks.append(current_memory - initial_memory)
                
                if self.config.enable_cpu_profiling:
                    cpu_after = psutil.cpu_percent(interval=0.1)
                    cpu_usages.append(max(cpu_before, cpu_after))
            
            # Calculate statistics
            avg_time = np.mean(times)
            avg_memory = np.mean(memory_peaks) if memory_peaks else 0.0
            avg_cpu = np.mean(cpu_usages) if cpu_usages else 0.0
            
            throughput = data_size[0] / avg_time if avg_time > 0 else 0.0
            
            # Create result
            benchmark_result = BenchmarkResult(
                component_name=component_name,
                test_name=test_name,
                data_size=data_size,
                execution_time=avg_time,
                memory_usage_mb=avg_memory,
                cpu_usage_percent=avg_cpu,
                throughput_samples_per_second=throughput,
                metadata=metadata or {}
            )
            
            # Store function result in metadata
            if result is not None:
                if hasattr(result, '__dict__'):
                    benchmark_result.metadata["result_type"] = type(result).__name__
                else:
                    benchmark_result.metadata["result"] = str(result)
            
            print(f"   ✅ {component_name}.{test_name}: {avg_time:.3f}s, {throughput:.0f} samples/s")
            
            return benchmark_result
            
        except Exception as e:
            print(f"   ❌ {component_name}.{test_name}: {str(e)}")
            return None
    
    def _generate_test_data(self, size: Tuple[int, int]) -> npt.NDArray[np.floating]:
        """Generate test data for benchmarking.
        
        Args:
            size: (n_samples, n_features)
            
        Returns:
            Generated test data
        """
        n_samples, n_features = size
        
        # Generate mostly normal data with some anomalies
        normal_samples = int(n_samples * 0.9)
        anomaly_samples = n_samples - normal_samples
        
        normal_data = np.random.normal(0, 1, (normal_samples, n_features))
        anomaly_data = np.random.normal(3, 1, (anomaly_samples, n_features))
        
        data = np.vstack([normal_data, anomaly_data])
        np.random.shuffle(data)
        
        return data
    
    def _generate_time_series_data(self, length: int) -> npt.NDArray[np.floating]:
        """Generate time series data for benchmarking.
        
        Args:
            length: Length of time series
            
        Returns:
            Generated time series data
        """
        # Generate time series with trend, seasonality, and anomalies
        t = np.arange(length)
        
        # Base signal with trend and seasonality
        trend = 0.01 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 100)
        noise = np.random.normal(0, 0.5, length)
        
        ts = trend + seasonal + noise
        
        # Add some anomalies
        anomaly_indices = np.random.choice(length, size=int(length * 0.05), replace=False)
        ts[anomaly_indices] += np.random.normal(5, 1, len(anomaly_indices))
        
        return ts
    
    def _generate_text_data(self, count: int) -> List[str]:
        """Generate text data for benchmarking.
        
        Args:
            count: Number of text samples
            
        Returns:
            Generated text data
        """
        # Generate mostly normal text with some anomalous ones
        normal_texts = []
        anomaly_texts = []
        
        # Normal texts (typical length and content)
        for i in range(int(count * 0.9)):
            length = np.random.randint(50, 200)
            text = " ".join([f"word{j}" for j in range(length)])
            normal_texts.append(text)
        
        # Anomalous texts (very short, very long, or unusual content)
        for i in range(count - len(normal_texts)):
            if i % 3 == 0:
                # Very short
                text = "short"
            elif i % 3 == 1:
                # Very long
                length = np.random.randint(500, 1000)
                text = " ".join([f"longword{j}" for j in range(length)])
            else:
                # Unusual content
                text = "!@#$%^&*()_+" * 20
            
            anomaly_texts.append(text)
        
        all_texts = normal_texts + anomaly_texts
        np.random.shuffle(all_texts)
        
        return all_texts
    
    def _generate_benchmark_report(self) -> None:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return
        
        # Generate JSON report
        json_report = {
            "benchmark_info": {
                "timestamp": time.time(),
                "total_tests": len(self.results),
                "configuration": {
                    "data_sizes": self.config.data_sizes,
                    "algorithms": self.config.algorithms,
                    "contamination_rates": self.config.contamination_rates,
                    "n_iterations": self.config.n_iterations
                }
            },
            "results": [
                {
                    "component_name": r.component_name,
                    "test_name": r.test_name,
                    "data_size": r.data_size,
                    "execution_time": r.execution_time,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                    "throughput_samples_per_second": r.throughput_samples_per_second,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ]
        }
        
        # Save JSON report
        json_path = self.output_dir / "benchmark_report.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report()
        
        print(f"📊 Benchmark report saved to: {self.output_dir}")
    
    def _generate_summary_report(self) -> None:
        """Generate human-readable summary report."""
        summary_lines = []
        summary_lines.append("# Phase 2 Performance Benchmark Report")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        
        # Group results by component
        by_component = {}
        for result in self.results:
            if result.component_name not in by_component:
                by_component[result.component_name] = []
            by_component[result.component_name].append(result)
        
        # Generate summary for each component
        for component, results in by_component.items():
            summary_lines.append(f"## {component}")
            summary_lines.append("-" * 30)
            
            # Calculate component statistics
            avg_time = np.mean([r.execution_time for r in results])
            avg_memory = np.mean([r.memory_usage_mb for r in results])
            avg_throughput = np.mean([r.throughput_samples_per_second for r in results])
            
            summary_lines.append(f"Tests: {len(results)}")
            summary_lines.append(f"Average execution time: {avg_time:.3f}s")
            summary_lines.append(f"Average memory usage: {avg_memory:.1f}MB")
            summary_lines.append(f"Average throughput: {avg_throughput:.0f} samples/s")
            summary_lines.append("")
            
            # Top performing tests
            best_throughput = max(results, key=lambda r: r.throughput_samples_per_second)
            worst_throughput = min(results, key=lambda r: r.throughput_samples_per_second)
            
            summary_lines.append("**Best Performance:**")
            summary_lines.append(f"  {best_throughput.test_name}: {best_throughput.throughput_samples_per_second:.0f} samples/s")
            summary_lines.append("**Worst Performance:**")
            summary_lines.append(f"  {worst_throughput.test_name}: {worst_throughput.throughput_samples_per_second:.0f} samples/s")
            summary_lines.append("")
        
        # Overall statistics
        summary_lines.append("## Overall Statistics")
        summary_lines.append("-" * 20)
        all_times = [r.execution_time for r in self.results]
        all_memory = [r.memory_usage_mb for r in self.results]
        all_throughput = [r.throughput_samples_per_second for r in self.results]
        
        summary_lines.append(f"Total tests executed: {len(self.results)}")
        summary_lines.append(f"Total execution time: {sum(all_times):.1f}s")
        summary_lines.append(f"Average test time: {np.mean(all_times):.3f}s")
        summary_lines.append(f"Peak memory usage: {max(all_memory):.1f}MB")
        summary_lines.append(f"Best throughput: {max(all_throughput):.0f} samples/s")
        summary_lines.append("")
        
        # Save summary report
        summary_path = self.output_dir / "benchmark_summary.md"
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary_lines))
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations.
        
        Returns:
            Performance insights and recommendations
        """
        if not self.results:
            return {"error": "No benchmark results available"}
        
        insights = {
            "total_tests": len(self.results),
            "components_tested": len(set(r.component_name for r in self.results)),
            "performance_summary": {},
            "recommendations": [],
            "bottlenecks": []
        }
        
        # Analyze performance by component
        by_component = {}
        for result in self.results:
            if result.component_name not in by_component:
                by_component[result.component_name] = []
            by_component[result.component_name].append(result)
        
        for component, results in by_component.items():
            avg_time = np.mean([r.execution_time for r in results])
            avg_memory = np.mean([r.memory_usage_mb for r in results])
            avg_throughput = np.mean([r.throughput_samples_per_second for r in results])
            
            insights["performance_summary"][component] = {
                "avg_execution_time": avg_time,
                "avg_memory_usage_mb": avg_memory,
                "avg_throughput_samples_per_second": avg_throughput,
                "test_count": len(results)
            }
            
            # Generate recommendations
            if avg_time > 5.0:
                insights["recommendations"].append(
                    f"{component}: Consider optimization - average execution time is {avg_time:.1f}s"
                )
            
            if avg_memory > 500:
                insights["recommendations"].append(
                    f"{component}: High memory usage detected - {avg_memory:.1f}MB average"
                )
            
            if avg_throughput < 100:
                insights["bottlenecks"].append(
                    f"{component}: Low throughput - {avg_throughput:.0f} samples/s"
                )
        
        return insights