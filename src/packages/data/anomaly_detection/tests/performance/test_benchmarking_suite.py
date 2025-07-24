"""Comprehensive performance benchmarking suite for anomaly detection system."""

import pytest
import time
import json
import csv
import statistics
import numpy as np
import psutil
import gc
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from pathlib import Path

from anomaly_detection.main import create_app
from anomaly_detection.application.services.performance.optimization.performance_optimizer import get_performance_optimizer


@dataclass
class BenchmarkResult:
    """Container for individual benchmark results."""
    test_name: str
    algorithm: str
    data_size: int
    feature_count: int
    response_time_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    accuracy_score: Optional[float]
    anomaly_count: int
    error_occurred: bool
    timestamp: str


@dataclass
class BenchmarkSuite:
    """Container for complete benchmark suite results."""
    suite_name: str
    start_time: str
    end_time: str
    total_duration_sec: float
    environment_info: Dict[str, Any]
    results: List[BenchmarkResult]
    summary_stats: Dict[str, Any]


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking framework."""
    
    def __init__(self, output_dir: str = "reports/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.process = psutil.Process()
        self.results: List[BenchmarkResult] = []
        
        # Performance optimizer integration
        self.performance_optimizer = get_performance_optimizer()
        self.performance_optimizer.start_monitoring()
        
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'performance_optimizer'):
            self.performance_optimizer.stop_monitoring()
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Collect system environment information."""
        import platform
        
        try:
            cpu_info = psutil.cpu_freq()
            cpu_freq = cpu_info.current if cpu_info else "Unknown"
        except:
            cpu_freq = "Unknown"
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": cpu_freq,
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "timestamp": datetime.now().isoformat()
        }
    
    def benchmark_single_request(
        self,
        client: TestClient,
        endpoint: str,
        payload: Dict[str, Any],
        test_name: str,
        algorithm: str = "unknown"
    ) -> BenchmarkResult:
        """Benchmark a single API request."""
        data_size = len(payload.get("data", []))
        feature_count = len(payload.get("data", [[]])[0]) if payload.get("data") else 0
        
        # Record initial state
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        gc.collect()  # Clean garbage before measurement
        
        # Measure CPU usage before request
        cpu_before = self.process.cpu_percent(interval=0.1)
        
        # Execute request with timing
        start_time = time.perf_counter()
        start_timestamp = datetime.now()
        
        error_occurred = False
        response_data = None
        
        try:
            response = client.post(endpoint, json=payload)
            if response.status_code == 200:
                response_data = response.json()
            else:
                error_occurred = True
        except Exception:
            error_occurred = True
        
        end_time = time.perf_counter()
        
        # Measure CPU usage after request
        cpu_after = self.process.cpu_percent(interval=0.1)
        cpu_usage = max(cpu_after, cpu_before)  # Take the higher reading
        
        # Measure memory usage
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_usage = final_memory - initial_memory
        
        # Calculate metrics
        response_time_ms = (end_time - start_time) * 1000
        throughput_samples_per_sec = data_size / (response_time_ms / 1000) if response_time_ms > 0 else 0
        
        # Extract anomaly information
        anomaly_count = 0
        accuracy_score = None
        
        if response_data and not error_occurred:
            anomaly_count = response_data.get("anomalies_detected", 0)
            # Calculate basic accuracy as anomaly rate (for benchmarking purposes)
            if data_size > 0:
                accuracy_score = anomaly_count / data_size
        
        return BenchmarkResult(
            test_name=test_name,
            algorithm=algorithm,
            data_size=data_size,
            feature_count=feature_count,
            response_time_ms=response_time_ms,
            throughput_samples_per_sec=throughput_samples_per_sec,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            accuracy_score=accuracy_score,
            anomaly_count=anomaly_count,
            error_occurred=error_occurred,
            timestamp=start_timestamp.isoformat()
        )
    
    def run_algorithm_comparison_benchmark(
        self,
        client: TestClient,
        test_datasets: Dict[str, List[List[float]]],
        algorithms: List[str]
    ) -> List[BenchmarkResult]:
        """Run comprehensive algorithm comparison benchmark."""
        results = []
        
        for dataset_name, data in test_datasets.items():
            for algorithm in algorithms:
                # Run multiple iterations for statistical significance
                for iteration in range(3):
                    test_name = f"algorithm_comparison_{dataset_name}_{algorithm}_iter{iteration+1}"
                    
                    payload = {
                        "data": data,
                        "algorithm": algorithm,
                        "contamination": 0.1
                    }
                    
                    result = self.benchmark_single_request(
                        client=client,
                        endpoint="/api/v1/detection/detect",
                        payload=payload,
                        test_name=test_name,
                        algorithm=algorithm
                    )
                    
                    results.append(result)
                    self.results.append(result)
                    
                    # Small delay between iterations
                    time.sleep(0.5)
        
        return results
    
    def run_data_size_scaling_benchmark(
        self,
        client: TestClient,
        base_algorithm: str = "isolation_forest"
    ) -> List[BenchmarkResult]:
        """Run data size scaling benchmark."""
        results = []
        
        # Generate datasets of increasing size
        np.random.seed(42)
        dataset_configs = [
            (50, 5, "small"),
            (100, 10, "medium"),
            (250, 15, "large"),
            (500, 20, "xlarge"),
            (1000, 25, "xxlarge")
        ]
        
        for samples, features, size_name in dataset_configs:
            # Generate dataset with some anomalies
            data = np.random.normal(0, 1, (samples, features))
            
            # Add anomalies (last 10% scaled up)
            anomaly_count = samples // 10
            for i in range(-anomaly_count, 0):
                data[i] *= 3  # Make anomalous
            
            data_list = data.tolist()
            
            # Run benchmark
            test_name = f"data_scaling_{size_name}_{samples}x{features}"
            
            payload = {
                "data": data_list,
                "algorithm": base_algorithm,
                "contamination": 0.1
            }
            
            result = self.benchmark_single_request(
                client=client,
                endpoint="/api/v1/detection/detect",
                payload=payload,
                test_name=test_name,
                algorithm=base_algorithm
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def run_ensemble_performance_benchmark(
        self,
        client: TestClient,
        test_data: List[List[float]]
    ) -> List[BenchmarkResult]:
        """Run ensemble method performance benchmark."""
        results = []
        
        ensemble_configs = [
            (["isolation_forest", "one_class_svm"], "majority", "ensemble_2alg_majority"),
            (["isolation_forest", "one_class_svm", "local_outlier_factor"], "majority", "ensemble_3alg_majority"),
            (["isolation_forest", "one_class_svm"], "average", "ensemble_2alg_average"),
        ]
        
        for algorithms, method, test_name in ensemble_configs:
            payload = {
                "data": test_data,
                "algorithms": algorithms,
                "method": method,
                "contamination": 0.1
            }
            
            result = self.benchmark_single_request(
                client=client,
                endpoint="/api/v1/detection/ensemble",
                payload=payload,
                test_name=test_name,
                algorithm=f"ensemble_{method}"
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def run_memory_efficiency_benchmark(
        self,
        client: TestClient,
        algorithm: str = "isolation_forest"
    ) -> List[BenchmarkResult]:
        """Run memory efficiency benchmark with varying data sizes."""
        results = []
        
        # Memory efficiency test with progressively larger datasets
        sizes = [100, 500, 1000, 2000, 5000]
        
        for size in sizes:
            # Generate large dataset
            np.random.seed(42)
            data = np.random.normal(0, 1, (size, 20)).tolist()
            
            # Force garbage collection before test
            gc.collect()
            
            test_name = f"memory_efficiency_{algorithm}_{size}_samples"
            
            payload = {
                "data": data,
                "algorithm": algorithm,
                "contamination": 0.1
            }
            
            result = self.benchmark_single_request(
                client=client,
                endpoint="/api/v1/detection/detect",
                payload=payload,
                test_name=test_name,
                algorithm=algorithm
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def calculate_summary_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics for benchmark results."""
        if not results:
            return {}
        
        # Group results by algorithm
        algorithm_stats = {}
        
        for result in results:
            if result.algorithm not in algorithm_stats:
                algorithm_stats[result.algorithm] = {
                    "response_times": [],
                    "throughputs": [],
                    "memory_usage": [],
                    "cpu_usage": [],
                    "data_sizes": [],
                    "error_count": 0,
                    "total_count": 0
                }
            
            stats = algorithm_stats[result.algorithm]
            stats["response_times"].append(result.response_time_ms)
            stats["throughputs"].append(result.throughput_samples_per_sec)
            stats["memory_usage"].append(result.memory_usage_mb)
            stats["cpu_usage"].append(result.cpu_usage_percent)
            stats["data_sizes"].append(result.data_size)
            
            if result.error_occurred:
                stats["error_count"] += 1
            stats["total_count"] += 1
        
        # Calculate statistics for each algorithm
        summary = {
            "total_tests": len(results),
            "algorithms_tested": list(algorithm_stats.keys()),
            "algorithm_performance": {}
        }
        
        for algorithm, stats in algorithm_stats.items():
            if stats["response_times"]:
                algorithm_summary = {
                    "test_count": stats["total_count"],
                    "error_rate": stats["error_count"] / stats["total_count"],
                    "response_time_ms": {
                        "mean": statistics.mean(stats["response_times"]),
                        "median": statistics.median(stats["response_times"]),
                        "min": min(stats["response_times"]),
                        "max": max(stats["response_times"]),
                        "std": statistics.stdev(stats["response_times"]) if len(stats["response_times"]) > 1 else 0
                    },
                    "throughput_samples_per_sec": {
                        "mean": statistics.mean(stats["throughputs"]),
                        "median": statistics.median(stats["throughputs"]),
                        "min": min(stats["throughputs"]),
                        "max": max(stats["throughputs"])
                    },
                    "memory_usage_mb": {
                        "mean": statistics.mean(stats["memory_usage"]),
                        "max": max(stats["memory_usage"])
                    },
                    "cpu_usage_percent": {
                        "mean": statistics.mean(stats["cpu_usage"]),
                        "max": max(stats["cpu_usage"])
                    },
                    "data_sizes_tested": list(set(stats["data_sizes"]))
                }
                
                summary["algorithm_performance"][algorithm] = algorithm_summary
        
        return summary
    
    def export_results(self, suite_name: str, format: str = "all") -> Dict[str, str]:
        """Export benchmark results in various formats."""
        if not self.results:
            return {}
        
        # Create benchmark suite
        suite = BenchmarkSuite(
            suite_name=suite_name,
            start_time=min(r.timestamp for r in self.results),
            end_time=max(r.timestamp for r in self.results),
            total_duration_sec=0,  # Will be calculated
            environment_info=self.get_environment_info(),
            results=self.results,
            summary_stats=self.calculate_summary_statistics(self.results)
        )
        
        # Calculate total duration
        start_dt = datetime.fromisoformat(suite.start_time)
        end_dt = datetime.fromisoformat(suite.end_time)
        suite.total_duration_sec = (end_dt - start_dt).total_seconds()
        
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format in ["json", "all"]:
            # Export as JSON
            json_file = self.output_dir / f"{suite_name}_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(asdict(suite), f, indent=2, default=str)
            exported_files["json"] = str(json_file)
        
        if format in ["csv", "all"]:
            # Export detailed results as CSV
            csv_file = self.output_dir / f"{suite_name}_{timestamp}_results.csv"
            with open(csv_file, 'w', newline='') as f:
                if self.results:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                    writer.writeheader()
                    for result in self.results:
                        writer.writerow(asdict(result))
            exported_files["csv"] = str(csv_file)
        
        if format in ["html", "all"]:
            # Export as HTML report
            html_file = self.output_dir / f"{suite_name}_{timestamp}_report.html"
            with open(html_file, 'w') as f:
                f.write(self._generate_html_report(suite))
            exported_files["html"] = str(html_file)
        
        return exported_files
    
    def _generate_html_report(self, suite: BenchmarkSuite) -> str:
        """Generate HTML benchmark report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Benchmark Report - {suite.suite_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px; border-radius: 3px; }}
                .algorithm {{ margin: 15px 0; padding: 15px; border: 1px solid #ccc; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Benchmark Report</h1>
                <h2>{suite.suite_name}</h2>
                <p><strong>Duration:</strong> {suite.total_duration_sec:.2f} seconds</p>
                <p><strong>Tests Run:</strong> {len(suite.results)}</p>
                <p><strong>Period:</strong> {suite.start_time} to {suite.end_time}</p>
            </div>
            
            <div class="section">
                <h3>Environment Information</h3>
                <div class="metric">
                    <strong>Platform:</strong> {suite.environment_info.get('platform', 'Unknown')}<br>
                    <strong>Python Version:</strong> {suite.environment_info.get('python_version', 'Unknown')}<br>
                    <strong>CPU Cores:</strong> {suite.environment_info.get('cpu_count', 'Unknown')}<br>
                    <strong>Total Memory:</strong> {suite.environment_info.get('total_memory_gb', 0):.1f} GB<br>
                    <strong>Available Memory:</strong> {suite.environment_info.get('available_memory_gb', 0):.1f} GB
                </div>
            </div>
            
            <div class="section">
                <h3>Algorithm Performance Summary</h3>
        """
        
        # Add algorithm performance sections
        for algorithm, stats in suite.summary_stats.get("algorithm_performance", {}).items():
            html += f"""
                <div class="algorithm">
                    <h4>{algorithm}</h4>
                    <div class="metric">
                        <strong>Test Count:</strong> {stats['test_count']}<br>
                        <strong>Error Rate:</strong> {stats['error_rate']:.1%}<br>
                        <strong>Avg Response Time:</strong> {stats['response_time_ms']['mean']:.2f} ms<br>
                        <strong>Avg Throughput:</strong> {stats['throughput_samples_per_sec']['mean']:.2f} samples/sec<br>
                        <strong>Peak Memory Usage:</strong> {stats['memory_usage_mb']['max']:.2f} MB<br>
                        <strong>Peak CPU Usage:</strong> {stats['cpu_usage_percent']['max']:.1f}%
                    </div>
                </div>
            """
        
        # Add detailed results table
        html += """
            </div>
            
            <div class="section">
                <h3>Detailed Results</h3>
                <table>
                    <tr>
                        <th>Test Name</th>
                        <th>Algorithm</th>
                        <th>Data Size</th>
                        <th>Response Time (ms)</th>
                        <th>Throughput (samples/sec)</th>
                        <th>Memory (MB)</th>
                        <th>CPU (%)</th>
                        <th>Anomalies</th>
                        <th>Error</th>
                    </tr>
        """
        
        for result in suite.results:
            error_class = "style='background-color: #ffe6e6;'" if result.error_occurred else ""
            html += f"""
                    <tr {error_class}>
                        <td>{result.test_name}</td>
                        <td>{result.algorithm}</td>
                        <td>{result.data_size}</td>
                        <td>{result.response_time_ms:.2f}</td>
                        <td>{result.throughput_samples_per_sec:.2f}</td>
                        <td>{result.memory_usage_mb:.2f}</td>
                        <td>{result.cpu_usage_percent:.1f}</td>
                        <td>{result.anomaly_count}</td>
                        <td>{'Yes' if result.error_occurred else 'No'}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html


@pytest.mark.performance
@pytest.mark.benchmark
class TestPerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking test suite."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('anomaly_detection.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.api.cors_origins = ["*"]
            mock_get_settings.return_value = mock_settings
            
            self.app = create_app()
            self.client = TestClient(self.app)
        
        self.benchmarker = PerformanceBenchmarker()
        
        # Generate test datasets
        np.random.seed(42)
        self.test_datasets = {
            "small": self._generate_dataset_with_anomalies(50, 5),
            "medium": self._generate_dataset_with_anomalies(200, 10),
            "large": self._generate_dataset_with_anomalies(500, 15)
        }
    
    def _generate_dataset_with_anomalies(self, samples: int, features: int) -> List[List[float]]:
        """Generate dataset with known anomalies."""
        data = np.random.normal(0, 1, (samples, features))
        
        # Add anomalies (last 10% scaled up)
        anomaly_count = samples // 10
        for i in range(-anomaly_count, 0):
            data[i] *= 3  # Make anomalous
        
        return data.tolist()
    
    def test_comprehensive_algorithm_benchmark(self):
        """Run comprehensive algorithm performance benchmark."""
        algorithms = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
        
        results = self.benchmarker.run_algorithm_comparison_benchmark(
            client=self.client,
            test_datasets=self.test_datasets,
            algorithms=algorithms
        )
        
        # Export results
        exported_files = self.benchmarker.export_results(
            suite_name="algorithm_comparison_benchmark",
            format="all"
        )
        
        # Verify benchmark results
        assert len(results) > 0, "No benchmark results generated"
        
        # Check that all algorithms were tested
        algorithms_tested = set(r.algorithm for r in results)
        assert len(algorithms_tested) == len(algorithms), f"Not all algorithms tested: {algorithms_tested}"
        
        # Check performance criteria
        for result in results:
            assert not result.error_occurred, f"Benchmark error in {result.test_name}"
            assert result.response_time_ms > 0, f"Invalid response time: {result.response_time_ms}"
            assert result.throughput_samples_per_sec > 0, f"Invalid throughput: {result.throughput_samples_per_sec}"
        
        # Print summary
        summary_stats = self.benchmarker.calculate_summary_statistics(results)
        self._print_benchmark_summary("Algorithm Comparison", summary_stats, exported_files)
    
    def test_data_size_scaling_benchmark(self):
        """Run data size scaling performance benchmark."""
        results = self.benchmarker.run_data_size_scaling_benchmark(
            client=self.client,
            base_algorithm="isolation_forest"
        )
        
        # Export results
        exported_files = self.benchmarker.export_results(
            suite_name="data_size_scaling_benchmark",
            format="all"
        )
        
        # Verify scaling characteristics
        assert len(results) >= 3, "Insufficient scaling test points"
        
        # Check that response time scales reasonably with data size
        results_by_size = sorted(results, key=lambda x: x.data_size)
        
        for i, result in enumerate(results_by_size):
            assert not result.error_occurred, f"Scaling benchmark error: {result.test_name}"
            
            # Verify reasonable performance scaling
            samples_per_ms = result.data_size / result.response_time_ms
            assert samples_per_ms > 0.1, f"Poor scaling performance: {samples_per_ms} samples/ms"
            
            print(f"Data size {result.data_size}: {result.response_time_ms:.2f}ms, "
                  f"{result.throughput_samples_per_sec:.2f} samples/sec")
        
        # Calculate scaling factor
        scaling_analysis = self._analyze_scaling_performance(results_by_size)
        
        # Print summary
        summary_stats = self.benchmarker.calculate_summary_statistics(results)
        summary_stats["scaling_analysis"] = scaling_analysis
        
        self._print_benchmark_summary("Data Size Scaling", summary_stats, exported_files)
    
    def test_ensemble_performance_benchmark(self):
        """Run ensemble method performance benchmark."""
        results = self.benchmarker.run_ensemble_performance_benchmark(
            client=self.client,
            test_data=self.test_datasets["medium"]
        )
        
        # Export results
        exported_files = self.benchmarker.export_results(
            suite_name="ensemble_performance_benchmark",
            format="all"
        )
        
        # Verify ensemble performance
        assert len(results) > 0, "No ensemble benchmark results"
        
        for result in results:
            assert not result.error_occurred, f"Ensemble benchmark error: {result.test_name}"
            
            # Ensemble methods should be slower but still reasonable
            assert result.response_time_ms < 30000, f"Ensemble too slow: {result.response_time_ms}ms"
            
            print(f"Ensemble {result.algorithm}: {result.response_time_ms:.2f}ms, "
                  f"{result.anomaly_count} anomalies detected")
        
        # Print summary
        summary_stats = self.benchmarker.calculate_summary_statistics(results)
        self._print_benchmark_summary("Ensemble Performance", summary_stats, exported_files)
    
    def test_memory_efficiency_benchmark(self):
        """Run memory efficiency benchmark."""
        results = self.benchmarker.run_memory_efficiency_benchmark(
            client=self.client,
            algorithm="isolation_forest"
        )
        
        # Export results
        exported_files = self.benchmarker.export_results(
            suite_name="memory_efficiency_benchmark",
            format="all"
        )
        
        # Verify memory efficiency
        assert len(results) > 0, "No memory benchmark results"
        
        for result in results:
            assert not result.error_occurred, f"Memory benchmark error: {result.test_name}"
            
            # Check memory usage is reasonable (should not exceed 100MB per request)
            memory_per_sample = result.memory_usage_mb / result.data_size if result.data_size > 0 else 0
            assert memory_per_sample < 0.1, f"Memory usage too high: {memory_per_sample:.4f}MB per sample"
            
            print(f"Data size {result.data_size}: {result.memory_usage_mb:.2f}MB, "
                  f"{memory_per_sample:.6f}MB per sample")
        
        # Print summary
        summary_stats = self.benchmarker.calculate_summary_statistics(results)
        self._print_benchmark_summary("Memory Efficiency", summary_stats, exported_files)
    
    def test_full_benchmark_suite(self):
        """Run complete benchmark suite with all tests."""
        print("\n" + "="*80)
        print("RUNNING COMPLETE PERFORMANCE BENCHMARK SUITE")
        print("="*80)
        
        suite_start = time.time()
        
        # Run all benchmark categories
        benchmarks = [
            ("Algorithm Comparison", self.benchmarker.run_algorithm_comparison_benchmark, 
             (self.client, {"medium": self.test_datasets["medium"]}, ["isolation_forest", "one_class_svm"])),
            ("Data Size Scaling", self.benchmarker.run_data_size_scaling_benchmark, 
             (self.client,)),
            ("Ensemble Performance", self.benchmarker.run_ensemble_performance_benchmark, 
             (self.client, self.test_datasets["small"])),
            ("Memory Efficiency", self.benchmarker.run_memory_efficiency_benchmark, 
             (self.client,))
        ]
        
        all_results = []
        
        for benchmark_name, benchmark_func, args in benchmarks:
            print(f"\nRunning {benchmark_name} benchmark...")
            benchmark_results = benchmark_func(*args)
            all_results.extend(benchmark_results)
        
        suite_end = time.time()
        suite_duration = suite_end - suite_start
        
        # Export comprehensive results
        exported_files = self.benchmarker.export_results(
            suite_name="complete_performance_benchmark_suite",
            format="all"
        )
        
        # Calculate overall statistics
        summary_stats = self.benchmarker.calculate_summary_statistics(all_results)
        summary_stats["suite_duration_seconds"] = suite_duration
        
        # Print comprehensive summary
        print(f"\n" + "="*80)
        print("COMPLETE BENCHMARK SUITE RESULTS")
        print("="*80)
        print(f"Total Duration: {suite_duration:.2f} seconds")
        print(f"Total Tests: {len(all_results)}")
        print(f"Algorithms Tested: {summary_stats.get('algorithms_tested', [])}")
        
        # Print top-level performance metrics
        if summary_stats.get("algorithm_performance"):
            print(f"\nPerformance Summary:")
            for algorithm, stats in summary_stats["algorithm_performance"].items():
                print(f"  {algorithm}:")
                print(f"    Avg Response Time: {stats['response_time_ms']['mean']:.2f}ms")
                print(f"    Avg Throughput: {stats['throughput_samples_per_sec']['mean']:.2f} samples/sec")
                print(f"    Error Rate: {stats['error_rate']:.1%}")
        
        print(f"\nExported Files:")
        for format_type, file_path in exported_files.items():
            print(f"  {format_type.upper()}: {file_path}")
        
        # Overall assertions
        assert len(all_results) > 10, "Insufficient benchmark coverage"
        error_count = sum(1 for r in all_results if r.error_occurred)
        error_rate = error_count / len(all_results)
        assert error_rate < 0.10, f"Too many benchmark errors: {error_rate:.1%}"
        
        print(f"\n✅ Complete benchmark suite passed with {error_rate:.1%} error rate")
    
    def _analyze_scaling_performance(self, results_by_size: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze scaling performance characteristics."""
        if len(results_by_size) < 2:
            return {"error": "Insufficient data for scaling analysis"}
        
        data_sizes = [r.data_size for r in results_by_size]
        response_times = [r.response_time_ms for r in results_by_size]
        
        # Calculate scaling factor using log-log regression
        log_sizes = np.log(data_sizes)
        log_times = np.log(response_times)
        
        # Linear regression on log-log data
        scaling_factor = np.polyfit(log_sizes, log_times, 1)[0]
        
        # Calculate efficiency metrics
        throughput_values = [r.throughput_samples_per_sec for r in results_by_size]
        efficiency_decline = (throughput_values[0] - throughput_values[-1]) / throughput_values[0] if throughput_values else 0
        
        return {
            "scaling_factor": scaling_factor,
            "scaling_interpretation": (
                "sublinear (better than linear)" if scaling_factor < 0.9 else
                "linear" if scaling_factor < 1.1 else
                "superlinear (worse than linear)"
            ),
            "efficiency_decline_percent": efficiency_decline * 100,
            "data_size_range": f"{min(data_sizes)}-{max(data_sizes)} samples",
            "response_time_range": f"{min(response_times):.2f}-{max(response_times):.2f}ms"
        }
    
    def _print_benchmark_summary(self, benchmark_name: str, summary_stats: Dict[str, Any], exported_files: Dict[str, str]):
        """Print formatted benchmark summary."""
        print(f"\n" + "="*60)
        print(f"{benchmark_name.upper()} BENCHMARK RESULTS")
        print("="*60)
        
        print(f"Total Tests: {summary_stats.get('total_tests', 0)}")
        print(f"Algorithms: {', '.join(summary_stats.get('algorithms_tested', []))}")
        
        if summary_stats.get("algorithm_performance"):
            for algorithm, stats in summary_stats["algorithm_performance"].items():
                print(f"\n{algorithm}:")
                print(f"  Tests: {stats['test_count']}")
                print(f"  Error Rate: {stats['error_rate']:.1%}")
                print(f"  Avg Response Time: {stats['response_time_ms']['mean']:.2f}ms")
                print(f"  Response Time Range: {stats['response_time_ms']['min']:.2f}-{stats['response_time_ms']['max']:.2f}ms")
                print(f"  Avg Throughput: {stats['throughput_samples_per_sec']['mean']:.2f} samples/sec")
                print(f"  Peak Memory Usage: {stats['memory_usage_mb']['max']:.2f}MB")
                print(f"  Peak CPU Usage: {stats['cpu_usage_percent']['max']:.1f}%")
        
        if "scaling_analysis" in summary_stats:
            scaling = summary_stats["scaling_analysis"]
            print(f"\nScaling Analysis:")
            print(f"  Scaling Factor: {scaling.get('scaling_factor', 0):.2f} ({scaling.get('scaling_interpretation', 'unknown')})")
            print(f"  Efficiency Decline: {scaling.get('efficiency_decline_percent', 0):.1f}%")
            print(f"  Data Size Range: {scaling.get('data_size_range', 'unknown')}")
        
        print(f"\nExported Files:")
        for format_type, file_path in exported_files.items():
            print(f"  {format_type.upper()}: {file_path}")


if __name__ == "__main__":
    # Run complete benchmark suite
    pytest.main([
        __file__ + "::TestPerformanceBenchmarkSuite::test_full_benchmark_suite",
        "-v", "-s", "--tb=short"
    ])