"""Performance testing and benchmarking service for comprehensive performance analysis."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import psutil

try:
    from memory_profiler import profile

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

    def profile(func):
        """Dummy decorator when memory_profiler is not available."""
        return func


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""

    benchmark_id: UUID = field(default_factory=uuid4)
    benchmark_name: str = ""
    description: str = ""

    # Test parameters
    dataset_sizes: list[int] = field(default_factory=lambda: [1000, 5000, 10000, 50000])
    feature_dimensions: list[int] = field(default_factory=lambda: [10, 50, 100, 500])
    contamination_rates: list[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.2]
    )
    algorithms: list[str] = field(default_factory=list)

    # Performance thresholds
    max_execution_time_seconds: float = 300.0
    max_memory_usage_mb: float = 4096.0
    min_throughput_samples_per_second: float = 10.0

    # Test configuration
    iterations: int = 5
    warmup_iterations: int = 2
    timeout_seconds: float = 600.0
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    enable_scalability_testing: bool = True

    # Output configuration
    save_detailed_results: bool = True
    generate_plots: bool = True
    export_formats: list[str] = field(default_factory=lambda: ["json", "csv", "html"])


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test run."""

    test_id: UUID = field(default_factory=uuid4)
    algorithm_name: str = ""
    dataset_size: int = 0
    feature_dimension: int = 0
    contamination_rate: float = 0.0

    # Execution metrics
    execution_time_seconds: float = 0.0
    training_time_seconds: float = 0.0
    prediction_time_seconds: float = 0.0
    preprocessing_time_seconds: float = 0.0

    # Memory metrics
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0

    # CPU metrics
    cpu_usage_percent: float = 0.0
    cpu_time_seconds: float = 0.0

    # Throughput metrics
    training_throughput: float = 0.0  # samples/second
    prediction_throughput: float = 0.0  # samples/second

    # Quality metrics
    accuracy_score: float = 0.0
    precision_score: float = 0.0
    recall_score: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0

    # Resource metrics
    disk_io_mb: float = 0.0
    network_io_mb: float = 0.0

    # Scalability metrics
    scalability_factor: float = 1.0
    efficiency_ratio: float = 1.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    test_environment: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    success: bool = True


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""

    suite_id: UUID = field(default_factory=uuid4)
    suite_name: str = ""
    description: str = ""
    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # Results
    individual_results: list[PerformanceMetrics] = field(default_factory=list)
    summary_stats: dict[str, Any] = field(default_factory=dict)
    comparative_analysis: dict[str, Any] = field(default_factory=dict)

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    total_duration_seconds: float = 0.0

    # Environment
    test_environment: dict[str, Any] = field(default_factory=dict)
    system_info: dict[str, Any] = field(default_factory=dict)

    # Quality assessment
    overall_score: float = 0.0
    performance_grade: str = "B"
    recommendations: list[str] = field(default_factory=list)


class PerformanceBenchmarkingService:
    """Comprehensive performance testing and benchmarking service."""

    def __init__(self, storage_path: Path):
        """Initialize performance benchmarking service."""
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.benchmark_results: dict[UUID, BenchmarkSuite] = {}
        self.active_benchmarks: set[UUID] = set()

        # Performance tracking
        self.performance_history: list[PerformanceMetrics] = []
        self.baseline_metrics: dict[str, PerformanceMetrics] = {}

        # System monitoring
        self.system_monitor = SystemMonitor()

    async def create_benchmark_suite(
        self, suite_name: str, description: str, config: BenchmarkConfig
    ) -> UUID:
        """Create new benchmark suite."""
        suite = BenchmarkSuite(
            suite_name=suite_name,
            description=description,
            config=config,
            test_environment=await self._get_test_environment(),
            system_info=await self._get_system_info(),
        )

        self.benchmark_results[suite.suite_id] = suite
        return suite.suite_id

    async def run_comprehensive_benchmark(
        self,
        suite_id: UUID,
        algorithms: list[str],
        datasets: list[pd.DataFrame] | None = None,
    ) -> BenchmarkSuite:
        """Run comprehensive performance benchmark."""
        if suite_id not in self.benchmark_results:
            raise ValueError(f"Benchmark suite {suite_id} not found")

        suite = self.benchmark_results[suite_id]
        self.active_benchmarks.add(suite_id)

        try:
            # Generate synthetic datasets if not provided
            if datasets is None:
                datasets = await self._generate_benchmark_datasets(suite.config)

            # Run benchmarks for each algorithm
            for algorithm in algorithms:
                await self._benchmark_algorithm(suite, algorithm, datasets)

            # Calculate summary statistics
            await self._calculate_summary_statistics(suite)

            # Perform comparative analysis
            await self._perform_comparative_analysis(suite)

            # Generate recommendations
            await self._generate_recommendations(suite)

            # Complete suite
            suite.end_time = datetime.utcnow()
            suite.total_duration_seconds = (
                suite.end_time - suite.start_time
            ).total_seconds()

            return suite

        finally:
            self.active_benchmarks.discard(suite_id)

    async def run_scalability_test(
        self,
        algorithm_name: str,
        base_dataset_size: int = 1000,
        scale_factors: list[int] = None,
        feature_dimension: int = 10,
    ) -> dict[str, Any]:
        """Run scalability test for algorithm."""
        if scale_factors is None:
            scale_factors = [1, 2, 4, 8, 16, 32]

        results = []
        baseline_time = None

        for scale_factor in scale_factors:
            dataset_size = base_dataset_size * scale_factor

            # Generate test dataset
            dataset = await self._generate_synthetic_dataset(
                size=dataset_size, features=feature_dimension, contamination=0.1
            )

            # Run benchmark
            metrics = await self._benchmark_single_run(
                algorithm_name=algorithm_name,
                dataset=dataset,
                dataset_size=dataset_size,
                feature_dimension=feature_dimension,
                contamination_rate=0.1,
            )

            if baseline_time is None:
                baseline_time = metrics.execution_time_seconds

            # Calculate scalability metrics
            metrics.scalability_factor = scale_factor
            metrics.efficiency_ratio = baseline_time / (
                metrics.execution_time_seconds / scale_factor
            )

            results.append(metrics)

        return {
            "algorithm": algorithm_name,
            "results": results,
            "scalability_summary": await self._analyze_scalability(results),
        }

    async def run_memory_stress_test(
        self,
        algorithm_name: str,
        max_dataset_size: int = 1000000,
        memory_limit_mb: float = 8192.0,
    ) -> dict[str, Any]:
        """Run memory stress test."""
        results = []
        current_size = 1000

        while current_size <= max_dataset_size:
            # Generate large dataset
            dataset = await self._generate_synthetic_dataset(
                size=current_size, features=50, contamination=0.1
            )

            # Monitor memory during execution
            memory_before = psutil.virtual_memory().used / 1024 / 1024

            try:
                metrics = await self._benchmark_single_run(
                    algorithm_name=algorithm_name,
                    dataset=dataset,
                    dataset_size=current_size,
                    feature_dimension=50,
                    contamination_rate=0.1,
                )

                memory_after = psutil.virtual_memory().used / 1024 / 1024
                metrics.memory_growth_mb = memory_after - memory_before

                results.append(metrics)

                # Check if we've hit memory limit
                if metrics.peak_memory_mb > memory_limit_mb:
                    break

                # Increase dataset size
                current_size = int(current_size * 1.5)

            except MemoryError:
                break
            except Exception:
                break

        return {
            "algorithm": algorithm_name,
            "max_dataset_size_tested": current_size,
            "memory_limit_mb": memory_limit_mb,
            "results": results,
            "memory_analysis": await self._analyze_memory_usage(results),
        }

    async def run_throughput_benchmark(
        self,
        algorithms: list[str],
        dataset_sizes: list[int] = None,
        duration_seconds: int = 60,
    ) -> dict[str, Any]:
        """Run throughput benchmark."""
        if dataset_sizes is None:
            dataset_sizes = [1000, 5000, 10000, 25000]

        results = {}

        for algorithm in algorithms:
            algorithm_results = []

            for size in dataset_sizes:
                # Generate dataset
                dataset = await self._generate_synthetic_dataset(
                    size=size, features=20, contamination=0.1
                )

                # Measure throughput
                throughput_metrics = await self._measure_throughput(
                    algorithm_name=algorithm,
                    dataset=dataset,
                    duration_seconds=duration_seconds,
                )

                algorithm_results.append(throughput_metrics)

            results[algorithm] = algorithm_results

        return {
            "results": results,
            "throughput_analysis": await self._analyze_throughput(results),
        }

    async def compare_algorithms(
        self,
        algorithms: list[str],
        dataset_sizes: list[int] = None,
        metrics: list[str] = None,
    ) -> dict[str, Any]:
        """Compare algorithms across multiple metrics."""
        if dataset_sizes is None:
            dataset_sizes = [1000, 5000, 10000]

        if metrics is None:
            metrics = ["execution_time", "memory_usage", "accuracy", "throughput"]

        comparison_results = defaultdict(dict)

        for algorithm in algorithms:
            for size in dataset_sizes:
                # Generate test dataset
                dataset = await self._generate_synthetic_dataset(
                    size=size, features=20, contamination=0.1
                )

                # Run benchmark
                result = await self._benchmark_single_run(
                    algorithm_name=algorithm,
                    dataset=dataset,
                    dataset_size=size,
                    feature_dimension=20,
                    contamination_rate=0.1,
                )

                comparison_results[algorithm][size] = result

        # Generate comparative analysis
        analysis = await self._generate_algorithm_comparison(
            comparison_results, metrics
        )

        return {
            "algorithms": algorithms,
            "dataset_sizes": dataset_sizes,
            "metrics": metrics,
            "results": dict(comparison_results),
            "analysis": analysis,
        }

    async def get_performance_trends(
        self, algorithm_name: str | None = None, days: int = 30
    ) -> dict[str, Any]:
        """Get performance trends over time."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Filter historical data
        filtered_history = [
            m
            for m in self.performance_history
            if m.timestamp >= cutoff_date
            and (algorithm_name is None or m.algorithm_name == algorithm_name)
        ]

        if not filtered_history:
            return {"message": "No historical data available"}

        # Analyze trends
        trends = await self._analyze_performance_trends(filtered_history)

        return {
            "algorithm": algorithm_name,
            "period_days": days,
            "data_points": len(filtered_history),
            "trends": trends,
            "recommendations": await self._generate_trend_recommendations(trends),
        }

    async def generate_benchmark_report(
        self, suite_id: UUID, output_path: Path, format: str = "html"
    ) -> Path:
        """Generate comprehensive benchmark report."""
        if suite_id not in self.benchmark_results:
            raise ValueError(f"Benchmark suite {suite_id} not found")

        suite = self.benchmark_results[suite_id]

        if format == "html":
            return await self._generate_html_report(suite, output_path)
        elif format == "json":
            return await self._generate_json_report(suite, output_path)
        elif format == "csv":
            return await self._generate_csv_report(suite, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    # Private methods

    async def _benchmark_algorithm(
        self, suite: BenchmarkSuite, algorithm: str, datasets: list[pd.DataFrame]
    ) -> None:
        """Benchmark single algorithm across all test configurations."""
        config = suite.config

        for dataset in datasets:
            for contamination_rate in config.contamination_rates:
                # Run multiple iterations
                iteration_results = []

                for _i in range(config.iterations):
                    try:
                        metrics = await self._benchmark_single_run(
                            algorithm_name=algorithm,
                            dataset=dataset,
                            dataset_size=len(dataset),
                            feature_dimension=dataset.shape[1],
                            contamination_rate=contamination_rate,
                        )

                        iteration_results.append(metrics)

                    except Exception as e:
                        # Record failed run
                        metrics = PerformanceMetrics(
                            algorithm_name=algorithm,
                            dataset_size=len(dataset),
                            feature_dimension=dataset.shape[1],
                            contamination_rate=contamination_rate,
                            error_message=str(e),
                            success=False,
                        )
                        iteration_results.append(metrics)

                # Calculate average metrics for this configuration
                if iteration_results:
                    avg_metrics = await self._calculate_average_metrics(
                        iteration_results
                    )
                    suite.individual_results.append(avg_metrics)
                    self.performance_history.append(avg_metrics)

    async def _benchmark_single_run(
        self,
        algorithm_name: str,
        dataset: pd.DataFrame,
        dataset_size: int,
        feature_dimension: int,
        contamination_rate: float,
    ) -> PerformanceMetrics:
        """Run single benchmark iteration."""
        metrics = PerformanceMetrics(
            algorithm_name=algorithm_name,
            dataset_size=dataset_size,
            feature_dimension=feature_dimension,
            contamination_rate=contamination_rate,
        )

        try:
            # Start system monitoring
            monitor = await self.system_monitor.start_monitoring()

            start_time = time.time()
            psutil.virtual_memory().used / 1024 / 1024

            # Run the actual algorithm (placeholder - would integrate with actual detection service)
            await self._run_detection_algorithm(
                algorithm_name, dataset, contamination_rate
            )

            end_time = time.time()
            memory_after = psutil.virtual_memory().used / 1024 / 1024

            # Stop monitoring and collect metrics
            monitoring_data = await self.system_monitor.stop_monitoring(monitor)

            # Calculate metrics
            metrics.execution_time_seconds = end_time - start_time
            metrics.peak_memory_mb = monitoring_data.get("peak_memory_mb", memory_after)
            metrics.average_memory_mb = monitoring_data.get(
                "avg_memory_mb", memory_after
            )
            metrics.cpu_usage_percent = monitoring_data.get("avg_cpu_percent", 0.0)
            metrics.training_throughput = dataset_size / metrics.execution_time_seconds

            # Calculate quality metrics (placeholder)
            metrics.accuracy_score = 0.85 + np.random.normal(0, 0.05)
            metrics.precision_score = 0.80 + np.random.normal(0, 0.05)
            metrics.recall_score = 0.75 + np.random.normal(0, 0.05)
            metrics.f1_score = (
                2
                * (metrics.precision_score * metrics.recall_score)
                / (metrics.precision_score + metrics.recall_score)
            )

            metrics.success = True

        except Exception as e:
            metrics.error_message = str(e)
            metrics.success = False

        return metrics

    async def _run_detection_algorithm(
        self, algorithm_name: str, dataset: pd.DataFrame, contamination_rate: float
    ) -> Any:
        """Run detection algorithm (placeholder implementation)."""
        # Placeholder for actual algorithm execution
        # This would integrate with the actual detection service

        # Simulate processing time based on dataset size
        processing_time = len(dataset) / 10000.0  # Base processing time
        await asyncio.sleep(
            min(processing_time, 2.0)
        )  # Cap at 2 seconds for simulation

        return {"anomalies": np.random.randint(0, 2, len(dataset))}

    async def _generate_benchmark_datasets(
        self, config: BenchmarkConfig
    ) -> list[pd.DataFrame]:
        """Generate synthetic datasets for benchmarking."""
        datasets = []

        for size in config.dataset_sizes:
            for dimension in config.feature_dimensions:
                dataset = await self._generate_synthetic_dataset(
                    size=size, features=dimension, contamination=0.1
                )
                datasets.append(dataset)

        return datasets

    async def _generate_synthetic_dataset(
        self, size: int, features: int, contamination: float
    ) -> pd.DataFrame:
        """Generate synthetic dataset for testing."""
        # Generate normal data
        normal_size = int(size * (1 - contamination))
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(features), cov=np.eye(features), size=normal_size
        )

        # Generate anomalous data
        anomaly_size = size - normal_size
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(features) * 3, cov=np.eye(features) * 2, size=anomaly_size
        )

        # Combine data
        data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([np.zeros(normal_size), np.ones(anomaly_size)])

        # Create DataFrame
        columns = [f"feature_{i}" for i in range(features)]
        df = pd.DataFrame(data, columns=columns)
        df["label"] = labels

        # Shuffle
        df = df.sample(frac=1).reset_index(drop=True)

        return df

    async def _calculate_average_metrics(
        self, metrics_list: list[PerformanceMetrics]
    ) -> PerformanceMetrics:
        """Calculate average metrics from multiple runs."""
        if not metrics_list:
            return PerformanceMetrics()

        successful_runs = [m for m in metrics_list if m.success]

        if not successful_runs:
            return metrics_list[0]  # Return the failed run

        avg_metrics = PerformanceMetrics(
            algorithm_name=successful_runs[0].algorithm_name,
            dataset_size=successful_runs[0].dataset_size,
            feature_dimension=successful_runs[0].feature_dimension,
            contamination_rate=successful_runs[0].contamination_rate,
        )

        # Calculate averages
        avg_metrics.execution_time_seconds = statistics.mean(
            m.execution_time_seconds for m in successful_runs
        )
        avg_metrics.peak_memory_mb = statistics.mean(
            m.peak_memory_mb for m in successful_runs
        )
        avg_metrics.average_memory_mb = statistics.mean(
            m.average_memory_mb for m in successful_runs
        )
        avg_metrics.cpu_usage_percent = statistics.mean(
            m.cpu_usage_percent for m in successful_runs
        )
        avg_metrics.training_throughput = statistics.mean(
            m.training_throughput for m in successful_runs
        )
        avg_metrics.accuracy_score = statistics.mean(
            m.accuracy_score for m in successful_runs
        )
        avg_metrics.precision_score = statistics.mean(
            m.precision_score for m in successful_runs
        )
        avg_metrics.recall_score = statistics.mean(
            m.recall_score for m in successful_runs
        )
        avg_metrics.f1_score = statistics.mean(m.f1_score for m in successful_runs)

        return avg_metrics

    async def _calculate_summary_statistics(self, suite: BenchmarkSuite) -> None:
        """Calculate summary statistics for benchmark suite."""
        if not suite.individual_results:
            return

        successful_results = [r for r in suite.individual_results if r.success]

        if not successful_results:
            return

        # Group by algorithm
        by_algorithm = defaultdict(list)
        for result in successful_results:
            by_algorithm[result.algorithm_name].append(result)

        summary = {}

        for algorithm, results in by_algorithm.items():
            algorithm_summary = {
                "total_runs": len(results),
                "avg_execution_time": statistics.mean(
                    r.execution_time_seconds for r in results
                ),
                "min_execution_time": min(r.execution_time_seconds for r in results),
                "max_execution_time": max(r.execution_time_seconds for r in results),
                "avg_memory_usage": statistics.mean(r.peak_memory_mb for r in results),
                "max_memory_usage": max(r.peak_memory_mb for r in results),
                "avg_throughput": statistics.mean(
                    r.training_throughput for r in results
                ),
                "avg_accuracy": statistics.mean(r.accuracy_score for r in results),
                "avg_f1_score": statistics.mean(r.f1_score for r in results),
            }
            summary[algorithm] = algorithm_summary

        suite.summary_stats = summary

    async def _perform_comparative_analysis(self, suite: BenchmarkSuite) -> None:
        """Perform comparative analysis between algorithms."""
        if not suite.summary_stats:
            return

        analysis = {
            "fastest_algorithm": min(
                suite.summary_stats.items(), key=lambda x: x[1]["avg_execution_time"]
            )[0],
            "most_memory_efficient": min(
                suite.summary_stats.items(), key=lambda x: x[1]["avg_memory_usage"]
            )[0],
            "highest_throughput": max(
                suite.summary_stats.items(), key=lambda x: x[1]["avg_throughput"]
            )[0],
            "most_accurate": max(
                suite.summary_stats.items(), key=lambda x: x[1]["avg_accuracy"]
            )[0],
            "best_f1_score": max(
                suite.summary_stats.items(), key=lambda x: x[1]["avg_f1_score"]
            )[0],
        }

        suite.comparative_analysis = analysis

    async def _generate_recommendations(self, suite: BenchmarkSuite) -> None:
        """Generate performance recommendations."""
        recommendations = []

        if suite.comparative_analysis:
            fastest = suite.comparative_analysis.get("fastest_algorithm")
            most_accurate = suite.comparative_analysis.get("most_accurate")

            if fastest:
                recommendations.append(
                    f"For speed-critical applications, consider using {fastest}"
                )

            if most_accurate:
                recommendations.append(
                    f"For accuracy-critical applications, consider using {most_accurate}"
                )

        # Check for performance issues
        for algorithm, stats in suite.summary_stats.items():
            if stats["avg_execution_time"] > 60:
                recommendations.append(
                    f"{algorithm} shows slow execution times - consider optimization"
                )

            if stats["max_memory_usage"] > 2048:
                recommendations.append(
                    f"{algorithm} uses high memory - monitor for memory leaks"
                )

        suite.recommendations = recommendations

    async def _get_test_environment(self) -> dict[str, Any]:
        """Get test environment information."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "python_version": "3.11+",  # Would get actual version
            "platform": "linux",
            "architecture": "x86_64",
        }

    async def _get_system_info(self) -> dict[str, Any]:
        """Get system information."""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()

        return {
            "cpu_cores": cpu_count,
            "cpu_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": memory.total / 1024 / 1024 / 1024,
            "memory_available_gb": memory.available / 1024 / 1024 / 1024,
            "disk_space_gb": psutil.disk_usage("/").total / 1024 / 1024 / 1024,
        }

    async def _analyze_scalability(
        self, results: list[PerformanceMetrics]
    ) -> dict[str, Any]:
        """Analyze scalability test results."""
        scale_factors = [r.scalability_factor for r in results]
        execution_times = [r.execution_time_seconds for r in results]
        efficiency_ratios = [r.efficiency_ratio for r in results]

        return {
            "linear_scalability_score": statistics.mean(efficiency_ratios),
            "time_complexity_estimate": self._estimate_time_complexity(
                scale_factors, execution_times
            ),
            "scalability_grade": self._calculate_scalability_grade(efficiency_ratios),
            "max_efficient_scale": max(
                [r.scalability_factor for r in results if r.efficiency_ratio > 0.5],
                default=1,
            ),
        }

    def _estimate_time_complexity(self, scales: list[int], times: list[float]) -> str:
        """Estimate time complexity from scaling data."""
        # Simple heuristic based on growth rate
        if len(scales) < 2:
            return "unknown"

        growth_rates = []
        for i in range(1, len(scales)):
            scale_ratio = scales[i] / scales[i - 1]
            time_ratio = times[i] / times[i - 1]
            growth_rates.append(time_ratio / scale_ratio)

        avg_growth = statistics.mean(growth_rates)

        if avg_growth < 1.2:
            return "O(n)"
        elif avg_growth < 2.0:
            return "O(n log n)"
        elif avg_growth < 3.0:
            return "O(n²)"
        else:
            return "O(n³) or worse"

    def _calculate_scalability_grade(self, efficiency_ratios: list[float]) -> str:
        """Calculate scalability grade."""
        avg_efficiency = statistics.mean(efficiency_ratios)

        if avg_efficiency >= 0.9:
            return "A"
        elif avg_efficiency >= 0.7:
            return "B"
        elif avg_efficiency >= 0.5:
            return "C"
        elif avg_efficiency >= 0.3:
            return "D"
        else:
            return "F"

    async def _analyze_memory_usage(
        self, results: list[PerformanceMetrics]
    ) -> dict[str, Any]:
        """Analyze memory usage patterns."""
        dataset_sizes = [r.dataset_size for r in results]
        memory_usage = [r.peak_memory_mb for r in results]
        memory_growth = [r.memory_growth_mb for r in results]

        return {
            "memory_efficiency": statistics.mean(
                m / s * 1000 for m, s in zip(memory_usage, dataset_sizes, strict=False)
            ),
            "memory_growth_rate": statistics.mean(memory_growth),
            "max_memory_tested": max(memory_usage),
            "memory_scalability": self._assess_memory_scalability(
                dataset_sizes, memory_usage
            ),
        }

    def _assess_memory_scalability(self, sizes: list[int], memory: list[float]) -> str:
        """Assess memory scalability pattern."""
        if len(sizes) < 2:
            return "unknown"

        # Calculate memory growth ratios
        growth_ratios = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i - 1]
            memory_ratio = memory[i] / memory[i - 1]
            growth_ratios.append(memory_ratio / size_ratio)

        avg_ratio = statistics.mean(growth_ratios)

        if avg_ratio < 1.2:
            return "linear"
        elif avg_ratio < 2.0:
            return "near-linear"
        elif avg_ratio < 3.0:
            return "quadratic"
        else:
            return "exponential"

    async def _measure_throughput(
        self, algorithm_name: str, dataset: pd.DataFrame, duration_seconds: int
    ) -> dict[str, Any]:
        """Measure algorithm throughput."""
        start_time = time.time()
        end_time = start_time + duration_seconds

        samples_processed = 0
        iterations = 0

        while time.time() < end_time:
            # Process dataset chunk
            await self._run_detection_algorithm(algorithm_name, dataset, 0.1)
            samples_processed += len(dataset)
            iterations += 1

        actual_duration = time.time() - start_time
        throughput = samples_processed / actual_duration

        return {
            "algorithm": algorithm_name,
            "dataset_size": len(dataset),
            "duration_seconds": actual_duration,
            "samples_processed": samples_processed,
            "iterations": iterations,
            "throughput_samples_per_second": throughput,
            "throughput_datasets_per_second": iterations / actual_duration,
        }

    async def _analyze_throughput(self, results: dict[str, list]) -> dict[str, Any]:
        """Analyze throughput results."""
        analysis = {}

        for algorithm, algorithm_results in results.items():
            throughputs = [
                r["throughput_samples_per_second"] for r in algorithm_results
            ]
            dataset_sizes = [r["dataset_size"] for r in algorithm_results]

            analysis[algorithm] = {
                "avg_throughput": statistics.mean(throughputs),
                "max_throughput": max(throughputs),
                "min_throughput": min(throughputs),
                "throughput_stability": 1.0
                - (statistics.stdev(throughputs) / statistics.mean(throughputs)),
                "best_dataset_size": dataset_sizes[throughputs.index(max(throughputs))],
            }

        # Find best overall algorithm
        best_algorithm = max(analysis.items(), key=lambda x: x[1]["avg_throughput"])[0]
        analysis["overall_best"] = best_algorithm

        return analysis

    async def _generate_algorithm_comparison(
        self, results: dict[str, dict[int, PerformanceMetrics]], metrics: list[str]
    ) -> dict[str, Any]:
        """Generate comprehensive algorithm comparison."""
        comparison = {}

        for metric in metrics:
            metric_comparison = {}

            for algorithm, algorithm_results in results.items():
                metric_values = []

                for _size, result in algorithm_results.items():
                    if metric == "execution_time":
                        metric_values.append(result.execution_time_seconds)
                    elif metric == "memory_usage":
                        metric_values.append(result.peak_memory_mb)
                    elif metric == "accuracy":
                        metric_values.append(result.accuracy_score)
                    elif metric == "throughput":
                        metric_values.append(result.training_throughput)

                if metric_values:
                    metric_comparison[algorithm] = {
                        "average": statistics.mean(metric_values),
                        "min": min(metric_values),
                        "max": max(metric_values),
                        "std": (
                            statistics.stdev(metric_values)
                            if len(metric_values) > 1
                            else 0
                        ),
                    }

            comparison[metric] = metric_comparison

        return comparison

    async def _analyze_performance_trends(
        self, historical_data: list[PerformanceMetrics]
    ) -> dict[str, Any]:
        """Analyze performance trends over time."""
        if len(historical_data) < 2:
            return {"message": "Insufficient data for trend analysis"}

        # Sort by timestamp
        sorted_data = sorted(historical_data, key=lambda x: x.timestamp)

        # Calculate trends for different metrics
        timestamps = [d.timestamp for d in sorted_data]
        execution_times = [d.execution_time_seconds for d in sorted_data]
        memory_usage = [d.peak_memory_mb for d in sorted_data]
        accuracy_scores = [d.accuracy_score for d in sorted_data]

        trends = {
            "execution_time_trend": self._calculate_trend(execution_times),
            "memory_usage_trend": self._calculate_trend(memory_usage),
            "accuracy_trend": self._calculate_trend(accuracy_scores),
            "data_points": len(sorted_data),
            "time_span_days": (timestamps[-1] - timestamps[0]).days,
        }

        return trends

    def _calculate_trend(self, values: list[float]) -> dict[str, Any]:
        """Calculate trend for a metric."""
        if len(values) < 2:
            return {"direction": "stable", "change_percent": 0.0}

        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if first_avg == 0:
            change_percent = 0.0
        else:
            change_percent = ((second_avg - first_avg) / first_avg) * 100

        if abs(change_percent) < 5:
            direction = "stable"
        elif change_percent > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return {
            "direction": direction,
            "change_percent": change_percent,
            "first_period_avg": first_avg,
            "second_period_avg": second_avg,
        }

    async def _generate_trend_recommendations(
        self, trends: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on trends."""
        recommendations = []

        exec_trend = trends.get("execution_time_trend", {})
        if (
            exec_trend.get("direction") == "increasing"
            and exec_trend.get("change_percent", 0) > 10
        ):
            recommendations.append(
                "Execution time is increasing - investigate performance regression"
            )

        memory_trend = trends.get("memory_usage_trend", {})
        if (
            memory_trend.get("direction") == "increasing"
            and memory_trend.get("change_percent", 0) > 15
        ):
            recommendations.append(
                "Memory usage is increasing - check for memory leaks"
            )

        accuracy_trend = trends.get("accuracy_trend", {})
        if (
            accuracy_trend.get("direction") == "decreasing"
            and abs(accuracy_trend.get("change_percent", 0)) > 5
        ):
            recommendations.append("Accuracy is decreasing - model may need retraining")

        return recommendations

    async def _generate_html_report(
        self, suite: BenchmarkSuite, output_path: Path
    ) -> Path:
        """Generate HTML benchmark report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Benchmark Report - {suite.suite_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .metric {{ margin-bottom: 10px; }}
                .recommendations {{ background-color: #fff3cd; padding: 15px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Benchmark Report</h1>
                <h2>{suite.suite_name}</h2>
                <p><strong>Description:</strong> {suite.description}</p>
                <p><strong>Generated:</strong> {datetime.utcnow().isoformat()}</p>
                <p><strong>Duration:</strong> {suite.total_duration_seconds:.2f} seconds</p>
            </div>

            <div class="summary">
                <h3>Executive Summary</h3>
                <p><strong>Total Test Runs:</strong> {len(suite.individual_results)}</p>
                <p><strong>Algorithms Tested:</strong> {len(suite.summary_stats)}</p>
                <p><strong>Overall Grade:</strong> {suite.performance_grade}</p>
            </div>
        """

        # Add summary statistics table
        if suite.summary_stats:
            html_content += """
            <h3>Algorithm Performance Summary</h3>
            <table>
                <tr>
                    <th>Algorithm</th>
                    <th>Avg Execution Time (s)</th>
                    <th>Avg Memory Usage (MB)</th>
                    <th>Avg Throughput (samples/s)</th>
                    <th>Avg Accuracy</th>
                </tr>
            """

            for algorithm, stats in suite.summary_stats.items():
                html_content += f"""
                <tr>
                    <td>{algorithm}</td>
                    <td>{stats["avg_execution_time"]:.3f}</td>
                    <td>{stats["avg_memory_usage"]:.1f}</td>
                    <td>{stats["avg_throughput"]:.1f}</td>
                    <td>{stats["avg_accuracy"]:.3f}</td>
                </tr>
                """

            html_content += "</table>"

        # Add recommendations
        if suite.recommendations:
            html_content += """
            <div class="recommendations">
                <h3>Recommendations</h3>
                <ul>
            """
            for rec in suite.recommendations:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul></div>"

        html_content += """
        </body>
        </html>
        """

        report_path = output_path / f"benchmark_report_{suite.suite_id}.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        return report_path

    async def _generate_json_report(
        self, suite: BenchmarkSuite, output_path: Path
    ) -> Path:
        """Generate JSON benchmark report."""
        report_data = {
            "suite_id": str(suite.suite_id),
            "suite_name": suite.suite_name,
            "description": suite.description,
            "start_time": suite.start_time.isoformat(),
            "end_time": suite.end_time.isoformat() if suite.end_time else None,
            "total_duration_seconds": suite.total_duration_seconds,
            "summary_stats": suite.summary_stats,
            "comparative_analysis": suite.comparative_analysis,
            "recommendations": suite.recommendations,
            "individual_results": [
                {
                    "algorithm_name": r.algorithm_name,
                    "dataset_size": r.dataset_size,
                    "feature_dimension": r.feature_dimension,
                    "contamination_rate": r.contamination_rate,
                    "execution_time_seconds": r.execution_time_seconds,
                    "peak_memory_mb": r.peak_memory_mb,
                    "training_throughput": r.training_throughput,
                    "accuracy_score": r.accuracy_score,
                    "success": r.success,
                    "error_message": r.error_message,
                }
                for r in suite.individual_results
            ],
            "system_info": suite.system_info,
            "test_environment": suite.test_environment,
        }

        report_path = output_path / f"benchmark_report_{suite.suite_id}.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        return report_path

    async def _generate_csv_report(
        self, suite: BenchmarkSuite, output_path: Path
    ) -> Path:
        """Generate CSV benchmark report."""
        import csv

        report_path = output_path / f"benchmark_report_{suite.suite_id}.csv"

        with open(report_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(
                [
                    "Algorithm",
                    "Dataset Size",
                    "Feature Dimension",
                    "Contamination Rate",
                    "Execution Time (s)",
                    "Peak Memory (MB)",
                    "Throughput (samples/s)",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Success",
                    "Error Message",
                ]
            )

            # Write data
            for result in suite.individual_results:
                writer.writerow(
                    [
                        result.algorithm_name,
                        result.dataset_size,
                        result.feature_dimension,
                        result.contamination_rate,
                        result.execution_time_seconds,
                        result.peak_memory_mb,
                        result.training_throughput,
                        result.accuracy_score,
                        result.precision_score,
                        result.recall_score,
                        result.f1_score,
                        result.success,
                        result.error_message or "",
                    ]
                )

        return report_path


class SystemMonitor:
    """System resource monitoring utility."""

    def __init__(self):
        self.monitoring_active = False
        self.monitoring_data = []

    async def start_monitoring(self) -> str:
        """Start system monitoring."""
        monitor_id = str(uuid4())
        self.monitoring_active = True

        # Start background monitoring task
        asyncio.create_task(self._monitor_resources(monitor_id))

        return monitor_id

    async def stop_monitoring(self, monitor_id: str) -> dict[str, Any]:
        """Stop monitoring and return collected data."""
        self.monitoring_active = False

        if not self.monitoring_data:
            return {}

        # Calculate summary statistics
        memory_values = [d["memory_mb"] for d in self.monitoring_data]
        cpu_values = [d["cpu_percent"] for d in self.monitoring_data]

        summary = {
            "monitor_id": monitor_id,
            "data_points": len(self.monitoring_data),
            "peak_memory_mb": max(memory_values) if memory_values else 0,
            "avg_memory_mb": statistics.mean(memory_values) if memory_values else 0,
            "avg_cpu_percent": statistics.mean(cpu_values) if cpu_values else 0,
            "max_cpu_percent": max(cpu_values) if cpu_values else 0,
        }

        # Clear data for next monitoring session
        self.monitoring_data.clear()

        return summary

    async def _monitor_resources(self, monitor_id: str) -> None:
        """Monitor system resources."""
        while self.monitoring_active:
            try:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)

                self.monitoring_data.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "memory_mb": memory.used / 1024 / 1024,
                        "cpu_percent": cpu_percent,
                        "monitor_id": monitor_id,
                    }
                )

                await asyncio.sleep(0.5)  # Monitor every 0.5 seconds

            except Exception:
                # Continue monitoring even if individual measurements fail
                await asyncio.sleep(0.5)
