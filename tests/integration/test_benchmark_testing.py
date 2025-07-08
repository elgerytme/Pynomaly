"""Benchmark testing for performance validation - Phase 4 Performance Testing."""

from __future__ import annotations

import gc
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pytest
from pynomaly.domain.entities import Detector
from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""

    algorithm: str
    dataset_size: tuple[int, int]  # (n_samples, n_features)
    training_time_ms: float
    prediction_time_ms: float
    total_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    accuracy_metrics: dict[str, float]
    success: bool
    error_message: str | None = None


@dataclass
class PerformanceProfile:
    """Performance profile for an algorithm."""

    algorithm: str
    avg_training_time_ms: float
    avg_prediction_time_ms: float
    time_complexity_factor: float  # Time scaling factor
    memory_efficiency: float  # MB per 1000 samples
    accuracy_score: float
    stability_score: float  # Consistency across runs
    overall_score: float


class PerformanceBenchmark:
    """Performance benchmarking framework."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []
        self.profiles: dict[str, PerformanceProfile] = {}

    def generate_benchmark_dataset(
        self,
        n_samples: int,
        n_features: int,
        contamination: float = 0.1,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Generate benchmark dataset with known anomalies."""
        np.random.seed(random_state)

        # Generate normal data
        normal_samples = int(n_samples * (1 - contamination))
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features), cov=np.eye(n_features), size=normal_samples
        )

        # Generate anomalous data
        anomaly_samples = n_samples - normal_samples
        if anomaly_samples > 0:
            # Anomalies are further from center with higher variance
            anomaly_data = np.random.multivariate_normal(
                mean=np.ones(n_features) * 3,  # Shifted mean
                cov=np.eye(n_features) * 4,  # Higher variance
                size=anomaly_samples,
            )

            # Combine normal and anomalous data
            data = np.vstack([normal_data, anomaly_data])
            labels = np.hstack([np.zeros(normal_samples), np.ones(anomaly_samples)])
        else:
            data = normal_data
            labels = np.zeros(normal_samples)

        # Shuffle data
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]

        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(data, columns=feature_names)
        df["true_label"] = labels

        return df

    def benchmark_algorithm(
        self,
        algorithm: str,
        dataset: pd.DataFrame,
        adapter_class: type,
        hyperparameters: dict[str, Any] = None,
    ) -> BenchmarkResult:
        """Benchmark a single algorithm on a dataset."""
        if hyperparameters is None:
            hyperparameters = {"contamination": 0.1, "random_state": 42}

        n_samples, n_features = (
            dataset.shape[0],
            dataset.shape[1] - 1,
        )  # Exclude label column

        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory

        try:
            # Create detector
            detector = Detector(
                id=f"benchmark_{algorithm}",
                name=f"Benchmark {algorithm}",
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                is_fitted=False,
            )

            # Prepare data (exclude true labels)
            X = dataset.drop("true_label", axis=1).values
            y_true = dataset["true_label"].values

            # Create adapter
            adapter = adapter_class()

            # Training phase
            gc.collect()  # Clean up before training
            training_start = time.time()

            success = adapter.train(detector, X)

            training_end = time.time()
            training_time_ms = (training_end - training_start) * 1000

            # Update peak memory
            current_memory = process.memory_info().rss / 1024 / 1024
            if current_memory > peak_memory:
                peak_memory = current_memory

            if not success:
                return BenchmarkResult(
                    algorithm=algorithm,
                    dataset_size=(n_samples, n_features),
                    training_time_ms=training_time_ms,
                    prediction_time_ms=0,
                    total_time_ms=training_time_ms,
                    memory_usage_mb=current_memory - initial_memory,
                    peak_memory_mb=peak_memory,
                    accuracy_metrics={},
                    success=False,
                    error_message="Training failed",
                )

            # Prediction phase
            gc.collect()
            prediction_start = time.time()

            predictions, scores = adapter.predict(detector, X)

            prediction_end = time.time()
            prediction_time_ms = (prediction_end - prediction_start) * 1000

            # Update peak memory
            current_memory = process.memory_info().rss / 1024 / 1024
            if current_memory > peak_memory:
                peak_memory = current_memory

            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(
                y_true, predictions, scores
            )

            return BenchmarkResult(
                algorithm=algorithm,
                dataset_size=(n_samples, n_features),
                training_time_ms=training_time_ms,
                prediction_time_ms=prediction_time_ms,
                total_time_ms=training_time_ms + prediction_time_ms,
                memory_usage_mb=current_memory - initial_memory,
                peak_memory_mb=peak_memory,
                accuracy_metrics=accuracy_metrics,
                success=True,
            )

        except Exception as e:
            current_memory = process.memory_info().rss / 1024 / 1024
            return BenchmarkResult(
                algorithm=algorithm,
                dataset_size=(n_samples, n_features),
                training_time_ms=0,
                prediction_time_ms=0,
                total_time_ms=0,
                memory_usage_mb=current_memory - initial_memory,
                peak_memory_mb=peak_memory,
                accuracy_metrics={},
                success=False,
                error_message=str(e),
            )

    def _calculate_accuracy_metrics(
        self, y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray
    ) -> dict[str, float]:
        """Calculate accuracy metrics for benchmark."""
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        try:
            metrics = {}

            # Basic classification metrics
            metrics["precision"] = precision_score(y_true, predictions, zero_division=0)
            metrics["recall"] = recall_score(y_true, predictions, zero_division=0)
            metrics["f1_score"] = f1_score(y_true, predictions, zero_division=0)

            # AUC score (if we have non-constant scores and labels)
            if len(np.unique(y_true)) > 1 and len(np.unique(scores)) > 1:
                metrics["roc_auc"] = roc_auc_score(y_true, scores)
            else:
                metrics["roc_auc"] = 0.5

            # Anomaly detection specific metrics
            n_anomalies_true = np.sum(y_true)
            n_anomalies_pred = np.sum(predictions)
            n_total = len(y_true)

            metrics["anomaly_rate_true"] = n_anomalies_true / n_total
            metrics["anomaly_rate_pred"] = n_anomalies_pred / n_total
            metrics["detection_rate"] = np.sum(
                (y_true == 1) & (predictions == 1)
            ) / max(n_anomalies_true, 1)

            return metrics

        except Exception:
            # Return default metrics if calculation fails
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "roc_auc": 0.5,
                "anomaly_rate_true": 0.0,
                "anomaly_rate_pred": 0.0,
                "detection_rate": 0.0,
            }

    def run_comprehensive_benchmark(
        self,
        algorithms: list[str],
        dataset_sizes: list[tuple[int, int]],
        n_runs: int = 3,
    ) -> dict[str, PerformanceProfile]:
        """Run comprehensive benchmark across algorithms and dataset sizes."""
        all_results = []

        for algorithm in algorithms:
            algorithm_results = []

            for n_samples, n_features in dataset_sizes:
                for run in range(n_runs):
                    print(
                        f"Benchmarking {algorithm} on {n_samples}x{n_features} dataset (run {run + 1}/{n_runs})"
                    )

                    # Generate dataset for this run
                    dataset = self.generate_benchmark_dataset(
                        n_samples, n_features, random_state=42 + run
                    )

                    # Determine adapter class
                    if algorithm in [
                        "IsolationForest",
                        "LocalOutlierFactor",
                        "OneClassSVM",
                    ]:
                        adapter_class = SklearnAdapter
                    else:
                        adapter_class = PyODAdapter

                    # Run benchmark
                    result = self.benchmark_algorithm(algorithm, dataset, adapter_class)
                    algorithm_results.append(result)
                    all_results.append(result)

                    # Force garbage collection between runs
                    gc.collect()

            # Create performance profile for this algorithm
            if algorithm_results:
                profile = self._create_performance_profile(algorithm, algorithm_results)
                self.profiles[algorithm] = profile

        self.results.extend(all_results)
        return self.profiles

    def _create_performance_profile(
        self, algorithm: str, results: list[BenchmarkResult]
    ) -> PerformanceProfile:
        """Create performance profile from benchmark results."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return PerformanceProfile(
                algorithm=algorithm,
                avg_training_time_ms=0,
                avg_prediction_time_ms=0,
                time_complexity_factor=float("inf"),
                memory_efficiency=float("inf"),
                accuracy_score=0,
                stability_score=0,
                overall_score=0,
            )

        # Calculate averages
        avg_training_time = statistics.mean(
            [r.training_time_ms for r in successful_results]
        )
        avg_prediction_time = statistics.mean(
            [r.prediction_time_ms for r in successful_results]
        )

        # Calculate time complexity factor (time increase per sample)
        if len(successful_results) > 1:
            time_complexity_factor = self._calculate_time_complexity_factor(
                successful_results
            )
        else:
            time_complexity_factor = 1.0

        # Calculate memory efficiency (MB per 1000 samples)
        memory_efficiency = statistics.mean(
            [
                r.memory_usage_mb / (r.dataset_size[0] / 1000)
                for r in successful_results
                if r.dataset_size[0] > 0
            ]
        )

        # Calculate average accuracy score
        accuracy_scores = []
        for r in successful_results:
            if "f1_score" in r.accuracy_metrics:
                accuracy_scores.append(r.accuracy_metrics["f1_score"])
        avg_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0

        # Calculate stability score (inverse of coefficient of variation)
        if len(successful_results) > 1:
            training_times = [r.training_time_ms for r in successful_results]
            cv = (
                statistics.stdev(training_times) / statistics.mean(training_times)
                if statistics.mean(training_times) > 0
                else 1
            )
            stability_score = 1 / (1 + cv)  # Higher score = more stable
        else:
            stability_score = 1.0

        # Calculate overall score (weighted combination)
        # Lower time and memory usage = higher score
        # Higher accuracy and stability = higher score
        time_score = 1 / (1 + avg_training_time / 1000)  # Normalize by seconds
        memory_score = 1 / (
            1 + memory_efficiency / 10
        )  # Normalize by 10MB per 1K samples
        accuracy_score = avg_accuracy

        overall_score = (
            time_score * 0.3
            + memory_score * 0.2
            + accuracy_score * 0.3
            + stability_score * 0.2
        )

        return PerformanceProfile(
            algorithm=algorithm,
            avg_training_time_ms=avg_training_time,
            avg_prediction_time_ms=avg_prediction_time,
            time_complexity_factor=time_complexity_factor,
            memory_efficiency=memory_efficiency,
            accuracy_score=avg_accuracy,
            stability_score=stability_score,
            overall_score=overall_score,
        )

    def _calculate_time_complexity_factor(
        self, results: list[BenchmarkResult]
    ) -> float:
        """Calculate time complexity scaling factor."""
        # Sort results by dataset size
        sorted_results = sorted(results, key=lambda r: r.dataset_size[0])

        if len(sorted_results) < 2:
            return 1.0

        # Calculate average time increase per sample increase
        time_ratios = []
        for i in range(1, len(sorted_results)):
            prev_result = sorted_results[i - 1]
            curr_result = sorted_results[i]

            size_ratio = curr_result.dataset_size[0] / prev_result.dataset_size[0]
            time_ratio = curr_result.total_time_ms / max(prev_result.total_time_ms, 1)

            if size_ratio > 1:
                complexity_factor = time_ratio / size_ratio
                time_ratios.append(complexity_factor)

        return statistics.mean(time_ratios) if time_ratios else 1.0

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.profiles:
            return "No benchmark results available."

        report = []
        report.append("# Pynomaly Performance Benchmark Report")
        report.append("=" * 50)
        report.append("")

        # Overall summary
        report.append("## Overall Performance Summary")
        report.append("")

        # Sort algorithms by overall score
        sorted_algorithms = sorted(
            self.profiles.items(), key=lambda x: x[1].overall_score, reverse=True
        )

        report.append(
            "| Rank | Algorithm | Overall Score | Training Time (ms) | Memory Efficiency | Accuracy | Stability |"
        )
        report.append(
            "|------|-----------|---------------|-------------------|-------------------|----------|-----------|"
        )

        for rank, (algorithm, profile) in enumerate(sorted_algorithms, 1):
            report.append(
                f"| {rank} | {algorithm} | {profile.overall_score:.3f} | "
                f"{profile.avg_training_time_ms:.1f} | {profile.memory_efficiency:.2f} MB/1K | "
                f"{profile.accuracy_score:.3f} | {profile.stability_score:.3f} |"
            )

        report.append("")

        # Detailed analysis
        report.append("## Detailed Algorithm Analysis")
        report.append("")

        for algorithm, profile in sorted_algorithms:
            report.append(f"### {algorithm}")
            report.append("")
            report.append(f"- **Overall Score**: {profile.overall_score:.3f}")
            report.append(
                f"- **Average Training Time**: {profile.avg_training_time_ms:.1f} ms"
            )
            report.append(
                f"- **Average Prediction Time**: {profile.avg_prediction_time_ms:.1f} ms"
            )
            report.append(
                f"- **Time Complexity Factor**: {profile.time_complexity_factor:.2f}"
            )
            report.append(
                f"- **Memory Efficiency**: {profile.memory_efficiency:.2f} MB per 1000 samples"
            )
            report.append(f"- **Accuracy Score**: {profile.accuracy_score:.3f}")
            report.append(f"- **Stability Score**: {profile.stability_score:.3f}")
            report.append("")

        # Performance recommendations
        report.append("## Performance Recommendations")
        report.append("")

        # Best for different scenarios
        best_overall = max(self.profiles.items(), key=lambda x: x[1].overall_score)
        best_speed = min(self.profiles.items(), key=lambda x: x[1].avg_training_time_ms)
        best_memory = min(self.profiles.items(), key=lambda x: x[1].memory_efficiency)
        best_accuracy = max(self.profiles.items(), key=lambda x: x[1].accuracy_score)

        report.append(
            f"- **Best Overall Performance**: {best_overall[0]} (score: {best_overall[1].overall_score:.3f})"
        )
        report.append(
            f"- **Fastest Training**: {best_speed[0]} ({best_speed[1].avg_training_time_ms:.1f} ms)"
        )
        report.append(
            f"- **Most Memory Efficient**: {best_memory[0]} ({best_memory[1].memory_efficiency:.2f} MB/1K)"
        )
        report.append(
            f"- **Highest Accuracy**: {best_accuracy[0]} ({best_accuracy[1].accuracy_score:.3f})"
        )
        report.append("")

        # Usage recommendations
        report.append("### Usage Recommendations")
        report.append("")
        report.append(
            "- **For real-time applications**: Choose algorithms with low training and prediction times"
        )
        report.append(
            "- **For large datasets**: Prioritize memory efficiency and time complexity"
        )
        report.append(
            "- **For high accuracy requirements**: Focus on accuracy score while considering computational cost"
        )
        report.append(
            "- **For production stability**: Consider stability score for consistent performance"
        )
        report.append("")

        return "\n".join(report)


class TestPerformanceBenchmarking:
    """Test performance benchmarking framework."""

    def test_algorithm_benchmark_small_dataset(self):
        """Test benchmarking on small dataset."""
        benchmark = PerformanceBenchmark()

        # Generate small test dataset
        dataset = benchmark.generate_benchmark_dataset(100, 5, contamination=0.1)

        # Benchmark IsolationForest
        result = benchmark.benchmark_algorithm(
            "IsolationForest",
            dataset,
            SklearnAdapter,
            {"n_estimators": 50, "contamination": 0.1, "random_state": 42},
        )

        # Validate result
        assert result.success, f"Benchmark failed: {result.error_message}"
        assert result.algorithm == "IsolationForest"
        assert result.dataset_size == (100, 5)
        assert result.training_time_ms > 0
        assert result.prediction_time_ms > 0
        assert (
            result.total_time_ms == result.training_time_ms + result.prediction_time_ms
        )
        assert result.memory_usage_mb >= 0
        assert "f1_score" in result.accuracy_metrics
        assert 0 <= result.accuracy_metrics["f1_score"] <= 1

    def test_multiple_algorithm_comparison(self):
        """Test comparison of multiple algorithms."""
        benchmark = PerformanceBenchmark()

        algorithms = ["IsolationForest", "LocalOutlierFactor"]
        dataset_sizes = [(50, 3), (100, 5)]

        # Run benchmark
        profiles = benchmark.run_comprehensive_benchmark(
            algorithms, dataset_sizes, n_runs=2
        )

        # Validate profiles
        assert len(profiles) == len(algorithms)

        for algorithm in algorithms:
            assert algorithm in profiles
            profile = profiles[algorithm]

            assert profile.algorithm == algorithm
            assert profile.avg_training_time_ms >= 0
            assert profile.avg_prediction_time_ms >= 0
            assert profile.time_complexity_factor > 0
            assert profile.memory_efficiency >= 0
            assert 0 <= profile.accuracy_score <= 1
            assert 0 <= profile.stability_score <= 1
            assert 0 <= profile.overall_score <= 1

    def test_dataset_generation(self):
        """Test benchmark dataset generation."""
        benchmark = PerformanceBenchmark()

        # Test different dataset configurations
        configs = [(100, 5, 0.1), (200, 10, 0.05), (50, 3, 0.2)]

        for n_samples, n_features, contamination in configs:
            dataset = benchmark.generate_benchmark_dataset(
                n_samples, n_features, contamination
            )

            # Validate dataset properties
            assert len(dataset) == n_samples
            assert len(dataset.columns) == n_features + 1  # +1 for true_label
            assert "true_label" in dataset.columns

            # Check contamination rate
            actual_contamination = dataset["true_label"].mean()
            assert (
                abs(actual_contamination - contamination) < 0.05
            )  # Allow 5% tolerance

            # Check data types
            feature_columns = [col for col in dataset.columns if col != "true_label"]
            for col in feature_columns:
                assert pd.api.types.is_numeric_dtype(dataset[col])

    def test_performance_report_generation(self):
        """Test performance report generation."""
        benchmark = PerformanceBenchmark()

        # Run small benchmark
        algorithms = ["IsolationForest"]
        dataset_sizes = [(50, 3)]

        benchmark.run_comprehensive_benchmark(algorithms, dataset_sizes, n_runs=1)

        # Generate report
        report = benchmark.generate_performance_report()

        # Validate report content
        assert "Performance Benchmark Report" in report
        assert "IsolationForest" in report
        assert "Overall Score" in report
        assert "Training Time" in report
        assert "Memory Efficiency" in report
        assert "Accuracy" in report
        assert "Performance Recommendations" in report

    def test_memory_usage_tracking(self):
        """Test memory usage tracking during benchmarks."""
        benchmark = PerformanceBenchmark()

        # Use larger dataset to see memory usage
        dataset = benchmark.generate_benchmark_dataset(500, 20, contamination=0.1)

        result = benchmark.benchmark_algorithm(
            "IsolationForest",
            dataset,
            SklearnAdapter,
            {"n_estimators": 100, "contamination": 0.1, "random_state": 42},
        )

        assert result.success
        assert result.memory_usage_mb >= 0
        assert result.peak_memory_mb >= result.memory_usage_mb

        # Memory usage should be reasonable (not excessive)
        assert (
            result.memory_usage_mb < 100
        )  # Should not use more than 100MB for this test

    def test_time_complexity_analysis(self):
        """Test time complexity analysis."""
        benchmark = PerformanceBenchmark()

        # Test with increasing dataset sizes
        algorithms = ["IsolationForest"]
        dataset_sizes = [(100, 5), (200, 5), (400, 5)]  # Double size each time

        profiles = benchmark.run_comprehensive_benchmark(
            algorithms, dataset_sizes, n_runs=1
        )

        profile = profiles["IsolationForest"]

        # Time complexity factor should be reasonable
        assert profile.time_complexity_factor > 0
        assert profile.time_complexity_factor < 10  # Should not be worse than O(n^3)

    def test_accuracy_metrics_calculation(self):
        """Test accuracy metrics calculation."""
        benchmark = PerformanceBenchmark()

        # Create test data with known results
        y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
        predictions = np.array([0, 0, 1, 1, 0, 0, 1, 0])
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.1, 0.7, 0.3])

        metrics = benchmark._calculate_accuracy_metrics(y_true, predictions, scores)

        # Validate metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "anomaly_rate_true" in metrics
        assert "anomaly_rate_pred" in metrics
        assert "detection_rate" in metrics

        # All metrics should be between 0 and 1
        for metric_name, value in metrics.items():
            assert (
                0 <= value <= 1
            ), f"Metric {metric_name} = {value} is out of range [0, 1]"

    def test_benchmark_error_handling(self):
        """Test error handling in benchmarks."""
        benchmark = PerformanceBenchmark()

        # Create dataset
        dataset = benchmark.generate_benchmark_dataset(50, 3)

        # Test with invalid hyperparameters that might cause errors
        result = benchmark.benchmark_algorithm(
            "IsolationForest",
            dataset,
            SklearnAdapter,
            {"n_estimators": -1, "contamination": 2.0},  # Invalid parameters
        )

        # Should handle error gracefully
        if not result.success:
            assert result.error_message is not None
            assert len(result.error_message) > 0

    def test_concurrent_benchmarking(self):
        """Test concurrent benchmark execution."""
        benchmark = PerformanceBenchmark()

        def run_single_benchmark(algorithm):
            dataset = benchmark.generate_benchmark_dataset(
                100, 5, random_state=hash(algorithm) % 1000
            )
            return benchmark.benchmark_algorithm(algorithm, dataset, SklearnAdapter)

        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]

        # Run benchmarks concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_single_benchmark, alg) for alg in algorithms]
            results = [future.result() for future in futures]

        # Validate all results
        assert len(results) == len(algorithms)

        for result in results:
            assert result.algorithm in algorithms
            # Results should either succeed or fail gracefully
            if result.success:
                assert result.training_time_ms > 0
                assert result.prediction_time_ms > 0
            else:
                assert result.error_message is not None


class TestPerformanceRegression:
    """Test for performance regression detection."""

    def test_performance_baseline_establishment(self):
        """Establish performance baselines for regression testing."""
        benchmark = PerformanceBenchmark()

        # Standard test configuration
        dataset = benchmark.generate_benchmark_dataset(
            1000, 10, contamination=0.1, random_state=42
        )

        # Benchmark standard algorithms
        baseline_algorithms = [
            (
                "IsolationForest",
                SklearnAdapter,
                {"n_estimators": 100, "contamination": 0.1, "random_state": 42},
            ),
            (
                "LocalOutlierFactor",
                SklearnAdapter,
                {"n_neighbors": 20, "contamination": 0.1},
            ),
        ]

        baselines = {}

        for algorithm, adapter_class, hyperparams in baseline_algorithms:
            result = benchmark.benchmark_algorithm(
                algorithm, dataset, adapter_class, hyperparams
            )

            if result.success:
                baselines[algorithm] = {
                    "training_time_ms": result.training_time_ms,
                    "prediction_time_ms": result.prediction_time_ms,
                    "memory_usage_mb": result.memory_usage_mb,
                    "f1_score": result.accuracy_metrics.get("f1_score", 0),
                }

        # Validate baselines were established
        assert len(baselines) > 0

        for algorithm, baseline in baselines.items():
            print(f"{algorithm} baseline:")
            print(f"  Training time: {baseline['training_time_ms']:.1f} ms")
            print(f"  Prediction time: {baseline['prediction_time_ms']:.1f} ms")
            print(f"  Memory usage: {baseline['memory_usage_mb']:.1f} MB")
            print(f"  F1 score: {baseline['f1_score']:.3f}")

            # Reasonable baseline expectations
            assert baseline["training_time_ms"] < 10000  # Under 10 seconds
            assert baseline["prediction_time_ms"] < 5000  # Under 5 seconds
            assert baseline["memory_usage_mb"] < 200  # Under 200 MB
            assert baseline["f1_score"] >= 0.0  # Valid F1 score

    def test_performance_consistency(self):
        """Test that performance is consistent across multiple runs."""
        benchmark = PerformanceBenchmark()

        dataset = benchmark.generate_benchmark_dataset(
            500, 8, contamination=0.1, random_state=42
        )

        # Run same benchmark multiple times
        n_runs = 5
        results = []

        for _run in range(n_runs):
            result = benchmark.benchmark_algorithm(
                "IsolationForest",
                dataset,
                SklearnAdapter,
                {
                    "n_estimators": 50,
                    "contamination": 0.1,
                    "random_state": 42,
                },  # Fixed seed for consistency
            )

            if result.success:
                results.append(result)

        assert len(results) >= n_runs * 0.8  # At least 80% success rate

        if len(results) > 1:
            # Check consistency across runs
            training_times = [r.training_time_ms for r in results]
            prediction_times = [r.prediction_time_ms for r in results]

            # Calculate coefficient of variation (CV)
            training_cv = statistics.stdev(training_times) / statistics.mean(
                training_times
            )
            prediction_cv = statistics.stdev(prediction_times) / statistics.mean(
                prediction_times
            )

            print(f"Training time CV: {training_cv:.3f}")
            print(f"Prediction time CV: {prediction_cv:.3f}")

            # Performance should be reasonably consistent (CV < 0.5)
            assert training_cv < 0.5, f"Training time too variable: {training_cv}"
            assert prediction_cv < 0.5, f"Prediction time too variable: {prediction_cv}"

    def test_scalability_characteristics(self):
        """Test algorithm scalability characteristics."""
        benchmark = PerformanceBenchmark()

        # Test with increasing dataset sizes
        sizes = [(100, 5), (500, 5), (1000, 5)]
        algorithm = "IsolationForest"

        results = []

        for n_samples, n_features in sizes:
            dataset = benchmark.generate_benchmark_dataset(
                n_samples, n_features, random_state=42
            )
            result = benchmark.benchmark_algorithm(
                algorithm,
                dataset,
                SklearnAdapter,
                {"n_estimators": 50, "contamination": 0.1, "random_state": 42},
            )

            if result.success:
                results.append(result)

        assert (
            len(results) >= 2
        ), "Need at least 2 successful results to test scalability"

        # Analyze scaling behavior
        for i in range(1, len(results)):
            prev_result = results[i - 1]
            curr_result = results[i]

            size_factor = curr_result.dataset_size[0] / prev_result.dataset_size[0]
            time_factor = curr_result.total_time_ms / prev_result.total_time_ms
            memory_factor = curr_result.memory_usage_mb / max(
                prev_result.memory_usage_mb, 1
            )

            print(
                f"Size factor: {size_factor:.2f}x, Time factor: {time_factor:.2f}x, Memory factor: {memory_factor:.2f}x"
            )

            # Time scaling should be reasonable (not exponential)
            assert (
                time_factor < size_factor**2
            ), f"Time scaling too poor: {time_factor} vs {size_factor}^2"

            # Memory scaling should be reasonable
            assert (
                memory_factor < size_factor * 2
            ), f"Memory scaling too poor: {memory_factor} vs {size_factor}*2"


if __name__ == "__main__":
    # Run performance benchmarks directly
    benchmark = PerformanceBenchmark()

    print("Running Pynomaly Performance Benchmark...")

    algorithms = ["IsolationForest", "LocalOutlierFactor"]
    dataset_sizes = [(100, 5), (500, 10), (1000, 15)]

    profiles = benchmark.run_comprehensive_benchmark(
        algorithms, dataset_sizes, n_runs=2
    )

    print("\n" + benchmark.generate_performance_report())

    # Run pytest tests
    pytest.main([__file__, "-v"])
