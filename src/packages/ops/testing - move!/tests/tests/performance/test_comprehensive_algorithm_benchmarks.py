"""
Comprehensive Performance Benchmark Tests for Anomaly Detection Algorithms.

This module provides extensive benchmarking tests for all anomaly detection algorithms
in the Pynomaly system, including scalability, memory usage, and optimization analysis.
"""

import asyncio
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Import Pynomaly components
try:
    from pynomaly.infrastructure.performance.advanced_benchmarking_service import (
        AdvancedBenchmarkConfig,
        AdvancedPerformanceBenchmarkingService,
        AdvancedPerformanceMetrics,
    )
    from pynomaly.infrastructure.performance.optimization_engine import (
        OptimizationConfig,
        PerformanceOptimizationEngine,
        create_optimization_engine,
    )

    PYNOMALY_AVAILABLE = True
except ImportError:
    PYNOMALY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlgorithmBenchmarkSuite:
    """Comprehensive algorithm benchmark suite."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize benchmarking service
        if PYNOMALY_AVAILABLE:
            self.benchmark_service = AdvancedPerformanceBenchmarkingService(
                storage_path
            )
            self.optimization_engine = create_optimization_engine(
                cache_size_mb=256, storage_path=storage_path / "optimization"
            )

        # Algorithm configurations
        self.algorithms = {
            "isolation_forest": self._create_isolation_forest,
            "local_outlier_factor": self._create_lof,
            "one_class_svm": self._create_one_class_svm,
            "ensemble_voting": self._create_ensemble_voting,
            "ensemble_stacking": self._create_ensemble_stacking,
        }

        # Test configurations
        self.dataset_sizes = [100, 500, 1000, 5000, 10000, 25000, 50000]
        self.feature_dimensions = [5, 10, 20, 50, 100]
        self.contamination_rates = [0.01, 0.05, 0.1, 0.2]

        logger.info(
            f"Algorithm benchmark suite initialized with {len(self.algorithms)} algorithms"
        )

    def _create_isolation_forest(self, contamination: float = 0.1, **kwargs):
        """Create Isolation Forest detector."""
        return IsolationForest(
            contamination=contamination, random_state=42, n_jobs=-1, **kwargs
        )

    def _create_lof(self, contamination: float = 0.1, **kwargs):
        """Create Local Outlier Factor detector."""
        return LocalOutlierFactor(contamination=contamination, n_jobs=-1, **kwargs)

    def _create_one_class_svm(self, contamination: float = 0.1, **kwargs):
        """Create One-Class SVM detector."""
        nu = min(contamination * 2, 0.5)  # Adjust nu based on contamination
        return OneClassSVM(nu=nu, gamma="scale", **kwargs)

    def _create_ensemble_voting(self, contamination: float = 0.1, **kwargs):
        """Create ensemble with voting."""
        return EnsembleVotingDetector(contamination=contamination, **kwargs)

    def _create_ensemble_stacking(self, contamination: float = 0.1, **kwargs):
        """Create ensemble with stacking."""
        return EnsembleStackingDetector(contamination=contamination, **kwargs)

    async def run_algorithm_performance_test(
        self,
        algorithm_name: str,
        dataset_size: int,
        feature_dimension: int,
        contamination_rate: float,
        iterations: int = 5,
    ) -> AdvancedPerformanceMetrics:
        """Run performance test for a specific algorithm configuration."""
        logger.info(
            f"Testing {algorithm_name} - Size: {dataset_size}, Features: {feature_dimension}"
        )

        # Generate test dataset
        dataset = self._generate_test_dataset(
            size=dataset_size,
            features=feature_dimension,
            contamination=contamination_rate,
        )

        # Initialize metrics
        metrics_list = []

        for iteration in range(iterations):
            start_time = time.time()

            # Memory tracking
            initial_memory = self._get_memory_usage()
            gc_before = gc.get_count()

            try:
                # Create algorithm instance
                algorithm = self.algorithms[algorithm_name](
                    contamination=contamination_rate
                )

                # Training phase
                train_start = time.time()
                algorithm.fit(dataset.drop("label", axis=1))
                train_end = time.time()

                # Prediction phase
                pred_start = time.time()
                predictions = algorithm.predict(dataset.drop("label", axis=1))
                pred_end = time.time()

                # Calculate metrics
                total_time = time.time() - start_time
                train_time = train_end - train_start
                pred_time = pred_end - pred_start

                # Memory metrics
                peak_memory = self._get_memory_usage()
                gc_after = gc.get_count()

                # Quality metrics
                quality_metrics = self._calculate_quality_metrics(
                    dataset["label"].values, predictions
                )

                # Create performance metrics
                metrics = AdvancedPerformanceMetrics(
                    algorithm_name=algorithm_name,
                    dataset_size=dataset_size,
                    feature_dimension=feature_dimension,
                    contamination_rate=contamination_rate,
                    total_execution_time_seconds=total_time,
                    training_time_seconds=train_time,
                    prediction_time_seconds=pred_time,
                    initial_memory_mb=initial_memory,
                    peak_memory_mb=peak_memory,
                    memory_growth_mb=peak_memory - initial_memory,
                    training_throughput=dataset_size / train_time
                    if train_time > 0
                    else 0,
                    prediction_throughput=dataset_size / pred_time
                    if pred_time > 0
                    else 0,
                    overall_throughput=dataset_size / total_time
                    if total_time > 0
                    else 0,
                    gc_collections=sum(gc_after) - sum(gc_before),
                    **quality_metrics,
                )

                metrics_list.append(metrics)

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {str(e)}")
                metrics = AdvancedPerformanceMetrics(
                    algorithm_name=algorithm_name,
                    dataset_size=dataset_size,
                    feature_dimension=feature_dimension,
                    contamination_rate=contamination_rate,
                    success=False,
                    error_message=str(e),
                )
                metrics_list.append(metrics)

        # Calculate average metrics
        return self._calculate_average_metrics(metrics_list)

    async def run_scalability_benchmark(
        self,
        algorithm_name: str,
        base_size: int = 1000,
        max_size: int = 50000,
        steps: int = 8,
    ) -> dict[str, Any]:
        """Run scalability benchmark for an algorithm."""
        logger.info(f"Running scalability benchmark for {algorithm_name}")

        # Generate size progression
        sizes = np.logspace(
            np.log10(base_size), np.log10(max_size), steps, dtype=int
        ).tolist()

        results = []
        baseline_time = None

        for size in sizes:
            # Run performance test
            metrics = await self.run_algorithm_performance_test(
                algorithm_name=algorithm_name,
                dataset_size=size,
                feature_dimension=20,
                contamination_rate=0.1,
                iterations=3,
            )

            if baseline_time is None:
                baseline_time = metrics.total_execution_time_seconds

            # Calculate scalability metrics
            scale_factor = size / base_size
            efficiency = baseline_time / (
                metrics.total_execution_time_seconds / scale_factor
            )

            metrics.scalability_factor = scale_factor
            metrics.efficiency_ratio = efficiency

            results.append(metrics)

            logger.info(
                f"Size {size}: {metrics.total_execution_time_seconds:.3f}s, "
                f"efficiency: {efficiency:.3f}"
            )

        # Analyze scalability pattern
        scalability_analysis = self._analyze_scalability_pattern(results)

        return {
            "algorithm": algorithm_name,
            "size_range": f"{base_size}-{max_size}",
            "results": results,
            "scalability_analysis": scalability_analysis,
        }

    async def run_memory_stress_test(
        self,
        algorithm_name: str,
        max_memory_mb: float = 2048.0,
        step_multiplier: float = 1.5,
    ) -> dict[str, Any]:
        """Run memory stress test for an algorithm."""
        logger.info(f"Running memory stress test for {algorithm_name}")

        results = []
        current_size = 1000

        while True:
            try:
                # Check current memory usage
                initial_memory = self._get_memory_usage()

                if initial_memory > max_memory_mb * 0.8:
                    logger.warning("Approaching memory limit, stopping test")
                    break

                # Run test
                metrics = await self.run_algorithm_performance_test(
                    algorithm_name=algorithm_name,
                    dataset_size=current_size,
                    feature_dimension=50,
                    contamination_rate=0.1,
                    iterations=1,
                )

                results.append(metrics)

                # Check if we exceeded memory limit
                if metrics.peak_memory_mb > max_memory_mb:
                    logger.info(f"Memory limit reached at size {current_size}")
                    break

                # Increase size
                current_size = int(current_size * step_multiplier)

                # Force garbage collection
                gc.collect()

            except MemoryError:
                logger.info(f"Memory error at size {current_size}")
                break
            except Exception as e:
                logger.error(f"Error in memory stress test: {str(e)}")
                break

        # Analyze memory usage pattern
        memory_analysis = self._analyze_memory_pattern(results)

        return {
            "algorithm": algorithm_name,
            "max_dataset_size_tested": current_size,
            "max_memory_limit_mb": max_memory_mb,
            "results": results,
            "memory_analysis": memory_analysis,
        }

    async def run_comparative_benchmark(
        self,
        algorithms: list[str] | None = None,
        dataset_sizes: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run comparative benchmark across multiple algorithms."""
        if algorithms is None:
            algorithms = list(self.algorithms.keys())

        if dataset_sizes is None:
            dataset_sizes = [1000, 5000, 10000, 25000]

        logger.info(f"Running comparative benchmark for {len(algorithms)} algorithms")

        comparison_results = {}

        for algorithm in algorithms:
            algorithm_results = []

            for size in dataset_sizes:
                metrics = await self.run_algorithm_performance_test(
                    algorithm_name=algorithm,
                    dataset_size=size,
                    feature_dimension=20,
                    contamination_rate=0.1,
                    iterations=3,
                )
                algorithm_results.append(metrics)

            comparison_results[algorithm] = algorithm_results

        # Perform comparative analysis
        comparative_analysis = self._perform_comparative_analysis(comparison_results)

        return {
            "algorithms": algorithms,
            "dataset_sizes": dataset_sizes,
            "results": comparison_results,
            "analysis": comparative_analysis,
            "recommendations": self._generate_algorithm_recommendations(
                comparative_analysis
            ),
        }

    def _generate_test_dataset(
        self, size: int, features: int, contamination: float, random_seed: int = 42
    ) -> pd.DataFrame:
        """Generate synthetic test dataset."""
        np.random.seed(random_seed)

        # Generate normal samples
        normal_size = int(size * (1 - contamination))
        X_normal, _ = make_classification(
            n_samples=normal_size,
            n_features=features,
            n_informative=features // 2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=random_seed,
        )

        # Generate anomalous samples
        anomaly_size = size - normal_size
        X_anomaly = np.random.multivariate_normal(
            mean=np.ones(features) * 5, cov=np.eye(features) * 4, size=anomaly_size
        )

        # Combine data
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([np.zeros(normal_size), np.ones(anomaly_size)])

        # Create DataFrame
        columns = [f"feature_{i}" for i in range(features)]
        df = pd.DataFrame(X, columns=columns)
        df["label"] = y

        # Shuffle
        return df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _calculate_quality_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate quality metrics for anomaly detection."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # Convert predictions to binary (anomaly detection often uses -1, 1)
        y_pred_binary = (y_pred == -1).astype(int)

        try:
            return {
                "accuracy_score": accuracy_score(y_true, y_pred_binary),
                "precision_score": precision_score(
                    y_true, y_pred_binary, zero_division=0
                ),
                "recall_score": recall_score(y_true, y_pred_binary, zero_division=0),
                "f1_score": f1_score(y_true, y_pred_binary, zero_division=0),
                "auc_score": roc_auc_score(y_true, y_pred_binary)
                if len(np.unique(y_true)) > 1
                else 0.5,
            }
        except Exception:
            return {
                "accuracy_score": 0.0,
                "precision_score": 0.0,
                "recall_score": 0.0,
                "f1_score": 0.0,
                "auc_score": 0.5,
            }

    def _calculate_average_metrics(
        self, metrics_list: list[AdvancedPerformanceMetrics]
    ) -> AdvancedPerformanceMetrics:
        """Calculate average metrics from multiple runs."""
        successful_metrics = [m for m in metrics_list if m.success]

        if not successful_metrics:
            return metrics_list[0] if metrics_list else AdvancedPerformanceMetrics()

        # Calculate averages
        avg_metrics = AdvancedPerformanceMetrics(
            algorithm_name=successful_metrics[0].algorithm_name,
            dataset_size=successful_metrics[0].dataset_size,
            feature_dimension=successful_metrics[0].feature_dimension,
            contamination_rate=successful_metrics[0].contamination_rate,
        )

        # Time metrics
        avg_metrics.total_execution_time_seconds = np.mean(
            [m.total_execution_time_seconds for m in successful_metrics]
        )
        avg_metrics.training_time_seconds = np.mean(
            [m.training_time_seconds for m in successful_metrics]
        )
        avg_metrics.prediction_time_seconds = np.mean(
            [m.prediction_time_seconds for m in successful_metrics]
        )

        # Memory metrics
        avg_metrics.peak_memory_mb = np.mean(
            [m.peak_memory_mb for m in successful_metrics]
        )
        avg_metrics.memory_growth_mb = np.mean(
            [m.memory_growth_mb for m in successful_metrics]
        )

        # Throughput metrics
        avg_metrics.training_throughput = np.mean(
            [m.training_throughput for m in successful_metrics]
        )
        avg_metrics.prediction_throughput = np.mean(
            [m.prediction_throughput for m in successful_metrics]
        )
        avg_metrics.overall_throughput = np.mean(
            [m.overall_throughput for m in successful_metrics]
        )

        # Quality metrics
        avg_metrics.accuracy_score = np.mean(
            [m.accuracy_score for m in successful_metrics]
        )
        avg_metrics.precision_score = np.mean(
            [m.precision_score for m in successful_metrics]
        )
        avg_metrics.recall_score = np.mean([m.recall_score for m in successful_metrics])
        avg_metrics.f1_score = np.mean([m.f1_score for m in successful_metrics])
        avg_metrics.auc_score = np.mean([m.auc_score for m in successful_metrics])

        return avg_metrics

    def _analyze_scalability_pattern(
        self, results: list[AdvancedPerformanceMetrics]
    ) -> dict[str, Any]:
        """Analyze scalability pattern from results."""
        if len(results) < 2:
            return {"pattern": "insufficient_data"}

        # Extract data
        sizes = [r.dataset_size for r in results]
        times = [r.total_execution_time_seconds for r in results]
        efficiencies = [r.efficiency_ratio for r in results]

        # Estimate time complexity
        complexity = self._estimate_time_complexity(sizes, times)

        # Calculate scalability score
        avg_efficiency = np.mean(efficiencies)
        scalability_grade = self._calculate_scalability_grade(avg_efficiency)

        return {
            "pattern": complexity,
            "average_efficiency": avg_efficiency,
            "scalability_grade": scalability_grade,
            "linear_scalability_score": avg_efficiency,
            "degradation_point": self._find_degradation_point(results),
        }

    def _estimate_time_complexity(self, sizes: list[int], times: list[float]) -> str:
        """Estimate time complexity from scaling data."""
        if len(sizes) < 3:
            return "O(?)"

        # Calculate growth ratios
        growth_ratios = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = times[i] / times[i - 1]
            growth_ratios.append(time_ratio / size_ratio)

        avg_growth = np.mean(growth_ratios)

        if avg_growth < 1.3:
            return "O(n)"
        elif avg_growth < 2.0:
            return "O(n log n)"
        elif avg_growth < 3.0:
            return "O(n²)"
        else:
            return "O(n³) or worse"

    def _calculate_scalability_grade(self, efficiency: float) -> str:
        """Calculate scalability grade based on efficiency."""
        if efficiency >= 0.9:
            return "A"
        elif efficiency >= 0.75:
            return "B"
        elif efficiency >= 0.6:
            return "C"
        elif efficiency >= 0.4:
            return "D"
        else:
            return "F"

    def _find_degradation_point(
        self, results: list[AdvancedPerformanceMetrics]
    ) -> int | None:
        """Find the point where performance starts to degrade significantly."""
        efficiencies = [r.efficiency_ratio for r in results]

        for i in range(1, len(efficiencies)):
            if efficiencies[i] < 0.5 and efficiencies[i - 1] >= 0.5:
                return results[i].dataset_size

        return None

    def _analyze_memory_pattern(
        self, results: list[AdvancedPerformanceMetrics]
    ) -> dict[str, Any]:
        """Analyze memory usage pattern."""
        if not results:
            return {"pattern": "no_data"}

        sizes = [r.dataset_size for r in results]
        memory_usage = [r.peak_memory_mb for r in results]

        # Calculate memory efficiency (samples per MB)
        memory_efficiency = [
            s / m if m > 0 else 0 for s, m in zip(sizes, memory_usage, strict=False)
        ]

        # Estimate memory complexity
        memory_growth_pattern = self._estimate_memory_complexity(sizes, memory_usage)

        return {
            "memory_growth_pattern": memory_growth_pattern,
            "average_memory_efficiency": np.mean(memory_efficiency),
            "max_memory_tested": max(memory_usage),
            "memory_scalability": self._assess_memory_scalability(sizes, memory_usage),
        }

    def _estimate_memory_complexity(self, sizes: list[int], memory: list[float]) -> str:
        """Estimate memory complexity pattern."""
        if len(sizes) < 2:
            return "unknown"

        # Calculate memory growth ratios
        growth_ratios = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i - 1]
            memory_ratio = memory[i] / memory[i - 1]
            growth_ratios.append(memory_ratio / size_ratio)

        avg_ratio = np.mean(growth_ratios)

        if avg_ratio < 1.2:
            return "linear"
        elif avg_ratio < 2.0:
            return "near-linear"
        elif avg_ratio < 3.0:
            return "quadratic"
        else:
            return "exponential"

    def _assess_memory_scalability(self, sizes: list[int], memory: list[float]) -> str:
        """Assess memory scalability."""
        if len(sizes) < 2:
            return "unknown"

        # Check if memory growth is reasonable
        final_efficiency = sizes[-1] / memory[-1] if memory[-1] > 0 else 0
        initial_efficiency = sizes[0] / memory[0] if memory[0] > 0 else 0

        efficiency_ratio = (
            final_efficiency / initial_efficiency if initial_efficiency > 0 else 0
        )

        if efficiency_ratio > 0.8:
            return "excellent"
        elif efficiency_ratio > 0.6:
            return "good"
        elif efficiency_ratio > 0.4:
            return "fair"
        else:
            return "poor"

    def _perform_comparative_analysis(
        self, results: dict[str, list[AdvancedPerformanceMetrics]]
    ) -> dict[str, Any]:
        """Perform comparative analysis across algorithms."""
        analysis = {}

        # Find best performers in each category
        categories = {
            "fastest": lambda m: m.total_execution_time_seconds,
            "most_memory_efficient": lambda m: m.peak_memory_mb,
            "highest_throughput": lambda m: -m.overall_throughput,  # Negative for min
            "most_accurate": lambda m: -m.accuracy_score,  # Negative for min
            "best_f1": lambda m: -m.f1_score,  # Negative for min
        }

        for category, key_func in categories.items():
            best_algorithm = None
            best_score = float("inf")

            for algorithm, metrics_list in results.items():
                avg_score = np.mean([key_func(m) for m in metrics_list if m.success])
                if avg_score < best_score:
                    best_score = avg_score
                    best_algorithm = algorithm

            analysis[category] = best_algorithm

        # Calculate overall rankings
        algorithm_scores = {}
        for algorithm, metrics_list in results.items():
            successful_metrics = [m for m in metrics_list if m.success]
            if successful_metrics:
                # Composite score (weighted)
                score = (
                    np.mean(
                        [m.total_execution_time_seconds for m in successful_metrics]
                    )
                    * 0.25
                    + np.mean([m.peak_memory_mb for m in successful_metrics]) * 0.25
                    + (
                        1
                        - np.mean([m.overall_throughput for m in successful_metrics])
                        / 10000
                    )
                    * 0.25
                    + (1 - np.mean([m.f1_score for m in successful_metrics])) * 0.25
                )
                algorithm_scores[algorithm] = score

        # Sort by composite score
        ranked_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1])
        analysis["overall_ranking"] = [alg for alg, _ in ranked_algorithms]

        return analysis

    def _generate_algorithm_recommendations(
        self, analysis: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on comparative analysis."""
        recommendations = []

        if "fastest" in analysis:
            recommendations.append(f"For speed: Use {analysis['fastest']}")

        if "most_memory_efficient" in analysis:
            recommendations.append(
                f"For memory efficiency: Use {analysis['most_memory_efficient']}"
            )

        if "most_accurate" in analysis:
            recommendations.append(f"For accuracy: Use {analysis['most_accurate']}")

        if "overall_ranking" in analysis and analysis["overall_ranking"]:
            recommendations.append(f"Overall best: {analysis['overall_ranking'][0]}")

        return recommendations


class EnsembleVotingDetector:
    """Simple ensemble detector using voting."""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.detectors = [
            IsolationForest(contamination=contamination, random_state=42),
            LocalOutlierFactor(contamination=contamination, novelty=True),
        ]

    def fit(self, X):
        for detector in self.detectors:
            detector.fit(X)
        return self

    def predict(self, X):
        predictions = np.array([detector.predict(X) for detector in self.detectors])
        # Majority vote
        return np.where(np.mean(predictions, axis=0) < 0, -1, 1)


class EnsembleStackingDetector:
    """Simple ensemble detector using stacking."""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.base_detectors = [
            IsolationForest(contamination=contamination, random_state=42),
            LocalOutlierFactor(contamination=contamination, novelty=True),
        ]
        self.meta_detector = OneClassSVM(nu=contamination)

    def fit(self, X):
        # Train base detectors
        for detector in self.base_detectors:
            detector.fit(X)

        # Get base predictions for meta-learning
        base_predictions = np.column_stack(
            [detector.decision_function(X) for detector in self.base_detectors]
        )

        # Train meta detector
        self.meta_detector.fit(base_predictions)
        return self

    def predict(self, X):
        # Get base predictions
        base_predictions = np.column_stack(
            [detector.decision_function(X) for detector in self.base_detectors]
        )

        # Meta prediction
        return self.meta_detector.predict(base_predictions)


# Pytest test functions
@pytest.fixture
def benchmark_suite():
    """Create benchmark suite for testing."""
    storage_path = Path("test_benchmarks")
    return AlgorithmBenchmarkSuite(storage_path)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_isolation_forest_performance(benchmark_suite):
    """Test Isolation Forest performance."""
    metrics = await benchmark_suite.run_algorithm_performance_test(
        algorithm_name="isolation_forest",
        dataset_size=1000,
        feature_dimension=10,
        contamination_rate=0.1,
        iterations=3,
    )

    assert metrics.success
    assert metrics.total_execution_time_seconds > 0
    assert metrics.peak_memory_mb > 0
    assert 0 <= metrics.accuracy_score <= 1


@pytest.mark.performance
@pytest.mark.asyncio
async def test_lof_performance(benchmark_suite):
    """Test Local Outlier Factor performance."""
    metrics = await benchmark_suite.run_algorithm_performance_test(
        algorithm_name="local_outlier_factor",
        dataset_size=500,  # Smaller size for LOF
        feature_dimension=10,
        contamination_rate=0.1,
        iterations=2,
    )

    assert metrics.success
    assert metrics.total_execution_time_seconds > 0
    assert metrics.peak_memory_mb > 0


@pytest.mark.performance
@pytest.mark.asyncio
async def test_scalability_benchmark(benchmark_suite):
    """Test algorithm scalability."""
    results = await benchmark_suite.run_scalability_benchmark(
        algorithm_name="isolation_forest", base_size=100, max_size=5000, steps=5
    )

    assert "algorithm" in results
    assert "results" in results
    assert "scalability_analysis" in results
    assert len(results["results"]) == 5


@pytest.mark.performance
@pytest.mark.asyncio
async def test_memory_stress_test(benchmark_suite):
    """Test memory stress test."""
    results = await benchmark_suite.run_memory_stress_test(
        algorithm_name="isolation_forest", max_memory_mb=512.0, step_multiplier=2.0
    )

    assert "algorithm" in results
    assert "results" in results
    assert "memory_analysis" in results


@pytest.mark.performance
@pytest.mark.asyncio
async def test_comparative_benchmark(benchmark_suite):
    """Test comparative benchmark."""
    algorithms = ["isolation_forest", "one_class_svm"]

    results = await benchmark_suite.run_comparative_benchmark(
        algorithms=algorithms, dataset_sizes=[100, 500, 1000]
    )

    assert "algorithms" in results
    assert "analysis" in results
    assert "recommendations" in results
    assert len(results["algorithms"]) == 2


@pytest.mark.performance
def test_ensemble_detectors():
    """Test ensemble detector implementations."""
    # Generate test data
    X = np.random.randn(100, 5)

    # Test voting ensemble
    voting_detector = EnsembleVotingDetector(contamination=0.1)
    voting_detector.fit(X)
    predictions = voting_detector.predict(X)
    assert len(predictions) == len(X)

    # Test stacking ensemble
    stacking_detector = EnsembleStackingDetector(contamination=0.1)
    stacking_detector.fit(X)
    predictions = stacking_detector.predict(X)
    assert len(predictions) == len(X)


if __name__ == "__main__":
    # Run standalone benchmark
    import asyncio

    async def main():
        storage_path = Path("benchmark_results")
        suite = AlgorithmBenchmarkSuite(storage_path)

        # Run comparative benchmark
        results = await suite.run_comparative_benchmark(
            algorithms=["isolation_forest", "local_outlier_factor"],
            dataset_sizes=[100, 500, 1000],
        )

        # Save results
        with open(storage_path / "comparative_results.json", "w") as f:
            # Convert results to JSON-serializable format
            json_results = {
                "algorithms": results["algorithms"],
                "dataset_sizes": results["dataset_sizes"],
                "analysis": results["analysis"],
                "recommendations": results["recommendations"],
            }
            json.dump(json_results, f, indent=2)

        print("Benchmark completed successfully!")
        print(f"Results saved to: {storage_path}")

    asyncio.run(main())
