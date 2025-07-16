"""Algorithm performance benchmarking and optimization service.

This module provides comprehensive benchmarking capabilities for anomaly detection
algorithms, enabling data-driven optimization decisions during Phase 2 enhancement.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ...domain.entities import Dataset, Detector
from ...infrastructure.adapters import PyODAdapter, SklearnAdapter
from ...infrastructure.config.feature_flags import require_feature


@dataclass
class BenchmarkResult:
    """Container for algorithm benchmark results."""

    algorithm_name: str
    dataset_name: str

    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    average_precision: float = 0.0
    matthews_correlation: float = 0.0

    # Timing metrics
    fit_time: float = 0.0
    predict_time: float = 0.0
    total_time: float = 0.0

    # Resource metrics
    memory_usage: float = 0.0
    cpu_utilization: float = 0.0

    # Data characteristics
    n_samples: int = 0
    n_features: int = 0
    contamination_rate: float = 0.0

    # Model characteristics
    model_size: int = 0  # Serialized model size in bytes

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    parameters: dict[str, Any] = field(default_factory=dict)

    def overall_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        # Weighted combination of metrics
        weights = {
            "accuracy": 0.15,
            "precision": 0.15,
            "recall": 0.15,
            "f1_score": 0.20,
            "roc_auc": 0.20,
            "matthews_correlation": 0.15,
        }

        score = 0.0
        for metric, weight in weights.items():
            value = getattr(self, metric, 0.0)
            # Handle NaN values
            if pd.isna(value):
                value = 0.0
            # Normalize negative scores
            if metric == "matthews_correlation":
                value = (value + 1) / 2  # Convert from [-1,1] to [0,1]
            score += value * weight

        return score * 100

    def efficiency_score(self) -> float:
        """Calculate efficiency score based on speed and memory."""
        if self.total_time <= 0:
            return 0.0

        # Samples per second
        throughput = self.n_samples / self.total_time

        # Memory efficiency (samples per MB)
        memory_efficiency = self.n_samples / max(self.memory_usage, 1.0)

        # Combined efficiency (normalized to 0-100)
        # These thresholds may need adjustment based on actual performance
        throughput_score = min(throughput / 10000, 1.0)  # 10k samples/sec = 100%
        memory_score = min(memory_efficiency / 1000, 1.0)  # 1k samples/MB = 100%

        return (throughput_score * 0.7 + memory_score * 0.3) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "algorithm_name": self.algorithm_name,
            "dataset_name": self.dataset_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "matthews_correlation": self.matthews_correlation,
            "fit_time": self.fit_time,
            "predict_time": self.predict_time,
            "total_time": self.total_time,
            "memory_usage": self.memory_usage,
            "cpu_utilization": self.cpu_utilization,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "contamination_rate": self.contamination_rate,
            "model_size": self.model_size,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters,
            "overall_score": self.overall_score(),
            "efficiency_score": self.efficiency_score(),
        }


class AlgorithmBenchmarkService:
    """Service for benchmarking anomaly detection algorithms."""

    def __init__(self):
        """Initialize the benchmarking service."""
        # Default algorithms to benchmark
        self.default_algorithms = {
            "isolation_forest": {
                "adapter_class": SklearnAdapter,
                "algorithm": "IsolationForest",
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "contamination": [0.05, 0.1, 0.15],
                    "max_features": [0.5, 1.0],
                },
            },
            "local_outlier_factor": {
                "adapter_class": SklearnAdapter,
                "algorithm": "LocalOutlierFactor",
                "param_grid": {
                    "n_neighbors": [10, 20, 30],
                    "contamination": [0.05, 0.1, 0.15],
                    "algorithm": ["auto", "ball_tree", "kd_tree"],
                },
            },
            "one_class_svm": {
                "adapter_class": SklearnAdapter,
                "algorithm": "OneClassSVM",
                "param_grid": {
                    "nu": [0.05, 0.1, 0.15],
                    "kernel": ["rbf", "linear", "poly"],
                    "gamma": ["scale", "auto"],
                },
            },
        }

        # Add PyOD algorithms if available
        try:
            import pyod

            self.default_algorithms.update(
                {
                    "knn": {
                        "adapter_class": PyODAdapter,
                        "algorithm": "KNN",
                        "param_grid": {
                            "n_neighbors": [5, 10, 20],
                            "contamination": [0.05, 0.1, 0.15],
                            "method": ["largest", "mean", "median"],
                        },
                    },
                    "abod": {
                        "adapter_class": PyODAdapter,
                        "algorithm": "ABOD",
                        "param_grid": {
                            "contamination": [0.05, 0.1, 0.15],
                            "n_neighbors": [5, 10, 20],
                        },
                    },
                }
            )
        except ImportError:
            pass

    @require_feature("algorithm_optimization")
    def benchmark_algorithm(
        self,
        algorithm_name: str,
        dataset: Dataset,
        parameters: dict[str, Any] | None = None,
        n_runs: int = 3,
    ) -> list[BenchmarkResult]:
        """Benchmark a single algorithm on a dataset."""
        if algorithm_name not in self.default_algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        alg_config = self.default_algorithms[algorithm_name]
        adapter_class = alg_config["adapter_class"]

        results = []

        for run in range(n_runs):
            # Create detector with specified parameters
            params = parameters or {}

            detector = Detector(
                name=f"{algorithm_name}_run_{run}",
                algorithm_name=alg_config["algorithm"],
                parameters=params,
            )

            # Create adapter instance
            adapter = adapter_class(detector)

            # Measure performance
            result = self._measure_performance(
                adapter, dataset, algorithm_name, f"{dataset.name}_run_{run}"
            )
            result.parameters = params

            results.append(result)

        return results

    @require_feature("algorithm_optimization")
    def benchmark_multiple_algorithms(
        self, dataset: Dataset, algorithms: list[str] | None = None, n_runs: int = 3
    ) -> dict[str, list[BenchmarkResult]]:
        """Benchmark multiple algorithms on a dataset."""
        if algorithms is None:
            algorithms = list(self.default_algorithms.keys())

        results = {}

        for algorithm in algorithms:
            try:
                results[algorithm] = self.benchmark_algorithm(
                    algorithm, dataset, n_runs=n_runs
                )
            except Exception as e:
                warnings.warn(f"Failed to benchmark {algorithm}: {e}", stacklevel=2)
                results[algorithm] = []

        return results

    @require_feature("algorithm_optimization")
    def hyperparameter_search(
        self,
        algorithm_name: str,
        dataset: Dataset,
        param_grid: dict[str, list] | None = None,
        max_combinations: int = 20,
    ) -> list[BenchmarkResult]:
        """Perform hyperparameter search for an algorithm."""
        if algorithm_name not in self.default_algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        alg_config = self.default_algorithms[algorithm_name]

        if param_grid is None:
            param_grid = alg_config["param_grid"]

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)

        # Limit combinations to prevent excessive runtime
        if len(param_combinations) > max_combinations:
            # Sample random combinations
            import random

            param_combinations = random.sample(param_combinations, max_combinations)

        results = []

        for params in param_combinations:
            try:
                result = self.benchmark_algorithm(
                    algorithm_name, dataset, parameters=params, n_runs=1
                )[0]
                results.append(result)
            except Exception as e:
                warnings.warn(
                    f"Failed parameter combination {params}: {e}", stacklevel=2
                )

        # Sort by overall score
        results.sort(key=lambda x: x.overall_score(), reverse=True)

        return results

    @require_feature("algorithm_optimization")
    def recommend_algorithm(
        self,
        dataset: Dataset,
        priority: str = "balanced",  # "speed", "accuracy", "balanced"
        max_evaluation_time: float = 300.0,  # seconds
    ) -> tuple[str, dict[str, Any], BenchmarkResult]:
        """Recommend the best algorithm for a dataset."""
        start_time = time.time()
        algorithm_results = {}

        for algorithm_name in self.default_algorithms:
            if time.time() - start_time > max_evaluation_time:
                break

            try:
                # Quick benchmark with default parameters
                results = self.benchmark_algorithm(algorithm_name, dataset, n_runs=1)
                algorithm_results[algorithm_name] = results[0]
            except Exception as e:
                warnings.warn(f"Failed to evaluate {algorithm_name}: {e}", stacklevel=2)

        if not algorithm_results:
            raise RuntimeError("No algorithms could be successfully evaluated")

        # Select best algorithm based on priority
        best_algorithm = self._select_best_algorithm(algorithm_results, priority)
        best_result = algorithm_results[best_algorithm]

        # Optionally perform hyperparameter optimization for the best algorithm
        if time.time() - start_time < max_evaluation_time * 0.5:
            remaining_time = max_evaluation_time - (time.time() - start_time)
            max_combinations = max(5, int(remaining_time / 10))  # ~10s per combination

            optimized_results = self.hyperparameter_search(
                best_algorithm, dataset, max_combinations=max_combinations
            )

            if (
                optimized_results
                and optimized_results[0].overall_score() > best_result.overall_score()
            ):
                best_result = optimized_results[0]

        return best_algorithm, best_result.parameters, best_result

    def generate_benchmark_report(
        self, results: dict[str, list[BenchmarkResult]]
    ) -> str:
        """Generate a human-readable benchmark report."""
        report = "# Algorithm Benchmark Report\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if not results:
            return report + "No benchmark results available.\n"

        # Summary table
        report += "## Performance Summary\n\n"
        report += "| Algorithm | Overall Score | Efficiency Score | Accuracy | F1 Score | Time (s) |\n"
        report += "|-----------|---------------|------------------|----------|----------|----------|\n"

        for algorithm, result_list in results.items():
            if not result_list:
                continue

            # Average metrics across runs
            avg_overall = np.mean([r.overall_score() for r in result_list])
            avg_efficiency = np.mean([r.efficiency_score() for r in result_list])
            avg_accuracy = np.mean([r.accuracy for r in result_list])
            avg_f1 = np.mean([r.f1_score for r in result_list])
            avg_time = np.mean([r.total_time for r in result_list])

            report += f"| {algorithm} | {avg_overall:.1f} | {avg_efficiency:.1f} | "
            report += f"{avg_accuracy:.3f} | {avg_f1:.3f} | {avg_time:.2f} |\n"

        # Detailed results
        report += "\n## Detailed Results\n\n"

        for algorithm, result_list in results.items():
            if not result_list:
                continue

            report += f"### {algorithm}\n\n"

            best_result = max(result_list, key=lambda x: x.overall_score())

            report += "**Best Configuration:**\n"
            report += f"- Parameters: {best_result.parameters}\n"
            report += f"- Overall Score: {best_result.overall_score():.1f}\n"
            report += f"- Accuracy: {best_result.accuracy:.3f}\n"
            report += f"- Precision: {best_result.precision:.3f}\n"
            report += f"- Recall: {best_result.recall:.3f}\n"
            report += f"- F1 Score: {best_result.f1_score:.3f}\n"
            report += f"- ROC AUC: {best_result.roc_auc:.3f}\n"
            report += f"- Training Time: {best_result.fit_time:.2f}s\n"
            report += f"- Prediction Time: {best_result.predict_time:.2f}s\n"
            report += f"- Memory Usage: {best_result.memory_usage:.1f}MB\n\n"

        return report

    def _measure_performance(
        self, adapter, dataset: Dataset, algorithm_name: str, dataset_name: str
    ) -> BenchmarkResult:
        """Measure comprehensive performance metrics for an algorithm."""
        import pickle

        import psutil

        result = BenchmarkResult(
            algorithm_name=algorithm_name, dataset_name=dataset_name
        )

        # Dataset characteristics
        X = dataset.data
        y = getattr(dataset, "labels", None)

        result.n_samples, result.n_features = X.shape
        if y is not None:
            result.contamination_rate = np.sum(y) / len(y)

        # Measure memory before training
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Measure training time
        start_time = time.time()
        adapter.fit(dataset)
        result.fit_time = time.time() - start_time

        # Measure prediction time
        start_time = time.time()
        detection_result = adapter.detect(dataset)
        result.predict_time = time.time() - start_time

        result.total_time = result.fit_time + result.predict_time

        # Measure memory after training
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        result.memory_usage = memory_after - memory_before

        # Measure model size
        try:
            model_bytes = pickle.dumps(adapter)
            result.model_size = len(model_bytes)
        except Exception:
            result.model_size = 0

        # Calculate performance metrics if labels are available
        if y is not None:
            predictions = detection_result.labels
            scores = detection_result.scores

            try:
                result.accuracy = accuracy_score(y, predictions)
                result.precision = precision_score(y, predictions, zero_division=0)
                result.recall = recall_score(y, predictions, zero_division=0)
                result.f1_score = f1_score(y, predictions, zero_division=0)
                result.matthews_correlation = matthews_corrcoef(y, predictions)

                if len(np.unique(y)) == 2:  # Binary classification
                    result.roc_auc = roc_auc_score(y, scores)
                    result.average_precision = average_precision_score(y, scores)

            except Exception as e:
                warnings.warn(f"Error calculating metrics: {e}", stacklevel=2)

        return result

    def _generate_param_combinations(
        self, param_grid: dict[str, list]
    ) -> list[dict[str, Any]]:
        """Generate all combinations of parameters."""
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination, strict=False))
            combinations.append(param_dict)

        return combinations

    def _select_best_algorithm(
        self, results: dict[str, BenchmarkResult], priority: str
    ) -> str:
        """Select the best algorithm based on priority."""
        if priority == "speed":
            # Prioritize efficiency
            return max(results.keys(), key=lambda k: results[k].efficiency_score())
        elif priority == "accuracy":
            # Prioritize overall performance
            return max(results.keys(), key=lambda k: results[k].overall_score())
        else:  # balanced
            # Weighted combination of performance and efficiency
            def combined_score(k):
                r = results[k]
                return 0.7 * r.overall_score() + 0.3 * r.efficiency_score()

            return max(results.keys(), key=combined_score)
