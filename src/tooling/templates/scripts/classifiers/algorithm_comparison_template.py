#!/usr/bin/env python3
"""
Algorithm Comparison Template

This template provides a comprehensive framework for comparing multiple anomaly detection
algorithms across different metrics, datasets, and performance criteria.
"""

import warnings
from datetime import datetime
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# Machine learning imports
import json
import logging
import time

# Statistical imports
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pynomaly imports (adjust path as needed)
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

try:
    from pynomaly.domain.entities.dataset import Dataset
    from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
    from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter
    from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
except ImportError:
    logger.warning("Pynomaly imports not available, using fallback implementations")


class AlgorithmComparator:
    """
    Comprehensive algorithm comparison framework for anomaly detection.

    Features:
    - Multi-algorithm benchmarking
    - Cross-validation and statistical testing
    - Performance metric analysis
    - Computational complexity assessment
    - Visualization and reporting
    - Algorithm selection recommendations
    """

    def __init__(
        self,
        config: dict[str, Any] = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize the algorithm comparator.

        Args:
            config: Configuration dictionary for comparison parameters
            random_state: Random seed for reproducibility
            verbose: Enable detailed logging
        """
        self.config = config or self._get_default_config()
        self.random_state = random_state
        self.verbose = verbose

        # Results storage
        self.results = {}
        self.algorithm_metadata = {}
        self.comparison_report = {}

        # Set random seeds
        np.random.seed(random_state)

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for algorithm comparison."""
        return {
            "algorithms": {
                "IsolationForest": {
                    "class": IsolationForest,
                    "params": {
                        "contamination": 0.1,
                        "random_state": 42,
                        "n_estimators": 100,
                    },
                    "param_grid": {
                        "contamination": [0.05, 0.1, 0.15],
                        "n_estimators": [50, 100, 200],
                    },
                },
                "LocalOutlierFactor": {
                    "class": LocalOutlierFactor,
                    "params": {
                        "contamination": 0.1,
                        "n_neighbors": 20,
                        "novelty": True,
                    },
                    "param_grid": {
                        "contamination": [0.05, 0.1, 0.15],
                        "n_neighbors": [10, 20, 30],
                    },
                },
                "OneClassSVM": {
                    "class": OneClassSVM,
                    "params": {"nu": 0.1, "kernel": "rbf", "gamma": "scale"},
                    "param_grid": {
                        "nu": [0.05, 0.1, 0.15],
                        "kernel": ["rbf", "linear"],
                        "gamma": ["scale", "auto"],
                    },
                },
                "EllipticEnvelope": {
                    "class": EllipticEnvelope,
                    "params": {"contamination": 0.1, "random_state": 42},
                    "param_grid": {"contamination": [0.05, 0.1, 0.15]},
                },
            },
            "evaluation": {
                "metrics": ["roc_auc", "precision", "recall", "f1"],
                "cross_validation": {"enable": True, "folds": 5, "stratified": True},
                "statistical_tests": {"enable": True, "significance_level": 0.05},
            },
            "performance": {
                "measure_time": True,
                "measure_memory": True,
                "scalability_test": True,
            },
            "hyperparameter_tuning": {
                "enable": True,
                "method": "grid_search",  # 'grid_search', 'random_search'
                "scoring": "roc_auc",
                "cv_folds": 3,
            },
            "reporting": {
                "generate_plots": True,
                "detailed_analysis": True,
                "recommendations": True,
            },
        }

    def compare_algorithms(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        dataset_name: str = "Unknown",
    ) -> dict[str, Any]:
        """
        Compare multiple algorithms on the given dataset.

        Args:
            X: Feature matrix
            y: True labels (optional, for supervised evaluation)
            dataset_name: Name of the dataset for reporting

        Returns:
            Comprehensive comparison results
        """
        logger.info(f"Starting algorithm comparison on dataset: {dataset_name}")

        # Initialize results
        algorithm_results = {}

        # Get algorithms to compare
        algorithms = self.config["algorithms"]

        for algo_name, algo_config in algorithms.items():
            logger.info(f"Evaluating algorithm: {algo_name}")

            try:
                # Evaluate single algorithm
                result = self._evaluate_algorithm(
                    algo_name, algo_config, X, y, dataset_name
                )
                algorithm_results[algo_name] = result

            except Exception as e:
                logger.error(f"Error evaluating {algo_name}: {str(e)}")
                algorithm_results[algo_name] = {"error": str(e), "status": "failed"}

        # Perform comparative analysis
        comparison_results = self._perform_comparative_analysis(
            algorithm_results, X, y, dataset_name
        )

        # Generate final report
        final_report = self._generate_comparison_report(
            algorithm_results, comparison_results, dataset_name
        )

        # Store results
        self.results[dataset_name] = {
            "algorithm_results": algorithm_results,
            "comparison_results": comparison_results,
            "final_report": final_report,
            "dataset_info": {
                "shape": X.shape,
                "has_labels": y is not None,
                "contamination_rate": np.mean(y) if y is not None else None,
            },
        }

        return self.results[dataset_name]

    def _evaluate_algorithm(
        self,
        algo_name: str,
        algo_config: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray | None,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Evaluate a single algorithm comprehensively."""

        # Initialize algorithm
        algorithm_class = algo_config["class"]
        base_params = algo_config["params"].copy()

        # Performance tracking
        start_time = time.time()

        # Basic evaluation with default parameters
        model = algorithm_class(**base_params)

        # Fit the model
        fit_start = time.time()
        model.fit(X)
        fit_time = time.time() - fit_start

        # Predict anomalies
        predict_start = time.time()
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            predictions = model.predict(X)
        else:
            predictions = model.fit_predict(X)
            scores = None
        predict_time = time.time() - predict_start

        # Convert predictions to binary (1 for normal, -1 for anomaly -> 0 for normal, 1 for anomaly)
        binary_predictions = (predictions == -1).astype(int)

        # Calculate metrics
        metrics = {}
        if y is not None:
            if scores is not None:
                metrics["roc_auc"] = roc_auc_score(
                    y, -scores
                )  # Negative because higher scores = more normal

            metrics["precision"] = precision_score(
                y, binary_predictions, zero_division=0
            )
            metrics["recall"] = recall_score(y, binary_predictions, zero_division=0)
            metrics["f1"] = f1_score(y, binary_predictions, zero_division=0)

            # Additional metrics
            tn, fp, fn, tp = confusion_matrix(y, binary_predictions).ravel()
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)

        # Cross-validation evaluation
        cv_results = {}
        if self.config["evaluation"]["cross_validation"]["enable"] and y is not None:
            cv_results = self._cross_validate_algorithm(
                algorithm_class, base_params, X, y
            )

        # Hyperparameter tuning
        tuning_results = {}
        if self.config["hyperparameter_tuning"]["enable"]:
            tuning_results = self._tune_hyperparameters(
                algorithm_class, algo_config, X, y
            )

        # Performance characteristics
        performance = {
            "fit_time": fit_time,
            "predict_time": predict_time,
            "total_time": time.time() - start_time,
            "predictions_per_second": (
                len(X) / predict_time if predict_time > 0 else float("inf")
            ),
            "scalability_score": self._calculate_scalability_score(
                X.shape, fit_time, predict_time
            ),
        }

        # Algorithm characteristics
        characteristics = self._analyze_algorithm_characteristics(
            algo_name, model, X, binary_predictions
        )

        return {
            "status": "success",
            "metrics": metrics,
            "cv_results": cv_results,
            "tuning_results": tuning_results,
            "performance": performance,
            "characteristics": characteristics,
            "model": model,
            "predictions": binary_predictions,
            "scores": scores,
        }

    def _cross_validate_algorithm(
        self, algorithm_class, params: dict[str, Any], X: np.ndarray, y: np.ndarray
    ) -> dict[str, Any]:
        """Perform cross-validation evaluation."""

        cv_config = self.config["evaluation"]["cross_validation"]
        n_folds = cv_config["folds"]

        if cv_config["stratified"]:
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_state
            )
        else:
            from sklearn.model_selection import KFold

            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        cv_scores = {metric: [] for metric in self.config["evaluation"]["metrics"]}
        fold_times = []

        for _fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            _y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            fold_start = time.time()
            model = algorithm_class(**params)
            model.fit(X_train)

            # Predict on test set
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                predictions = model.predict(X_test)
            else:
                predictions = model.fit_predict(X_test)
                scores = None

            fold_time = time.time() - fold_start
            fold_times.append(fold_time)

            # Convert predictions
            binary_predictions = (predictions == -1).astype(int)

            # Calculate metrics for this fold
            if "roc_auc" in cv_scores and scores is not None:
                try:
                    cv_scores["roc_auc"].append(roc_auc_score(y_test, -scores))
                except:
                    cv_scores["roc_auc"].append(0.5)  # Random performance

            if "precision" in cv_scores:
                cv_scores["precision"].append(
                    precision_score(y_test, binary_predictions, zero_division=0)
                )

            if "recall" in cv_scores:
                cv_scores["recall"].append(
                    recall_score(y_test, binary_predictions, zero_division=0)
                )

            if "f1" in cv_scores:
                cv_scores["f1"].append(
                    f1_score(y_test, binary_predictions, zero_division=0)
                )

        # Calculate statistics
        cv_statistics = {}
        for metric, scores in cv_scores.items():
            if scores:  # Only if we have scores
                cv_statistics[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "scores": scores,
                }

        return {
            "cv_statistics": cv_statistics,
            "fold_times": fold_times,
            "avg_fold_time": np.mean(fold_times),
        }

    def _tune_hyperparameters(
        self,
        algorithm_class,
        algo_config: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray | None,
    ) -> dict[str, Any]:
        """Perform hyperparameter tuning."""

        if y is None:
            return {"status": "skipped", "reason": "No labels available for tuning"}

        param_grid = algo_config.get("param_grid", {})
        if not param_grid:
            return {"status": "skipped", "reason": "No parameter grid specified"}

        tuning_config = self.config["hyperparameter_tuning"]

        from sklearn.metrics import make_scorer
        from sklearn.model_selection import GridSearchCV

        # Custom scorer for anomaly detection
        def anomaly_scorer(estimator, X, y):
            if hasattr(estimator, "decision_function"):
                scores = estimator.decision_function(X)
                return roc_auc_score(y, -scores)
            else:
                predictions = estimator.fit_predict(X)
                binary_predictions = (predictions == -1).astype(int)
                return f1_score(y, binary_predictions, zero_division=0)

        scorer = make_scorer(anomaly_scorer)

        # Perform grid search
        start_time = time.time()

        grid_search = GridSearchCV(
            estimator=algorithm_class(),
            param_grid=param_grid,
            scoring=scorer,
            cv=tuning_config["cv_folds"],
            n_jobs=-1,
            error_score=0,
        )

        try:
            grid_search.fit(X, y)
            tuning_time = time.time() - start_time

            return {
                "status": "success",
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "tuning_time": tuning_time,
                "n_combinations": len(grid_search.cv_results_["params"]),
                "cv_results": {
                    "mean_test_score": grid_search.cv_results_["mean_test_score"],
                    "std_test_score": grid_search.cv_results_["std_test_score"],
                    "params": grid_search.cv_results_["params"],
                },
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "tuning_time": time.time() - start_time,
            }

    def _calculate_scalability_score(
        self, data_shape: tuple[int, int], fit_time: float, predict_time: float
    ) -> float:
        """Calculate a scalability score based on time complexity."""
        n_samples, n_features = data_shape

        # Normalize times by data size
        fit_time_per_sample = fit_time / n_samples
        predict_time_per_sample = predict_time / n_samples

        # Calculate scalability score (lower is better, normalize to 0-1)
        total_time_per_sample = fit_time_per_sample + predict_time_per_sample

        # Use exponential decay to map time to score
        scalability_score = np.exp(-total_time_per_sample * 1000)  # Scale factor

        return min(1.0, max(0.0, scalability_score))

    def _analyze_algorithm_characteristics(
        self, algo_name: str, model: Any, X: np.ndarray, predictions: np.ndarray
    ) -> dict[str, Any]:
        """Analyze algorithm-specific characteristics."""

        characteristics = {
            "algorithm_type": self._get_algorithm_type(algo_name),
            "complexity_class": self._get_complexity_class(algo_name),
            "parameter_sensitivity": self._assess_parameter_sensitivity(algo_name),
            "data_requirements": self._get_data_requirements(algo_name),
            "interpretability": self._assess_interpretability(algo_name),
            "anomaly_detection_ratio": np.mean(predictions),
            "supports_online_learning": self._supports_online_learning(algo_name),
            "handles_categorical": self._handles_categorical_features(algo_name),
            "scalability_rating": self._get_scalability_rating(algo_name),
        }

        return characteristics

    def _get_algorithm_type(self, algo_name: str) -> str:
        """Get the type/category of the algorithm."""
        type_mapping = {
            "IsolationForest": "ensemble",
            "LocalOutlierFactor": "density-based",
            "OneClassSVM": "boundary-based",
            "EllipticEnvelope": "statistical",
        }
        return type_mapping.get(algo_name, "unknown")

    def _get_complexity_class(self, algo_name: str) -> str:
        """Get the computational complexity class."""
        complexity_mapping = {
            "IsolationForest": "O(n log n)",
            "LocalOutlierFactor": "O(n²)",
            "OneClassSVM": "O(n²) to O(n³)",
            "EllipticEnvelope": "O(n³)",
        }
        return complexity_mapping.get(algo_name, "unknown")

    def _assess_parameter_sensitivity(self, algo_name: str) -> str:
        """Assess sensitivity to hyperparameters."""
        sensitivity_mapping = {
            "IsolationForest": "low",
            "LocalOutlierFactor": "medium",
            "OneClassSVM": "high",
            "EllipticEnvelope": "low",
        }
        return sensitivity_mapping.get(algo_name, "unknown")

    def _get_data_requirements(self, algo_name: str) -> dict[str, Any]:
        """Get data requirements for the algorithm."""
        requirements_mapping = {
            "IsolationForest": {
                "min_samples": 100,
                "handles_high_dim": True,
                "requires_scaling": False,
                "handles_missing": False,
            },
            "LocalOutlierFactor": {
                "min_samples": 50,
                "handles_high_dim": False,
                "requires_scaling": True,
                "handles_missing": False,
            },
            "OneClassSVM": {
                "min_samples": 50,
                "handles_high_dim": True,
                "requires_scaling": True,
                "handles_missing": False,
            },
            "EllipticEnvelope": {
                "min_samples": 100,
                "handles_high_dim": False,
                "requires_scaling": True,
                "handles_missing": False,
            },
        }
        return requirements_mapping.get(algo_name, {})

    def _assess_interpretability(self, algo_name: str) -> str:
        """Assess algorithm interpretability."""
        interpretability_mapping = {
            "IsolationForest": "medium",
            "LocalOutlierFactor": "high",
            "OneClassSVM": "low",
            "EllipticEnvelope": "high",
        }
        return interpretability_mapping.get(algo_name, "unknown")

    def _supports_online_learning(self, algo_name: str) -> bool:
        """Check if algorithm supports online learning."""
        online_support = {
            "IsolationForest": False,
            "LocalOutlierFactor": False,
            "OneClassSVM": False,
            "EllipticEnvelope": False,
        }
        return online_support.get(algo_name, False)

    def _handles_categorical_features(self, algo_name: str) -> bool:
        """Check if algorithm handles categorical features."""
        categorical_support = {
            "IsolationForest": True,
            "LocalOutlierFactor": False,
            "OneClassSVM": False,
            "EllipticEnvelope": False,
        }
        return categorical_support.get(algo_name, False)

    def _get_scalability_rating(self, algo_name: str) -> str:
        """Get scalability rating."""
        scalability_mapping = {
            "IsolationForest": "high",
            "LocalOutlierFactor": "medium",
            "OneClassSVM": "low",
            "EllipticEnvelope": "low",
        }
        return scalability_mapping.get(algo_name, "unknown")

    def _perform_comparative_analysis(
        self,
        algorithm_results: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray | None,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Perform comparative analysis across algorithms."""

        successful_results = {
            name: result
            for name, result in algorithm_results.items()
            if result.get("status") == "success"
        }

        if len(successful_results) < 2:
            return {
                "status": "insufficient_algorithms",
                "message": "Need at least 2 successful algorithms for comparison",
            }

        # Metric comparison
        metric_comparison = self._compare_metrics(successful_results)

        # Performance comparison
        performance_comparison = self._compare_performance(successful_results)

        # Statistical testing
        statistical_results = {}
        if self.config["evaluation"]["statistical_tests"]["enable"] and y is not None:
            statistical_results = self._perform_statistical_tests(successful_results)

        # Algorithm ranking
        algorithm_ranking = self._rank_algorithms(successful_results, X.shape)

        # Similarity analysis
        similarity_analysis = self._analyze_algorithm_similarity(successful_results)

        return {
            "status": "success",
            "metric_comparison": metric_comparison,
            "performance_comparison": performance_comparison,
            "statistical_results": statistical_results,
            "algorithm_ranking": algorithm_ranking,
            "similarity_analysis": similarity_analysis,
        }

    def _compare_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Compare metrics across algorithms."""

        metrics_data = {}

        # Collect metrics from all algorithms
        for algo_name, result in results.items():
            metrics = result.get("metrics", {})
            for metric_name, value in metrics.items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = {}
                metrics_data[metric_name][algo_name] = value

        # Calculate statistics for each metric
        metric_statistics = {}
        for metric_name, algo_values in metrics_data.items():
            if algo_values:
                values = list(algo_values.values())
                metric_statistics[metric_name] = {
                    "best_algorithm": max(algo_values.items(), key=lambda x: x[1]),
                    "worst_algorithm": min(algo_values.items(), key=lambda x: x[1]),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "range": max(values) - min(values),
                    "all_values": algo_values,
                }

        return {"metrics_data": metrics_data, "statistics": metric_statistics}

    def _compare_performance(self, results: dict[str, Any]) -> dict[str, Any]:
        """Compare performance characteristics across algorithms."""

        performance_data = {}

        # Collect performance metrics
        for algo_name, result in results.items():
            performance = result.get("performance", {})
            performance_data[algo_name] = performance

        # Calculate performance rankings
        performance_rankings = {}

        if performance_data:
            # Fit time ranking (lower is better)
            fit_times = {
                name: perf.get("fit_time", float("inf"))
                for name, perf in performance_data.items()
            }
            performance_rankings["fit_time"] = sorted(
                fit_times.items(), key=lambda x: x[1]
            )

            # Predict time ranking (lower is better)
            predict_times = {
                name: perf.get("predict_time", float("inf"))
                for name, perf in performance_data.items()
            }
            performance_rankings["predict_time"] = sorted(
                predict_times.items(), key=lambda x: x[1]
            )

            # Predictions per second ranking (higher is better)
            throughput = {
                name: perf.get("predictions_per_second", 0)
                for name, perf in performance_data.items()
            }
            performance_rankings["throughput"] = sorted(
                throughput.items(), key=lambda x: x[1], reverse=True
            )

            # Scalability score ranking (higher is better)
            scalability = {
                name: perf.get("scalability_score", 0)
                for name, perf in performance_data.items()
            }
            performance_rankings["scalability"] = sorted(
                scalability.items(), key=lambda x: x[1], reverse=True
            )

        return {"performance_data": performance_data, "rankings": performance_rankings}

    def _perform_statistical_tests(self, results: dict[str, Any]) -> dict[str, Any]:
        """Perform statistical significance testing."""

        # Collect CV scores for statistical testing
        cv_scores = {}
        for algo_name, result in results.items():
            cv_results = result.get("cv_results", {})
            cv_statistics = cv_results.get("cv_statistics", {})

            for metric_name, stats in cv_statistics.items():
                if metric_name not in cv_scores:
                    cv_scores[metric_name] = {}
                cv_scores[metric_name][algo_name] = stats.get("scores", [])

        statistical_results = {}

        for metric_name, algo_scores in cv_scores.items():
            if len(algo_scores) >= 2:
                # Perform pairwise Wilcoxon signed-rank tests
                algorithms = list(algo_scores.keys())
                pairwise_tests = {}

                for i, algo1 in enumerate(algorithms):
                    for algo2 in algorithms[i + 1 :]:
                        scores1 = algo_scores[algo1]
                        scores2 = algo_scores[algo2]

                        if len(scores1) == len(scores2) and len(scores1) > 0:
                            try:
                                statistic, p_value = wilcoxon(scores1, scores2)

                                pairwise_tests[f"{algo1}_vs_{algo2}"] = {
                                    "statistic": statistic,
                                    "p_value": p_value,
                                    "significant": p_value
                                    < self.config["evaluation"]["statistical_tests"][
                                        "significance_level"
                                    ],
                                    "winner": (
                                        algo1
                                        if np.mean(scores1) > np.mean(scores2)
                                        else algo2
                                    ),
                                }
                            except Exception as e:
                                pairwise_tests[f"{algo1}_vs_{algo2}"] = {
                                    "error": str(e)
                                }

                # Perform Friedman test for overall significance
                if len(algorithms) > 2:
                    try:
                        score_matrix = [algo_scores[algo] for algo in algorithms]
                        statistic, p_value = friedmanchisquare(*score_matrix)

                        friedman_result = {
                            "statistic": statistic,
                            "p_value": p_value,
                            "significant": p_value
                            < self.config["evaluation"]["statistical_tests"][
                                "significance_level"
                            ],
                        }
                    except Exception as e:
                        friedman_result = {"error": str(e)}
                else:
                    friedman_result = {
                        "status": "skipped",
                        "reason": "Less than 3 algorithms",
                    }

                statistical_results[metric_name] = {
                    "pairwise_tests": pairwise_tests,
                    "friedman_test": friedman_result,
                }

        return statistical_results

    def _rank_algorithms(
        self, results: dict[str, Any], data_shape: tuple[int, int]
    ) -> dict[str, Any]:
        """Rank algorithms based on multiple criteria."""

        algorithms = list(results.keys())
        n_samples, n_features = data_shape

        # Define ranking criteria
        ranking_criteria = {
            "overall_performance": 0.4,
            "speed": 0.2,
            "scalability": 0.2,
            "interpretability": 0.1,
            "robustness": 0.1,
        }

        algorithm_scores = {}

        for algo_name in algorithms:
            result = results[algo_name]
            scores = {}

            # Overall performance score (average of key metrics)
            metrics = result.get("metrics", {})
            key_metrics = ["roc_auc", "f1", "precision", "recall"]
            performance_scores = [
                metrics.get(m, 0) for m in key_metrics if m in metrics
            ]
            scores["overall_performance"] = (
                np.mean(performance_scores) if performance_scores else 0
            )

            # Speed score (based on predictions per second)
            performance = result.get("performance", {})
            throughput = performance.get("predictions_per_second", 0)
            scores["speed"] = min(
                1.0, throughput / 1000
            )  # Normalize to 1000 predictions/sec

            # Scalability score
            scores["scalability"] = performance.get("scalability_score", 0)

            # Interpretability score
            characteristics = result.get("characteristics", {})
            interpretability = characteristics.get("interpretability", "unknown")
            interpretability_scores = {
                "high": 1.0,
                "medium": 0.6,
                "low": 0.3,
                "unknown": 0.0,
            }
            scores["interpretability"] = interpretability_scores.get(
                interpretability, 0
            )

            # Robustness score (based on CV std)
            cv_results = result.get("cv_results", {})
            cv_statistics = cv_results.get("cv_statistics", {})
            if cv_statistics:
                # Lower std = higher robustness
                stds = [stats.get("std", 1) for stats in cv_statistics.values()]
                avg_std = np.mean(stds)
                scores["robustness"] = max(0, 1 - avg_std)
            else:
                scores["robustness"] = 0.5  # Default for unknown

            # Calculate weighted final score
            final_score = sum(
                scores[criterion] * weight
                for criterion, weight in ranking_criteria.items()
            )

            algorithm_scores[algo_name] = {
                "final_score": final_score,
                "component_scores": scores,
            }

        # Sort by final score
        ranked_algorithms = sorted(
            algorithm_scores.items(), key=lambda x: x[1]["final_score"], reverse=True
        )

        return {
            "ranking_criteria": ranking_criteria,
            "algorithm_scores": algorithm_scores,
            "ranked_list": ranked_algorithms,
            "recommended_algorithm": (
                ranked_algorithms[0][0] if ranked_algorithms else None
            ),
        }

    def _analyze_algorithm_similarity(self, results: dict[str, Any]) -> dict[str, Any]:
        """Analyze similarity between algorithm predictions."""

        algorithms = list(results.keys())
        predictions = {}

        # Collect predictions
        for algo_name in algorithms:
            pred = results[algo_name].get("predictions")
            if pred is not None:
                predictions[algo_name] = pred

        if len(predictions) < 2:
            return {"status": "insufficient_data"}

        # Calculate pairwise similarities
        similarity_matrix = {}

        for _i, algo1 in enumerate(predictions.keys()):
            similarity_matrix[algo1] = {}
            for algo2 in predictions.keys():
                if algo1 == algo2:
                    similarity_matrix[algo1][algo2] = 1.0
                else:
                    # Calculate Jaccard similarity for binary predictions
                    pred1 = predictions[algo1]
                    pred2 = predictions[algo2]

                    intersection = np.sum((pred1 == 1) & (pred2 == 1))
                    union = np.sum((pred1 == 1) | (pred2 == 1))

                    jaccard_sim = intersection / union if union > 0 else 0
                    similarity_matrix[algo1][algo2] = jaccard_sim

        # Find most and least similar pairs
        similarities = []
        for algo1 in similarity_matrix:
            for algo2 in similarity_matrix[algo1]:
                if algo1 < algo2:  # Avoid duplicates
                    sim = similarity_matrix[algo1][algo2]
                    similarities.append((algo1, algo2, sim))

        similarities.sort(key=lambda x: x[2])

        return {
            "status": "success",
            "similarity_matrix": similarity_matrix,
            "most_similar_pair": similarities[-1] if similarities else None,
            "least_similar_pair": similarities[0] if similarities else None,
            "average_similarity": (
                np.mean([s[2] for s in similarities]) if similarities else 0
            ),
        }

    def _generate_comparison_report(
        self,
        algorithm_results: dict[str, Any],
        comparison_results: dict[str, Any],
        dataset_name: str,
    ) -> dict[str, Any]:
        """Generate comprehensive comparison report."""

        report = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "recommendations": {},
            "detailed_analysis": {},
        }

        # Summary
        successful_algorithms = [
            name
            for name, result in algorithm_results.items()
            if result.get("status") == "success"
        ]

        report["summary"] = {
            "total_algorithms_tested": len(algorithm_results),
            "successful_algorithms": len(successful_algorithms),
            "failed_algorithms": len(algorithm_results) - len(successful_algorithms),
            "algorithms_tested": list(algorithm_results.keys()),
        }

        # Recommendations
        if comparison_results.get("status") == "success":
            ranking = comparison_results.get("algorithm_ranking", {})
            recommended = ranking.get("recommended_algorithm")

            if recommended:
                recommended_result = algorithm_results[recommended]

                report["recommendations"] = {
                    "best_overall_algorithm": recommended,
                    "reasons": self._generate_recommendation_reasons(
                        recommended, recommended_result, comparison_results
                    ),
                    "alternative_choices": self._get_alternative_recommendations(
                        ranking, comparison_results
                    ),
                    "use_case_specific": self._get_use_case_recommendations(
                        algorithm_results, comparison_results
                    ),
                }

        # Detailed analysis
        report["detailed_analysis"] = {
            "performance_summary": self._summarize_performance(algorithm_results),
            "metric_analysis": self._analyze_metric_trends(comparison_results),
            "statistical_significance": self._summarize_statistical_tests(
                comparison_results
            ),
            "algorithm_characteristics": self._summarize_characteristics(
                algorithm_results
            ),
        }

        return report

    def _generate_recommendation_reasons(
        self,
        algorithm_name: str,
        algorithm_result: dict[str, Any],
        comparison_results: dict[str, Any],
    ) -> list[str]:
        """Generate reasons for algorithm recommendation."""

        reasons = []

        # Performance reasons
        metrics = algorithm_result.get("metrics", {})
        if metrics.get("roc_auc", 0) > 0.8:
            reasons.append(f"Excellent ROC-AUC score of {metrics['roc_auc']:.3f}")

        if metrics.get("f1", 0) > 0.7:
            reasons.append(f"Strong F1 score of {metrics['f1']:.3f}")

        # Speed reasons
        performance = algorithm_result.get("performance", {})
        if performance.get("scalability_score", 0) > 0.8:
            reasons.append("High scalability for large datasets")

        # Robustness reasons
        cv_results = algorithm_result.get("cv_results", {})
        cv_stats = cv_results.get("cv_statistics", {})
        if cv_stats:
            avg_std = np.mean([stats.get("std", 1) for stats in cv_stats.values()])
            if avg_std < 0.1:
                reasons.append("Consistent performance across cross-validation folds")

        # Characteristics reasons
        characteristics = algorithm_result.get("characteristics", {})
        if characteristics.get("interpretability") == "high":
            reasons.append("High interpretability for understanding results")

        if not reasons:
            reasons.append("Best overall performance across multiple criteria")

        return reasons

    def _get_alternative_recommendations(
        self, ranking: dict[str, Any], comparison_results: dict[str, Any]
    ) -> dict[str, str]:
        """Get alternative algorithm recommendations for specific scenarios."""

        ranked_list = ranking.get("ranked_list", [])
        alternatives = {}

        if len(ranked_list) > 1:
            alternatives["second_best"] = {
                "algorithm": ranked_list[1][0],
                "reason": "Close alternative with different characteristics",
            }

        # Find best for specific criteria
        performance_ranking = comparison_results.get("performance_comparison", {}).get(
            "rankings", {}
        )

        if "throughput" in performance_ranking:
            fastest = performance_ranking["throughput"][0][0]
            alternatives["fastest"] = {
                "algorithm": fastest,
                "reason": "Best choice for real-time applications",
            }

        return alternatives

    def _get_use_case_recommendations(
        self, algorithm_results: dict[str, Any], comparison_results: dict[str, Any]
    ) -> dict[str, str]:
        """Get recommendations for specific use cases."""

        use_cases = {}

        # Find best for different scenarios
        for algo_name, result in algorithm_results.items():
            if result.get("status") != "success":
                continue

            characteristics = result.get("characteristics", {})
            performance = result.get("performance", {})

            # Real-time processing
            if performance.get("predictions_per_second", 0) > 1000:
                use_cases["real_time"] = algo_name

            # High interpretability
            if characteristics.get("interpretability") == "high":
                use_cases["interpretable"] = algo_name

            # Large datasets
            if characteristics.get("scalability_rating") == "high":
                use_cases["large_scale"] = algo_name

            # High precision requirements
            metrics = result.get("metrics", {})
            if metrics.get("precision", 0) > 0.9:
                use_cases["high_precision"] = algo_name

        return use_cases

    def _summarize_performance(
        self, algorithm_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Summarize performance across all algorithms."""

        successful_results = {
            name: result
            for name, result in algorithm_results.items()
            if result.get("status") == "success"
        }

        if not successful_results:
            return {"status": "no_successful_algorithms"}

        # Collect all performance metrics
        all_metrics = {}
        for _algo_name, result in successful_results.items():
            metrics = result.get("metrics", {})
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # Calculate summary statistics
        summary = {}
        for metric_name, values in all_metrics.items():
            summary[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "range": np.max(values) - np.min(values),
            }

        return summary

    def _analyze_metric_trends(
        self, comparison_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze trends in metrics across algorithms."""

        metric_comparison = comparison_results.get("metric_comparison", {})
        if not metric_comparison:
            return {}

        statistics = metric_comparison.get("statistics", {})
        trends = {}

        for metric_name, stats in statistics.items():
            metric_range = stats.get("range", 0)
            stats.get("mean", 0)

            if metric_range > 0.3:
                trends[metric_name] = "high_variance"
            elif metric_range > 0.1:
                trends[metric_name] = "moderate_variance"
            else:
                trends[metric_name] = "low_variance"

        return trends

    def _summarize_statistical_tests(
        self, comparison_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Summarize statistical test results."""

        statistical_results = comparison_results.get("statistical_results", {})
        if not statistical_results:
            return {}

        summary = {}

        for metric_name, results in statistical_results.items():
            pairwise_tests = results.get("pairwise_tests", {})
            significant_pairs = [
                pair
                for pair, test in pairwise_tests.items()
                if test.get("significant", False)
            ]

            friedman_test = results.get("friedman_test", {})

            summary[metric_name] = {
                "significant_differences": len(significant_pairs),
                "total_comparisons": len(pairwise_tests),
                "overall_significant": friedman_test.get("significant", False),
            }

        return summary

    def _summarize_characteristics(
        self, algorithm_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Summarize algorithm characteristics."""

        characteristics_summary = {}

        for algo_name, result in algorithm_results.items():
            if result.get("status") != "success":
                continue

            characteristics = result.get("characteristics", {})
            characteristics_summary[algo_name] = {
                "type": characteristics.get("algorithm_type", "unknown"),
                "complexity": characteristics.get("complexity_class", "unknown"),
                "interpretability": characteristics.get("interpretability", "unknown"),
                "scalability": characteristics.get("scalability_rating", "unknown"),
            }

        return characteristics_summary

    def save_results(self, filepath: str):
        """Save comparison results to file."""

        # Prepare serializable results
        serializable_results = {}

        for dataset_name, results in self.results.items():
            serializable_results[dataset_name] = {
                "comparison_results": results.get("comparison_results", {}),
                "final_report": results.get("final_report", {}),
                "dataset_info": results.get("dataset_info", {}),
                "algorithm_summary": {},
            }

            # Extract key information from algorithm results (exclude model objects)
            algorithm_results = results.get("algorithm_results", {})
            for algo_name, result in algorithm_results.items():
                if result.get("status") == "success":
                    serializable_results[dataset_name]["algorithm_summary"][
                        algo_name
                    ] = {
                        "metrics": result.get("metrics", {}),
                        "performance": result.get("performance", {}),
                        "characteristics": result.get("characteristics", {}),
                        "cv_results": result.get("cv_results", {}),
                        "tuning_results": result.get("tuning_results", {}),
                    }

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")

    def load_results(self, filepath: str):
        """Load comparison results from file."""

        with open(filepath) as f:
            loaded_results = json.load(f)

        self.results.update(loaded_results)
        logger.info(f"Results loaded from {filepath}")


def main():
    """Example usage of the Algorithm Comparator."""

    # Generate sample anomaly detection dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    contamination = 0.1

    # Generate normal data
    X_normal = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=int(n_samples * (1 - contamination)),
    )

    # Generate anomalous data
    X_anomaly = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,
        cov=np.eye(n_features) * 2,
        size=int(n_samples * contamination),
    )

    # Combine data
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])

    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Contamination rate: {np.mean(y):.3f}")

    # Initialize comparator
    comparator = AlgorithmComparator(verbose=True)

    # Run comparison
    results = comparator.compare_algorithms(X, y, "Synthetic Dataset")

    # Print summary
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON RESULTS")
    print("=" * 60)

    final_report = results["final_report"]

    # Summary
    summary = final_report["summary"]
    print("\nSummary:")
    print(f"- Algorithms tested: {summary['total_algorithms_tested']}")
    print(f"- Successful: {summary['successful_algorithms']}")
    print(f"- Failed: {summary['failed_algorithms']}")

    # Recommendations
    recommendations = final_report.get("recommendations", {})
    if recommendations:
        print(
            f"\nRecommended Algorithm: {recommendations.get('best_overall_algorithm', 'None')}"
        )

        reasons = recommendations.get("reasons", [])
        if reasons:
            print("Reasons:")
            for reason in reasons:
                print(f"  - {reason}")

        alternatives = recommendations.get("alternative_choices", {})
        if alternatives:
            print("\nAlternative Choices:")
            for choice_type, choice_info in alternatives.items():
                print(
                    f"  - {choice_type}: {choice_info['algorithm']} ({choice_info['reason']})"
                )

    # Performance summary
    detailed_analysis = final_report.get("detailed_analysis", {})
    performance_summary = detailed_analysis.get("performance_summary", {})

    if performance_summary:
        print("\nPerformance Summary:")
        for metric, stats in performance_summary.items():
            print(
                f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} (range: {stats['min']:.3f} - {stats['max']:.3f})"
            )

    # Algorithm characteristics
    characteristics = detailed_analysis.get("algorithm_characteristics", {})
    if characteristics:
        print("\nAlgorithm Characteristics:")
        for algo, chars in characteristics.items():
            print(
                f"  {algo}: {chars['type']}, {chars['complexity']}, interpretability: {chars['interpretability']}"
            )

    # Save results
    comparator.save_results("algorithm_comparison_results.json")

    print("\nDetailed results saved to 'algorithm_comparison_results.json'")
    print("Algorithm comparison completed successfully!")


if __name__ == "__main__":
    main()
