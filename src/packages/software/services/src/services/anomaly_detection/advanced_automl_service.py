"""Advanced AutoML service with intelligent hyperparameter optimization.

This service provides sophisticated AutoML capabilities including:
- Multi-objective optimization (accuracy, speed, interpretability)
- Adaptive learning from user feedback and historical results
- Resource-aware optimization based on computational constraints
- Bayesian optimization with advanced acquisition functions
- Intelligent algorithm selection with performance prediction
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

# Application layer imports
# Core domain imports
from monorepo.domain.entities import Dataset, Detector

# Infrastructure imports - handle optional dependencies
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    MedianPruner = None
    OPTUNA_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Suppress warnings during optimization
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class OptimizationObjective(BaseModel):
    """Optimization objective configuration."""

    name: str
    weight: float = Field(gt=0.0, le=1.0)
    direction: str = Field(pattern="^(maximize|minimize)$")
    threshold: float | None = None
    description: str = ""


class ResourceConstraints(BaseModel):
    """Resource constraints for optimization."""

    max_time_seconds: int = Field(default=3600, ge=60)  # 1 hour default
    max_trials: int = Field(default=100, ge=10)
    max_memory_mb: int = Field(default=4096, ge=512)
    max_cpu_cores: int = Field(default=4, ge=1)
    gpu_available: bool = False
    prefer_speed: bool = False


class OptimizationHistory(BaseModel):
    """Historical optimization results for learning."""

    data_collection_characteristics: dict[str, Any]
    algorithm_name: str
    best_parameters: dict[str, Any]
    performance_measurements: dict[str, float]
    optimization_time: float
    resource_usage: dict[str, float]
    user_feedback: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class AdvancedAutoMLService:
    """Advanced AutoML service with intelligent optimization."""

    def __init__(
        self,
        optimization_storage_path: Path = Path("./automl_storage"),
        enable_distributed: bool = False,
        n_parallel_jobs: int = 1,
    ):
        """Initialize advanced AutoML service.

        Args:
            optimization_storage_path: Path for storing optimization history
            enable_distributed: Enable distributed optimization
            n_parallel_jobs: Number of parallel optimization jobs
        """
        self.storage_path = optimization_storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.enable_distributed = enable_distributed
        self.n_parallel_jobs = n_parallel_jobs

        # Optimization history for learning
        self.optimization_history: list[OptimizationHistory] = []
        self.load_optimization_history()

        # Algorithm knowledge base
        self.algorithm_knowledge = self._initialize_algorithm_knowledge()

        # Default objectives
        self.default_objectives = [
            OptimizationObjective(
                name="accuracy",
                weight=0.4,
                direction="maximize",
                description="Processing accuracy (AUC-ROC)",
            ),
            OptimizationObjective(
                name="speed",
                weight=0.3,
                direction="maximize",
                description="Training and inference speed",
            ),
            OptimizationObjective(
                name="interpretability",
                weight=0.2,
                direction="maximize",
                description="Processor interpretability score",
            ),
            OptimizationObjective(
                name="memory_efficiency",
                weight=0.1,
                direction="maximize",
                description="Memory usage efficiency",
            ),
        ]

    def _initialize_algorithm_knowledge(self) -> dict[str, dict[str, Any]]:
        """Initialize algorithm knowledge base."""
        return {
            "IsolationForest": {
                "complexity": "low",
                "interpretability": "high",
                "scalability": "high",
                "typical_performance": {"accuracy": 0.8, "speed": 0.9},
                "parameter_importance": {
                    "n_estimators": 0.8,
                    "contamination": 0.9,
                    "max_features": 0.6,
                },
                "resource_requirements": {"memory": "medium", "cpu": "medium"},
            },
            "LocalOutlierFactor": {
                "complexity": "medium",
                "interpretability": "medium",
                "scalability": "medium",
                "typical_performance": {"accuracy": 0.75, "speed": 0.7},
                "parameter_importance": {
                    "n_neighbors": 0.9,
                    "contamination": 0.8,
                    "algorithm": 0.5,
                },
                "resource_requirements": {"memory": "high", "cpu": "medium"},
            },
            "OneClassSVM": {
                "complexity": "high",
                "interpretability": "low",
                "scalability": "low",
                "typical_performance": {"accuracy": 0.82, "speed": 0.4},
                "parameter_importance": {"nu": 0.9, "gamma": 0.8, "kernel": 0.7},
                "resource_requirements": {"memory": "high", "cpu": "high"},
            },
        }

    async def optimize_detector_advanced(
        self,
        data_collection: DataCollection,
        algorithm_name: str,
        objectives: list[OptimizationObjective] | None = None,
        constraints: ResourceConstraints | None = None,
        enable_learning: bool = True,
    ) -> tuple[Detector, dict[str, Any]]:
        """Perform advanced multi-objective detector optimization.

        Args:
            data_collection: DataCollection for optimization
            algorithm_name: Algorithm to optimize
            objectives: Optimization objectives
            constraints: Resource constraints
            enable_learning: Enable learning from optimization history

        Returns:
            Tuple of (optimized_detector, optimization_report)
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is required for advanced AutoML optimization")

        objectives = objectives or self.default_objectives
        constraints = constraints or ResourceConstraints()

        logger.info(f"Starting advanced optimization for {algorithm_name}")

        # Analyze data_collection characteristics
        data_collection_chars = self._analyze_data_collection_characteristics(data_collection)

        # Predict optimal parameters using historical data
        initial_suggestions = None
        if enable_learning:
            initial_suggestions = self._predict_optimal_parameters(
                data_collection_chars, algorithm_name
            )

        # Create optimization study
        study_name = f"{algorithm_name}_{data_collection.name}_{int(time.time())}"
        study = optuna.create_study(
            study_name=study_name,
            directions=[
                "maximize" if obj.direction == "maximize" else "minimize"
                for obj in objectives
            ],
            sampler=TPESampler(n_startup_trials=20, n_ei_candidates=24, seed=42),
            pruner=MedianPruner(
                n_startup_trials=5, n_warmup_steps=10, interval_steps=5
            ),
        )

        # Define optimization objective function
        def objective_function(trial):
            return self._evaluate_trial(
                trial,
                data_collection,
                algorithm_name,
                objectives,
                constraints,
                initial_suggestions,
            )

        # Run optimization with resource constraints
        start_time = time.time()

        try:
            study.optimize(
                objective_function,
                n_trials=constraints.max_trials,
                timeout=constraints.max_time_seconds,
                n_jobs=self.n_parallel_jobs if not self.enable_distributed else 1,
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

        optimization_time = time.time() - start_time

        # Get best parameters and create optimized detector
        if study.best_trials:
            best_trial = study.best_trials[0]  # Get first Pareto-optimal solution
            best_params = best_trial.params
        else:
            # Fallback to default parameters
            best_params = self._get_default_parameters(algorithm_name)
            logger.warning("No trials completed successfully, using default parameters")

        # Create optimized detector
        optimized_detector = await self._create_optimized_detector(
            algorithm_name, best_params, data_collection
        )

        # Generate optimization report
        optimization_report = self._generate_optimization_report(
            study, objectives, constraints, optimization_time, data_collection_chars
        )

        # Store optimization history for learning
        if enable_learning:
            history_entry = OptimizationHistory(
                data_collection_characteristics=data_collection_chars,
                algorithm_name=algorithm_name,
                best_parameters=best_params,
                performance_measurements=optimization_report["best_measurements"],
                optimization_time=optimization_time,
                resource_usage=optimization_report["resource_usage"],
            )
            self.optimization_history.append(history_entry)
            self.save_optimization_history()

        logger.info(f"Optimization completed in {optimization_time:.2f}s")

        return optimized_detector, optimization_report

    def _analyze_dataset_characteristics(self, dataset: Dataset) -> dict[str, Any]:
        """Analyze data_collection characteristics for optimization guidance."""
        data = data_collection.data
        n_samples, n_features = data.shape

        characteristics = {
            "n_samples": n_samples,
            "n_features": n_features,
            "size_category": self._categorize_size(n_samples, n_features),
            "feature_types": self._analyze_feature_types(data),
            "data_distribution": self._analyze_distribution(data),
            "sparsity": self._calculate_sparsity(data),
            "correlation_structure": self._analyze_correlations(data),
            "outlier_characteristics": self._analyze_outlier_patterns(data),
        }

        return characteristics

    def _categorize_size(self, n_samples: int, n_features: int) -> str:
        """Categorize data_collection size."""
        if n_samples < 1000 and n_features < 10:
            return "small"
        elif n_samples < 10000 and n_features < 100:
            return "medium"
        elif n_samples < 100000 and n_features < 1000:
            return "large"
        else:
            return "very_large"

    def _analyze_feature_types(self, data: np.ndarray) -> dict[str, float]:
        """Analyze feature type distribution."""
        # Simple heuristic for feature type analysis
        n_features = data.shape[1]

        # Check for integer-like features
        integer_features = 0
        continuous_features = 0

        for i in range(min(n_features, 20)):  # Sample first 20 features
            feature_data = data[:, i]
            if np.all(feature_data == feature_data.astype(int)):
                integer_features += 1
            else:
                continuous_features += 1

        total_sampled = integer_features + continuous_features

        return {
            "integer_ratio": (
                integer_features / total_sampled if total_sampled > 0 else 0
            ),
            "continuous_ratio": (
                continuous_features / total_sampled if total_sampled > 0 else 0
            ),
        }

    def _analyze_distribution(self, data: np.ndarray) -> dict[str, float]:
        """Analyze data distribution characteristics."""
        flattened = data.flatten()

        return {
            "mean": float(np.mean(flattened)),
            "std": float(np.std(flattened)),
            "skewness": float(self._calculate_skewness(flattened)),
            "kurtosis": float(self._calculate_kurtosis(flattened)),
        }

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3.0

    def _calculate_sparsity(self, data: np.ndarray) -> float:
        """Calculate data sparsity ratio."""
        zero_count = np.count_nonzero(data == 0)
        total_elements = data.size
        return zero_count / total_elements if total_elements > 0 else 0.0

    def _analyze_correlations(self, data: np.ndarray) -> dict[str, float]:
        """Analyze correlation structure."""
        try:
            if data.shape[1] > 1:
                corr_matrix = np.corrcoef(data.T)
                # Remove diagonal elements
                corr_matrix = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]

                return {
                    "max_correlation": float(np.max(np.abs(corr_matrix))),
                    "mean_correlation": float(np.mean(np.abs(corr_matrix))),
                    "high_correlation_ratio": float(np.mean(np.abs(corr_matrix) > 0.8)),
                }
            else:
                return {
                    "max_correlation": 0.0,
                    "mean_correlation": 0.0,
                    "high_correlation_ratio": 0.0,
                }
        except Exception:
            return {
                "max_correlation": 0.0,
                "mean_correlation": 0.0,
                "high_correlation_ratio": 0.0,
            }

    def _analyze_outlier_patterns(self, data: np.ndarray) -> dict[str, float]:
        """Analyze existing outlier patterns in data."""
        outlier_stats = {}

        for i in range(min(data.shape[1], 10)):  # Sample first 10 features
            feature_data = data[:, i]
            q1, q3 = np.percentile(feature_data, [25, 75])
            iqr = q3 - q1

            if iqr > 0:
                outlier_threshold = 1.5 * iqr
                outliers = np.sum(
                    (feature_data < q1 - outlier_threshold)
                    | (feature_data > q3 + outlier_threshold)
                )
                outlier_ratio = outliers / len(feature_data)
                outlier_stats[f"feature_{i}_outlier_ratio"] = outlier_ratio

        if outlier_stats:
            return {
                "mean_outlier_ratio": np.mean(list(outlier_stats.values())),
                "max_outlier_ratio": np.max(list(outlier_stats.values())),
            }
        else:
            return {"mean_outlier_ratio": 0.0, "max_outlier_ratio": 0.0}

    def _predict_optimal_parameters(
        self, data_collection_chars: dict[str, Any], algorithm_name: str
    ) -> dict[str, Any] | None:
        """Predict optimal parameters based on historical data."""
        if not self.optimization_history:
            return None

        # Find similar datasets and algorithms
        similar_histories = []
        for history in self.optimization_history:
            if history.algorithm_name == algorithm_name:
                similarity = self._calculate_data_collection_similarity(
                    data_collection_chars, history.data_collection_characteristics
                )
                if similarity > 0.7:  # Similarity threshold
                    similar_histories.append((history, similarity))

        if not similar_histories:
            return None

        # Weight parameters by similarity and performance
        weighted_params = {}
        total_weight = 0

        for history, similarity in similar_histories:
            performance_weight = history.performance_measurements.get("accuracy", 0.5)
            weight = similarity * performance_weight
            total_weight += weight

            for param, value in history.best_parameters.items():
                if param not in weighted_params:
                    weighted_params[param] = 0
                weighted_params[param] += weight * value

        if total_weight > 0:
            # Normalize weighted parameters
            for param in weighted_params:
                weighted_params[param] /= total_weight

        return weighted_params if weighted_params else None

    def _calculate_dataset_similarity(
        self, chars1: dict[str, Any], chars2: dict[str, Any]
    ) -> float:
        """Calculate similarity between data_collection characteristics."""
        # Simple similarity based on key characteristics
        size_similarity = 1.0 - abs(chars1["n_samples"] - chars2["n_samples"]) / max(
            chars1["n_samples"], chars2["n_samples"]
        )

        feature_similarity = 1.0 - abs(
            chars1["n_features"] - chars2["n_features"]
        ) / max(chars1["n_features"], chars2["n_features"])

        # Distribution similarity
        dist1 = chars1.get("data_distribution", {})
        dist2 = chars2.get("data_distribution", {})

        distribution_similarity = 0.5  # Default
        if dist1 and dist2:
            std_diff = abs(dist1.get("std", 1) - dist2.get("std", 1))
            distribution_similarity = 1.0 / (1.0 + std_diff)

        # Weighted average
        overall_similarity = (
            0.4 * size_similarity
            + 0.3 * feature_similarity
            + 0.3 * distribution_similarity
        )

        return overall_similarity

    def _evaluate_trial(
        self,
        trial,
        data_collection: DataCollection,
        algorithm_name: str,
        objectives: list[OptimizationObjective],
        constraints: ResourceConstraints,
        initial_suggestions: dict[str, Any] | None,
    ) -> list[float]:
        """Evaluate a single optimization trial."""
        try:
            # Generate parameters for this trial
            params = self._generate_trial_parameters(
                trial, algorithm_name, initial_suggestions
            )

            # Create detector with trial parameters
            detector = self._create_detector_from_params(algorithm_name, params)

            # Measure performance
            start_time = time.time()
            start_memory = self._get_memory_usage()

            # Fit detector
            detector.fit(data_collection)

            # Generate synthetic test data for evaluation
            test_data = self._generate_evaluation_data(data_collection)

            # Get predictions
            scores = detector.predict(test_data)

            end_time = time.time()
            end_memory = self._get_memory_usage()

            # Calculate measurements for each objective
            objective_values = []

            for objective in objectives:
                if objective.name == "accuracy":
                    # Evaluate accuracy using synthetic anomalies
                    accuracy = self._evaluate_accuracy(scores, test_data)
                    objective_values.append(accuracy)

                elif objective.name == "speed":
                    # Speed score (inverse of time)
                    training_time = end_time - start_time
                    speed_score = 1.0 / (1.0 + training_time)
                    objective_values.append(speed_score)

                elif objective.name == "interpretability":
                    # Algorithm-specific interpretability score
                    interpretability = self._calculate_interpretability(
                        algorithm_name, params
                    )
                    objective_values.append(interpretability)

                elif objective.name == "memory_efficiency":
                    # Memory efficiency score
                    memory_used = max(end_memory - start_memory, 0.1)
                    memory_score = 1.0 / (1.0 + memory_used / 100)  # Normalize to MB
                    objective_values.append(memory_score)

                else:
                    # Default score for unknown objectives
                    objective_values.append(0.5)

            # Check resource constraints
            if (
                end_time - start_time > constraints.max_time_seconds / 10
            ):  # Per trial limit
                raise optuna.TrialPruned()

            if end_memory - start_memory > constraints.max_memory_mb:
                raise optuna.TrialPruned()

            return objective_values

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            # Return poor scores for failed trials
            return [0.1] * len(objectives)

    def _generate_trial_parameters(
        self, trial, algorithm_name: str, initial_suggestions: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Generate parameters for optimization trial."""
        params = {}

        if algorithm_name == "IsolationForest":
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
            params["max_features"] = trial.suggest_float("max_features", 0.1, 1.0)
            params["contamination"] = trial.suggest_float("contamination", 0.01, 0.5)
            params["random_state"] = 42

        elif algorithm_name == "LocalOutlierFactor":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 5, 50)
            params["contamination"] = trial.suggest_float("contamination", 0.01, 0.5)
            params["algorithm"] = trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            )

        elif algorithm_name == "OneClassSVM":
            params["nu"] = trial.suggest_float("nu", 0.01, 0.5)
            params["gamma"] = trial.suggest_categorical(
                "gamma", ["scale", "auto"]
            ) or trial.suggest_float("gamma_value", 1e-6, 1e-1, log=True)
            params["kernel"] = trial.suggest_categorical(
                "kernel", ["rbf", "linear", "poly", "sigmoid"]
            )

        # Apply initial suggestions if available
        if initial_suggestions:
            for param, value in initial_suggestions.items():
                if param in params:
                    # Use suggestion as a hint (add some variation)
                    if isinstance(value, int | float):
                        current_value = params[param]
                        if isinstance(current_value, int):
                            # For integer parameters, use suggestion ± 20%
                            variation = max(1, int(value * 0.2))
                            params[param] = trial.suggest_int(
                                f"{param}_suggested",
                                max(1, int(value - variation)),
                                int(value + variation),
                            )
                        else:
                            # For float parameters, use suggestion ± 20%
                            variation = value * 0.2
                            params[param] = trial.suggest_float(
                                f"{param}_suggested",
                                max(0.001, value - variation),
                                value + variation,
                            )

        return params

    def _create_detector_from_params(self, algorithm_name: str, params: dict[str, Any]):
        """Create detector instance from parameters."""
        # This would integrate with the existing adapter system
        # For now, return a mock detector
        from monorepo.infrastructure.adapters import SklearnAdapter

        try:
            adapter = SklearnAdapter(algorithm_name)
            adapter.algorithm_params = params
            return adapter
        except Exception:
            # Fallback to basic configuration
            adapter = SklearnAdapter(algorithm_name)
            return adapter

    def _generate_evaluation_data(self, dataset: Dataset) -> Dataset:
        """Generate evaluation data with known anomalies."""
        # Create synthetic test data with controlled anomalies
        original_data = data_collection.data
        n_samples, n_features = original_data.shape

        # Use subset of original data as normal samples
        normal_samples = original_data[: int(n_samples * 0.8)]

        # Generate synthetic anomalies
        n_anomalies = max(1, int(n_samples * 0.1))
        anomalies = self._generate_synthetic_anomalies(normal_samples, n_anomalies)

        # Combine normal and anomalous samples
        test_data = np.vstack([normal_samples, anomalies])
        test_labels = np.hstack(
            [
                np.zeros(len(normal_samples)),  # Normal samples
                np.ones(len(anomalies)),  # Anomalies
            ]
        )

        # Create test data_collection
        test_data_collection = DataCollection(
            name=f"{data_collection.name}_test",
            data=test_data,
            features=data_collection.features,
            metadata={"labels": test_labels},
        )

        return test_data_collection

    def _generate_synthetic_anomalies(
        self, normal_data: np.ndarray, n_anomalies: int
    ) -> np.ndarray:
        """Generate synthetic anomalies for evaluation."""
        anomalies = []

        for _ in range(n_anomalies):
            # Method 1: Add noise to random normal sample
            if len(anomalies) < n_anomalies // 2:
                base_sample = normal_data[np.random.randint(len(normal_data))]
                noise = np.random.normal(0, np.std(normal_data) * 2, base_sample.shape)
                anomaly = base_sample + noise

            # Method 2: Create extreme values
            else:
                anomaly = np.random.uniform(
                    np.min(normal_data, axis=0) - 2 * np.std(normal_data, axis=0),
                    np.max(normal_data, axis=0) + 2 * np.std(normal_data, axis=0),
                )

            anomalies.append(anomaly)

        return np.array(anomalies)

    def _evaluate_accuracy(self, scores: np.ndarray, test_dataset: Dataset) -> float:
        """Evaluate processing accuracy."""
        if "labels" not in test_data_collection.metadata:
            return 0.5  # Default score if no labels available

        true_labels = test_data_collection.metadata["labels"]

        try:
            # Convert scores to binary predictions using threshold
            threshold = np.percentile(scores, 90)  # Top 10% as anomalies
            predictions = (scores > threshold).astype(int)

            # Calculate accuracy measurements
            if SKLEARN_AVAILABLE and len(np.unique(true_labels)) > 1:
                auc_score = roc_auc_score(true_labels, scores)
                return auc_score
            else:
                # Simple accuracy if sklearn not available
                accuracy = np.mean(predictions == true_labels)
                return accuracy

        except Exception:
            return 0.5

    def _calculate_interpretability(
        self, algorithm_name: str, params: dict[str, Any]
    ) -> float:
        """Calculate interpretability score for algorithm and parameters."""
        base_scores = {
            "IsolationForest": 0.8,
            "LocalOutlierFactor": 0.6,
            "OneClassSVM": 0.3,
        }

        base_score = base_scores.get(algorithm_name, 0.5)

        # Adjust based on parameters
        if algorithm_name == "IsolationForest":
            # Fewer estimators = more interpretable
            n_estimators = params.get("n_estimators", 100)
            complexity_penalty = min(0.2, (n_estimators - 50) / 500)
            return max(0.1, base_score - complexity_penalty)

        elif algorithm_name == "OneClassSVM":
            # Linear kernel is more interpretable
            kernel = params.get("kernel", "rbf")
            if kernel == "linear":
                return base_score + 0.2
            elif kernel in ["poly", "sigmoid"]:
                return base_score - 0.1

        return base_score

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0

    def _get_default_parameters(self, algorithm_name: str) -> dict[str, Any]:
        """Get default parameters for algorithm."""
        defaults = {
            "IsolationForest": {
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42,
            },
            "LocalOutlierFactor": {"n_neighbors": 20, "contamination": 0.1},
            "OneClassSVM": {"nu": 0.1, "gamma": "scale", "kernel": "rbf"},
        }

        return defaults.get(algorithm_name, {})

    async def _create_optimized_detector(
        self, algorithm_name: str, parameters: dict[str, Any], data_collection: DataCollection
    ) -> Detector:
        """Create and train optimized detector."""
        detector = self._create_detector_from_params(algorithm_name, parameters)
        detector.fit(data_collection)
        return detector

    def _generate_optimization_report(
        self,
        study,
        objectives: list[OptimizationObjective],
        constraints: ResourceConstraints,
        optimization_time: float,
        data_collection_chars: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            "optimization_summary": {
                "total_trials": len(study.trials),
                "optimization_time": optimization_time,
                "successful_trials": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                ),
                "pruned_trials": len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED
                    ]
                ),
            },
            "data_collection_characteristics": data_collection_chars,
            "objectives": [obj.dict() for obj in objectives],
            "constraints": constraints.dict(),
            "best_measurements": {},
            "best_parameters": {},
            "resource_usage": {
                "peak_memory_mb": constraints.max_memory_mb,
                "optimization_time": optimization_time,
            },
            "pareto_optimal_solutions": [],
        }

        # Add best trial information
        if study.best_trials:
            best_trial = study.best_trials[0]
            report["best_parameters"] = best_trial.params
            report["best_measurements"] = {
                f"objective_{i}": value for i, value in enumerate(best_trial.values)
            }

            # Add all Pareto-optimal solutions
            for trial in study.best_trials[:5]:  # Top 5
                solution = {
                    "parameters": trial.params,
                    "objectives": trial.values,
                    "trial_number": trial.number,
                }
                report["pareto_optimal_solutions"].append(solution)

        return report

    def save_optimization_history(self):
        """Save optimization history to disk."""
        history_file = self.storage_path / "optimization_history.json"

        try:
            history_data = [
                {**history.dict(), "timestamp": history.timestamp.isoformat()}
                for history in self.optimization_history
            ]

            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save optimization history: {e}")

    def load_optimization_history(self):
        """Load optimization history from disk."""
        history_file = self.storage_path / "optimization_history.json"

        if not history_file.exists():
            return

        try:
            with open(history_file) as f:
                history_data = json.load(f)

            self.optimization_history = []
            for item in history_data:
                # Convert timestamp back to datetime
                item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                history = OptimizationHistory(**item)
                self.optimization_history.append(history)

        except Exception as e:
            logger.error(f"Failed to load optimization history: {e}")
            self.optimization_history = []

    async def analyze_optimization_trends(self) -> dict[str, Any]:
        """Analyze optimization trends and learning progress."""
        if not self.optimization_history:
            return {"message": "No optimization history available"}

        # Group by algorithm
        algorithm_trends = {}

        for history in self.optimization_history:
            algorithm = history.algorithm_name
            if algorithm not in algorithm_trends:
                algorithm_trends[algorithm] = {
                    "optimizations": [],
                    "performance_trend": [],
                    "parameter_preferences": {},
                }

            algorithm_trends[algorithm]["optimizations"].append(history)

            # Track performance over time
            accuracy = history.performance_measurements.get("accuracy", 0)
            algorithm_trends[algorithm]["performance_trend"].append(accuracy)

            # Track parameter preferences
            for param, value in history.best_parameters.items():
                if param not in algorithm_trends[algorithm]["parameter_preferences"]:
                    algorithm_trends[algorithm]["parameter_preferences"][param] = []
                algorithm_trends[algorithm]["parameter_preferences"][param].append(
                    value
                )

        # Calculate trends
        trends_analysis = {}

        for algorithm, data in algorithm_trends.items():
            performance_trend = data["performance_trend"]

            trends_analysis[algorithm] = {
                "total_optimizations": len(data["optimizations"]),
                "average_performance": (
                    np.mean(performance_trend) if performance_trend else 0
                ),
                "performance_improvement": self._calculate_trend(performance_trend),
                "preferred_parameters": self._analyze_parameter_preferences(
                    data["parameter_preferences"]
                ),
                "learning_rate": self._calculate_learning_rate(performance_trend),
            }

        return {
            "algorithm_trends": trends_analysis,
            "total_optimizations": len(self.optimization_history),
            "learning_insights": self._generate_learning_insights(algorithm_trends),
        }

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"

    def _analyze_parameter_preferences(
        self, param_data: dict[str, list]
    ) -> dict[str, Any]:
        """Analyze parameter preferences from optimization history."""
        preferences = {}

        for param, values in param_data.items():
            if not values:
                continue

            if all(isinstance(v, int | float) for v in values):
                # Numeric parameter
                preferences[param] = {
                    "type": "numeric",
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "range": [min(values), max(values)],
                }
            else:
                # Categorical parameter
                from collections import Counter

                counts = Counter(values)
                preferences[param] = {
                    "type": "categorical",
                    "most_common": counts.most_common(3),
                    "distribution": dict(counts),
                }

        return preferences

    def _calculate_learning_rate(self, performance_values: list[float]) -> float:
        """Calculate learning rate (improvement over time)."""
        if len(performance_values) < 3:
            return 0.0

        # Calculate improvement rate using exponential smoothing
        improvements = []
        for i in range(1, len(performance_values)):
            improvement = performance_values[i] - performance_values[i - 1]
            improvements.append(improvement)

        if improvements:
            return np.mean(improvements)
        return 0.0

    def _generate_learning_insights(self, algorithm_trends: dict) -> list[str]:
        """Generate insights from learning analysis."""
        insights = []

        for algorithm, data in algorithm_trends.items():
            optimizations = data["optimizations"]

            if len(optimizations) >= 3:
                recent_performance = np.mean(
                    [
                        opt.performance_measurements.get("accuracy", 0)
                        for opt in optimizations[-3:]
                    ]
                )
                early_performance = np.mean(
                    [
                        opt.performance_measurements.get("accuracy", 0)
                        for opt in optimizations[:3]
                    ]
                )

                if recent_performance > early_performance + 0.05:
                    insights.append(
                        f"Learning progress detected for {algorithm}: "
                        f"Performance improved by {recent_performance - early_performance:.3f}"
                    )

        if not insights:
            insights.append("Insufficient optimization history for learning insights")

        return insights
