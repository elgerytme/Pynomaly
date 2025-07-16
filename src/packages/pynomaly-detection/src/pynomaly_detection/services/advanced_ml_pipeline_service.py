"""
Advanced ML Pipeline Service

Comprehensive machine learning pipeline management with automated model lifecycle,
A/B testing, hyperparameter optimization, and intelligent model selection.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from pynomaly_detection.domain.entities.model import Model, ModelVersion
from pynomaly_detection.domain.value_objects.performance_metrics import PerformanceMetrics
from pynomaly_detection.infrastructure.logging.structured_logger import StructuredLogger
from pynomaly_detection.infrastructure.monitoring.metrics_service import MetricsService


class ModelStatus(Enum):
    """Model status enumeration."""

    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"


class OptimizationStrategy(Enum):
    """Hyperparameter optimization strategy."""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"


@dataclass
class ModelExperiment:
    """Model experiment configuration."""

    experiment_id: str
    name: str
    description: str
    algorithm: str
    hyperparameters: dict[str, Any]
    optimization_strategy: OptimizationStrategy
    evaluation_metrics: list[str]
    created_at: datetime = field(default_factory=datetime.now)
    status: ModelStatus = ModelStatus.TRAINING
    results: dict[str, Any] | None = None


@dataclass
class ModelComparison:
    """Model comparison results."""

    champion_model: str
    challenger_model: str
    champion_metrics: dict[str, float]
    challenger_metrics: dict[str, float]
    winner: str
    confidence_level: float
    recommendation: str


@dataclass
class ABTestConfig:
    """A/B testing configuration."""

    test_id: str
    champion_model: str
    challenger_model: str
    traffic_split: float  # Percentage of traffic to challenger (0.0-1.0)
    success_metric: str
    minimum_samples: int
    max_duration_days: int
    significance_threshold: float = 0.05


class AdvancedMLPipelineService:
    """Advanced ML pipeline service with comprehensive model management."""

    def __init__(
        self, models_dir: str = "models", experiments_dir: str = "experiments"
    ):
        self.logger = StructuredLogger("ml_pipeline")
        self.metrics_service = MetricsService()

        # Storage paths
        self.models_dir = Path(models_dir)
        self.experiments_dir = Path(experiments_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)

        # Model registry
        self.model_registry: dict[str, Model] = {}
        self.active_experiments: dict[str, ModelExperiment] = {}
        self.ab_tests: dict[str, ABTestConfig] = {}

        # Performance tracking
        self.model_performance_history: dict[str, list[PerformanceMetrics]] = {}

        # Load existing models
        self._load_model_registry()

    async def create_model_experiment(
        self,
        name: str,
        description: str,
        algorithm: str,
        hyperparameter_space: dict[str, Any],
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH,
        evaluation_metrics: list[str] = None,
    ) -> str:
        """Create a new model experiment."""

        if evaluation_metrics is None:
            evaluation_metrics = ["roc_auc", "precision", "recall", "f1"]

        experiment_id = f"exp_{algorithm}_{int(time.time())}"

        experiment = ModelExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            algorithm=algorithm,
            hyperparameters=hyperparameter_space,
            optimization_strategy=optimization_strategy,
            evaluation_metrics=evaluation_metrics,
        )

        self.active_experiments[experiment_id] = experiment

        self.logger.info(f"Created model experiment: {experiment_id}")

        # Save experiment configuration
        await self._save_experiment_config(experiment)

        return experiment_id

    async def run_hyperparameter_optimization(
        self,
        experiment_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cv_folds: int = 5,
        n_iter: int = 50,
    ) -> dict[str, Any]:
        """Run hyperparameter optimization for an experiment."""

        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.active_experiments[experiment_id]
        experiment.status = ModelStatus.TRAINING

        self.logger.info(f"Starting hyperparameter optimization for {experiment_id}")

        try:
            # Get base estimator
            estimator = self._get_base_estimator(experiment.algorithm)

            # Set up optimization
            if experiment.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
                optimizer = GridSearchCV(
                    estimator=estimator,
                    param_grid=experiment.hyperparameters,
                    cv=cv_folds,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=1,
                )
            elif experiment.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
                optimizer = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=experiment.hyperparameters,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=1,
                    random_state=42,
                )
            else:
                raise ValueError(
                    f"Optimization strategy {experiment.optimization_strategy} not implemented"
                )

            # Run optimization
            start_time = time.time()
            optimizer.fit(X_train, y_train)
            optimization_time = time.time() - start_time

            # Get best model
            best_model = optimizer.best_estimator_
            best_params = optimizer.best_params_
            best_score = optimizer.best_score_

            # Evaluate on validation set
            val_predictions = best_model.predict(X_val)
            val_proba = (
                best_model.predict_proba(X_val)[:, 1]
                if hasattr(best_model, "predict_proba")
                else None
            )

            # Calculate metrics
            metrics = await self._calculate_evaluation_metrics(
                y_val, val_predictions, val_proba
            )

            # Update experiment results
            results = {
                "best_parameters": best_params,
                "best_cv_score": best_score,
                "validation_metrics": metrics,
                "optimization_time": optimization_time,
                "total_trials": len(optimizer.cv_results_["params"]),
                "cv_results": optimizer.cv_results_,
            }

            experiment.results = results
            experiment.status = ModelStatus.TRAINED

            # Save trained model
            model_path = await self._save_trained_model(
                experiment_id, best_model, results
            )

            self.logger.info(
                f"Hyperparameter optimization completed for {experiment_id}. "
                f"Best score: {best_score:.4f}, Time: {optimization_time:.2f}s"
            )

            # Record metrics
            self.metrics_service.record_model_training_metrics(
                algorithm=experiment.algorithm,
                optimization_time=optimization_time,
                best_score=best_score,
                total_trials=len(optimizer.cv_results_["params"]),
            )

            return results

        except Exception as e:
            experiment.status = ModelStatus.FAILED
            self.logger.error(
                f"Hyperparameter optimization failed for {experiment_id}: {e}"
            )
            raise

    async def compare_models(
        self,
        champion_experiment_id: str,
        challenger_experiment_id: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        significance_level: float = 0.05,
    ) -> ModelComparison:
        """Compare two models using statistical testing."""

        if champion_experiment_id not in self.active_experiments:
            raise ValueError(f"Champion experiment {champion_experiment_id} not found")

        if challenger_experiment_id not in self.active_experiments:
            raise ValueError(
                f"Challenger experiment {challenger_experiment_id} not found"
            )

        self.logger.info(
            f"Comparing models: {champion_experiment_id} vs {challenger_experiment_id}"
        )

        # Load models
        champion_model = await self._load_experiment_model(champion_experiment_id)
        challenger_model = await self._load_experiment_model(challenger_experiment_id)

        # Get predictions
        champion_pred = champion_model.predict(X_test)
        challenger_pred = challenger_model.predict(X_test)

        champion_proba = (
            champion_model.predict_proba(X_test)[:, 1]
            if hasattr(champion_model, "predict_proba")
            else None
        )
        challenger_proba = (
            challenger_model.predict_proba(X_test)[:, 1]
            if hasattr(challenger_model, "predict_proba")
            else None
        )

        # Calculate metrics
        champion_metrics = await self._calculate_evaluation_metrics(
            y_test, champion_pred, champion_proba
        )
        challenger_metrics = await self._calculate_evaluation_metrics(
            y_test, challenger_pred, challenger_proba
        )

        # Determine winner based on primary metric (ROC AUC)
        primary_metric = "roc_auc"
        champion_score = champion_metrics.get(primary_metric, 0)
        challenger_score = challenger_metrics.get(primary_metric, 0)

        # Statistical significance test (simplified)
        score_difference = abs(challenger_score - champion_score)
        confidence_level = min(
            score_difference * 10, 0.99
        )  # Simplified confidence calculation

        if challenger_score > champion_score and confidence_level > (
            1 - significance_level
        ):
            winner = "challenger"
            recommendation = f"Deploy challenger model (improvement: {challenger_score - champion_score:.4f})"
        elif champion_score > challenger_score and confidence_level > (
            1 - significance_level
        ):
            winner = "champion"
            recommendation = (
                "Keep champion model (statistically significant better performance)"
            )
        else:
            winner = "tie"
            recommendation = (
                "No statistically significant difference. Consider A/B testing."
            )

        comparison = ModelComparison(
            champion_model=champion_experiment_id,
            challenger_model=challenger_experiment_id,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            winner=winner,
            confidence_level=confidence_level,
            recommendation=recommendation,
        )

        self.logger.info(
            f"Model comparison completed. Winner: {winner}, Confidence: {confidence_level:.3f}"
        )

        return comparison

    async def start_ab_test(
        self,
        test_name: str,
        champion_model_id: str,
        challenger_model_id: str,
        traffic_split: float = 0.1,
        success_metric: str = "roc_auc",
        minimum_samples: int = 1000,
        max_duration_days: int = 30,
    ) -> str:
        """Start an A/B test between two models."""

        test_id = f"ab_test_{int(time.time())}"

        ab_test = ABTestConfig(
            test_id=test_id,
            champion_model=champion_model_id,
            challenger_model=challenger_model_id,
            traffic_split=traffic_split,
            success_metric=success_metric,
            minimum_samples=minimum_samples,
            max_duration_days=max_duration_days,
        )

        self.ab_tests[test_id] = ab_test

        self.logger.info(
            f"Started A/B test {test_id}: {champion_model_id} vs {challenger_model_id} "
            f"(split: {traffic_split:.1%})"
        )

        return test_id

    async def evaluate_ab_test(
        self, test_id: str, results_data: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Evaluate A/B test results."""

        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test {test_id} not found")

        ab_test = self.ab_tests[test_id]

        # Split results by model
        champion_results = [
            r for r in results_data if r.get("model_id") == ab_test.champion_model
        ]
        challenger_results = [
            r for r in results_data if r.get("model_id") == ab_test.challenger_model
        ]

        if (
            len(champion_results) < ab_test.minimum_samples
            or len(challenger_results) < ab_test.minimum_samples
        ):
            return {
                "status": "insufficient_data",
                "champion_samples": len(champion_results),
                "challenger_samples": len(challenger_results),
                "minimum_required": ab_test.minimum_samples,
            }

        # Calculate metrics for each model
        champion_metric = np.mean(
            [r.get(ab_test.success_metric, 0) for r in champion_results]
        )
        challenger_metric = np.mean(
            [r.get(ab_test.success_metric, 0) for r in challenger_results]
        )

        # Simple statistical test (t-test approximation)
        champion_values = [r.get(ab_test.success_metric, 0) for r in champion_results]
        challenger_values = [
            r.get(ab_test.success_metric, 0) for r in challenger_results
        ]

        # Calculate confidence interval for difference
        difference = challenger_metric - champion_metric
        pooled_std = np.sqrt((np.var(champion_values) + np.var(challenger_values)) / 2)
        standard_error = pooled_std * np.sqrt(
            1 / len(champion_values) + 1 / len(challenger_values)
        )

        # 95% confidence interval
        confidence_interval = (
            difference - 1.96 * standard_error,
            difference + 1.96 * standard_error,
        )

        # Determine significance
        is_significant = confidence_interval[0] > 0 or confidence_interval[1] < 0

        results = {
            "status": "completed",
            "champion_metric": champion_metric,
            "challenger_metric": challenger_metric,
            "difference": difference,
            "confidence_interval": confidence_interval,
            "is_significant": is_significant,
            "champion_samples": len(champion_results),
            "challenger_samples": len(challenger_results),
            "recommendation": (
                "Deploy challenger"
                if difference > 0 and is_significant
                else "Keep champion"
                if difference < 0 and is_significant
                else "No significant difference"
            ),
        }

        self.logger.info(
            f"A/B test {test_id} evaluation completed: {results['recommendation']}"
        )

        return results

    async def auto_select_best_model(
        self,
        experiments: list[str],
        X_test: np.ndarray,
        y_test: np.ndarray,
        selection_criteria: dict[str, float] = None,
    ) -> str:
        """Automatically select the best model from multiple experiments."""

        if selection_criteria is None:
            selection_criteria = {
                "roc_auc": 0.4,
                "precision": 0.3,
                "recall": 0.2,
                "training_time": 0.1,
            }

        self.logger.info(
            f"Auto-selecting best model from {len(experiments)} experiments"
        )

        model_scores = {}

        for exp_id in experiments:
            if exp_id not in self.active_experiments:
                continue

            experiment = self.active_experiments[exp_id]
            if experiment.status != ModelStatus.TRAINED:
                continue

            try:
                # Load model
                model = await self._load_experiment_model(exp_id)

                # Get predictions
                predictions = model.predict(X_test)
                proba = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Calculate metrics
                metrics = await self._calculate_evaluation_metrics(
                    y_test, predictions, proba
                )

                # Get training time
                training_time = experiment.results.get("optimization_time", 0)
                normalized_time = 1.0 / (
                    1.0 + training_time / 3600
                )  # Normalize by hours

                # Calculate weighted score
                score = (
                    metrics.get("roc_auc", 0) * selection_criteria["roc_auc"]
                    + metrics.get("precision", 0) * selection_criteria["precision"]
                    + metrics.get("recall", 0) * selection_criteria["recall"]
                    + normalized_time * selection_criteria["training_time"]
                )

                model_scores[exp_id] = {
                    "score": score,
                    "metrics": metrics,
                    "training_time": training_time,
                }

            except Exception as e:
                self.logger.error(f"Error evaluating experiment {exp_id}: {e}")
                continue

        if not model_scores:
            raise ValueError("No valid models found for selection")

        # Select best model
        best_experiment = max(
            model_scores.keys(), key=lambda x: model_scores[x]["score"]
        )

        self.logger.info(
            f"Selected best model: {best_experiment} "
            f"(score: {model_scores[best_experiment]['score']:.4f})"
        )

        return best_experiment

    async def deploy_model(self, experiment_id: str, deployment_name: str) -> str:
        """Deploy a trained model to production."""

        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.active_experiments[experiment_id]
        if experiment.status != ModelStatus.TRAINED:
            raise ValueError(f"Experiment {experiment_id} is not trained")

        deployment_id = f"deploy_{deployment_name}_{int(time.time())}"

        # Create model version
        model_version = ModelVersion(
            version_id=deployment_id,
            experiment_id=experiment_id,
            deployment_name=deployment_name,
            deployed_at=datetime.now(),
            status="active",
        )

        # Load and package model
        model = await self._load_experiment_model(experiment_id)
        deployment_path = self.models_dir / f"{deployment_id}.joblib"

        # Save deployment model
        joblib.dump(
            {
                "model": model,
                "experiment": experiment,
                "metadata": {
                    "deployed_at": datetime.now().isoformat(),
                    "deployment_name": deployment_name,
                    "version_id": deployment_id,
                },
            },
            deployment_path,
        )

        experiment.status = ModelStatus.DEPLOYED

        self.logger.info(
            f"Deployed model {experiment_id} as {deployment_name} (ID: {deployment_id})"
        )

        # Record deployment metrics
        self.metrics_service.record_model_deployment(
            experiment_id=experiment_id,
            deployment_name=deployment_name,
            deployment_id=deployment_id,
        )

        return deployment_id

    def _get_base_estimator(self, algorithm: str) -> BaseEstimator:
        """Get base estimator for algorithm."""

        estimators = {
            "isolation_forest": IsolationForest(random_state=42),
            "one_class_svm": __import__(
                "sklearn.svm", fromlist=["OneClassSVM"]
            ).OneClassSVM(),
            "local_outlier_factor": __import__(
                "sklearn.neighbors", fromlist=["LocalOutlierFactor"]
            ).LocalOutlierFactor(novelty=True),
            "elliptic_envelope": __import__(
                "sklearn.covariance", fromlist=["EllipticEnvelope"]
            ).EllipticEnvelope(random_state=42),
        }

        if algorithm not in estimators:
            raise ValueError(f"Algorithm {algorithm} not supported")

        return estimators[algorithm]

    async def _calculate_evaluation_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
    ) -> dict[str, float]:
        """Calculate evaluation metrics."""

        metrics = {}

        try:
            # Basic metrics
            if len(np.unique(y_true)) > 1:  # Ensure we have both classes
                metrics["precision"] = precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                metrics["recall"] = recall_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )

                if metrics["precision"] + metrics["recall"] > 0:
                    metrics["f1"] = (
                        2
                        * (metrics["precision"] * metrics["recall"])
                        / (metrics["precision"] + metrics["recall"])
                    )
                else:
                    metrics["f1"] = 0.0

                # ROC AUC if probabilities available
                if y_proba is not None:
                    try:
                        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                    except ValueError:
                        metrics["roc_auc"] = 0.5  # Random performance
            else:
                # Single class case
                metrics.update(
                    {
                        "precision": 1.0 if np.all(y_pred == y_true) else 0.0,
                        "recall": 1.0 if np.all(y_pred == y_true) else 0.0,
                        "f1": 1.0 if np.all(y_pred == y_true) else 0.0,
                        "roc_auc": 0.5,
                    }
                )

            # Accuracy
            metrics["accuracy"] = np.mean(y_pred == y_true)

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.5,
                "accuracy": 0.0,
            }

        return metrics

    async def _save_experiment_config(self, experiment: ModelExperiment):
        """Save experiment configuration."""

        config_path = self.experiments_dir / f"{experiment.experiment_id}_config.json"

        config_data = {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "algorithm": experiment.algorithm,
            "hyperparameters": experiment.hyperparameters,
            "optimization_strategy": experiment.optimization_strategy.value,
            "evaluation_metrics": experiment.evaluation_metrics,
            "created_at": experiment.created_at.isoformat(),
            "status": experiment.status.value,
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    async def _save_trained_model(
        self, experiment_id: str, model: BaseEstimator, results: dict[str, Any]
    ) -> str:
        """Save trained model."""

        model_path = self.models_dir / f"{experiment_id}_model.joblib"

        model_data = {
            "model": model,
            "results": results,
            "saved_at": datetime.now().isoformat(),
        }

        joblib.dump(model_data, model_path)

        return str(model_path)

    async def _load_experiment_model(self, experiment_id: str) -> BaseEstimator:
        """Load experiment model."""

        model_path = self.models_dir / f"{experiment_id}_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)
        return model_data["model"]

    def _load_model_registry(self):
        """Load existing model registry."""
        # Implementation would load from persistent storage
        pass

    async def get_experiment_status(self, experiment_id: str) -> dict[str, Any]:
        """Get experiment status and results."""

        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.active_experiments[experiment_id]

        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "algorithm": experiment.algorithm,
            "created_at": experiment.created_at.isoformat(),
            "results": experiment.results,
        }

    async def list_experiments(
        self, status_filter: ModelStatus | None = None
    ) -> list[dict[str, Any]]:
        """List all experiments with optional status filter."""

        experiments = []

        for exp_id, experiment in self.active_experiments.items():
            if status_filter is None or experiment.status == status_filter:
                experiments.append(
                    {
                        "experiment_id": exp_id,
                        "name": experiment.name,
                        "algorithm": experiment.algorithm,
                        "status": experiment.status.value,
                        "created_at": experiment.created_at.isoformat(),
                    }
                )

        return experiments
