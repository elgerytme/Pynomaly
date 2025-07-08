"""
Automated Model Training Service

Production-ready service for automated machine learning model training
with hyperparameter optimization, performance monitoring, and intelligent
model selection for anomaly detection algorithms.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from pynomaly.application.dto.training_dto import (
    TrainingConfigDTO,
    TrainingRequestDTO,
    TrainingResultDTO,
)
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.model import ModelMetrics, ModelVersion
from pynomaly.domain.entities.training_job import TrainingJob, TrainingStatus
from pynomaly.domain.optimization.pareto_optimizer import ParetoOptimizer
from pynomaly.domain.services.metrics_calculator import MetricsCalculator
from pynomaly.domain.services.model_selector import ModelSelector
from pynomaly.domain.services.model_service import ModelService
from pynomaly.domain.services.statistical_tester import StatisticalTester
from pynomaly.domain.value_objects.hyperparameters import HyperparameterSet
from pynomaly.infrastructure.adapters.algorithm_adapter import AlgorithmAdapter
from pynomaly.infrastructure.config.training_config import TrainingConfig
from pynomaly.infrastructure.persistence.model_repository import ModelRepository
from pynomaly.infrastructure.persistence.training_repository import TrainingRepository
from pynomaly.shared.exceptions import (
    DataValidationError,
    ModelValidationError,
    TrainingError,
)

logger = logging.getLogger(__name__)


class AutomatedTrainingService:
    """
    Automated training service with hyperparameter optimization and intelligent model selection.

    Features:
    - Multi-algorithm parallel training
    - Automated hyperparameter optimization
    - Cross-validation and model evaluation
    - Performance-based model selection
    - Experiment tracking and versioning
    - Resource-aware training scheduling
    """

    def __init__(
        self,
        training_repository: TrainingRepository,
        model_repository: ModelRepository,
        model_service: ModelService,
        algorithm_adapters: Dict[str, AlgorithmAdapter],
        config: TrainingConfig,
    ):
        self.training_repository = training_repository
        self.model_repository = model_repository
        self.model_service = model_service
        self.algorithm_adapters = algorithm_adapters
        self.config = config

        # Training state
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.job_queue: List[TrainingJob] = []
        self.resource_lock = asyncio.Lock()

        # Performance tracking
        self.training_metrics = {}
        self.best_models = {}

    async def start_automated_training(
        self, request: TrainingRequestDTO
    ) -> TrainingJob:
        """
        Start automated training pipeline with multiple algorithms and optimization.

        Args:
            request: Training request with dataset, algorithms, and configuration

        Returns:
            TrainingJob: Created training job
        """
        logger.info(f"Starting automated training for dataset: {request.dataset_id}")

        # Validate request
        await self._validate_training_request(request)

        # Create training job
        job = TrainingJob(
            id=str(uuid.uuid4()),
            dataset_id=request.dataset_id,
            algorithms=request.algorithms or self._get_default_algorithms(),
            config=request.config,
            status=TrainingStatus.PENDING,
            created_at=datetime.utcnow(),
            progress=0.0,
        )

        # Store job
        await self.training_repository.save_job(job)
        self.active_jobs[job.id] = job

        # Start training asynchronously
        asyncio.create_task(self._execute_training_pipeline(job))

        logger.info(f"Training job created: {job.id}")
        return job

    async def _execute_training_pipeline(self, job: TrainingJob) -> None:
        """
        Execute the complete training pipeline for a job.

        Args:
            job: Training job to execute
        """
        try:
            await self._update_job_status(job, TrainingStatus.RUNNING)

            # Load and prepare dataset
            dataset = await self._load_dataset(job.dataset_id)
            X, y = await self._prepare_training_data(dataset, job.config)

            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.config.validation_split,
                random_state=self.config.random_seed,
                stratify=y if self._is_classification_task(y) else None,
            )

            job.progress = 10.0
            await self._update_job_progress(job)

            # Train models for each algorithm
            training_results = []
            algorithms_count = len(job.algorithms)

            for i, algorithm_name in enumerate(job.algorithms):
                logger.info(f"Training {algorithm_name} for job {job.id}")

                try:
                    result = await self._train_algorithm(
                        algorithm_name, X_train, y_train, X_val, y_val, job
                    )
                    training_results.append(result)

                except Exception as e:
                    logger.error(f"Failed to train {algorithm_name}: {e}")
                    continue

                # Update progress
                progress = 10.0 + (80.0 * (i + 1) / algorithms_count)
                job.progress = progress
                await self._update_job_progress(job)

            # Collect ModelEvaluation
            metrics_calculator = MetricsCalculator()
            model_evaluations = []
            for result in training_results:
                evaluation = metrics_calculator.compute(
                    y_true=y_val,
                    y_pred=result["metrics"]["y_pred"],
                    proba=result["metrics"].get("proba"),
                    task_type="classification",
                    confidence_level=0.95,
                )
                model_evaluations.append(evaluation)

            # Call StatisticalTester & ParetoOptimizer
            statistical_tester = StatisticalTester()
            for i, result_a in enumerate(training_results):
                for j, result_b in enumerate(training_results[i + 1 :], i + 1):
                    significant = statistical_tester.test_significance(
                        result_a["metrics"], result_b["metrics"]
                    )
                    # Handle significance result

            pareto_optimizer = ParetoOptimizer(
                objectives=[
                    {"name": "f1_score", "direction": "max"},
                    {"name": "roc_auc", "direction": "max"},
                ]
            )
            pareto_optimal_models = pareto_optimizer.find_pareto_optimal(
                training_results
            )

            # Feed into ModelSelector
            model_selector = ModelSelector(
                primary_metric="f1_score", secondary_metrics=["roc_auc"]
            )
            best_model_info = model_selector.select_best_model(pareto_optimal_models)
            job.best_model_id = (
                best_model_info["selected_model"] if best_model_info else None
            )

            # Persist comparison artifacts
            await self.training_repository.save_comparison_artifacts(
                job.id, best_model_info
            )
            self.best_models[job.id] = best_model_info

            # Finalize job
            job.results = training_results
            job.completed_at = datetime.utcnow()
            job.progress = 100.0

            await self._update_job_status(job, TrainingStatus.COMPLETED)

            # Trigger async progress events
            await self._trigger_async_event("MODEL_TRAINING_COMPLETED", job.id)
            await self._trigger_async_event("MODEL_EVALUATION_COMPLETED", job.id)
            await self._trigger_async_event("SIGNIFICANCE_TESTING_COMPLETED", job.id)
            await self._trigger_async_event("PARETO_OPTIMIZATION_COMPLETED", job.id)
            await self._trigger_async_event("MODEL_SELECTION_COMPLETED", job.id)

            logger.info(f"Training job {job.id} completed successfully")

        except Exception as e:
            logger.error(f"Training job {job.id} failed: {e}")
            job.error_message = str(e)
            await self._update_job_status(job, TrainingStatus.FAILED)

        finally:
            # Cleanup
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]

    async def _train_algorithm(
        self,
        algorithm_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        job: TrainingJob,
    ) -> Dict[str, Any]:
        """
        Train a specific algorithm with hyperparameter optimization.

        Args:
            algorithm_name: Name of the algorithm to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            job: Training job context

        Returns:
            Training result with model and metrics
        """
        adapter = self.algorithm_adapters.get(algorithm_name)
        if not adapter:
            raise TrainingError(f"Algorithm adapter not found: {algorithm_name}")

        # Get hyperparameter search space
        search_space = adapter.get_hyperparameter_space()

        # Perform hyperparameter optimization
        best_params, optimization_history = await self._optimize_hyperparameters(
            adapter, search_space, X_train, y_train, X_val, y_val, job.config
        )

        # Train final model with best parameters
        model = await adapter.train(X_train, y_train, best_params)

        # Evaluate model
        metrics = await self._evaluate_model(model, X_val, y_val, algorithm_name)

        # Create model version
        model_version = ModelVersion(
            id=str(uuid.uuid4()),
            algorithm=algorithm_name,
            hyperparameters=HyperparameterSet(best_params),
            metrics=metrics,
            training_job_id=job.id,
            created_at=datetime.utcnow(),
            model_data=await self._serialize_model(model),
        )

        # Save model
        await self.model_repository.save_version(model_version)

        return {
            "algorithm": algorithm_name,
            "model_id": model_version.id,
            "hyperparameters": best_params,
            "metrics": metrics.to_dict(),
            "optimization_history": optimization_history,
            "training_time": metrics.training_time,
        }

    async def _optimize_hyperparameters(
        self,
        adapter: AlgorithmAdapter,
        search_space: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrainingConfigDTO,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Optimize hyperparameters using the configured optimization strategy.

        Args:
            adapter: Algorithm adapter
            search_space: Hyperparameter search space
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            config: Training configuration

        Returns:
            Tuple of (best_parameters, optimization_history)
        """
        optimization_strategy = config.optimization_strategy or "optuna"

        if optimization_strategy == "optuna":
            return await self._optimize_with_optuna(
                adapter, search_space, X_train, y_train, X_val, y_val, config
            )
        elif optimization_strategy == "grid_search":
            return await self._optimize_with_grid_search(
                adapter, search_space, X_train, y_train, X_val, y_val, config
            )
        elif optimization_strategy == "random_search":
            return await self._optimize_with_random_search(
                adapter, search_space, X_train, y_train, X_val, y_val, config
            )
        else:
            raise TrainingError(
                f"Unknown optimization strategy: {optimization_strategy}"
            )

    async def _optimize_with_optuna(
        self,
        adapter: AlgorithmAdapter,
        search_space: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrainingConfigDTO,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            adapter: Algorithm adapter
            search_space: Hyperparameter search space
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            config: Training configuration

        Returns:
            Tuple of (best_parameters, optimization_history)
        """
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from optuna.samplers import TPESampler
        except ImportError:
            raise TrainingError(
                "Optuna not available. Install with: pip install optuna"
            )

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=config.random_seed),
            pruner=MedianPruner(),
        )

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in search_space.items():
                if param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False),
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )

            # Train and evaluate model
            try:
                model = asyncio.run(adapter.train(X_train, y_train, params))
                score = asyncio.run(self._evaluate_model_score(model, X_val, y_val))
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        # Run optimization
        n_trials = config.optimization_trials or self.config.default_optimization_trials
        study.optimize(objective, n_trials=n_trials)

        # Extract results
        best_params = study.best_params
        optimization_history = [
            {
                "trial": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name,
            }
            for trial in study.trials
        ]

        return best_params, optimization_history

    async def _evaluate_model(
        self, model: Any, X_val: np.ndarray, y_val: np.ndarray, algorithm_name: str
    ) -> ModelMetrics:
        """
        Evaluate a trained model and return comprehensive metrics.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            algorithm_name: Name of the algorithm

        Returns:
            ModelMetrics: Comprehensive model evaluation metrics
        """
        start_time = datetime.utcnow()

        # Make predictions
        y_pred = model.predict(X_val)

        # Handle anomaly detection vs classification
        if self._is_anomaly_detection_task(y_val):
            # For anomaly detection, predictions are anomaly scores
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics_dict = self._calculate_anomaly_metrics(y_val, y_pred, y_pred_binary)
        else:
            # For classification, predictions are class labels
            y_pred_proba = (
                model.predict_proba(X_val)[:, 1]
                if hasattr(model, "predict_proba")
                else y_pred
            )
            metrics_dict = self._calculate_classification_metrics(
                y_val, y_pred, y_pred_proba
            )

        inference_time = (datetime.utcnow() - start_time).total_seconds()

        return ModelMetrics(
            accuracy=metrics_dict["accuracy"],
            precision=metrics_dict["precision"],
            recall=metrics_dict["recall"],
            f1_score=metrics_dict["f1_score"],
            roc_auc=metrics_dict["roc_auc"],
            confusion_matrix=metrics_dict["confusion_matrix"],
            training_time=0.0,  # Will be set by caller
            inference_time=inference_time,
            algorithm=algorithm_name,
            additional_metrics=metrics_dict.get("additional_metrics", {}),
        )

    def _calculate_anomaly_metrics(
        self, y_true: np.ndarray, y_scores: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate metrics for anomaly detection tasks."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": (
                roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "additional_metrics": {
                "anomaly_ratio": np.mean(y_pred),
                "score_mean": np.mean(y_scores),
                "score_std": np.std(y_scores),
            },
        }

    def _calculate_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate metrics for classification tasks."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc": (
                roc_auc_score(y_true, y_pred_proba, multi_class="ovr")
                if len(np.unique(y_true)) > 1
                else 0.0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "additional_metrics": {
                "classification_report": classification_report(
                    y_true, y_pred, output_dict=True
                )
            },
        }

    async def _select_best_model(
        self, training_results: List[Dict[str, Any]], config: TrainingConfigDTO
    ) -> Optional[ModelVersion]:
        """
        Select the best model based on specified criteria.

        Args:
            training_results: Results from all trained models
            config: Training configuration with selection criteria

        Returns:
            Best model version or None if no models trained successfully
        """
        if not training_results:
            return None

        # Define selection criteria
        primary_metric = config.primary_metric or "f1_score"
        selection_strategy = config.selection_strategy or "best_single_metric"

        if selection_strategy == "best_single_metric":
            # Select model with best primary metric
            best_result = max(
                training_results, key=lambda x: x["metrics"].get(primary_metric, 0.0)
            )

        elif selection_strategy == "weighted_score":
            # Select model with best weighted score across multiple metrics
            weights = config.metric_weights or {
                "f1_score": 0.4,
                "precision": 0.3,
                "recall": 0.3,
            }

            def calculate_weighted_score(result):
                metrics = result["metrics"]
                return sum(
                    weights.get(metric, 0.0) * metrics.get(metric, 0.0)
                    for metric in weights
                )

            best_result = max(training_results, key=calculate_weighted_score)

        elif selection_strategy == "pareto_optimal":
            # Select from Pareto-optimal solutions
            pareto_optimal = self._find_pareto_optimal_models(training_results)
            # For simplicity, select the one with best F1 score among Pareto-optimal
            best_result = max(
                pareto_optimal, key=lambda x: x["metrics"].get("f1_score", 0.0)
            )

        else:
            raise TrainingError(f"Unknown selection strategy: {selection_strategy}")

        # Load and return the best model
        model_id = best_result["model_id"]
        return await self.model_repository.get_version(model_id)

    def _find_pareto_optimal_models(
        self, training_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find Pareto-optimal models considering multiple objectives.

        Args:
            training_results: Results from all trained models

        Returns:
            List of Pareto-optimal models
        """
        objectives = ["f1_score", "precision", "recall"]

        def dominates(result1, result2):
            """Check if result1 dominates result2."""
            better_in_all = True
            strictly_better_in_one = False

            for objective in objectives:
                val1 = result1["metrics"].get(objective, 0.0)
                val2 = result2["metrics"].get(objective, 0.0)

                if val1 < val2:
                    better_in_all = False
                    break
                elif val1 > val2:
                    strictly_better_in_one = True

            return better_in_all and strictly_better_in_one

        pareto_optimal = []

        for result in training_results:
            is_dominated = False

            for other_result in training_results:
                if dominates(other_result, result):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_optimal.append(result)

        return pareto_optimal

    # Helper methods
    async def _validate_training_request(self, request: TrainingRequestDTO) -> None:
        """Validate training request."""
        if not request.dataset_id:
            raise DataValidationError("Dataset ID is required")

        # Check if dataset exists
        dataset = await self._load_dataset(request.dataset_id)
        if not dataset:
            raise DataValidationError(f"Dataset not found: {request.dataset_id}")

        # Validate algorithms
        if request.algorithms:
            for algorithm in request.algorithms:
                if algorithm not in self.algorithm_adapters:
                    raise TrainingError(f"Algorithm not supported: {algorithm}")

    async def _load_dataset(self, dataset_id: str) -> Dataset:
        """Load dataset by ID."""
        # This would typically load from a dataset repository
        # For now, return a placeholder
        return Dataset(id=dataset_id, name=f"dataset_{dataset_id}")

    async def _prepare_training_data(
        self, dataset: Dataset, config: TrainingConfigDTO
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from dataset."""
        # This would typically load and preprocess the actual data
        # For now, return placeholder data
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        return X, y

    def _get_default_algorithms(self) -> List[str]:
        """Get default algorithms for training."""
        return list(self.algorithm_adapters.keys())[:3]  # Use first 3 algorithms

    def _is_classification_task(self, y: np.ndarray) -> bool:
        """Check if this is a classification task."""
        return len(np.unique(y)) <= 10  # Arbitrary threshold

    def _is_anomaly_detection_task(self, y: np.ndarray) -> bool:
        """Check if this is an anomaly detection task."""
        return len(np.unique(y)) == 2 and np.min(y) == 0 and np.max(y) == 1

    async def _evaluate_model_score(
        self, model: Any, X_val: np.ndarray, y_val: np.ndarray
    ) -> float:
        """Get a single score for model comparison."""
        y_pred = model.predict(X_val)
        if self._is_classification_task(y_val):
            return f1_score(y_val, y_pred, average="weighted", zero_division=0)
        else:
            # For anomaly detection, use AUC if possible
            if len(np.unique(y_val)) > 1:
                return roc_auc_score(y_val, y_pred)
            else:
                return 0.0

    async def _serialize_model(self, model: Any) -> bytes:
        """Serialize model for storage."""
        import pickle

        return pickle.dumps(model)

    async def _update_job_status(
        self, job: TrainingJob, status: TrainingStatus
    ) -> None:
        """Update job status."""
        job.status = status
        job.updated_at = datetime.utcnow()
        await self.training_repository.update_job(job)

    async def _update_job_progress(self, job: TrainingJob) -> None:
        """Update job progress."""
        job.updated_at = datetime.utcnow()
        await self.training_repository.update_job(job)

    # Public API methods
    async def get_training_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID."""
        return await self.training_repository.get_job(job_id)

    async def list_training_jobs(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
        limit: int = 100,
    ) -> List[TrainingJob]:
        """List training jobs with optional filtering."""
        return await self.training_repository.list_jobs(
            dataset_id=dataset_id, status=status, limit=limit
        )

    async def cancel_training_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        job = await self.training_repository.get_job(job_id)
        if not job:
            return False

        if job.status in [TrainingStatus.PENDING, TrainingStatus.RUNNING]:
            await self._update_job_status(job, TrainingStatus.CANCELLED)

            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

            return True

        return False

    async def get_training_metrics(self) -> dict[str, Any]:
        """Get training service metrics."""
        return {
            "active_jobs": len(self.active_jobs),
            "queued_jobs": len(self.job_queue),
            "training_metrics": self.training_metrics,
            "best_models": self.best_models,
        }

    async def _trigger_async_event(self, event_type: str, job_id: str) -> None:
        """Trigger async progress events for workflow stages."""
        try:
            logger.info(f"Training pipeline event: {event_type} for job {job_id}")
            # Here you could integrate with event bus, websockets, etc.
            # For now, just log the event
        except Exception as e:
            logger.error(f"Failed to trigger event {event_type} for job {job_id}: {e}")
