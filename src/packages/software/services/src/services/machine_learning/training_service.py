"""
Automated Processor Training Service

Production-ready service for automated machine learning processor training
with hyperparameter optimization, performance monitoring, and intelligent
processor selection for anomaly processing algorithms.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

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

from monorepo.application.dto.training_dto import TrainingConfigDTO, TrainingRequestDTO
from monorepo.application.services.algorithm_adapter_registry import AlgorithmAdapter
from monorepo.application.services.model_persistence_service import (
    ModelPersistenceService,
)
from monorepo.domain.entities.dataset import Dataset
from monorepo.domain.entities.model_version import ModelVersion
from monorepo.domain.value_objects.model_metrics import ModelMetrics
from monorepo.infrastructure.persistence.training_repository import TrainingRepository
from monorepo.shared.exceptions import DataValidationError, TrainingError
from monorepo.shared.protocols.repository_protocol import ModelRepositoryProtocol

logger = logging.getLogger(__name__)


class AutomatedTrainingService:
    """
    Automated training service with hyperparameter optimization and intelligent processor selection.

    Features:
    - Multi-algorithm parallel training
    - Automated hyperparameter optimization
    - Cross-validation and processor evaluation
    - Performance-based processor selection
    - Experiment tracking and versioning
    - Resource-aware training scheduling
    """

    def __init__(
        self,
        training_repository: TrainingRepository,
        processor_repository: ModelRepositoryProtocol,
        processor_service: ModelPersistenceService,
        algorithm_adapters: dict[str, AlgorithmAdapter],
        config: dict[str, Any] | None = None,
    ):
        self.training_repository = training_repository
        self.processor_repository = processor_repository
        self.processor_service = processor_service
        self.algorithm_adapters = algorithm_adapters
        self.config = config

        # Training state
        self.active_jobs: dict[str, Any] = {}  # dict[str, TrainingJob]
        self.job_queue: list[Any] = []  # list[TrainingJob]
        self.resource_lock = asyncio.Lock()

        # Performance tracking
        self.training_measurements = {}
        self.best_processors = {}

    async def start_automated_training(
        self, request: TrainingRequestDTO
    ) -> Any:  # TrainingJob
        """
        Start automated training pipeline with multiple algorithms and optimization.

        Args:
            request: Training request with data_collection, algorithms, and configuration

        Returns:
            Any: Created training job  # TrainingJob
        """
        logger.info(f"Starting automated training for data_collection: {request.data_collection_id}")

        # Validate request
        await self._validate_training_request(request)

        # Create training job
        job = {
            "id": str(uuid.uuid4()),
            "data_collection_id": request.data_collection_id,
            "algorithms": request.algorithms or self._get_default_algorithms(),
            "config": request.config,
            "status": "PENDING",  # TrainingStatus.PENDING
            "created_at": datetime.utcnow(),
            "progress": 0.0,
        }

        # Store job
        await self.training_repository.save_job(job)
        self.active_jobs[job.id] = job

        # Start training asynchronously
        asyncio.create_task(self._execute_training_pipeline(job))

        logger.info(f"Training job created: {job.id}")
        return job

    async def _execute_training_pipeline(self, job: Any) -> None:  # TrainingJob
        """
        Execute the complete training pipeline for a job.

        Args:
            job: Training job to execute
        """
        try:
            await self._update_job_status(job, "RUNNING")  # TrainingStatus.RUNNING

            # Load and prepare data_collection
            data_collection = await self._load_data_collection(job.data_collection_id)
            X, y = await self._prepare_training_data(data_collection, job.config)

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

            # Select best processor
            best_processor = await self._select_best_processor(training_results, job.config)
            job.best_processor_id = best_processor.id if best_processor else None

            # Finalize job
            job.results = training_results
            job.completed_at = datetime.utcnow()
            job.progress = 100.0

            await self._update_job_status(job, "COMPLETED")  # TrainingStatus.COMPLETED

            logger.info(f"Training job {job.id} completed successfully")

        except Exception as e:
            logger.error(f"Training job {job.id} failed: {e}")
            job.error_message = str(e)
            await self._update_job_status(job, "FAILED")  # TrainingStatus.FAILED

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
        job: Any,  # TrainingJob
    ) -> dict[str, Any]:
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
            Training result with processor and measurements
        """
        adapter = self.algorithm_adapters.get(algorithm_name)
        if not adapter:
            raise TrainingError(f"Algorithm adapter not found: {algorithm_name}")

        # Get hyperparameter search space
        search_space = self._get_default_hyperparameter_space(algorithm_name)

        # Perform hyperparameter optimization
        best_params, optimization_history = await self._optimize_hyperparameters(
            adapter, search_space, X_train, y_train, X_val, y_val, job.config
        )

        # Create a detector with the best parameters
        from uuid import uuid4

        from monorepo.domain.entities.detector import Detector
        from monorepo.domain.value_objects import ContaminationRate

        detector = Detector(
            id=uuid4(),
            name=f"{algorithm_name}_training_{job.id}",
            algorithm_name=algorithm_name,
            parameters=best_params,
            contamination_rate=ContaminationRate(0.1),  # Default value
        )

        # Create training data_collection entity
        train_df = pd.DataFrame(
            X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])]
        )
        if y_train is not None:
            train_df["target"] = y_train

        training_data_collection = DataCollection(
            name=f"training_data_{job.id}",
            data=train_df,
            feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
            target_column="target" if y_train is not None else None,
        )

        # Fit the processor using the adapter
        adapter.fit(detector, training_data_collection)

        # Create validation data_collection for evaluation
        val_df = pd.DataFrame(
            X_val, columns=[f"feature_{i}" for i in range(X_val.shape[1])]
        )
        if y_val is not None:
            val_df["target"] = y_val

        val_data_collection = DataCollection(
            name=f"validation_data_{job.id}",
            data=val_df,
            feature_names=[f"feature_{i}" for i in range(X_val.shape[1])],
            target_column="target" if y_val is not None else None,
        )

        # Evaluate processor
        measurements = await self._evaluate_processor(
            detector, adapter, val_data_collection, algorithm_name
        )

        # Create processor version
        processor_version = ModelVersion(
            id=str(uuid.uuid4()),
            algorithm=algorithm_name,
            hyperparameters=best_params,  # HyperparameterSet(best_params)
            measurements=measurements,
            training_job_id=job.id,
            created_at=datetime.utcnow(),
            processor_data=await self._serialize_processor(detector),
        )

        # Save processor
        await self.processor_repository.save_version(processor_version)

        return {
            "algorithm": algorithm_name,
            "processor_id": processor_version.id,
            "hyperparameters": best_params,
            "measurements": measurements.to_dict(),
            "optimization_history": optimization_history,
            "training_time": measurements.training_time,
        }

    async def _optimize_hyperparameters(
        self,
        adapter: AlgorithmAdapter,
        search_space: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrainingConfigDTO,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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

    async def _optimize_with_grid_search(
        self,
        adapter: AlgorithmAdapter,
        search_space: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrainingConfigDTO,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Optimize hyperparameters using grid search.

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
            from sklearn.model_selection import ParameterGrid
        except ImportError:
            raise TrainingError(
                "sklearn not available. Install with: pip install scikit-learn"
            )

        # Convert search space to parameter grid format
        param_grid = {}
        for param_name, param_config in search_space.items():
            if param_config["type"] == "categorical":
                param_grid[param_name] = param_config["choices"]
            elif param_config["type"] == "float":
                # Create discrete grid for float parameters
                low, high = param_config["low"], param_config["high"]
                n_steps = min(param_config.get("n_steps", 10), 20)  # Limit grid size
                if param_config.get("log", False):
                    import math

                    param_grid[param_name] = [
                        math.exp(x)
                        for x in np.linspace(math.log(low), math.log(high), n_steps)
                    ]
                else:
                    param_grid[param_name] = list(np.linspace(low, high, n_steps))
            elif param_config["type"] == "int":
                low, high = param_config["low"], param_config["high"]
                n_steps = min(high - low + 1, 20)  # Limit grid size
                param_grid[param_name] = list(
                    range(low, high + 1, max(1, (high - low) // n_steps))
                )

        # Create parameter grid
        grid = ParameterGrid(param_grid)

        # Limit grid size to prevent excessive computation
        max_trials = (
            config.optimization_trials or self.config.default_optimization_trials
        )
        if len(grid) > max_trials:
            # Sample random subset of parameter combinations
            import random

            grid = random.sample(list(grid), max_trials)

        best_params = None
        best_score = -np.inf
        optimization_history = []

        for trial_num, params in enumerate(grid):
            try:
                # Create temporary detector for this trial
                temp_detector = self._create_temp_detector(
                    "grid_search", params, f"trial_{trial_num}"
                )
                temp_train_data_collection = self._create_temp_data_collection(
                    X_train, y_train, f"train_trial_{trial_num}"
                )

                # Train and evaluate processor
                adapter.fit(temp_detector, temp_train_data_collection)
                temp_val_data_collection = self._create_temp_data_collection(
                    X_val, y_val, f"val_trial_{trial_num}"
                )
                score = await self._evaluate_processor_score(
                    temp_detector, adapter, temp_val_data_collection
                )

                # Track history
                optimization_history.append(
                    {
                        "trial": trial_num,
                        "params": params,
                        "value": score,
                        "state": "COMPLETE",
                    }
                )

                # Update best if better
                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.warning(f"Grid search trial {trial_num} failed: {e}")
                optimization_history.append(
                    {
                        "trial": trial_num,
                        "params": params,
                        "value": 0.0,
                        "state": "FAIL",
                    }
                )
                continue

        if best_params is None:
            # Fallback to default parameters
            best_params = {
                param: config["choices"][0]
                if config["type"] == "categorical"
                else config["low"]
                for param, config in search_space.items()
            }

        return best_params, optimization_history

    async def _optimize_with_random_search(
        self,
        adapter: AlgorithmAdapter,
        search_space: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrainingConfigDTO,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Optimize hyperparameters using random search.

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
        import math
        import random

        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        def sample_parameter(param_name: str, param_config: dict[str, Any]) -> Any:
            """Sample a single parameter value."""
            if param_config["type"] == "categorical":
                return random.choice(param_config["choices"])
            elif param_config["type"] == "float":
                low, high = param_config["low"], param_config["high"]
                if param_config.get("log", False):
                    log_low, log_high = math.log(low), math.log(high)
                    return math.exp(random.uniform(log_low, log_high))
                else:
                    return random.uniform(low, high)
            elif param_config["type"] == "int":
                low, high = param_config["low"], param_config["high"]
                return random.randint(low, high)
            else:
                raise ValueError(f"Unknown parameter type: {param_config['type']}")

        best_params = None
        best_score = -np.inf
        optimization_history = []

        n_trials = config.optimization_trials or self.config.default_optimization_trials

        for trial_num in range(n_trials):
            try:
                # Sample random parameters
                params = {}
                for param_name, param_config in search_space.items():
                    params[param_name] = sample_parameter(param_name, param_config)

                # Create temporary detector for this trial
                temp_detector = self._create_temp_detector(
                    "random_search", params, f"trial_{trial_num}"
                )
                temp_train_data_collection = self._create_temp_data_collection(
                    X_train, y_train, f"train_trial_{trial_num}"
                )

                # Train and evaluate processor
                adapter.fit(temp_detector, temp_train_data_collection)
                temp_val_data_collection = self._create_temp_data_collection(
                    X_val, y_val, f"val_trial_{trial_num}"
                )
                score = await self._evaluate_processor_score(
                    temp_detector, adapter, temp_val_data_collection
                )

                # Track history
                optimization_history.append(
                    {
                        "trial": trial_num,
                        "params": params,
                        "value": score,
                        "state": "COMPLETE",
                    }
                )

                # Update best if better
                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.warning(f"Random search trial {trial_num} failed: {e}")
                optimization_history.append(
                    {
                        "trial": trial_num,
                        "params": params if "params" in locals() else {},
                        "value": 0.0,
                        "state": "FAIL",
                    }
                )
                continue

        if best_params is None:
            # Fallback to default parameters
            best_params = {
                param: sample_parameter(param, config)
                for param, config in search_space.items()
            }

        return best_params, optimization_history

    async def _optimize_with_optuna(
        self,
        adapter: AlgorithmAdapter,
        search_space: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: TrainingConfigDTO,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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

            # Train and evaluate processor
            try:
                temp_detector = self._create_temp_detector(
                    "optuna", params, f"trial_{trial.number}"
                )
                temp_train_data_collection = self._create_temp_data_collection(
                    X_train, y_train, f"train_trial_{trial.number}"
                )

                adapter.fit(temp_detector, temp_train_data_collection)
                temp_val_data_collection = self._create_temp_data_collection(
                    X_val, y_val, f"val_trial_{trial.number}"
                )
                score = asyncio.run(
                    self._evaluate_processor_score(temp_detector, adapter, temp_val_data_collection)
                )
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

    async def _evaluate_processor(
        self,
        detector: Any,
        adapter: AlgorithmAdapter,
        val_data_collection: DataCollection,
        algorithm_name: str,
    ) -> ModelMetrics:
        """Evaluate a trained processor and return comprehensive measurements."""
        start_time = datetime.utcnow()

        try:
            # Get predictions and scores using the adapter
            predictions = adapter.predict(detector, val_data_collection)
            scores = adapter.score(detector, val_data_collection)

            # Extract actual score values
            score_values = np.array([score.value for score in scores])

            # Get true labels if available
            if val_data_collection.has_target:
                y_true = val_data_collection.target.values

                # Handle anomaly processing vs classification
                if self._is_anomaly_processing_task(y_true):
                    measurements_dict = self._calculate_anomaly_measurements(
                        y_true, score_values, np.array(predictions)
                    )
                else:
                    # For classification, use predictions directly
                    measurements_dict = self._calculate_classification_measurements(
                        y_true, np.array(predictions), score_values
                    )
            else:
                # For unsupervised learning, create dummy measurements
                measurements_dict = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "roc_auc": 0.0,
                    "confusion_matrix": [[0, 0], [0, 0]],
                    "additional_measurements": {
                        "anomaly_ratio": np.mean(predictions),
                        "score_mean": np.mean(score_values),
                        "score_std": np.std(score_values),
                    },
                }

            inference_time = (datetime.utcnow() - start_time).total_seconds()

            return ModelMetrics(
                accuracy=measurements_dict["accuracy"],
                precision=measurements_dict["precision"],
                recall=measurements_dict["recall"],
                f1_score=measurements_dict["f1_score"],
                roc_auc=measurements_dict["roc_auc"],
                confusion_matrix=measurements_dict["confusion_matrix"],
                training_time=0.0,  # Will be set by caller
                inference_time=inference_time,
                algorithm=algorithm_name,
                additional_measurements=measurements_dict.get("additional_measurements", {}),
            )

        except Exception as e:
            logger.error(f"Error evaluating processor: {e}")
            # Return default measurements
            return ModelMetrics(
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                roc_auc=0.0,
                confusion_matrix=[[0, 0], [0, 0]],
                training_time=0.0,
                inference_time=0.0,
                algorithm=algorithm_name,
                additional_measurements={},
            )

    def _calculate_anomaly_metrics(
        self, y_true: np.ndarray, y_scores: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, Any]:
        """Calculate measurements for anomaly processing tasks."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": (
                roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "additional_measurements": {
                "anomaly_ratio": np.mean(y_pred),
                "score_mean": np.mean(y_scores),
                "score_std": np.std(y_scores),
            },
        }

    def _calculate_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> dict[str, Any]:
        """Calculate measurements for classification tasks."""
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
            "additional_measurements": {
                "classification_report": classification_report(
                    y_true, y_pred, output_dict=True
                )
            },
        }

    async def _select_best_processor(
        self, training_results: list[dict[str, Any]], config: TrainingConfigDTO
    ) -> ModelVersion | None:
        """
        Select the best processor based on specified criteria.

        Args:
            training_results: Results from all trained models
            config: Training configuration with selection criteria

        Returns:
            Best processor version or None if no models trained successfully
        """
        if not training_results:
            return None

        # Define selection criteria
        primary_metric = config.primary_metric or "f1_score"
        selection_strategy = config.selection_strategy or "best_single_metric"

        if selection_strategy == "best_single_metric":
            # Select processor with best primary metric
            best_result = max(
                training_results, key=lambda x: x["measurements"].get(primary_metric, 0.0)
            )

        elif selection_strategy == "weighted_score":
            # Select processor with best weighted score across multiple measurements
            weights = config.metric_weights or {
                "f1_score": 0.4,
                "precision": 0.3,
                "recall": 0.3,
            }

            def calculate_weighted_score(result):
                measurements = result["measurements"]
                return sum(
                    weights.get(metric, 0.0) * measurements.get(metric, 0.0)
                    for metric in weights
                )

            best_result = max(training_results, key=calculate_weighted_score)

        elif selection_strategy == "pareto_optimal":
            # Select from Pareto-optimal solutions
            pareto_optimal = self._find_pareto_optimal_processors(training_results)
            # For simplicity, select the one with best F1 score among Pareto-optimal
            best_result = max(
                pareto_optimal, key=lambda x: x["measurements"].get("f1_score", 0.0)
            )

        else:
            raise TrainingError(f"Unknown selection strategy: {selection_strategy}")

        # Load and return the best processor
        processor_id = best_result["processor_id"]
        return await self.processor_repository.get_version(processor_id)

    def _find_pareto_optimal_models(
        self, training_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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
                val1 = result1["measurements"].get(objective, 0.0)
                val2 = result2["measurements"].get(objective, 0.0)

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
        if not request.data_collection_id:
            raise DataValidationError("DataCollection ID is required")

        # Check if data_collection exists
        data_collection = await self._load_data_collection(request.data_collection_id)
        if not data_collection:
            raise DataValidationError(f"DataCollection not found: {request.data_collection_id}")

        # Validate algorithms
        if request.algorithms:
            for algorithm in request.algorithms:
                if algorithm not in self.algorithm_adapters:
                    raise TrainingError(f"Algorithm not supported: {algorithm}")

    async def _load_data_collection(self, data_collection_id: str) -> DataCollection:
        """Load data_collection by ID."""
        # Try to load from data_collection repository if available
        try:
            from monorepo.infrastructure.config.container import Container

            # Get data_collection repository from container
            container = Container()
            data_collection_repo = container.get_data_collection_repository()

            # Find data_collection by ID
            data_collection = data_collection_repo.find_by_id(data_collection_id)
            if data_collection:
                return data_collection

        except Exception as e:
            logger.warning(f"Could not load data_collection from repository: {e}")

        # Fallback: generate synthetic data_collection for testing
        logger.info(f"Generating synthetic data_collection for ID: {data_collection_id}")

        # Create different types of synthetic data based on data_collection_id
        if "anomaly" in data_collection_id.lower():
            # Generate anomaly processing data_collection
            n_samples = 1000
            n_features = 10
            contamination = 0.1

            # Normal data
            normal_samples = int(n_samples * (1 - contamination))
            X_normal = np.random.multivariate_normal(
                mean=np.zeros(n_features), cov=np.eye(n_features), size=normal_samples
            )
            y_normal = np.zeros(normal_samples)

            # Anomalous data
            anomaly_samples = n_samples - normal_samples
            X_anomaly = np.random.multivariate_normal(
                mean=np.ones(n_features) * 3,  # Shifted mean
                cov=np.eye(n_features) * 4,  # Larger variance
                size=anomaly_samples,
            )
            y_anomaly = np.ones(anomaly_samples)

            # Combine data
            X = np.vstack([X_normal, X_anomaly])
            y = np.hstack([y_normal, y_anomaly])

            # Shuffle
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

        elif "classification" in data_collection_id.lower():
            # Generate classification data_collection
            from sklearn.datasets import make_classification

            X, y = make_classification(
                n_samples=1000,
                n_features=10,
                n_informative=8,
                n_redundant=2,
                n_classes=2,
                random_state=42,
            )

        else:
            # Generate general data_collection
            n_samples = 1000
            n_features = 10
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 2, n_samples)

        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        data_df = pd.DataFrame(X, columns=feature_names)
        data_df["target"] = y

        # Create DataCollection entity
        data_collection = DataCollection(
            name=f"synthetic_data_collection_{data_collection_id}",
            data=data_df,
            feature_names=feature_names,
            target_column="target",
            metadata={
                "type": "synthetic",
                "n_samples": len(X),
                "n_features": X.shape[1],
                "contamination": np.mean(y)
                if "anomaly" in data_collection_id.lower()
                else None,
            },
        )

        return data_collection

    async def _prepare_training_data(
        self, data_collection: DataCollection, config: TrainingConfigDTO
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data from data_collection."""
        try:
            # Get feature data
            X = data_collection.features.values

            # Get target data
            if data_collection.has_target:
                y = data_collection.target.values
            else:
                # For unsupervised learning, create dummy targets
                y = np.zeros(len(X))
                logger.info(
                    "No target column found, using dummy targets for unsupervised learning"
                )

            # Data preprocessing based on configuration
            if (
                config.preprocessing
                and hasattr(config.preprocessing, "enabled")
                and config.preprocessing.enabled
            ):
                X, y = await self._preprocess_data(X, y, config.preprocessing)
            else:
                # Basic preprocessing
                X, y = await self._basic_preprocessing(X, y)

            logger.info(
                f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features"
            )

            return X, y

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            # Fallback to basic data extraction
            X = data_collection.data.select_dtypes(include=[np.number]).values
            if data_collection.has_target:
                y = data_collection.target.values
            else:
                y = np.zeros(len(X))

            return X, y

    async def _basic_preprocessing(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply basic preprocessing to training data."""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        # Handle missing values
        if np.any(np.isnan(X)):
            logger.info("Handling missing values with median imputation")
            imputer = SimpleImputer(strategy="median")
            X = imputer.fit_transform(X)

        # Handle infinite values
        if np.any(np.isinf(X)):
            logger.info("Handling infinite values")
            X = np.where(np.isinf(X), np.nan, X)
            imputer = SimpleImputer(strategy="median")
            X = imputer.fit_transform(X)

        # Standard scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Store preprocessing metadata for later use
        if not hasattr(self, "_preprocessing_metadata"):
            self._preprocessing_metadata = {}

        self._preprocessing_metadata["scaler"] = scaler
        self._preprocessing_metadata["imputer"] = (
            imputer if "imputer" in locals() else None
        )

        return X_scaled, y

    async def _preprocess_data(
        self, X: np.ndarray, y: np.ndarray, preprocessing_config: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply advanced preprocessing based on configuration."""
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.impute import KNNImputer, SimpleImputer
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        # Initialize preprocessing metadata
        if not hasattr(self, "_preprocessing_metadata"):
            self._preprocessing_metadata = {}

        # Handle missing values
        if np.any(np.isnan(X)):
            imputation_strategy = getattr(
                preprocessing_config, "imputation_strategy", "median"
            )
            if imputation_strategy == "knn":
                logger.info("Using KNN imputation for missing values")
                imputer = KNNImputer(n_neighbors=5)
            else:
                logger.info(
                    f"Using {imputation_strategy} imputation for missing values"
                )
                imputer = SimpleImputer(strategy=imputation_strategy)

            X = imputer.fit_transform(X)
            self._preprocessing_metadata["imputer"] = imputer

        # Handle infinite values
        if np.any(np.isinf(X)):
            logger.info("Handling infinite values")
            X = np.where(np.isinf(X), np.nan, X)
            if "imputer" not in self._preprocessing_metadata:
                imputer = SimpleImputer(strategy="median")
                X = imputer.fit_transform(X)
                self._preprocessing_metadata["imputer"] = imputer

        # Feature scaling
        scaling_method = getattr(preprocessing_config, "scaling_method", "standard")
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()  # Default fallback

        logger.info(f"Applying {scaling_method} scaling")
        X_scaled = scaler.fit_transform(X)
        self._preprocessing_metadata["scaler"] = scaler

        # Feature selection
        if (
            hasattr(preprocessing_config, "feature_selection")
            and preprocessing_config.feature_selection
        ):
            n_features = getattr(
                preprocessing_config, "n_features_to_select", min(20, X_scaled.shape[1])
            )
            if n_features < X_scaled.shape[1]:
                logger.info(f"Selecting top {n_features} features")
                selector = SelectKBest(score_func=f_classif, k=n_features)
                X_selected = selector.fit_transform(X_scaled, y)
                self._preprocessing_metadata["feature_selector"] = selector
                X_scaled = X_selected

        # Outlier handling (basic implementation)
        if (
            hasattr(preprocessing_config, "outlier_handling")
            and preprocessing_config.outlier_handling
        ):
            outlier_method = getattr(preprocessing_config, "outlier_method", "iqr")
            if outlier_method == "iqr":
                logger.info("Applying IQR-based outlier clipping")
                Q1 = np.percentile(X_scaled, 25, axis=0)
                Q3 = np.percentile(X_scaled, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X_scaled = np.clip(X_scaled, lower_bound, upper_bound)

        return X_scaled, y

    def _get_default_algorithms(self) -> list[str]:
        """Get default algorithms for training."""
        return list(self.algorithm_adapters.keys())[:3]  # Use first 3 algorithms

    def _is_classification_task(self, y: np.ndarray) -> bool:
        """Check if this is a classification task."""
        return len(np.unique(y)) <= 10  # Arbitrary threshold

    def _is_anomaly_detection_task(self, y: np.ndarray) -> bool:
        """Check if this is an anomaly processing task."""
        return len(np.unique(y)) == 2 and np.min(y) == 0 and np.max(y) == 1

    async def _serialize_processor(self, processor: Any) -> bytes:
        """Serialize processor for storage."""
        import pickle

        return pickle.dumps(processor)

    async def _update_job_status(
        self,
        job: Any,
        status: str,  # TrainingJob, TrainingStatus
    ) -> None:
        """Update job status."""
        job.status = status
        job.updated_at = datetime.utcnow()
        await self.training_repository.update_job(job)

    async def _update_job_progress(self, job: Any) -> None:  # TrainingJob
        """Update job progress."""
        job.updated_at = datetime.utcnow()
        await self.training_repository.update_job(job)

    # Public API methods
    async def get_training_job(self, job_id: str) -> Any | None:  # TrainingJob
        """Get training job by ID."""
        return await self.training_repository.get_job(job_id)

    async def list_training_jobs(
        self,
        data_collection_id: str | None = None,
        status: str | None = None,  # TrainingStatus
        limit: int = 100,
    ) -> list[Any]:  # list[TrainingJob]
        """List training jobs with optional filtering."""
        return await self.training_repository.list_jobs(
            data_collection_id=data_collection_id, status=status, limit=limit
        )

    async def cancel_training_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        job = await self.training_repository.get_job(job_id)
        if not job:
            return False

        if job.get("status") in ["PENDING", "RUNNING"]:  # TrainingStatus values
            await self._update_job_status(job, "CANCELLED")  # TrainingStatus.CANCELLED

            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

            return True

        return False

    async def get_training_measurements(self) -> dict[str, Any]:
        """Get training service measurements."""
        return {
            "active_jobs": len(self.active_jobs),
            "queued_jobs": len(self.job_queue),
            "training_measurements": self.training_measurements,
            "best_processors": self.best_processors,
        }

    def _get_default_hyperparameter_space(self, algorithm_name: str) -> dict[str, Any]:
        """Get default hyperparameter search space for an algorithm."""
        # Basic hyperparameter spaces for common algorithms
        hyperparameter_spaces = {
            "IsolationForest": {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "max_samples": {"type": "float", "low": 0.1, "high": 1.0},
                "contamination": {"type": "float", "low": 0.05, "high": 0.3},
                "max_features": {"type": "float", "low": 0.5, "high": 1.0},
            },
            "LocalOutlierFactor": {
                "n_neighbors": {"type": "int", "low": 5, "high": 50},
                "algorithm": {
                    "type": "categorical",
                    "choices": ["auto", "ball_tree", "kd_tree", "brute"],
                },
                "leaf_size": {"type": "int", "low": 10, "high": 50},
                "contamination": {"type": "float", "low": 0.05, "high": 0.3},
            },
            "OneClassSVM": {
                "kernel": {
                    "type": "categorical",
                    "choices": ["rbf", "linear", "poly", "sigmoid"],
                },
                "gamma": {"type": "float", "low": 0.001, "high": 1.0, "log": True},
                "nu": {"type": "float", "low": 0.01, "high": 1.0},
                "degree": {"type": "int", "low": 2, "high": 5},
            },
            "LOF": {
                "n_neighbors": {"type": "int", "low": 5, "high": 50},
                "algorithm": {
                    "type": "categorical",
                    "choices": ["auto", "ball_tree", "kd_tree", "brute"],
                },
                "contamination": {"type": "float", "low": 0.05, "high": 0.3},
            },
            "ABOD": {
                "contamination": {"type": "float", "low": 0.05, "high": 0.3},
                "n_neighbors": {"type": "int", "low": 5, "high": 20},
            },
            "CBLOF": {
                "n_clusters": {"type": "int", "low": 3, "high": 20},
                "contamination": {"type": "float", "low": 0.05, "high": 0.3},
                "clustering_estimator": {
                    "type": "categorical",
                    "choices": ["KMeans", "MiniBatchKMeans"],
                },
            },
            "HBOS": {
                "n_bins": {"type": "int", "low": 5, "high": 50},
                "alpha": {"type": "float", "low": 0.01, "high": 1.0},
                "tol": {"type": "float", "low": 0.1, "high": 1.0},
            },
            "KNN": {
                "n_neighbors": {"type": "int", "low": 3, "high": 30},
                "method": {
                    "type": "categorical",
                    "choices": ["largest", "mean", "median"],
                },
                "contamination": {"type": "float", "low": 0.05, "high": 0.3},
            },
            "PCA": {
                "n_components": {"type": "float", "low": 0.5, "high": 0.99},
                "contamination": {"type": "float", "low": 0.05, "high": 0.3},
                "standardization": {"type": "categorical", "choices": [True, False]},
            },
        }

        # Return algorithm-specific space or default
        return hyperparameter_spaces.get(
            algorithm_name,
            {
                "contamination": {"type": "float", "low": 0.05, "high": 0.3},
                "random_state": {"type": "int", "low": 1, "high": 1000},
            },
        )

    def _create_temp_detector(
        self, algorithm_name: str, params: dict[str, Any], suffix: str
    ) -> Any:
        """Create a temporary detector for hyperparameter optimization."""
        from uuid import uuid4

        from monorepo.domain.entities.detector import Detector
        from monorepo.domain.value_objects import ContaminationRate

        return Detector(
            id=uuid4(),
            name=f"{algorithm_name}_{suffix}",
            algorithm_name=algorithm_name,
            parameters=params,
            contamination_rate=ContaminationRate(params.get("contamination", 0.1)),
        )

    def _create_temp_dataset(self, X: np.ndarray, y: np.ndarray, name: str) -> Dataset:
        """Create a temporary data_collection for hyperparameter optimization."""
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        data_df = pd.DataFrame(X, columns=feature_names)

        if y is not None:
            data_df["target"] = y
            target_column = "target"
        else:
            target_column = None

        return DataCollection(
            name=name,
            data=data_df,
            feature_names=feature_names,
            target_column=target_column,
        )

    async def _evaluate_processor_score(
        self, detector: Any, adapter: AlgorithmAdapter, val_data_collection: DataCollection
    ) -> float:
        """Get a single score for processor comparison during optimization."""
        try:
            # Get predictions using the adapter
            predictions = adapter.predict(detector, val_data_collection)
            scores = adapter.score(detector, val_data_collection)

            # Extract actual scores as floats
            score_values = [score.value for score in scores]

            # Get true labels
            if val_data_collection.has_target:
                y_true = val_data_collection.target.values

                # Calculate F1 score if we have binary classification
                if len(np.unique(y_true)) == 2:
                    return f1_score(
                        y_true, predictions, average="weighted", zero_division=0
                    )
                # For anomaly processing, use AUC if possible
                elif len(np.unique(y_true)) > 1:
                    return roc_auc_score(y_true, score_values)
                else:
                    return 0.0
            else:
                # For unsupervised learning, use mean anomaly score as proxy
                return np.mean(score_values)

        except Exception as e:
            logger.warning(f"Error evaluating processor score: {e}")
            return 0.0
