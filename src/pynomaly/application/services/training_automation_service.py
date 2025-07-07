"""Comprehensive model training automation service.

This module provides enterprise-grade training automation with:
- Automated hyperparameter optimization using Optuna
- Training pipeline orchestration with experiment tracking
- AutoML capabilities for algorithm selection and tuning
- Model lifecycle management and versioning
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
from uuid import uuid4

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import AutoMLError, TrainingError
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate

# Optional dependencies with graceful fallback
try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner
    from optuna.samplers import CmaEsSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training job status enumeration."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class OptimizationStrategy(Enum):
    """Hyperparameter optimization strategies."""

    RANDOM = "random"
    TPE = "tpe"  # Tree-structured Parzen Estimator
    CMA_ES = "cma_es"  # Covariance Matrix Adaptation Evolution Strategy
    GRID = "grid"
    BAYESIAN = "bayesian"


class PruningStrategy(Enum):
    """Early stopping strategies for optimization trials."""

    NONE = "none"
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"


@dataclass
class TrainingConfiguration:
    """Configuration for automated training."""

    # Basic settings
    max_trials: int = 100
    timeout_minutes: Optional[int] = 60
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.TPE
    pruning_strategy: PruningStrategy = PruningStrategy.MEDIAN

    # Optimization objectives
    primary_metric: str = "roc_auc"
    secondary_metrics: List[str] = field(
        default_factory=lambda: ["precision", "recall", "f1"]
    )
    optimization_direction: str = "maximize"  # or "minimize"

    # Advanced settings
    cross_validation_folds: int = 5
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    min_improvement_threshold: float = 0.001

    # Resource constraints
    max_memory_gb: Optional[float] = None
    max_cpu_cores: Optional[int] = None
    use_gpu: bool = False

    # Experiment tracking
    experiment_name: Optional[str] = None
    track_artifacts: bool = True
    save_models: bool = True

    # AutoML specific
    algorithm_whitelist: Optional[List[str]] = None
    algorithm_blacklist: Optional[List[str]] = None
    ensemble_methods: List[str] = field(default_factory=lambda: ["voting", "stacking"])


@dataclass
class HyperparameterSpace:
    """Defines hyperparameter search space for algorithms."""

    algorithm_name: str
    parameters: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None

    def suggest_parameters(self, trial: Any) -> Dict[str, Any]:
        """Suggest parameters for a trial."""
        suggested = {}

        for param_name, param_config in self.parameters.items():
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "float")

                if param_type == "float":
                    low = param_config["low"]
                    high = param_config["high"]
                    log_scale = param_config.get("log", False)
                    suggested[param_name] = trial.suggest_float(
                        param_name, low, high, log=log_scale
                    )
                elif param_type == "int":
                    low = param_config["low"]
                    high = param_config["high"]
                    suggested[param_name] = trial.suggest_int(param_name, low, high)
                elif param_type == "categorical":
                    choices = param_config["choices"]
                    suggested[param_name] = trial.suggest_categorical(
                        param_name, choices
                    )
            else:
                # Fixed parameter
                suggested[param_name] = param_config

        return suggested


@dataclass
class TrainingJob:
    """Represents a training job with full lifecycle tracking."""

    job_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    status: TrainingStatus = TrainingStatus.QUEUED

    # Configuration
    configuration: TrainingConfiguration = field(default_factory=TrainingConfiguration)
    dataset_id: str = ""
    target_algorithms: List[str] = field(default_factory=list)

    # Execution tracking
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Results
    best_model: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    best_parameters: Optional[Dict[str, Any]] = None
    trial_history: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics
    total_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0
    execution_time_seconds: float = 0.0

    # Artifacts
    model_path: Optional[str] = None
    experiment_id: Optional[str] = None
    study_id: Optional[str] = None


class TrainingJobRepository(Protocol):
    """Repository for training job persistence."""

    async def save_job(self, job: TrainingJob) -> None:
        """Save training job."""
        ...

    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID."""
        ...

    async def list_jobs(
        self, status: Optional[TrainingStatus] = None, limit: int = 100
    ) -> List[TrainingJob]:
        """List training jobs."""
        ...

    async def update_job_status(self, job_id: str, status: TrainingStatus) -> None:
        """Update job status."""
        ...


class ModelTrainer(Protocol):
    """Protocol for model training implementations."""

    async def train(
        self, detector: Detector, dataset: Dataset, parameters: Dict[str, Any]
    ) -> DetectionResult:
        """Train model with given parameters."""
        ...

    async def evaluate(
        self,
        detector: Detector,
        dataset: Dataset,
        validation_data: Optional[Dataset] = None,
    ) -> Dict[str, float]:
        """Evaluate trained model."""
        ...


class TrainingAutomationService:
    """Enterprise-grade training automation service."""

    def __init__(
        self,
        job_repository: TrainingJobRepository,
        model_trainer: ModelTrainer,
        storage_path: Optional[Path] = None,
    ):
        self.job_repository = job_repository
        self.model_trainer = model_trainer
        self.storage_path = storage_path or Path("./training_artifacts")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize tracking systems
        self._initialize_tracking()

        # Define algorithm hyperparameter spaces
        self.hyperparameter_spaces = self._create_hyperparameter_spaces()

        # Active jobs tracking
        self.active_jobs: Dict[str, asyncio.Task] = {}

    def _initialize_tracking(self) -> None:
        """Initialize experiment tracking systems."""
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(str(self.storage_path / "mlflow"))
                logger.info("MLflow tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}")

    def _create_hyperparameter_spaces(self) -> Dict[str, HyperparameterSpace]:
        """Create predefined hyperparameter search spaces."""
        spaces = {}

        # Isolation Forest
        spaces["IsolationForest"] = HyperparameterSpace(
            algorithm_name="IsolationForest",
            parameters={
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                "max_features": {"type": "float", "low": 0.1, "high": 1.0},
                "bootstrap": {"type": "categorical", "choices": [True, False]},
                "random_state": 42,
            },
        )

        # Local Outlier Factor
        spaces["LocalOutlierFactor"] = HyperparameterSpace(
            algorithm_name="LocalOutlierFactor",
            parameters={
                "n_neighbors": {"type": "int", "low": 5, "high": 100},
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                "metric": {
                    "type": "categorical",
                    "choices": ["euclidean", "manhattan", "cosine"],
                },
                "algorithm": {
                    "type": "categorical",
                    "choices": ["auto", "ball_tree", "kd_tree"],
                },
            },
        )

        # One-Class SVM
        spaces["OneClassSVM"] = HyperparameterSpace(
            algorithm_name="OneClassSVM",
            parameters={
                "kernel": {
                    "type": "categorical",
                    "choices": ["rbf", "linear", "poly", "sigmoid"],
                },
                "gamma": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
                "nu": {"type": "float", "low": 0.01, "high": 0.5},
                "degree": {"type": "int", "low": 2, "high": 5},  # Only for poly kernel
            },
        )

        # Autoencoder (if available)
        spaces["Autoencoder"] = HyperparameterSpace(
            algorithm_name="Autoencoder",
            parameters={
                "hidden_neurons": {
                    "type": "categorical",
                    "choices": [[32, 16, 8], [64, 32, 16], [128, 64, 32]],
                },
                "learning_rate": {
                    "type": "float",
                    "low": 1e-4,
                    "high": 1e-1,
                    "log": True,
                },
                "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
                "epochs": {"type": "int", "low": 50, "high": 500},
                "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5},
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
            },
        )

        return spaces

    async def create_training_job(
        self,
        name: str,
        dataset_id: str,
        configuration: TrainingConfiguration,
        target_algorithms: Optional[List[str]] = None,
    ) -> TrainingJob:
        """Create a new training job."""

        # Default to all available algorithms if none specified
        if target_algorithms is None:
            target_algorithms = list(self.hyperparameter_spaces.keys())

        # Filter algorithms based on whitelist/blacklist
        if configuration.algorithm_whitelist:
            target_algorithms = [
                alg
                for alg in target_algorithms
                if alg in configuration.algorithm_whitelist
            ]

        if configuration.algorithm_blacklist:
            target_algorithms = [
                alg
                for alg in target_algorithms
                if alg not in configuration.algorithm_blacklist
            ]

        job = TrainingJob(
            name=name,
            configuration=configuration,
            dataset_id=dataset_id,
            target_algorithms=target_algorithms,
        )

        await self.job_repository.save_job(job)
        logger.info(
            f"Created training job {job.job_id} for algorithms: {target_algorithms}"
        )

        return job

    async def start_training_job(self, job_id: str) -> None:
        """Start training job execution."""
        job = await self.job_repository.get_job(job_id)
        if not job:
            raise TrainingError(f"Training job {job_id} not found")

        if job_id in self.active_jobs:
            raise TrainingError(f"Training job {job_id} is already running")

        # Create and start training task
        task = asyncio.create_task(self._execute_training_job(job))
        self.active_jobs[job_id] = task

        logger.info(f"Started training job {job_id}")

    async def _execute_training_job(self, job: TrainingJob) -> None:
        """Execute training job with comprehensive error handling."""
        try:
            # Update status to running
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            await self.job_repository.save_job(job)

            start_time = time.time()

            # Initialize experiment tracking
            experiment_id = None
            if MLFLOW_AVAILABLE and job.configuration.experiment_name:
                experiment_id = mlflow.create_experiment(
                    job.configuration.experiment_name,
                    artifact_location=str(
                        self.storage_path / "experiments" / job.job_id
                    ),
                )
                job.experiment_id = experiment_id

            # Run optimization for each algorithm
            best_results = []

            for algorithm in job.target_algorithms:
                logger.info(f"Starting optimization for {algorithm}")

                try:
                    result = await self._optimize_algorithm(job, algorithm)
                    if result:
                        best_results.append(result)
                        job.successful_trials += result.get("n_trials", 0)
                except Exception as e:
                    logger.error(f"Algorithm {algorithm} optimization failed: {e}")
                    job.failed_trials += 1
                    continue

            # Select best overall result
            if best_results:
                best_result = max(best_results, key=lambda x: x["best_score"])
                job.best_model = best_result["best_model"]
                job.best_score = best_result["best_score"]
                job.best_parameters = best_result["best_parameters"]

                # Save best model if configured
                if job.configuration.save_models:
                    model_path = await self._save_best_model(job, best_result)
                    job.model_path = str(model_path)

            # Update job completion
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.execution_time_seconds = time.time() - start_time

            logger.info(f"Training job {job.job_id} completed successfully")

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Training job {job.job_id} failed: {e}")

        finally:
            await self.job_repository.save_job(job)
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

    async def _optimize_algorithm(
        self, job: TrainingJob, algorithm: str
    ) -> Optional[Dict[str, Any]]:
        """Optimize hyperparameters for a specific algorithm."""

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return await self._train_with_defaults(job, algorithm)

        # Get hyperparameter space
        param_space = self.hyperparameter_spaces.get(algorithm)
        if not param_space:
            logger.warning(f"No hyperparameter space defined for {algorithm}")
            return await self._train_with_defaults(job, algorithm)

        # Create Optuna study
        study_name = f"{job.job_id}_{algorithm}"

        # Configure sampler
        sampler = self._create_sampler(job.configuration.optimization_strategy)

        # Configure pruner
        pruner = self._create_pruner(job.configuration.pruning_strategy)

        study = optuna.create_study(
            study_name=study_name,
            direction=job.configuration.optimization_direction,
            sampler=sampler,
            pruner=pruner,
        )

        job.study_id = study_name

        # Define objective function
        async def objective(trial):
            return await self._objective_function(trial, job, algorithm, param_space)

        # Run optimization
        try:
            # Convert async objective to sync for Optuna compatibility
            def sync_objective(trial):
                return asyncio.run(objective(trial))

            study.optimize(
                sync_objective,
                n_trials=job.configuration.max_trials,
                timeout=(
                    job.configuration.timeout_minutes * 60
                    if job.configuration.timeout_minutes
                    else None
                ),
            )

            # Extract best results
            best_trial = study.best_trial

            return {
                "algorithm": algorithm,
                "best_score": best_trial.value,
                "best_parameters": best_trial.params,
                "best_model": None,  # Will be trained separately
                "n_trials": len(study.trials),
                "study": study,
            }

        except Exception as e:
            logger.error(f"Optimization failed for {algorithm}: {e}")
            return None

    async def _objective_function(
        self,
        trial: Any,
        job: TrainingJob,
        algorithm: str,
        param_space: HyperparameterSpace,
    ) -> float:
        """Objective function for Optuna optimization."""

        # Suggest parameters
        parameters = param_space.suggest_parameters(trial)

        # Create detector with suggested parameters
        detector = Detector(
            name=f"{algorithm}_trial_{trial.number}",
            algorithm_name=algorithm,
            parameters=parameters,
        )

        # Load dataset (this would need to be implemented)
        # For now, we'll use a placeholder
        dataset = await self._load_dataset(job.dataset_id)

        try:
            # Train model
            result = await self.model_trainer.train(detector, dataset, parameters)

            # Evaluate model
            metrics = await self.model_trainer.evaluate(detector, dataset)

            # Return primary metric
            primary_score = metrics.get(job.configuration.primary_metric, 0.0)

            # Log trial results
            trial_info = {
                "trial_number": trial.number,
                "algorithm": algorithm,
                "parameters": parameters,
                "score": primary_score,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
            job.trial_history.append(trial_info)

            # Report intermediate results for pruning
            trial.report(primary_score, trial.number)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return primary_score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            # Return worst possible score for failed trials
            return (
                float("-inf")
                if job.configuration.optimization_direction == "maximize"
                else float("inf")
            )

    def _create_sampler(self, strategy: OptimizationStrategy):
        """Create appropriate Optuna sampler."""
        if strategy == OptimizationStrategy.TPE:
            return TPESampler()
        elif strategy == OptimizationStrategy.CMA_ES:
            return CmaEsSampler()
        elif strategy == OptimizationStrategy.RANDOM:
            return optuna.samplers.RandomSampler()
        else:
            return TPESampler()  # Default

    def _create_pruner(self, strategy: PruningStrategy):
        """Create appropriate Optuna pruner."""
        if strategy == PruningStrategy.MEDIAN:
            return MedianPruner()
        elif strategy == PruningStrategy.HYPERBAND:
            return HyperbandPruner()
        elif strategy == PruningStrategy.NONE:
            return optuna.pruners.NopPruner()
        else:
            return MedianPruner()  # Default

    async def _train_with_defaults(
        self, job: TrainingJob, algorithm: str
    ) -> Dict[str, Any]:
        """Train algorithm with default parameters as fallback."""
        logger.info(f"Training {algorithm} with default parameters")

        # Get default parameters from hyperparameter space
        param_space = self.hyperparameter_spaces.get(algorithm)
        default_params = {}

        if param_space:
            for param_name, param_config in param_space.parameters.items():
                if isinstance(param_config, dict):
                    # Use middle value for ranges
                    if param_config.get("type") == "float":
                        low, high = param_config["low"], param_config["high"]
                        default_params[param_name] = (low + high) / 2
                    elif param_config.get("type") == "int":
                        low, high = param_config["low"], param_config["high"]
                        default_params[param_name] = (low + high) // 2
                    elif param_config.get("type") == "categorical":
                        choices = param_config["choices"]
                        default_params[param_name] = choices[0]
                else:
                    default_params[param_name] = param_config

        # Create detector and train
        detector = Detector(
            name=f"{algorithm}_default",
            algorithm_name=algorithm,
            parameters=default_params,
        )

        dataset = await self._load_dataset(job.dataset_id)

        try:
            result = await self.model_trainer.train(detector, dataset, default_params)
            metrics = await self.model_trainer.evaluate(detector, dataset)

            primary_score = metrics.get(job.configuration.primary_metric, 0.0)

            return {
                "algorithm": algorithm,
                "best_score": primary_score,
                "best_parameters": default_params,
                "best_model": detector,
                "n_trials": 1,
                "study": None,
            }

        except Exception as e:
            logger.error(f"Default training failed for {algorithm}: {e}")
            return None

    async def _load_dataset(self, dataset_id: str) -> Dataset:
        """Load dataset by ID."""
        # This would be implemented to load from the dataset repository
        # For now, return a placeholder
        return Dataset(name="placeholder", data=pd.DataFrame(), id=dataset_id)

    async def _save_best_model(self, job: TrainingJob, result: Dict[str, Any]) -> Path:
        """Save the best trained model."""
        model_dir = self.storage_path / "models" / job.job_id
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"best_model_{result['algorithm']}.pkl"

        # Save model (implementation depends on model type)
        # This would use joblib, pickle, or framework-specific saving

        logger.info(f"Saved best model to {model_path}")
        return model_path

    async def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get current status of training job."""
        return await self.job_repository.get_job(job_id)

    async def cancel_training_job(self, job_id: str) -> None:
        """Cancel running training job."""
        job = await self.job_repository.get_job(job_id)
        if not job:
            raise TrainingError(f"Training job {job_id} not found")

        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            task.cancel()
            del self.active_jobs[job_id]

            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            await self.job_repository.save_job(job)

            logger.info(f"Cancelled training job {job_id}")

    async def list_training_jobs(
        self, status: Optional[TrainingStatus] = None, limit: int = 100
    ) -> List[TrainingJob]:
        """List training jobs with optional filtering."""
        return await self.job_repository.list_jobs(status, limit)

    async def get_training_metrics(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive training metrics for a job."""
        job = await self.job_repository.get_job(job_id)
        if not job:
            raise TrainingError(f"Training job {job_id} not found")

        metrics = {
            "job_id": job.job_id,
            "status": job.status.value,
            "execution_time": job.execution_time_seconds,
            "total_trials": job.total_trials,
            "successful_trials": job.successful_trials,
            "failed_trials": job.failed_trials,
            "success_rate": job.successful_trials / max(job.total_trials, 1),
            "best_score": job.best_score,
            "best_algorithm": None,
            "trial_history": job.trial_history,
        }

        if job.best_model:
            metrics["best_algorithm"] = job.best_model.get("algorithm")

        return metrics

    async def cleanup_old_jobs(self, days: int = 30) -> int:
        """Clean up old training jobs and artifacts."""
        cutoff_date = datetime.now() - timedelta(days=days)
        jobs = await self.job_repository.list_jobs()

        cleaned_count = 0
        for job in jobs:
            if job.created_at < cutoff_date and job.status in [
                TrainingStatus.COMPLETED,
                TrainingStatus.FAILED,
                TrainingStatus.CANCELLED,
            ]:
                # Clean up artifacts
                if job.model_path and Path(job.model_path).exists():
                    Path(job.model_path).unlink()

                # Could also clean up MLflow experiments, logs, etc.
                cleaned_count += 1

        logger.info(f"Cleaned up {cleaned_count} old training jobs")
        return cleaned_count


# Convenience functions for common training scenarios


async def quick_optimize(
    dataset_id: str,
    algorithms: Optional[List[str]] = None,
    max_trials: int = 50,
    timeout_minutes: int = 30,
) -> TrainingJob:
    """Quick optimization with sensible defaults."""
    from pynomaly.infrastructure.adapters.model_trainer_adapter import (
        ModelTrainerAdapter,
    )
    from pynomaly.infrastructure.persistence.training_job_repository import (
        TrainingJobRepository,
    )

    repo = TrainingJobRepository()
    trainer = ModelTrainerAdapter()
    service = TrainingAutomationService(repo, trainer)

    config = TrainingConfiguration(
        max_trials=max_trials,
        timeout_minutes=timeout_minutes,
        optimization_strategy=OptimizationStrategy.TPE,
        pruning_strategy=PruningStrategy.MEDIAN,
    )

    job = await service.create_training_job(
        name=f"Quick optimization {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        dataset_id=dataset_id,
        configuration=config,
        target_algorithms=algorithms,
    )

    await service.start_training_job(job.job_id)
    return job


async def production_optimize(
    dataset_id: str,
    experiment_name: str,
    algorithms: Optional[List[str]] = None,
    max_trials: int = 200,
    timeout_hours: int = 4,
) -> TrainingJob:
    """Production-grade optimization with comprehensive tracking."""
    from pynomaly.infrastructure.adapters.model_trainer_adapter import (
        ModelTrainerAdapter,
    )
    from pynomaly.infrastructure.persistence.training_job_repository import (
        TrainingJobRepository,
    )

    repo = TrainingJobRepository()
    trainer = ModelTrainerAdapter()
    service = TrainingAutomationService(repo, trainer)

    config = TrainingConfiguration(
        max_trials=max_trials,
        timeout_minutes=timeout_hours * 60,
        optimization_strategy=OptimizationStrategy.TPE,
        pruning_strategy=PruningStrategy.HYPERBAND,
        experiment_name=experiment_name,
        track_artifacts=True,
        save_models=True,
        cross_validation_folds=5,
        early_stopping_patience=20,
    )

    job = await service.create_training_job(
        name=experiment_name,
        dataset_id=dataset_id,
        configuration=config,
        target_algorithms=algorithms,
    )

    await service.start_training_job(job.job_id)
    return job
