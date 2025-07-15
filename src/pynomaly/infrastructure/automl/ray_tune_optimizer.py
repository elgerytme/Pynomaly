"""Ray Tune integration for distributed hyperparameter optimization.

This module provides distributed hyperparameter optimization using Ray Tune,
enabling scalable optimization across multiple machines and workers.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from pynomaly.application.services.automl_service import (
    AutoMLResult,
    OptimizationObjective,
)
from pynomaly.domain.exceptions import AutoMLError

# Optional Ray imports
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import (
        ASHAScheduler,
        HyperBandScheduler,
        MedianStoppingRule,
        PopulationBasedTraining,
    )
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.bayesopt import BayesOptSearch
    from ray.tune.stopper import TrialPlateauStopper

    RAY_TUNE_AVAILABLE = True
except ImportError:
    RAY_TUNE_AVAILABLE = False

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import hyperopt

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RayTuneConfig:
    """Configuration for Ray Tune optimization."""

    # Ray cluster settings
    ray_address: str | None = None  # None for local, "ray://head-node:port" for cluster
    num_workers: int = 4
    resources_per_trial: dict[str, float] = field(
        default_factory=lambda: {"cpu": 1, "gpu": 0}
    )

    # Optimization settings
    max_concurrent_trials: int = 4
    max_total_trials: int = 100
    max_training_time: int = 3600  # seconds
    trial_timeout: int = 300  # seconds per trial

    # Search algorithm
    search_algorithm: str = "optuna"  # "optuna", "hyperopt", "bayesopt", "random"
    scheduler: str = "asha"  # "asha", "hyperband", "median", "pbt"

    # Early stopping
    enable_early_stopping: bool = True
    patience: int = 10
    min_improvement: float = 0.001

    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 10
    keep_checkpoints: int = 3

    # Results
    local_dir: Path = field(default_factory=lambda: Path("./ray_results"))
    experiment_name: str = "automl_optimization"

    # Advanced settings
    resume: bool = False
    verbose: int = 1
    log_to_file: bool = True
    metric_columns: list[str] = field(
        default_factory=lambda: ["training_iteration", "score", "time_total_s"]
    )


@dataclass
class RayTuneResult:
    """Result from Ray Tune optimization."""

    best_config: dict[str, Any]
    best_score: float
    best_trial: str
    total_trials: int
    total_time: float
    results_df: pd.DataFrame
    analysis: Any  # tune.Analysis object
    experiment_path: Path


class RayTuneOptimizer:
    """Distributed hyperparameter optimizer using Ray Tune."""

    def __init__(self, config: RayTuneConfig | None = None):
        """Initialize Ray Tune optimizer.

        Args:
            config: Ray Tune configuration
        """
        self.config = config or RayTuneConfig()
        self._ray_initialized = False
        self._current_objective_func = None

        if not RAY_TUNE_AVAILABLE:
            raise AutoMLError(
                "Ray Tune is not available. Install with: pip install ray[tune]"
            )

        logger.info("Ray Tune optimizer initialized")

    def _initialize_ray(self) -> None:
        """Initialize Ray cluster."""
        if self._ray_initialized:
            return

        try:
            if self.config.ray_address:
                # Connect to existing cluster
                ray.init(address=self.config.ray_address)
                logger.info(f"Connected to Ray cluster at {self.config.ray_address}")
            else:
                # Start local Ray instance
                ray.init(num_cpus=self.config.num_workers, ignore_reinit_error=True)
                logger.info(f"Started local Ray instance with {self.config.num_workers} workers")

            self._ray_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise AutoMLError(f"Failed to initialize Ray: {e}")

    def _shutdown_ray(self) -> None:
        """Shutdown Ray if we initialized it."""
        if self._ray_initialized:
            try:
                ray.shutdown()
                self._ray_initialized = False
                logger.info("Ray cluster shutdown")
            except Exception as e:
                logger.warning(f"Error shutting down Ray: {e}")

    def _create_search_algorithm(self, param_space: dict[str, Any]):
        """Create search algorithm based on configuration."""
        if self.config.search_algorithm == "optuna" and OPTUNA_AVAILABLE:
            search_alg = OptunaSearch(metric="score", mode="max")
        elif self.config.search_algorithm == "hyperopt" and HYPEROPT_AVAILABLE:
            search_alg = HyperOptSearch(metric="score", mode="max")
        elif self.config.search_algorithm == "bayesopt":
            search_alg = BayesOptSearch(metric="score", mode="max")
        else:
            # Default to random search
            search_alg = None

        # Limit concurrency
        if search_alg:
            search_alg = ConcurrencyLimiter(
                search_alg, max_concurrent=self.config.max_concurrent_trials
            )

        return search_alg

    def _create_scheduler(self):
        """Create scheduler based on configuration."""
        if self.config.scheduler == "asha":
            return ASHAScheduler(
                metric="score",
                mode="max",
                max_t=self.config.max_training_time,
                grace_period=10,
                reduction_factor=2,
            )
        elif self.config.scheduler == "hyperband":
            return HyperBandScheduler(
                metric="score",
                mode="max",
                max_t=self.config.max_training_time,
                reduction_factor=2,
            )
        elif self.config.scheduler == "median":
            return MedianStoppingRule(
                metric="score",
                mode="max",
                grace_period=10,
            )
        elif self.config.scheduler == "pbt":
            return PopulationBasedTraining(
                metric="score",
                mode="max",
                perturbation_interval=20,
                hyperparam_mutations={},
            )
        else:
            return None

    def _create_stopper(self):
        """Create early stopping criterion."""
        if self.config.enable_early_stopping:
            return TrialPlateauStopper(
                metric="score",
                std=self.config.min_improvement,
                num_results=self.config.patience,
                grace_period=5,
                mode="max",
            )
        return None

    def _create_reporter(self):
        """Create progress reporter."""
        return CLIReporter(
            metric_columns=self.config.metric_columns,
            max_progress_rows=20,
            max_report_frequency=30,
        )

    def optimize(
        self,
        objective_function: Callable[[dict], dict],
        param_space: dict[str, Any],
        experiment_name: str | None = None,
    ) -> RayTuneResult:
        """Run distributed hyperparameter optimization.

        Args:
            objective_function: Function to optimize (returns dict with 'score' key)
            param_space: Parameter search space
            experiment_name: Optional experiment name

        Returns:
            Ray Tune optimization result
        """
        try:
            # Initialize Ray
            self._initialize_ray()

            # Set up experiment
            exp_name = experiment_name or f"{self.config.experiment_name}_{int(time.time())}"
            
            # Create search algorithm and scheduler
            search_alg = self._create_search_algorithm(param_space)
            scheduler = self._create_scheduler()
            stopper = self._create_stopper()
            reporter = self._create_reporter()

            # Configure trial resources
            resources = tune.PlacementGroupFactory(
                [self.config.resources_per_trial],
                strategy="PACK",
            )

            # Set up checkpoint configuration
            checkpoint_config = None
            if self.config.enable_checkpointing:
                checkpoint_config = tune.CheckpointConfig(
                    num_to_keep=self.config.keep_checkpoints,
                    checkpoint_score_attribute="score",
                    checkpoint_score_order="max",
                )

            # Set up run configuration
            run_config = tune.RunConfig(
                name=exp_name,
                local_dir=str(self.config.local_dir),
                checkpoint_config=checkpoint_config,
                stop=stopper,
                verbose=self.config.verbose,
                progress_reporter=reporter,
            )

            # Convert parameter space to Ray Tune format
            tune_param_space = self._convert_param_space(param_space)

            logger.info(f"Starting Ray Tune optimization: {exp_name}")
            logger.info(f"Parameter space: {tune_param_space}")
            logger.info(f"Max trials: {self.config.max_total_trials}")
            logger.info(f"Max concurrent: {self.config.max_concurrent_trials}")

            # Run optimization
            start_time = time.time()

            analysis = tune.run(
                objective_function,
                config=tune_param_space,
                num_samples=self.config.max_total_trials,
                search_alg=search_alg,
                scheduler=scheduler,
                resources_per_trial=resources,
                time_budget_s=self.config.max_training_time,
                run_config=run_config,
                resume=self.config.resume,
            )

            total_time = time.time() - start_time

            # Extract results
            best_trial = analysis.get_best_trial("score", "max")
            best_config = best_trial.config
            best_score = best_trial.last_result["score"]

            # Create results DataFrame
            results_df = analysis.results_df

            # Create result object
            result = RayTuneResult(
                best_config=best_config,
                best_score=best_score,
                best_trial=best_trial.trial_id,
                total_trials=len(analysis.trials),
                total_time=total_time,
                results_df=results_df,
                analysis=analysis,
                experiment_path=Path(analysis.get_best_logdir()),
            )

            logger.info(f"Ray Tune optimization completed in {total_time:.2f}s")
            logger.info(f"Best score: {best_score:.4f}")
            logger.info(f"Best config: {best_config}")

            return result

        except Exception as e:
            logger.error(f"Ray Tune optimization failed: {e}")
            raise AutoMLError(f"Distributed optimization failed: {e}")

        finally:
            # Note: We don't shutdown Ray here as it might be used by other processes
            pass

    def _convert_param_space(self, param_space: dict[str, Any]) -> dict[str, Any]:
        """Convert parameter space to Ray Tune format."""
        tune_space = {}

        for param_name, param_config in param_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "float")

                if param_type == "float":
                    low = param_config["low"]
                    high = param_config["high"]
                    tune_space[param_name] = tune.uniform(low, high)

                elif param_type == "int":
                    low = param_config["low"]
                    high = param_config["high"]
                    tune_space[param_name] = tune.randint(low, high + 1)

                elif param_type == "categorical":
                    choices = param_config["choices"]
                    tune_space[param_name] = tune.choice(choices)

                elif param_type == "loguniform":
                    low = param_config["low"]
                    high = param_config["high"]
                    tune_space[param_name] = tune.loguniform(low, high)

                else:
                    # Default to grid search for unknown types
                    if "choices" in param_config:
                        tune_space[param_name] = tune.grid_search(param_config["choices"])
                    else:
                        tune_space[param_name] = param_config

            else:
                # Direct value
                tune_space[param_name] = param_config

        return tune_space

    async def optimize_automl(
        self,
        automl_service,
        dataset_id: str,
        algorithm: str,
        objective: OptimizationObjective = OptimizationObjective.AUC,
        experiment_name: str | None = None,
    ) -> AutoMLResult:
        """Optimize AutoML using distributed Ray Tune.

        Args:
            automl_service: AutoML service instance
            dataset_id: Dataset ID
            algorithm: Algorithm to optimize
            objective: Optimization objective
            experiment_name: Optional experiment name

        Returns:
            AutoML result with distributed optimization
        """
        try:
            logger.info(f"Starting distributed AutoML optimization for {algorithm}")

            # Get algorithm configuration
            if algorithm not in automl_service.algorithm_configs:
                raise AutoMLError(f"Algorithm {algorithm} not supported")

            config = automl_service.algorithm_configs[algorithm]

            # Create objective function for Ray Tune
            def objective_function(tune_config):
                """Objective function for Ray Tune."""
                try:
                    # Create a mock trial object for evaluation
                    class MockTrial:
                        def __init__(self, params):
                            self.params = params

                        def suggest_float(self, name, low, high):
                            return params.get(name, (low + high) / 2)

                        def suggest_int(self, name, low, high):
                            return params.get(name, (low + high) // 2)

                        def suggest_categorical(self, name, choices):
                            return params.get(name, choices[0])

                    # Evaluate using AutoML service
                    trial = MockTrial(tune_config)
                    score = automl_service._evaluate_trial(
                        trial, dataset_id, config, objective
                    )

                    return {"score": score, "done": True}

                except Exception as e:
                    logger.error(f"Trial evaluation failed: {e}")
                    return {"score": 0.0, "done": True}

            # Run distributed optimization
            result = self.optimize(
                objective_function=objective_function,
                param_space=config.param_space,
                experiment_name=experiment_name or f"{algorithm}_optimization",
            )

            # Convert to AutoML result
            automl_result = AutoMLResult(
                best_algorithm=algorithm,
                best_params=result.best_config,
                best_score=result.best_score,
                optimization_time=result.total_time,
                trials_completed=result.total_trials,
                algorithm_rankings=[(algorithm, result.best_score)],
            )

            # Add Ray Tune specific metadata
            automl_result.metadata = {
                "ray_tune_analysis": str(result.experiment_path),
                "distributed_optimization": True,
                "total_workers": self.config.num_workers,
                "search_algorithm": self.config.search_algorithm,
                "scheduler": self.config.scheduler,
            }

            logger.info(f"Distributed AutoML optimization completed for {algorithm}")
            return automl_result

        except Exception as e:
            logger.error(f"Distributed AutoML optimization failed: {e}")
            raise AutoMLError(f"Distributed optimization failed: {e}")

    def get_cluster_status(self) -> dict[str, Any]:
        """Get Ray cluster status information."""
        if not self._ray_initialized:
            return {"status": "not_initialized"}

        try:
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()

            return {
                "status": "active",
                "cluster_resources": cluster_resources,
                "available_resources": available_resources,
                "num_nodes": len(ray.nodes()),
                "ray_version": ray.__version__,
                "dashboard_url": ray.get_dashboard_url(),
            }

        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {"status": "error", "error": str(e)}

    def cleanup(self) -> None:
        """Clean up resources."""
        self._shutdown_ray()

    def __enter__(self):
        """Context manager entry."""
        self._initialize_ray()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class DistributedAutoMLService:
    """AutoML service with distributed Ray Tune optimization."""

    def __init__(
        self,
        automl_service,
        ray_config: RayTuneConfig | None = None,
    ):
        """Initialize distributed AutoML service.

        Args:
            automl_service: Base AutoML service
            ray_config: Ray Tune configuration
        """
        self.automl_service = automl_service
        self.ray_optimizer = RayTuneOptimizer(ray_config)

        logger.info("Distributed AutoML service initialized")

    async def optimize_algorithm_distributed(
        self,
        dataset_id: str,
        algorithm: str,
        objective: OptimizationObjective = OptimizationObjective.AUC,
        experiment_name: str | None = None,
    ) -> AutoMLResult:
        """Optimize single algorithm using distributed computing.

        Args:
            dataset_id: Dataset ID
            algorithm: Algorithm to optimize
            objective: Optimization objective
            experiment_name: Optional experiment name

        Returns:
            AutoML result with distributed optimization
        """
        return await self.ray_optimizer.optimize_automl(
            self.automl_service, dataset_id, algorithm, objective, experiment_name
        )

    async def auto_select_and_optimize_distributed(
        self,
        dataset_id: str,
        objective: OptimizationObjective = OptimizationObjective.AUC,
        max_algorithms: int = 3,
        enable_ensemble: bool = True,
    ) -> AutoMLResult:
        """Run full AutoML with distributed optimization.

        Args:
            dataset_id: Dataset ID
            objective: Optimization objective
            max_algorithms: Maximum algorithms to try
            enable_ensemble: Whether to create ensemble

        Returns:
            Complete AutoML result with distributed optimization
        """
        try:
            start_time = time.time()
            logger.info(f"Starting distributed AutoML for dataset {dataset_id}")

            # Profile dataset
            profile = await self.automl_service.profile_dataset(dataset_id)

            # Recommend algorithms
            recommended_algorithms = self.automl_service.recommend_algorithms(
                profile, max_algorithms
            )

            # Optimize each algorithm in parallel using Ray Tune
            optimization_results = []

            for algorithm in recommended_algorithms:
                try:
                    result = await self.optimize_algorithm_distributed(
                        dataset_id, algorithm, objective, f"{algorithm}_optimization"
                    )
                    optimization_results.append(result)

                except Exception as e:
                    logger.warning(f"Distributed optimization failed for {algorithm}: {e}")
                    continue

            if not optimization_results:
                raise AutoMLError("No algorithms could be successfully optimized")

            # Find best result
            best_result = max(optimization_results, key=lambda x: x.best_score)

            # Create ensemble if enabled
            if enable_ensemble and len(optimization_results) > 1:
                ensemble_config = self._create_ensemble_config(optimization_results)
                best_result.ensemble_config = ensemble_config

            # Update timing
            total_time = time.time() - start_time
            best_result.optimization_time = total_time

            # Add distributed metadata
            best_result.metadata = best_result.metadata or {}
            best_result.metadata.update({
                "distributed_automl": True,
                "algorithms_optimized": len(optimization_results),
                "ray_tune_enabled": True,
                "total_optimization_time": total_time,
            })

            logger.info(
                f"Distributed AutoML completed in {total_time:.2f}s. "
                f"Best algorithm: {best_result.best_algorithm} "
                f"(score: {best_result.best_score:.4f})"
            )

            return best_result

        except Exception as e:
            logger.error(f"Distributed AutoML failed: {e}")
            raise AutoMLError(f"Distributed AutoML failed: {e}")

    def _create_ensemble_config(self, results: list[AutoMLResult]) -> dict[str, Any]:
        """Create ensemble configuration from optimization results."""
        # Select top 3 algorithms
        top_results = sorted(results, key=lambda x: x.best_score, reverse=True)[:3]

        # Calculate weights based on performance
        scores = [r.best_score for r in top_results]
        total_score = sum(scores)
        weights = [score / total_score for score in scores] if total_score > 0 else [1/3] * 3

        return {
            "method": "distributed_weighted_voting",
            "algorithms": [
                {
                    "name": result.best_algorithm,
                    "params": result.best_params,
                    "weight": weight,
                    "score": result.best_score,
                    "distributed_optimization": True,
                }
                for result, weight in zip(top_results, weights)
            ],
            "voting_strategy": "soft",
            "normalize_scores": True,
            "distributed_ensemble": True,
        }

    def get_service_status(self) -> dict[str, Any]:
        """Get distributed service status."""
        base_status = {
            "distributed_automl_available": True,
            "ray_tune_available": RAY_TUNE_AVAILABLE,
            "optuna_available": OPTUNA_AVAILABLE,
            "hyperopt_available": HYPEROPT_AVAILABLE,
        }

        if RAY_TUNE_AVAILABLE:
            cluster_status = self.ray_optimizer.get_cluster_status()
            base_status.update({"cluster_status": cluster_status})

        return base_status

    def cleanup(self) -> None:
        """Clean up distributed resources."""
        self.ray_optimizer.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Utility functions for Ray Tune integration
def check_ray_tune_availability() -> dict[str, bool]:
    """Check availability of Ray Tune and related libraries."""
    return {
        "ray_tune": RAY_TUNE_AVAILABLE,
        "optuna": OPTUNA_AVAILABLE,
        "hyperopt": HYPEROPT_AVAILABLE,
    }


def create_distributed_automl_service(
    automl_service,
    num_workers: int = 4,
    max_concurrent_trials: int = 4,
    ray_address: str | None = None,
) -> DistributedAutoMLService:
    """Create distributed AutoML service with sensible defaults.

    Args:
        automl_service: Base AutoML service
        num_workers: Number of Ray workers
        max_concurrent_trials: Maximum concurrent trials
        ray_address: Ray cluster address (None for local)

    Returns:
        Configured distributed AutoML service
    """
    ray_config = RayTuneConfig(
        ray_address=ray_address,
        num_workers=num_workers,
        max_concurrent_trials=max_concurrent_trials,
        search_algorithm="optuna" if OPTUNA_AVAILABLE else "random",
        scheduler="asha",
        enable_early_stopping=True,
    )

    return DistributedAutoMLService(automl_service, ray_config)