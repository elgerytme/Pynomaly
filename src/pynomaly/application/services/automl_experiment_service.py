"""AutoML Experiment Service with comprehensive database persistence.

This service provides database persistence for AutoML experiments, integrating with the 
domain experiment entity and the MLOps experiment tracker for comprehensive tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pynomaly.application.services.automl_service import (
    AutoMLResult,
    AutoMLService,
    OptimizationObjective,
)
from pynomaly.application.services.experiment_tracking_service import (
    ExperimentTrackingService,
)
from pynomaly.domain.entities.experiment import (
    Experiment,
    ExperimentRun,
    ExperimentStatus,
    ExperimentType,
)
from pynomaly.domain.exceptions import AutoMLError
from pynomaly.mlops.experiment_tracker import experiment_tracker

logger = logging.getLogger(__name__)


class AutoMLExperimentService:
    """Service for managing AutoML experiments with full database persistence."""

    def __init__(
        self,
        automl_service: AutoMLService,
        experiment_tracking_service: ExperimentTrackingService,
        experiment_repository=None,
        storage_path: Path | None = None,
    ):
        """Initialize AutoML experiment service.

        Args:
            automl_service: Base AutoML service
            experiment_tracking_service: Application experiment tracking service
            experiment_repository: Optional repository for experiment entities
            storage_path: Path for additional storage
        """
        self.automl_service = automl_service
        self.experiment_tracking_service = experiment_tracking_service
        self.experiment_repository = experiment_repository
        self.storage_path = storage_path or Path("./automl_experiments")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache for active experiments
        self.active_experiments: dict[UUID, Experiment] = {}

        logger.info("AutoML Experiment Service initialized")

    async def create_experiment(
        self,
        name: str,
        description: str,
        dataset_id: str,
        objective: OptimizationObjective = OptimizationObjective.AUC,
        tags: list[str] | None = None,
        created_by: str = "AutoML Service",
    ) -> UUID:
        """Create a new AutoML experiment with full persistence.

        Args:
            name: Experiment name
            description: Experiment description
            dataset_id: ID of the dataset to use
            objective: Optimization objective
            tags: Optional tags for categorization
            created_by: User creating the experiment

        Returns:
            Experiment ID
        """
        try:
            # Validate dataset exists
            dataset = await self.automl_service.dataset_repository.get(dataset_id)
            if not dataset:
                raise AutoMLError(f"Dataset {dataset_id} not found")

            # Create domain experiment entity
            experiment = Experiment(
                name=name,
                description=description,
                experiment_type=ExperimentType.HYPERPARAMETER_TUNING,
                objective=f"AutoML optimization for {objective.value}",
                created_by=created_by,
                tags=tags or [],
            )

            # Add optimization metric
            experiment.add_optimization_metric(objective.value, "maximize")

            # Add dataset to experiment
            experiment.dataset_ids.add(UUID(dataset_id))

            # Store metadata
            experiment.metadata.update({
                "automl_objective": objective.value,
                "dataset_id": dataset_id,
                "automl_created": True,
                "optimization_metric": objective.value,
            })

            # Save to repository if available
            if self.experiment_repository:
                await self.experiment_repository.save(experiment)

            # Cache experiment
            self.active_experiments[experiment.id] = experiment

            # Create MLOps tracking experiment
            mlops_experiment_id = experiment_tracker.create_experiment(
                name=name,
                description=f"AutoML: {description}",
                tags=(tags or []) + ["automl", objective.value],
            )

            # Create application-level experiment
            app_experiment_id = await self.experiment_tracking_service.create_experiment(
                name=name,
                description=description,
                tags=(tags or []) + ["automl"],
            )

            # Link experiments in metadata
            experiment.metadata.update({
                "mlops_experiment_id": mlops_experiment_id,
                "app_experiment_id": app_experiment_id,
            })

            # Save updated experiment
            if self.experiment_repository:
                await self.experiment_repository.save(experiment)

            # Save to file storage as backup
            await self._save_experiment_to_file(experiment)

            logger.info(
                f"Created AutoML experiment: {name} ({experiment.id}) "
                f"with MLOps ID: {mlops_experiment_id}"
            )

            return experiment.id

        except Exception as e:
            logger.error(f"Failed to create AutoML experiment: {e}")
            raise AutoMLError(f"Failed to create experiment: {e}")

    async def run_automl_optimization(
        self,
        experiment_id: UUID,
        max_algorithms: int = 5,
        enable_ensemble: bool = True,
        background: bool = False,
    ) -> AutoMLResult:
        """Run AutoML optimization for an experiment with full tracking.

        Args:
            experiment_id: ID of the experiment
            max_algorithms: Maximum algorithms to try
            enable_ensemble: Whether to create ensemble
            background: Whether to run in background (not implemented yet)

        Returns:
            AutoML optimization result
        """
        try:
            # Get experiment
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                raise AutoMLError(f"Experiment {experiment_id} not found")

            # Start experiment
            experiment.start_experiment()
            if self.experiment_repository:
                await self.experiment_repository.save(experiment)

            # Get dataset ID from metadata
            dataset_id = experiment.metadata.get("dataset_id")
            if not dataset_id:
                raise AutoMLError("Dataset ID not found in experiment metadata")

            objective_str = experiment.metadata.get("automl_objective", "auc")
            objective = OptimizationObjective(objective_str)

            # Profile dataset
            profile = await self.automl_service.profile_dataset(dataset_id)

            # Update experiment with dataset profile
            experiment.metadata.update({
                "dataset_profile": {
                    "n_samples": profile.n_samples,
                    "n_features": profile.n_features,
                    "contamination_estimate": profile.contamination_estimate,
                    "complexity_score": profile.complexity_score,
                }
            })

            # Recommend algorithms
            recommended_algorithms = self.automl_service.recommend_algorithms(
                profile, max_algorithms
            )

            # Update experiment metadata
            experiment.metadata["recommended_algorithms"] = recommended_algorithms
            experiment.metadata["optimization_started_at"] = datetime.now(UTC).isoformat()

            if self.experiment_repository:
                await self.experiment_repository.save(experiment)

            # Run optimization for each algorithm
            optimization_results = []

            for i, algorithm in enumerate(recommended_algorithms):
                try:
                    logger.info(f"Optimizing algorithm {i+1}/{len(recommended_algorithms)}: {algorithm}")

                    # Run optimization with tracking
                    result = await self._run_algorithm_optimization_with_tracking(
                        experiment, algorithm, objective, dataset_id
                    )
                    optimization_results.append(result)

                except Exception as e:
                    logger.warning(f"Failed to optimize {algorithm}: {e}")

                    # Create failed run
                    failed_run = ExperimentRun(
                        name=f"{algorithm}_optimization",
                        dataset_id=UUID(dataset_id),
                        parameters={"algorithm": algorithm},
                        status="failed",
                        error_message=str(e),
                    )
                    failed_run.fail(str(e))
                    experiment.add_run(failed_run)

            if not optimization_results:
                experiment.fail_experiment("No algorithms could be successfully optimized")
                if self.experiment_repository:
                    await self.experiment_repository.save(experiment)
                raise AutoMLError("No algorithms could be successfully optimized")

            # Find best result
            best_result = max(optimization_results, key=lambda x: x.best_score)

            # Create ensemble if enabled and beneficial
            if enable_ensemble and len(optimization_results) > 1:
                ensemble_config = self._create_ensemble_config(optimization_results)
                best_result.ensemble_config = ensemble_config

                # Create ensemble run
                ensemble_run = ExperimentRun(
                    name="ensemble_creation",
                    dataset_id=UUID(dataset_id),
                    parameters={
                        "ensemble_method": "weighted_voting",
                        "algorithms": [r.best_algorithm for r in optimization_results[:3]],
                    },
                    metadata={"ensemble_config": ensemble_config},
                )
                ensemble_run.start()
                ensemble_run.complete({"ensemble_score": best_result.best_score})
                experiment.add_run(ensemble_run)

            # Complete experiment
            experiment.complete_experiment()
            experiment.metadata.update({
                "optimization_completed_at": datetime.now(UTC).isoformat(),
                "best_algorithm": best_result.best_algorithm,
                "best_score": best_result.best_score,
                "total_algorithms_tried": len(optimization_results),
                "ensemble_enabled": enable_ensemble,
            })

            # Save final experiment state
            if self.experiment_repository:
                await self.experiment_repository.save(experiment)

            await self._save_experiment_to_file(experiment)

            # Log final result to MLOps tracker
            await self._log_final_result_to_mlops(experiment, best_result)

            logger.info(
                f"AutoML optimization completed for experiment {experiment_id}. "
                f"Best algorithm: {best_result.best_algorithm} "
                f"(score: {best_result.best_score:.4f})"
            )

            return best_result

        except Exception as e:
            logger.error(f"AutoML optimization failed: {e}")

            # Update experiment status
            if experiment_id in self.active_experiments:
                experiment = self.active_experiments[experiment_id]
                experiment.fail_experiment(str(e))
                if self.experiment_repository:
                    await self.experiment_repository.save(experiment)

            raise AutoMLError(f"AutoML optimization failed: {e}")

    async def _run_algorithm_optimization_with_tracking(
        self,
        experiment: Experiment,
        algorithm: str,
        objective: OptimizationObjective,
        dataset_id: str,
    ) -> AutoMLResult:
        """Run algorithm optimization with comprehensive tracking."""
        # Create experiment run
        run = ExperimentRun(
            name=f"{algorithm}_optimization",
            dataset_id=UUID(dataset_id),
            parameters={"algorithm": algorithm, "objective": objective.value},
            metadata={"algorithm_family": self._get_algorithm_family(algorithm)},
        )
        run.start()
        experiment.add_run(run)

        # Save experiment state
        if self.experiment_repository:
            await self.experiment_repository.save(experiment)

        try:
            # Get MLOps experiment ID for tracking
            mlops_experiment_id = experiment.metadata.get("mlops_experiment_id")

            # Run optimization with MLOps tracking
            if mlops_experiment_id:
                with experiment_tracker.start_run(
                    experiment_id=mlops_experiment_id,
                    run_name=f"{algorithm}_optimization",
                    parameters={"algorithm": algorithm, "objective": objective.value},
                    tags=["automl", algorithm],
                ):
                    # Log algorithm parameters
                    experiment_tracker.log_parameter("algorithm", algorithm)
                    experiment_tracker.log_parameter("objective", objective.value)

                    # Run actual optimization
                    result = await self.automl_service.optimize_hyperparameters(
                        dataset_id, algorithm, objective
                    )

                    # Log results to MLOps
                    experiment_tracker.log_metrics({
                        "best_score": result.best_score,
                        "optimization_time": result.optimization_time,
                        "trials_completed": result.trials_completed,
                    })

                    # Log hyperparameters
                    for param, value in result.best_params.items():
                        experiment_tracker.log_parameter(f"best_{param}", value)

            else:
                # Run without MLOps tracking
                result = await self.automl_service.optimize_hyperparameters(
                    dataset_id, algorithm, objective
                )

            # Complete experiment run
            run.complete({
                objective.value: result.best_score,
                "optimization_time": result.optimization_time,
                "trials_completed": result.trials_completed,
            })

            # Update run metadata
            run.metadata.update({
                "best_params": result.best_params,
                "algorithm_rankings": result.algorithm_rankings,
            })

            # Log to application tracking service
            app_experiment_id = experiment.metadata.get("app_experiment_id")
            if app_experiment_id:
                await self.experiment_tracking_service.log_run(
                    experiment_id=app_experiment_id,
                    detector_name=algorithm,
                    dataset_name=dataset_id,
                    parameters=result.best_params,
                    metrics={objective.value: result.best_score},
                    artifacts={},
                )

            logger.info(f"Algorithm {algorithm} optimization completed: {result.best_score:.4f}")
            return result

        except Exception as e:
            # Mark run as failed
            run.fail(str(e))
            logger.error(f"Algorithm {algorithm} optimization failed: {e}")
            raise

    def _get_algorithm_family(self, algorithm: str) -> str:
        """Get algorithm family for categorization."""
        family_mapping = {
            "IsolationForest": "Isolation-based",
            "OneClassSVM": "Distance-based",
            "LOF": "Distance-based",
            "ECOD": "Statistical",
            "COPOD": "Statistical",
        }
        return family_mapping.get(algorithm, "Unknown")

    def _create_ensemble_config(self, results: list[AutoMLResult]) -> dict[str, Any]:
        """Create ensemble configuration from optimization results."""
        # Select top 3 algorithms
        top_results = sorted(results, key=lambda x: x.best_score, reverse=True)[:3]

        # Calculate weights based on performance
        scores = [r.best_score for r in top_results]
        total_score = sum(scores)
        weights = [score / total_score for score in scores] if total_score > 0 else [1/3] * 3

        return {
            "method": "weighted_voting",
            "algorithms": [
                {
                    "name": result.best_algorithm,
                    "params": result.best_params,
                    "weight": weight,
                    "score": result.best_score,
                }
                for result, weight in zip(top_results, weights, strict=False)
            ],
            "voting_strategy": "soft",
            "normalize_scores": True,
        }

    async def _log_final_result_to_mlops(
        self, experiment: Experiment, result: AutoMLResult
    ) -> None:
        """Log final AutoML result to MLOps tracker."""
        try:
            mlops_experiment_id = experiment.metadata.get("mlops_experiment_id")
            if not mlops_experiment_id:
                return

            with experiment_tracker.start_run(
                experiment_id=mlops_experiment_id,
                run_name="automl_final_result",
                parameters={
                    "best_algorithm": result.best_algorithm,
                    "ensemble_enabled": result.ensemble_config is not None,
                    "total_algorithms_tried": len(result.algorithm_rankings),
                },
                tags=["automl", "final_result"],
            ):
                # Log final metrics
                experiment_tracker.log_metrics({
                    "final_best_score": result.best_score,
                    "total_optimization_time": result.optimization_time,
                    "total_trials": result.trials_completed,
                })

                # Log best hyperparameters
                for param, value in result.best_params.items():
                    experiment_tracker.log_parameter(f"final_best_{param}", value)

                # Log algorithm rankings
                for i, (algo, score) in enumerate(result.algorithm_rankings):
                    experiment_tracker.log_metric(f"ranking_{i+1}_score", score)
                    experiment_tracker.log_parameter(f"ranking_{i+1}_algorithm", algo)

        except Exception as e:
            logger.warning(f"Failed to log final result to MLOps: {e}")

    async def _get_experiment(self, experiment_id: UUID) -> Experiment | None:
        """Get experiment from cache or repository."""
        # Check cache first
        if experiment_id in self.active_experiments:
            return self.active_experiments[experiment_id]

        # Try repository
        if self.experiment_repository:
            try:
                experiment = await self.experiment_repository.get(experiment_id)
                if experiment:
                    self.active_experiments[experiment_id] = experiment
                    return experiment
            except Exception as e:
                logger.warning(f"Failed to load experiment from repository: {e}")

        # Try file storage
        try:
            experiment = await self._load_experiment_from_file(experiment_id)
            if experiment:
                self.active_experiments[experiment_id] = experiment
                return experiment
        except Exception as e:
            logger.warning(f"Failed to load experiment from file: {e}")

        return None

    async def _save_experiment_to_file(self, experiment: Experiment) -> None:
        """Save experiment to file storage as backup."""
        try:
            experiment_file = self.storage_path / f"{experiment.id}.json"
            experiment_data = {
                "id": str(experiment.id),
                "name": experiment.name,
                "description": experiment.description,
                "experiment_type": experiment.experiment_type.value,
                "objective": experiment.objective,
                "created_by": experiment.created_by,
                "created_at": experiment.created_at.isoformat(),
                "status": experiment.status.value,
                "tags": experiment.tags,
                "dataset_ids": [str(d) for d in experiment.dataset_ids],
                "metadata": experiment.metadata,
                "runs": [
                    {
                        "id": str(run.id),
                        "name": run.name,
                        "dataset_id": str(run.dataset_id) if run.dataset_id else None,
                        "parameters": run.parameters,
                        "metrics": run.metrics,
                        "status": run.status,
                        "started_at": run.started_at.isoformat() if run.started_at else None,
                        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                        "error_message": run.error_message,
                        "metadata": run.metadata,
                    }
                    for run in experiment.runs
                ],
            }

            with open(experiment_file, "w") as f:
                json.dump(experiment_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save experiment to file: {e}")

    async def _load_experiment_from_file(self, experiment_id: UUID) -> Experiment | None:
        """Load experiment from file storage."""
        try:
            experiment_file = self.storage_path / f"{experiment_id}.json"
            if not experiment_file.exists():
                return None

            with open(experiment_file) as f:
                data = json.load(f)

            # Reconstruct experiment
            experiment = Experiment(
                name=data["name"],
                description=data["description"],
                experiment_type=ExperimentType(data["experiment_type"]),
                objective=data["objective"],
                created_by=data["created_by"],
            )
            experiment.id = UUID(data["id"])
            experiment.created_at = datetime.fromisoformat(data["created_at"])
            experiment.status = ExperimentStatus(data["status"])
            experiment.tags = data["tags"]
            experiment.dataset_ids = {UUID(d) for d in data["dataset_ids"]}
            experiment.metadata = data["metadata"]

            # Reconstruct runs
            for run_data in data["runs"]:
                run = ExperimentRun(
                    name=run_data["name"],
                    dataset_id=UUID(run_data["dataset_id"]) if run_data["dataset_id"] else None,
                    parameters=run_data["parameters"],
                    status=run_data["status"],
                    error_message=run_data["error_message"],
                    metadata=run_data["metadata"],
                )
                run.id = UUID(run_data["id"])
                run.metrics = run_data["metrics"]

                if run_data["started_at"]:
                    run.started_at = datetime.fromisoformat(run_data["started_at"])
                if run_data["completed_at"]:
                    run.completed_at = datetime.fromisoformat(run_data["completed_at"])

                experiment.runs.append(run)

            return experiment

        except Exception as e:
            logger.error(f"Failed to load experiment from file: {e}")
            return None

    async def get_experiment(self, experiment_id: UUID) -> Experiment | None:
        """Get experiment by ID."""
        return await self._get_experiment(experiment_id)

    async def list_experiments(
        self, status: ExperimentStatus | None = None, limit: int = 50
    ) -> list[Experiment]:
        """List AutoML experiments."""
        experiments = []

        # Get from cache
        for experiment in self.active_experiments.values():
            if not status or experiment.status == status:
                experiments.append(experiment)

        # Get from repository if available
        if self.experiment_repository:
            try:
                # This would require repository to support filtering
                # For now, just return cached experiments
                pass
            except Exception as e:
                logger.warning(f"Failed to list experiments from repository: {e}")

        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x.created_at, reverse=True)

        return experiments[:limit]

    async def get_experiment_summary(self, experiment_id: UUID) -> dict[str, Any]:
        """Get comprehensive experiment summary."""
        experiment = await self._get_experiment(experiment_id)
        if not experiment:
            raise AutoMLError(f"Experiment {experiment_id} not found")

        summary = experiment.get_info()

        # Add AutoML-specific information
        summary.update({
            "automl_metadata": {
                "dataset_profile": experiment.metadata.get("dataset_profile", {}),
                "recommended_algorithms": experiment.metadata.get("recommended_algorithms", []),
                "best_algorithm": experiment.metadata.get("best_algorithm"),
                "best_score": experiment.metadata.get("best_score"),
                "ensemble_enabled": experiment.metadata.get("ensemble_enabled", False),
            },
            "tracking_info": {
                "mlops_experiment_id": experiment.metadata.get("mlops_experiment_id"),
                "app_experiment_id": experiment.metadata.get("app_experiment_id"),
            },
        })

        return summary

    async def compare_experiments(self, experiment_ids: list[UUID]) -> dict[str, Any]:
        """Compare multiple AutoML experiments."""
        experiments = []
        for exp_id in experiment_ids:
            experiment = await self._get_experiment(exp_id)
            if experiment:
                experiments.append(experiment)

        if not experiments:
            return {"error": "No valid experiments found"}

        comparison = {
            "experiments": [],
            "best_overall": None,
            "algorithm_performance": {},
            "dataset_comparison": {},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        best_score = 0.0
        best_experiment = None

        for exp in experiments:
            exp_data = {
                "id": str(exp.id),
                "name": exp.name,
                "status": exp.status.value,
                "success_rate": exp.success_rate,
                "best_score": exp.metadata.get("best_score"),
                "best_algorithm": exp.metadata.get("best_algorithm"),
                "total_runs": exp.run_count,
                "successful_runs": len(exp.successful_runs),
            }

            comparison["experiments"].append(exp_data)

            # Track best overall
            exp_score = exp.metadata.get("best_score", 0.0)
            if exp_score > best_score:
                best_score = exp_score
                best_experiment = exp

        if best_experiment:
            comparison["best_overall"] = {
                "experiment_id": str(best_experiment.id),
                "experiment_name": best_experiment.name,
                "best_score": best_score,
                "best_algorithm": best_experiment.metadata.get("best_algorithm"),
            }

        return comparison

    async def delete_experiment(self, experiment_id: UUID) -> bool:
        """Delete an AutoML experiment."""
        try:
            experiment = await self._get_experiment(experiment_id)
            if not experiment:
                return False

            # Remove from cache
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]

            # Delete from repository
            if self.experiment_repository:
                await self.experiment_repository.delete(experiment_id)

            # Delete file storage
            experiment_file = self.storage_path / f"{experiment_id}.json"
            if experiment_file.exists():
                experiment_file.unlink()

            # Clean up MLOps tracking (if supported)
            mlops_experiment_id = experiment.metadata.get("mlops_experiment_id")
            if mlops_experiment_id:
                # MLOps cleanup would go here
                logger.info(f"MLOps experiment {mlops_experiment_id} should be cleaned up manually")

            logger.info(f"Deleted AutoML experiment: {experiment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False

    def get_service_status(self) -> dict[str, Any]:
        """Get service status and statistics."""
        return {
            "active_experiments": len(self.active_experiments),
            "storage_path": str(self.storage_path),
            "has_experiment_repository": self.experiment_repository is not None,
            "automl_service_available": True,
            "experiment_tracking_available": True,
            "mlops_tracking_available": True,
        }
