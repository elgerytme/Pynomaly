"""AutoML Configuration Integration Service.

This module extends AutoML services to automatically capture and save
optimization configurations and results for future reference and learning.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from pynomaly.application.dto.configuration_dto import (
    ConfigurationCaptureRequestDTO,
    ConfigurationSource,
    PerformanceResultsDTO,
)
from pynomaly.application.services.advanced_automl_service import AdvancedAutoMLService
from pynomaly.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.infrastructure.config.feature_flags import require_feature

logger = logging.getLogger(__name__)


class AutoMLConfigurationIntegration:
    """Integration service for AutoML configuration capture and management."""

    def __init__(
        self,
        automl_service: AdvancedAutoMLService,
        configuration_service: ConfigurationCaptureService,
        auto_save_successful: bool = True,
        auto_save_failed: bool = False,
        save_optimization_history: bool = True,
    ):
        """Initialize AutoML configuration integration.

        Args:
            automl_service: Advanced AutoML service instance
            configuration_service: Configuration capture service
            auto_save_successful: Automatically save successful optimization configurations
            auto_save_failed: Automatically save failed optimization configurations
            save_optimization_history: Save complete optimization history
        """
        self.automl_service = automl_service
        self.configuration_service = configuration_service
        self.auto_save_successful = auto_save_successful
        self.auto_save_failed = auto_save_failed
        self.save_optimization_history = save_optimization_history

        # Integration statistics
        self.integration_stats = {
            "total_optimizations": 0,
            "successful_captures": 0,
            "failed_captures": 0,
            "configurations_saved": 0,
            "optimization_histories_saved": 0,
        }

    @require_feature("advanced_automl")
    async def optimize_with_configuration_capture(
        self,
        dataset: Dataset,
        algorithm_name: str,
        objectives: list[Any] | None = None,
        constraints: Any | None = None,
        enable_learning: bool = True,
        capture_config: dict[str, Any] | None = None,
    ) -> tuple[Detector, dict[str, Any], str | None]:
        """Run AutoML optimization with automatic configuration capture.

        Args:
            dataset: Dataset for optimization
            algorithm_name: Algorithm to optimize
            objectives: Optimization objectives
            constraints: Resource constraints
            enable_learning: Enable learning from optimization history
            capture_config: Additional configuration capture settings

        Returns:
            Tuple of (optimized_detector, optimization_report, configuration_id)
        """
        self.integration_stats["total_optimizations"] += 1

        # Extract optimization parameters before running
        optimization_params = self._extract_optimization_parameters(
            dataset, algorithm_name, objectives, constraints, enable_learning
        )

        start_time = datetime.now()
        configuration_id = None

        try:
            logger.info(
                f"Starting AutoML optimization with configuration capture for {algorithm_name}"
            )

            # Run AutoML optimization
            (
                optimized_detector,
                optimization_report,
            ) = await self.automl_service.optimize(
                dataset=dataset,
                algorithm_name=algorithm_name,
                objectives=objectives,
                constraints=constraints,
                enable_learning=enable_learning,
            )

            end_time = datetime.now()
            optimization_duration = (end_time - start_time).total_seconds()

            # Extract performance results from optimization report
            performance_results = self._extract_performance_results(
                optimization_report, optimization_duration
            )

            # Capture configuration if successful optimization and auto-save enabled
            if self.auto_save_successful:
                configuration_id = await self._capture_optimization_configuration(
                    optimization_params=optimization_params,
                    performance_results=performance_results,
                    optimization_report=optimization_report,
                    optimized_detector=optimized_detector,
                    success=True,
                    capture_config=capture_config,
                )

            # Save optimization history if enabled
            if self.save_optimization_history:
                await self._save_optimization_history(
                    optimization_params,
                    performance_results,
                    optimization_report,
                    configuration_id,
                )

            logger.info(
                f"AutoML optimization completed successfully. Configuration ID: {configuration_id}"
            )
            return optimized_detector, optimization_report, configuration_id

        except Exception as e:
            logger.error(f"AutoML optimization failed: {e}")

            # Capture configuration if failed optimization and auto-save enabled
            if self.auto_save_failed:
                try:
                    end_time = datetime.now()
                    optimization_duration = (end_time - start_time).total_seconds()

                    # Create minimal performance results for failed optimization
                    failed_performance = PerformanceResultsDTO(
                        training_time_seconds=optimization_duration,
                        accuracy=0.0,
                        precision=0.0,
                        recall=0.0,
                        f1_score=0.0,
                    )

                    configuration_id = await self._capture_optimization_configuration(
                        optimization_params=optimization_params,
                        performance_results=failed_performance,
                        optimization_report={"error": str(e), "success": False},
                        optimized_detector=None,
                        success=False,
                        capture_config=capture_config,
                    )

                except Exception as capture_error:
                    logger.error(
                        f"Failed to capture configuration for failed optimization: {capture_error}"
                    )

            raise e

    async def capture_manual_configuration(
        self,
        dataset: Dataset,
        detector: Detector,
        performance_results: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Capture configuration for manually created detector.

        Args:
            dataset: Dataset used for training
            detector: Trained detector
            performance_results: Manual performance evaluation results
            metadata: Additional metadata

        Returns:
            Configuration ID
        """
        try:
            # Extract detector parameters
            optimization_params = {
                "dataset_path": getattr(dataset, "path", None)
                or f"dataset_{dataset.name}",
                "dataset_name": dataset.name,
                "algorithm": detector.algorithm_name,
                "contamination": getattr(detector, "contamination", 0.1),
                "random_state": getattr(detector, "random_state", None),
                "hyperparameters": getattr(detector, "params", {}),
                "manual_creation": True,
            }

            # Convert performance results if provided
            perf_results = None
            if performance_results:
                perf_results = PerformanceResultsDTO(
                    accuracy=performance_results.get("accuracy"),
                    precision=performance_results.get("precision"),
                    recall=performance_results.get("recall"),
                    f1_score=performance_results.get("f1_score"),
                    roc_auc=performance_results.get("roc_auc"),
                    training_time_seconds=performance_results.get("training_time"),
                    memory_usage_mb=performance_results.get("memory_usage"),
                )

            # Capture configuration
            configuration_id = await self._capture_optimization_configuration(
                optimization_params=optimization_params,
                performance_results=perf_results,
                optimization_report={"manual_creation": True},
                optimized_detector=detector,
                success=True,
                capture_config=metadata or {},
            )

            logger.info(
                f"Manual configuration captured. Configuration ID: {configuration_id}"
            )
            return configuration_id

        except Exception as e:
            logger.error(f"Failed to capture manual configuration: {e}")
            raise

    async def batch_capture_configurations(
        self,
        optimization_results: list[dict[str, Any]],
        batch_metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Capture configurations for batch optimization results.

        Args:
            optimization_results: List of optimization result dictionaries
            batch_metadata: Metadata for the entire batch

        Returns:
            List of configuration IDs
        """
        configuration_ids = []

        try:
            for i, result in enumerate(optimization_results):
                try:
                    # Extract required data from result
                    dataset = result["dataset"]
                    detector = result["detector"]
                    performance = result.get("performance", {})

                    # Add batch information to metadata
                    metadata = batch_metadata.copy() if batch_metadata else {}
                    metadata.update(
                        {
                            "batch_index": i,
                            "batch_size": len(optimization_results),
                            "batch_id": str(uuid4()),
                        }
                    )

                    # Capture configuration
                    config_id = await self.capture_manual_configuration(
                        dataset=dataset,
                        detector=detector,
                        performance_results=performance,
                        metadata=metadata,
                    )

                    configuration_ids.append(config_id)

                except Exception as e:
                    logger.warning(
                        f"Failed to capture configuration for batch item {i}: {e}"
                    )
                    continue

            logger.info(
                f"Batch configuration capture completed. Captured {len(configuration_ids)} configurations"
            )
            return configuration_ids

        except Exception as e:
            logger.error(f"Batch configuration capture failed: {e}")
            raise

    def get_integration_statistics(self) -> dict[str, Any]:
        """Get integration statistics.

        Returns:
            Integration statistics dictionary
        """
        config_stats = asyncio.run(
            self.configuration_service.get_configuration_statistics()
        )

        return {
            "integration_stats": self.integration_stats,
            "configuration_service_stats": config_stats,
            "automl_service_info": {
                "available_algorithms": len(
                    getattr(self.automl_service, "algorithm_configs", {})
                ),
                "optimization_history_size": len(
                    getattr(self.automl_service, "optimization_history", [])
                ),
                "enable_distributed": getattr(
                    self.automl_service, "enable_distributed", False
                ),
                "n_parallel_jobs": getattr(self.automl_service, "n_parallel_jobs", 1),
            },
        }

    # Private methods

    def _extract_optimization_parameters(
        self,
        dataset: Dataset,
        algorithm_name: str,
        objectives: list[Any] | None,
        constraints: Any | None,
        enable_learning: bool,
    ) -> dict[str, Any]:
        """Extract optimization parameters for configuration capture."""
        params = {
            "dataset_path": getattr(dataset, "path", None),
            "dataset_name": dataset.name,
            "dataset_shape": dataset.data.shape
            if hasattr(dataset.data, "shape")
            else None,
            "algorithm": algorithm_name,
            "objectives": [
                obj.name if hasattr(obj, "name") else str(obj)
                for obj in (objectives or [])
            ],
            "enable_learning": enable_learning,
            "optimization_mode": "advanced_automl",
        }

        # Extract constraints if provided
        if constraints:
            params.update(
                {
                    "max_trials": getattr(constraints, "max_trials", None),
                    "max_time_seconds": getattr(constraints, "max_time_seconds", None),
                    "max_memory_mb": getattr(constraints, "max_memory_mb", None),
                    "max_cpu_cores": getattr(constraints, "max_cpu_cores", None),
                }
            )

        return params

    def _extract_performance_results(
        self, optimization_report: dict[str, Any], optimization_duration: float
    ) -> PerformanceResultsDTO:
        """Extract performance results from optimization report."""
        # Extract best metrics from the report
        best_metrics = optimization_report.get("best_metrics", {})
        resource_usage = optimization_report.get("resource_usage", {})

        return PerformanceResultsDTO(
            accuracy=best_metrics.get("accuracy"),
            precision=best_metrics.get("precision"),
            recall=best_metrics.get("recall"),
            f1_score=best_metrics.get("f1_score"),
            roc_auc=best_metrics.get("roc_auc"),
            training_time_seconds=optimization_duration,
            memory_usage_mb=resource_usage.get("peak_memory_mb"),
            cpu_usage_percent=resource_usage.get("cpu_usage_percent"),
            cv_scores=optimization_report.get("cv_scores"),
            cv_mean=optimization_report.get("cv_mean"),
            cv_std=optimization_report.get("cv_std"),
        )

    async def _capture_optimization_configuration(
        self,
        optimization_params: dict[str, Any],
        performance_results: PerformanceResultsDTO | None,
        optimization_report: dict[str, Any],
        optimized_detector: Detector | None,
        success: bool,
        capture_config: dict[str, Any] | None = None,
    ) -> str | None:
        """Capture optimization configuration."""
        try:
            # Create capture request
            capture_request = ConfigurationCaptureRequestDTO(
                source=ConfigurationSource.AUTOML,
                raw_parameters=optimization_params,
                execution_results=performance_results.model_dump()
                if performance_results
                else None,
                source_context={
                    "optimization_report": optimization_report,
                    "success": success,
                    "detector_algorithm": optimized_detector.algorithm_name
                    if optimized_detector
                    else None,
                    "detector_params": getattr(optimized_detector, "params", {})
                    if optimized_detector
                    else {},
                    **(capture_config or {}),
                },
                auto_save=True,
                generate_name=True,
                tags=[
                    "automl",
                    "optimization",
                    optimization_params.get("algorithm", "unknown"),
                ],
            )

            # Capture configuration
            response = await self.configuration_service.capture_configuration(
                capture_request
            )

            if response.success:
                self.integration_stats["successful_captures"] += 1
                self.integration_stats["configurations_saved"] += 1
                return str(response.configuration.id)
            else:
                self.integration_stats["failed_captures"] += 1
                logger.error(f"Configuration capture failed: {response.message}")
                return None

        except Exception as e:
            self.integration_stats["failed_captures"] += 1
            logger.error(f"Configuration capture error: {e}")
            return None

    async def _save_optimization_history(
        self,
        optimization_params: dict[str, Any],
        performance_results: PerformanceResultsDTO | None,
        optimization_report: dict[str, Any],
        configuration_id: str | None,
    ) -> None:
        """Save optimization history for future learning."""
        try:
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "configuration_id": configuration_id,
                "optimization_params": optimization_params,
                "performance_results": performance_results.model_dump()
                if performance_results
                else None,
                "optimization_report_summary": {
                    "n_trials": optimization_report.get("n_trials"),
                    "optimization_time": optimization_report.get("optimization_time"),
                    "best_trial_number": optimization_report.get("best_trial_number"),
                    "pareto_front_size": optimization_report.get("pareto_front_size"),
                },
            }

            # Store in service's optimization history
            if hasattr(self.automl_service, "optimization_history"):
                self.automl_service.optimization_history.append(history_entry)
                self.integration_stats["optimization_histories_saved"] += 1

            logger.debug(
                f"Optimization history saved for configuration {configuration_id}"
            )

        except Exception as e:
            logger.warning(f"Failed to save optimization history: {e}")


class AutoMLConfigurationManager:
    """Manager for AutoML configuration operations and analytics."""

    def __init__(self, configuration_service: ConfigurationCaptureService):
        """Initialize AutoML configuration manager.

        Args:
            configuration_service: Configuration capture service
        """
        self.configuration_service = configuration_service

    async def get_automl_configurations(
        self,
        algorithm: str | None = None,
        min_accuracy: float | None = None,
        max_training_time: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get AutoML configurations with filtering.

        Args:
            algorithm: Filter by algorithm name
            min_accuracy: Minimum accuracy threshold
            max_training_time: Maximum training time in seconds
            limit: Maximum number of configurations to return

        Returns:
            List of AutoML configuration dictionaries
        """
        from pynomaly.application.dto.configuration_dto import (
            ConfigurationSearchRequestDTO,
        )

        search_request = ConfigurationSearchRequestDTO(
            source=ConfigurationSource.AUTOML,
            algorithm=algorithm,
            min_accuracy=min_accuracy,
            limit=limit,
            sort_by="accuracy" if min_accuracy else "created_at",
            sort_order="desc",
        )

        configurations = await self.configuration_service.search_configurations(
            search_request
        )

        # Additional filtering for training time
        if max_training_time:
            configurations = [
                config
                for config in configurations
                if (
                    config.performance_results
                    and config.performance_results.training_time_seconds
                    and config.performance_results.training_time_seconds
                    <= max_training_time
                )
            ]

        return [config.model_dump() for config in configurations]

    async def analyze_automl_performance_trends(self) -> dict[str, Any]:
        """Analyze AutoML performance trends.

        Returns:
            Performance trends analysis
        """
        # Get all AutoML configurations
        configs = await self.get_automl_configurations(limit=1000)

        if not configs:
            return {"error": "No AutoML configurations found"}

        # Analyze by algorithm
        algorithm_performance = {}
        for config in configs:
            algorithm = config.get("algorithm_config", {}).get(
                "algorithm_name", "unknown"
            )
            if algorithm not in algorithm_performance:
                algorithm_performance[algorithm] = []

            if config.get("performance_results", {}).get("accuracy"):
                algorithm_performance[algorithm].append(
                    config["performance_results"]["accuracy"]
                )

        # Calculate statistics
        algorithm_stats = {}
        for algorithm, accuracies in algorithm_performance.items():
            if accuracies:
                algorithm_stats[algorithm] = {
                    "count": len(accuracies),
                    "mean_accuracy": sum(accuracies) / len(accuracies),
                    "max_accuracy": max(accuracies),
                    "min_accuracy": min(accuracies),
                }

        # Time-based trends
        from datetime import datetime

        import pandas as pd

        config_df = pd.DataFrame(configs)
        if not config_df.empty and "metadata" in config_df.columns:
            config_df["created_at"] = pd.to_datetime(
                config_df["metadata"].apply(lambda x: x.get("created_at"))
            )

            # Weekly performance trend
            config_df["week"] = config_df["created_at"].dt.to_period("W")
            weekly_stats = (
                config_df.groupby("week")
                .agg(
                    {
                        "performance_results": lambda x: [
                            r.get("accuracy") for r in x if r and r.get("accuracy")
                        ]
                    }
                )
                .to_dict()
            )
        else:
            weekly_stats = {}

        return {
            "total_configurations": len(configs),
            "algorithm_performance": algorithm_stats,
            "weekly_trends": weekly_stats,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    async def export_automl_configurations(
        self,
        algorithm: str | None = None,
        export_format: str = "json",
        include_performance: bool = True,
    ) -> str:
        """Export AutoML configurations for analysis.

        Args:
            algorithm: Filter by algorithm name
            export_format: Export format (json, yaml, csv)
            include_performance: Include performance results

        Returns:
            Exported configuration data as string
        """
        from pynomaly.application.dto.configuration_dto import (
            ConfigurationExportRequestDTO,
            ExportFormat,
        )

        # Get AutoML configurations
        configs = await self.get_automl_configurations(algorithm=algorithm, limit=1000)
        config_ids = [config["id"] for config in configs]

        if not config_ids:
            return ""

        # Create export request
        export_request = ConfigurationExportRequestDTO(
            configuration_ids=config_ids,
            export_format=ExportFormat(export_format.lower()),
            include_metadata=True,
            include_performance=include_performance,
            include_lineage=False,
        )

        # Export configurations
        response = await self.configuration_service.export_configurations(
            export_request
        )

        if response.success:
            return response.export_data
        else:
            raise RuntimeError(f"Export failed: {response.message}")
