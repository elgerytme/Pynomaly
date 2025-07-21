"""Autonomous Mode Configuration Integration Service.

This module extends the autonomous detection service to automatically capture
and save successful configurations for future learning and reuse.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

# TODO: Create local configuration DTOs
from monorepo.application.services.autonomous_service import (
    AutonomousConfig,
    AutonomousDetectionService,
)
# TODO: Create local configuration service
from monorepo.domain.entities import DetectionResult

logger = logging.getLogger(__name__)


class AutonomousConfigurationIntegration:
    """Integration service for autonomous mode configuration capture."""

    def __init__(
        self,
        autonomous_service: AutonomousDetectionService,
        configuration_service: ConfigurationCaptureService,
        auto_save_successful: bool = True,
        auto_save_threshold: float = 0.7,
        save_preprocessing_configs: bool = True,
        save_algorithm_selections: bool = True,
    ):
        """Initialize autonomous configuration integration.

        Args:
            autonomous_service: Autonomous detection service
            configuration_service: Configuration capture service
            auto_save_successful: Automatically save successful configurations
            auto_save_threshold: Minimum performance threshold for auto-save
            save_preprocessing_configs: Save preprocessing configurations
            save_algorithm_selections: Save algorithm selection decisions
        """
        self.autonomous_service = autonomous_service
        self.configuration_service = configuration_service
        self.auto_save_successful = auto_save_successful
        self.auto_save_threshold = auto_save_threshold
        self.save_preprocessing_configs = save_preprocessing_configs
        self.save_algorithm_selections = save_algorithm_selections

        # Integration statistics
        self.integration_stats = {
            "total_autonomous_runs": 0,
            "configurations_saved": 0,
            "preprocessing_configs_saved": 0,
            "algorithm_selection_configs_saved": 0,
            "successful_captures": 0,
            "failed_captures": 0,
            "configurations_above_threshold": 0,
        }

    async def detect_with_configuration_capture(
        self,
        data_source: str | Path | pd.DataFrame,
        config: AutonomousConfig | None = None,
        capture_metadata: dict[str, Any] | None = None,
    ) -> tuple[DetectionResult, str | None]:
        """Run autonomous detection with configuration capture.

        Args:
            data_source: Data source for detection
            config: Autonomous detection configuration
            capture_metadata: Additional metadata for capture

        Returns:
            Tuple of (detection_result, configuration_id)
        """
        self.integration_stats["total_autonomous_runs"] += 1

        config = config or AutonomousConfig()
        start_time = datetime.now()
        configuration_id = None

        try:
            logger.info("Starting autonomous detection with configuration capture")

            # Extract initial parameters for configuration capture
            initial_params = self._extract_initial_parameters(
                data_source, config, capture_metadata
            )

            # Run autonomous detection
            detection_result = await self.autonomous_service.detect_anomalies(
                data_source=data_source, config=config
            )

            end_time = datetime.now()
            detection_duration = (end_time - start_time).total_seconds()

            # Extract comprehensive results for configuration capture
            if self.auto_save_successful and self._should_save_configuration(
                detection_result
            ):
                configuration_id = await self._capture_autonomous_configuration(
                    initial_params=initial_params,
                    detection_result=detection_result,
                    config=config,
                    detection_duration=detection_duration,
                    capture_metadata=capture_metadata,
                )

                if configuration_id:
                    self.integration_stats["configurations_saved"] += 1

                    # Analyze performance
                    performance_score = self._calculate_performance_score(
                        detection_result
                    )
                    if performance_score >= self.auto_save_threshold:
                        self.integration_stats["configurations_above_threshold"] += 1

            # Save preprocessing configuration if enabled
            if self.save_preprocessing_configs and hasattr(
                detection_result, "preprocessing_report"
            ):
                await self._save_preprocessing_configuration(
                    detection_result, initial_params, capture_metadata
                )

            # Save algorithm selection configuration if enabled
            if self.save_algorithm_selections and hasattr(
                detection_result, "algorithm_explanations"
            ):
                await self._save_algorithm_selection_configuration(
                    detection_result, initial_params, capture_metadata
                )

            logger.info(
                f"Autonomous detection completed. Configuration ID: {configuration_id}"
            )
            return detection_result, configuration_id

        except Exception as e:
            logger.error(f"Autonomous detection with configuration capture failed: {e}")
            raise e

    async def capture_autonomous_experiment(
        self,
        experiment_name: str,
        data_sources: list[str | Path | pd.DataFrame],
        configs: list[AutonomousConfig],
        experiment_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[DetectionResult, str | None]]:
        """Run autonomous experiment with configuration capture.

        Args:
            experiment_name: Name of the experiment
            data_sources: List of data sources
            configs: List of configurations to test
            experiment_metadata: Experiment-level metadata

        Returns:
            List of (detection_result, configuration_id) tuples
        """
        experiment_id = str(uuid4())
        results = []

        logger.info(
            f"Starting autonomous experiment '{experiment_name}' with {len(data_sources)} datasets and {len(configs)} configurations"
        )

        for i, data_source in enumerate(data_sources):
            for j, config in enumerate(configs):
                try:
                    # Prepare capture metadata with experiment context
                    capture_metadata = {
                        "experiment_name": experiment_name,
                        "experiment_id": experiment_id,
                        "dataset_index": i,
                        "config_index": j,
                        "total_datasets": len(data_sources),
                        "total_configs": len(configs),
                        **(experiment_metadata or {}),
                    }

                    # Run detection with capture
                    result, config_id = await self.detect_with_configuration_capture(
                        data_source=data_source,
                        config=config,
                        capture_metadata=capture_metadata,
                    )

                    results.append((result, config_id))

                except Exception as e:
                    logger.warning(
                        f"Experiment run failed for dataset {i}, config {j}: {e}"
                    )
                    results.append((None, None))

        logger.info(
            f"Autonomous experiment '{experiment_name}' completed with {len([r for r in results if r[0] is not None])} successful runs"
        )
        return results

    async def analyze_autonomous_configurations(
        self, days_back: int = 30, min_performance: float | None = None
    ) -> dict[str, Any]:
        """Analyze autonomous mode configuration patterns.

        Args:
            days_back: Number of days to analyze
            min_performance: Minimum performance threshold

        Returns:
            Analysis results dictionary
        """
        from datetime import timedelta

        # TODO: Create local configuration DTOs

        # Search for autonomous configurations
        search_request = ConfigurationSearchRequestDTO(
            source=ConfigurationSource.AUTONOMOUS,
            created_after=datetime.now() - timedelta(days=days_back),
            min_accuracy=min_performance,
            limit=1000,
            sort_by="created_at",
            sort_order="desc",
        )

        configurations = await self.configuration_service.search_configurations(
            search_request
        )

        if not configurations:
            return {
                "message": "No autonomous configurations found",
                "configurations": 0,
            }

        # Analyze patterns
        algorithm_usage = {}
        preprocessing_patterns = {}
        performance_distribution = []

        for config in configurations:
            # Algorithm usage
            algorithm = config.algorithm_config.algorithm_name
            algorithm_usage[algorithm] = algorithm_usage.get(algorithm, 0) + 1

            # Preprocessing patterns
            if config.preprocessing_config:
                strategy = f"{config.preprocessing_config.scaling_method}_{config.preprocessing_config.missing_value_strategy}"
                preprocessing_patterns[strategy] = (
                    preprocessing_patterns.get(strategy, 0) + 1
                )

            # Performance distribution
            if config.performance_results and config.performance_results.accuracy:
                performance_distribution.append(config.performance_results.accuracy)

        # Calculate statistics
        if performance_distribution:
            avg_performance = sum(performance_distribution) / len(
                performance_distribution
            )
            max_performance = max(performance_distribution)
            min_performance = min(performance_distribution)
        else:
            avg_performance = max_performance = min_performance = 0.0

        return {
            "analysis_period_days": days_back,
            "total_configurations": len(configurations),
            "algorithm_usage": algorithm_usage,
            "preprocessing_patterns": preprocessing_patterns,
            "performance_statistics": {
                "average": avg_performance,
                "maximum": max_performance,
                "minimum": min_performance,
                "distribution": performance_distribution,
            },
            "most_used_algorithm": (
                max(algorithm_usage.items(), key=lambda x: x[1])[0]
                if algorithm_usage
                else None
            ),
            "most_used_preprocessing": (
                max(preprocessing_patterns.items(), key=lambda x: x[1])[0]
                if preprocessing_patterns
                else None
            ),
            "integration_statistics": self.integration_stats,
        }

    def get_integration_statistics(self) -> dict[str, Any]:
        """Get integration statistics.

        Returns:
            Integration statistics dictionary
        """
        return {
            "integration_stats": self.integration_stats,
            "autonomous_service_info": {
                "available_algorithms": len(
                    getattr(self.autonomous_service, "algorithm_adapters", {})
                ),
                "data_loaders": list(
                    getattr(self.autonomous_service, "data_loaders", {}).keys()
                ),
                "preprocessing_enabled": getattr(
                    self.autonomous_service, "preprocessing_orchestrator", None
                )
                is not None,
            },
            "capture_settings": {
                "auto_save_successful": self.auto_save_successful,
                "auto_save_threshold": self.auto_save_threshold,
                "save_preprocessing_configs": self.save_preprocessing_configs,
                "save_algorithm_selections": self.save_algorithm_selections,
            },
        }

    # Private methods

    def _extract_initial_parameters(
        self,
        data_source: str | Path | pd.DataFrame,
        config: AutonomousConfig,
        capture_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Extract initial parameters for configuration capture."""
        params = {
            "autonomous_mode": True,
            "data_source_type": self._get_data_source_type(data_source),
            "max_samples_analysis": config.max_samples_analysis,
            "confidence_threshold": config.confidence_threshold,
            "max_algorithms": config.max_algorithms,
            "auto_tune_hyperparams": config.auto_tune_hyperparams,
            "enable_preprocessing": config.enable_preprocessing,
            "preprocessing_strategy": config.preprocessing_strategy,
            "enable_explainability": config.enable_explainability,
            "explain_algorithm_choices": config.explain_algorithm_choices,
            "explain_anomalies": config.explain_anomalies,
        }

        # Add data source information
        if isinstance(data_source, str | Path):
            params["dataset_path"] = str(data_source)
            params["dataset_name"] = Path(data_source).stem
        elif isinstance(data_source, pd.DataFrame):
            params["dataset_shape"] = data_source.shape
            params["dataset_name"] = (
                f"dataframe_{data_source.shape[0]}x{data_source.shape[1]}"
            )

        # Add capture metadata
        if capture_metadata:
            params.update(capture_metadata)

        return params

    def _get_data_source_type(self, data_source: str | Path | pd.DataFrame) -> str:
        """Get data source type for configuration."""
        if isinstance(data_source, pd.DataFrame):
            return "dataframe"
        elif isinstance(data_source, str | Path):
            extension = Path(data_source).suffix.lower()
            if extension in [".csv", ".tsv", ".txt"]:
                return "csv"
            elif extension in [".parquet", ".pq"]:
                return "parquet"
            elif extension in [".xlsx", ".xls"]:
                return "excel"
            elif extension in [".json", ".jsonl"]:
                return "json"
            else:
                return "unknown"
        else:
            return "unknown"

    def _should_save_configuration(self, detection_result: DetectionResult) -> bool:
        """Determine if configuration should be saved based on results."""
        if not detection_result:
            return False

        # Check performance threshold
        performance_score = self._calculate_performance_score(detection_result)

        # Save if performance is above threshold
        if performance_score >= self.auto_save_threshold:
            return True

        # Save if detection was successful and produced reasonable results
        if (
            hasattr(detection_result, "anomalies")
            and len(detection_result.anomalies) > 0
            and len(detection_result.anomalies) < len(detection_result.scores) * 0.5
        ):  # Not too many anomalies
            return True

        return False

    def _calculate_performance_score(self, detection_result: DetectionResult) -> float:
        """Calculate overall performance score for detection result."""
        if not detection_result:
            return 0.0

        # Base score from anomaly detection quality
        base_score = 0.5

        # Adjust based on anomaly ratio (reasonable range is 5-20%)
        if hasattr(detection_result, "anomaly_rate"):
            anomaly_rate = detection_result.anomaly_rate
            if 0.05 <= anomaly_rate <= 0.20:
                base_score += 0.2  # Good anomaly rate
            elif 0.01 <= anomaly_rate <= 0.30:
                base_score += 0.1  # Acceptable anomaly rate

        # Adjust based on confidence in results
        if hasattr(detection_result, "confidence_scores"):
            avg_confidence = sum(detection_result.confidence_scores) / len(
                detection_result.confidence_scores
            )
            base_score += (avg_confidence - 0.5) * 0.3

        # Adjust based on processing success
        if (
            hasattr(detection_result, "processing_time")
            and detection_result.processing_time > 0
        ):
            base_score += 0.1  # Successfully completed processing

        # Adjust based on explanations available
        if hasattr(detection_result, "explanations") and detection_result.explanations:
            base_score += 0.1  # Explanations available

        return max(0.0, min(1.0, base_score))

    async def _capture_autonomous_configuration(
        self,
        initial_params: dict[str, Any],
        detection_result: DetectionResult,
        config: AutonomousConfig,
        detection_duration: float,
        capture_metadata: dict[str, Any] | None,
    ) -> str | None:
        """Capture autonomous detection configuration."""
        try:
            # Extract performance results
            performance_results = self._extract_performance_results(
                detection_result, detection_duration
            )

            # Create capture request
            capture_request = ConfigurationCaptureRequestDTO(
                source=ConfigurationSource.AUTONOMOUS,
                raw_parameters=initial_params,
                execution_results=(
                    performance_results.model_dump() if performance_results else None
                ),
                source_context={
                    "autonomous_config": config.__dict__,
                    "detection_summary": self._create_detection_summary(
                        detection_result
                    ),
                    "performance_score": self._calculate_performance_score(
                        detection_result
                    ),
                    **(capture_metadata or {}),
                },
                auto_save=True,
                generate_name=True,
                tags=["autonomous", "auto_detection", "successful"],
            )

            # Capture configuration
            response = await self.configuration_service.capture_configuration(
                capture_request
            )

            if response.success:
                self.integration_stats["successful_captures"] += 1
                return str(response.configuration.id)
            else:
                self.integration_stats["failed_captures"] += 1
                logger.error(
                    f"Autonomous configuration capture failed: {response.message}"
                )
                return None

        except Exception as e:
            self.integration_stats["failed_captures"] += 1
            logger.error(f"Autonomous configuration capture error: {e}")
            return None

    def _extract_performance_results(
        self, detection_result: DetectionResult, detection_duration: float
    ) -> PerformanceResultsDTO | None:
        """Extract performance results from detection result."""
        if not detection_result:
            return None

        return PerformanceResultsDTO(
            training_time_seconds=detection_duration,
            prediction_time_ms=getattr(detection_result, "prediction_time_ms", None),
            memory_usage_mb=getattr(detection_result, "memory_usage_mb", None),
            anomaly_scores=(
                detection_result.scores.tolist()
                if hasattr(detection_result.scores, "tolist")
                else None
            ),
            # Additional autonomous-specific metrics
            stability_score=getattr(detection_result, "stability_score", None),
            robustness_score=getattr(detection_result, "robustness_score", None),
            interpretability_score=getattr(
                detection_result, "interpretability_score", None
            ),
        )

    def _create_detection_summary(
        self, detection_result: DetectionResult
    ) -> dict[str, Any]:
        """Create summary of detection results."""
        if not detection_result:
            return {}

        return {
            "n_samples": (
                len(detection_result.scores)
                if detection_result.scores is not None
                else 0
            ),
            "n_anomalies": detection_result.n_anomalies,
            "anomaly_rate": detection_result.anomaly_rate,
            "detector_algorithm": (
                detection_result.detector_name
                if hasattr(detection_result, "detector_name")
                else None
            ),
            "timestamp": detection_result.timestamp.isoformat(),
            "has_explanations": hasattr(detection_result, "explanations")
            and detection_result.explanations is not None,
            "processing_successful": True,
        }

    async def _save_preprocessing_configuration(
        self,
        detection_result: DetectionResult,
        initial_params: dict[str, Any],
        capture_metadata: dict[str, Any] | None,
    ) -> None:
        """Save preprocessing configuration separately."""
        try:
            if not hasattr(detection_result, "preprocessing_report"):
                return

            preprocessing_params = {
                **initial_params,
                "preprocessing_report": detection_result.preprocessing_report,
                "configuration_type": "preprocessing",
            }

            capture_request = ConfigurationCaptureRequestDTO(
                source=ConfigurationSource.AUTONOMOUS,
                raw_parameters=preprocessing_params,
                source_context={
                    "configuration_focus": "preprocessing",
                    **(capture_metadata or {}),
                },
                auto_save=True,
                generate_name=True,
                tags=["autonomous", "preprocessing", "data_quality"],
            )

            response = await self.configuration_service.capture_configuration(
                capture_request
            )

            if response.success:
                self.integration_stats["preprocessing_configs_saved"] += 1

        except Exception as e:
            logger.warning(f"Failed to save preprocessing configuration: {e}")

    async def _save_algorithm_selection_configuration(
        self,
        detection_result: DetectionResult,
        initial_params: dict[str, Any],
        capture_metadata: dict[str, Any] | None,
    ) -> None:
        """Save algorithm selection configuration separately."""
        try:
            if not hasattr(detection_result, "algorithm_explanations"):
                return

            selection_params = {
                **initial_params,
                "algorithm_explanations": detection_result.algorithm_explanations,
                "selected_algorithms": getattr(
                    detection_result, "selected_algorithms", []
                ),
                "configuration_type": "algorithm_selection",
            }

            capture_request = ConfigurationCaptureRequestDTO(
                source=ConfigurationSource.AUTONOMOUS,
                raw_parameters=selection_params,
                source_context={
                    "configuration_focus": "algorithm_selection",
                    **(capture_metadata or {}),
                },
                auto_save=True,
                generate_name=True,
                tags=["autonomous", "algorithm_selection", "ml_reasoning"],
            )

            response = await self.configuration_service.capture_configuration(
                capture_request
            )

            if response.success:
                self.integration_stats["algorithm_selection_configs_saved"] += 1

        except Exception as e:
            logger.warning(f"Failed to save algorithm selection configuration: {e}")


class AutonomousConfigurationManager:
    """Manager for autonomous mode configuration analytics and optimization."""

    def __init__(self, configuration_service: ConfigurationCaptureService):
        """Initialize autonomous configuration manager.

        Args:
            configuration_service: Configuration capture service
        """
        self.configuration_service = configuration_service

    async def recommend_autonomous_config(
        self,
        dataset_characteristics: dict[str, Any],
        performance_requirements: dict[str, float] | None = None,
    ) -> AutonomousConfig:
        """Recommend autonomous configuration based on similar successful runs.

        Args:
            dataset_characteristics: Characteristics of the target dataset
            performance_requirements: Performance requirements

        Returns:
            Recommended autonomous configuration
        """
        # TODO: Create local configuration DTOs

        # Search for successful autonomous configurations
        search_request = ConfigurationSearchRequestDTO(
            source=ConfigurationSource.AUTONOMOUS,
            tags=["successful"],
            min_accuracy=(
                performance_requirements.get("min_accuracy", 0.7)
                if performance_requirements
                else 0.7
            ),
            limit=50,
            sort_by="accuracy",
            sort_order="desc",
        )

        configurations = await self.configuration_service.search_configurations(
            search_request
        )

        if not configurations:
            return AutonomousConfig()  # Return default config

        # Analyze successful configurations to find common patterns
        confidence_thresholds = []
        max_algorithms_values = []
        preprocessing_strategies = []

        for config in configurations:
            if config.source_context:
                autonomous_config = config.source_context.get("autonomous_config", {})
                confidence_thresholds.append(
                    autonomous_config.get("confidence_threshold", 0.8)
                )
                max_algorithms_values.append(autonomous_config.get("max_algorithms", 5))
                preprocessing_strategies.append(
                    autonomous_config.get("preprocessing_strategy", "auto")
                )

        # Calculate optimal values
        recommended_config = AutonomousConfig()

        if confidence_thresholds:
            recommended_config.confidence_threshold = sum(confidence_thresholds) / len(
                confidence_thresholds
            )

        if max_algorithms_values:
            # Use mode for max_algorithms
            from collections import Counter

            recommended_config.max_algorithms = Counter(
                max_algorithms_values
            ).most_common(1)[0][0]

        if preprocessing_strategies:
            # Use most common strategy
            from collections import Counter

            recommended_config.preprocessing_strategy = Counter(
                preprocessing_strategies
            ).most_common(1)[0][0]

        return recommended_config

    async def export_autonomous_configurations(
        self,
        include_performance: bool = True,
        include_explanations: bool = False,
        export_format: str = "json",
    ) -> str:
        """Export autonomous configurations for analysis.

        Args:
            include_performance: Include performance results
            include_explanations: Include explanation data
            export_format: Export format (json, yaml, csv)

        Returns:
            Exported configuration data as string
        """
        # TODO: Create local configuration DTOs

        # Get all autonomous configurations
        search_request = ConfigurationSearchRequestDTO(
            source=ConfigurationSource.AUTONOMOUS,
            limit=1000,
            sort_by="created_at",
            sort_order="desc",
        )

        configurations = await self.configuration_service.search_configurations(
            search_request
        )
        config_ids = [config.id for config in configurations]

        if not config_ids:
            return ""

        # Create export request
        export_request = ConfigurationExportRequestDTO(
            configuration_ids=config_ids,
            export_format=ExportFormat(export_format.lower()),
            include_metadata=True,
            include_performance=include_performance,
            include_lineage=include_explanations,
        )

        # Export configurations
        response = await self.configuration_service.export_configurations(
            export_request
        )

        if response.success:
            return response.export_data
        else:
            raise RuntimeError(f"Export failed: {response.message}")
