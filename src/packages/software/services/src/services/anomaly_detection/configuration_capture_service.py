"""Configuration capture service for automatic experiment configuration management."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from monorepo.application.dto.configuration_dto import (
    AlgorithmConfigDTO,
    ConfigurationCaptureRequestDTO,
    ConfigurationExportRequestDTO,
    ConfigurationLineageDTO,
    ConfigurationMetadataDTO,
    ConfigurationResponseDTO,
    ConfigurationSearchRequestDTO,
    ConfigurationSource,
    ConfigurationValidationResultDTO,
    DatasetConfigDTO,
    EvaluationConfigDTO,
    ExperimentConfigurationDTO,
    ExportFormat,
    PerformanceResultsDTO,
    PreprocessingConfigDTO,
)

logger = logging.getLogger(__name__)


class ConfigurationCaptureService:
    """Service for capturing, managing, and exporting experiment configurations."""

    def __init__(
        self,
        storage_path: Path | None = None,
        auto_capture: bool = True,
        enable_lineage_tracking: bool = True,
        max_configurations: int = 10000,
    ):
        """Initialize configuration capture service.

        Args:
            storage_path: Path to store configurations
            auto_capture: Enable automatic configuration capture
            enable_lineage_tracking: Enable configuration lineage tracking
            max_configurations: Maximum number of configurations to store
        """
        self.storage_path = storage_path or Path("data/configurations")
        self.auto_capture = auto_capture
        self.enable_lineage_tracking = enable_lineage_tracking
        self.max_configurations = max_configurations

        # Initialize storage
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory configuration cache
        self.configuration_cache: dict[UUID, ExperimentConfigurationDTO] = {}
        self.configuration_index: dict[
            str, list[UUID]
        ] = {}  # Tag/algorithm -> config IDs

        # Capture state
        self.active_captures: dict[
            str, dict[str, Any]
        ] = {}  # session_id -> capture context
        self.capture_stats = {
            "total_captures": 0,
            "successful_captures": 0,
            "failed_captures": 0,
            "auto_captures": 0,
            "manual_captures": 0,
        }

        # Load existing configurations
        asyncio.create_task(self._load_existing_configurations())

    async def capture_configuration(
        self, request: ConfigurationCaptureRequestDTO
    ) -> ConfigurationResponseDTO:
        """Capture configuration from execution context.

        Args:
            request: Configuration capture request

        Returns:
            Configuration response with captured configuration
        """
        try:
            logger.info(f"Capturing configuration from {request.source}")

            # Extract configuration from raw parameters
            config = await self._extract_configuration_from_parameters(
                request.raw_parameters, request.source, request.source_context
            )

            # Add metadata
            config.metadata.created_by = request.user_id
            config.metadata.source = request.source
            config.metadata.tags.extend(request.tags)

            # Add performance results if available
            if request.execution_results:
                config.performance_results = await self._extract_performance_results(
                    request.execution_results
                )

            # Add lineage information
            if self.enable_lineage_tracking:
                config.lineage = await self._extract_lineage_information(
                    request.source_context
                )

            # Validate configuration
            validation_result = await self.validate_configuration(config)
            config.is_valid = validation_result.is_valid
            config.validation_errors = validation_result.errors
            config.validation_warnings = validation_result.warnings

            # Generate name if requested
            if request.generate_name:
                config.name = await self._generate_configuration_name(config)

            # Save configuration if requested
            if request.auto_save:
                await self.save_configuration(config)

            # Update statistics
            self.capture_stats["total_captures"] += 1
            self.capture_stats["successful_captures"] += 1
            if request.source == ConfigurationSource.AUTOML:
                self.capture_stats["auto_captures"] += 1
            else:
                self.capture_stats["manual_captures"] += 1

            logger.info(f"Successfully captured configuration: {config.name}")

            return ConfigurationResponseDTO(
                success=True,
                message="Configuration captured successfully",
                configuration=config,
            )

        except Exception as e:
            logger.error(f"Failed to capture configuration: {e}")
            self.capture_stats["failed_captures"] += 1

            return ConfigurationResponseDTO(
                success=False,
                message=f"Configuration capture failed: {e}",
                errors=[str(e)],
            )

    async def save_configuration(self, config: ExperimentConfigurationDTO) -> bool:
        """Save configuration to storage.

        Args:
            config: Configuration to save

        Returns:
            True if saved successfully
        """
        try:
            # Add to cache
            self.configuration_cache[config.id] = config

            # Update indexes
            await self._update_configuration_indexes(config)

            # Save to disk
            config_file = self.storage_path / f"{config.id}.json"
            with open(config_file, "w") as f:
                json.dump(config.processor_dump(), f, indent=2, default=str)

            # Maintain cache size
            if len(self.configuration_cache) > self.max_configurations:
                await self._cleanup_old_configurations()

            logger.info(f"Saved configuration {config.name} to {config_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration {config.name}: {e}")
            return False

    async def load_configuration(
        self, config_id: UUID
    ) -> ExperimentConfigurationDTO | None:
        """Load configuration by ID.

        Args:
            config_id: Configuration ID

        Returns:
            Configuration if found, None otherwise
        """
        # Check cache first
        if config_id in self.configuration_cache:
            return self.configuration_cache[config_id]

        # Load from disk
        config_file = self.storage_path / f"{config_id}.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config_data = json.load(f)

                config = ExperimentConfigurationDTO(**config_data)
                self.configuration_cache[config_id] = config
                return config

            except Exception as e:
                logger.error(f"Failed to load configuration {config_id}: {e}")

        return None

    async def search_configurations(
        self, request: ConfigurationSearchRequestDTO
    ) -> ConfigurationResponseDTO:
        """Search configurations based on criteria.

        Args:
            request: Search request parameters

        Returns:
            Response with matching configurations
        """
        try:
            # Get all configurations
            all_configs = []
            for config_file in self.storage_path.glob("*.json"):
                try:
                    with open(config_file) as f:
                        config_data = json.load(f)
                    config = ExperimentConfigurationDTO(**config_data)
                    all_configs.append(config)
                except Exception as e:
                    logger.warning(f"Failed to load config {config_file}: {e}")
                    continue

            # Apply filters
            filtered_configs = await self._apply_search_filters(all_configs, request)

            # Sort results
            sorted_configs = await self._sort_configurations(filtered_configs, request)

            # Apply pagination
            start_idx = request.offset
            end_idx = start_idx + request.limit
            paginated_configs = sorted_configs[start_idx:end_idx]

            return ConfigurationResponseDTO(
                success=True,
                message=f"Found {len(filtered_configs)} configurations",
                configurations=paginated_configs,
                total_count=len(filtered_configs),
            )

        except Exception as e:
            logger.error(f"Configuration search failed: {e}")
            return ConfigurationResponseDTO(
                success=False, message=f"Search failed: {e}", errors=[str(e)]
            )

    async def export_configurations(
        self, request: ConfigurationExportRequestDTO
    ) -> ConfigurationResponseDTO:
        """Export configurations in specified format.

        Args:
            request: Export request parameters

        Returns:
            Response with exported configuration data
        """
        try:
            # Load configurations
            configurations = []
            for config_id in request.configuration_ids:
                config = await self.load_configuration(config_id)
                if config:
                    configurations.append(config)

            if not configurations:
                return ConfigurationResponseDTO(
                    success=False,
                    message="No configurations found to export",
                    errors=["No valid configuration IDs provided"],
                )

            # Generate export data based on format
            if request.export_format == ExportFormat.YAML:
                export_data = await self._export_to_yaml(configurations, request)
            elif request.export_format == ExportFormat.JSON:
                export_data = await self._export_to_json(configurations, request)
            elif request.export_format == ExportFormat.PYTHON:
                export_data = await self._export_to_python(configurations, request)
            elif request.export_format == ExportFormat.NOTEBOOK:
                export_data = await self._export_to_notebook(configurations, request)
            elif request.export_format == ExportFormat.DOCKER:
                export_data = await self._export_to_docker(configurations, request)
            else:
                raise ValueError(f"Unsupported export format: {request.export_format}")

            # Save to file if path specified
            export_files = []
            if request.output_path:
                output_path = Path(request.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w") as f:
                    f.write(export_data)

                export_files.append(str(output_path))

            return ConfigurationResponseDTO(
                success=True,
                message=f"Exported {len(configurations)} configurations",
                export_data=export_data,
                export_files=export_files,
            )

        except Exception as e:
            logger.error(f"Configuration export failed: {e}")
            return ConfigurationResponseDTO(
                success=False, message=f"Export failed: {e}", errors=[str(e)]
            )

    async def validate_configuration(
        self, config: ExperimentConfigurationDTO
    ) -> ConfigurationValidationResultDTO:
        """Validate configuration for completeness and correctness.

        Args:
            config: Configuration to validate

        Returns:
            Validation results
        """
        errors = []
        warnings = []
        suggestions = []

        # Validate data_collection configuration
        if (
            not config.data_collection_config.data_collection_path
            and not config.data_collection_config.data_collection_name
        ):
            errors.append("DataCollection path or name must be specified")

        # Validate algorithm configuration
        if not config.algorithm_config.algorithm_name:
            errors.append("Algorithm name must be specified")

        # Validate hyperparameters
        if config.algorithm_config.contamination and (
            config.algorithm_config.contamination < 0
            or config.algorithm_config.contamination > 0.5
        ):
            errors.append("Contamination rate must be between 0 and 0.5")

        # Check for common issues
        if config.preprocessing_config and config.preprocessing_config.apply_pca:
            if not config.preprocessing_config.pca_components:
                warnings.append("PCA enabled but number of components not specified")

        # Estimate resource requirements
        estimated_runtime = await self._estimate_runtime(config)
        estimated_memory = await self._estimate_memory_usage(config)

        # Calculate validation score
        validation_score = 1.0
        validation_score -= len(errors) * 0.2
        validation_score -= len(warnings) * 0.1
        validation_score = max(0.0, validation_score)

        return ConfigurationValidationResultDTO(
            is_valid=len(errors) == 0,
            validation_score=validation_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            estimated_runtime=estimated_runtime,
            estimated_memory=estimated_memory,
        )

    async def get_configuration_statistics(self) -> dict[str, Any]:
        """Get configuration management statistics.

        Returns:
            Statistics dictionary
        """
        # Count configurations by source
        source_counts = {}
        algorithm_counts = {}

        for config in self.configuration_cache.values():
            source = config.metadata.source
            source_counts[source] = source_counts.get(source, 0) + 1

            algorithm = config.algorithm_config.algorithm_name
            algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1

        return {
            "total_configurations": len(self.configuration_cache),
            "capture_statistics": self.capture_stats,
            "configurations_by_source": source_counts,
            "configurations_by_algorithm": algorithm_counts,
            "storage_path": str(self.storage_path),
            "cache_size": len(self.configuration_cache),
            "auto_capture_enabled": self.auto_capture,
            "lineage_tracking_enabled": self.enable_lineage_tracking,
        }

    # Private methods

    async def _extract_configuration_from_parameters(
        self,
        raw_parameters: dict[str, Any],
        source: ConfigurationSource,
        context: dict[str, Any],
    ) -> ExperimentConfigurationDTO:
        """Extract structured configuration from raw parameters."""
        # Generate configuration ID and name
        config_id = uuid4()
        config_name = context.get(
            "name", f"config_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Extract data_collection configuration
        data_collection_config = DatasetConfigDTO(
            data_collection_path=raw_parameters.get("data_collection_path"),
            data_collection_name=raw_parameters.get("data_collection_name"),
            file_format=raw_parameters.get("file_format"),
            feature_columns=raw_parameters.get("feature_columns"),
            target_column=raw_parameters.get("target_column"),
        )

        # Extract algorithm configuration
        algorithm_config = AlgorithmConfigDTO(
            algorithm_name=raw_parameters.get("algorithm", "isolation_forest"),
            hyperparameters=raw_parameters.get("hyperparameters", {}),
            contamination=raw_parameters.get("contamination", 0.1),
            random_state=raw_parameters.get("random_state", 42),
            n_jobs=raw_parameters.get("n_jobs", 1),
        )

        # Extract preprocessing configuration
        preprocessing_config = None
        if any(key.startswith("preprocess") for key in raw_parameters.keys()):
            preprocessing_config = PreprocessingConfigDTO(
                missing_value_strategy=raw_parameters.get(
                    "missing_value_strategy", "mean"
                ),
                outlier_processing_method=raw_parameters.get("outlier_method", "iqr"),
                scaling_method=raw_parameters.get("scaling_method", "standard"),
                feature_selection_method=raw_parameters.get("feature_selection"),
                apply_pca=raw_parameters.get("apply_pca", False),
                pca_components=raw_parameters.get("pca_components"),
            )

        # Extract evaluation configuration
        evaluation_config = EvaluationConfigDTO(
            primary_metric=raw_parameters.get("metric", "roc_auc"),
            cv_folds=raw_parameters.get("cv_folds", 5),
            test_size=raw_parameters.get("test_size", 0.2),
            calculate_feature_importance=raw_parameters.get("feature_importance", True),
        )

        # Create metadata
        metadata = ConfigurationMetadataDTO(
            source=source,
            tags=context.get("tags", []),
            description=context.get("description"),
            version="1.0.0",
        )

        return ExperimentConfigurationDTO(
            id=config_id,
            name=config_name,
            data_collection_config=data_collection_config,
            algorithm_config=algorithm_config,
            preprocessing_config=preprocessing_config,
            evaluation_config=evaluation_config,
            metadata=metadata,
        )

    async def _extract_performance_results(
        self, execution_results: dict[str, Any]
    ) -> PerformanceResultsDTO:
        """Extract performance results from execution context."""
        return PerformanceResultsDTO(
            accuracy=execution_results.get("accuracy"),
            precision=execution_results.get("precision"),
            recall=execution_results.get("recall"),
            f1_score=execution_results.get("f1_score"),
            roc_auc=execution_results.get("roc_auc"),
            training_time_seconds=execution_results.get("training_time"),
            prediction_time_ms=execution_results.get("prediction_time"),
            memory_usage_mb=execution_results.get("memory_usage"),
            cv_scores=execution_results.get("cv_scores"),
            cv_mean=execution_results.get("cv_mean"),
            cv_std=execution_results.get("cv_std"),
            feature_importance=execution_results.get("feature_importance"),
        )

    async def _extract_lineage_information(
        self, context: dict[str, Any]
    ) -> ConfigurationLineageDTO:
        """Extract configuration lineage information."""
        return ConfigurationLineageDTO(
            parent_configurations=context.get("parent_configs", []),
            derivation_method=context.get("derivation_method"),
            modifications_made=context.get("modifications", []),
            git_commit=context.get("git_commit"),
            git_branch=context.get("git_branch"),
            experiment_id=context.get("experiment_id"),
            run_id=context.get("run_id"),
        )

    async def _generate_configuration_name(
        self, config: ExperimentConfigurationDTO
    ) -> str:
        """Generate a descriptive name for the configuration."""
        algorithm = config.algorithm_config.algorithm_name
        source = config.metadata.source
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config.data_collection_config.data_collection_name:
            data_collection_part = config.data_collection_config.data_collection_name
        elif config.data_collection_config.data_collection_path:
            data_collection_part = Path(config.data_collection_config.data_collection_path).stem
        else:
            data_collection_part = "unknown"

        return f"{source}_{algorithm}_{data_collection_part}_{timestamp}"

    async def _update_configuration_indexes(
        self, config: ExperimentConfigurationDTO
    ) -> None:
        """Update configuration indexes for fast searching."""
        # Index by algorithm
        algorithm = config.algorithm_config.algorithm_name
        if algorithm not in self.configuration_index:
            self.configuration_index[algorithm] = []
        self.configuration_index[algorithm].append(config.id)

        # Index by tags
        for tag in config.metadata.tags:
            if tag not in self.configuration_index:
                self.configuration_index[tag] = []
            self.configuration_index[tag].append(config.id)

        # Index by source
        source = config.metadata.source
        if source not in self.configuration_index:
            self.configuration_index[source] = []
        self.configuration_index[source].append(config.id)

    async def _apply_search_filters(
        self,
        configurations: list[ExperimentConfigurationDTO],
        request: ConfigurationSearchRequestDTO,
    ) -> list[ExperimentConfigurationDTO]:
        """Apply search filters to configuration list."""
        filtered = configurations

        # Filter by query
        if request.query:
            query_lower = request.query.lower()
            filtered = [
                config
                for config in filtered
                if (
                    query_lower in config.name.lower()
                    or query_lower in config.algorithm_config.algorithm_name.lower()
                    or (
                        config.metadata.description
                        and query_lower in config.metadata.description.lower()
                    )
                )
            ]

        # Filter by tags
        if request.tags:
            filtered = [
                config
                for config in filtered
                if all(tag in config.metadata.tags for tag in request.tags)
            ]

        # Filter by source
        if request.source:
            filtered = [
                config
                for config in filtered
                if config.metadata.source == request.source
            ]

        # Filter by algorithm
        if request.algorithm:
            filtered = [
                config
                for config in filtered
                if config.algorithm_config.algorithm_name == request.algorithm
            ]

        # Filter by date range
        if request.created_after:
            filtered = [
                config
                for config in filtered
                if config.metadata.created_at >= request.created_after
            ]

        if request.created_before:
            filtered = [
                config
                for config in filtered
                if config.metadata.created_at <= request.created_before
            ]

        # Filter by performance
        if request.min_accuracy and request.min_accuracy > 0:
            filtered = [
                config
                for config in filtered
                if (
                    config.performance_results
                    and config.performance_results.accuracy
                    and config.performance_results.accuracy >= request.min_accuracy
                )
            ]

        return filtered

    async def _sort_configurations(
        self,
        configurations: list[ExperimentConfigurationDTO],
        request: ConfigurationSearchRequestDTO,
    ) -> list[ExperimentConfigurationDTO]:
        """Sort configurations based on request parameters."""
        reverse = request.sort_order.lower() == "desc"

        if request.sort_by == "created_at":
            return sorted(
                configurations, key=lambda x: x.metadata.created_at, reverse=reverse
            )
        elif request.sort_by == "name":
            return sorted(configurations, key=lambda x: x.name.lower(), reverse=reverse)
        elif request.sort_by == "algorithm":
            return sorted(
                configurations,
                key=lambda x: x.algorithm_config.algorithm_name,
                reverse=reverse,
            )
        elif request.sort_by == "accuracy" and any(
            c.performance_results for c in configurations
        ):
            return sorted(
                configurations,
                key=lambda x: (
                    x.performance_results.accuracy
                    if x.performance_results and x.performance_results.accuracy
                    else 0
                ),
                reverse=reverse,
            )
        else:
            # Default to created_at
            return sorted(
                configurations, key=lambda x: x.metadata.created_at, reverse=reverse
            )

    async def _export_to_yaml(
        self,
        configurations: list[ExperimentConfigurationDTO],
        request: ConfigurationExportRequestDTO,
    ) -> str:
        """Export configurations to YAML format."""
        import yaml

        export_data = []
        for config in configurations:
            config_dict = config.processor_dump()

            # Remove unwanted fields based on request
            if not request.include_metadata:
                config_dict.pop("metadata", None)
            if not request.include_performance:
                config_dict.pop("performance_results", None)
            if not request.include_lineage:
                config_dict.pop("lineage", None)

            export_data.append(config_dict)

        return yaml.dump(export_data, default_flow_style=False, sort_keys=False)

    async def _export_to_json(
        self,
        configurations: list[ExperimentConfigurationDTO],
        request: ConfigurationExportRequestDTO,
    ) -> str:
        """Export configurations to JSON format."""
        export_data = []
        for config in configurations:
            config_dict = config.processor_dump()

            # Remove unwanted fields based on request
            if not request.include_metadata:
                config_dict.pop("metadata", None)
            if not request.include_performance:
                config_dict.pop("performance_results", None)
            if not request.include_lineage:
                config_dict.pop("lineage", None)

            export_data.append(config_dict)

        return json.dumps(export_data, indent=2, default=str)

    async def _export_to_python(
        self,
        configurations: list[ExperimentConfigurationDTO],
        request: ConfigurationExportRequestDTO,
    ) -> str:
        """Export configurations to Python script format."""
        script_lines = [
            "#!/usr/bin/env python3",
            '"""Generated configuration script from Software."""',
            "",
            "import monorepo",
            "from monorepo.application.services import DetectionService",
            "from monorepo.infrastructure.data_loaders import CSVLoader",
            "",
            "def run_configurations():",
            '    """Execute all configurations."""',
        ]

        for i, config in enumerate(configurations):
            script_lines.extend(
                [
                    "",
                    f"    # Configuration {i + 1}: {config.name}",
                    f"    print('Running configuration: {config.name}')",
                    "    ",
                    "    # Load data_collection",
                    "    loader = CSVLoader()",
                    f"    data_collection = loader.load('{config.data_collection_config.data_collection_path}')",
                    "    ",
                    "    # Configure algorithm",
                    "    algorithm_config = {",
                    f"        'algorithm': '{config.algorithm_config.algorithm_name}',",
                    f"        'contamination': {config.algorithm_config.contamination},",
                    f"        'random_state': {config.algorithm_config.random_state}",
                    "    }",
                    "    ",
                    "    # Run processing",
                    "    service = DetectionService()",
                    "    result = service.detect_anomalies(data_collection, algorithm_config)",
                    "    print(f'Found {len(result.anomalies)} anomalies')",
                ]
            )

        script_lines.extend(
            ["", "if __name__ == '__main__':", "    run_configurations()"]
        )

        return "\n".join(script_lines)

    async def _export_to_notebook(
        self,
        configurations: list[ExperimentConfigurationDTO],
        request: ConfigurationExportRequestDTO,
    ) -> str:
        """Export configurations to Jupyter notebook format."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Software Configuration Notebook\n",
                        "\n",
                        f"Generated from {len(configurations)} configurations\n",
                        f"Export date: {datetime.now().isoformat()}",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "import monorepo\n",
                        "from monorepo.application.services import DetectionService\n",
                        "from monorepo.infrastructure.data_loaders import CSVLoader\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11.0"},
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        for i, config in enumerate(configurations):
            # Add configuration cell
            notebook["cells"].extend(
                [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [f"## Configuration {i + 1}: {config.name}"],
                    },
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "source": [
                            f"# Configuration: {config.name}\n",
                            f"data_collection_path = '{config.data_collection_config.data_collection_path}'\n",
                            f"algorithm = '{config.algorithm_config.algorithm_name}'\n",
                            f"contamination = {config.algorithm_config.contamination}\n",
                            f"random_state = {config.algorithm_config.random_state}\n",
                            "\n",
                            "# Load and process data\n",
                            "loader = CSVLoader()\n",
                            "data_collection = loader.load(data_collection_path)\n",
                            "print(f'DataCollection shape: {data_collection.shape}')",
                        ],
                    },
                ]
            )

        return json.dumps(notebook, indent=2)

    async def _export_to_docker(
        self,
        configurations: list[ExperimentConfigurationDTO],
        request: ConfigurationExportRequestDTO,
    ) -> str:
        """Export configurations to Docker Compose format."""
        docker_compose = {"version": "3.8", "services": {}}

        for i, config in enumerate(configurations):
            service_name = f"software-config-{i + 1}"
            docker_compose["services"][service_name] = {
                "image": "software:latest",
                "environment": {
                    "DATASET_PATH": config.data_collection_config.data_collection_path,
                    "ALGORITHM": config.algorithm_config.algorithm_name,
                    "CONTAMINATION": str(config.algorithm_config.contamination),
                    "RANDOM_STATE": str(config.algorithm_config.random_state),
                },
                "volumes": ["./data:/app/data", "./results:/app/results"],
                "command": f"python -m software detect --config /app/configs/config_{i + 1}.json",
            }

        import yaml

        return yaml.dump(docker_compose, default_flow_style=False)

    async def _estimate_runtime(
        self, config: ExperimentConfigurationDTO
    ) -> float | None:
        """Estimate configuration runtime in seconds."""
        # Simple heuristic based on algorithm and data_collection size
        base_times = {
            "isolation_forest": 10.0,
            "local_outlier_factor": 30.0,
            "one_class_svm": 60.0,
            "elliptic_envelope": 20.0,
            "autoencoder": 300.0,
        }

        algorithm = config.algorithm_config.algorithm_name
        base_time = base_times.get(algorithm, 30.0)

        # Adjust for data_collection size (if known)
        if config.data_collection_config.expected_shape:
            n_samples = config.data_collection_config.expected_shape[0]
            n_features = config.data_collection_config.expected_shape[1]

            # Scale time based on data size
            size_factor = (n_samples * n_features) / 10000
            base_time *= max(1.0, size_factor**0.5)

        # Adjust for preprocessing
        if config.preprocessing_config:
            if config.preprocessing_config.apply_pca:
                base_time *= 1.5
            if config.preprocessing_config.feature_selection_method:
                base_time *= 1.3

        return base_time

    async def _estimate_memory_usage(
        self, config: ExperimentConfigurationDTO
    ) -> float | None:
        """Estimate configuration memory usage in MB."""
        # Simple heuristic based on algorithm and data_collection size
        base_memory = {
            "isolation_forest": 100.0,
            "local_outlier_factor": 500.0,
            "one_class_svm": 1000.0,
            "elliptic_envelope": 200.0,
            "autoencoder": 2000.0,
        }

        algorithm = config.algorithm_config.algorithm_name
        memory = base_memory.get(algorithm, 300.0)

        # Adjust for data_collection size
        if config.data_collection_config.expected_shape:
            n_samples = config.data_collection_config.expected_shape[0]
            n_features = config.data_collection_config.expected_shape[1]

            # Estimate data memory usage
            data_memory = (n_samples * n_features * 8) / (
                1024 * 1024
            )  # 8 bytes per float64
            memory += data_memory * 3  # Factor for algorithm overhead

        return memory

    async def _cleanup_old_configurations(self) -> None:
        """Remove old configurations to maintain cache size."""
        # Sort by last used timestamp, remove oldest
        configs_by_usage = sorted(
            self.configuration_cache.items(),
            key=lambda x: x[1].metadata.last_used or x[1].metadata.created_at,
        )

        # Remove oldest 10% of configurations
        num_to_remove = len(configs_by_usage) // 10
        for config_id, _ in configs_by_usage[:num_to_remove]:
            del self.configuration_cache[config_id]

            # Remove from disk
            config_file = self.storage_path / f"{config_id}.json"
            if config_file.exists():
                config_file.unlink()

    async def _load_existing_configurations(self) -> None:
        """Load existing configurations from storage."""
        try:
            for config_file in self.storage_path.glob("*.json"):
                try:
                    with open(config_file) as f:
                        config_data = json.load(f)

                    config = ExperimentConfigurationDTO(**config_data)
                    self.configuration_cache[config.id] = config
                    await self._update_configuration_indexes(config)

                except Exception as e:
                    logger.warning(f"Failed to load configuration {config_file}: {e}")
                    continue

            logger.info(
                f"Loaded {len(self.configuration_cache)} existing configurations"
            )

        except Exception as e:
            logger.error(f"Failed to load existing configurations: {e}")
