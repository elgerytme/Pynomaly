"""Comprehensive tests for ConfigurationCaptureService."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from pynomaly.application.dto.configuration_dto import (
    AlgorithmConfigDTO,
    ConfigurationCaptureRequestDTO,
    ConfigurationExportRequestDTO,
    ConfigurationMetadataDTO,
    ConfigurationSearchRequestDTO,
    ConfigurationSource,
    DatasetConfigDTO,
    EvaluationConfigDTO,
    ExperimentConfigurationDTO,
    ExportFormat,
    PreprocessingConfigDTO,
)
from pynomaly.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)


class TestConfigurationCaptureService:
    """Test configuration capture service functionality."""

    def test_service_initialization(self):
        """Test service initialization with default settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(
                storage_path=storage_path,
                auto_capture=True,
                enable_lineage_tracking=True,
                max_configurations=100,
            )

            assert service.storage_path == storage_path
            assert service.auto_capture is True
            assert service.enable_lineage_tracking is True
            assert service.max_configurations == 100
            assert service.configuration_cache == {}
            assert service.configuration_index == {}
            assert service.active_captures == {}
            assert service.capture_stats["total_captures"] == 0

    def test_service_initialization_defaults(self):
        """Test service initialization with default values."""
        service = ConfigurationCaptureService()

        assert service.storage_path == Path("data/configurations")
        assert service.auto_capture is True
        assert service.enable_lineage_tracking is True
        assert service.max_configurations == 10000
        assert service.capture_stats["total_captures"] == 0

    @pytest.mark.asyncio
    async def test_capture_configuration_basic(self):
        """Test basic configuration capture."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create capture request
            raw_parameters = {
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "random_state": 42,
                "dataset_path": "/path/to/dataset.csv",
                "dataset_name": "test_dataset",
            }

            request = ConfigurationCaptureRequestDTO(
                raw_parameters=raw_parameters,
                source=ConfigurationSource.MANUAL,
                source_context={"name": "test_config"},
                user_id="user123",
                tags=["test", "isolation"],
                generate_name=True,
                auto_save=True,
            )

            # Capture configuration
            response = await service.capture_configuration(request)

            assert response.success is True
            assert response.message == "Configuration captured successfully"
            assert response.configuration is not None
            assert response.configuration.algorithm_config.algorithm_name == "isolation_forest"
            assert response.configuration.algorithm_config.contamination == 0.1
            assert response.configuration.dataset_config.dataset_path == "/path/to/dataset.csv"
            assert response.configuration.metadata.source == ConfigurationSource.MANUAL
            assert "test" in response.configuration.metadata.tags
            assert response.configuration.metadata.created_by == "user123"
            assert response.configuration.is_valid is True

    @pytest.mark.asyncio
    async def test_capture_configuration_with_performance_results(self):
        """Test configuration capture with performance results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            raw_parameters = {
                "algorithm": "local_outlier_factor",
                "contamination": 0.05,
                "dataset_path": "/path/to/dataset.csv",
            }

            execution_results = {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.88,
                "f1_score": 0.90,
                "roc_auc": 0.94,
                "training_time": 15.5,
                "prediction_time": 2.3,
                "memory_usage": 512.0,
            }

            request = ConfigurationCaptureRequestDTO(
                raw_parameters=raw_parameters,
                source=ConfigurationSource.AUTOML,
                source_context={"experiment_id": "exp123"},
                execution_results=execution_results,
                user_id="user456",
                auto_save=True,
            )

            response = await service.capture_configuration(request)

            assert response.success is True
            assert response.configuration.performance_results is not None
            assert response.configuration.performance_results.accuracy == 0.95
            assert response.configuration.performance_results.precision == 0.92
            assert response.configuration.performance_results.training_time_seconds == 15.5
            assert response.configuration.performance_results.memory_usage_mb == 512.0

    @pytest.mark.asyncio
    async def test_capture_configuration_with_lineage(self):
        """Test configuration capture with lineage tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(
                storage_path=storage_path, enable_lineage_tracking=True
            )

            raw_parameters = {
                "algorithm": "one_class_svm",
                "contamination": 0.08,
                "dataset_path": "/path/to/dataset.csv",
            }

            source_context = {
                "parent_configs": [str(uuid4())],
                "derivation_method": "hyperparameter_tuning",
                "modifications": ["contamination_rate_optimized"],
                "git_commit": "abc123",
                "git_branch": "feature/optimization",
                "experiment_id": "exp456",
                "run_id": "run789",
            }

            request = ConfigurationCaptureRequestDTO(
                raw_parameters=raw_parameters,
                source=ConfigurationSource.OPTIMIZATION,
                source_context=source_context,
                user_id="user789",
                auto_save=True,
            )

            response = await service.capture_configuration(request)

            assert response.success is True
            assert response.configuration.lineage is not None
            assert len(response.configuration.lineage.parent_configurations) == 1
            assert response.configuration.lineage.derivation_method == "hyperparameter_tuning"
            assert response.configuration.lineage.git_commit == "abc123"
            assert response.configuration.lineage.experiment_id == "exp456"

    @pytest.mark.asyncio
    async def test_capture_configuration_with_preprocessing(self):
        """Test configuration capture with preprocessing settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            raw_parameters = {
                "algorithm": "elliptic_envelope",
                "contamination": 0.1,
                "dataset_path": "/path/to/dataset.csv",
                "missing_value_strategy": "median",
                "outlier_method": "isolation_forest",
                "scaling_method": "minmax",
                "feature_selection": "variance_threshold",
                "apply_pca": True,
                "pca_components": 10,
            }

            request = ConfigurationCaptureRequestDTO(
                raw_parameters=raw_parameters,
                source=ConfigurationSource.MANUAL,
                source_context={"name": "preprocessing_config"},
                user_id="user123",
                auto_save=True,
            )

            response = await service.capture_configuration(request)

            assert response.success is True
            assert response.configuration.preprocessing_config is not None
            assert response.configuration.preprocessing_config.missing_value_strategy == "median"
            assert response.configuration.preprocessing_config.outlier_detection_method == "isolation_forest"
            assert response.configuration.preprocessing_config.scaling_method == "minmax"
            assert response.configuration.preprocessing_config.feature_selection_method == "variance_threshold"
            assert response.configuration.preprocessing_config.apply_pca is True
            assert response.configuration.preprocessing_config.pca_components == 10

    @pytest.mark.asyncio
    async def test_capture_configuration_error_handling(self):
        """Test configuration capture error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create invalid request (missing required fields)
            raw_parameters = {}  # Empty parameters

            request = ConfigurationCaptureRequestDTO(
                raw_parameters=raw_parameters,
                source=ConfigurationSource.MANUAL,
                source_context={},
                user_id="user123",
                auto_save=True,
            )

            # Mock validation to raise an error
            with patch.object(service, 'validate_configuration') as mock_validate:
                mock_validate.side_effect = Exception("Validation error")

                response = await service.capture_configuration(request)

                assert response.success is False
                assert "Configuration capture failed" in response.message
                assert len(response.errors) > 0
                assert service.capture_stats["failed_captures"] == 1

    @pytest.mark.asyncio
    async def test_save_configuration(self):
        """Test saving configuration to storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="test_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                    dataset_name="test_dataset",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                    contamination=0.1,
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                    cv_folds=5,
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                    tags=["test"],
                ),
            )

            # Save configuration
            result = await service.save_configuration(config)

            assert result is True
            assert config.id in service.configuration_cache
            assert (storage_path / f"{config.id}.json").exists()

            # Verify file content
            with open(storage_path / f"{config.id}.json") as f:
                saved_data = json.load(f)

            assert saved_data["name"] == "test_config"
            assert saved_data["algorithm_config"]["algorithm_name"] == "isolation_forest"

    @pytest.mark.asyncio
    async def test_save_configuration_error_handling(self):
        """Test save configuration error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="test_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            # Mock file write to raise an error
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                result = await service.save_configuration(config)

                assert result is False
                assert config.id in service.configuration_cache  # Should still be in cache

    @pytest.mark.asyncio
    async def test_load_configuration(self):
        """Test loading configuration from storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create and save test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="test_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            await service.save_configuration(config)

            # Clear cache to test loading from disk
            service.configuration_cache.clear()

            # Load configuration
            loaded_config = await service.load_configuration(config.id)

            assert loaded_config is not None
            assert loaded_config.id == config.id
            assert loaded_config.name == "test_config"
            assert loaded_config.algorithm_config.algorithm_name == "isolation_forest"
            assert config.id in service.configuration_cache  # Should be cached

    @pytest.mark.asyncio
    async def test_load_configuration_from_cache(self):
        """Test loading configuration from cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="cached_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="local_outlier_factor",
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="f1_score",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.AUTOML,
                ),
            )

            # Add to cache manually
            service.configuration_cache[config.id] = config

            # Load configuration (should come from cache)
            loaded_config = await service.load_configuration(config.id)

            assert loaded_config is not None
            assert loaded_config.id == config.id
            assert loaded_config.name == "cached_config"

    @pytest.mark.asyncio
    async def test_load_configuration_not_found(self):
        """Test loading non-existent configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Try to load non-existent configuration
            non_existent_id = uuid4()
            loaded_config = await service.load_configuration(non_existent_id)

            assert loaded_config is None

    @pytest.mark.asyncio
    async def test_search_configurations_basic(self):
        """Test basic configuration search."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configurations
            configs = []
            for i in range(3):
                config = ExperimentConfigurationDTO(
                    id=uuid4(),
                    name=f"config_{i}",
                    dataset_config=DatasetConfigDTO(
                        dataset_path=f"/path/to/dataset_{i}.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="isolation_forest" if i % 2 == 0 else "local_outlier_factor",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="roc_auc",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.MANUAL,
                        tags=[f"tag_{i}"],
                    ),
                )
                configs.append(config)
                await service.save_configuration(config)

            # Search all configurations
            request = ConfigurationSearchRequestDTO(
                limit=10,
                offset=0,
                sort_by="name",
                sort_order="asc",
            )

            response = await service.search_configurations(request)

            assert response.success is True
            assert len(response.configurations) == 3
            assert response.total_count == 3
            assert response.configurations[0].name == "config_0"

    @pytest.mark.asyncio
    async def test_search_configurations_with_filters(self):
        """Test configuration search with filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configurations
            configs = []
            for i in range(5):
                config = ExperimentConfigurationDTO(
                    id=uuid4(),
                    name=f"config_{i}",
                    dataset_config=DatasetConfigDTO(
                        dataset_path=f"/path/to/dataset_{i}.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="isolation_forest" if i % 2 == 0 else "local_outlier_factor",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="roc_auc",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.MANUAL if i < 3 else ConfigurationSource.AUTOML,
                        tags=[f"tag_{i}", "common"] if i < 2 else [f"tag_{i}"],
                    ),
                )
                configs.append(config)
                await service.save_configuration(config)

            # Search with algorithm filter
            request = ConfigurationSearchRequestDTO(
                algorithm="isolation_forest",
                limit=10,
                offset=0,
            )

            response = await service.search_configurations(request)

            assert response.success is True
            assert len(response.configurations) == 3  # configs 0, 2, 4
            assert all(c.algorithm_config.algorithm_name == "isolation_forest" for c in response.configurations)

            # Search with source filter
            request = ConfigurationSearchRequestDTO(
                source=ConfigurationSource.AUTOML,
                limit=10,
                offset=0,
            )

            response = await service.search_configurations(request)

            assert response.success is True
            assert len(response.configurations) == 2  # configs 3, 4
            assert all(c.metadata.source == ConfigurationSource.AUTOML for c in response.configurations)

            # Search with tags filter
            request = ConfigurationSearchRequestDTO(
                tags=["common"],
                limit=10,
                offset=0,
            )

            response = await service.search_configurations(request)

            assert response.success is True
            assert len(response.configurations) == 2  # configs 0, 1
            assert all("common" in c.metadata.tags for c in response.configurations)

    @pytest.mark.asyncio
    async def test_search_configurations_with_query(self):
        """Test configuration search with text query."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configurations
            configs = [
                ExperimentConfigurationDTO(
                    id=uuid4(),
                    name="isolation_forest_config",
                    dataset_config=DatasetConfigDTO(
                        dataset_path="/path/to/dataset.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="isolation_forest",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="roc_auc",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.MANUAL,
                        description="Configuration for isolation forest algorithm",
                    ),
                ),
                ExperimentConfigurationDTO(
                    id=uuid4(),
                    name="lof_config",
                    dataset_config=DatasetConfigDTO(
                        dataset_path="/path/to/dataset.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="local_outlier_factor",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="f1_score",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.AUTOML,
                        description="Configuration for LOF algorithm",
                    ),
                ),
            ]

            for config in configs:
                await service.save_configuration(config)

            # Search by name
            request = ConfigurationSearchRequestDTO(
                query="isolation",
                limit=10,
                offset=0,
            )

            response = await service.search_configurations(request)

            assert response.success is True
            assert len(response.configurations) == 1
            assert response.configurations[0].name == "isolation_forest_config"

            # Search by description
            request = ConfigurationSearchRequestDTO(
                query="LOF",
                limit=10,
                offset=0,
            )

            response = await service.search_configurations(request)

            assert response.success is True
            assert len(response.configurations) == 1
            assert response.configurations[0].name == "lof_config"

    @pytest.mark.asyncio
    async def test_search_configurations_pagination(self):
        """Test configuration search with pagination."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configurations
            configs = []
            for i in range(10):
                config = ExperimentConfigurationDTO(
                    id=uuid4(),
                    name=f"config_{i:02d}",
                    dataset_config=DatasetConfigDTO(
                        dataset_path=f"/path/to/dataset_{i}.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="isolation_forest",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="roc_auc",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.MANUAL,
                    ),
                )
                configs.append(config)
                await service.save_configuration(config)

            # First page
            request = ConfigurationSearchRequestDTO(
                limit=3,
                offset=0,
                sort_by="name",
                sort_order="asc",
            )

            response = await service.search_configurations(request)

            assert response.success is True
            assert len(response.configurations) == 3
            assert response.total_count == 10
            assert response.configurations[0].name == "config_00"
            assert response.configurations[2].name == "config_02"

            # Second page
            request = ConfigurationSearchRequestDTO(
                limit=3,
                offset=3,
                sort_by="name",
                sort_order="asc",
            )

            response = await service.search_configurations(request)

            assert response.success is True
            assert len(response.configurations) == 3
            assert response.total_count == 10
            assert response.configurations[0].name == "config_03"
            assert response.configurations[2].name == "config_05"

    @pytest.mark.asyncio
    async def test_search_configurations_error_handling(self):
        """Test search configurations error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Mock glob to raise an error
            with patch.object(storage_path, 'glob', side_effect=Exception("File system error")):
                request = ConfigurationSearchRequestDTO(
                    limit=10,
                    offset=0,
                )

                response = await service.search_configurations(request)

                assert response.success is False
                assert "Search failed" in response.message
                assert len(response.errors) > 0

    @pytest.mark.asyncio
    async def test_export_configurations_json(self):
        """Test exporting configurations to JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="test_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            await service.save_configuration(config)

            # Export to JSON
            request = ConfigurationExportRequestDTO(
                configuration_ids=[config.id],
                export_format=ExportFormat.JSON,
                include_metadata=True,
                include_performance=False,
                include_lineage=False,
            )

            response = await service.export_configurations(request)

            assert response.success is True
            assert response.export_data is not None
            assert "test_config" in response.export_data
            assert "isolation_forest" in response.export_data

            # Parse JSON to verify structure
            exported_configs = json.loads(response.export_data)
            assert len(exported_configs) == 1
            assert exported_configs[0]["name"] == "test_config"
            assert "metadata" in exported_configs[0]
            assert "performance_results" not in exported_configs[0]

    @pytest.mark.asyncio
    async def test_export_configurations_yaml(self):
        """Test exporting configurations to YAML format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="yaml_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="local_outlier_factor",
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="f1_score",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.AUTOML,
                ),
            )

            await service.save_configuration(config)

            # Export to YAML
            request = ConfigurationExportRequestDTO(
                configuration_ids=[config.id],
                export_format=ExportFormat.YAML,
                include_metadata=True,
                include_performance=True,
                include_lineage=True,
            )

            response = await service.export_configurations(request)

            assert response.success is True
            assert response.export_data is not None
            assert "yaml_config" in response.export_data
            assert "local_outlier_factor" in response.export_data

    @pytest.mark.asyncio
    async def test_export_configurations_python(self):
        """Test exporting configurations to Python script format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="python_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="one_class_svm",
                    contamination=0.05,
                    random_state=42,
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            await service.save_configuration(config)

            # Export to Python
            request = ConfigurationExportRequestDTO(
                configuration_ids=[config.id],
                export_format=ExportFormat.PYTHON,
            )

            response = await service.export_configurations(request)

            assert response.success is True
            assert response.export_data is not None
            assert "#!/usr/bin/env python3" in response.export_data
            assert "import pynomaly" in response.export_data
            assert "one_class_svm" in response.export_data
            assert "contamination': 0.05" in response.export_data
            assert "random_state': 42" in response.export_data

    @pytest.mark.asyncio
    async def test_export_configurations_notebook(self):
        """Test exporting configurations to Jupyter notebook format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="notebook_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="elliptic_envelope",
                    contamination=0.1,
                    random_state=123,
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="precision",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.OPTIMIZATION,
                ),
            )

            await service.save_configuration(config)

            # Export to notebook
            request = ConfigurationExportRequestDTO(
                configuration_ids=[config.id],
                export_format=ExportFormat.NOTEBOOK,
            )

            response = await service.export_configurations(request)

            assert response.success is True
            assert response.export_data is not None

            # Parse notebook JSON
            notebook = json.loads(response.export_data)
            assert "cells" in notebook
            assert notebook["nbformat"] == 4
            assert len(notebook["cells"]) >= 3  # Header, imports, config cell

            # Check for configuration content
            notebook_content = response.export_data
            assert "elliptic_envelope" in notebook_content
            assert "contamination = 0.1" in notebook_content
            assert "random_state = 123" in notebook_content

    @pytest.mark.asyncio
    async def test_export_configurations_docker(self):
        """Test exporting configurations to Docker Compose format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="docker_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/data/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                    contamination=0.1,
                    random_state=42,
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            await service.save_configuration(config)

            # Export to Docker Compose
            request = ConfigurationExportRequestDTO(
                configuration_ids=[config.id],
                export_format=ExportFormat.DOCKER,
            )

            response = await service.export_configurations(request)

            assert response.success is True
            assert response.export_data is not None
            assert "version: '3.8'" in response.export_data
            assert "services:" in response.export_data
            assert "pynomaly-config-1:" in response.export_data
            assert "image: pynomaly:latest" in response.export_data
            assert "DATASET_PATH: /data/dataset.csv" in response.export_data
            assert "ALGORITHM: isolation_forest" in response.export_data
            assert "CONTAMINATION: '0.1'" in response.export_data

    @pytest.mark.asyncio
    async def test_export_configurations_to_file(self):
        """Test exporting configurations to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="file_export_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            await service.save_configuration(config)

            # Export to file
            output_path = Path(temp_dir) / "export.json"
            request = ConfigurationExportRequestDTO(
                configuration_ids=[config.id],
                export_format=ExportFormat.JSON,
                output_path=str(output_path),
            )

            response = await service.export_configurations(request)

            assert response.success is True
            assert response.export_data is not None
            assert len(response.export_files) == 1
            assert response.export_files[0] == str(output_path)
            assert output_path.exists()

            # Verify file content
            with open(output_path) as f:
                content = f.read()
            assert "file_export_config" in content

    @pytest.mark.asyncio
    async def test_export_configurations_empty_list(self):
        """Test exporting empty configuration list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Try to export non-existent configurations
            request = ConfigurationExportRequestDTO(
                configuration_ids=[uuid4(), uuid4()],
                export_format=ExportFormat.JSON,
            )

            response = await service.export_configurations(request)

            assert response.success is False
            assert "No configurations found to export" in response.message
            assert len(response.errors) > 0

    @pytest.mark.asyncio
    async def test_export_configurations_unsupported_format(self):
        """Test exporting configurations with unsupported format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="test_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            await service.save_configuration(config)

            # Mock an unsupported export format
            request = ConfigurationExportRequestDTO(
                configuration_ids=[config.id],
                export_format="UNSUPPORTED",  # This will cause an error
            )

            response = await service.export_configurations(request)

            assert response.success is False
            assert "Export failed" in response.message
            assert len(response.errors) > 0

    @pytest.mark.asyncio
    async def test_validate_configuration_valid(self):
        """Test configuration validation for valid configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create valid configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="valid_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                    dataset_name="test_dataset",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                    contamination=0.1,
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            # Validate configuration
            result = await service.validate_configuration(config)

            assert result.is_valid is True
            assert result.validation_score == 1.0
            assert len(result.errors) == 0
            assert result.estimated_runtime is not None
            assert result.estimated_memory is not None

    @pytest.mark.asyncio
    async def test_validate_configuration_invalid(self):
        """Test configuration validation for invalid configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create invalid configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="invalid_config",
                dataset_config=DatasetConfigDTO(
                    # Missing dataset_path and dataset_name
                ),
                algorithm_config=AlgorithmConfigDTO(
                    # Missing algorithm_name
                    contamination=1.5,  # Invalid contamination rate
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            # Validate configuration
            result = await service.validate_configuration(config)

            assert result.is_valid is False
            assert result.validation_score < 1.0
            assert len(result.errors) > 0
            assert "Dataset path or name must be specified" in result.errors
            assert "Algorithm name must be specified" in result.errors
            assert "Contamination rate must be between 0 and 0.5" in result.errors

    @pytest.mark.asyncio
    async def test_validate_configuration_warnings(self):
        """Test configuration validation with warnings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create configuration with warnings
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="warning_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                    contamination=0.1,
                ),
                preprocessing_config=PreprocessingConfigDTO(
                    apply_pca=True,
                    # Missing pca_components
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            # Validate configuration
            result = await service.validate_configuration(config)

            assert result.is_valid is True
            assert result.validation_score < 1.0
            assert len(result.errors) == 0
            assert len(result.warnings) > 0
            assert "PCA enabled but number of components not specified" in result.warnings

    @pytest.mark.asyncio
    async def test_get_configuration_statistics(self):
        """Test getting configuration statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configurations
            configs = [
                ExperimentConfigurationDTO(
                    id=uuid4(),
                    name="config_1",
                    dataset_config=DatasetConfigDTO(
                        dataset_path="/path/to/dataset.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="isolation_forest",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="roc_auc",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.MANUAL,
                    ),
                ),
                ExperimentConfigurationDTO(
                    id=uuid4(),
                    name="config_2",
                    dataset_config=DatasetConfigDTO(
                        dataset_path="/path/to/dataset.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="local_outlier_factor",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="f1_score",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.AUTOML,
                    ),
                ),
                ExperimentConfigurationDTO(
                    id=uuid4(),
                    name="config_3",
                    dataset_config=DatasetConfigDTO(
                        dataset_path="/path/to/dataset.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="isolation_forest",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="roc_auc",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.MANUAL,
                    ),
                ),
            ]

            for config in configs:
                await service.save_configuration(config)

            # Update capture statistics
            service.capture_stats["manual_captures"] = 2
            service.capture_stats["auto_captures"] = 1
            service.capture_stats["successful_captures"] = 3
            service.capture_stats["total_captures"] = 3

            # Get statistics
            stats = await service.get_configuration_statistics()

            assert stats["total_configurations"] == 3
            assert stats["cache_size"] == 3
            assert stats["storage_path"] == str(storage_path)
            assert stats["auto_capture_enabled"] is True
            assert stats["lineage_tracking_enabled"] is True
            assert stats["capture_statistics"]["manual_captures"] == 2
            assert stats["capture_statistics"]["auto_captures"] == 1
            assert stats["capture_statistics"]["successful_captures"] == 3
            assert stats["configurations_by_source"]["MANUAL"] == 2
            assert stats["configurations_by_source"]["AUTOML"] == 1
            assert stats["configurations_by_algorithm"]["isolation_forest"] == 2
            assert stats["configurations_by_algorithm"]["local_outlier_factor"] == 1

    @pytest.mark.asyncio
    async def test_cache_size_management(self):
        """Test cache size management and cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(
                storage_path=storage_path,
                max_configurations=5,
            )

            # Create more configurations than the limit
            configs = []
            for i in range(8):
                config = ExperimentConfigurationDTO(
                    id=uuid4(),
                    name=f"config_{i}",
                    dataset_config=DatasetConfigDTO(
                        dataset_path=f"/path/to/dataset_{i}.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="isolation_forest",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="roc_auc",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.MANUAL,
                    ),
                )
                configs.append(config)
                await service.save_configuration(config)

                # Add small delay to ensure different timestamps
                await asyncio.sleep(0.01)

            # Check that cache size is limited
            assert len(service.configuration_cache) <= service.max_configurations

            # Check that files were cleaned up
            json_files = list(storage_path.glob("*.json"))
            assert len(json_files) <= service.max_configurations

    @pytest.mark.asyncio
    async def test_configuration_indexing(self):
        """Test configuration indexing for fast search."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create test configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="indexed_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                    tags=["test", "experiment"],
                ),
            )

            await service.save_configuration(config)

            # Check indexes
            assert "isolation_forest" in service.configuration_index
            assert config.id in service.configuration_index["isolation_forest"]
            assert "test" in service.configuration_index
            assert config.id in service.configuration_index["test"]
            assert "experiment" in service.configuration_index
            assert config.id in service.configuration_index["experiment"]
            assert "MANUAL" in service.configuration_index
            assert config.id in service.configuration_index["MANUAL"]

    @pytest.mark.asyncio
    async def test_concurrent_configuration_operations(self):
        """Test concurrent configuration operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create multiple configurations concurrently
            async def create_config(index):
                config = ExperimentConfigurationDTO(
                    id=uuid4(),
                    name=f"concurrent_config_{index}",
                    dataset_config=DatasetConfigDTO(
                        dataset_path=f"/path/to/dataset_{index}.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="isolation_forest",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="roc_auc",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.MANUAL,
                    ),
                )
                return await service.save_configuration(config)

            # Run concurrent operations
            tasks = [create_config(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

            # All operations should succeed
            assert all(result for result in results)
            assert len(service.configuration_cache) == 10

    @pytest.mark.asyncio
    async def test_configuration_service_performance(self):
        """Test configuration service performance under load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Measure performance for creating many configurations
            start_time = time.time()

            configs = []
            for i in range(100):
                config = ExperimentConfigurationDTO(
                    id=uuid4(),
                    name=f"perf_config_{i}",
                    dataset_config=DatasetConfigDTO(
                        dataset_path=f"/path/to/dataset_{i}.csv",
                    ),
                    algorithm_config=AlgorithmConfigDTO(
                        algorithm_name="isolation_forest",
                    ),
                    evaluation_config=EvaluationConfigDTO(
                        primary_metric="roc_auc",
                    ),
                    metadata=ConfigurationMetadataDTO(
                        source=ConfigurationSource.MANUAL,
                    ),
                )
                configs.append(config)
                await service.save_configuration(config)

            end_time = time.time()
            duration = end_time - start_time

            # Should complete within reasonable time (less than 5 seconds)
            assert duration < 5.0

            # Test search performance
            start_time = time.time()

            request = ConfigurationSearchRequestDTO(
                limit=50,
                offset=0,
                sort_by="name",
                sort_order="asc",
            )

            response = await service.search_configurations(request)

            end_time = time.time()
            search_duration = end_time - start_time

            # Search should be fast (less than 1 second)
            assert search_duration < 1.0
            assert response.success is True
            assert len(response.configurations) == 50

    @pytest.mark.asyncio
    async def test_configuration_error_recovery(self):
        """Test configuration service error recovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create valid configuration
            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="error_recovery_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            await service.save_configuration(config)

            # Simulate file system error during search
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                request = ConfigurationSearchRequestDTO(
                    limit=10,
                    offset=0,
                )

                response = await service.search_configurations(request)

                assert response.success is False
                assert "Search failed" in response.message

            # Service should still work after error
            stats = await service.get_configuration_statistics()
            assert stats["total_configurations"] == 1

    @pytest.mark.asyncio
    async def test_configuration_memory_management(self):
        """Test configuration memory management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            service = ConfigurationCaptureService(storage_path=storage_path)

            # Create configuration with large data
            large_hyperparameters = {f"param_{i}": f"value_{i}" for i in range(1000)}

            config = ExperimentConfigurationDTO(
                id=uuid4(),
                name="large_config",
                dataset_config=DatasetConfigDTO(
                    dataset_path="/path/to/dataset.csv",
                ),
                algorithm_config=AlgorithmConfigDTO(
                    algorithm_name="isolation_forest",
                    hyperparameters=large_hyperparameters,
                ),
                evaluation_config=EvaluationConfigDTO(
                    primary_metric="roc_auc",
                ),
                metadata=ConfigurationMetadataDTO(
                    source=ConfigurationSource.MANUAL,
                ),
            )

            await service.save_configuration(config)

            # Load configuration multiple times
            for _ in range(10):
                loaded_config = await service.load_configuration(config.id)
                assert loaded_config is not None
                assert loaded_config.name == "large_config"

            # Service should handle large configurations gracefully
            stats = await service.get_configuration_statistics()
            assert stats["total_configurations"] == 1
