"""Comprehensive tests for configuration management system."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from pynomaly.application.dto.configuration_dto import (
    AlgorithmConfigDTO,
    ConfigurationCaptureRequestDTO,
    ConfigurationExportRequestDTO,
    ConfigurationMetadataDTO,
    ConfigurationSearchRequestDTO,
    ConfigurationSource,
    ConfigurationStatus,
    DatasetConfigDTO,
    EvaluationConfigDTO,
    ExperimentConfigurationDTO,
    ExportFormat,
    create_basic_configuration,
)
from pynomaly.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)
from pynomaly.infrastructure.persistence.configuration_repository import (
    ConfigurationRepository,
)


class TestConfigurationDTOs:
    """Test configuration data transfer objects."""

    def test_create_basic_configuration(self):
        """Test basic configuration creation."""
        config = create_basic_configuration(
            name="test_config",
            dataset_path="/path/to/data.csv",
            algorithm_name="isolation_forest",
            source=ConfigurationSource.MANUAL,
        )

        assert config.name == "test_config"
        assert config.dataset_config.dataset_path == "/path/to/data.csv"
        assert config.algorithm_config.algorithm_name == "isolation_forest"
        assert config.metadata.source == ConfigurationSource.MANUAL
        assert config.status == ConfigurationStatus.DRAFT
        assert config.is_valid is True

    def test_experiment_configuration_validation(self):
        """Test experiment configuration validation."""
        config = ExperimentConfigurationDTO(
            name="test_experiment",
            dataset_config=DatasetConfigDTO(dataset_path="/test/data.csv"),
            algorithm_config=AlgorithmConfigDTO(algorithm_name="isolation_forest"),
            evaluation_config=EvaluationConfigDTO(),
            metadata=ConfigurationMetadataDTO(source=ConfigurationSource.AUTOML),
        )

        assert config.id is not None
        assert isinstance(config.id, type(uuid4()))
        assert config.metadata.created_at is not None
        assert config.execution_count == 0

    def test_configuration_capture_request_dto(self):
        """Test configuration capture request DTO."""
        request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource.CLI,
            raw_parameters={
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "dataset_path": "/test/data.csv",
            },
            source_context={"cli_command": "detect"},
            auto_save=True,
            generate_name=True,
            tags=["test", "cli"],
        )

        assert request.source == ConfigurationSource.CLI
        assert request.raw_parameters["algorithm"] == "isolation_forest"
        assert request.auto_save is True
        assert "test" in request.tags

    def test_configuration_export_request_dto(self):
        """Test configuration export request DTO."""
        config_ids = [uuid4(), uuid4()]
        request = ConfigurationExportRequestDTO(
            configuration_ids=config_ids,
            export_format=ExportFormat.YAML,
            include_metadata=True,
            include_performance=False,
            output_path="/tmp/export.yaml",
        )

        assert len(request.configuration_ids) == 2
        assert request.export_format == ExportFormat.YAML
        assert request.include_metadata is True
        assert request.include_performance is False

    def test_configuration_search_request_dto(self):
        """Test configuration search request DTO."""
        request = ConfigurationSearchRequestDTO(
            query="isolation_forest",
            tags=["experiment"],
            source=ConfigurationSource.AUTOML,
            algorithm="isolation_forest",
            min_accuracy=0.8,
            limit=25,
            sort_by="accuracy",
            sort_order="desc",
        )

        assert request.query == "isolation_forest"
        assert request.tags == ["experiment"]
        assert request.source == ConfigurationSource.AUTOML
        assert request.min_accuracy == 0.8
        assert request.limit == 25


class TestConfigurationCaptureService:
    """Test configuration capture service."""

    @pytest.fixture
    def capture_service(self):
        """Create configuration capture service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationCaptureService(
                storage_path=Path(temp_dir),
                auto_capture=True,
                enable_lineage_tracking=True,
            )

    @pytest.mark.asyncio
    async def test_capture_configuration_basic(self, capture_service):
        """Test basic configuration capture."""
        request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource.CLI,
            raw_parameters={
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "dataset_path": "/test/data.csv",
                "random_state": 42,
            },
            source_context={"name": "test_config"},
            auto_save=False,
            generate_name=True,
        )

        response = await capture_service.capture_configuration(request)

        assert response.success is True
        assert response.configuration is not None
        assert response.configuration.name == "test_config"
        assert (
            response.configuration.algorithm_config.algorithm_name == "isolation_forest"
        )
        assert response.configuration.algorithm_config.contamination == 0.1
        assert response.configuration.is_valid is True

    @pytest.mark.asyncio
    async def test_capture_configuration_with_performance(self, capture_service):
        """Test configuration capture with performance results."""
        request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource.AUTOML,
            raw_parameters={
                "algorithm": "local_outlier_factor",
                "contamination": 0.05,
                "dataset_path": "/test/data.csv",
            },
            execution_results={
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
                "training_time": 120.5,
            },
            auto_save=False,
        )

        response = await capture_service.capture_configuration(request)

        assert response.success is True
        config = response.configuration
        assert config.performance_results is not None
        assert config.performance_results.accuracy == 0.85
        assert config.performance_results.precision == 0.82
        assert config.performance_results.training_time_seconds == 120.5

    @pytest.mark.asyncio
    async def test_capture_configuration_with_preprocessing(self, capture_service):
        """Test configuration capture with preprocessing parameters."""
        request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource.WEB_API,
            raw_parameters={
                "algorithm": "one_class_svm",
                "contamination": 0.1,
                "dataset_path": "/test/data.csv",
                "preprocess_missing": "mean",
                "scaling_method": "standard",
                "apply_pca": True,
                "pca_components": 10,
            },
            auto_save=False,
        )

        response = await capture_service.capture_configuration(request)

        assert response.success is True
        config = response.configuration
        assert config.preprocessing_config is not None
        assert config.preprocessing_config.missing_value_strategy == "mean"
        assert config.preprocessing_config.scaling_method == "standard"
        assert config.preprocessing_config.apply_pca is True
        assert config.preprocessing_config.pca_components == 10

    @pytest.mark.asyncio
    async def test_configuration_validation(self, capture_service):
        """Test configuration validation."""
        # Valid configuration
        request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource.MANUAL,
            raw_parameters={
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "dataset_path": "/test/data.csv",
            },
            auto_save=False,
        )

        response = await capture_service.capture_configuration(request)
        config = response.configuration

        validation_result = await capture_service.validate_configuration(config)
        assert validation_result.is_valid is True
        assert validation_result.validation_score > 0.8
        assert len(validation_result.errors) == 0

    @pytest.mark.asyncio
    async def test_configuration_validation_errors(self, capture_service):
        """Test configuration validation with errors."""
        # Invalid configuration - no dataset path and invalid contamination
        request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource.MANUAL,
            raw_parameters={
                "algorithm": "",  # Empty algorithm
                "contamination": 0.8,  # Invalid contamination rate
            },
            auto_save=False,
        )

        response = await capture_service.capture_configuration(request)
        config = response.configuration

        validation_result = await capture_service.validate_configuration(config)
        assert validation_result.is_valid is False
        assert validation_result.validation_score < 0.5
        assert len(validation_result.errors) > 0

    @pytest.mark.asyncio
    async def test_export_configurations_json(self, capture_service):
        """Test exporting configurations to JSON."""
        # Create test configurations
        configs = []
        for i in range(3):
            request = ConfigurationCaptureRequestDTO(
                source=ConfigurationSource.TEST,
                raw_parameters={
                    "algorithm": f"algorithm_{i}",
                    "contamination": 0.1 + i * 0.05,
                    "dataset_path": f"/test/data_{i}.csv",
                },
                auto_save=True,
            )
            response = await capture_service.capture_configuration(request)
            configs.append(response.configuration)

        # Export configurations
        export_request = ConfigurationExportRequestDTO(
            configuration_ids=[config.id for config in configs],
            export_format=ExportFormat.JSON,
            include_metadata=True,
            include_performance=False,
        )

        export_response = await capture_service.export_configurations(export_request)

        assert export_response.success is True
        assert export_response.export_data is not None

        # Parse exported data
        exported_data = json.loads(export_response.export_data)
        assert len(exported_data) == 3

        for i, config_data in enumerate(exported_data):
            assert config_data["algorithm_config"]["algorithm_name"] == f"algorithm_{i}"
            assert "metadata" in config_data  # Metadata included

    @pytest.mark.asyncio
    async def test_export_configurations_yaml(self, capture_service):
        """Test exporting configurations to YAML."""
        # Create test configuration
        request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource.AUTOML,
            raw_parameters={
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "dataset_path": "/test/data.csv",
            },
            auto_save=True,
        )

        response = await capture_service.capture_configuration(request)
        config = response.configuration

        # Export to YAML
        export_request = ConfigurationExportRequestDTO(
            configuration_ids=[config.id],
            export_format=ExportFormat.YAML,
            include_metadata=False,
            include_performance=False,
        )

        export_response = await capture_service.export_configurations(export_request)

        assert export_response.success is True
        assert export_response.export_data is not None
        assert export_response.export_data.startswith("- algorithm_config:")

    @pytest.mark.asyncio
    async def test_export_configurations_python(self, capture_service):
        """Test exporting configurations to Python script."""
        request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource.CLI,
            raw_parameters={
                "algorithm": "local_outlier_factor",
                "contamination": 0.05,
                "dataset_path": "/data/test.csv",
            },
            auto_save=True,
        )

        response = await capture_service.capture_configuration(request)
        config = response.configuration

        # Export to Python
        export_request = ConfigurationExportRequestDTO(
            configuration_ids=[config.id], export_format=ExportFormat.PYTHON
        )

        export_response = await capture_service.export_configurations(export_request)

        assert export_response.success is True
        assert export_response.export_data is not None
        assert "#!/usr/bin/env python3" in export_response.export_data
        assert "import pynomaly" in export_response.export_data
        assert "local_outlier_factor" in export_response.export_data

    @pytest.mark.asyncio
    async def test_get_configuration_statistics(self, capture_service):
        """Test getting configuration statistics."""
        # Create some test configurations
        for i in range(5):
            request = ConfigurationCaptureRequestDTO(
                source=(
                    ConfigurationSource.AUTOML
                    if i % 2 == 0
                    else ConfigurationSource.CLI
                ),
                raw_parameters={
                    "algorithm": (
                        "isolation_forest" if i < 3 else "local_outlier_factor"
                    ),
                    "contamination": 0.1,
                    "dataset_path": f"/test/data_{i}.csv",
                },
                auto_save=True,
            )
            await capture_service.capture_configuration(request)

        stats = await capture_service.get_configuration_statistics()

        assert stats["total_configurations"] == 5
        assert stats["capture_statistics"]["total_captures"] == 5
        assert stats["capture_statistics"]["successful_captures"] == 5
        assert "configurations_by_source" in stats
        assert "configurations_by_algorithm" in stats
        assert stats["auto_capture_enabled"] is True


class TestConfigurationRepository:
    """Test configuration repository."""

    @pytest.fixture
    def repository(self):
        """Create configuration repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationRepository(
                storage_path=Path(temp_dir), enable_versioning=True, backup_enabled=True
            )

    @pytest.mark.asyncio
    async def test_save_and_load_configuration(self, repository):
        """Test saving and loading configuration."""
        config = create_basic_configuration(
            name="test_save_load",
            dataset_path="/test/data.csv",
            algorithm_name="isolation_forest",
        )

        # Save configuration
        success = await repository.save_configuration(config)
        assert success is True

        # Load configuration
        loaded_config = await repository.load_configuration(config.id)
        assert loaded_config is not None
        assert loaded_config.id == config.id
        assert loaded_config.name == config.name
        assert loaded_config.algorithm_config.algorithm_name == "isolation_forest"

    @pytest.mark.asyncio
    async def test_delete_configuration(self, repository):
        """Test deleting configuration."""
        config = create_basic_configuration(
            name="test_delete",
            dataset_path="/test/data.csv",
            algorithm_name="local_outlier_factor",
        )

        # Save and then delete
        await repository.save_configuration(config)
        success = await repository.delete_configuration(config.id)
        assert success is True

        # Verify deletion
        loaded_config = await repository.load_configuration(config.id)
        assert loaded_config is None

    @pytest.mark.asyncio
    async def test_search_configurations(self, repository):
        """Test searching configurations."""
        # Create test configurations
        configs = []
        for i in range(3):
            config = create_basic_configuration(
                name=f"search_test_{i}",
                dataset_path=f"/test/data_{i}.csv",
                algorithm_name="isolation_forest" if i < 2 else "local_outlier_factor",
            )
            config.metadata.tags = ["test", f"dataset_{i}"]
            configs.append(config)
            await repository.save_configuration(config)

        # Search by query
        search_request = ConfigurationSearchRequestDTO(query="search_test", limit=10)
        results = await repository.search_configurations(search_request)
        assert len(results) == 3

        # Search by algorithm
        search_request = ConfigurationSearchRequestDTO(
            algorithm="isolation_forest", limit=10
        )
        results = await repository.search_configurations(search_request)
        assert len(results) == 2

        # Search by tags
        search_request = ConfigurationSearchRequestDTO(tags=["dataset_1"], limit=10)
        results = await repository.search_configurations(search_request)
        assert len(results) == 1
        assert results[0].name == "search_test_1"

    @pytest.mark.asyncio
    async def test_list_configurations(self, repository):
        """Test listing configurations."""
        # Create test configurations with different sources
        for i in range(3):
            config = create_basic_configuration(
                name=f"list_test_{i}",
                dataset_path=f"/test/data_{i}.csv",
                algorithm_name="isolation_forest",
                source=ConfigurationSource.AUTOML if i < 2 else ConfigurationSource.CLI,
            )
            await repository.save_configuration(config)

        # List all configurations
        all_configs = await repository.list_configurations(limit=10)
        assert len(all_configs) == 3

        # List by source
        automl_configs = await repository.list_configurations(
            source=ConfigurationSource.AUTOML, limit=10
        )
        assert len(automl_configs) == 2

        cli_configs = await repository.list_configurations(
            source=ConfigurationSource.CLI, limit=10
        )
        assert len(cli_configs) == 1

    @pytest.mark.asyncio
    async def test_export_import_configurations(self, repository):
        """Test exporting and importing configurations."""
        # Create test configurations
        original_configs = []
        for i in range(2):
            config = create_basic_configuration(
                name=f"export_import_test_{i}",
                dataset_path=f"/test/data_{i}.csv",
                algorithm_name="isolation_forest",
            )
            original_configs.append(config)
            await repository.save_configuration(config)

        # Export configurations
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = Path(f.name)

        config_ids = [config.id for config in original_configs]
        success = await repository.export_configurations(
            config_ids, export_path, include_metadata=True
        )
        assert success is True
        assert export_path.exists()

        # Create new repository for import
        with tempfile.TemporaryDirectory() as temp_dir2:
            new_repository = ConfigurationRepository(Path(temp_dir2))

            # Import configurations
            imported_count = await new_repository.import_configurations(
                export_path, overwrite_existing=False
            )
            assert imported_count == 2

            # Verify imported configurations
            for original_config in original_configs:
                imported_config = await new_repository.load_configuration(
                    original_config.id
                )
                assert imported_config is not None
                assert imported_config.name == original_config.name

        # Cleanup
        export_path.unlink()

    def test_repository_statistics(self, repository):
        """Test repository statistics."""
        stats = repository.get_repository_statistics()

        assert "total_configurations" in stats
        assert "storage_size_bytes" in stats
        assert "storage_path" in stats
        assert "versioning_enabled" in stats
        assert "compression_enabled" in stats
        assert "backup_enabled" in stats

        assert stats["versioning_enabled"] is True
        assert stats["backup_enabled"] is True


class TestConfigurationIntegration:
    """Test integration between capture service and repository."""

    @pytest.fixture
    def integrated_service(self):
        """Create integrated configuration service and repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ConfigurationCaptureService(
                storage_path=Path(temp_dir), auto_capture=True
            )
            repository = ConfigurationRepository(Path(temp_dir))
            yield service, repository

    @pytest.mark.asyncio
    async def test_capture_and_retrieve_workflow(self, integrated_service):
        """Test complete capture and retrieve workflow."""
        service, repository = integrated_service

        # Capture configuration
        request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource.AUTOML,
            raw_parameters={
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "dataset_path": "/test/data.csv",
                "random_state": 42,
            },
            execution_results={
                "accuracy": 0.92,
                "precision": 0.88,
                "recall": 0.94,
                "training_time": 45.2,
            },
            auto_save=True,
            tags=["integration_test", "automl"],
        )

        response = await service.capture_configuration(request)
        assert response.success is True
        config_id = response.configuration.id

        # Retrieve from repository
        stored_config = await repository.load_configuration(config_id)
        assert stored_config is not None
        assert stored_config.metadata.source == ConfigurationSource.AUTOML
        assert stored_config.performance_results.accuracy == 0.92
        assert "integration_test" in stored_config.metadata.tags

    @pytest.mark.asyncio
    async def test_search_and_export_workflow(self, integrated_service):
        """Test search and export workflow."""
        service, repository = integrated_service

        # Create multiple configurations
        algorithms = ["isolation_forest", "local_outlier_factor", "one_class_svm"]
        config_ids = []

        for i, algorithm in enumerate(algorithms):
            request = ConfigurationCaptureRequestDTO(
                source=ConfigurationSource.EXPERIMENT,
                raw_parameters={
                    "algorithm": algorithm,
                    "contamination": 0.1,
                    "dataset_path": f"/test/data_{i}.csv",
                },
                execution_results={
                    "accuracy": 0.8 + i * 0.05,
                    "precision": 0.75 + i * 0.05,
                },
                auto_save=True,
                tags=["workflow_test", algorithm],
            )

            response = await service.capture_configuration(request)
            config_ids.append(response.configuration.id)

        # Search configurations
        search_request = ConfigurationSearchRequestDTO(
            query="workflow_test",
            min_accuracy=0.82,
            sort_by="accuracy",
            sort_order="desc",
        )

        search_results = await repository.search_configurations(search_request)
        assert len(search_results) == 2  # Only configs with accuracy >= 0.82
        assert (
            search_results[0].performance_results.accuracy
            >= search_results[1].performance_results.accuracy
        )

        # Export search results
        export_request = ConfigurationExportRequestDTO(
            configuration_ids=[config.id for config in search_results],
            export_format=ExportFormat.JSON,
            include_performance=True,
        )

        export_response = await service.export_configurations(export_request)
        assert export_response.success is True

        # Verify export contains performance data
        exported_data = json.loads(export_response.export_data)
        assert len(exported_data) == 2
        for config_data in exported_data:
            assert "performance_results" in config_data
            assert config_data["performance_results"]["accuracy"] >= 0.82


if __name__ == "__main__":
    pytest.main([__file__])
