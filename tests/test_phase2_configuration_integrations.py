"""Tests for Phase 2 configuration integrations (AutoML and Autonomous mode)."""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from pynomaly.application.dto.configuration_dto import (
    ConfigurationSource,
)
from pynomaly.application.services.automl_configuration_integration import (
    AutoMLConfigurationIntegration,
    AutoMLConfigurationManager,
)
from pynomaly.application.services.autonomous_configuration_integration import (
    AutonomousConfigurationIntegration,
    AutonomousConfigurationManager,
)
from pynomaly.application.services.autonomous_service import AutonomousConfig
from pynomaly.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)
from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.infrastructure.monitoring.cli_parameter_interceptor import (
    CLIParameterInterceptor,
    capture_detection_command,
    initialize_cli_interceptor,
)


class TestAutoMLConfigurationIntegration:
    """Test AutoML configuration integration."""

    @pytest.fixture
    def mock_automl_service(self):
        """Create mock AutoML service."""
        service = Mock()
        service.optimize = AsyncMock()
        service.algorithm_configs = {
            "isolation_forest": {"complexity": 0.5},
            "local_outlier_factor": {"complexity": 0.7},
        }
        service.optimization_history = []
        return service

    @pytest.fixture
    def capture_service(self):
        """Create configuration capture service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationCaptureService(
                storage_path=Path(temp_dir), auto_capture=True
            )

    @pytest.fixture
    def integration_service(self, mock_automl_service, capture_service):
        """Create AutoML configuration integration service."""
        return AutoMLConfigurationIntegration(
            automl_service=mock_automl_service,
            configuration_service=capture_service,
            auto_save_successful=True,
            auto_save_failed=False,
        )

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(1, 2, 100),
                "feature3": np.random.uniform(0, 1, 100),
            }
        )
        return Dataset(
            name="test_dataset",
            data=data,
            feature_names=["feature1", "feature2", "feature3"],
        )

    @pytest.fixture
    def mock_detector(self):
        """Create mock detector."""
        detector = Mock(spec=Detector)
        detector.algorithm_name = "isolation_forest"
        detector.contamination = 0.1
        detector.random_state = 42
        detector.params = {"n_estimators": 100, "contamination": 0.1}
        return detector

    @pytest.mark.asyncio
    async def test_optimize_with_configuration_capture_success(
        self, integration_service, sample_dataset, mock_detector
    ):
        """Test successful optimization with configuration capture."""
        # Setup mock optimization result
        optimization_report = {
            "best_metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
            },
            "resource_usage": {"peak_memory_mb": 256, "cpu_usage_percent": 75},
            "n_trials": 50,
            "optimization_time": 120.5,
        }

        integration_service.automl_service.optimize.return_value = (
            mock_detector,
            optimization_report,
        )

        # Run optimization with capture
        (
            detector,
            report,
            config_id,
        ) = await integration_service.optimize_with_configuration_capture(
            dataset=sample_dataset,
            algorithm_name="isolation_forest",
            objectives=None,
            constraints=None,
        )

        # Verify results
        assert detector == mock_detector
        assert report == optimization_report
        assert config_id is not None
        assert integration_service.integration_stats["total_optimizations"] == 1
        assert integration_service.integration_stats["successful_captures"] == 1
        assert integration_service.integration_stats["configurations_saved"] == 1

    @pytest.mark.asyncio
    async def test_optimize_with_configuration_capture_failure(
        self, integration_service, sample_dataset
    ):
        """Test optimization failure handling."""
        # Setup mock to raise exception
        integration_service.automl_service.optimize.side_effect = RuntimeError(
            "Optimization failed"
        )
        integration_service.auto_save_failed = True

        # Run optimization and expect failure
        with pytest.raises(RuntimeError, match="Optimization failed"):
            await integration_service.optimize_with_configuration_capture(
                dataset=sample_dataset, algorithm_name="isolation_forest"
            )

        # Verify statistics
        assert integration_service.integration_stats["total_optimizations"] == 1
        assert integration_service.integration_stats["failed_captures"] == 1

    @pytest.mark.asyncio
    async def test_capture_manual_configuration(
        self, integration_service, sample_dataset, mock_detector
    ):
        """Test manual configuration capture."""
        performance_results = {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92,
            "training_time": 45.2,
        }

        metadata = {"experiment_name": "manual_test", "user": "test_user"}

        # Capture manual configuration
        config_id = await integration_service.capture_manual_configuration(
            dataset=sample_dataset,
            detector=mock_detector,
            performance_results=performance_results,
            metadata=metadata,
        )

        # Verify capture
        assert config_id is not None
        assert integration_service.integration_stats["successful_captures"] == 1

    @pytest.mark.asyncio
    async def test_batch_capture_configurations(
        self, integration_service, sample_dataset, mock_detector
    ):
        """Test batch configuration capture."""
        # Create batch results
        optimization_results = [
            {
                "dataset": sample_dataset,
                "detector": mock_detector,
                "performance": {"accuracy": 0.85, "precision": 0.82},
            },
            {
                "dataset": sample_dataset,
                "detector": mock_detector,
                "performance": {"accuracy": 0.87, "precision": 0.84},
            },
        ]

        batch_metadata = {"experiment_type": "batch_test", "researcher": "test_user"}

        # Capture batch configurations
        config_ids = await integration_service.batch_capture_configurations(
            optimization_results=optimization_results, batch_metadata=batch_metadata
        )

        # Verify results
        assert len(config_ids) == 2
        assert all(config_id is not None for config_id in config_ids)
        assert integration_service.integration_stats["successful_captures"] == 2

    def test_get_integration_statistics(self, integration_service):
        """Test integration statistics retrieval."""
        stats = integration_service.get_integration_statistics()

        assert "integration_stats" in stats
        assert "configuration_service_stats" in stats
        assert "automl_service_info" in stats
        assert stats["automl_service_info"]["available_algorithms"] == 2


class TestAutonomousConfigurationIntegration:
    """Test autonomous mode configuration integration."""

    @pytest.fixture
    def mock_autonomous_service(self):
        """Create mock autonomous service."""
        service = Mock()
        service.detect_anomalies = AsyncMock()
        service.algorithm_adapters = {"isolation_forest": Mock(), "lof": Mock()}
        service.data_loaders = {"csv": Mock(), "parquet": Mock()}
        service.preprocessing_orchestrator = Mock()
        return service

    @pytest.fixture
    def capture_service(self):
        """Create configuration capture service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationCaptureService(
                storage_path=Path(temp_dir), auto_capture=True
            )

    @pytest.fixture
    def integration_service(self, mock_autonomous_service, capture_service):
        """Create autonomous configuration integration service."""
        return AutonomousConfigurationIntegration(
            autonomous_service=mock_autonomous_service,
            configuration_service=capture_service,
            auto_save_successful=True,
            auto_save_threshold=0.7,
        )

    @pytest.fixture
    def sample_detection_result(self):
        """Create sample detection result."""
        scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.8),
            AnomalyScore(0.2),
            AnomalyScore(0.9),
        ]
        return DetectionResult(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            scores=np.array([s.value for s in scores]),
            anomalies=np.array([1, 3]),  # Indices of anomalies
            timestamp=datetime.now(),
            metadata={"algorithm": "isolation_forest"},
        )

    @pytest.mark.asyncio
    async def test_detect_with_configuration_capture_success(
        self, integration_service, sample_detection_result
    ):
        """Test successful autonomous detection with configuration capture."""
        # Setup mock detection result
        integration_service.autonomous_service.detect_anomalies.return_value = (
            sample_detection_result
        )

        # Create test data source
        data_source = pd.DataFrame(
            {"x": np.random.normal(0, 1, 100), "y": np.random.normal(0, 1, 100)}
        )

        config = AutonomousConfig(
            max_algorithms=3, confidence_threshold=0.8, enable_preprocessing=True
        )

        # Run detection with capture
        result, config_id = await integration_service.detect_with_configuration_capture(
            data_source=data_source, config=config, capture_metadata={"test_run": True}
        )

        # Verify results
        assert result == sample_detection_result
        assert config_id is not None
        assert integration_service.integration_stats["total_autonomous_runs"] == 1
        assert integration_service.integration_stats["configurations_saved"] == 1

    @pytest.mark.asyncio
    async def test_capture_autonomous_experiment(
        self, integration_service, sample_detection_result
    ):
        """Test autonomous experiment configuration capture."""
        # Setup mock detection results
        integration_service.autonomous_service.detect_anomalies.return_value = (
            sample_detection_result
        )

        # Create test experiment
        data_sources = [
            pd.DataFrame({"x": np.random.normal(0, 1, 50)}),
            pd.DataFrame({"y": np.random.normal(1, 1, 50)}),
        ]

        configs = [
            AutonomousConfig(max_algorithms=2, confidence_threshold=0.7),
            AutonomousConfig(max_algorithms=3, confidence_threshold=0.8),
        ]

        experiment_metadata = {
            "researcher": "test_user",
            "experiment_type": "comparative",
        }

        # Run experiment
        results = await integration_service.capture_autonomous_experiment(
            experiment_name="test_experiment",
            data_sources=data_sources,
            configs=configs,
            experiment_metadata=experiment_metadata,
        )

        # Verify results
        assert len(results) == 4  # 2 datasets × 2 configs
        assert all(result[0] is not None for result in results)  # All successful
        assert integration_service.integration_stats["total_autonomous_runs"] == 4

    @pytest.mark.asyncio
    async def test_analyze_autonomous_configurations(
        self, integration_service, capture_service
    ):
        """Test autonomous configuration analysis."""
        # Create some test configurations first
        from pynomaly.application.dto.configuration_dto import (
            ConfigurationCaptureRequestDTO,
        )

        for i in range(3):
            request = ConfigurationCaptureRequestDTO(
                source=ConfigurationSource.AUTONOMOUS,
                raw_parameters={
                    "algorithm": f"algorithm_{i}",
                    "confidence_threshold": 0.7 + i * 0.1,
                    "max_algorithms": 3 + i,
                },
                execution_results={"accuracy": 0.8 + i * 0.05},
                auto_save=True,
                tags=["test", "analysis"],
            )
            await capture_service.capture_configuration(request)

        # Analyze configurations
        analysis = await integration_service.analyze_autonomous_configurations(
            days_back=1
        )

        # Verify analysis
        assert analysis["total_configurations"] == 3
        assert "algorithm_usage" in analysis
        assert "preprocessing_patterns" in analysis
        assert "performance_statistics" in analysis
        assert analysis["performance_statistics"]["average"] > 0.8

    def test_get_integration_statistics(self, integration_service):
        """Test integration statistics retrieval."""
        stats = integration_service.get_integration_statistics()

        assert "integration_stats" in stats
        assert "autonomous_service_info" in stats
        assert "capture_settings" in stats
        assert stats["autonomous_service_info"]["available_algorithms"] == 2
        assert stats["capture_settings"]["auto_save_threshold"] == 0.7


class TestCLIParameterInterceptor:
    """Test CLI parameter interceptor."""

    @pytest.fixture
    def capture_service(self):
        """Create configuration capture service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationCaptureService(
                storage_path=Path(temp_dir), auto_capture=True
            )

    @pytest.fixture
    def interceptor(self, capture_service):
        """Create CLI parameter interceptor."""
        return CLIParameterInterceptor(
            configuration_service=capture_service,
            enable_automatic_capture=True,
            min_execution_time=0.1,  # Short time for testing
        )

    @pytest.mark.asyncio
    async def test_capture_cli_command_decorator_sync(self, interceptor):
        """Test CLI command capture decorator for sync functions."""

        @interceptor.capture_cli_command("test_command")
        def test_function(param1: str, param2: int = 42, param3: bool = True):
            """Test function."""
            return {"result": f"processed {param1} with {param2}"}

        # Call decorated function
        result = test_function("test_value", param2=100)

        # Verify result
        assert result == {"result": "processed test_value with 100"}

        # Verify statistics
        stats = interceptor.get_capture_statistics()
        assert stats["capture_stats"]["total_commands"] == 1
        assert stats["capture_stats"]["commands_by_type"]["test_command"] == 1

    @pytest.mark.asyncio
    async def test_capture_cli_command_decorator_async(self, interceptor):
        """Test CLI command capture decorator for async functions."""

        @interceptor.capture_cli_command("async_test_command")
        async def async_test_function(
            dataset_path: str, algorithm: str = "isolation_forest"
        ):
            """Async test function."""
            await asyncio.sleep(0.2)  # Simulate processing time
            return {"accuracy": 0.85, "algorithm": algorithm}

        # Call decorated function
        result = await async_test_function("/path/to/data.csv", algorithm="lof")

        # Verify result
        assert result == {"accuracy": 0.85, "algorithm": "lof"}

        # Verify statistics
        stats = interceptor.get_capture_statistics()
        assert stats["capture_stats"]["total_commands"] == 1
        assert (
            stats["capture_stats"]["captured_commands"] == 1
        )  # Should capture due to execution time

    @pytest.mark.asyncio
    async def test_capture_cli_command_with_exception(self, interceptor):
        """Test CLI command capture when function raises exception."""
        interceptor.capture_successful_only = False  # Allow capturing failed commands

        @interceptor.capture_cli_command("failing_command")
        def failing_function(param: str):
            """Function that raises exception."""
            raise ValueError(f"Test error with {param}")

        # Call function and expect exception
        with pytest.raises(ValueError, match="Test error with test_param"):
            failing_function("test_param")

        # Verify statistics
        stats = interceptor.get_capture_statistics()
        assert stats["capture_stats"]["total_commands"] == 1
        assert (
            stats["capture_stats"]["captured_commands"] == 1
        )  # Should capture failed command

    def test_parameter_extraction(self, interceptor):
        """Test parameter extraction from function calls."""

        def test_function(a: str, b: int, c: list, d: dict):
            return "test"

        # Extract parameters
        args = ("string_value", 42)
        kwargs = {"c": [1, 2, 3], "d": {"key": "value"}}
        params = interceptor._extract_parameters(test_function, args, kwargs)

        # Verify extraction
        assert params["a"] == "string_value"
        assert params["b"] == 42
        assert params["c"] == [1, 2, 3]
        assert params["d"] == {"key": "value"}

    def test_serialize_parameter_value(self, interceptor):
        """Test parameter value serialization."""
        # Test basic types
        assert interceptor._serialize_parameter_value("string") == "string"
        assert interceptor._serialize_parameter_value(42) == 42
        assert interceptor._serialize_parameter_value([1, 2, 3]) == [1, 2, 3]

        # Test Path objects
        path_value = Path("/test/path")
        assert interceptor._serialize_parameter_value(path_value) == "/test/path"

        # Test custom objects
        class TestObject:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42

        obj = TestObject()
        serialized = interceptor._serialize_parameter_value(obj)
        assert serialized["type"] == "TestObject"
        assert serialized["attributes"]["attr1"] == "value1"
        assert serialized["attributes"]["attr2"] == 42

    @pytest.mark.asyncio
    async def test_global_interceptor_initialization(self, capture_service):
        """Test global interceptor initialization and usage."""
        # Initialize global interceptor
        global_interceptor = initialize_cli_interceptor(capture_service)
        assert global_interceptor is not None

        # Test global decorator
        @capture_detection_command(tags=["global_test"])
        def global_test_function(param: str):
            return f"global_result_{param}"

        # Call function
        result = global_test_function("test")
        assert result == "global_result_test"

        # Verify capture
        stats = global_interceptor.get_capture_statistics()
        assert stats["capture_stats"]["total_commands"] == 1


class TestConfigurationManagers:
    """Test configuration manager classes."""

    @pytest.fixture
    def capture_service(self):
        """Create configuration capture service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationCaptureService(
                storage_path=Path(temp_dir), auto_capture=True
            )

    @pytest.mark.asyncio
    async def test_automl_configuration_manager(self, capture_service):
        """Test AutoML configuration manager."""
        manager = AutoMLConfigurationManager(capture_service)

        # Create test configurations
        from pynomaly.application.dto.configuration_dto import (
            ConfigurationCaptureRequestDTO,
        )

        for i in range(3):
            request = ConfigurationCaptureRequestDTO(
                source=ConfigurationSource.AUTOML,
                raw_parameters={
                    "algorithm": "isolation_forest",
                    "contamination": 0.1,
                    "n_estimators": 100 + i * 50,
                },
                execution_results={
                    "accuracy": 0.8 + i * 0.05,
                    "training_time": 60 + i * 10,
                },
                auto_save=True,
                tags=["automl", "test"],
            )
            await capture_service.capture_configuration(request)

        # Get AutoML configurations
        configs = await manager.get_automl_configurations(
            algorithm="isolation_forest", min_accuracy=0.8, limit=10
        )

        assert len(configs) == 3
        assert all(
            config["algorithm_config"]["algorithm_name"] == "isolation_forest"
            for config in configs
        )

        # Analyze performance trends
        trends = await manager.analyze_automl_performance_trends()
        assert trends["total_configurations"] == 3
        assert "isolation_forest" in trends["algorithm_performance"]
        assert trends["algorithm_performance"]["isolation_forest"]["count"] == 3

    @pytest.mark.asyncio
    async def test_autonomous_configuration_manager(self, capture_service):
        """Test autonomous configuration manager."""
        manager = AutonomousConfigurationManager(capture_service)

        # Create test configurations
        from pynomaly.application.dto.configuration_dto import (
            ConfigurationCaptureRequestDTO,
        )

        successful_configs = []
        for i in range(3):
            request = ConfigurationCaptureRequestDTO(
                source=ConfigurationSource.AUTONOMOUS,
                raw_parameters={
                    "confidence_threshold": 0.7 + i * 0.1,
                    "max_algorithms": 3 + i,
                    "preprocessing_strategy": "auto",
                },
                execution_results={"accuracy": 0.8 + i * 0.05},
                source_context={
                    "autonomous_config": {
                        "confidence_threshold": 0.7 + i * 0.1,
                        "max_algorithms": 3 + i,
                        "preprocessing_strategy": "auto",
                    }
                },
                auto_save=True,
                tags=["autonomous", "successful"],
            )
            response = await capture_service.capture_configuration(request)
            successful_configs.append(response.configuration)

        # Test configuration recommendation
        dataset_characteristics = {"n_samples": 1000, "n_features": 10, "sparsity": 0.1}

        recommended_config = await manager.recommend_autonomous_config(
            dataset_characteristics=dataset_characteristics,
            performance_requirements={"min_accuracy": 0.8},
        )

        # Verify recommendation
        assert isinstance(recommended_config, AutonomousConfig)
        assert 0.7 <= recommended_config.confidence_threshold <= 1.0
        assert 3 <= recommended_config.max_algorithms <= 6


if __name__ == "__main__":
    pytest.main([__file__])
