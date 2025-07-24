"""Integration tests for the hexagonal architecture implementation.

These tests verify that the new architecture works correctly with
dependency injection and interface-based design.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from anomaly_detection.infrastructure.container.container import Container
from anomaly_detection.domain.interfaces.ml_operations import (
    MLModelTrainingPort,
    TrainingRequest,
    TrainingResult,
    ModelStatus,
)
from anomaly_detection.domain.interfaces.mlops_operations import (
    MLOpsExperimentTrackingPort,
    MLOpsModelRegistryPort,
    ExperimentMetadata,
    RunMetadata,
    ExperimentStatus,
    RunStatus,
)
from anomaly_detection.domain.entities.dataset import Dataset
from anomaly_detection.domain.entities.model import Model
from anomaly_detection.application.services.model_training_service import (
    ModelTrainingApplicationService
)


class TestHexagonalArchitecture:
    """Test suite for hexagonal architecture implementation."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        
        from anomaly_detection.domain.entities.dataset import DatasetMetadata
        
        metadata = DatasetMetadata(
            name="test_dataset",
            description="Test dataset for integration tests",
            source="test",
            feature_names=["feature_1", "feature_2", "feature_3"]
        )
        
        return Dataset(
            data=data,
            feature_names=["feature_1", "feature_2", "feature_3"],
            metadata=metadata
        )
    
    @pytest.fixture
    def mock_ml_training(self):
        """Create a mock ML training port."""
        mock = AsyncMock(spec=MLModelTrainingPort)
        
        # Mock training result
        from anomaly_detection.domain.entities.model import ModelMetadata, ModelStatus
        
        mock_metadata = ModelMetadata(
            model_id="test_model_123",
            name="test_isolation_forest",
            algorithm="isolation_forest",
            version="1.0.0",
            status=ModelStatus.TRAINED,
            hyperparameters={"contamination": 0.1},
            accuracy=0.85,
            precision=0.80,
            recall=0.90,
            f1_score=0.85,
            description="Test model"
        )
        
        mock_model = Model(
            metadata=mock_metadata,
            model_object=Mock()
        )
        
        mock_result = TrainingResult(
            model=mock_model,
            training_metrics={"accuracy": 0.85, "precision": 0.80, "recall": 0.90},
            validation_metrics={"accuracy": 0.83, "precision": 0.78, "recall": 0.88},
            training_duration_seconds=10.5,
            feature_importance={"feature_1": 0.5, "feature_2": 0.3, "feature_3": 0.2},
            model_artifacts={"model_path": "/tmp/model.pkl"},
            status=ModelStatus.TRAINED
        )
        
        mock.train_model.return_value = mock_result
        mock.get_supported_algorithms.return_value = ["isolation_forest", "one_class_svm"]
        mock.get_algorithm_parameters.return_value = {
            "contamination": {"type": "float", "default": 0.1, "range": [0.0, 0.5]}
        }
        
        return mock
    
    @pytest.fixture
    def mock_experiment_tracking(self):
        """Create a mock experiment tracking port."""
        mock = AsyncMock(spec=MLOpsExperimentTrackingPort)
        
        mock.create_experiment.return_value = "exp_123"
        mock.start_run.return_value = "run_456"
        mock.log_parameters.return_value = None
        mock.log_metrics.return_value = None
        mock.end_run.return_value = None
        
        return mock
    
    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock model registry port."""
        mock = AsyncMock(spec=MLOpsModelRegistryPort)
        
        mock.register_model.return_value = "model_789"
        mock.create_model_version.return_value = "version_001"
        
        return mock
    
    @pytest.fixture
    def container(self, mock_ml_training, mock_experiment_tracking, mock_model_registry):
        """Create a container with mock implementations."""
        container = Container()
        
        # Register mock implementations
        container.register_instance(MLModelTrainingPort, mock_ml_training)
        container.register_instance(MLOpsExperimentTrackingPort, mock_experiment_tracking)
        container.register_instance(MLOpsModelRegistryPort, mock_model_registry)
        
        return container
    
    @pytest.fixture
    def training_service(self, mock_ml_training, mock_experiment_tracking, mock_model_registry):
        """Create a training service with mock dependencies."""
        return ModelTrainingApplicationService(
            ml_training=mock_ml_training,
            experiment_tracking=mock_experiment_tracking,
            model_registry=mock_model_registry
        )
    
    @pytest.mark.asyncio
    async def test_dependency_injection_container(self, container):
        """Test that the dependency injection container works correctly."""
        # Test that we can resolve dependencies
        ml_training = container.get(MLModelTrainingPort)
        experiment_tracking = container.get(MLOpsExperimentTrackingPort)
        model_registry = container.get(MLOpsModelRegistryPort)
        
        assert ml_training is not None
        assert experiment_tracking is not None
        assert model_registry is not None
        
        # Test that singletons return the same instance
        ml_training_2 = container.get(MLModelTrainingPort)
        assert ml_training is ml_training_2
    
    @pytest.mark.asyncio
    async def test_interface_based_architecture(self, training_service, sample_dataset):
        """Test that the architecture works through interfaces."""
        # Test getting supported algorithms
        algorithms = await training_service.get_supported_algorithms()
        assert len(algorithms) > 0
        assert "isolation_forest" in algorithms
        
        # Test getting algorithm parameters
        params = await training_service.get_algorithm_parameters("isolation_forest")
        assert "contamination" in params
        
        # Test model training workflow
        result = await training_service.train_anomaly_detection_model(
            algorithm_name="isolation_forest",
            training_data=sample_dataset,
            parameters={"contamination": 0.1},
            experiment_name="test_experiment",
            register_model=True,
            created_by="test_user"
        )
        
        # Verify the result structure
        assert result["success"] is True
        assert "model" in result
        assert "training" in result
        assert "experiment" in result
        assert "registry" in result
        
        # Verify model information
        model_info = result["model"]
        assert model_info["id"] == "test_model_123"
        assert model_info["algorithm"] == "isolation_forest"
        assert model_info["status"] == "trained"
        
        # Verify training information
        training_info = result["training"]
        assert "metrics" in training_info
        assert "duration_seconds" in training_info
        assert training_info["metrics"]["accuracy"] == 0.85
    
    @pytest.mark.asyncio
    async def test_experiment_tracking_integration(
        self, 
        training_service, 
        sample_dataset, 
        mock_experiment_tracking
    ):
        """Test that experiment tracking integration works correctly."""
        await training_service.train_anomaly_detection_model(
            algorithm_name="isolation_forest",
            training_data=sample_dataset,
            register_model=False,
            created_by="test_user"
        )
        
        # Verify experiment tracking calls
        mock_experiment_tracking.create_experiment.assert_called_once()
        mock_experiment_tracking.start_run.assert_called_once()
        mock_experiment_tracking.log_parameters.assert_called()
        mock_experiment_tracking.log_metrics.assert_called()
        mock_experiment_tracking.end_run.assert_called_once()
        
        # Check that experiment was created with correct parameters
        create_call = mock_experiment_tracking.create_experiment.call_args
        assert "isolation_forest_training" in create_call[1]["name"]
        assert create_call[1]["created_by"] == "test_user"
    
    @pytest.mark.asyncio
    async def test_model_registry_integration(
        self,
        training_service,
        sample_dataset,
        mock_model_registry
    ):
        """Test that model registry integration works correctly."""
        result = await training_service.train_anomaly_detection_model(
            algorithm_name="isolation_forest",
            training_data=sample_dataset,
            register_model=True,
            created_by="test_user"
        )
        
        # Verify model registry calls
        mock_model_registry.register_model.assert_called_once()
        mock_model_registry.create_model_version.assert_called_once()
        
        # Check registration parameters
        register_call = mock_model_registry.register_model.call_args
        assert "isolation_forest_anomaly_detector" in register_call[1]["name"]
        assert register_call[1]["created_by"] == "test_user"
        
        # Verify registry information in result
        assert result["registry"]["registered"] is True
        assert result["registry"]["model_id"] == "model_789"
        assert result["registry"]["version_id"] == "version_001"
    
    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        training_service,
        sample_dataset,
        mock_ml_training,
        mock_experiment_tracking
    ):
        """Test error handling in the hexagonal architecture."""
        # Make ML training fail
        mock_ml_training.train_model.side_effect = Exception("Training failed")
        
        result = await training_service.train_anomaly_detection_model(
            algorithm_name="isolation_forest",
            training_data=sample_dataset,
            created_by="test_user"
        )
        
        # Verify error handling
        assert result["success"] is False
        assert "error" in result
        assert "Training failed" in result["error"]
        
        # Verify that run was marked as failed
        mock_experiment_tracking.end_run.assert_called_with("run_456", RunStatus.FAILED)
    
    @pytest.mark.asyncio
    async def test_domain_service_refactoring(self, mock_experiment_tracking, mock_model_registry):
        """Test that domain services work with the new interface-based approach."""
        from anomaly_detection.domain.services.ab_testing_service import (
            ABTestingService,
            ABTestConfig,
            TestVariant,
            SplitType
        )
        
        # Mock model registry to return empty versions (to make validation fail gracefully)
        mock_model_registry.list_model_versions.return_value = []
        
        # Create AB testing service with interface dependencies
        ab_service = ABTestingService(
            experiment_tracking=mock_experiment_tracking,
            model_registry=mock_model_registry
        )
        
        # Create test configuration with 2 variants to satisfy validation
        config = ABTestConfig(
            test_name="test_ab_experiment",
            description="Test A/B experiment",
            variants=[
                TestVariant(
                    variant_id="variant_a",
                    name="Variant A",
                    model_id="model_a",
                    model_version=1,
                    traffic_percentage=50.0
                ),
                TestVariant(
                    variant_id="variant_b",
                    name="Variant B",
                    model_id="model_b",
                    model_version=1,
                    traffic_percentage=50.0
                )
            ],
            split_type=SplitType.RANDOM,
            duration_days=7
        )
        
        # This should fail validation since model doesn't exist, but it demonstrates
        # that the service is using the injected interfaces
        with pytest.raises(ValueError, match="Model version 1 not found"):
            await ab_service.create_test(config, created_by="test_user")
        
        # Verify that model registry was called
        mock_model_registry.list_model_versions.assert_called_with("model_a")
    
    def test_stub_implementations_available(self):
        """Test that stub implementations are available when external packages aren't."""
        from anomaly_detection.infrastructure.adapters.stubs.ml_stubs import MLTrainingStub
        from anomaly_detection.infrastructure.adapters.stubs.mlops_stubs import (
            MLOpsExperimentTrackingStub,
            MLOpsModelRegistryStub
        )
        
        # Verify stub implementations can be instantiated
        ml_stub = MLTrainingStub()
        experiment_stub = MLOpsExperimentTrackingStub()
        registry_stub = MLOpsModelRegistryStub()
        
        assert ml_stub is not None
        assert experiment_stub is not None
        assert registry_stub is not None
    
    @pytest.mark.asyncio
    async def test_stub_functionality(self, sample_dataset):
        """Test that stub implementations provide basic functionality."""
        from anomaly_detection.infrastructure.adapters.stubs.ml_stubs import MLTrainingStub
        from anomaly_detection.infrastructure.adapters.stubs.mlops_stubs import (
            MLOpsExperimentTrackingStub,
            MLOpsModelRegistryStub
        )
        from anomaly_detection.domain.interfaces.ml_operations import TrainingRequest
        
        # Test ML stub
        ml_stub = MLTrainingStub()
        algorithms = await ml_stub.get_supported_algorithms()
        assert len(algorithms) > 0
        
        # Test training with stub
        request = TrainingRequest(
            algorithm_name="isolation_forest",
            training_data=sample_dataset,
            parameters={"contamination": 0.1},
            created_by="test"
        )
        result = await ml_stub.train_model(request)
        assert result.model is not None
        assert result.status == ModelStatus.TRAINED
        
        # Test MLOps stubs
        exp_stub = MLOpsExperimentTrackingStub()
        exp_id = await exp_stub.create_experiment("test_exp", created_by="test")
        assert exp_id is not None
        
        run_id = await exp_stub.start_run(exp_id, created_by="test")
        assert run_id is not None
        
        registry_stub = MLOpsModelRegistryStub()
        model_id = await registry_stub.register_model("test_model", created_by="test")
        assert model_id is not None