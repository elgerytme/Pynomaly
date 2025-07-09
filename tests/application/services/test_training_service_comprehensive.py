"""
Comprehensive tests for training service.
Tests model training orchestration, validation, and training pipeline management.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from pynomaly.application.services.training_service import AutomatedTrainingService as TrainingService
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.training_job import TrainingJob
from pynomaly.domain.entities.training_result import TrainingResult
from pynomaly.domain.exceptions import DatasetError, DetectorError, TrainingError
from pynomaly.domain.value_objects import PerformanceMetrics


class TestTrainingService:
    """Test suite for TrainingService application service."""

    @pytest.fixture
    def mock_dataset_repository(self):
        """Mock dataset repository."""
        mock_repo = AsyncMock()
        mock_repo.find_by_id.return_value = None
        mock_repo.save.return_value = None
        return mock_repo

    @pytest.fixture
    def mock_detector_repository(self):
        """Mock detector repository."""
        mock_repo = AsyncMock()
        mock_repo.find_by_id.return_value = None
        mock_repo.save.return_value = None
        return mock_repo

    @pytest.fixture
    def mock_training_job_repository(self):
        """Mock training job repository."""
        mock_repo = AsyncMock()
        mock_repo.save.return_value = None
        mock_repo.find_by_detector_id.return_value = []
        return mock_repo

    @pytest.fixture
    def mock_algorithm_registry(self):
        """Mock algorithm registry."""
        registry = Mock()
        adapter = Mock()
        adapter.fit.return_value = None
        adapter.is_fitted = True
        adapter.get_params.return_value = {"n_estimators": 100}
        adapter.set_params.return_value = adapter
        registry.get_adapter.return_value = adapter
        return registry

    @pytest.fixture
    def mock_validation_service(self):
        """Mock validation service."""
        service = Mock()
        service.validate_training_data.return_value = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
        }
        service.cross_validate.return_value = {
            "mean_score": 0.85,
            "std_score": 0.03,
            "fold_scores": [0.83, 0.87, 0.84, 0.86, 0.85],
        }
        return service

    @pytest.fixture
    def mock_performance_evaluator(self):
        """Mock performance evaluator."""
        evaluator = Mock()
        evaluator.evaluate.return_value = PerformanceMetrics(
            precision=0.85,
            recall=0.78,
            f1_score=0.814,
            accuracy=0.92,
            roc_auc=0.88,
        )
        return evaluator

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        return Dataset(
            id=uuid4(),
            name="training-dataset",
            file_path="/tmp/train.csv",
            features=["feature1", "feature2", "feature3"],
            feature_types={"feature1": "numeric", "feature2": "numeric", "feature3": "numeric"},
            target_column=None,
            data_shape=(1000, 3),
        )

    @pytest.fixture
    def sample_detector(self):
        """Create sample detector."""
        return Detector(
            id=uuid4(),
            name="training-detector",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100, "contamination": 0.1},
            is_fitted=False,
        )

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        return np.random.randn(1000, 3)

    @pytest.fixture
    def sample_validation_data(self):
        """Create sample validation data."""
        return np.random.randn(200, 3)

    @pytest.fixture
    def training_service(
        self,
        mock_dataset_repository,
        mock_detector_repository,
        mock_training_job_repository,
        mock_algorithm_registry,
        mock_validation_service,
        mock_performance_evaluator,
    ):
        """Create training service with mocked dependencies."""
        return TrainingService(
            dataset_repository=mock_dataset_repository,
            detector_repository=mock_detector_repository,
            training_job_repository=mock_training_job_repository,
            algorithm_registry=mock_algorithm_registry,
            validation_service=mock_validation_service,
            performance_evaluator=mock_performance_evaluator,
        )

    @pytest.mark.asyncio
    async def test_train_detector_basic(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
        mock_algorithm_registry,
    ):
        """Test basic detector training."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
        )

        # Verify
        assert training_result is not None
        assert isinstance(training_result, TrainingResult)
        assert training_result.detector_id == detector_id
        assert training_result.training_successful is True
        assert training_result.performance_metrics is not None

        # Verify detector was fitted and saved
        mock_algorithm_registry.get_adapter.assert_called_once()
        mock_detector_repository.save.assert_called_once()

        # Verify detector is now fitted
        saved_detector = mock_detector_repository.save.call_args[0][0]
        assert saved_detector.is_fitted is True

    @pytest.mark.asyncio
    async def test_train_detector_with_validation(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        sample_validation_data,
        mock_detector_repository,
        mock_validation_service,
    ):
        """Test detector training with validation."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
            validation_data=sample_validation_data,
        )

        # Verify
        assert training_result is not None
        assert training_result.validation_results is not None
        assert "validation_score" in training_result.validation_results
        assert "validation_metrics" in training_result.validation_results

        # Verify validation service was called
        mock_validation_service.validate_training_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_detector_with_cross_validation(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
        mock_validation_service,
    ):
        """Test detector training with cross-validation."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
            use_cross_validation=True,
            cv_folds=5,
        )

        # Verify
        assert training_result is not None
        assert training_result.cross_validation_results is not None
        assert "mean_score" in training_result.cross_validation_results
        assert "std_score" in training_result.cross_validation_results
        assert "fold_scores" in training_result.cross_validation_results

        # Verify cross-validation was performed
        mock_validation_service.cross_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_detector_not_found(
        self,
        training_service,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test training with non-existent detector."""
        # Setup
        detector_id = uuid4()
        mock_detector_repository.find_by_id.return_value = None

        # Execute & Verify
        with pytest.raises(DetectorError, match="Detector not found"):
            await training_service.train_detector(
                detector_id=detector_id,
                training_data=sample_training_data,
            )

    @pytest.mark.asyncio
    async def test_train_detector_invalid_data(
        self,
        training_service,
        sample_detector,
        mock_detector_repository,
    ):
        """Test training with invalid data."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Test with None data
        with pytest.raises(ValueError, match="Training data cannot be None"):
            await training_service.train_detector(
                detector_id=detector_id,
                training_data=None,
            )

        # Test with empty data
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            await training_service.train_detector(
                detector_id=detector_id,
                training_data=np.array([]),
            )

    @pytest.mark.asyncio
    async def test_train_detector_already_fitted(
        self,
        training_service,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test training already fitted detector."""
        # Setup
        fitted_detector = Detector(
            id=uuid4(),
            name="fitted-detector",
            algorithm_name="IsolationForest",
            hyperparameters={},
            is_fitted=True,
        )
        mock_detector_repository.find_by_id.return_value = fitted_detector

        # Execute with force retrain
        training_result = await training_service.train_detector(
            detector_id=fitted_detector.id,
            training_data=sample_training_data,
            force_retrain=True,
        )

        # Verify
        assert training_result is not None
        assert training_result.was_retrained is True

    @pytest.mark.asyncio
    async def test_train_detector_with_early_stopping(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        sample_validation_data,
        mock_detector_repository,
    ):
        """Test detector training with early stopping."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        early_stopping_config = {
            "patience": 10,
            "min_delta": 0.001,
            "monitor": "validation_score",
            "mode": "max",
        }

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
            validation_data=sample_validation_data,
            early_stopping=early_stopping_config,
        )

        # Verify
        assert training_result is not None
        assert training_result.early_stopping_triggered is not None
        assert training_result.early_stopping_config == early_stopping_config

    @pytest.mark.asyncio
    async def test_train_detector_with_callbacks(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test detector training with callbacks."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Mock callbacks
        progress_callback = Mock()
        validation_callback = Mock()
        
        callbacks = {
            "on_epoch_end": progress_callback,
            "on_validation_end": validation_callback,
        }

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
            callbacks=callbacks,
        )

        # Verify
        assert training_result is not None
        assert training_result.callbacks_used is True

    @pytest.mark.asyncio
    async def test_train_detector_batch_processing(
        self,
        training_service,
        sample_detector,
        mock_detector_repository,
    ):
        """Test detector training with batch processing."""
        # Setup
        detector_id = sample_detector.id
        large_training_data = np.random.randn(10000, 3)
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=large_training_data,
            batch_size=1000,
            use_batch_processing=True,
        )

        # Verify
        assert training_result is not None
        assert training_result.batch_processing_used is True
        assert training_result.batch_size == 1000
        assert training_result.total_batches == 10

    @pytest.mark.asyncio
    async def test_train_detector_with_hyperparameter_tuning(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test detector training with hyperparameter tuning."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        hyperparameter_space = {
            "n_estimators": [50, 100, 200],
            "contamination": [0.05, 0.1, 0.15],
        }

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
            tune_hyperparameters=True,
            hyperparameter_space=hyperparameter_space,
        )

        # Verify
        assert training_result is not None
        assert training_result.hyperparameter_tuning_results is not None
        assert "best_params" in training_result.hyperparameter_tuning_results
        assert "best_score" in training_result.hyperparameter_tuning_results

    @pytest.mark.asyncio
    async def test_create_training_job(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
        mock_training_job_repository,
    ):
        """Test training job creation."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        training_job = await training_service.create_training_job(
            detector_id=detector_id,
            training_data=sample_training_data,
            job_name="test-training-job",
        )

        # Verify
        assert training_job is not None
        assert isinstance(training_job, TrainingJob)
        assert training_job.detector_id == detector_id
        assert training_job.job_name == "test-training-job"
        assert training_job.status == "pending"

        # Verify job was saved
        mock_training_job_repository.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_training_job(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
        mock_training_job_repository,
    ):
        """Test training job execution."""
        # Setup
        training_job = TrainingJob(
            id=uuid4(),
            detector_id=sample_detector.id,
            job_name="test-job",
            training_data=sample_training_data,
            status="pending",
        )
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        training_result = await training_service.execute_training_job(training_job)

        # Verify
        assert training_result is not None
        assert training_job.status == "completed"
        assert training_job.training_result is not None

    @pytest.mark.asyncio
    async def test_train_multiple_detectors(
        self,
        training_service,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test training multiple detectors concurrently."""
        # Setup
        detectors = [
            Detector(
                id=uuid4(),
                name=f"detector-{i}",
                algorithm_name="IsolationForest",
                hyperparameters={},
                is_fitted=False,
            )
            for i in range(3)
        ]

        def find_by_id_side_effect(detector_id):
            return next((d for d in detectors if d.id == detector_id), None)

        mock_detector_repository.find_by_id.side_effect = find_by_id_side_effect

        # Execute
        detector_ids = [d.id for d in detectors]
        training_results = await training_service.train_multiple_detectors(
            detector_ids=detector_ids,
            training_data=sample_training_data,
        )

        # Verify
        assert len(training_results) == 3
        for result in training_results:
            assert isinstance(result, TrainingResult)
            assert result.training_successful is True

    @pytest.mark.asyncio
    async def test_train_detector_with_data_augmentation(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test detector training with data augmentation."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        augmentation_config = {
            "noise_injection": {"noise_level": 0.1, "probability": 0.3},
            "feature_dropout": {"dropout_rate": 0.1, "probability": 0.2},
            "synthetic_generation": {"method": "gaussian", "count": 100},
        }

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
            use_data_augmentation=True,
            augmentation_config=augmentation_config,
        )

        # Verify
        assert training_result is not None
        assert training_result.data_augmentation_used is True
        assert training_result.augmentation_config == augmentation_config

    @pytest.mark.asyncio
    async def test_train_detector_with_incremental_learning(
        self,
        training_service,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test detector training with incremental learning."""
        # Setup
        fitted_detector = Detector(
            id=uuid4(),
            name="incremental-detector",
            algorithm_name="IsolationForest",
            hyperparameters={},
            is_fitted=True,
        )
        mock_detector_repository.find_by_id.return_value = fitted_detector

        # Execute
        training_result = await training_service.train_detector(
            detector_id=fitted_detector.id,
            training_data=sample_training_data,
            incremental_learning=True,
        )

        # Verify
        assert training_result is not None
        assert training_result.incremental_learning_used is True
        assert training_result.was_retrained is False  # Should be incremental update

    @pytest.mark.asyncio
    async def test_train_detector_error_handling(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
        mock_algorithm_registry,
    ):
        """Test training error handling."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Mock algorithm adapter to raise error
        mock_algorithm_registry.get_adapter.return_value.fit.side_effect = RuntimeError("Training failed")

        # Execute & Verify
        with pytest.raises(TrainingError, match="Training failed"):
            await training_service.train_detector(
                detector_id=detector_id,
                training_data=sample_training_data,
            )

    @pytest.mark.asyncio
    async def test_train_detector_performance_monitoring(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test training performance monitoring."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
            monitor_performance=True,
        )

        # Verify
        assert training_result is not None
        assert training_result.performance_monitoring is not None
        assert "training_time" in training_result.performance_monitoring
        assert "memory_usage" in training_result.performance_monitoring
        assert "cpu_usage" in training_result.performance_monitoring

    @pytest.mark.asyncio
    async def test_train_detector_with_custom_loss_function(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test detector training with custom loss function."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        # Define custom loss function
        def custom_loss(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
            custom_loss_function=custom_loss,
        )

        # Verify
        assert training_result is not None
        assert training_result.custom_loss_used is True

    @pytest.mark.asyncio
    async def test_train_detector_with_regularization(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test detector training with regularization."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        regularization_config = {
            "l1_regularization": 0.01,
            "l2_regularization": 0.001,
            "dropout_rate": 0.1,
        }

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
            regularization_config=regularization_config,
        )

        # Verify
        assert training_result is not None
        assert training_result.regularization_used is True
        assert training_result.regularization_config == regularization_config

    @pytest.mark.asyncio
    async def test_train_detector_distributed_training(
        self,
        training_service,
        sample_detector,
        sample_training_data,
        mock_detector_repository,
    ):
        """Test distributed training."""
        # Setup
        detector_id = sample_detector.id
        mock_detector_repository.find_by_id.return_value = sample_detector

        distributed_config = {
            "num_workers": 4,
            "distribution_strategy": "data_parallel",
            "communication_backend": "nccl",
        }

        # Execute
        training_result = await training_service.train_detector(
            detector_id=detector_id,
            training_data=sample_training_data,
            distributed_training=True,
            distributed_config=distributed_config,
        )

        # Verify
        assert training_result is not None
        assert training_result.distributed_training_used is True
        assert training_result.distributed_config == distributed_config

    @pytest.mark.asyncio
    async def test_train_detector_integration_workflow(
        self,
        training_service,
        sample_detector,
        sample_dataset,
        sample_training_data,
        sample_validation_data,
        mock_detector_repository,
        mock_dataset_repository,
    ):
        """Test complete training integration workflow."""
        # Setup
        detector_id = sample_detector.id
        dataset_id = sample_dataset.id
        mock_detector_repository.find_by_id.return_value = sample_detector
        mock_dataset_repository.find_by_id.return_value = sample_dataset

        # Execute complete workflow
        training_result = await training_service.train_detector_from_dataset(
            detector_id=detector_id,
            dataset_id=dataset_id,
            validation_split=0.2,
            use_cross_validation=True,
            tune_hyperparameters=True,
            monitor_performance=True,
        )

        # Verify
        assert training_result is not None
        assert training_result.dataset_id == dataset_id
        assert training_result.validation_split == 0.2
        assert training_result.cross_validation_results is not None
        assert training_result.hyperparameter_tuning_results is not None
        assert training_result.performance_monitoring is not None

        # Verify all repositories were called
        mock_detector_repository.find_by_id.assert_called_once_with(detector_id)
        mock_dataset_repository.find_by_id.assert_called_once_with(dataset_id)