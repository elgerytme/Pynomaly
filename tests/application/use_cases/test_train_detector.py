"""Tests for TrainDetector use case."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pandas as pd
import pytest

from pynomaly.application.use_cases.train_detector import (
    TrainDetectorRequest,
    TrainDetectorResponse,
    TrainDetectorUseCase,
)
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.training_result import TrainingResult
from pynomaly.domain.exceptions import FittingError, InsufficientDataError
from pynomaly.domain.value_objects import ContaminationRate


class TestTrainDetectorUseCase:
    """Test train detector use case execution."""

    @pytest.fixture
    def mock_detector_repository(self):
        """Mock detector repository."""
        repository = Mock()
        repository.find_by_id = Mock()
        repository.save = Mock()
        return repository

    @pytest.fixture
    def mock_feature_validator(self):
        """Mock feature validator."""
        validator = Mock()
        validator.validate_numeric_features = Mock()
        validator.check_data_quality = Mock()
        return validator

    @pytest.fixture
    def mock_adapter_registry(self):
        """Mock algorithm adapter registry."""
        registry = Mock()
        registry.fit_detector = Mock()
        return registry

    @pytest.fixture
    def sample_detector(self):
        """Sample detector for testing."""
        return Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.1, "n_estimators": 100},
        )

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for testing."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],  # 100 is outlier
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 200],  # 200 is outlier
            }
        )
        return Dataset(name="Test Data", data=data)

    @pytest.mark.asyncio
    async def test_execute_orchestrates_complete_training_workflow(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        sample_detector,
        sample_dataset,
    ):
        """Test that execute orchestrates the complete training workflow."""
        # Arrange
        use_case = TrainDetectorUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = sample_detector
        mock_feature_validator.validate_numeric_features.return_value = [
            "feature1",
            "feature2",
        ]
        mock_feature_validator.check_data_quality.return_value = {
            "quality_score": 0.9,
            "missing_values": [],
            "constant_features": [],
        }

        request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=sample_dataset,
            save_model=True,
            validate_data=True,
        )

        # Act
        response = await use_case.execute(request)

        # Assert - Validate inputs
        mock_detector_repository.find_by_id.assert_called_once_with(sample_detector.id)

        # Assert - Load detector
        assert response.trained_detector is not None
        assert response.trained_detector.id == sample_detector.id

        # Assert - Process dataset
        mock_feature_validator.validate_numeric_features.assert_called_once_with(
            sample_dataset
        )
        mock_feature_validator.check_data_quality.assert_called_once_with(
            sample_dataset
        )

        # Assert - Execute training
        mock_adapter_registry.fit_detector.assert_called_once_with(
            sample_detector, sample_dataset
        )

        # Assert - Validate model performance (basic checks)
        assert response.trained_detector.is_fitted is True
        assert response.trained_detector.trained_at is not None

        # Assert - Persist trained model
        mock_detector_repository.save.assert_called_once_with(sample_detector)

        # Assert - Return structured results with comprehensive metadata
        assert isinstance(response, TrainDetectorResponse)
        assert response.training_metrics is not None
        assert "training_time" in response.training_metrics
        assert "dataset_summary" in response.training_metrics
        assert "training_samples" in response.training_metrics
        assert response.training_metrics["training_samples"] == len(sample_dataset.data)
        assert response.model_path is not None

    @pytest.mark.asyncio
    async def test_execute_validates_detector_exists(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        sample_dataset,
    ):
        """Test that execute validates detector exists."""
        # Arrange
        use_case = TrainDetectorUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        detector_id = uuid4()
        mock_detector_repository.find_by_id.return_value = None

        request = TrainDetectorRequest(
            detector_id=detector_id, training_data=sample_dataset
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Detector .* not found"):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_execute_validates_dataset_size(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        sample_detector,
    ):
        """Test that execute validates dataset has sufficient samples."""
        # Arrange
        use_case = TrainDetectorUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
            min_samples=20,  # Require more samples than our test data
        )

        mock_detector_repository.find_by_id.return_value = sample_detector

        # Create small dataset
        small_data = pd.DataFrame({"feature1": [1, 2, 3]})
        small_dataset = Dataset(name="Small Data", data=small_data)

        request = TrainDetectorRequest(
            detector_id=sample_detector.id, training_data=small_dataset
        )

        # Act & Assert
        with pytest.raises(InsufficientDataError):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_execute_handles_training_failures(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        sample_detector,
        sample_dataset,
    ):
        """Test that execute handles training failures gracefully."""
        # Arrange
        use_case = TrainDetectorUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = sample_detector
        mock_feature_validator.validate_numeric_features.return_value = [
            "feature1",
            "feature2",
        ]
        mock_feature_validator.check_data_quality.return_value = {
            "quality_score": 0.9,
            "missing_values": [],
            "constant_features": [],
        }
        mock_adapter_registry.fit_detector.side_effect = RuntimeError("Training failed")

        request = TrainDetectorRequest(
            detector_id=sample_detector.id, training_data=sample_dataset
        )

        # Act & Assert
        with pytest.raises(FittingError, match="Training failed"):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_execute_handles_hyperparameter_tuning(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        sample_detector,
        sample_dataset,
    ):
        """Test that execute handles hyperparameter tuning."""
        # Arrange
        use_case = TrainDetectorUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = sample_detector
        mock_feature_validator.validate_numeric_features.return_value = [
            "feature1",
            "feature2",
        ]
        mock_feature_validator.check_data_quality.return_value = {
            "quality_score": 0.9,
            "missing_values": [],
            "constant_features": [],
        }

        request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=sample_dataset,
            hyperparameter_grid={
                "n_estimators": [50, 100, 200],
                "contamination": [0.05, 0.1, 0.15],
            },
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.training_metrics["best_parameters"] is not None
        assert "n_estimators" in response.training_metrics["best_parameters"]
        assert "contamination" in response.training_metrics["best_parameters"]

    @pytest.mark.asyncio
    async def test_execute_validates_data_quality(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        sample_detector,
        sample_dataset,
    ):
        """Test that execute validates data quality when requested."""
        # Arrange
        use_case = TrainDetectorUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = sample_detector
        mock_feature_validator.validate_numeric_features.return_value = [
            "feature1",
            "feature2",
        ]
        mock_feature_validator.check_data_quality.return_value = {
            "quality_score": 0.7,
            "missing_values": ["feature1"],
            "constant_features": ["feature2"],
        }

        request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=sample_dataset,
            validate_data=True,
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.training_warnings is not None
        assert any(
            "Missing values" in warning for warning in response.training_warnings
        )
        assert any(
            "constant features" in warning for warning in response.training_warnings
        )
        assert response.training_metrics["validation_results"] is not None
        assert response.training_metrics["validation_results"]["quality_score"] == 0.7

    @pytest.mark.asyncio
    async def test_execute_skips_data_validation_when_disabled(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        sample_detector,
        sample_dataset,
    ):
        """Test that execute skips data validation when disabled."""
        # Arrange
        use_case = TrainDetectorUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = sample_detector

        request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=sample_dataset,
            validate_data=False,
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        mock_feature_validator.validate_numeric_features.assert_not_called()
        mock_feature_validator.check_data_quality.assert_not_called()
        assert response.training_metrics["validation_results"] is None

    @pytest.mark.asyncio
    async def test_execute_updates_detector_parameters(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        sample_detector,
        sample_dataset,
    ):
        """Test that execute updates detector parameters as specified."""
        # Arrange
        use_case = TrainDetectorUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = sample_detector

        new_contamination = ContaminationRate(0.15)
        new_parameters = {"n_estimators": 150, "max_samples": 256}

        request = TrainDetectorRequest(
            detector_id=sample_detector.id,
            training_data=sample_dataset,
            contamination_rate=new_contamination,
            parameters=new_parameters,
            validate_data=False,
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.trained_detector.contamination_rate == new_contamination
        assert response.trained_detector.parameters["n_estimators"] == 150
        assert response.trained_detector.parameters["max_samples"] == 256
