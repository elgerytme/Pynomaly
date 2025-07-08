"""Tests for DetectAnomalies use case."""

from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import pandas as pd
import pytest
from pynomaly.application.use_cases.detect_anomalies import (
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    DetectAnomaliesUseCase,
)
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.exceptions import DatasetError, DetectorNotFittedError
from pynomaly.domain.value_objects import AnomalyScore


class TestDetectAnomaliesUseCase:
    """Test detect anomalies use case execution."""

    @pytest.fixture
    def mock_detector_repository(self):
        """Mock detector repository."""
        repository = Mock()
        repository.find_by_id = Mock()
        return repository

    @pytest.fixture
    def mock_feature_validator(self):
        """Mock feature validator."""
        validator = Mock()
        validator.check_data_quality = Mock()
        validator.suggest_preprocessing = Mock()
        return validator

    @pytest.fixture
    def mock_adapter_registry(self):
        """Mock algorithm adapter registry."""
        registry = Mock()
        registry.score_with_detector = Mock()
        registry.predict_with_detector = Mock()
        return registry

    @pytest.fixture
    def fitted_detector(self):
        """Fitted detector for testing."""
        return Detector(
            name="Test Detector",
            algorithm_name="IsolationForest",
            is_fitted=True,
            trained_at=datetime.utcnow(),
            parameters={"contamination": 0.1, "n_estimators": 100},
        )

    @pytest.fixture
    def unfitted_detector(self):
        """Unfitted detector for testing."""
        return Detector(
            name="Unfitted Detector",
            algorithm_name="IsolationForest",
            is_fitted=False,
            parameters={"contamination": 0.1, "n_estimators": 100},
        )

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for testing."""
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 100],  # 100 is outlier
                "feature2": [2, 4, 6, 8, 10, 200],  # 200 is outlier
            }
        )
        return Dataset(name="Test Data", data=data)

    @pytest.mark.asyncio
    async def test_execute_orchestrates_complete_detection_workflow(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        fitted_detector,
        sample_dataset,
    ):
        """Test that execute orchestrates the complete anomaly detection workflow."""
        # Arrange
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = fitted_detector
        mock_feature_validator.check_data_quality.return_value = {
            "quality_score": 0.9,
            "missing_values": [],
            "constant_features": [],
        }
        mock_feature_validator.suggest_preprocessing.return_value = []

        # Mock adapter responses
        mock_scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.15),
            AnomalyScore(0.12),
            AnomalyScore(0.18),
            AnomalyScore(0.14),
            AnomalyScore(0.95),
        ]
        mock_predictions = [0, 0, 0, 0, 0, 1]  # Last sample is anomaly

        mock_adapter_registry.score_with_detector.return_value = mock_scores
        mock_adapter_registry.predict_with_detector.return_value = mock_predictions

        request = DetectAnomaliesRequest(
            detector_id=fitted_detector.id,
            dataset=sample_dataset,
            validate_features=True,
            save_results=True,
        )

        # Act
        response = await use_case.execute(request)

        # Assert - Validate inputs
        mock_detector_repository.find_by_id.assert_called_once_with(fitted_detector.id)

        # Assert - Load detector
        assert response.result is not None
        assert response.result.detector_id == fitted_detector.id

        # Assert - Process dataset
        mock_feature_validator.check_data_quality.assert_called_once_with(
            sample_dataset
        )

        # Assert - Detect anomalies
        mock_adapter_registry.score_with_detector.assert_called_once_with(
            fitted_detector, sample_dataset
        )
        mock_adapter_registry.predict_with_detector.assert_called_once_with(
            fitted_detector, sample_dataset
        )

        # Assert - Calculate metrics
        assert len(response.result.scores) == 6
        assert len(response.result.labels) == 6
        assert response.result.n_anomalies == 1
        assert response.result.anomaly_rate == 1 / 6

        # Assert - Return structured results with comprehensive metadata
        assert isinstance(response, DetectAnomaliesResponse)
        assert isinstance(response.result, DetectionResult)
        assert response.quality_report is not None
        assert "execution_time_ms" in response.result.metadata
        assert "algorithm" in response.result.metadata
        assert response.result.metadata["algorithm"] == "IsolationForest"

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
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        detector_id = uuid4()
        mock_detector_repository.find_by_id.return_value = None

        request = DetectAnomaliesRequest(
            detector_id=detector_id, dataset=sample_dataset
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Detector .* not found"):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_execute_validates_detector_is_fitted(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        unfitted_detector,
        sample_dataset,
    ):
        """Test that execute validates detector is fitted."""
        # Arrange
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = unfitted_detector

        request = DetectAnomaliesRequest(
            detector_id=unfitted_detector.id, dataset=sample_dataset
        )

        # Act & Assert
        with pytest.raises(DetectorNotFittedError):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_execute_handles_detection_errors(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        fitted_detector,
        sample_dataset,
    ):
        """Test that execute handles detection errors gracefully."""
        # Arrange
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = fitted_detector

        # The feature validator will be called before the adapter
        mock_feature_validator.check_data_quality.return_value = {
            "quality_score": 0.9,
            "missing_values": [],
            "constant_features": [],
        }
        mock_feature_validator.suggest_preprocessing.return_value = []

        # Simulate adapter failure
        mock_adapter_registry.score_with_detector.side_effect = RuntimeError(
            "Detection failed"
        )

        request = DetectAnomaliesRequest(
            detector_id=fitted_detector.id, dataset=sample_dataset
        )

        # Act & Assert
        with pytest.raises(DatasetError, match="Detection failed"):
            await use_case.execute(request)

    @pytest.mark.asyncio
    async def test_execute_validates_data_quality_when_enabled(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        fitted_detector,
        sample_dataset,
    ):
        """Test that execute validates data quality when feature validation is enabled."""
        # Arrange
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = fitted_detector
        mock_feature_validator.check_data_quality.return_value = {
            "quality_score": 0.6,  # Low quality score
            "missing_values": ["feature1"],
            "constant_features": [],
        }
        mock_feature_validator.suggest_preprocessing.return_value = [
            "Remove missing values"
        ]

        mock_scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.15),
            AnomalyScore(0.12),
            AnomalyScore(0.18),
            AnomalyScore(0.14),
            AnomalyScore(0.95),
        ]
        mock_predictions = [0, 0, 0, 0, 0, 1]

        mock_adapter_registry.score_with_detector.return_value = mock_scores
        mock_adapter_registry.predict_with_detector.return_value = mock_predictions

        request = DetectAnomaliesRequest(
            detector_id=fitted_detector.id,
            dataset=sample_dataset,
            validate_features=True,
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        mock_feature_validator.check_data_quality.assert_called_once_with(
            sample_dataset
        )
        mock_feature_validator.suggest_preprocessing.assert_called_once()

        assert response.quality_report is not None
        assert response.quality_report["quality_score"] == 0.6
        assert response.warnings is not None
        assert any(
            "Data quality score is low" in warning for warning in response.warnings
        )
        assert any("Remove missing values" in warning for warning in response.warnings)

    @pytest.mark.asyncio
    async def test_execute_skips_validation_when_disabled(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        fitted_detector,
        sample_dataset,
    ):
        """Test that execute skips feature validation when disabled."""
        # Arrange
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = fitted_detector

        mock_scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.15),
            AnomalyScore(0.12),
            AnomalyScore(0.18),
            AnomalyScore(0.14),
            AnomalyScore(0.95),
        ]
        mock_predictions = [0, 0, 0, 0, 0, 1]

        mock_adapter_registry.score_with_detector.return_value = mock_scores
        mock_adapter_registry.predict_with_detector.return_value = mock_predictions

        request = DetectAnomaliesRequest(
            detector_id=fitted_detector.id,
            dataset=sample_dataset,
            validate_features=False,
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        mock_feature_validator.check_data_quality.assert_not_called()
        mock_feature_validator.suggest_preprocessing.assert_not_called()
        assert response.quality_report is None

    @pytest.mark.asyncio
    async def test_execute_handles_empty_dataset(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        fitted_detector,
    ):
        """Test that execute handles empty datasets correctly."""
        # Arrange
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = fitted_detector

        # Create dataset with data then make it empty to bypass Dataset validation
        data = pd.DataFrame({"feature1": [1, 2, 3]})
        empty_dataset = Dataset(name="Empty Data", data=data)
        empty_dataset.data = pd.DataFrame()  # Make it empty

        mock_adapter_registry.score_with_detector.return_value = []
        mock_adapter_registry.predict_with_detector.return_value = []

        request = DetectAnomaliesRequest(
            detector_id=fitted_detector.id,
            dataset=empty_dataset,
            validate_features=False,
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert len(response.result.scores) == 0
        assert len(response.result.labels) == 0
        assert response.result.n_anomalies == 0
        assert response.result.anomaly_rate == 0.0

    @pytest.mark.asyncio
    async def test_execute_calculates_appropriate_threshold(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        fitted_detector,
        sample_dataset,
    ):
        """Test that execute calculates appropriate anomaly threshold."""
        # Arrange
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = fitted_detector

        # Mock scores where the last one is clearly anomalous
        mock_scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.15),
            AnomalyScore(0.12),
            AnomalyScore(0.18),
            AnomalyScore(0.14),
            AnomalyScore(0.95),
        ]
        mock_predictions = [0, 0, 0, 0, 0, 1]  # Last sample is anomaly

        mock_adapter_registry.score_with_detector.return_value = mock_scores
        mock_adapter_registry.predict_with_detector.return_value = mock_predictions

        request = DetectAnomaliesRequest(
            detector_id=fitted_detector.id,
            dataset=sample_dataset,
            validate_features=False,
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        # Threshold should be the minimum score of detected anomalies
        assert response.result.threshold == 0.95
        assert response.result.threshold > 0.18  # Higher than normal samples

    @pytest.mark.asyncio
    async def test_execute_saves_results_when_requested(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        fitted_detector,
        sample_dataset,
    ):
        """Test that execute saves results when requested."""
        # Arrange
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = fitted_detector

        mock_scores = [AnomalyScore(0.1), AnomalyScore(0.95)]
        mock_predictions = [0, 1]

        mock_adapter_registry.score_with_detector.return_value = mock_scores
        mock_adapter_registry.predict_with_detector.return_value = mock_predictions

        # Create smaller dataset to match scores
        small_data = pd.DataFrame({"feature1": [1, 100], "feature2": [2, 200]})
        small_dataset = Dataset(name="Small Data", data=small_data)

        request = DetectAnomaliesRequest(
            detector_id=fitted_detector.id,
            dataset=small_dataset,
            validate_features=False,
            save_results=True,
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.result.metadata.get("saved") is True

    @pytest.mark.asyncio
    async def test_execute_provides_backward_compatibility(
        self,
        mock_detector_repository,
        mock_feature_validator,
        mock_adapter_registry,
        fitted_detector,
        sample_dataset,
    ):
        """Test that execute provides backward compatibility properties."""
        # Arrange
        use_case = DetectAnomaliesUseCase(
            detector_repository=mock_detector_repository,
            feature_validator=mock_feature_validator,
            adapter_registry=mock_adapter_registry,
        )

        mock_detector_repository.find_by_id.return_value = fitted_detector

        mock_scores = [AnomalyScore(0.1), AnomalyScore(0.15), AnomalyScore(0.95)]
        mock_predictions = [0, 0, 1]  # Last sample is anomaly

        mock_adapter_registry.score_with_detector.return_value = mock_scores
        mock_adapter_registry.predict_with_detector.return_value = mock_predictions

        # Create smaller dataset to match scores
        small_data = pd.DataFrame({"feature1": [1, 2, 100], "feature2": [2, 4, 200]})
        small_dataset = Dataset(name="Small Data", data=small_data)

        request = DetectAnomaliesRequest(
            detector_id=fitted_detector.id,
            dataset=small_dataset,
            validate_features=False,
        )

        # Act
        response = await use_case.execute(request)

        # Assert backward compatibility properties
        assert response.anomaly_indices == [2]  # Index of anomaly
        assert response.anomaly_scores == [0.1, 0.15, 0.95]  # All scores as floats
        assert len(response.anomaly_scores) == 3
