"""Comprehensive application service layer tests for anomaly detection."""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
import numpy as np
from typing import Dict, List, Any

from anomaly_detection.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from anomaly_detection.application.use_cases.train_model import TrainModelUseCase
from anomaly_detection.domain.entities.anomaly import Anomaly, AnomalyType, AnomalySeverity
from anomaly_detection.domain.value_objects.algorithm_config import AlgorithmConfig, AlgorithmType
from anomaly_detection.domain.value_objects.detection_metrics import DetectionMetrics


class TestDetectAnomaliesUseCase:
    """Test cases for the DetectAnomalies use case."""
    
    @pytest.fixture
    def mock_model_repository(self):
        """Mock model repository."""
        repository = Mock()
        repository.get_by_id = AsyncMock()
        repository.save = AsyncMock()
        return repository
    
    @pytest.fixture
    def mock_anomaly_repository(self):
        """Mock anomaly repository."""
        repository = Mock()
        repository.save = AsyncMock()
        repository.save_batch = AsyncMock()
        return repository
    
    @pytest.fixture
    def mock_detector(self):
        """Mock anomaly detector."""
        detector = Mock()
        detector.detect = AsyncMock()
        detector.fit = AsyncMock()
        detector.predict = AsyncMock()
        return detector
    
    @pytest.fixture
    def detect_anomalies_use_case(self, mock_model_repository, mock_anomaly_repository, mock_detector):
        """Create DetectAnomaliesUseCase with mocked dependencies."""
        return DetectAnomaliesUseCase(
            model_repository=mock_model_repository,
            anomaly_repository=mock_anomaly_repository,
            detector=mock_detector
        )
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_single_point(self, detect_anomalies_use_case, mock_detector):
        """Test detecting anomalies for a single data point."""
        # Arrange
        input_data = np.array([[1.5, 2.0, 3.5]])
        model_id = "test_model_123"
        
        # Mock detector response
        mock_detector.detect.return_value = [
            Anomaly(
                index=0,
                confidence_score=0.85,
                anomaly_type=AnomalyType.POINT,
                severity=AnomalySeverity.HIGH
            )
        ]
        
        # Act
        anomalies = await detect_anomalies_use_case.execute(
            data=input_data,
            model_id=model_id
        )
        
        # Assert
        assert len(anomalies) == 1
        assert anomalies[0].confidence_score == 0.85
        assert anomalies[0].anomaly_type == AnomalyType.POINT
        assert anomalies[0].severity == AnomalySeverity.HIGH
        
        mock_detector.detect.assert_called_once_with(input_data, model_id)
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_batch_processing(self, detect_anomalies_use_case, mock_detector):
        """Test detecting anomalies for batch data."""
        # Arrange
        input_data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [100.0, 200.0, 300.0]  # Anomalous point
        ])
        model_id = "batch_model_456"
        
        # Mock detector response - only last point is anomalous
        mock_detector.detect.return_value = [
            Anomaly(
                index=3,
                confidence_score=0.95,
                anomaly_type=AnomalyType.POINT,
                severity=AnomalySeverity.CRITICAL,
                feature_values=np.array([100.0, 200.0, 300.0])
            )
        ]
        
        # Act
        anomalies = await detect_anomalies_use_case.execute(
            data=input_data,
            model_id=model_id,
            batch_size=2
        )
        
        # Assert
        assert len(anomalies) == 1
        assert anomalies[0].index == 3
        assert anomalies[0].confidence_score == 0.95
        assert anomalies[0].severity == AnomalySeverity.CRITICAL
        assert np.array_equal(anomalies[0].feature_values, np.array([100.0, 200.0, 300.0]))
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_with_threshold(self, detect_anomalies_use_case, mock_detector):
        """Test anomaly detection with confidence threshold filtering."""
        # Arrange
        input_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model_id = "threshold_model_789"
        confidence_threshold = 0.8
        
        # Mock detector response with varying confidence scores
        mock_detector.detect.return_value = [
            Anomaly(index=0, confidence_score=0.6, anomaly_type=AnomalyType.POINT),  # Below threshold
            Anomaly(index=1, confidence_score=0.9, anomaly_type=AnomalyType.POINT),  # Above threshold
            Anomaly(index=2, confidence_score=0.85, anomaly_type=AnomalyType.POINT)  # Above threshold
        ]
        
        # Act
        anomalies = await detect_anomalies_use_case.execute(
            data=input_data,
            model_id=model_id,
            confidence_threshold=confidence_threshold
        )
        
        # Assert - Only anomalies above threshold should be returned
        assert len(anomalies) == 2
        assert all(a.confidence_score >= confidence_threshold for a in anomalies)
        assert anomalies[0].index == 1
        assert anomalies[1].index == 2
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_error_handling(self, detect_anomalies_use_case, mock_detector):
        """Test error handling in anomaly detection."""
        # Arrange
        input_data = np.array([[1.0, 2.0]])
        model_id = "error_model_999"
        
        # Mock detector to raise an exception
        mock_detector.detect.side_effect = ValueError("Model not found")
        
        # Act & Assert
        with pytest.raises(ValueError, match="Model not found"):
            await detect_anomalies_use_case.execute(
                data=input_data,
                model_id=model_id
            )
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_empty_data(self, detect_anomalies_use_case):
        """Test detection with empty input data."""
        # Arrange
        input_data = np.array([]).reshape(0, 3)
        model_id = "empty_model"
        
        # Act
        anomalies = await detect_anomalies_use_case.execute(
            data=input_data,
            model_id=model_id
        )
        
        # Assert
        assert len(anomalies) == 0


class TestTrainModelUseCase:
    """Test cases for the TrainModel use case."""
    
    @pytest.fixture
    def mock_model_repository(self):
        """Mock model repository."""
        repository = Mock()
        repository.save = AsyncMock()
        repository.get_by_id = AsyncMock()
        return repository
    
    @pytest.fixture
    def mock_trainer(self):
        """Mock model trainer."""
        trainer = Mock()
        trainer.train = AsyncMock()
        trainer.evaluate = AsyncMock()
        return trainer
    
    @pytest.fixture
    def train_model_use_case(self, mock_model_repository, mock_trainer):
        """Create TrainModelUseCase with mocked dependencies."""
        return TrainModelUseCase(
            model_repository=mock_model_repository,
            trainer=mock_trainer
        )
    
    @pytest.mark.asyncio
    async def test_train_model_success(self, train_model_use_case, mock_trainer, mock_model_repository):
        """Test successful model training."""
        # Arrange
        training_data = np.random.rand(100, 5)
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            contamination=0.1,
            random_state=42,
            hyperparameters={"n_estimators": 100}
        )
        model_name = "test_model"
        
        # Mock training results
        mock_trainer.train.return_value = {
            "model_id": "trained_model_123",
            "training_time": 45.2,
            "status": "completed"
        }
        
        mock_trainer.evaluate.return_value = DetectionMetrics(
            accuracy=0.92,
            precision=0.88,
            recall=0.94,
            f1_score=0.91,
            auc_roc=0.96
        )
        
        # Act
        result = await train_model_use_case.execute(
            training_data=training_data,
            algorithm_config=config,
            model_name=model_name
        )
        
        # Assert
        assert result["model_id"] == "trained_model_123"
        assert result["status"] == "completed"
        assert result["training_time"] == 45.2
        
        mock_trainer.train.assert_called_once_with(training_data, config, model_name)
        mock_trainer.evaluate.assert_called_once()
        mock_model_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_train_model_with_validation_split(self, train_model_use_case, mock_trainer):
        """Test model training with validation data split."""
        # Arrange
        training_data = np.random.rand(200, 3)
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.LOCAL_OUTLIER_FACTOR,
            contamination=0.05
        )
        model_name = "validation_model"
        validation_split = 0.2
        
        # Mock training with validation
        mock_trainer.train.return_value = {
            "model_id": "validated_model_456",
            "training_time": 78.5,
            "validation_score": 0.89,
            "status": "completed"
        }
        
        # Act
        result = await train_model_use_case.execute(
            training_data=training_data,
            algorithm_config=config,
            model_name=model_name,
            validation_split=validation_split
        )
        
        # Assert
        assert result["validation_score"] == 0.89
        assert "validation_split" in str(mock_trainer.train.call_args)
    
    @pytest.mark.asyncio
    async def test_train_model_hyperparameter_validation(self, train_model_use_case):
        """Test training with invalid hyperparameters."""
        # Arrange
        training_data = np.random.rand(50, 2)
        
        # Invalid configuration - contamination too high
        with pytest.raises(ValueError, match="Contamination must be between 0.0 and 0.5"):
            invalid_config = AlgorithmConfig(
                algorithm_type=AlgorithmType.ISOLATION_FOREST,
                contamination=0.8  # Invalid
            )
    
    @pytest.mark.asyncio
    async def test_train_model_insufficient_data(self, train_model_use_case, mock_trainer):
        """Test training with insufficient data."""
        # Arrange
        small_data = np.random.rand(5, 3)  # Too small dataset
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            contamination=0.1
        )
        
        # Mock trainer to raise error for insufficient data
        mock_trainer.train.side_effect = ValueError("Insufficient training data")
        
        # Act & Assert
        with pytest.raises(ValueError, match="Insufficient training data"):
            await train_model_use_case.execute(
                training_data=small_data,
                algorithm_config=config,
                model_name="small_data_model"
            )
    
    @pytest.mark.asyncio
    async def test_train_model_performance_metrics(self, train_model_use_case, mock_trainer):
        """Test training with comprehensive performance evaluation."""
        # Arrange
        training_data = np.random.rand(300, 4)
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ENSEMBLE,
            contamination=0.15,
            hyperparameters={
                "estimators": ["isolation_forest", "lof"],
                "voting": "soft"
            }
        )
        
        # Mock comprehensive evaluation metrics
        mock_trainer.evaluate.return_value = DetectionMetrics(
            accuracy=0.94,
            precision=0.91,
            recall=0.89,
            f1_score=0.90,
            auc_roc=0.95,
            false_positive_rate=0.06,
            false_negative_rate=0.11
        )
        
        mock_trainer.train.return_value = {
            "model_id": "ensemble_model_789",
            "status": "completed",
            "training_time": 156.8
        }
        
        # Act
        result = await train_model_use_case.execute(
            training_data=training_data,
            algorithm_config=config,
            model_name="ensemble_model",
            evaluate_performance=True
        )
        
        # Assert
        assert result["model_id"] == "ensemble_model_789"
        mock_trainer.evaluate.assert_called_once()


class TestApplicationServiceIntegration:
    """Integration tests for application services."""
    
    @pytest.mark.asyncio
    async def test_train_and_detect_workflow(self):
        """Test complete workflow: train model then detect anomalies."""
        # Arrange
        training_data = np.random.rand(150, 3)
        test_data = np.random.rand(20, 3)
        
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            contamination=0.1,
            random_state=42
        )
        
        # Mock repositories and services
        model_repo = Mock()
        anomaly_repo = Mock()
        trainer = Mock()
        detector = Mock()
        
        model_repo.save = AsyncMock()
        model_repo.get_by_id = AsyncMock()
        anomaly_repo.save_batch = AsyncMock()
        
        # Mock training
        trainer.train = AsyncMock(return_value={
            "model_id": "workflow_model_123",
            "status": "completed",
            "training_time": 23.5
        })
        trainer.evaluate = AsyncMock(return_value=DetectionMetrics(
            accuracy=0.88, precision=0.85, recall=0.90, f1_score=0.87
        ))
        
        # Mock detection
        detector.detect = AsyncMock(return_value=[
            Anomaly(index=5, confidence_score=0.92, anomaly_type=AnomalyType.POINT),
            Anomaly(index=12, confidence_score=0.87, anomaly_type=AnomalyType.POINT)
        ])
        
        # Create use cases
        train_use_case = TrainModelUseCase(model_repo, trainer)
        detect_use_case = DetectAnomaliesUseCase(model_repo, anomaly_repo, detector)
        
        # Act - Train model
        train_result = await train_use_case.execute(
            training_data=training_data,
            algorithm_config=config,
            model_name="workflow_test_model"
        )
        
        # Act - Detect anomalies
        anomalies = await detect_use_case.execute(
            data=test_data,
            model_id=train_result["model_id"]
        )
        
        # Assert
        assert train_result["model_id"] == "workflow_model_123"
        assert train_result["status"] == "completed"
        assert len(anomalies) == 2
        assert anomalies[0].index == 5
        assert anomalies[1].index == 12
        
        # Verify interactions
        trainer.train.assert_called_once()
        detector.detect.assert_called_once_with(test_data, "workflow_model_123")
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error propagation through application layer."""
        # Arrange
        model_repo = Mock()
        trainer = Mock()
        
        # Mock repository error
        model_repo.save = AsyncMock(side_effect=ConnectionError("Database unavailable"))
        trainer.train = AsyncMock(return_value={"model_id": "test", "status": "completed"})
        trainer.evaluate = AsyncMock(return_value=DetectionMetrics(
            accuracy=0.8, precision=0.8, recall=0.8, f1_score=0.8
        ))
        
        train_use_case = TrainModelUseCase(model_repo, trainer)
        
        # Act & Assert
        with pytest.raises(ConnectionError, match="Database unavailable"):
            await train_use_case.execute(
                training_data=np.random.rand(100, 2),
                algorithm_config=AlgorithmConfig(
                    algorithm_type=AlgorithmType.LOCAL_OUTLIER_FACTOR,
                    contamination=0.1
                ),
                model_name="error_test_model"
            )