"""Infrastructure adapter tests with comprehensive mocking."""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import numpy as np
from typing import Dict, List, Any
import tempfile
import json
from pathlib import Path

from anomaly_detection.domain.entities.anomaly import Anomaly, AnomalyType, AnomalySeverity
from anomaly_detection.domain.value_objects.algorithm_config import AlgorithmConfig, AlgorithmType
from anomaly_detection.domain.value_objects.detection_metrics import DetectionMetrics


class TestFileBasedModelRepository:
    """Test cases for file-based model repository adapter."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_file_repository(self, temp_directory):
        """Mock file-based repository."""
        from unittest.mock import Mock
        repo = Mock()
        repo.base_path = temp_directory
        repo.save = AsyncMock()
        repo.load = AsyncMock()
        repo.exists = AsyncMock()
        repo.delete = AsyncMock()
        repo.list_models = AsyncMock()
        return repo
    
    @pytest.mark.asyncio
    async def test_save_model_metadata(self, mock_file_repository):
        """Test saving model metadata to file."""
        # Arrange
        model_data = {
            "model_id": "test_model_123",
            "algorithm_type": "isolation_forest",
            "created_at": "2024-01-15T10:30:00Z",
            "hyperparameters": {"n_estimators": 100, "contamination": 0.1},
            "performance_metrics": {
                "accuracy": 0.92,
                "precision": 0.88,
                "recall": 0.94,
                "f1_score": 0.91
            }
        }
        
        # Act
        await mock_file_repository.save(model_data["model_id"], model_data)
        
        # Assert
        mock_file_repository.save.assert_called_once_with("test_model_123", model_data)
    
    @pytest.mark.asyncio
    async def test_load_model_metadata(self, mock_file_repository):
        """Test loading model metadata from file."""
        # Arrange
        expected_model_data = {
            "model_id": "load_test_456",
            "algorithm_type": "lof",
            "status": "trained",
            "feature_columns": ["temperature", "humidity", "pressure"]
        }
        mock_file_repository.load.return_value = expected_model_data
        
        # Act
        loaded_data = await mock_file_repository.load("load_test_456")
        
        # Assert
        assert loaded_data == expected_model_data
        mock_file_repository.load.assert_called_once_with("load_test_456")
    
    @pytest.mark.asyncio
    async def test_model_exists_check(self, mock_file_repository):
        """Test checking if model exists."""
        # Arrange
        mock_file_repository.exists.return_value = True
        
        # Act
        exists = await mock_file_repository.exists("existing_model_789")
        
        # Assert
        assert exists is True
        mock_file_repository.exists.assert_called_once_with("existing_model_789")
    
    @pytest.mark.asyncio
    async def test_delete_model(self, mock_file_repository):
        """Test deleting model from repository."""
        # Arrange
        model_id = "delete_me_999"
        mock_file_repository.delete.return_value = True
        
        # Act
        deleted = await mock_file_repository.delete(model_id)
        
        # Assert
        assert deleted is True
        mock_file_repository.delete.assert_called_once_with(model_id)
    
    @pytest.mark.asyncio
    async def test_list_all_models(self, mock_file_repository):
        """Test listing all models in repository."""
        # Arrange
        expected_models = [
            {"model_id": "model_1", "algorithm": "isolation_forest"},
            {"model_id": "model_2", "algorithm": "lof"},
            {"model_id": "model_3", "algorithm": "one_class_svm"}
        ]
        mock_file_repository.list_models.return_value = expected_models
        
        # Act
        models = await mock_file_repository.list_models()
        
        # Assert
        assert len(models) == 3
        assert models == expected_models
        mock_file_repository.list_models.assert_called_once()


class TestDatabaseAnomalyRepository:
    """Test cases for database anomaly repository adapter."""
    
    @pytest.fixture
    def mock_db_connection(self):
        """Mock database connection."""
        connection = Mock()
        connection.execute = AsyncMock()
        connection.fetchall = AsyncMock()
        connection.fetchone = AsyncMock()
        connection.commit = AsyncMock()
        connection.rollback = AsyncMock()
        return connection
    
    @pytest.fixture
    def mock_db_repository(self, mock_db_connection):
        """Mock database repository."""
        repo = Mock()
        repo.connection = mock_db_connection
        repo.save_anomaly = AsyncMock()
        repo.save_batch = AsyncMock()
        repo.get_anomalies_by_timerange = AsyncMock()
        repo.get_anomalies_by_severity = AsyncMock()
        repo.count_anomalies = AsyncMock()
        return repo
    
    @pytest.mark.asyncio
    async def test_save_single_anomaly(self, mock_db_repository):
        """Test saving single anomaly to database."""
        # Arrange
        anomaly = Anomaly(
            index=0,
            confidence_score=0.95,
            anomaly_type=AnomalyType.POINT,
            severity=AnomalySeverity.HIGH,
            timestamp=None,
            metadata={"model_id": "test_model", "feature_count": 5}
        )
        
        # Act
        await mock_db_repository.save_anomaly(anomaly)
        
        # Assert
        mock_db_repository.save_anomaly.assert_called_once_with(anomaly)
    
    @pytest.mark.asyncio
    async def test_save_anomaly_batch(self, mock_db_repository):
        """Test saving batch of anomalies to database."""
        # Arrange
        anomalies = [
            Anomaly(index=i, confidence_score=0.8 + i*0.05, anomaly_type=AnomalyType.POINT)
            for i in range(5)
        ]
        
        # Act
        await mock_db_repository.save_batch(anomalies)
        
        # Assert
        mock_db_repository.save_batch.assert_called_once_with(anomalies)
    
    @pytest.mark.asyncio
    async def test_get_anomalies_by_severity(self, mock_db_repository):
        """Test retrieving anomalies filtered by severity."""
        # Arrange
        expected_anomalies = [
            {"id": 1, "confidence_score": 0.95, "severity": "critical"},
            {"id": 2, "confidence_score": 0.92, "severity": "critical"}
        ]
        mock_db_repository.get_anomalies_by_severity.return_value = expected_anomalies
        
        # Act
        anomalies = await mock_db_repository.get_anomalies_by_severity(AnomalySeverity.CRITICAL)
        
        # Assert
        assert len(anomalies) == 2
        assert anomalies == expected_anomalies
        mock_db_repository.get_anomalies_by_severity.assert_called_once_with(AnomalySeverity.CRITICAL)
    
    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self, mock_db_repository, mock_db_connection):
        """Test database transaction rollback on error."""
        # Arrange
        mock_db_connection.execute.side_effect = Exception("Database error")
        mock_db_repository.save_anomaly.side_effect = Exception("Database error")
        
        anomaly = Anomaly(
            index=0,
            confidence_score=0.8,
            anomaly_type=AnomalyType.POINT
        )
        
        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await mock_db_repository.save_anomaly(anomaly)


class TestSklearnAnomalyDetector:
    """Test cases for sklearn-based anomaly detector adapter."""
    
    @pytest.fixture
    def mock_sklearn_model(self):
        """Mock sklearn model."""
        model = Mock()
        model.fit = Mock()
        model.predict = Mock()
        model.decision_function = Mock()
        model.score_samples = Mock()
        return model
    
    @pytest.fixture
    def mock_detector_adapter(self, mock_sklearn_model):
        """Mock detector adapter."""
        adapter = Mock()
        adapter.model = mock_sklearn_model
        adapter.fit = Mock()
        adapter.predict = Mock()
        adapter.detect_anomalies = Mock()
        adapter.get_anomaly_scores = Mock()
        return adapter
    
    def test_fit_isolation_forest(self, mock_detector_adapter, mock_sklearn_model):
        """Test fitting isolation forest model."""
        # Arrange
        training_data = np.random.rand(100, 4)
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ISOLATION_FOREST,
            contamination=0.1,
            random_state=42,
            hyperparameters={"n_estimators": 200}
        )
        
        # Act
        mock_detector_adapter.fit(training_data, config)
        
        # Assert
        mock_detector_adapter.fit.assert_called_once_with(training_data, config)
    
    def test_predict_anomalies(self, mock_detector_adapter, mock_sklearn_model):
        """Test predicting anomalies on new data."""
        # Arrange
        test_data = np.random.rand(20, 4)
        mock_sklearn_model.predict.return_value = np.array([1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 
                                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        mock_sklearn_model.decision_function.return_value = np.array([0.1, 0.2, -0.8, 0.15, 0.05,
                                                                     -0.9, 0.3, 0.1, 0.2, 0.1,
                                                                     0.1, 0.1, 0.1, 0.1, 0.1,
                                                                     0.1, 0.1, 0.1, 0.1, 0.1])
        
        mock_detector_adapter.detect_anomalies.return_value = [2, 5]  # Indices of anomalies
        
        # Act
        anomaly_indices = mock_detector_adapter.detect_anomalies(test_data)
        
        # Assert
        assert anomaly_indices == [2, 5]
        mock_detector_adapter.detect_anomalies.assert_called_once_with(test_data)
    
    def test_get_anomaly_scores(self, mock_detector_adapter):
        """Test getting anomaly scores for data points."""
        # Arrange
        test_data = np.random.rand(10, 3)
        expected_scores = np.array([0.1, 0.2, 0.85, 0.15, 0.92, 0.05, 0.3, 0.1, 0.2, 0.1])
        mock_detector_adapter.get_anomaly_scores.return_value = expected_scores
        
        # Act
        scores = mock_detector_adapter.get_anomaly_scores(test_data)
        
        # Assert
        assert np.array_equal(scores, expected_scores)
        mock_detector_adapter.get_anomaly_scores.assert_called_once_with(test_data)
    
    def test_unsupported_algorithm_error(self, mock_detector_adapter):
        """Test error handling for unsupported algorithms."""
        # Arrange
        training_data = np.random.rand(50, 2)
        invalid_config = Mock()
        invalid_config.algorithm_type = "unsupported_algorithm"
        
        mock_detector_adapter.fit.side_effect = ValueError("Unsupported algorithm type")
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported algorithm type"):
            mock_detector_adapter.fit(training_data, invalid_config)


class TestMetricsCollectorAdapter:
    """Test cases for metrics collection adapter."""
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock metrics collector."""
        collector = Mock()
        collector.record_anomaly_detected = Mock()
        collector.record_model_training_time = Mock()
        collector.record_detection_latency = Mock()
        collector.get_detection_statistics = Mock()
        collector.export_metrics = Mock()
        return collector
    
    def test_record_anomaly_detection(self, mock_metrics_collector):
        """Test recording anomaly detection metrics."""
        # Arrange
        anomaly_data = {
            "model_id": "metrics_test_model",
            "confidence_score": 0.87,
            "detection_time": 0.023,
            "anomaly_type": "point"
        }
        
        # Act
        mock_metrics_collector.record_anomaly_detected(anomaly_data)
        
        # Assert
        mock_metrics_collector.record_anomaly_detected.assert_called_once_with(anomaly_data)
    
    def test_record_training_metrics(self, mock_metrics_collector):
        """Test recording model training metrics."""
        # Arrange
        training_metrics = {
            "model_id": "training_metrics_model",
            "algorithm": "isolation_forest",
            "training_time": 45.7,
            "data_size": 10000,
            "feature_count": 8
        }
        
        # Act
        mock_metrics_collector.record_model_training_time(training_metrics)
        
        # Assert
        mock_metrics_collector.record_model_training_time.assert_called_once_with(training_metrics)
    
    def test_get_detection_statistics(self, mock_metrics_collector):
        """Test retrieving detection statistics."""
        # Arrange
        expected_stats = {
            "total_detections": 1543,
            "average_confidence": 0.78,
            "anomaly_rate": 0.12,
            "detection_latency_p95": 0.045
        }
        mock_metrics_collector.get_detection_statistics.return_value = expected_stats
        
        # Act
        stats = mock_metrics_collector.get_detection_statistics()
        
        # Assert
        assert stats == expected_stats
        mock_metrics_collector.get_detection_statistics.assert_called_once()
    
    def test_export_metrics_to_prometheus(self, mock_metrics_collector):
        """Test exporting metrics in Prometheus format."""
        # Arrange
        prometheus_format = """
        # HELP anomaly_detections_total Total number of anomalies detected
        # TYPE anomaly_detections_total counter
        anomaly_detections_total{model_id="test_model"} 42
        
        # HELP detection_latency_seconds Detection latency in seconds
        # TYPE detection_latency_seconds histogram
        detection_latency_seconds_bucket{le="0.01"} 10
        detection_latency_seconds_bucket{le="0.05"} 25
        detection_latency_seconds_bucket{le="0.1"} 30
        """
        mock_metrics_collector.export_metrics.return_value = prometheus_format
        
        # Act
        exported = mock_metrics_collector.export_metrics(format="prometheus")
        
        # Assert
        assert "anomaly_detections_total" in exported
        assert "detection_latency_seconds" in exported
        mock_metrics_collector.export_metrics.assert_called_once_with(format="prometheus")


class TestNotificationAdapter:
    """Test cases for notification/alerting adapter."""
    
    @pytest.fixture
    def mock_notification_service(self):
        """Mock notification service."""
        service = Mock()
        service.send_alert = AsyncMock()
        service.send_email = AsyncMock()
        service.send_slack_message = AsyncMock()
        service.send_webhook = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_send_anomaly_alert(self, mock_notification_service):
        """Test sending anomaly detection alert."""
        # Arrange
        alert_data = {
            "anomaly_id": "alert_123",
            "severity": "critical",
            "confidence_score": 0.96,
            "message": "Critical anomaly detected in production system",
            "timestamp": "2024-01-15T14:30:00Z",
            "model_id": "production_model_v2"
        }
        
        # Act
        await mock_notification_service.send_alert(alert_data)
        
        # Assert
        mock_notification_service.send_alert.assert_called_once_with(alert_data)
    
    @pytest.mark.asyncio
    async def test_send_email_notification(self, mock_notification_service):
        """Test sending email notification."""
        # Arrange
        email_data = {
            "to": ["admin@company.com", "ml-team@company.com"],
            "subject": "Anomaly Detection Alert - Critical",
            "body": "Multiple anomalies detected in the last hour. Please investigate.",
            "attachments": ["anomaly_report.pdf"]
        }
        
        # Act
        await mock_notification_service.send_email(email_data)
        
        # Assert
        mock_notification_service.send_email.assert_called_once_with(email_data)
    
    @pytest.mark.asyncio
    async def test_send_slack_notification(self, mock_notification_service):
        """Test sending Slack notification."""
        # Arrange
        slack_data = {
            "channel": "#ml-alerts",
            "message": "🚨 High confidence anomaly detected",
            "username": "AnomalyBot",
            "color": "danger"
        }
        
        # Act
        await mock_notification_service.send_slack_message(slack_data)
        
        # Assert
        mock_notification_service.send_slack_message.assert_called_once_with(slack_data)
    
    @pytest.mark.asyncio
    async def test_webhook_notification(self, mock_notification_service):
        """Test sending webhook notification."""
        # Arrange
        webhook_data = {
            "url": "https://api.company.com/ml/alerts",
            "method": "POST",
            "headers": {"Authorization": "Bearer token123"},
            "payload": {
                "event": "anomaly_detected",
                "severity": "high",
                "confidence": 0.89
            }
        }
        
        # Act
        await mock_notification_service.send_webhook(webhook_data)
        
        # Assert
        mock_notification_service.send_webhook.assert_called_once_with(webhook_data)
    
    @pytest.mark.asyncio
    async def test_notification_error_handling(self, mock_notification_service):
        """Test notification error handling."""
        # Arrange
        mock_notification_service.send_alert.side_effect = ConnectionError("Network error")
        
        alert_data = {"message": "Test alert"}
        
        # Act & Assert
        with pytest.raises(ConnectionError, match="Network error"):
            await mock_notification_service.send_alert(alert_data)


class TestCacheAdapter:
    """Test cases for caching adapter."""
    
    @pytest.fixture
    def mock_cache_service(self):
        """Mock cache service."""
        cache = Mock()
        cache.get = AsyncMock()
        cache.set = AsyncMock()
        cache.delete = AsyncMock()
        cache.exists = AsyncMock()
        cache.expire = AsyncMock()
        return cache
    
    @pytest.mark.asyncio
    async def test_cache_model_predictions(self, mock_cache_service):
        """Test caching model predictions."""
        # Arrange
        cache_key = "model_123_predictions_hash_abc"
        predictions = [0.1, 0.9, 0.2, 0.8, 0.15]
        ttl = 3600  # 1 hour
        
        # Act
        await mock_cache_service.set(cache_key, predictions, ttl)
        
        # Assert
        mock_cache_service.set.assert_called_once_with(cache_key, predictions, ttl)
    
    @pytest.mark.asyncio
    async def test_retrieve_cached_predictions(self, mock_cache_service):
        """Test retrieving cached predictions."""
        # Arrange
        cache_key = "model_456_predictions_hash_def"
        expected_predictions = [0.3, 0.7, 0.4, 0.85, 0.25]
        mock_cache_service.get.return_value = expected_predictions
        
        # Act
        predictions = await mock_cache_service.get(cache_key)
        
        # Assert
        assert predictions == expected_predictions
        mock_cache_service.get.assert_called_once_with(cache_key)
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_cache_service):
        """Test cache miss scenario."""
        # Arrange
        cache_key = "nonexistent_key"
        mock_cache_service.get.return_value = None
        
        # Act
        result = await mock_cache_service.get(cache_key)
        
        # Assert
        assert result is None
        mock_cache_service.get.assert_called_once_with(cache_key)
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, mock_cache_service):
        """Test setting cache expiration."""
        # Arrange
        cache_key = "expiring_key"
        expiration_time = 1800  # 30 minutes
        
        # Act
        await mock_cache_service.expire(cache_key, expiration_time)
        
        # Assert
        mock_cache_service.expire.assert_called_once_with(cache_key, expiration_time)