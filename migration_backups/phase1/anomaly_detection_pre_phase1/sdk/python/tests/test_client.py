"""Tests for the synchronous anomaly detection client."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from anomaly_detection_sdk import AnomalyDetectionClient, AlgorithmType
from anomaly_detection_sdk.models import DetectionResult, ModelInfo, TrainingRequest, BatchProcessingRequest
from anomaly_detection_sdk.exceptions import (
    ValidationError, APIError, ConnectionError, TimeoutError, AuthenticationError
)


@pytest.fixture
def client():
    """Create a test client."""
    return AnomalyDetectionClient(
        base_url="http://localhost:8000",
        timeout=30.0,
        max_retries=3
    )


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        [1.0, 2.0],
        [1.1, 2.1],
        [1.2, 1.9],
        [10.0, 20.0],  # Anomaly
        [0.9, 2.2]
    ]


@pytest.fixture
def mock_response():
    """Mock HTTP response."""
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "anomalies": [
            {
                "index": 3,
                "score": 0.8,
                "data_point": [10.0, 20.0],
                "confidence": 0.9,
                "timestamp": "2023-01-01T00:00:00Z"
            }
        ],
        "total_points": 5,
        "anomaly_count": 1,
        "algorithm_used": "isolation_forest",
        "execution_time": 0.1,
        "metadata": {}
    }
    return mock_resp


class TestAnomalyDetectionClient:
    """Test the AnomalyDetectionClient class."""

    def test_client_initialization(self):
        """Test client initialization with various parameters."""
        # Basic initialization
        client = AnomalyDetectionClient(base_url="http://localhost:8000")
        assert client.config.base_url == "http://localhost:8000"
        assert client.config.timeout == 30.0
        assert client.config.max_retries == 3

        # With all parameters
        client = AnomalyDetectionClient(
            base_url="http://example.com",
            api_key="test-key",
            timeout=60.0,
            max_retries=5,
            headers={"Custom": "Header"}
        )
        assert client.config.base_url == "http://example.com"
        assert client.config.api_key == "test-key"
        assert client.config.timeout == 60.0
        assert client.config.max_retries == 5
        assert client.config.headers["Custom"] == "Header"

    @patch('requests.Session.request')
    def test_detect_anomalies_success(self, mock_request, client, sample_data, mock_response):
        """Test successful anomaly detection."""
        mock_request.return_value = mock_response

        result = client.detect_anomalies(
            data=sample_data,
            algorithm=AlgorithmType.ISOLATION_FOREST,
            parameters={'contamination': 0.2}
        )

        # Verify request was made correctly
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]['method'] == 'POST'
        assert '/api/v1/detect' in call_args[1]['url']

        # Verify result
        assert isinstance(result, DetectionResult)
        assert result.anomaly_count == 1
        assert result.total_points == 5
        assert len(result.anomalies) == 1
        assert result.anomalies[0].index == 3
        assert result.anomalies[0].score == 0.8

    def test_detect_anomalies_empty_data(self, client):
        """Test detection with empty data."""
        with pytest.raises(ValidationError) as exc_info:
            client.detect_anomalies(data=[])
        
        assert "Data cannot be empty" in str(exc_info.value)
        assert exc_info.value.field == "data"

    def test_detect_anomalies_invalid_data_format(self, client):
        """Test detection with invalid data format."""
        invalid_data = [1, 2, 3]  # Should be list of lists

        with pytest.raises(ValidationError) as exc_info:
            client.detect_anomalies(data=invalid_data)
        
        assert "All data points must be lists" in str(exc_info.value)

    @patch('requests.Session.request')
    def test_detect_anomalies_api_error(self, mock_request, client, sample_data):
        """Test API error handling."""
        mock_resp = Mock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"detail": "Invalid algorithm"}
        mock_request.return_value = mock_resp

        with pytest.raises(APIError) as exc_info:
            client.detect_anomalies(data=sample_data)
        
        assert exc_info.value.status_code == 400
        assert "Invalid algorithm" in str(exc_info.value)

    @patch('requests.Session.request')
    def test_detect_anomalies_connection_error(self, mock_request, client, sample_data):
        """Test connection error handling."""
        mock_request.side_effect = ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            client.detect_anomalies(data=sample_data)

    @patch('requests.Session.request')
    def test_detect_anomalies_timeout_error(self, mock_request, client, sample_data):
        """Test timeout error handling."""
        from requests.exceptions import Timeout
        mock_request.side_effect = Timeout("Request timed out")

        with pytest.raises(TimeoutError):
            client.detect_anomalies(data=sample_data)

    @patch('requests.Session.request')
    def test_batch_detect(self, mock_request, client, sample_data, mock_response):
        """Test batch detection."""
        mock_request.return_value = mock_response

        request = BatchProcessingRequest(
            data=sample_data,
            algorithm=AlgorithmType.ISOLATION_FOREST,
            parameters={'contamination': 0.2},
            return_explanations=True
        )

        result = client.batch_detect(request)

        # Verify request
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert '/api/v1/batch/detect' in call_args[1]['url']

        # Verify result
        assert isinstance(result, DetectionResult)
        assert result.anomaly_count == 1

    @patch('requests.Session.request')
    def test_train_model(self, mock_request, client, sample_data):
        """Test model training."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model_id": "test-model-123",
            "training_time": 5.2,
            "performance_metrics": {"accuracy": 0.95},
            "validation_metrics": {"f1_score": 0.92},
            "model_info": {
                "model_id": "test-model-123",
                "algorithm": "isolation_forest",
                "created_at": "2023-01-01T00:00:00Z",
                "training_data_size": 1000,
                "performance_metrics": {"accuracy": 0.95},
                "hyperparameters": {"n_estimators": 100},
                "version": "1.0",
                "status": "trained"
            }
        }
        mock_request.return_value = mock_resp

        training_request = TrainingRequest(
            data=sample_data,
            algorithm=AlgorithmType.ISOLATION_FOREST,
            model_name="test-model"
        )

        result = client.train_model(training_request)

        # Verify request
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert '/api/v1/models/train' in call_args[1]['url']

        # Verify result
        assert result.model_id == "test-model-123"
        assert result.training_time == 5.2

    @patch('requests.Session.request')
    def test_get_model(self, mock_request, client):
        """Test getting model information."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model_id": "test-model-123",
            "algorithm": "isolation_forest",
            "created_at": "2023-01-01T00:00:00Z",
            "training_data_size": 1000,
            "performance_metrics": {"accuracy": 0.95},
            "hyperparameters": {"n_estimators": 100},
            "version": "1.0",
            "status": "trained"
        }
        mock_request.return_value = mock_resp

        result = client.get_model("test-model-123")

        # Verify request
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert '/api/v1/models/test-model-123' in call_args[1]['url']

        # Verify result
        assert isinstance(result, ModelInfo)
        assert result.model_id == "test-model-123"
        assert result.algorithm == AlgorithmType.ISOLATION_FOREST

    def test_get_model_empty_id(self, client):
        """Test getting model with empty ID."""
        with pytest.raises(ValidationError) as exc_info:
            client.get_model("")
        
        assert "Model ID is required" in str(exc_info.value)

    @patch('requests.Session.request')
    def test_list_models(self, mock_request, client):
        """Test listing models."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {
                    "model_id": "model-1",
                    "algorithm": "isolation_forest",
                    "created_at": "2023-01-01T00:00:00Z",
                    "training_data_size": 1000,
                    "performance_metrics": {"accuracy": 0.95},
                    "hyperparameters": {"n_estimators": 100},
                    "version": "1.0",
                    "status": "trained"
                },
                {
                    "model_id": "model-2",
                    "algorithm": "local_outlier_factor",
                    "created_at": "2023-01-01T01:00:00Z",
                    "training_data_size": 500,
                    "performance_metrics": {"accuracy": 0.88},
                    "hyperparameters": {"n_neighbors": 20},
                    "version": "1.0",
                    "status": "trained"
                }
            ]
        }
        mock_request.return_value = mock_resp

        result = client.list_models()

        # Verify request
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert '/api/v1/models' in call_args[1]['url']

        # Verify result
        assert len(result) == 2
        assert all(isinstance(model, ModelInfo) for model in result)
        assert result[0].model_id == "model-1"
        assert result[1].model_id == "model-2"

    @patch('requests.Session.request')
    def test_delete_model(self, mock_request, client):
        """Test model deletion."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": "Model deleted successfully"}
        mock_request.return_value = mock_resp

        result = client.delete_model("test-model-123")

        # Verify request
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]['method'] == 'DELETE'
        assert '/api/v1/models/test-model-123' in call_args[1]['url']

        # Verify result
        assert result["message"] == "Model deleted successfully"

    @patch('requests.Session.request')
    def test_explain_anomaly(self, mock_request, client):
        """Test anomaly explanation."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "anomaly_index": 0,
            "feature_importance": {"feature_0": 0.8, "feature_1": 0.2},
            "shap_values": [0.3, 0.1],
            "explanation_text": "High values in feature 0",
            "confidence": 0.9
        }
        mock_request.return_value = mock_resp

        result = client.explain_anomaly(
            data_point=[10.0, 20.0],
            algorithm=AlgorithmType.ISOLATION_FOREST,
            method="shap"
        )

        # Verify request
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert '/api/v1/explain' in call_args[1]['url']

        # Verify result
        assert result.anomaly_index == 0
        assert result.feature_importance["feature_0"] == 0.8
        assert result.explanation_text == "High values in feature 0"

    def test_explain_anomaly_empty_data_point(self, client):
        """Test explanation with empty data point."""
        with pytest.raises(ValidationError) as exc_info:
            client.explain_anomaly(data_point=[])
        
        assert "Data point must be an array" in str(exc_info.value)

    @patch('requests.Session.request')
    def test_get_health(self, mock_request, client):
        """Test health check."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "healthy",
            "timestamp": "2023-01-01T00:00:00Z",
            "version": "1.0.0",
            "uptime": 3600.5,
            "components": {"database": "healthy", "cache": "healthy"},
            "metrics": {"requests_per_second": 100}
        }
        mock_request.return_value = mock_resp

        result = client.get_health()

        # Verify request
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert '/api/v1/health' in call_args[1]['url']

        # Verify result
        assert result.status == "healthy"
        assert result.version == "1.0.0"
        assert result.uptime == 3600.5

    @patch('requests.Session.request')
    def test_get_metrics(self, mock_request, client):
        """Test getting metrics."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "requests_per_second": 100,
            "average_response_time": 0.05,
            "active_connections": 25
        }
        mock_request.return_value = mock_resp

        result = client.get_metrics()

        # Verify request
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert '/api/v1/metrics' in call_args[1]['url']

        # Verify result
        assert result["requests_per_second"] == 100
        assert result["average_response_time"] == 0.05

    @patch('requests.Session.request')
    def test_upload_data(self, mock_request, client, sample_data):
        """Test data upload."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "dataset_id": "dataset-123",
            "message": "Data uploaded successfully"
        }
        mock_request.return_value = mock_resp

        result = client.upload_data(
            data=sample_data,
            dataset_name="test-dataset",
            description="Test dataset for anomaly detection"
        )

        # Verify request
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert '/api/v1/data/upload' in call_args[1]['url']

        # Verify result
        assert result["dataset_id"] == "dataset-123"
        assert result["message"] == "Data uploaded successfully"

    def test_upload_data_empty_data(self, client):
        """Test upload with empty data."""
        with pytest.raises(ValidationError) as exc_info:
            client.upload_data(data=[], dataset_name="test")
        
        assert "Data cannot be empty" in str(exc_info.value)

    def test_upload_data_empty_name(self, client, sample_data):
        """Test upload with empty dataset name."""
        with pytest.raises(ValidationError) as exc_info:
            client.upload_data(data=sample_data, dataset_name="")
        
        assert "Dataset name is required" in str(exc_info.value)

    @patch('requests.Session.request')
    def test_authentication_error(self, mock_request, client, sample_data):
        """Test authentication error handling."""
        mock_resp = Mock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"detail": "Invalid API key"}
        mock_request.return_value = mock_resp

        with pytest.raises(APIError) as exc_info:
            client.detect_anomalies(data=sample_data)
        
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in str(exc_info.value)

    @patch('requests.Session.request')
    def test_rate_limit_error(self, mock_request, client, sample_data):
        """Test rate limit error handling."""
        mock_resp = Mock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "60"}
        mock_resp.json.return_value = {"detail": "Rate limit exceeded"}
        mock_request.return_value = mock_resp

        with pytest.raises(APIError) as exc_info:
            client.detect_anomalies(data=sample_data)
        
        assert exc_info.value.status_code == 429

    def test_context_manager(self):
        """Test client as context manager."""
        with AnomalyDetectionClient(base_url="http://localhost:8000") as client:
            assert client is not None
        # Client should be closed after exiting context

    @patch('requests.Session.request')
    def test_retry_logic(self, mock_request, client, sample_data):
        """Test retry logic on transient failures."""
        # First call fails, second succeeds
        mock_request.side_effect = [
            ConnectionError("Temporary failure"),
            mock_response
        ]

        result = client.detect_anomalies(data=sample_data)
        
        # Should have retried once
        assert mock_request.call_count == 2
        assert isinstance(result, DetectionResult)

    @patch('requests.Session.request')
    def test_max_retries_exceeded(self, mock_request, client, sample_data):
        """Test behavior when max retries are exceeded."""
        mock_request.side_effect = ConnectionError("Persistent failure")

        with pytest.raises(ConnectionError):
            client.detect_anomalies(data=sample_data)
        
        # Should have tried max_retries + 1 times
        assert mock_request.call_count == client.config.max_retries + 1

    def test_headers_configuration(self):
        """Test custom headers configuration."""
        custom_headers = {"Custom-Header": "custom-value"}
        client = AnomalyDetectionClient(
            base_url="http://localhost:8000",
            headers=custom_headers
        )
        
        assert "Custom-Header" in client.headers
        assert client.headers["Custom-Header"] == "custom-value"
        assert client.headers["Content-Type"] == "application/json"

    def test_api_key_configuration(self):
        """Test API key configuration."""
        client = AnomalyDetectionClient(
            base_url="http://localhost:8000",
            api_key="test-api-key"
        )
        
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer test-api-key"