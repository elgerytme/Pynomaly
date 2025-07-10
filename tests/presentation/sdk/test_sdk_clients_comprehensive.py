"""
Comprehensive tests for SDK clients.

This module provides extensive testing for both synchronous and asynchronous
SDK clients, including configuration, error handling, and API method coverage.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import tempfile
import os

from pynomaly.presentation.sdk.client import PynomályClient
from pynomaly.presentation.sdk.async_client import AsyncPynomályClient
from pynomaly.presentation.sdk.config import SDKConfig
from pynomaly.presentation.sdk.exceptions import (
    SDKError,
    AuthenticationError,
    ValidationError,
    NetworkError,
    TimeoutError,
    RateLimitError
)
from pynomaly.presentation.sdk.models import (
    Dataset,
    Detector,
    DetectionResult,
    TrainingJob,
    Ensemble
)


class TestSDKConfiguration:
    """Test SDK configuration management."""

    def test_sdk_config_creation(self):
        """Test SDK configuration creation."""
        config = SDKConfig(
            api_url="http://localhost:8000",
            api_key="test-key",
            timeout=30
        )
        
        assert config.api_url == "http://localhost:8000"
        assert config.api_key == "test-key"
        assert config.timeout == 30

    def test_sdk_config_from_dict(self):
        """Test SDK configuration from dictionary."""
        config_dict = {
            "api_url": "http://test:8000",
            "api_key": "test-key",
            "timeout": 60,
            "retries": 3
        }
        
        config = SDKConfig.from_dict(config_dict)
        
        assert config.api_url == "http://test:8000"
        assert config.api_key == "test-key"
        assert config.timeout == 60

    def test_sdk_config_from_file(self):
        """Test SDK configuration from file."""
        config_data = {
            "api_url": "http://file:8000",
            "api_key": "file-key",
            "timeout": 45
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = SDKConfig.from_file(config_file)
            
            assert config.api_url == "http://file:8000"
            assert config.api_key == "file-key"
            assert config.timeout == 45
        finally:
            os.unlink(config_file)

    def test_sdk_config_from_env(self):
        """Test SDK configuration from environment variables."""
        os.environ['PYNOMALY_API_URL'] = 'http://env:8000'
        os.environ['PYNOMALY_API_KEY'] = 'env-key'
        os.environ['PYNOMALY_TIMEOUT'] = '90'
        
        try:
            config = SDKConfig.from_env()
            
            assert config.api_url == "http://env:8000"
            assert config.api_key == "env-key"
            assert config.timeout == 90
        finally:
            # Clean up
            for key in ['PYNOMALY_API_URL', 'PYNOMALY_API_KEY', 'PYNOMALY_TIMEOUT']:
                if key in os.environ:
                    del os.environ[key]

    def test_sdk_config_validation(self):
        """Test SDK configuration validation."""
        # Test invalid URL
        with pytest.raises(ValidationError):
            SDKConfig(api_url="invalid-url", api_key="test-key")
        
        # Test missing API key
        with pytest.raises(ValidationError):
            SDKConfig(api_url="http://localhost:8000", api_key="")
        
        # Test invalid timeout
        with pytest.raises(ValidationError):
            SDKConfig(api_url="http://localhost:8000", api_key="test-key", timeout=-1)

    def test_sdk_config_defaults(self):
        """Test SDK configuration defaults."""
        config = SDKConfig(
            api_url="http://localhost:8000",
            api_key="test-key"
        )
        
        assert config.timeout == 30  # Default timeout
        assert config.retries == 3  # Default retries
        assert config.verify_ssl is True  # Default SSL verification

    def test_sdk_config_serialization(self):
        """Test SDK configuration serialization."""
        config = SDKConfig(
            api_url="http://localhost:8000",
            api_key="test-key",
            timeout=60
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict['api_url'] == "http://localhost:8000"
        assert config_dict['api_key'] == "test-key"
        assert config_dict['timeout'] == 60
        
        # Test from_dict round-trip
        config_restored = SDKConfig.from_dict(config_dict)
        assert config_restored.api_url == config.api_url
        assert config_restored.api_key == config.api_key
        assert config_restored.timeout == config.timeout


class TestSynchronousSDKClient:
    """Test synchronous SDK client."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SDKConfig(
            api_url="http://localhost:8000",
            api_key="test-key",
            timeout=30
        )

    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return PynomályClient(config)

    @pytest.fixture
    def mock_response(self):
        """Mock HTTP response."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": "success"}
        mock_resp.text = '{"message": "success"}'
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    def test_client_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client.session is not None
        assert client.base_url == "http://localhost:8000"

    def test_client_authentication(self, client, mock_response):
        """Test client authentication."""
        with patch('requests.Session.request', return_value=mock_response):
            # Test API key authentication
            response = client._make_request('GET', '/test')
            assert response.status_code == 200
            
            # Verify auth header was set
            client.session.request.assert_called_once()
            call_args = client.session.request.call_args
            assert 'Authorization' in call_args[1]['headers']
            assert call_args[1]['headers']['Authorization'] == 'Bearer test-key'

    def test_client_request_retry(self, client):
        """Test client request retry mechanism."""
        # Mock network error followed by success
        mock_error = Mock()
        mock_error.side_effect = [
            ConnectionError("Network error"),
            Mock(status_code=200, json=lambda: {"message": "success"})
        ]
        
        with patch('requests.Session.request', side_effect=mock_error):
            # Should retry and succeed
            response = client._make_request('GET', '/test')
            assert response.status_code == 200

    def test_client_timeout_handling(self, client):
        """Test client timeout handling."""
        with patch('requests.Session.request', side_effect=TimeoutError("Request timeout")):
            with pytest.raises(TimeoutError):
                client._make_request('GET', '/test')

    def test_client_rate_limit_handling(self, client):
        """Test client rate limit handling."""
        mock_resp = Mock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": "Rate limit exceeded"}
        mock_resp.headers = {"Retry-After": "60"}
        
        with patch('requests.Session.request', return_value=mock_resp):
            with pytest.raises(RateLimitError):
                client._make_request('GET', '/test')

    def test_client_error_handling(self, client):
        """Test client error handling."""
        # Test 401 Unauthorized
        mock_resp = Mock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": "Unauthorized"}
        
        with patch('requests.Session.request', return_value=mock_resp):
            with pytest.raises(AuthenticationError):
                client._make_request('GET', '/test')
        
        # Test 400 Bad Request
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "Bad request"}
        
        with patch('requests.Session.request', return_value=mock_resp):
            with pytest.raises(ValidationError):
                client._make_request('GET', '/test')

    # Dataset methods
    def test_client_list_datasets(self, client, mock_response):
        """Test list datasets method."""
        mock_response.json.return_value = [
            {"id": "dataset1", "name": "Dataset 1"},
            {"id": "dataset2", "name": "Dataset 2"}
        ]
        
        with patch('requests.Session.request', return_value=mock_response):
            datasets = client.list_datasets()
            assert len(datasets) == 2
            assert datasets[0]['id'] == 'dataset1'

    def test_client_get_dataset(self, client, mock_response):
        """Test get dataset method."""
        mock_response.json.return_value = {"id": "dataset1", "name": "Dataset 1"}
        
        with patch('requests.Session.request', return_value=mock_response):
            dataset = client.get_dataset("dataset1")
            assert dataset['id'] == 'dataset1'
            assert dataset['name'] == 'Dataset 1'

    def test_client_create_dataset(self, client, mock_response):
        """Test create dataset method."""
        mock_response.json.return_value = {"id": "new-dataset", "name": "New Dataset"}
        
        with patch('requests.Session.request', return_value=mock_response):
            dataset = client.create_dataset(
                name="New Dataset",
                data=[[1, 2, 3], [4, 5, 6]]
            )
            assert dataset['id'] == 'new-dataset'

    def test_client_upload_dataset(self, client, mock_response):
        """Test upload dataset method."""
        mock_response.json.return_value = {"id": "uploaded-dataset", "name": "Uploaded Dataset"}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("col1,col2\n1,2\n3,4\n")
            file_path = f.name
        
        try:
            with patch('requests.Session.request', return_value=mock_response):
                dataset = client.upload_dataset(
                    file_path=file_path,
                    name="Uploaded Dataset"
                )
                assert dataset['id'] == 'uploaded-dataset'
        finally:
            os.unlink(file_path)

    def test_client_delete_dataset(self, client, mock_response):
        """Test delete dataset method."""
        mock_response.status_code = 204
        
        with patch('requests.Session.request', return_value=mock_response):
            result = client.delete_dataset("dataset1")
            assert result is True

    # Detector methods
    def test_client_list_detectors(self, client, mock_response):
        """Test list detectors method."""
        mock_response.json.return_value = [
            {"id": "detector1", "name": "Detector 1"},
            {"id": "detector2", "name": "Detector 2"}
        ]
        
        with patch('requests.Session.request', return_value=mock_response):
            detectors = client.list_detectors()
            assert len(detectors) == 2
            assert detectors[0]['id'] == 'detector1'

    def test_client_create_detector(self, client, mock_response):
        """Test create detector method."""
        mock_response.json.return_value = {
            "id": "new-detector",
            "name": "New Detector",
            "algorithm": "isolation_forest"
        }
        
        with patch('requests.Session.request', return_value=mock_response):
            detector = client.create_detector(
                name="New Detector",
                algorithm="isolation_forest",
                contamination=0.1
            )
            assert detector['id'] == 'new-detector'

    def test_client_train_detector(self, client, mock_response):
        """Test train detector method."""
        mock_response.json.return_value = {
            "job_id": "training-job-1",
            "status": "started"
        }
        
        with patch('requests.Session.request', return_value=mock_response):
            job = client.train_detector("detector1", "dataset1")
            assert job['job_id'] == 'training-job-1'

    def test_client_detect_anomalies(self, client, mock_response):
        """Test detect anomalies method."""
        mock_response.json.return_value = {
            "anomaly_scores": [0.1, 0.2, 0.8, 0.1],
            "predictions": [0, 0, 1, 0]
        }
        
        with patch('requests.Session.request', return_value=mock_response):
            result = client.detect_anomalies(
                detector_id="detector1",
                data=[[1, 2], [3, 4], [100, 200], [5, 6]]
            )
            assert len(result['anomaly_scores']) == 4
            assert result['predictions'][2] == 1

    # Training methods
    def test_client_get_training_job(self, client, mock_response):
        """Test get training job method."""
        mock_response.json.return_value = {
            "id": "job1",
            "status": "running",
            "progress": 0.5
        }
        
        with patch('requests.Session.request', return_value=mock_response):
            job = client.get_training_job("job1")
            assert job['id'] == 'job1'
            assert job['status'] == 'running'

    def test_client_stop_training_job(self, client, mock_response):
        """Test stop training job method."""
        mock_response.json.return_value = {
            "id": "job1",
            "status": "stopped"
        }
        
        with patch('requests.Session.request', return_value=mock_response):
            job = client.stop_training_job("job1")
            assert job['status'] == 'stopped'

    # Ensemble methods
    def test_client_create_ensemble(self, client, mock_response):
        """Test create ensemble method."""
        mock_response.json.return_value = {
            "id": "ensemble1",
            "name": "Test Ensemble",
            "detectors": ["detector1", "detector2"]
        }
        
        with patch('requests.Session.request', return_value=mock_response):
            ensemble = client.create_ensemble(
                name="Test Ensemble",
                detector_ids=["detector1", "detector2"],
                aggregation_method="mean"
            )
            assert ensemble['id'] == 'ensemble1'

    def test_client_run_ensemble(self, client, mock_response):
        """Test run ensemble method."""
        mock_response.json.return_value = {
            "anomaly_scores": [0.2, 0.3, 0.7, 0.1],
            "predictions": [0, 0, 1, 0]
        }
        
        with patch('requests.Session.request', return_value=mock_response):
            result = client.run_ensemble(
                ensemble_id="ensemble1",
                data=[[1, 2], [3, 4], [100, 200], [5, 6]]
            )
            assert len(result['anomaly_scores']) == 4

    # AutoML methods
    def test_client_run_automl(self, client, mock_response):
        """Test run AutoML method."""
        mock_response.json.return_value = {
            "experiment_id": "automl-exp-1",
            "status": "started"
        }
        
        with patch('requests.Session.request', return_value=mock_response):
            experiment = client.run_automl(
                dataset_id="dataset1",
                target_column="anomaly",
                time_budget=3600
            )
            assert experiment['experiment_id'] == 'automl-exp-1'

    def test_client_get_automl_results(self, client, mock_response):
        """Test get AutoML results method."""
        mock_response.json.return_value = {
            "experiment_id": "automl-exp-1",
            "best_model": "isolation_forest",
            "score": 0.95
        }
        
        with patch('requests.Session.request', return_value=mock_response):
            results = client.get_automl_results("automl-exp-1")
            assert results['best_model'] == 'isolation_forest'

    # Export methods
    def test_client_export_dataset(self, client, mock_response):
        """Test export dataset method."""
        mock_response.content = b"csv,data\n1,2\n3,4\n"
        mock_response.headers = {"content-type": "text/csv"}
        
        with patch('requests.Session.request', return_value=mock_response):
            data = client.export_dataset("dataset1", format="csv")
            assert data == b"csv,data\n1,2\n3,4\n"

    def test_client_export_results(self, client, mock_response):
        """Test export results method."""
        mock_response.content = b'{"results": [0.1, 0.2, 0.8]}'
        mock_response.headers = {"content-type": "application/json"}
        
        with patch('requests.Session.request', return_value=mock_response):
            data = client.export_results("result1", format="json")
            assert b"results" in data

    # Utility methods
    def test_client_health_check(self, client, mock_response):
        """Test health check method."""
        mock_response.json.return_value = {"status": "healthy"}
        
        with patch('requests.Session.request', return_value=mock_response):
            health = client.health_check()
            assert health['status'] == 'healthy'

    def test_client_get_version(self, client, mock_response):
        """Test get version method."""
        mock_response.json.return_value = {"version": "1.0.0"}
        
        with patch('requests.Session.request', return_value=mock_response):
            version = client.get_version()
            assert version['version'] == '1.0.0'

    def test_client_context_manager(self, config):
        """Test client context manager."""
        with PynomályClient(config) as client:
            assert client.session is not None
        
        # Session should be closed after exiting context
        assert client.session is None

    def test_client_connection_pooling(self, client, mock_response):
        """Test client connection pooling."""
        with patch('requests.Session.request', return_value=mock_response):
            # Make multiple requests
            for i in range(5):
                response = client._make_request('GET', f'/test{i}')
                assert response.status_code == 200
            
            # Should reuse the same session
            assert client.session.request.call_count == 5


class TestAsynchronousSDKClient:
    """Test asynchronous SDK client."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SDKConfig(
            api_url="http://localhost:8000",
            api_key="test-key",
            timeout=30
        )

    @pytest.fixture
    def client(self, config):
        """Create test async client."""
        return AsyncPynomályClient(config)

    @pytest.fixture
    def mock_response(self):
        """Mock async HTTP response."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"message": "success"})
        mock_resp.text = AsyncMock(return_value='{"message": "success"}')
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    @pytest.mark.asyncio
    async def test_async_client_initialization(self, client, config):
        """Test async client initialization."""
        assert client.config == config
        assert client.base_url == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_async_client_authentication(self, client, mock_response):
        """Test async client authentication."""
        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            async with client:
                response = await client._make_request('GET', '/test')
                assert response.status == 200

    @pytest.mark.asyncio
    async def test_async_client_error_handling(self, client):
        """Test async client error handling."""
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_resp.json = AsyncMock(return_value={"error": "Unauthorized"})
        
        with patch('aiohttp.ClientSession.request', return_value=mock_resp):
            async with client:
                with pytest.raises(AuthenticationError):
                    await client._make_request('GET', '/test')

    @pytest.mark.asyncio
    async def test_async_client_timeout_handling(self, client):
        """Test async client timeout handling."""
        with patch('aiohttp.ClientSession.request', side_effect=asyncio.TimeoutError):
            async with client:
                with pytest.raises(TimeoutError):
                    await client._make_request('GET', '/test')

    @pytest.mark.asyncio
    async def test_async_client_list_datasets(self, client, mock_response):
        """Test async list datasets method."""
        mock_response.json.return_value = [
            {"id": "dataset1", "name": "Dataset 1"},
            {"id": "dataset2", "name": "Dataset 2"}
        ]
        
        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            async with client:
                datasets = await client.list_datasets()
                assert len(datasets) == 2
                assert datasets[0]['id'] == 'dataset1'

    @pytest.mark.asyncio
    async def test_async_client_create_dataset(self, client, mock_response):
        """Test async create dataset method."""
        mock_response.json.return_value = {"id": "new-dataset", "name": "New Dataset"}
        
        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            async with client:
                dataset = await client.create_dataset(
                    name="New Dataset",
                    data=[[1, 2, 3], [4, 5, 6]]
                )
                assert dataset['id'] == 'new-dataset'

    @pytest.mark.asyncio
    async def test_async_client_detect_anomalies(self, client, mock_response):
        """Test async detect anomalies method."""
        mock_response.json.return_value = {
            "anomaly_scores": [0.1, 0.2, 0.8, 0.1],
            "predictions": [0, 0, 1, 0]
        }
        
        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            async with client:
                result = await client.detect_anomalies(
                    detector_id="detector1",
                    data=[[1, 2], [3, 4], [100, 200], [5, 6]]
                )
                assert len(result['anomaly_scores']) == 4

    @pytest.mark.asyncio
    async def test_async_client_concurrent_requests(self, client, mock_response):
        """Test async client concurrent requests."""
        mock_response.json.return_value = {"message": "success"}
        
        with patch('aiohttp.ClientSession.request', return_value=mock_response):
            async with client:
                # Make concurrent requests
                tasks = [
                    client._make_request('GET', f'/test{i}')
                    for i in range(5)
                ]
                responses = await asyncio.gather(*tasks)
                
                # All requests should succeed
                for response in responses:
                    assert response.status == 200

    @pytest.mark.asyncio
    async def test_async_client_context_manager(self, config):
        """Test async client context manager."""
        async with AsyncPynomályClient(config) as client:
            assert client.session is not None
        
        # Session should be closed after exiting context
        assert client.session is None

    @pytest.mark.asyncio
    async def test_async_client_retry_mechanism(self, client):
        """Test async client retry mechanism."""
        # Mock network error followed by success
        mock_error = AsyncMock()
        mock_error.side_effect = [
            aiohttp.ClientError("Network error"),
            AsyncMock(status=200, json=AsyncMock(return_value={"message": "success"}))
        ]
        
        with patch('aiohttp.ClientSession.request', side_effect=mock_error):
            async with client:
                # Should retry and succeed
                response = await client._make_request('GET', '/test')
                assert response.status == 200

    @pytest.mark.asyncio
    async def test_async_client_streaming_responses(self, client):
        """Test async client streaming responses."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.content.read = AsyncMock(return_value=b"streaming data")
        
        with patch('aiohttp.ClientSession.request', return_value=mock_resp):
            async with client:
                response = await client._make_request('GET', '/stream')
                content = await response.content.read()
                assert content == b"streaming data"


class TestSDKModels:
    """Test SDK data models."""

    def test_dataset_model(self):
        """Test Dataset model."""
        dataset = Dataset(
            id="dataset1",
            name="Test Dataset",
            created_at=datetime.now(),
            size=1000,
            columns=["col1", "col2"],
            data_type="csv"
        )
        
        assert dataset.id == "dataset1"
        assert dataset.name == "Test Dataset"
        assert dataset.size == 1000
        assert len(dataset.columns) == 2

    def test_detector_model(self):
        """Test Detector model."""
        detector = Detector(
            id="detector1",
            name="Test Detector",
            algorithm="isolation_forest",
            contamination=0.1,
            created_at=datetime.now(),
            status="trained"
        )
        
        assert detector.id == "detector1"
        assert detector.algorithm == "isolation_forest"
        assert detector.contamination == 0.1
        assert detector.status == "trained"

    def test_detection_result_model(self):
        """Test DetectionResult model."""
        result = DetectionResult(
            id="result1",
            detector_id="detector1",
            dataset_id="dataset1",
            anomaly_scores=[0.1, 0.2, 0.8],
            predictions=[0, 0, 1],
            created_at=datetime.now(),
            processing_time=1.5
        )
        
        assert result.id == "result1"
        assert len(result.anomaly_scores) == 3
        assert len(result.predictions) == 3
        assert result.processing_time == 1.5

    def test_training_job_model(self):
        """Test TrainingJob model."""
        job = TrainingJob(
            id="job1",
            detector_id="detector1",
            dataset_id="dataset1",
            status="running",
            progress=0.5,
            created_at=datetime.now(),
            parameters={"epochs": 10, "batch_size": 32}
        )
        
        assert job.id == "job1"
        assert job.status == "running"
        assert job.progress == 0.5
        assert job.parameters["epochs"] == 10

    def test_ensemble_model(self):
        """Test Ensemble model."""
        ensemble = Ensemble(
            id="ensemble1",
            name="Test Ensemble",
            detector_ids=["detector1", "detector2"],
            aggregation_method="mean",
            created_at=datetime.now(),
            weights=[0.6, 0.4]
        )
        
        assert ensemble.id == "ensemble1"
        assert len(ensemble.detector_ids) == 2
        assert ensemble.aggregation_method == "mean"
        assert sum(ensemble.weights) == 1.0

    def test_model_serialization(self):
        """Test model serialization."""
        dataset = Dataset(
            id="dataset1",
            name="Test Dataset",
            created_at=datetime.now(),
            size=1000,
            columns=["col1", "col2"],
            data_type="csv"
        )
        
        # Test to_dict
        data_dict = dataset.to_dict()
        assert data_dict['id'] == "dataset1"
        assert data_dict['name'] == "Test Dataset"
        
        # Test from_dict
        dataset_restored = Dataset.from_dict(data_dict)
        assert dataset_restored.id == dataset.id
        assert dataset_restored.name == dataset.name

    def test_model_validation(self):
        """Test model validation."""
        # Test invalid contamination
        with pytest.raises(ValidationError):
            Detector(
                id="detector1",
                name="Test Detector",
                algorithm="isolation_forest",
                contamination=1.5,  # Invalid: > 1.0
                created_at=datetime.now(),
                status="untrained"
            )
        
        # Test invalid ensemble weights
        with pytest.raises(ValidationError):
            Ensemble(
                id="ensemble1",
                name="Test Ensemble",
                detector_ids=["detector1", "detector2"],
                aggregation_method="weighted_mean",
                created_at=datetime.now(),
                weights=[0.3, 0.3]  # Invalid: doesn't sum to 1.0
            )


class TestSDKExceptions:
    """Test SDK exception handling."""

    def test_sdk_error_hierarchy(self):
        """Test SDK error hierarchy."""
        # Test base exception
        base_error = SDKError("Base error")
        assert str(base_error) == "Base error"
        
        # Test specific exceptions
        auth_error = AuthenticationError("Auth failed")
        assert isinstance(auth_error, SDKError)
        
        validation_error = ValidationError("Validation failed")
        assert isinstance(validation_error, SDKError)
        
        network_error = NetworkError("Network failed")
        assert isinstance(network_error, SDKError)
        
        timeout_error = TimeoutError("Request timeout")
        assert isinstance(timeout_error, SDKError)
        
        rate_limit_error = RateLimitError("Rate limit exceeded")
        assert isinstance(rate_limit_error, SDKError)

    def test_exception_with_context(self):
        """Test exceptions with additional context."""
        error = ValidationError(
            "Validation failed",
            field="contamination",
            value=1.5,
            allowed_range="0.0-1.0"
        )
        
        assert error.field == "contamination"
        assert error.value == 1.5
        assert error.allowed_range == "0.0-1.0"

    def test_exception_serialization(self):
        """Test exception serialization."""
        error = NetworkError(
            "Connection failed",
            url="http://localhost:8000",
            status_code=503
        )
        
        error_dict = error.to_dict()
        assert error_dict['message'] == "Connection failed"
        assert error_dict['url'] == "http://localhost:8000"
        assert error_dict['status_code'] == 503

    def test_rate_limit_exception(self):
        """Test rate limit exception."""
        error = RateLimitError(
            "Rate limit exceeded",
            retry_after=60,
            limit=100,
            remaining=0
        )
        
        assert error.retry_after == 60
        assert error.limit == 100
        assert error.remaining == 0


class TestSDKIntegration:
    """Test SDK integration scenarios."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SDKConfig(
            api_url="http://localhost:8000",
            api_key="test-key",
            timeout=30
        )

    def test_end_to_end_workflow(self, config):
        """Test end-to-end workflow with SDK."""
        client = PynomályClient(config)
        
        # Mock all responses
        mock_responses = {
            'datasets': [{"id": "dataset1", "name": "Dataset 1"}],
            'dataset': {"id": "dataset1", "name": "Dataset 1"},
            'detector': {"id": "detector1", "name": "Detector 1"},
            'training_job': {"id": "job1", "status": "completed"},
            'detection_result': {"anomaly_scores": [0.1, 0.8], "predictions": [0, 1]}
        }
        
        def mock_request(method, url, **kwargs):
            mock_resp = Mock()
            mock_resp.status_code = 200
            
            if '/datasets' in url:
                mock_resp.json.return_value = mock_responses['datasets']
            elif '/detectors' in url:
                mock_resp.json.return_value = mock_responses['detector']
            elif '/training' in url:
                mock_resp.json.return_value = mock_responses['training_job']
            elif '/detect' in url:
                mock_resp.json.return_value = mock_responses['detection_result']
            else:
                mock_resp.json.return_value = {"message": "success"}
            
            return mock_resp
        
        with patch('requests.Session.request', side_effect=mock_request):
            # Step 1: List datasets
            datasets = client.list_datasets()
            assert len(datasets) == 1
            
            # Step 2: Create detector
            detector = client.create_detector(
                name="Test Detector",
                algorithm="isolation_forest"
            )
            assert detector['id'] == 'detector1'
            
            # Step 3: Train detector
            job = client.train_detector("detector1", "dataset1")
            assert job['id'] == 'job1'
            
            # Step 4: Detect anomalies
            result = client.detect_anomalies(
                detector_id="detector1",
                data=[[1, 2], [100, 200]]
            )
            assert len(result['anomaly_scores']) == 2

    @pytest.mark.asyncio
    async def test_async_end_to_end_workflow(self, config):
        """Test async end-to-end workflow with SDK."""
        client = AsyncPynomályClient(config)
        
        # Mock all responses
        mock_responses = {
            'datasets': [{"id": "dataset1", "name": "Dataset 1"}],
            'detector': {"id": "detector1", "name": "Detector 1"},
            'detection_result': {"anomaly_scores": [0.1, 0.8], "predictions": [0, 1]}
        }
        
        async def mock_request(method, url, **kwargs):
            mock_resp = AsyncMock()
            mock_resp.status = 200
            
            if '/datasets' in url:
                mock_resp.json.return_value = mock_responses['datasets']
            elif '/detectors' in url:
                mock_resp.json.return_value = mock_responses['detector']
            elif '/detect' in url:
                mock_resp.json.return_value = mock_responses['detection_result']
            else:
                mock_resp.json.return_value = {"message": "success"}
            
            return mock_resp
        
        with patch('aiohttp.ClientSession.request', side_effect=mock_request):
            async with client:
                # Step 1: List datasets
                datasets = await client.list_datasets()
                assert len(datasets) == 1
                
                # Step 2: Create detector
                detector = await client.create_detector(
                    name="Test Detector",
                    algorithm="isolation_forest"
                )
                assert detector['id'] == 'detector1'
                
                # Step 3: Detect anomalies
                result = await client.detect_anomalies(
                    detector_id="detector1",
                    data=[[1, 2], [100, 200]]
                )
                assert len(result['anomaly_scores']) == 2

    def test_sdk_performance_monitoring(self, config):
        """Test SDK performance monitoring."""
        client = PynomályClient(config)
        
        # Mock successful response
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": "success"}
        
        with patch('requests.Session.request', return_value=mock_resp):
            import time
            
            # Measure request time
            start_time = time.time()
            response = client._make_request('GET', '/test')
            end_time = time.time()
            
            # Should complete quickly
            assert (end_time - start_time) < 1.0
            assert response.status_code == 200

    def test_sdk_connection_management(self, config):
        """Test SDK connection management."""
        # Test connection pooling
        client = PynomályClient(config)
        
        # Mock response
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": "success"}
        
        with patch('requests.Session.request', return_value=mock_resp):
            # Make multiple requests
            responses = []
            for i in range(10):
                response = client._make_request('GET', f'/test{i}')
                responses.append(response)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
            
            # Should reuse the same session
            assert client.session.request.call_count == 10

    def test_sdk_error_recovery(self, config):
        """Test SDK error recovery."""
        client = PynomályClient(config)
        
        # Mock intermittent failures
        call_count = 0
        def mock_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:  # Fail first 2 requests
                raise ConnectionError("Connection failed")
            
            # Succeed on third request
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"message": "success"}
            return mock_resp
        
        with patch('requests.Session.request', side_effect=mock_request):
            # Should retry and eventually succeed
            response = client._make_request('GET', '/test')
            assert response.status_code == 200
            assert call_count == 3  # Should have retried twice