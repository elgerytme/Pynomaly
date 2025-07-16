"""
Test suite for Pynomaly Python SDK Client

Comprehensive tests for the main client functionality including
async operations, error handling, caching, and integration tests.
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import aiohttp

from src.packages.python_sdk.client import (
    PynomagyClient, 
    ClientConfig, 
    RetryConfig,
    create_client
)
from src.packages.python_sdk.application.dto.detection_dto import DetectionResponseDTO
from src.packages.python_sdk.domain.exceptions.validation_exceptions import ValidationError


class TestClientConfig:
    """Test cases for ClientConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ClientConfig()
        
        assert config.base_url == "https://api.monorepo.com/v1"
        assert config.api_key is None
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.backoff_factor == 2.0
        assert config.verify_ssl is True
        assert config.connection_pool_size == 10
        assert config.max_concurrent_requests == 10
        assert config.enable_caching is True
        assert config.cache_ttl == 300
        assert config.debug is False
        assert config.custom_headers == {}
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ClientConfig(
            base_url="https://custom.api.com",
            api_key="test-key",
            timeout=60,
            debug=True,
            custom_headers={"X-Custom": "header"}
        )
        
        assert config.base_url == "https://custom.api.com"
        assert config.api_key == "test-key"
        assert config.timeout == 60
        assert config.debug is True
        assert config.custom_headers == {"X-Custom": "header"}


class TestRetryConfig:
    """Test cases for RetryConfig."""
    
    def test_default_retry_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.backoff_factor == 2.0
        assert config.max_backoff == 60.0
        assert config.retry_on_status == [429, 500, 502, 503, 504]
        assert aiohttp.ClientError in config.retry_on_exceptions
        assert asyncio.TimeoutError in config.retry_on_exceptions


class TestPynomagyClient:
    """Test cases for PynomagyClient."""
    
    @pytest.fixture
    def client_config(self):
        """Create test client configuration."""
        return ClientConfig(
            base_url="https://test.api.com",
            api_key="test-key",
            timeout=10,
            debug=True
        )
    
    @pytest.fixture
    def client(self, client_config):
        """Create test client instance."""
        return PynomagyClient(client_config)
    
    def test_client_initialization(self, client_config):
        """Test client initialization."""
        client = PynomagyClient(client_config)
        
        assert client.config == client_config
        assert client.retry_config is not None
        assert client._session is None
        assert client._detection_service is None
        assert client._pyod_adapter is None
        assert client._cache == {}
        assert client._executor is not None
    
    def test_client_initialization_with_defaults(self):
        """Test client initialization with default config."""
        client = PynomagyClient()
        
        assert client.config.base_url == "https://api.monorepo.com/v1"
        assert client.config.api_key is None
        assert client.config.timeout == 30
    
    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test async context manager functionality."""
        with patch.object(client, '_initialize_session') as mock_init:
            with patch.object(client, '_cleanup') as mock_cleanup:
                mock_init.return_value = None
                mock_cleanup.return_value = None
                
                async with client:
                    assert mock_init.called
                
                assert mock_cleanup.called
    
    @pytest.mark.asyncio
    async def test_initialize_session(self, client):
        """Test session initialization."""
        with patch('aiohttp.ClientSession') as mock_session:
            with patch('aiohttp.TCPConnector') as mock_connector:
                mock_session_instance = Mock()
                mock_session.return_value = mock_session_instance
                
                await client._initialize_session()
                
                assert mock_session.called
                assert mock_connector.called
                assert client._session == mock_session_instance
                assert client._detection_service is not None
                assert client._pyod_adapter is not None
    
    @pytest.mark.asyncio
    async def test_cleanup(self, client):
        """Test resource cleanup."""
        mock_session = Mock()
        mock_session.close = AsyncMock()
        client._session = mock_session
        
        await client._cleanup()
        
        assert mock_session.close.called
    
    def test_get_cache_key(self, client):
        """Test cache key generation."""
        key1 = client._get_cache_key("GET", "/test")
        key2 = client._get_cache_key("POST", "/test", {"param": "value"})
        key3 = client._get_cache_key("GET", "/test")
        
        assert key1 == key3
        assert key1 != key2
        assert "GET" in key1
        assert "/test" in key1
        assert "param" in key2
    
    @pytest.mark.asyncio
    async def test_make_request_success(self, client):
        """Test successful HTTP request."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})
        
        mock_session = Mock()
        mock_session.request = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client._session = mock_session
        
        result = await client._make_request("GET", "/test")
        
        assert result == {"result": "success"}
        assert mock_session.request.called
    
    @pytest.mark.asyncio
    async def test_make_request_retry_on_error(self, client):
        """Test request retry on recoverable error."""
        mock_response = Mock()
        mock_response.status = 500
        mock_response.request_info = Mock()
        mock_response.history = []
        
        mock_session = Mock()
        mock_session.request = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client._session = mock_session
        client.retry_config.max_attempts = 2
        
        with pytest.raises(aiohttp.ClientResponseError):
            await client._make_request("GET", "/test")
        
        # Should have called request twice (original + 1 retry)
        assert mock_session.request.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cached_request_cache_hit(self, client):
        """Test cached request with cache hit."""
        # Setup cache
        cache_key = client._get_cache_key("GET", "/test")
        cached_result = {"cached": True}
        client._cache[cache_key] = (cached_result, 9999999999)  # Far future timestamp
        
        result = await client._cached_request("GET", "/test", use_cache=True)
        
        assert result == cached_result
    
    @pytest.mark.asyncio
    async def test_cached_request_cache_miss(self, client):
        """Test cached request with cache miss."""
        with patch.object(client, '_make_request') as mock_make_request:
            mock_make_request.return_value = {"result": "new"}
            
            result = await client._cached_request("GET", "/test", use_cache=True)
            
            assert result == {"result": "new"}
            assert mock_make_request.called
            
            # Check that result is cached
            cache_key = client._get_cache_key("GET", "/test")
            assert cache_key in client._cache
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_with_dataframe(self, client):
        """Test anomaly detection with pandas DataFrame."""
        # Create test data
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        # Mock the detection service
        mock_service = Mock()
        mock_response = DetectionResponseDTO(
            request_id="test-123",
            anomaly_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
            anomaly_labels=[0, 0, 0, 1, 1],
            algorithm_used="isolation_forest",
            execution_time=0.5,
            metadata={}
        )
        mock_service.detect_anomalies = AsyncMock(return_value=mock_response)
        client._detection_service = mock_service
        
        result = await client.detect_anomalies(df, algorithm="isolation_forest")
        
        assert isinstance(result, DetectionResponseDTO)
        assert result.request_id == "test-123"
        assert len(result.anomaly_scores) == 5
        assert len(result.anomaly_labels) == 5
        assert result.algorithm_used == "isolation_forest"
        assert mock_service.detect_anomalies.called
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_with_numpy_array(self, client):
        """Test anomaly detection with numpy array."""
        # Create test data
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Mock the detection service
        mock_service = Mock()
        mock_response = DetectionResponseDTO(
            request_id="test-456",
            anomaly_scores=[0.1, 0.2, 0.3],
            anomaly_labels=[0, 0, 1],
            algorithm_used="lof",
            execution_time=0.3,
            metadata={}
        )
        mock_service.detect_anomalies = AsyncMock(return_value=mock_response)
        client._detection_service = mock_service
        
        result = await client.detect_anomalies(data, algorithm="lof")
        
        assert isinstance(result, DetectionResponseDTO)
        assert result.request_id == "test-456"
        assert len(result.anomaly_scores) == 3
        assert result.algorithm_used == "lof"
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_batch(self, client):
        """Test batch anomaly detection."""
        # Create test datasets
        datasets = [
            {
                "data": [[1, 2], [3, 4]],
                "algorithm": "isolation_forest"
            },
            {
                "data": [[5, 6], [7, 8]],
                "algorithm": "lof"
            }
        ]
        
        # Mock the detection service
        mock_service = Mock()
        mock_responses = [
            DetectionResponseDTO(
                request_id="batch-1",
                anomaly_scores=[0.1, 0.2],
                anomaly_labels=[0, 1],
                algorithm_used="isolation_forest",
                execution_time=0.1,
                metadata={}
            ),
            DetectionResponseDTO(
                request_id="batch-2",
                anomaly_scores=[0.3, 0.4],
                anomaly_labels=[1, 0],
                algorithm_used="lof",
                execution_time=0.2,
                metadata={}
            )
        ]
        mock_service.detect_anomalies = AsyncMock(side_effect=mock_responses)
        client._detection_service = mock_service
        
        # Mock progress callback
        progress_callback = Mock()
        
        results = await client.detect_anomalies_batch(
            datasets, 
            max_concurrent=2,
            progress_callback=progress_callback
        )
        
        assert len(results) == 2
        assert all(isinstance(r, DetectionResponseDTO) for r in results)
        assert progress_callback.call_count == 2
    
    @pytest.mark.asyncio
    async def test_assess_data_quality_with_dataframe(self, client):
        """Test data quality assessment with DataFrame."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, None, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        expected_result = {
            "completeness": 0.9,
            "uniqueness": 1.0,
            "validity": 0.8,
            "consistency": 0.95
        }
        
        with patch.object(client, '_cached_request') as mock_request:
            mock_request.return_value = expected_result
            
            result = await client.assess_data_quality(df)
            
            assert result == expected_result
            assert mock_request.called
            
            # Check that the request was made with correct data
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/data-quality/assess"
    
    @pytest.mark.asyncio
    async def test_assess_data_quality_with_numpy_array(self, client):
        """Test data quality assessment with numpy array."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        expected_result = {
            "completeness": 1.0,
            "uniqueness": 1.0,
            "validity": 1.0,
            "consistency": 1.0
        }
        
        with patch.object(client, '_cached_request') as mock_request:
            mock_request.return_value = expected_result
            
            result = await client.assess_data_quality(data)
            
            assert result == expected_result
            assert mock_request.called
    
    @pytest.mark.asyncio
    async def test_assess_data_quality_invalid_data(self, client):
        """Test data quality assessment with invalid data type."""
        invalid_data = "invalid_data"
        
        with pytest.raises(ValidationError):
            await client.assess_data_quality(invalid_data)
    
    @pytest.mark.asyncio
    async def test_evaluate_model_performance(self, client):
        """Test model performance evaluation."""
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        
        expected_result = {
            "accuracy": 0.85,
            "precision": 0.8,
            "recall": 0.9,
            "f1_score": 0.84
        }
        
        with patch.object(client, '_cached_request') as mock_request:
            mock_request.return_value = expected_result
            
            result = await client.evaluate_model_performance(
                model_id="test-model-123",
                test_data=test_data,
                metrics=["accuracy", "precision", "recall", "f1_score"]
            )
            
            assert result == expected_result
            assert mock_request.called
            
            # Check that the request was made with correct data
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/model-performance/evaluate"
    
    @pytest.mark.asyncio
    async def test_list_available_algorithms(self, client):
        """Test listing available algorithms."""
        expected_algorithms = [
            {"name": "isolation_forest", "description": "Isolation Forest"},
            {"name": "lof", "description": "Local Outlier Factor"},
            {"name": "one_class_svm", "description": "One-Class SVM"}
        ]
        
        with patch.object(client, '_cached_request') as mock_request:
            mock_request.return_value = {"algorithms": expected_algorithms}
            
            result = await client.list_available_algorithms()
            
            assert result == expected_algorithms
            assert mock_request.called
    
    @pytest.mark.asyncio
    async def test_get_algorithm_info(self, client):
        """Test getting algorithm information."""
        expected_info = {
            "name": "isolation_forest",
            "description": "Isolation Forest algorithm",
            "parameters": {
                "n_estimators": {"type": "int", "default": 100},
                "contamination": {"type": "float", "default": 0.1}
            }
        }
        
        with patch.object(client, '_cached_request') as mock_request:
            mock_request.return_value = expected_info
            
            result = await client.get_algorithm_info("isolation_forest")
            
            assert result == expected_info
            assert mock_request.called
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test API health check."""
        expected_health = {
            "status": "healthy",
            "timestamp": "2023-01-01T00:00:00Z",
            "version": "1.0.0"
        }
        
        with patch.object(client, '_cached_request') as mock_request:
            mock_request.return_value = expected_health
            
            result = await client.health_check()
            
            assert result == expected_health
            assert mock_request.called
    
    def test_clear_cache(self, client):
        """Test cache clearing."""
        # Add some items to cache
        client._cache["key1"] = ("value1", 123456789)
        client._cache["key2"] = ("value2", 123456790)
        
        assert len(client._cache) == 2
        
        client.clear_cache()
        
        assert len(client._cache) == 0
    
    def test_get_cache_stats(self, client):
        """Test cache statistics."""
        # Add some items to cache
        client._cache["key1"] = ("value1", 123456789)
        client._cache["key2"] = ("value2", 123456790)
        
        stats = client.get_cache_stats()
        
        assert stats["cache_size"] == 2
        assert stats["cache_enabled"] == client.config.enable_caching
        assert stats["cache_ttl"] == client.config.cache_ttl


class TestCreateClient:
    """Test cases for create_client convenience function."""
    
    def test_create_client_with_defaults(self):
        """Test creating client with default configuration."""
        client = create_client()
        
        assert isinstance(client, PynomagyClient)
        assert client.config.base_url == "https://api.monorepo.com/v1"
        assert client.config.api_key is None
    
    def test_create_client_with_custom_params(self):
        """Test creating client with custom parameters."""
        client = create_client(
            api_key="test-key",
            base_url="https://custom.api.com",
            timeout=60,
            debug=True
        )
        
        assert isinstance(client, PynomagyClient)
        assert client.config.api_key == "test-key"
        assert client.config.base_url == "https://custom.api.com"
        assert client.config.timeout == 60
        assert client.config.debug is True


class TestIntegration:
    """Integration tests for the Python SDK."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_mock_api(self):
        """Test complete workflow with mocked API responses."""
        # Create client
        client = create_client(api_key="test-key", debug=True)
        
        # Mock API responses
        mock_health = {"status": "healthy"}
        mock_algorithms = {"algorithms": [{"name": "isolation_forest"}]}
        mock_detection = {
            "request_id": "test-123",
            "anomaly_scores": [0.1, 0.2, 0.3],
            "anomaly_labels": [0, 0, 1],
            "algorithm_used": "isolation_forest",
            "execution_time": 0.5,
            "metadata": {}
        }
        
        with patch.object(client, '_cached_request') as mock_request:
            mock_request.side_effect = [mock_health, mock_algorithms, mock_detection]
            
            # Test health check
            health = await client.health_check()
            assert health["status"] == "healthy"
            
            # Test list algorithms
            algorithms = await client.list_available_algorithms()
            assert len(algorithms) == 1
            assert algorithms[0]["name"] == "isolation_forest"
            
            # Test detect anomalies (will fall back to API since no local service)
            test_data = [[1, 2], [3, 4], [5, 6]]
            
            # Mock the detection service to fail so it falls back to API
            mock_service = Mock()
            mock_service.detect_anomalies = AsyncMock(side_effect=Exception("Service unavailable"))
            client._detection_service = mock_service
            
            result = await client.detect_anomalies(test_data, algorithm="isolation_forest")
            
            assert isinstance(result, DetectionResponseDTO)
            assert result.request_id == "test-123"
            assert len(result.anomaly_scores) == 3
    
    @pytest.mark.asyncio
    async def test_error_handling_and_retries(self):
        """Test error handling and retry mechanisms."""
        config = ClientConfig(timeout=1, max_retries=2)
        client = PynomagyClient(config)
        
        # Mock session with failing requests
        mock_response = Mock()
        mock_response.status = 500
        mock_response.request_info = Mock()
        mock_response.history = []
        
        mock_session = Mock()
        mock_session.request = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client._session = mock_session
        
        # Test that retries are attempted
        with pytest.raises(aiohttp.ClientResponseError):
            await client._make_request("GET", "/test")
        
        # Should have made 2 attempts (original + 1 retry)
        assert mock_session.request.call_count == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request handling."""
        client = create_client(max_concurrent_requests=3)
        
        # Create multiple datasets for batch processing
        datasets = [
            {"data": [[1, 2]], "algorithm": "isolation_forest"},
            {"data": [[3, 4]], "algorithm": "lof"},
            {"data": [[5, 6]], "algorithm": "one_class_svm"}
        ]
        
        # Mock detection service to simulate processing
        mock_service = Mock()
        mock_responses = [
            DetectionResponseDTO(
                request_id=f"batch-{i}",
                anomaly_scores=[0.1 * i],
                anomaly_labels=[i % 2],
                algorithm_used=dataset["algorithm"],
                execution_time=0.1,
                metadata={}
            )
            for i, dataset in enumerate(datasets)
        ]
        mock_service.detect_anomalies = AsyncMock(side_effect=mock_responses)
        client._detection_service = mock_service
        
        # Test batch processing
        results = await client.detect_anomalies_batch(datasets, max_concurrent=2)
        
        assert len(results) == 3
        assert all(isinstance(r, DetectionResponseDTO) for r in results)
        
        # Verify all requests were processed
        assert mock_service.detect_anomalies.call_count == 3