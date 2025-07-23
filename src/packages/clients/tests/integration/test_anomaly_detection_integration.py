#!/usr/bin/env python3
"""
Integration tests for the anomaly detection client.

These tests verify that the client works correctly against a real or mock API server.
"""

import asyncio
import os
import pytest
import sys
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

# Add SDK packages to path
sdk_core_path = Path(__file__).parent.parent.parent / "shared" / "sdk_core" / "src"
client_path = Path(__file__).parent.parent.parent / "clients" / "anomaly_detection_client" / "src"
sys.path.insert(0, str(sdk_core_path))
sys.path.insert(0, str(client_path))

from anomaly_detection_client import (
    AnomalyDetectionClient,
    AnomalyDetectionSyncClient,
    DetectionResponse,
    TrainingResponse,
    ModelInfo,
    AlgorithmInfo,
    ValidationError,
    RateLimitError,
    ServerError,
)
from sdk_core import ClientConfig, Environment


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Normal points in 0-10 range
    normal_data = [[i + 0.5, i + 1.0] for i in range(20)]
    # Clear anomalies
    anomalies = [[100.0, 100.0], [-50.0, -50.0]]
    return normal_data + anomalies


@pytest.fixture
def mock_client_config():
    """Create a test client configuration."""
    return ClientConfig.for_environment(
        Environment.LOCAL,
        api_key=os.getenv("TEST_INTEGRATION_API_KEY", "test_api_key_placeholder"),
        timeout=30.0,
        max_retries=1  # Reduce retries for faster tests
    )


@pytest.fixture
def mock_detection_response():
    """Mock detection response."""
    return {
        "success": True,
        "anomalies": [20, 21],  # Last two indices (the anomalies)
        "scores": [0.1] * 20 + [0.9, 0.8],  # High scores for anomalies
        "algorithm": "isolation_forest",
        "total_samples": 22,
        "anomaly_count": 2,
        "processing_time_ms": 156.7,
        "timestamp": "2025-01-01T12:00:00Z",
        "request_id": "test-request-123"
    }


@pytest.fixture
def mock_training_response():
    """Mock training response."""
    return {
        "success": True,
        "model": {
            "id": "model-123",
            "name": "test_model",
            "version": "1.0.0",
            "algorithm": "isolation_forest",
            "status": "trained",
            "created_at": "2025-01-01T12:00:00Z",
            "contamination": 0.1,
            "training_samples": 20,
            "parameters": {"n_estimators": 100}
        },
        "training_metrics": {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.91
        },
        "training_time_ms": 2500.0,
        "timestamp": "2025-01-01T12:00:00Z"
    }


@pytest.fixture
def mock_algorithms_response():
    """Mock algorithms response."""
    return {
        "success": True,
        "data": [
            {
                "name": "isolation_forest",
                "display_name": "Isolation Forest",
                "description": "Isolation-based anomaly detection",
                "parameters": {
                    "n_estimators": {
                        "type": "int",
                        "default": 100,
                        "range": [10, 1000]
                    }
                },
                "supports_online_learning": False,
                "supports_feature_importance": True,
                "computational_complexity": "O(n log n)"
            }
        ]
    }


class TestAnomalyDetectionIntegration:
    """Integration tests for anomaly detection client."""
    
    @pytest.mark.asyncio
    async def test_basic_detection(self, mock_client_config, sample_data, mock_detection_response):
        """Test basic anomaly detection functionality."""
        
        with patch('sdk_core.client.BaseClient._make_request') as mock_request:
            mock_request.return_value = mock_detection_response
            
            async with AnomalyDetectionClient(config=mock_client_config) as client:
                result = await client.detect(
                    data=sample_data,
                    algorithm="isolation_forest",
                    contamination=0.1
                )
                
                # Verify the response structure
                assert isinstance(result, DetectionResponse)
                assert result.success is True
                assert result.anomalies == [20, 21]
                assert result.total_samples == 22
                assert result.anomaly_count == 2
                assert result.algorithm == "isolation_forest"
                assert result.processing_time_ms > 0
                
                # Verify the request was made correctly
                mock_request.assert_called_once()
                call_args = mock_request.call_args
                assert call_args[0][0] == "POST"  # Method
                assert "detect" in call_args[0][1]  # URL contains 'detect'
                
                # Verify request data
                request_data = call_args[1]["json_data"]
                assert request_data["algorithm"] == "isolation_forest"
                assert request_data["contamination"] == 0.1
                assert len(request_data["data"]) == 22
    
    @pytest.mark.asyncio
    async def test_ensemble_detection(self, mock_client_config, sample_data):
        """Test ensemble detection functionality."""
        
        ensemble_response = {
            "success": True,
            "anomalies": [20, 21],
            "ensemble_scores": [0.1] * 20 + [0.9, 0.8],
            "individual_results": {
                "isolation_forest": {"anomalies": [20, 21], "scores": [0.1] * 22},
                "one_class_svm": {"anomalies": [20], "scores": [0.2] * 22}
            },
            "voting_strategy": "majority",
            "algorithms_used": ["isolation_forest", "one_class_svm"],
            "total_samples": 22,
            "anomaly_count": 2,
            "processing_time_ms": 245.8,
            "timestamp": "2025-01-01T12:00:00Z"
        }
        
        with patch('sdk_core.client.BaseClient._make_request') as mock_request:
            mock_request.return_value = ensemble_response
            
            async with AnomalyDetectionClient(config=mock_client_config) as client:
                result = await client.detect_ensemble(
                    data=sample_data,
                    algorithms=["isolation_forest", "one_class_svm"],
                    voting_strategy="majority"
                )
                
                assert result.success is True
                assert result.anomalies == [20, 21]
                assert result.voting_strategy == "majority"
                assert "isolation_forest" in result.algorithms_used
                assert "one_class_svm" in result.algorithms_used
                assert len(result.individual_results) == 2
    
    @pytest.mark.asyncio
    async def test_model_training(self, mock_client_config, sample_data, mock_training_response):
        """Test model training functionality."""
        
        with patch('sdk_core.client.BaseClient._make_request') as mock_request:
            mock_request.return_value = mock_training_response
            
            async with AnomalyDetectionClient(config=mock_client_config) as client:
                result = await client.train_model(
                    data=sample_data,
                    algorithm="isolation_forest",
                    name="test_model",
                    description="Test model for integration testing"
                )
                
                assert isinstance(result, TrainingResponse)
                assert result.success is True
                assert isinstance(result.model, ModelInfo)
                assert result.model.name == "test_model"
                assert result.model.algorithm == "isolation_forest"
                assert result.training_time_ms > 0
                assert "accuracy" in result.training_metrics
    
    @pytest.mark.asyncio
    async def test_model_prediction(self, mock_client_config, sample_data):
        """Test prediction with trained model."""
        
        prediction_response = {
            "success": True,
            "anomalies": [1, 2],
            "scores": [0.9, 0.8, 0.1, 0.1, 0.1],
            "model_id": "model-123",
            "model_name": "test_model",
            "total_samples": 5,
            "anomaly_count": 2,
            "processing_time_ms": 89.3,
            "timestamp": "2025-01-01T12:00:00Z"
        }
        
        with patch('sdk_core.client.BaseClient._make_request') as mock_request:
            mock_request.return_value = prediction_response
            
            async with AnomalyDetectionClient(config=mock_client_config) as client:
                result = await client.predict(
                    data=sample_data[:5],  # Use subset for prediction
                    model_id="model-123"
                )
                
                assert result.success is True
                assert result.model_id == "model-123"
                assert result.model_name == "test_model"
                assert result.anomaly_count == 2
                assert len(result.scores) == 5
    
    @pytest.mark.asyncio
    async def test_get_algorithms(self, mock_client_config, mock_algorithms_response):
        """Test getting available algorithms."""
        
        with patch('sdk_core.client.BaseClient._make_request') as mock_request:
            mock_request.return_value = mock_algorithms_response
            
            async with AnomalyDetectionClient(config=mock_client_config) as client:
                algorithms = await client.get_algorithms()
                
                assert len(algorithms) == 1
                algo = algorithms[0]
                assert isinstance(algo, AlgorithmInfo)
                assert algo.name == "isolation_forest"
                assert algo.display_name == "Isolation Forest"
                assert algo.supports_feature_importance is True
                assert "n_estimators" in algo.parameters
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_client_config):
        """Test health check functionality."""
        
        health_response = {
            "success": True,
            "status": "healthy",
            "version": "1.0.0",
            "uptime": 3600.5,
            "checks": {
                "database": {"status": "healthy", "response_time": 15.2},
                "redis": {"status": "healthy", "response_time": 2.1}
            },
            "timestamp": "2025-01-01T12:00:00Z"
        }
        
        with patch('sdk_core.client.BaseClient.health_check') as mock_health:
            mock_health.return_value = health_response
            
            async with AnomalyDetectionClient(config=mock_client_config) as client:
                result = await client.health_check()
                
                assert result["success"] is True
                assert result["status"] == "healthy"
                assert "checks" in result
    
    @pytest.mark.asyncio
    async def test_error_handling_validation_error(self, mock_client_config, sample_data):
        """Test handling of validation errors."""
        
        from sdk_core.exceptions import ValidationError as SDKValidationError
        
        with patch('sdk_core.client.BaseClient._make_request') as mock_request:
            mock_request.side_effect = SDKValidationError(
                "Invalid data format",
                details=[{"field": "data", "code": "invalid_format", "message": "Data must be 2D array"}]
            )
            
            async with AnomalyDetectionClient(config=mock_client_config) as client:
                with pytest.raises(SDKValidationError) as exc_info:
                    await client.detect(data=sample_data, algorithm="invalid_algorithm")
                
                assert "Invalid data format" in str(exc_info.value)
                assert exc_info.value.status_code == 422
    
    @pytest.mark.asyncio
    async def test_error_handling_rate_limit(self, mock_client_config, sample_data):
        """Test handling of rate limit errors."""
        
        from sdk_core.exceptions import RateLimitError as SDKRateLimitError
        
        with patch('sdk_core.client.BaseClient._make_request') as mock_request:
            mock_request.side_effect = SDKRateLimitError(
                "Rate limit exceeded",
                retry_after=60
            )
            
            async with AnomalyDetectionClient(config=mock_client_config) as client:
                with pytest.raises(SDKRateLimitError) as exc_info:
                    await client.detect(data=sample_data)
                
                assert exc_info.value.retry_after == 60
                assert exc_info.value.status_code == 429
    
    @pytest.mark.asyncio
    async def test_data_format_conversion(self, mock_client_config, mock_detection_response):
        """Test automatic data format conversion."""
        
        # Test with different input formats
        import numpy as np
        import pandas as pd
        
        # NumPy array
        np_data = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]])
        
        # Pandas DataFrame
        df_data = pd.DataFrame({'x': [1.0, 3.0, 100.0], 'y': [2.0, 4.0, 200.0]})
        
        # 1D list (should be converted to 2D)
        list_1d = [1.0, 2.0, 3.0]
        
        with patch('sdk_core.client.BaseClient._make_request') as mock_request:
            mock_request.return_value = mock_detection_response
            
            async with AnomalyDetectionClient(config=mock_client_config) as client:
                # Test NumPy array
                result1 = await client.detect(data=np_data)
                assert result1.success is True
                
                # Test DataFrame
                result2 = await client.detect(data=df_data)
                assert result2.success is True
                
                # Test 1D list
                result3 = await client.detect(data=list_1d)
                assert result3.success is True
                
                # Verify all calls were made
                assert mock_request.call_count == 3


class TestSyncClient:
    """Test synchronous client functionality."""
    
    def test_sync_detection(self, mock_client_config, sample_data, mock_detection_response):
        """Test synchronous detection."""
        
        with patch('sdk_core.client.BaseClient._make_request') as mock_request:
            mock_request.return_value = mock_detection_response
            
            with AnomalyDetectionSyncClient(config=mock_client_config) as client:
                result = client.detect(
                    data=sample_data,
                    algorithm="isolation_forest"
                )
                
                assert isinstance(result, DetectionResponse)
                assert result.success is True
                assert result.anomalies == [20, 21]
    
    def test_sync_health_check(self, mock_client_config):
        """Test synchronous health check."""
        
        health_response = {"success": True, "status": "healthy"}
        
        with patch('sdk_core.client.SyncClient.health_check') as mock_health:
            mock_health.return_value = health_response
            
            with AnomalyDetectionSyncClient(config=mock_client_config) as client:
                result = client.health_check()
                
                assert result["success"] is True
                assert result["status"] == "healthy"


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_detection(self, mock_client_config):
        """Test batch detection functionality."""
        
        batch_response = {
            "success": True,
            "results": [
                {"id": "dataset1", "result": {"anomalies": [0], "anomaly_count": 1}},
                {"id": "dataset2", "result": {"anomalies": [], "anomaly_count": 0}},
                {"id": "dataset3", "error": "Processing failed"}
            ],
            "total_datasets": 3,
            "successful_count": 2,
            "failed_count": 1,
            "total_processing_time_ms": 450.2,
            "timestamp": "2025-01-01T12:00:00Z"
        }
        
        datasets = [
            {"id": "dataset1", "data": [[1, 2], [100, 200]]},
            {"id": "dataset2", "data": [[1, 2], [3, 4]]},
            {"id": "dataset3", "data": [[1, 2], [3, 4]]}
        ]
        
        with patch('sdk_core.client.BaseClient._make_request') as mock_request:
            mock_request.return_value = batch_response
            
            async with AnomalyDetectionClient(config=mock_client_config) as client:
                result = await client.batch_detect(
                    datasets=datasets,
                    algorithm="isolation_forest",
                    parallel_processing=True
                )
                
                assert result["total_datasets"] == 3
                assert result["successful_count"] == 2
                assert result["failed_count"] == 1
                assert len(result["results"]) == 3


@pytest.mark.integration
class TestRealAPI:
    """
    Integration tests against a real API server.
    These tests are skipped unless INTEGRATION_TEST=true is set.
    """
    
    @pytest.mark.skipif(
        os.getenv("INTEGRATION_TEST") != "true",
        reason="Integration tests disabled. Set INTEGRATION_TEST=true to enable."
    )
    @pytest.mark.asyncio
    async def test_real_api_health_check(self):
        """Test against real API health endpoint."""
        
        api_key = os.getenv("ANOMALY_DETECTION_API_KEY")
        if not api_key:
            pytest.skip("ANOMALY_DETECTION_API_KEY not set")
        
        config = ClientConfig.for_environment(
            Environment.DEVELOPMENT,  # Use dev environment for integration tests
            api_key=api_key,
            timeout=30.0
        )
        
        async with AnomalyDetectionClient(config=config) as client:
            try:
                result = await client.health_check()
                assert "status" in result
                print(f"API Health Status: {result.get('status')}")
            except Exception as e:
                pytest.fail(f"Health check failed: {e}")
    
    @pytest.mark.skipif(
        os.getenv("INTEGRATION_TEST") != "true",
        reason="Integration tests disabled. Set INTEGRATION_TEST=true to enable."
    )
    @pytest.mark.asyncio
    async def test_real_api_algorithms(self):
        """Test fetching algorithms from real API."""
        
        api_key = os.getenv("ANOMALY_DETECTION_API_KEY")
        if not api_key:
            pytest.skip("ANOMALY_DETECTION_API_KEY not set")
        
        config = ClientConfig.for_environment(
            Environment.DEVELOPMENT,
            api_key=api_key
        )
        
        async with AnomalyDetectionClient(config=config) as client:
            try:
                algorithms = await client.get_algorithms()
                assert len(algorithms) > 0
                assert any(algo.name == "isolation_forest" for algo in algorithms)
                print(f"Available algorithms: {[algo.name for algo in algorithms]}")
            except Exception as e:
                pytest.fail(f"Get algorithms failed: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])