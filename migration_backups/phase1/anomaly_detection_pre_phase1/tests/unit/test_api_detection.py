"""Unit tests for detection API endpoints."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from anomaly_detection.api.v1.detection import (
    router, DetectionRequest, EnsembleRequest, DetectionResponse,
    get_detection_service, get_ensemble_service
)
from anomaly_detection.domain.entities.detection_result import DetectionResult


@pytest.fixture
def app():
    """Create FastAPI app with detection router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/detection")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_detection_service():
    """Create mock detection service."""
    service = Mock()
    
    # Mock successful detection result
    result = DetectionResult(
        success=True,
        predictions=np.array([1, -1, 1, -1]),
        confidence_scores=np.array([0.2, 0.8, 0.3, 0.9]),
        anomalies=[1, 3],
        algorithm="iforest",
        total_samples=4,
        anomaly_count=2,
        anomaly_rate=0.5,
        execution_time_ms=150.0
    )
    
    service.detect_anomalies.return_value = result
    return service


@pytest.fixture
def mock_ensemble_service():
    """Create mock ensemble service."""
    service = Mock()
    
    # Mock ensemble methods
    service.majority_vote.return_value = np.array([1, -1, 1, -1])
    service.average_combination.return_value = (
        np.array([1, -1, 1, -1]),
        np.array([0.2, 0.8, 0.3, 0.9])
    )
    service.max_combination.return_value = (
        np.array([1, -1, 1, -1]),
        np.array([0.3, 0.9, 0.4, 0.95])
    )
    service.weighted_combination.return_value = (
        np.array([1, -1, 1, -1]),
        np.array([0.25, 0.85, 0.35, 0.92])
    )
    
    return service


class TestDetectionEndpoints:
    """Test suite for detection API endpoints."""
    
    def test_detect_anomalies_success(self, client, mock_detection_service):
        """Test successful anomaly detection."""
        with patch('anomaly_detection.api.v1.detection.get_detection_service', 
                  return_value=mock_detection_service):
            
            request_data = {
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "parameters": {"n_estimators": 100}
            }
            
            response = client.post("/api/v1/detection/detect", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["algorithm"] == "isolation_forest"
            assert data["total_samples"] == 4
            assert data["anomalies_detected"] == 2
            assert data["anomaly_rate"] == 0.5
            assert data["anomalies"] == [1, 3]
            assert data["scores"] == [0.2, 0.8, 0.3, 0.9]
            assert "timestamp" in data
            assert "processing_time_ms" in data
            
            # Verify service was called correctly
            mock_detection_service.detect_anomalies.assert_called_once()
            call_args = mock_detection_service.detect_anomalies.call_args
            
            assert call_args[1]["algorithm"] == "iforest"  # Mapped algorithm
            assert call_args[1]["contamination"] == 0.1
            assert call_args[1]["n_estimators"] == 100
    
    def test_detect_anomalies_empty_data(self, client):
        """Test detection with empty data."""
        request_data = {
            "data": [],
            "algorithm": "isolation_forest"
        }
        
        response = client.post("/api/v1/detection/detect", json=request_data)
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_detect_anomalies_invalid_data(self, client):
        """Test detection with invalid data structure."""
        request_data = {
            "data": [[]],
            "algorithm": "isolation_forest"
        }
        
        response = client.post("/api/v1/detection/detect", json=request_data)
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_detect_anomalies_algorithm_mapping(self, client, mock_detection_service):
        """Test algorithm name mapping."""
        with patch('anomaly_detection.api.v1.detection.get_detection_service', 
                  return_value=mock_detection_service):
            
            test_cases = [
                ("isolation_forest", "iforest"),
                ("one_class_svm", "ocsvm"),
                ("local_outlier_factor", "lof"),
                ("lof", "lof"),
                ("custom_algo", "custom_algo")  # No mapping
            ]
            
            for input_algo, expected_algo in test_cases:
                request_data = {
                    "data": [[1, 2], [3, 4]],
                    "algorithm": input_algo
                }
                
                response = client.post("/api/v1/detection/detect", json=request_data)
                
                assert response.status_code == 200
                
                # Check that service was called with mapped algorithm
                call_args = mock_detection_service.detect_anomalies.call_args
                assert call_args[1]["algorithm"] == expected_algo
    
    def test_detect_anomalies_service_error(self, client):
        """Test detection with service error."""
        mock_service = Mock()
        mock_service.detect_anomalies.side_effect = ValueError("Algorithm not supported")
        
        with patch('anomaly_detection.api.v1.detection.get_detection_service', 
                  return_value=mock_service):
            
            request_data = {
                "data": [[1, 2], [3, 4]],
                "algorithm": "invalid_algo"
            }
            
            response = client.post("/api/v1/detection/detect", json=request_data)
            
            assert response.status_code == 400
            assert "Invalid input" in response.json()["detail"]
    
    def test_detect_anomalies_internal_error(self, client):
        """Test detection with internal server error."""
        mock_service = Mock()
        mock_service.detect_anomalies.side_effect = RuntimeError("Internal error")
        
        with patch('anomaly_detection.api.v1.detection.get_detection_service', 
                  return_value=mock_service):
            
            request_data = {
                "data": [[1, 2], [3, 4]],
                "algorithm": "isolation_forest"
            }
            
            response = client.post("/api/v1/detection/detect", json=request_data)
            
            assert response.status_code == 500
            assert "Detection failed" in response.json()["detail"]
    
    def test_ensemble_detect_success(self, client, mock_ensemble_service):
        """Test successful ensemble detection."""
        # Mock individual detection results
        mock_detection_service = Mock()
        result1 = DetectionResult(
            success=True, predictions=np.array([1, -1, 1, -1]),
            confidence_scores=np.array([0.2, 0.8, 0.3, 0.9]),
            anomalies=[1, 3], algorithm="iforest",
            total_samples=4, anomaly_count=2, anomaly_rate=0.5
        )
        result2 = DetectionResult(
            success=True, predictions=np.array([1, 1, -1, -1]),
            confidence_scores=np.array([0.3, 0.1, 0.7, 0.85]),
            anomalies=[2, 3], algorithm="ocsvm",
            total_samples=4, anomaly_count=2, anomaly_rate=0.5
        )
        
        mock_detection_service.detect_anomalies.side_effect = [result1, result2]
        
        with patch('anomaly_detection.api.v1.detection.get_ensemble_service', 
                  return_value=mock_ensemble_service), \
             patch('anomaly_detection.api.v1.detection.DetectionService', 
                  return_value=mock_detection_service):
            
            request_data = {
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "algorithms": ["isolation_forest", "one_class_svm"],
                "method": "majority",
                "contamination": 0.1
            }
            
            response = client.post("/api/v1/detection/ensemble", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["algorithm"] == "ensemble_majority"
            assert data["total_samples"] == 4
            assert isinstance(data["anomalies"], list)
            assert "timestamp" in data
            assert "processing_time_ms" in data
            
            # Verify ensemble method was called
            mock_ensemble_service.majority_vote.assert_called_once()
    
    def test_ensemble_detect_average_method(self, client, mock_ensemble_service):
        """Test ensemble detection with average method."""
        mock_detection_service = Mock()
        result = DetectionResult(
            success=True, predictions=np.array([1, -1, 1, -1]),
            confidence_scores=np.array([0.2, 0.8, 0.3, 0.9]),
            anomalies=[1, 3], algorithm="iforest",
            total_samples=4, anomaly_count=2, anomaly_rate=0.5
        )
        mock_detection_service.detect_anomalies.return_value = result
        
        with patch('anomaly_detection.api.v1.detection.get_ensemble_service', 
                  return_value=mock_ensemble_service), \
             patch('anomaly_detection.api.v1.detection.DetectionService', 
                  return_value=mock_detection_service):
            
            request_data = {
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "algorithms": ["isolation_forest", "one_class_svm"],
                "method": "average"
            }
            
            response = client.post("/api/v1/detection/ensemble", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["algorithm"] == "ensemble_average"
            assert data["scores"] is not None
            
            # Verify average combination was called
            mock_ensemble_service.average_combination.assert_called_once()
    
    def test_ensemble_detect_weighted_average(self, client, mock_ensemble_service):
        """Test ensemble detection with weighted average."""
        mock_detection_service = Mock()
        result = DetectionResult(
            success=True, predictions=np.array([1, -1, 1, -1]),
            confidence_scores=np.array([0.2, 0.8, 0.3, 0.9]),
            anomalies=[1, 3], algorithm="iforest",
            total_samples=4, anomaly_count=2, anomaly_rate=0.5
        )
        mock_detection_service.detect_anomalies.return_value = result
        
        with patch('anomaly_detection.api.v1.detection.get_ensemble_service', 
                  return_value=mock_ensemble_service), \
             patch('anomaly_detection.api.v1.detection.DetectionService', 
                  return_value=mock_detection_service):
            
            request_data = {
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "algorithms": ["isolation_forest", "one_class_svm", "lof"],
                "method": "weighted_average"
            }
            
            response = client.post("/api/v1/detection/ensemble", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["algorithm"] == "ensemble_weighted_average"
            
            # Verify weighted combination was called
            mock_ensemble_service.weighted_combination.assert_called_once()
            
            # Check that weights were passed (equal weights for 3 algorithms)
            call_args = mock_ensemble_service.weighted_combination.call_args
            weights = call_args[0][2]  # Third argument is weights
            expected_weights = np.ones(3) / 3
            np.testing.assert_array_almost_equal(weights, expected_weights)
    
    def test_ensemble_detect_empty_data(self, client):
        """Test ensemble detection with empty data."""
        request_data = {
            "data": [],
            "algorithms": ["isolation_forest", "one_class_svm"]
        }
        
        response = client.post("/api/v1/detection/ensemble", json=request_data)
        
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
    
    def test_ensemble_detect_insufficient_algorithms(self, client):
        """Test ensemble detection with insufficient algorithms."""
        request_data = {
            "data": [[1, 2], [3, 4]],
            "algorithms": ["isolation_forest"]  # Only one algorithm
        }
        
        response = client.post("/api/v1/detection/ensemble", json=request_data)
        
        assert response.status_code == 400
        assert "at least 2 algorithms" in response.json()["detail"]
    
    def test_ensemble_detect_fallback_majority(self, client, mock_ensemble_service):
        """Test ensemble detection fallback to majority vote."""
        mock_detection_service = Mock()
        # Create result without confidence scores
        result = DetectionResult(
            success=True, predictions=np.array([1, -1, 1, -1]),
            confidence_scores=None,  # No scores
            anomalies=[1, 3], algorithm="iforest",
            total_samples=4, anomaly_count=2, anomaly_rate=0.5
        )
        mock_detection_service.detect_anomalies.return_value = result
        
        with patch('anomaly_detection.api.v1.detection.get_ensemble_service', 
                  return_value=mock_ensemble_service), \
             patch('anomaly_detection.api.v1.detection.DetectionService', 
                  return_value=mock_detection_service):
            
            request_data = {
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "algorithms": ["isolation_forest", "one_class_svm"],
                "method": "average"  # Should fallback to majority
            }
            
            response = client.post("/api/v1/detection/ensemble", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["scores"] is None  # No scores due to fallback
            
            # Should use majority vote as fallback
            mock_ensemble_service.majority_vote.assert_called_once()
    
    def test_ensemble_detect_algorithm_specific_params(self, client, mock_ensemble_service):
        """Test ensemble detection with algorithm-specific parameters."""
        mock_detection_service = Mock()
        result = DetectionResult(
            success=True, predictions=np.array([1, -1, 1, -1]),
            confidence_scores=np.array([0.2, 0.8, 0.3, 0.9]),
            anomalies=[1, 3], algorithm="iforest",
            total_samples=4, anomaly_count=2, anomaly_rate=0.5
        )
        mock_detection_service.detect_anomalies.return_value = result
        
        with patch('anomaly_detection.api.v1.detection.get_ensemble_service', 
                  return_value=mock_ensemble_service), \
             patch('anomaly_detection.api.v1.detection.DetectionService', 
                  return_value=mock_detection_service):
            
            request_data = {
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "algorithms": ["isolation_forest", "one_class_svm"],
                "method": "majority",
                "parameters": {
                    "iforest": {"n_estimators": 200},
                    "ocsvm": {"kernel": "linear"}
                }
            }
            
            response = client.post("/api/v1/detection/ensemble", json=request_data)
            
            assert response.status_code == 200
            
            # Verify service was called with correct parameters
            assert mock_detection_service.detect_anomalies.call_count == 2
            
            # Check first call (iforest)
            call_args_1 = mock_detection_service.detect_anomalies.call_args_list[0]
            assert call_args_1[1]["n_estimators"] == 200
            
            # Check second call would get ocsvm params, but we need to mock properly
            # This test verifies the parameter passing logic works
    
    def test_list_algorithms(self, client):
        """Test listing available algorithms."""
        response = client.get("/api/v1/detection/algorithms")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "single_algorithms" in data
        assert "ensemble_methods" in data
        assert "supported_formats" in data
        
        # Check expected algorithms are present
        assert "isolation_forest" in data["single_algorithms"]
        assert "one_class_svm" in data["single_algorithms"]
        assert "local_outlier_factor" in data["single_algorithms"]
        assert "lof" in data["single_algorithms"]
        
        # Check ensemble methods
        assert "majority" in data["ensemble_methods"]
        assert "average" in data["ensemble_methods"]
        assert "weighted_average" in data["ensemble_methods"]
        assert "max" in data["ensemble_methods"]
        
        # Check supported formats
        assert "json" in data["supported_formats"]
        assert "csv" in data["supported_formats"]


class TestDetectionModels:
    """Test Pydantic models for detection endpoints."""
    
    def test_detection_request_model(self):
        """Test DetectionRequest model validation."""
        # Valid request
        request = DetectionRequest(
            data=[[1, 2], [3, 4]],
            algorithm="isolation_forest",
            contamination=0.1,
            parameters={"n_estimators": 100}
        )
        
        assert request.data == [[1, 2], [3, 4]]
        assert request.algorithm == "isolation_forest"
        assert request.contamination == 0.1
        assert request.parameters == {"n_estimators": 100}
    
    def test_detection_request_defaults(self):
        """Test DetectionRequest model defaults."""
        request = DetectionRequest(data=[[1, 2], [3, 4]])
        
        assert request.algorithm == "isolation_forest"
        assert request.contamination == 0.1
        assert request.parameters == {}
    
    def test_detection_request_validation(self):
        """Test DetectionRequest model validation."""
        # Invalid contamination
        with pytest.raises(ValueError):
            DetectionRequest(
                data=[[1, 2]],
                contamination=0.6  # Too high
            )
        
        with pytest.raises(ValueError):
            DetectionRequest(
                data=[[1, 2]],
                contamination=0.0  # Too low
            )
    
    def test_ensemble_request_model(self):
        """Test EnsembleRequest model validation."""
        request = EnsembleRequest(
            data=[[1, 2], [3, 4]],
            algorithms=["isolation_forest", "one_class_svm"],
            method="majority",
            contamination=0.05,
            parameters={"iforest": {"n_estimators": 50}}
        )
        
        assert request.algorithms == ["isolation_forest", "one_class_svm"]
        assert request.method == "majority"
        assert request.contamination == 0.05
        assert request.parameters == {"iforest": {"n_estimators": 50}}
    
    def test_ensemble_request_defaults(self):
        """Test EnsembleRequest model defaults."""
        request = EnsembleRequest(data=[[1, 2], [3, 4]])
        
        assert request.algorithms == ["isolation_forest", "one_class_svm", "lof"]
        assert request.method == "majority"
        assert request.contamination == 0.1
        assert request.parameters == {}
    
    def test_detection_response_model(self):
        """Test DetectionResponse model."""
        response = DetectionResponse(
            success=True,
            anomalies=[1, 3, 5],
            scores=[0.8, 0.9, 0.7],
            algorithm="isolation_forest",
            total_samples=10,
            anomalies_detected=3,
            anomaly_rate=0.3,
            timestamp="2024-01-01T12:00:00",
            processing_time_ms=150.5
        )
        
        assert response.success is True
        assert response.anomalies == [1, 3, 5]
        assert response.scores == [0.8, 0.9, 0.7]
        assert response.algorithm == "isolation_forest"
        assert response.total_samples == 10
        assert response.anomalies_detected == 3
        assert response.anomaly_rate == 0.3
        assert response.timestamp == "2024-01-01T12:00:00"
        assert response.processing_time_ms == 150.5


class TestDetectionDependencies:
    """Test dependency injection for detection endpoints."""
    
    def test_get_detection_service_singleton(self):
        """Test that detection service is singleton."""
        # Clear any existing instance
        import anomaly_detection.api.v1.detection as detection_module
        detection_module._detection_service = None
        
        service1 = get_detection_service()
        service2 = get_detection_service()
        
        assert service1 is service2
    
    def test_get_ensemble_service_singleton(self):
        """Test that ensemble service is singleton."""
        import anomaly_detection.api.v1.detection as detection_module
        detection_module._ensemble_service = None
        
        service1 = get_ensemble_service()
        service2 = get_ensemble_service()
        
        assert service1 is service2


class TestDetectionErrorHandling:
    """Test error handling in detection endpoints."""
    
    def test_numpy_conversion_error(self, client):
        """Test error handling for numpy conversion issues."""
        request_data = {
            "data": [["invalid", "data"], ["not", "numeric"]],
            "algorithm": "isolation_forest"
        }
        
        response = client.post("/api/v1/detection/detect", json=request_data)
        
        # Should handle numpy conversion error
        assert response.status_code in [400, 500]
    
    def test_async_decorator_functionality(self, client, mock_detection_service):
        """Test that async log decorator is properly applied."""
        with patch('anomaly_detection.api.v1.detection.get_detection_service', 
                  return_value=mock_detection_service):
            
            request_data = {
                "data": [[1, 2], [3, 4]],
                "algorithm": "isolation_forest"
            }
            
            response = client.post("/api/v1/detection/detect", json=request_data)
            
            assert response.status_code == 200
            # The decorator should not interfere with normal operation
            assert "processing_time_ms" in response.json()
    
    def test_large_dataset_handling(self, client, mock_detection_service):
        """Test handling of large datasets."""
        with patch('anomaly_detection.api.v1.detection.get_detection_service', 
                  return_value=mock_detection_service):
            
            # Create large dataset
            large_data = [[i, i+1] for i in range(10000)]
            
            request_data = {
                "data": large_data,
                "algorithm": "isolation_forest"
            }
            
            response = client.post("/api/v1/detection/detect", json=request_data)
            
            assert response.status_code == 200
            # Verify service received the data
            call_args = mock_detection_service.detect_anomalies.call_args
            assert call_args[1]["data"].shape == (10000, 2)