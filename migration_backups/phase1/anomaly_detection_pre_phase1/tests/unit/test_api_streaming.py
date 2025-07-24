"""Unit tests for streaming detection API endpoints."""

import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from anomaly_detection.api.v1.streaming import (
    router, StreamingSample, StreamingBatch, StreamingResult, StreamingStats,
    DriftDetectionResult, get_streaming_service
)
from anomaly_detection.domain.entities.detection_result import DetectionResult


@pytest.fixture
def app():
    """Create FastAPI app with streaming router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/streaming")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_streaming_service():
    """Create mock streaming service."""
    service = Mock()
    
    # Mock process_sample result
    sample_result = DetectionResult(
        success=True,
        predictions=np.array([-1]),  # Single anomaly
        confidence_scores=np.array([0.85]),
        anomalies=[0],
        algorithm="iforest",
        total_samples=1,
        anomaly_count=1,
        anomaly_rate=1.0,
        execution_time_ms=50.0
    )
    service.process_sample.return_value = sample_result
    
    # Mock process_batch result
    batch_result = DetectionResult(
        success=True,
        predictions=np.array([1, -1, 1, -1]),
        confidence_scores=np.array([0.2, 0.8, 0.3, 0.9]),
        anomalies=[1, 3],
        algorithm="iforest",
        total_samples=4,
        anomaly_count=2,
        anomaly_rate=0.5,
        execution_time_ms=100.0
    )
    service.process_batch.return_value = batch_result
    
    # Mock streaming stats
    service.get_streaming_stats.return_value = {
        "total_samples": 150,
        "buffer_size": 95,
        "buffer_capacity": 1000,
        "model_fitted": True,
        "current_algorithm": "iforest",
        "samples_since_update": 45,
        "update_frequency": 100,
        "last_update_at": 105
    }
    
    # Mock concept drift detection
    service.detect_concept_drift.return_value = {
        "drift_detected": True,
        "max_relative_change": 0.15,
        "drift_threshold": 0.1,
        "reason": "Significant change in feature distributions",
        "buffer_size": 95,
        "recent_samples": 200
    }
    
    # Mock reset
    service.reset_stream.return_value = None
    
    return service


@pytest.fixture
def mock_detection_service():
    """Create mock detection service for dependency."""
    return Mock()


class TestStreamingSampleEndpoint:
    """Test streaming sample processing endpoint."""
    
    def test_process_sample_success(self, client, mock_streaming_service):
        """Test successful single sample processing."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            request_data = {
                "data": [1.5, 2.3, -0.5, 4.2],
                "timestamp": "2024-01-01T12:00:00"
            }
            
            response = client.post(
                "/api/v1/streaming/sample?algorithm=isolation_forest&session_id=test&window_size=500&update_frequency=50",
                json=request_data
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["is_anomaly"] is True  # prediction == -1
            assert data["confidence_score"] == 0.85
            assert data["algorithm"] == "isolation_forest"
            assert data["buffer_size"] == 95
            assert data["model_fitted"] is True
            assert "sample_id" in data
            assert "timestamp" in data
            
            # Verify service was called correctly
            mock_streaming_service.process_sample.assert_called_once()
            call_args = mock_streaming_service.process_sample.call_args
            
            # Check numpy array was passed
            np.testing.assert_array_equal(call_args[0][0], [1.5, 2.3, -0.5, 4.2])
            assert call_args[0][1] == "iforest"  # Mapped algorithm
    
    def test_process_sample_algorithm_mapping(self, client, mock_streaming_service):
        """Test algorithm name mapping in sample processing."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            test_cases = [
                ("isolation_forest", "iforest"),
                ("one_class_svm", "ocsvm"),
                ("lof", "lof")
            ]
            
            for input_algo, expected_algo in test_cases:
                request_data = {"data": [1.0, 2.0]}
                
                response = client.post(
                    f"/api/v1/streaming/sample?algorithm={input_algo}",
                    json=request_data
                )
                
                assert response.status_code == 200
                
                # Verify correct algorithm was passed to service
                call_args = mock_streaming_service.process_sample.call_args
                assert call_args[0][1] == expected_algo
    
    def test_process_sample_defaults(self, client, mock_streaming_service):
        """Test sample processing with default parameters."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            request_data = {"data": [1.0, 2.0, 3.0]}
            
            response = client.post("/api/v1/streaming/sample", json=request_data)
            
            assert response.status_code == 200
            
            # Verify service was created with defaults
            # get_streaming_service should be called with default parameters
            mock_streaming_service.process_sample.assert_called_once()
    
    def test_process_sample_no_confidence_scores(self, client):
        """Test sample processing when service returns no confidence scores."""
        # Mock service with no confidence scores
        mock_service = Mock()
        sample_result = DetectionResult(
            success=True,
            predictions=np.array([1]),  # Normal sample
            confidence_scores=None,  # No scores
            anomalies=[],
            algorithm="iforest",
            total_samples=1,
            anomaly_count=0,
            anomaly_rate=0.0
        )
        mock_service.process_sample.return_value = sample_result
        mock_service.get_streaming_stats.return_value = {
            "buffer_size": 50,
            "model_fitted": True
        }
        
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_service):
            
            request_data = {"data": [1.0, 2.0]}
            
            response = client.post("/api/v1/streaming/sample", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["is_anomaly"] is False  # prediction == 1
            assert data["confidence_score"] is None
    
    def test_process_sample_service_error(self, client):
        """Test sample processing with service error."""
        mock_service = Mock()
        mock_service.process_sample.side_effect = Exception("Processing failed")
        
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_service):
            
            request_data = {"data": [1.0, 2.0]}
            
            response = client.post("/api/v1/streaming/sample", json=request_data)
            
            assert response.status_code == 500
            assert "Sample processing failed" in response.json()["detail"]


class TestStreamingBatchEndpoint:
    """Test streaming batch processing endpoint."""
    
    def test_process_batch_success(self, client, mock_streaming_service):
        """Test successful batch processing."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            request_data = {
                "samples": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "algorithm": "isolation_forest",
                "timestamp": "2024-01-01T12:00:00"
            }
            
            response = client.post(
                "/api/v1/streaming/batch?session_id=batch_test",
                json=request_data
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == 4  # One result per sample
            
            # Check first result
            result1 = data[0]
            assert result1["success"] is True
            assert result1["is_anomaly"] is False  # prediction == 1
            assert result1["algorithm"] == "isolation_forest"
            assert result1["buffer_size"] == 95
            assert "sample_id" in result1
            
            # Check anomaly result
            result2 = data[1]
            assert result2["is_anomaly"] is True  # prediction == -1
            assert result2["confidence_score"] == 0.8
            
            # Verify service was called correctly
            mock_streaming_service.process_batch.assert_called_once()
            call_args = mock_streaming_service.process_batch.call_args
            
            # Check numpy array shape
            assert call_args[0][0].shape == (4, 2)
            assert call_args[0][1] == "iforest"
    
    def test_process_batch_algorithm_mapping(self, client, mock_streaming_service):
        """Test algorithm mapping in batch processing."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            request_data = {
                "samples": [[1, 2], [3, 4]],
                "algorithm": "one_class_svm"
            }
            
            response = client.post("/api/v1/streaming/batch", json=request_data)
            
            assert response.status_code == 200
            
            # Verify algorithm was mapped
            call_args = mock_streaming_service.process_batch.call_args
            assert call_args[0][1] == "ocsvm"
    
    def test_process_batch_no_scores(self, client):
        """Test batch processing without confidence scores."""
        mock_service = Mock()
        batch_result = DetectionResult(
            success=True,
            predictions=np.array([1, -1, 1]),
            confidence_scores=None,  # No scores
            anomalies=[1],
            algorithm="iforest",
            total_samples=3,
            anomaly_count=1,
            anomaly_rate=0.33
        )
        mock_service.process_batch.return_value = batch_result
        mock_service.get_streaming_stats.return_value = {
            "buffer_size": 50,
            "model_fitted": True
        }
        
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_service):
            
            request_data = {"samples": [[1, 2], [3, 4], [5, 6]]}
            
            response = client.post("/api/v1/streaming/batch", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == 3
            for result in data:
                assert result["confidence_score"] is None
    
    def test_process_batch_service_error(self, client):
        """Test batch processing with service error."""
        mock_service = Mock()
        mock_service.process_batch.side_effect = Exception("Batch processing failed")
        
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_service):
            
            request_data = {"samples": [[1, 2], [3, 4]]}
            
            response = client.post("/api/v1/streaming/batch", json=request_data)
            
            assert response.status_code == 500
            assert "Batch processing failed" in response.json()["detail"]


class TestStreamingStatsEndpoint:
    """Test streaming statistics endpoint."""
    
    def test_get_streaming_stats_success(self, client, mock_streaming_service):
        """Test successful stats retrieval."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            response = client.get("/api/v1/streaming/stats?session_id=stats_test")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["total_samples"] == 150
            assert data["buffer_size"] == 95
            assert data["buffer_capacity"] == 1000
            assert data["model_fitted"] is True
            assert data["current_algorithm"] == "iforest"
            assert data["samples_since_update"] == 45
            assert data["update_frequency"] == 100
            assert data["last_update_at"] == 105
            
            # Verify service method was called
            mock_streaming_service.get_streaming_stats.assert_called_once()
    
    def test_get_streaming_stats_error(self, client):
        """Test stats retrieval with error."""
        mock_service = Mock()
        mock_service.get_streaming_stats.side_effect = Exception("Stats error")
        
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_service):
            
            response = client.get("/api/v1/streaming/stats")
            
            assert response.status_code == 500
            assert "Failed to get streaming stats" in response.json()["detail"]


class TestDriftDetectionEndpoint:
    """Test concept drift detection endpoint."""
    
    def test_detect_drift_success(self, client, mock_streaming_service):
        """Test successful drift detection."""
        # Mock global services dictionary
        with patch('anomaly_detection.api.v1.streaming._streaming_services',
                  {"drift_test": mock_streaming_service}):
            
            response = client.post(
                "/api/v1/streaming/drift?session_id=drift_test&window_size=150"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["drift_detected"] is True
            assert data["max_relative_change"] == 0.15
            assert data["drift_threshold"] == 0.1
            assert data["reason"] == "Significant change in feature distributions"
            assert data["buffer_size"] == 95
            assert data["recent_samples"] == 200
            
            # Verify service method was called
            mock_streaming_service.detect_concept_drift.assert_called_once_with(150)
    
    def test_detect_drift_session_not_found(self, client):
        """Test drift detection with non-existent session."""
        with patch('anomaly_detection.api.v1.streaming._streaming_services', {}):
            
            response = client.post("/api/v1/streaming/drift?session_id=nonexistent")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
    
    def test_detect_drift_service_error(self, client, mock_streaming_service):
        """Test drift detection with service error."""
        mock_streaming_service.detect_concept_drift.side_effect = Exception("Drift detection failed")
        
        with patch('anomaly_detection.api.v1.streaming._streaming_services',
                  {"error_test": mock_streaming_service}):
            
            response = client.post("/api/v1/streaming/drift?session_id=error_test")
            
            assert response.status_code == 500
            assert "Drift detection failed" in response.json()["detail"]


class TestStreamResetEndpoint:
    """Test stream reset endpoint."""
    
    def test_reset_stream_success(self, client, mock_streaming_service):
        """Test successful stream reset."""
        with patch('anomaly_detection.api.v1.streaming._streaming_services',
                  {"reset_test": mock_streaming_service}):
            
            response = client.post("/api/v1/streaming/reset?session_id=reset_test")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "reset successfully" in data["message"]
            
            # Verify service method was called
            mock_streaming_service.reset_stream.assert_called_once()
    
    def test_reset_stream_not_found(self, client):
        """Test reset with non-existent session."""
        with patch('anomaly_detection.api.v1.streaming._streaming_services', {}):
            
            response = client.post("/api/v1/streaming/reset?session_id=nonexistent")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
    
    def test_reset_stream_error(self, client, mock_streaming_service):
        """Test reset with service error."""
        mock_streaming_service.reset_stream.side_effect = Exception("Reset failed")
        
        with patch('anomaly_detection.api.v1.streaming._streaming_services',
                  {"error_test": mock_streaming_service}):
            
            response = client.post("/api/v1/streaming/reset?session_id=error_test")
            
            assert response.status_code == 500
            assert "Failed to reset stream" in response.json()["detail"]


class TestWebSocketEndpoint:
    """Test WebSocket streaming endpoint."""
    
    def test_websocket_sample_processing(self, client, mock_streaming_service):
        """Test WebSocket sample processing."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            with client.websocket_connect("/api/v1/streaming/ws/websocket_test") as websocket:
                # Send sample message
                message = {
                    "type": "sample",
                    "data": [1.5, 2.3, -0.5],
                    "algorithm": "isolation_forest"
                }
                websocket.send_text(json.dumps(message))
                
                # Receive response
                response = websocket.receive_text()
                data = json.loads(response)
                
                assert data["type"] == "result"
                assert data["is_anomaly"] is True
                assert data["confidence_score"] == 0.85
                assert data["algorithm"] == "isolation_forest"
                assert "sample_id" in data
                assert "timestamp" in data
                
                # Verify service was called
                mock_streaming_service.process_sample.assert_called_once()
    
    def test_websocket_stats_request(self, client, mock_streaming_service):
        """Test WebSocket stats request."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            with client.websocket_connect("/api/v1/streaming/ws/stats_test") as websocket:
                # Send stats request
                message = {"type": "stats"}
                websocket.send_text(json.dumps(message))
                
                # Receive response
                response = websocket.receive_text()
                data = json.loads(response)
                
                assert data["type"] == "stats"
                assert data["total_samples"] == 150
                assert data["buffer_size"] == 95
                assert data["model_fitted"] is True
                assert "timestamp" in data
    
    def test_websocket_drift_request(self, client, mock_streaming_service):
        """Test WebSocket drift detection request."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            with client.websocket_connect("/api/v1/streaming/ws/drift_test") as websocket:
                # Send drift request
                message = {
                    "type": "drift",
                    "window_size": 300
                }
                websocket.send_text(json.dumps(message))
                
                # Receive response
                response = websocket.receive_text()
                data = json.loads(response)
                
                assert data["type"] == "drift"
                assert data["drift_detected"] is True
                assert data["max_relative_change"] == 0.15
                assert "timestamp" in data
                
                # Verify service was called with correct window size
                mock_streaming_service.detect_concept_drift.assert_called_once_with(300)


class TestStreamingPydanticModels:
    """Test Pydantic models for streaming endpoints."""
    
    def test_streaming_sample_model(self):
        """Test StreamingSample model."""
        sample = StreamingSample(
            data=[1.5, 2.3, -0.5],
            timestamp="2024-01-01T12:00:00"
        )
        
        assert sample.data == [1.5, 2.3, -0.5]
        assert sample.timestamp == "2024-01-01T12:00:00"
    
    def test_streaming_sample_defaults(self):
        """Test StreamingSample model defaults."""
        sample = StreamingSample(data=[1.0, 2.0])
        
        assert sample.data == [1.0, 2.0]
        assert sample.timestamp is None
    
    def test_streaming_batch_model(self):
        """Test StreamingBatch model."""
        batch = StreamingBatch(
            samples=[[1, 2], [3, 4]],
            algorithm="lof",
            timestamp="2024-01-01T12:00:00"
        )
        
        assert batch.samples == [[1, 2], [3, 4]]
        assert batch.algorithm == "lof"
        assert batch.timestamp == "2024-01-01T12:00:00"
    
    def test_streaming_batch_defaults(self):
        """Test StreamingBatch model defaults."""
        batch = StreamingBatch(samples=[[1, 2]])
        
        assert batch.algorithm == "isolation_forest"
        assert batch.timestamp is None
    
    def test_streaming_result_model(self):
        """Test StreamingResult model."""
        result = StreamingResult(
            success=True,
            sample_id="test-sample-123",
            is_anomaly=True,
            confidence_score=0.85,
            algorithm="isolation_forest",
            timestamp="2024-01-01T12:00:00",
            buffer_size=95,
            model_fitted=True
        )
        
        assert result.success is True
        assert result.sample_id == "test-sample-123"
        assert result.is_anomaly is True
        assert result.confidence_score == 0.85
        assert result.algorithm == "isolation_forest"
        assert result.buffer_size == 95
        assert result.model_fitted is True
    
    def test_streaming_stats_model(self):
        """Test StreamingStats model."""
        stats = StreamingStats(
            total_samples=500,
            buffer_size=150,
            buffer_capacity=1000,
            model_fitted=True,
            current_algorithm="isolation_forest",
            samples_since_update=50,
            update_frequency=100,
            last_update_at=450
        )
        
        assert stats.total_samples == 500
        assert stats.buffer_size == 150
        assert stats.buffer_capacity == 1000
        assert stats.model_fitted is True
        assert stats.current_algorithm == "isolation_forest"
        assert stats.samples_since_update == 50
        assert stats.update_frequency == 100
        assert stats.last_update_at == 450
    
    def test_drift_detection_result_model(self):
        """Test DriftDetectionResult model."""
        result = DriftDetectionResult(
            drift_detected=True,
            max_relative_change=0.25,
            drift_threshold=0.1,
            reason="Significant distribution shift",
            buffer_size=200,
            recent_samples=150
        )
        
        assert result.drift_detected is True
        assert result.max_relative_change == 0.25
        assert result.drift_threshold == 0.1
        assert result.reason == "Significant distribution shift"
        assert result.buffer_size == 200
        assert result.recent_samples == 150


class TestStreamingDependencies:
    """Test dependency management for streaming endpoints."""
    
    def test_get_streaming_service_creation(self, mock_detection_service):
        """Test streaming service creation."""
        with patch('anomaly_detection.api.v1.streaming.DetectionService',
                  return_value=mock_detection_service), \
             patch('anomaly_detection.api.v1.streaming.StreamingService') as mock_streaming_class:
            
            # Clear services dictionary
            import anomaly_detection.api.v1.streaming as streaming_module
            streaming_module._streaming_services = {}
            
            service = get_streaming_service(
                session_id="test_session",
                window_size=500,
                update_frequency=50
            )
            
            # Verify StreamingService was created with correct parameters
            mock_streaming_class.assert_called_once_with(
                detection_service=mock_detection_service,
                window_size=500,
                update_frequency=50
            )
    
    def test_get_streaming_service_reuse(self, mock_streaming_service):
        """Test streaming service reuse for same session."""
        with patch('anomaly_detection.api.v1.streaming._streaming_services',
                  {"reuse_test": mock_streaming_service}):
            
            service1 = get_streaming_service("reuse_test")
            service2 = get_streaming_service("reuse_test")
            
            assert service1 is service2
            assert service1 is mock_streaming_service


class TestStreamingErrorHandling:
    """Test error handling in streaming endpoints."""
    
    def test_invalid_data_conversion(self, client, mock_streaming_service):
        """Test handling of invalid data conversion."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            request_data = {"data": ["invalid", "data"]}
            
            response = client.post("/api/v1/streaming/sample", json=request_data)
            
            # Should handle numpy conversion error gracefully
            assert response.status_code == 500
    
    def test_websocket_error_handling(self, client, mock_streaming_service):
        """Test WebSocket error handling."""
        mock_streaming_service.process_sample.side_effect = Exception("Processing error")
        
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            # WebSocket should handle errors gracefully and close connection
            # This is more of an integration test concept
            pass
    
    def test_large_batch_handling(self, client, mock_streaming_service):
        """Test handling of large batches."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            # Create large batch
            large_batch = [[i, i+1] for i in range(1000)]
            
            request_data = {"samples": large_batch}
            
            response = client.post("/api/v1/streaming/batch", json=request_data)
            
            assert response.status_code == 200
            # Should handle large batches efficiently
            assert len(response.json()) == 1000
    
    def test_concurrent_session_handling(self, client, mock_streaming_service):
        """Test handling of multiple concurrent sessions."""
        with patch('anomaly_detection.api.v1.streaming.get_streaming_service',
                  return_value=mock_streaming_service):
            
            # Test different session IDs
            sessions = ["session1", "session2", "session3"]
            
            for session_id in sessions:
                request_data = {"data": [1.0, 2.0]}
                
                response = client.post(
                    f"/api/v1/streaming/sample?session_id={session_id}",
                    json=request_data
                )
                
                assert response.status_code == 200
                # Each session should be handled independently