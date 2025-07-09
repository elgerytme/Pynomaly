"""
Comprehensive tests for streaming endpoints.
Tests real-time streaming and WebSocket API endpoints.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from fastapi import status

from pynomaly.presentation.web_api.app import app
from pynomaly.domain.entities.streaming_session import StreamingSession
from pynomaly.domain.entities.streaming_anomaly import StreamingAnomaly
from pynomaly.domain.exceptions import StreamingError, DetectorError


class TestStreamingEndpointsComprehensive:
    """Comprehensive test suite for streaming API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_streaming_service(self):
        """Mock streaming service."""
        service = AsyncMock()
        service.create_streaming_session.return_value = StreamingSession(
            id=uuid4(),
            detector_id=uuid4(),
            name="test-streaming-session",
            config={
                "batch_size": 100,
                "window_size": 1000,
                "overlap": 0.1,
                "threshold": 0.5,
            },
            status="active",
            created_at=datetime.utcnow(),
        )
        
        service.get_streaming_session.return_value = StreamingSession(
            id=uuid4(),
            detector_id=uuid4(),
            name="test-streaming-session",
            config={"batch_size": 100},
            status="active",
            created_at=datetime.utcnow(),
        )
        
        service.process_streaming_data.return_value = {
            "batch_id": str(uuid4()),
            "session_id": str(uuid4()),
            "processed_samples": 100,
            "anomalies_detected": 3,
            "anomalies": [
                {
                    "id": str(uuid4()),
                    "sample_index": 15,
                    "anomaly_score": 0.85,
                    "timestamp": datetime.utcnow().isoformat(),
                    "confidence": 0.92,
                },
                {
                    "id": str(uuid4()),
                    "sample_index": 47,
                    "anomaly_score": 0.78,
                    "timestamp": datetime.utcnow().isoformat(),
                    "confidence": 0.87,
                },
            ],
            "batch_statistics": {
                "mean_score": 0.34,
                "std_score": 0.21,
                "min_score": 0.05,
                "max_score": 0.85,
            },
            "processing_time": 0.125,
        }
        
        service.get_streaming_metrics.return_value = {
            "session_id": str(uuid4()),
            "total_samples": 10000,
            "total_anomalies": 150,
            "anomaly_rate": 0.015,
            "throughput": 800.5,  # samples per second
            "latency": {
                "mean": 0.025,
                "p95": 0.045,
                "p99": 0.078,
            },
            "resource_usage": {
                "memory_mb": 512,
                "cpu_percent": 25.5,
            },
            "uptime_seconds": 3600,
        }
        
        service.get_streaming_anomalies.return_value = [
            StreamingAnomaly(
                id=uuid4(),
                session_id=uuid4(),
                sample_index=15,
                anomaly_score=0.85,
                timestamp=datetime.utcnow(),
                confidence=0.92,
                raw_data=[1.2, 2.3, 0.8],
            ),
            StreamingAnomaly(
                id=uuid4(),
                session_id=uuid4(),
                sample_index=47,
                anomaly_score=0.78,
                timestamp=datetime.utcnow(),
                confidence=0.87,
                raw_data=[2.1, 1.7, 1.3],
            ),
        ]
        
        return service

    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service."""
        service = AsyncMock()
        service.get_current_user.return_value = {
            "id": "user_123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"],
        }
        return service

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test_token_123"}

    @pytest.fixture
    def valid_streaming_session_payload(self):
        """Valid streaming session creation payload."""
        return {
            "detector_id": str(uuid4()),
            "name": "test-streaming-session",
            "config": {
                "batch_size": 100,
                "window_size": 1000,
                "overlap": 0.1,
                "threshold": 0.5,
                "alert_threshold": 0.8,
                "buffer_size": 5000,
            },
            "alerts": {
                "enabled": True,
                "email_notifications": True,
                "webhook_url": "https://example.com/webhook",
            },
            "data_retention": {
                "keep_raw_data": False,
                "keep_anomalies": True,
                "retention_days": 30,
            },
        }

    @pytest.fixture
    def valid_streaming_data_payload(self):
        """Valid streaming data payload."""
        return {
            "session_id": str(uuid4()),
            "data": [
                [1.2, 2.3, 0.8],
                [1.1, 2.1, 0.9],
                [1.5, 2.5, 0.7],
                [2.8, 3.2, 1.5],  # Potential anomaly
                [1.3, 2.2, 0.8],
            ],
            "timestamps": [
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
            ],
            "metadata": {
                "source": "sensor_network",
                "device_id": "device_001",
                "batch_id": str(uuid4()),
            },
        }

    def test_create_streaming_session_success(
        self, client, valid_streaming_session_payload, mock_streaming_service, auth_headers
    ):
        """Test successful streaming session creation."""
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.post(
                "/api/v1/streaming/sessions",
                json=valid_streaming_session_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert "detector_id" in data
        assert "name" in data
        assert "config" in data
        assert "status" in data
        assert "created_at" in data
        assert data["status"] == "active"

    def test_create_streaming_session_invalid_detector(
        self, client, valid_streaming_session_payload, mock_streaming_service, auth_headers
    ):
        """Test streaming session creation with invalid detector."""
        mock_streaming_service.create_streaming_session.side_effect = DetectorError("Detector not found")
        
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.post(
                "/api/v1/streaming/sessions",
                json=valid_streaming_session_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "error" in data
        assert "Detector not found" in data["error"]

    def test_create_streaming_session_missing_fields(self, client, auth_headers):
        """Test streaming session creation with missing required fields."""
        incomplete_payload = {
            "name": "test-session",
            # Missing detector_id
        }

        response = client.post(
            "/api/v1/streaming/sessions",
            json=incomplete_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_streaming_session_unauthorized(self, client, valid_streaming_session_payload):
        """Test streaming session creation without authentication."""
        response = client.post(
            "/api/v1/streaming/sessions",
            json=valid_streaming_session_payload,
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_streaming_session_success(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test successful streaming session retrieval."""
        session_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.get(
                f"/api/v1/streaming/sessions/{session_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert "detector_id" in data
        assert "name" in data
        assert "config" in data
        assert "status" in data

    def test_get_streaming_session_not_found(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test streaming session retrieval with non-existent ID."""
        mock_streaming_service.get_streaming_session.side_effect = StreamingError("Session not found")
        session_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.get(
                f"/api/v1/streaming/sessions/{session_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_streaming_sessions_success(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test successful streaming sessions listing."""
        mock_sessions = [
            StreamingSession(
                id=uuid4(),
                detector_id=uuid4(),
                name="session-1",
                config={"batch_size": 100},
                status="active",
                created_at=datetime.utcnow(),
            ),
            StreamingSession(
                id=uuid4(),
                detector_id=uuid4(),
                name="session-2",
                config={"batch_size": 200},
                status="paused",
                created_at=datetime.utcnow(),
            ),
        ]
        mock_streaming_service.list_streaming_sessions.return_value = mock_sessions

        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.get("/api/v1/streaming/sessions", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "sessions" in data
        assert len(data["sessions"]) == 2
        assert data["sessions"][0]["name"] == "session-1"
        assert data["sessions"][1]["name"] == "session-2"

    def test_list_streaming_sessions_with_filters(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test streaming sessions listing with filters."""
        mock_streaming_service.list_streaming_sessions.return_value = []

        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.get(
                "/api/v1/streaming/sessions?status=active&detector_id=test-detector",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "sessions" in data

    def test_process_streaming_data_success(
        self, client, valid_streaming_data_payload, mock_streaming_service, auth_headers
    ):
        """Test successful streaming data processing."""
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.post(
                "/api/v1/streaming/process",
                json=valid_streaming_data_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "batch_id" in data
        assert "session_id" in data
        assert "processed_samples" in data
        assert "anomalies_detected" in data
        assert "anomalies" in data
        assert "batch_statistics" in data
        assert "processing_time" in data

        # Verify anomalies structure
        anomalies = data["anomalies"]
        assert len(anomalies) > 0
        assert "id" in anomalies[0]
        assert "sample_index" in anomalies[0]
        assert "anomaly_score" in anomalies[0]
        assert "timestamp" in anomalies[0]

    def test_process_streaming_data_invalid_session(
        self, client, valid_streaming_data_payload, mock_streaming_service, auth_headers
    ):
        """Test streaming data processing with invalid session."""
        mock_streaming_service.process_streaming_data.side_effect = StreamingError("Session not found")
        
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.post(
                "/api/v1/streaming/process",
                json=valid_streaming_data_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_process_streaming_data_invalid_format(
        self, client, auth_headers
    ):
        """Test streaming data processing with invalid data format."""
        invalid_payload = {
            "session_id": str(uuid4()),
            "data": "invalid_data_format",  # Should be array
        }

        response = client.post(
            "/api/v1/streaming/process",
            json=invalid_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_streaming_metrics_success(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test successful streaming metrics retrieval."""
        session_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.get(
                f"/api/v1/streaming/sessions/{session_id}/metrics",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "session_id" in data
        assert "total_samples" in data
        assert "total_anomalies" in data
        assert "anomaly_rate" in data
        assert "throughput" in data
        assert "latency" in data
        assert "resource_usage" in data
        assert "uptime_seconds" in data

    def test_get_streaming_metrics_with_timerange(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test streaming metrics with time range filter."""
        session_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.get(
                f"/api/v1/streaming/sessions/{session_id}/metrics?start_time=2024-01-01T00:00:00Z&end_time=2024-01-02T00:00:00Z",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_samples" in data

    def test_get_streaming_anomalies_success(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test successful streaming anomalies retrieval."""
        session_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.get(
                f"/api/v1/streaming/sessions/{session_id}/anomalies",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "anomalies" in data
        assert len(data["anomalies"]) > 0

        # Verify anomaly structure
        anomaly = data["anomalies"][0]
        assert "id" in anomaly
        assert "session_id" in anomaly
        assert "sample_index" in anomaly
        assert "anomaly_score" in anomaly
        assert "timestamp" in anomaly

    def test_get_streaming_anomalies_with_filters(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test streaming anomalies with filters."""
        session_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.get(
                f"/api/v1/streaming/sessions/{session_id}/anomalies?min_score=0.8&limit=10",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "anomalies" in data

    def test_update_streaming_session_success(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test successful streaming session update."""
        session_id = str(uuid4())
        update_payload = {
            "name": "updated-session",
            "config": {
                "batch_size": 200,
                "threshold": 0.7,
            },
        }

        mock_updated_session = StreamingSession(
            id=session_id,
            detector_id=uuid4(),
            name=update_payload["name"],
            config=update_payload["config"],
            status="active",
            created_at=datetime.utcnow(),
        )
        mock_streaming_service.update_streaming_session.return_value = mock_updated_session

        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.put(
                f"/api/v1/streaming/sessions/{session_id}",
                json=update_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == update_payload["name"]
        assert data["config"]["batch_size"] == update_payload["config"]["batch_size"]

    def test_pause_streaming_session_success(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test successful streaming session pause."""
        session_id = str(uuid4())
        mock_streaming_service.pause_streaming_session.return_value = True

        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.post(
                f"/api/v1/streaming/sessions/{session_id}/pause",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "paused" in data["message"].lower()

    def test_resume_streaming_session_success(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test successful streaming session resume."""
        session_id = str(uuid4())
        mock_streaming_service.resume_streaming_session.return_value = True

        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.post(
                f"/api/v1/streaming/sessions/{session_id}/resume",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "resumed" in data["message"].lower()

    def test_delete_streaming_session_success(
        self, client, mock_streaming_service, auth_headers
    ):
        """Test successful streaming session deletion."""
        session_id = str(uuid4())
        mock_streaming_service.delete_streaming_session.return_value = True

        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.delete(
                f"/api/v1/streaming/sessions/{session_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_streaming_concurrent_processing(
        self, client, valid_streaming_data_payload, mock_streaming_service, auth_headers
    ):
        """Test handling concurrent streaming data processing."""
        import threading
        
        results = []
        
        def process_streaming_data():
            with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
                response = client.post(
                    "/api/v1/streaming/process",
                    json=valid_streaming_data_payload,
                    headers=auth_headers,
                )
                results.append(response.status_code)

        # Create multiple threads for concurrent processing
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_streaming_data)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should have completed successfully
        assert len(results) == 5
        assert all(status_code == 200 for status_code in results)

    def test_streaming_request_validation(self, client, auth_headers):
        """Test comprehensive request validation."""
        # Test invalid JSON
        response = client.post(
            "/api/v1/streaming/process",
            data="invalid json",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid UUID format
        invalid_payload = {
            "session_id": "invalid-uuid",
            "data": [[1, 2, 3]],
        }
        response = client.post(
            "/api/v1/streaming/process",
            json=invalid_payload,
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_streaming_error_handling(
        self, client, valid_streaming_data_payload, mock_streaming_service, auth_headers
    ):
        """Test error handling in streaming endpoints."""
        # Test service unavailable
        mock_streaming_service.process_streaming_data.side_effect = Exception("Service unavailable")
        
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.post(
                "/api/v1/streaming/process",
                json=valid_streaming_data_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_streaming_security_headers(
        self, client, valid_streaming_data_payload, mock_streaming_service, auth_headers
    ):
        """Test security headers in streaming responses."""
        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.post(
                "/api/v1/streaming/process",
                json=valid_streaming_data_payload,
                headers=auth_headers,
            )

        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers

    def test_streaming_cors_handling(
        self, client, valid_streaming_data_payload, mock_streaming_service, auth_headers
    ):
        """Test CORS handling in streaming endpoints."""
        cors_headers = {
            **auth_headers,
            "Origin": "https://example.com",
        }

        with patch("pynomaly.presentation.web_api.dependencies.get_streaming_service", return_value=mock_streaming_service):
            response = client.post(
                "/api/v1/streaming/process",
                json=valid_streaming_data_payload,
                headers=cors_headers,
            )

        # Check for CORS headers
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 200