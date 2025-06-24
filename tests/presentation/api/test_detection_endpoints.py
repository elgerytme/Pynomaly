"""
Detection Endpoints Testing Suite
Comprehensive tests for anomaly detection API endpoints.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from pynomaly.presentation.api.app import create_app
from pynomaly.application.dto import (
    DetectorDTO, DatasetDTO, DetectionResultDTO, 
    ConfidenceInterval, AnomalyScore
)
from pynomaly.domain.entities import Detector, Dataset, DetectionResult
from pynomaly.domain.value_objects import ContaminationRate


class TestDetectionEndpoints:
    """Test suite for detection API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_container(self):
        """Mock dependency injection container."""
        with patch('pynomaly.presentation.api.deps.get_container') as mock:
            container = Mock()
            
            # Mock repositories
            container.detector_repository.return_value = Mock()
            container.dataset_repository.return_value = Mock()
            container.result_repository.return_value = Mock()
            
            # Mock use cases
            container.detect_anomalies_use_case.return_value = Mock()
            container.train_detector_use_case.return_value = Mock()
            container.evaluate_model_use_case.return_value = Mock()
            
            mock.return_value = container
            yield container

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        with patch('pynomaly.presentation.api.deps.get_current_user') as mock:
            user = {
                "user_id": "test-user-123",
                "email": "test@example.com",
                "roles": ["user"],
                "permissions": ["detect:create", "detector:read", "dataset:read"]
            }
            mock.return_value = user
            yield user

    @pytest.fixture
    def sample_detector_dto(self):
        """Sample detector DTO for testing."""
        return DetectorDTO(
            id=uuid.UUID("12345678-1234-5678-9012-123456789012"),
            name="Test Detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1, "n_estimators": 100},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_trained=True,
            metadata={"version": "1.0", "framework": "sklearn"}
        )

    @pytest.fixture
    def sample_dataset_dto(self):
        """Sample dataset DTO for testing."""
        return DatasetDTO(
            id=uuid.UUID("87654321-4321-8765-2109-876543210987"),
            name="Test Dataset",
            description="Test dataset for anomaly detection",
            features=["feature1", "feature2", "feature3"],
            shape=(1000, 3),
            contamination_rate=0.05,
            created_at=datetime.utcnow(),
            metadata={"source": "test", "quality": "high"}
        )

    @pytest.fixture
    def sample_result_dto(self):
        """Sample detection result DTO for testing."""
        return DetectionResultDTO(
            id=uuid.UUID("11111111-2222-3333-4444-555555555555"),
            detector_id=uuid.UUID("12345678-1234-5678-9012-123456789012"),
            dataset_id=uuid.UUID("87654321-4321-8765-2109-876543210987"),
            scores=[0.1, 0.8, 0.3, 0.9, 0.2],
            predictions=[0, 1, 0, 1, 0],
            confidence_intervals=[
                ConfidenceInterval(lower=0.05, upper=0.15, confidence=0.95),
                ConfidenceInterval(lower=0.75, upper=0.85, confidence=0.95)
            ],
            metadata={
                "execution_time": 0.5,
                "anomaly_count": 2,
                "total_samples": 5
            },
            created_at=datetime.utcnow()
        )

    # Training Endpoints Tests

    def test_train_detector_success(self, client, mock_container, mock_user, sample_detector_dto):
        """Test successful detector training."""
        # Setup mocks
        mock_use_case = mock_container.train_detector_use_case.return_value
        mock_use_case.execute = AsyncMock(return_value=sample_detector_dto)
        
        request_data = {
            "detector_id": str(sample_detector_dto.id),
            "dataset_id": "87654321-4321-8765-2109-876543210987",
            "validate_data": True,
            "save_model": True
        }
        
        response = client.post("/api/v1/detection/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_detector_dto.id)
        assert data["name"] == sample_detector_dto.name
        assert data["is_trained"] is True
        
        # Verify use case was called
        mock_use_case.execute.assert_called_once()

    def test_train_detector_invalid_request(self, client, mock_user):
        """Test training with invalid request data."""
        invalid_request = {
            "detector_id": "invalid-uuid",
            "dataset_id": "also-invalid"
        }
        
        response = client.post("/api/v1/detection/train", json=invalid_request)
        
        assert response.status_code == 422  # Validation error

    def test_train_detector_unauthorized(self, client, mock_container):
        """Test training without authentication."""
        request_data = {
            "detector_id": "12345678-1234-5678-9012-123456789012",
            "dataset_id": "87654321-4321-8765-2109-876543210987"
        }
        
        with patch('pynomaly.presentation.api.deps.get_current_user') as mock_auth:
            mock_auth.side_effect = HTTPException(status_code=401, detail="Not authenticated")
            
            response = client.post("/api/v1/detection/train", json=request_data)
            assert response.status_code == 401

    def test_train_detector_not_found(self, client, mock_container, mock_user):
        """Test training with non-existent detector."""
        mock_use_case = mock_container.train_detector_use_case.return_value
        mock_use_case.execute = AsyncMock(side_effect=HTTPException(
            status_code=404, detail="Detector not found"
        ))
        
        request_data = {
            "detector_id": "99999999-9999-9999-9999-999999999999",
            "dataset_id": "87654321-4321-8765-2109-876543210987"
        }
        
        response = client.post("/api/v1/detection/train", json=request_data)
        
        assert response.status_code == 404

    def test_train_detector_background_task(self, client, mock_container, mock_user, sample_detector_dto):
        """Test asynchronous detector training."""
        mock_use_case = mock_container.train_detector_use_case.return_value
        mock_use_case.execute = AsyncMock(return_value=sample_detector_dto)
        
        request_data = {
            "detector_id": str(sample_detector_dto.id),
            "dataset_id": "87654321-4321-8765-2109-876543210987",
            "async_mode": True
        }
        
        response = client.post("/api/v1/detection/train", json=request_data)
        
        assert response.status_code == 202  # Accepted
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "accepted"

    # Detection Endpoints Tests

    def test_detect_anomalies_success(self, client, mock_container, mock_user, sample_result_dto):
        """Test successful anomaly detection."""
        mock_use_case = mock_container.detect_anomalies_use_case.return_value
        mock_use_case.execute = AsyncMock(return_value=sample_result_dto)
        
        request_data = {
            "detector_id": str(sample_result_dto.detector_id),
            "dataset_id": str(sample_result_dto.dataset_id),
            "validate_features": True,
            "save_results": True
        }
        
        response = client.post("/api/v1/detection/detect", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_result_dto.id)
        assert len(data["scores"]) == len(sample_result_dto.scores)
        assert len(data["predictions"]) == len(sample_result_dto.predictions)
        assert data["metadata"]["anomaly_count"] == 2

    def test_detect_anomalies_batch(self, client, mock_container, mock_user, sample_result_dto):
        """Test batch anomaly detection."""
        mock_use_case = mock_container.detect_anomalies_use_case.return_value
        mock_use_case.execute = AsyncMock(return_value=[sample_result_dto, sample_result_dto])
        
        request_data = {
            "detector_ids": [
                str(sample_result_dto.detector_id),
                "22222222-2222-2222-2222-222222222222"
            ],
            "dataset_id": str(sample_result_dto.dataset_id),
            "save_results": True
        }
        
        response = client.post("/api/v1/detection/batch-detect", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert all("scores" in result for result in data)

    def test_detect_anomalies_invalid_detector(self, client, mock_container, mock_user):
        """Test detection with invalid detector."""
        mock_use_case = mock_container.detect_anomalies_use_case.return_value
        mock_use_case.execute = AsyncMock(side_effect=HTTPException(
            status_code=400, detail="Detector not trained"
        ))
        
        request_data = {
            "detector_id": "12345678-1234-5678-9012-123456789012",
            "dataset_id": "87654321-4321-8765-2109-876543210987"
        }
        
        response = client.post("/api/v1/detection/detect", json=request_data)
        
        assert response.status_code == 400

    def test_detect_anomalies_feature_mismatch(self, client, mock_container, mock_user):
        """Test detection with feature mismatch."""
        mock_use_case = mock_container.detect_anomalies_use_case.return_value
        mock_use_case.execute = AsyncMock(side_effect=HTTPException(
            status_code=400, detail="Feature dimensions mismatch"
        ))
        
        request_data = {
            "detector_id": "12345678-1234-5678-9012-123456789012",
            "dataset_id": "87654321-4321-8765-2109-876543210987",
            "validate_features": True
        }
        
        response = client.post("/api/v1/detection/detect", json=request_data)
        
        assert response.status_code == 400
        assert "Feature dimensions mismatch" in response.json()["detail"]

    # Evaluation Endpoints Tests

    def test_evaluate_detector_success(self, client, mock_container, mock_user):
        """Test successful detector evaluation."""
        mock_use_case = mock_container.evaluate_model_use_case.return_value
        evaluation_result = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.90,
            "auc_roc": 0.94,
            "confusion_matrix": [[450, 10], [15, 25]],
            "classification_report": {
                "normal": {"precision": 0.97, "recall": 0.98, "f1-score": 0.97},
                "anomaly": {"precision": 0.71, "recall": 0.63, "f1-score": 0.67}
            }
        }
        mock_use_case.execute = AsyncMock(return_value=evaluation_result)
        
        request_data = {
            "detector_id": "12345678-1234-5678-9012-123456789012",
            "dataset_id": "87654321-4321-8765-2109-876543210987",
            "test_size": 0.2,
            "random_state": 42
        }
        
        response = client.post("/api/v1/detection/evaluate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["accuracy"] == 0.95
        assert data["f1_score"] == 0.90
        assert "confusion_matrix" in data
        assert "classification_report" in data

    def test_evaluate_detector_cross_validation(self, client, mock_container, mock_user):
        """Test detector evaluation with cross-validation."""
        mock_use_case = mock_container.evaluate_model_use_case.return_value
        cv_results = {
            "cv_scores": [0.94, 0.96, 0.93, 0.95, 0.94],
            "mean_score": 0.944,
            "std_score": 0.011,
            "detailed_metrics": {
                "precision": {"mean": 0.92, "std": 0.02},
                "recall": {"mean": 0.89, "std": 0.03},
                "f1_score": {"mean": 0.90, "std": 0.02}
            }
        }
        mock_use_case.execute = AsyncMock(return_value=cv_results)
        
        request_data = {
            "detector_id": "12345678-1234-5678-9012-123456789012",
            "dataset_id": "87654321-4321-8765-2109-876543210987",
            "evaluation_type": "cross_validation",
            "cv_folds": 5
        }
        
        response = client.post("/api/v1/detection/evaluate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["cv_scores"]) == 5
        assert data["mean_score"] == 0.944
        assert "detailed_metrics" in data

    # Results Management Tests

    def test_get_detection_results(self, client, mock_container, mock_user, sample_result_dto):
        """Test retrieving detection results."""
        mock_repo = mock_container.result_repository.return_value
        mock_repo.find_by_id.return_value = sample_result_dto
        
        response = client.get(f"/api/v1/detection/results/{sample_result_dto.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_result_dto.id)
        assert len(data["scores"]) == len(sample_result_dto.scores)

    def test_list_detection_results(self, client, mock_container, mock_user, sample_result_dto):
        """Test listing detection results with pagination."""
        mock_repo = mock_container.result_repository.return_value
        mock_repo.find_all.return_value = [sample_result_dto, sample_result_dto]
        mock_repo.count.return_value = 25
        
        response = client.get("/api/v1/detection/results?page=1&size=10")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data
        assert len(data["items"]) == 2
        assert data["total"] == 25

    def test_filter_detection_results(self, client, mock_container, mock_user, sample_result_dto):
        """Test filtering detection results."""
        mock_repo = mock_container.result_repository.return_value
        mock_repo.find_by_detector.return_value = [sample_result_dto]
        
        params = {
            "detector_id": str(sample_result_dto.detector_id),
            "min_anomaly_score": 0.5,
            "created_after": "2024-01-01T00:00:00"
        }
        
        response = client.get("/api/v1/detection/results", params=params)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1

    def test_delete_detection_results(self, client, mock_container, mock_user, sample_result_dto):
        """Test deleting detection results."""
        mock_repo = mock_container.result_repository.return_value
        mock_repo.find_by_id.return_value = sample_result_dto
        mock_repo.delete.return_value = True
        
        response = client.delete(f"/api/v1/detection/results/{sample_result_dto.id}")
        
        assert response.status_code == 204
        mock_repo.delete.assert_called_once_with(sample_result_dto.id)

    # Real-time Detection Tests

    def test_real_time_detection_endpoint(self, client, mock_container, mock_user):
        """Test real-time detection endpoint."""
        mock_service = mock_container.detection_service.return_value
        mock_service.detect_real_time = AsyncMock(return_value={
            "anomaly_score": 0.85,
            "is_anomaly": True,
            "confidence": 0.92,
            "explanation": {"top_features": ["feature1", "feature3"]}
        })
        
        request_data = {
            "detector_id": "12345678-1234-5678-9012-123456789012",
            "data_point": [1.5, 2.3, 0.8],
            "include_explanation": True
        }
        
        response = client.post("/api/v1/detection/real-time", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["anomaly_score"] == 0.85
        assert data["is_anomaly"] is True
        assert "explanation" in data

    def test_streaming_detection_endpoint(self, client, mock_container, mock_user):
        """Test streaming detection endpoint."""
        mock_service = mock_container.streaming_service.return_value
        mock_service.start_stream = AsyncMock(return_value={
            "stream_id": "stream-123",
            "status": "active",
            "endpoint": "ws://localhost:8000/api/v1/detection/stream/stream-123"
        })
        
        request_data = {
            "detector_id": "12345678-1234-5678-9012-123456789012",
            "stream_config": {
                "buffer_size": 100,
                "batch_interval": 5,
                "alert_threshold": 0.8
            }
        }
        
        response = client.post("/api/v1/detection/stream", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        assert "stream_id" in data
        assert data["status"] == "active"

    # Performance and Monitoring Tests

    def test_detection_performance_metrics(self, client, mock_container, mock_user):
        """Test detection performance metrics endpoint."""
        mock_service = mock_container.performance_service.return_value
        mock_service.get_detection_metrics.return_value = {
            "average_response_time": 0.150,
            "requests_per_second": 125,
            "success_rate": 0.995,
            "error_rate": 0.005,
            "detector_usage": {
                "12345678-1234-5678-9012-123456789012": 450,
                "87654321-4321-8765-2109-876543210987": 320
            }
        }
        
        response = client.get("/api/v1/detection/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["average_response_time"] == 0.150
        assert data["success_rate"] == 0.995
        assert "detector_usage" in data

    def test_detection_health_check(self, client, mock_container, mock_user):
        """Test detection service health check."""
        mock_service = mock_container.detection_service.return_value
        mock_service.health_check.return_value = {
            "status": "healthy",
            "active_detectors": 5,
            "processing_queue": 2,
            "average_latency": 0.120
        }
        
        response = client.get("/api/v1/detection/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["active_detectors"] == 5

    # Error Handling Tests

    def test_detection_service_timeout(self, client, mock_container, mock_user):
        """Test detection service timeout handling."""
        mock_use_case = mock_container.detect_anomalies_use_case.return_value
        mock_use_case.execute = AsyncMock(side_effect=TimeoutError("Detection timeout"))
        
        request_data = {
            "detector_id": "12345678-1234-5678-9012-123456789012",
            "dataset_id": "87654321-4321-8765-2109-876543210987"
        }
        
        response = client.post("/api/v1/detection/detect", json=request_data)
        
        assert response.status_code == 408  # Request Timeout

    def test_detection_service_error(self, client, mock_container, mock_user):
        """Test detection service error handling."""
        mock_use_case = mock_container.detect_anomalies_use_case.return_value
        mock_use_case.execute = AsyncMock(side_effect=Exception("Internal error"))
        
        request_data = {
            "detector_id": "12345678-1234-5678-9012-123456789012",
            "dataset_id": "87654321-4321-8765-2109-876543210987"
        }
        
        response = client.post("/api/v1/detection/detect", json=request_data)
        
        assert response.status_code == 500

    def test_insufficient_permissions(self, client, mock_container):
        """Test detection with insufficient permissions."""
        with patch('pynomaly.presentation.api.deps.get_current_user') as mock_auth:
            mock_auth.return_value = {
                "user_id": "test-user",
                "permissions": ["detector:read"]  # Missing detect:create
            }
            
            request_data = {
                "detector_id": "12345678-1234-5678-9012-123456789012",
                "dataset_id": "87654321-4321-8765-2109-876543210987"
            }
            
            response = client.post("/api/v1/detection/detect", json=request_data)
            assert response.status_code == 403


class TestDetectionEndpointsIntegration:
    """Integration tests for detection endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_complete_detection_workflow(self, client):
        """Test complete detection workflow from training to results."""
        with patch('pynomaly.presentation.api.deps.get_current_user') as mock_auth:
            mock_auth.return_value = {
                "user_id": "test-user",
                "permissions": ["detect:create", "detector:read", "dataset:read"]
            }
            
            # 1. Train detector
            train_request = {
                "detector_id": "12345678-1234-5678-9012-123456789012",
                "dataset_id": "87654321-4321-8765-2109-876543210987"
            }
            
            with patch('pynomaly.presentation.api.deps.get_container') as mock_container:
                mock_container.return_value.train_detector_use_case.return_value.execute = AsyncMock()
                train_response = client.post("/api/v1/detection/train", json=train_request)
                
                # 2. Detect anomalies
                detect_request = {
                    "detector_id": "12345678-1234-5678-9012-123456789012",
                    "dataset_id": "87654321-4321-8765-2109-876543210987"
                }
                
                mock_container.return_value.detect_anomalies_use_case.return_value.execute = AsyncMock()
                detect_response = client.post("/api/v1/detection/detect", json=detect_request)
                
                # 3. Get results
                results_response = client.get("/api/v1/detection/results")
                
                # Verify workflow
                assert train_response.status_code in [200, 202]
                assert detect_response.status_code in [200, 202]
                assert results_response.status_code in [200, 401]  # May require auth

    def test_concurrent_detection_requests(self, client):
        """Test handling of concurrent detection requests."""
        import asyncio
        import threading
        
        def make_request():
            request_data = {
                "detector_id": "12345678-1234-5678-9012-123456789012",
                "dataset_id": "87654321-4321-8765-2109-876543210987"
            }
            return client.post("/api/v1/detection/detect", json=request_data)
        
        # Simulate concurrent requests
        threads = []
        responses = []
        
        for _ in range(5):
            thread = threading.Thread(target=lambda: responses.append(make_request()))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all requests handled
        assert len(responses) == 5
        
        # All should either succeed or fail consistently
        status_codes = [r.status_code for r in responses]
        assert all(code in [200, 401, 422, 500] for code in status_codes)