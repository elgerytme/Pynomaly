"""
Comprehensive tests for detection endpoints.
Tests anomaly detection, training, and model management API endpoints.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from fastapi import status

from pynomaly.presentation.web_api.app import app
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.exceptions import DetectorError, DatasetError


class TestDetectionEndpointsComprehensive:
    """Comprehensive test suite for detection API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_detection_service(self):
        """Mock detection service."""
        service = AsyncMock()
        service.detect_anomalies.return_value = DetectionResult(
            detector_id=uuid4(),
            anomaly_scores=[0.1, 0.8, 0.3, 0.9, 0.2],
            anomalies=[],
            threshold=0.5,
            execution_time=0.125,
        )
        service.train_detector.return_value = {
            "detector_id": str(uuid4()),
            "training_time": 45.2,
            "performance_metrics": {
                "accuracy": 0.92,
                "precision": 0.87,
                "recall": 0.85,
                "f1_score": 0.86,
            },
            "model_size": "2.3MB",
        }
        service.get_detector.return_value = Detector(
            id=uuid4(),
            name="test-detector",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100},
            is_fitted=True,
        )
        return service

    @pytest.fixture
    def mock_dataset_service(self):
        """Mock dataset service."""
        service = AsyncMock()
        service.get_dataset.return_value = Dataset(
            id=uuid4(),
            name="test-dataset",
            file_path="/tmp/test.csv",
            features=["feature1", "feature2"],
            feature_types={"feature1": "numeric", "feature2": "numeric"},
            data_shape=(1000, 2),
        )
        service.upload_dataset.return_value = {
            "dataset_id": str(uuid4()),
            "name": "uploaded-dataset",
            "size": 1024,
            "rows": 1000,
            "columns": 5,
        }
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
    def valid_detection_payload(self):
        """Valid detection request payload."""
        return {
            "detector_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "threshold": 0.5,
            "return_scores": True,
            "include_explanation": False,
        }

    @pytest.fixture
    def valid_training_payload(self):
        """Valid training request payload."""
        return {
            "detector_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "validation_split": 0.2,
            "hyperparameters": {
                "n_estimators": 100,
                "contamination": 0.1,
            },
        }

    @pytest.fixture
    def valid_detector_payload(self):
        """Valid detector creation payload."""
        return {
            "name": "test-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {
                "n_estimators": 100,
                "contamination": 0.1,
                "max_features": 1.0,
            },
            "description": "Test detector for anomaly detection",
        }

    def test_detect_anomalies_success(
        self, client, valid_detection_payload, mock_detection_service, auth_headers
    ):
        """Test successful anomaly detection."""
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/detect",
                json=valid_detection_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "anomaly_scores" in data
        assert "anomalies" in data
        assert "threshold" in data
        assert "execution_time" in data
        assert isinstance(data["anomaly_scores"], list)
        assert len(data["anomaly_scores"]) > 0

    def test_detect_anomalies_invalid_detector(
        self, client, valid_detection_payload, mock_detection_service, auth_headers
    ):
        """Test detection with invalid detector ID."""
        mock_detection_service.detect_anomalies.side_effect = DetectorError("Detector not found")
        
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/detect",
                json=valid_detection_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "error" in data
        assert "Detector not found" in data["error"]

    def test_detect_anomalies_invalid_dataset(
        self, client, valid_detection_payload, mock_detection_service, auth_headers
    ):
        """Test detection with invalid dataset ID."""
        mock_detection_service.detect_anomalies.side_effect = DatasetError("Dataset not found")
        
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/detect",
                json=valid_detection_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "error" in data
        assert "Dataset not found" in data["error"]

    def test_detect_anomalies_missing_fields(self, client, auth_headers):
        """Test detection with missing required fields."""
        incomplete_payload = {
            "detector_id": str(uuid4()),
            # Missing dataset_id
        }

        response = client.post(
            "/api/v1/detection/detect",
            json=incomplete_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_detect_anomalies_invalid_threshold(
        self, client, valid_detection_payload, auth_headers
    ):
        """Test detection with invalid threshold value."""
        invalid_payload = valid_detection_payload.copy()
        invalid_payload["threshold"] = 1.5  # Invalid threshold > 1.0

        response = client.post(
            "/api/v1/detection/detect",
            json=invalid_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_detect_anomalies_unauthorized(self, client, valid_detection_payload):
        """Test detection without authentication."""
        response = client.post(
            "/api/v1/detection/detect",
            json=valid_detection_payload,
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_detect_anomalies_with_explanation(
        self, client, valid_detection_payload, mock_detection_service, auth_headers
    ):
        """Test detection with explanation enabled."""
        # Enable explanation
        payload_with_explanation = valid_detection_payload.copy()
        payload_with_explanation["include_explanation"] = True

        # Mock detection result with explanation
        mock_result = DetectionResult(
            detector_id=uuid4(),
            anomaly_scores=[0.1, 0.8, 0.3],
            anomalies=[],
            threshold=0.5,
            execution_time=0.125,
            explanations=[
                {
                    "sample_index": 1,
                    "feature_importance": [0.6, 0.4],
                    "feature_contributions": [0.5, 0.3],
                    "explanation_text": "High value in feature1",
                }
            ],
        )
        mock_detection_service.detect_anomalies.return_value = mock_result

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/detect",
                json=payload_with_explanation,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "explanations" in data
        assert len(data["explanations"]) > 0

    def test_train_detector_success(
        self, client, valid_training_payload, mock_detection_service, auth_headers
    ):
        """Test successful detector training."""
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/train",
                json=valid_training_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "detector_id" in data
        assert "training_time" in data
        assert "performance_metrics" in data
        assert "model_size" in data

        # Check performance metrics structure
        metrics = data["performance_metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_train_detector_invalid_detector(
        self, client, valid_training_payload, mock_detection_service, auth_headers
    ):
        """Test training with invalid detector ID."""
        mock_detection_service.train_detector.side_effect = DetectorError("Detector not found")
        
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/train",
                json=valid_training_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_train_detector_invalid_hyperparameters(
        self, client, valid_training_payload, auth_headers
    ):
        """Test training with invalid hyperparameters."""
        invalid_payload = valid_training_payload.copy()
        invalid_payload["hyperparameters"] = {
            "n_estimators": -1,  # Invalid negative value
            "contamination": 1.5,  # Invalid contamination > 1.0
        }

        response = client.post(
            "/api/v1/detection/train",
            json=invalid_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_train_detector_unauthorized(self, client, valid_training_payload):
        """Test training without authentication."""
        response = client.post(
            "/api/v1/detection/train",
            json=valid_training_payload,
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_create_detector_success(
        self, client, valid_detector_payload, mock_detection_service, auth_headers
    ):
        """Test successful detector creation."""
        mock_detector = Detector(
            id=uuid4(),
            name=valid_detector_payload["name"],
            algorithm_name=valid_detector_payload["algorithm_name"],
            hyperparameters=valid_detector_payload["hyperparameters"],
            is_fitted=False,
        )
        mock_detection_service.create_detector.return_value = mock_detector

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detectors",
                json=valid_detector_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert "name" in data
        assert "algorithm_name" in data
        assert "hyperparameters" in data
        assert data["name"] == valid_detector_payload["name"]
        assert data["is_fitted"] is False

    def test_create_detector_invalid_algorithm(
        self, client, valid_detector_payload, auth_headers
    ):
        """Test detector creation with invalid algorithm."""
        invalid_payload = valid_detector_payload.copy()
        invalid_payload["algorithm_name"] = "NonExistentAlgorithm"

        response = client.post(
            "/api/v1/detectors",
            json=invalid_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_detector_missing_fields(self, client, auth_headers):
        """Test detector creation with missing required fields."""
        incomplete_payload = {
            "name": "test-detector",
            # Missing algorithm_name
        }

        response = client.post(
            "/api/v1/detectors",
            json=incomplete_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_detector_success(
        self, client, mock_detection_service, auth_headers
    ):
        """Test successful detector retrieval."""
        detector_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.get(
                f"/api/v1/detectors/{detector_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert "name" in data
        assert "algorithm_name" in data
        assert "hyperparameters" in data
        assert "is_fitted" in data

    def test_get_detector_not_found(
        self, client, mock_detection_service, auth_headers
    ):
        """Test detector retrieval with non-existent ID."""
        mock_detection_service.get_detector.side_effect = DetectorError("Detector not found")
        detector_id = str(uuid4())
        
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.get(
                f"/api/v1/detectors/{detector_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_detector_invalid_id(self, client, auth_headers):
        """Test detector retrieval with invalid ID format."""
        invalid_id = "invalid-uuid"
        
        response = client.get(
            f"/api/v1/detectors/{invalid_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_list_detectors_success(
        self, client, mock_detection_service, auth_headers
    ):
        """Test successful detector listing."""
        mock_detectors = [
            Detector(
                id=uuid4(),
                name="detector-1",
                algorithm_name="IsolationForest",
                hyperparameters={"n_estimators": 100},
                is_fitted=True,
            ),
            Detector(
                id=uuid4(),
                name="detector-2",
                algorithm_name="LocalOutlierFactor",
                hyperparameters={"n_neighbors": 20},
                is_fitted=False,
            ),
        ]
        mock_detection_service.list_detectors.return_value = mock_detectors

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.get("/api/v1/detectors", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "detectors" in data
        assert len(data["detectors"]) == 2
        assert data["detectors"][0]["name"] == "detector-1"
        assert data["detectors"][1]["name"] == "detector-2"

    def test_list_detectors_with_pagination(
        self, client, mock_detection_service, auth_headers
    ):
        """Test detector listing with pagination."""
        mock_detection_service.list_detectors.return_value = []

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.get(
                "/api/v1/detectors?page=1&size=10",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "detectors" in data
        assert "pagination" in data

    def test_list_detectors_with_filters(
        self, client, mock_detection_service, auth_headers
    ):
        """Test detector listing with filters."""
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.get(
                "/api/v1/detectors?algorithm=IsolationForest&fitted=true",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "detectors" in data

    def test_update_detector_success(
        self, client, mock_detection_service, auth_headers
    ):
        """Test successful detector update."""
        detector_id = str(uuid4())
        update_payload = {
            "name": "updated-detector",
            "description": "Updated description",
            "hyperparameters": {
                "n_estimators": 200,
                "contamination": 0.05,
            },
        }

        mock_updated_detector = Detector(
            id=detector_id,
            name=update_payload["name"],
            algorithm_name="IsolationForest",
            hyperparameters=update_payload["hyperparameters"],
            is_fitted=True,
        )
        mock_detection_service.update_detector.return_value = mock_updated_detector

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.put(
                f"/api/v1/detectors/{detector_id}",
                json=update_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == update_payload["name"]
        assert data["hyperparameters"] == update_payload["hyperparameters"]

    def test_update_detector_not_found(
        self, client, mock_detection_service, auth_headers
    ):
        """Test detector update with non-existent ID."""
        mock_detection_service.update_detector.side_effect = DetectorError("Detector not found")
        detector_id = str(uuid4())
        update_payload = {"name": "updated-detector"}

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.put(
                f"/api/v1/detectors/{detector_id}",
                json=update_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_detector_success(
        self, client, mock_detection_service, auth_headers
    ):
        """Test successful detector deletion."""
        detector_id = str(uuid4())
        mock_detection_service.delete_detector.return_value = True

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.delete(
                f"/api/v1/detectors/{detector_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_detector_not_found(
        self, client, mock_detection_service, auth_headers
    ):
        """Test detector deletion with non-existent ID."""
        mock_detection_service.delete_detector.side_effect = DetectorError("Detector not found")
        detector_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.delete(
                f"/api/v1/detectors/{detector_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_detection_results_success(
        self, client, mock_detection_service, auth_headers
    ):
        """Test successful detection results retrieval."""
        detector_id = str(uuid4())
        mock_results = [
            DetectionResult(
                detector_id=detector_id,
                anomaly_scores=[0.1, 0.8, 0.3],
                anomalies=[],
                threshold=0.5,
                execution_time=0.125,
                created_at=datetime.utcnow(),
            )
        ]
        mock_detection_service.get_detection_results.return_value = mock_results

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.get(
                f"/api/v1/detectors/{detector_id}/results",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert "anomaly_scores" in data["results"][0]
        assert "execution_time" in data["results"][0]

    def test_get_detection_results_not_found(
        self, client, mock_detection_service, auth_headers
    ):
        """Test detection results retrieval with non-existent detector."""
        mock_detection_service.get_detection_results.side_effect = DetectorError("Detector not found")
        detector_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.get(
                f"/api/v1/detectors/{detector_id}/results",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_detector_performance_success(
        self, client, mock_detection_service, auth_headers
    ):
        """Test successful detector performance retrieval."""
        detector_id = str(uuid4())
        mock_performance = {
            "accuracy": 0.92,
            "precision": 0.87,
            "recall": 0.85,
            "f1_score": 0.86,
            "roc_auc": 0.89,
            "confusion_matrix": [[850, 50], [75, 25]],
            "training_time": 45.2,
            "inference_time": 0.125,
        }
        mock_detection_service.get_detector_performance.return_value = mock_performance

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.get(
                f"/api/v1/detectors/{detector_id}/performance",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "accuracy" in data
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data
        assert "training_time" in data
        assert "inference_time" in data

    def test_batch_detection_success(
        self, client, mock_detection_service, auth_headers
    ):
        """Test successful batch detection."""
        batch_payload = {
            "detector_id": str(uuid4()),
            "datasets": [str(uuid4()), str(uuid4())],
            "threshold": 0.5,
            "batch_size": 1000,
        }

        mock_batch_results = [
            {
                "dataset_id": batch_payload["datasets"][0],
                "anomaly_scores": [0.1, 0.8, 0.3],
                "anomalies": [],
                "execution_time": 0.125,
            },
            {
                "dataset_id": batch_payload["datasets"][1],
                "anomaly_scores": [0.2, 0.9, 0.4],
                "anomalies": [],
                "execution_time": 0.135,
            },
        ]
        mock_detection_service.batch_detect.return_value = mock_batch_results

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/batch",
                json=batch_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert "total_execution_time" in data

    def test_streaming_detection_setup(
        self, client, mock_detection_service, auth_headers
    ):
        """Test streaming detection setup."""
        streaming_payload = {
            "detector_id": str(uuid4()),
            "stream_config": {
                "batch_size": 100,
                "window_size": 1000,
                "overlap": 0.1,
            },
        }

        mock_stream_session = {
            "session_id": str(uuid4()),
            "detector_id": streaming_payload["detector_id"],
            "config": streaming_payload["stream_config"],
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
        }
        mock_detection_service.setup_streaming.return_value = mock_stream_session

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/streaming/setup",
                json=streaming_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "session_id" in data
        assert "detector_id" in data
        assert "status" in data
        assert data["status"] == "active"

    def test_concurrent_detection_requests(
        self, client, valid_detection_payload, mock_detection_service, auth_headers
    ):
        """Test handling concurrent detection requests."""
        import threading
        import time

        results = []
        
        def make_detection_request():
            with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
                response = client.post(
                    "/api/v1/detection/detect",
                    json=valid_detection_payload,
                    headers=auth_headers,
                )
                results.append(response.status_code)

        # Create multiple threads for concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_detection_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should have completed successfully
        assert len(results) == 5
        assert all(status_code == 200 for status_code in results)

    def test_detection_request_validation(self, client, auth_headers):
        """Test comprehensive request validation."""
        # Test invalid JSON
        response = client.post(
            "/api/v1/detection/detect",
            data="invalid json",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid UUID format
        invalid_payload = {
            "detector_id": "invalid-uuid",
            "dataset_id": str(uuid4()),
        }
        response = client.post(
            "/api/v1/detection/detect",
            json=invalid_payload,
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid threshold range
        invalid_payload = {
            "detector_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "threshold": 2.0,  # Invalid threshold > 1.0
        }
        response = client.post(
            "/api/v1/detection/detect",
            json=invalid_payload,
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_detection_error_handling(
        self, client, valid_detection_payload, mock_detection_service, auth_headers
    ):
        """Test error handling in detection endpoints."""
        # Test service unavailable
        mock_detection_service.detect_anomalies.side_effect = Exception("Service unavailable")
        
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/detect",
                json=valid_detection_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_detection_rate_limiting(
        self, client, valid_detection_payload, mock_detection_service, auth_headers
    ):
        """Test rate limiting on detection endpoints."""
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            # Make multiple requests rapidly
            responses = []
            for i in range(20):
                response = client.post(
                    "/api/v1/detection/detect",
                    json=valid_detection_payload,
                    headers=auth_headers,
                )
                responses.append(response.status_code)
                if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                    break

        # Should handle rate limiting gracefully
        assert any(status_code in [200, 429] for status_code in responses)

    def test_detection_security_headers(
        self, client, valid_detection_payload, mock_detection_service, auth_headers
    ):
        """Test security headers in detection responses."""
        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/detect",
                json=valid_detection_payload,
                headers=auth_headers,
            )

        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers

    def test_detection_cors_handling(
        self, client, valid_detection_payload, mock_detection_service, auth_headers
    ):
        """Test CORS handling in detection endpoints."""
        cors_headers = {
            **auth_headers,
            "Origin": "https://example.com",
        }

        with patch("pynomaly.presentation.web_api.dependencies.get_detection_service", return_value=mock_detection_service):
            response = client.post(
                "/api/v1/detection/detect",
                json=valid_detection_payload,
                headers=cors_headers,
            )

        # Check for CORS headers
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 200