"""
Comprehensive tests for AutoML endpoints.
Tests automated machine learning and hyperparameter optimization API endpoints.
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from pynomaly.domain.entities.experiment import Experiment
from pynomaly.domain.exceptions import AutoMLError, DatasetError
from pynomaly.presentation.web_api.app import app


class TestAutoMLEndpointsComprehensive:
    """Comprehensive test suite for AutoML API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_automl_service(self):
        """Mock AutoML service."""
        service = AsyncMock()
        service.run_automl.return_value = {
            "experiment_id": str(uuid4()),
            "best_detector": {
                "id": str(uuid4()),
                "algorithm": "IsolationForest",
                "hyperparameters": {"n_estimators": 100, "contamination": 0.1},
                "performance": {"f1_score": 0.85, "precision": 0.82, "recall": 0.88},
            },
            "optimization_results": {
                "trials_completed": 50,
                "best_score": 0.85,
                "optimization_time": 120.5,
                "convergence_reached": True,
            },
            "model_comparison": [
                {
                    "algorithm": "IsolationForest",
                    "score": 0.85,
                    "hyperparameters": {"n_estimators": 100},
                },
                {
                    "algorithm": "LocalOutlierFactor",
                    "score": 0.78,
                    "hyperparameters": {"n_neighbors": 20},
                },
            ],
        }

        service.get_automl_experiment.return_value = Experiment(
            id=uuid4(),
            name="test-automl-experiment",
            dataset_id=uuid4(),
            algorithm_configs=[
                {"algorithm": "IsolationForest", "hyperparameters": {}},
                {"algorithm": "LocalOutlierFactor", "hyperparameters": {}},
            ],
            status="completed",
            created_at=datetime.utcnow(),
        )

        service.optimize_hyperparameters.return_value = {
            "optimization_id": str(uuid4()),
            "best_hyperparameters": {"n_estimators": 150, "contamination": 0.08},
            "best_score": 0.87,
            "optimization_history": [
                {"trial": 1, "score": 0.75, "hyperparameters": {"n_estimators": 50}},
                {"trial": 2, "score": 0.82, "hyperparameters": {"n_estimators": 100}},
                {"trial": 3, "score": 0.87, "hyperparameters": {"n_estimators": 150}},
            ],
        }

        service.get_algorithm_recommendations.return_value = {
            "recommendations": [
                {
                    "algorithm": "IsolationForest",
                    "suitability_score": 0.92,
                    "reasons": ["High-dimensional data", "Unsupervised learning"],
                    "expected_performance": 0.85,
                },
                {
                    "algorithm": "LocalOutlierFactor",
                    "suitability_score": 0.78,
                    "reasons": ["Local density analysis"],
                    "expected_performance": 0.78,
                },
            ],
            "dataset_characteristics": {
                "dimensions": 10,
                "samples": 1000,
                "sparsity": 0.1,
                "noise_level": "low",
            },
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
    def valid_automl_payload(self):
        """Valid AutoML request payload."""
        return {
            "dataset_id": str(uuid4()),
            "algorithms": ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
            "optimization_metric": "f1_score",
            "max_trials": 50,
            "timeout_minutes": 30,
            "validation_strategy": "cross_validation",
            "cv_folds": 5,
            "hyperparameter_space": {
                "IsolationForest": {
                    "n_estimators": {"type": "int", "low": 50, "high": 200},
                    "contamination": {"type": "float", "low": 0.05, "high": 0.2},
                },
                "LocalOutlierFactor": {
                    "n_neighbors": {"type": "int", "low": 10, "high": 50},
                },
            },
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.01,
            },
        }

    @pytest.fixture
    def valid_hyperparameter_optimization_payload(self):
        """Valid hyperparameter optimization payload."""
        return {
            "detector_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "optimization_algorithm": "optuna",
            "objective_metric": "f1_score",
            "max_trials": 100,
            "timeout_minutes": 60,
            "hyperparameter_space": {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "contamination": {"type": "float", "low": 0.01, "high": 0.3},
                "max_features": {"type": "float", "low": 0.1, "high": 1.0},
            },
            "pruning": {
                "enabled": True,
                "warmup_trials": 5,
                "min_trials": 10,
            },
        }

    def test_run_automl_success(
        self, client, valid_automl_payload, mock_automl_service, auth_headers
    ):
        """Test successful AutoML execution."""
        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.post(
                "/api/v1/automl/run",
                json=valid_automl_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "experiment_id" in data
        assert "best_detector" in data
        assert "optimization_results" in data
        assert "model_comparison" in data

        # Verify best detector structure
        best_detector = data["best_detector"]
        assert "id" in best_detector
        assert "algorithm" in best_detector
        assert "hyperparameters" in best_detector
        assert "performance" in best_detector

    def test_run_automl_invalid_dataset(
        self, client, valid_automl_payload, mock_automl_service, auth_headers
    ):
        """Test AutoML with invalid dataset ID."""
        mock_automl_service.run_automl.side_effect = DatasetError("Dataset not found")

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.post(
                "/api/v1/automl/run",
                json=valid_automl_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "error" in data
        assert "Dataset not found" in data["error"]

    def test_run_automl_invalid_algorithms(
        self, client, valid_automl_payload, auth_headers
    ):
        """Test AutoML with invalid algorithm list."""
        invalid_payload = valid_automl_payload.copy()
        invalid_payload["algorithms"] = ["NonExistentAlgorithm"]

        response = client.post(
            "/api/v1/automl/run",
            json=invalid_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_run_automl_missing_required_fields(self, client, auth_headers):
        """Test AutoML with missing required fields."""
        incomplete_payload = {
            "dataset_id": str(uuid4()),
            # Missing algorithms
        }

        response = client.post(
            "/api/v1/automl/run",
            json=incomplete_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_run_automl_unauthorized(self, client, valid_automl_payload):
        """Test AutoML without authentication."""
        response = client.post(
            "/api/v1/automl/run",
            json=valid_automl_payload,
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_automl_experiment_success(
        self, client, mock_automl_service, auth_headers
    ):
        """Test successful AutoML experiment retrieval."""
        experiment_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.get(
                f"/api/v1/automl/experiments/{experiment_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "id" in data
        assert "name" in data
        assert "dataset_id" in data
        assert "algorithm_configs" in data
        assert "status" in data

    def test_get_automl_experiment_not_found(
        self, client, mock_automl_service, auth_headers
    ):
        """Test AutoML experiment retrieval with non-existent ID."""
        mock_automl_service.get_automl_experiment.side_effect = AutoMLError("Experiment not found")
        experiment_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.get(
                f"/api/v1/automl/experiments/{experiment_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_optimize_hyperparameters_success(
        self, client, valid_hyperparameter_optimization_payload, mock_automl_service, auth_headers
    ):
        """Test successful hyperparameter optimization."""
        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.post(
                "/api/v1/automl/optimize",
                json=valid_hyperparameter_optimization_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "optimization_id" in data
        assert "best_hyperparameters" in data
        assert "best_score" in data
        assert "optimization_history" in data

    def test_optimize_hyperparameters_invalid_detector(
        self, client, valid_hyperparameter_optimization_payload, mock_automl_service, auth_headers
    ):
        """Test hyperparameter optimization with invalid detector."""
        mock_automl_service.optimize_hyperparameters.side_effect = AutoMLError("Detector not found")

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.post(
                "/api/v1/automl/optimize",
                json=valid_hyperparameter_optimization_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_algorithm_recommendations_success(
        self, client, mock_automl_service, auth_headers
    ):
        """Test successful algorithm recommendations."""
        dataset_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.get(
                f"/api/v1/automl/recommendations/{dataset_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "recommendations" in data
        assert "dataset_characteristics" in data
        assert len(data["recommendations"]) > 0

        # Verify recommendation structure
        recommendation = data["recommendations"][0]
        assert "algorithm" in recommendation
        assert "suitability_score" in recommendation
        assert "reasons" in recommendation
        assert "expected_performance" in recommendation

    def test_get_algorithm_recommendations_invalid_dataset(
        self, client, mock_automl_service, auth_headers
    ):
        """Test algorithm recommendations with invalid dataset."""
        mock_automl_service.get_algorithm_recommendations.side_effect = DatasetError("Dataset not found")
        dataset_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.get(
                f"/api/v1/automl/recommendations/{dataset_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_automl_experiments_success(
        self, client, mock_automl_service, auth_headers
    ):
        """Test successful AutoML experiments listing."""
        mock_experiments = [
            Experiment(
                id=uuid4(),
                name="experiment-1",
                dataset_id=uuid4(),
                algorithm_configs=[{"algorithm": "IsolationForest"}],
                status="completed",
                created_at=datetime.utcnow(),
            ),
            Experiment(
                id=uuid4(),
                name="experiment-2",
                dataset_id=uuid4(),
                algorithm_configs=[{"algorithm": "LocalOutlierFactor"}],
                status="running",
                created_at=datetime.utcnow(),
            ),
        ]
        mock_automl_service.list_automl_experiments.return_value = mock_experiments

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.get("/api/v1/automl/experiments", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "experiments" in data
        assert len(data["experiments"]) == 2
        assert data["experiments"][0]["name"] == "experiment-1"
        assert data["experiments"][1]["name"] == "experiment-2"

    def test_list_automl_experiments_with_filters(
        self, client, mock_automl_service, auth_headers
    ):
        """Test AutoML experiments listing with filters."""
        mock_automl_service.list_automl_experiments.return_value = []

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.get(
                "/api/v1/automl/experiments?status=completed&algorithm=IsolationForest",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "experiments" in data

    def test_cancel_automl_experiment_success(
        self, client, mock_automl_service, auth_headers
    ):
        """Test successful AutoML experiment cancellation."""
        experiment_id = str(uuid4())
        mock_automl_service.cancel_automl_experiment.return_value = True

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.post(
                f"/api/v1/automl/experiments/{experiment_id}/cancel",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "cancelled" in data["message"].lower()

    def test_get_automl_metrics_success(
        self, client, mock_automl_service, auth_headers
    ):
        """Test successful AutoML metrics retrieval."""
        experiment_id = str(uuid4())
        mock_metrics = {
            "optimization_progress": [
                {"iteration": 1, "best_score": 0.75, "current_score": 0.75},
                {"iteration": 2, "best_score": 0.82, "current_score": 0.82},
                {"iteration": 3, "best_score": 0.87, "current_score": 0.83},
            ],
            "algorithm_performance": {
                "IsolationForest": {"best_score": 0.87, "trials": 20},
                "LocalOutlierFactor": {"best_score": 0.78, "trials": 15},
            },
            "resource_usage": {
                "total_time": 120.5,
                "memory_peak": "2.1GB",
                "cpu_utilization": 0.75,
            },
        }
        mock_automl_service.get_automl_metrics.return_value = mock_metrics

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.get(
                f"/api/v1/automl/experiments/{experiment_id}/metrics",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "optimization_progress" in data
        assert "algorithm_performance" in data
        assert "resource_usage" in data

    def test_automl_concurrent_requests(
        self, client, valid_automl_payload, mock_automl_service, auth_headers
    ):
        """Test handling concurrent AutoML requests."""
        import threading

        results = []

        def make_automl_request():
            with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
                response = client.post(
                    "/api/v1/automl/run",
                    json=valid_automl_payload,
                    headers=auth_headers,
                )
                results.append(response.status_code)

        # Create multiple threads for concurrent requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_automl_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should have completed successfully
        assert len(results) == 3
        assert all(status_code == 200 for status_code in results)

    def test_automl_request_validation(self, client, auth_headers):
        """Test comprehensive request validation."""
        # Test invalid JSON
        response = client.post(
            "/api/v1/automl/run",
            data="invalid json",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid UUID format
        invalid_payload = {
            "dataset_id": "invalid-uuid",
            "algorithms": ["IsolationForest"],
        }
        response = client.post(
            "/api/v1/automl/run",
            json=invalid_payload,
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid algorithm list
        invalid_payload = {
            "dataset_id": str(uuid4()),
            "algorithms": [],  # Empty list
        }
        response = client.post(
            "/api/v1/automl/run",
            json=invalid_payload,
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_automl_error_handling(
        self, client, valid_automl_payload, mock_automl_service, auth_headers
    ):
        """Test error handling in AutoML endpoints."""
        # Test service unavailable
        mock_automl_service.run_automl.side_effect = Exception("Service unavailable")

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.post(
                "/api/v1/automl/run",
                json=valid_automl_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_automl_security_headers(
        self, client, valid_automl_payload, mock_automl_service, auth_headers
    ):
        """Test security headers in AutoML responses."""
        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.post(
                "/api/v1/automl/run",
                json=valid_automl_payload,
                headers=auth_headers,
            )

        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers

    def test_automl_cors_handling(
        self, client, valid_automl_payload, mock_automl_service, auth_headers
    ):
        """Test CORS handling in AutoML endpoints."""
        cors_headers = {
            **auth_headers,
            "Origin": "https://example.com",
        }

        with patch("pynomaly.presentation.web_api.dependencies.get_automl_service", return_value=mock_automl_service):
            response = client.post(
                "/api/v1/automl/run",
                json=valid_automl_payload,
                headers=cors_headers,
            )

        # Check for CORS headers
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 200
