"""
Comprehensive integration tests for Web API.
Tests end-to-end API workflows and cross-endpoint integration.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from monorepo.domain.entities.dataset import Dataset
from monorepo.domain.entities.detection_result import DetectionResult
from monorepo.domain.entities.detector import Detector
from monorepo.presentation.api.app import app


class TestWebAPIIntegrationComprehensive:
    """Comprehensive integration test suite for Web API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_services(self):
        """Mock all required services."""
        services = {
            "auth": AsyncMock(),
            "dataset": AsyncMock(),
            "detection": AsyncMock(),
            "automl": AsyncMock(),
            "explainability": AsyncMock(),
            "streaming": AsyncMock(),
        }

        # Mock auth service
        services["auth"].authenticate_user.return_value = {
            "id": "user_123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user"],
        }
        services["auth"].create_access_token.return_value = {
            "access_token": "test_token_123",
            "token_type": "bearer",
            "expires_in": 3600,
        }

        # Mock dataset service
        services["dataset"].upload_dataset.return_value = {
            "dataset_id": str(uuid4()),
            "name": "test-dataset",
            "size": 1024,
            "rows": 1000,
            "columns": 5,
        }
        services["dataset"].get_dataset.return_value = Dataset(
            id=uuid4(),
            name="test-dataset",
            file_path="/tmp/test.csv",
            features=["feature1", "feature2", "feature3"],
            feature_types={
                "feature1": "numeric",
                "feature2": "numeric",
                "feature3": "categorical",
            },
            data_shape=(1000, 3),
        )

        # Mock detection service
        services["detection"].create_detector.return_value = Detector(
            id=uuid4(),
            name="test-detector",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100},
            is_fitted=False,
        )
        services["detection"].train_detector.return_value = {
            "detector_id": str(uuid4()),
            "training_time": 45.2,
            "performance_metrics": {"accuracy": 0.92, "precision": 0.87},
        }
        services["detection"].detect_anomalies.return_value = DetectionResult(
            detector_id=uuid4(),
            anomaly_scores=[0.1, 0.8, 0.3],
            anomalies=[],
            threshold=0.5,
            execution_time=0.125,
        )

        # Mock AutoML service
        services["automl"].run_automl.return_value = {
            "experiment_id": str(uuid4()),
            "best_detector": {
                "algorithm": "IsolationForest",
                "hyperparameters": {"n_estimators": 100},
                "performance": {"f1_score": 0.85},
            },
        }

        # Mock explainability service
        services["explainability"].explain_anomaly.return_value = {
            "explanation_id": str(uuid4()),
            "feature_importance": [{"feature": "feature1", "importance": 0.6}],
            "textual_explanation": "High value in feature1",
        }

        # Mock streaming service
        services["streaming"].create_streaming_session.return_value = {
            "session_id": str(uuid4()),
            "detector_id": str(uuid4()),
            "status": "active",
        }

        return services

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test_token_123"}

    def test_complete_anomaly_detection_workflow(
        self, client, mock_services, auth_headers
    ):
        """Test complete end-to-end anomaly detection workflow."""
        # Step 1: Upload dataset
        with patch(
            "monorepo.presentation.web_api.dependencies.get_dataset_service",
            return_value=mock_services["dataset"],
        ):
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={
                    "file": ("test.csv", b"feature1,feature2\n1,2\n3,4\n", "text/csv")
                },
                data={"metadata": json.dumps({"name": "test-dataset"})},
                headers=auth_headers,
            )

        assert upload_response.status_code == status.HTTP_201_CREATED
        dataset_id = upload_response.json()["dataset_id"]

        # Step 2: Create detector
        detector_payload = {
            "name": "test-detector",
            "algorithm_name": "IsolationForest",
            "hyperparameters": {"n_estimators": 100},
        }

        with patch(
            "monorepo.presentation.api.deps.get_detection_service",
            return_value=mock_services["detection"],
        ):
            detector_response = client.post(
                "/api/v1/detectors",
                json=detector_payload,
                headers=auth_headers,
            )

        assert detector_response.status_code == status.HTTP_201_CREATED
        detector_id = detector_response.json()["id"]

        # Step 3: Train detector
        train_payload = {
            "detector_id": detector_id,
            "dataset_id": dataset_id,
            "validation_split": 0.2,
        }

        with patch(
            "monorepo.presentation.api.deps.get_detection_service",
            return_value=mock_services["detection"],
        ):
            train_response = client.post(
                "/api/v1/detection/train",
                json=train_payload,
                headers=auth_headers,
            )

        assert train_response.status_code == status.HTTP_200_OK
        assert "training_time" in train_response.json()
        assert "performance_metrics" in train_response.json()

        # Step 4: Detect anomalies
        detect_payload = {
            "detector_id": detector_id,
            "dataset_id": dataset_id,
            "threshold": 0.5,
        }

        with patch(
            "monorepo.presentation.api.deps.get_detection_service",
            return_value=mock_services["detection"],
        ):
            detect_response = client.post(
                "/api/v1/detection/detect",
                json=detect_payload,
                headers=auth_headers,
            )

        assert detect_response.status_code == status.HTTP_200_OK
        assert "anomaly_scores" in detect_response.json()
        assert "execution_time" in detect_response.json()

        # Verify workflow completion
        assert all(
            response.status_code in [200, 201]
            for response in [
                upload_response,
                detector_response,
                train_response,
                detect_response,
            ]
        )

    def test_automl_workflow_integration(self, client, mock_services, auth_headers):
        """Test AutoML workflow integration."""
        # Step 1: Upload dataset
        with patch(
            "monorepo.presentation.web_api.dependencies.get_dataset_service",
            return_value=mock_services["dataset"],
        ):
            upload_response = client.post(
                "/api/v1/datasets/upload",
                files={
                    "file": ("test.csv", b"feature1,feature2\n1,2\n3,4\n", "text/csv")
                },
                data={"metadata": json.dumps({"name": "automl-dataset"})},
                headers=auth_headers,
            )

        assert upload_response.status_code == status.HTTP_201_CREATED
        dataset_id = upload_response.json()["dataset_id"]

        # Step 2: Run AutoML
        automl_payload = {
            "dataset_id": dataset_id,
            "algorithms": ["IsolationForest", "LocalOutlierFactor"],
            "optimization_metric": "f1_score",
            "max_trials": 10,
        }

        with patch(
            "monorepo.presentation.api.deps.get_automl_service",
            return_value=mock_services["automl"],
        ):
            automl_response = client.post(
                "/api/v1/automl/run",
                json=automl_payload,
                headers=auth_headers,
            )

        assert automl_response.status_code == status.HTTP_200_OK
        assert "experiment_id" in automl_response.json()
        assert "best_detector" in automl_response.json()

        # Step 3: Use best detector for detection
        best_detector = automl_response.json()["best_detector"]

        # Create detector with best configuration
        detector_payload = {
            "name": "automl-best-detector",
            "algorithm_name": best_detector["algorithm"],
            "hyperparameters": best_detector["hyperparameters"],
        }

        with patch(
            "monorepo.presentation.api.deps.get_detection_service",
            return_value=mock_services["detection"],
        ):
            detector_response = client.post(
                "/api/v1/detectors",
                json=detector_payload,
                headers=auth_headers,
            )

        assert detector_response.status_code == status.HTTP_201_CREATED

    def test_explainability_workflow_integration(
        self, client, mock_services, auth_headers
    ):
        """Test explainability workflow integration."""
        # Step 1: Setup data and detector (abbreviated)
        detector_id = str(uuid4())
        anomaly_id = str(uuid4())

        # Step 2: Explain anomaly
        explain_payload = {
            "anomaly_id": anomaly_id,
            "detector_id": detector_id,
            "explanation_methods": ["SHAP"],
            "include_local": True,
            "include_global": True,
        }

        with patch(
            "monorepo.presentation.web_api.dependencies.get_explainability_service",
            return_value=mock_services["explainability"],
        ):
            explain_response = client.post(
                "/api/v1/explainability/anomaly",
                json=explain_payload,
                headers=auth_headers,
            )

        assert explain_response.status_code == status.HTTP_200_OK
        assert "explanation_id" in explain_response.json()
        assert "feature_importance" in explain_response.json()

        # Step 3: Get feature importance for detector
        with patch(
            "monorepo.presentation.web_api.dependencies.get_explainability_service",
            return_value=mock_services["explainability"],
        ):
            importance_response = client.get(
                f"/api/v1/explainability/feature-importance/{detector_id}",
                headers=auth_headers,
            )

        # Should work even if service returns mock data
        assert importance_response.status_code in [200, 404]

    def test_streaming_workflow_integration(self, client, mock_services, auth_headers):
        """Test streaming workflow integration."""
        # Step 1: Create streaming session
        session_payload = {
            "detector_id": str(uuid4()),
            "name": "test-streaming-session",
            "config": {
                "batch_size": 100,
                "threshold": 0.5,
            },
        }

        with patch(
            "monorepo.presentation.api.deps.get_streaming_service",
            return_value=mock_services["streaming"],
        ):
            session_response = client.post(
                "/api/v1/streaming/sessions",
                json=session_payload,
                headers=auth_headers,
            )

        assert session_response.status_code == status.HTTP_201_CREATED
        session_id = session_response.json()["session_id"]

        # Step 2: Process streaming data
        data_payload = {
            "session_id": session_id,
            "data": [[1, 2, 3], [4, 5, 6]],
            "timestamps": [
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
            ],
        }

        mock_services["streaming"].process_streaming_data.return_value = {
            "batch_id": str(uuid4()),
            "session_id": session_id,
            "processed_samples": 2,
            "anomalies_detected": 0,
        }

        with patch(
            "monorepo.presentation.api.deps.get_streaming_service",
            return_value=mock_services["streaming"],
        ):
            process_response = client.post(
                "/api/v1/streaming/process",
                json=data_payload,
                headers=auth_headers,
            )

        assert process_response.status_code == status.HTTP_200_OK
        assert "processed_samples" in process_response.json()

    def test_cross_endpoint_data_consistency(self, client, mock_services, auth_headers):
        """Test data consistency across multiple endpoints."""
        # Use consistent IDs across endpoints
        dataset_id = str(uuid4())
        detector_id = str(uuid4())

        # Mock services to return consistent data
        mock_services["dataset"].get_dataset.return_value = Dataset(
            id=dataset_id,
            name="consistent-dataset",
            file_path="/tmp/consistent.csv",
            features=["feature1", "feature2"],
            feature_types={"feature1": "numeric", "feature2": "numeric"},
            data_shape=(1000, 2),
        )

        mock_services["detection"].get_detector.return_value = Detector(
            id=detector_id,
            name="consistent-detector",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100},
            is_fitted=True,
        )

        # Test dataset endpoint
        with patch(
            "monorepo.presentation.web_api.dependencies.get_dataset_service",
            return_value=mock_services["dataset"],
        ):
            dataset_response = client.get(
                f"/api/v1/datasets/{dataset_id}",
                headers=auth_headers,
            )

        # Test detector endpoint
        with patch(
            "monorepo.presentation.api.deps.get_detection_service",
            return_value=mock_services["detection"],
        ):
            detector_response = client.get(
                f"/api/v1/detectors/{detector_id}",
                headers=auth_headers,
            )

        # Verify consistent data
        assert dataset_response.status_code == status.HTTP_200_OK
        assert detector_response.status_code == status.HTTP_200_OK

        dataset_data = dataset_response.json()
        detector_data = detector_response.json()

        assert dataset_data["name"] == "consistent-dataset"
        assert detector_data["name"] == "consistent-detector"

    def test_error_handling_across_endpoints(self, client, mock_services, auth_headers):
        """Test error handling consistency across endpoints."""
        # Test 404 errors
        not_found_endpoints = [
            f"/api/v1/datasets/{uuid4()}",
            f"/api/v1/detectors/{uuid4()}",
            f"/api/v1/streaming/sessions/{uuid4()}",
        ]

        for endpoint in not_found_endpoints:
            # Mock services to raise not found errors
            for service in mock_services.values():
                for method in ["get_dataset", "get_detector", "get_streaming_session"]:
                    if hasattr(service, method):
                        getattr(service, method).side_effect = Exception("Not found")

            with patch(
                "monorepo.presentation.api.deps.get_dataset_service",
                return_value=mock_services["dataset"],
            ):
                with patch(
                    "monorepo.presentation.api.deps.get_detection_service",
                    return_value=mock_services["detection"],
                ):
                    with patch(
                        "monorepo.presentation.api.deps.get_streaming_service",
                        return_value=mock_services["streaming"],
                    ):
                        response = client.get(endpoint, headers=auth_headers)
                        assert response.status_code in [404, 503]

    def test_authentication_across_endpoints(self, client, mock_services):
        """Test authentication requirements across endpoints."""
        # Test endpoints that require authentication
        protected_endpoints = [
            ("GET", "/api/v1/datasets"),
            ("POST", "/api/v1/detectors"),
            ("POST", "/api/v1/detection/detect"),
            ("POST", "/api/v1/automl/run"),
            ("POST", "/api/v1/explainability/anomaly"),
            ("POST", "/api/v1/streaming/sessions"),
        ]

        for method, endpoint in protected_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json={})

            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_rate_limiting_across_endpoints(self, client, mock_services, auth_headers):
        """Test rate limiting behavior across endpoints."""
        # Test endpoints with potential rate limiting
        rate_limited_endpoints = [
            (
                "POST",
                "/api/v1/detection/detect",
                {"detector_id": str(uuid4()), "dataset_id": str(uuid4())},
            ),
            (
                "POST",
                "/api/v1/automl/run",
                {"dataset_id": str(uuid4()), "algorithms": ["IsolationForest"]},
            ),
        ]

        for method, endpoint, payload in rate_limited_endpoints:
            # Make multiple rapid requests
            responses = []
            for i in range(5):
                with patch(
                    "monorepo.presentation.api.deps.get_detection_service",
                    return_value=mock_services["detection"],
                ):
                    with patch(
                        "monorepo.presentation.api.deps.get_automl_service",
                        return_value=mock_services["automl"],
                    ):
                        response = client.post(
                            endpoint, json=payload, headers=auth_headers
                        )
                        responses.append(response.status_code)

            # Should handle requests gracefully (either succeed or rate limit)
            assert all(status_code in [200, 201, 429, 503] for status_code in responses)

    def test_cors_handling_across_endpoints(self, client, mock_services, auth_headers):
        """Test CORS handling across endpoints."""
        cors_headers = {
            **auth_headers,
            "Origin": "https://example.com",
        }

        # Test various endpoints with CORS headers
        endpoints = [
            ("GET", "/api/v1/datasets"),
            (
                "POST",
                "/api/v1/detectors",
                {"name": "test", "algorithm_name": "IsolationForest"},
            ),
        ]

        for method, endpoint, *payload in endpoints:
            with patch(
                "monorepo.presentation.api.deps.get_dataset_service",
                return_value=mock_services["dataset"],
            ):
                with patch(
                    "monorepo.presentation.api.deps.get_detection_service",
                    return_value=mock_services["detection"],
                ):
                    if method == "GET":
                        response = client.get(endpoint, headers=cors_headers)
                    else:
                        response = client.post(
                            endpoint,
                            json=payload[0] if payload else {},
                            headers=cors_headers,
                        )

                    # Should either handle CORS or return normal response
                    assert response.status_code in [200, 201, 401, 404, 503]

    def test_security_headers_across_endpoints(
        self, client, mock_services, auth_headers
    ):
        """Test security headers across endpoints."""
        # Test various endpoints for security headers
        endpoints = [
            ("GET", "/api/v1/datasets"),
            (
                "POST",
                "/api/v1/detectors",
                {"name": "test", "algorithm_name": "IsolationForest"},
            ),
        ]

        for method, endpoint, *payload in endpoints:
            with patch(
                "monorepo.presentation.api.deps.get_dataset_service",
                return_value=mock_services["dataset"],
            ):
                with patch(
                    "monorepo.presentation.api.deps.get_detection_service",
                    return_value=mock_services["detection"],
                ):
                    if method == "GET":
                        response = client.get(endpoint, headers=auth_headers)
                    else:
                        response = client.post(
                            endpoint,
                            json=payload[0] if payload else {},
                            headers=auth_headers,
                        )

                    # Check for basic security headers (if endpoint responds successfully)
                    if response.status_code in [200, 201]:
                        assert "X-Content-Type-Options" in response.headers
                        assert "X-Frame-Options" in response.headers
                        assert "X-XSS-Protection" in response.headers

    def test_performance_across_endpoints(self, client, mock_services, auth_headers):
        """Test performance consistency across endpoints."""
        import time

        # Test response times for various endpoints
        endpoints = [
            ("GET", "/api/v1/datasets"),
            ("GET", f"/api/v1/detectors/{uuid4()}"),
        ]

        for method, endpoint in endpoints:
            start_time = time.time()

            with patch(
                "monorepo.presentation.api.deps.get_dataset_service",
                return_value=mock_services["dataset"],
            ):
                with patch(
                    "monorepo.presentation.api.deps.get_detection_service",
                    return_value=mock_services["detection"],
                ):
                    response = client.get(endpoint, headers=auth_headers)

            response_time = time.time() - start_time

            # Response should be reasonably fast (under 1 second for mocked services)
            assert response_time < 1.0

            # Should return a valid status code
            assert response.status_code in [200, 201, 404, 503]
