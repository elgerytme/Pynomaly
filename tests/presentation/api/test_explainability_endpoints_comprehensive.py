"""
Comprehensive tests for explainability endpoints.
Tests explainable AI and model interpretation API endpoints.
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from pynomaly.domain.exceptions import DetectorError, ExplainabilityError
from pynomaly.presentation.web_api.app import app


class TestExplainabilityEndpointsComprehensive:
    """Comprehensive test suite for explainability API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_explainability_service(self):
        """Mock explainability service."""
        service = AsyncMock()
        service.explain_anomaly.return_value = {
            "explanation_id": str(uuid4()),
            "anomaly_id": str(uuid4()),
            "detector_id": str(uuid4()),
            "feature_importance": [
                {"feature": "feature1", "importance": 0.65, "contribution": 0.4},
                {"feature": "feature2", "importance": 0.35, "contribution": 0.3},
                {"feature": "feature3", "importance": 0.25, "contribution": 0.1},
            ],
            "local_explanation": {
                "method": "SHAP",
                "values": [0.4, 0.3, 0.1],
                "expected_value": 0.2,
                "confidence": 0.87,
            },
            "global_explanation": {
                "feature_ranking": ["feature1", "feature2", "feature3"],
                "importance_scores": [0.65, 0.35, 0.25],
                "interaction_effects": [
                    {"features": ["feature1", "feature2"], "interaction": 0.15}
                ],
            },
            "counterfactual": {
                "original_values": [2.5, 1.8, 0.9],
                "counterfactual_values": [1.2, 1.5, 0.8],
                "changes_needed": [
                    {"feature": "feature1", "change": -1.3, "direction": "decrease"},
                    {"feature": "feature2", "change": -0.3, "direction": "decrease"},
                ],
            },
            "textual_explanation": "This sample is anomalous primarily due to high values in feature1 (2.5) and feature2 (1.8), which deviate significantly from the expected normal range.",
        }

        service.explain_detector.return_value = {
            "detector_id": str(uuid4()),
            "global_importance": [
                {"feature": "feature1", "importance": 0.65},
                {"feature": "feature2", "importance": 0.35},
                {"feature": "feature3", "importance": 0.25},
            ],
            "decision_boundaries": {
                "method": "isolation_paths",
                "boundary_points": [
                    {"feature1": 2.0, "feature2": 1.5, "anomaly_score": 0.7},
                    {"feature1": 1.5, "feature2": 2.0, "anomaly_score": 0.6},
                ],
            },
            "model_behavior": {
                "linearity": 0.45,
                "interaction_strength": 0.32,
                "feature_sensitivity": {
                    "feature1": {"min": 0.1, "max": 0.9, "std": 0.2},
                    "feature2": {"min": 0.05, "max": 0.7, "std": 0.15},
                },
            },
            "interpretation": "The detector primarily uses feature1 and feature2 to identify anomalies, with feature1 being the most discriminative.",
        }

        service.compare_explanations.return_value = {
            "comparison_id": str(uuid4()),
            "explanations": [
                {
                    "explanation_id": str(uuid4()),
                    "anomaly_id": str(uuid4()),
                    "feature_importance": [{"feature": "feature1", "importance": 0.6}],
                },
                {
                    "explanation_id": str(uuid4()),
                    "anomaly_id": str(uuid4()),
                    "feature_importance": [{"feature": "feature1", "importance": 0.7}],
                },
            ],
            "similarity_score": 0.85,
            "common_features": ["feature1", "feature2"],
            "differing_features": ["feature3"],
            "explanation_consistency": 0.82,
        }

        service.get_feature_importance.return_value = {
            "detector_id": str(uuid4()),
            "importance_method": "permutation",
            "global_importance": [
                {"feature": "feature1", "importance": 0.65, "std": 0.05},
                {"feature": "feature2", "importance": 0.35, "std": 0.03},
                {"feature": "feature3", "importance": 0.25, "std": 0.02},
            ],
            "importance_plot_data": {
                "features": ["feature1", "feature2", "feature3"],
                "importance_values": [0.65, 0.35, 0.25],
                "error_bars": [0.05, 0.03, 0.02],
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
    def valid_explain_anomaly_payload(self):
        """Valid explain anomaly request payload."""
        return {
            "anomaly_id": str(uuid4()),
            "detector_id": str(uuid4()),
            "explanation_methods": ["SHAP", "LIME", "counterfactual"],
            "include_global": True,
            "include_local": True,
            "include_counterfactual": True,
            "confidence_threshold": 0.8,
            "max_features": 10,
        }

    @pytest.fixture
    def valid_explain_detector_payload(self):
        """Valid explain detector request payload."""
        return {
            "detector_id": str(uuid4()),
            "explanation_methods": ["feature_importance", "decision_boundaries"],
            "include_interactions": True,
            "include_sensitivity": True,
            "sample_size": 1000,
        }

    @pytest.fixture
    def valid_compare_explanations_payload(self):
        """Valid compare explanations request payload."""
        return {
            "explanation_ids": [str(uuid4()), str(uuid4())],
            "comparison_method": "cosine_similarity",
            "include_feature_analysis": True,
            "include_consistency_metrics": True,
        }

    def test_explain_anomaly_success(
        self, client, valid_explain_anomaly_payload, mock_explainability_service, auth_headers
    ):
        """Test successful anomaly explanation."""
        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.post(
                "/api/v1/explainability/anomaly",
                json=valid_explain_anomaly_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "explanation_id" in data
        assert "anomaly_id" in data
        assert "detector_id" in data
        assert "feature_importance" in data
        assert "local_explanation" in data
        assert "global_explanation" in data
        assert "counterfactual" in data
        assert "textual_explanation" in data

        # Verify feature importance structure
        feature_importance = data["feature_importance"]
        assert len(feature_importance) > 0
        assert "feature" in feature_importance[0]
        assert "importance" in feature_importance[0]
        assert "contribution" in feature_importance[0]

    def test_explain_anomaly_invalid_detector(
        self, client, valid_explain_anomaly_payload, mock_explainability_service, auth_headers
    ):
        """Test anomaly explanation with invalid detector."""
        mock_explainability_service.explain_anomaly.side_effect = DetectorError("Detector not found")

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.post(
                "/api/v1/explainability/anomaly",
                json=valid_explain_anomaly_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "error" in data
        assert "Detector not found" in data["error"]

    def test_explain_anomaly_missing_fields(self, client, auth_headers):
        """Test anomaly explanation with missing required fields."""
        incomplete_payload = {
            "anomaly_id": str(uuid4()),
            # Missing detector_id
        }

        response = client.post(
            "/api/v1/explainability/anomaly",
            json=incomplete_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_explain_anomaly_unauthorized(self, client, valid_explain_anomaly_payload):
        """Test anomaly explanation without authentication."""
        response = client.post(
            "/api/v1/explainability/anomaly",
            json=valid_explain_anomaly_payload,
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_explain_detector_success(
        self, client, valid_explain_detector_payload, mock_explainability_service, auth_headers
    ):
        """Test successful detector explanation."""
        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.post(
                "/api/v1/explainability/detector",
                json=valid_explain_detector_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "detector_id" in data
        assert "global_importance" in data
        assert "decision_boundaries" in data
        assert "model_behavior" in data
        assert "interpretation" in data

        # Verify global importance structure
        global_importance = data["global_importance"]
        assert len(global_importance) > 0
        assert "feature" in global_importance[0]
        assert "importance" in global_importance[0]

    def test_explain_detector_invalid_detector(
        self, client, valid_explain_detector_payload, mock_explainability_service, auth_headers
    ):
        """Test detector explanation with invalid detector."""
        mock_explainability_service.explain_detector.side_effect = DetectorError("Detector not found")

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.post(
                "/api/v1/explainability/detector",
                json=valid_explain_detector_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_compare_explanations_success(
        self, client, valid_compare_explanations_payload, mock_explainability_service, auth_headers
    ):
        """Test successful explanation comparison."""
        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.post(
                "/api/v1/explainability/compare",
                json=valid_compare_explanations_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "comparison_id" in data
        assert "explanations" in data
        assert "similarity_score" in data
        assert "common_features" in data
        assert "differing_features" in data
        assert "explanation_consistency" in data

    def test_compare_explanations_insufficient_data(
        self, client, auth_headers
    ):
        """Test explanation comparison with insufficient data."""
        invalid_payload = {
            "explanation_ids": [str(uuid4())],  # Need at least 2 explanations
        }

        response = client.post(
            "/api/v1/explainability/compare",
            json=invalid_payload,
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_feature_importance_success(
        self, client, mock_explainability_service, auth_headers
    ):
        """Test successful feature importance retrieval."""
        detector_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.get(
                f"/api/v1/explainability/feature-importance/{detector_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "detector_id" in data
        assert "importance_method" in data
        assert "global_importance" in data
        assert "importance_plot_data" in data

    def test_get_feature_importance_with_method(
        self, client, mock_explainability_service, auth_headers
    ):
        """Test feature importance with specific method."""
        detector_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.get(
                f"/api/v1/explainability/feature-importance/{detector_id}?method=shap&top_k=5",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "global_importance" in data

    def test_get_feature_importance_invalid_detector(
        self, client, mock_explainability_service, auth_headers
    ):
        """Test feature importance with invalid detector."""
        mock_explainability_service.get_feature_importance.side_effect = DetectorError("Detector not found")
        detector_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.get(
                f"/api/v1/explainability/feature-importance/{detector_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_explanation_success(
        self, client, mock_explainability_service, auth_headers
    ):
        """Test successful explanation retrieval."""
        explanation_id = str(uuid4())
        mock_explanation = {
            "explanation_id": explanation_id,
            "anomaly_id": str(uuid4()),
            "detector_id": str(uuid4()),
            "feature_importance": [{"feature": "feature1", "importance": 0.6}],
            "created_at": datetime.utcnow().isoformat(),
            "method": "SHAP",
        }
        mock_explainability_service.get_explanation.return_value = mock_explanation

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.get(
                f"/api/v1/explainability/explanations/{explanation_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "explanation_id" in data
        assert "anomaly_id" in data
        assert "detector_id" in data
        assert "feature_importance" in data

    def test_get_explanation_not_found(
        self, client, mock_explainability_service, auth_headers
    ):
        """Test explanation retrieval with non-existent ID."""
        mock_explainability_service.get_explanation.side_effect = ExplainabilityError("Explanation not found")
        explanation_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.get(
                f"/api/v1/explainability/explanations/{explanation_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_explanations_success(
        self, client, mock_explainability_service, auth_headers
    ):
        """Test successful explanations listing."""
        mock_explanations = [
            {
                "explanation_id": str(uuid4()),
                "anomaly_id": str(uuid4()),
                "detector_id": str(uuid4()),
                "method": "SHAP",
                "created_at": datetime.utcnow().isoformat(),
            },
            {
                "explanation_id": str(uuid4()),
                "anomaly_id": str(uuid4()),
                "detector_id": str(uuid4()),
                "method": "LIME",
                "created_at": datetime.utcnow().isoformat(),
            },
        ]
        mock_explainability_service.list_explanations.return_value = mock_explanations

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.get("/api/v1/explainability/explanations", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "explanations" in data
        assert len(data["explanations"]) == 2

    def test_list_explanations_with_filters(
        self, client, mock_explainability_service, auth_headers
    ):
        """Test explanations listing with filters."""
        mock_explainability_service.list_explanations.return_value = []

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.get(
                "/api/v1/explainability/explanations?detector_id=test-detector&method=SHAP",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "explanations" in data

    def test_delete_explanation_success(
        self, client, mock_explainability_service, auth_headers
    ):
        """Test successful explanation deletion."""
        explanation_id = str(uuid4())
        mock_explainability_service.delete_explanation.return_value = True

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.delete(
                f"/api/v1/explainability/explanations/{explanation_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_delete_explanation_not_found(
        self, client, mock_explainability_service, auth_headers
    ):
        """Test explanation deletion with non-existent ID."""
        mock_explainability_service.delete_explanation.side_effect = ExplainabilityError("Explanation not found")
        explanation_id = str(uuid4())

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.delete(
                f"/api/v1/explainability/explanations/{explanation_id}",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_explainability_concurrent_requests(
        self, client, valid_explain_anomaly_payload, mock_explainability_service, auth_headers
    ):
        """Test handling concurrent explainability requests."""
        import threading

        results = []

        def make_explain_request():
            with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
                response = client.post(
                    "/api/v1/explainability/anomaly",
                    json=valid_explain_anomaly_payload,
                    headers=auth_headers,
                )
                results.append(response.status_code)

        # Create multiple threads for concurrent requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_explain_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should have completed successfully
        assert len(results) == 3
        assert all(status_code == 200 for status_code in results)

    def test_explainability_request_validation(self, client, auth_headers):
        """Test comprehensive request validation."""
        # Test invalid JSON
        response = client.post(
            "/api/v1/explainability/anomaly",
            data="invalid json",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test invalid UUID format
        invalid_payload = {
            "anomaly_id": "invalid-uuid",
            "detector_id": str(uuid4()),
        }
        response = client.post(
            "/api/v1/explainability/anomaly",
            json=invalid_payload,
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_explainability_error_handling(
        self, client, valid_explain_anomaly_payload, mock_explainability_service, auth_headers
    ):
        """Test error handling in explainability endpoints."""
        # Test service unavailable
        mock_explainability_service.explain_anomaly.side_effect = Exception("Service unavailable")

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.post(
                "/api/v1/explainability/anomaly",
                json=valid_explain_anomaly_payload,
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_explainability_security_headers(
        self, client, valid_explain_anomaly_payload, mock_explainability_service, auth_headers
    ):
        """Test security headers in explainability responses."""
        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.post(
                "/api/v1/explainability/anomaly",
                json=valid_explain_anomaly_payload,
                headers=auth_headers,
            )

        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers

    def test_explainability_cors_handling(
        self, client, valid_explain_anomaly_payload, mock_explainability_service, auth_headers
    ):
        """Test CORS handling in explainability endpoints."""
        cors_headers = {
            **auth_headers,
            "Origin": "https://example.com",
        }

        with patch("pynomaly.presentation.web_api.dependencies.get_explainability_service", return_value=mock_explainability_service):
            response = client.post(
                "/api/v1/explainability/anomaly",
                json=valid_explain_anomaly_payload,
                headers=cors_headers,
            )

        # Check for CORS headers
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 200
