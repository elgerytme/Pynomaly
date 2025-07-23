"""Integration tests for AutoML API endpoints."""

import json
import pytest
import httpx
from typing import Any, Dict
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from test_utilities.api_test_base import APITestBase
from test_utilities.fixtures.api import (
    api_client,
    authenticated_client,
    mock_container,
    mock_auth_dependencies
)
from test_utilities.fixtures.data import sample_datasets, sample_model_data


class TestAutoMLEndpointsIntegration(APITestBase):
    """Integration tests for AutoML API endpoints."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with AutoML router."""
        from machine_learning.presentation.api.endpoints.automl import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1/automl")
        return app

    @pytest.fixture
    def mock_automl_use_case(self):
        """Mock AutoML use case."""
        mock = AsyncMock()
        mock.profile_dataset.return_value = MagicMock(
            n_samples=1000,
            n_features=10,
            contamination_estimate=0.1,
            feature_types={"feature_1": "numerical", "feature_2": "categorical"},
            profile_metadata={"missing_values_ratio": 0.05, "sparsity_ratio": 0.1},
            has_temporal_structure=False,
            complexity_score=0.7
        )
        mock.get_algorithm_recommendations.return_value = MagicMock(
            recommended_algorithms=["IsolationForest", "LOF"],
            algorithm_scores={"IsolationForest": 0.85, "LOF": 0.78},
            reasoning={"IsolationForest": "Good for high-dimensional data"}
        )
        mock.auto_optimize.return_value = MagicMock(
            success=True,
            best_algorithm="IsolationForest",
            best_parameters={"contamination": 0.1, "n_estimators": 100},
            best_score=0.85,
            optimization_time_seconds=120.5,
            trials_completed=50,
            algorithm_rankings=["IsolationForest", "LOF"],
            optimized_detector_id=str(uuid4()),
            optimization_summary="Completed successfully"
        )
        return mock

    @pytest.fixture
    def mock_algorithm_registry(self):
        """Mock algorithm registry."""
        mock = MagicMock()
        mock.get_supported_algorithms.return_value = [
            "IsolationForest", "LOF", "KNN", "OneClassSVM", "AutoEncoder"
        ]
        return mock

    @pytest.mark.asyncio
    async def test_profile_dataset_success(
        self, 
        authenticated_client: TestClient,
        mock_automl_use_case: AsyncMock,
        mock_container: MagicMock
    ):
        """Test successful dataset profiling."""
        # Setup mocks
        mock_container.automl_optimization_use_case.return_value = mock_automl_use_case
        
        request_data = {
            "dataset_id": str(uuid4()),
            "include_recommendations": True,
            "max_recommendations": 5
        }
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/profile", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "dataset_profile" in data
        assert "algorithm_recommendations" in data
        assert data["dataset_profile"]["n_samples"] == 1000
        assert data["dataset_profile"]["n_features"] == 10
        assert len(data["algorithm_recommendations"]) == 2
        
        # Verify use case was called correctly
        mock_automl_use_case.profile_dataset.assert_called_once()
        mock_automl_use_case.get_algorithm_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_profile_dataset_without_recommendations(
        self, 
        authenticated_client: TestClient,
        mock_automl_use_case: AsyncMock,
        mock_container: MagicMock
    ):
        """Test dataset profiling without algorithm recommendations."""
        mock_container.automl_optimization_use_case.return_value = mock_automl_use_case
        
        request_data = {
            "dataset_id": str(uuid4()),
            "include_recommendations": False
        }
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/profile", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["algorithm_recommendations"] is None
        
        # Verify recommendations were not requested
        mock_automl_use_case.get_algorithm_recommendations.assert_not_called()

    @pytest.mark.asyncio
    async def test_profile_dataset_failure(
        self, 
        authenticated_client: TestClient,
        mock_automl_use_case: AsyncMock,
        mock_container: MagicMock
    ):
        """Test dataset profiling failure handling."""
        mock_container.automl_optimization_use_case.return_value = mock_automl_use_case
        mock_automl_use_case.profile_dataset.side_effect = Exception("Dataset not found")
        
        request_data = {
            "dataset_id": str(uuid4()),
            "include_recommendations": False
        }
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/profile", json=request_data)
        
        assert response.status_code == 200  # Endpoint handles errors gracefully
        data = response.json()
        
        assert data["success"] is False
        assert "Dataset not found" in data["error"]

    @pytest.mark.asyncio
    async def test_optimize_automl_success(
        self, 
        authenticated_client: TestClient,
        mock_automl_use_case: AsyncMock,
        mock_container: MagicMock
    ):
        """Test successful AutoML optimization."""
        mock_container.automl_optimization_use_case.return_value = mock_automl_use_case
        
        request_data = {
            "dataset_id": str(uuid4()),
            "objective": "f1_score",
            "max_algorithms": 3,
            "max_optimization_time": 1800,
            "enable_ensemble": True,
            "detector_name": "test_detector",
            "cross_validation_folds": 5,
            "random_state": 42
        }
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/optimize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "detector_id" in data
        assert "automl_result" in data
        assert data["automl_result"]["best_algorithm"] == "IsolationForest"
        assert data["automl_result"]["best_score"] == 0.85
        assert data["automl_result"]["trials_completed"] == 50

    @pytest.mark.asyncio
    async def test_optimize_single_algorithm_success(
        self, 
        authenticated_client: TestClient,
        mock_automl_use_case: AsyncMock,
        mock_container: MagicMock
    ):
        """Test successful single algorithm optimization."""
        mock_container.automl_optimization_use_case.return_value = mock_automl_use_case
        mock_automl_use_case.optimize_single_algorithm.return_value = MagicMock(
            success=True,
            best_parameters={"contamination": 0.1},
            best_score=0.82,
            trials_completed=20
        )
        
        request_data = {
            "dataset_id": str(uuid4()),
            "algorithm": "IsolationForest",
            "objective": "f1_score",
            "n_trials": 20
        }
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/optimize-algorithm", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["best_score"] == 0.82
        assert data["algorithm"] == "IsolationForest"
        assert data["trials_completed"] == 20

    @pytest.mark.asyncio
    async def test_list_supported_algorithms(
        self, 
        authenticated_client: TestClient,
        mock_algorithm_registry: MagicMock,
        mock_container: MagicMock
    ):
        """Test listing supported algorithms."""
        mock_container.algorithm_adapter_registry.return_value = mock_algorithm_registry
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            with patch("machine_learning.presentation.api.endpoints.automl.feature_flags") as mock_flags:
                mock_flags.is_enabled.return_value = True
                response = authenticated_client.get("/api/v1/automl/algorithms")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_algorithms"] == 5
        assert "IsolationForest" in data["all_algorithms"]
        assert "by_family" in data
        assert data["automl_enabled"] is True

    @pytest.mark.asyncio
    async def test_get_algorithm_recommendations(
        self, 
        authenticated_client: TestClient,
        mock_automl_use_case: AsyncMock,
        mock_container: MagicMock
    ):
        """Test getting algorithm recommendations."""
        mock_container.automl_optimization_use_case.return_value = mock_automl_use_case
        
        dataset_id = str(uuid4())
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.get(f"/api/v1/automl/recommendations/{dataset_id}?max_recommendations=3")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["dataset_id"] == dataset_id
        assert len(data["recommendations"]) == 2
        assert data["recommendations"][0]["algorithm"] == "IsolationForest"
        assert data["recommendations"][0]["score"] == 0.85

    @pytest.mark.asyncio
    async def test_run_automl_with_valid_algorithm(
        self, 
        authenticated_client: TestClient,
        mock_automl_use_case: AsyncMock,
        mock_container: MagicMock,
        tmp_path
    ):
        """Test running AutoML with a valid algorithm."""
        mock_container.automl_optimization_use_case.return_value = mock_automl_use_case
        
        # Create temporary dataset file
        dataset_path = str(tmp_path / "test_dataset.csv")
        with open(dataset_path, "w") as f:
            f.write("feature1,feature2,target\n1,2,0\n3,4,1\n")
        
        params = {
            "dataset_path": dataset_path,
            "algorithm_name": "IsolationForest",
            "max_trials": 50,
            "storage_path": str(tmp_path / "storage")
        }
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/run", params=params)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["algorithm"] == "IsolationForest"
        assert "detector_id" in data
        assert "storage_file" in data

    @pytest.mark.asyncio
    async def test_run_automl_with_invalid_algorithm(
        self, 
        authenticated_client: TestClient,
        mock_container: MagicMock
    ):
        """Test running AutoML with an invalid algorithm."""
        params = {
            "dataset_path": "/path/to/dataset.csv",
            "algorithm_name": "InvalidAlgorithm",
            "max_trials": 50
        }
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/run", params=params)
        
        assert response.status_code == 400
        assert "Unsupported algorithm" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_batch_optimize_success(
        self, 
        authenticated_client: TestClient,
        mock_container: MagicMock
    ):
        """Test successful batch optimization."""
        dataset_ids = [str(uuid4()) for _ in range(3)]
        
        request_data = {
            "dataset_ids": dataset_ids,
            "optimization_objective": "auc",
            "max_algorithms_per_dataset": 2,
            "enable_ensemble": True
        }
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/batch-optimize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "queued"
        assert data["total_datasets"] == 3
        assert data["dataset_ids"] == dataset_ids
        assert "batch_id" in data

    @pytest.mark.asyncio
    async def test_batch_optimize_too_many_datasets(
        self, 
        authenticated_client: TestClient,
        mock_container: MagicMock
    ):
        """Test batch optimization with too many datasets."""
        dataset_ids = [str(uuid4()) for _ in range(15)]  # Exceeds limit of 10
        
        request_data = {
            "dataset_ids": dataset_ids,
            "optimization_objective": "auc"
        }
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/batch-optimize", json=request_data)
        
        assert response.status_code == 400
        assert "Maximum 10 datasets" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_optimization_status(
        self, 
        authenticated_client: TestClient,
        mock_container: MagicMock
    ):
        """Test getting optimization status."""
        optimization_id = uuid4()
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.get(f"/api/v1/automl/status/{optimization_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["optimization_id"] == str(optimization_id)
        assert data["status"] == "completed"
        assert "progress_percentage" in data

    @pytest.mark.asyncio
    async def test_cancel_optimization(
        self, 
        authenticated_client: TestClient,
        mock_container: MagicMock
    ):
        """Test canceling optimization."""
        optimization_id = uuid4()
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.delete(f"/api/v1/automl/optimization/{optimization_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["optimization_id"] == str(optimization_id)
        assert data["cancelled"] is True

    @pytest.mark.asyncio
    async def test_authentication_required(self, api_client: TestClient):
        """Test that authentication is required for all endpoints."""
        endpoints = [
            ("POST", "/api/v1/automl/profile", {"dataset_id": str(uuid4())}),
            ("POST", "/api/v1/automl/optimize", {"dataset_id": str(uuid4()), "objective": "f1_score"}),
            ("GET", "/api/v1/automl/algorithms", None),
            ("POST", "/api/v1/automl/run", None)
        ]
        
        for method, endpoint, data in endpoints:
            if method == "POST":
                response = api_client.post(endpoint, json=data or {})
            else:
                response = api_client.get(endpoint)
            
            assert response.status_code == 401  # Unauthorized

    @pytest.mark.asyncio
    async def test_feature_flag_enforcement(
        self, 
        authenticated_client: TestClient,
        mock_container: MagicMock
    ):
        """Test that feature flags are properly enforced."""
        request_data = {"dataset_id": str(uuid4())}
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            with patch("machine_learning.presentation.api.endpoints.automl.require_feature") as mock_require:
                mock_require.side_effect = Exception("Feature not enabled")
                
                response = authenticated_client.post("/api/v1/automl/profile", json=request_data)
                
                # Feature flag decorator should prevent access
                assert response.status_code != 200

    @pytest.mark.asyncio
    async def test_concurrent_requests(
        self, 
        authenticated_client: TestClient,
        mock_automl_use_case: AsyncMock,
        mock_container: MagicMock
    ):
        """Test handling concurrent requests."""
        import asyncio
        
        mock_container.automl_optimization_use_case.return_value = mock_automl_use_case
        
        async def make_request():
            request_data = {"dataset_id": str(uuid4()), "include_recommendations": False}
            with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
                return authenticated_client.post("/api/v1/automl/profile", json=request_data)
        
        # Make multiple concurrent requests
        tasks = [make_request() for _ in range(5)]
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            assert response.json()["success"] is True

    @pytest.mark.asyncio
    async def test_request_validation(
        self, 
        authenticated_client: TestClient,
        mock_container: MagicMock
    ):
        """Test request validation."""
        # Test missing required fields
        response = authenticated_client.post("/api/v1/automl/profile", json={})
        assert response.status_code == 422  # Validation error
        
        # Test invalid field types
        request_data = {
            "dataset_id": 123,  # Should be string
            "include_recommendations": "yes"  # Should be boolean
        }
        response = authenticated_client.post("/api/v1/automl/profile", json=request_data)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_error_logging(
        self, 
        authenticated_client: TestClient,
        mock_automl_use_case: AsyncMock,
        mock_container: MagicMock,
        caplog
    ):
        """Test that errors are properly logged."""
        mock_container.automl_optimization_use_case.return_value = mock_automl_use_case
        mock_automl_use_case.profile_dataset.side_effect = Exception("Test error")
        
        request_data = {"dataset_id": str(uuid4()), "include_recommendations": False}
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/profile", json=request_data)
        
        assert response.status_code == 200
        assert "Dataset profiling failed" in caplog.text
        assert "Test error" in caplog.text

    @pytest.mark.asyncio
    async def test_performance_metrics(
        self, 
        authenticated_client: TestClient,
        mock_automl_use_case: AsyncMock,
        mock_container: MagicMock
    ):
        """Test that performance metrics are included in responses."""
        mock_container.automl_optimization_use_case.return_value = mock_automl_use_case
        
        request_data = {"dataset_id": str(uuid4()), "include_recommendations": False}
        
        with patch("machine_learning.presentation.api.endpoints.automl.get_container", return_value=mock_container):
            response = authenticated_client.post("/api/v1/automl/profile", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "execution_time" in data
        assert isinstance(data["execution_time"], float)
        assert data["execution_time"] > 0

    def test_api_documentation_coverage(self):
        """Test that all endpoints have proper documentation."""
        from machine_learning.presentation.api.endpoints.automl import router
        
        for route in router.routes:
            if hasattr(route, 'endpoint'):
                assert route.endpoint.__doc__ is not None, f"Endpoint {route.path} missing documentation"
                assert len(route.endpoint.__doc__.strip()) > 10, f"Endpoint {route.path} has insufficient documentation"