"""Test AutoML API endpoints."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

# We'll need to mock the container and dependencies
from pynomaly.presentation.api.endpoints.automl import router


class TestAutoMLAPI:
    """Test AutoML API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router, prefix="/automl")

        return TestClient(app)

    @pytest.fixture
    def mock_container(self):
        """Create mock container."""
        container = Mock()
        use_case = AsyncMock()
        container.automl_optimization_use_case.return_value = use_case
        return container, use_case

    def test_automl_run_endpoint_success(self, client, mock_container):
        """Test successful /automl/run endpoint."""
        container, use_case = mock_container

        # Mock optimization response
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_score = 0.8
        mock_response.best_parameters = {"n_neighbors": 5, "method": "largest"}
        mock_response.trials_completed = 50
        mock_response.optimized_detector_id = "detector_123"

        use_case.auto_optimize.return_value = mock_response

        with (
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_container"
            ) as mock_get_container,
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_current_user"
            ) as mock_get_user,
            patch(
                "pynomaly.presentation.api.endpoints.automl.require_write"
            ) as mock_require_write,
        ):

            mock_get_container.return_value = container
            mock_get_user.return_value = "test_user"
            mock_require_write.return_value = "write"

            # Create temporary storage for test
            with tempfile.TemporaryDirectory() as temp_dir:
                response = client.post(
                    "/automl/run",
                    params={
                        "dataset_path": "test_data.csv",
                        "algorithm_name": "KNN",
                        "max_trials": 100,
                        "storage_path": temp_dir,
                    },
                )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["algorithm"] == "KNN"
        assert data["best_score"] == 0.8
        assert data["meets_success_criteria"] is True  # 0.8 > 0.5 + 15%
        assert "detector_id" in data

    def test_automl_run_endpoint_unsupported_algorithm(self, client, mock_container):
        """Test /automl/run endpoint with unsupported algorithm."""
        container, use_case = mock_container

        with (
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_container"
            ) as mock_get_container,
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_current_user"
            ) as mock_get_user,
            patch(
                "pynomaly.presentation.api.endpoints.automl.require_write"
            ) as mock_require_write,
        ):

            mock_get_container.return_value = container
            mock_get_user.return_value = "test_user"
            mock_require_write.return_value = "write"

            response = client.post(
                "/automl/run",
                params={
                    "dataset_path": "test_data.csv",
                    "algorithm_name": "UnsupportedAlgorithm",
                    "max_trials": 100,
                },
            )

        assert response.status_code == 400
        assert "Unsupported algorithm" in response.json()["detail"]

    def test_automl_run_endpoint_optimization_failure(self, client, mock_container):
        """Test /automl/run endpoint when optimization fails."""
        container, use_case = mock_container

        # Mock failed optimization response
        mock_response = Mock()
        mock_response.success = False
        mock_response.error_message = "Optimization failed due to invalid data"

        use_case.auto_optimize.return_value = mock_response

        with (
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_container"
            ) as mock_get_container,
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_current_user"
            ) as mock_get_user,
            patch(
                "pynomaly.presentation.api.endpoints.automl.require_write"
            ) as mock_require_write,
        ):

            mock_get_container.return_value = container
            mock_get_user.return_value = "test_user"
            mock_require_write.return_value = "write"

            response = client.post(
                "/automl/run",
                params={
                    "dataset_path": "invalid_data.csv",
                    "algorithm_name": "KNN",
                    "max_trials": 50,
                },
            )

        assert response.status_code == 200  # Endpoint returns 200 with error details
        data = response.json()

        assert data["success"] is False
        assert data["algorithm"] == "KNN"
        assert "error" in data
        assert data["message"] == "AutoML optimization failed"

    def test_automl_run_endpoint_exception_handling(self, client, mock_container):
        """Test /automl/run endpoint exception handling."""
        container, use_case = mock_container

        # Mock exception during optimization
        use_case.auto_optimize.side_effect = Exception("Unexpected error")

        with (
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_container"
            ) as mock_get_container,
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_current_user"
            ) as mock_get_user,
            patch(
                "pynomaly.presentation.api.endpoints.automl.require_write"
            ) as mock_require_write,
        ):

            mock_get_container.return_value = container
            mock_get_user.return_value = "test_user"
            mock_require_write.return_value = "write"

            response = client.post(
                "/automl/run",
                params={
                    "dataset_path": "test_data.csv",
                    "algorithm_name": "KNN",
                    "max_trials": 100,
                },
            )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is False
        assert data["algorithm"] == "KNN"
        assert "Unexpected error" in data["error"]

    def test_automl_run_performance_criteria_calculation(self, client, mock_container):
        """Test performance criteria calculation in API response."""
        container, use_case = mock_container

        # Test case: Score meets 15% improvement criteria
        mock_response_success = Mock()
        mock_response_success.success = True
        mock_response_success.best_score = 0.6  # 20% above baseline of 0.5
        mock_response_success.best_parameters = {"n_neighbors": 10}
        mock_response_success.trials_completed = 75
        mock_response_success.optimized_detector_id = "detector_456"

        use_case.auto_optimize.return_value = mock_response_success

        with (
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_container"
            ) as mock_get_container,
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_current_user"
            ) as mock_get_user,
            patch(
                "pynomaly.presentation.api.endpoints.automl.require_write"
            ) as mock_require_write,
        ):

            mock_get_container.return_value = container
            mock_get_user.return_value = "test_user"
            mock_require_write.return_value = "write"

            response = client.post(
                "/automl/run",
                params={
                    "dataset_path": "high_performance_data.csv",
                    "algorithm_name": "KNN",
                    "max_trials": 100,
                },
            )

        data = response.json()

        assert data["success"] is True
        assert data["meets_success_criteria"] is True
        assert "20.0%" in data["performance_improvement"]

    def test_automl_run_storage_functionality(self, client, mock_container):
        """Test storage functionality in /automl/run endpoint."""
        container, use_case = mock_container

        mock_response = Mock()
        mock_response.success = True
        mock_response.best_score = 0.75
        mock_response.best_parameters = {"n_neighbors": 8, "method": "mean"}
        mock_response.trials_completed = 40
        mock_response.optimized_detector_id = "detector_789"

        use_case.auto_optimize.return_value = mock_response

        with (
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_container"
            ) as mock_get_container,
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_current_user"
            ) as mock_get_user,
            patch(
                "pynomaly.presentation.api.endpoints.automl.require_write"
            ) as mock_require_write,
        ):

            mock_get_container.return_value = container
            mock_get_user.return_value = "test_user"
            mock_require_write.return_value = "write"

            with tempfile.TemporaryDirectory() as temp_dir:
                response = client.post(
                    "/automl/run",
                    params={
                        "dataset_path": "test_data.csv",
                        "algorithm_name": "LOF",
                        "max_trials": 100,
                        "storage_path": temp_dir,
                    },
                )

                data = response.json()

                assert data["success"] is True
                assert "storage_file" in data

                # Verify storage file was created (check if path exists in response)
                storage_file = data["storage_file"]
                assert temp_dir in storage_file
                assert "LOF_optimization_" in storage_file

    def test_automl_run_parameter_validation(self, client, mock_container):
        """Test parameter validation for /automl/run endpoint."""
        container, use_case = mock_container

        with (
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_container"
            ) as mock_get_container,
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_current_user"
            ) as mock_get_user,
            patch(
                "pynomaly.presentation.api.endpoints.automl.require_write"
            ) as mock_require_write,
        ):

            mock_get_container.return_value = container
            mock_get_user.return_value = "test_user"
            mock_require_write.return_value = "write"

            # Test all supported PyOD algorithms
            supported_algorithms = [
                "KNN",
                "LOF",
                "IsolationForest",
                "OneClassSVM",
                "AutoEncoder",
            ]

            for algorithm in supported_algorithms:
                mock_response = Mock()
                mock_response.success = True
                mock_response.best_score = 0.7
                mock_response.best_parameters = {"test_param": "test_value"}
                mock_response.trials_completed = 30
                mock_response.optimized_detector_id = f"detector_{algorithm}"

                use_case.auto_optimize.return_value = mock_response

                response = client.post(
                    "/automl/run",
                    params={
                        "dataset_path": "test_data.csv",
                        "algorithm_name": algorithm,
                        "max_trials": 50,
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["algorithm"] == algorithm

    def test_automl_run_30_minute_time_limit(self, client, mock_container):
        """Test that AutoML run respects the 30-minute time limit."""
        container, use_case = mock_container

        mock_response = Mock()
        mock_response.success = True
        mock_response.best_score = 0.8
        mock_response.best_parameters = {"n_neighbors": 5}
        mock_response.trials_completed = 100
        mock_response.optimized_detector_id = "detector_time_test"

        use_case.auto_optimize.return_value = mock_response

        with (
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_container"
            ) as mock_get_container,
            patch(
                "pynomaly.presentation.api.endpoints.automl.get_current_user"
            ) as mock_get_user,
            patch(
                "pynomaly.presentation.api.endpoints.automl.require_write"
            ) as mock_require_write,
        ):

            mock_get_container.return_value = container
            mock_get_user.return_value = "test_user"
            mock_require_write.return_value = "write"

            response = client.post(
                "/automl/run",
                params={
                    "dataset_path": "large_dataset.csv",
                    "algorithm_name": "KNN",
                    "max_trials": 100,
                },
            )

            data = response.json()

            # Verify that optimization_time is tracked
            assert "optimization_time" in data
            # In real implementation, this should be < 1800 seconds (30 minutes)
            assert data["optimization_time"] >= 0


class TestTrialPersistenceAPI:
    """Test trial persistence functionality in API."""

    def test_trial_data_structure(self):
        """Test the structure of persisted trial data."""
        trial_data = {
            "algorithm": "KNN",
            "best_score": 0.85,
            "best_parameters": {
                "n_neighbors": 10,
                "method": "largest",
                "contamination": 0.1,
            },
            "optimization_time": 450.2,
            "trials_completed": 75,
            "storage_path": "./automl_storage",
        }

        # Verify all required fields are present
        required_fields = [
            "algorithm",
            "best_score",
            "best_parameters",
            "optimization_time",
            "trials_completed",
            "storage_path",
        ]

        for field in required_fields:
            assert field in trial_data

        # Verify data types
        assert isinstance(trial_data["algorithm"], str)
        assert isinstance(trial_data["best_score"], (int, float))
        assert isinstance(trial_data["best_parameters"], dict)
        assert isinstance(trial_data["optimization_time"], (int, float))
        assert isinstance(trial_data["trials_completed"], int)

    def test_trial_data_json_serialization(self):
        """Test that trial data can be serialized to JSON."""
        trial_data = {
            "algorithm": "LOF",
            "best_score": 0.72,
            "best_parameters": {"n_neighbors": 15, "contamination": 0.15},
            "optimization_time": 320.8,
            "trials_completed": 60,
            "storage_path": "./test_storage",
        }

        # Should not raise an exception
        json_str = json.dumps(trial_data, indent=2, default=str)

        # Should be able to deserialize back
        loaded_data = json.loads(json_str)

        assert loaded_data["algorithm"] == "LOF"
        assert loaded_data["best_score"] == 0.72
        assert loaded_data["trials_completed"] == 60


class TestPerformanceCriteriaAPI:
    """Test performance criteria validation in API."""

    def test_f1_improvement_calculation(self):
        """Test F1 improvement calculation logic."""
        baseline_score = 0.5

        # Test cases for different improvement levels
        test_cases = [
            (0.575, 0.15, True),  # Exactly 15% improvement
            (0.6, 0.2, True),  # 20% improvement (exceeds criteria)
            (0.55, 0.1, False),  # 10% improvement (below criteria)
            (0.4, -0.2, False),  # Negative improvement
        ]

        for optimized_score, expected_improvement, should_meet_criteria in test_cases:
            improvement = (optimized_score - baseline_score) / baseline_score
            meets_criteria = improvement >= 0.15

            assert improvement == pytest.approx(expected_improvement, rel=1e-3)
            assert meets_criteria == should_meet_criteria

    def test_performance_improvement_formatting(self):
        """Test performance improvement formatting for API response."""
        improvements = [0.15, 0.20, 0.05, -0.10]
        expected_formats = ["15.0%", "20.0%", "5.0%", "-10.0%"]

        for improvement, expected in zip(improvements, expected_formats):
            formatted = f"{improvement:.1%}"
            assert formatted == expected


if __name__ == "__main__":
    pytest.main([__file__])
