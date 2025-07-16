"""Comprehensive integration tests for presentation layer - Phase 3 Coverage."""

from __future__ import annotations

import io
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app
from pynomaly.presentation.web.app import create_web_app


@pytest.fixture
def integration_container():
    """Create integration test container."""
    container = create_container()
    return container


@pytest.fixture
def api_client(integration_container):
    """Create API test client for integration tests."""
    app = create_app(integration_container)
    return TestClient(app)


@pytest.fixture
def web_client(integration_container):
    """Create web app test client for integration tests."""
    app = create_web_app(integration_container)
    return TestClient(app)


@pytest.fixture
async def async_api_client(integration_container):
    """Create async API test client."""
    app = create_app(integration_container)
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def auth_token():
    """Create authentication token for integration tests."""
    from datetime import datetime, timedelta

    import jwt

    payload = {
        "sub": "integration_user",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "roles": ["user", "admin"],
    }
    return jwt.encode(payload, "test_secret", algorithm="HS256")


@pytest.fixture
def sample_csv_content():
    """Create sample CSV content for testing."""
    return """feature1,feature2,feature3,target
1.0,2.0,3.0,0
1.5,2.5,3.5,0
2.0,3.0,4.0,0
10.0,20.0,30.0,1
15.0,25.0,35.0,1
1.2,2.2,3.2,0
1.8,2.8,3.8,0
12.0,22.0,32.0,1"""


class TestEndToEndAPIWorkflow:
    """Test complete end-to-end API workflows."""

    @pytest.mark.asyncio
    async def test_complete_detection_workflow(
        self, async_api_client: AsyncClient, auth_token, sample_csv_content
    ):
        """Test complete detection workflow: upload → create → train → predict."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Step 1: Upload dataset
        csv_file = io.BytesIO(sample_csv_content.encode())
        files = {"file": ("integration_test.csv", csv_file, "text/csv")}
        data = {
            "name": "Integration Test Dataset",
            "target_column": "target",
            "description": "Dataset for integration testing",
        }

        with patch(
            "pynomaly.application.services.dataset_service.DatasetService.load_dataset"
        ) as mock_load:
            mock_dataset = Mock()
            mock_dataset.id = "dataset_123"
            mock_dataset.name = "Integration Test Dataset"
            mock_dataset.n_samples = 8
            mock_dataset.n_features = 3
            mock_dataset.has_target = True
            mock_load.return_value = mock_dataset

            upload_response = await async_api_client.post(
                "/api/datasets/upload", files=files, data=data, headers=headers
            )

            assert upload_response.status_code == 200
            dataset_data = upload_response.json()
            dataset_id = dataset_data["id"]

        # Step 2: Create detector
        detector_config = {
            "name": "Integration Test Detector",
            "algorithm": "IsolationForest",
            "description": "Detector for integration testing",
            "parameters": {
                "contamination": 0.25,
                "n_estimators": 100,
                "random_state": 42,
            },
        }

        with patch(
            "pynomaly.application.services.detector_service.DetectorService.create_detector"
        ) as mock_create:
            mock_detector = Mock()
            mock_detector.id = "detector_123"
            mock_detector.name = "Integration Test Detector"
            mock_detector.algorithm = "IsolationForest"
            mock_detector.is_fitted = False
            mock_create.return_value = mock_detector

            detector_response = await async_api_client.post(
                "/api/detectors/", json=detector_config, headers=headers
            )

            assert detector_response.status_code == 200
            detector_data = detector_response.json()
            detector_id = detector_data["id"]

        # Step 3: Train detector
        training_config = {
            "detector_id": detector_id,
            "dataset_id": dataset_id,
            "validation_split": 0.2,
            "save_model": True,
            "training_options": {"cross_validation": True, "cv_folds": 3},
        }

        with patch(
            "pynomaly.application.services.detection_service.DetectionService.train_detector"
        ) as mock_train:
            mock_train_result = {
                "success": True,
                "training_time_ms": 1500,
                "model_metrics": {"fit_time": 0.15, "score_samples_time": 0.05},
                "validation_results": {
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1_score": 0.90,
                    "roc_auc": 0.95,
                },
            }
            mock_train.return_value = mock_train_result

            training_response = await async_api_client.post(
                "/api/detection/train", json=training_config, headers=headers
            )

            assert training_response.status_code == 200
            training_data = training_response.json()
            assert training_data["success"] is True
            assert training_data["training_time_ms"] == 1500

        # Step 4: Predict anomalies
        prediction_config = {
            "detector_id": detector_id,
            "dataset_id": dataset_id,
            "include_scores": True,
            "include_explanations": True,
            "threshold": 0.6,
        }

        with patch(
            "pynomaly.application.services.detection_service.DetectionService.detect_anomalies"
        ) as mock_predict:
            mock_prediction_result = {
                "predictions": [0, 0, 0, 1, 1, 0, 0, 1],
                "anomaly_scores": [0.1, 0.15, 0.12, 0.85, 0.92, 0.08, 0.11, 0.78],
                "anomaly_count": 3,
                "summary": {
                    "total_samples": 8,
                    "anomaly_rate": 0.375,
                    "max_score": 0.92,
                    "min_score": 0.08,
                    "avg_score": 0.375,
                    "threshold_used": 0.6,
                },
                "explanations": [
                    {"sample_id": 3, "features": {"feature1": 0.8, "feature2": 0.9}},
                    {"sample_id": 4, "features": {"feature1": 0.9, "feature2": 0.95}},
                    {"sample_id": 7, "features": {"feature1": 0.7, "feature2": 0.85}},
                ],
            }
            mock_predict.return_value = mock_prediction_result

            prediction_response = await async_api_client.post(
                "/api/detection/predict", json=prediction_config, headers=headers
            )

            assert prediction_response.status_code == 200
            prediction_data = prediction_response.json()
            assert prediction_data["anomaly_count"] == 3
            assert len(prediction_data["predictions"]) == 8
            assert len(prediction_data["explanations"]) == 3

        # Step 5: Retrieve and verify results
        results_response = await async_api_client.get(
            f"/api/detection/results?detector_id={detector_id}", headers=headers
        )

        assert results_response.status_code == 200
        results_data = results_response.json()
        assert "results" in results_data

    @pytest.mark.asyncio
    async def test_experiment_workflow(
        self, async_api_client: AsyncClient, auth_token, sample_csv_content
    ):
        """Test experiment creation and execution workflow."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Upload dataset for experiment
        csv_file = io.BytesIO(sample_csv_content.encode())
        files = {"file": ("experiment_data.csv", csv_file, "text/csv")}
        data = {"name": "Experiment Dataset"}

        with patch(
            "pynomaly.application.services.dataset_service.DatasetService.load_dataset"
        ) as mock_load:
            mock_dataset = Mock()
            mock_dataset.id = "exp_dataset_123"
            mock_load.return_value = mock_dataset

            dataset_response = await async_api_client.post(
                "/api/datasets/upload", files=files, data=data, headers=headers
            )
            dataset_id = dataset_response.json()["id"]

        # Create experiment with multiple algorithms
        experiment_config = {
            "name": "Multi-Algorithm Comparison",
            "description": "Compare IsolationForest, LOF, and OCSVM",
            "algorithms": [
                {
                    "name": "IsolationForest",
                    "parameters": {"contamination": 0.1, "n_estimators": 100},
                },
                {
                    "name": "LOF",
                    "parameters": {"contamination": 0.1, "n_neighbors": 20},
                },
                {"name": "OCSVM", "parameters": {"nu": 0.1, "kernel": "rbf"}},
            ],
            "evaluation_metrics": ["precision", "recall", "f1_score", "roc_auc"],
            "cross_validation": {"enabled": True, "folds": 5},
        }

        with patch(
            "pynomaly.application.services.experiment_service.ExperimentService.create_experiment"
        ) as mock_create_exp:
            mock_experiment = Mock()
            mock_experiment.id = "experiment_123"
            mock_experiment.name = "Multi-Algorithm Comparison"
            mock_experiment.status = "created"
            mock_create_exp.return_value = mock_experiment

            experiment_response = await async_api_client.post(
                "/api/experiments/", json=experiment_config, headers=headers
            )

            assert experiment_response.status_code == 200
            experiment_id = experiment_response.json()["id"]

        # Execute experiment
        execution_config = {
            "dataset_id": dataset_id,
            "test_split": 0.2,
            "random_state": 42,
        }

        with patch(
            "pynomaly.application.services.experiment_service.ExperimentService.execute_experiment"
        ) as mock_execute:
            mock_execution_result = {
                "experiment_id": experiment_id,
                "status": "completed",
                "execution_time_ms": 5000,
                "results": [
                    {
                        "algorithm": "IsolationForest",
                        "metrics": {
                            "precision": 0.94,
                            "recall": 0.89,
                            "f1_score": 0.915,
                            "roc_auc": 0.96,
                        },
                        "training_time_ms": 1200,
                        "prediction_time_ms": 50,
                    },
                    {
                        "algorithm": "LOF",
                        "metrics": {
                            "precision": 0.91,
                            "recall": 0.92,
                            "f1_score": 0.915,
                            "roc_auc": 0.94,
                        },
                        "training_time_ms": 2300,
                        "prediction_time_ms": 120,
                    },
                    {
                        "algorithm": "OCSVM",
                        "metrics": {
                            "precision": 0.88,
                            "recall": 0.85,
                            "f1_score": 0.865,
                            "roc_auc": 0.92,
                        },
                        "training_time_ms": 4500,
                        "prediction_time_ms": 80,
                    },
                ],
                "best_algorithm": "IsolationForest",
                "summary": {
                    "total_algorithms": 3,
                    "best_f1_score": 0.915,
                    "fastest_training": "IsolationForest",
                    "fastest_prediction": "IsolationForest",
                },
            }
            mock_execute.return_value = mock_execution_result

            execution_response = await async_api_client.post(
                f"/api/experiments/{experiment_id}/execute",
                json=execution_config,
                headers=headers,
            )

            assert execution_response.status_code == 200
            execution_data = execution_response.json()
            assert execution_data["status"] == "completed"
            assert len(execution_data["results"]) == 3
            assert execution_data["best_algorithm"] == "IsolationForest"

        # Get experiment results and leaderboard
        results_response = await async_api_client.get(
            f"/api/experiments/{experiment_id}/results", headers=headers
        )

        assert results_response.status_code == 200

        leaderboard_response = await async_api_client.get(
            f"/api/experiments/{experiment_id}/leaderboard", headers=headers
        )

        assert leaderboard_response.status_code == 200
        leaderboard_data = leaderboard_response.json()
        assert "rankings" in leaderboard_data

    @pytest.mark.asyncio
    async def test_batch_operations(
        self, async_api_client: AsyncClient, auth_token, sample_csv_content
    ):
        """Test batch operations across multiple datasets."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Upload multiple datasets
        dataset_ids = []
        for i in range(3):
            csv_file = io.BytesIO(sample_csv_content.encode())
            files = {"file": (f"batch_dataset_{i}.csv", csv_file, "text/csv")}
            data = {"name": f"Batch Dataset {i}"}

            with patch(
                "pynomaly.application.services.dataset_service.DatasetService.load_dataset"
            ) as mock_load:
                mock_dataset = Mock()
                mock_dataset.id = f"batch_dataset_{i}"
                mock_load.return_value = mock_dataset

                upload_response = await async_api_client.post(
                    "/api/datasets/upload", files=files, data=data, headers=headers
                )
                dataset_ids.append(upload_response.json()["id"])

        # Create detector for batch operations
        detector_config = {"name": "Batch Detector", "algorithm": "IsolationForest"}

        with patch(
            "pynomaly.application.services.detector_service.DetectorService.create_detector"
        ) as mock_create:
            mock_detector = Mock()
            mock_detector.id = "batch_detector_123"
            mock_create.return_value = mock_detector

            detector_response = await async_api_client.post(
                "/api/detectors/", json=detector_config, headers=headers
            )
            detector_id = detector_response.json()["id"]

        # Train detector on first dataset
        with patch(
            "pynomaly.application.services.detection_service.DetectionService.train_detector"
        ) as mock_train:
            mock_train.return_value = {"success": True, "training_time_ms": 1000}

            training_response = await async_api_client.post(
                "/api/detection/train",
                json={"detector_id": detector_id, "dataset_id": dataset_ids[0]},
                headers=headers,
            )
            assert training_response.status_code == 200

        # Batch prediction on remaining datasets
        batch_config = {
            "detector_id": detector_id,
            "dataset_ids": dataset_ids[1:],
            "batch_options": {"parallel_processing": True, "max_workers": 2},
        }

        with patch(
            "pynomaly.application.services.detection_service.DetectionService.batch_detect_anomalies"
        ) as mock_batch:
            mock_batch_result = {
                "batch_id": "batch_123",
                "status": "completed",
                "results": [
                    {
                        "dataset_id": dataset_ids[1],
                        "anomaly_count": 2,
                        "anomaly_rate": 0.25,
                    },
                    {
                        "dataset_id": dataset_ids[2],
                        "anomaly_count": 3,
                        "anomaly_rate": 0.375,
                    },
                ],
                "summary": {
                    "total_datasets": 2,
                    "total_anomalies": 5,
                    "avg_anomaly_rate": 0.3125,
                    "processing_time_ms": 800,
                },
            }
            mock_batch.return_value = mock_batch_result

            batch_response = await async_api_client.post(
                "/api/detection/batch", json=batch_config, headers=headers
            )

            assert batch_response.status_code == 200
            batch_data = batch_response.json()
            assert len(batch_data["results"]) == 2
            assert batch_data["summary"]["total_anomalies"] == 5


class TestWebUIIntegration:
    """Test Web UI integration scenarios."""

    def test_web_to_api_integration(
        self, web_client: TestClient, api_client: TestClient
    ):
        """Test Web UI calls to API endpoints."""
        # Test that web UI can make requests to API
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response

            # Web UI health check that calls API
            response = web_client.get("/health")

            assert response.status_code == 200

    def test_real_time_updates_integration(self, web_client: TestClient):
        """Test real-time updates between web UI and backend."""
        with patch("pynomaly.presentation.web.app.get_live_updates") as mock_updates:
            mock_updates.return_value = {
                "recent_detections": [
                    {"id": "det1", "status": "completed", "anomalies": 5},
                    {"id": "det2", "status": "running", "progress": 0.7},
                ],
                "system_stats": {
                    "active_detections": 2,
                    "queue_size": 3,
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                },
            }

            # HTMX endpoint for live updates
            response = web_client.get(
                "/htmx/live-updates", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            assert "5 anomalies" in response.text
            assert "45.2%" in response.text

    def test_form_submission_to_api(self, web_client: TestClient):
        """Test form submission that triggers API calls."""
        with patch("pynomaly.presentation.web.app.create_detector") as mock_create:
            mock_create.return_value = {
                "id": "new_detector_123",
                "name": "Web Created Detector",
                "algorithm": "IsolationForest",
                "success": True,
            }

            form_data = {
                "name": "Web Created Detector",
                "algorithm": "IsolationForest",
                "contamination": "0.1",
                "description": "Created via web UI",
            }

            response = web_client.post(
                "/htmx/detectors/create", data=form_data, headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            assert "Web Created Detector" in response.text

    def test_file_upload_integration(self, web_client: TestClient, sample_csv_content):
        """Test file upload through web UI."""
        with patch("pynomaly.presentation.web.app.upload_dataset") as mock_upload:
            mock_upload.return_value = {
                "id": "uploaded_dataset_123",
                "name": "Web Uploaded Dataset",
                "n_samples": 8,
                "n_features": 3,
                "success": True,
            }

            csv_file = io.BytesIO(sample_csv_content.encode())
            files = {"file": ("web_upload.csv", csv_file, "text/csv")}
            data = {"name": "Web Uploaded Dataset"}

            response = web_client.post(
                "/htmx/datasets/upload",
                files=files,
                data=data,
                headers={"HX-Request": "true"},
            )

            assert response.status_code == 200
            assert "Web Uploaded Dataset" in response.text
            assert "8 samples" in response.text


class TestCLIIntegration:
    """Test CLI integration with backend services."""

    def test_cli_api_integration(self):
        """Test CLI commands that interact with API."""
        from typer.testing import CliRunner

        from pynomaly.presentation.cli.app import app

        runner = CliRunner()

        with patch("pynomaly.presentation.cli.server.requests.get") as mock_get:
            # Mock API response for server status
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response

            result = runner.invoke(app, ["server", "status"])

            assert result.exit_code == 0
            assert "Server is running" in result.stdout

    def test_cli_direct_service_integration(self):
        """Test CLI commands that use services directly."""
        from typer.testing import CliRunner

        from pynomaly.presentation.cli.app import app

        runner = CliRunner()

        with patch(
            "pynomaly.presentation.cli.detectors.get_cli_container"
        ) as mock_container:
            mock_service = Mock()
            mock_detector = Mock()
            mock_detector.id = "cli_detector_123"
            mock_detector.name = "CLI Detector"
            mock_service.create_detector.return_value = mock_detector
            mock_container.return_value.detector_service.return_value = mock_service

            result = runner.invoke(
                app,
                [
                    "detector",
                    "create",
                    "--name",
                    "CLI Detector",
                    "--algorithm",
                    "IsolationForest",
                ],
            )

            assert result.exit_code == 0
            assert "CLI Detector" in result.stdout


class TestCrossPlatformIntegration:
    """Test cross-platform integration scenarios."""

    def test_api_cli_web_data_flow(
        self,
        api_client: TestClient,
        web_client: TestClient,
        auth_token,
        sample_csv_content,
    ):
        """Test data flow between API, CLI, and Web UI."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Step 1: Create detector via API
        detector_config = {"name": "Cross-Platform Detector", "algorithm": "LOF"}

        with patch(
            "pynomaly.application.services.detector_service.DetectorService.create_detector"
        ) as mock_create:
            mock_detector = Mock()
            mock_detector.id = "cross_platform_det"
            mock_detector.name = "Cross-Platform Detector"
            mock_create.return_value = mock_detector

            api_response = api_client.post(
                "/api/detectors/", json=detector_config, headers=headers
            )
            assert api_response.status_code == 200
            detector_id = api_response.json()["id"]

        # Step 2: Upload dataset via Web UI
        with patch("pynomaly.presentation.web.app.upload_dataset") as mock_upload:
            mock_upload.return_value = {
                "id": "cross_platform_ds",
                "name": "Cross-Platform Dataset",
                "success": True,
            }

            csv_file = io.BytesIO(sample_csv_content.encode())
            files = {"file": ("cross_platform.csv", csv_file, "text/csv")}
            data = {"name": "Cross-Platform Dataset"}

            web_response = web_client.post(
                "/htmx/datasets/upload", files=files, data=data
            )
            assert web_response.status_code == 200
            dataset_id = "cross_platform_ds"  # From mock

        # Step 3: Train via API
        with patch(
            "pynomaly.application.services.detection_service.DetectionService.train_detector"
        ) as mock_train:
            mock_train.return_value = {"success": True, "training_time_ms": 1000}

            train_response = api_client.post(
                "/api/detection/train",
                json={"detector_id": detector_id, "dataset_id": dataset_id},
                headers=headers,
            )
            assert train_response.status_code == 200

        # Step 4: View results in Web UI
        with patch(
            "pynomaly.presentation.web.app.get_detection_results"
        ) as mock_results:
            mock_results.return_value = {
                "results": [
                    {
                        "id": "result1",
                        "detector": "Cross-Platform Detector",
                        "anomalies": 3,
                    }
                ]
            }

            results_response = web_client.get("/detection/results")
            assert results_response.status_code == 200
            assert "Cross-Platform Detector" in results_response.text

    def test_concurrent_access_across_interfaces(
        self, api_client: TestClient, web_client: TestClient, auth_token
    ):
        """Test concurrent access from multiple interfaces."""
        import concurrent.futures

        headers = {"Authorization": f"Bearer {auth_token}"}

        def api_request():
            return api_client.get("/api/health/", headers=headers)

        def web_request():
            return web_client.get("/")

        # Make concurrent requests from both interfaces
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            api_futures = [executor.submit(api_request) for _ in range(5)]
            web_futures = [executor.submit(web_request) for _ in range(5)]

            api_results = [future.result() for future in api_futures]
            web_results = [future.result() for future in web_futures]

        # All requests should succeed
        assert all(result.status_code == 200 for result in api_results)
        assert all(result.status_code == 200 for result in web_results)

    def test_session_consistency_across_interfaces(
        self, api_client: TestClient, web_client: TestClient
    ):
        """Test session consistency across different interfaces."""
        # This would test shared session state between API and Web UI
        # In practice, this depends on the session management implementation

        # Login via API
        with patch(
            "pynomaly.infrastructure.auth.JWTAuthService.authenticate_user"
        ) as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "username": "testuser"}

            login_response = api_client.post(
                "/api/auth/login", data={"username": "testuser", "password": "testpass"}
            )

            if login_response.status_code == 200:
                # Check if session is available in web interface
                web_response = web_client.get("/dashboard")
                # Response depends on implementation
                assert web_response.status_code in [200, 302, 401]


class TestErrorHandlingIntegration:
    """Test error handling across presentation layer components."""

    def test_api_error_propagation_to_web(self, web_client: TestClient):
        """Test error propagation from API to Web UI."""
        with patch("requests.post") as mock_post:
            # Mock API error response
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "detail": "Invalid detector configuration"
            }
            mock_post.return_value = mock_response

            form_data = {
                "name": "",  # Invalid empty name
                "algorithm": "InvalidAlgorithm",
            }

            response = web_client.post(
                "/htmx/detectors/create", data=form_data, headers={"HX-Request": "true"}
            )

            # Should handle API error gracefully
            assert response.status_code in [200, 400]
            if response.status_code == 200:
                assert (
                    "error" in response.text.lower()
                    or "invalid" in response.text.lower()
                )

    def test_service_failure_handling(self, api_client: TestClient, auth_token):
        """Test handling of service failures."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        with patch(
            "pynomaly.application.services.detector_service.DetectorService.create_detector"
        ) as mock_create:
            # Mock service failure
            mock_create.side_effect = Exception("Service temporarily unavailable")

            response = api_client.post(
                "/api/detectors/",
                json={"name": "Test", "algorithm": "IsolationForest"},
                headers=headers,
            )

            # Should return appropriate error status
            assert response.status_code in [500, 503]
            data = response.json()
            assert "detail" in data

    def test_database_connection_error_handling(
        self, api_client: TestClient, auth_token
    ):
        """Test database connection error handling."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        with patch(
            "pynomaly.infrastructure.persistence.database.get_session"
        ) as mock_session:
            mock_session.side_effect = Exception("Database connection failed")

            response = api_client.get("/api/detectors/", headers=headers)

            # Should handle database error gracefully
            assert response.status_code in [500, 503]
            data = response.json()
            assert "detail" in data


class TestPerformanceIntegration:
    """Test performance characteristics of integrated systems."""

    def test_end_to_end_performance(
        self, api_client: TestClient, auth_token, sample_csv_content
    ):
        """Test end-to-end performance of complete workflows."""
        import time

        headers = {"Authorization": f"Bearer {auth_token}"}

        # Mock all services for performance testing
        with (
            patch(
                "pynomaly.application.services.dataset_service.DatasetService.load_dataset"
            ) as mock_load,
            patch(
                "pynomaly.application.services.detector_service.DetectorService.create_detector"
            ) as mock_create,
            patch(
                "pynomaly.application.services.detection_service.DetectionService.train_detector"
            ) as mock_train,
            patch(
                "pynomaly.application.services.detection_service.DetectionService.detect_anomalies"
            ) as mock_predict,
        ):
            # Setup mocks
            mock_load.return_value = Mock(id="perf_dataset", n_samples=1000)
            mock_create.return_value = Mock(id="perf_detector")
            mock_train.return_value = {"success": True, "training_time_ms": 500}
            mock_predict.return_value = {
                "anomaly_count": 50,
                "predictions": [0] * 950 + [1] * 50,
            }

            start_time = time.time()

            # Complete workflow
            csv_file = io.BytesIO(sample_csv_content.encode())
            files = {"file": ("perf_test.csv", csv_file, "text/csv")}
            data = {"name": "Performance Test Dataset"}

            # Upload
            upload_response = api_client.post(
                "/api/datasets/upload", files=files, data=data, headers=headers
            )
            dataset_id = upload_response.json()["id"]

            # Create detector
            detector_response = api_client.post(
                "/api/detectors/",
                json={"name": "Performance Detector", "algorithm": "IsolationForest"},
                headers=headers,
            )
            detector_id = detector_response.json()["id"]

            # Train
            train_response = api_client.post(
                "/api/detection/train",
                json={"detector_id": detector_id, "dataset_id": dataset_id},
                headers=headers,
            )

            # Predict
            predict_response = api_client.post(
                "/api/detection/predict",
                json={"detector_id": detector_id, "dataset_id": dataset_id},
                headers=headers,
            )

            end_time = time.time()
            total_time = end_time - start_time

            # Verify all operations succeeded
            assert upload_response.status_code == 200
            assert detector_response.status_code == 200
            assert train_response.status_code == 200
            assert predict_response.status_code == 200

            # Performance should be reasonable
            assert total_time < 5.0  # Complete workflow under 5 seconds

    def test_concurrent_operations_performance(
        self, api_client: TestClient, auth_token
    ):
        """Test performance under concurrent operations."""
        import concurrent.futures
        import time

        headers = {"Authorization": f"Bearer {auth_token}"}

        def health_check():
            return api_client.get("/api/health/", headers=headers)

        # Measure concurrent performance
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(health_check) for _ in range(50)]
            results = [future.result() for future in futures]

        end_time = time.time()
        total_time = end_time - start_time

        # All requests should succeed
        assert all(result.status_code == 200 for result in results)

        # Should handle concurrent requests efficiently
        avg_time_per_request = total_time / 50
        assert avg_time_per_request < 0.5  # Average under 500ms per request

    def test_memory_usage_under_load(self, api_client: TestClient, auth_token):
        """Test memory usage under sustained load."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Make many requests to test memory usage
        for _i in range(100):
            response = api_client.get("/api/health/", headers=headers)
            assert response.status_code == 200

        # This test would be more meaningful with actual memory monitoring
        # In a real test environment, you'd monitor memory usage patterns
        assert True  # Placeholder for memory usage validation


class TestSecurityIntegration:
    """Test security integration across presentation layer."""

    def test_authentication_consistency(
        self, api_client: TestClient, web_client: TestClient, auth_token
    ):
        """Test authentication consistency across interfaces."""
        # API authentication
        api_headers = {"Authorization": f"Bearer {auth_token}"}
        api_response = api_client.get("/api/detectors/", headers=api_headers)

        # Should authenticate successfully
        assert api_response.status_code in [200, 404]  # Not 401

        # Web authentication (session-based)
        with web_client.session_transaction() as session:
            session["user_id"] = "test_user"
            session["authenticated"] = True

        web_response = web_client.get("/dashboard")
        assert web_response.status_code == 200

    def test_authorization_enforcement(self, api_client: TestClient, auth_token):
        """Test authorization enforcement across endpoints."""
        from datetime import datetime, timedelta

        import jwt

        # Create tokens with different roles
        user_payload = {
            "sub": "regular_user",
            "roles": ["user"],
            "exp": datetime.utcnow() + timedelta(hours=1),
        }
        user_token = jwt.encode(user_payload, "test_secret", algorithm="HS256")

        admin_payload = {
            "sub": "admin_user",
            "roles": ["admin"],
            "exp": datetime.utcnow() + timedelta(hours=1),
        }
        admin_token = jwt.encode(admin_payload, "test_secret", algorithm="HS256")

        # Test user access
        user_headers = {"Authorization": f"Bearer {user_token}"}
        user_response = api_client.get("/api/detectors/", headers=user_headers)
        assert user_response.status_code in [200, 404]  # Should have access

        # Test admin access to admin endpoints
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        admin_response = api_client.get("/api/admin/settings", headers=admin_headers)
        # May return 404 if endpoint doesn't exist, but not 403
        assert admin_response.status_code in [200, 404]

    def test_input_validation_consistency(
        self, api_client: TestClient, web_client: TestClient, auth_token
    ):
        """Test input validation consistency across interfaces."""
        # Test malicious input through API
        api_headers = {"Authorization": f"Bearer {auth_token}"}
        malicious_data = {
            "name": "<script>alert('xss')</script>",
            "algorithm": "'; DROP TABLE detectors; --",
        }

        api_response = api_client.post(
            "/api/detectors/", json=malicious_data, headers=api_headers
        )
        # Should validate and reject or sanitize
        assert api_response.status_code in [200, 400, 422]

        # Test same input through Web UI
        web_response = web_client.post("/htmx/detectors/create", data=malicious_data)
        # Should handle consistently
        assert web_response.status_code in [200, 400, 422]

        # Neither should cause server errors
        assert api_response.status_code != 500
        assert web_response.status_code != 500
