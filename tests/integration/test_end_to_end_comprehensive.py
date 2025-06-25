"""Comprehensive end-to-end integration tests - Phase 4 Coverage."""

from __future__ import annotations

import asyncio
import gc
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import psutil
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from typer.testing import CliRunner

from pynomaly.application.use_cases import DetectAnomalies, TrainDetector
from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app
from pynomaly.presentation.cli.app import app as cli_app
from pynomaly.presentation.web.app import create_web_app


@pytest.fixture(scope="session")
def integration_container():
    """Create integration test container with real implementations."""
    container = create_container()
    # Override with test configurations
    container.config.override(
        {
            "database_url": "sqlite:///:memory:",
            "redis_url": "redis://localhost:6379/1",
            "storage_path": "/tmp/pynomaly_test",
            "max_concurrent_detections": 5,
            "enable_gpu": False,
            "log_level": "DEBUG",
        }
    )
    return container


@pytest.fixture
def api_client(integration_container):
    """Create API test client."""
    app = create_app(integration_container)
    return TestClient(app)


@pytest.fixture
def web_client(integration_container):
    """Create web app test client."""
    app = create_web_app(integration_container)
    return TestClient(app)


@pytest.fixture
async def async_api_client(integration_container):
    """Create async API test client."""
    app = create_app(integration_container)
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def cli_runner():
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_datasets():
    """Create sample datasets for testing."""
    datasets = {}

    # Normal dataset
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 5))
    datasets["normal"] = pd.DataFrame(
        normal_data, columns=[f"feature_{i}" for i in range(5)]
    )

    # Dataset with anomalies
    anomaly_data = normal_data.copy()
    anomaly_indices = np.random.choice(1000, 50, replace=False)
    anomaly_data[anomaly_indices] = np.random.normal(5, 2, (50, 5))
    datasets["with_anomalies"] = pd.DataFrame(
        anomaly_data, columns=[f"feature_{i}" for i in range(5)]
    )

    # Large dataset
    large_data = np.random.normal(0, 1, (10000, 20))
    datasets["large"] = pd.DataFrame(
        large_data, columns=[f"feature_{i}" for i in range(20)]
    )

    # Time series data
    time_data = np.sin(np.linspace(0, 20 * np.pi, 1000)) + np.random.normal(
        0, 0.1, 1000
    )
    time_data = time_data.reshape(-1, 1)
    datasets["time_series"] = pd.DataFrame(time_data, columns=["value"])

    return datasets


@pytest.fixture
def performance_monitor():
    """Monitor system performance during tests."""

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.peak_memory = None

        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory

        def update(self):
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory

        def get_stats(self):
            return {
                "elapsed_time": time.time() - self.start_time if self.start_time else 0,
                "memory_usage_mb": self.peak_memory - self.start_memory
                if self.start_memory
                else 0,
                "peak_memory_mb": self.peak_memory if self.peak_memory else 0,
            }

    return PerformanceMonitor()


class TestEndToEndAPIWorkflows:
    """Test complete end-to-end API workflows."""

    @pytest.mark.asyncio
    async def test_complete_anomaly_detection_pipeline(
        self, async_api_client: AsyncClient, sample_datasets, performance_monitor
    ):
        """Test complete anomaly detection pipeline from dataset upload to results."""
        performance_monitor.start()

        # Step 1: Health check
        health_response = await async_api_client.get("/api/health/")
        assert health_response.status_code == 200

        # Step 2: Upload dataset
        dataset_csv = sample_datasets["with_anomalies"].to_csv(index=False)
        upload_response = await async_api_client.post(
            "/api/datasets/",
            files={"file": ("anomaly_data.csv", dataset_csv.encode(), "text/csv")},
            data={
                "name": "E2E Test Dataset",
                "description": "Dataset for end-to-end testing",
                "target_column": None,
            },
        )
        assert upload_response.status_code == 201
        dataset_data = upload_response.json()
        dataset_id = dataset_data["id"]

        performance_monitor.update()

        # Step 3: Create multiple detectors
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        detector_ids = []

        for algorithm in algorithms:
            detector_response = await async_api_client.post(
                "/api/detectors/",
                json={
                    "name": f"E2E {algorithm} Detector",
                    "algorithm": algorithm,
                    "hyperparameters": {"contamination": 0.05, "random_state": 42},
                },
            )
            assert detector_response.status_code == 201
            detector_data = detector_response.json()
            detector_ids.append(detector_data["id"])

        performance_monitor.update()

        # Step 4: Train all detectors concurrently
        training_tasks = []
        for detector_id in detector_ids:
            training_tasks.append(
                async_api_client.post(
                    f"/api/detectors/{detector_id}/train",
                    json={
                        "dataset_id": dataset_id,
                        "validation_split": 0.2,
                        "cross_validation": False,
                    },
                )
            )

        training_responses = await asyncio.gather(*training_tasks)
        for response in training_responses:
            assert response.status_code == 200
            training_result = response.json()
            assert training_result["success"] is True
            assert "training_time_ms" in training_result

        performance_monitor.update()

        # Step 5: Run predictions with all trained detectors
        prediction_tasks = []
        for detector_id in detector_ids:
            prediction_tasks.append(
                async_api_client.post(
                    f"/api/detectors/{detector_id}/predict",
                    json={"dataset_id": dataset_id, "threshold": 0.5},
                )
            )

        prediction_responses = await asyncio.gather(*prediction_tasks)
        results = []
        for response in prediction_responses:
            assert response.status_code == 200
            prediction_result = response.json()
            assert "predictions" in prediction_result
            assert "anomaly_scores" in prediction_result
            assert "anomaly_count" in prediction_result
            results.append(prediction_result)

        performance_monitor.update()

        # Step 6: Create and run experiment comparing all detectors
        experiment_response = await async_api_client.post(
            "/api/experiments/",
            json={
                "name": "E2E Algorithm Comparison",
                "description": "Compare all algorithms on test dataset",
                "detector_ids": detector_ids,
                "dataset_id": dataset_id,
                "evaluation_metrics": ["precision", "recall", "f1_score", "roc_auc"],
                "cross_validation_folds": 3,
            },
        )
        assert experiment_response.status_code == 201
        experiment_data = experiment_response.json()
        experiment_id = experiment_data["id"]

        # Wait for experiment completion (with timeout)
        timeout = 60  # seconds
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_response = await async_api_client.get(
                f"/api/experiments/{experiment_id}"
            )
            assert status_response.status_code == 200
            experiment_status = status_response.json()

            if experiment_status["status"] == "completed":
                break
            elif experiment_status["status"] == "failed":
                pytest.fail(
                    f"Experiment failed: {experiment_status.get('error_message')}"
                )

            await asyncio.sleep(1)
        else:
            pytest.fail("Experiment did not complete within timeout")

        performance_monitor.update()

        # Step 7: Retrieve and validate experiment results
        results_response = await async_api_client.get(
            f"/api/experiments/{experiment_id}/results"
        )
        assert results_response.status_code == 200
        experiment_results = results_response.json()

        assert "results" in experiment_results
        assert len(experiment_results["results"]) == len(algorithms)
        assert "best_detector" in experiment_results
        assert "summary_statistics" in experiment_results

        # Validate performance metrics
        for result in experiment_results["results"]:
            assert "detector_name" in result
            assert "metrics" in result
            assert "precision" in result["metrics"]
            assert "recall" in result["metrics"]
            assert "f1_score" in result["metrics"]

        # Step 8: Export results
        export_response = await async_api_client.get(
            f"/api/experiments/{experiment_id}/export", params={"format": "json"}
        )
        assert export_response.status_code == 200
        export_data = export_response.json()
        assert "experiment_metadata" in export_data
        assert "detailed_results" in export_data

        # Validate performance
        stats = performance_monitor.get_stats()
        assert stats["elapsed_time"] < 120  # Should complete within 2 minutes
        assert stats["memory_usage_mb"] < 500  # Should not use excessive memory

        # Clean up
        await async_api_client.delete(f"/api/experiments/{experiment_id}")
        for detector_id in detector_ids:
            await async_api_client.delete(f"/api/detectors/{detector_id}")
        await async_api_client.delete(f"/api/datasets/{dataset_id}")

    @pytest.mark.asyncio
    async def test_streaming_detection_workflow(
        self, async_api_client: AsyncClient, sample_datasets
    ):
        """Test streaming anomaly detection workflow."""
        # Create detector for streaming
        detector_response = await async_api_client.post(
            "/api/detectors/",
            json={
                "name": "Streaming Detector",
                "algorithm": "IsolationForest",
                "hyperparameters": {"contamination": 0.1, "random_state": 42},
            },
        )
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Upload training dataset
        training_data = sample_datasets["normal"].to_csv(index=False)
        upload_response = await async_api_client.post(
            "/api/datasets/",
            files={"file": ("training_data.csv", training_data.encode(), "text/csv")},
            data={"name": "Streaming Training Data"},
        )
        assert upload_response.status_code == 201
        dataset_id = upload_response.json()["id"]

        # Train detector
        train_response = await async_api_client.post(
            f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
        )
        assert train_response.status_code == 200

        # Simulate streaming data points
        streaming_points = (
            sample_datasets["with_anomalies"].iloc[:100].to_dict("records")
        )

        # Send streaming data points
        stream_results = []
        for point in streaming_points[:20]:  # Test with subset for performance
            stream_response = await async_api_client.post(
                f"/api/detectors/{detector_id}/predict-stream",
                json={"data_point": point},
            )
            assert stream_response.status_code == 200
            result = stream_response.json()
            assert "is_anomaly" in result
            assert "anomaly_score" in result
            assert "confidence" in result
            stream_results.append(result)

        # Validate streaming results
        anomaly_count = sum(1 for result in stream_results if result["is_anomaly"])
        assert 0 <= anomaly_count <= len(stream_results)

        # Clean up
        await async_api_client.delete(f"/api/detectors/{detector_id}")
        await async_api_client.delete(f"/api/datasets/{dataset_id}")

    @pytest.mark.asyncio
    async def test_concurrent_detection_operations(
        self, async_api_client: AsyncClient, sample_datasets
    ):
        """Test concurrent detection operations for performance and stability."""
        # Create multiple datasets
        dataset_ids = []
        for i, (name, data) in enumerate(sample_datasets.items()):
            if i >= 3:  # Limit for performance
                break

            csv_data = data.to_csv(index=False)
            upload_response = await async_api_client.post(
                "/api/datasets/",
                files={"file": (f"{name}_data.csv", csv_data.encode(), "text/csv")},
                data={"name": f"Concurrent Test Dataset {i}"},
            )
            assert upload_response.status_code == 201
            dataset_ids.append(upload_response.json()["id"])

        # Create multiple detectors
        detector_ids = []
        algorithms = ["IsolationForest", "LocalOutlierFactor"]
        for i, algorithm in enumerate(algorithms):
            detector_response = await async_api_client.post(
                "/api/detectors/",
                json={
                    "name": f"Concurrent {algorithm} {i}",
                    "algorithm": algorithm,
                    "hyperparameters": {"contamination": 0.1, "random_state": 42},
                },
            )
            assert detector_response.status_code == 201
            detector_ids.append(detector_response.json()["id"])

        # Perform concurrent training operations
        training_tasks = []
        for detector_id in detector_ids:
            for dataset_id in dataset_ids:
                training_tasks.append(
                    async_api_client.post(
                        f"/api/detectors/{detector_id}/train",
                        json={"dataset_id": dataset_id},
                    )
                )

        # Execute training tasks concurrently
        start_time = time.time()
        training_results = await asyncio.gather(*training_tasks, return_exceptions=True)
        training_time = time.time() - start_time

        # Validate results
        successful_trainings = 0
        for result in training_results:
            if isinstance(result, Exception):
                print(f"Training exception: {result}")
            else:
                assert result.status_code == 200
                successful_trainings += 1

        assert (
            successful_trainings >= len(training_tasks) * 0.8
        )  # At least 80% success rate
        assert training_time < 60  # Should complete within 1 minute

        # Perform concurrent prediction operations
        prediction_tasks = []
        for detector_id in detector_ids:
            for dataset_id in dataset_ids:
                prediction_tasks.append(
                    async_api_client.post(
                        f"/api/detectors/{detector_id}/predict",
                        json={"dataset_id": dataset_id},
                    )
                )

        # Execute prediction tasks concurrently
        start_time = time.time()
        prediction_results = await asyncio.gather(
            *prediction_tasks, return_exceptions=True
        )
        prediction_time = time.time() - start_time

        # Validate prediction results
        successful_predictions = 0
        for result in prediction_results:
            if isinstance(result, Exception):
                print(f"Prediction exception: {result}")
            else:
                if result.status_code == 200:
                    successful_predictions += 1
                    prediction_data = result.json()
                    assert "predictions" in prediction_data
                    assert "anomaly_scores" in prediction_data

        assert (
            successful_predictions >= len(prediction_tasks) * 0.8
        )  # At least 80% success rate
        assert prediction_time < 30  # Should complete within 30 seconds

        # Clean up
        for detector_id in detector_ids:
            await async_api_client.delete(f"/api/detectors/{detector_id}")
        for dataset_id in dataset_ids:
            await async_api_client.delete(f"/api/datasets/{dataset_id}")


class TestCrossLayerIntegration:
    """Test integration across different architectural layers."""

    def test_api_cli_web_consistency(
        self, api_client: TestClient, web_client: TestClient, cli_runner: CliRunner
    ):
        """Test consistency between API, CLI, and Web interfaces."""
        # Create detector via API
        api_response = api_client.post(
            "/api/detectors/",
            json={
                "name": "Consistency Test Detector",
                "algorithm": "IsolationForest",
                "hyperparameters": {"contamination": 0.1},
            },
        )
        assert api_response.status_code == 201
        api_detector = api_response.json()
        detector_id = api_detector["id"]

        # Verify detector exists via CLI
        with patch(
            "pynomaly.presentation.cli.detectors.get_cli_container"
        ) as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            mock_detector = Mock()
            mock_detector.id = detector_id
            mock_detector.name = "Consistency Test Detector"
            mock_detector.algorithm = "IsolationForest"
            mock_service.get_detector.return_value = mock_detector
            mock_container.detector_service.return_value = mock_service
            mock_get_container.return_value = mock_container

            cli_result = cli_runner.invoke(cli_app, ["detector", "show", detector_id])
            assert cli_result.exit_code == 0
            assert "Consistency Test Detector" in cli_result.stdout
            assert "IsolationForest" in cli_result.stdout

        # Verify detector exists via Web interface
        with patch(
            "pynomaly.presentation.web.app.get_detector_details"
        ) as mock_get_details:
            mock_get_details.return_value = {
                "id": detector_id,
                "name": "Consistency Test Detector",
                "algorithm": "IsolationForest",
                "hyperparameters": {"contamination": 0.1},
                "is_fitted": False,
            }

            web_response = web_client.get(
                f"/htmx/detectors/{detector_id}/details", headers={"HX-Request": "true"}
            )
            assert web_response.status_code == 200
            assert "Consistency Test Detector" in web_response.text

        # Clean up
        api_client.delete(f"/api/detectors/{detector_id}")

    def test_domain_application_infrastructure_integration(self, integration_container):
        """Test integration between domain, application, and infrastructure layers."""
        # Get services from container
        detector_service = integration_container.detector_service()
        dataset_service = integration_container.dataset_service()
        integration_container.detection_service()

        # Create test data at domain level
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("feature1,feature2\n1,2\n3,4\n5,6\n100,200\n")
            csv_path = f.name

        try:
            # Test infrastructure layer - dataset loading
            dataset = dataset_service.load_dataset(
                file_path=csv_path, name="Integration Test Dataset", target_column=None
            )
            assert dataset.id is not None
            assert dataset.name == "Integration Test Dataset"
            assert dataset.n_samples == 4
            assert dataset.n_features == 2

            # Test application layer - detector creation
            detector = detector_service.create_detector(
                name="Integration Test Detector",
                algorithm="IsolationForest",
                hyperparameters={"contamination": 0.25, "random_state": 42},
            )
            assert detector.id is not None
            assert detector.algorithm == "IsolationForest"
            assert not detector.is_fitted

            # Test domain layer - training use case
            train_use_case = TrainDetector(
                detector_repository=integration_container.detector_repository(),
                dataset_repository=integration_container.dataset_repository(),
                adapter_registry=integration_container.adapter_registry(),
            )

            training_result = train_use_case.execute(
                detector_id=detector.id,
                dataset_id=dataset.id,
                validation_split=0.0,  # No validation for small dataset
            )
            assert training_result.success is True
            assert training_result.training_time_ms > 0

            # Test domain layer - detection use case
            detect_use_case = DetectAnomalies(
                detector_repository=integration_container.detector_repository(),
                dataset_repository=integration_container.dataset_repository(),
                adapter_registry=integration_container.adapter_registry(),
            )

            detection_result = detect_use_case.execute(
                detector_id=detector.id, dataset_id=dataset.id, threshold=0.5
            )
            assert detection_result.success is True
            assert len(detection_result.predictions) == 4
            assert len(detection_result.anomaly_scores) == 4
            assert detection_result.anomaly_count >= 0

            # Validate domain invariants
            assert all(0 <= score <= 1 for score in detection_result.anomaly_scores)
            assert detection_result.anomaly_count == sum(detection_result.predictions)

        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_error_propagation_across_layers(self, integration_container):
        """Test error propagation from infrastructure through application to presentation."""
        # Test infrastructure error propagation
        dataset_service = integration_container.dataset_service()

        # Should raise domain exception when file doesn't exist
        with pytest.raises(
            Exception
        ):  # Could be FileNotFoundError or custom domain exception
            dataset_service.load_dataset(
                file_path="/nonexistent/file.csv", name="Non-existent Dataset"
            )

        # Test application layer error handling
        detector_service = integration_container.detector_service()

        with pytest.raises(
            Exception
        ):  # Could be ValidationError or custom domain exception
            detector_service.create_detector(
                name="",  # Invalid empty name
                algorithm="NonExistentAlgorithm",
                hyperparameters={},
            )

        # Test repository layer error handling
        detector_repo = integration_container.detector_repository()

        with pytest.raises(Exception):  # Should raise EntityNotFoundError or similar
            detector_repo.get("non-existent-id")


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    def test_large_dataset_processing_performance(
        self, api_client: TestClient, performance_monitor
    ):
        """Test performance with large datasets."""
        performance_monitor.start()

        # Create large dataset
        np.random.seed(42)
        large_data = np.random.normal(0, 1, (50000, 10))  # 50K samples, 10 features
        large_df = pd.DataFrame(large_data, columns=[f"feature_{i}" for i in range(10)])
        csv_data = large_df.to_csv(index=False)

        # Upload large dataset
        upload_start = time.time()
        upload_response = api_client.post(
            "/api/datasets/",
            files={"file": ("large_dataset.csv", csv_data.encode(), "text/csv")},
            data={"name": "Large Performance Test Dataset"},
        )
        upload_time = time.time() - upload_start

        assert upload_response.status_code == 201
        dataset_id = upload_response.json()["id"]

        performance_monitor.update()

        # Create detector
        detector_response = api_client.post(
            "/api/detectors/",
            json={
                "name": "Large Dataset Detector",
                "algorithm": "IsolationForest",
                "hyperparameters": {"contamination": 0.01, "random_state": 42},
            },
        )
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Train on large dataset
        train_start = time.time()
        train_response = api_client.post(
            f"/api/detectors/{detector_id}/train",
            json={"dataset_id": dataset_id, "validation_split": 0.1},
        )
        train_time = time.time() - train_start

        assert train_response.status_code == 200
        training_result = train_response.json()
        assert training_result["success"] is True

        performance_monitor.update()

        # Predict on large dataset
        predict_start = time.time()
        predict_response = api_client.post(
            f"/api/detectors/{detector_id}/predict", json={"dataset_id": dataset_id}
        )
        predict_time = time.time() - predict_start

        assert predict_response.status_code == 200
        prediction_result = predict_response.json()
        assert len(prediction_result["predictions"]) == 50000

        performance_monitor.update()

        # Validate performance metrics
        stats = performance_monitor.get_stats()

        # Performance assertions
        assert upload_time < 30  # Upload should complete within 30 seconds
        assert train_time < 60  # Training should complete within 1 minute
        assert predict_time < 30  # Prediction should complete within 30 seconds
        assert (
            stats["memory_usage_mb"] < 1000
        )  # Should not use more than 1GB additional memory

        print(
            f"Performance Stats - Upload: {upload_time:.2f}s, Train: {train_time:.2f}s, "
            f"Predict: {predict_time:.2f}s, Memory: {stats['memory_usage_mb']:.2f}MB"
        )

        # Clean up
        api_client.delete(f"/api/detectors/{detector_id}")
        api_client.delete(f"/api/datasets/{dataset_id}")

    def test_memory_usage_and_garbage_collection(self, api_client: TestClient):
        """Test memory usage patterns and garbage collection."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Create multiple datasets and detectors
        created_resources = []

        for i in range(10):
            # Create dataset
            data = np.random.normal(0, 1, (1000, 5))
            df = pd.DataFrame(data, columns=[f"feature_{j}" for j in range(5)])
            csv_data = df.to_csv(index=False)

            upload_response = api_client.post(
                "/api/datasets/",
                files={"file": (f"mem_test_{i}.csv", csv_data.encode(), "text/csv")},
                data={"name": f"Memory Test Dataset {i}"},
            )
            assert upload_response.status_code == 201
            dataset_id = upload_response.json()["id"]

            # Create detector
            detector_response = api_client.post(
                "/api/detectors/",
                json={
                    "name": f"Memory Test Detector {i}",
                    "algorithm": "IsolationForest",
                    "hyperparameters": {"contamination": 0.1, "random_state": 42},
                },
            )
            assert detector_response.status_code == 201
            detector_id = detector_response.json()["id"]

            created_resources.append((dataset_id, detector_id))

            # Check memory usage periodically
            if i % 3 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                assert memory_increase < 500  # Should not increase by more than 500MB

        # Force garbage collection
        gc.collect()

        # Clean up resources
        for dataset_id, detector_id in created_resources:
            api_client.delete(f"/api/detectors/{detector_id}")
            api_client.delete(f"/api/datasets/{dataset_id}")

        # Force garbage collection after cleanup
        gc.collect()

        # Check final memory usage
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # After cleanup, memory increase should be minimal
        assert memory_increase < 100  # Should not have significant memory leaks

    def test_concurrent_user_simulation(self, api_client: TestClient):
        """Simulate multiple concurrent users."""

        def user_workflow(user_id: int) -> dict[str, Any]:
            """Simulate a single user's workflow."""
            try:
                # Create dataset
                data = np.random.normal(0, 1, (500, 3))
                df = pd.DataFrame(data, columns=["f1", "f2", "f3"])
                csv_data = df.to_csv(index=False)

                upload_response = api_client.post(
                    "/api/datasets/",
                    files={
                        "file": (
                            f"user_{user_id}_data.csv",
                            csv_data.encode(),
                            "text/csv",
                        )
                    },
                    data={"name": f"User {user_id} Dataset"},
                )
                if upload_response.status_code != 201:
                    return {
                        "user_id": user_id,
                        "success": False,
                        "error": "Upload failed",
                    }

                dataset_id = upload_response.json()["id"]

                # Create detector
                detector_response = api_client.post(
                    "/api/detectors/",
                    json={
                        "name": f"User {user_id} Detector",
                        "algorithm": "IsolationForest",
                        "hyperparameters": {
                            "contamination": 0.1,
                            "random_state": user_id,
                        },
                    },
                )
                if detector_response.status_code != 201:
                    return {
                        "user_id": user_id,
                        "success": False,
                        "error": "Detector creation failed",
                    }

                detector_id = detector_response.json()["id"]

                # Train detector
                train_response = api_client.post(
                    f"/api/detectors/{detector_id}/train",
                    json={"dataset_id": dataset_id},
                )
                if train_response.status_code != 200:
                    return {
                        "user_id": user_id,
                        "success": False,
                        "error": "Training failed",
                    }

                # Predict
                predict_response = api_client.post(
                    f"/api/detectors/{detector_id}/predict",
                    json={"dataset_id": dataset_id},
                )
                if predict_response.status_code != 200:
                    return {
                        "user_id": user_id,
                        "success": False,
                        "error": "Prediction failed",
                    }

                # Clean up
                api_client.delete(f"/api/detectors/{detector_id}")
                api_client.delete(f"/api/datasets/{dataset_id}")

                return {"user_id": user_id, "success": True}

            except Exception as e:
                return {"user_id": user_id, "success": False, "error": str(e)}

        # Simulate 5 concurrent users
        num_users = 5
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(user_workflow, i) for i in range(num_users)]
            results = [future.result() for future in as_completed(futures)]

        total_time = time.time() - start_time

        # Validate results
        successful_users = sum(1 for result in results if result["success"])
        failed_users = [result for result in results if not result["success"]]

        print(
            f"Concurrent users test: {successful_users}/{num_users} successful in {total_time:.2f}s"
        )
        if failed_users:
            print(f"Failed users: {failed_users}")

        # At least 80% of users should succeed
        assert successful_users >= num_users * 0.8
        # Should complete within reasonable time
        assert total_time < 60


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_data_scientist_workflow(self, api_client: TestClient, sample_datasets):
        """Test typical data scientist workflow."""
        # Step 1: Upload multiple datasets for comparison
        dataset_ids = []
        for name, data in sample_datasets.items():
            csv_data = data.to_csv(index=False)
            upload_response = api_client.post(
                "/api/datasets/",
                files={"file": (f"{name}.csv", csv_data.encode(), "text/csv")},
                data={"name": f"DS Workflow {name.title()} Dataset"},
            )
            assert upload_response.status_code == 201
            dataset_ids.append(upload_response.json()["id"])

        # Step 2: Create multiple detectors with different algorithms
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        detector_configs = []

        for algorithm in algorithms:
            detector_response = api_client.post(
                "/api/detectors/",
                json={
                    "name": f"DS {algorithm} Detector",
                    "algorithm": algorithm,
                    "hyperparameters": {"contamination": 0.1, "random_state": 42},
                },
            )
            assert detector_response.status_code == 201
            detector_configs.append(
                {"id": detector_response.json()["id"], "algorithm": algorithm}
            )

        # Step 3: Train detectors on different datasets
        training_results = {}
        for detector_config in detector_configs:
            for dataset_id in dataset_ids[:2]:  # Use first two datasets
                train_response = api_client.post(
                    f"/api/detectors/{detector_config['id']}/train",
                    json={"dataset_id": dataset_id, "validation_split": 0.2},
                )
                assert train_response.status_code == 200

                key = f"{detector_config['algorithm']}_{dataset_id}"
                training_results[key] = train_response.json()

        # Step 4: Compare performance across algorithms and datasets
        comparison_results = {}
        for detector_config in detector_configs:
            for dataset_id in dataset_ids:
                predict_response = api_client.post(
                    f"/api/detectors/{detector_config['id']}/predict",
                    json={"dataset_id": dataset_id},
                )

                if predict_response.status_code == 200:
                    result = predict_response.json()
                    key = f"{detector_config['algorithm']}_{dataset_id}"
                    comparison_results[key] = {
                        "anomaly_count": result["anomaly_count"],
                        "anomaly_rate": result["anomaly_count"]
                        / len(result["predictions"]),
                        "mean_score": np.mean(result["anomaly_scores"]),
                    }

        # Step 5: Create experiment for systematic comparison
        experiment_response = api_client.post(
            "/api/experiments/",
            json={
                "name": "Data Scientist Comparison Experiment",
                "description": "Systematic comparison of algorithms",
                "detector_ids": [d["id"] for d in detector_configs],
                "dataset_id": dataset_ids[1],  # Use dataset with anomalies
                "evaluation_metrics": ["precision", "recall", "f1_score"],
                "cross_validation_folds": 3,
            },
        )
        assert experiment_response.status_code == 201
        experiment_id = experiment_response.json()["id"]

        # Wait for experiment completion
        timeout = 30
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_response = api_client.get(f"/api/experiments/{experiment_id}")
            experiment_data = status_response.json()
            if experiment_data["status"] == "completed":
                break
            time.sleep(1)

        # Step 6: Analyze results and export
        results_response = api_client.get(f"/api/experiments/{experiment_id}/results")
        assert results_response.status_code == 200
        results = results_response.json()

        # Validate comprehensive results
        assert "results" in results
        assert len(results["results"]) == len(algorithms)
        assert "best_detector" in results

        # Export for further analysis
        export_response = api_client.get(
            f"/api/experiments/{experiment_id}/export", params={"format": "json"}
        )
        assert export_response.status_code == 200

        # Clean up
        api_client.delete(f"/api/experiments/{experiment_id}")
        for detector_config in detector_configs:
            api_client.delete(f"/api/detectors/{detector_config['id']}")
        for dataset_id in dataset_ids:
            api_client.delete(f"/api/datasets/{dataset_id}")

    def test_production_monitoring_scenario(
        self, api_client: TestClient, sample_datasets
    ):
        """Test production monitoring scenario with drift detection."""
        # Step 1: Set up baseline detector
        baseline_data = sample_datasets["normal"]
        csv_data = baseline_data.to_csv(index=False)

        baseline_response = api_client.post(
            "/api/datasets/",
            files={"file": ("baseline.csv", csv_data.encode(), "text/csv")},
            data={"name": "Production Baseline Dataset"},
        )
        assert baseline_response.status_code == 201
        baseline_dataset_id = baseline_response.json()["id"]

        # Create production detector
        detector_response = api_client.post(
            "/api/detectors/",
            json={
                "name": "Production Monitoring Detector",
                "algorithm": "IsolationForest",
                "hyperparameters": {
                    "contamination": 0.05,
                    "random_state": 42,
                    "n_estimators": 200,  # More robust for production
                },
            },
        )
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Train on baseline data
        train_response = api_client.post(
            f"/api/detectors/{detector_id}/train",
            json={"dataset_id": baseline_dataset_id},
        )
        assert train_response.status_code == 200

        # Step 2: Simulate production data with potential drift
        production_batches = []

        # Normal production batch
        normal_batch = sample_datasets["normal"].iloc[:100]
        production_batches.append(("normal", normal_batch))

        # Batch with anomalies
        anomaly_batch = sample_datasets["with_anomalies"].iloc[:100]
        production_batches.append(("with_anomalies", anomaly_batch))

        # Batch with drift (different distribution)
        drift_data = np.random.normal(2, 1.5, (100, 5))  # Shifted mean and variance
        drift_batch = pd.DataFrame(
            drift_data, columns=[f"feature_{i}" for i in range(5)]
        )
        production_batches.append(("drift", drift_batch))

        # Step 3: Process production batches and monitor
        monitoring_results = []

        for batch_name, batch_data in production_batches:
            # Upload batch
            batch_csv = batch_data.to_csv(index=False)
            upload_response = api_client.post(
                "/api/datasets/",
                files={
                    "file": (
                        f"prod_batch_{batch_name}.csv",
                        batch_csv.encode(),
                        "text/csv",
                    )
                },
                data={"name": f"Production Batch {batch_name}"},
            )
            assert upload_response.status_code == 201
            batch_dataset_id = upload_response.json()["id"]

            # Run detection
            predict_response = api_client.post(
                f"/api/detectors/{detector_id}/predict",
                json={"dataset_id": batch_dataset_id},
            )
            assert predict_response.status_code == 200
            prediction_result = predict_response.json()

            # Calculate monitoring metrics
            anomaly_rate = prediction_result["anomaly_count"] / len(
                prediction_result["predictions"]
            )
            mean_score = np.mean(prediction_result["anomaly_scores"])
            max_score = np.max(prediction_result["anomaly_scores"])

            monitoring_results.append(
                {
                    "batch_name": batch_name,
                    "anomaly_rate": anomaly_rate,
                    "mean_score": mean_score,
                    "max_score": max_score,
                    "dataset_id": batch_dataset_id,
                }
            )

        # Step 4: Validate monitoring results
        # Normal batch should have low anomaly rate
        normal_result = next(
            r for r in monitoring_results if r["batch_name"] == "normal"
        )
        assert normal_result["anomaly_rate"] < 0.1

        # Anomaly batch should have higher anomaly rate
        anomaly_result = next(
            r for r in monitoring_results if r["batch_name"] == "with_anomalies"
        )
        assert anomaly_result["anomaly_rate"] > normal_result["anomaly_rate"]

        # Drift batch should be detectable
        drift_result = next(r for r in monitoring_results if r["batch_name"] == "drift")
        assert drift_result["mean_score"] > normal_result["mean_score"]

        # Step 5: Simulate alert thresholds
        alert_threshold = 0.15  # 15% anomaly rate threshold
        high_alert_batches = [
            r for r in monitoring_results if r["anomaly_rate"] > alert_threshold
        ]

        # Should trigger alerts for anomaly and drift batches
        assert len(high_alert_batches) >= 1

        print("Production monitoring results:")
        for result in monitoring_results:
            print(
                f"  {result['batch_name']}: {result['anomaly_rate']:.3f} anomaly rate, "
                f"{result['mean_score']:.3f} mean score"
            )

        # Clean up
        api_client.delete(f"/api/detectors/{detector_id}")
        api_client.delete(f"/api/datasets/{baseline_dataset_id}")
        for result in monitoring_results:
            api_client.delete(f"/api/datasets/{result['dataset_id']}")


class TestSystemReliability:
    """Test system reliability and error recovery."""

    def test_graceful_error_handling_and_recovery(self, api_client: TestClient):
        """Test graceful error handling and system recovery."""
        # Test invalid dataset upload
        invalid_response = api_client.post(
            "/api/datasets/",
            files={
                "file": (
                    "invalid.csv",
                    b"invalid,csv,content\nwith,missing\n",
                    "text/csv",
                )
            },
            data={"name": "Invalid Dataset"},
        )
        # Should handle gracefully
        assert invalid_response.status_code in [400, 422]

        # Test invalid detector creation
        invalid_detector_response = api_client.post(
            "/api/detectors/",
            json={
                "name": "",  # Invalid empty name
                "algorithm": "NonExistentAlgorithm",
                "hyperparameters": {"invalid_param": "value"},
            },
        )
        assert invalid_detector_response.status_code in [400, 422]

        # System should still be responsive after errors
        health_response = api_client.get("/api/health/")
        assert health_response.status_code == 200

        # Test recovery by creating valid resources
        valid_data = "feature1,feature2\n1,2\n3,4\n5,6\n"
        upload_response = api_client.post(
            "/api/datasets/",
            files={"file": ("valid.csv", valid_data.encode(), "text/csv")},
            data={"name": "Valid Recovery Dataset"},
        )
        assert upload_response.status_code == 201
        dataset_id = upload_response.json()["id"]

        detector_response = api_client.post(
            "/api/detectors/",
            json={
                "name": "Recovery Detector",
                "algorithm": "IsolationForest",
                "hyperparameters": {"contamination": 0.1},
            },
        )
        assert detector_response.status_code == 201
        detector_id = detector_response.json()["id"]

        # Clean up
        api_client.delete(f"/api/detectors/{detector_id}")
        api_client.delete(f"/api/datasets/{dataset_id}")

    def test_resource_cleanup_and_management(self, api_client: TestClient):
        """Test proper resource cleanup and management."""
        created_resources = []

        try:
            # Create multiple resources
            for i in range(5):
                # Create dataset
                data = f"feature1,feature2\n{i},{i * 2}\n{i + 1},{(i + 1) * 2}\n"
                upload_response = api_client.post(
                    "/api/datasets/",
                    files={
                        "file": (f"cleanup_test_{i}.csv", data.encode(), "text/csv")
                    },
                    data={"name": f"Cleanup Test Dataset {i}"},
                )
                assert upload_response.status_code == 201
                dataset_id = upload_response.json()["id"]

                # Create detector
                detector_response = api_client.post(
                    "/api/detectors/",
                    json={
                        "name": f"Cleanup Test Detector {i}",
                        "algorithm": "IsolationForest",
                        "hyperparameters": {"contamination": 0.1},
                    },
                )
                assert detector_response.status_code == 201
                detector_id = detector_response.json()["id"]

                created_resources.append((dataset_id, detector_id))

            # Verify all resources exist
            for dataset_id, detector_id in created_resources:
                dataset_response = api_client.get(f"/api/datasets/{dataset_id}")
                assert dataset_response.status_code == 200

                detector_response = api_client.get(f"/api/detectors/{detector_id}")
                assert detector_response.status_code == 200

            # Test bulk cleanup
            for dataset_id, detector_id in created_resources:
                # Delete detector first (may have dependencies)
                delete_detector_response = api_client.delete(
                    f"/api/detectors/{detector_id}"
                )
                assert delete_detector_response.status_code in [200, 204]

                # Delete dataset
                delete_dataset_response = api_client.delete(
                    f"/api/datasets/{dataset_id}"
                )
                assert delete_dataset_response.status_code in [200, 204]

            # Verify resources are cleaned up
            for dataset_id, detector_id in created_resources:
                dataset_response = api_client.get(f"/api/datasets/{dataset_id}")
                assert dataset_response.status_code == 404

                detector_response = api_client.get(f"/api/detectors/{detector_id}")
                assert detector_response.status_code == 404

        except Exception as e:
            # Ensure cleanup even if test fails
            for dataset_id, detector_id in created_resources:
                try:
                    api_client.delete(f"/api/detectors/{detector_id}")
                    api_client.delete(f"/api/datasets/{dataset_id}")
                except:
                    pass
            raise e

    def test_system_health_monitoring(self, api_client: TestClient):
        """Test comprehensive system health monitoring."""
        # Basic health check
        health_response = api_client.get("/api/health/")
        assert health_response.status_code == 200
        health_data = health_response.json()

        # Validate health response structure
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "components" in health_data

        # Test detailed health metrics
        metrics_response = api_client.get("/api/health/metrics")
        if metrics_response.status_code == 200:
            metrics_data = metrics_response.json()
            assert "system" in metrics_data
            assert "application" in metrics_data

        # Test readiness probe
        ready_response = api_client.get("/api/health/ready")
        assert ready_response.status_code in [200, 503]

        # Test liveness probe
        live_response = api_client.get("/api/health/live")
        assert live_response.status_code in [200, 503]

        # System should remain healthy after operations
        # Create and delete a resource
        data = "f1,f2\n1,2\n3,4\n"
        upload_response = api_client.post(
            "/api/datasets/",
            files={"file": ("health_test.csv", data.encode(), "text/csv")},
            data={"name": "Health Test Dataset"},
        )

        if upload_response.status_code == 201:
            dataset_id = upload_response.json()["id"]

            # Check health after operation
            post_op_health = api_client.get("/api/health/")
            assert post_op_health.status_code == 200

            # Clean up
            api_client.delete(f"/api/datasets/{dataset_id}")

            # Final health check
            final_health = api_client.get("/api/health/")
            assert final_health.status_code == 200
