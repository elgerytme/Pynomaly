"""Complete user journey end-to-end tests.

This module tests complete user workflows from data upload to anomaly detection results,
covering all major user paths through the system.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app


class TestCompleteUserJourney:
    """Test complete user workflows end-to-end."""

    @pytest.fixture
    def app_client(self):
        """Create test client for API."""
        container = create_container()
        app = create_app(container)
        return TestClient(app)

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        import numpy as np

        np.random.seed(42)

        # Create realistic anomaly detection dataset
        normal_data = np.random.normal(0, 1, (800, 5))
        anomalous_data = np.random.normal(3, 0.5, (50, 5))  # Clear outliers

        all_data = pd.DataFrame(
            np.vstack([normal_data, anomalous_data]),
            columns=[f"feature_{i}" for i in range(5)],
        )

        return all_data

    def test_complete_api_workflow(self, app_client, sample_dataset):
        """Test complete workflow through API endpoints."""
        # Step 1: Health check
        health_response = app_client.get("/api/health/")
        assert health_response.status_code == 200

        # Step 2: Create detector
        detector_data = {
            "name": "E2E Test Detector",
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1, "random_state": 42},
        }

        create_detector_response = app_client.post(
            "/api/detectors/", json=detector_data
        )
        assert create_detector_response.status_code == 200
        detector_id = create_detector_response.json()["id"]

        # Step 3: Upload dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as f:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("test_data.csv", f, "text/csv")},
                    data={"name": "E2E Test Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Step 4: Train detector
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
            )
            assert train_response.status_code == 200

            # Step 5: Run detection
            detect_response = app_client.post(
                f"/api/detectors/{detector_id}/detect", json={"dataset_id": dataset_id}
            )
            assert detect_response.status_code == 200
            result = detect_response.json()

            # Verify results
            assert "anomalies" in result
            assert "anomaly_rate" in result
            assert len(result["anomalies"]) > 0
            assert 0.05 <= result["anomaly_rate"] <= 0.15  # Expected range

            # Step 6: Get detection history
            history_response = app_client.get(f"/api/detectors/{detector_id}/results")
            assert history_response.status_code == 200
            history = history_response.json()
            assert len(history) >= 1

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_batch_processing_workflow(self, app_client, sample_dataset):
        """Test batch processing of multiple datasets."""
        # Create detector
        detector_data = {
            "name": "Batch Test Detector",
            "algorithm_name": "LocalOutlierFactor",
            "parameters": {"n_neighbors": 20},
        }

        create_response = app_client.post("/api/detectors/", json=detector_data)
        assert create_response.status_code == 200
        detector_id = create_response.json()["id"]

        # Create multiple datasets
        dataset_ids = []
        temp_files = []

        try:
            for i in range(3):
                # Create variant datasets
                variant_data = sample_dataset.copy()
                variant_data.iloc[:, 0] += i * 0.5  # Add variation

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False
                ) as f:
                    variant_data.to_csv(f.name, index=False)
                    temp_files.append(f.name)

                with open(f.name, "rb") as file:
                    upload_response = app_client.post(
                        "/api/datasets/upload",
                        files={"file": (f"batch_data_{i}.csv", file, "text/csv")},
                        data={"name": f"Batch Dataset {i}"},
                    )
                assert upload_response.status_code == 200
                dataset_ids.append(upload_response.json()["id"])

            # Train on first dataset
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train",
                json={"dataset_id": dataset_ids[0]},
            )
            assert train_response.status_code == 200

            # Run batch detection
            batch_request = {
                "dataset_ids": dataset_ids,
                "detector_id": detector_id,
                "options": {"parallel": True, "timeout": 300},
            }

            batch_response = app_client.post("/api/detection/batch", json=batch_request)
            assert batch_response.status_code == 200
            batch_results = batch_response.json()

            # Verify batch results
            assert len(batch_results["results"]) == 3
            for result in batch_results["results"]:
                assert "dataset_id" in result
                assert "anomalies" in result
                assert "status" in result
                assert result["status"] == "completed"

        finally:
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)

    def test_ensemble_detection_workflow(self, app_client, sample_dataset):
        """Test ensemble detection with multiple algorithms."""
        # Create multiple detectors
        algorithms = [
            ("IsolationForest", {"contamination": 0.1}),
            ("LocalOutlierFactor", {"n_neighbors": 20}),
            ("OneClassSVM", {"gamma": "scale"}),
        ]

        detector_ids = []
        for i, (algorithm, params) in enumerate(algorithms):
            detector_data = {
                "name": f"Ensemble Detector {i}",
                "algorithm_name": algorithm,
                "parameters": params,
            }

            response = app_client.post("/api/detectors/", json=detector_data)
            assert response.status_code == 200
            detector_ids.append(response.json()["id"])

        # Upload dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("ensemble_data.csv", file, "text/csv")},
                    data={"name": "Ensemble Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Train all detectors
            for detector_id in detector_ids:
                train_response = app_client.post(
                    f"/api/detectors/{detector_id}/train",
                    json={"dataset_id": dataset_id},
                )
                assert train_response.status_code == 200

            # Run ensemble detection
            ensemble_request = {
                "detector_ids": detector_ids,
                "dataset_id": dataset_id,
                "method": "voting",
                "weights": [0.4, 0.3, 0.3],
            }

            ensemble_response = app_client.post(
                "/api/detection/ensemble", json=ensemble_request
            )
            assert ensemble_response.status_code == 200
            ensemble_result = ensemble_response.json()

            # Verify ensemble results
            assert "anomalies" in ensemble_result
            assert "individual_results" in ensemble_result
            assert "ensemble_score" in ensemble_result
            assert len(ensemble_result["individual_results"]) == 3

            # Verify ensemble improves confidence
            ensemble_scores = ensemble_result["ensemble_score"]
            assert len(ensemble_scores) == len(sample_dataset)
            assert all(0 <= score <= 1 for score in ensemble_scores)

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_real_time_detection_workflow(self, app_client, sample_dataset):
        """Test real-time detection capabilities."""
        # Create streaming detector
        detector_data = {
            "name": "Streaming Detector",
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1, "random_state": 42},
        }

        response = app_client.post("/api/detectors/", json=detector_data)
        assert response.status_code == 200
        detector_id = response.json()["id"]

        # Upload training dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("streaming_data.csv", file, "text/csv")},
                    data={"name": "Streaming Dataset"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            # Train detector
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
            )
            assert train_response.status_code == 200

            # Test single sample detection
            sample_data = {
                "features": sample_dataset.iloc[0].tolist(),
                "detector_id": detector_id,
            }

            single_response = app_client.post("/api/detection/single", json=sample_data)
            assert single_response.status_code == 200
            single_result = single_response.json()

            assert "is_anomaly" in single_result
            assert "anomaly_score" in single_result
            assert "confidence" in single_result
            assert isinstance(single_result["is_anomaly"], bool)
            assert 0 <= single_result["anomaly_score"] <= 1

            # Test streaming batch
            streaming_data = {
                "samples": sample_dataset.iloc[:10].values.tolist(),
                "detector_id": detector_id,
                "stream_id": "test_stream_001",
            }

            stream_response = app_client.post(
                "/api/detection/stream", json=streaming_data
            )
            assert stream_response.status_code == 200
            stream_result = stream_response.json()

            assert "results" in stream_result
            assert "stream_id" in stream_result
            assert len(stream_result["results"]) == 10

            for result in stream_result["results"]:
                assert "is_anomaly" in result
                assert "anomaly_score" in result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_error_handling_workflow(self, app_client):
        """Test error handling in complete workflows."""
        # Test invalid detector creation
        invalid_detector = {
            "name": "",  # Invalid empty name
            "algorithm_name": "NonExistentAlgorithm",
            "parameters": {"invalid_param": "value"},
        }

        error_response = app_client.post("/api/detectors/", json=invalid_detector)
        assert error_response.status_code == 422

        # Test detection without training
        valid_detector = {
            "name": "Error Test Detector",
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1},
        }

        create_response = app_client.post("/api/detectors/", json=valid_detector)
        assert create_response.status_code == 200
        detector_id = create_response.json()["id"]

        # Try to detect without training
        detect_response = app_client.post(
            f"/api/detectors/{detector_id}/detect",
            json={"dataset_id": "non_existent_dataset"},
        )
        assert detect_response.status_code in [400, 404, 422]

        # Test invalid file upload
        invalid_upload = app_client.post(
            "/api/datasets/upload",
            files={"file": ("test.txt", b"invalid,data\nformat", "text/plain")},
            data={"name": "Invalid Dataset"},
        )
        assert invalid_upload.status_code in [400, 422]

    def test_performance_monitoring_workflow(self, app_client, sample_dataset):
        """Test performance monitoring throughout workflow."""
        import time

        start_time = time.time()

        # Create detector with timing
        detector_data = {
            "name": "Performance Test Detector",
            "algorithm_name": "IsolationForest",
            "parameters": {"contamination": 0.1},
        }

        create_start = time.time()
        create_response = app_client.post("/api/detectors/", json=detector_data)
        create_time = time.time() - create_start

        assert create_response.status_code == 200
        assert create_time < 5.0  # Should be fast
        detector_id = create_response.json()["id"]

        # Upload dataset with timing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_dataset.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            upload_start = time.time()
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("perf_data.csv", file, "text/csv")},
                    data={"name": "Performance Dataset"},
                )
            upload_time = time.time() - upload_start

            assert upload_response.status_code == 200
            assert upload_time < 10.0  # Should be reasonable
            dataset_id = upload_response.json()["id"]

            # Train with timing
            train_start = time.time()
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
            )
            train_time = time.time() - train_start

            assert train_response.status_code == 200
            assert train_time < 30.0  # Training should be reasonable

            # Detect with timing
            detect_start = time.time()
            detect_response = app_client.post(
                f"/api/detectors/{detector_id}/detect", json={"dataset_id": dataset_id}
            )
            detect_time = time.time() - detect_start

            assert detect_response.status_code == 200
            assert detect_time < 15.0  # Detection should be fast

            total_time = time.time() - start_time
            assert total_time < 60.0  # Complete workflow under 1 minute

            # Verify performance metrics in response
            result = detect_response.json()
            if "performance_metrics" in result:
                metrics = result["performance_metrics"]
                assert "execution_time" in metrics
                assert "memory_usage" in metrics
                assert metrics["execution_time"] > 0

        finally:
            Path(dataset_file).unlink(missing_ok=True)
