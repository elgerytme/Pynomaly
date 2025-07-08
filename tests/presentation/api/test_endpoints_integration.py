"""Integration tests for API endpoints with real algorithms."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pynomaly.application.services.algorithm_adapter_registry import (
    AlgorithmAdapterRegistry,
)
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.presentation.api.app import app


class TestDetectorEndpointsIntegration:
    """Integration tests for detector endpoints with real algorithms."""

    @pytest.fixture
    def client(self):
        """Test client."""
        return TestClient(app)

    @pytest.fixture
    def test_dataset(self):
        """Create test dataset with outliers."""
        np.random.seed(42)
        # Normal data
        normal_data = np.random.normal(0, 1, (100, 3))
        # Clear outliers
        outliers = np.random.normal(5, 0.5, (10, 3))

        data = np.vstack([normal_data, outliers])
        df = pd.DataFrame(data, columns=["x", "y", "z"])

        return Dataset(name="Integration Test Data", data=df)

    @pytest.fixture
    def mock_container(self, test_dataset):
        """Mock container with real services."""
        container = Mock()

        # Mock repositories
        detector_repo = Mock()
        dataset_repo = Mock()

        # Mock detector repository methods
        detector_repo.find_all.return_value = []
        detector_repo.find_by_id.return_value = None
        detector_repo.save.return_value = None
        detector_repo.exists.return_value = False

        # Mock dataset repository methods
        dataset_repo.find_by_id.return_value = test_dataset

        # Mock use cases with real implementations
        from pynomaly.application.use_cases.detect_anomalies import (
            DetectAnomaliesUseCase,
        )
        from pynomaly.application.use_cases.train_detector import TrainDetectorUseCase
        from pynomaly.domain.services import FeatureValidator

        feature_validator = FeatureValidator()
        adapter_registry = AlgorithmAdapterRegistry()

        train_use_case = TrainDetectorUseCase(
            detector_repository=detector_repo,
            feature_validator=feature_validator,
            adapter_registry=adapter_registry,
        )

        detect_use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repo,
            feature_validator=feature_validator,
            adapter_registry=adapter_registry,
        )

        # Configure container
        container.detector_repository.return_value = detector_repo
        container.dataset_repository.return_value = dataset_repo
        container.train_detector_use_case.return_value = train_use_case
        container.detect_anomalies_use_case.return_value = detect_use_case

        return container

    @pytest.mark.integration
    def test_list_available_algorithms(self, client):
        """Test listing available algorithms."""
        response = client.get("/api/detectors/algorithms")

        assert response.status_code == 200
        data = response.json()

        # Should have PyOD and sklearn algorithms
        assert "pyod" in data
        assert "sklearn" in data
        assert len(data["pyod"]) > 0
        assert len(data["sklearn"]) > 0

        # Should include key algorithms
        assert "IsolationForest" in data["pyod"]
        assert "LOF" in data["pyod"]
        assert "OneClassSVM" in data["pyod"]

    @pytest.mark.integration
    def test_create_detector_with_real_algorithm(self, client, mock_container):
        """Test creating detector with real PyOD algorithm."""
        with app.container.override(mock_container):
            detector_data = {
                "name": "Test Isolation Forest",
                "algorithm_name": "IsolationForest",
                "contamination_rate": 0.1,
                "parameters": {"n_estimators": 100, "random_state": 42},
                "metadata": {"description": "Integration test detector"},
            }

            response = client.post("/api/detectors/", json=detector_data)

            assert response.status_code == 200
            data = response.json()

            assert data["name"] == "Test Isolation Forest"
            assert data["algorithm_name"] == "IsolationForest"
            assert data["contamination_rate"] == 0.1
            assert data["is_fitted"] is False
            assert data["parameters"]["n_estimators"] == 100

    @pytest.mark.integration
    def test_full_training_workflow_api(self, client, mock_container, test_dataset):
        """Test complete training workflow through API."""
        # First create a detector
        detector = Detector(
            name="API Test Detector",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 50, "random_state": 42},
        )

        # Configure mocks to return our detector
        mock_container.detector_repository().find_by_id.return_value = detector

        with app.container.override(mock_container):
            # Test training endpoint
            train_request = {
                "detector_id": str(detector.id),
                "dataset_id": str(test_dataset.id),
                "validate_data": True,
                "save_model": True,
            }

            response = client.post("/api/detection/train", json=train_request)

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "training_time_ms" in data
            assert "dataset_summary" in data
            assert data["dataset_summary"]["training_samples"] == len(test_dataset.data)

    @pytest.mark.integration
    def test_full_detection_workflow_api(self, client, mock_container, test_dataset):
        """Test complete detection workflow through API."""
        # Create a fitted detector
        detector = Detector(
            name="API Detection Test",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 50, "random_state": 42},
            is_fitted=True,
        )

        # Configure mocks
        mock_container.detector_repository().find_by_id.return_value = detector

        with app.container.override(mock_container):
            # First train the detector to get it fitted with real algorithm
            train_request = {
                "detector_id": str(detector.id),
                "dataset_id": str(test_dataset.id),
                "validate_data": True,
                "save_model": True,
            }

            train_response = client.post("/api/detection/train", json=train_request)
            assert train_response.status_code == 200

            # Now test detection
            detect_request = {
                "detector_id": str(detector.id),
                "dataset_id": str(test_dataset.id),
                "validate_features": True,
                "save_results": True,
            }

            response = client.post("/api/detection/detect", json=detect_request)

            assert response.status_code == 200
            data = response.json()

            assert "detector_id" in data
            assert "dataset_id" in data
            assert "n_samples" in data
            assert "n_anomalies" in data
            assert "anomaly_rate" in data
            assert "threshold" in data
            assert "execution_time_ms" in data

            # Should detect some anomalies (we have clear outliers)
            assert data["n_samples"] == len(test_dataset.data)
            assert data["n_anomalies"] > 0
            assert 0.0 <= data["anomaly_rate"] <= 1.0

    @pytest.mark.integration
    def test_batch_detection_api(self, client, mock_container, test_dataset):
        """Test batch detection with multiple detectors."""
        # Create multiple fitted detectors
        detectors = [
            Detector(
                name=f"Batch Test {i}",
                algorithm_name="IsolationForest",
                contamination_rate=ContaminationRate(0.1),
                parameters={"n_estimators": 50, "random_state": 42 + i},
                is_fitted=True,
            )
            for i in range(3)
        ]

        # Mock repository to return different detectors
        def mock_find_by_id(detector_id):
            for detector in detectors:
                if detector.id == detector_id:
                    return detector
            return None

        mock_container.detector_repository().find_by_id.side_effect = mock_find_by_id

        # Mock detection service
        detection_service = Mock()
        detection_results = {}

        for detector in detectors:
            # Create mock detection result
            result = Mock()
            result.n_anomalies = 10 + detectors.index(detector)
            result.anomaly_rate = (10 + detectors.index(detector)) / len(
                test_dataset.data
            )
            result.threshold = 0.5 + detectors.index(detector) * 0.1
            result.execution_time_ms = 100.0 + detectors.index(detector) * 50
            detection_results[detector.id] = result

        detection_service.detect_with_multiple_detectors.return_value = (
            detection_results
        )
        mock_container.detection_service.return_value = detection_service

        with app.container.override(mock_container):
            batch_request = {
                "detector_ids": [str(d.id) for d in detectors],
                "dataset_id": str(test_dataset.id),
                "save_results": True,
            }

            response = client.post("/api/detection/detect/batch", json=batch_request)

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["n_detectors"] == 3
            assert len(data["results"]) == 3

            # Verify results for each detector
            for detector in detectors:
                detector_id_str = str(detector.id)
                assert detector_id_str in data["results"]
                result = data["results"][detector_id_str]
                assert "n_anomalies" in result
                assert "anomaly_rate" in result
                assert "threshold" in result
                assert "execution_time_ms" in result

    @pytest.mark.integration
    def test_algorithm_comparison_api(self, client, mock_container, test_dataset):
        """Test comparing different algorithms through API."""
        # Create detectors with different algorithms
        algorithms = ["IsolationForest", "LOF", "OneClassSVM"]
        detectors = []

        for algorithm in algorithms:
            detector = Detector(
                name=f"Compare {algorithm}",
                algorithm_name=algorithm,
                contamination_rate=ContaminationRate(0.1),
                parameters=(
                    {"random_state": 42} if algorithm != "LOF" else {"n_neighbors": 20}
                ),
                is_fitted=True,
            )
            detectors.append(detector)

        # Mock repository
        def mock_find_by_id(detector_id):
            for detector in detectors:
                if detector.id == detector_id:
                    return detector
            return None

        mock_container.detector_repository().find_by_id.side_effect = mock_find_by_id

        # Mock detection service for comparison
        detection_service = Mock()
        comparison_result = {
            "algorithms": algorithms,
            "dataset_size": len(test_dataset.data),
            "results": {
                algorithm: {
                    "n_anomalies": 10 + i,
                    "anomaly_rate": (10 + i) / len(test_dataset.data),
                    "execution_time_ms": 100 + i * 50,
                    "performance_score": 0.8 + i * 0.05,
                }
                for i, algorithm in enumerate(algorithms)
            },
            "best_algorithm": "OneClassSVM",
            "recommendation": "OneClassSVM shows the best balance of detection rate and performance",
        }

        detection_service.compare_detectors.return_value = comparison_result
        mock_container.detection_service.return_value = detection_service

        # Add target to dataset for comparison
        test_dataset.has_target = True

        with app.container.override(mock_container):
            response = client.get(
                f"/api/detection/compare?dataset_id={test_dataset.id}",
                params={"detector_ids": [str(d.id) for d in detectors]},
            )

            assert response.status_code == 200
            data = response.json()

            assert "algorithms" in data
            assert "results" in data
            assert "best_algorithm" in data
            assert len(data["algorithms"]) == 3
            assert len(data["results"]) == 3

    @pytest.mark.integration
    def test_error_handling_api(self, client, mock_container):
        """Test API error handling."""
        with app.container.override(mock_container):
            # Test training with non-existent detector
            train_request = {
                "detector_id": "00000000-0000-0000-0000-000000000000",
                "dataset_id": "00000000-0000-0000-0000-000000000000",
                "validate_data": True,
                "save_model": True,
            }

            response = client.post("/api/detection/train", json=train_request)
            assert response.status_code == 404
            assert "Detector not found" in response.json()["detail"]

            # Test detection with non-existent detector
            detect_request = {
                "detector_id": "00000000-0000-0000-0000-000000000000",
                "dataset_id": "00000000-0000-0000-0000-000000000000",
                "validate_features": True,
                "save_results": True,
            }

            response = client.post("/api/detection/detect", json=detect_request)
            assert response.status_code == 404
            assert "Detector not found" in response.json()["detail"]

    @pytest.mark.integration
    def test_api_performance_monitoring(self, client, mock_container, test_dataset):
        """Test API performance monitoring."""
        # Create detector
        detector = Detector(
            name="Performance Test",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 50, "random_state": 42},
            is_fitted=True,
        )

        mock_container.detector_repository().find_by_id.return_value = detector

        with app.container.override(mock_container):
            import time

            # Test training performance
            start_time = time.time()

            train_request = {
                "detector_id": str(detector.id),
                "dataset_id": str(test_dataset.id),
                "validate_data": True,
                "save_model": True,
            }

            response = client.post("/api/detection/train", json=train_request)
            training_time = time.time() - start_time

            assert response.status_code == 200
            assert training_time < 5.0  # Should complete within 5 seconds

            # Test detection performance
            start_time = time.time()

            detect_request = {
                "detector_id": str(detector.id),
                "dataset_id": str(test_dataset.id),
                "validate_features": False,  # Skip validation for speed
                "save_results": False,
            }

            response = client.post("/api/detection/detect", json=detect_request)
            detection_time = time.time() - start_time

            assert response.status_code == 200
            assert detection_time < 2.0  # Should complete within 2 seconds

            # Verify execution time is reported in response
            data = response.json()
            assert "execution_time_ms" in data
            assert data["execution_time_ms"] > 0

    @pytest.mark.integration
    def test_concurrent_api_requests(self, client, mock_container, test_dataset):
        """Test handling concurrent API requests."""
        import threading
        import time

        detector = Detector(
            name="Concurrent Test",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 20, "random_state": 42},  # Smaller for speed
            is_fitted=True,
        )

        mock_container.detector_repository().find_by_id.return_value = detector

        results = []
        errors = []

        def make_detection_request():
            try:
                with app.container.override(mock_container):
                    detect_request = {
                        "detector_id": str(detector.id),
                        "dataset_id": str(test_dataset.id),
                        "validate_features": False,
                        "save_results": False,
                    }

                    response = client.post("/api/detection/detect", json=detect_request)
                    results.append(response.status_code == 200)
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads for concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_detection_request)
            threads.append(thread)

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert all(results), "Some requests failed"
        assert total_time < 10.0, "Concurrent requests took too long"
        assert len(results) == 5, "Not all requests completed"
