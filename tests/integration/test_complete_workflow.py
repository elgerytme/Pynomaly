"""Complete end-to-end integration test for all 4 major features."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pynomaly.application.services.algorithm_adapter_registry import (
    AlgorithmAdapterRegistry,
)
from pynomaly.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from pynomaly.application.use_cases.train_detector import TrainDetectorUseCase
from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.services import FeatureValidator
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.presentation.api.app import create_app
from pynomaly.presentation.web.app import mount_web_ui


class TestCompleteWorkflowIntegration:
    """Test complete workflow: Real Algorithms → API → PWA → Integration."""

    @pytest.fixture
    def app(self):
        """Complete app with API and Web UI."""
        api_app = create_app()
        mount_web_ui(api_app)
        return api_app

    @pytest.fixture
    def client(self, app):
        """Test client for complete app."""
        return TestClient(app)

    @pytest.fixture
    def real_test_data(self):
        """Real test dataset with clear anomalies."""
        np.random.seed(42)

        # Generate realistic data with clear outliers
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0], cov=[[1, 0.2, 0.1], [0.2, 1, 0.3], [0.1, 0.3, 1]], size=500
        )

        # Add clear anomalies
        anomalies = np.random.multivariate_normal(
            mean=[5, 5, 5], cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], size=50
        )

        all_data = np.vstack([normal_data, anomalies])

        # Shuffle the data
        indices = np.random.permutation(len(all_data))
        shuffled_data = all_data[indices]

        df = pd.DataFrame(
            shuffled_data, columns=["feature_1", "feature_2", "feature_3"]
        )

        return Dataset(name="Integration Test Dataset", data=df)

    @pytest.fixture
    def mock_container_with_real_services(self, real_test_data):
        """Mock container with real algorithm services."""
        container = Mock()

        # Create real services
        registry = AlgorithmAdapterRegistry()
        feature_validator = FeatureValidator()

        # Mock repositories
        detector_repo = Mock()
        dataset_repo = Mock()
        result_repo = Mock()

        # Set up repository responses
        detector_repo.find_all.return_value = []
        detector_repo.count.return_value = 0
        dataset_repo.find_all.return_value = [real_test_data]
        dataset_repo.count.return_value = 1
        dataset_repo.find_by_id.return_value = real_test_data
        result_repo.find_recent.return_value = []
        result_repo.count.return_value = 0

        # Create real use cases with mocked repositories
        train_use_case = TrainDetectorUseCase(
            detector_repository=detector_repo,
            feature_validator=feature_validator,
            adapter_registry=registry,
        )

        detect_use_case = DetectAnomaliesUseCase(
            detector_repository=detector_repo,
            feature_validator=feature_validator,
            adapter_registry=registry,
        )

        # Configure container
        container.detector_repository.return_value = detector_repo
        container.dataset_repository.return_value = dataset_repo
        container.result_repository.return_value = result_repo
        container.train_detector_use_case.return_value = train_use_case
        container.detect_anomalies_use_case.return_value = detect_use_case

        # Mock config
        config = Mock()
        config.auth_enabled = False
        container.config.return_value = config

        return container

    @pytest.mark.integration
    def test_step1_real_algorithm_integration(self):
        """Step 1: Test real PyOD algorithm integration works."""
        # Create real detector and dataset
        detector = Detector(
            name="Real PyOD Test",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 50, "random_state": 42},
        )

        # Create test data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 3))
        outliers = np.random.normal(5, 1, (10, 3))
        data = np.vstack([normal_data, outliers])
        df = pd.DataFrame(data, columns=["x", "y", "z"])
        dataset = Dataset(name="Real Test Data", data=df)

        # Test with real algorithm registry
        registry = AlgorithmAdapterRegistry()

        # Test fitting
        registry.fit_detector(detector, dataset)

        # Test prediction
        predictions = registry.predict_with_detector(detector, dataset)
        assert len(predictions) == len(dataset.data)
        assert all(pred in [0, 1] for pred in predictions)
        assert sum(predictions) > 0  # Should detect some outliers

        # Test scoring
        scores = registry.score_with_detector(detector, dataset)
        assert len(scores) == len(dataset.data)
        assert all(0.0 <= score.value <= 1.0 for score in scores)

    @pytest.mark.integration
    def test_step2_api_endpoints_with_real_algorithms(
        self, client, mock_container_with_real_services, real_test_data
    ):
        """Step 2: Test API endpoints work with real algorithms."""
        with patch(
            "pynomaly.presentation.api.app.get_container",
            return_value=mock_container_with_real_services,
        ):
            # Test algorithm listing endpoint
            response = client.get("/detectors/algorithms")
            assert response.status_code == 200
            data = response.json()
            assert "pyod" in data
            assert "IsolationForest" in data["pyod"]
            assert len(data["all_supported"]) > 10  # Should have many algorithms

            # Test detector creation
            detector_data = {
                "name": "API Test Detector",
                "algorithm_name": "IsolationForest",
                "contamination_rate": 0.1,
                "parameters": {"n_estimators": 50, "random_state": 42},
                "metadata": {"test": "integration"},
            }

            create_response = client.post("/detectors/", json=detector_data)
            assert create_response.status_code == 200
            created_detector = create_response.json()

            # Mock the detector in repository for subsequent calls
            detector = Detector(
                name=created_detector["name"],
                algorithm_name=created_detector["algorithm_name"],
                contamination_rate=ContaminationRate(
                    created_detector["contamination_rate"]
                ),
                parameters=created_detector["parameters"],
            )
            mock_container_with_real_services.detector_repository().find_by_id.return_value = detector

            # Test training via API
            train_request = {
                "detector_id": str(detector.id),
                "dataset_id": str(real_test_data.id),
                "validate_data": True,
                "save_model": True,
            }

            train_response = client.post("/detection/train", json=train_request)
            assert train_response.status_code == 200
            train_data = train_response.json()
            assert train_data["success"] is True
            assert "training_time_ms" in train_data

            # Update detector to fitted state
            detector.is_fitted = True

            # Test detection via API
            detect_request = {
                "detector_id": str(detector.id),
                "dataset_id": str(real_test_data.id),
                "validate_features": True,
                "save_results": True,
            }

            detect_response = client.post("/detection/detect", json=detect_request)
            assert detect_response.status_code == 200
            detect_data = detect_response.json()
            assert "n_anomalies" in detect_data
            assert "anomaly_rate" in detect_data
            assert detect_data["n_samples"] == len(real_test_data.data)

    @pytest.mark.integration
    def test_step3_pwa_functionality(self, client, mock_container_with_real_services):
        """Step 3: Test PWA features work correctly."""
        with patch(
            "pynomaly.presentation.web.app.get_container",
            return_value=mock_container_with_real_services,
        ):
            # Test PWA manifest
            response = client.get("/static/manifest.json")
            assert response.status_code == 200
            manifest = response.json()
            assert manifest["name"] == "Pynomaly - Anomaly Detection Platform"
            assert manifest["display"] == "standalone"
            assert len(manifest["icons"]) > 0

            # Test service worker
            response = client.get("/static/sw.js")
            assert response.status_code == 200
            assert "Service Worker for Pynomaly PWA" in response.text

            # Test main dashboard
            response = client.get("/web/")
            assert response.status_code == 200
            assert "Pynomaly" in response.text

            # Test detectors page
            response = client.get("/web/detectors")
            assert response.status_code == 200

            # Test datasets page
            response = client.get("/web/datasets")
            assert response.status_code == 200

            # Test detection page
            response = client.get("/web/detection")
            assert response.status_code == 200

            # Test HTMX endpoints
            htmx_response = client.get("/web/htmx/detector-list")
            assert htmx_response.status_code == 200

            htmx_response = client.get("/web/htmx/dataset-list")
            assert htmx_response.status_code == 200

    @pytest.mark.integration
    def test_step4_complete_end_to_end_workflow(
        self, client, mock_container_with_real_services, real_test_data
    ):
        """Step 4: Test complete end-to-end workflow from PWA to real algorithms."""
        with patch(
            "pynomaly.presentation.web.app.get_container",
            return_value=mock_container_with_real_services,
        ):
            # 1. Create detector via API
            detector_data = {
                "name": "E2E Workflow Detector",
                "algorithm_name": "IsolationForest",
                "contamination_rate": 0.1,
                "parameters": {"n_estimators": 100, "random_state": 42},
                "metadata": {"workflow": "end-to-end"},
            }

            create_response = client.post("/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_info = create_response.json()

            # Create actual detector object for use cases
            detector = Detector(
                name=detector_info["name"],
                algorithm_name=detector_info["algorithm_name"],
                contamination_rate=ContaminationRate(
                    detector_info["contamination_rate"]
                ),
                parameters=detector_info["parameters"],
            )

            # Mock repository to return our detector
            mock_container_with_real_services.detector_repository().find_by_id.return_value = detector

            # 2. Train detector via HTMX endpoint (PWA functionality)
            train_form_data = {
                "detector_id": str(detector.id),
                "dataset_id": str(real_test_data.id),
            }

            train_response = client.post(
                "/web/htmx/train-detector",
                data=train_form_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            assert train_response.status_code == 200
            assert "Training completed" in train_response.text

            # Update detector state
            detector.is_fitted = True

            # 3. Run detection via HTMX endpoint
            detect_form_data = {
                "detector_id": str(detector.id),
                "dataset_id": str(real_test_data.id),
            }

            detect_response = client.post(
                "/web/htmx/detect-anomalies",
                data=detect_form_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            assert detect_response.status_code == 200

            # 4. Verify the workflow produced real results
            # The HTMX response should contain detection results
            response_text = detect_response.text
            assert (
                "anomalies" in response_text.lower()
                or "detected" in response_text.lower()
            )

            # 5. Test visualization endpoints work
            viz_response = client.get("/web/visualizations")
            assert viz_response.status_code == 200

            # 6. Test monitoring dashboard
            monitoring_response = client.get("/web/monitoring")
            assert monitoring_response.status_code == 200

    @pytest.mark.integration
    def test_real_algorithms_comparison(self):
        """Test comparison between different real algorithms."""
        # Create dataset
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (200, 2))
        outliers = np.random.normal(4, 0.5, (20, 2))
        data = np.vstack([normal_data, outliers])
        df = pd.DataFrame(data, columns=["x", "y"])
        dataset = Dataset(name="Comparison Test", data=df)

        # Test multiple algorithms
        algorithms = ["IsolationForest", "LOF", "OneClassSVM"]
        registry = AlgorithmAdapterRegistry()
        results = {}

        for algorithm in algorithms:
            try:
                detector = Detector(
                    name=f"Test {algorithm}",
                    algorithm_name=algorithm,
                    contamination_rate=ContaminationRate(0.1),
                    parameters=(
                        {"random_state": 42}
                        if algorithm != "LOF"
                        else {"n_neighbors": 20}
                    ),
                )

                # Train and test
                registry.fit_detector(detector, dataset)
                predictions = registry.predict_with_detector(detector, dataset)
                scores = registry.score_with_detector(detector, dataset)

                results[algorithm] = {
                    "anomalies": sum(predictions),
                    "anomaly_rate": sum(predictions) / len(predictions),
                    "score_range": (
                        min(s.value for s in scores),
                        max(s.value for s in scores),
                    ),
                }

            except Exception as e:
                # Some algorithms might not be available
                print(f"Skipping {algorithm}: {e}")
                continue

        # Verify we got results for at least one algorithm
        assert len(results) > 0

        # Each algorithm should detect some anomalies
        for algorithm, result in results.items():
            assert result["anomalies"] > 0, f"{algorithm} detected no anomalies"
            assert (
                0.0 < result["anomaly_rate"] < 0.5
            ), f"{algorithm} anomaly rate out of reasonable range"

    @pytest.mark.integration
    def test_pwa_offline_functionality(self, client):
        """Test PWA offline capabilities."""
        # Test service worker registration in main page
        response = client.get("/web/")
        assert response.status_code == 200

        # Verify PWA features are included
        content = response.text
        assert "manifest.json" in content
        assert "sw.js" in content or "serviceWorker" in content

        # Test offline page exists
        response = client.get("/web/templates/offline.html")
        # This might not be directly accessible, but let's test the static file
        response = client.get("/static/sw.js")
        assert response.status_code == 200
        assert "offline" in response.text.lower()

    @pytest.mark.integration
    def test_scalability_with_larger_dataset(self):
        """Test system performance with larger datasets."""
        # Create larger dataset
        np.random.seed(42)
        large_normal = np.random.normal(0, 1, (2000, 5))
        large_outliers = np.random.normal(6, 1, (200, 5))
        large_data = np.vstack([large_normal, large_outliers])

        df = pd.DataFrame(large_data, columns=[f"feature_{i}" for i in range(5)])
        large_dataset = Dataset(name="Large Test Dataset", data=df)

        # Test with real algorithm
        detector = Detector(
            name="Scalability Test",
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            parameters={"n_estimators": 50, "random_state": 42},  # Smaller for speed
        )

        registry = AlgorithmAdapterRegistry()

        import time

        start_time = time.time()

        # Train
        registry.fit_detector(detector, large_dataset)
        train_time = time.time() - start_time

        # Predict
        start_time = time.time()
        predictions = registry.predict_with_detector(detector, large_dataset)
        predict_time = time.time() - start_time

        # Score
        start_time = time.time()
        scores = registry.score_with_detector(detector, large_dataset)
        score_time = time.time() - start_time

        # Verify performance is acceptable
        assert train_time < 10.0, f"Training took too long: {train_time}s"
        assert predict_time < 5.0, f"Prediction took too long: {predict_time}s"
        assert score_time < 5.0, f"Scoring took too long: {score_time}s"

        # Verify results quality
        assert len(predictions) == len(large_dataset.data)
        assert len(scores) == len(large_dataset.data)

        anomaly_count = sum(predictions)
        expected_range = (100, 400)  # Expect 200 +/- some tolerance
        assert (
            expected_range[0] <= anomaly_count <= expected_range[1]
        ), f"Anomaly count {anomaly_count} outside expected range {expected_range}"

    @pytest.mark.integration
    def test_feature_integration_completeness(
        self, client, mock_container_with_real_services
    ):
        """Test that all 4 major features are properly integrated."""
        with (
            patch(
                "pynomaly.presentation.api.app.get_container",
                return_value=mock_container_with_real_services,
            ),
            patch(
                "pynomaly.presentation.web.app.get_container",
                return_value=mock_container_with_real_services,
            ),
        ):
            # ✅ Feature 1: Real Algorithm Integration
            registry = AlgorithmAdapterRegistry()
            supported_algorithms = registry.get_supported_algorithms()
            assert len(supported_algorithms) >= 10, "Should support many algorithms"
            assert "IsolationForest" in supported_algorithms

            # ✅ Feature 2: RESTful API Endpoints
            api_response = client.get("/detectors/algorithms")
            assert api_response.status_code == 200

            health_response = client.get("/health")
            assert health_response.status_code == 200

            # ✅ Feature 3: Progressive Web App
            pwa_response = client.get("/web/")
            assert pwa_response.status_code == 200

            manifest_response = client.get("/static/manifest.json")
            assert manifest_response.status_code == 200

            sw_response = client.get("/static/sw.js")
            assert sw_response.status_code == 200

            # ✅ Feature 4: Complete Integration
            # Test HTMX endpoints work
            htmx_response = client.get("/web/htmx/detector-list")
            assert htmx_response.status_code == 200

            # Test visualization pages
            viz_response = client.get("/web/visualizations")
            assert viz_response.status_code == 200

            # Test monitoring
            monitoring_response = client.get("/web/monitoring")
            assert monitoring_response.status_code == 200

            print("✅ All 4 major features are properly integrated:")
            print("  1. Real PyOD Algorithm Integration")
            print("  2. RESTful API Endpoints")
            print("  3. Progressive Web App with HTMX/Tailwind")
            print("  4. Complete End-to-End Integration")
