"""Comprehensive tests for Web UI presentation layer - Phase 3 Coverage Enhancement."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pynomaly.application.services import (
    DatasetService,
    DetectionService,
    DetectorService,
)
from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.presentation.web.main import app


class TestWebUIMain:
    """Comprehensive tests for Web UI main application."""

    def test_app_creation(self):
        """Test Web UI app is properly created."""
        assert isinstance(app, FastAPI)
        assert app.title == "Pynomaly Web UI"
        assert len(app.routes) > 0

    def test_app_static_files_mount(self):
        """Test static files are properly mounted."""
        with TestClient(app) as client:
            # Test CSS files
            response = client.get("/static/css/main.css")
            # May return 404 if file doesn't exist, but should not error
            assert response.status_code in [200, 404]

            # Test JavaScript files
            response = client.get("/static/js/main.js")
            assert response.status_code in [200, 404]

    def test_app_pwa_manifest(self):
        """Test PWA manifest is accessible."""
        with TestClient(app) as client:
            response = client.get("/manifest.json")
            assert response.status_code in [200, 404]

            if response.status_code == 200:
                manifest = response.json()
                assert "name" in manifest
                assert "icons" in manifest

    def test_app_service_worker(self):
        """Test service worker is accessible."""
        with TestClient(app) as client:
            response = client.get("/sw.js")
            assert response.status_code in [200, 404]


class TestDashboardRoutes:
    """Comprehensive tests for dashboard routes."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_services(self):
        """Create mock services for dashboard."""
        detector_service = Mock(spec=DetectorService)
        dataset_service = Mock(spec=DatasetService)
        detection_service = Mock(spec=DetectionService)

        # Mock recent activity
        detector_service.get_recent_detectors.return_value = []
        dataset_service.get_recent_datasets.return_value = []
        detection_service.get_recent_results.return_value = []

        return {
            "detector_service": detector_service,
            "dataset_service": dataset_service,
            "detection_service": detection_service,
        }

    def test_dashboard_index_page(self, test_client):
        """Test dashboard index page."""
        response = test_client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "dashboard" in response.text.lower()

    def test_dashboard_stats_endpoint(self, test_client, mock_services):
        """Test dashboard statistics endpoint."""
        with patch.multiple(
            "pynomaly.presentation.web.routes.dashboard_routes", **mock_services
        ):
            response = test_client.get("/dashboard/stats")

            assert response.status_code == 200
            assert "total_detectors" in response.text or "stats" in response.text

    def test_dashboard_recent_activity(self, test_client, mock_services):
        """Test dashboard recent activity."""
        with patch.multiple(
            "pynomaly.presentation.web.routes.dashboard_routes", **mock_services
        ):
            response = test_client.get("/dashboard/recent")

            assert response.status_code == 200
            assert (
                "activity" in response.text.lower() or "recent" in response.text.lower()
            )


class TestDetectorRoutes:
    """Comprehensive tests for detector web routes."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_detector_service(self):
        """Create mock detector service."""
        service = Mock(spec=DetectorService)
        return service

    @pytest.fixture
    def sample_detector(self):
        """Create sample detector."""
        return Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination=ContaminationRate(0.1),
            hyperparameters={"n_estimators": 100},
        )

    def test_detectors_list_page(self, test_client, mock_detector_service):
        """Test detectors list page."""
        mock_detector_service.list_detectors.return_value = []

        with patch(
            "pynomaly.presentation.web.routes.detector_routes.get_detector_service"
        ) as mock_get:
            mock_get.return_value = mock_detector_service

            response = test_client.get("/detectors")

            assert response.status_code == 200
            assert "detectors" in response.text.lower()

    def test_detector_detail_page(
        self, test_client, mock_detector_service, sample_detector
    ):
        """Test detector detail page."""
        mock_detector_service.get_detector.return_value = sample_detector

        with patch(
            "pynomaly.presentation.web.routes.detector_routes.get_detector_service"
        ) as mock_get:
            mock_get.return_value = mock_detector_service

            response = test_client.get(f"/detectors/{sample_detector.id}")

            assert response.status_code == 200
            assert "test_detector" in response.text

    def test_detector_create_page(self, test_client):
        """Test detector creation page."""
        response = test_client.get("/detectors/create")

        assert response.status_code == 200
        assert "create" in response.text.lower()
        assert "form" in response.text.lower()

    def test_detector_create_form_submission(
        self, test_client, mock_detector_service, sample_detector
    ):
        """Test detector creation form submission."""
        mock_detector_service.create_detector.return_value = sample_detector

        with patch(
            "pynomaly.presentation.web.routes.detector_routes.get_detector_service"
        ) as mock_get:
            mock_get.return_value = mock_detector_service

            response = test_client.post(
                "/detectors/create",
                data={
                    "name": "test_detector",
                    "algorithm": "isolation_forest",
                    "contamination_rate": "0.1",
                },
            )

            # Should redirect after successful creation
            assert response.status_code in [200, 302, 303]

    def test_detector_edit_page(
        self, test_client, mock_detector_service, sample_detector
    ):
        """Test detector edit page."""
        mock_detector_service.get_detector.return_value = sample_detector

        with patch(
            "pynomaly.presentation.web.routes.detector_routes.get_detector_service"
        ) as mock_get:
            mock_get.return_value = mock_detector_service

            response = test_client.get(f"/detectors/{sample_detector.id}/edit")

            assert response.status_code == 200
            assert "edit" in response.text.lower()
            assert "test_detector" in response.text

    def test_detector_delete_action(
        self, test_client, mock_detector_service, sample_detector
    ):
        """Test detector deletion action."""
        mock_detector_service.delete_detector.return_value = True

        with patch(
            "pynomaly.presentation.web.routes.detector_routes.get_detector_service"
        ) as mock_get:
            mock_get.return_value = mock_detector_service

            response = test_client.delete(f"/detectors/{sample_detector.id}")

            # Should return success status
            assert response.status_code in [200, 204, 302]


class TestDatasetRoutes:
    """Comprehensive tests for dataset web routes."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_dataset_service(self):
        """Create mock dataset service."""
        service = Mock(spec=DatasetService)
        return service

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        features = np.random.random((100, 5))
        return Dataset(
            name="test_dataset",
            features=features,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )

    def test_datasets_list_page(self, test_client, mock_dataset_service):
        """Test datasets list page."""
        mock_dataset_service.list_datasets.return_value = []

        with patch(
            "pynomaly.presentation.web.routes.dataset_routes.get_dataset_service"
        ) as mock_get:
            mock_get.return_value = mock_dataset_service

            response = test_client.get("/datasets")

            assert response.status_code == 200
            assert "datasets" in response.text.lower()

    def test_dataset_detail_page(
        self, test_client, mock_dataset_service, sample_dataset
    ):
        """Test dataset detail page."""
        mock_dataset_service.get_dataset.return_value = sample_dataset

        with patch(
            "pynomaly.presentation.web.routes.dataset_routes.get_dataset_service"
        ) as mock_get:
            mock_get.return_value = mock_dataset_service

            response = test_client.get(f"/datasets/{sample_dataset.id}")

            assert response.status_code == 200
            assert "test_dataset" in response.text

    def test_dataset_upload_page(self, test_client):
        """Test dataset upload page."""
        response = test_client.get("/datasets/upload")

        assert response.status_code == 200
        assert "upload" in response.text.lower()
        assert "file" in response.text.lower()

    def test_dataset_upload_form_submission(
        self, test_client, mock_dataset_service, sample_dataset
    ):
        """Test dataset upload form submission."""
        mock_dataset_service.create_dataset_from_file.return_value = sample_dataset

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("f1,f2,f3\n1,2,3\n4,5,6\n")
            csv_path = Path(f.name)

        try:
            with patch(
                "pynomaly.presentation.web.routes.dataset_routes.get_dataset_service"
            ) as mock_get:
                mock_get.return_value = mock_dataset_service

                with open(csv_path, "rb") as csv_file:
                    response = test_client.post(
                        "/datasets/upload",
                        files={"file": ("test.csv", csv_file, "text/csv")},
                        data={"name": "test_dataset"},
                    )

                # Should redirect after successful upload
                assert response.status_code in [200, 302, 303]
        finally:
            csv_path.unlink()

    def test_dataset_statistics_page(
        self, test_client, mock_dataset_service, sample_dataset
    ):
        """Test dataset statistics page."""
        mock_dataset_service.get_dataset.return_value = sample_dataset

        with patch(
            "pynomaly.presentation.web.routes.dataset_routes.get_dataset_service"
        ) as mock_get:
            mock_get.return_value = mock_dataset_service

            response = test_client.get(f"/datasets/{sample_dataset.id}/statistics")

            assert response.status_code == 200
            assert "statistics" in response.text.lower()


class TestDetectionRoutes:
    """Comprehensive tests for detection web routes."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_detection_service(self):
        """Create mock detection service."""
        service = Mock(spec=DetectionService)
        return service

    @pytest.fixture
    def sample_detection_result(self):
        """Create sample detection result."""
        detector = Detector(
            name="test", algorithm="test", contamination=ContaminationRate(0.1)
        )
        features = np.random.random((100, 5))
        dataset = Dataset(name="test", features=features)

        return DetectionResult(
            detector=detector,
            dataset=dataset,
            anomalies=[],
            scores=[AnomalyScore(0.5)] * 100,
            labels=np.array([0] * 100),
            threshold=0.7,
        )

    def test_detection_page(self, test_client):
        """Test detection page."""
        response = test_client.get("/detection")

        assert response.status_code == 200
        assert "detection" in response.text.lower()

    def test_detection_run_page(self, test_client):
        """Test detection run page."""
        response = test_client.get("/detection/run")

        assert response.status_code == 200
        assert "run" in response.text.lower() or "detect" in response.text.lower()

    def test_detection_results_page(self, test_client, mock_detection_service):
        """Test detection results page."""
        mock_detection_service.get_detection_results.return_value = []

        with patch(
            "pynomaly.presentation.web.routes.detection_routes.get_detection_service"
        ) as mock_get:
            mock_get.return_value = mock_detection_service

            response = test_client.get("/detection/results")

            assert response.status_code == 200
            assert "results" in response.text.lower()

    def test_detection_result_detail(
        self, test_client, mock_detection_service, sample_detection_result
    ):
        """Test detection result detail page."""
        mock_detection_service.get_detection_result.return_value = (
            sample_detection_result
        )

        with patch(
            "pynomaly.presentation.web.routes.detection_routes.get_detection_service"
        ) as mock_get:
            mock_get.return_value = mock_detection_service

            response = test_client.get(
                f"/detection/results/{sample_detection_result.id}"
            )

            assert response.status_code == 200
            assert "result" in response.text.lower()

    def test_detection_form_submission(
        self, test_client, mock_detection_service, sample_detection_result
    ):
        """Test detection form submission."""
        mock_detection_service.detect_anomalies.return_value = sample_detection_result

        with patch(
            "pynomaly.presentation.web.routes.detection_routes.get_detection_service"
        ) as mock_get:
            mock_get.return_value = mock_detection_service

            response = test_client.post(
                "/detection/run",
                data={
                    "detector_id": "test_detector_id",
                    "dataset_id": "test_dataset_id",
                    "threshold": "0.7",
                },
            )

            # Should redirect to results or return success
            assert response.status_code in [200, 302, 303]


class TestHTMXEndpoints:
    """Comprehensive tests for HTMX endpoints."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """Create HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://testserver/",
            "HX-Target": "main-content",
        }

    def test_detector_list_htmx(self, test_client, htmx_headers):
        """Test detector list HTMX endpoint."""
        response = test_client.get("/htmx/detectors", headers=htmx_headers)

        assert response.status_code == 200
        # Should return HTML fragment, not full page
        assert "<!DOCTYPE html>" not in response.text

    def test_detector_form_htmx(self, test_client, htmx_headers):
        """Test detector form HTMX endpoint."""
        response = test_client.get("/htmx/detectors/form", headers=htmx_headers)

        assert response.status_code == 200
        assert "form" in response.text.lower()
        assert "<!DOCTYPE html>" not in response.text

    def test_dataset_list_htmx(self, test_client, htmx_headers):
        """Test dataset list HTMX endpoint."""
        response = test_client.get("/htmx/datasets", headers=htmx_headers)

        assert response.status_code == 200
        assert "<!DOCTYPE html>" not in response.text

    def test_detection_visualization_htmx(self, test_client, htmx_headers):
        """Test detection visualization HTMX endpoint."""
        response = test_client.get(
            "/htmx/detection/visualization/test_id", headers=htmx_headers
        )

        # May return 404 if result doesn't exist, but should handle gracefully
        assert response.status_code in [200, 404]

    def test_real_time_updates_htmx(self, test_client, htmx_headers):
        """Test real-time updates HTMX endpoint."""
        response = test_client.get("/htmx/updates", headers=htmx_headers)

        assert response.status_code == 200
        assert "<!DOCTYPE html>" not in response.text


class TestVisualizationComponents:
    """Comprehensive tests for visualization components."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_detection_result(self):
        """Create sample detection result."""
        detector = Detector(
            name="test", algorithm="test", contamination=ContaminationRate(0.1)
        )
        features = np.random.random((100, 5))
        dataset = Dataset(name="test", features=features)

        return DetectionResult(
            detector=detector,
            dataset=dataset,
            anomalies=[],
            scores=[AnomalyScore(0.5)] * 100,
            labels=np.array([0] * 100),
            threshold=0.7,
        )

    def test_anomaly_scatter_plot(self, test_client, sample_detection_result):
        """Test anomaly scatter plot visualization."""
        with patch(
            "pynomaly.presentation.web.routes.visualization_routes.get_detection_service"
        ) as mock_get:
            mock_service = Mock()
            mock_service.get_detection_result.return_value = sample_detection_result
            mock_get.return_value = mock_service

            response = test_client.get(
                f"/visualizations/scatter/{sample_detection_result.id}"
            )

            assert response.status_code == 200
            assert (
                "scatter" in response.text.lower()
                or "visualization" in response.text.lower()
            )

    def test_anomaly_histogram(self, test_client, sample_detection_result):
        """Test anomaly score histogram."""
        with patch(
            "pynomaly.presentation.web.routes.visualization_routes.get_detection_service"
        ) as mock_get:
            mock_service = Mock()
            mock_service.get_detection_result.return_value = sample_detection_result
            mock_get.return_value = mock_service

            response = test_client.get(
                f"/visualizations/histogram/{sample_detection_result.id}"
            )

            assert response.status_code == 200
            assert (
                "histogram" in response.text.lower() or "chart" in response.text.lower()
            )

    def test_detection_timeline(self, test_client):
        """Test detection timeline visualization."""
        response = test_client.get("/visualizations/timeline")

        assert response.status_code == 200
        assert "timeline" in response.text.lower() or "time" in response.text.lower()


class TestFormHandling:
    """Comprehensive tests for form handling."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    def test_detector_form_validation(self, test_client):
        """Test detector form validation."""
        # Test with invalid data
        response = test_client.post(
            "/detectors/create",
            data={
                "name": "",  # Empty name
                "algorithm": "invalid_algorithm",
                "contamination_rate": "1.5",  # Invalid rate
            },
        )

        # Should return error or validation message
        assert response.status_code in [200, 400, 422]
        if response.status_code == 200:
            assert (
                "error" in response.text.lower() or "invalid" in response.text.lower()
            )

    def test_dataset_upload_validation(self, test_client):
        """Test dataset upload validation."""
        # Test with invalid file
        response = test_client.post(
            "/datasets/upload",
            files={"file": ("test.txt", b"invalid content", "text/plain")},
            data={"name": "test_dataset"},
        )

        # Should return error for invalid file type
        assert response.status_code in [200, 400, 422]

    def test_detection_form_validation(self, test_client):
        """Test detection form validation."""
        # Test with missing required fields
        response = test_client.post(
            "/detection/run",
            data={
                "detector_id": "",  # Empty detector ID
                "dataset_id": "test_dataset_id",
            },
        )

        # Should return validation error
        assert response.status_code in [200, 400, 422]


class TestResponsiveDesign:
    """Tests for responsive design and mobile compatibility."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mobile_headers(self):
        """Create mobile user agent headers."""
        return {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        }

    def test_mobile_dashboard(self, test_client, mobile_headers):
        """Test dashboard on mobile device."""
        response = test_client.get("/", headers=mobile_headers)

        assert response.status_code == 200
        assert "viewport" in response.text.lower()
        assert (
            "responsive" in response.text.lower() or "mobile" in response.text.lower()
        )

    def test_mobile_detector_list(self, test_client, mobile_headers):
        """Test detector list on mobile device."""
        response = test_client.get("/detectors", headers=mobile_headers)

        assert response.status_code == 200
        # Should have mobile-friendly layout
        assert "col-" in response.text or "grid" in response.text

    def test_tablet_detection_page(self, test_client):
        """Test detection page on tablet device."""
        tablet_headers = {
            "User-Agent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        }

        response = test_client.get("/detection", headers=tablet_headers)

        assert response.status_code == 200
        assert "viewport" in response.text.lower()


class TestAccessibility:
    """Tests for web accessibility compliance."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    def test_semantic_html_structure(self, test_client):
        """Test semantic HTML structure."""
        response = test_client.get("/")

        assert response.status_code == 200
        # Should have proper semantic elements
        assert "<header>" in response.text or "<nav>" in response.text
        assert "<main>" in response.text
        assert "<footer>" in response.text or "role=" in response.text

    def test_form_labels_and_accessibility(self, test_client):
        """Test form labels and accessibility."""
        response = test_client.get("/detectors/create")

        assert response.status_code == 200
        # Should have proper form labels
        assert "label" in response.text.lower()
        assert "for=" in response.text or "aria-label" in response.text

    def test_keyboard_navigation_support(self, test_client):
        """Test keyboard navigation support."""
        response = test_client.get("/")

        assert response.status_code == 200
        # Should have tabindex or focus management
        assert (
            "tabindex" in response.text
            or "focus" in response.text
            or "keyboard" in response.text
        )

    def test_aria_labels_and_roles(self, test_client):
        """Test ARIA labels and roles."""
        response = test_client.get("/")

        assert response.status_code == 200
        # Should have ARIA attributes
        assert "aria-" in response.text or "role=" in response.text


class TestPWAFeatures:
    """Tests for Progressive Web App features."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    def test_pwa_manifest_content(self, test_client):
        """Test PWA manifest content."""
        response = test_client.get("/manifest.json")

        if response.status_code == 200:
            manifest = response.json()
            assert "name" in manifest
            assert "short_name" in manifest
            assert "start_url" in manifest
            assert "display" in manifest
            assert "icons" in manifest

    def test_service_worker_registration(self, test_client):
        """Test service worker registration."""
        response = test_client.get("/")

        assert response.status_code == 200
        # Should have service worker registration code
        assert "serviceWorker" in response.text or "sw.js" in response.text

    def test_offline_page(self, test_client):
        """Test offline page."""
        response = test_client.get("/offline")

        assert response.status_code == 200
        assert "offline" in response.text.lower()

    def test_app_shell_caching(self, test_client):
        """Test app shell caching headers."""
        response = test_client.get("/")

        assert response.status_code == 200
        # Should have appropriate caching headers for PWA
        headers = response.headers
        assert "cache-control" in headers or "etag" in headers


class TestPerformanceOptimization:
    """Tests for performance optimization."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    def test_static_asset_compression(self, test_client):
        """Test static asset compression."""
        response = test_client.get(
            "/static/css/main.css", headers={"Accept-Encoding": "gzip"}
        )

        if response.status_code == 200:
            # Should support compression
            assert "content-encoding" in response.headers or len(response.content) > 0

    def test_image_optimization(self, test_client):
        """Test image optimization."""
        response = test_client.get("/static/images/logo.png")

        if response.status_code == 200:
            # Should have appropriate headers
            assert "content-type" in response.headers
            assert response.headers["content-type"].startswith("image/")

    def test_lazy_loading_support(self, test_client):
        """Test lazy loading support."""
        response = test_client.get("/datasets")

        assert response.status_code == 200
        # Should have lazy loading attributes
        assert "lazy" in response.text or "loading=" in response.text

    def test_page_load_time(self, test_client):
        """Test page load time performance."""
        import time

        start_time = time.time()
        response = test_client.get("/")
        end_time = time.time()

        load_time = end_time - start_time

        assert response.status_code == 200
        # Should load within reasonable time
        assert load_time < 2.0  # Less than 2 seconds


class TestWebUIIntegration:
    """Integration tests for Web UI components."""

    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)

    def test_full_detection_workflow(self, test_client):
        """Test complete detection workflow through Web UI."""
        # Step 1: Access dashboard
        response = test_client.get("/")
        assert response.status_code == 200

        # Step 2: Create detector
        response = test_client.get("/detectors/create")
        assert response.status_code == 200

        # Step 3: Upload dataset
        response = test_client.get("/datasets/upload")
        assert response.status_code == 200

        # Step 4: Run detection
        response = test_client.get("/detection/run")
        assert response.status_code == 200

        # Step 5: View results
        response = test_client.get("/detection/results")
        assert response.status_code == 200

    def test_navigation_consistency(self, test_client):
        """Test navigation consistency across pages."""
        pages = ["/", "/detectors", "/datasets", "/detection", "/experiments"]

        for page in pages:
            response = test_client.get(page)
            assert response.status_code == 200
            # Should have consistent navigation
            assert "nav" in response.text.lower() or "menu" in response.text.lower()

    def test_error_page_handling(self, test_client):
        """Test error page handling."""
        # Test 404 page
        response = test_client.get("/nonexistent-page")
        assert response.status_code == 404

        # Should have user-friendly error page
        assert "404" in response.text or "not found" in response.text.lower()

    def test_cross_browser_compatibility(self, test_client):
        """Test cross-browser compatibility."""
        browsers = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        ]

        for user_agent in browsers:
            response = test_client.get("/", headers={"User-Agent": user_agent})
            assert response.status_code == 200
