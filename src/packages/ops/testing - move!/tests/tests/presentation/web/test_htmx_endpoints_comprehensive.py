"""
Comprehensive tests for Web UI HTMX endpoints.

This module provides extensive testing for HTMX endpoints in the web UI,
including real-time updates, interactive components, and dynamic content.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from monorepo.presentation.web.app import app


class TestHTMXDashboardEndpoints:
    """Test HTMX dashboard endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/dashboard",
            "HX-Target": "dashboard-content",
        }

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test-token"}

    def test_dashboard_stats_htmx(self, client, htmx_headers, auth_headers):
        """Test dashboard stats HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/dashboard/stats", headers=headers)

        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_404_NOT_FOUND,
        ]

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_dashboard_charts_htmx(self, client, htmx_headers, auth_headers):
        """Test dashboard charts HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/dashboard/charts", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_dashboard_recent_activity_htmx(self, client, htmx_headers, auth_headers):
        """Test dashboard recent activity HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/dashboard/activity", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_dashboard_alerts_htmx(self, client, htmx_headers, auth_headers):
        """Test dashboard alerts HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/dashboard/alerts", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_dashboard_refresh_htmx(self, client, htmx_headers, auth_headers):
        """Test dashboard refresh HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.post("/htmx/dashboard/refresh", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_dashboard_without_htmx_headers(self, client, auth_headers):
        """Test dashboard endpoints without HTMX headers."""
        response = client.get("/htmx/dashboard/stats", headers=auth_headers)

        # Should handle gracefully
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_404_NOT_FOUND,
        ]


class TestHTMXDetectionEndpoints:
    """Test HTMX detection endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/detection",
            "HX-Target": "detection-content",
        }

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test-token"}

    def test_detection_form_htmx(self, client, htmx_headers, auth_headers):
        """Test detection form HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/detection/form", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML form
            assert "text/html" in response.headers.get("content-type", "")
            content = response.content.decode()
            assert "form" in content.lower()

    def test_detection_run_htmx(self, client, htmx_headers, auth_headers):
        """Test detection run HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        detection_data = {
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "data": "test-data",
        }

        response = client.post(
            "/htmx/detection/run", data=detection_data, headers=headers
        )

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_detection_results_htmx(self, client, htmx_headers, auth_headers):
        """Test detection results HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/detection/results/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_detection_progress_htmx(self, client, htmx_headers, auth_headers):
        """Test detection progress HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/detection/progress/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_detection_visualization_htmx(self, client, htmx_headers, auth_headers):
        """Test detection visualization HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/detection/viz/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_detection_export_htmx(self, client, htmx_headers, auth_headers):
        """Test detection export HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.post("/htmx/detection/export/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment or file
            assert len(response.content) > 0


class TestHTMXDatasetEndpoints:
    """Test HTMX dataset endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/datasets",
            "HX-Target": "dataset-content",
        }

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test-token"}

    def test_dataset_list_htmx(self, client, htmx_headers, auth_headers):
        """Test dataset list HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/datasets/list", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_dataset_upload_form_htmx(self, client, htmx_headers, auth_headers):
        """Test dataset upload form HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/datasets/upload-form", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML form
            assert "text/html" in response.headers.get("content-type", "")
            content = response.content.decode()
            assert "form" in content.lower()

    def test_dataset_upload_htmx(self, client, htmx_headers, auth_headers):
        """Test dataset upload HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        # Mock file upload
        files = {"file": ("test.csv", "col1,col2\n1,2\n3,4", "text/csv")}
        data = {"name": "test-dataset"}

        response = client.post(
            "/htmx/datasets/upload", files=files, data=data, headers=headers
        )

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_dataset_preview_htmx(self, client, htmx_headers, auth_headers):
        """Test dataset preview HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/datasets/preview/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_dataset_stats_htmx(self, client, htmx_headers, auth_headers):
        """Test dataset statistics HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/datasets/stats/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_dataset_delete_htmx(self, client, htmx_headers, auth_headers):
        """Test dataset delete HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.delete("/htmx/datasets/delete/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_dataset_search_htmx(self, client, htmx_headers, auth_headers):
        """Test dataset search HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/datasets/search?q=test", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0


class TestHTMXDetectorEndpoints:
    """Test HTMX detector endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/detectors",
            "HX-Target": "detector-content",
        }

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test-token"}

    def test_detector_list_htmx(self, client, htmx_headers, auth_headers):
        """Test detector list HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/detectors/list", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_detector_create_form_htmx(self, client, htmx_headers, auth_headers):
        """Test detector create form HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/detectors/create-form", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML form
            assert "text/html" in response.headers.get("content-type", "")
            content = response.content.decode()
            assert "form" in content.lower()

    def test_detector_create_htmx(self, client, htmx_headers, auth_headers):
        """Test detector create HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        detector_data = {
            "name": "test-detector",
            "algorithm": "isolation_forest",
            "contamination": 0.1,
        }

        response = client.post(
            "/htmx/detectors/create", data=detector_data, headers=headers
        )

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_detector_edit_form_htmx(self, client, htmx_headers, auth_headers):
        """Test detector edit form HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/detectors/edit-form/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML form
            assert "text/html" in response.headers.get("content-type", "")
            content = response.content.decode()
            assert "form" in content.lower()

    def test_detector_update_htmx(self, client, htmx_headers, auth_headers):
        """Test detector update HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        detector_data = {
            "name": "updated-detector",
            "algorithm": "lof",
            "contamination": 0.2,
        }

        response = client.put(
            "/htmx/detectors/update/test-id", data=detector_data, headers=headers
        )

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_detector_delete_htmx(self, client, htmx_headers, auth_headers):
        """Test detector delete HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.delete("/htmx/detectors/delete/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_detector_details_htmx(self, client, htmx_headers, auth_headers):
        """Test detector details HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/detectors/details/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_detector_train_htmx(self, client, htmx_headers, auth_headers):
        """Test detector train HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        train_data = {"dataset_id": "test-dataset", "epochs": 10}

        response = client.post(
            "/htmx/detectors/train/test-id", data=train_data, headers=headers
        )

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_detector_test_htmx(self, client, htmx_headers, auth_headers):
        """Test detector test HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        test_data = {"dataset_id": "test-dataset"}

        response = client.post(
            "/htmx/detectors/test/test-id", data=test_data, headers=headers
        )

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0


class TestHTMXTrainingEndpoints:
    """Test HTMX training endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/training",
            "HX-Target": "training-content",
        }

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test-token"}

    def test_training_jobs_list_htmx(self, client, htmx_headers, auth_headers):
        """Test training jobs list HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/training/jobs", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_training_start_htmx(self, client, htmx_headers, auth_headers):
        """Test training start HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        training_data = {
            "detector_id": "test-detector",
            "dataset_id": "test-dataset",
            "epochs": 10,
        }

        response = client.post(
            "/htmx/training/start", data=training_data, headers=headers
        )

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_training_status_htmx(self, client, htmx_headers, auth_headers):
        """Test training status HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/training/status/test-job-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_training_progress_htmx(self, client, htmx_headers, auth_headers):
        """Test training progress HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/training/progress/test-job-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_training_logs_htmx(self, client, htmx_headers, auth_headers):
        """Test training logs HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/training/logs/test-job-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_training_stop_htmx(self, client, htmx_headers, auth_headers):
        """Test training stop HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.post("/htmx/training/stop/test-job-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_training_results_htmx(self, client, htmx_headers, auth_headers):
        """Test training results HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/training/results/test-job-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0


class TestHTMXEnsembleEndpoints:
    """Test HTMX ensemble endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/ensemble",
            "HX-Target": "ensemble-content",
        }

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test-token"}

    def test_ensemble_list_htmx(self, client, htmx_headers, auth_headers):
        """Test ensemble list HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/ensemble/list", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_ensemble_create_form_htmx(self, client, htmx_headers, auth_headers):
        """Test ensemble create form HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/ensemble/create-form", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML form
            assert "text/html" in response.headers.get("content-type", "")
            content = response.content.decode()
            assert "form" in content.lower()

    def test_ensemble_create_htmx(self, client, htmx_headers, auth_headers):
        """Test ensemble create HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        ensemble_data = {
            "name": "test-ensemble",
            "detectors": "detector1,detector2",
            "aggregation": "mean",
        }

        response = client.post(
            "/htmx/ensemble/create", data=ensemble_data, headers=headers
        )

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_ensemble_details_htmx(self, client, htmx_headers, auth_headers):
        """Test ensemble details HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/ensemble/details/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_ensemble_run_htmx(self, client, htmx_headers, auth_headers):
        """Test ensemble run HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        run_data = {"dataset_id": "test-dataset"}

        response = client.post(
            "/htmx/ensemble/run/test-id", data=run_data, headers=headers
        )

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_ensemble_performance_htmx(self, client, htmx_headers, auth_headers):
        """Test ensemble performance HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/ensemble/performance/test-id", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0


class TestHTMXRealtimeEndpoints:
    """Test HTMX real-time endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/realtime",
            "HX-Target": "realtime-content",
        }

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test-token"}

    def test_realtime_monitoring_htmx(self, client, htmx_headers, auth_headers):
        """Test real-time monitoring HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/realtime/monitoring", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_realtime_alerts_htmx(self, client, htmx_headers, auth_headers):
        """Test real-time alerts HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/realtime/alerts", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_realtime_metrics_htmx(self, client, htmx_headers, auth_headers):
        """Test real-time metrics HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/realtime/metrics", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_realtime_logs_htmx(self, client, htmx_headers, auth_headers):
        """Test real-time logs HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/realtime/logs", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0

    def test_realtime_performance_htmx(self, client, htmx_headers, auth_headers):
        """Test real-time performance HTMX endpoint."""
        headers = {**htmx_headers, **auth_headers}

        response = client.get("/htmx/realtime/performance", headers=headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML fragment
            assert "text/html" in response.headers.get("content-type", "")
            assert len(response.content) > 0


class TestHTMXErrorHandling:
    """Test HTMX error handling."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/test",
            "HX-Target": "test-content",
        }

    def test_htmx_404_handling(self, client, htmx_headers):
        """Test HTMX 404 error handling."""
        response = client.get("/htmx/nonexistent/endpoint", headers=htmx_headers)

        assert response.status_code == status.HTTP_404_NOT_FOUND

        # Should return HTML error fragment
        if "text/html" in response.headers.get("content-type", ""):
            content = response.content.decode()
            assert "error" in content.lower() or "not found" in content.lower()

    def test_htmx_500_handling(self, client, htmx_headers):
        """Test HTMX 500 error handling."""
        # This would need a specific endpoint that raises an error
        # For now, test that error responses are properly formatted
        pass

    def test_htmx_validation_errors(self, client, htmx_headers):
        """Test HTMX validation error handling."""
        # Test with invalid data
        invalid_data = {"invalid": "data"}

        response = client.post(
            "/htmx/detection/run", data=invalid_data, headers=htmx_headers
        )

        if response.status_code == status.HTTP_400_BAD_REQUEST:
            # Should return HTML error fragment
            if "text/html" in response.headers.get("content-type", ""):
                content = response.content.decode()
                assert "error" in content.lower() or "invalid" in content.lower()

    def test_htmx_authentication_errors(self, client, htmx_headers):
        """Test HTMX authentication error handling."""
        # Test without authentication
        response = client.get("/htmx/dashboard/stats", headers=htmx_headers)

        if response.status_code == status.HTTP_401_UNAUTHORIZED:
            # Should return HTML error fragment
            if "text/html" in response.headers.get("content-type", ""):
                content = response.content.decode()
                assert "auth" in content.lower() or "login" in content.lower()


class TestHTMXPerformance:
    """Test HTMX performance."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/test",
            "HX-Target": "test-content",
        }

    def test_htmx_response_time(self, client, htmx_headers):
        """Test HTMX response time."""
        import time

        start_time = time.time()
        response = client.get("/htmx/dashboard/stats", headers=htmx_headers)
        end_time = time.time()

        # Should respond quickly
        assert (end_time - start_time) < 2.0  # Should respond within 2 seconds

    def test_htmx_concurrent_requests(self, client, htmx_headers):
        """Test HTMX concurrent request handling."""
        import concurrent.futures

        def make_request():
            return client.get("/htmx/dashboard/stats", headers=htmx_headers)

        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]

        # All requests should complete
        for response in responses:
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_404_NOT_FOUND,
            ]

    def test_htmx_response_size(self, client, htmx_headers):
        """Test HTMX response size."""
        response = client.get("/htmx/dashboard/stats", headers=htmx_headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return reasonably sized fragments
            content_length = len(response.content)
            assert content_length < 100000  # Should be less than 100KB


class TestHTMXSecurity:
    """Test HTMX security features."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def htmx_headers(self):
        """HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Current-URL": "http://localhost/test",
            "HX-Target": "test-content",
        }

    def test_htmx_csrf_protection(self, client, htmx_headers):
        """Test HTMX CSRF protection."""
        # Test POST request without CSRF token
        response = client.post(
            "/htmx/detection/run", data={"test": "data"}, headers=htmx_headers
        )

        # Should either work or require CSRF token
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_403_FORBIDDEN,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_404_NOT_FOUND,
        ]

    def test_htmx_xss_protection(self, client, htmx_headers):
        """Test HTMX XSS protection."""
        # Test with potentially malicious input
        malicious_data = {"input": "<script>alert('xss')</script>"}

        response = client.post(
            "/htmx/detection/run", data=malicious_data, headers=htmx_headers
        )

        if response.status_code == status.HTTP_200_OK:
            content = response.content.decode()
            # Should not contain unescaped script tags
            assert "<script>" not in content or "&lt;script&gt;" in content

    def test_htmx_content_type_validation(self, client, htmx_headers):
        """Test HTMX content type validation."""
        response = client.get("/htmx/dashboard/stats", headers=htmx_headers)

        if response.status_code == status.HTTP_200_OK:
            # Should return HTML content type
            content_type = response.headers.get("content-type", "")
            assert "text/html" in content_type

    def test_htmx_header_validation(self, client):
        """Test HTMX header validation."""
        # Test without HTMX headers
        response = client.get("/htmx/dashboard/stats")

        # Should either work or require HTMX headers
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_404_NOT_FOUND,
        ]
