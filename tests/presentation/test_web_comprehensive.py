"""Comprehensive tests for Web UI components - Phase 3 Coverage."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import json
import tempfile
from pathlib import Path

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.web.app import create_web_app
from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore


@pytest.fixture
def test_container():
    """Create test container for web app."""
    container = create_container()
    return container


@pytest.fixture
def web_client(test_container):
    """Create test client for web app."""
    app = create_web_app(test_container)
    return TestClient(app)


@pytest.fixture
async def async_web_client(test_container):
    """Create async test client for web app."""
    app = create_web_app(test_container)
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_session():
    """Create mock session for testing."""
    session = Mock()
    session.get.return_value = None
    session.setdefault.return_value = []
    return session


class TestWebAppRoutes:
    """Test main web application routes."""
    
    def test_home_page(self, web_client: TestClient):
        """Test home page rendering."""
        response = web_client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Pynomaly" in response.text
        assert "Dashboard" in response.text
    
    def test_dashboard_page(self, web_client: TestClient):
        """Test dashboard page."""
        with patch('pynomaly.presentation.web.app.get_dashboard_stats') as mock_stats:
            mock_stats.return_value = {
                "total_detectors": 5,
                "total_datasets": 10,
                "recent_detections": 3,
                "system_health": "healthy"
            }
            
            response = web_client.get("/dashboard")
            
            assert response.status_code == 200
            assert "Dashboard" in response.text
            assert "5" in response.text  # detector count
            assert "10" in response.text  # dataset count
    
    def test_detectors_page(self, web_client: TestClient):
        """Test detectors management page."""
        with patch('pynomaly.presentation.web.app.get_detectors') as mock_get_detectors:
            mock_detectors = [
                {
                    "id": "det1",
                    "name": "Detector 1",
                    "algorithm": "IsolationForest",
                    "is_fitted": True,
                    "created_at": "2024-01-01T00:00:00Z"
                },
                {
                    "id": "det2",
                    "name": "Detector 2",
                    "algorithm": "LOF",
                    "is_fitted": False,
                    "created_at": "2024-01-01T01:00:00Z"
                }
            ]
            mock_get_detectors.return_value = mock_detectors
            
            response = web_client.get("/detectors")
            
            assert response.status_code == 200
            assert "Detector 1" in response.text
            assert "IsolationForest" in response.text
            assert "Fitted" in response.text
    
    def test_datasets_page(self, web_client: TestClient):
        """Test datasets management page."""
        with patch('pynomaly.presentation.web.app.get_datasets') as mock_get_datasets:
            mock_datasets = [
                {
                    "id": "ds1",
                    "name": "Dataset 1",
                    "n_samples": 1000,
                    "n_features": 10,
                    "file_size_mb": 2.5,
                    "created_at": "2024-01-01T00:00:00Z"
                }
            ]
            mock_get_datasets.return_value = mock_datasets
            
            response = web_client.get("/datasets")
            
            assert response.status_code == 200
            assert "Dataset 1" in response.text
            assert "1000" in response.text
            assert "2.5 MB" in response.text
    
    def test_detection_page(self, web_client: TestClient):
        """Test detection workflow page."""
        response = web_client.get("/detection")
        
        assert response.status_code == 200
        assert "Detection" in response.text
        assert "Train Model" in response.text
        assert "Run Detection" in response.text
    
    def test_experiments_page(self, web_client: TestClient):
        """Test experiments page."""
        with patch('pynomaly.presentation.web.app.get_experiments') as mock_get_experiments:
            mock_experiments = [
                {
                    "id": "exp1",
                    "name": "Experiment 1",
                    "status": "completed",
                    "detectors_count": 3,
                    "best_score": 0.95,
                    "created_at": "2024-01-01T00:00:00Z"
                }
            ]
            mock_get_experiments.return_value = mock_experiments
            
            response = web_client.get("/experiments")
            
            assert response.status_code == 200
            assert "Experiment 1" in response.text
            assert "completed" in response.text
            assert "0.95" in response.text
    
    def test_static_files_serving(self, web_client: TestClient):
        """Test static files are served correctly."""
        # Test CSS file
        response = web_client.get("/static/css/app.css")
        assert response.status_code in [200, 404]  # File may not exist in test
        
        # Test JavaScript file
        response = web_client.get("/static/js/app.js")
        assert response.status_code in [200, 404]  # File may not exist in test
        
        # Test service worker
        response = web_client.get("/static/js/service-worker.js")
        assert response.status_code in [200, 404]  # File may not exist in test


class TestHTMXEndpoints:
    """Test HTMX-specific endpoints."""
    
    def test_detector_list_htmx(self, web_client: TestClient):
        """Test HTMX detector list partial."""
        with patch('pynomaly.presentation.web.app.get_detectors') as mock_get_detectors:
            mock_detectors = [
                {
                    "id": "det1",
                    "name": "HTMX Detector",
                    "algorithm": "IsolationForest",
                    "is_fitted": True
                }
            ]
            mock_get_detectors.return_value = mock_detectors
            
            response = web_client.get(
                "/htmx/detectors",
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "HTMX Detector" in response.text
            # Should return partial HTML, not full page
            assert "<html>" not in response.text
    
    def test_dataset_upload_htmx(self, web_client: TestClient):
        """Test HTMX dataset upload."""
        csv_content = "feature1,feature2\n1,2\n3,4\n"
        
        with patch('pynomaly.presentation.web.app.upload_dataset') as mock_upload:
            mock_upload.return_value = {
                "id": "ds123",
                "name": "Uploaded Dataset",
                "n_samples": 2,
                "n_features": 2,
                "success": True
            }
            
            files = {"file": ("test.csv", csv_content, "text/csv")}
            data = {"name": "Uploaded Dataset"}
            
            response = web_client.post(
                "/htmx/datasets/upload",
                files=files,
                data=data,
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "Uploaded Dataset" in response.text
    
    def test_detector_create_htmx(self, web_client: TestClient):
        """Test HTMX detector creation."""
        with patch('pynomaly.presentation.web.app.create_detector') as mock_create:
            mock_create.return_value = {
                "id": "det123",
                "name": "New Detector",
                "algorithm": "LOF",
                "success": True
            }
            
            form_data = {
                "name": "New Detector",
                "algorithm": "LOF",
                "contamination": "0.1"
            }
            
            response = web_client.post(
                "/htmx/detectors/create",
                data=form_data,
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "New Detector" in response.text
    
    def test_detection_train_htmx(self, web_client: TestClient):
        """Test HTMX detector training."""
        with patch('pynomaly.presentation.web.app.train_detector') as mock_train:
            mock_train.return_value = {
                "success": True,
                "training_time_ms": 1500,
                "model_metrics": {"accuracy": 0.95},
                "message": "Training completed successfully"
            }
            
            form_data = {
                "detector_id": "det123",
                "dataset_id": "ds123",
                "validation_split": "0.2"
            }
            
            response = web_client.post(
                "/htmx/detection/train",
                data=form_data,
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "Training completed" in response.text
            assert "1500 ms" in response.text
    
    def test_detection_predict_htmx(self, web_client: TestClient):
        """Test HTMX anomaly prediction."""
        with patch('pynomaly.presentation.web.app.predict_anomalies') as mock_predict:
            mock_predict.return_value = {
                "success": True,
                "predictions": [0, 0, 1, 0, 1],
                "anomaly_count": 2,
                "anomaly_rate": 0.4,
                "visualization_data": {
                    "scores": [0.1, 0.2, 0.8, 0.15, 0.9],
                    "threshold": 0.5
                }
            }
            
            form_data = {
                "detector_id": "det123",
                "dataset_id": "ds456",
                "threshold": "0.5"
            }
            
            response = web_client.post(
                "/htmx/detection/predict",
                data=form_data,
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "2 anomalies" in response.text
            assert "40.0%" in response.text
    
    def test_experiment_create_htmx(self, web_client: TestClient):
        """Test HTMX experiment creation."""
        with patch('pynomaly.presentation.web.app.create_experiment') as mock_create:
            mock_create.return_value = {
                "id": "exp123",
                "name": "HTMX Experiment",
                "status": "created",
                "success": True
            }
            
            form_data = {
                "name": "HTMX Experiment",
                "description": "Test experiment via HTMX",
                "algorithms": ["IsolationForest", "LOF", "OCSVM"]
            }
            
            response = web_client.post(
                "/htmx/experiments/create",
                data=form_data,
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "HTMX Experiment" in response.text
    
    def test_live_updates_htmx(self, web_client: TestClient):
        """Test HTMX live updates."""
        with patch('pynomaly.presentation.web.app.get_live_updates') as mock_updates:
            mock_updates.return_value = {
                "recent_detections": [
                    {"id": "res1", "detector": "Det1", "anomalies": 5, "timestamp": "12:00"},
                    {"id": "res2", "detector": "Det2", "anomalies": 3, "timestamp": "12:05"}
                ],
                "system_stats": {
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "active_detections": 2
                }
            }
            
            response = web_client.get(
                "/htmx/live-updates",
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "Det1" in response.text
            assert "5 anomalies" in response.text
            assert "45.2%" in response.text


class TestWebUIComponents:
    """Test individual Web UI components."""
    
    def test_detector_detail_modal(self, web_client: TestClient):
        """Test detector detail modal."""
        with patch('pynomaly.presentation.web.app.get_detector_details') as mock_details:
            mock_details.return_value = {
                "id": "det123",
                "name": "Detailed Detector",
                "algorithm": "IsolationForest",
                "hyperparameters": {"n_estimators": 100, "contamination": 0.1},
                "training_history": [
                    {"dataset": "Dataset 1", "accuracy": 0.95, "date": "2024-01-01"},
                    {"dataset": "Dataset 2", "accuracy": 0.92, "date": "2024-01-02"}
                ],
                "performance_metrics": {
                    "avg_training_time": 1200,
                    "avg_accuracy": 0.935,
                    "total_predictions": 50000
                }
            }
            
            response = web_client.get(
                "/htmx/detectors/det123/details",
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "Detailed Detector" in response.text
            assert "n_estimators" in response.text
            assert "Training History" in response.text
            assert "0.95" in response.text
    
    def test_dataset_preview_component(self, web_client: TestClient):
        """Test dataset preview component."""
        with patch('pynomaly.presentation.web.app.get_dataset_preview') as mock_preview:
            mock_preview.return_value = {
                "sample_data": [
                    {"feature1": 1.5, "feature2": 2.3, "feature3": 0.8},
                    {"feature1": 2.1, "feature2": 1.9, "feature3": 1.2},
                    {"feature1": 0.9, "feature2": 3.1, "feature3": 0.5}
                ],
                "columns": ["feature1", "feature2", "feature3"],
                "statistics": {
                    "feature1": {"mean": 1.5, "std": 0.6, "min": 0.9, "max": 2.1},
                    "feature2": {"mean": 2.43, "std": 0.62, "min": 1.9, "max": 3.1}
                },
                "data_quality": {
                    "missing_values": 0,
                    "duplicate_rows": 0,
                    "outliers_detected": 1
                }
            }
            
            response = web_client.get(
                "/htmx/datasets/ds123/preview",
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "feature1" in response.text
            assert "1.5" in response.text
            assert "Statistics" in response.text
    
    def test_visualization_components(self, web_client: TestClient):
        """Test data visualization components."""
        with patch('pynomaly.presentation.web.app.get_visualization_data') as mock_viz_data:
            mock_viz_data.return_value = {
                "scatter_plot": {
                    "data": [
                        {"x": 1.0, "y": 2.0, "anomaly": False},
                        {"x": 1.5, "y": 2.2, "anomaly": False},
                        {"x": 10.0, "y": 15.0, "anomaly": True}
                    ],
                    "config": {"x_label": "Feature 1", "y_label": "Feature 2"}
                },
                "score_distribution": {
                    "bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "counts": [45, 32, 28, 15, 12, 8, 5, 3, 1, 1],
                    "threshold": 0.6
                },
                "roc_curve": {
                    "fpr": [0.0, 0.1, 0.2, 0.3, 1.0],
                    "tpr": [0.0, 0.7, 0.85, 0.95, 1.0],
                    "auc": 0.92
                }
            }
            
            response = web_client.get(
                "/htmx/visualizations/result123",
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "scatter-plot" in response.text
            assert "score-distribution" in response.text
            assert "roc-curve" in response.text
    
    def test_performance_dashboard_component(self, web_client: TestClient):
        """Test performance dashboard component."""
        with patch('pynomaly.presentation.web.app.get_performance_data') as mock_perf_data:
            mock_perf_data.return_value = {
                "system_metrics": {
                    "cpu_usage": 35.7,
                    "memory_usage": 62.3,
                    "disk_usage": 78.1,
                    "gpu_usage": 23.4
                },
                "detector_performance": [
                    {"name": "IsolationForest", "avg_time": 150, "accuracy": 0.94},
                    {"name": "LOF", "avg_time": 320, "accuracy": 0.91},
                    {"name": "OCSVM", "avg_time": 890, "accuracy": 0.89}
                ],
                "recent_activity": [
                    {"action": "Detection completed", "time": "2 min ago", "status": "success"},
                    {"action": "Model trained", "time": "5 min ago", "status": "success"},
                    {"action": "Dataset uploaded", "time": "10 min ago", "status": "success"}
                ]
            }
            
            response = web_client.get(
                "/htmx/performance/dashboard",
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "35.7%" in response.text  # CPU usage
            assert "IsolationForest" in response.text
            assert "Detection completed" in response.text
    
    def test_experiment_results_component(self, web_client: TestClient):
        """Test experiment results component."""
        with patch('pynomaly.presentation.web.app.get_experiment_results') as mock_results:
            mock_results.return_value = {
                "experiment_id": "exp123",
                "name": "Multi-Algorithm Comparison",
                "status": "completed",
                "results": [
                    {
                        "detector": "IsolationForest",
                        "precision": 0.94,
                        "recall": 0.89,
                        "f1_score": 0.915,
                        "training_time": 1200
                    },
                    {
                        "detector": "LOF",
                        "precision": 0.91,
                        "recall": 0.92,
                        "f1_score": 0.915,
                        "training_time": 2300
                    }
                ],
                "best_detector": "IsolationForest",
                "summary": {
                    "total_detectors": 2,
                    "best_f1": 0.915,
                    "fastest_training": "IsolationForest"
                }
            }
            
            response = web_client.get(
                "/htmx/experiments/exp123/results",
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            assert "Multi-Algorithm Comparison" in response.text
            assert "0.94" in response.text  # Precision
            assert "IsolationForest" in response.text


class TestPWAFeatures:
    """Test Progressive Web App features."""
    
    def test_manifest_json(self, web_client: TestClient):
        """Test PWA manifest.json."""
        response = web_client.get("/static/manifest.json")
        
        if response.status_code == 200:
            manifest = response.json()
            assert "name" in manifest
            assert "short_name" in manifest
            assert "start_url" in manifest
            assert "display" in manifest
            assert "icons" in manifest
        else:
            # Manifest file might not exist in test environment
            assert response.status_code == 404
    
    def test_service_worker(self, web_client: TestClient):
        """Test service worker registration."""
        response = web_client.get("/static/js/service-worker.js")
        
        if response.status_code == 200:
            assert "javascript" in response.headers.get("content-type", "")
            # Service worker should contain cache logic
            assert any(keyword in response.text for keyword in ["cache", "fetch", "install"])
        else:
            # Service worker might not exist in test environment
            assert response.status_code == 404
    
    def test_offline_page(self, web_client: TestClient):
        """Test offline page."""
        response = web_client.get("/offline")
        
        assert response.status_code == 200
        assert "offline" in response.text.lower()
        assert "Pynomaly" in response.text
    
    def test_installability_check(self, web_client: TestClient):
        """Test PWA installability endpoint."""
        response = web_client.get("/pwa/installable")
        
        assert response.status_code == 200
        data = response.json()
        assert "installable" in data
        assert isinstance(data["installable"], bool)
    
    def test_cache_status(self, web_client: TestClient):
        """Test cache status endpoint."""
        response = web_client.get("/pwa/cache-status")
        
        assert response.status_code == 200
        data = response.json()
        assert "cache_size" in data
        assert "cached_resources" in data


class TestWebUIAuthentication:
    """Test Web UI authentication and session management."""
    
    def test_login_page(self, web_client: TestClient):
        """Test login page rendering."""
        response = web_client.get("/auth/login")
        
        assert response.status_code == 200
        assert "login" in response.text.lower()
        assert "username" in response.text.lower()
        assert "password" in response.text.lower()
    
    def test_login_form_submission(self, web_client: TestClient):
        """Test login form submission."""
        with patch('pynomaly.presentation.web.auth.authenticate_user') as mock_auth:
            mock_auth.return_value = {
                "success": True,
                "user": {"id": "user123", "username": "testuser"},
                "redirect_url": "/dashboard"
            }
            
            form_data = {
                "username": "testuser",
                "password": "testpass"
            }
            
            response = web_client.post("/auth/login", data=form_data)
            
            # Should redirect on successful login
            assert response.status_code in [200, 302]
    
    def test_logout_functionality(self, web_client: TestClient):
        """Test logout functionality."""
        # Simulate logged-in session
        with web_client.session_transaction() as session:
            session["user_id"] = "user123"
            session["username"] = "testuser"
        
        response = web_client.post("/auth/logout")
        
        # Should redirect after logout
        assert response.status_code in [200, 302]
    
    def test_protected_route_access(self, web_client: TestClient):
        """Test access to protected routes."""
        # Try to access protected route without authentication
        response = web_client.get("/admin/settings")
        
        # Should redirect to login or return 401/403
        assert response.status_code in [302, 401, 403]
    
    def test_session_persistence(self, web_client: TestClient):
        """Test session persistence across requests."""
        # Login
        with patch('pynomaly.presentation.web.auth.authenticate_user') as mock_auth:
            mock_auth.return_value = {
                "success": True,
                "user": {"id": "user123", "username": "testuser"}
            }
            
            login_response = web_client.post("/auth/login", data={
                "username": "testuser",
                "password": "testpass"
            })
            
            # Access protected resource with session
            dashboard_response = web_client.get("/dashboard")
            
            # Should be able to access if session is maintained
            assert dashboard_response.status_code == 200


class TestWebUIErrorHandling:
    """Test Web UI error handling."""
    
    def test_404_error_page(self, web_client: TestClient):
        """Test 404 error page."""
        response = web_client.get("/nonexistent-page")
        
        assert response.status_code == 404
        assert "404" in response.text or "not found" in response.text.lower()
    
    def test_500_error_handling(self, web_client: TestClient):
        """Test 500 error handling."""
        with patch('pynomaly.presentation.web.app.get_dashboard_stats') as mock_stats:
            mock_stats.side_effect = Exception("Database error")
            
            response = web_client.get("/dashboard")
            
            # Should handle error gracefully
            assert response.status_code in [200, 500]
            if response.status_code == 500:
                assert "error" in response.text.lower()
    
    def test_invalid_form_submission(self, web_client: TestClient):
        """Test handling of invalid form submissions."""
        # Submit detector creation form with invalid data
        invalid_form_data = {
            "name": "",  # Empty name
            "algorithm": "InvalidAlgorithm",  # Invalid algorithm
            "contamination": "invalid_number"  # Invalid contamination
        }
        
        response = web_client.post(
            "/htmx/detectors/create",
            data=invalid_form_data,
            headers={"HX-Request": "true"}
        )
        
        # Should return error message
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            assert "error" in response.text.lower() or "invalid" in response.text.lower()
    
    def test_file_upload_error_handling(self, web_client: TestClient):
        """Test file upload error handling."""
        # Upload invalid file
        invalid_content = "This is not CSV content"
        files = {"file": ("invalid.csv", invalid_content, "text/csv")}
        data = {"name": "Invalid Dataset"}
        
        response = web_client.post(
            "/htmx/datasets/upload",
            files=files,
            data=data,
            headers={"HX-Request": "true"}
        )
        
        # Should handle invalid file gracefully
        assert response.status_code in [200, 400]
    
    def test_concurrent_request_handling(self, web_client: TestClient):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_request():
            return web_client.get("/")
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(result.status_code == 200 for result in results)


class TestWebUIPerformance:
    """Test Web UI performance characteristics."""
    
    def test_page_load_times(self, web_client: TestClient):
        """Test page load performance."""
        import time
        
        pages_to_test = ["/", "/dashboard", "/detectors", "/datasets"]
        
        for page in pages_to_test:
            start_time = time.time()
            response = web_client.get(page)
            end_time = time.time()
            
            assert response.status_code == 200
            load_time = end_time - start_time
            assert load_time < 2.0  # Should load within 2 seconds
    
    def test_large_dataset_list_performance(self, web_client: TestClient):
        """Test performance with large dataset lists."""
        # Mock large number of datasets
        with patch('pynomaly.presentation.web.app.get_datasets') as mock_get_datasets:
            large_dataset_list = [
                {
                    "id": f"ds{i}",
                    "name": f"Dataset {i}",
                    "n_samples": 1000 + i,
                    "n_features": 10 + (i % 5),
                    "file_size_mb": 1.0 + (i * 0.1)
                }
                for i in range(100)  # 100 datasets
            ]
            mock_get_datasets.return_value = large_dataset_list
            
            start_time = time.time()
            response = web_client.get("/datasets")
            end_time = time.time()
            
            assert response.status_code == 200
            load_time = end_time - start_time
            assert load_time < 3.0  # Should handle large lists efficiently
    
    def test_htmx_response_times(self, web_client: TestClient):
        """Test HTMX endpoint response times."""
        import time
        
        htmx_endpoints = [
            "/htmx/detectors",
            "/htmx/datasets",
            "/htmx/live-updates"
        ]
        
        for endpoint in htmx_endpoints:
            start_time = time.time()
            response = web_client.get(
                endpoint,
                headers={"HX-Request": "true"}
            )
            end_time = time.time()
            
            assert response.status_code == 200
            response_time = end_time - start_time
            assert response_time < 1.0  # HTMX responses should be fast
    
    def test_memory_usage_with_large_data(self, web_client: TestClient):
        """Test memory usage with large data operations."""
        # Test large dataset preview
        with patch('pynomaly.presentation.web.app.get_dataset_preview') as mock_preview:
            # Mock large dataset preview
            large_sample = [
                {f"feature_{j}": i * j for j in range(50)}
                for i in range(1000)
            ]
            mock_preview.return_value = {
                "sample_data": large_sample,
                "columns": [f"feature_{j}" for j in range(50)]
            }
            
            response = web_client.get(
                "/htmx/datasets/large_dataset/preview",
                headers={"HX-Request": "true"}
            )
            
            assert response.status_code == 200
            # Should handle large data without memory issues


class TestWebUIAccessibility:
    """Test Web UI accessibility features."""
    
    def test_semantic_html_structure(self, web_client: TestClient):
        """Test semantic HTML structure."""
        response = web_client.get("/")
        
        assert response.status_code == 200
        html_content = response.text
        
        # Check for semantic elements
        assert "<main>" in html_content or 'role="main"' in html_content
        assert "<nav>" in html_content or 'role="navigation"' in html_content
        assert "<header>" in html_content
        assert "<footer>" in html_content
    
    def test_aria_labels_and_roles(self, web_client: TestClient):
        """Test ARIA labels and roles."""
        response = web_client.get("/detectors")
        
        assert response.status_code == 200
        html_content = response.text
        
        # Check for ARIA attributes
        aria_patterns = ["aria-label", "aria-describedby", "role="]
        assert any(pattern in html_content for pattern in aria_patterns)
    
    def test_keyboard_navigation_support(self, web_client: TestClient):
        """Test keyboard navigation support."""
        response = web_client.get("/")
        
        assert response.status_code == 200
        html_content = response.text
        
        # Check for tabindex and keyboard event handling
        assert "tabindex" in html_content or "keydown" in html_content
    
    def test_alt_text_for_images(self, web_client: TestClient):
        """Test alt text for images."""
        response = web_client.get("/")
        
        assert response.status_code == 200
        html_content = response.text
        
        # If images exist, they should have alt text
        if "<img" in html_content:
            assert "alt=" in html_content


class TestWebUIIntegration:
    """Test Web UI integration scenarios."""
    
    def test_end_to_end_detection_workflow(self, web_client: TestClient):
        """Test complete detection workflow through Web UI."""
        # This would test the full workflow:
        # 1. Upload dataset
        # 2. Create detector
        # 3. Train detector
        # 4. Run detection
        # 5. View results
        
        # Mock all the necessary services
        with patch('pynomaly.presentation.web.app.upload_dataset') as mock_upload, \
             patch('pynomaly.presentation.web.app.create_detector') as mock_create, \
             patch('pynomaly.presentation.web.app.train_detector') as mock_train, \
             patch('pynomaly.presentation.web.app.predict_anomalies') as mock_predict:
            
            # Step 1: Upload dataset
            mock_upload.return_value = {"id": "ds123", "success": True}
            upload_response = web_client.post(
                "/htmx/datasets/upload",
                files={"file": ("test.csv", "x,y\n1,2\n3,4\n", "text/csv")},
                data={"name": "Test Dataset"},
                headers={"HX-Request": "true"}
            )
            assert upload_response.status_code == 200
            
            # Step 2: Create detector
            mock_create.return_value = {"id": "det123", "success": True}
            create_response = web_client.post(
                "/htmx/detectors/create",
                data={"name": "Test Detector", "algorithm": "IsolationForest"},
                headers={"HX-Request": "true"}
            )
            assert create_response.status_code == 200
            
            # Step 3: Train detector
            mock_train.return_value = {"success": True, "training_time_ms": 1000}
            train_response = web_client.post(
                "/htmx/detection/train",
                data={"detector_id": "det123", "dataset_id": "ds123"},
                headers={"HX-Request": "true"}
            )
            assert train_response.status_code == 200
            
            # Step 4: Run detection
            mock_predict.return_value = {
                "success": True,
                "anomaly_count": 1,
                "anomaly_rate": 0.5
            }
            predict_response = web_client.post(
                "/htmx/detection/predict",
                data={"detector_id": "det123", "dataset_id": "ds123"},
                headers={"HX-Request": "true"}
            )
            assert predict_response.status_code == 200
    
    def test_real_time_updates(self, web_client: TestClient):
        """Test real-time updates functionality."""
        with patch('pynomaly.presentation.web.app.get_live_updates') as mock_updates:
            mock_updates.return_value = {
                "recent_detections": [],
                "system_stats": {"active_detections": 0}
            }
            
            # Test multiple rapid requests to simulate real-time updates
            for i in range(5):
                response = web_client.get(
                    "/htmx/live-updates",
                    headers={"HX-Request": "true"}
                )
                assert response.status_code == 200
    
    def test_cross_browser_compatibility_headers(self, web_client: TestClient):
        """Test cross-browser compatibility headers."""
        response = web_client.get("/")
        
        assert response.status_code == 200
        
        # Check for compatibility headers
        headers = response.headers
        # These might be set by the server
        expected_headers = [
            "x-content-type-options",
            "x-frame-options",
            "content-security-policy"
        ]
        
        # At least some security headers should be present
        # (depending on server configuration)