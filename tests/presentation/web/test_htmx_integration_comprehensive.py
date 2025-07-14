"""
Comprehensive test suite for HTMX integration in Pynomaly web UI.

This module provides extensive testing coverage for HTMX functionality,
including dynamic content updates, form submissions, and real-time features.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Import the web application
from pynomaly.presentation.web.app import create_web_app


class TestHTMXDetectionResults:
    """Test HTMX detection results functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.app = create_web_app()
        self.client = TestClient(self.app)

    def test_htmx_detection_results_success(self):
        """Test successful HTMX detection results."""
        # Mock detection service
        with patch(
            "pynomaly.presentation.web.dependencies.get_detection_service"
        ) as mock_service:
            mock_service.return_value.detect_anomalies.return_value = {
                "anomaly_scores": [0.1, 0.9, 0.2, 0.8],
                "anomaly_labels": [0, 1, 0, 1],
                "execution_time": 0.123,
                "model_info": {"name": "isolation_forest", "version": "1.0"},
            }

            # Make HTMX request
            response = self.client.post(
                "/htmx/detection/results",
                json={
                    "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                    "algorithm": "isolation_forest",
                },
                headers={"HX-Request": "true", "HX-Target": "detection-results"},
            )

            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

            # Check response content
            html_content = response.text
            assert "detection-results" in html_content
            assert "Anomaly Scores" in html_content
            assert "0.9" in html_content  # High anomaly score
            assert "0.123" in html_content  # Execution time

    def test_htmx_detection_results_with_visualization(self):
        """Test HTMX detection results with visualization."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_detection_service"
        ) as mock_service:
            mock_service.return_value.detect_anomalies.return_value = {
                "anomaly_scores": [0.1, 0.9, 0.2],
                "anomaly_labels": [0, 1, 0],
                "execution_time": 0.456,
                "visualization_data": {
                    "scatter_plot": {"x": [1, 2, 3], "y": [1, 2, 3]},
                    "histogram": {"bins": [0, 0.5, 1.0], "counts": [2, 0, 1]},
                },
            }

            response = self.client.post(
                "/htmx/detection/results",
                json={
                    "data": [[1, 2], [3, 4], [5, 6]],
                    "algorithm": "isolation_forest",
                    "include_visualization": True,
                },
                headers={"HX-Request": "true"},
            )

            assert response.status_code == 200
            html_content = response.text
            assert "visualization-container" in html_content
            assert "scatter-plot" in html_content
            assert "histogram-plot" in html_content

    def test_htmx_detection_results_error_handling(self):
        """Test HTMX detection results error handling."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_detection_service"
        ) as mock_service:
            mock_service.return_value.detect_anomalies.side_effect = Exception(
                "Detection failed"
            )

            response = self.client.post(
                "/htmx/detection/results",
                json={"data": [[1, 2], [3, 4]], "algorithm": "invalid_algorithm"},
                headers={"HX-Request": "true"},
            )

            assert response.status_code == 200
            html_content = response.text
            assert "alert-error" in html_content
            assert "Detection failed" in html_content

    def test_htmx_detection_results_streaming(self):
        """Test HTMX detection results with streaming."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_detection_service"
        ) as mock_service:
            mock_service.return_value.detect_anomalies_stream.return_value = [
                {"progress": 25, "partial_results": [0.1, 0.2]},
                {"progress": 50, "partial_results": [0.1, 0.2, 0.9]},
                {
                    "progress": 100,
                    "final_results": {"anomaly_scores": [0.1, 0.2, 0.9, 0.1]},
                },
            ]

            response = self.client.post(
                "/htmx/detection/results",
                json={
                    "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                    "algorithm": "isolation_forest",
                    "streaming": True,
                },
                headers={"HX-Request": "true"},
            )

            assert response.status_code == 200
            html_content = response.text
            assert "streaming-results" in html_content
            assert "progress-bar" in html_content


class TestHTMXTrainingProgress:
    """Test HTMX training progress functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.app = create_web_app()
        self.client = TestClient(self.app)

    def test_htmx_training_progress_running(self):
        """Test HTMX training progress for running job."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_training_service"
        ) as mock_service:
            mock_service.return_value.get_training_status.return_value = {
                "job_id": "train-123",
                "status": "running",
                "progress": 65,
                "current_epoch": 13,
                "total_epochs": 20,
                "current_loss": 0.045,
                "estimated_time_remaining": 120,
            }

            response = self.client.get(
                "/htmx/training/progress/train-123", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "progress-bar" in html_content
            assert "65%" in html_content
            assert "13/20" in html_content  # Current epoch
            assert "0.045" in html_content  # Current loss
            assert "2 minutes" in html_content  # Estimated time

    def test_htmx_training_progress_completed(self):
        """Test HTMX training progress for completed job."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_training_service"
        ) as mock_service:
            mock_service.return_value.get_training_status.return_value = {
                "job_id": "train-123",
                "status": "completed",
                "progress": 100,
                "final_metrics": {
                    "accuracy": 0.95,
                    "precision": 0.93,
                    "recall": 0.91,
                    "f1_score": 0.92,
                },
                "model_id": "model-456",
                "training_time": 300,
            }

            response = self.client.get(
                "/htmx/training/progress/train-123", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "training-completed" in html_content
            assert "100%" in html_content
            assert "0.95" in html_content  # Accuracy
            assert "model-456" in html_content
            assert "5 minutes" in html_content  # Training time

    def test_htmx_training_progress_failed(self):
        """Test HTMX training progress for failed job."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_training_service"
        ) as mock_service:
            mock_service.return_value.get_training_status.return_value = {
                "job_id": "train-123",
                "status": "failed",
                "progress": 45,
                "error_message": "Insufficient memory for training",
                "last_checkpoint": "epoch-8",
            }

            response = self.client.get(
                "/htmx/training/progress/train-123", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "training-failed" in html_content
            assert "Insufficient memory" in html_content
            assert "epoch-8" in html_content

    def test_htmx_training_progress_with_metrics_chart(self):
        """Test HTMX training progress with metrics chart."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_training_service"
        ) as mock_service:
            mock_service.return_value.get_training_status.return_value = {
                "job_id": "train-123",
                "status": "running",
                "progress": 80,
                "metrics_history": {
                    "epochs": [1, 2, 3, 4, 5],
                    "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2],
                    "val_loss": [0.9, 0.7, 0.5, 0.4, 0.3],
                    "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
                },
            }

            response = self.client.get(
                "/htmx/training/progress/train-123", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "metrics-chart" in html_content
            assert "train_loss" in html_content
            assert "val_loss" in html_content
            assert "accuracy" in html_content


class TestHTMXDatasetPreview:
    """Test HTMX dataset preview functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.app = create_web_app()
        self.client = TestClient(self.app)

    def test_htmx_dataset_preview_success(self):
        """Test successful HTMX dataset preview."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_dataset_service"
        ) as mock_service:
            mock_service.return_value.get_dataset_preview.return_value = {
                "dataset_id": "dataset-123",
                "name": "test_dataset",
                "size": 1000,
                "features": 5,
                "preview_data": {
                    "columns": [
                        "feature1",
                        "feature2",
                        "feature3",
                        "feature4",
                        "feature5",
                    ],
                    "data": [
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [1.1, 2.1, 3.1, 4.1, 5.1],
                        [1.2, 2.2, 3.2, 4.2, 5.2],
                    ],
                },
                "statistics": {
                    "feature1": {"mean": 1.1, "std": 0.1, "min": 1.0, "max": 1.2},
                    "feature2": {"mean": 2.1, "std": 0.1, "min": 2.0, "max": 2.2},
                },
            }

            response = self.client.get(
                "/htmx/dataset/preview/dataset-123", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "dataset-preview" in html_content
            assert "test_dataset" in html_content
            assert "1000" in html_content  # Size
            assert "5" in html_content  # Features
            assert "feature1" in html_content  # Column names
            assert "1.1" in html_content  # Mean value

    def test_htmx_dataset_preview_with_pagination(self):
        """Test HTMX dataset preview with pagination."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_dataset_service"
        ) as mock_service:
            mock_service.return_value.get_dataset_preview.return_value = {
                "dataset_id": "dataset-123",
                "name": "large_dataset",
                "size": 10000,
                "features": 3,
                "preview_data": {
                    "columns": ["col1", "col2", "col3"],
                    "data": [[i, i + 1, i + 2] for i in range(50)],  # 50 rows
                },
                "pagination": {
                    "page": 1,
                    "per_page": 50,
                    "total_pages": 200,
                    "total_rows": 10000,
                },
            }

            response = self.client.get(
                "/htmx/dataset/preview/dataset-123?page=1&per_page=50",
                headers={"HX-Request": "true"},
            )

            assert response.status_code == 200
            html_content = response.text
            assert "pagination-controls" in html_content
            assert "Page 1 of 200" in html_content
            assert "10000 total rows" in html_content

    def test_htmx_dataset_preview_with_filtering(self):
        """Test HTMX dataset preview with filtering."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_dataset_service"
        ) as mock_service:
            mock_service.return_value.get_dataset_preview.return_value = {
                "dataset_id": "dataset-123",
                "name": "filtered_dataset",
                "size": 500,
                "features": 4,
                "preview_data": {
                    "columns": ["feature1", "feature2", "feature3", "feature4"],
                    "data": [[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1]],
                },
                "filter_info": {
                    "column": "feature1",
                    "operator": ">",
                    "value": 1.0,
                    "rows_matched": 500,
                },
            }

            response = self.client.get(
                "/htmx/dataset/preview/dataset-123?filter_column=feature1&filter_operator=>&filter_value=1.0",
                headers={"HX-Request": "true"},
            )

            assert response.status_code == 200
            html_content = response.text
            assert "filter-info" in html_content
            assert "feature1 > 1.0" in html_content
            assert "500 rows matched" in html_content


class TestHTMXModelMetrics:
    """Test HTMX model metrics functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.app = create_web_app()
        self.client = TestClient(self.app)

    def test_htmx_model_metrics_success(self):
        """Test successful HTMX model metrics."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_model_service"
        ) as mock_service:
            mock_service.return_value.get_model_metrics.return_value = {
                "model_id": "model-123",
                "name": "test_model",
                "algorithm": "isolation_forest",
                "metrics": {
                    "accuracy": 0.95,
                    "precision": 0.93,
                    "recall": 0.91,
                    "f1_score": 0.92,
                    "roc_auc": 0.94,
                },
                "confusion_matrix": [[450, 50], [30, 470]],
                "feature_importance": {
                    "feature1": 0.4,
                    "feature2": 0.3,
                    "feature3": 0.2,
                    "feature4": 0.1,
                },
            }

            response = self.client.get(
                "/htmx/model/metrics/model-123", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "model-metrics" in html_content
            assert "test_model" in html_content
            assert "0.95" in html_content  # Accuracy
            assert "0.93" in html_content  # Precision
            assert "confusion-matrix" in html_content
            assert "feature-importance" in html_content

    def test_htmx_model_metrics_with_charts(self):
        """Test HTMX model metrics with charts."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_model_service"
        ) as mock_service:
            mock_service.return_value.get_model_metrics.return_value = {
                "model_id": "model-123",
                "name": "chart_model",
                "metrics": {
                    "accuracy": 0.88,
                    "precision": 0.85,
                    "recall": 0.90,
                    "f1_score": 0.87,
                },
                "roc_curve": {
                    "fpr": [0.0, 0.1, 0.2, 0.3, 1.0],
                    "tpr": [0.0, 0.7, 0.8, 0.9, 1.0],
                    "auc": 0.88,
                },
                "precision_recall_curve": {
                    "precision": [1.0, 0.9, 0.8, 0.7, 0.6],
                    "recall": [0.0, 0.2, 0.4, 0.6, 0.8],
                    "average_precision": 0.82,
                },
            }

            response = self.client.get(
                "/htmx/model/metrics/model-123?include_charts=true",
                headers={"HX-Request": "true"},
            )

            assert response.status_code == 200
            html_content = response.text
            assert "roc-curve-chart" in html_content
            assert "precision-recall-chart" in html_content
            assert "AUC: 0.88" in html_content
            assert "AP: 0.82" in html_content

    def test_htmx_model_metrics_comparison(self):
        """Test HTMX model metrics comparison."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_model_service"
        ) as mock_service:
            mock_service.return_value.compare_models.return_value = {
                "models": [
                    {"model_id": "model-1", "name": "Model A", "accuracy": 0.85},
                    {"model_id": "model-2", "name": "Model B", "accuracy": 0.90},
                    {"model_id": "model-3", "name": "Model C", "accuracy": 0.88},
                ],
                "comparison_metrics": {
                    "best_accuracy": "model-2",
                    "best_precision": "model-2",
                    "best_recall": "model-1",
                    "best_f1": "model-2",
                },
            }

            response = self.client.get(
                "/htmx/model/metrics/comparison?models=model-1,model-2,model-3",
                headers={"HX-Request": "true"},
            )

            assert response.status_code == 200
            html_content = response.text
            assert "model-comparison" in html_content
            assert "Model A" in html_content
            assert "Model B" in html_content
            assert "Model C" in html_content
            assert "best-accuracy" in html_content


class TestHTMXSystemStatus:
    """Test HTMX system status functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.app = create_web_app()
        self.client = TestClient(self.app)

    def test_htmx_system_status_healthy(self):
        """Test HTMX system status when healthy."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_system_service"
        ) as mock_service:
            mock_service.return_value.get_system_status.return_value = {
                "overall_status": "healthy",
                "services": {
                    "database": {"status": "healthy", "response_time": 0.05},
                    "cache": {"status": "healthy", "response_time": 0.01},
                    "ml_engine": {"status": "healthy", "response_time": 0.1},
                },
                "system_metrics": {
                    "cpu_usage": 25.5,
                    "memory_usage": 60.2,
                    "disk_usage": 45.8,
                    "active_connections": 15,
                },
                "last_updated": "2023-01-01T12:00:00Z",
            }

            response = self.client.get(
                "/htmx/system/status", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "system-status" in html_content
            assert "status-healthy" in html_content
            assert "25.5%" in html_content  # CPU usage
            assert "60.2%" in html_content  # Memory usage
            assert "Database: Healthy" in html_content

    def test_htmx_system_status_degraded(self):
        """Test HTMX system status when degraded."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_system_service"
        ) as mock_service:
            mock_service.return_value.get_system_status.return_value = {
                "overall_status": "degraded",
                "services": {
                    "database": {"status": "healthy", "response_time": 0.05},
                    "cache": {"status": "slow", "response_time": 0.5},
                    "ml_engine": {"status": "healthy", "response_time": 0.1},
                },
                "system_metrics": {
                    "cpu_usage": 85.0,
                    "memory_usage": 90.5,
                    "disk_usage": 75.0,
                    "active_connections": 50,
                },
                "alerts": [
                    {"level": "warning", "message": "High memory usage detected"},
                    {"level": "warning", "message": "Cache performance degraded"},
                ],
            }

            response = self.client.get(
                "/htmx/system/status", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "status-degraded" in html_content
            assert "85.0%" in html_content  # CPU usage
            assert "Cache: Slow" in html_content
            assert "High memory usage" in html_content

    def test_htmx_system_status_error(self):
        """Test HTMX system status error handling."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_system_service"
        ) as mock_service:
            mock_service.return_value.get_system_status.side_effect = Exception(
                "System status unavailable"
            )

            response = self.client.get(
                "/htmx/system/status", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "system-error" in html_content
            assert "System status unavailable" in html_content


class TestHTMXFileUploadProgress:
    """Test HTMX file upload progress functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.app = create_web_app()
        self.client = TestClient(self.app)

    def test_htmx_file_upload_progress_success(self):
        """Test successful HTMX file upload progress."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_upload_service"
        ) as mock_service:
            mock_service.return_value.get_upload_progress.return_value = {
                "upload_id": "upload-123",
                "filename": "test_dataset.csv",
                "status": "uploading",
                "progress": 65,
                "bytes_uploaded": 65000,
                "total_bytes": 100000,
                "upload_speed": 1000,  # bytes per second
                "estimated_time_remaining": 35,
            }

            response = self.client.get(
                "/htmx/upload/progress/upload-123", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "upload-progress" in html_content
            assert "65%" in html_content
            assert "test_dataset.csv" in html_content
            assert "65,000 / 100,000 bytes" in html_content
            assert "1.0 KB/s" in html_content
            assert "35 seconds" in html_content

    def test_htmx_file_upload_progress_completed(self):
        """Test HTMX file upload progress when completed."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_upload_service"
        ) as mock_service:
            mock_service.return_value.get_upload_progress.return_value = {
                "upload_id": "upload-123",
                "filename": "completed_dataset.csv",
                "status": "completed",
                "progress": 100,
                "bytes_uploaded": 100000,
                "total_bytes": 100000,
                "upload_time": 120,  # seconds
                "dataset_id": "dataset-456",
                "validation_results": {
                    "valid": True,
                    "rows": 1000,
                    "columns": 5,
                    "data_types": {"numeric": 4, "categorical": 1},
                },
            }

            response = self.client.get(
                "/htmx/upload/progress/upload-123", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "upload-completed" in html_content
            assert "100%" in html_content
            assert "completed_dataset.csv" in html_content
            assert "2 minutes" in html_content  # Upload time
            assert "1000 rows" in html_content
            assert "5 columns" in html_content
            assert "dataset-456" in html_content

    def test_htmx_file_upload_progress_failed(self):
        """Test HTMX file upload progress when failed."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_upload_service"
        ) as mock_service:
            mock_service.return_value.get_upload_progress.return_value = {
                "upload_id": "upload-123",
                "filename": "failed_dataset.csv",
                "status": "failed",
                "progress": 45,
                "bytes_uploaded": 45000,
                "total_bytes": 100000,
                "error_message": "Invalid file format",
                "error_details": "File contains non-numeric data in column 3",
            }

            response = self.client.get(
                "/htmx/upload/progress/upload-123", headers={"HX-Request": "true"}
            )

            assert response.status_code == 200
            html_content = response.text
            assert "upload-failed" in html_content
            assert "Invalid file format" in html_content
            assert "non-numeric data in column 3" in html_content
            assert "45%" in html_content


class TestHTMXWebSocketIntegration:
    """Test HTMX WebSocket integration."""

    def setup_method(self):
        """Set up test environment."""
        self.app = create_web_app()
        self.client = TestClient(self.app)

    def test_htmx_websocket_connection_info(self):
        """Test HTMX WebSocket connection information."""
        response = self.client.get(
            "/htmx/websocket/info", headers={"HX-Request": "true"}
        )

        assert response.status_code == 200
        html_content = response.text
        assert "websocket-info" in html_content
        assert "ws://" in html_content or "wss://" in html_content
        assert "connection-status" in html_content

    def test_htmx_websocket_message_template(self):
        """Test HTMX WebSocket message template."""
        response = self.client.get(
            "/htmx/websocket/message-template", headers={"HX-Request": "true"}
        )

        assert response.status_code == 200
        html_content = response.text
        assert "websocket-message" in html_content
        assert "hx-swap-oob" in html_content
        assert "message-content" in html_content


class TestHTMXErrorHandling:
    """Test HTMX error handling."""

    def setup_method(self):
        """Set up test environment."""
        self.app = create_web_app()
        self.client = TestClient(self.app)

    def test_htmx_error_response(self):
        """Test HTMX error response handling."""
        response = self.client.get(
            "/htmx/nonexistent/endpoint", headers={"HX-Request": "true"}
        )

        assert response.status_code == 404
        html_content = response.text
        assert "error-message" in html_content
        assert "404" in html_content

    def test_htmx_validation_error(self):
        """Test HTMX validation error handling."""
        response = self.client.post(
            "/htmx/detection/results",
            json={"data": "invalid_data_format", "algorithm": ""},
            headers={"HX-Request": "true"},
        )

        assert response.status_code == 422
        html_content = response.text
        assert "validation-error" in html_content
        assert (
            "Invalid data format" in html_content or "Validation failed" in html_content
        )

    def test_htmx_server_error(self):
        """Test HTMX server error handling."""
        with patch(
            "pynomaly.presentation.web.dependencies.get_detection_service"
        ) as mock_service:
            mock_service.return_value.detect_anomalies.side_effect = Exception(
                "Internal server error"
            )

            response = self.client.post(
                "/htmx/detection/results",
                json={"data": [[1, 2], [3, 4]], "algorithm": "isolation_forest"},
                headers={"HX-Request": "true"},
            )

            assert response.status_code == 500
            html_content = response.text
            assert "server-error" in html_content
            assert "Internal server error" in html_content


# Test fixtures
@pytest.fixture
def htmx_headers():
    """Standard HTMX headers for testing."""
    return {
        "HX-Request": "true",
        "HX-Current-URL": "http://localhost:8080/",
        "HX-Target": "main-content",
    }


@pytest.fixture
def sample_detection_data():
    """Sample detection data for testing."""
    return {
        "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "algorithm": "isolation_forest",
        "parameters": {"contamination": 0.1, "n_estimators": 100},
    }


@pytest.fixture
def sample_training_data():
    """Sample training data for testing."""
    return {
        "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        "algorithm": "isolation_forest",
        "hyperparameters": {"n_estimators": 100, "contamination": 0.1},
    }
