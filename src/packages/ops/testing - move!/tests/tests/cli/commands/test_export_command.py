"""
Comprehensive tests for export CLI commands.
Tests for exporting detection results, models, and reports to various formats.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from monorepo.domain.exceptions import DatasetError, DetectorError, ValidationError
from monorepo.presentation.cli.export import export_app


class TestExportCommand:
    """Test suite for export CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_export_service(self):
        """Mock export service."""
        with patch("monorepo.presentation.cli.export.export_service") as mock:
            yield mock

    @pytest.fixture
    def mock_detector_service(self):
        """Mock detector service."""
        with patch("monorepo.presentation.cli.export.detector_service") as mock:
            yield mock

    @pytest.fixture
    def mock_dataset_service(self):
        """Mock dataset service."""
        with patch("monorepo.presentation.cli.export.dataset_service") as mock:
            yield mock

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container."""
        with patch("monorepo.presentation.cli.export.get_cli_container") as mock:
            container = Mock()
            container.config.return_value.storage_path = Path("/tmp/pynomaly")
            container.config.return_value.export_formats = [
                "csv",
                "json",
                "excel",
                "pdf",
            ]
            mock.return_value = container
            yield container

    @pytest.fixture
    def sample_detection_results(self):
        """Sample detection results."""
        return {
            "detector_id": "test-detector",
            "dataset_id": "test-dataset",
            "timestamp": "2024-01-01T00:00:00Z",
            "results": [
                {
                    "index": 0,
                    "score": 0.1,
                    "is_anomaly": False,
                    "confidence": 0.9,
                    "features": {"feature1": 1.0, "feature2": 2.0},
                },
                {
                    "index": 1,
                    "score": 0.8,
                    "is_anomaly": True,
                    "confidence": 0.7,
                    "features": {"feature1": 3.0, "feature2": 4.0},
                },
            ],
            "summary": {
                "total_samples": 2,
                "anomalies_detected": 1,
                "anomaly_rate": 0.5,
            },
        }

    @pytest.fixture
    def sample_report_data(self):
        """Sample report data."""
        return {
            "report_id": "test-report",
            "generated_at": "2024-01-01T00:00:00Z",
            "detector_summary": {
                "detector_id": "test-detector",
                "algorithm": "IsolationForest",
                "metrics": {"precision": 0.85, "recall": 0.78, "f1": 0.81},
            },
            "dataset_summary": {
                "dataset_id": "test-dataset",
                "rows": 1000,
                "columns": 5,
                "anomalies": 50,
            },
            "analysis": {
                "patterns": ["Feature1 values > 10 are anomalous"],
                "recommendations": ["Investigate samples with high feature1 values"],
            },
        }

    # Export Results Command Tests

    def test_export_results_basic_csv(
        self, runner, mock_export_service, sample_detection_results
    ):
        """Test basic export results to CSV."""
        mock_export_service.export_results.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.csv"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                ],
            )

            assert result.exit_code == 0
            assert "Results exported successfully" in result.stdout
            mock_export_service.export_results.assert_called_once()

    def test_export_results_basic_json(
        self, runner, mock_export_service, sample_detection_results
    ):
        """Test basic export results to JSON."""
        mock_export_service.export_results.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.json"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "json",
                ],
            )

            assert result.exit_code == 0
            assert "Results exported successfully" in result.stdout
            mock_export_service.export_results.assert_called_once()

    def test_export_results_with_filter(
        self, runner, mock_export_service, sample_detection_results
    ):
        """Test export results with anomaly filter."""
        mock_export_service.export_results.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "anomalies.csv"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                    "--filter",
                    "anomalies",
                ],
            )

            assert result.exit_code == 0
            assert "Results exported successfully" in result.stdout
            mock_export_service.export_results.assert_called_once()

    def test_export_results_with_threshold(
        self, runner, mock_export_service, sample_detection_results
    ):
        """Test export results with custom threshold."""
        mock_export_service.export_results.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.csv"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                    "--threshold",
                    "0.7",
                ],
            )

            assert result.exit_code == 0
            assert "Results exported successfully" in result.stdout
            mock_export_service.export_results.assert_called_once()

    def test_export_results_with_include_features(
        self, runner, mock_export_service, sample_detection_results
    ):
        """Test export results with feature inclusion."""
        mock_export_service.export_results.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results_with_features.csv"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                    "--include-features",
                ],
            )

            assert result.exit_code == 0
            assert "Results exported successfully" in result.stdout
            mock_export_service.export_results.assert_called_once()

    def test_export_results_detector_not_found(self, runner, mock_export_service):
        """Test export results with non-existent detector."""
        mock_export_service.export_results.side_effect = DetectorError(
            "Detector not found"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.csv"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "non-existent-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                ],
            )

            assert result.exit_code == 1
            assert "Detector not found" in result.stdout

    def test_export_results_dataset_not_found(self, runner, mock_export_service):
        """Test export results with non-existent dataset."""
        mock_export_service.export_results.side_effect = DatasetError(
            "Dataset not found"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.csv"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "non-existent-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                ],
            )

            assert result.exit_code == 1
            assert "Dataset not found" in result.stdout

    def test_export_results_unsupported_format(self, runner, mock_export_service):
        """Test export results with unsupported format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.xyz"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "xyz",
                ],
            )

            assert result.exit_code == 1
            assert "Unsupported export format" in result.stdout

    # Export Model Command Tests

    def test_export_model_basic(self, runner, mock_export_service):
        """Test basic model export."""
        mock_export_service.export_model.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "model.pkl"

            result = runner.invoke(
                export_app, ["model", "test-detector", "--output", str(output_file)]
            )

            assert result.exit_code == 0
            assert "Model exported successfully" in result.stdout
            mock_export_service.export_model.assert_called_once()

    def test_export_model_with_metadata(self, runner, mock_export_service):
        """Test model export with metadata."""
        mock_export_service.export_model.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "model_with_metadata.pkl"

            result = runner.invoke(
                export_app,
                [
                    "model",
                    "test-detector",
                    "--output",
                    str(output_file),
                    "--include-metadata",
                ],
            )

            assert result.exit_code == 0
            assert "Model exported successfully" in result.stdout
            mock_export_service.export_model.assert_called_once()

    def test_export_model_with_preprocessing(self, runner, mock_export_service):
        """Test model export with preprocessing pipeline."""
        mock_export_service.export_model.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "model_with_preprocessing.pkl"

            result = runner.invoke(
                export_app,
                [
                    "model",
                    "test-detector",
                    "--output",
                    str(output_file),
                    "--include-preprocessing",
                ],
            )

            assert result.exit_code == 0
            assert "Model exported successfully" in result.stdout
            mock_export_service.export_model.assert_called_once()

    def test_export_model_different_formats(self, runner, mock_export_service):
        """Test model export in different formats."""
        mock_export_service.export_model.return_value = None

        formats = ["pickle", "joblib", "onnx", "tensorflow"]

        for format_type in formats:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = Path(temp_dir) / f"model.{format_type}"

                result = runner.invoke(
                    export_app,
                    [
                        "model",
                        "test-detector",
                        "--output",
                        str(output_file),
                        "--format",
                        format_type,
                    ],
                )

                assert result.exit_code == 0
                assert "Model exported successfully" in result.stdout

    def test_export_model_not_found(self, runner, mock_export_service):
        """Test model export with non-existent detector."""
        mock_export_service.export_model.side_effect = DetectorError(
            "Detector not found"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "model.pkl"

            result = runner.invoke(
                export_app,
                ["model", "non-existent-detector", "--output", str(output_file)],
            )

            assert result.exit_code == 1
            assert "Detector not found" in result.stdout

    # Export Report Command Tests

    def test_export_report_basic(self, runner, mock_export_service, sample_report_data):
        """Test basic report export."""
        mock_export_service.export_report.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "report.pdf"

            result = runner.invoke(
                export_app,
                [
                    "report",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "pdf",
                ],
            )

            assert result.exit_code == 0
            assert "Report exported successfully" in result.stdout
            mock_export_service.export_report.assert_called_once()

    def test_export_report_html(self, runner, mock_export_service, sample_report_data):
        """Test HTML report export."""
        mock_export_service.export_report.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "report.html"

            result = runner.invoke(
                export_app,
                [
                    "report",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "html",
                ],
            )

            assert result.exit_code == 0
            assert "Report exported successfully" in result.stdout
            mock_export_service.export_report.assert_called_once()

    def test_export_report_with_sections(
        self, runner, mock_export_service, sample_report_data
    ):
        """Test report export with specific sections."""
        mock_export_service.export_report.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "report.pdf"

            result = runner.invoke(
                export_app,
                [
                    "report",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "pdf",
                    "--sections",
                    "summary,metrics,analysis",
                ],
            )

            assert result.exit_code == 0
            assert "Report exported successfully" in result.stdout
            mock_export_service.export_report.assert_called_once()

    def test_export_report_with_charts(
        self, runner, mock_export_service, sample_report_data
    ):
        """Test report export with charts."""
        mock_export_service.export_report.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "report_with_charts.pdf"

            result = runner.invoke(
                export_app,
                [
                    "report",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "pdf",
                    "--include-charts",
                ],
            )

            assert result.exit_code == 0
            assert "Report exported successfully" in result.stdout
            mock_export_service.export_report.assert_called_once()

    def test_export_report_with_template(
        self, runner, mock_export_service, sample_report_data
    ):
        """Test report export with custom template."""
        mock_export_service.export_report.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            template_file = Path(temp_dir) / "template.html"
            template_file.write_text("<html><body>Custom template</body></html>")

            output_file = Path(temp_dir) / "report.pdf"

            result = runner.invoke(
                export_app,
                [
                    "report",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "pdf",
                    "--template",
                    str(template_file),
                ],
            )

            assert result.exit_code == 0
            assert "Report exported successfully" in result.stdout
            mock_export_service.export_report.assert_called_once()

    # Export Dashboard Command Tests

    def test_export_dashboard_basic(self, runner, mock_export_service):
        """Test basic dashboard export."""
        mock_export_service.export_dashboard.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "dashboard.html"

            result = runner.invoke(
                export_app, ["dashboard", "test-detector", "--output", str(output_file)]
            )

            assert result.exit_code == 0
            assert "Dashboard exported successfully" in result.stdout
            mock_export_service.export_dashboard.assert_called_once()

    def test_export_dashboard_with_datasets(self, runner, mock_export_service):
        """Test dashboard export with multiple datasets."""
        mock_export_service.export_dashboard.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "dashboard.html"

            result = runner.invoke(
                export_app,
                [
                    "dashboard",
                    "test-detector",
                    "--datasets",
                    "dataset1,dataset2,dataset3",
                    "--output",
                    str(output_file),
                ],
            )

            assert result.exit_code == 0
            assert "Dashboard exported successfully" in result.stdout
            mock_export_service.export_dashboard.assert_called_once()

    def test_export_dashboard_with_time_range(self, runner, mock_export_service):
        """Test dashboard export with time range."""
        mock_export_service.export_dashboard.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "dashboard.html"

            result = runner.invoke(
                export_app,
                [
                    "dashboard",
                    "test-detector",
                    "--output",
                    str(output_file),
                    "--start-date",
                    "2024-01-01",
                    "--end-date",
                    "2024-01-31",
                ],
            )

            assert result.exit_code == 0
            assert "Dashboard exported successfully" in result.stdout
            mock_export_service.export_dashboard.assert_called_once()

    def test_export_dashboard_interactive(self, runner, mock_export_service):
        """Test interactive dashboard export."""
        mock_export_service.export_dashboard.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "dashboard_interactive.html"

            result = runner.invoke(
                export_app,
                [
                    "dashboard",
                    "test-detector",
                    "--output",
                    str(output_file),
                    "--interactive",
                ],
            )

            assert result.exit_code == 0
            assert "Dashboard exported successfully" in result.stdout
            mock_export_service.export_dashboard.assert_called_once()

    # Export Metrics Command Tests

    def test_export_metrics_basic(self, runner, mock_export_service):
        """Test basic metrics export."""
        mock_export_service.export_metrics.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "metrics.json"

            result = runner.invoke(
                export_app, ["metrics", "test-detector", "--output", str(output_file)]
            )

            assert result.exit_code == 0
            assert "Metrics exported successfully" in result.stdout
            mock_export_service.export_metrics.assert_called_once()

    def test_export_metrics_with_comparison(self, runner, mock_export_service):
        """Test metrics export with detector comparison."""
        mock_export_service.export_metrics.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "metrics_comparison.json"

            result = runner.invoke(
                export_app,
                [
                    "metrics",
                    "detector1,detector2,detector3",
                    "--output",
                    str(output_file),
                    "--compare",
                ],
            )

            assert result.exit_code == 0
            assert "Metrics exported successfully" in result.stdout
            mock_export_service.export_metrics.assert_called_once()

    def test_export_metrics_with_time_series(self, runner, mock_export_service):
        """Test metrics export with time series data."""
        mock_export_service.export_metrics.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "metrics_timeseries.csv"

            result = runner.invoke(
                export_app,
                [
                    "metrics",
                    "test-detector",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                    "--time-series",
                ],
            )

            assert result.exit_code == 0
            assert "Metrics exported successfully" in result.stdout
            mock_export_service.export_metrics.assert_called_once()

    # Export Configuration Command Tests

    def test_export_config_basic(self, runner, mock_export_service):
        """Test basic configuration export."""
        mock_export_service.export_configuration.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "config.json"

            result = runner.invoke(
                export_app, ["config", "test-detector", "--output", str(output_file)]
            )

            assert result.exit_code == 0
            assert "Configuration exported successfully" in result.stdout
            mock_export_service.export_configuration.assert_called_once()

    def test_export_config_with_hyperparameters(self, runner, mock_export_service):
        """Test configuration export with hyperparameters."""
        mock_export_service.export_configuration.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "config_with_hyperparams.yaml"

            result = runner.invoke(
                export_app,
                [
                    "config",
                    "test-detector",
                    "--output",
                    str(output_file),
                    "--format",
                    "yaml",
                    "--include-hyperparameters",
                ],
            )

            assert result.exit_code == 0
            assert "Configuration exported successfully" in result.stdout
            mock_export_service.export_configuration.assert_called_once()

    def test_export_config_with_training_history(self, runner, mock_export_service):
        """Test configuration export with training history."""
        mock_export_service.export_configuration.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "config_with_history.json"

            result = runner.invoke(
                export_app,
                [
                    "config",
                    "test-detector",
                    "--output",
                    str(output_file),
                    "--include-training-history",
                ],
            )

            assert result.exit_code == 0
            assert "Configuration exported successfully" in result.stdout
            mock_export_service.export_configuration.assert_called_once()

    # Batch Export Command Tests

    def test_export_batch_basic(self, runner, mock_export_service):
        """Test basic batch export."""
        mock_export_service.export_batch.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "batch_export"

            result = runner.invoke(
                export_app,
                [
                    "batch",
                    "detector1,detector2,detector3",
                    "--output-dir",
                    str(output_dir),
                    "--format",
                    "json",
                ],
            )

            assert result.exit_code == 0
            assert "Batch export completed successfully" in result.stdout
            mock_export_service.export_batch.assert_called_once()

    def test_export_batch_with_datasets(self, runner, mock_export_service):
        """Test batch export with datasets."""
        mock_export_service.export_batch.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "batch_export"

            result = runner.invoke(
                export_app,
                [
                    "batch",
                    "detector1,detector2",
                    "--datasets",
                    "dataset1,dataset2",
                    "--output-dir",
                    str(output_dir),
                    "--format",
                    "csv",
                ],
            )

            assert result.exit_code == 0
            assert "Batch export completed successfully" in result.stdout
            mock_export_service.export_batch.assert_called_once()

    def test_export_batch_with_parallel_processing(self, runner, mock_export_service):
        """Test batch export with parallel processing."""
        mock_export_service.export_batch.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "batch_export"

            result = runner.invoke(
                export_app,
                [
                    "batch",
                    "detector1,detector2,detector3",
                    "--output-dir",
                    str(output_dir),
                    "--format",
                    "json",
                    "--parallel",
                    "--workers",
                    "4",
                ],
            )

            assert result.exit_code == 0
            assert "Batch export completed successfully" in result.stdout
            mock_export_service.export_batch.assert_called_once()

    # Error Handling Tests

    def test_export_invalid_output_path(self, runner, mock_export_service):
        """Test export with invalid output path."""
        result = runner.invoke(
            export_app,
            [
                "results",
                "test-detector",
                "--dataset",
                "test-dataset",
                "--output",
                "/invalid/path/results.csv",
                "--format",
                "csv",
            ],
        )

        assert result.exit_code == 1
        assert "Output directory does not exist" in result.stdout

    def test_export_permission_denied(self, runner, mock_export_service):
        """Test export with permission denied."""
        mock_export_service.export_results.side_effect = PermissionError(
            "Permission denied"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.csv"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                ],
            )

            assert result.exit_code == 1
            assert "Permission denied" in result.stdout

    def test_export_service_error(self, runner, mock_export_service):
        """Test export with service error."""
        mock_export_service.export_results.side_effect = ValidationError(
            "Service error"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.csv"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                ],
            )

            assert result.exit_code == 1
            assert "Service error" in result.stdout

    def test_export_keyboard_interrupt(self, runner, mock_export_service):
        """Test export with keyboard interrupt."""
        mock_export_service.export_results.side_effect = KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "results.csv"

            result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                ],
            )

            assert result.exit_code == 1
            assert "Operation cancelled by user" in result.stdout

    # Integration Tests

    def test_export_complete_workflow(self, runner, mock_export_service):
        """Test complete export workflow."""
        mock_export_service.export_results.return_value = None
        mock_export_service.export_model.return_value = None
        mock_export_service.export_report.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            # Export results
            results_file = Path(temp_dir) / "results.csv"
            results_result = runner.invoke(
                export_app,
                [
                    "results",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(results_file),
                    "--format",
                    "csv",
                ],
            )
            assert results_result.exit_code == 0

            # Export model
            model_file = Path(temp_dir) / "model.pkl"
            model_result = runner.invoke(
                export_app, ["model", "test-detector", "--output", str(model_file)]
            )
            assert model_result.exit_code == 0

            # Export report
            report_file = Path(temp_dir) / "report.pdf"
            report_result = runner.invoke(
                export_app,
                [
                    "report",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(report_file),
                    "--format",
                    "pdf",
                ],
            )
            assert report_result.exit_code == 0

    def test_export_help_commands(self, runner):
        """Test help commands for export."""
        result = runner.invoke(export_app, ["--help"])
        assert result.exit_code == 0
        assert "Commands:" in result.stdout
        assert "results" in result.stdout
        assert "model" in result.stdout
        assert "report" in result.stdout
        assert "dashboard" in result.stdout
        assert "metrics" in result.stdout

        # Test subcommand help
        result = runner.invoke(export_app, ["results", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--output" in result.stdout
        assert "--format" in result.stdout

        result = runner.invoke(export_app, ["model", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--output" in result.stdout
        assert "--include-metadata" in result.stdout

        result = runner.invoke(export_app, ["report", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--output" in result.stdout
        assert "--format" in result.stdout
        assert "--sections" in result.stdout
