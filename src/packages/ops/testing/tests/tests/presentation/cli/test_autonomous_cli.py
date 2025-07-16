"""
Autonomous CLI Testing Suite
Comprehensive tests for autonomous anomaly detection CLI commands.
"""

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from pynomaly.presentation.cli.autonomous import app


class TestAutonomousCLI:
    """Test suite for autonomous CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_data_file(self):
        """Create sample data file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "cpu_usage", "memory_usage", "network_io", "disk_io"]
            )

            # Generate normal data
            for i in range(100):
                writer.writerow(
                    [
                        f"2024-01-{i // 10 + 1:02d} {i % 24:02d}:00:00",
                        50 + (i % 20),  # CPU usage 50-70%
                        60 + (i % 15),  # Memory usage 60-75%
                        100 + (i % 50),  # Network I/O
                        200 + (i % 30),  # Disk I/O
                    ]
                )

            # Add some anomalies
            anomaly_data = [
                ["2024-01-15 10:00:00", 95, 90, 500, 800],  # High CPU/Memory/IO
                ["2024-01-16 14:30:00", 98, 95, 600, 900],  # Very high usage
                ["2024-01-17 09:15:00", 5, 10, 10, 5],  # Very low usage
            ]

            for row in anomaly_data:
                writer.writerow(row)

            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def mock_autonomous_service(self):
        """Mock autonomous detection service."""
        with patch(
            "pynomaly.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock:
            service = Mock()

            # Mock data profiling results
            service.profile_data.return_value = {
                "shape": (103, 5),
                "data_types": {
                    "timestamp": "datetime",
                    "cpu_usage": "numeric",
                    "memory_usage": "numeric",
                    "network_io": "numeric",
                    "disk_io": "numeric",
                },
                "missing_values": 0,
                "outlier_percentage": 0.029,
                "recommended_algorithms": [
                    "IsolationForest",
                    "LOF",
                    "EllipticEnvelope",
                ],
                "seasonality_detected": False,
                "time_series_features": ["timestamp"],
                "numerical_features": [
                    "cpu_usage",
                    "memory_usage",
                    "network_io",
                    "disk_io",
                ],
                "categorical_features": [],
                "complexity_score": 0.3,
                "quality_score": 0.95,
            }

            # Mock autonomous detection results
            service.detect_anomalies.return_value = {
                "best_detector": "IsolationForest",
                "best_score": 0.87,
                "anomalies_found": 3,
                "anomaly_indices": [100, 101, 102],
                "confidence_scores": [0.92, 0.89, 0.85],
                "algorithm_comparison": {
                    "IsolationForest": {"score": 0.87, "anomalies": 3},
                    "LOF": {"score": 0.82, "anomalies": 4},
                    "EllipticEnvelope": {"score": 0.78, "anomalies": 5},
                },
                "feature_importance": {
                    "cpu_usage": 0.35,
                    "memory_usage": 0.28,
                    "network_io": 0.22,
                    "disk_io": 0.15,
                },
                "runtime_seconds": 2.3,
                "hyperparameters_used": {
                    "contamination": 0.05,
                    "n_estimators": 100,
                    "random_state": 42,
                },
            }

            # Mock quick detection results
            service.quick_detect.return_value = {
                "anomalies": [100, 101, 102],
                "scores": [0.92, 0.89, 0.85],
                "algorithm": "IsolationForest",
                "runtime_seconds": 0.8,
            }

            mock.return_value = service
            yield service

    # Basic Command Tests

    def test_autonomous_help(self, runner):
        """Test autonomous CLI help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "autonomous anomaly detection" in result.stdout.lower()
        assert "Commands:" in result.stdout
        assert "detect" in result.stdout
        assert "profile" in result.stdout
        assert "quick" in result.stdout

    def test_detect_help(self, runner):
        """Test detect command help."""
        result = runner.invoke(app, ["detect", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "data_path" in result.stdout
        assert "--output" in result.stdout
        assert "--contamination" in result.stdout

    def test_profile_help(self, runner):
        """Test profile command help."""
        result = runner.invoke(app, ["profile", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "data_path" in result.stdout
        assert "--verbose" in result.stdout

    def test_quick_help(self, runner):
        """Test quick command help."""
        result = runner.invoke(app, ["quick", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "data_path" in result.stdout
        assert "--contamination" in result.stdout

    # Profile Command Tests

    def test_profile_basic(self, runner, mock_autonomous_service, sample_data_file):
        """Test basic data profiling."""
        result = runner.invoke(app, ["profile", sample_data_file])

        assert result.exit_code == 0
        assert "Data Profile" in result.stdout
        assert "Shape:" in result.stdout
        assert "103" in result.stdout
        assert "5" in result.stdout
        assert "Recommended algorithms:" in result.stdout
        assert "IsolationForest" in result.stdout

    def test_profile_verbose(self, runner, mock_autonomous_service, sample_data_file):
        """Test verbose data profiling."""
        result = runner.invoke(app, ["profile", sample_data_file, "--verbose"])

        assert result.exit_code == 0
        assert "Data Profile" in result.stdout
        assert "Feature Analysis:" in result.stdout
        assert "Quality Score:" in result.stdout
        assert "0.95" in result.stdout

    def test_profile_save_report(
        self, runner, mock_autonomous_service, sample_data_file
    ):
        """Test saving profile report."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report_path = f.name

        try:
            result = runner.invoke(
                app, ["profile", sample_data_file, "--save-report", report_path]
            )

            assert result.exit_code == 0
            assert "Profile saved" in result.stdout

            # Verify report was saved
            assert Path(report_path).exists()
            with open(report_path) as f:
                report = json.load(f)

            assert "shape" in report
            assert "data_types" in report

        finally:
            Path(report_path).unlink(missing_ok=True)

    def test_profile_nonexistent_file(self, runner):
        """Test profiling nonexistent file."""
        result = runner.invoke(app, ["profile", "/path/to/nonexistent.csv"])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    # Quick Detection Tests

    def test_quick_detect_basic(
        self, runner, mock_autonomous_service, sample_data_file
    ):
        """Test basic quick detection."""
        result = runner.invoke(app, ["quick", sample_data_file])

        assert result.exit_code == 0
        assert "Quick Anomaly Detection" in result.stdout
        assert "Found 3 anomalies" in result.stdout
        assert "Algorithm used: IsolationForest" in result.stdout

    def test_quick_detect_with_contamination(
        self, runner, mock_autonomous_service, sample_data_file
    ):
        """Test quick detection with custom contamination rate."""
        result = runner.invoke(
            app, ["quick", sample_data_file, "--contamination", "0.1"]
        )

        assert result.exit_code == 0
        assert "Quick Anomaly Detection" in result.stdout

    def test_quick_detect_verbose(
        self, runner, mock_autonomous_service, sample_data_file
    ):
        """Test verbose quick detection."""
        result = runner.invoke(app, ["quick", sample_data_file, "--verbose"])

        assert result.exit_code == 0
        assert "Runtime:" in result.stdout
        assert "0.8 seconds" in result.stdout

    def test_quick_detect_save_results(
        self, runner, mock_autonomous_service, sample_data_file
    ):
        """Test quick detection with result saving."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            results_path = f.name

        try:
            result = runner.invoke(
                app, ["quick", sample_data_file, "--save-results", results_path]
            )

            assert result.exit_code == 0
            assert "Results saved" in result.stdout

            # Verify results were saved
            assert Path(results_path).exists()
            with open(results_path) as f:
                results = json.load(f)

            assert "anomalies" in results
            assert "scores" in results

        finally:
            Path(results_path).unlink(missing_ok=True)

    # Full Detection Tests

    def test_detect_basic(self, runner, mock_autonomous_service, sample_data_file):
        """Test basic autonomous detection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result = runner.invoke(
                app, ["detect", sample_data_file, "--output", output_path]
            )

            assert result.exit_code == 0
            assert "Autonomous Anomaly Detection" in result.stdout
            assert "Best detector: IsolationForest" in result.stdout
            assert "Score: 0.87" in result.stdout
            assert "Found 3 anomalies" in result.stdout

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_detect_with_custom_options(
        self, runner, mock_autonomous_service, sample_data_file
    ):
        """Test detection with custom options."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "detect",
                    sample_data_file,
                    "--output",
                    output_path,
                    "--contamination",
                    "0.08",
                    "--max-algorithms",
                    "3",
                    "--confidence-threshold",
                    "0.85",
                ],
            )

            assert result.exit_code == 0
            assert "Autonomous Anomaly Detection" in result.stdout

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_detect_verbose_output(
        self, runner, mock_autonomous_service, sample_data_file
    ):
        """Test detection with verbose output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result = runner.invoke(
                app, ["detect", sample_data_file, "--output", output_path, "--verbose"]
            )

            assert result.exit_code == 0
            assert "Algorithm Comparison:" in result.stdout
            assert "Feature Importance:" in result.stdout
            assert "cpu_usage" in result.stdout
            assert "Runtime:" in result.stdout

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_detect_no_tuning(self, runner, mock_autonomous_service, sample_data_file):
        """Test detection without hyperparameter tuning."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result = runner.invoke(
                app,
                ["detect", sample_data_file, "--output", output_path, "--no-tuning"],
            )

            assert result.exit_code == 0

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_detect_save_model(self, runner, mock_autonomous_service, sample_data_file):
        """Test detection with model saving."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "detect",
                    sample_data_file,
                    "--output",
                    output_path,
                    "--save-model",
                    model_path,
                ],
            )

            assert result.exit_code == 0
            assert "Model saved" in result.stdout

        finally:
            Path(output_path).unlink(missing_ok=True)
            Path(model_path).unlink(missing_ok=True)

    # Configuration Tests

    def test_detect_with_config_file(
        self, runner, mock_autonomous_service, sample_data_file
    ):
        """Test detection with configuration file."""
        # Create config file
        config = {
            "analysis": {"max_samples": 1000, "profile_data": True},
            "detection": {
                "max_algorithms": 2,
                "confidence_threshold": 0.9,
                "auto_tune_hyperparams": False,
            },
            "output": {"verbose": True, "export_format": "csv"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "detect",
                    sample_data_file,
                    "--output",
                    output_path,
                    "--config",
                    config_path,
                ],
            )

            assert result.exit_code == 0

        finally:
            Path(config_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    # Error Handling Tests

    def test_detect_invalid_contamination(self, runner, sample_data_file):
        """Test detection with invalid contamination rate."""
        result = runner.invoke(
            app, ["detect", sample_data_file, "--contamination", "1.5"]
        )

        assert result.exit_code != 0

    def test_detect_invalid_max_algorithms(self, runner, sample_data_file):
        """Test detection with invalid max algorithms."""
        result = runner.invoke(
            app, ["detect", sample_data_file, "--max-algorithms", "0"]
        )

        assert result.exit_code != 0

    def test_detect_missing_output_file(self, runner, sample_data_file):
        """Test detection without required output file."""
        result = runner.invoke(app, ["detect", sample_data_file])

        assert result.exit_code != 0

    def test_detect_invalid_output_directory(self, runner, sample_data_file):
        """Test detection with invalid output directory."""
        result = runner.invoke(
            app,
            [
                "detect",
                sample_data_file,
                "--output",
                "/nonexistent/directory/output.csv",
            ],
        )

        assert result.exit_code == 1

    # Service Integration Tests

    def test_autonomous_service_error_handling(self, runner, sample_data_file):
        """Test handling of autonomous service errors."""
        with patch(
            "pynomaly.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock_service:
            mock_service.side_effect = Exception("Service unavailable")

            result = runner.invoke(app, ["profile", sample_data_file])

            assert result.exit_code == 1
            assert "Error" in result.stdout

    def test_autonomous_service_memory_error(self, runner, sample_data_file):
        """Test handling of memory errors."""
        with patch(
            "pynomaly.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock_service:
            service = Mock()
            service.profile_data.side_effect = MemoryError("Out of memory")
            mock_service.return_value = service

            result = runner.invoke(app, ["profile", sample_data_file])

            assert result.exit_code == 1
            assert "memory" in result.stdout.lower()

    # Data Format Tests

    def test_detect_json_data(self, runner, mock_autonomous_service):
        """Test detection with JSON data format."""
        json_data = {
            "data": [
                {"cpu": 50, "memory": 60, "disk": 100},
                {"cpu": 55, "memory": 65, "disk": 110},
                {"cpu": 95, "memory": 90, "disk": 500},  # Anomaly
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result = runner.invoke(app, ["detect", json_path, "--output", output_path])

            assert result.exit_code == 0

        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_detect_parquet_data(self, runner, mock_autonomous_service):
        """Test detection with Parquet data format."""
        # Mock parquet file handling
        with patch("pandas.read_parquet") as mock_read:
            import pandas as pd

            mock_read.return_value = pd.DataFrame(
                {
                    "feature1": [1, 2, 100],  # Anomaly
                    "feature2": [10, 20, 30],
                }
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".parquet", delete=False
            ) as f:
                parquet_path = f.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                output_path = f.name

            try:
                result = runner.invoke(
                    app, ["detect", parquet_path, "--output", output_path]
                )

                assert result.exit_code == 0

            finally:
                Path(parquet_path).unlink(missing_ok=True)
                Path(output_path).unlink(missing_ok=True)

    # Performance Tests

    def test_detect_large_dataset_warning(self, runner, mock_autonomous_service):
        """Test warning for large datasets."""
        # Mock large dataset profiling
        mock_autonomous_service.profile_data.return_value = {
            "shape": (1000000, 50),  # Large dataset
            "data_types": {},
            "missing_values": 0,
            "recommended_algorithms": ["IsolationForest"],
            "complexity_score": 0.8,
            "quality_score": 0.9,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            large_data_path = f.name

        try:
            result = runner.invoke(app, ["profile", large_data_path])

            assert result.exit_code == 0
            assert (
                "large dataset" in result.stdout.lower() or "1000000" in result.stdout
            )

        finally:
            Path(large_data_path).unlink(missing_ok=True)

    def test_detect_performance_monitoring(
        self, runner, mock_autonomous_service, sample_data_file
    ):
        """Test performance monitoring output."""
        result = runner.invoke(app, ["quick", sample_data_file, "--verbose"])

        assert result.exit_code == 0
        assert "Runtime:" in result.stdout
        assert "seconds" in result.stdout
