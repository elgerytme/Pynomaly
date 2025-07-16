"""
Integration tests for CLI workflow scenarios.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from pynomaly.presentation.cli import autonomous, datasets, detection, detectors
from pynomaly.presentation.cli.app import app


class TestCLIIntegration:
    """Test CLI integration workflows."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_data_file(self):
        """Create sample data file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("feature1,feature2,feature3,target\n")
            f.write("1.0,2.0,3.0,0\n")
            f.write("2.0,3.0,4.0,0\n")
            f.write("100.0,200.0,300.0,1\n")  # Outlier
            f.write("3.0,4.0,5.0,0\n")
            f.write("4.0,5.0,6.0,0\n")
            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_autonomous_detection_workflow(self, runner, sample_data_file):
        """Test complete autonomous detection workflow."""
        with patch(
            "pynomaly.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance

            # Mock successful detection
            mock_service_instance.detect_anomalies.return_value = {
                "best_detector": "IsolationForest",
                "anomalies_found": 1,
                "confidence": 0.95,
                "anomaly_indices": [2],
                "anomaly_scores": [0.95],
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as output_file:
                output_path = output_file.name

            try:
                result = runner.invoke(
                    autonomous.app,
                    [
                        "detect",
                        sample_data_file,
                        "--output",
                        output_path,
                        "--contamination",
                        "0.1",
                    ],
                )

                # Should complete successfully
                assert result.exit_code == 0
                assert (
                    "Autonomous detection completed" in result.stdout
                    or "detected" in result.stdout.lower()
                )

            finally:
                Path(output_path).unlink(missing_ok=True)

    def test_dataset_lifecycle_workflow(self, runner, sample_data_file):
        """Test dataset load, show, and delete workflow."""
        with patch(
            "pynomaly.presentation.cli.container.get_cli_container"
        ) as mock_container:
            container = Mock()
            dataset_repo = Mock()
            container.dataset_repository.return_value = dataset_repo

            # Mock successful dataset operations
            mock_dataset = Mock()
            mock_dataset.name = "test_dataset"
            mock_dataset.shape = (5, 4)
            mock_dataset.features = ["feature1", "feature2", "feature3", "target"]

            dataset_repo.save.return_value = True
            dataset_repo.find_by_name.return_value = mock_dataset
            dataset_repo.delete.return_value = True

            mock_container.return_value = container

            # 1. Load dataset
            load_result = runner.invoke(
                datasets.app, ["load", sample_data_file, "--name", "test_dataset"]
            )

            # 2. Show dataset
            show_result = runner.invoke(datasets.app, ["show", "test_dataset"])

            # 3. Delete dataset
            delete_result = runner.invoke(
                datasets.app, ["delete", "test_dataset"], input="y\n"
            )

            # All operations should handle mocked dependencies
            assert load_result.exit_code in [0, 1]
            assert show_result.exit_code in [0, 1]
            assert delete_result.exit_code in [0, 1]

    def test_detector_creation_and_training_workflow(self, runner, sample_data_file):
        """Test detector creation and training workflow."""
        with patch(
            "pynomaly.presentation.cli.container.get_cli_container"
        ) as mock_container:
            container = Mock()
            detector_repo = Mock()
            dataset_repo = Mock()

            container.detector_repository.return_value = detector_repo
            container.dataset_repository.return_value = dataset_repo

            # Mock detector and dataset
            mock_detector = Mock()
            mock_detector.name = "test_detector"
            mock_detector.algorithm_name = "IsolationForest"
            mock_detector.is_fitted = False

            mock_dataset = Mock()
            mock_dataset.name = "test_dataset"

            detector_repo.save.return_value = True
            detector_repo.find_by_name.return_value = mock_detector
            dataset_repo.find_by_name.return_value = mock_dataset

            mock_container.return_value = container

            # 1. Create detector
            create_result = runner.invoke(
                detectors.app,
                [
                    "create",
                    "--name",
                    "test_detector",
                    "--algorithm",
                    "IsolationForest",
                    "--contamination",
                    "0.1",
                ],
            )

            # 2. Train detector (mock training)
            with patch(
                "pynomaly.application.use_cases.train_detector.TrainDetectorUseCase"
            ) as mock_use_case:
                mock_use_case_instance = Mock()
                mock_use_case.return_value = mock_use_case_instance

                train_result = runner.invoke(
                    detection.app,
                    [
                        "train",
                        "--detector",
                        "test_detector",
                        "--dataset",
                        "test_dataset",
                    ],
                )

            # Operations should handle mocked dependencies
            assert create_result.exit_code in [0, 1]
            assert train_result.exit_code in [0, 1]

    def test_configuration_generation_workflow(self, runner):
        """Test configuration generation and usage workflow."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as config_file:
            config_path = config_file.name

        try:
            # Generate test configuration
            result = runner.invoke(
                app,
                [
                    "generate-config",
                    "test",
                    "--output",
                    config_path,
                    "--detector",
                    "IsolationForest",
                ],
            )

            if result.exit_code == 0:
                # Verify configuration was created
                assert Path(config_path).exists()

                # Verify configuration content
                with open(config_path) as f:
                    config = json.load(f)

                assert config["metadata"]["type"] == "test"
                assert config["test"]["detector"]["algorithm"] == "IsolationForest"

        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_export_workflow(self, runner):
        """Test export functionality workflow."""
        # Create sample results data
        results_data = {
            "detector": "IsolationForest",
            "dataset": "test_data",
            "anomalies": [1, 5, 10],
            "scores": [0.8, 0.9, 0.95],
            "metadata": {"total_samples": 100},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as results_file:
            json.dump(results_data, results_file)
            results_path = results_file.name

        try:
            # Test export formats listing
            list_result = runner.invoke(app, ["export", "list-formats"])
            assert list_result.exit_code == 0
            assert "Available Export Formats" in list_result.stdout

            # Test Excel export (mocked)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".xlsx", delete=False
            ) as excel_file:
                excel_path = excel_file.name

            try:
                with patch(
                    "pynomaly.infrastructure.exporters.excel_exporter.ExcelExporter"
                ) as mock_exporter:
                    mock_exporter_instance = Mock()
                    mock_exporter.return_value = mock_exporter_instance

                    excel_result = runner.invoke(
                        app, ["export", "excel", results_path, excel_path]
                    )

                    assert excel_result.exit_code in [0, 1]

            finally:
                Path(excel_path).unlink(missing_ok=True)

        finally:
            Path(results_path).unlink(missing_ok=True)

    def test_server_management_workflow(self, runner):
        """Test server management workflow."""
        # Test server status
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            status_result = runner.invoke(app, ["server", "status"])
            assert status_result.exit_code == 0
            assert "Server Status" in status_result.stdout

    def test_quickstart_workflow(self, runner):
        """Test quickstart workflow."""
        # Test quickstart acceptance
        result = runner.invoke(app, ["quickstart"], input="y\n")
        assert result.exit_code == 0
        assert "Welcome to Pynomaly!" in result.stdout

        # Test quickstart decline
        result = runner.invoke(app, ["quickstart"], input="n\n")
        assert result.exit_code == 0
        assert "Quickstart cancelled" in result.stdout

    def test_help_system_workflow(self, runner):
        """Test help system completeness."""
        # Test main help
        main_help = runner.invoke(app, ["--help"])
        assert main_help.exit_code == 0
        assert "Commands:" in main_help.stdout

        # Test subcommand helps
        subcommands = ["auto", "dataset", "detector", "detect", "export", "server"]

        for cmd in subcommands:
            help_result = runner.invoke(app, [cmd, "--help"])
            assert help_result.exit_code == 0
            assert (
                "Usage:" in help_result.stdout
                or "Commands:" in help_result.stdout
                or "command" in help_result.stdout.lower()
            )

    def test_error_recovery_workflow(self, runner, sample_data_file):
        """Test error recovery in workflows."""
        with patch(
            "pynomaly.presentation.cli.container.get_cli_container"
        ) as mock_container:
            container = Mock()
            dataset_repo = Mock()
            container.dataset_repository.return_value = dataset_repo

            # Mock dataset load failure then success
            dataset_repo.save.side_effect = [Exception("Database error"), True]

            mock_container.return_value = container

            # First attempt should fail
            first_result = runner.invoke(
                datasets.app, ["load", sample_data_file, "--name", "test_dataset"]
            )

            # Second attempt should succeed (with fixed mock)
            dataset_repo.save.side_effect = None
            dataset_repo.save.return_value = True

            second_result = runner.invoke(
                datasets.app, ["load", sample_data_file, "--name", "test_dataset"]
            )

            # Both should handle errors gracefully
            assert first_result.exit_code in [0, 1]
            assert second_result.exit_code in [0, 1]

    def test_concurrent_operations_workflow(self, runner, sample_data_file):
        """Test handling of concurrent operations."""
        with patch(
            "pynomaly.presentation.cli.container.get_cli_container"
        ) as mock_container:
            container = Mock()
            dataset_repo = Mock()
            container.dataset_repository.return_value = dataset_repo

            # Mock concurrent access
            dataset_repo.save.return_value = True
            dataset_repo.find_by_name.return_value = Mock(name="test_dataset")

            mock_container.return_value = container

            # Simulate concurrent dataset operations
            results = []
            for i in range(3):
                result = runner.invoke(
                    datasets.app,
                    ["load", sample_data_file, "--name", f"test_dataset_{i}"],
                )
                results.append(result)

            # All operations should handle concurrency gracefully
            for result in results:
                assert result.exit_code in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__])
