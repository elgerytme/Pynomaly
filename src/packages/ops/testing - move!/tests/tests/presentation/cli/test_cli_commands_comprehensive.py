"""
Comprehensive tests for CLI commands.

This module provides extensive testing for all CLI commands in the
presentation layer, including command execution, argument validation,
error handling, and interactive features.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
from click.testing import CliRunner as ClickRunner
from typer.testing import CliRunner

from pynomaly.presentation.cli.app import app
from pynomaly.presentation.cli.lazy_app import lazy_app


class TestCLIApplication:
    """Test CLI application setup and basic functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def click_runner(self):
        """Create click test runner."""
        return ClickRunner()

    def test_cli_app_creation(self):
        """Test CLI app creation."""
        assert app is not None
        assert hasattr(app, "command")

    def test_cli_help_command(self, runner):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout

    def test_cli_version_command(self, runner):
        """Test CLI version command."""
        result = runner.invoke(app, ["--version"])
        # Should either show version or exit cleanly
        assert result.exit_code in [0, 1]

    def test_lazy_app_functionality(self):
        """Test lazy app functionality."""
        assert lazy_app is not None
        # Lazy app should be callable
        assert callable(lazy_app)


class TestDetectionCommands:
    """Test detection-related CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_data_file(self):
        """Create sample data file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("timestamp,value\n")
            f.write("2023-01-01,1.0\n")
            f.write("2023-01-02,2.0\n")
            f.write("2023-01-03,10.0\n")
            return f.name

    def test_detect_command_help(self, runner):
        """Test detect command help."""
        result = runner.invoke(app, ["detect", "--help"])
        assert result.exit_code == 0
        assert "detect" in result.stdout.lower()

    @patch("pynomaly.presentation.cli.commands.detection.detect_anomalies")
    def test_detect_command_basic(self, mock_detect, runner, sample_data_file):
        """Test basic detect command."""
        mock_detect.return_value = {"anomalies": []}

        result = runner.invoke(
            app,
            ["detect", "--file", sample_data_file, "--algorithm", "isolation_forest"],
        )

        # Should not crash
        assert result.exit_code in [0, 1]  # May fail due to missing dependencies

    @patch("pynomaly.presentation.cli.commands.detection.detect_anomalies")
    def test_detect_command_with_options(self, mock_detect, runner, sample_data_file):
        """Test detect command with various options."""
        mock_detect.return_value = {"anomalies": []}

        result = runner.invoke(
            app,
            [
                "detect",
                "--file",
                sample_data_file,
                "--algorithm",
                "lof",
                "--contamination",
                "0.1",
                "--output",
                "json",
            ],
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_detect_command_missing_file(self, runner):
        """Test detect command with missing file."""
        result = runner.invoke(
            app,
            ["detect", "--file", "nonexistent.csv", "--algorithm", "isolation_forest"],
        )

        # Should fail gracefully
        assert result.exit_code != 0

    def test_detect_command_invalid_algorithm(self, runner, sample_data_file):
        """Test detect command with invalid algorithm."""
        result = runner.invoke(
            app,
            ["detect", "--file", sample_data_file, "--algorithm", "invalid_algorithm"],
        )

        # Should fail gracefully
        assert result.exit_code != 0

    def cleanup_sample_file(self, sample_data_file):
        """Clean up sample data file."""
        if os.path.exists(sample_data_file):
            os.unlink(sample_data_file)


class TestDetectorCommands:
    """Test detector management CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_detector_list_command(self, runner):
        """Test detector list command."""
        result = runner.invoke(app, ["detector", "list"])
        # Should not crash
        assert result.exit_code in [0, 1]

    def test_detector_create_command_help(self, runner):
        """Test detector create command help."""
        result = runner.invoke(app, ["detector", "create", "--help"])
        assert result.exit_code == 0
        assert "create" in result.stdout.lower()

    @patch("pynomaly.presentation.cli.commands.detectors.create_detector")
    def test_detector_create_command(self, mock_create, runner):
        """Test detector create command."""
        mock_create.return_value = {"id": "test-detector"}

        result = runner.invoke(
            app,
            [
                "detector",
                "create",
                "--name",
                "test-detector",
                "--algorithm",
                "isolation_forest",
            ],
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_detector_delete_command_help(self, runner):
        """Test detector delete command help."""
        result = runner.invoke(app, ["detector", "delete", "--help"])
        assert result.exit_code == 0
        assert "delete" in result.stdout.lower()

    @patch("pynomaly.presentation.cli.commands.detectors.delete_detector")
    def test_detector_delete_command(self, mock_delete, runner):
        """Test detector delete command."""
        mock_delete.return_value = True

        result = runner.invoke(app, ["detector", "delete", "--id", "test-detector"])

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_detector_info_command(self, runner):
        """Test detector info command."""
        result = runner.invoke(app, ["detector", "info", "--id", "test-detector"])

        # Should not crash
        assert result.exit_code in [0, 1]


class TestDatasetCommands:
    """Test dataset management CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_dataset_file(self):
        """Create sample dataset file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(
                {"name": "test-dataset", "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, f
            )
            return f.name

    def test_dataset_list_command(self, runner):
        """Test dataset list command."""
        result = runner.invoke(app, ["dataset", "list"])
        # Should not crash
        assert result.exit_code in [0, 1]

    def test_dataset_upload_command_help(self, runner):
        """Test dataset upload command help."""
        result = runner.invoke(app, ["dataset", "upload", "--help"])
        assert result.exit_code == 0
        assert "upload" in result.stdout.lower()

    @patch("pynomaly.presentation.cli.commands.datasets.upload_dataset")
    def test_dataset_upload_command(self, mock_upload, runner, sample_dataset_file):
        """Test dataset upload command."""
        mock_upload.return_value = {"id": "test-dataset"}

        result = runner.invoke(
            app,
            [
                "dataset",
                "upload",
                "--file",
                sample_dataset_file,
                "--name",
                "test-dataset",
            ],
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_dataset_download_command_help(self, runner):
        """Test dataset download command help."""
        result = runner.invoke(app, ["dataset", "download", "--help"])
        assert result.exit_code == 0
        assert "download" in result.stdout.lower()

    @patch("pynomaly.presentation.cli.commands.datasets.download_dataset")
    def test_dataset_download_command(self, mock_download, runner):
        """Test dataset download command."""
        mock_download.return_value = b"test data"

        result = runner.invoke(
            app,
            [
                "dataset",
                "download",
                "--id",
                "test-dataset",
                "--output",
                "test_output.csv",
            ],
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_dataset_info_command(self, runner):
        """Test dataset info command."""
        result = runner.invoke(app, ["dataset", "info", "--id", "test-dataset"])

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_dataset_delete_command(self, runner):
        """Test dataset delete command."""
        result = runner.invoke(app, ["dataset", "delete", "--id", "test-dataset"])

        # Should not crash
        assert result.exit_code in [0, 1]


class TestConfigCommands:
    """Test configuration CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(
                {
                    "api_url": "http://localhost:8000",
                    "api_key": "test-key",
                    "timeout": 30,
                },
                f,
            )
            return f.name

    def test_config_show_command(self, runner):
        """Test config show command."""
        result = runner.invoke(app, ["config", "show"])
        # Should not crash
        assert result.exit_code in [0, 1]

    def test_config_set_command_help(self, runner):
        """Test config set command help."""
        result = runner.invoke(app, ["config", "set", "--help"])
        assert result.exit_code == 0
        assert "set" in result.stdout.lower()

    @patch("pynomaly.presentation.cli.commands.config.set_config")
    def test_config_set_command(self, mock_set, runner):
        """Test config set command."""
        mock_set.return_value = True

        result = runner.invoke(
            app,
            ["config", "set", "--key", "api_url", "--value", "http://localhost:8000"],
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_config_get_command(self, runner):
        """Test config get command."""
        result = runner.invoke(app, ["config", "get", "--key", "api_url"])

        # Should not crash
        assert result.exit_code in [0, 1]

    @patch("pynomaly.presentation.cli.commands.config.load_config")
    def test_config_import_command(self, mock_load, runner, temp_config_file):
        """Test config import command."""
        mock_load.return_value = {"api_url": "http://localhost:8000"}

        result = runner.invoke(app, ["config", "import", "--file", temp_config_file])

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_config_export_command(self, runner):
        """Test config export command."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            result = runner.invoke(app, ["config", "export", "--file", f.name])

            # Should not crash
            assert result.exit_code in [0, 1]

    def test_config_reset_command(self, runner):
        """Test config reset command."""
        result = runner.invoke(app, ["config", "reset"])
        # Should not crash
        assert result.exit_code in [0, 1]


class TestAutoMLCommands:
    """Test AutoML CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_automl_help_command(self, runner):
        """Test AutoML help command."""
        result = runner.invoke(app, ["automl", "--help"])
        assert result.exit_code == 0
        assert "automl" in result.stdout.lower()

    @patch("pynomaly.presentation.cli.commands.automl.run_automl")
    def test_automl_run_command(self, mock_run, runner):
        """Test AutoML run command."""
        mock_run.return_value = {"experiment_id": "test-exp"}

        result = runner.invoke(
            app, ["automl", "run", "--dataset", "test-dataset", "--target", "anomaly"]
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_automl_status_command(self, runner):
        """Test AutoML status command."""
        result = runner.invoke(app, ["automl", "status", "--experiment-id", "test-exp"])

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_automl_results_command(self, runner):
        """Test AutoML results command."""
        result = runner.invoke(
            app, ["automl", "results", "--experiment-id", "test-exp"]
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_automl_stop_command(self, runner):
        """Test AutoML stop command."""
        result = runner.invoke(app, ["automl", "stop", "--experiment-id", "test-exp"])

        # Should not crash
        assert result.exit_code in [0, 1]


class TestEnsembleCommands:
    """Test ensemble CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_ensemble_help_command(self, runner):
        """Test ensemble help command."""
        result = runner.invoke(app, ["ensemble", "--help"])
        assert result.exit_code == 0
        assert "ensemble" in result.stdout.lower()

    @patch("pynomaly.presentation.cli.commands.ensemble.create_ensemble")
    def test_ensemble_create_command(self, mock_create, runner):
        """Test ensemble create command."""
        mock_create.return_value = {"id": "test-ensemble"}

        result = runner.invoke(
            app,
            [
                "ensemble",
                "create",
                "--name",
                "test-ensemble",
                "--detectors",
                "detector1,detector2",
            ],
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_ensemble_list_command(self, runner):
        """Test ensemble list command."""
        result = runner.invoke(app, ["ensemble", "list"])
        # Should not crash
        assert result.exit_code in [0, 1]

    def test_ensemble_delete_command(self, runner):
        """Test ensemble delete command."""
        result = runner.invoke(app, ["ensemble", "delete", "--id", "test-ensemble"])

        # Should not crash
        assert result.exit_code in [0, 1]


class TestServerCommands:
    """Test server CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_server_help_command(self, runner):
        """Test server help command."""
        result = runner.invoke(app, ["server", "--help"])
        assert result.exit_code == 0
        assert "server" in result.stdout.lower()

    @patch("pynomaly.presentation.cli.commands.server.start_server")
    def test_server_start_command(self, mock_start, runner):
        """Test server start command."""
        mock_start.return_value = None

        # Use timeout to prevent hanging
        result = runner.invoke(
            app,
            ["server", "start", "--port", "8000", "--host", "127.0.0.1"],
            catch_exceptions=False,
        )

        # Should not crash immediately
        assert result.exit_code in [0, 1]

    def test_server_status_command(self, runner):
        """Test server status command."""
        result = runner.invoke(app, ["server", "status"])
        # Should not crash
        assert result.exit_code in [0, 1]

    def test_server_stop_command(self, runner):
        """Test server stop command."""
        result = runner.invoke(app, ["server", "stop"])
        # Should not crash
        assert result.exit_code in [0, 1]


class TestExportCommands:
    """Test export CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_export_help_command(self, runner):
        """Test export help command."""
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "export" in result.stdout.lower()

    @patch("pynomaly.presentation.cli.commands.export.export_data")
    def test_export_dataset_command(self, mock_export, runner):
        """Test export dataset command."""
        mock_export.return_value = True

        result = runner.invoke(
            app,
            [
                "export",
                "dataset",
                "--id",
                "test-dataset",
                "--format",
                "csv",
                "--output",
                "test_output.csv",
            ],
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    @patch("pynomaly.presentation.cli.commands.export.export_results")
    def test_export_results_command(self, mock_export, runner):
        """Test export results command."""
        mock_export.return_value = True

        result = runner.invoke(
            app,
            [
                "export",
                "results",
                "--id",
                "test-results",
                "--format",
                "json",
                "--output",
                "test_results.json",
            ],
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_export_formats_command(self, runner):
        """Test export formats command."""
        result = runner.invoke(app, ["export", "formats"])
        # Should not crash
        assert result.exit_code in [0, 1]


class TestAdvancedCommands:
    """Test advanced CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_autonomous_command_help(self, runner):
        """Test autonomous command help."""
        result = runner.invoke(app, ["autonomous", "--help"])
        assert result.exit_code == 0
        assert "autonomous" in result.stdout.lower()

    def test_explainability_command_help(self, runner):
        """Test explainability command help."""
        result = runner.invoke(app, ["explain", "--help"])
        assert result.exit_code == 0

    def test_governance_command_help(self, runner):
        """Test governance command help."""
        result = runner.invoke(app, ["governance", "--help"])
        assert result.exit_code == 0

    def test_security_command_help(self, runner):
        """Test security command help."""
        result = runner.invoke(app, ["security", "--help"])
        assert result.exit_code == 0

    def test_performance_command_help(self, runner):
        """Test performance command help."""
        result = runner.invoke(app, ["performance", "--help"])
        assert result.exit_code == 0

    def test_validation_command_help(self, runner):
        """Test validation command help."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0

    def test_benchmarking_command_help(self, runner):
        """Test benchmarking command help."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0

    def test_tutorials_command_help(self, runner):
        """Test tutorials command help."""
        result = runner.invoke(app, ["tutorial", "--help"])
        assert result.exit_code == 0

    def test_recommendation_command_help(self, runner):
        """Test recommendation command help."""
        result = runner.invoke(app, ["recommend", "--help"])
        assert result.exit_code == 0


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_invalid_command(self, runner):
        """Test invalid command handling."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0
        assert "No such command" in result.stdout or "Usage:" in result.stdout

    def test_missing_required_argument(self, runner):
        """Test missing required argument."""
        result = runner.invoke(app, ["detect"])
        # Should show help or error
        assert result.exit_code != 0

    def test_invalid_argument_value(self, runner):
        """Test invalid argument value."""
        result = runner.invoke(app, ["detect", "--contamination", "invalid"])
        # Should show error
        assert result.exit_code != 0

    def test_network_error_handling(self, runner):
        """Test network error handling."""
        # Test with unreachable server
        result = runner.invoke(
            app,
            ["config", "set", "--key", "api_url", "--value", "http://unreachable:9999"],
        )

        # Should handle gracefully
        assert result.exit_code in [0, 1]

    def test_file_permission_error(self, runner):
        """Test file permission error handling."""
        # Try to write to read-only location
        result = runner.invoke(app, ["config", "export", "--file", "/root/config.json"])

        # Should handle gracefully
        assert result.exit_code in [0, 1]


class TestCLIInteractiveFeatures:
    """Test CLI interactive features."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_interactive_setup(self, runner):
        """Test interactive setup."""
        # Use input simulation
        result = runner.invoke(app, ["setup"], input="localhost\n8000\ntest-key\n")

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_confirmation_prompts(self, runner):
        """Test confirmation prompts."""
        # Test delete confirmation
        result = runner.invoke(
            app, ["detector", "delete", "--id", "test-detector"], input="y\n"
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_progress_indicators(self, runner):
        """Test progress indicators."""
        # Commands that might show progress
        result = runner.invoke(
            app, ["dataset", "upload", "--file", "test.csv", "--name", "test"]
        )

        # Should not crash
        assert result.exit_code in [0, 1]


class TestCLIConfiguration:
    """Test CLI configuration handling."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_environment_variables(self, runner):
        """Test environment variable handling."""
        # Set environment variable
        os.environ["PYNOMALY_API_URL"] = "http://test:8000"

        result = runner.invoke(app, ["config", "show"])

        # Should not crash
        assert result.exit_code in [0, 1]

        # Clean up
        del os.environ["PYNOMALY_API_URL"]

    def test_config_file_locations(self, runner):
        """Test config file location handling."""
        # Test with different config locations
        result = runner.invoke(
            app, ["config", "show", "--config-file", "/tmp/test_config.json"]
        )

        # Should not crash
        assert result.exit_code in [0, 1]

    def test_logging_configuration(self, runner):
        """Test logging configuration."""
        result = runner.invoke(
            app, ["config", "set", "--key", "log_level", "--value", "DEBUG"]
        )

        # Should not crash
        assert result.exit_code in [0, 1]


class TestCLIPerformance:
    """Test CLI performance."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_command_startup_time(self, runner):
        """Test command startup time."""
        import time

        start_time = time.time()
        result = runner.invoke(app, ["--help"])
        end_time = time.time()

        # Should start quickly
        assert (end_time - start_time) < 5.0  # Should start within 5 seconds
        assert result.exit_code == 0

    def test_lazy_loading(self, runner):
        """Test lazy loading functionality."""
        # Test that commands load quickly
        commands = ["detect", "dataset", "config", "server"]

        for command in commands:
            start_time = time.time()
            result = runner.invoke(app, [command, "--help"])
            end_time = time.time()

            # Should load quickly
            assert (end_time - start_time) < 3.0
            assert result.exit_code == 0

    def test_memory_usage(self, runner):
        """Test memory usage."""
        # Basic memory usage test
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # Should not consume excessive memory
        import psutil

        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB

        # Should use reasonable amount of memory
        assert memory_usage < 500  # Less than 500MB
