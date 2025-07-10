"""Comprehensive tests for main CLI app commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import typer
from typer.testing import CliRunner

from pynomaly.presentation.cli.app import app


class TestAppCommands:
    """Test suite for main CLI app commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()
    
    @pytest.fixture
    def mock_container(self):
        """Mock CLI container with all required dependencies."""
        with patch('pynomaly.presentation.cli.app.get_cli_container') as mock_get_container:
            # Create mock container
            container = Mock()
            
            # Mock config
            config = Mock()
            config.app.version = "1.0.0"
            config.app_name = "pynomaly"
            config.version = "1.0.0"
            config.debug = False
            config.storage_path = "/tmp/pynomaly"
            config.api_host = "localhost"
            config.api_port = 8000
            config.max_dataset_size_mb = 1000
            config.default_contamination_rate = 0.1
            config.gpu_enabled = True
            
            # Mock repositories
            detector_repo = Mock()
            detector_repo.count.return_value = 5
            dataset_repo = Mock()
            dataset_repo.count.return_value = 3
            result_repo = Mock()
            result_repo.count.return_value = 12
            
            # Mock recent results
            mock_result = Mock()
            mock_result.detector_id = "detector_1"
            mock_result.dataset_id = "dataset_1"
            mock_result.timestamp = Mock()
            mock_result.timestamp.strftime.return_value = "2025-01-01 12:00"
            mock_result.n_anomalies = 25
            mock_result.anomaly_rate = 0.05
            result_repo.find_recent.return_value = [mock_result]
            
            # Mock detector and dataset for recent results
            mock_detector = Mock()
            mock_detector.name = "Test Detector"
            detector_repo.find_by_id.return_value = mock_detector
            
            mock_dataset = Mock()
            mock_dataset.name = "Test Dataset"
            dataset_repo.find_by_id.return_value = mock_dataset
            
            # Configure container
            container.config.return_value = config
            container.detector_repository.return_value = detector_repo
            container.dataset_repository.return_value = dataset_repo
            container.result_repository.return_value = result_repo
            
            mock_get_container.return_value = container
            return container

    def test_version_command(self, runner, mock_container):
        """Test version command displays correct information."""
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "Pynomaly v1.0.0" in result.stdout
        assert "Python" in result.stdout
        assert "Storage:" in result.stdout
        
        # Verify container was called
        mock_container.config.assert_called_once()

    def test_settings_command_show(self, runner, mock_container):
        """Test settings command with --show flag."""
        result = runner.invoke(app, ["settings", "--show"])
        
        assert result.exit_code == 0
        assert "Pynomaly Settings" in result.stdout
        assert "App Name" in result.stdout
        assert "Version" in result.stdout
        assert "Debug Mode" in result.stdout
        assert "Storage Path" in result.stdout
        assert "API Host" in result.stdout
        assert "API Port" in result.stdout
        assert "Max Dataset Size" in result.stdout
        assert "Default Contamination Rate" in result.stdout
        assert "GPU Enabled" in result.stdout
        
        # Verify container was called
        mock_container.config.assert_called_once()

    def test_settings_command_set_valid(self, runner, mock_container):
        """Test settings command with valid --set parameter."""
        result = runner.invoke(app, ["settings", "--set", "debug=true"])
        
        assert result.exit_code == 0
        assert "Setting update not yet implemented" in result.stdout
        assert "Would set: debug = true" in result.stdout

    def test_settings_command_set_invalid_format(self, runner, mock_container):
        """Test settings command with invalid --set format."""
        result = runner.invoke(app, ["settings", "--set", "debug"])
        
        assert result.exit_code == 1
        assert "Error: Use format: --set key=value" in result.stdout

    def test_settings_command_no_options(self, runner, mock_container):
        """Test settings command without any options."""
        result = runner.invoke(app, ["settings"])
        
        assert result.exit_code == 0
        assert "Use --show to display settings" in result.stdout

    def test_status_command(self, runner, mock_container):
        """Test status command displays system status."""
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "Pynomaly System Status" in result.stdout
        assert "Detectors" in result.stdout
        assert "Datasets" in result.stdout
        assert "Results" in result.stdout
        assert "API Server" in result.stdout
        assert "✓ Active" in result.stdout
        assert "○ Not running" in result.stdout
        assert "Recent Detection Results" in result.stdout
        assert "Test Detector" in result.stdout
        assert "Test Dataset" in result.stdout
        
        # Verify repositories were called
        mock_container.detector_repository.assert_called()
        mock_container.dataset_repository.assert_called()
        mock_container.result_repository.assert_called()

    def test_generate_config_test_json(self, runner, mock_container):
        """Test generate_config command for test configuration in JSON format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_config.json"
            
            result = runner.invoke(app, [
                "generate-config", "test",
                "--output", str(output_path),
                "--format", "json",
                "--detector", "LOF",
                "--contamination", "0.05"
            ])
            
            assert result.exit_code == 0
            assert "Test configuration generated" in result.stdout
            assert "Usage Examples:" in result.stdout
            
            # Verify file was created and has correct content
            assert output_path.exists()
            with open(output_path) as f:
                config = json.load(f)
            
            assert config["metadata"]["type"] == "test"
            assert config["test"]["detector"]["algorithm"] == "LOF"
            assert config["test"]["detector"]["parameters"]["contamination"] == 0.05
            assert "examples" in config

    def test_generate_config_experiment_yaml(self, runner, mock_container):
        """Test generate_config command for experiment configuration in YAML format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "experiment_config.yaml"
            
            result = runner.invoke(app, [
                "generate-config", "experiment",
                "--output", str(output_path),
                "--format", "yaml",
                "--contamination", "0.08",
                "--cv", "true",
                "--folds", "3"
            ])
            
            assert result.exit_code == 0
            assert "Experiment configuration generated" in result.stdout
            assert output_path.exists()

    def test_generate_config_autonomous(self, runner, mock_container):
        """Test generate_config command for autonomous configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "autonomous_config.json"
            
            result = runner.invoke(app, [
                "generate-config", "autonomous",
                "--output", str(output_path),
                "--max-algorithms", "7",
                "--auto-tune", "true",
                "--verbose", "true"
            ])
            
            assert result.exit_code == 0
            assert "Autonomous configuration generated" in result.stdout
            
            # Verify file content
            with open(output_path) as f:
                config = json.load(f)
            
            assert config["metadata"]["type"] == "autonomous"
            assert config["autonomous"]["detection"]["max_algorithms"] == 7
            assert config["autonomous"]["detection"]["auto_tune_hyperparams"] == True
            assert config["autonomous"]["output"]["verbose"] == True

    def test_generate_config_invalid_type(self, runner, mock_container):
        """Test generate_config command with invalid config type."""
        result = runner.invoke(app, ["generate-config", "invalid"])
        
        assert result.exit_code == 1
        assert "Unknown config type 'invalid'" in result.stdout

    def test_generate_config_no_examples(self, runner, mock_container):
        """Test generate_config command with --no-examples flag."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "config.json"
            
            result = runner.invoke(app, [
                "generate-config", "test",
                "--output", str(output_path),
                "--no-examples"
            ])
            
            assert result.exit_code == 0
            assert "Usage Examples:" not in result.stdout
            
            # Verify examples are not included
            with open(output_path) as f:
                config = json.load(f)
            
            assert "examples" not in config

    @patch('typer.confirm')
    def test_quickstart_command_confirmed(self, mock_confirm, runner, mock_container):
        """Test quickstart command when user confirms."""
        mock_confirm.return_value = True
        
        result = runner.invoke(app, ["quickstart"])
        
        assert result.exit_code == 0
        assert "Welcome to Pynomaly!" in result.stdout
        assert "Step 1: Load a dataset" in result.stdout
        assert "Step 2: Clean and preprocess data" in result.stdout
        assert "Step 3: Create a detector" in result.stdout
        assert "Step 4: Train the detector" in result.stdout
        assert "Step 5: Detect anomalies" in result.stdout
        assert "Step 6: View and export results" in result.stdout
        assert "Ready to start!" in result.stdout
        
        mock_confirm.assert_called_once()

    @patch('typer.confirm')
    def test_quickstart_command_cancelled(self, mock_confirm, runner, mock_container):
        """Test quickstart command when user cancels."""
        mock_confirm.return_value = False
        
        result = runner.invoke(app, ["quickstart"])
        
        assert result.exit_code == 1
        assert "Quickstart cancelled." in result.stdout
        mock_confirm.assert_called_once()

    @patch('pynomaly.presentation.cli.ux_improvements.create_setup_wizard')
    def test_setup_command_success(self, mock_wizard, runner, mock_container):
        """Test setup command when wizard completes successfully."""
        mock_config = {
            'data_path': '/path/to/data.csv',
            'algorithm': 'IsolationForest',
            'contamination': 0.1,
            'output_format': 'csv'
        }
        mock_wizard.return_value = mock_config
        
        result = runner.invoke(app, ["setup"])
        
        assert result.exit_code == 0
        assert "Commands to run:" in result.stdout
        assert "pynomaly dataset load /path/to/data.csv" in result.stdout
        assert "pynomaly detector create my-detector --algorithm IsolationForest" in result.stdout
        assert "pynomaly detect train my-detector my-dataset" in result.stdout
        assert "pynomaly detect run my-detector my-dataset --output results.csv" in result.stdout
        
        mock_wizard.assert_called_once()

    @patch('pynomaly.presentation.cli.ux_improvements.create_setup_wizard')
    def test_setup_command_failure(self, mock_wizard, runner, mock_container):
        """Test setup command when wizard fails."""
        mock_wizard.side_effect = Exception("Setup wizard failed")
        
        result = runner.invoke(app, ["setup"])
        
        assert result.exit_code == 1
        assert "Setup wizard failed: Setup wizard failed" in result.stdout
        mock_wizard.assert_called_once()

    @patch('pynomaly.presentation.cli.ux_improvements.create_setup_wizard')
    def test_setup_command_no_config(self, mock_wizard, runner, mock_container):
        """Test setup command when wizard returns no config."""
        mock_wizard.return_value = None
        
        result = runner.invoke(app, ["setup"])
        
        assert result.exit_code == 0
        assert "Commands to run:" not in result.stdout
        mock_wizard.assert_called_once()

    def test_main_callback_verbose_and_quiet(self, runner, mock_container):
        """Test main callback with conflicting verbose and quiet flags."""
        result = runner.invoke(app, ["--verbose", "--quiet", "version"])
        
        assert result.exit_code == 1
        assert "Cannot use --verbose and --quiet together" in result.stdout

    def test_main_callback_verbose_only(self, runner, mock_container):
        """Test main callback with verbose flag only."""
        result = runner.invoke(app, ["--verbose", "version"])
        
        assert result.exit_code == 0
        assert "Pynomaly v1.0.0" in result.stdout

    def test_main_callback_quiet_only(self, runner, mock_container):
        """Test main callback with quiet flag only."""
        result = runner.invoke(app, ["--quiet", "version"])
        
        assert result.exit_code == 0
        # In quiet mode, version should still show but other output suppressed


class TestAppCommandsIntegration:
    """Integration tests for CLI app commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_help_command(self, runner):
        """Test that help command works."""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Pynomaly - State-of-the-art anomaly detection" in result.stdout
        assert "Commands:" in result.stdout

    def test_version_help(self, runner):
        """Test version command help."""
        result = runner.invoke(app, ["version", "--help"])
        
        assert result.exit_code == 0
        assert "Show version information" in result.stdout

    def test_settings_help(self, runner):
        """Test settings command help."""
        result = runner.invoke(app, ["settings", "--help"])
        
        assert result.exit_code == 0
        assert "Manage application settings" in result.stdout
        assert "--show" in result.stdout
        assert "--set" in result.stdout

    def test_status_help(self, runner):
        """Test status command help."""
        result = runner.invoke(app, ["status", "--help"])
        
        assert result.exit_code == 0
        assert "Show system status" in result.stdout

    def test_generate_config_help(self, runner):
        """Test generate-config command help."""
        result = runner.invoke(app, ["generate-config", "--help"])
        
        assert result.exit_code == 0
        assert "Generate configuration files" in result.stdout
        assert "--output" in result.stdout
        assert "--format" in result.stdout

    def test_quickstart_help(self, runner):
        """Test quickstart command help."""
        result = runner.invoke(app, ["quickstart", "--help"])
        
        assert result.exit_code == 0
        assert "Run interactive quickstart guide" in result.stdout

    def test_setup_help(self, runner):
        """Test setup command help."""
        result = runner.invoke(app, ["setup", "--help"])
        
        assert result.exit_code == 0
        assert "Interactive setup wizard" in result.stdout


class TestAppCommandsErrorHandling:
    """Test error handling for CLI app commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_generate_config_file_write_error(self, runner):
        """Test generate_config command with file write permission error."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            result = runner.invoke(app, ["generate-config", "test"])
            
            assert result.exit_code == 1
            assert "Failed to save config: Permission denied" in result.stdout

    @patch('pynomaly.presentation.cli.app.get_cli_container')
    def test_version_command_container_error(self, mock_get_container, runner):
        """Test version command when container fails."""
        mock_get_container.side_effect = Exception("Container initialization failed")
        
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code != 0

    @patch('pynomaly.presentation.cli.app.get_cli_container')
    def test_settings_command_container_error(self, mock_get_container, runner):
        """Test settings command when container fails."""
        mock_get_container.side_effect = Exception("Container initialization failed")
        
        result = runner.invoke(app, ["settings", "--show"])
        
        assert result.exit_code != 0

    @patch('pynomaly.presentation.cli.app.get_cli_container')
    def test_status_command_repository_error(self, mock_get_container, runner):
        """Test status command when repository calls fail."""
        container = Mock()
        container.detector_repository.side_effect = Exception("Repository error")
        mock_get_container.return_value = container
        
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code != 0

    def test_generate_config_invalid_yaml_format(self, runner):
        """Test generate_config command with invalid YAML format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test.yaml"
            
            with patch('yaml.dump', side_effect=Exception("YAML error")):
                result = runner.invoke(app, [
                    "generate-config", "test",
                    "--output", str(output_path),
                    "--format", "yaml"
                ])
                
                assert result.exit_code == 1
                assert "Failed to save config: YAML error" in result.stdout


class TestAppCommandsParameterValidation:
    """Test parameter validation for CLI app commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_generate_config_contamination_range(self, runner):
        """Test generate_config command with various contamination values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test valid contamination
            output_path = Path(tmp_dir) / "test1.json"
            result = runner.invoke(app, [
                "generate-config", "test",
                "--output", str(output_path),
                "--contamination", "0.05"
            ])
            assert result.exit_code == 0
            
            # Test edge case contamination
            output_path = Path(tmp_dir) / "test2.json"
            result = runner.invoke(app, [
                "generate-config", "test",
                "--output", str(output_path),
                "--contamination", "0.5"
            ])
            assert result.exit_code == 0

    def test_generate_config_cv_folds_validation(self, runner):
        """Test generate_config command with various CV folds values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test valid folds
            output_path = Path(tmp_dir) / "test1.json"
            result = runner.invoke(app, [
                "generate-config", "experiment",
                "--output", str(output_path),
                "--folds", "10"
            ])
            assert result.exit_code == 0
            
            # Test minimum folds
            output_path = Path(tmp_dir) / "test2.json"
            result = runner.invoke(app, [
                "generate-config", "experiment",
                "--output", str(output_path),
                "--folds", "2"
            ])
            assert result.exit_code == 0

    def test_generate_config_max_algorithms_validation(self, runner):
        """Test generate_config command with various max algorithms values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test valid max algorithms
            output_path = Path(tmp_dir) / "test1.json"
            result = runner.invoke(app, [
                "generate-config", "autonomous",
                "--output", str(output_path),
                "--max-algorithms", "3"
            ])
            assert result.exit_code == 0
            
            # Test higher max algorithms
            output_path = Path(tmp_dir) / "test2.json"
            result = runner.invoke(app, [
                "generate-config", "autonomous",
                "--output", str(output_path),
                "--max-algorithms", "15"
            ])
            assert result.exit_code == 0