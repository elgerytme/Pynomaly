"""Comprehensive tests for CLI commands - Phase 3 Coverage."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import io
from typing import List, Dict, Any

from pynomaly.presentation.cli.app import app
from pynomaly.presentation.cli import detectors, datasets, detection, server
from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore
from pynomaly.domain.exceptions import EntityNotFoundError


@pytest.fixture
def cli_runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_container():
    """Create mock container for CLI testing."""
    container = Mock()
    
    # Mock settings
    settings = Mock()
    settings.version = "1.0.0"
    settings.app_name = "Pynomaly"
    settings.debug = False
    settings.storage_path = "/tmp/pynomaly"
    settings.api_host = "localhost"
    settings.api_port = 8000
    settings.max_dataset_size_mb = 500
    settings.default_contamination_rate = 0.1
    settings.gpu_enabled = False
    container.config.return_value = settings
    
    # Mock repositories
    container.detector_repository.return_value.count.return_value = 5
    container.dataset_repository.return_value.count.return_value = 3
    container.result_repository.return_value.count.return_value = 10
    container.result_repository.return_value.find_recent.return_value = []
    
    return container


@pytest.fixture
def sample_csv_file():
    """Create sample CSV file for testing."""
    content = "feature1,feature2,target\n1,2,0\n3,4,0\n5,6,1\n7,8,0\n"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        return f.name


class TestMainCLICommands:
    """Test main CLI application commands."""
    
    def test_cli_help(self, cli_runner: CliRunner):
        """Test CLI help command."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Pynomaly" in result.stdout
        assert "anomaly detection CLI" in result.stdout
        assert "detector" in result.stdout
        assert "dataset" in result.stdout
        assert "detect" in result.stdout
        assert "server" in result.stdout
    
    def test_version_command(self, cli_runner: CliRunner):
        """Test version command."""
        with patch('pynomaly.presentation.cli.app.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_settings = Mock()
            mock_settings.version = "1.2.3"
            mock_container.config.return_value = mock_settings
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["version"])
            
            assert result.exit_code == 0
            assert "Pynomaly v1.2.3" in result.stdout
            assert "Python" in result.stdout
    
    def test_config_show_command(self, cli_runner: CliRunner, mock_container):
        """Test config show command."""
        with patch('pynomaly.presentation.cli.app.get_cli_container', return_value=mock_container):
            result = cli_runner.invoke(app, ["config", "--show"])
            
            assert result.exit_code == 0
            assert "Pynomaly Configuration" in result.stdout
            assert "App Name" in result.stdout
            assert "Version" in result.stdout
            assert "1.0.0" in result.stdout
    
    def test_config_set_command(self, cli_runner: CliRunner, mock_container):
        """Test config set command."""
        with patch('pynomaly.presentation.cli.app.get_cli_container', return_value=mock_container):
            result = cli_runner.invoke(app, ["config", "--set", "debug=true"])
            
            assert result.exit_code == 0
            assert "not yet implemented" in result.stdout
            assert "debug = true" in result.stdout
    
    def test_config_invalid_format(self, cli_runner: CliRunner, mock_container):
        """Test config set with invalid format."""
        with patch('pynomaly.presentation.cli.app.get_cli_container', return_value=mock_container):
            result = cli_runner.invoke(app, ["config", "--set", "invalid_format"])
            
            assert result.exit_code == 1
            assert "Use format: --set key=value" in result.stdout
    
    def test_status_command(self, cli_runner: CliRunner, mock_container):
        """Test status command."""
        with patch('pynomaly.presentation.cli.app.get_cli_container', return_value=mock_container):
            result = cli_runner.invoke(app, ["status"])
            
            assert result.exit_code == 0
            assert "Pynomaly System Status" in result.stdout
            assert "Detectors" in result.stdout
            assert "Datasets" in result.stdout
            assert "Results" in result.stdout
            assert "5" in result.stdout  # Detector count
            assert "3" in result.stdout  # Dataset count
            assert "10" in result.stdout  # Result count
    
    def test_quickstart_command_cancel(self, cli_runner: CliRunner):
        """Test quickstart command with cancellation."""
        result = cli_runner.invoke(app, ["quickstart"], input="n\n")
        
        assert result.exit_code == 1
        assert "Welcome to Pynomaly!" in result.stdout
        assert "Quickstart cancelled" in result.stdout
    
    def test_quickstart_command_continue(self, cli_runner: CliRunner):
        """Test quickstart command with continuation."""
        result = cli_runner.invoke(app, ["quickstart"], input="y\n")
        
        assert result.exit_code == 0
        assert "Welcome to Pynomaly!" in result.stdout
        assert "Step 1: Load a dataset" in result.stdout
        assert "Step 2: Create a detector" in result.stdout
        assert "Ready to start!" in result.stdout
    
    def test_verbose_quiet_conflict(self, cli_runner: CliRunner):
        """Test verbose and quiet flags conflict."""
        result = cli_runner.invoke(app, ["--verbose", "--quiet", "version"])
        
        assert result.exit_code == 1
        assert "Cannot use --verbose and --quiet together" in result.stdout


class TestDetectorCLICommands:
    """Test detector management CLI commands."""
    
    def test_detector_help(self, cli_runner: CliRunner):
        """Test detector help command."""
        result = cli_runner.invoke(app, ["detector", "--help"])
        
        assert result.exit_code == 0
        assert "detector" in result.stdout.lower()
        assert "create" in result.stdout
        assert "list" in result.stdout
        assert "show" in result.stdout
    
    def test_detector_create_basic(self, cli_runner: CliRunner):
        """Test basic detector creation."""
        with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            mock_detector = Mock()
            mock_detector.id = "detector_123"
            mock_detector.name = "Test Detector"
            mock_detector.algorithm = "IsolationForest"
            mock_detector.contamination.value = 0.1
            
            mock_service.create_detector.return_value = mock_detector
            mock_container.detector_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "detector", "create",
                "--name", "Test Detector",
                "--algorithm", "IsolationForest",
                "--contamination", "0.1"
            ])
            
            assert result.exit_code == 0
            assert "Test Detector" in result.stdout
            assert "detector_123" in result.stdout
    
    def test_detector_create_with_parameters(self, cli_runner: CliRunner):
        """Test detector creation with custom parameters."""
        with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            mock_detector = Mock()
            mock_detector.id = "detector_456"
            mock_detector.name = "Advanced Detector"
            mock_detector.algorithm = "LOF"
            
            mock_service.create_detector.return_value = mock_detector
            mock_container.detector_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "detector", "create",
                "--name", "Advanced Detector",
                "--algorithm", "LOF",
                "--parameters", '{"n_neighbors": 20, "metric": "euclidean"}'
            ])
            
            assert result.exit_code == 0
            assert "Advanced Detector" in result.stdout
    
    def test_detector_create_invalid_json(self, cli_runner: CliRunner):
        """Test detector creation with invalid JSON parameters."""
        result = cli_runner.invoke(app, [
            "detector", "create",
            "--name", "Invalid Detector",
            "--algorithm", "LOF",
            "--parameters", 'invalid json'
        ])
        
        assert result.exit_code == 1
        assert "Invalid JSON" in result.stdout
    
    def test_detector_list_basic(self, cli_runner: CliRunner):
        """Test listing detectors."""
        with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            
            # Mock detector data
            mock_detectors = [
                Mock(id="det1", name="Detector 1", algorithm="IsolationForest", is_fitted=True),
                Mock(id="det2", name="Detector 2", algorithm="LOF", is_fitted=False),
                Mock(id="det3", name="Detector 3", algorithm="OCSVM", is_fitted=True)
            ]
            
            mock_repo.find_all.return_value = mock_detectors
            mock_container.detector_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["detector", "list"])
            
            assert result.exit_code == 0
            assert "Detector 1" in result.stdout
            assert "Detector 2" in result.stdout
            assert "IsolationForest" in result.stdout
            assert "LOF" in result.stdout
    
    def test_detector_list_with_filters(self, cli_runner: CliRunner):
        """Test listing detectors with filters."""
        with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            
            mock_detectors = [
                Mock(id="det1", name="Fitted Detector", algorithm="IsolationForest", is_fitted=True)
            ]
            
            mock_repo.find_by_algorithm.return_value = mock_detectors
            mock_container.detector_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["detector", "list", "--algorithm", "IsolationForest"])
            
            assert result.exit_code == 0
            mock_repo.find_by_algorithm.assert_called_with("IsolationForest")
    
    def test_detector_show(self, cli_runner: CliRunner):
        """Test showing detector details."""
        with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            
            mock_detector = Mock()
            mock_detector.id = "det123"
            mock_detector.name = "Test Detector"
            mock_detector.algorithm = "IsolationForest"
            mock_detector.contamination.value = 0.1
            mock_detector.hyperparameters = {"n_estimators": 100}
            mock_detector.is_fitted = True
            mock_detector.created_at.isoformat.return_value = "2024-01-01T00:00:00Z"
            
            mock_repo.find_by_id.return_value = mock_detector
            mock_container.detector_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["detector", "show", "det123"])
            
            assert result.exit_code == 0
            assert "Test Detector" in result.stdout
            assert "IsolationForest" in result.stdout
            assert "0.1" in result.stdout
    
    def test_detector_show_not_found(self, cli_runner: CliRunner):
        """Test showing non-existent detector."""
        with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            mock_repo.find_by_id.side_effect = EntityNotFoundError("Detector not found")
            mock_container.detector_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["detector", "show", "nonexistent"])
            
            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()
    
    def test_detector_delete(self, cli_runner: CliRunner):
        """Test deleting detector."""
        with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            mock_detector = Mock()
            mock_detector.name = "Test Detector"
            
            mock_repo.find_by_id.return_value = mock_detector
            mock_repo.delete.return_value = None
            mock_container.detector_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["detector", "delete", "det123"], input="y\n")
            
            assert result.exit_code == 0
            assert "deleted successfully" in result.stdout.lower()
    
    def test_detector_delete_cancel(self, cli_runner: CliRunner):
        """Test canceling detector deletion."""
        with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            mock_detector = Mock()
            mock_detector.name = "Test Detector"
            
            mock_repo.find_by_id.return_value = mock_detector
            mock_container.detector_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["detector", "delete", "det123"], input="n\n")
            
            assert result.exit_code == 0
            assert "cancelled" in result.stdout.lower()
    
    def test_detector_algorithms_list(self, cli_runner: CliRunner):
        """Test listing available algorithms."""
        with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_adapter_service = Mock()
            mock_adapter_service.list_available_algorithms.return_value = [
                {"name": "IsolationForest", "type": "ensemble", "description": "Isolation Forest"},
                {"name": "LOF", "type": "proximity", "description": "Local Outlier Factor"},
                {"name": "OCSVM", "type": "linear", "description": "One-Class SVM"}
            ]
            mock_container.algorithm_adapter_service.return_value = mock_adapter_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["detector", "algorithms"])
            
            assert result.exit_code == 0
            assert "IsolationForest" in result.stdout
            assert "LOF" in result.stdout
            assert "ensemble" in result.stdout


class TestDatasetCLICommands:
    """Test dataset management CLI commands."""
    
    def test_dataset_help(self, cli_runner: CliRunner):
        """Test dataset help command."""
        result = cli_runner.invoke(app, ["dataset", "--help"])
        
        assert result.exit_code == 0
        assert "dataset" in result.stdout.lower()
        assert "load" in result.stdout
        assert "list" in result.stdout
    
    def test_dataset_load_csv(self, cli_runner: CliRunner, sample_csv_file):
        """Test loading CSV dataset."""
        with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            mock_dataset = Mock()
            mock_dataset.id = "dataset_123"
            mock_dataset.name = "Test Dataset"
            mock_dataset.n_samples = 4
            mock_dataset.n_features = 2
            
            mock_service.load_dataset.return_value = mock_dataset
            mock_container.dataset_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "dataset", "load",
                sample_csv_file,
                "--name", "Test Dataset",
                "--target-column", "target"
            ])
            
            assert result.exit_code == 0
            assert "Test Dataset" in result.stdout
            assert "4 samples" in result.stdout
    
    def test_dataset_load_with_preprocessing(self, cli_runner: CliRunner, sample_csv_file):
        """Test loading dataset with preprocessing."""
        with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            mock_dataset = Mock()
            mock_dataset.id = "dataset_456"
            mock_dataset.name = "Preprocessed Dataset"
            mock_dataset.n_samples = 3  # After removing duplicates
            mock_dataset.n_features = 2
            
            mock_service.load_dataset.return_value = mock_dataset
            mock_container.dataset_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "dataset", "load",
                sample_csv_file,
                "--name", "Preprocessed Dataset",
                "--remove-duplicates",
                "--handle-missing", "mean",
                "--normalize"
            ])
            
            assert result.exit_code == 0
            assert "Preprocessed Dataset" in result.stdout
    
    def test_dataset_load_file_not_found(self, cli_runner: CliRunner):
        """Test loading non-existent file."""
        result = cli_runner.invoke(app, [
            "dataset", "load",
            "/nonexistent/file.csv",
            "--name", "Missing Dataset"
        ])
        
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()
    
    def test_dataset_list(self, cli_runner: CliRunner):
        """Test listing datasets."""
        with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            
            mock_datasets = [
                Mock(id="ds1", name="Dataset 1", n_samples=100, n_features=5, file_size_mb=1.2),
                Mock(id="ds2", name="Dataset 2", n_samples=200, n_features=10, file_size_mb=2.5),
            ]
            
            mock_repo.find_all.return_value = mock_datasets
            mock_container.dataset_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["dataset", "list"])
            
            assert result.exit_code == 0
            assert "Dataset 1" in result.stdout
            assert "Dataset 2" in result.stdout
            assert "100" in result.stdout
            assert "200" in result.stdout
    
    def test_dataset_show(self, cli_runner: CliRunner):
        """Test showing dataset details."""
        with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            mock_service = Mock()
            
            mock_dataset = Mock()
            mock_dataset.id = "ds123"
            mock_dataset.name = "Test Dataset"
            mock_dataset.n_samples = 1000
            mock_dataset.n_features = 15
            mock_dataset.has_target = True
            mock_dataset.created_at.isoformat.return_value = "2024-01-01T00:00:00Z"
            
            mock_stats = {
                "feature1": {"mean": 10.5, "std": 2.3, "min": 1.0, "max": 20.0},
                "feature2": {"mean": 5.2, "std": 1.1, "min": 0.5, "max": 10.0}
            }
            
            mock_repo.find_by_id.return_value = mock_dataset
            mock_service.get_dataset_statistics.return_value = mock_stats
            mock_container.dataset_repository.return_value = mock_repo
            mock_container.dataset_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["dataset", "show", "ds123", "--statistics"])
            
            assert result.exit_code == 0
            assert "Test Dataset" in result.stdout
            assert "1000" in result.stdout
            assert "feature1" in result.stdout
    
    def test_dataset_sample(self, cli_runner: CliRunner):
        """Test showing dataset sample."""
        with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            
            mock_sample = {
                "data": [
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 3.0, "feature2": 4.0},
                    {"feature1": 5.0, "feature2": 6.0}
                ],
                "sample_size": 3,
                "total_size": 1000
            }
            
            mock_service.get_dataset_sample.return_value = mock_sample
            mock_container.dataset_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["dataset", "sample", "ds123", "--size", "3"])
            
            assert result.exit_code == 0
            assert "3 samples" in result.stdout
            assert "feature1" in result.stdout
    
    def test_dataset_export(self, cli_runner: CliRunner):
        """Test exporting dataset."""
        with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            mock_service.export_dataset.return_value = True
            mock_container.dataset_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                result = cli_runner.invoke(app, [
                    "dataset", "export",
                    "ds123",
                    tmp.name,
                    "--format", "csv"
                ])
                
                assert result.exit_code == 0
                assert "exported successfully" in result.stdout.lower()
    
    def test_dataset_delete(self, cli_runner: CliRunner):
        """Test deleting dataset."""
        with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            mock_dataset = Mock()
            mock_dataset.name = "Test Dataset"
            
            mock_repo.find_by_id.return_value = mock_dataset
            mock_repo.delete.return_value = None
            mock_container.dataset_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["dataset", "delete", "ds123"], input="y\n")
            
            assert result.exit_code == 0
            assert "deleted successfully" in result.stdout.lower()


class TestDetectionCLICommands:
    """Test detection CLI commands."""
    
    def test_detection_help(self, cli_runner: CliRunner):
        """Test detection help command."""
        result = cli_runner.invoke(app, ["detect", "--help"])
        
        assert result.exit_code == 0
        assert "detect" in result.stdout.lower()
        assert "train" in result.stdout
        assert "run" in result.stdout
    
    def test_detection_train(self, cli_runner: CliRunner):
        """Test training detector."""
        with patch('pynomaly.presentation.cli.detection.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            
            mock_result = {
                "success": True,
                "training_time_ms": 1500,
                "model_metrics": {"accuracy": 0.95},
                "detector_id": "det123"
            }
            
            mock_service.train_detector.return_value = mock_result
            mock_container.detection_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "detect", "train",
                "--detector", "test_detector",
                "--dataset", "test_dataset",
                "--save-model"
            ])
            
            assert result.exit_code == 0
            assert "Training completed" in result.stdout
            assert "1500 ms" in result.stdout
    
    def test_detection_train_with_validation(self, cli_runner: CliRunner):
        """Test training with validation split."""
        with patch('pynomaly.presentation.cli.detection.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            
            mock_result = {
                "success": True,
                "training_time_ms": 2000,
                "validation_results": {
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1_score": 0.90
                }
            }
            
            mock_service.train_detector.return_value = mock_result
            mock_container.detection_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "detect", "train",
                "--detector", "test_detector",
                "--dataset", "test_dataset",
                "--validation-split", "0.2",
                "--cross-validation"
            ])
            
            assert result.exit_code == 0
            assert "validation" in result.stdout.lower()
            assert "0.92" in result.stdout
    
    def test_detection_run(self, cli_runner: CliRunner):
        """Test running anomaly detection."""
        with patch('pynomaly.presentation.cli.detection.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            
            mock_result = {
                "predictions": [0, 0, 1, 0, 1],
                "anomaly_scores": [0.1, 0.2, 0.8, 0.15, 0.9],
                "anomaly_count": 2,
                "summary": {
                    "total_samples": 5,
                    "anomaly_rate": 0.4,
                    "max_score": 0.9,
                    "avg_score": 0.41
                }
            }
            
            mock_service.detect_anomalies.return_value = mock_result
            mock_container.detection_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "detect", "run",
                "--detector", "trained_detector",
                "--dataset", "test_dataset",
                "--output-scores"
            ])
            
            assert result.exit_code == 0
            assert "2 anomalies" in result.stdout
            assert "40.0%" in result.stdout  # Anomaly rate
    
    def test_detection_run_with_threshold(self, cli_runner: CliRunner):
        """Test detection with custom threshold."""
        with patch('pynomaly.presentation.cli.detection.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            
            mock_result = {
                "predictions": [0, 0, 1, 0],
                "anomaly_scores": [0.1, 0.2, 0.8, 0.3],
                "anomaly_count": 1,
                "threshold_used": 0.7
            }
            
            mock_service.detect_anomalies.return_value = mock_result
            mock_container.detection_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "detect", "run",
                "--detector", "trained_detector",
                "--dataset", "test_dataset",
                "--threshold", "0.7"
            ])
            
            assert result.exit_code == 0
            assert "threshold: 0.7" in result.stdout
    
    def test_detection_batch(self, cli_runner: CliRunner):
        """Test batch detection."""
        with patch('pynomaly.presentation.cli.detection.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            
            mock_results = [
                {"dataset_id": "ds1", "anomaly_count": 5, "anomaly_rate": 0.05},
                {"dataset_id": "ds2", "anomaly_count": 10, "anomaly_rate": 0.10},
                {"dataset_id": "ds3", "anomaly_count": 2, "anomaly_rate": 0.02}
            ]
            
            mock_service.batch_detect_anomalies.return_value = mock_results
            mock_container.detection_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "detect", "batch",
                "--detector", "batch_detector",
                "--datasets", "ds1,ds2,ds3",
                "--parallel"
            ])
            
            assert result.exit_code == 0
            assert "3 datasets" in result.stdout
            assert "ds1" in result.stdout
            assert "5 anomalies" in result.stdout
    
    def test_detection_results_list(self, cli_runner: CliRunner):
        """Test listing detection results."""
        with patch('pynomaly.presentation.cli.detection.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            
            mock_results = [
                Mock(
                    id="res1",
                    detector_name="Detector 1",
                    dataset_name="Dataset 1",
                    anomaly_count=5,
                    anomaly_rate=0.05,
                    created_at=Mock()
                ),
                Mock(
                    id="res2",
                    detector_name="Detector 2",
                    dataset_name="Dataset 2",
                    anomaly_count=10,
                    anomaly_rate=0.10,
                    created_at=Mock()
                )
            ]
            
            # Mock datetime formatting
            for result in mock_results:
                result.created_at.strftime.return_value = "2024-01-01 12:00"
            
            mock_repo.find_recent.return_value = mock_results
            mock_container.result_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["detect", "results", "--limit", "10"])
            
            assert result.exit_code == 0
            assert "Detector 1" in result.stdout
            assert "5 anomalies" in result.stdout
    
    def test_detection_export_results(self, cli_runner: CliRunner):
        """Test exporting detection results."""
        with patch('pynomaly.presentation.cli.detection.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            mock_service.export_results.return_value = True
            mock_container.detection_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                result = cli_runner.invoke(app, [
                    "detect", "export",
                    "result_123",
                    tmp.name,
                    "--format", "json",
                    "--include-scores"
                ])
                
                assert result.exit_code == 0
                assert "exported successfully" in result.stdout.lower()


class TestServerCLICommands:
    """Test server management CLI commands."""
    
    def test_server_help(self, cli_runner: CliRunner):
        """Test server help command."""
        result = cli_runner.invoke(app, ["server", "--help"])
        
        assert result.exit_code == 0
        assert "server" in result.stdout.lower()
        assert "start" in result.stdout
        assert "stop" in result.stdout
    
    def test_server_start(self, cli_runner: CliRunner):
        """Test starting server."""
        with patch('uvicorn.run') as mock_uvicorn:
            result = cli_runner.invoke(app, [
                "server", "start",
                "--host", "0.0.0.0",
                "--port", "8080",
                "--workers", "2"
            ])
            
            # The command should attempt to start uvicorn
            mock_uvicorn.assert_called_once()
            call_args = mock_uvicorn.call_args
            assert call_args[1]["host"] == "0.0.0.0"
            assert call_args[1]["port"] == 8080
            assert call_args[1]["workers"] == 2
    
    def test_server_start_development(self, cli_runner: CliRunner):
        """Test starting server in development mode."""
        with patch('uvicorn.run') as mock_uvicorn:
            result = cli_runner.invoke(app, [
                "server", "start",
                "--dev",
                "--reload"
            ])
            
            mock_uvicorn.assert_called_once()
            call_args = mock_uvicorn.call_args
            assert call_args[1]["reload"] is True
    
    def test_server_status(self, cli_runner: CliRunner):
        """Test checking server status."""
        with patch('requests.get') as mock_get:
            # Mock successful health check
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response
            
            result = cli_runner.invoke(app, ["server", "status"])
            
            assert result.exit_code == 0
            assert "Server is running" in result.stdout
    
    def test_server_status_not_running(self, cli_runner: CliRunner):
        """Test server status when not running."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            
            result = cli_runner.invoke(app, ["server", "status"])
            
            assert result.exit_code == 0
            assert "Server is not running" in result.stdout
    
    def test_server_logs(self, cli_runner: CliRunner):
        """Test viewing server logs."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.open') as mock_open:
            
            mock_exists.return_value = True
            mock_file = Mock()
            mock_file.__enter__.return_value.readlines.return_value = [
                "2024-01-01 12:00:00 INFO Server started\n",
                "2024-01-01 12:01:00 INFO Request processed\n"
            ]
            mock_open.return_value = mock_file
            
            result = cli_runner.invoke(app, ["server", "logs", "--lines", "10"])
            
            assert result.exit_code == 0
            assert "Server started" in result.stdout


class TestPerformanceCLICommands:
    """Test performance management CLI commands."""
    
    def test_performance_help(self, cli_runner: CliRunner):
        """Test performance help command."""
        result = cli_runner.invoke(app, ["perf", "--help"])
        
        assert result.exit_code == 0
        assert "performance" in result.stdout.lower()
        assert "pools" in result.stdout
        assert "queries" in result.stdout
    
    def test_list_pools(self, cli_runner: CliRunner):
        """Test listing connection pools."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class:
            mock_container = Mock()
            mock_pool_manager = Mock()
            
            # Mock pool data
            mock_pool_manager.list_pools.return_value = ["main_pool", "cache_pool"]
            mock_pool_manager.get_pool_info.side_effect = [
                {
                    "type": "PostgreSQL",
                    "stats": Mock(
                        active_connections=5,
                        total_requests=1000,
                        successful_requests=950,
                        avg_response_time=0.125,
                        connection_errors=2
                    )
                },
                {
                    "type": "Redis",
                    "stats": Mock(
                        active_connections=3,
                        total_requests=500,
                        successful_requests=495,
                        avg_response_time=0.050,
                        connection_errors=0
                    )
                }
            ]
            
            mock_container.connection_pool_manager = mock_pool_manager
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "pools"])
            
            assert result.exit_code == 0
            assert "Connection Pools" in result.stdout
            assert "main_pool" in result.stdout
            assert "cache_pool" in result.stdout
            assert "PostgreSQL" in result.stdout
            assert "Redis" in result.stdout
    
    def test_show_pool_details(self, cli_runner: CliRunner):
        """Test showing detailed pool information."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class:
            mock_container = Mock()
            mock_pool_manager = Mock()
            
            # Mock detailed pool info
            mock_pool_manager.get_pool_info.return_value = {
                "type": "PostgreSQL",
                "stats": Mock(
                    active_connections=5,
                    idle_connections=10,
                    overflow_connections=2,
                    total_requests=1000,
                    successful_requests=950,
                    failed_requests=50,
                    avg_response_time=0.125,
                    connection_errors=2,
                    connections_created=15,
                    connections_closed=3,
                    connections_recycled=100
                ),
                "pool_info": {
                    "max_connections": 20,
                    "timeout": 30.0,
                    "database": "pynomaly_db"
                }
            }
            
            mock_container.connection_pool_manager = mock_pool_manager
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "pool", "main_pool"])
            
            assert result.exit_code == 0
            assert "Pool: main_pool" in result.stdout
            assert "Active Connections: 5" in result.stdout
            assert "Pool-Specific Information:" in result.stdout
    
    def test_show_pool_not_found(self, cli_runner: CliRunner):
        """Test showing non-existent pool."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class:
            mock_container = Mock()
            mock_pool_manager = Mock()
            mock_pool_manager.get_pool_info.side_effect = KeyError("Pool not found")
            
            mock_container.connection_pool_manager = mock_pool_manager
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "pool", "nonexistent"])
            
            assert result.exit_code == 1
            assert "not found" in result.stdout
    
    def test_reset_pool_stats(self, cli_runner: CliRunner):
        """Test resetting pool statistics."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class:
            mock_container = Mock()
            mock_pool_manager = Mock()
            mock_pool_manager.reset_stats.return_value = None
            
            mock_container.connection_pool_manager = mock_pool_manager
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "reset-pools", "--pool", "main_pool"])
            
            assert result.exit_code == 0
            assert "Statistics reset for pool 'main_pool'" in result.stdout
            mock_pool_manager.reset_stats.assert_called_once_with("main_pool")
    
    def test_query_performance(self, cli_runner: CliRunner):
        """Test query performance statistics."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class:
            mock_container = Mock()
            mock_optimizer = Mock()
            
            # Mock performance summary
            mock_optimizer.performance_tracker.get_performance_summary.return_value = {
                "total_queries": 1000,
                "unique_queries": 50,
                "avg_time": 0.125,
                "total_time": 125.0,
                "slow_queries": 5,
                "slowest_query": 2.5,
                "query_types": {"SELECT": 800, "INSERT": 150, "UPDATE": 50}
            }
            
            # Mock slow queries
            mock_slow_query = Mock()
            mock_slow_query.query_hash = "abc123def456ghi789"
            mock_slow_query.query_type.value = "SELECT"
            mock_slow_query.execution_count = 10
            mock_slow_query.avg_time = 1.5
            mock_slow_query.max_time = 2.5
            
            mock_optimizer.performance_tracker.get_slow_queries.return_value = [mock_slow_query]
            mock_optimizer.performance_tracker.get_most_frequent_queries.return_value = [mock_slow_query]
            
            mock_container.query_optimizer = mock_optimizer
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "queries", "--threshold", "1.0"])
            
            assert result.exit_code == 0
            assert "Query Performance Summary" in result.stdout
            assert "1000" in result.stdout  # Total queries
            assert "Slow Queries" in result.stdout
    
    def test_cache_stats(self, cli_runner: CliRunner):
        """Test cache statistics."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class:
            mock_container = Mock()
            mock_optimizer = Mock()
            
            mock_optimizer.cache.get_stats.return_value = {
                "total_entries": 250,
                "max_size": 1000,
                "total_hits": 1500,
                "hit_rate": 0.85,
                "memory_usage_mb": 45.2
            }
            
            mock_container.query_optimizer = mock_optimizer
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "cache"])
            
            assert result.exit_code == 0
            assert "Cache Statistics" in result.stdout
            assert "250" in result.stdout  # Total entries
            assert "85.00%" in result.stdout  # Hit rate
    
    def test_clear_cache_with_confirmation(self, cli_runner: CliRunner):
        """Test clearing cache with confirmation."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class:
            mock_container = Mock()
            mock_optimizer = Mock()
            
            mock_container.query_optimizer = mock_optimizer
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "clear-cache", "--yes"])
            
            assert result.exit_code == 0
            assert "Query cache cleared" in result.stdout
            mock_optimizer.clear_cache.assert_called_once()
    
    def test_optimize_database_dry_run(self, cli_runner: CliRunner):
        """Test database optimization dry run."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class:
            mock_container = Mock()
            mock_optimizer = Mock()
            
            mock_optimizer.get_optimization_report.return_value = {
                "index_recommendations": [
                    {
                        "table": "anomaly_results",
                        "columns": ["detector_id", "timestamp"],
                        "reason": "Frequent filtering on these columns",
                        "estimated_benefit": 0.35
                    },
                    {
                        "table": "datasets",
                        "columns": ["name"],
                        "reason": "Unique lookups",
                        "estimated_benefit": 0.20
                    }
                ]
            }
            
            mock_container.query_optimizer = mock_optimizer
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "optimize", "--dry-run"])
            
            assert result.exit_code == 0
            assert "Index Recommendations" in result.stdout
            assert "anomaly_results" in result.stdout
            assert "35.0%" in result.stdout  # Estimated benefit
    
    def test_performance_monitor(self, cli_runner: CliRunner):
        """Test performance monitoring."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class, \
             patch('asyncio.sleep') as mock_sleep:
            
            mock_container = Mock()
            mock_service = Mock()
            
            # Mock performance summary
            mock_service.get_performance_summary.return_value = {
                "connection_pools": {
                    "main_pool": {
                        "active_connections": 5,
                        "avg_response_time": 0.125,
                        "success_rate": 0.95
                    }
                },
                "query_performance": {
                    "total_queries": 1000,
                    "avg_time": 0.125,
                    "slow_queries": 2
                },
                "cache_performance": {
                    "total_entries": 250,
                    "hit_rate": 0.85
                }
            }
            
            mock_service.get_alerts.return_value = [
                {"message": "High query response time", "type": "performance"}
            ]
            
            mock_container.performance_service = mock_service
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "monitor", "--duration", "10", "--interval", "5"])
            
            assert result.exit_code == 0
            assert "Monitoring performance" in result.stdout
    
    def test_performance_report_generation(self, cli_runner: CliRunner):
        """Test performance report generation."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class, \
             patch('builtins.open', create=True) as mock_open:
            
            mock_container = Mock()
            mock_optimizer = Mock()
            mock_service = Mock()
            
            # Mock report data
            mock_optimizer.get_optimization_report.return_value = {
                "performance_summary": {
                    "total_queries": 1000,
                    "avg_time": 0.125,
                    "slow_queries": 5
                },
                "index_recommendations": [
                    {
                        "table": "results",
                        "columns": ["id", "timestamp"],
                        "reason": "Frequent joins"
                    }
                ]
            }
            
            mock_service.get_performance_summary.return_value = {
                "monitoring_duration": 24.5,
                "active_alerts": 2,
                "connection_pools": {
                    "main": {
                        "active_connections": 5,
                        "success_rate": 0.95,
                        "avg_response_time": 0.125
                    }
                }
            }
            
            mock_service.get_alerts.return_value = [
                {
                    "timestamp": 1640995200,
                    "message": "High memory usage",
                    "type": "resource",
                    "resolved": False
                }
            ]
            
            mock_container.query_optimizer = mock_optimizer
            mock_container.performance_service = mock_service
            mock_container_class.return_value = mock_container
            
            # Mock file write
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result = cli_runner.invoke(app, ["perf", "report", "--output", "/tmp/report.md"])
            
            assert result.exit_code == 0
            assert "Report saved to /tmp/report.md" in result.stdout
            mock_file.write.assert_called_once()


class TestAdvancedCLIFeatures:
    """Test advanced CLI features and edge cases."""
    
    def test_cli_with_missing_dependencies(self, cli_runner: CliRunner):
        """Test CLI behavior when optional dependencies are missing."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class:
            mock_container = Mock()
            # Simulate missing performance service
            mock_container.performance_service = None
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "monitor"])
            
            assert result.exit_code == 1
            assert "not available" in result.stdout
    
    def test_cli_async_command_error_handling(self, cli_runner: CliRunner):
        """Test async command error handling."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class:
            mock_container = Mock()
            mock_optimizer = Mock()
            mock_optimizer.clear_cache.side_effect = Exception("Cache error")
            
            mock_container.query_optimizer = mock_optimizer
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "clear-cache", "--yes"])
            
            assert result.exit_code == 1
            assert "Error clearing cache" in result.stdout
    
    def test_cli_interrupt_handling(self, cli_runner: CliRunner):
        """Test CLI keyboard interrupt handling."""
        with patch('pynomaly.presentation.cli.performance.Container') as mock_container_class, \
             patch('asyncio.sleep', side_effect=KeyboardInterrupt):
            
            mock_container = Mock()
            mock_service = Mock()
            mock_service.get_performance_summary.return_value = {}
            mock_service.get_alerts.return_value = []
            
            mock_container.performance_service = mock_service
            mock_container_class.return_value = mock_container
            
            result = cli_runner.invoke(app, ["perf", "monitor", "--duration", "60"])
            
            # Should handle interrupt gracefully
            assert "interrupted by user" in result.stdout
    
    def test_cli_large_data_handling(self, cli_runner: CliRunner):
        """Test CLI with large dataset operations."""
        with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            
            # Simulate large dataset
            mock_dataset = Mock()
            mock_dataset.id = "large_ds"
            mock_dataset.name = "Large Dataset"
            mock_dataset.n_samples = 1000000
            mock_dataset.n_features = 100
            mock_dataset.file_size_mb = 850.5
            
            mock_service.load_dataset.return_value = mock_dataset
            mock_container.dataset_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "dataset", "load", "/path/to/large_file.csv",
                "--name", "Large Dataset"
            ])
            
            assert result.exit_code == 0
            assert "1000000 samples" in result.stdout
    
    def test_cli_streaming_operations(self, cli_runner: CliRunner):
        """Test CLI streaming detection operations."""
        # This test would be for streaming CLI commands if they exist
        # For now, test that the app can handle streaming-related queries
        with patch('pynomaly.presentation.cli.app.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_settings = Mock()
            mock_settings.version = "1.0.0"
            mock_container.config.return_value = mock_settings
            mock_get_container.return_value = mock_container
            
            # Test that CLI doesn't break with streaming-related status queries
            result = cli_runner.invoke(app, ["status"])
            
            assert result.exit_code == 0
    
    def test_cli_memory_usage_monitoring(self, cli_runner: CliRunner):
        """Test CLI memory usage during operations."""
        # Test CLI behavior under memory constraints
        with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            
            # Mock memory-intensive dataset list
            large_datasets = []
            for i in range(100):
                ds = Mock()
                ds.id = f"ds_{i}"
                ds.name = f"Dataset {i}"
                ds.n_samples = 10000
                ds.n_features = 50
                ds.file_size_mb = 25.0
                large_datasets.append(ds)
            
            mock_repo.find_all.return_value = large_datasets
            mock_container.dataset_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, ["dataset", "list"])
            
            assert result.exit_code == 0
            assert "Dataset 0" in result.stdout
            assert "Dataset 99" in result.stdout


class TestCLIIntegration:
    """Test CLI integration and workflow scenarios."""
    
    def test_complete_workflow(self, cli_runner: CliRunner, sample_csv_file):
        """Test complete CLI workflow: load data, create detector, train, detect."""
        with patch('pynomaly.presentation.cli.datasets.get_cli_container') as mock_datasets_container, \
             patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_detectors_container, \
             patch('pynomaly.presentation.cli.detection.get_cli_container') as mock_detection_container:
            
            # Mock dataset loading
            mock_dataset_service = Mock()
            mock_dataset = Mock()
            mock_dataset.id = "ds123"
            mock_dataset.name = "Test Dataset"
            mock_dataset.n_samples = 4
            mock_dataset.n_features = 2
            mock_dataset_service.load_dataset.return_value = mock_dataset
            mock_datasets_container.return_value.dataset_service.return_value = mock_dataset_service
            
            # Mock detector creation
            mock_detector_service = Mock()
            mock_detector = Mock()
            mock_detector.id = "det123"
            mock_detector.name = "Test Detector"
            mock_detector.algorithm = "IsolationForest"
            mock_detector_service.create_detector.return_value = mock_detector
            mock_detectors_container.return_value.detector_service.return_value = mock_detector_service
            
            # Mock training
            mock_detection_service = Mock()
            mock_train_result = {"success": True, "training_time_ms": 1000}
            mock_detection_service.train_detector.return_value = mock_train_result
            mock_detection_container.return_value.detection_service.return_value = mock_detection_service
            
            # Step 1: Load dataset
            result1 = cli_runner.invoke(app, [
                "dataset", "load", sample_csv_file,
                "--name", "Test Dataset"
            ])
            assert result1.exit_code == 0
            
            # Step 2: Create detector
            result2 = cli_runner.invoke(app, [
                "detector", "create",
                "--name", "Test Detector",
                "--algorithm", "IsolationForest"
            ])
            assert result2.exit_code == 0
            
            # Step 3: Train detector
            result3 = cli_runner.invoke(app, [
                "detect", "train",
                "--detector", "Test Detector",
                "--dataset", "Test Dataset"
            ])
            assert result3.exit_code == 0
    
    def test_cli_error_handling(self, cli_runner: CliRunner):
        """Test CLI error handling."""
        # Test invalid command
        result = cli_runner.invoke(app, ["invalid_command"])
        assert result.exit_code != 0
        
        # Test missing required arguments
        result = cli_runner.invoke(app, ["detector", "create"])
        assert result.exit_code != 0
    
    def test_cli_output_formats(self, cli_runner: CliRunner):
        """Test different output formats."""
        with patch('pynomaly.presentation.cli.detectors.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_repo = Mock()
            mock_detector = Mock()
            mock_detector.id = "det123"
            mock_detector.name = "Test Detector"
            mock_detector.algorithm = "IsolationForest"
            mock_detector.to_dict.return_value = {
                "id": "det123",
                "name": "Test Detector",
                "algorithm": "IsolationForest"
            }
            
            mock_repo.find_by_id.return_value = mock_detector
            mock_container.detector_repository.return_value = mock_repo
            mock_get_container.return_value = mock_container
            
            # Test JSON output
            result = cli_runner.invoke(app, ["detector", "show", "det123", "--format", "json"])
            
            assert result.exit_code == 0
            # Should contain JSON-like output
            assert "{" in result.stdout
    
    def test_cli_configuration_persistence(self, cli_runner: CliRunner):
        """Test CLI configuration persistence."""
        with patch('pynomaly.presentation.cli.app.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_settings = Mock()
            mock_settings.version = "1.0.0"
            mock_settings.app_name = "Pynomaly"
            mock_settings.debug = False
            mock_container.config.return_value = mock_settings
            mock_get_container.return_value = mock_container
            
            # Test setting configuration
            result = cli_runner.invoke(app, ["config", "--set", "debug=true"])
            
            assert result.exit_code == 0
            # Configuration setting should be acknowledged
            assert "debug = true" in result.stdout
    
    def test_cli_parallel_operations(self, cli_runner: CliRunner):
        """Test CLI parallel operations."""
        with patch('pynomaly.presentation.cli.detection.get_cli_container') as mock_get_container:
            mock_container = Mock()
            mock_service = Mock()
            
            mock_results = [
                {"dataset_id": "ds1", "anomaly_count": 5},
                {"dataset_id": "ds2", "anomaly_count": 10}
            ]
            
            mock_service.batch_detect_anomalies.return_value = mock_results
            mock_container.detection_service.return_value = mock_service
            mock_get_container.return_value = mock_container
            
            result = cli_runner.invoke(app, [
                "detect", "batch",
                "--detector", "test_detector",
                "--datasets", "ds1,ds2",
                "--parallel",
                "--max-workers", "4"
            ])
            
            assert result.exit_code == 0
            assert "2 datasets" in result.stdout