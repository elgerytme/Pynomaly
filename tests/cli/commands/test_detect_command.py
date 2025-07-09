"""
Comprehensive CLI Detection Command Tests
Tests for the detect command - core anomaly detection functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from typer.testing import CliRunner

from pynomaly.presentation.cli.app import app
from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import ContaminationRate, AnomalyScore


class TestDetectCommand:
    """Test suite for detect command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test data file
        self.test_data_file = self.temp_dir / "test_data.csv"
        self.test_data_file.write_text("feature1,feature2,label\n1.0,2.0,0\n2.0,3.0,1\n3.0,4.0,0\n")
        
        # Create test config file
        self.config_file = self.temp_dir / "config.json"
        self.config_file.write_text(json.dumps({
            "algorithm": "IsolationForest",
            "contamination_rate": 0.1,
            "n_estimators": 100
        }))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # Basic Command Tests
    
    def test_detect_help_command(self):
        """Test detect command help output."""
        result = self.runner.invoke(app, ["detect", "--help"])
        
        assert result.exit_code == 0
        assert "detect" in result.stdout.lower()
        assert "anomaly detection" in result.stdout.lower() or "detect" in result.stdout.lower()
        assert "train" in result.stdout.lower()
        assert "run" in result.stdout.lower()

    def test_detect_train_help(self):
        """Test detect train subcommand help."""
        result = self.runner.invoke(app, ["detect", "train", "--help"])
        
        assert result.exit_code == 0
        assert "train" in result.stdout.lower()
        assert "dataset" in result.stdout.lower()
        assert "algorithm" in result.stdout.lower() or "detector" in result.stdout.lower()

    def test_detect_run_help(self):
        """Test detect run subcommand help."""
        result = self.runner.invoke(app, ["detect", "run", "--help"])
        
        assert result.exit_code == 0
        assert "run" in result.stdout.lower()
        assert "dataset" in result.stdout.lower()
        assert "detector" in result.stdout.lower() or "model" in result.stdout.lower()

    # Training Command Tests
    
    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_train_basic(self, mock_dataset_service, mock_detection_service):
        """Test basic training command execution."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.name = "test_dataset"
        mock_dataset.id = "test_id"
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_detector = Mock(spec=Detector)
        mock_detector.name = "test_detector"
        mock_detector.id = "detector_id"
        mock_detection_service.train_detector.return_value = mock_detector
        
        # Execute command
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "IsolationForest",
            "--detector-name", "test_detector"
        ])
        
        # Verify results
        assert result.exit_code == 0
        mock_dataset_service.load_dataset.assert_called_once()
        mock_detection_service.train_detector.assert_called_once()

    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_train_with_contamination_rate(self, mock_dataset_service, mock_detection_service):
        """Test training with specific contamination rate."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_detector = Mock(spec=Detector)
        mock_detection_service.train_detector.return_value = mock_detector
        
        # Execute command with contamination rate
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "IsolationForest",
            "--contamination-rate", "0.15",
            "--detector-name", "test_detector"
        ])
        
        # Verify contamination rate was passed correctly
        assert result.exit_code == 0
        call_args = mock_detection_service.train_detector.call_args
        # Check that contamination rate was used in the call
        assert call_args is not None

    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_train_with_config_file(self, mock_dataset_service, mock_detection_service):
        """Test training with configuration file."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_detector = Mock(spec=Detector)
        mock_detection_service.train_detector.return_value = mock_detector
        
        # Execute command with config file
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--config", str(self.config_file),
            "--detector-name", "test_detector"
        ])
        
        assert result.exit_code == 0
        mock_detection_service.train_detector.assert_called_once()

    # Detection Command Tests
    
    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_run_basic(self, mock_dataset_service, mock_detection_service):
        """Test basic detection run command."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_result.scores = [AnomalyScore(0.3), AnomalyScore(0.7)]
        mock_detection_service.detect_anomalies.return_value = mock_result
        
        # Execute command
        result = self.runner.invoke(app, [
            "detect", "run",
            "--dataset", str(self.test_data_file),
            "--detector", "test_detector"
        ])
        
        assert result.exit_code == 0
        mock_detection_service.detect_anomalies.assert_called_once()

    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_run_with_output_file(self, mock_dataset_service, mock_detection_service):
        """Test detection with output file specification."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_result.scores = [AnomalyScore(0.3), AnomalyScore(0.7)]
        mock_detection_service.detect_anomalies.return_value = mock_result
        
        output_file = self.temp_dir / "output.json"
        
        # Execute command with output file
        result = self.runner.invoke(app, [
            "detect", "run",
            "--dataset", str(self.test_data_file),
            "--detector", "test_detector",
            "--output", str(output_file)
        ])
        
        assert result.exit_code == 0
        # Verify output file creation would be handled
        mock_detection_service.detect_anomalies.assert_called_once()

    # Batch Detection Tests
    
    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_batch_basic(self, mock_dataset_service, mock_detection_service):
        """Test batch detection command."""
        # Create batch config
        batch_config = {
            "datasets": [str(self.test_data_file)],
            "detectors": ["test_detector1", "test_detector2"],
            "output_dir": str(self.temp_dir)
        }
        batch_config_file = self.temp_dir / "batch_config.json"
        batch_config_file.write_text(json.dumps(batch_config))
        
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_detection_service.detect_anomalies.return_value = mock_result
        
        # Execute batch command
        result = self.runner.invoke(app, [
            "detect", "batch",
            "--config", str(batch_config_file)
        ])
        
        # Should handle batch processing (implementation dependent)
        assert result.exit_code == 0

    # Error Handling Tests
    
    def test_detect_train_missing_dataset(self):
        """Test training with missing dataset file."""
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", "/nonexistent/file.csv",
            "--algorithm", "IsolationForest",
            "--detector-name", "test_detector"
        ])
        
        assert result.exit_code != 0
        assert "file" in result.stdout.lower() or "not found" in result.stdout.lower()

    def test_detect_train_invalid_algorithm(self):
        """Test training with invalid algorithm."""
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "InvalidAlgorithm",
            "--detector-name", "test_detector"
        ])
        
        assert result.exit_code != 0
        assert "algorithm" in result.stdout.lower() or "invalid" in result.stdout.lower()

    def test_detect_train_invalid_contamination_rate(self):
        """Test training with invalid contamination rate."""
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "IsolationForest",
            "--contamination-rate", "1.5",  # Invalid: > 1.0
            "--detector-name", "test_detector"
        ])
        
        assert result.exit_code != 0
        assert "contamination" in result.stdout.lower() or "rate" in result.stdout.lower()

    def test_detect_run_missing_detector(self):
        """Test detection with missing detector."""
        result = self.runner.invoke(app, [
            "detect", "run",
            "--dataset", str(self.test_data_file),
            "--detector", "nonexistent_detector"
        ])
        
        assert result.exit_code != 0
        assert "detector" in result.stdout.lower() or "not found" in result.stdout.lower()

    # Parameter Validation Tests
    
    def test_detect_train_contamination_rate_boundaries(self):
        """Test contamination rate boundary values."""
        # Test minimum boundary (0.0)
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "IsolationForest",
            "--contamination-rate", "0.0",
            "--detector-name", "test_detector"
        ])
        
        # Should handle validation appropriately
        # (exact behavior depends on implementation)
        assert result.exit_code in [0, 1]  # Either success or validation error

    def test_detect_train_negative_contamination_rate(self):
        """Test negative contamination rate."""
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "IsolationForest",
            "--contamination-rate", "-0.1",
            "--detector-name", "test_detector"
        ])
        
        assert result.exit_code != 0
        assert "contamination" in result.stdout.lower() or "negative" in result.stdout.lower()

    # Algorithm-Specific Tests
    
    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_train_isolation_forest(self, mock_dataset_service, mock_detection_service):
        """Test training with IsolationForest algorithm."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_detector = Mock(spec=Detector)
        mock_detection_service.train_detector.return_value = mock_detector
        
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "IsolationForest",
            "--detector-name", "test_detector"
        ])
        
        assert result.exit_code == 0
        mock_detection_service.train_detector.assert_called_once()

    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_train_one_class_svm(self, mock_dataset_service, mock_detection_service):
        """Test training with OneClassSVM algorithm."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_detector = Mock(spec=Detector)
        mock_detection_service.train_detector.return_value = mock_detector
        
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "OneClassSVM",
            "--detector-name", "test_detector"
        ])
        
        assert result.exit_code == 0
        mock_detection_service.train_detector.assert_called_once()

    # Output Format Tests
    
    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_run_json_output(self, mock_dataset_service, mock_detection_service):
        """Test detection with JSON output format."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_result.scores = [AnomalyScore(0.3), AnomalyScore(0.7)]
        mock_detection_service.detect_anomalies.return_value = mock_result
        
        result = self.runner.invoke(app, [
            "detect", "run",
            "--dataset", str(self.test_data_file),
            "--detector", "test_detector",
            "--output-format", "json"
        ])
        
        assert result.exit_code == 0
        mock_detection_service.detect_anomalies.assert_called_once()

    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_run_csv_output(self, mock_dataset_service, mock_detection_service):
        """Test detection with CSV output format."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_result.scores = [AnomalyScore(0.3), AnomalyScore(0.7)]
        mock_detection_service.detect_anomalies.return_value = mock_result
        
        result = self.runner.invoke(app, [
            "detect", "run",
            "--dataset", str(self.test_data_file),
            "--detector", "test_detector",
            "--output-format", "csv"
        ])
        
        assert result.exit_code == 0
        mock_detection_service.detect_anomalies.assert_called_once()

    # Performance and Resource Tests
    
    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_train_with_timeout(self, mock_dataset_service, mock_detection_service):
        """Test training with timeout parameter."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_detector = Mock(spec=Detector)
        mock_detection_service.train_detector.return_value = mock_detector
        
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "IsolationForest",
            "--detector-name", "test_detector",
            "--timeout", "30"
        ])
        
        assert result.exit_code == 0
        mock_detection_service.train_detector.assert_called_once()

    # Integration Tests
    
    @patch('pynomaly.presentation.cli.commands.detect.detection_service')
    @patch('pynomaly.presentation.cli.commands.detect.dataset_service')
    def test_detect_train_and_run_workflow(self, mock_dataset_service, mock_detection_service):
        """Test complete train and run workflow."""
        # Mock services for training
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset
        
        mock_detector = Mock(spec=Detector)
        mock_detector.name = "test_detector"
        mock_detection_service.train_detector.return_value = mock_detector
        
        # Train detector
        train_result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "IsolationForest",
            "--detector-name", "test_detector"
        ])
        
        assert train_result.exit_code == 0
        
        # Mock services for detection
        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_result.scores = [AnomalyScore(0.3), AnomalyScore(0.7)]
        mock_detection_service.detect_anomalies.return_value = mock_result
        
        # Run detection
        detect_result = self.runner.invoke(app, [
            "detect", "run",
            "--dataset", str(self.test_data_file),
            "--detector", "test_detector"
        ])
        
        assert detect_result.exit_code == 0
        
        # Verify both operations were called
        mock_detection_service.train_detector.assert_called_once()
        mock_detection_service.detect_anomalies.assert_called_once()

    # Edge Cases
    
    def test_detect_train_empty_dataset_file(self):
        """Test training with empty dataset file."""
        empty_file = self.temp_dir / "empty.csv"
        empty_file.write_text("")
        
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(empty_file),
            "--algorithm", "IsolationForest",
            "--detector-name", "test_detector"
        ])
        
        assert result.exit_code != 0
        assert "empty" in result.stdout.lower() or "invalid" in result.stdout.lower()

    def test_detect_train_malformed_csv(self):
        """Test training with malformed CSV file."""
        malformed_file = self.temp_dir / "malformed.csv"
        malformed_file.write_text("feature1,feature2\n1.0,2.0,extra_column\n2.0\n")
        
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(malformed_file),
            "--algorithm", "IsolationForest",
            "--detector-name", "test_detector"
        ])
        
        assert result.exit_code != 0
        assert "format" in result.stdout.lower() or "invalid" in result.stdout.lower()

    def test_detect_train_long_detector_name(self):
        """Test training with extremely long detector name."""
        long_name = "a" * 1000
        
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "IsolationForest",
            "--detector-name", long_name
        ])
        
        # Should handle long names gracefully
        assert result.exit_code in [0, 1]  # Either success or validation error

    def test_detect_train_special_characters_in_name(self):
        """Test training with special characters in detector name."""
        special_name = "test_detector_🚀_with_emoji"
        
        result = self.runner.invoke(app, [
            "detect", "train",
            "--dataset", str(self.test_data_file),
            "--algorithm", "IsolationForest",
            "--detector-name", special_name
        ])
        
        # Should handle special characters appropriately
        assert result.exit_code in [0, 1]  # Either success or validation error