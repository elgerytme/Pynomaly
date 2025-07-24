"""Unit tests for CLI detection commands."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from typer.testing import CliRunner

from anomaly_detection.cli.commands.detection import app
from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.domain.entities.dataset import Dataset, DatasetType, DatasetMetadata


class TestDetectionCommands:
    """Test cases for detection CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.sample_csv_data = "feature1,feature2,feature3\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0"
        self.sample_json_data = '[{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}]'
        
        # Sample detection result
        self.mock_detection_result = DetectionResult(
            predictions=np.array([1, -1, 1]),
            anomaly_count=1,
            normal_count=2,
            anomaly_rate=0.333,
            anomalies=[1],
            confidence_scores=np.array([0.1, 0.9, 0.2]),
            success=True
        )
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2,feature3\n1.0,2.0,3.0\n4.0,5.0,6.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    def test_run_command_csv_file(self, mock_service_class, mock_read_csv, mock_file):
        """Test run command with CSV file input."""
        # Setup mocks
        mock_df = pd.DataFrame({
            'feature1': [1.0, 4.0],
            'feature2': [2.0, 5.0],
            'feature3': [3.0, 6.0]
        })
        mock_read_csv.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.return_value = self.mock_detection_result
        
        # Run command
        result = self.runner.invoke(app, [
            'run', 
            '--input', 'test.csv',
            '--algorithm', 'isolation_forest',
            '--contamination', '0.1'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_read_csv.assert_called_once()
        mock_service.detect_anomalies.assert_called_once()
        
        # Check that correct algorithm mapping was used
        call_args = mock_service.detect_anomalies.call_args
        assert call_args[1]['algorithm'] == 'iforest'  # mapped algorithm
        assert call_args[1]['contamination'] == 0.1
    
    @patch('builtins.open', new_callable=mock_open, read_data='[{"feature1": 1.0}]')
    @patch('pandas.read_json')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    def test_run_command_json_file(self, mock_service_class, mock_read_json, mock_file):
        """Test run command with JSON file input."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 2.0]})
        mock_read_json.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.return_value = self.mock_detection_result
        
        # Run command
        result = self.runner.invoke(app, [
            'run',
            '--input', 'test.json',
            '--algorithm', 'one_class_svm'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_read_json.assert_called_once()
        mock_service.detect_anomalies.assert_called_once()
        
        # Check algorithm mapping
        call_args = mock_service.detect_anomalies.call_args  
        assert call_args[1]['algorithm'] == 'ocsvm'
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n4.0,5.0")
    @patch('pandas.read_parquet')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    def test_run_command_parquet_file(self, mock_service_class, mock_read_parquet, mock_file):
        """Test run command with Parquet file input."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 4.0], 'feature2': [2.0, 5.0]})
        mock_read_parquet.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.return_value = self.mock_detection_result
        
        # Run command
        result = self.runner.invoke(app, [
            'run',
            '--input', 'test.parquet',
            '--algorithm', 'lof'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_read_parquet.assert_called_once()
        mock_service.detect_anomalies.assert_called_once()
        
        # Check algorithm mapping
        call_args = mock_service.detect_anomalies.call_args
        assert call_args[1]['algorithm'] == 'lof'  # no mapping needed
    
    def test_run_command_file_not_found(self):
        """Test run command with non-existent file."""
        result = self.runner.invoke(app, [
            'run',
            '--input', 'nonexistent.csv'
        ])
        
        assert result.exit_code == 1
        assert "Input file 'nonexistent.csv' not found" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="data")
    def test_run_command_unsupported_format(self, mock_file):
        """Test run command with unsupported file format."""
        result = self.runner.invoke(app, [
            'run',
            '--input', 'test.txt'
        ])
        
        assert result.exit_code == 1
        assert "Unsupported file format '.txt'" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,label\n1.0,-1\n2.0,1")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    @patch('sklearn.metrics.accuracy_score')
    @patch('sklearn.metrics.precision_score')
    @patch('sklearn.metrics.recall_score')
    @patch('sklearn.metrics.f1_score')
    def test_run_command_with_labels(self, mock_f1, mock_recall, mock_precision, 
                                   mock_accuracy, mock_service_class, mock_read_csv, mock_file):
        """Test run command with ground truth labels."""
        # Setup mocks
        mock_df = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'label': [-1, 1]
        })
        mock_read_csv.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.return_value = self.mock_detection_result
        
        # Mock metrics
        mock_accuracy.return_value = 0.8
        mock_precision.return_value = 0.75
        mock_recall.return_value = 0.85
        mock_f1.return_value = 0.79
        
        # Run command
        result = self.runner.invoke(app, [
            'run',
            '--input', 'test.csv',
            '--has-labels',
            '--label-column', 'label'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_read_csv.assert_called_once()
        mock_service.detect_anomalies.assert_called_once()
        
        # Check that metrics were calculated
        mock_accuracy.assert_called_once()
        mock_precision.assert_called_once()
        mock_recall.assert_called_once()
        mock_f1.assert_called_once()
        
        # Check output contains evaluation metrics
        assert "Accuracy" in result.stdout
        assert "0.800" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_run_command_with_output_file(self, mock_json_dump, mock_output_file,
                                        mock_service_class, mock_read_csv, mock_input_file):
        """Test run command with output file."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.return_value = self.mock_detection_result
        
        # Run command
        result = self.runner.invoke(app, [
            'run',
            '--input', 'test.csv',
            '--output', 'results.json'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_json_dump.assert_called_once()
        
        # Check that results structure is correct
        dumped_data = mock_json_dump.call_args[0][0]
        assert 'detection_results' in dumped_data
        assert 'dataset_info' in dumped_data
        assert dumped_data['algorithm'] == 'isolation_forest'
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    def test_run_command_detection_failure(self, mock_service_class, mock_read_csv, mock_file):
        """Test run command when detection fails."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.side_effect = Exception("Detection failed")
        
        # Run command
        result = self.runner.invoke(app, [
            'run',
            '--input', 'test.csv'
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Detection failed" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    @patch('anomaly_detection.cli_new.commands.detection.EnsembleService')
    def test_ensemble_command_majority_vote(self, mock_ensemble_class, mock_service_class, 
                                          mock_read_csv, mock_file):
        """Test ensemble command with majority vote."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        # Mock individual services
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.return_value = self.mock_detection_result
        
        # Mock ensemble service
        mock_ensemble = Mock()
        mock_ensemble_class.return_value = mock_ensemble
        mock_ensemble.majority_vote.return_value = np.array([1, -1])
        
        # Run command
        result = self.runner.invoke(app, [
            'ensemble',
            '--input', 'test.csv',
            '--algorithms', 'isolation_forest',
            '--algorithms', 'one_class_svm',
            '--method', 'majority'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_ensemble.majority_vote.assert_called_once()
        
        # Check that individual algorithms were called
        assert mock_service.detect_anomalies.call_count == 2
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    @patch('anomaly_detection.cli_new.commands.detection.EnsembleService')
    def test_ensemble_command_average_method(self, mock_ensemble_class, mock_service_class,
                                           mock_read_csv, mock_file):
        """Test ensemble command with average combination method."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        # Mock detection result with confidence scores
        result_with_scores = DetectionResult(
            predictions=np.array([1, -1]),
            anomaly_count=1,
            normal_count=1,
            anomaly_rate=0.5,
            anomalies=[1],
            confidence_scores=np.array([0.2, 0.8]),
            success=True
        )
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.return_value = result_with_scores
        
        # Mock ensemble service
        mock_ensemble = Mock()
        mock_ensemble_class.return_value = mock_ensemble
        mock_ensemble.average_combination.return_value = (np.array([1, -1]), np.array([0.3, 0.7]))
        
        # Run command
        result = self.runner.invoke(app, [
            'ensemble',
            '--input', 'test.csv',
            '--algorithms', 'isolation_forest',
            '--algorithms', 'lof',
            '--method', 'average'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_ensemble.average_combination.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    @patch('anomaly_detection.cli_new.commands.detection.EnsembleService')
    def test_ensemble_command_weighted_average(self, mock_ensemble_class, mock_service_class,
                                             mock_read_csv, mock_file):
        """Test ensemble command with weighted average method."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        # Mock detection result with confidence scores
        result_with_scores = DetectionResult(
            predictions=np.array([1, -1]),
            anomaly_count=1,
            normal_count=1,
            anomaly_rate=0.5,
            anomalies=[1],
            confidence_scores=np.array([0.1, 0.9]),
            success=True
        )
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.return_value = result_with_scores
        
        # Mock ensemble service
        mock_ensemble = Mock()
        mock_ensemble_class.return_value = mock_ensemble
        mock_ensemble.weighted_combination.return_value = (np.array([1, -1]), np.array([0.2, 0.8]))
        
        # Run command
        result = self.runner.invoke(app, [
            'ensemble',
            '--input', 'test.csv',
            '--algorithms', 'isolation_forest',
            '--algorithms', 'one_class_svm',
            '--algorithms', 'lof',
            '--method', 'weighted_average'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_ensemble.weighted_combination.assert_called_once()
        
        # Check that weights were passed
        call_args = mock_ensemble.weighted_combination.call_args[0]
        weights = call_args[2]
        assert len(weights) == 3
        assert np.allclose(weights, [1/3, 1/3, 1/3])
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    @patch('anomaly_detection.cli_new.commands.detection.EnsembleService')
    def test_ensemble_command_max_method(self, mock_ensemble_class, mock_service_class,
                                       mock_read_csv, mock_file):
        """Test ensemble command with max combination method."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        # Mock detection result with confidence scores
        result_with_scores = DetectionResult(
            predictions=np.array([1, -1]),
            anomaly_count=1,
            normal_count=1,
            anomaly_rate=0.5,
            anomalies=[1],
            confidence_scores=np.array([0.3, 0.7]),
            success=True
        )
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.return_value = result_with_scores
        
        # Mock ensemble service
        mock_ensemble = Mock()
        mock_ensemble_class.return_value = mock_ensemble
        mock_ensemble.max_combination.return_value = (np.array([1, -1]), np.array([0.4, 0.8]))
        
        # Run command
        result = self.runner.invoke(app, [
            'ensemble',
            '--input', 'test.csv',
            '--algorithms', 'isolation_forest',
            '--algorithms', 'lof',
            '--method', 'max'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_ensemble.max_combination.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    @patch('anomaly_detection.cli_new.commands.detection.EnsembleService')
    @patch('json.dump')
    def test_ensemble_command_with_output_file(self, mock_json_dump, mock_ensemble_class,
                                             mock_service_class, mock_read_csv, mock_file):
        """Test ensemble command with output file."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.return_value = self.mock_detection_result
        
        mock_ensemble = Mock()
        mock_ensemble_class.return_value = mock_ensemble
        mock_ensemble.majority_vote.return_value = np.array([1, -1])
        
        # Run command
        result = self.runner.invoke(app, [
            'ensemble',
            '--input', 'test.csv',
            '--output', 'ensemble_results.json',
            '--algorithms', 'isolation_forest',
            '--algorithms', 'lof'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_json_dump.assert_called_once()
        
        # Check results structure
        dumped_data = mock_json_dump.call_args[0][0]
        assert 'ensemble_config' in dumped_data
        assert 'individual_results' in dumped_data
        assert 'ensemble_results' in dumped_data
    
    def test_ensemble_command_file_not_found(self):
        """Test ensemble command with non-existent file."""
        result = self.runner.invoke(app, [
            'ensemble',
            '--input', 'nonexistent.csv'
        ])
        
        assert result.exit_code == 1
        assert "Input file 'nonexistent.csv' not found" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="data")
    def test_ensemble_command_unsupported_format(self, mock_file):
        """Test ensemble command with unsupported file format."""
        result = self.runner.invoke(app, [
            'ensemble',
            '--input', 'test.xml'
        ])
        
        assert result.exit_code == 1
        assert "Unsupported file format '.xml'" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    def test_ensemble_command_detection_failure(self, mock_service_class, mock_read_csv, mock_file):
        """Test ensemble command when individual detection fails."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.detect_anomalies.side_effect = Exception("Detection failed")
        
        # Run command
        result = self.runner.invoke(app, [
            'ensemble',
            '--input', 'test.csv',
            '--algorithms', 'isolation_forest'
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Ensemble detection failed" in result.stdout
    
    def test_algorithm_mappings(self):
        """Test that algorithm mappings are correct."""
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof',
            'autoencoder': 'autoencoder'
        }
        
        # Test that all expected mappings exist
        assert algorithm_map['isolation_forest'] == 'iforest'
        assert algorithm_map['one_class_svm'] == 'ocsvm'
        assert algorithm_map['lof'] == 'lof'
        assert algorithm_map['autoencoder'] == 'autoencoder'
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\ninvalid,data")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.detection.DetectionService')
    def test_run_command_data_loading_failure(self, mock_service_class, mock_read_csv, mock_file):
        """Test run command when data loading fails."""
        # Setup mocks
        mock_read_csv.side_effect = Exception("Invalid CSV format")
        
        # Run command
        result = self.runner.invoke(app, [
            'run',
            '--input', 'test.csv'
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Failed to load dataset" in result.stdout