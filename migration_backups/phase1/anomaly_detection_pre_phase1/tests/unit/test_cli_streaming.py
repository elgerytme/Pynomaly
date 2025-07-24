"""Unit tests for CLI streaming commands."""

import json
import pytest
import numpy as np
import pandas as pd
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock, AsyncMock
from typer.testing import CliRunner

from anomaly_detection.cli_new.commands.streaming import app


class TestStreamingCommands:
    """Test cases for streaming CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.sample_streaming_stats = {
            'buffer_size': 100,
            'buffer_capacity': 1000,
            'model_fitted': True,
            'samples_since_update': 50,
            'update_frequency': 100,
            'total_samples_processed': 500,
            'last_update_timestamp': '2023-01-01T00:00:00'
        }
        
        self.sample_drift_result = {
            'drift_detected': True,
            'drift_score': 0.75,
            'p_value': 0.02,
            'threshold': 0.05,
            'statistical_test': 'kolmogorov_smirnov',
            'window_size': 200
        }
    
    @patch('anomaly_detection.cli_new.commands.streaming.StreamingService')
    @patch('anomaly_detection.cli_new.commands.streaming.DetectionService')
    @patch('anomaly_detection.cli_new.commands.streaming._create_data_generator')
    @patch('time.sleep')
    def test_monitor_command_random_data(self, mock_sleep, mock_generator, 
                                        mock_detection_service, mock_streaming_service):
        """Test monitor command with random data generator."""
        # Setup mocks
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_streaming = Mock()
        mock_streaming_service.return_value = mock_streaming
        
        # Mock data generator
        sample_data = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
        mock_generator.return_value = iter(sample_data)
        
        # Mock streaming service methods
        mock_batch_result = Mock()
        mock_batch_result.predictions = np.array([1])
        mock_streaming.process_batch.return_value = mock_batch_result
        mock_streaming.get_streaming_stats.return_value = self.sample_streaming_stats
        
        # Mock sleep to avoid actual delays
        mock_sleep.return_value = None
        
        # Run command with short duration
        result = self.runner.invoke(app, [
            'monitor',
            '--input', 'random',
            '--algorithm', 'isolation_forest',
            '--duration', '1',
            '--sample-rate', '1.0'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_streaming_service.assert_called_once()
        mock_generator.assert_called_once_with('random', 1.0)
        mock_streaming.get_streaming_stats.assert_called()
        assert "Monitoring completed" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.streaming.StreamingService')
    @patch('anomaly_detection.cli_new.commands.streaming.DetectionService')
    @patch('anomaly_detection.cli_new.commands.streaming._create_data_generator')
    @patch('time.sleep')
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_monitor_command_with_output_file(self, mock_file, mock_json_dump, mock_sleep,
                                            mock_generator, mock_detection_service, 
                                            mock_streaming_service):
        """Test monitor command with output file."""
        # Setup mocks
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_streaming = Mock()
        mock_streaming_service.return_value = mock_streaming
        
        # Mock data generator
        sample_data = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
        mock_generator.return_value = iter(sample_data)
        
        # Mock streaming service methods
        mock_batch_result = Mock()
        mock_batch_result.predictions = np.array([1])
        mock_streaming.process_batch.return_value = mock_batch_result
        mock_streaming.get_streaming_stats.return_value = self.sample_streaming_stats
        
        # Mock sleep
        mock_sleep.return_value = None
        
        # Run command
        result = self.runner.invoke(app, [
            'monitor',
            '--input', 'random',
            '--duration', '1',
            '--output', 'results.json'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_json_dump.assert_called_once()
        
        # Check that results were saved
        dumped_data = mock_json_dump.call_args[0][0]
        assert 'monitoring_session' in dumped_data
        assert 'summary' in dumped_data
        assert 'streaming_stats' in dumped_data
        assert 'batch_results' in dumped_data
    
    @patch('anomaly_detection.cli_new.commands.streaming.StreamingService')
    @patch('anomaly_detection.cli_new.commands.streaming.DetectionService')
    @patch('anomaly_detection.cli_new.commands.streaming._create_data_generator')
    @patch('time.sleep')
    def test_monitor_command_with_anomalies(self, mock_sleep, mock_generator,
                                          mock_detection_service, mock_streaming_service):
        """Test monitor command detecting anomalies."""
        # Setup mocks
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_streaming = Mock()
        mock_streaming_service.return_value = mock_streaming
        
        # Mock data generator
        sample_data = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
        mock_generator.return_value = iter(sample_data)
        
        # Mock streaming service methods - return anomaly
        mock_batch_result = Mock()
        mock_batch_result.predictions = np.array([-1])  # Anomaly
        mock_streaming.process_batch.return_value = mock_batch_result
        mock_streaming.get_streaming_stats.return_value = self.sample_streaming_stats
        
        # Mock sleep
        mock_sleep.return_value = None
        
        # Run command
        result = self.runner.invoke(app, [
            'monitor',
            '--input', 'random',
            '--duration', '1'
        ])
        
        # Assertions
        assert result.exit_code == 0
        assert "Monitoring completed" in result.stdout
        # Should show non-zero anomaly count
    
    @patch('anomaly_detection.cli_new.commands.streaming.StreamingService')
    @patch('anomaly_detection.cli_new.commands.streaming.DetectionService')
    @patch('anomaly_detection.cli_new.commands.streaming._create_data_generator')
    def test_monitor_command_keyboard_interrupt(self, mock_generator, mock_detection_service,
                                              mock_streaming_service):
        """Test monitor command with keyboard interrupt."""
        # Setup mocks
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_streaming = Mock()
        mock_streaming_service.return_value = mock_streaming
        
        # Mock data generator
        sample_data = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
        mock_generator.return_value = iter(sample_data)
        
        # Mock streaming service methods
        mock_batch_result = Mock()
        mock_batch_result.predictions = np.array([1])
        mock_streaming.process_batch.return_value = mock_batch_result
        mock_streaming.get_streaming_stats.return_value = self.sample_streaming_stats
        
        # Mock keyboard interrupt during processing
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            result = self.runner.invoke(app, [
                'monitor',
                '--input', 'random',
                '--duration', '60'
            ])
        
        # Assertions
        assert result.exit_code == 0
        assert "interrupted by user" in result.stdout
    
    @patch('anomaly_detection.cli_new.commands.streaming.StreamingService')
    def test_stats_command_success(self, mock_streaming_service):
        """Test stats command with successful statistics retrieval."""
        # Setup mock
        mock_streaming = Mock()
        mock_streaming_service.return_value = mock_streaming
        mock_streaming.get_streaming_stats.return_value = self.sample_streaming_stats
        
        # Run command
        result = self.runner.invoke(app, [
            'stats',
            '--window-size', '500',
            '--update-freq', '50'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_streaming_service.assert_called_once_with(window_size=500, update_frequency=50)
        mock_streaming.get_streaming_stats.assert_called_once()
        assert "Buffer Size" in result.stdout
        assert "Model Fitted" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.streaming.StreamingService')
    def test_drift_command_csv_success(self, mock_streaming_service, mock_read_csv, mock_file):
        """Test drift command with CSV file."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_streaming = Mock()
        mock_streaming_service.return_value = mock_streaming
        mock_streaming.detect_concept_drift.return_value = self.sample_drift_result
        
        # Run command
        result = self.runner.invoke(app, [
            'drift',
            '--input', 'test.csv',
            '--window-size', '200',
            '--algorithm', 'isolation_forest'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_read_csv.assert_called_once()
        mock_streaming.detect_concept_drift.assert_called()
        assert "Concept drift detected" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data='[{"feature1": 1.0}]')
    @patch('pandas.read_json')
    @patch('anomaly_detection.cli_new.commands.streaming.StreamingService')
    def test_drift_command_json_success(self, mock_streaming_service, mock_read_json, mock_file):
        """Test drift command with JSON file."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 2.0, 3.0]})
        mock_read_json.return_value = mock_df
        
        mock_streaming = Mock()
        mock_streaming_service.return_value = mock_streaming
        
        # Mock no drift detected
        no_drift_result = self.sample_drift_result.copy()
        no_drift_result['drift_detected'] = False
        mock_streaming.detect_concept_drift.return_value = no_drift_result
        
        # Run command
        result = self.runner.invoke(app, [
            'drift',
            '--input', 'test.json',
            '--algorithm', 'lof'
        ])
        
        # Assertions
        assert result.exit_code == 0
        mock_read_json.assert_called_once()
        assert "No significant concept drift detected" in result.stdout
    
    def test_drift_command_file_not_found(self):
        """Test drift command with non-existent file."""
        result = self.runner.invoke(app, [
            'drift',
            '--input', 'nonexistent.csv'
        ])
        
        assert result.exit_code == 1
        assert "Input file 'nonexistent.csv' not found" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="data")
    def test_drift_command_unsupported_format(self, mock_file):
        """Test drift command with unsupported file format."""
        result = self.runner.invoke(app, [
            'drift',
            '--input', 'test.xml'
        ])
        
        assert result.exit_code == 1
        assert "Unsupported file format '.xml'" in result.stdout
    
    @patch('builtins.open', new_callable=mock_open, read_data="feature1,feature2\n1.0,2.0\n3.0,4.0")
    @patch('pandas.read_csv')
    @patch('anomaly_detection.cli_new.commands.streaming.StreamingService')
    def test_drift_command_processing_failure(self, mock_streaming_service, mock_read_csv, mock_file):
        """Test drift command when drift analysis fails."""
        # Setup mocks
        mock_df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
        mock_read_csv.return_value = mock_df
        
        mock_streaming = Mock()
        mock_streaming_service.return_value = mock_streaming
        mock_streaming.process_sample.side_effect = Exception("Processing failed")
        
        # Run command
        result = self.runner.invoke(app, [
            'drift',
            '--input', 'test.csv'
        ])
        
        # Assertions
        assert result.exit_code == 1
        assert "Drift analysis failed" in result.stdout
    
    def test_create_data_generator_random(self):
        """Test _create_data_generator with random source."""
        from anomaly_detection.cli_new.commands.streaming import _create_data_generator
        
        generator = _create_data_generator("random", 1.0)
        
        # Get a few samples
        samples = []
        for i in range(5):
            samples.append(next(generator))
        
        # Assertions
        assert len(samples) == 5
        for sample in samples:
            assert isinstance(sample, np.ndarray)
            assert len(sample) == 5  # Expected feature count
    
    @patch('pandas.read_csv')
    def test_create_data_generator_file_csv(self, mock_read_csv):
        """Test _create_data_generator with CSV file source."""
        from anomaly_detection.cli_new.commands.streaming import _create_data_generator
        
        # Setup mock
        mock_df = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'feature2': [3.0, 4.0]
        })
        mock_read_csv.return_value = mock_df
        
        with patch('pathlib.Path.exists', return_value=True):
            generator = _create_data_generator("file:test.csv", 1.0)
            
            # Get samples
            sample1 = next(generator)
            sample2 = next(generator)
            sample3 = next(generator)  # Should cycle back to first
            
            # Assertions
            assert isinstance(sample1, np.ndarray)
            assert len(sample1) == 2
            np.testing.assert_array_equal(sample1, sample3)  # Should cycle
    
    @patch('pandas.read_json')
    def test_create_data_generator_file_json(self, mock_read_json):
        """Test _create_data_generator with JSON file source."""
        from anomaly_detection.cli_new.commands.streaming import _create_data_generator
        
        # Setup mock
        mock_df = pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]})
        mock_read_json.return_value = mock_df
        
        with patch('pathlib.Path.exists', return_value=True):
            generator = _create_data_generator("file:test.json", 1.0)
            sample = next(generator)
            
            # Assertions
            assert isinstance(sample, np.ndarray)
            assert len(sample) == 2
    
    def test_create_data_generator_file_not_found(self):
        """Test _create_data_generator with non-existent file."""
        from anomaly_detection.cli_new.commands.streaming import _create_data_generator
        
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(ValueError, match="File not found"):
                _create_data_generator("file:nonexistent.csv", 1.0)
    
    def test_create_data_generator_unsupported_file_format(self):
        """Test _create_data_generator with unsupported file format."""
        from anomaly_detection.cli_new.commands.streaming import _create_data_generator
        
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(ValueError, match="Unsupported file format"):
                _create_data_generator("file:test.xml", 1.0)
    
    def test_create_data_generator_unsupported_source(self):
        """Test _create_data_generator with unsupported input source."""
        from anomaly_detection.cli_new.commands.streaming import _create_data_generator
        
        with pytest.raises(ValueError, match="Unsupported input source"):
            _create_data_generator("kafka:topic", 1.0)
    
    def test_update_monitoring_display(self):
        """Test _update_monitoring_display function."""
        from anomaly_detection.cli_new.commands.streaming import _update_monitoring_display
        from rich.layout import Layout
        
        # Setup mock streaming service
        mock_streaming_service = Mock()
        mock_streaming_service.get_streaming_stats.return_value = self.sample_streaming_stats
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        # Call function
        _update_monitoring_display(
            layout=layout,
            streaming_service=mock_streaming_service,
            total_samples=100,
            anomaly_count=5,
            elapsed=30,
            duration=60,
            algorithm="isolation_forest"
        )
        
        # Assertions - function should execute without error
        mock_streaming_service.get_streaming_stats.assert_called_once()
    
    def test_algorithm_mappings(self):
        """Test that algorithm mappings are correct."""
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        
        # Test that all expected mappings exist
        assert algorithm_map['isolation_forest'] == 'iforest'
        assert algorithm_map['one_class_svm'] == 'ocsvm'
        assert algorithm_map['lof'] == 'lof'
    
    @patch('anomaly_detection.cli_new.commands.streaming.StreamingService')
    @patch('anomaly_detection.cli_new.commands.streaming.DetectionService')
    @patch('anomaly_detection.cli_new.commands.streaming._create_data_generator')
    @patch('time.sleep')
    def test_monitor_command_different_algorithms(self, mock_sleep, mock_generator,
                                                 mock_detection_service, mock_streaming_service):
        """Test monitor command with different algorithms."""
        # Setup mocks
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_streaming = Mock()
        mock_streaming_service.return_value = mock_streaming
        
        # Mock data generator
        sample_data = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
        mock_generator.return_value = iter(sample_data)
        
        # Mock streaming service methods
        mock_batch_result = Mock()
        mock_batch_result.predictions = np.array([1])
        mock_streaming.process_batch.return_value = mock_batch_result
        mock_streaming.get_streaming_stats.return_value = self.sample_streaming_stats
        
        # Mock sleep
        mock_sleep.return_value = None
        
        # Test with different algorithms
        algorithms = ['isolation_forest', 'one_class_svm', 'lof']
        for algorithm in algorithms:
            result = self.runner.invoke(app, [
                'monitor',
                '--input', 'random',
                '--algorithm', algorithm,
                '--duration', '1'
            ])
            
            assert result.exit_code == 0
    
    @patch('anomaly_detection.cli_new.commands.streaming.StreamingService')
    @patch('anomaly_detection.cli_new.commands.streaming.DetectionService')
    @patch('anomaly_detection.cli_new.commands.streaming._create_data_generator')
    @patch('time.sleep')
    def test_monitor_command_custom_parameters(self, mock_sleep, mock_generator,
                                             mock_detection_service, mock_streaming_service):
        """Test monitor command with custom parameters."""
        # Setup mocks
        mock_detection = Mock()
        mock_detection_service.return_value = mock_detection
        
        mock_streaming = Mock()
        mock_streaming_service.return_value = mock_streaming
        
        # Mock data generator
        sample_data = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
        mock_generator.return_value = iter(sample_data)
        
        # Mock streaming service methods
        mock_batch_result = Mock()
        mock_batch_result.predictions = np.array([1])
        mock_streaming.process_batch.return_value = mock_batch_result
        mock_streaming.get_streaming_stats.return_value = self.sample_streaming_stats
        
        # Mock sleep
        mock_sleep.return_value = None
        
        # Run command with custom parameters
        result = self.runner.invoke(app, [
            'monitor',
            '--input', 'random',
            '--algorithm', 'isolation_forest',
            '--window-size', '2000',
            '--update-freq', '200',
            '--contamination', '0.05',
            '--duration', '1',
            '--sample-rate', '0.5'
        ])
        
        # Assertions
        assert result.exit_code == 0
        
        # Check that streaming service was initialized with correct parameters
        mock_streaming_service.assert_called_once_with(
            detection_service=mock_detection,
            window_size=2000,
            update_frequency=200
        )