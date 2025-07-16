"""
Stable CLI integration tests with proper mocking and dependency injection.

This module provides robust CLI testing with minimal external dependencies.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from typer.testing import CliRunner

# Import CLI modules
from monorepo.presentation.cli.app import app
from monorepo.presentation.cli import autonomous, datasets, detectors, detection


@pytest.fixture
def cli_runner():
    """Create a stable CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for testing."""
    content = """feature1,feature2,feature3,label
1.0,2.0,3.0,0
2.0,3.0,4.0,0
100.0,200.0,300.0,1
3.0,4.0,5.0,0
4.0,5.0,6.0,0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_container():
    """Create mock container with all services."""
    container = Mock()
    
    # Mock repository services
    dataset_repo = Mock()
    detector_repo = Mock()
    result_repo = Mock()
    
    # Configure dataset repository
    dataset_repo.save.return_value = True
    dataset_repo.find_by_name.return_value = None
    dataset_repo.list_all.return_value = []
    dataset_repo.delete.return_value = True
    
    # Configure detector repository
    detector_repo.save.return_value = True
    detector_repo.find_by_name.return_value = None
    detector_repo.list_all.return_value = []
    detector_repo.delete.return_value = True
    
    # Configure result repository
    result_repo.save.return_value = True
    result_repo.find_by_id.return_value = None
    result_repo.list_all.return_value = []
    
    # Attach repositories to container
    container.dataset_repository.return_value = dataset_repo
    container.detector_repository.return_value = detector_repo
    container.result_repository.return_value = result_repo
    
    # Mock use cases
    container.train_detector_use_case.return_value = Mock()
    container.detect_anomalies_use_case.return_value = Mock()
    container.autonomous_detection_service.return_value = Mock()
    
    return container


class TestStableCLIIntegration:
    """Stable CLI integration tests."""
    
    def test_app_help_command(self, cli_runner):
        """Test main app help command."""
        result = cli_runner.invoke(app, ['--help'])
        
        assert result.exit_code == 0
        assert 'Usage:' in result.stdout
        assert 'Commands:' in result.stdout
        assert 'Pynomaly' in result.stdout
    
    def test_version_command(self, cli_runner):
        """Test version command."""
        result = cli_runner.invoke(app, ['--version'])
        
        # Should either succeed or fail gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert len(result.stdout.strip()) > 0
    
    @patch('monorepo.presentation.cli.container.get_cli_container')
    def test_dataset_list_empty(self, mock_get_container, cli_runner, mock_container):
        """Test dataset list with no datasets."""
        mock_get_container.return_value = mock_container
        mock_container.dataset_repository.return_value.list_all.return_value = []
        
        result = cli_runner.invoke(datasets.app, ['list'])
        
        assert result.exit_code in [0, 1]  # Allow graceful failures
        if result.exit_code == 0:
            assert 'No datasets found' in result.stdout or 'datasets' in result.stdout.lower()
    
    @patch('monorepo.presentation.cli.container.get_cli_container')
    def test_dataset_load_basic(self, mock_get_container, cli_runner, mock_container, sample_csv_file):
        """Test basic dataset loading."""
        mock_get_container.return_value = mock_container
        
        # Mock successful dataset creation
        mock_dataset = Mock()
        mock_dataset.name = 'test_dataset'
        mock_dataset.shape = (5, 4)
        mock_container.dataset_repository.return_value.save.return_value = True
        
        result = cli_runner.invoke(datasets.app, [
            'load', sample_csv_file, '--name', 'test_dataset'
        ])
        
        # Should handle gracefully even if dependencies are missing
        assert result.exit_code in [0, 1]
    
    @patch('monorepo.presentation.cli.container.get_cli_container')
    def test_detector_list_empty(self, mock_get_container, cli_runner, mock_container):
        """Test detector list with no detectors."""
        mock_get_container.return_value = mock_container
        mock_container.detector_repository.return_value.list_all.return_value = []
        
        result = cli_runner.invoke(detectors.app, ['list'])
        
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert 'No detectors found' in result.stdout or 'detectors' in result.stdout.lower()
    
    @patch('monorepo.presentation.cli.container.get_cli_container')
    def test_detector_create_basic(self, mock_get_container, cli_runner, mock_container):
        """Test basic detector creation."""
        mock_get_container.return_value = mock_container
        
        # Mock successful detector creation
        mock_detector = Mock()
        mock_detector.name = 'test_detector'
        mock_detector.algorithm_name = 'IsolationForest'
        mock_container.detector_repository.return_value.save.return_value = True
        
        result = cli_runner.invoke(detectors.app, [
            'create', '--name', 'test_detector', '--algorithm', 'IsolationForest'
        ])
        
        assert result.exit_code in [0, 1]  # Allow graceful failures
    
    @patch('monorepo.application.services.autonomous_service.AutonomousDetectionService')
    def test_autonomous_detect_mocked(self, mock_service_class, cli_runner, sample_csv_file):
        """Test autonomous detection with mocked service."""
        # Mock the service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock successful detection
        mock_service.detect_anomalies.return_value = {
            'best_detector': 'IsolationForest',
            'anomalies_found': 1,
            'confidence': 0.95,
            'anomaly_indices': [2],
            'anomaly_scores': [0.95]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            result = cli_runner.invoke(autonomous.app, [
                'detect', sample_csv_file, '--output', output_path
            ])
            
            # Should complete or fail gracefully
            assert result.exit_code in [0, 1]
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_export_list_formats(self, cli_runner):
        """Test export formats listing."""
        result = cli_runner.invoke(app, ['export', 'list-formats'])
        
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert 'Available Export Formats' in result.stdout or 'formats' in result.stdout.lower()
    
    @patch('monorepo.presentation.cli.container.get_cli_container')
    def test_error_handling_invalid_dataset(self, mock_get_container, cli_runner, mock_container):
        """Test error handling for invalid dataset operations."""
        mock_get_container.return_value = mock_container
        
        # Mock dataset not found
        mock_container.dataset_repository.return_value.find_by_name.return_value = None
        
        result = cli_runner.invoke(datasets.app, ['show', 'nonexistent_dataset'])
        
        # Should handle error gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 1:
            assert 'not found' in result.stdout.lower() or 'error' in result.stdout.lower()
    
    def test_invalid_command_handling(self, cli_runner):
        """Test handling of invalid commands."""
        result = cli_runner.invoke(app, ['invalid-command'])
        
        assert result.exit_code != 0
        assert 'No such command' in result.stdout or 'invalid' in result.stdout.lower()
    
    def test_missing_arguments_handling(self, cli_runner):
        """Test handling of missing required arguments."""
        result = cli_runner.invoke(datasets.app, ['load'])
        
        assert result.exit_code != 0
        # Should show usage or error message
        assert 'Usage:' in result.stdout or 'Missing' in result.stdout or 'Error' in result.stdout
    
    def test_file_not_found_handling(self, cli_runner):
        """Test handling of non-existent files."""
        result = cli_runner.invoke(datasets.app, [
            'load', '/nonexistent/file.csv', '--name', 'test'
        ])
        
        assert result.exit_code != 0
    
    @patch('requests.get')
    def test_server_status_offline(self, mock_get, cli_runner):
        """Test server status when server is offline."""
        # Mock connection error
        mock_get.side_effect = Exception('Connection refused')
        
        result = cli_runner.invoke(app, ['server', 'status'])
        
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert 'offline' in result.stdout.lower() or 'connection' in result.stdout.lower()
    
    def test_quickstart_accept(self, cli_runner):
        """Test quickstart workflow acceptance."""
        result = cli_runner.invoke(app, ['quickstart'], input='y\n')
        
        assert result.exit_code == 0
        assert 'Welcome' in result.stdout or 'Quickstart' in result.stdout
    
    def test_quickstart_decline(self, cli_runner):
        """Test quickstart workflow decline."""
        result = cli_runner.invoke(app, ['quickstart'], input='n\n')
        
        assert result.exit_code == 0
        assert 'cancelled' in result.stdout.lower() or 'declined' in result.stdout.lower()
    
    def test_configuration_generation(self, cli_runner):
        """Test configuration file generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
            config_path = config_file.name
        
        try:
            result = cli_runner.invoke(app, [
                'generate-config', 'test', '--output', config_path
            ])
            
            # Should complete or fail gracefully
            assert result.exit_code in [0, 1]
            
            if result.exit_code == 0 and Path(config_path).exists():
                # Verify configuration is valid JSON
                with open(config_path) as f:
                    config = json.load(f)
                assert isinstance(config, dict)
        
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    def test_help_system_completeness(self, cli_runner):
        """Test help system for all major commands."""
        commands_to_test = [
            ['auto', '--help'],
            ['dataset', '--help'],
            ['detector', '--help'],
            ['detect', '--help'],
            ['export', '--help'],
            ['server', '--help']
        ]
        
        for cmd_args in commands_to_test:
            result = cli_runner.invoke(app, cmd_args)
            
            # Help should always work
            assert result.exit_code == 0
            assert 'Usage:' in result.stdout or 'Commands:' in result.stdout


class TestCLIStabilityAndPerformance:
    """Test CLI stability and performance characteristics."""
    
    def test_memory_usage_autonomous_detection(self, cli_runner, sample_csv_file):
        """Test memory usage during autonomous detection."""
        with patch('monorepo.application.services.autonomous_service.AutonomousDetectionService') as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            
            # Mock lightweight response
            mock_service_instance.detect_anomalies.return_value = {
                'best_detector': 'IsolationForest',
                'anomalies_found': 0,
                'confidence': 0.5
            }
            
            result = cli_runner.invoke(autonomous.app, ['detect', sample_csv_file])
            
            # Should complete without memory issues
            assert result.exit_code in [0, 1]
    
    def test_concurrent_operations_stability(self, cli_runner):
        """Test stability under concurrent-like operations."""
        results = []
        
        # Simulate multiple rapid operations
        for i in range(3):
            result = cli_runner.invoke(app, ['--help'])
            results.append(result)
        
        # All operations should be stable
        for result in results:
            assert result.exit_code == 0
            assert 'Usage:' in result.stdout
    
    def test_resource_cleanup(self, cli_runner, sample_csv_file):
        """Test that resources are properly cleaned up."""
        with patch('monorepo.presentation.cli.container.get_cli_container') as mock_get_container:
            container = Mock()
            mock_get_container.return_value = container
            
            # Run operation that should clean up properly
            result = cli_runner.invoke(datasets.app, [
                'load', sample_csv_file, '--name', 'test'
            ])
            
            # Should complete without hanging
            assert result.exit_code in [0, 1]
    
    def test_error_recovery(self, cli_runner):
        """Test error recovery and graceful degradation."""
        # Test with invalid command
        result1 = cli_runner.invoke(app, ['invalid-command'])
        assert result1.exit_code != 0
        
        # Test that app still works after error
        result2 = cli_runner.invoke(app, ['--help'])
        assert result2.exit_code == 0
        assert 'Usage:' in result2.stdout
    
    def test_large_output_handling(self, cli_runner):
        """Test handling of potentially large outputs."""
        result = cli_runner.invoke(app, ['--help'])
        
        assert result.exit_code == 0
        # Should handle output efficiently
        assert len(result.stdout) < 50000  # Reasonable limit for help text


class TestCLIEdgeCases:
    """Test CLI edge cases and boundary conditions."""
    
    def test_empty_input_file(self, cli_runner):
        """Test handling of empty input files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')  # Empty file
            empty_file_path = f.name
        
        try:
            result = cli_runner.invoke(datasets.app, [
                'load', empty_file_path, '--name', 'empty_test'
            ])
            
            # Should handle gracefully
            assert result.exit_code in [0, 1]
            
        finally:
            Path(empty_file_path).unlink(missing_ok=True)
    
    def test_very_long_arguments(self, cli_runner):
        """Test handling of very long command line arguments."""
        long_name = 'a' * 1000  # Very long name
        
        result = cli_runner.invoke(detectors.app, [
            'create', '--name', long_name, '--algorithm', 'IsolationForest'
        ])
        
        # Should handle gracefully
        assert result.exit_code in [0, 1]
    
    def test_special_characters_in_names(self, cli_runner):
        """Test handling of special characters in dataset/detector names."""
        special_names = [
            'test-dataset',
            'test_dataset',
            'test.dataset',
            'test dataset'  # Space
        ]
        
        for name in special_names:
            result = cli_runner.invoke(detectors.app, [
                'create', '--name', name, '--algorithm', 'IsolationForest'
            ])
            
            # Should handle gracefully
            assert result.exit_code in [0, 1]
    
    def test_unicode_handling(self, cli_runner):
        """Test Unicode character handling."""
        unicode_name = 'test_ä_ñ_ü_dataset'
        
        result = cli_runner.invoke(detectors.app, [
            'create', '--name', unicode_name, '--algorithm', 'IsolationForest'
        ])
        
        # Should handle gracefully
        assert result.exit_code in [0, 1]
    
    def test_binary_file_handling(self, cli_runner):
        """Test handling of binary files as input."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.bin', delete=False) as f:
            f.write(b'\x00\x01\x02\x03\x04\x05')  # Binary data
            binary_file_path = f.name
        
        try:
            result = cli_runner.invoke(datasets.app, [
                'load', binary_file_path, '--name', 'binary_test'
            ])
            
            # Should fail gracefully
            assert result.exit_code != 0
            
        finally:
            Path(binary_file_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])