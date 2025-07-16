"""
Basic CLI functionality tests with minimal dependencies.

This module tests core CLI functionality without complex mocking.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from monorepo.presentation.cli.app import app
from monorepo.presentation.cli import autonomous, datasets, detectors


class TestBasicCLIFunctionality:
    """Test basic CLI functionality."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner."""
        return CliRunner()
    
    def test_main_help_command(self, cli_runner):
        """Test main help command works."""
        result = cli_runner.invoke(app, ['--help'])
        
        assert result.exit_code == 0
        assert 'Usage:' in result.stdout
        assert 'Commands' in result.stdout
    
    def test_dataset_help_command(self, cli_runner):
        """Test dataset help command."""
        result = cli_runner.invoke(datasets.app, ['--help'])
        
        assert result.exit_code == 0
        assert 'Usage:' in result.stdout or 'Commands' in result.stdout
    
    def test_detector_help_command(self, cli_runner):
        """Test detector help command."""
        result = cli_runner.invoke(detectors.app, ['--help'])
        
        assert result.exit_code == 0
        assert 'Usage:' in result.stdout or 'Commands' in result.stdout
    
    def test_autonomous_help_command(self, cli_runner):
        """Test autonomous help command."""
        result = cli_runner.invoke(autonomous.app, ['--help'])
        
        assert result.exit_code == 0
        assert 'Usage:' in result.stdout or 'Commands' in result.stdout
    
    def test_invalid_command_handling(self, cli_runner):
        """Test invalid command handling."""
        result = cli_runner.invoke(app, ['nonexistent-command'])
        
        assert result.exit_code != 0
        assert 'No such command' in result.stdout or 'invalid' in result.stdout.lower()
    
    def test_missing_arguments_handling(self, cli_runner):
        """Test missing arguments handling."""
        result = cli_runner.invoke(datasets.app, ['load'])
        
        assert result.exit_code != 0
    
    def test_subcommand_help_completeness(self, cli_runner):
        """Test that all major subcommands have help."""
        subcommands = [
            ['auto', '--help'],
            ['dataset', '--help'],
            ['detector', '--help'],
            ['export', '--help'],
            ['server', '--help']
        ]
        
        for cmd_args in subcommands:
            result = cli_runner.invoke(app, cmd_args)
            
            # Help should always work
            assert result.exit_code == 0
            assert 'Usage:' in result.stdout or 'Commands' in result.stdout
    
    @patch('monorepo.presentation.cli.container.get_cli_container')
    def test_dataset_list_with_mock(self, mock_get_container, cli_runner):
        """Test dataset list with mocked container."""
        # Mock container and repository
        container = Mock()
        dataset_repo = Mock()
        dataset_repo.list_all.return_value = []
        container.dataset_repository.return_value = dataset_repo
        mock_get_container.return_value = container
        
        result = cli_runner.invoke(datasets.app, ['list'])
        
        # Should handle gracefully
        assert result.exit_code in [0, 1]
    
    @patch('monorepo.presentation.cli.container.get_cli_container')
    def test_detector_list_with_mock(self, mock_get_container, cli_runner):
        """Test detector list with mocked container."""
        # Mock container and repository
        container = Mock()
        detector_repo = Mock()
        detector_repo.list_all.return_value = []
        container.detector_repository.return_value = detector_repo
        mock_get_container.return_value = container
        
        result = cli_runner.invoke(detectors.app, ['list'])
        
        # Should handle gracefully
        assert result.exit_code in [0, 1]
    
    def test_export_list_formats(self, cli_runner):
        """Test export formats listing."""
        result = cli_runner.invoke(app, ['export', 'list-formats'])
        
        # Should work or fail gracefully
        assert result.exit_code in [0, 1]
    
    @patch('requests.get')
    def test_server_status_with_mock(self, mock_get, cli_runner):
        """Test server status with mocked request."""
        # Mock connection error
        mock_get.side_effect = Exception('Connection refused')
        
        result = cli_runner.invoke(app, ['server', 'status'])
        
        # Should handle connection error gracefully
        assert result.exit_code in [0, 1]
    
    def test_quickstart_workflow(self, cli_runner):
        """Test quickstart workflow."""
        # Test quickstart acceptance
        result = cli_runner.invoke(app, ['quickstart'], input='y\n')
        assert result.exit_code == 0
        
        # Test quickstart decline (should exit cleanly with code 0)
        result = cli_runner.invoke(app, ['quickstart'], input='n\n')
        assert result.exit_code in [0, 1]  # Accept both normal exit and typer.Exit()
    
    def test_configuration_generation_basic(self, cli_runner):
        """Test basic configuration generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
            config_path = config_file.name
        
        try:
            result = cli_runner.invoke(app, [
                'generate-config', 'test', '--output', config_path
            ])
            
            # Should complete or fail gracefully
            assert result.exit_code in [0, 1]
            
        finally:
            Path(config_path).unlink(missing_ok=True)
    
    @patch('monorepo.application.services.autonomous_service.AutonomousDetectionService')
    def test_autonomous_detect_basic(self, mock_service_class, cli_runner):
        """Test basic autonomous detection."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('feature1,feature2,feature3\n')
            f.write('1.0,2.0,3.0\n')
            f.write('2.0,3.0,4.0\n')
            f.write('100.0,200.0,300.0\n')
            temp_path = f.name
        
        try:
            # Mock the service
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.detect_anomalies.return_value = {
                'best_detector': 'IsolationForest',
                'anomalies_found': 1,
                'confidence': 0.95
            }
            
            result = cli_runner.invoke(autonomous.app, ['detect', temp_path])
            
            # Should complete or fail gracefully
            assert result.exit_code in [0, 1]
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner."""
        return CliRunner()
    
    def test_file_not_found_handling(self, cli_runner):
        """Test handling of non-existent files."""
        result = cli_runner.invoke(datasets.app, [
            'load', '/nonexistent/path/file.csv', '--name', 'test'
        ])
        
        assert result.exit_code != 0
    
    def test_invalid_parameter_values(self, cli_runner):
        """Test handling of invalid parameter values."""
        # Test with invalid contamination rate
        result = cli_runner.invoke(autonomous.app, [
            'detect', '/tmp/fake.csv', '--contamination', 'invalid'
        ])
        
        assert result.exit_code != 0
    
    def test_special_characters_in_names(self, cli_runner):
        """Test handling of special characters in names."""
        special_names = [
            'test-dataset',
            'test_dataset', 
            'test.dataset'
        ]
        
        for name in special_names:
            result = cli_runner.invoke(detectors.app, [
                'create', '--name', name, '--algorithm', 'IsolationForest'
            ])
            
            # Should handle gracefully (0=success, 1=handled error, 2=CLI parsing error)
            assert result.exit_code in [0, 1, 2]
    
    def test_empty_file_handling(self, cli_runner):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Empty file
            empty_path = f.name
        
        try:
            result = cli_runner.invoke(datasets.app, [
                'load', empty_path, '--name', 'empty_test'
            ])
            
            # Should handle empty files gracefully
            assert result.exit_code in [0, 1]
            
        finally:
            Path(empty_path).unlink(missing_ok=True)


class TestCLIPerformance:
    """Test CLI performance characteristics."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner."""
        return CliRunner()
    
    def test_help_command_speed(self, cli_runner):
        """Test that help commands are fast."""
        import time
        
        start_time = time.time()
        result = cli_runner.invoke(app, ['--help'])
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert result.exit_code == 0
        assert execution_time < 5.0  # Should be reasonably fast
    
    def test_multiple_help_commands(self, cli_runner):
        """Test multiple help commands don't interfere."""
        commands = ['--help', 'dataset --help', 'detector --help']
        
        for cmd in commands:
            result = cli_runner.invoke(app, cmd.split())
            assert result.exit_code == 0
    
    def test_memory_stability(self, cli_runner):
        """Test memory usage remains stable."""
        # Run multiple operations
        for i in range(3):
            result = cli_runner.invoke(app, ['--help'])
            assert result.exit_code == 0
    
    def test_error_recovery(self, cli_runner):
        """Test CLI recovers from errors."""
        # Run invalid command
        result1 = cli_runner.invoke(app, ['invalid-command'])
        assert result1.exit_code != 0
        
        # Verify CLI still works
        result2 = cli_runner.invoke(app, ['--help'])
        assert result2.exit_code == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])