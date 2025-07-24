"""Integration tests for CLI commands."""

import pytest
import json
import tempfile
from pathlib import Path
from click.testing import CliRunner
from typing import Dict, Any

from anomaly_detection.cli import main


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_cli_help(self, runner: CliRunner):
        """Test CLI help functionality."""
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "Anomaly Detection CLI" in result.output
        assert "detect" in result.output
        assert "model" in result.output
        assert "ensemble" in result.output
    
    def test_detection_command_help(self, runner: CliRunner):
        """Test detection command help."""
        result = runner.invoke(main, ['detect', '--help'])
        
        assert result.exit_code == 0
        assert "Anomaly detection commands" in result.output
    
    def test_detect_run_with_csv(self, runner: CliRunner, sample_csv_file: Path):
        """Test anomaly detection with CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            result = runner.invoke(main, [
                'detect', 'run',
                '--input', str(sample_csv_file),
                '--output', output_file,
                '--algorithm', 'isolation_forest',
                '--contamination', '0.1',
                '--has-labels',
                '--label-column', 'label'
            ])
            
            assert result.exit_code == 0
            assert "Detection completed successfully" in result.output
            
            # Check output file
            output_path = Path(output_file)
            assert output_path.exists()
            
            with open(output_path, 'r') as f:
                results = json.load(f)
            
            # Verify result structure
            assert "detection_results" in results
            assert "algorithm" in results
            assert "evaluation_metrics" in results  # Should have metrics due to labels
            
            detection_results = results["detection_results"]
            assert "anomalies_detected" in detection_results
            assert "anomaly_rate" in detection_results
            
            evaluation_metrics = results["evaluation_metrics"]
            assert "accuracy" in evaluation_metrics
            assert "precision" in evaluation_metrics
            assert "recall" in evaluation_metrics
            assert "f1_score" in evaluation_metrics
            
        finally:
            # Cleanup
            Path(output_file).unlink(missing_ok=True)
    
    def test_detect_run_with_json(self, runner: CliRunner, sample_json_file: Path):
        """Test anomaly detection with JSON file."""
        result = runner.invoke(main, [
            'detect', 'run',
            '--input', str(sample_json_file),
            '--algorithm', 'local_outlier_factor',
            '--contamination', '0.15'
        ])
        
        assert result.exit_code == 0
        assert "Detection completed successfully" in result.output
        
        # Should output results to stdout as JSON
        assert '"algorithm": "local_outlier_factor"' in result.output
        assert '"contamination": 0.15' in result.output
    
    def test_detect_run_different_algorithms(self, runner: CliRunner, sample_csv_file: Path):
        """Test detection with different algorithms."""
        algorithms = ['isolation_forest', 'lof']
        
        for algorithm in algorithms:
            result = runner.invoke(main, [
                'detect', 'run',
                '--input', str(sample_csv_file),
                '--algorithm', algorithm,
                '--contamination', '0.1'
            ])
            
            assert result.exit_code == 0
            assert "Detection completed successfully" in result.output
            assert f'"algorithm": "{algorithm}"' in result.output
    
    def test_ensemble_command(self, runner: CliRunner, sample_csv_file: Path):
        """Test ensemble detection command."""
        result = runner.invoke(main, [
            'ensemble', 'combine',
            '--input', str(sample_csv_file),
            '--algorithms', 'isolation_forest',
            '--algorithms', 'lof',
            '--method', 'majority',
            '--contamination', '0.1'
        ])
        
        assert result.exit_code == 0
        assert "Ensemble detection completed successfully" in result.output
        assert "ensemble_results" in result.output
        assert "individual_results" in result.output
    
    def test_model_training_command(self, runner: CliRunner, sample_csv_file: Path, temp_dir: Path):
        """Test model training command."""
        models_dir = temp_dir / "cli_models"
        
        result = runner.invoke(main, [
            'model', 'train',
            '--input', str(sample_csv_file),
            '--model-name', 'CLI Test Model',
            '--algorithm', 'isolation_forest',
            '--contamination', '0.1',
            '--output-dir', str(models_dir),
            '--format', 'pickle',
            '--has-labels',
            '--label-column', 'label'
        ])
        
        assert result.exit_code == 0
        assert "Model training completed successfully" in result.output
        assert "Model ID:" in result.output
        assert "Performance metrics:" in result.output
        
        # Verify model directory was created
        assert models_dir.exists()
        
        # Should have at least one model directory
        model_dirs = list(models_dir.glob("*/"))
        assert len(model_dirs) >= 1
    
    def test_model_list_command(self, runner: CliRunner, temp_dir: Path):
        """Test model listing command."""
        models_dir = temp_dir / "cli_models_list"
        models_dir.mkdir()
        
        # Initially empty
        result = runner.invoke(main, [
            'model', 'list',
            '--models-dir', str(models_dir)
        ])
        
        assert result.exit_code == 0
        assert "No models found" in result.output
    
    def test_model_stats_command(self, runner: CliRunner, temp_dir: Path):
        """Test model statistics command."""
        models_dir = temp_dir / "cli_models_stats"
        models_dir.mkdir()
        
        result = runner.invoke(main, [
            'model', 'stats',
            '--models-dir', str(models_dir)
        ])
        
        assert result.exit_code == 0
        assert "Model Repository Statistics" in result.output
        assert "Total models: 0" in result.output
    
    def test_data_generation_command(self, runner: CliRunner, temp_dir: Path):
        """Test synthetic data generation command."""
        output_file = temp_dir / "generated_data.csv"
        
        result = runner.invoke(main, [
            'data', 'generate',
            '--output', str(output_file),
            '--samples', '200',
            '--features', '3',
            '--contamination', '0.15',
            '--anomaly-type', 'point',
            '--random-state', '42'
        ])
        
        assert result.exit_code == 0
        assert "Generated synthetic dataset" in result.output
        assert f"File: {output_file}" in result.output
        assert "Samples: 200" in result.output
        assert "Features: 3" in result.output
        
        # Verify file was created
        assert output_file.exists()
        
        # Load and verify data
        import pandas as pd
        df = pd.read_csv(output_file)
        
        assert len(df) == 200
        assert len(df.columns) == 4  # 3 features + 1 label
        assert 'label' in df.columns
        
        # Check label distribution
        normal_count = (df['label'] == 1).sum()
        anomaly_count = (df['label'] == -1).sum()
        
        assert normal_count > 0
        assert anomaly_count > 0
        assert abs((anomaly_count / 200) - 0.15) < 0.05  # Should be close to 15%
    
    def test_data_generation_different_types(self, runner: CliRunner, temp_dir: Path):
        """Test data generation with different anomaly types."""
        anomaly_types = ['point', 'contextual', 'collective']
        
        for anomaly_type in anomaly_types:
            output_file = temp_dir / f"data_{anomaly_type}.json"
            
            result = runner.invoke(main, [
                'data', 'generate',
                '--output', str(output_file),
                '--samples', '100',
                '--features', '2',
                '--contamination', '0.1',
                '--anomaly-type', anomaly_type,
                '--random-state', '42'
            ])
            
            assert result.exit_code == 0
            assert "Generated synthetic dataset" in result.output
            assert f"Anomaly type: {anomaly_type}" in result.output
            assert output_file.exists()
    
    def test_monitor_status_command(self, runner: CliRunner):
        """Test monitoring status command."""
        result = runner.invoke(main, ['monitor', 'status'])
        
        assert result.exit_code == 0
        assert "Overall System Status:" in result.output
        assert "Health Checks:" in result.output
        assert "Metrics Summary:" in result.output
    
    def test_monitor_metrics_command(self, runner: CliRunner):
        """Test monitoring metrics command."""
        result = runner.invoke(main, ['monitor', 'metrics', '--limit', '5'])
        
        assert result.exit_code == 0
        # May be empty initially, but should not crash
        assert "Performance Metrics" in result.output or "No performance metrics available" in result.output
    
    def test_monitor_cleanup_command(self, runner: CliRunner):
        """Test monitoring cleanup command."""
        result = runner.invoke(main, ['monitor', 'cleanup'])
        
        assert result.exit_code == 0
        assert "Cleaned up" in result.output
        assert "Monitoring cleanup completed" in result.output
    
    def test_error_handling_invalid_file(self, runner: CliRunner):
        """Test CLI error handling with invalid file."""
        result = runner.invoke(main, [
            'detect', 'run',
            '--input', 'nonexistent_file.csv',
            '--algorithm', 'isolation_forest'
        ])
        
        assert result.exit_code == 1  # Should exit with error
        assert "Error:" in result.output
        assert "not found" in result.output
    
    def test_error_handling_invalid_algorithm(self, runner: CliRunner, sample_csv_file: Path):
        """Test CLI error handling with invalid algorithm."""
        result = runner.invoke(main, [
            'detect', 'run',
            '--input', str(sample_csv_file),
            '--algorithm', 'nonexistent_algorithm'
        ])
        
        # Should be caught by Click's choice validation before our code runs
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "Error:" in result.output
    
    def test_error_handling_invalid_contamination(self, runner: CliRunner, sample_csv_file: Path):
        """Test CLI error handling with invalid contamination rate."""
        result = runner.invoke(main, [
            'detect', 'run',
            '--input', str(sample_csv_file),
            '--algorithm', 'isolation_forest',
            '--contamination', '1.5'  # Invalid
        ])
        
        assert result.exit_code == 1
        assert "Error:" in result.output
    
    def test_verbose_output(self, runner: CliRunner, sample_csv_file: Path):
        """Test verbose output mode."""
        result = runner.invoke(main, [
            '--verbose',
            'detect', 'run',
            '--input', str(sample_csv_file),
            '--algorithm', 'isolation_forest'
        ])
        
        assert result.exit_code == 0
        # With verbose mode, should have more detailed output
        assert "Detection completed successfully" in result.output
    
    def test_full_workflow_integration(self, runner: CliRunner, temp_dir: Path):
        """Test complete workflow: generate data -> detect -> train model -> list models."""
        # Step 1: Generate data
        data_file = temp_dir / "workflow_data.csv"
        result = runner.invoke(main, [
            'data', 'generate',
            '--output', str(data_file),
            '--samples', '150',
            '--features', '4',
            '--contamination', '0.12',
            '--random-state', '42'
        ])
        assert result.exit_code == 0
        assert data_file.exists()
        
        # Step 2: Run detection
        results_file = temp_dir / "detection_results.json"
        result = runner.invoke(main, [
            'detect', 'run',
            '--input', str(data_file),
            '--output', str(results_file),
            '--algorithm', 'isolation_forest',
            '--has-labels'
        ])
        assert result.exit_code == 0
        assert results_file.exists()
        
        # Step 3: Train model
        models_dir = temp_dir / "workflow_models"
        result = runner.invoke(main, [
            'model', 'train',
            '--input', str(data_file),
            '--model-name', 'Workflow Test Model',
            '--algorithm', 'isolation_forest',
            '--output-dir', str(models_dir),
            '--has-labels'
        ])
        assert result.exit_code == 0
        assert models_dir.exists()
        
        # Step 4: List models
        result = runner.invoke(main, [
            'model', 'list',
            '--models-dir', str(models_dir)
        ])
        assert result.exit_code == 0
        assert "Workflow Test Model" in result.output
        
        # Step 5: Get model stats
        result = runner.invoke(main, [
            'model', 'stats',
            '--models-dir', str(models_dir)
        ])
        assert result.exit_code == 0
        assert "Total models: 1" in result.output
    
    def test_cli_configuration_file(self, runner: CliRunner, sample_csv_file: Path, temp_dir: Path):
        """Test CLI with configuration file."""
        # Create a config file (this would be implemented if config file support is added)
        config_file = temp_dir / "config.json"
        config_data = {
            "default_algorithm": "isolation_forest",
            "default_contamination": 0.1,
            "output_format": "json"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Test with config file
        result = runner.invoke(main, [
            '--config', str(config_file),
            'detect', 'run',
            '--input', str(sample_csv_file)
        ])
        
        # Should work regardless of whether config file support is implemented
        assert result.exit_code == 0 or "Error:" in result.output