"""Comprehensive tests for configuration management CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from uuid import UUID, uuid4

import pytest
from typer.testing import CliRunner

from pynomaly.presentation.cli.config import app


class TestConfigCommands:
    """Test suite for configuration management CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()
    
    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            "name": "test_config",
            "algorithm": "IsolationForest",
            "contamination": 0.1,
            "random_state": 42,
            "hyperparameters": {
                "n_estimators": 100,
                "max_features": 1.0
            }
        }
    
    @pytest.fixture
    def sample_configuration(self):
        """Sample configuration object for testing."""
        config = Mock()
        config.id = uuid4()
        config.name = "test_config"
        config.status = "active"
        config.is_valid = True
        config.validation_errors = []
        config.validation_warnings = []
        
        # Algorithm config
        algo_config = Mock()
        algo_config.algorithm_name = "IsolationForest"
        algo_config.contamination = 0.1
        algo_config.random_state = 42
        algo_config.hyperparameters = {"n_estimators": 100}
        config.algorithm_config = algo_config
        
        # Dataset config
        dataset_config = Mock()
        dataset_config.dataset_path = "/path/to/dataset.csv"
        dataset_config.dataset_name = "test_dataset"
        dataset_config.feature_columns = ["feature1", "feature2"]
        config.dataset_config = dataset_config
        
        # Metadata
        metadata = Mock()
        metadata.source = "cli"
        metadata.created_at = Mock()
        metadata.created_at.strftime.return_value = "2025-01-01 12:00"
        metadata.tags = ["test", "experiment"]
        config.metadata = metadata
        
        # Performance results
        performance = Mock()
        performance.accuracy = 0.95
        performance.precision = 0.92
        performance.recall = 0.88
        performance.f1_score = 0.90
        performance.training_time_seconds = 10.5
        config.performance_results = performance
        
        # Lineage
        lineage = Mock()
        lineage.parent_configurations = []
        lineage.derivation_method = "manual"
        lineage.git_commit = "abc123"
        config.lineage = lineage
        
        # Model dump
        config.model_dump.return_value = {
            "id": str(config.id),
            "name": config.name,
            "algorithm_config": {"algorithm_name": "IsolationForest"},
            "metadata": {"source": "cli", "tags": ["test"]}
        }
        
        return config

    @patch('pynomaly.infrastructure.config.feature_flags.require_feature')
    @patch('pynomaly.presentation.cli.config.ConfigurationCaptureService')
    def test_capture_configuration_success(self, mock_service_class, mock_require_feature, runner, sample_config_data):
        """Test successful configuration capture."""
        mock_require_feature.return_value = lambda f: f  # Mock decorator
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock response
        mock_response = Mock()
        mock_response.success = True
        mock_config = Mock()
        mock_config.name = "test_config"
        mock_config.id = uuid4()
        mock_config.algorithm_config.algorithm_name = "IsolationForest"
        mock_config.is_valid = True
        mock_config.validation_errors = []
        mock_config.validation_warnings = []
        mock_config.model_dump.return_value = {"name": "test_config"}
        mock_response.configuration = mock_config
        
        # Mock async call
        with patch('asyncio.run', return_value=mock_response):
            # Create test parameters file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(sample_config_data, f)
                params_file = f.name
            
            result = runner.invoke(app, [
                "capture", "automl",
                "--params", params_file,
                "--description", "Test configuration",
                "--tag", "test",
                "--tag", "experiment"
            ])
            
            assert result.exit_code == 0
            assert "Configuration captured successfully" in result.stdout
            assert "Configuration is valid" in result.stdout
            assert str(mock_config.id) in result.stdout
            
            # Verify service was called
            mock_service.capture_configuration.assert_called_once()

    @patch('pynomaly.infrastructure.config.feature_flags.require_feature')
    @patch('pynomaly.presentation.cli.config.ConfigurationCaptureService')
    def test_capture_configuration_with_output_file(self, mock_service_class, mock_require_feature, runner, sample_config_data):
        """Test configuration capture with output file."""
        mock_require_feature.return_value = lambda f: f
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock response
        mock_response = Mock()
        mock_response.success = True
        mock_config = Mock()
        mock_config.name = "test_config"
        mock_config.id = uuid4()
        mock_config.algorithm_config.algorithm_name = "IsolationForest"
        mock_config.is_valid = True
        mock_config.validation_errors = []
        mock_config.validation_warnings = []
        mock_config.model_dump.return_value = {"name": "test_config"}
        mock_response.configuration = mock_config
        
        with patch('asyncio.run', return_value=mock_response):
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create test parameters file
                params_file = Path(tmp_dir) / "params.json"
                params_file.write_text(json.dumps(sample_config_data))
                
                # Output file
                output_file = Path(tmp_dir) / "config.json"
                
                result = runner.invoke(app, [
                    "capture", "cli",
                    "--params", str(params_file),
                    "--output", str(output_file)
                ])
                
                assert result.exit_code == 0
                assert "Configuration captured successfully" in result.stdout
                assert f"Saved to: {output_file}" in result.stdout
                assert output_file.exists()

    @patch('pynomaly.infrastructure.config.feature_flags.require_feature')
    @patch('pynomaly.presentation.cli.config.ConfigurationCaptureService')
    def test_capture_configuration_validation_issues(self, mock_service_class, mock_require_feature, runner, sample_config_data):
        """Test configuration capture with validation issues."""
        mock_require_feature.return_value = lambda f: f
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock response with validation issues
        mock_response = Mock()
        mock_response.success = True
        mock_config = Mock()
        mock_config.name = "test_config"
        mock_config.id = uuid4()
        mock_config.algorithm_config.algorithm_name = "IsolationForest"
        mock_config.is_valid = False
        mock_config.validation_errors = ["Invalid contamination value"]
        mock_config.validation_warnings = ["Missing feature columns"]
        mock_config.model_dump.return_value = {"name": "test_config"}
        mock_response.configuration = mock_config
        
        with patch('asyncio.run', return_value=mock_response):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(sample_config_data, f)
                params_file = f.name
            
            result = runner.invoke(app, [
                "capture", "automl",
                "--params", params_file
            ])
            
            assert result.exit_code == 0
            assert "Configuration has validation issues" in result.stdout
            assert "Invalid contamination value" in result.stdout
            assert "Missing feature columns" in result.stdout

    @patch('pynomaly.infrastructure.config.feature_flags.require_feature')
    @patch('pynomaly.presentation.cli.config.ConfigurationCaptureService')
    def test_capture_configuration_failure(self, mock_service_class, mock_require_feature, runner, sample_config_data):
        """Test configuration capture failure."""
        mock_require_feature.return_value = lambda f: f
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock failure response
        mock_response = Mock()
        mock_response.success = False
        mock_response.message = "Configuration validation failed"
        mock_response.errors = ["Missing required parameter", "Invalid algorithm"]
        
        with patch('asyncio.run', return_value=mock_response):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(sample_config_data, f)
                params_file = f.name
            
            result = runner.invoke(app, [
                "capture", "automl",
                "--params", params_file
            ])
            
            assert result.exit_code == 1
            assert "Configuration validation failed" in result.stdout
            assert "Missing required parameter" in result.stdout
            assert "Invalid algorithm" in result.stdout

    def test_capture_configuration_params_file_not_found(self, runner):
        """Test configuration capture with missing parameters file."""
        result = runner.invoke(app, [
            "capture", "automl",
            "--params", "/nonexistent/params.json"
        ])
        
        assert result.exit_code == 1
        assert "Parameters file not found" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationCaptureService')
    def test_export_configurations_success(self, mock_service_class, runner):
        """Test successful configuration export."""
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock response
        mock_response = Mock()
        mock_response.success = True
        mock_response.export_files = ["/tmp/config1.json", "/tmp/config2.json"]
        
        with patch('asyncio.run', return_value=mock_response):
            config_id = str(uuid4())
            
            result = runner.invoke(app, [
                "export", config_id,
                "--output", "/tmp/export.json",
                "--format", "json",
                "--metadata",
                "--performance"
            ])
            
            assert result.exit_code == 0
            assert "Exported 1 configurations" in result.stdout
            assert "Format: json" in result.stdout
            assert "Output: /tmp/export.json" in result.stdout
            assert "/tmp/config1.json" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationCaptureService')
    def test_export_configurations_multiple_ids(self, mock_service_class, runner):
        """Test exporting multiple configurations."""
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock response
        mock_response = Mock()
        mock_response.success = True
        mock_response.export_files = ["/tmp/export.json"]
        
        with patch('asyncio.run', return_value=mock_response):
            config_id1 = str(uuid4())
            config_id2 = str(uuid4())
            
            result = runner.invoke(app, [
                "export", config_id1, config_id2,
                "--output", "/tmp/export.json",
                "--format", "yaml",
                "--template", "experiment_template"
            ])
            
            assert result.exit_code == 0
            assert "Exported 2 configurations" in result.stdout
            assert "Format: yaml" in result.stdout

    def test_export_configurations_invalid_uuid(self, runner):
        """Test export with invalid UUID."""
        result = runner.invoke(app, [
            "export", "invalid-uuid",
            "--output", "/tmp/export.json"
        ])
        
        assert result.exit_code == 1
        assert "Invalid UUID format: invalid-uuid" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationCaptureService')
    def test_export_configurations_failure(self, mock_service_class, runner):
        """Test export configuration failure."""
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock failure response
        mock_response = Mock()
        mock_response.success = False
        mock_response.message = "Export failed"
        mock_response.errors = ["Configuration not found", "Export format not supported"]
        
        with patch('asyncio.run', return_value=mock_response):
            config_id = str(uuid4())
            
            result = runner.invoke(app, [
                "export", config_id,
                "--output", "/tmp/export.json"
            ])
            
            assert result.exit_code == 1
            assert "Export failed" in result.stdout
            assert "Configuration not found" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_import_configurations_success(self, mock_repo_class, runner):
        """Test successful configuration import."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock async import
        with patch('asyncio.run', return_value=3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                import_data = {
                    "configurations": [
                        {"name": "config1", "algorithm_config": {"algorithm_name": "IsolationForest"}},
                        {"name": "config2", "algorithm_config": {"algorithm_name": "LOF"}},
                        {"name": "config3", "algorithm_config": {"algorithm_name": "OneClassSVM"}}
                    ]
                }
                json.dump(import_data, f)
                import_file = f.name
            
            result = runner.invoke(app, [
                "import", import_file,
                "--storage", "/tmp/configs"
            ])
            
            assert result.exit_code == 0
            assert "Successfully imported 3 configurations" in result.stdout
            assert "Storage: /tmp/configs" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_import_configurations_dry_run(self, mock_repo_class, runner):
        """Test configuration import dry run."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import_data = {
                "configurations": [
                    {"name": "config1", "algorithm_config": {"algorithm_name": "IsolationForest"}},
                    {"name": "config2", "algorithm_config": {"algorithm_name": "LOF"}}
                ]
            }
            json.dump(import_data, f)
            import_file = f.name
        
        result = runner.invoke(app, [
            "import", import_file,
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        assert "Import Preview:" in result.stdout
        assert "Found 2 configurations to import" in result.stdout
        assert "config1 (IsolationForest)" in result.stdout
        assert "config2 (LOF)" in result.stdout

    def test_import_configurations_file_not_found(self, runner):
        """Test import with missing file."""
        result = runner.invoke(app, [
            "import", "/nonexistent/file.json"
        ])
        
        assert result.exit_code == 1
        assert "Import file not found" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_list_configurations_success(self, mock_repo_class, runner, sample_configuration):
        """Test successful configuration listing."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock async list
        with patch('asyncio.run', return_value=[sample_configuration]):
            result = runner.invoke(app, [
                "list",
                "--storage", "/tmp/configs",
                "--limit", "10"
            ])
            
            assert result.exit_code == 0
            assert "Configurations (1 found)" in result.stdout
            assert "test_config" in result.stdout
            assert "IsolationForest" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_list_configurations_detailed(self, mock_repo_class, runner, sample_configuration):
        """Test detailed configuration listing."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock async list
        with patch('asyncio.run', return_value=[sample_configuration]):
            result = runner.invoke(app, [
                "list",
                "--details",
                "--source", "cli",
                "--algorithm", "IsolationForest"
            ])
            
            assert result.exit_code == 0
            assert "test_config" in result.stdout
            assert "IsolationForest" in result.stdout
            assert "0.950" in result.stdout  # Accuracy

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_list_configurations_empty(self, mock_repo_class, runner):
        """Test listing when no configurations exist."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock async list returning empty
        with patch('asyncio.run', return_value=[]):
            result = runner.invoke(app, ["list"])
            
            assert result.exit_code == 0
            assert "No configurations found" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_search_configurations_success(self, mock_repo_class, runner, sample_configuration):
        """Test successful configuration search."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock async search
        with patch('asyncio.run', return_value=[sample_configuration]):
            result = runner.invoke(app, [
                "search", "isolation",
                "--tag", "test",
                "--source", "cli",
                "--min-accuracy", "0.9",
                "--sort", "accuracy",
                "--order", "desc"
            ])
            
            assert result.exit_code == 0
            assert "Found 1 configurations matching 'isolation'" in result.stdout
            assert "test_config" in result.stdout
            assert "IsolationForest" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_search_configurations_no_results(self, mock_repo_class, runner):
        """Test search with no results."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock async search returning empty
        with patch('asyncio.run', return_value=[]):
            result = runner.invoke(app, [
                "search", "nonexistent"
            ])
            
            assert result.exit_code == 0
            assert "No configurations found matching query: 'nonexistent'" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_show_configuration_success(self, mock_repo_class, runner, sample_configuration):
        """Test successful configuration show."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock async load
        with patch('asyncio.run', return_value=sample_configuration):
            config_id = str(uuid4())
            
            result = runner.invoke(app, [
                "show", config_id,
                "--performance",
                "--lineage"
            ])
            
            assert result.exit_code == 0
            assert "test_config" in result.stdout
            assert "IsolationForest" in result.stdout
            assert "Basic Information" in result.stdout
            assert "Algorithm Configuration" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_show_configuration_json_format(self, mock_repo_class, runner, sample_configuration):
        """Test show configuration in JSON format."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock async load
        with patch('asyncio.run', return_value=sample_configuration):
            config_id = str(uuid4())
            
            result = runner.invoke(app, [
                "show", config_id,
                "--format", "json"
            ])
            
            assert result.exit_code == 0
            assert "test_config" in result.stdout
            assert "IsolationForest" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_show_configuration_yaml_format(self, mock_repo_class, runner, sample_configuration):
        """Test show configuration in YAML format."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock async load
        with patch('asyncio.run', return_value=sample_configuration):
            config_id = str(uuid4())
            
            with patch('yaml.dump', return_value="name: test_config\n"):
                result = runner.invoke(app, [
                    "show", config_id,
                    "--format", "yaml"
                ])
                
                assert result.exit_code == 0
                assert "test_config" in result.stdout

    def test_show_configuration_invalid_uuid(self, runner):
        """Test show configuration with invalid UUID."""
        result = runner.invoke(app, [
            "show", "invalid-uuid"
        ])
        
        assert result.exit_code == 1
        assert "Invalid UUID format: invalid-uuid" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_show_configuration_not_found(self, mock_repo_class, runner):
        """Test show configuration when not found."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock async load returning None
        with patch('asyncio.run', return_value=None):
            config_id = str(uuid4())
            
            result = runner.invoke(app, [
                "show", config_id
            ])
            
            assert result.exit_code == 1
            assert "Configuration not found" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_show_statistics_success(self, mock_repo_class, runner):
        """Test successful statistics display."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock statistics
        stats = {
            "total_configurations": 15,
            "total_collections": 3,
            "total_templates": 2,
            "total_backups": 5,
            "storage_size_bytes": 1024 * 1024 * 10,  # 10 MB
            "storage_path": "/tmp/configs",
            "versioning_enabled": True,
            "compression_enabled": False,
            "backup_enabled": True,
            "last_modified": "2025-01-01 12:00:00",
            "repository_created": "2024-12-01 10:00:00"
        }
        mock_repo.get_repository_statistics.return_value = stats
        
        result = runner.invoke(app, ["stats"])
        
        assert result.exit_code == 0
        assert "Configuration Repository Statistics" in result.stdout
        assert "Total Configurations: 15" in result.stdout
        assert "10.00 MB" in result.stdout
        assert "Versioning Enabled: ✓" in result.stdout
        assert "Compression Enabled: ✗" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_show_statistics_no_data(self, mock_repo_class, runner):
        """Test statistics when no data available."""
        # Mock repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Mock empty statistics
        mock_repo.get_repository_statistics.return_value = None
        
        result = runner.invoke(app, ["stats"])
        
        assert result.exit_code == 0
        assert "No statistics available" in result.stdout


class TestConfigCommandsHelp:
    """Test help functionality for config commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_config_help(self, runner):
        """Test config command help."""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Configuration management commands" in result.stdout

    def test_capture_help(self, runner):
        """Test capture command help."""
        result = runner.invoke(app, ["capture", "--help"])
        
        assert result.exit_code == 0
        assert "Capture configuration from raw parameters" in result.stdout

    def test_export_help(self, runner):
        """Test export command help."""
        result = runner.invoke(app, ["export", "--help"])
        
        assert result.exit_code == 0
        assert "Export configurations to specified format" in result.stdout

    def test_import_help(self, runner):
        """Test import command help."""
        result = runner.invoke(app, ["import", "--help"])
        
        assert result.exit_code == 0
        assert "Import configurations from file" in result.stdout

    def test_list_help(self, runner):
        """Test list command help."""
        result = runner.invoke(app, ["list", "--help"])
        
        assert result.exit_code == 0
        assert "List stored configurations" in result.stdout

    def test_search_help(self, runner):
        """Test search command help."""
        result = runner.invoke(app, ["search", "--help"])
        
        assert result.exit_code == 0
        assert "Search configurations by query and filters" in result.stdout

    def test_show_help(self, runner):
        """Test show command help."""
        result = runner.invoke(app, ["show", "--help"])
        
        assert result.exit_code == 0
        assert "Show detailed configuration information" in result.stdout

    def test_stats_help(self, runner):
        """Test stats command help."""
        result = runner.invoke(app, ["stats", "--help"])
        
        assert result.exit_code == 0
        assert "Show repository statistics" in result.stdout


class TestConfigCommandsErrorHandling:
    """Test error handling for config commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    @patch('pynomaly.infrastructure.config.feature_flags.require_feature')
    def test_capture_configuration_exception(self, mock_require_feature, runner):
        """Test capture command with exception."""
        mock_require_feature.return_value = lambda f: f
        
        with patch('pynomaly.presentation.cli.config.ConfigurationCaptureService', side_effect=Exception("Service error")):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({"test": "data"}, f)
                params_file = f.name
            
            result = runner.invoke(app, [
                "capture", "automl",
                "--params", params_file
            ])
            
            assert result.exit_code == 1
            assert "Error capturing configuration: Service error" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationCaptureService')
    def test_export_configuration_exception(self, mock_service_class, runner):
        """Test export command with exception."""
        mock_service_class.side_effect = Exception("Export error")
        
        config_id = str(uuid4())
        result = runner.invoke(app, [
            "export", config_id,
            "--output", "/tmp/export.json"
        ])
        
        assert result.exit_code == 1
        assert "Error exporting configurations: Export error" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_import_configuration_exception(self, mock_repo_class, runner):
        """Test import command with exception."""
        mock_repo_class.side_effect = Exception("Import error")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"configurations": []}, f)
            import_file = f.name
        
        result = runner.invoke(app, [
            "import", import_file
        ])
        
        assert result.exit_code == 1
        assert "Error importing configurations: Import error" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_list_configurations_exception(self, mock_repo_class, runner):
        """Test list command with exception."""
        mock_repo_class.side_effect = Exception("List error")
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 1
        assert "Error listing configurations: List error" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_search_configurations_exception(self, mock_repo_class, runner):
        """Test search command with exception."""
        mock_repo_class.side_effect = Exception("Search error")
        
        result = runner.invoke(app, ["search", "test"])
        
        assert result.exit_code == 1
        assert "Error searching configurations: Search error" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_show_configuration_exception(self, mock_repo_class, runner):
        """Test show command with exception."""
        mock_repo_class.side_effect = Exception("Show error")
        
        config_id = str(uuid4())
        result = runner.invoke(app, ["show", config_id])
        
        assert result.exit_code == 1
        assert "Error showing configuration: Show error" in result.stdout

    @patch('pynomaly.presentation.cli.config.ConfigurationRepository')
    def test_stats_exception(self, mock_repo_class, runner):
        """Test stats command with exception."""
        mock_repo_class.side_effect = Exception("Stats error")
        
        result = runner.invoke(app, ["stats"])
        
        assert result.exit_code == 1
        assert "Error getting statistics: Stats error" in result.stdout