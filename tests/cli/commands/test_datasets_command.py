"""
Comprehensive tests for datasets CLI commands.
Tests for dataset management, CRUD operations, and validation.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from pynomaly.domain.exceptions import DatasetError, ValidationError
from pynomaly.presentation.cli.datasets import app as datasets_app


class TestDatasetsCommand:
    """Test suite for datasets CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container."""
        with patch(
            "pynomaly.presentation.cli.datasets.get_cli_container"
        ) as mock:
            container = Mock()
            
            # Mock repository
            mock_repo = Mock()
            container.dataset_repository.return_value = mock_repo
            
            # Mock config
            mock_config = Mock()
            mock_config.storage_path = Path("/tmp/pynomaly")
            mock_config.max_dataset_size_mb = 100
            container.config.return_value = mock_config
            
            mock.return_value = container
            yield container, mock_repo

    @pytest.fixture
    def sample_dataset_data(self):
        """Sample dataset data."""
        return {
            "id": "test-dataset",
            "name": "Test Dataset",
            "description": "A test dataset for unit testing",
            "format": "csv",
            "size": 1024,
            "columns": ["feature1", "feature2", "target"],
            "rows": 100,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

    @pytest.fixture
    def sample_csv_file(self):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("feature1,feature2,target\n")
            f.write("1.0,2.0,0\n")
            f.write("2.0,3.0,1\n")
            f.write("3.0,4.0,0\n")
            yield f.name
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    # Dataset List Command Tests

    def test_datasets_list_basic(
        self, runner, mock_container, sample_dataset_data
    ):
        """Test basic dataset list command."""
        container, mock_repo = mock_container
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.id = "test-dataset"
        mock_dataset.name = "Test Dataset"
        mock_dataset.n_samples = 100
        mock_dataset.n_features = 3
        mock_dataset.has_target = True
        mock_dataset.memory_usage = 1024 * 1024  # 1MB
        mock_dataset.created_at = Mock()
        mock_dataset.created_at.strftime.return_value = "2024-01-01 00:00"
        
        mock_repo.find_all.return_value = [mock_dataset]

        result = runner.invoke(datasets_app, ["list"])

        assert result.exit_code == 0
        assert "Test Dataset" in result.stdout
        assert "test-dataset" in result.stdout
        mock_repo.find_all.assert_called_once()

    def test_datasets_list_empty(self, runner, mock_dataset_service):
        """Test dataset list with no datasets."""
        mock_dataset_service.list_datasets.return_value = []

        result = runner.invoke(datasets_app, ["list"])

        assert result.exit_code == 0
        assert "No datasets found" in result.stdout
        mock_dataset_service.list_datasets.assert_called_once()

    def test_datasets_list_with_format_json(
        self, runner, mock_dataset_service, sample_dataset_data
    ):
        """Test dataset list with JSON format."""
        mock_dataset = Mock()
        mock_dataset.to_dict.return_value = sample_dataset_data
        mock_dataset_service.list_datasets.return_value = [mock_dataset]

        result = runner.invoke(datasets_app, ["list", "--format", "json"])

        assert result.exit_code == 0
        assert "test-dataset" in result.stdout
        # Should contain JSON-formatted output
        assert "{" in result.stdout
        mock_dataset_service.list_datasets.assert_called_once()

    def test_datasets_list_with_limit(
        self, runner, mock_dataset_service, sample_dataset_data
    ):
        """Test dataset list with limit parameter."""
        mock_datasets = []
        for i in range(5):
            mock_dataset = Mock()
            dataset_data = sample_dataset_data.copy()
            dataset_data["id"] = f"dataset-{i}"
            dataset_data["name"] = f"Dataset {i}"
            mock_dataset.to_dict.return_value = dataset_data
            mock_datasets.append(mock_dataset)

        mock_dataset_service.list_datasets.return_value = mock_datasets

        result = runner.invoke(datasets_app, ["list", "--limit", "3"])

        assert result.exit_code == 0
        mock_dataset_service.list_datasets.assert_called_once()

    # Dataset Show Command Tests

    def test_datasets_show_basic(
        self, runner, mock_dataset_service, sample_dataset_data
    ):
        """Test basic dataset show command."""
        mock_dataset = Mock()
        mock_dataset.to_dict.return_value = sample_dataset_data
        mock_dataset_service.get_dataset.return_value = mock_dataset

        result = runner.invoke(datasets_app, ["show", "test-dataset"])

        assert result.exit_code == 0
        assert "Test Dataset" in result.stdout
        assert "test-dataset" in result.stdout
        mock_dataset_service.get_dataset.assert_called_once_with("test-dataset")

    def test_datasets_show_not_found(self, runner, mock_dataset_service):
        """Test dataset show with non-existent dataset."""
        mock_dataset_service.get_dataset.side_effect = DatasetError("Dataset not found")

        result = runner.invoke(datasets_app, ["show", "non-existent"])

        assert result.exit_code == 1
        assert "Dataset not found" in result.stdout
        mock_dataset_service.get_dataset.assert_called_once_with("non-existent")

    def test_datasets_show_with_format_json(
        self, runner, mock_dataset_service, sample_dataset_data
    ):
        """Test dataset show with JSON format."""
        mock_dataset = Mock()
        mock_dataset.to_dict.return_value = sample_dataset_data
        mock_dataset_service.get_dataset.return_value = mock_dataset

        result = runner.invoke(
            datasets_app, ["show", "test-dataset", "--format", "json"]
        )

        assert result.exit_code == 0
        assert "{" in result.stdout
        mock_dataset_service.get_dataset.assert_called_once_with("test-dataset")

    def test_datasets_show_with_stats(
        self, runner, mock_dataset_service, sample_dataset_data
    ):
        """Test dataset show with statistics."""
        mock_dataset = Mock()
        mock_dataset.to_dict.return_value = sample_dataset_data
        mock_dataset_service.get_dataset.return_value = mock_dataset

        # Mock statistics
        mock_stats = {
            "mean": [1.5, 2.5],
            "std": [0.5, 0.5],
            "min": [1.0, 2.0],
            "max": [3.0, 4.0],
        }
        mock_dataset_service.get_dataset_statistics.return_value = mock_stats

        result = runner.invoke(datasets_app, ["show", "test-dataset", "--stats"])

        assert result.exit_code == 0
        assert "Statistics" in result.stdout
        mock_dataset_service.get_dataset.assert_called_once_with("test-dataset")
        mock_dataset_service.get_dataset_statistics.assert_called_once_with(
            "test-dataset"
        )

    # Dataset Create Command Tests

    def test_datasets_create_basic(self, runner, mock_dataset_service, sample_csv_file):
        """Test basic dataset create command."""
        mock_dataset = Mock()
        mock_dataset.id = "new-dataset"
        mock_dataset_service.create_dataset.return_value = mock_dataset

        result = runner.invoke(
            datasets_app,
            [
                "create",
                "new-dataset",
                "--file",
                sample_csv_file,
                "--description",
                "Test dataset",
            ],
        )

        assert result.exit_code == 0
        assert "Dataset created successfully" in result.stdout
        mock_dataset_service.create_dataset.assert_called_once()

    def test_datasets_create_with_invalid_name(
        self, runner, mock_dataset_service, sample_csv_file
    ):
        """Test dataset create with invalid name."""
        result = runner.invoke(
            datasets_app, ["create", "invalid/name", "--file", sample_csv_file]
        )

        assert result.exit_code == 1
        assert "invalid character" in result.stdout

    def test_datasets_create_with_missing_file(self, runner, mock_dataset_service):
        """Test dataset create with missing file."""
        result = runner.invoke(
            datasets_app, ["create", "test-dataset", "--file", "/non/existent/file.csv"]
        )

        assert result.exit_code == 1
        assert "File not found" in result.stdout

    def test_datasets_create_with_unsupported_format(
        self, runner, mock_dataset_service
    ):
        """Test dataset create with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"some text data")
            temp_file = f.name

        try:
            result = runner.invoke(
                datasets_app, ["create", "test-dataset", "--file", temp_file]
            )

            assert result.exit_code == 1
            assert "Unsupported file format" in result.stdout
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_datasets_create_with_options(
        self, runner, mock_dataset_service, sample_csv_file
    ):
        """Test dataset create with various options."""
        mock_dataset = Mock()
        mock_dataset.id = "advanced-dataset"
        mock_dataset_service.create_dataset.return_value = mock_dataset

        result = runner.invoke(
            datasets_app,
            [
                "create",
                "advanced-dataset",
                "--file",
                sample_csv_file,
                "--description",
                "Advanced test dataset",
                "--format",
                "csv",
                "--separator",
                ";",
                "--has-header",
            ],
        )

        assert result.exit_code == 0
        assert "Dataset created successfully" in result.stdout
        mock_dataset_service.create_dataset.assert_called_once()

    # Dataset Update Command Tests

    def test_datasets_update_basic(self, runner, mock_dataset_service):
        """Test basic dataset update command."""
        mock_dataset = Mock()
        mock_dataset.id = "test-dataset"
        mock_dataset_service.update_dataset.return_value = mock_dataset

        result = runner.invoke(
            datasets_app,
            ["update", "test-dataset", "--description", "Updated description"],
        )

        assert result.exit_code == 0
        assert "Dataset updated successfully" in result.stdout
        mock_dataset_service.update_dataset.assert_called_once()

    def test_datasets_update_not_found(self, runner, mock_dataset_service):
        """Test dataset update with non-existent dataset."""
        mock_dataset_service.update_dataset.side_effect = DatasetError(
            "Dataset not found"
        )

        result = runner.invoke(
            datasets_app, ["update", "non-existent", "--description", "New description"]
        )

        assert result.exit_code == 1
        assert "Dataset not found" in result.stdout

    def test_datasets_update_with_new_file(
        self, runner, mock_dataset_service, sample_csv_file
    ):
        """Test dataset update with new file."""
        mock_dataset = Mock()
        mock_dataset.id = "test-dataset"
        mock_dataset_service.update_dataset.return_value = mock_dataset

        result = runner.invoke(
            datasets_app,
            [
                "update",
                "test-dataset",
                "--file",
                sample_csv_file,
                "--description",
                "Updated with new file",
            ],
        )

        assert result.exit_code == 0
        assert "Dataset updated successfully" in result.stdout
        mock_dataset_service.update_dataset.assert_called_once()

    # Dataset Delete Command Tests

    def test_datasets_delete_basic(self, runner, mock_dataset_service):
        """Test basic dataset delete command."""
        mock_dataset_service.delete_dataset.return_value = None

        result = runner.invoke(datasets_app, ["delete", "test-dataset"], input="y\n")

        assert result.exit_code == 0
        assert "Dataset deleted successfully" in result.stdout
        mock_dataset_service.delete_dataset.assert_called_once_with("test-dataset")

    def test_datasets_delete_not_found(self, runner, mock_dataset_service):
        """Test dataset delete with non-existent dataset."""
        mock_dataset_service.delete_dataset.side_effect = DatasetError(
            "Dataset not found"
        )

        result = runner.invoke(datasets_app, ["delete", "non-existent"], input="y\n")

        assert result.exit_code == 1
        assert "Dataset not found" in result.stdout

    def test_datasets_delete_with_force(self, runner, mock_dataset_service):
        """Test dataset delete with force flag."""
        mock_dataset_service.delete_dataset.return_value = None

        result = runner.invoke(datasets_app, ["delete", "test-dataset", "--force"])

        assert result.exit_code == 0
        assert "Dataset deleted successfully" in result.stdout
        mock_dataset_service.delete_dataset.assert_called_once_with("test-dataset")

    def test_datasets_delete_cancelled(self, runner, mock_dataset_service):
        """Test dataset delete cancelled by user."""
        result = runner.invoke(datasets_app, ["delete", "test-dataset"], input="n\n")

        assert result.exit_code == 0
        assert "Deletion cancelled" in result.stdout
        mock_dataset_service.delete_dataset.assert_not_called()

    # Dataset Import Command Tests

    def test_datasets_import_basic(self, runner, mock_dataset_service, sample_csv_file):
        """Test basic dataset import command."""
        mock_dataset = Mock()
        mock_dataset.id = "imported-dataset"
        mock_dataset_service.import_dataset.return_value = mock_dataset

        result = runner.invoke(
            datasets_app, ["import", sample_csv_file, "--name", "imported-dataset"]
        )

        assert result.exit_code == 0
        assert "Dataset imported successfully" in result.stdout
        mock_dataset_service.import_dataset.assert_called_once()

    def test_datasets_import_with_auto_name(
        self, runner, mock_dataset_service, sample_csv_file
    ):
        """Test dataset import with auto-generated name."""
        mock_dataset = Mock()
        mock_dataset.id = "auto-generated-name"
        mock_dataset_service.import_dataset.return_value = mock_dataset

        result = runner.invoke(datasets_app, ["import", sample_csv_file])

        assert result.exit_code == 0
        assert "Dataset imported successfully" in result.stdout
        mock_dataset_service.import_dataset.assert_called_once()

    def test_datasets_import_with_preprocessing(
        self, runner, mock_dataset_service, sample_csv_file
    ):
        """Test dataset import with preprocessing options."""
        mock_dataset = Mock()
        mock_dataset.id = "preprocessed-dataset"
        mock_dataset_service.import_dataset.return_value = mock_dataset

        result = runner.invoke(
            datasets_app,
            [
                "import",
                sample_csv_file,
                "--name",
                "preprocessed-dataset",
                "--normalize",
                "--remove-duplicates",
                "--handle-missing",
                "drop",
            ],
        )

        assert result.exit_code == 0
        assert "Dataset imported successfully" in result.stdout
        mock_dataset_service.import_dataset.assert_called_once()

    # Dataset Export Command Tests

    def test_datasets_export_basic(self, runner, mock_dataset_service):
        """Test basic dataset export command."""
        mock_dataset_service.export_dataset.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "exported.csv"

            result = runner.invoke(
                datasets_app, ["export", "test-dataset", "--output", str(output_file)]
            )

            assert result.exit_code == 0
            assert "Dataset exported successfully" in result.stdout
            mock_dataset_service.export_dataset.assert_called_once()

    def test_datasets_export_with_format(self, runner, mock_dataset_service):
        """Test dataset export with specific format."""
        mock_dataset_service.export_dataset.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "exported.json"

            result = runner.invoke(
                datasets_app,
                [
                    "export",
                    "test-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "json",
                ],
            )

            assert result.exit_code == 0
            assert "Dataset exported successfully" in result.stdout
            mock_dataset_service.export_dataset.assert_called_once()

    def test_datasets_export_not_found(self, runner, mock_dataset_service):
        """Test dataset export with non-existent dataset."""
        mock_dataset_service.export_dataset.side_effect = DatasetError(
            "Dataset not found"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "exported.csv"

            result = runner.invoke(
                datasets_app, ["export", "non-existent", "--output", str(output_file)]
            )

            assert result.exit_code == 1
            assert "Dataset not found" in result.stdout

    # Dataset Validation Tests

    def test_datasets_validate_basic(self, runner, mock_dataset_service):
        """Test basic dataset validation command."""
        mock_validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": "Dataset is valid",
        }
        mock_dataset_service.validate_dataset.return_value = mock_validation_result

        result = runner.invoke(datasets_app, ["validate", "test-dataset"])

        assert result.exit_code == 0
        assert "Dataset is valid" in result.stdout
        mock_dataset_service.validate_dataset.assert_called_once_with("test-dataset")

    def test_datasets_validate_with_errors(self, runner, mock_dataset_service):
        """Test dataset validation with errors."""
        mock_validation_result = {
            "valid": False,
            "errors": [
                "Missing values in column 'feature1'",
                "Invalid data type in column 'target'",
            ],
            "warnings": ["Column 'feature2' has high cardinality"],
            "summary": "Dataset has validation errors",
        }
        mock_dataset_service.validate_dataset.return_value = mock_validation_result

        result = runner.invoke(datasets_app, ["validate", "test-dataset"])

        assert result.exit_code == 1
        assert "Dataset has validation errors" in result.stdout
        assert "Missing values in column" in result.stdout
        assert "Invalid data type" in result.stdout
        mock_dataset_service.validate_dataset.assert_called_once_with("test-dataset")

    def test_datasets_validate_not_found(self, runner, mock_dataset_service):
        """Test dataset validation with non-existent dataset."""
        mock_dataset_service.validate_dataset.side_effect = DatasetError(
            "Dataset not found"
        )

        result = runner.invoke(datasets_app, ["validate", "non-existent"])

        assert result.exit_code == 1
        assert "Dataset not found" in result.stdout

    # Dataset Statistics Tests

    def test_datasets_stats_basic(self, runner, mock_dataset_service):
        """Test basic dataset statistics command."""
        mock_stats = {
            "rows": 1000,
            "columns": 5,
            "missing_values": 10,
            "duplicates": 5,
            "data_types": {"numeric": 3, "categorical": 2},
            "memory_usage": "1.2 MB",
        }
        mock_dataset_service.get_dataset_statistics.return_value = mock_stats

        result = runner.invoke(datasets_app, ["stats", "test-dataset"])

        assert result.exit_code == 0
        assert "1000" in result.stdout  # rows
        assert "5" in result.stdout  # columns
        mock_dataset_service.get_dataset_statistics.assert_called_once_with(
            "test-dataset"
        )

    def test_datasets_stats_with_detailed(self, runner, mock_dataset_service):
        """Test dataset statistics with detailed option."""
        mock_stats = {
            "rows": 1000,
            "columns": 5,
            "column_stats": {
                "feature1": {"mean": 5.0, "std": 1.5, "min": 1.0, "max": 10.0},
                "feature2": {"mean": 3.0, "std": 0.8, "min": 0.5, "max": 6.0},
            },
        }
        mock_dataset_service.get_dataset_statistics.return_value = mock_stats

        result = runner.invoke(datasets_app, ["stats", "test-dataset", "--detailed"])

        assert result.exit_code == 0
        assert "feature1" in result.stdout
        assert "feature2" in result.stdout
        mock_dataset_service.get_dataset_statistics.assert_called_once_with(
            "test-dataset"
        )

    def test_datasets_stats_not_found(self, runner, mock_dataset_service):
        """Test dataset statistics with non-existent dataset."""
        mock_dataset_service.get_dataset_statistics.side_effect = DatasetError(
            "Dataset not found"
        )

        result = runner.invoke(datasets_app, ["stats", "non-existent"])

        assert result.exit_code == 1
        assert "Dataset not found" in result.stdout

    # Error Handling Tests

    def test_datasets_service_error_handling(self, runner, mock_dataset_service):
        """Test error handling for dataset service errors."""
        mock_dataset_service.list_datasets.side_effect = ValidationError(
            "Service error"
        )

        result = runner.invoke(datasets_app, ["list"])

        assert result.exit_code == 1
        assert "Service error" in result.stdout

    def test_datasets_validation_error_handling(self, runner, mock_dataset_service):
        """Test error handling for validation errors."""
        result = runner.invoke(datasets_app, ["create", ""])

        assert result.exit_code == 1
        assert "Dataset name cannot be empty" in result.stdout

    def test_datasets_keyboard_interrupt(self, runner, mock_dataset_service):
        """Test keyboard interrupt handling."""
        mock_dataset_service.list_datasets.side_effect = KeyboardInterrupt()

        result = runner.invoke(datasets_app, ["list"])

        assert result.exit_code == 1
        assert "Operation cancelled by user" in result.stdout

    # Integration Tests

    def test_datasets_workflow_integration(
        self, runner, mock_dataset_service, sample_csv_file
    ):
        """Test complete dataset workflow integration."""
        # Create dataset
        mock_dataset = Mock()
        mock_dataset.id = "workflow-dataset"
        mock_dataset_service.create_dataset.return_value = mock_dataset

        create_result = runner.invoke(
            datasets_app,
            [
                "create",
                "workflow-dataset",
                "--file",
                sample_csv_file,
                "--description",
                "Workflow test dataset",
            ],
        )
        assert create_result.exit_code == 0

        # List datasets
        mock_dataset.to_dict.return_value = {
            "id": "workflow-dataset",
            "name": "Workflow Test Dataset",
            "description": "Workflow test dataset",
        }
        mock_dataset_service.list_datasets.return_value = [mock_dataset]

        list_result = runner.invoke(datasets_app, ["list"])
        assert list_result.exit_code == 0
        assert "workflow-dataset" in list_result.stdout

        # Show dataset
        mock_dataset_service.get_dataset.return_value = mock_dataset

        show_result = runner.invoke(datasets_app, ["show", "workflow-dataset"])
        assert show_result.exit_code == 0
        assert "workflow-dataset" in show_result.stdout

        # Delete dataset
        mock_dataset_service.delete_dataset.return_value = None

        delete_result = runner.invoke(
            datasets_app, ["delete", "workflow-dataset", "--force"]
        )
        assert delete_result.exit_code == 0
        assert "Dataset deleted successfully" in delete_result.stdout

    def test_datasets_help_commands(self, runner):
        """Test help commands for datasets."""
        result = runner.invoke(datasets_app, ["--help"])
        assert result.exit_code == 0
        assert "Commands:" in result.stdout
        assert "list" in result.stdout
        assert "create" in result.stdout
        assert "delete" in result.stdout

        # Test subcommand help
        result = runner.invoke(datasets_app, ["create", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--file" in result.stdout
        assert "--description" in result.stdout
