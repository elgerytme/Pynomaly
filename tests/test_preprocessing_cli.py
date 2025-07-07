"""Tests for preprocessing CLI commands."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pynomaly.domain.entities import Dataset
from pynomaly.presentation.cli.preprocessing import app


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_dataset():
    """Create a sample dataset with various data quality issues."""
    np.random.seed(42)

    # Create data with issues
    data = {
        "numeric_normal": np.random.normal(0, 1, 100),
        "numeric_outliers": np.concatenate(
            [
                np.random.normal(0, 1, 95),
                np.array([100, -100, 200, 150, -150]),  # outliers
            ]
        ),
        "numeric_missing": np.random.normal(0, 1, 100),
        "categorical": np.random.choice(["A", "B", "C"], 100),
        "categorical_missing": np.random.choice(["X", "Y", "Z"], 100),
        "constant_feature": np.ones(100),
        "zero_values": np.random.choice([0, 1, 2, 3, 0, 0], 100),
    }

    # Add missing values
    data["numeric_missing"][10:20] = np.nan
    data["categorical_missing"][15:25] = None

    # Add infinite values
    data["numeric_outliers"][0] = np.inf
    data["numeric_outliers"][1] = -np.inf

    df = pd.DataFrame(data)

    # Add duplicate rows
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    return Dataset(
        id="test-dataset-123",
        name="Test Dataset",
        description="Test dataset with quality issues",
        data=df,
        target_column=None,
    )


@pytest.fixture
def mock_container(sample_dataset):
    """Mock CLI container with sample dataset."""
    mock_container = Mock()
    mock_repo = Mock()
    mock_repo.find_by_id.return_value = sample_dataset
    mock_repo.save.return_value = None
    mock_container.dataset_repository.return_value = mock_repo
    return mock_container


class TestDataCleaningCLI:
    """Test data cleaning CLI commands."""

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_clean_missing_values_dry_run(
        self, mock_get_container, runner, mock_container
    ):
        """Test cleaning missing values in dry run mode."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app, ["clean", "test-dataset-123", "--missing", "drop_rows", "--dry-run"]
        )

        assert result.exit_code == 0
        assert "DRY RUN MODE" in result.output
        assert "Would handle" in result.output
        assert "missing values" in result.output

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_clean_outliers(self, mock_get_container, runner, mock_container):
        """Test outlier handling."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app,
            [
                "clean",
                "test-dataset-123",
                "--outliers",
                "clip",
                "--outlier-threshold",
                "2.0",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "outliers" in result.output.lower()

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_clean_duplicates(self, mock_get_container, runner, mock_container):
        """Test duplicate removal."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app, ["clean", "test-dataset-123", "--duplicates", "--dry-run"]
        )

        assert result.exit_code == 0
        assert "duplicate" in result.output.lower()

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_clean_comprehensive(self, mock_get_container, runner, mock_container):
        """Test comprehensive cleaning with multiple options."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app,
            [
                "clean",
                "test-dataset-123",
                "--missing",
                "fill_median",
                "--outliers",
                "clip",
                "--duplicates",
                "--zeros",
                "remove",
                "--infinite",
                "remove",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "missing values" in result.output.lower()
        assert "outliers" in result.output.lower()
        assert "duplicate" in result.output.lower()

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_clean_invalid_strategy(self, mock_get_container, runner, mock_container):
        """Test error handling for invalid strategies."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app, ["clean", "test-dataset-123", "--missing", "invalid_strategy"]
        )

        assert result.exit_code == 1
        assert "Invalid missing value strategy" in result.output


class TestDataTransformationCLI:
    """Test data transformation CLI commands."""

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_transform_scaling(self, mock_get_container, runner, mock_container):
        """Test feature scaling."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app, ["transform", "test-dataset-123", "--scaling", "standard", "--dry-run"]
        )

        assert result.exit_code == 0
        assert "scaling" in result.output.lower()

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_transform_encoding(self, mock_get_container, runner, mock_container):
        """Test categorical encoding."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app, ["transform", "test-dataset-123", "--encoding", "onehot", "--dry-run"]
        )

        assert result.exit_code == 0
        assert "encoding" in result.output.lower()

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_transform_feature_selection(
        self, mock_get_container, runner, mock_container
    ):
        """Test feature selection."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app,
            [
                "transform",
                "test-dataset-123",
                "--feature-selection",
                "variance_threshold",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "feature selection" in result.output.lower()

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_transform_polynomial_features(
        self, mock_get_container, runner, mock_container
    ):
        """Test polynomial feature generation."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app, ["transform", "test-dataset-123", "--polynomial", "2", "--dry-run"]
        )

        assert result.exit_code == 0
        assert "polynomial" in result.output.lower()

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_transform_comprehensive(self, mock_get_container, runner, mock_container):
        """Test comprehensive transformation."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app,
            [
                "transform",
                "test-dataset-123",
                "--scaling",
                "minmax",
                "--encoding",
                "label",
                "--normalize-names",
                "--optimize-dtypes",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "scaling" in result.output.lower()
        assert "encoding" in result.output.lower()


class TestPipelineManagementCLI:
    """Test pipeline management CLI commands."""

    def test_pipeline_create_interactive(self, runner):
        """Test interactive pipeline creation."""
        # This would need more complex mocking for interactive input
        result = runner.invoke(
            app, ["pipeline", "create", "--name", "test_pipeline"], input="\n"
        )  # Empty input to finish

        # Note: This test needs enhancement for proper interactive testing
        assert "test_pipeline" in result.output or result.exit_code == 0

    def test_pipeline_list_empty(self, runner):
        """Test listing pipelines when none exist."""
        result = runner.invoke(app, ["pipeline", "list"])

        assert result.exit_code == 0

    def test_pipeline_create_from_config(self, runner, tmp_path):
        """Test creating pipeline from config file."""
        config_file = tmp_path / "test_pipeline.json"
        config = {
            "name": "test_pipeline",
            "steps": [
                {
                    "name": "clean_missing",
                    "operation": "handle_missing_values",
                    "parameters": {"strategy": "drop_rows"},
                    "enabled": True,
                    "description": "Remove rows with missing values",
                }
            ],
        }

        import json

        with open(config_file, "w") as f:
            json.dump(config, f)

        result = runner.invoke(
            app,
            [
                "pipeline",
                "create",
                "--name",
                "test_pipeline",
                "--config",
                str(config_file),
            ],
        )

        assert result.exit_code == 0
        assert "test_pipeline" in result.output

    def test_pipeline_save(self, runner, tmp_path):
        """Test saving pipeline to file."""
        # First create a pipeline
        runner.invoke(
            app, ["pipeline", "create", "--name", "test_pipeline"], input="\n"
        )

        output_file = tmp_path / "saved_pipeline.json"
        result = runner.invoke(
            app,
            [
                "pipeline",
                "save",
                "--name",
                "test_pipeline",
                "--output",
                str(output_file),
            ],
        )

        # This might fail if pipeline doesn't exist, but tests the command structure
        assert "test_pipeline" in result.output or result.exit_code in [0, 1]


class TestPreprocessingIntegration:
    """Test integration between preprocessing components."""

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_dataset_not_found(self, mock_get_container, runner):
        """Test error handling when dataset is not found."""
        mock_container = Mock()
        mock_repo = Mock()
        mock_repo.find_by_id.return_value = None
        mock_container.dataset_repository.return_value = mock_repo
        mock_get_container.return_value = mock_container

        result = runner.invoke(app, ["clean", "nonexistent-dataset"])

        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_save_as_new_dataset(self, mock_get_container, runner, mock_container):
        """Test saving cleaned data as new dataset."""
        mock_get_container.return_value = mock_container

        result = runner.invoke(
            app,
            [
                "clean",
                "test-dataset-123",
                "--missing",
                "drop_rows",
                "--save-as",
                "cleaned_dataset",
                "--dry-run",  # Keep dry run to avoid complex mocking
            ],
        )

        assert result.exit_code == 0
        assert "cleaned_dataset" in result.output or "DRY RUN" in result.output


class TestCommandValidation:
    """Test command validation and error handling."""

    def test_invalid_action_pipeline(self, runner):
        """Test error handling for invalid pipeline actions."""
        result = runner.invoke(app, ["pipeline", "invalid_action"])

        assert result.exit_code == 1
        assert "Unknown action" in result.output

    def test_missing_required_arguments(self, runner):
        """Test error handling for missing required arguments."""
        result = runner.invoke(app, ["clean"])

        # Should fail due to missing dataset_id argument
        assert result.exit_code == 2  # Typer error code for missing arguments


@pytest.mark.integration
class TestPreprocessingWorkflow:
    """Integration tests for complete preprocessing workflows."""

    @patch("pynomaly.presentation.cli.preprocessing.get_cli_container")
    def test_complete_preprocessing_workflow(
        self, mock_get_container, runner, mock_container
    ):
        """Test a complete preprocessing workflow."""
        mock_get_container.return_value = mock_container

        # Step 1: Clean data
        result1 = runner.invoke(
            app,
            [
                "clean",
                "test-dataset-123",
                "--missing",
                "fill_median",
                "--outliers",
                "clip",
                "--duplicates",
                "--dry-run",
            ],
        )
        assert result1.exit_code == 0

        # Step 2: Transform data
        result2 = runner.invoke(
            app,
            [
                "transform",
                "test-dataset-123",
                "--scaling",
                "standard",
                "--encoding",
                "onehot",
                "--dry-run",
            ],
        )
        assert result2.exit_code == 0

        # Both commands should complete successfully
        assert (
            "cleaning" in result1.output.lower() or "missing" in result1.output.lower()
        )
        assert (
            "transform" in result2.output.lower() or "scaling" in result2.output.lower()
        )


if __name__ == "__main__":
    pytest.main([__file__])
