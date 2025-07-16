"""
Preprocessing CLI Testing Suite
Comprehensive tests for data preprocessing CLI commands.
"""

import csv
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from monorepo.presentation.cli.preprocessing import app


class TestPreprocessingCLI:
    """Test suite for preprocessing CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container for preprocessing operations."""
        with patch("monorepo.presentation.cli.container.get_cli_container") as mock:
            container = Mock()

            # Mock dataset repository
            dataset_repo = Mock()
            container.dataset_repository.return_value = dataset_repo

            # Mock preprocessing service
            preprocessing_service = Mock()
            container.preprocessing_service.return_value = preprocessing_service

            # Mock sample dataset
            mock_dataset = Mock()
            mock_dataset.id = "test-dataset-123"
            mock_dataset.name = "Test Dataset"
            mock_dataset.shape = (1000, 5)
            mock_dataset.features = [
                "feature1",
                "feature2",
                "feature3",
                "feature4",
                "target",
            ]
            mock_dataset.data = pd.DataFrame(
                {
                    "feature1": [1, 2, None, 4, 100],  # Has missing value and outlier
                    "feature2": [10, 20, 30, 40, 50],
                    "feature3": ["A", "B", "A", "C", "A"],  # Categorical
                    "feature4": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "target": [0, 0, 1, 0, 1],
                }
            )

            dataset_repo.find_by_name.return_value = mock_dataset
            dataset_repo.save.return_value = True

            mock.return_value = container
            yield container

    @pytest.fixture
    def sample_dirty_data_file(self):
        """Create sample dirty data file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["feature1", "feature2", "feature3", "category", "target"])

            # Normal data
            writer.writerow([1.0, 10.0, 0.1, "A", 0])
            writer.writerow([2.0, 20.0, 0.2, "B", 0])
            writer.writerow([3.0, 30.0, 0.3, "A", 1])

            # Dirty data
            writer.writerow([None, 40.0, 0.4, "C", 0])  # Missing value
            writer.writerow([100.0, 500.0, 5.0, "A", 1])  # Outliers
            writer.writerow([2.0, 20.0, 0.2, "B", 0])  # Duplicate
            writer.writerow(["invalid", "invalid", "invalid", "D", 1])  # Invalid values

            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    # Basic Command Tests

    def test_preprocessing_help(self, runner):
        """Test preprocessing CLI help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Data preprocessing" in result.stdout
        assert "Commands:" in result.stdout
        assert "clean" in result.stdout
        assert "transform" in result.stdout
        assert "pipeline" in result.stdout

    def test_clean_help(self, runner):
        """Test clean command help."""
        result = runner.invoke(app, ["clean", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--missing" in result.stdout
        assert "--outliers" in result.stdout
        assert "--duplicates" in result.stdout

    def test_transform_help(self, runner):
        """Test transform command help."""
        result = runner.invoke(app, ["transform", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--scaling" in result.stdout
        assert "--encoding" in result.stdout

    # Data Cleaning Tests

    def test_clean_missing_values_drop_rows(self, runner, mock_container):
        """Test cleaning missing values by dropping rows."""
        result = runner.invoke(app, ["clean", "Test Dataset", "--missing", "drop_rows"])

        assert result.exit_code == 0
        assert "Cleaning dataset" in result.stdout
        assert "Missing values: drop_rows" in result.stdout

        # Verify preprocessing service was called
        preprocessing_service = mock_container.preprocessing_service.return_value
        preprocessing_service.clean_missing_values.assert_called_once()

    def test_clean_missing_values_fill_mean(self, runner, mock_container):
        """Test cleaning missing values by filling with mean."""
        result = runner.invoke(app, ["clean", "Test Dataset", "--missing", "fill_mean"])

        assert result.exit_code == 0
        assert "Missing values: fill_mean" in result.stdout

    def test_clean_missing_values_fill_median(self, runner, mock_container):
        """Test cleaning missing values by filling with median."""
        result = runner.invoke(
            app, ["clean", "Test Dataset", "--missing", "fill_median"]
        )

        assert result.exit_code == 0
        assert "Missing values: fill_median" in result.stdout

    def test_clean_missing_values_interpolate(self, runner, mock_container):
        """Test cleaning missing values by interpolation."""
        result = runner.invoke(
            app, ["clean", "Test Dataset", "--missing", "interpolate"]
        )

        assert result.exit_code == 0
        assert "Missing values: interpolate" in result.stdout

    def test_clean_outliers_clip(self, runner, mock_container):
        """Test cleaning outliers by clipping."""
        result = runner.invoke(app, ["clean", "Test Dataset", "--outliers", "clip"])

        assert result.exit_code == 0
        assert "Outliers: clip" in result.stdout

    def test_clean_outliers_remove(self, runner, mock_container):
        """Test cleaning outliers by removal."""
        result = runner.invoke(app, ["clean", "Test Dataset", "--outliers", "remove"])

        assert result.exit_code == 0
        assert "Outliers: remove" in result.stdout

    def test_clean_outliers_zscore(self, runner, mock_container):
        """Test cleaning outliers using z-score method."""
        result = runner.invoke(
            app,
            [
                "clean",
                "Test Dataset",
                "--outliers",
                "zscore",
                "--outlier-threshold",
                "3.0",
            ],
        )

        assert result.exit_code == 0
        assert "Outliers: zscore" in result.stdout

    def test_clean_outliers_iqr(self, runner, mock_container):
        """Test cleaning outliers using IQR method."""
        result = runner.invoke(app, ["clean", "Test Dataset", "--outliers", "iqr"])

        assert result.exit_code == 0
        assert "Outliers: iqr" in result.stdout

    def test_clean_duplicates_remove(self, runner, mock_container):
        """Test removing duplicate rows."""
        result = runner.invoke(app, ["clean", "Test Dataset", "--duplicates", "remove"])

        assert result.exit_code == 0
        assert "Duplicates: remove" in result.stdout

    def test_clean_duplicates_keep_first(self, runner, mock_container):
        """Test keeping first occurrence of duplicates."""
        result = runner.invoke(
            app, ["clean", "Test Dataset", "--duplicates", "keep_first"]
        )

        assert result.exit_code == 0
        assert "Duplicates: keep_first" in result.stdout

    def test_clean_duplicates_keep_last(self, runner, mock_container):
        """Test keeping last occurrence of duplicates."""
        result = runner.invoke(
            app, ["clean", "Test Dataset", "--duplicates", "keep_last"]
        )

        assert result.exit_code == 0
        assert "Duplicates: keep_last" in result.stdout

    def test_clean_all_options(self, runner, mock_container):
        """Test cleaning with all options."""
        result = runner.invoke(
            app,
            [
                "clean",
                "Test Dataset",
                "--missing",
                "fill_mean",
                "--outliers",
                "clip",
                "--duplicates",
                "remove",
                "--outlier-threshold",
                "2.5",
            ],
        )

        assert result.exit_code == 0
        assert "Missing values: fill_mean" in result.stdout
        assert "Outliers: clip" in result.stdout
        assert "Duplicates: remove" in result.stdout

    def test_clean_dataset_not_found(self, runner, mock_container):
        """Test cleaning non-existent dataset."""
        mock_container.dataset_repository.return_value.find_by_name.return_value = None

        result = runner.invoke(
            app, ["clean", "NonExistent Dataset", "--missing", "drop_rows"]
        )

        assert result.exit_code == 1
        assert "not found" in result.stdout

    # Data Transformation Tests

    def test_transform_scaling_standard(self, runner, mock_container):
        """Test standard scaling transformation."""
        result = runner.invoke(
            app, ["transform", "Test Dataset", "--scaling", "standard"]
        )

        assert result.exit_code == 0
        assert "Scaling: standard" in result.stdout

    def test_transform_scaling_minmax(self, runner, mock_container):
        """Test min-max scaling transformation."""
        result = runner.invoke(
            app, ["transform", "Test Dataset", "--scaling", "minmax"]
        )

        assert result.exit_code == 0
        assert "Scaling: minmax" in result.stdout

    def test_transform_scaling_robust(self, runner, mock_container):
        """Test robust scaling transformation."""
        result = runner.invoke(
            app, ["transform", "Test Dataset", "--scaling", "robust"]
        )

        assert result.exit_code == 0
        assert "Scaling: robust" in result.stdout

    def test_transform_encoding_onehot(self, runner, mock_container):
        """Test one-hot encoding transformation."""
        result = runner.invoke(
            app, ["transform", "Test Dataset", "--encoding", "onehot"]
        )

        assert result.exit_code == 0
        assert "Encoding: onehot" in result.stdout

    def test_transform_encoding_label(self, runner, mock_container):
        """Test label encoding transformation."""
        result = runner.invoke(
            app, ["transform", "Test Dataset", "--encoding", "label"]
        )

        assert result.exit_code == 0
        assert "Encoding: label" in result.stdout

    def test_transform_encoding_target(self, runner, mock_container):
        """Test target encoding transformation."""
        result = runner.invoke(
            app, ["transform", "Test Dataset", "--encoding", "target"]
        )

        assert result.exit_code == 0
        assert "Encoding: target" in result.stdout

    def test_transform_feature_selection(self, runner, mock_container):
        """Test feature selection transformation."""
        result = runner.invoke(
            app,
            [
                "transform",
                "Test Dataset",
                "--feature-selection",
                "variance",
                "--n-features",
                "3",
            ],
        )

        assert result.exit_code == 0
        assert "Feature selection: variance" in result.stdout

    def test_transform_pca(self, runner, mock_container):
        """Test PCA transformation."""
        result = runner.invoke(app, ["transform", "Test Dataset", "--pca", "2"])

        assert result.exit_code == 0
        assert "PCA: 2 components" in result.stdout

    def test_transform_all_options(self, runner, mock_container):
        """Test transformation with multiple options."""
        result = runner.invoke(
            app,
            [
                "transform",
                "Test Dataset",
                "--scaling",
                "standard",
                "--encoding",
                "onehot",
                "--feature-selection",
                "variance",
                "--n-features",
                "5",
            ],
        )

        assert result.exit_code == 0
        assert "Scaling: standard" in result.stdout
        assert "Encoding: onehot" in result.stdout
        assert "Feature selection: variance" in result.stdout

    # Pipeline Tests

    def test_pipeline_create(self, runner, mock_container):
        """Test creating preprocessing pipeline."""
        result = runner.invoke(
            app,
            [
                "pipeline",
                "create",
                "--name",
                "test_pipeline",
                "--steps",
                "clean,transform,validate",
            ],
        )

        assert result.exit_code == 0
        assert "Pipeline created: test_pipeline" in result.stdout

    def test_pipeline_list(self, runner, mock_container):
        """Test listing preprocessing pipelines."""
        # Mock pipeline repository
        pipeline_repo = Mock()
        mock_container.pipeline_repository.return_value = pipeline_repo

        mock_pipeline = Mock()
        mock_pipeline.name = "test_pipeline"
        mock_pipeline.steps = ["clean", "transform"]
        mock_pipeline.created_at = datetime.utcnow()

        pipeline_repo.find_all.return_value = [mock_pipeline]

        result = runner.invoke(app, ["pipeline", "list"])

        assert result.exit_code == 0
        assert "Available Pipelines:" in result.stdout
        assert "test_pipeline" in result.stdout

    def test_pipeline_run(self, runner, mock_container):
        """Test running preprocessing pipeline."""
        # Mock pipeline repository
        pipeline_repo = Mock()
        mock_container.pipeline_repository.return_value = pipeline_repo

        mock_pipeline = Mock()
        mock_pipeline.name = "test_pipeline"
        pipeline_repo.find_by_name.return_value = mock_pipeline

        result = runner.invoke(
            app, ["pipeline", "run", "test_pipeline", "--dataset", "Test Dataset"]
        )

        assert result.exit_code == 0
        assert "Running pipeline: test_pipeline" in result.stdout

    def test_pipeline_delete(self, runner, mock_container):
        """Test deleting preprocessing pipeline."""
        # Mock pipeline repository
        pipeline_repo = Mock()
        mock_container.pipeline_repository.return_value = pipeline_repo

        result = runner.invoke(
            app, ["pipeline", "delete", "test_pipeline"], input="y\n"
        )

        assert result.exit_code == 0

    # Validation Tests

    def test_validate_dataset(self, runner, mock_container):
        """Test dataset validation."""
        # Mock validation service
        validation_service = Mock()
        mock_container.validation_service.return_value = validation_service

        validation_service.validate_dataset.return_value = {
            "is_valid": True,
            "warnings": ["Column 'feature1' has missing values"],
            "errors": [],
            "quality_score": 0.85,
            "recommendations": ["Consider imputing missing values"],
        }

        result = runner.invoke(app, ["validate", "Test Dataset"])

        assert result.exit_code == 0
        assert "Dataset Validation" in result.stdout
        assert "Quality Score: 0.85" in result.stdout
        assert "Warnings:" in result.stdout

    def test_validate_dataset_with_errors(self, runner, mock_container):
        """Test dataset validation with errors."""
        # Mock validation service
        validation_service = Mock()
        mock_container.validation_service.return_value = validation_service

        validation_service.validate_dataset.return_value = {
            "is_valid": False,
            "warnings": [],
            "errors": ["Invalid data types in column 'feature1'"],
            "quality_score": 0.45,
            "recommendations": ["Fix data type issues before processing"],
        }

        result = runner.invoke(app, ["validate", "Test Dataset"])

        assert result.exit_code == 0
        assert "❌ Dataset validation failed" in result.stdout
        assert "Errors:" in result.stdout

    # Feature Engineering Tests

    def test_feature_engineering_create_features(self, runner, mock_container):
        """Test creating new features."""
        result = runner.invoke(
            app,
            [
                "features",
                "create",
                "--dataset",
                "Test Dataset",
                "--polynomial",
                "2",
                "--interactions",
            ],
        )

        assert result.exit_code == 0
        assert "Feature Engineering" in result.stdout
        assert "Polynomial features: degree 2" in result.stdout
        assert "Interaction features: enabled" in result.stdout

    def test_feature_engineering_datetime_features(self, runner, mock_container):
        """Test creating datetime features."""
        result = runner.invoke(
            app,
            [
                "features",
                "datetime",
                "--dataset",
                "Test Dataset",
                "--column",
                "timestamp",
                "--extract",
                "year,month,day,hour",
            ],
        )

        assert result.exit_code == 0
        assert "DateTime Features" in result.stdout

    def test_feature_engineering_text_features(self, runner, mock_container):
        """Test creating text features."""
        result = runner.invoke(
            app,
            [
                "features",
                "text",
                "--dataset",
                "Test Dataset",
                "--column",
                "description",
                "--method",
                "tfidf",
                "--max-features",
                "100",
            ],
        )

        assert result.exit_code == 0
        assert "Text Features" in result.stdout

    # Analysis Tests

    def test_analyze_missing_values(self, runner, mock_container):
        """Test missing values analysis."""
        result = runner.invoke(app, ["analyze", "Test Dataset", "--missing"])

        assert result.exit_code == 0
        assert "Missing Values Analysis" in result.stdout

    def test_analyze_outliers(self, runner, mock_container):
        """Test outliers analysis."""
        result = runner.invoke(app, ["analyze", "Test Dataset", "--outliers"])

        assert result.exit_code == 0
        assert "Outliers Analysis" in result.stdout

    def test_analyze_correlations(self, runner, mock_container):
        """Test correlations analysis."""
        result = runner.invoke(app, ["analyze", "Test Dataset", "--correlations"])

        assert result.exit_code == 0
        assert "Correlation Analysis" in result.stdout

    def test_analyze_distribution(self, runner, mock_container):
        """Test distribution analysis."""
        result = runner.invoke(app, ["analyze", "Test Dataset", "--distribution"])

        assert result.exit_code == 0
        assert "Distribution Analysis" in result.stdout

    def test_analyze_all(self, runner, mock_container):
        """Test comprehensive analysis."""
        result = runner.invoke(
            app,
            [
                "analyze",
                "Test Dataset",
                "--missing",
                "--outliers",
                "--correlations",
                "--distribution",
            ],
        )

        assert result.exit_code == 0
        assert "Missing Values Analysis" in result.stdout
        assert "Outliers Analysis" in result.stdout
        assert "Correlation Analysis" in result.stdout
        assert "Distribution Analysis" in result.stdout

    # Error Handling Tests

    def test_clean_invalid_missing_strategy(self, runner, mock_container):
        """Test clean with invalid missing value strategy."""
        result = runner.invoke(
            app, ["clean", "Test Dataset", "--missing", "invalid_strategy"]
        )

        assert result.exit_code != 0

    def test_clean_invalid_outlier_method(self, runner, mock_container):
        """Test clean with invalid outlier method."""
        result = runner.invoke(
            app, ["clean", "Test Dataset", "--outliers", "invalid_method"]
        )

        assert result.exit_code != 0

    def test_transform_invalid_scaling_method(self, runner, mock_container):
        """Test transform with invalid scaling method."""
        result = runner.invoke(
            app, ["transform", "Test Dataset", "--scaling", "invalid_scaling"]
        )

        assert result.exit_code != 0

    def test_transform_invalid_encoding_method(self, runner, mock_container):
        """Test transform with invalid encoding method."""
        result = runner.invoke(
            app, ["transform", "Test Dataset", "--encoding", "invalid_encoding"]
        )

        assert result.exit_code != 0

    def test_preprocessing_service_error(self, runner, mock_container):
        """Test preprocessing service error handling."""
        preprocessing_service = mock_container.preprocessing_service.return_value
        preprocessing_service.clean_missing_values.side_effect = Exception(
            "Processing error"
        )

        result = runner.invoke(app, ["clean", "Test Dataset", "--missing", "drop_rows"])

        assert result.exit_code == 1
        assert "Error" in result.stdout

    # Integration Tests

    def test_complete_preprocessing_workflow(self, runner, mock_container):
        """Test complete preprocessing workflow."""
        # 1. Analyze dataset
        analyze_result = runner.invoke(
            app, ["analyze", "Test Dataset", "--missing", "--outliers"]
        )

        # 2. Clean dataset
        clean_result = runner.invoke(
            app,
            ["clean", "Test Dataset", "--missing", "fill_mean", "--outliers", "clip"],
        )

        # 3. Transform dataset
        transform_result = runner.invoke(
            app,
            [
                "transform",
                "Test Dataset",
                "--scaling",
                "standard",
                "--encoding",
                "onehot",
            ],
        )

        # 4. Validate processed dataset
        validate_result = runner.invoke(app, ["validate", "Test Dataset"])

        # All steps should succeed
        assert analyze_result.exit_code == 0
        assert clean_result.exit_code == 0
        assert transform_result.exit_code == 0
        assert validate_result.exit_code == 0

    def test_file_preprocessing_workflow(self, runner, sample_dirty_data_file):
        """Test preprocessing workflow with file input."""
        with patch("pandas.read_csv") as mock_read_csv:
            # Mock reading dirty data
            dirty_df = pd.DataFrame(
                {
                    "feature1": [1.0, 2.0, None, 100.0],  # Missing value and outlier
                    "feature2": [10.0, 20.0, 30.0, 500.0],  # Outlier
                    "category": ["A", "B", "A", "C"],
                    "target": [0, 0, 1, 1],
                }
            )
            mock_read_csv.return_value = dirty_df

            with patch(
                "monorepo.application.services.preprocessing_service.PreprocessingService"
            ) as mock_service:
                service_instance = Mock()
                mock_service.return_value = service_instance

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False
                ) as f:
                    output_path = f.name

                try:
                    result = runner.invoke(
                        app,
                        [
                            "file",
                            "clean",
                            sample_dirty_data_file,
                            "--output",
                            output_path,
                            "--missing",
                            "fill_mean",
                            "--outliers",
                            "clip",
                        ],
                    )

                    assert result.exit_code == 0

                finally:
                    Path(output_path).unlink(missing_ok=True)
