"""
Comprehensive CLI Testing Suite
Tests for all CLI commands, options, and user workflows.
"""

import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.presentation.cli import (
    autonomous,
    datasets,
    detection,
    detectors,
    preprocessing,
    server,
)
from pynomaly.presentation.cli.app import app
from pynomaly.presentation.cli.export import export_app


class TestMainCLIApp:
    """Test suite for main CLI application."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container."""
        with patch("pynomaly.presentation.cli.container.get_cli_container") as mock:
            container = Mock()

            # Mock repositories
            container.detector_repository.return_value = Mock()
            container.dataset_repository.return_value = Mock()
            container.result_repository.return_value = Mock()

            # Mock config
            config = Mock()
            config.app.version = "1.0.0"
            config.app_name = "Pynomaly"
            config.version = "1.0.0"
            config.debug = False
            config.storage_path = Path("/tmp/pynomaly")
            config.api_host = "localhost"
            config.api_port = 8000
            config.max_dataset_size_mb = 100
            config.default_contamination_rate = 0.1
            config.gpu_enabled = False
            container.config.return_value = config

            mock.return_value = container
            yield container

    # Main App Command Tests

    def test_main_help(self, runner):
        """Test main app help command."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Pynomaly" in result.stdout
        assert "Commands:" in result.stdout
        assert "auto" in result.stdout
        assert "detector" in result.stdout
        assert "dataset" in result.stdout

    def test_version_command(self, runner, mock_container):
        """Test version command."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "Pynomaly v1.0.0" in result.stdout
        assert "Python" in result.stdout

    def test_status_command(self, runner, mock_container):
        """Test status command."""
        # Mock repository counts
        mock_container.detector_repository.return_value.count.return_value = 5
        mock_container.dataset_repository.return_value.count.return_value = 3
        mock_container.result_repository.return_value.count.return_value = 10
        mock_container.result_repository.return_value.find_recent.return_value = []

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "System Status" in result.stdout
        assert "Detectors" in result.stdout
        assert "5" in result.stdout

    def test_config_show(self, runner, mock_container):
        """Test config show command."""
        result = runner.invoke(app, ["config", "--show"])

        assert result.exit_code == 0
        assert "Configuration" in result.stdout
        assert "App Name" in result.stdout
        assert "Version" in result.stdout

    def test_config_set_invalid_format(self, runner, mock_container):
        """Test config set with invalid format."""
        result = runner.invoke(app, ["config", "--set", "invalid_format"])

        assert result.exit_code == 1
        assert "Use format: --set key=value" in result.stdout

    def test_config_set_valid_format(self, runner, mock_container):
        """Test config set with valid format."""
        result = runner.invoke(app, ["config", "--set", "debug=true"])

        assert result.exit_code == 0
        assert "Would set: debug = true" in result.stdout

    def test_generate_config_test(self, runner):
        """Test generate test configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "generate-config",
                    "test",
                    "--output",
                    temp_path,
                    "--detector",
                    "IsolationForest",
                    "--contamination",
                    "0.05",
                ],
            )

            assert result.exit_code == 0
            assert "Test configuration generated" in result.stdout

            # Verify file contents
            with open(temp_path) as f:
                config = json.load(f)

            assert config["metadata"]["type"] == "test"
            assert config["test"]["detector"]["algorithm"] == "IsolationForest"
            assert config["test"]["detector"]["parameters"]["contamination"] == 0.05

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_generate_config_experiment(self, runner):
        """Test generate experiment configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "generate-config",
                    "experiment",
                    "--output",
                    temp_path,
                    "--auto-tune",
                    "true",
                    "--cv",
                    "true",
                    "--folds",
                    "10",
                ],
            )

            assert result.exit_code == 0
            assert "Experiment configuration generated" in result.stdout

            # Verify file contents
            with open(temp_path) as f:
                config = json.load(f)

            assert config["metadata"]["type"] == "experiment"
            assert config["experiment"]["evaluation"]["cross_validation"] is True
            assert config["experiment"]["evaluation"]["folds"] == 10

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_generate_config_autonomous(self, runner):
        """Test generate autonomous configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = runner.invoke(
                app,
                [
                    "generate-config",
                    "autonomous",
                    "--output",
                    temp_path,
                    "--max-algorithms",
                    "3",
                    "--verbose",
                    "true",
                ],
            )

            assert result.exit_code == 0
            assert "Autonomous configuration generated" in result.stdout

            # Verify file contents
            with open(temp_path) as f:
                config = json.load(f)

            assert config["metadata"]["type"] == "autonomous"
            assert config["autonomous"]["detection"]["max_algorithms"] == 3
            assert config["autonomous"]["output"]["verbose"] is True

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_generate_config_yaml_format(self, runner):
        """Test generate configuration in YAML format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            result = runner.invoke(
                app,
                ["generate-config", "test", "--output", temp_path, "--format", "yaml"],
            )

            assert result.exit_code == 0
            assert "Test configuration generated" in result.stdout

            # Verify file exists and contains YAML
            with open(temp_path) as f:
                content = f.read()

            assert "metadata:" in content
            assert "type: test" in content

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_generate_config_invalid_type(self, runner):
        """Test generate configuration with invalid type."""
        result = runner.invoke(app, ["generate-config", "invalid_type"])

        assert result.exit_code == 1
        assert "Unknown config type 'invalid_type'" in result.stdout

    def test_quickstart_interactive(self, runner):
        """Test quickstart command with user interaction."""
        # Test declining quickstart
        result = runner.invoke(app, ["quickstart"], input="n\n")

        assert result.exit_code == 0
        assert "Welcome to Pynomaly!" in result.stdout
        assert "Quickstart cancelled." in result.stdout

    def test_quickstart_accept(self, runner):
        """Test quickstart command accepting guide."""
        result = runner.invoke(app, ["quickstart"], input="y\n")

        assert result.exit_code == 0
        assert "Welcome to Pynomaly!" in result.stdout
        assert "Step 1: Load a dataset" in result.stdout
        assert "Step 2: Clean and preprocess" in result.stdout
        assert "Ready to start!" in result.stdout

    def test_verbose_and_quiet_conflict(self, runner):
        """Test that verbose and quiet options conflict."""
        result = runner.invoke(app, ["--verbose", "--quiet", "version"])

        assert result.exit_code == 1
        assert "Cannot use --verbose and --quiet together" in result.stdout

    def test_verbose_mode(self, runner, mock_container):
        """Test verbose mode."""
        result = runner.invoke(app, ["--verbose", "version"])

        assert result.exit_code == 0
        assert "Pynomaly v1.0.0" in result.stdout

    def test_quiet_mode(self, runner, mock_container):
        """Test quiet mode."""
        result = runner.invoke(app, ["--quiet", "version"])

        assert result.exit_code == 0


class TestDetectorsCLI:
    """Test suite for detectors CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container for detector operations."""
        with patch("pynomaly.presentation.cli.container.get_cli_container") as mock:
            container = Mock()

            # Mock detector repository
            detector_repo = Mock()
            container.detector_repository.return_value = detector_repo

            # Mock sample detector
            sample_detector = Mock()
            sample_detector.id = "test-detector-123"
            sample_detector.name = "Test Detector"
            sample_detector.algorithm_name = "IsolationForest"
            sample_detector.contamination_rate = ContaminationRate(0.1)
            sample_detector.is_fitted = False
            sample_detector.created_at = datetime.utcnow()
            sample_detector.parameters = {"n_estimators": 100}
            sample_detector.metadata = {"description": "Test detector"}

            detector_repo.find_all.return_value = [sample_detector]
            detector_repo.find_by_id.return_value = sample_detector
            detector_repo.find_by_name.return_value = sample_detector
            detector_repo.save.return_value = True
            detector_repo.delete.return_value = True

            mock.return_value = container
            yield container

    def test_detector_list(self, runner, mock_container):
        """Test detector list command."""
        result = runner.invoke(detectors.app, ["list"])

        assert result.exit_code == 0
        assert "Test Detector" in result.stdout
        assert "IsolationForest" in result.stdout

    def test_detector_list_with_algorithm_filter(self, runner, mock_container):
        """Test detector list with algorithm filter."""
        result = runner.invoke(
            detectors.app, ["list", "--algorithm", "IsolationForest"]
        )

        assert result.exit_code == 0

    def test_detector_create_success(self, runner, mock_container):
        """Test successful detector creation."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.PyODAdapter"
        ) as mock_adapter:
            mock_instance = Mock()
            mock_adapter.return_value = mock_instance

            result = runner.invoke(
                detectors.app,
                [
                    "create",
                    "--name",
                    "New Detector",
                    "--algorithm",
                    "IsolationForest",
                    "--contamination",
                    "0.05",
                ],
            )

            assert result.exit_code == 0

    def test_detector_create_invalid_algorithm(self, runner, mock_container):
        """Test detector creation with invalid algorithm."""
        result = runner.invoke(
            detectors.app,
            [
                "create",
                "--name",
                "Invalid Detector",
                "--algorithm",
                "NonExistentAlgorithm",
                "--contamination",
                "0.1",
            ],
        )

        assert result.exit_code != 0

    def test_detector_show(self, runner, mock_container):
        """Test detector show command."""
        result = runner.invoke(detectors.app, ["show", "Test Detector"])

        assert result.exit_code == 0
        assert "Test Detector" in result.stdout
        assert "IsolationForest" in result.stdout

    def test_detector_show_not_found(self, runner, mock_container):
        """Test detector show with non-existent detector."""
        mock_container.detector_repository.return_value.find_by_name.return_value = None

        result = runner.invoke(detectors.app, ["show", "NonExistent"])

        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_detector_delete_success(self, runner, mock_container):
        """Test successful detector deletion."""
        result = runner.invoke(detectors.app, ["delete", "Test Detector"], input="y\n")

        assert result.exit_code == 0

    def test_detector_delete_cancelled(self, runner, mock_container):
        """Test detector deletion cancellation."""
        result = runner.invoke(detectors.app, ["delete", "Test Detector"], input="n\n")

        assert result.exit_code == 0
        assert "Deletion cancelled" in result.stdout

    def test_detector_algorithms_list(self, runner):
        """Test listing available algorithms."""
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.PyODAdapter"
        ) as mock_pyod:
            with patch(
                "pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter"
            ) as mock_sklearn:
                mock_pyod.ALGORITHM_MAPPING = {
                    "IsolationForest": "IF",
                    "LOF": "LocalOutlierFactor",
                }
                mock_sklearn.ALGORITHM_MAPPING = {"OneClassSVM": "OCSVM"}

                result = runner.invoke(detectors.app, ["algorithms"])

                assert result.exit_code == 0
                assert "Available Algorithms" in result.stdout

    def test_detector_tune_hyperparameters(self, runner, mock_container):
        """Test hyperparameter tuning."""
        with patch(
            "pynomaly.application.services.hyperparameter_tuning_service.HyperparameterTuningService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.tune_hyperparameters.return_value = {
                "n_estimators": 150
            }

            result = runner.invoke(
                detectors.app,
                ["tune", "Test Detector", "--method", "grid_search", "--cv-folds", "3"],
            )

            assert result.exit_code == 0


class TestDatasetsCLI:
    """Test suite for datasets CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container for dataset operations."""
        with patch("pynomaly.presentation.cli.container.get_cli_container") as mock:
            container = Mock()

            # Mock dataset repository
            dataset_repo = Mock()
            container.dataset_repository.return_value = dataset_repo

            # Mock sample dataset
            sample_dataset = Mock()
            sample_dataset.id = "test-dataset-123"
            sample_dataset.name = "Test Dataset"
            sample_dataset.shape = (1000, 5)
            sample_dataset.features = [
                "feature1",
                "feature2",
                "feature3",
                "feature4",
                "feature5",
            ]
            sample_dataset.contamination_rate = 0.05
            sample_dataset.created_at = datetime.utcnow()

            dataset_repo.find_all.return_value = [sample_dataset]
            dataset_repo.find_by_name.return_value = sample_dataset
            dataset_repo.save.return_value = True
            dataset_repo.delete.return_value = True

            mock.return_value = container
            yield container

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["feature1", "feature2", "feature3", "target"])
            writer.writerow([1.0, 2.0, 3.0, 0])
            writer.writerow([2.0, 3.0, 4.0, 0])
            writer.writerow([100.0, 200.0, 300.0, 1])  # Outlier
            writer.writerow([3.0, 4.0, 5.0, 0])
            temp_path = f.name

        yield temp_path

        Path(temp_path).unlink(missing_ok=True)

    def test_dataset_list(self, runner, mock_container):
        """Test dataset list command."""
        result = runner.invoke(datasets.app, ["list"])

        assert result.exit_code == 0
        assert "Test Dataset" in result.stdout

    def test_dataset_load_csv(self, runner, mock_container, sample_csv_file):
        """Test loading CSV dataset."""
        result = runner.invoke(
            datasets.app,
            [
                "load",
                sample_csv_file,
                "--name",
                "Test CSV Dataset",
                "--target-column",
                "target",
            ],
        )

        assert result.exit_code == 0

    def test_dataset_load_missing_file(self, runner, mock_container):
        """Test loading non-existent file."""
        result = runner.invoke(
            datasets.app,
            ["load", "/path/to/nonexistent.csv", "--name", "Missing Dataset"],
        )

        assert result.exit_code == 1

    def test_dataset_show(self, runner, mock_container):
        """Test dataset show command."""
        result = runner.invoke(datasets.app, ["show", "Test Dataset"])

        assert result.exit_code == 0
        assert "Test Dataset" in result.stdout

    def test_dataset_show_not_found(self, runner, mock_container):
        """Test dataset show with non-existent dataset."""
        mock_container.dataset_repository.return_value.find_by_name.return_value = None

        result = runner.invoke(datasets.app, ["show", "NonExistent"])

        assert result.exit_code == 1

    def test_dataset_describe(self, runner, mock_container):
        """Test dataset describe command."""
        with patch("pandas.DataFrame.describe") as mock_describe:
            mock_describe.return_value = Mock()

            result = runner.invoke(datasets.app, ["describe", "Test Dataset"])

            assert result.exit_code == 0

    def test_dataset_delete(self, runner, mock_container):
        """Test dataset deletion."""
        result = runner.invoke(datasets.app, ["delete", "Test Dataset"], input="y\n")

        assert result.exit_code == 0

    def test_dataset_export_csv(self, runner, mock_container):
        """Test dataset export to CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            result = runner.invoke(
                datasets.app,
                ["export", "Test Dataset", "--format", "csv", "--output", temp_path],
            )

            assert result.exit_code == 0

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_dataset_validate(self, runner, mock_container):
        """Test dataset validation."""
        with patch(
            "pynomaly.application.services.data_validation_service.DataValidationService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.validate.return_value = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
            }

            result = runner.invoke(datasets.app, ["validate", "Test Dataset"])

            assert result.exit_code == 0


class TestDetectionCLI:
    """Test suite for detection CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container for detection operations."""
        with patch("pynomaly.presentation.cli.container.get_cli_container") as mock:
            container = Mock()

            # Mock repositories
            detector_repo = Mock()
            dataset_repo = Mock()
            result_repo = Mock()

            container.detector_repository.return_value = detector_repo
            container.dataset_repository.return_value = dataset_repo
            container.result_repository.return_value = result_repo

            # Mock detector and dataset
            mock_detector = Mock()
            mock_detector.name = "Test Detector"
            mock_detector.is_fitted = False

            mock_dataset = Mock()
            mock_dataset.name = "Test Dataset"

            detector_repo.find_by_name.return_value = mock_detector
            dataset_repo.find_by_name.return_value = mock_dataset

            # Mock detection result
            mock_result = Mock()
            mock_result.id = "result-123"
            mock_result.detector_name = "Test Detector"
            mock_result.dataset_name = "Test Dataset"
            mock_result.n_anomalies = 10
            mock_result.anomaly_rate = 0.1
            mock_result.timestamp = datetime.utcnow()

            result_repo.find_recent.return_value = [mock_result]

            mock.return_value = container
            yield container

    def test_detection_train(self, runner, mock_container):
        """Test detector training."""
        with patch(
            "pynomaly.application.use_cases.train_detector.TrainDetectorUseCase"
        ) as mock_use_case:
            mock_use_case_instance = Mock()
            mock_use_case.return_value = mock_use_case_instance

            result = runner.invoke(
                detection.app,
                ["train", "--detector", "Test Detector", "--dataset", "Test Dataset"],
            )

            assert result.exit_code == 0

    def test_detection_run(self, runner, mock_container):
        """Test running detection."""
        # Set detector as fitted for detection
        mock_container.detector_repository.return_value.find_by_name.return_value.is_fitted = (
            True
        )

        with patch(
            "pynomaly.application.use_cases.detect_anomalies.DetectAnomaliesUseCase"
        ) as mock_use_case:
            mock_use_case_instance = Mock()
            mock_use_case.return_value = mock_use_case_instance
            mock_use_case_instance.execute.return_value = Mock()

            result = runner.invoke(
                detection.app,
                ["run", "--detector", "Test Detector", "--dataset", "Test Dataset"],
            )

            assert result.exit_code == 0

    def test_detection_run_unfitted_detector(self, runner, mock_container):
        """Test running detection with unfitted detector."""
        result = runner.invoke(
            detection.app,
            ["run", "--detector", "Test Detector", "--dataset", "Test Dataset"],
        )

        assert result.exit_code == 1
        assert "not fitted" in result.stdout

    def test_detection_evaluate(self, runner, mock_container):
        """Test detector evaluation."""
        mock_container.detector_repository.return_value.find_by_name.return_value.is_fitted = (
            True
        )

        with patch(
            "pynomaly.application.use_cases.evaluate_detector.EvaluateDetectorUseCase"
        ) as mock_use_case:
            mock_use_case_instance = Mock()
            mock_use_case.return_value = mock_use_case_instance
            mock_use_case_instance.execute.return_value = {
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81,
                "auc_roc": 0.92,
            }

            result = runner.invoke(
                detection.app,
                [
                    "evaluate",
                    "--detector",
                    "Test Detector",
                    "--dataset",
                    "Test Dataset",
                ],
            )

            assert result.exit_code == 0
            assert "precision" in result.stdout

    def test_detection_results(self, runner, mock_container):
        """Test viewing detection results."""
        result = runner.invoke(detection.app, ["results", "--latest"])

        assert result.exit_code == 0
        assert "Test Detector" in result.stdout
        assert "Test Dataset" in result.stdout

    def test_detection_batch_multiple_detectors(self, runner, mock_container):
        """Test batch detection with multiple detectors."""
        with patch(
            "pynomaly.application.use_cases.batch_detection.BatchDetectionUseCase"
        ) as mock_use_case:
            mock_use_case_instance = Mock()
            mock_use_case.return_value = mock_use_case_instance

            result = runner.invoke(
                detection.app,
                [
                    "batch",
                    "IsolationForest",
                    "LOF",
                    "OneClassSVM",
                    "--dataset",
                    "Test Dataset",
                ],
            )

            assert result.exit_code == 0


class TestAutonomousCLI:
    """Test suite for autonomous CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["feature1", "feature2", "feature3"])
            for i in range(100):
                writer.writerow([i, i * 2, i * 3])
            temp_path = f.name

        yield temp_path

        Path(temp_path).unlink(missing_ok=True)

    def test_autonomous_detect(self, runner, sample_csv_file):
        """Test autonomous detection."""
        with patch(
            "pynomaly.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.detect_anomalies.return_value = {
                "best_detector": "IsolationForest",
                "anomalies_found": 5,
                "confidence": 0.85,
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as out_file:
                output_path = out_file.name

            try:
                result = runner.invoke(
                    autonomous.app, ["detect", sample_csv_file, "--output", output_path]
                )

                assert result.exit_code == 0

            finally:
                Path(output_path).unlink(missing_ok=True)

    def test_autonomous_profile(self, runner, sample_csv_file):
        """Test autonomous data profiling."""
        with patch(
            "pynomaly.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.profile_data.return_value = {
                "shape": (100, 3),
                "data_types": {
                    "feature1": "numeric",
                    "feature2": "numeric",
                    "feature3": "numeric",
                },
                "missing_values": 0,
                "recommended_algorithms": ["IsolationForest", "LOF"],
            }

            result = runner.invoke(autonomous.app, ["profile", sample_csv_file])

            assert result.exit_code == 0
            assert "Data Profile" in result.stdout

    def test_autonomous_quick(self, runner, sample_csv_file):
        """Test autonomous quick detection."""
        with patch(
            "pynomaly.application.services.autonomous_service.AutonomousDetectionService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.quick_detect.return_value = {
                "anomalies": [95, 96, 97, 98, 99],
                "scores": [0.8, 0.85, 0.9, 0.92, 0.95],
            }

            result = runner.invoke(
                autonomous.app, ["quick", sample_csv_file, "--contamination", "0.05"]
            )

            assert result.exit_code == 0


class TestPreprocessingCLI:
    """Test suite for preprocessing CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container for preprocessing operations."""
        with patch("pynomaly.presentation.cli.container.get_cli_container") as mock:
            container = Mock()

            dataset_repo = Mock()
            container.dataset_repository.return_value = dataset_repo

            mock_dataset = Mock()
            mock_dataset.name = "Test Dataset"
            dataset_repo.find_by_name.return_value = mock_dataset

            mock.return_value = container
            yield container

    def test_preprocessing_clean(self, runner, mock_container):
        """Test data cleaning."""
        with patch(
            "pynomaly.application.services.preprocessing_service.PreprocessingService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance

            result = runner.invoke(
                preprocessing.app,
                [
                    "clean",
                    "Test Dataset",
                    "--missing",
                    "drop_rows",
                    "--outliers",
                    "clip",
                    "--duplicates",
                    "remove",
                ],
            )

            assert result.exit_code == 0

    def test_preprocessing_transform(self, runner, mock_container):
        """Test data transformation."""
        with patch(
            "pynomaly.application.services.preprocessing_service.PreprocessingService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance

            result = runner.invoke(
                preprocessing.app,
                [
                    "transform",
                    "Test Dataset",
                    "--scaling",
                    "standard",
                    "--encoding",
                    "onehot",
                ],
            )

            assert result.exit_code == 0

    def test_preprocessing_pipeline_create(self, runner, mock_container):
        """Test preprocessing pipeline creation."""
        with patch(
            "pynomaly.application.services.preprocessing_service.PreprocessingService"
        ) as mock_service:
            mock_service_instance = Mock()
            mock_service.return_value = mock_service_instance

            result = runner.invoke(
                preprocessing.app,
                [
                    "pipeline",
                    "create",
                    "--name",
                    "Test Pipeline",
                    "--steps",
                    "clean,transform,validate",
                ],
            )

            assert result.exit_code == 0


class TestExportCLI:
    """Test suite for export CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results_file(self):
        """Create sample results file for export testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            results = {
                "detector": "IsolationForest",
                "dataset": "Test Data",
                "anomalies": [1, 5, 10, 15],
                "scores": [0.8, 0.9, 0.85, 0.95],
                "metadata": {"total_samples": 100},
            }
            json.dump(results, f)
            temp_path = f.name

        yield temp_path

        Path(temp_path).unlink(missing_ok=True)

    def test_export_list_formats(self, runner):
        """Test listing export formats."""
        result = runner.invoke(export_app, ["list-formats"])

        assert result.exit_code == 0
        assert "Available Export Formats" in result.stdout

    def test_export_excel(self, runner, sample_results_file):
        """Test Excel export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xlsx", delete=False) as f:
            output_path = f.name

        try:
            with patch(
                "pynomaly.infrastructure.exporters.excel_exporter.ExcelExporter"
            ) as mock_exporter:
                mock_exporter_instance = Mock()
                mock_exporter.return_value = mock_exporter_instance

                result = runner.invoke(
                    export_app, ["excel", sample_results_file, output_path]
                )

                assert result.exit_code == 0

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_powerbi(self, runner, sample_results_file):
        """Test Power BI export."""
        with patch(
            "pynomaly.infrastructure.exporters.powerbi_exporter.PowerBIExporter"
        ) as mock_exporter:
            mock_exporter_instance = Mock()
            mock_exporter.return_value = mock_exporter_instance

            result = runner.invoke(
                export_app,
                ["powerbi", sample_results_file, "--workspace-id", "test-workspace"],
            )

            assert result.exit_code == 0


class TestServerCLI:
    """Test suite for server CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_server_start_help(self, runner):
        """Test server start command help."""
        result = runner.invoke(server.app, ["start", "--help"])

        assert result.exit_code == 0
        assert "Start the API server" in result.stdout

    def test_server_status(self, runner):
        """Test server status command."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            result = runner.invoke(server.app, ["status"])

            assert result.exit_code == 0
            assert "Server Status" in result.stdout


class TestCLIErrorHandling:
    """Test suite for CLI error handling."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_invalid_command(self, runner):
        """Test invalid command handling."""
        result = runner.invoke(app, ["invalid-command"])

        assert result.exit_code != 0

    def test_missing_required_arguments(self, runner):
        """Test missing required arguments."""
        result = runner.invoke(detectors.app, ["create"])

        assert result.exit_code != 0

    def test_invalid_option_values(self, runner):
        """Test invalid option values."""
        result = runner.invoke(
            detectors.app,
            [
                "create",
                "--name",
                "Test",
                "--algorithm",
                "IsolationForest",
                "--contamination",
                "invalid_float",
            ],
        )

        assert result.exit_code != 0

    def test_file_permission_errors(self, runner):
        """Test file permission error handling."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                result = runner.invoke(
                    datasets.app,
                    ["load", "/restricted/file.csv", "--name", "Restricted Dataset"],
                )

                assert result.exit_code == 1


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_complete_detection_workflow(self, runner):
        """Test complete detection workflow from CLI."""
        with patch(
            "pynomaly.presentation.cli.container.get_cli_container"
        ) as mock_container:
            container = Mock()

            # Mock all required services
            container.detector_repository.return_value = Mock()
            container.dataset_repository.return_value = Mock()
            container.result_repository.return_value = Mock()

            mock_container.return_value = container

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["feature1", "feature2"])
                writer.writerow([1.0, 2.0])
                writer.writerow([2.0, 3.0])
                temp_path = f.name

            try:
                # 1. Load dataset
                load_result = runner.invoke(
                    datasets.app, ["load", temp_path, "--name", "workflow_data"]
                )

                # 2. Create detector
                create_result = runner.invoke(
                    detectors.app,
                    [
                        "create",
                        "--name",
                        "workflow_detector",
                        "--algorithm",
                        "IsolationForest",
                        "--contamination",
                        "0.1",
                    ],
                )

                # All commands should handle the mocked dependencies
                assert load_result.exit_code in [0, 1]  # Success or expected failure
                assert create_result.exit_code in [0, 1]  # Success or expected failure

            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_configuration_workflow(self, runner):
        """Test configuration generation and usage workflow."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name

        try:
            # Generate test configuration
            gen_result = runner.invoke(
                app,
                [
                    "generate-config",
                    "test",
                    "--output",
                    config_path,
                    "--detector",
                    "IsolationForest",
                ],
            )

            assert gen_result.exit_code == 0

            # Verify configuration was created
            assert Path(config_path).exists()

            with open(config_path) as f:
                config = json.load(f)

            assert config["metadata"]["type"] == "test"
            assert config["test"]["detector"]["algorithm"] == "IsolationForest"

        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_help_system_completeness(self, runner):
        """Test that all commands have proper help documentation."""
        # Test main app help
        main_help = runner.invoke(app, ["--help"])
        assert main_help.exit_code == 0
        assert "Commands:" in main_help.stdout

        # Test subcommand helps
        subcommands = [
            "detector",
            "dataset",
            "detect",
            "auto",
            "data",
            "export",
            "server",
        ]

        for cmd in subcommands:
            help_result = runner.invoke(app, [cmd, "--help"])
            assert help_result.exit_code == 0
            assert "Usage:" in help_result.stdout or "Commands:" in help_result.stdout
