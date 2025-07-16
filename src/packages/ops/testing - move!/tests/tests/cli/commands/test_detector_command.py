"""
Comprehensive tests for detector CLI commands.
Tests for detector management, training, evaluation, and model lifecycle.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from monorepo.domain.exceptions import DetectorError, ValidationError
from monorepo.presentation.cli.detectors import app as detector_app


class TestDetectorCommand:
    """Test suite for detector CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_detector_service(self):
        """Mock detector service."""
        with patch(
            "monorepo.presentation.cli.commands.detector.detector_service"
        ) as mock:
            yield mock

    @pytest.fixture
    def mock_training_service(self):
        """Mock training service."""
        with patch(
            "monorepo.presentation.cli.commands.detector.training_service"
        ) as mock:
            yield mock

    @pytest.fixture
    def mock_evaluation_service(self):
        """Mock evaluation service."""
        with patch(
            "monorepo.presentation.cli.commands.detector.evaluation_service"
        ) as mock:
            yield mock

    @pytest.fixture
    def mock_container(self):
        """Mock CLI container."""
        with patch(
            "monorepo.presentation.cli.commands.detector.get_cli_container"
        ) as mock:
            container = Mock()
            container.config.return_value.storage_path = Path("/tmp/pynomaly")
            container.config.return_value.default_contamination_rate = 0.1
            container.config.return_value.gpu_enabled = False
            mock.return_value = container
            yield container

    @pytest.fixture
    def sample_detector_data(self):
        """Sample detector data."""
        return {
            "id": "test-detector",
            "name": "Test Detector",
            "algorithm": "IsolationForest",
            "hyperparameters": {"n_estimators": 100, "contamination": 0.1},
            "status": "trained",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "training_dataset_id": "training-dataset",
            "metrics": {"precision": 0.85, "recall": 0.78, "f1": 0.81},
        }

    @pytest.fixture
    def sample_training_result(self):
        """Sample training result."""
        return {
            "detector_id": "test-detector",
            "training_time": 45.2,
            "metrics": {"precision": 0.85, "recall": 0.78, "f1": 0.81, "auc": 0.83},
            "hyperparameters": {"n_estimators": 100, "contamination": 0.1},
            "status": "completed",
        }

    # Detector List Command Tests

    def test_detector_list_basic(
        self, runner, mock_detector_service, sample_detector_data
    ):
        """Test basic detector list command."""
        mock_detector = Mock()
        mock_detector.to_dict.return_value = sample_detector_data
        mock_detector_service.list_detectors.return_value = [mock_detector]

        result = runner.invoke(detector_app, ["list"])

        assert result.exit_code == 0
        assert "Test Detector" in result.stdout
        assert "test-detector" in result.stdout
        assert "IsolationForest" in result.stdout
        mock_detector_service.list_detectors.assert_called_once()

    def test_detector_list_empty(self, runner, mock_detector_service):
        """Test detector list with no detectors."""
        mock_detector_service.list_detectors.return_value = []

        result = runner.invoke(detector_app, ["list"])

        assert result.exit_code == 0
        assert "No detectors found" in result.stdout
        mock_detector_service.list_detectors.assert_called_once()

    def test_detector_list_with_filter(
        self, runner, mock_detector_service, sample_detector_data
    ):
        """Test detector list with algorithm filter."""
        mock_detector = Mock()
        mock_detector.to_dict.return_value = sample_detector_data
        mock_detector_service.list_detectors.return_value = [mock_detector]

        result = runner.invoke(detector_app, ["list", "--algorithm", "IsolationForest"])

        assert result.exit_code == 0
        assert "IsolationForest" in result.stdout
        mock_detector_service.list_detectors.assert_called_once()

    def test_detector_list_with_status_filter(
        self, runner, mock_detector_service, sample_detector_data
    ):
        """Test detector list with status filter."""
        mock_detector = Mock()
        mock_detector.to_dict.return_value = sample_detector_data
        mock_detector_service.list_detectors.return_value = [mock_detector]

        result = runner.invoke(detector_app, ["list", "--status", "trained"])

        assert result.exit_code == 0
        assert "trained" in result.stdout
        mock_detector_service.list_detectors.assert_called_once()

    def test_detector_list_with_format_json(
        self, runner, mock_detector_service, sample_detector_data
    ):
        """Test detector list with JSON format."""
        mock_detector = Mock()
        mock_detector.to_dict.return_value = sample_detector_data
        mock_detector_service.list_detectors.return_value = [mock_detector]

        result = runner.invoke(detector_app, ["list", "--format", "json"])

        assert result.exit_code == 0
        assert "{" in result.stdout
        assert "test-detector" in result.stdout
        mock_detector_service.list_detectors.assert_called_once()

    # Detector Show Command Tests

    def test_detector_show_basic(
        self, runner, mock_detector_service, sample_detector_data
    ):
        """Test basic detector show command."""
        mock_detector = Mock()
        mock_detector.to_dict.return_value = sample_detector_data
        mock_detector_service.get_detector.return_value = mock_detector

        result = runner.invoke(detector_app, ["show", "test-detector"])

        assert result.exit_code == 0
        assert "Test Detector" in result.stdout
        assert "IsolationForest" in result.stdout
        mock_detector_service.get_detector.assert_called_once_with("test-detector")

    def test_detector_show_not_found(self, runner, mock_detector_service):
        """Test detector show with non-existent detector."""
        mock_detector_service.get_detector.side_effect = DetectorError(
            "Detector not found"
        )

        result = runner.invoke(detector_app, ["show", "non-existent"])

        assert result.exit_code == 1
        assert "Detector not found" in result.stdout
        mock_detector_service.get_detector.assert_called_once_with("non-existent")

    def test_detector_show_with_metrics(
        self, runner, mock_detector_service, sample_detector_data
    ):
        """Test detector show with metrics."""
        mock_detector = Mock()
        mock_detector.to_dict.return_value = sample_detector_data
        mock_detector_service.get_detector.return_value = mock_detector

        result = runner.invoke(detector_app, ["show", "test-detector", "--metrics"])

        assert result.exit_code == 0
        assert "Metrics" in result.stdout
        assert "precision" in result.stdout
        assert "0.85" in result.stdout
        mock_detector_service.get_detector.assert_called_once_with("test-detector")

    def test_detector_show_with_hyperparameters(
        self, runner, mock_detector_service, sample_detector_data
    ):
        """Test detector show with hyperparameters."""
        mock_detector = Mock()
        mock_detector.to_dict.return_value = sample_detector_data
        mock_detector_service.get_detector.return_value = mock_detector

        result = runner.invoke(
            detector_app, ["show", "test-detector", "--hyperparameters"]
        )

        assert result.exit_code == 0
        assert "Hyperparameters" in result.stdout
        assert "n_estimators" in result.stdout
        assert "100" in result.stdout
        mock_detector_service.get_detector.assert_called_once_with("test-detector")

    # Detector Create Command Tests

    def test_detector_create_basic(self, runner, mock_detector_service):
        """Test basic detector create command."""
        mock_detector = Mock()
        mock_detector.id = "new-detector"
        mock_detector_service.create_detector.return_value = mock_detector

        result = runner.invoke(
            detector_app,
            [
                "create",
                "new-detector",
                "--algorithm",
                "IsolationForest",
                "--description",
                "Test detector",
            ],
        )

        assert result.exit_code == 0
        assert "Detector created successfully" in result.stdout
        mock_detector_service.create_detector.assert_called_once()

    def test_detector_create_with_hyperparameters(self, runner, mock_detector_service):
        """Test detector create with hyperparameters."""
        mock_detector = Mock()
        mock_detector.id = "advanced-detector"
        mock_detector_service.create_detector.return_value = mock_detector

        result = runner.invoke(
            detector_app,
            [
                "create",
                "advanced-detector",
                "--algorithm",
                "IsolationForest",
                "--hyperparameters",
                '{"n_estimators": 200, "contamination": 0.05}',
            ],
        )

        assert result.exit_code == 0
        assert "Detector created successfully" in result.stdout
        mock_detector_service.create_detector.assert_called_once()

    def test_detector_create_with_invalid_algorithm(
        self, runner, mock_detector_service
    ):
        """Test detector create with invalid algorithm."""
        result = runner.invoke(
            detector_app, ["create", "test-detector", "--algorithm", "InvalidAlgorithm"]
        )

        assert result.exit_code == 1
        assert "Unknown algorithm" in result.stdout

    def test_detector_create_with_invalid_hyperparameters(
        self, runner, mock_detector_service
    ):
        """Test detector create with invalid hyperparameters JSON."""
        result = runner.invoke(
            detector_app,
            [
                "create",
                "test-detector",
                "--algorithm",
                "IsolationForest",
                "--hyperparameters",
                "invalid-json",
            ],
        )

        assert result.exit_code == 1
        assert "Invalid JSON" in result.stdout

    # Detector Train Command Tests

    def test_detector_train_basic(
        self, runner, mock_training_service, sample_training_result
    ):
        """Test basic detector train command."""
        mock_training_service.train_detector.return_value = sample_training_result

        result = runner.invoke(
            detector_app, ["train", "test-detector", "--dataset", "training-dataset"]
        )

        assert result.exit_code == 0
        assert "Training completed successfully" in result.stdout
        assert "45.2" in result.stdout  # training time
        mock_training_service.train_detector.assert_called_once()

    def test_detector_train_with_validation(
        self, runner, mock_training_service, sample_training_result
    ):
        """Test detector train with validation dataset."""
        mock_training_service.train_detector.return_value = sample_training_result

        result = runner.invoke(
            detector_app,
            [
                "train",
                "test-detector",
                "--dataset",
                "training-dataset",
                "--validation-dataset",
                "validation-dataset",
            ],
        )

        assert result.exit_code == 0
        assert "Training completed successfully" in result.stdout
        mock_training_service.train_detector.assert_called_once()

    def test_detector_train_with_custom_contamination(
        self, runner, mock_training_service, sample_training_result
    ):
        """Test detector train with custom contamination rate."""
        mock_training_service.train_detector.return_value = sample_training_result

        result = runner.invoke(
            detector_app,
            [
                "train",
                "test-detector",
                "--dataset",
                "training-dataset",
                "--contamination",
                "0.05",
            ],
        )

        assert result.exit_code == 0
        assert "Training completed successfully" in result.stdout
        mock_training_service.train_detector.assert_called_once()

    def test_detector_train_with_hyperparameter_tuning(
        self, runner, mock_training_service, sample_training_result
    ):
        """Test detector train with hyperparameter tuning."""
        mock_training_service.train_detector.return_value = sample_training_result

        result = runner.invoke(
            detector_app,
            [
                "train",
                "test-detector",
                "--dataset",
                "training-dataset",
                "--tune-hyperparameters",
            ],
        )

        assert result.exit_code == 0
        assert "Training completed successfully" in result.stdout
        mock_training_service.train_detector.assert_called_once()

    def test_detector_train_not_found(self, runner, mock_training_service):
        """Test detector train with non-existent detector."""
        mock_training_service.train_detector.side_effect = DetectorError(
            "Detector not found"
        )

        result = runner.invoke(
            detector_app, ["train", "non-existent", "--dataset", "training-dataset"]
        )

        assert result.exit_code == 1
        assert "Detector not found" in result.stdout

    def test_detector_train_with_invalid_contamination(
        self, runner, mock_training_service
    ):
        """Test detector train with invalid contamination rate."""
        result = runner.invoke(
            detector_app,
            [
                "train",
                "test-detector",
                "--dataset",
                "training-dataset",
                "--contamination",
                "1.5",
            ],
        )

        assert result.exit_code == 1
        assert "Contamination rate must be between 0.0 and 1.0" in result.stdout

    # Detector Evaluate Command Tests

    def test_detector_evaluate_basic(self, runner, mock_evaluation_service):
        """Test basic detector evaluate command."""
        mock_evaluation_result = {
            "detector_id": "test-detector",
            "dataset_id": "test-dataset",
            "metrics": {
                "precision": 0.85,
                "recall": 0.78,
                "f1": 0.81,
                "auc": 0.83,
                "accuracy": 0.92,
            },
            "confusion_matrix": [[950, 50], [22, 78]],
            "evaluation_time": 12.5,
        }
        mock_evaluation_service.evaluate_detector.return_value = mock_evaluation_result

        result = runner.invoke(
            detector_app, ["evaluate", "test-detector", "--dataset", "test-dataset"]
        )

        assert result.exit_code == 0
        assert "Evaluation completed successfully" in result.stdout
        assert "precision" in result.stdout
        assert "0.85" in result.stdout
        mock_evaluation_service.evaluate_detector.assert_called_once()

    def test_detector_evaluate_with_detailed_metrics(
        self, runner, mock_evaluation_service
    ):
        """Test detector evaluate with detailed metrics."""
        mock_evaluation_result = {
            "detector_id": "test-detector",
            "dataset_id": "test-dataset",
            "metrics": {"precision": 0.85, "recall": 0.78, "f1": 0.81, "auc": 0.83},
            "confusion_matrix": [[950, 50], [22, 78]],
            "roc_curve": {"fpr": [0.0, 0.1, 1.0], "tpr": [0.0, 0.8, 1.0]},
            "precision_recall_curve": {
                "precision": [1.0, 0.9, 0.8],
                "recall": [0.0, 0.5, 1.0],
            },
        }
        mock_evaluation_service.evaluate_detector.return_value = mock_evaluation_result

        result = runner.invoke(
            detector_app,
            ["evaluate", "test-detector", "--dataset", "test-dataset", "--detailed"],
        )

        assert result.exit_code == 0
        assert "Confusion Matrix" in result.stdout
        assert "ROC Curve" in result.stdout
        mock_evaluation_service.evaluate_detector.assert_called_once()

    def test_detector_evaluate_with_threshold(self, runner, mock_evaluation_service):
        """Test detector evaluate with custom threshold."""
        mock_evaluation_result = {
            "detector_id": "test-detector",
            "dataset_id": "test-dataset",
            "metrics": {"precision": 0.85, "recall": 0.78, "f1": 0.81},
            "threshold": 0.7,
        }
        mock_evaluation_service.evaluate_detector.return_value = mock_evaluation_result

        result = runner.invoke(
            detector_app,
            [
                "evaluate",
                "test-detector",
                "--dataset",
                "test-dataset",
                "--threshold",
                "0.7",
            ],
        )

        assert result.exit_code == 0
        assert "Evaluation completed successfully" in result.stdout
        mock_evaluation_service.evaluate_detector.assert_called_once()

    def test_detector_evaluate_not_found(self, runner, mock_evaluation_service):
        """Test detector evaluate with non-existent detector."""
        mock_evaluation_service.evaluate_detector.side_effect = DetectorError(
            "Detector not found"
        )

        result = runner.invoke(
            detector_app, ["evaluate", "non-existent", "--dataset", "test-dataset"]
        )

        assert result.exit_code == 1
        assert "Detector not found" in result.stdout

    # Detector Predict Command Tests

    def test_detector_predict_basic(self, runner, mock_detector_service):
        """Test basic detector predict command."""
        mock_predictions = {
            "detector_id": "test-detector",
            "dataset_id": "test-dataset",
            "predictions": [
                {"index": 0, "score": 0.1, "is_anomaly": False},
                {"index": 1, "score": 0.8, "is_anomaly": True},
                {"index": 2, "score": 0.3, "is_anomaly": False},
            ],
            "summary": {
                "total_samples": 3,
                "anomalies_detected": 1,
                "anomaly_rate": 0.33,
            },
        }
        mock_detector_service.predict.return_value = mock_predictions

        result = runner.invoke(
            detector_app, ["predict", "test-detector", "--dataset", "test-dataset"]
        )

        assert result.exit_code == 0
        assert "Prediction completed successfully" in result.stdout
        assert "total_samples: 3" in result.stdout
        assert "anomalies_detected: 1" in result.stdout
        mock_detector_service.predict.assert_called_once()

    def test_detector_predict_with_output_file(self, runner, mock_detector_service):
        """Test detector predict with output file."""
        mock_predictions = {
            "predictions": [
                {"index": 0, "score": 0.1, "is_anomaly": False},
                {"index": 1, "score": 0.8, "is_anomaly": True},
            ]
        }
        mock_detector_service.predict.return_value = mock_predictions

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "predictions.json"

            result = runner.invoke(
                detector_app,
                [
                    "predict",
                    "test-detector",
                    "--dataset",
                    "test-dataset",
                    "--output",
                    str(output_file),
                ],
            )

            assert result.exit_code == 0
            assert "Predictions saved to" in result.stdout
            mock_detector_service.predict.assert_called_once()

    def test_detector_predict_with_threshold(self, runner, mock_detector_service):
        """Test detector predict with custom threshold."""
        mock_predictions = {
            "predictions": [{"index": 0, "score": 0.7, "is_anomaly": True}],
            "threshold": 0.6,
        }
        mock_detector_service.predict.return_value = mock_predictions

        result = runner.invoke(
            detector_app,
            [
                "predict",
                "test-detector",
                "--dataset",
                "test-dataset",
                "--threshold",
                "0.6",
            ],
        )

        assert result.exit_code == 0
        assert "Prediction completed successfully" in result.stdout
        mock_detector_service.predict.assert_called_once()

    # Detector Update Command Tests

    def test_detector_update_basic(self, runner, mock_detector_service):
        """Test basic detector update command."""
        mock_detector = Mock()
        mock_detector.id = "test-detector"
        mock_detector_service.update_detector.return_value = mock_detector

        result = runner.invoke(
            detector_app,
            ["update", "test-detector", "--description", "Updated description"],
        )

        assert result.exit_code == 0
        assert "Detector updated successfully" in result.stdout
        mock_detector_service.update_detector.assert_called_once()

    def test_detector_update_hyperparameters(self, runner, mock_detector_service):
        """Test detector update with hyperparameters."""
        mock_detector = Mock()
        mock_detector.id = "test-detector"
        mock_detector_service.update_detector.return_value = mock_detector

        result = runner.invoke(
            detector_app,
            ["update", "test-detector", "--hyperparameters", '{"n_estimators": 150}'],
        )

        assert result.exit_code == 0
        assert "Detector updated successfully" in result.stdout
        mock_detector_service.update_detector.assert_called_once()

    def test_detector_update_not_found(self, runner, mock_detector_service):
        """Test detector update with non-existent detector."""
        mock_detector_service.update_detector.side_effect = DetectorError(
            "Detector not found"
        )

        result = runner.invoke(
            detector_app, ["update", "non-existent", "--description", "New description"]
        )

        assert result.exit_code == 1
        assert "Detector not found" in result.stdout

    # Detector Delete Command Tests

    def test_detector_delete_basic(self, runner, mock_detector_service):
        """Test basic detector delete command."""
        mock_detector_service.delete_detector.return_value = None

        result = runner.invoke(detector_app, ["delete", "test-detector"], input="y\n")

        assert result.exit_code == 0
        assert "Detector deleted successfully" in result.stdout
        mock_detector_service.delete_detector.assert_called_once_with("test-detector")

    def test_detector_delete_with_force(self, runner, mock_detector_service):
        """Test detector delete with force flag."""
        mock_detector_service.delete_detector.return_value = None

        result = runner.invoke(detector_app, ["delete", "test-detector", "--force"])

        assert result.exit_code == 0
        assert "Detector deleted successfully" in result.stdout
        mock_detector_service.delete_detector.assert_called_once_with("test-detector")

    def test_detector_delete_cancelled(self, runner, mock_detector_service):
        """Test detector delete cancelled by user."""
        result = runner.invoke(detector_app, ["delete", "test-detector"], input="n\n")

        assert result.exit_code == 0
        assert "Deletion cancelled" in result.stdout
        mock_detector_service.delete_detector.assert_not_called()

    def test_detector_delete_not_found(self, runner, mock_detector_service):
        """Test detector delete with non-existent detector."""
        mock_detector_service.delete_detector.side_effect = DetectorError(
            "Detector not found"
        )

        result = runner.invoke(detector_app, ["delete", "non-existent"], input="y\n")

        assert result.exit_code == 1
        assert "Detector not found" in result.stdout

    # Detector Export Command Tests

    def test_detector_export_basic(self, runner, mock_detector_service):
        """Test basic detector export command."""
        mock_detector_service.export_detector.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "detector.pkl"

            result = runner.invoke(
                detector_app, ["export", "test-detector", "--output", str(output_file)]
            )

            assert result.exit_code == 0
            assert "Detector exported successfully" in result.stdout
            mock_detector_service.export_detector.assert_called_once()

    def test_detector_export_with_format(self, runner, mock_detector_service):
        """Test detector export with specific format."""
        mock_detector_service.export_detector.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "detector.json"

            result = runner.invoke(
                detector_app,
                [
                    "export",
                    "test-detector",
                    "--output",
                    str(output_file),
                    "--format",
                    "json",
                ],
            )

            assert result.exit_code == 0
            assert "Detector exported successfully" in result.stdout
            mock_detector_service.export_detector.assert_called_once()

    def test_detector_export_not_found(self, runner, mock_detector_service):
        """Test detector export with non-existent detector."""
        mock_detector_service.export_detector.side_effect = DetectorError(
            "Detector not found"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "detector.pkl"

            result = runner.invoke(
                detector_app, ["export", "non-existent", "--output", str(output_file)]
            )

            assert result.exit_code == 1
            assert "Detector not found" in result.stdout

    # Detector Import Command Tests

    def test_detector_import_basic(self, runner, mock_detector_service):
        """Test basic detector import command."""
        mock_detector = Mock()
        mock_detector.id = "imported-detector"
        mock_detector_service.import_detector.return_value = mock_detector

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"dummy model data")
            model_file = f.name

        try:
            result = runner.invoke(
                detector_app, ["import", model_file, "--name", "imported-detector"]
            )

            assert result.exit_code == 0
            assert "Detector imported successfully" in result.stdout
            mock_detector_service.import_detector.assert_called_once()
        finally:
            Path(model_file).unlink(missing_ok=True)

    def test_detector_import_with_auto_name(self, runner, mock_detector_service):
        """Test detector import with auto-generated name."""
        mock_detector = Mock()
        mock_detector.id = "auto-generated-name"
        mock_detector_service.import_detector.return_value = mock_detector

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"dummy model data")
            model_file = f.name

        try:
            result = runner.invoke(detector_app, ["import", model_file])

            assert result.exit_code == 0
            assert "Detector imported successfully" in result.stdout
            mock_detector_service.import_detector.assert_called_once()
        finally:
            Path(model_file).unlink(missing_ok=True)

    # Error Handling Tests

    def test_detector_service_error_handling(self, runner, mock_detector_service):
        """Test error handling for detector service errors."""
        mock_detector_service.list_detectors.side_effect = ValidationError(
            "Service error"
        )

        result = runner.invoke(detector_app, ["list"])

        assert result.exit_code == 1
        assert "Service error" in result.stdout

    def test_detector_validation_error_handling(self, runner, mock_detector_service):
        """Test error handling for validation errors."""
        result = runner.invoke(detector_app, ["create", ""])

        assert result.exit_code == 1
        assert "Detector name cannot be empty" in result.stdout

    def test_detector_keyboard_interrupt(self, runner, mock_detector_service):
        """Test keyboard interrupt handling."""
        mock_detector_service.list_detectors.side_effect = KeyboardInterrupt()

        result = runner.invoke(detector_app, ["list"])

        assert result.exit_code == 1
        assert "Operation cancelled by user" in result.stdout

    # Integration Tests

    def test_detector_complete_workflow(
        self,
        runner,
        mock_detector_service,
        mock_training_service,
        mock_evaluation_service,
    ):
        """Test complete detector workflow."""
        # Create detector
        mock_detector = Mock()
        mock_detector.id = "workflow-detector"
        mock_detector_service.create_detector.return_value = mock_detector

        create_result = runner.invoke(
            detector_app,
            [
                "create",
                "workflow-detector",
                "--algorithm",
                "IsolationForest",
                "--description",
                "Workflow test detector",
            ],
        )
        assert create_result.exit_code == 0

        # Train detector
        mock_training_result = {
            "detector_id": "workflow-detector",
            "training_time": 30.5,
            "metrics": {"precision": 0.85, "recall": 0.78, "f1": 0.81},
        }
        mock_training_service.train_detector.return_value = mock_training_result

        train_result = runner.invoke(
            detector_app,
            ["train", "workflow-detector", "--dataset", "training-dataset"],
        )
        assert train_result.exit_code == 0

        # Evaluate detector
        mock_evaluation_result = {
            "detector_id": "workflow-detector",
            "metrics": {"precision": 0.85, "recall": 0.78, "f1": 0.81},
        }
        mock_evaluation_service.evaluate_detector.return_value = mock_evaluation_result

        evaluate_result = runner.invoke(
            detector_app, ["evaluate", "workflow-detector", "--dataset", "test-dataset"]
        )
        assert evaluate_result.exit_code == 0

        # Make predictions
        mock_predictions = {
            "predictions": [{"index": 0, "score": 0.1, "is_anomaly": False}],
            "summary": {"total_samples": 1, "anomalies_detected": 0},
        }
        mock_detector_service.predict.return_value = mock_predictions

        predict_result = runner.invoke(
            detector_app,
            ["predict", "workflow-detector", "--dataset", "prediction-dataset"],
        )
        assert predict_result.exit_code == 0

        # Delete detector
        mock_detector_service.delete_detector.return_value = None

        delete_result = runner.invoke(
            detector_app, ["delete", "workflow-detector", "--force"]
        )
        assert delete_result.exit_code == 0

    def test_detector_help_commands(self, runner):
        """Test help commands for detector."""
        result = runner.invoke(detector_app, ["--help"])
        assert result.exit_code == 0
        assert "Commands:" in result.stdout
        assert "list" in result.stdout
        assert "create" in result.stdout
        assert "train" in result.stdout
        assert "evaluate" in result.stdout
        assert "predict" in result.stdout
        assert "delete" in result.stdout

        # Test subcommand help
        result = runner.invoke(detector_app, ["create", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--algorithm" in result.stdout
        assert "--description" in result.stdout

        result = runner.invoke(detector_app, ["train", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "--dataset" in result.stdout
        assert "--contamination" in result.stdout
