"""
Comprehensive CLI workflow integration tests.
Tests complete end-to-end workflows across multiple CLI commands.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from pynomaly.presentation.cli.app import app


class TestCLIWorkflows:
    """Test suite for CLI workflow integration."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_services(self):
        """Mock all CLI services."""
        with (
            patch(
                "pynomaly.presentation.cli.commands.datasets.dataset_service"
            ) as mock_dataset,
            patch(
                "pynomaly.presentation.cli.commands.detector.detector_service"
            ) as mock_detector,
            patch(
                "pynomaly.presentation.cli.commands.detector.training_service"
            ) as mock_training,
            patch(
                "pynomaly.presentation.cli.commands.detector.evaluation_service"
            ) as mock_evaluation,
            patch("pynomaly.presentation.cli.export.export_service") as mock_export,
            patch(
                "pynomaly.presentation.cli.commands.detect.detection_service"
            ) as mock_detection,
            patch(
                "pynomaly.presentation.cli.commands.autonomous.autonomous_service"
            ) as mock_autonomous,
        ):
            services = {
                "dataset": mock_dataset,
                "detector": mock_detector,
                "training": mock_training,
                "evaluation": mock_evaluation,
                "export": mock_export,
                "detection": mock_detection,
                "autonomous": mock_autonomous,
            }
            yield services

    @pytest.fixture
    def sample_csv_file(self):
        """Create temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("feature1,feature2,target\n")
            f.write("1.0,2.0,0\n")
            f.write("2.0,3.0,1\n")
            f.write("3.0,4.0,0\n")
            f.write("15.0,20.0,1\n")  # Potential anomaly
            f.write("5.0,6.0,0\n")
            yield f.name
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    # Complete Detection Workflow Tests

    def test_complete_detection_workflow(self, runner, mock_services, sample_csv_file):
        """Test complete detection workflow: dataset -> detector -> train -> detect."""
        # Step 1: Create dataset
        mock_dataset = Mock()
        mock_dataset.id = "workflow-dataset"
        mock_dataset.to_dict.return_value = {
            "id": "workflow-dataset",
            "name": "Workflow Dataset",
            "rows": 5,
            "columns": 3,
        }
        mock_services["dataset"].create_dataset.return_value = mock_dataset

        dataset_result = runner.invoke(
            app,
            [
                "dataset",
                "create",
                "workflow-dataset",
                "--file",
                sample_csv_file,
                "--description",
                "Test workflow dataset",
            ],
        )
        assert dataset_result.exit_code == 0
        assert "Dataset created successfully" in dataset_result.stdout

        # Step 2: Create detector
        mock_detector = Mock()
        mock_detector.id = "workflow-detector"
        mock_detector.to_dict.return_value = {
            "id": "workflow-detector",
            "name": "Workflow Detector",
            "algorithm": "IsolationForest",
        }
        mock_services["detector"].create_detector.return_value = mock_detector

        detector_result = runner.invoke(
            app,
            [
                "detector",
                "create",
                "workflow-detector",
                "--algorithm",
                "IsolationForest",
                "--description",
                "Test workflow detector",
            ],
        )
        assert detector_result.exit_code == 0
        assert "Detector created successfully" in detector_result.stdout

        # Step 3: Train detector
        mock_training_result = {
            "detector_id": "workflow-detector",
            "training_time": 30.5,
            "metrics": {"precision": 0.85, "recall": 0.78, "f1": 0.81},
            "status": "completed",
        }
        mock_services["training"].train_detector.return_value = mock_training_result

        train_result = runner.invoke(
            app,
            ["detector", "train", "workflow-detector", "--dataset", "workflow-dataset"],
        )
        assert train_result.exit_code == 0
        assert "Training completed successfully" in train_result.stdout

        # Step 4: Run detection
        mock_detection_result = {
            "detector_id": "workflow-detector",
            "dataset_id": "workflow-dataset",
            "results": [
                {"index": 0, "score": 0.1, "is_anomaly": False},
                {"index": 1, "score": 0.2, "is_anomaly": False},
                {"index": 2, "score": 0.3, "is_anomaly": False},
                {"index": 3, "score": 0.9, "is_anomaly": True},
                {"index": 4, "score": 0.15, "is_anomaly": False},
            ],
            "summary": {"total_samples": 5, "anomalies_detected": 1},
        }
        mock_services["detection"].run_detection.return_value = mock_detection_result

        detect_result = runner.invoke(
            app, ["detect", "run", "workflow-detector", "--dataset", "workflow-dataset"]
        )
        assert detect_result.exit_code == 0
        assert "Detection completed" in detect_result.stdout

        # Verify all services were called
        mock_services["dataset"].create_dataset.assert_called_once()
        mock_services["detector"].create_detector.assert_called_once()
        mock_services["training"].train_detector.assert_called_once()
        mock_services["detection"].run_detection.assert_called_once()

    def test_autonomous_detection_workflow(
        self, runner, mock_services, sample_csv_file
    ):
        """Test autonomous detection workflow."""
        # Mock autonomous detection service
        mock_autonomous_result = {
            "selected_algorithm": "IsolationForest",
            "detector_id": "auto-detector-12345",
            "dataset_info": {"rows": 5, "columns": 3, "anomaly_rate": 0.2},
            "detection_results": {
                "anomalies_detected": 1,
                "confidence": 0.87,
                "results": [{"index": 3, "score": 0.9, "is_anomaly": True}],
            },
            "recommendations": [
                "Sample at index 3 shows anomalous behavior",
                "Consider investigating high feature values",
            ],
        }
        mock_services[
            "autonomous"
        ].run_autonomous_detection.return_value = mock_autonomous_result

        result = runner.invoke(
            app, ["auto", "detect", sample_csv_file, "--contamination", "0.2"]
        )

        assert result.exit_code == 0
        assert "Autonomous detection completed" in result.stdout
        assert "IsolationForest" in result.stdout
        assert "anomalies_detected: 1" in result.stdout
        mock_services["autonomous"].run_autonomous_detection.assert_called_once()

    def test_evaluation_and_export_workflow(
        self, runner, mock_services, sample_csv_file
    ):
        """Test evaluation and export workflow."""
        # Setup: Assume we have a trained detector
        mock_detector = Mock()
        mock_detector.id = "eval-detector"
        mock_detector.to_dict.return_value = {
            "id": "eval-detector",
            "status": "trained",
            "algorithm": "IsolationForest",
        }
        mock_services["detector"].get_detector.return_value = mock_detector

        # Step 1: Evaluate detector
        mock_evaluation_result = {
            "detector_id": "eval-detector",
            "dataset_id": "eval-dataset",
            "metrics": {"precision": 0.85, "recall": 0.78, "f1": 0.81, "auc": 0.83},
            "confusion_matrix": [[95, 5], [3, 12]],
            "evaluation_time": 15.2,
        }
        mock_services[
            "evaluation"
        ].evaluate_detector.return_value = mock_evaluation_result

        eval_result = runner.invoke(
            app, ["detector", "evaluate", "eval-detector", "--dataset", "eval-dataset"]
        )
        assert eval_result.exit_code == 0
        assert "Evaluation completed successfully" in eval_result.stdout
        assert "precision" in eval_result.stdout

        # Step 2: Export results
        mock_services["export"].export_results.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "evaluation_results.csv"

            export_result = runner.invoke(
                app,
                [
                    "export",
                    "results",
                    "eval-detector",
                    "--dataset",
                    "eval-dataset",
                    "--output",
                    str(output_file),
                    "--format",
                    "csv",
                ],
            )
            assert export_result.exit_code == 0
            assert "Results exported successfully" in export_result.stdout

        # Step 3: Export model
        mock_services["export"].export_model.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "model.pkl"

            model_export_result = runner.invoke(
                app, ["export", "model", "eval-detector", "--output", str(model_file)]
            )
            assert model_export_result.exit_code == 0
            assert "Model exported successfully" in model_export_result.stdout

        # Verify all services were called
        mock_services["evaluation"].evaluate_detector.assert_called_once()
        mock_services["export"].export_results.assert_called_once()
        mock_services["export"].export_model.assert_called_once()

    def test_dataset_management_workflow(self, runner, mock_services, sample_csv_file):
        """Test dataset management workflow."""
        # Step 1: Import dataset
        mock_dataset = Mock()
        mock_dataset.id = "imported-dataset"
        mock_dataset.to_dict.return_value = {
            "id": "imported-dataset",
            "name": "Imported Dataset",
            "rows": 5,
            "columns": 3,
            "format": "csv",
        }
        mock_services["dataset"].import_dataset.return_value = mock_dataset

        import_result = runner.invoke(
            app, ["dataset", "import", sample_csv_file, "--name", "imported-dataset"]
        )
        assert import_result.exit_code == 0
        assert "Dataset imported successfully" in import_result.stdout

        # Step 2: Validate dataset
        mock_validation_result = {
            "valid": True,
            "errors": [],
            "warnings": ["Column 'target' has limited unique values"],
            "summary": "Dataset is valid with minor warnings",
        }
        mock_services["dataset"].validate_dataset.return_value = mock_validation_result

        validate_result = runner.invoke(
            app, ["dataset", "validate", "imported-dataset"]
        )
        assert validate_result.exit_code == 0
        assert "Dataset is valid" in validate_result.stdout

        # Step 3: Get statistics
        mock_stats = {
            "rows": 5,
            "columns": 3,
            "missing_values": 0,
            "duplicates": 0,
            "data_types": {"numeric": 2, "categorical": 1},
        }
        mock_services["dataset"].get_dataset_statistics.return_value = mock_stats

        stats_result = runner.invoke(app, ["dataset", "stats", "imported-dataset"])
        assert stats_result.exit_code == 0
        assert "5" in stats_result.stdout  # rows
        assert "3" in stats_result.stdout  # columns

        # Step 4: Export dataset
        mock_services["dataset"].export_dataset.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            export_file = Path(temp_dir) / "exported_dataset.json"

            export_result = runner.invoke(
                app,
                [
                    "dataset",
                    "export",
                    "imported-dataset",
                    "--output",
                    str(export_file),
                    "--format",
                    "json",
                ],
            )
            assert export_result.exit_code == 0
            assert "Dataset exported successfully" in export_result.stdout

        # Verify all services were called
        mock_services["dataset"].import_dataset.assert_called_once()
        mock_services["dataset"].validate_dataset.assert_called_once()
        mock_services["dataset"].get_dataset_statistics.assert_called_once()
        mock_services["dataset"].export_dataset.assert_called_once()

    def test_detector_lifecycle_workflow(self, runner, mock_services):
        """Test complete detector lifecycle workflow."""
        # Step 1: Create detector
        mock_detector = Mock()
        mock_detector.id = "lifecycle-detector"
        mock_detector.to_dict.return_value = {
            "id": "lifecycle-detector",
            "name": "Lifecycle Detector",
            "algorithm": "LOF",
            "status": "created",
        }
        mock_services["detector"].create_detector.return_value = mock_detector

        create_result = runner.invoke(
            app,
            [
                "detector",
                "create",
                "lifecycle-detector",
                "--algorithm",
                "LOF",
                "--hyperparameters",
                '{"n_neighbors": 20}',
            ],
        )
        assert create_result.exit_code == 0

        # Step 2: Update detector
        mock_detector.to_dict.return_value["status"] = "configured"
        mock_services["detector"].update_detector.return_value = mock_detector

        update_result = runner.invoke(
            app,
            [
                "detector",
                "update",
                "lifecycle-detector",
                "--description",
                "Updated lifecycle detector",
            ],
        )
        assert update_result.exit_code == 0

        # Step 3: Show detector details
        mock_services["detector"].get_detector.return_value = mock_detector

        show_result = runner.invoke(
            app, ["detector", "show", "lifecycle-detector", "--hyperparameters"]
        )
        assert show_result.exit_code == 0
        assert "lifecycle-detector" in show_result.stdout

        # Step 4: List detectors
        mock_services["detector"].list_detectors.return_value = [mock_detector]

        list_result = runner.invoke(app, ["detector", "list", "--algorithm", "LOF"])
        assert list_result.exit_code == 0
        assert "lifecycle-detector" in list_result.stdout

        # Step 5: Delete detector
        mock_services["detector"].delete_detector.return_value = None

        delete_result = runner.invoke(
            app, ["detector", "delete", "lifecycle-detector", "--force"]
        )
        assert delete_result.exit_code == 0
        assert "Detector deleted successfully" in delete_result.stdout

        # Verify all operations
        mock_services["detector"].create_detector.assert_called_once()
        mock_services["detector"].update_detector.assert_called_once()
        mock_services["detector"].get_detector.assert_called_once()
        mock_services["detector"].list_detectors.assert_called_once()
        mock_services["detector"].delete_detector.assert_called_once()

    def test_batch_processing_workflow(self, runner, mock_services, sample_csv_file):
        """Test batch processing workflow."""
        # Setup multiple datasets and detectors
        datasets = []
        detectors = []

        for i in range(3):
            # Create datasets
            dataset = Mock()
            dataset.id = f"batch-dataset-{i}"
            dataset.to_dict.return_value = {
                "id": f"batch-dataset-{i}",
                "name": f"Batch Dataset {i}",
                "rows": 100 + i * 10,
            }
            datasets.append(dataset)

            # Create detectors
            detector = Mock()
            detector.id = f"batch-detector-{i}"
            detector.to_dict.return_value = {
                "id": f"batch-detector-{i}",
                "name": f"Batch Detector {i}",
                "algorithm": ["IsolationForest", "LOF", "OneClassSVM"][i],
                "status": "trained",
            }
            detectors.append(detector)

        # Mock batch operations
        mock_services["dataset"].list_datasets.return_value = datasets
        mock_services["detector"].list_detectors.return_value = detectors

        # Test: List all datasets
        dataset_list_result = runner.invoke(app, ["dataset", "list"])
        assert dataset_list_result.exit_code == 0
        for i in range(3):
            assert f"batch-dataset-{i}" in dataset_list_result.stdout

        # Test: List all detectors
        detector_list_result = runner.invoke(app, ["detector", "list"])
        assert detector_list_result.exit_code == 0
        for i in range(3):
            assert f"batch-detector-{i}" in detector_list_result.stdout

        # Test: Batch export
        mock_services["export"].export_batch.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            batch_export_result = runner.invoke(
                app,
                [
                    "export",
                    "batch",
                    "batch-detector-0,batch-detector-1,batch-detector-2",
                    "--output-dir",
                    temp_dir,
                    "--format",
                    "json",
                ],
            )
            assert batch_export_result.exit_code == 0
            assert "Batch export completed successfully" in batch_export_result.stdout

        mock_services["export"].export_batch.assert_called_once()

    def test_error_handling_workflow(self, runner, mock_services):
        """Test error handling across workflow steps."""
        # Test: Dataset not found error propagation
        from pynomaly.domain.exceptions import DatasetError

        mock_services["dataset"].get_dataset.side_effect = DatasetError(
            "Dataset not found"
        )

        dataset_error_result = runner.invoke(
            app, ["dataset", "show", "non-existent-dataset"]
        )
        assert dataset_error_result.exit_code == 1
        assert "Dataset not found" in dataset_error_result.stdout

        # Test: Detector not found error propagation
        from pynomaly.domain.exceptions import DetectorError

        mock_services["detector"].get_detector.side_effect = DetectorError(
            "Detector not found"
        )

        detector_error_result = runner.invoke(
            app, ["detector", "show", "non-existent-detector"]
        )
        assert detector_error_result.exit_code == 1
        assert "Detector not found" in detector_error_result.stdout

        # Test: Training error handling
        mock_services["training"].train_detector.side_effect = Exception(
            "Training failed"
        )

        training_error_result = runner.invoke(
            app, ["detector", "train", "test-detector", "--dataset", "test-dataset"]
        )
        assert training_error_result.exit_code == 1

    def test_help_system_workflow(self, runner):
        """Test comprehensive help system."""
        # Test main help
        main_help_result = runner.invoke(app, ["--help"])
        assert main_help_result.exit_code == 0
        assert "Commands:" in main_help_result.stdout

        # Test command group help
        command_groups = ["dataset", "detector", "detect", "auto", "export"]
        for group in command_groups:
            group_help_result = runner.invoke(app, [group, "--help"])
            assert group_help_result.exit_code == 0
            assert "Usage:" in group_help_result.stdout

        # Test specific command help
        specific_commands = [
            ["dataset", "create", "--help"],
            ["detector", "train", "--help"],
            ["detect", "run", "--help"],
            ["export", "results", "--help"],
        ]

        for cmd in specific_commands:
            cmd_help_result = runner.invoke(app, cmd)
            assert cmd_help_result.exit_code == 0
            assert "Usage:" in cmd_help_result.stdout

    def test_configuration_workflow(self, runner, mock_services):
        """Test configuration and settings workflow."""
        # Test version command
        version_result = runner.invoke(app, ["version"])
        assert version_result.exit_code == 0
        assert "Pynomaly" in version_result.stdout

        # Test status command
        mock_services["detector"].count.return_value = 5
        mock_services["dataset"].count.return_value = 3

        with patch("pynomaly.presentation.cli.app.get_cli_container") as mock_container:
            container = Mock()
            container.detector_repository.return_value.count.return_value = 5
            container.dataset_repository.return_value.count.return_value = 3
            container.result_repository.return_value.count.return_value = 10
            container.result_repository.return_value.find_recent.return_value = []
            mock_container.return_value = container

            status_result = runner.invoke(app, ["status"])
            assert status_result.exit_code == 0
            assert "System Status" in status_result.stdout

    def test_performance_workflow(self, runner, mock_services, sample_csv_file):
        """Test performance-focused workflow."""
        # Test: Large dataset handling simulation
        mock_large_dataset = Mock()
        mock_large_dataset.id = "large-dataset"
        mock_large_dataset.to_dict.return_value = {
            "id": "large-dataset",
            "rows": 100000,
            "columns": 50,
            "size_mb": 100,
        }
        mock_services["dataset"].create_dataset.return_value = mock_large_dataset

        large_dataset_result = runner.invoke(
            app,
            [
                "dataset",
                "create",
                "large-dataset",
                "--file",
                sample_csv_file,
                "--description",
                "Large dataset for performance testing",
            ],
        )
        assert large_dataset_result.exit_code == 0

        # Test: Performance-optimized detector
        mock_fast_detector = Mock()
        mock_fast_detector.id = "fast-detector"
        mock_services["detector"].create_detector.return_value = mock_fast_detector

        fast_detector_result = runner.invoke(
            app,
            [
                "detector",
                "create",
                "fast-detector",
                "--algorithm",
                "IsolationForest",
                "--hyperparameters",
                '{"n_estimators": 50, "max_samples": 256}',
            ],
        )
        assert fast_detector_result.exit_code == 0

        # Test: Fast training
        mock_training_result = {
            "detector_id": "fast-detector",
            "training_time": 5.2,  # Fast training
            "metrics": {"precision": 0.80, "recall": 0.75},
        }
        mock_services["training"].train_detector.return_value = mock_training_result

        fast_training_result = runner.invoke(
            app, ["detector", "train", "fast-detector", "--dataset", "large-dataset"]
        )
        assert fast_training_result.exit_code == 0
        assert "5.2" in fast_training_result.stdout  # Training time

    def test_cross_command_state_workflow(self, runner, mock_services, sample_csv_file):
        """Test workflow that maintains state across commands."""
        # Create shared state through file system or service calls
        shared_state = {
            "last_dataset": None,
            "last_detector": None,
            "last_results": None,
        }

        def track_dataset_creation(dataset_data):
            dataset = Mock()
            dataset.id = dataset_data["name"]
            shared_state["last_dataset"] = dataset.id
            return dataset

        def track_detector_creation(detector_data):
            detector = Mock()
            detector.id = detector_data["name"]
            shared_state["last_detector"] = detector.id
            return detector

        mock_services["dataset"].create_dataset.side_effect = (
            lambda **kwargs: track_dataset_creation(kwargs)
        )
        mock_services["detector"].create_detector.side_effect = (
            lambda **kwargs: track_detector_creation(kwargs)
        )

        # Step 1: Create dataset (updates shared state)
        dataset_result = runner.invoke(
            app, ["dataset", "create", "state-dataset", "--file", sample_csv_file]
        )
        assert dataset_result.exit_code == 0
        assert shared_state["last_dataset"] == "state-dataset"

        # Step 2: Create detector (updates shared state)
        detector_result = runner.invoke(
            app,
            ["detector", "create", "state-detector", "--algorithm", "IsolationForest"],
        )
        assert detector_result.exit_code == 0
        assert shared_state["last_detector"] == "state-detector"

        # Verify state consistency
        assert shared_state["last_dataset"] is not None
        assert shared_state["last_detector"] is not None
