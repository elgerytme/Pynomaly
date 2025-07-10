"""
Comprehensive CLI Autonomous Command Tests
Tests for the autonomous command - automated anomaly detection workflows.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.presentation.cli.app import app
from typer.testing import CliRunner


class TestAutonomousCommand:
    """Test suite for autonomous command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test data files
        self.test_data_file = self.temp_dir / "test_data.csv"
        self.test_data_file.write_text(
            "feature1,feature2,feature3\n"
            "1.0,2.0,0.5\n"
            "2.0,3.0,1.0\n"
            "3.0,4.0,1.5\n"
            "100.0,200.0,50.0\n"  # Obvious outlier
            "4.0,5.0,2.0\n"
        )

        # Create larger test dataset
        self.large_dataset_file = self.temp_dir / "large_data.csv"
        with open(self.large_dataset_file, "w") as f:
            f.write("feature1,feature2,feature3\n")
            for i in range(1000):
                f.write(f"{i},{i*2},{i*0.5}\n")
            f.write("1000,2000,500\n")  # Outlier

        # Create autonomous config
        self.autonomous_config = {
            "data_profiling": {
                "enable_profiling": True,
                "statistical_analysis": True,
                "correlation_analysis": True,
            },
            "algorithm_selection": {
                "strategy": "auto",
                "evaluation_metrics": ["f1_score", "precision", "recall"],
                "cross_validation": True,
            },
            "preprocessing": {
                "auto_scaling": True,
                "handle_missing": True,
                "feature_selection": True,
            },
            "output": {
                "generate_report": True,
                "include_visualizations": True,
                "export_results": True,
            },
        }

        self.config_file = self.temp_dir / "autonomous_config.json"
        self.config_file.write_text(json.dumps(self.autonomous_config))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # Basic Command Tests

    def test_autonomous_help_command(self):
        """Test autonomous command help output."""
        result = self.runner.invoke(app, ["auto", "--help"])

        assert result.exit_code == 0
        assert "autonomous" in result.stdout.lower() or "auto" in result.stdout.lower()
        assert "detect" in result.stdout.lower()
        assert "profile" in result.stdout.lower() or "quick" in result.stdout.lower()

    def test_autonomous_detect_help(self):
        """Test autonomous detect subcommand help."""
        result = self.runner.invoke(app, ["auto", "detect", "--help"])

        assert result.exit_code == 0
        assert "detect" in result.stdout.lower()
        assert "dataset" in result.stdout.lower()
        assert (
            "automatic" in result.stdout.lower()
            or "autonomous" in result.stdout.lower()
        )

    def test_autonomous_profile_help(self):
        """Test autonomous profile subcommand help."""
        result = self.runner.invoke(app, ["auto", "profile", "--help"])

        assert result.exit_code == 0
        assert "profile" in result.stdout.lower()
        assert "data" in result.stdout.lower()
        assert (
            "analysis" in result.stdout.lower() or "profiling" in result.stdout.lower()
        )

    def test_autonomous_quick_help(self):
        """Test autonomous quick subcommand help."""
        result = self.runner.invoke(app, ["auto", "quick", "--help"])

        assert result.exit_code == 0
        assert "quick" in result.stdout.lower()
        assert "minimal configuration" in result.stdout.lower()

    # Autonomous Detection Tests

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_basic(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test basic autonomous detection."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.name = "test_dataset"
        mock_dataset.n_samples = 5
        mock_dataset.n_features = 3
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_result.scores = [AnomalyScore(0.3), AnomalyScore(0.9), AnomalyScore(0.2)]
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        # Execute command
        result = self.runner.invoke(
            app, ["auto", "detect", "--dataset", str(self.test_data_file)]
        )

        assert result.exit_code == 0
        mock_dataset_service.load_dataset.assert_called_once()
        mock_autonomous_service.autonomous_detect.assert_called_once()

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_with_config(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous detection with configuration file."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        # Execute command with config
        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--config",
                str(self.config_file),
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.autonomous_detect.assert_called_once()

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_with_output_dir(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous detection with output directory."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        output_dir = self.temp_dir / "output"
        output_dir.mkdir()

        # Execute command with output directory
        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.autonomous_detect.assert_called_once()

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_with_algorithm_strategy(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous detection with specific algorithm strategy."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        # Execute command with algorithm strategy
        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--algorithm-strategy",
                "ensemble",
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.autonomous_detect.assert_called_once()

    # Data Profiling Tests

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_profile_basic(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test basic data profiling."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.name = "test_dataset"
        mock_dataset.n_samples = 5
        mock_dataset.n_features = 3
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_profile = {
            "dataset_info": {"samples": 5, "features": 3, "missing_values": 0},
            "statistical_summary": {"mean": [2.0, 4.0, 1.0], "std": [1.0, 2.0, 0.5]},
            "anomaly_indicators": {"outlier_candidates": 1, "suspicious_patterns": []},
        }
        mock_autonomous_service.profile_dataset.return_value = mock_profile

        # Execute command
        result = self.runner.invoke(
            app, ["auto", "profile", "--dataset", str(self.test_data_file)]
        )

        assert result.exit_code == 0
        mock_dataset_service.load_dataset.assert_called_once()
        mock_autonomous_service.profile_dataset.assert_called_once()

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_profile_with_detailed_analysis(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test data profiling with detailed analysis."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_profile = {
            "dataset_info": {"samples": 5, "features": 3},
            "detailed_analysis": {
                "correlation_matrix": [
                    [1.0, 0.8, 0.6],
                    [0.8, 1.0, 0.7],
                    [0.6, 0.7, 1.0],
                ],
                "feature_importance": [0.3, 0.5, 0.2],
                "anomaly_scores": [0.1, 0.2, 0.15, 0.9, 0.12],
            },
        }
        mock_autonomous_service.profile_dataset.return_value = mock_profile

        # Execute command with detailed analysis
        result = self.runner.invoke(
            app,
            ["auto", "profile", "--dataset", str(self.test_data_file), "--detailed"],
        )

        assert result.exit_code == 0
        mock_autonomous_service.profile_dataset.assert_called_once()

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_profile_with_visualizations(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test data profiling with visualizations."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_profile = {
            "dataset_info": {"samples": 5, "features": 3},
            "visualizations": {
                "generated_plots": [
                    "distribution.png",
                    "correlation.png",
                    "outliers.png",
                ],
                "plot_directory": str(self.temp_dir / "plots"),
            },
        }
        mock_autonomous_service.profile_dataset.return_value = mock_profile

        # Execute command with visualizations
        result = self.runner.invoke(
            app,
            [
                "auto",
                "profile",
                "--dataset",
                str(self.test_data_file),
                "--visualizations",
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.profile_dataset.assert_called_once()

    # Quick Detection Tests

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_quick_basic(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test basic quick detection."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.name = "test_dataset"
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_result.scores = [AnomalyScore(0.3), AnomalyScore(0.9)]
        mock_autonomous_service.quick_detect.return_value = mock_result

        # Execute command
        result = self.runner.invoke(
            app, ["auto", "quick", "--dataset", str(self.test_data_file)]
        )

        assert result.exit_code == 0
        mock_dataset_service.load_dataset.assert_called_once()
        mock_autonomous_service.quick_detect.assert_called_once()

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_quick_with_threshold(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test quick detection with custom threshold."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.quick_detect.return_value = mock_result

        # Execute command with custom threshold
        result = self.runner.invoke(
            app,
            [
                "auto",
                "quick",
                "--dataset",
                str(self.test_data_file),
                "--threshold",
                "0.8",
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.quick_detect.assert_called_once()

    # Error Handling Tests

    def test_autonomous_detect_missing_dataset(self):
        """Test autonomous detection with missing dataset."""
        result = self.runner.invoke(
            app, ["auto", "detect", "--dataset", "/nonexistent/file.csv"]
        )

        assert result.exit_code != 0
        assert "file" in result.stdout.lower() or "not found" in result.stdout.lower()

    def test_autonomous_detect_invalid_config(self):
        """Test autonomous detection with invalid configuration."""
        invalid_config = self.temp_dir / "invalid_config.json"
        invalid_config.write_text("{invalid json")

        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--config",
                str(invalid_config),
            ],
        )

        assert result.exit_code != 0
        assert "config" in result.stdout.lower() or "json" in result.stdout.lower()

    def test_autonomous_detect_invalid_algorithm_strategy(self):
        """Test autonomous detection with invalid algorithm strategy."""
        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--algorithm-strategy",
                "invalid_strategy",
            ],
        )

        assert result.exit_code != 0
        assert "strategy" in result.stdout.lower() or "invalid" in result.stdout.lower()

    # Performance Tests

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_large_dataset(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous detection with large dataset."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.name = "large_dataset"
        mock_dataset.n_samples = 1000
        mock_dataset.n_features = 3
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        # Execute command with large dataset
        result = self.runner.invoke(
            app, ["auto", "detect", "--dataset", str(self.large_dataset_file)]
        )

        assert result.exit_code == 0
        mock_autonomous_service.autonomous_detect.assert_called_once()

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_with_timeout(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous detection with timeout."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        # Execute command with timeout
        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--timeout",
                "60",
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.autonomous_detect.assert_called_once()

    # Integration Tests

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_profile_then_detect(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test profile then detect workflow."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        # Mock profile result
        mock_profile = {
            "dataset_info": {"samples": 5, "features": 3},
            "recommendations": {
                "suggested_algorithms": ["IsolationForest", "LocalOutlierFactor"],
                "contamination_estimate": 0.2,
            },
        }
        mock_autonomous_service.profile_dataset.return_value = mock_profile

        # Profile first
        profile_result = self.runner.invoke(
            app, ["auto", "profile", "--dataset", str(self.test_data_file)]
        )

        assert profile_result.exit_code == 0

        # Mock detection result
        mock_detection_result = Mock(spec=DetectionResult)
        mock_detection_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_detection_result

        # Then detect
        detect_result = self.runner.invoke(
            app, ["auto", "detect", "--dataset", str(self.test_data_file)]
        )

        assert detect_result.exit_code == 0

        # Verify both operations were called
        mock_autonomous_service.profile_dataset.assert_called_once()
        mock_autonomous_service.autonomous_detect.assert_called_once()

    # Output Format Tests

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_json_output(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous detection with JSON output."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        # Execute command with JSON output
        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--output-format",
                "json",
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.autonomous_detect.assert_called_once()

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_profile_html_report(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous profiling with HTML report."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_profile = {
            "dataset_info": {"samples": 5, "features": 3},
            "html_report": {
                "generated": True,
                "path": str(self.temp_dir / "profile_report.html"),
            },
        }
        mock_autonomous_service.profile_dataset.return_value = mock_profile

        # Execute command with HTML report
        result = self.runner.invoke(
            app,
            [
                "auto",
                "profile",
                "--dataset",
                str(self.test_data_file),
                "--format",
                "html",
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.profile_dataset.assert_called_once()

    # Advanced Configuration Tests

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_custom_preprocessing(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous detection with custom preprocessing."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        # Execute command with custom preprocessing
        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--preprocessing",
                "robust",
                "--scaling",
                "minmax",
                "--feature-selection",
                "variance",
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.autonomous_detect.assert_called_once()

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_ensemble_mode(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous detection with ensemble mode."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        # Execute command with ensemble mode
        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--ensemble-mode",
                "--algorithms",
                "IsolationForest,LocalOutlierFactor,OneClassSVM",
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.autonomous_detect.assert_called_once()

    # Edge Cases

    def test_autonomous_detect_empty_dataset(self):
        """Test autonomous detection with empty dataset."""
        empty_file = self.temp_dir / "empty.csv"
        empty_file.write_text("feature1,feature2\n")

        result = self.runner.invoke(
            app, ["auto", "detect", "--dataset", str(empty_file)]
        )

        assert result.exit_code != 0
        assert "empty" in result.stdout.lower() or "no data" in result.stdout.lower()

    def test_autonomous_detect_single_feature(self):
        """Test autonomous detection with single feature dataset."""
        single_feature_file = self.temp_dir / "single_feature.csv"
        single_feature_file.write_text("feature1\n1.0\n2.0\n3.0\n100.0\n4.0\n")

        result = self.runner.invoke(
            app, ["auto", "detect", "--dataset", str(single_feature_file)]
        )

        # Should handle single feature appropriately
        assert result.exit_code in [0, 1]

    def test_autonomous_detect_all_identical_values(self):
        """Test autonomous detection with all identical values."""
        identical_file = self.temp_dir / "identical.csv"
        identical_file.write_text("feature1,feature2\n1.0,1.0\n1.0,1.0\n1.0,1.0\n")

        result = self.runner.invoke(
            app, ["auto", "detect", "--dataset", str(identical_file)]
        )

        # Should handle constant values appropriately
        assert result.exit_code in [0, 1]

    def test_autonomous_detect_with_missing_values(self):
        """Test autonomous detection with missing values."""
        missing_values_file = self.temp_dir / "missing_values.csv"
        missing_values_file.write_text(
            "feature1,feature2\n1.0,2.0\n,3.0\n3.0,\n4.0,5.0\n"
        )

        result = self.runner.invoke(
            app, ["auto", "detect", "--dataset", str(missing_values_file)]
        )

        # Should handle missing values appropriately
        assert result.exit_code in [0, 1]

    # Resource Management Tests

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_memory_limit(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous detection with memory limit."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        # Execute command with memory limit
        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--memory-limit",
                "1GB",
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.autonomous_detect.assert_called_once()

    @patch("pynomaly.presentation.cli.commands.autonomous.autonomous_service")
    @patch("pynomaly.presentation.cli.commands.autonomous.dataset_service")
    def test_autonomous_detect_parallel_processing(
        self, mock_dataset_service, mock_autonomous_service
    ):
        """Test autonomous detection with parallel processing."""
        # Mock services
        mock_dataset = Mock(spec=Dataset)
        mock_dataset_service.load_dataset.return_value = mock_dataset

        mock_result = Mock(spec=DetectionResult)
        mock_result.anomalies = []
        mock_autonomous_service.autonomous_detect.return_value = mock_result

        # Execute command with parallel processing
        result = self.runner.invoke(
            app,
            [
                "auto",
                "detect",
                "--dataset",
                str(self.test_data_file),
                "--parallel",
                "--n-jobs",
                "4",
            ],
        )

        assert result.exit_code == 0
        mock_autonomous_service.autonomous_detect.assert_called_once()
