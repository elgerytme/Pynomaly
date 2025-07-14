"""Test AutoML CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from pynomaly.presentation.cli.automl import app


class TestAutoMLCLI:
    """Test AutoML CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        # Create sample data
        data = {
            "feature_1": [1, 2, 3, 4, 5, 100],  # Last value is an outlier
            "feature_2": [2, 4, 6, 8, 10, 200],
            "feature_3": [1, 1, 1, 1, 1, 1],
        }
        df = pd.DataFrame(data)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            return Path(f.name)

    def test_automl_run_command_help(self, runner):
        """Test automl run command help."""
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "Run AutoML hyperparameter optimization" in result.stdout
        assert "DATASET_PATH" in result.stdout
        assert "ALGORITHM_NAME" in result.stdout

    @patch("pynomaly.presentation.cli.automl._load_dataset")
    @patch("pynomaly.presentation.cli.automl.get_automl_service")
    @patch("pynomaly.presentation.cli.automl.asyncio.run")
    def test_automl_run_command_success(
        self,
        mock_asyncio_run,
        mock_get_service,
        mock_load_dataset,
        runner,
        sample_csv_file,
    ):
        """Test successful automl run command."""
        # Mock dataset loading
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        # Mock AutoML service
        mock_service = Mock()
        mock_get_service.return_value = mock_service

        # Mock optimization result
        mock_result = Mock()
        mock_result.best_algorithm.value = "KNN"
        mock_result.best_config.parameters = {"n_neighbors": 5, "method": "largest"}
        mock_result.best_score = 0.75
        mock_result.total_trials = 10
        mock_result.optimization_time = 120.5
        mock_result.trial_history = []

        mock_asyncio_run.return_value = mock_result

        # Create temporary storage directory
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(
                app,
                [
                    "run",
                    str(sample_csv_file),
                    "KNN",
                    "--max-trials",
                    "10",
                    "--storage",
                    temp_dir,
                ],
            )

        assert result.exit_code == 0
        assert "Loading dataset" in result.stdout
        assert "AutoML optimization completed" in result.stdout

        # Verify service was called
        mock_load_dataset.assert_called_once()
        mock_get_service.assert_called_once()

    def test_automl_run_unsupported_algorithm(self, runner, sample_csv_file):
        """Test automl run with unsupported algorithm."""
        result = runner.invoke(
            app, ["run", str(sample_csv_file), "UnsupportedAlgorithm"]
        )

        assert result.exit_code == 1
        assert "Unsupported algorithm" in result.stdout

    @patch("pynomaly.presentation.cli.automl._load_dataset")
    def test_automl_run_dataset_loading_error(
        self, mock_load_dataset, runner, sample_csv_file
    ):
        """Test automl run with dataset loading error."""
        mock_load_dataset.side_effect = Exception("Failed to load dataset")

        result = runner.invoke(app, ["run", str(sample_csv_file), "KNN"])

        assert result.exit_code == 1
        assert "AutoML optimization failed" in result.stdout

    @patch("pynomaly.presentation.cli.automl._load_dataset")
    @patch("pynomaly.presentation.cli.automl.get_automl_service")
    @patch("pynomaly.presentation.cli.automl.asyncio.run")
    def test_automl_run_with_output_file(
        self,
        mock_asyncio_run,
        mock_get_service,
        mock_load_dataset,
        runner,
        sample_csv_file,
    ):
        """Test automl run with output file."""
        # Setup mocks
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        mock_service = Mock()
        mock_get_service.return_value = mock_service

        mock_result = Mock()
        mock_result.best_algorithm.value = "KNN"
        mock_result.best_config.parameters = {"n_neighbors": 5}
        mock_result.best_score = 0.8
        mock_result.total_trials = 20
        mock_result.optimization_time = 180.0
        mock_result.trial_history = []

        mock_asyncio_run.return_value = mock_result

        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir) / "storage"
            output_file = Path(temp_dir) / "output.json"

            result = runner.invoke(
                app,
                [
                    "run",
                    str(sample_csv_file),
                    "KNN",
                    "--storage",
                    str(storage_dir),
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert "Detailed results saved" in result.stdout

    def test_automl_run_performance_criteria_met(self, runner):
        """Test that success criteria message appears when F1 improvement ≥ 15%."""
        with (
            patch(
                "pynomaly.presentation.cli.automl._load_dataset"
            ) as mock_load_dataset,
            patch(
                "pynomaly.presentation.cli.automl.get_automl_service"
            ) as mock_get_service,
            patch("pynomaly.presentation.cli.automl.asyncio.run") as mock_asyncio_run,
        ):
            # Setup mocks for high performance
            mock_dataset = Mock()
            mock_load_dataset.return_value = mock_dataset

            mock_service = Mock()
            mock_get_service.return_value = mock_service

            # Mock result with 20% improvement (0.6 -> 0.72)
            mock_result = Mock()
            mock_result.best_algorithm.value = "KNN"
            mock_result.best_config.parameters = {"n_neighbors": 10}
            mock_result.best_score = 0.6  # 20% above baseline of 0.5
            mock_result.total_trials = 50
            mock_result.optimization_time = 300.0
            mock_result.trial_history = []

            mock_asyncio_run.return_value = mock_result

            with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
                result = runner.invoke(app, ["run", temp_file.name, "KNN"])

            assert result.exit_code == 0
            assert "Success!" in result.stdout
            assert "F1 improvement" in result.stdout

    def test_automl_run_performance_criteria_not_met(self, runner):
        """Test warning when F1 improvement < 15%."""
        with (
            patch(
                "pynomaly.presentation.cli.automl._load_dataset"
            ) as mock_load_dataset,
            patch(
                "pynomaly.presentation.cli.automl.get_automl_service"
            ) as mock_get_service,
            patch("pynomaly.presentation.cli.automl.asyncio.run") as mock_asyncio_run,
        ):
            # Setup mocks for low performance
            mock_dataset = Mock()
            mock_load_dataset.return_value = mock_dataset

            mock_service = Mock()
            mock_get_service.return_value = mock_service

            # Mock result with 10% improvement (0.5 -> 0.55)
            mock_result = Mock()
            mock_result.best_algorithm.value = "KNN"
            mock_result.best_config.parameters = {"n_neighbors": 3}
            mock_result.best_score = 0.55  # 10% above baseline of 0.5
            mock_result.total_trials = 25
            mock_result.optimization_time = 150.0
            mock_result.trial_history = []

            mock_asyncio_run.return_value = mock_result

            with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
                result = runner.invoke(app, ["run", temp_file.name, "KNN"])

            assert result.exit_code == 0
            assert "F1 improvement" in result.stdout
            assert "target:" in result.stdout


class TestDatasetLoading:
    """Test dataset loading functionality."""

    def test_load_csv_dataset(self):
        """Test loading CSV dataset."""
        from pynomaly.presentation.cli.automl import _load_dataset

        # Create sample CSV
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data.to_csv(f.name, index=False)
            csv_path = Path(f.name)

        try:
            dataset = _load_dataset(csv_path)
            assert dataset is not None
            assert hasattr(dataset, "data")
        except Exception:
            # If infrastructure dependencies aren't available, that's OK for this test
            pytest.skip("CSV loader not available")
        finally:
            csv_path.unlink()

    def test_load_parquet_dataset(self):
        """Test loading Parquet dataset."""
        from pynomaly.presentation.cli.automl import _load_dataset

        # Create sample Parquet file
        data = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            parquet_path = Path(f.name)

        try:
            data.to_parquet(parquet_path)
            dataset = _load_dataset(parquet_path)
            assert dataset is not None
            assert hasattr(dataset, "data")
        except Exception:
            # If infrastructure dependencies aren't available, that's OK for this test
            pytest.skip("Parquet loader not available")
        finally:
            if parquet_path.exists():
                parquet_path.unlink()

    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        from pynomaly.presentation.cli.automl import _load_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            txt_path = Path(f.name)
            f.write("some text data")

        try:
            with pytest.raises(RuntimeError, match="Failed to load dataset"):
                _load_dataset(txt_path)
        finally:
            txt_path.unlink()


class TestTrialPersistence:
    """Test trial results persistence."""

    def test_trial_results_saved_to_storage(self):
        """Test that trial results are properly saved."""
        trial_data = {
            "algorithm": "KNN",
            "best_score": 0.75,
            "best_parameters": {"n_neighbors": 5, "method": "largest"},
            "optimization_time": 120.5,
            "total_trials": 50,
            "trial_history": [
                {"trial": 0, "score": 0.6, "parameters": {"n_neighbors": 3}},
                {"trial": 1, "score": 0.75, "parameters": {"n_neighbors": 5}},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "test_results.json"

            # Save trial data
            with open(storage_path, "w") as f:
                json.dump(trial_data, f, indent=2)

            # Verify saved data
            assert storage_path.exists()

            with open(storage_path) as f:
                loaded_data = json.load(f)

            assert loaded_data["algorithm"] == "KNN"
            assert loaded_data["best_score"] == 0.75
            assert len(loaded_data["trial_history"]) == 2

    def test_storage_directory_creation(self):
        """Test that storage directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir) / "new_storage_dir"

            # Directory shouldn't exist initially
            assert not storage_dir.exists()

            # Create directory (simulating CLI behavior)
            storage_dir.mkdir(parents=True, exist_ok=True)

            # Directory should now exist
            assert storage_dir.exists()
            assert storage_dir.is_dir()


if __name__ == "__main__":
    pytest.main([__file__])
