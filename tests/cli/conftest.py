"""
CLI-specific fixtures that extend the root conftest.py.
Import root conftest to prevent conflicts.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import all base fixtures from root conftest
from tests.conftest import *
from typer.testing import CliRunner


@pytest.fixture(scope="session")
def cli_runner():
    """Create a stable CLI runner for the session."""
    return CliRunner()


@pytest.fixture
def sample_dataset():
    """Create a sample dataset file for testing."""
    content = """feature1,feature2,feature3,target
1.0,2.0,3.0,0
2.0,3.0,4.0,0
100.0,200.0,300.0,1
3.0,4.0,5.0,0
4.0,5.0,6.0,0
5.0,6.0,7.0,0
150.0,250.0,350.0,1
6.0,7.0,8.0,0
7.0,8.0,9.0,0
8.0,9.0,10.0,0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_json_dataset():
    """Create a sample JSON dataset file."""
    import json

    data = {
        "data": [
            {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0, "target": 0},
            {"feature1": 2.0, "feature2": 3.0, "feature3": 4.0, "target": 0},
            {"feature1": 100.0, "feature2": 200.0, "feature3": 300.0, "target": 1},
            {"feature1": 3.0, "feature2": 4.0, "feature3": 5.0, "target": 0},
            {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0, "target": 0},
        ],
        "metadata": {
            "features": ["feature1", "feature2", "feature3"],
            "target": "target",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset object."""
    dataset = Mock()
    dataset.name = "test_dataset"
    dataset.shape = (10, 4)
    dataset.features = ["feature1", "feature2", "feature3", "target"]
    dataset.target_column = "target"
    dataset.created_at = "2024-01-01T00:00:00"
    dataset.data = Mock()
    dataset.data.shape = (10, 4)
    dataset.data.columns = ["feature1", "feature2", "feature3", "target"]
    return dataset


@pytest.fixture
def mock_detector():
    """Create a mock detector object."""
    detector = Mock()
    detector.name = "test_detector"
    detector.algorithm_name = "IsolationForest"
    detector.is_fitted = False
    detector.contamination_rate = Mock()
    detector.contamination_rate.value = 0.1
    detector.parameters = {"contamination": 0.1}
    detector.created_at = "2024-01-01T00:00:00"
    return detector


@pytest.fixture
def mock_detection_result():
    """Create a mock detection result."""
    result = Mock()
    result.detector_id = "test_detector_id"
    result.dataset_name = "test_dataset"
    result.anomalies = []
    result.scores = [0.1, 0.2, 0.9, 0.3, 0.4]  # Sample scores
    result.labels = [0, 0, 1, 0, 0]  # Sample labels
    result.threshold = 0.8
    result.execution_time_ms = 100.0
    result.metadata = {"algorithm": "IsolationForest"}
    return result


@pytest.fixture
def mock_container():
    """Create a comprehensive mock container."""
    container = Mock()

    # Mock repositories
    dataset_repo = Mock()
    detector_repo = Mock()
    result_repo = Mock()

    # Configure dataset repository
    dataset_repo.save.return_value = True
    dataset_repo.find_by_name.return_value = None
    dataset_repo.list_all.return_value = []
    dataset_repo.delete.return_value = True
    dataset_repo.exists.return_value = False

    # Configure detector repository
    detector_repo.save.return_value = True
    detector_repo.find_by_name.return_value = None
    detector_repo.list_all.return_value = []
    detector_repo.delete.return_value = True
    detector_repo.exists.return_value = False

    # Configure result repository
    result_repo.save.return_value = True
    result_repo.find_by_id.return_value = None
    result_repo.list_all.return_value = []
    result_repo.delete.return_value = True

    # Attach repositories
    container.dataset_repository.return_value = dataset_repo
    container.detector_repository.return_value = detector_repo
    container.result_repository.return_value = result_repo

    # Mock use cases
    container.train_detector_use_case.return_value = Mock()
    container.detect_anomalies_use_case.return_value = Mock()
    container.autonomous_detection_service.return_value = Mock()

    # Mock services
    container.data_loader_service.return_value = Mock()
    container.export_service.return_value = Mock()
    container.model_service.return_value = Mock()

    return container


@pytest.fixture
def mock_autonomous_service():
    """Create a mock autonomous detection service."""
    service = Mock()

    # Mock detect_autonomous method (the actual method used)
    service.detect_autonomous.return_value = {
        "autonomous_detection_results": {
            "success": True,
            "data_profile": {
                "samples": 100,
                "features": 3,
                "numeric_features": 3,
                "missing_ratio": 0.0,
                "complexity_score": 0.75,
                "recommended_contamination": 0.1,
            },
            "algorithm_recommendations": [
                {
                    "algorithm": "IsolationForest",
                    "confidence": 0.85,
                    "reasoning": "Good for high-dimensional data",
                },
                {
                    "algorithm": "LocalOutlierFactor",
                    "confidence": 0.75,
                    "reasoning": "Suitable for local outliers",
                },
            ],
            "detection_results": {
                "selected_algorithm": "IsolationForest",
                "anomalies_found": 2,
                "anomaly_indices": [2, 6],
                "anomaly_scores": [0.9, 0.95],
                "execution_time": 1.5,
                "hyperparameters": {"contamination": 0.1, "n_estimators": 100},
            },
            "preprocessing_results": {
                "applied_steps": ["StandardScaler"],
                "quality_score": 0.95,
                "processing_time": 0.5,
            },
            "metadata": {
                "total_time": 2.0,
                "dataset_name": "autonomous_data",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        }
    }

    # Mock detect_anomalies method (legacy compatibility)
    service.detect_anomalies.return_value = {
        "best_detector": "IsolationForest",
        "anomalies_found": 2,
        "confidence": 0.85,
        "anomaly_indices": [2, 6],
        "anomaly_scores": [0.9, 0.95],
        "algorithm_performance": {"IsolationForest": 0.85, "LocalOutlierFactor": 0.75},
        "preprocessing_applied": ["StandardScaler"],
        "execution_time": 1.5,
    }

    # Mock profile_dataset method
    service.profile_dataset.return_value = {
        "dataset_summary": {
            "rows": 10,
            "columns": 4,
            "numeric_features": 3,
            "categorical_features": 0,
            "missing_values": 0,
        },
        "recommended_algorithms": ["IsolationForest", "LocalOutlierFactor"],
        "contamination_estimate": 0.2,
        "feature_analysis": {
            "feature1": {"type": "numeric", "outliers": 1},
            "feature2": {"type": "numeric", "outliers": 1},
            "feature3": {"type": "numeric", "outliers": 1},
        },
    }

    return service


@pytest.fixture
def mock_export_service():
    """Create a mock export service."""
    service = Mock()

    # Mock export methods
    service.export_to_csv.return_value = True
    service.export_to_json.return_value = True
    service.export_to_excel.return_value = True
    service.export_to_pdf.return_value = True
    service.export_dashboard.return_value = True

    # Mock format listing
    service.get_available_formats.return_value = ["csv", "json", "excel", "pdf", "html"]

    return service


# Mock optional dependencies (non-autouse to prevent conflicts with root conftest)
@pytest.fixture
def mock_optional_dependencies():
    """Mock optional dependencies to prevent import errors - use only when needed."""
    # This fixture is available but not auto-used to prevent conflicts
    # Use explicitly in tests that need dependency mocking
    yield


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    import shutil
    import tempfile

    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_cli_dependencies():
    """Mock CLI-specific dependencies and services."""
    with patch(
        "pynomaly.presentation.cli.container.get_cli_container"
    ) as mock_get_container:
        # Create a comprehensive mock container
        container = Mock()

        # Mock repositories
        dataset_repo = Mock()
        detector_repo = Mock()
        result_repo = Mock()

        # Configure repository mocks with safe defaults
        dataset_repo.save.return_value = True
        dataset_repo.find_by_name.return_value = None
        dataset_repo.list_all.return_value = []
        dataset_repo.delete.return_value = True
        dataset_repo.exists.return_value = False

        detector_repo.save.return_value = True
        detector_repo.find_by_name.return_value = None
        detector_repo.list_all.return_value = []
        detector_repo.delete.return_value = True
        detector_repo.exists.return_value = False

        result_repo.save.return_value = True
        result_repo.find_by_id.return_value = None
        result_repo.list_all.return_value = []
        result_repo.delete.return_value = True

        # Attach repositories to container
        container.dataset_repository.return_value = dataset_repo
        container.detector_repository.return_value = detector_repo
        container.result_repository.return_value = result_repo

        # Mock use cases
        train_use_case = Mock()
        detect_use_case = Mock()
        autonomous_service = Mock()

        train_use_case.execute.return_value = Mock(success=True)
        detect_use_case.execute.return_value = Mock(success=True)
        autonomous_service.detect_anomalies.return_value = {
            "best_detector": "IsolationForest",
            "anomalies_found": 2,
            "confidence": 0.85,
            "anomaly_indices": [2, 6],
            "anomaly_scores": [0.9, 0.95],
        }
        autonomous_service.profile_dataset.return_value = {
            "dataset_summary": {"rows": 10, "columns": 4, "numeric_features": 3},
            "recommended_algorithms": ["IsolationForest"],
        }

        container.train_detector_use_case.return_value = train_use_case
        container.detect_anomalies_use_case.return_value = detect_use_case
        container.autonomous_detection_service.return_value = autonomous_service

        # Mock services
        data_loader = Mock()
        export_service = Mock()
        model_service = Mock()

        container.data_loader_service.return_value = data_loader
        container.export_service.return_value = export_service
        container.model_service.return_value = model_service

        mock_get_container.return_value = container

        yield container


@pytest.fixture
def mock_file_operations():
    """Mock file operations to prevent actual file system access."""
    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.is_file") as mock_is_file,
        patch("pathlib.Path.is_dir") as mock_is_dir,
        patch("os.path.exists") as mock_os_exists,
    ):
        # Default to files existing unless specified otherwise
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_is_dir.return_value = True
        mock_os_exists.return_value = True

        yield {
            "exists": mock_exists,
            "is_file": mock_is_file,
            "is_dir": mock_is_dir,
            "os_exists": mock_os_exists,
        }


@pytest.fixture
def mock_network_operations():
    """Mock network operations for server status checks."""
    with patch("requests.get") as mock_get:
        # Default to connection refused (server offline)
        mock_get.side_effect = ConnectionError("Connection refused")
        yield mock_get


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file."""
    import json

    config = {
        "metadata": {"type": "test", "version": "1.0"},
        "test": {
            "detector": {"algorithm": "IsolationForest", "contamination": 0.1},
            "preprocessing": {"enabled": True, "scalers": ["StandardScaler"]},
            "output": {"format": "json", "include_metadata": True},
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


# Test data generators
@pytest.fixture
def large_dataset():
    """Create a large dataset for performance testing."""
    import numpy as np
    import pandas as pd

    # Generate larger dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Normal data
    normal_data = np.random.normal(0, 1, (n_samples, n_features))

    # Add some anomalies
    n_anomalies = int(n_samples * 0.05)  # 5% anomalies
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    normal_data[anomaly_indices] += np.random.normal(5, 1, (n_anomalies, n_features))

    # Create DataFrame
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(normal_data, columns=columns)

    # Add target column
    df["target"] = 0
    df.loc[anomaly_indices, "target"] = 1

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


# Performance monitoring fixtures
@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import os
    import time

    import psutil

    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss

    yield

    end_time = time.time()
    end_memory = process.memory_info().rss

    execution_time = end_time - start_time
    memory_used = end_memory - start_memory

    # Log performance metrics if test is slow or uses too much memory
    if execution_time > 10:  # 10 seconds
        print(f"⚠️  Slow test: {execution_time:.2f}s")

    if memory_used > 100 * 1024 * 1024:  # 100MB
        print(f"⚠️  High memory usage: {memory_used / 1024 / 1024:.2f}MB")


# Error simulation fixtures
@pytest.fixture
def simulate_file_errors():
    """Fixture to simulate file system errors."""

    def _simulate_error(error_type="permission"):
        if error_type == "permission":
            return PermissionError("Permission denied")
        elif error_type == "not_found":
            return FileNotFoundError("File not found")
        elif error_type == "disk_full":
            return OSError("No space left on device")
        else:
            return OSError("Generic IO error")

    return _simulate_error


@pytest.fixture
def simulate_network_errors():
    """Fixture to simulate network errors."""

    def _simulate_error(error_type="connection"):
        if error_type == "connection":
            return ConnectionError("Connection refused")
        elif error_type == "timeout":
            return TimeoutError("Request timed out")
        elif error_type == "dns":
            return OSError("Name resolution failed")
        else:
            return Exception("Generic network error")

    return _simulate_error


# Additional CLI service mocks for improved test coverage
@pytest.fixture
def mock_dataset_service():
    """Mock dataset service for CLI tests."""
    with patch(
        "pynomaly.presentation.cli.datasets.get_cli_container"
    ) as mock_get_container:
        container = Mock()

        # Mock repository (this is what datasets CLI actually uses)
        dataset_repo = Mock()
        dataset_repo.find_all.return_value = []
        dataset_repo.find_by_id.return_value = None
        dataset_repo.save.return_value = True
        dataset_repo.delete.return_value = True
        container.dataset_repository.return_value = dataset_repo

        # Mock config
        mock_config = Mock()
        mock_config.storage_path = "/tmp/pynomaly"
        mock_config.max_dataset_size_mb = 100
        container.config.return_value = mock_config

        # Mock other dependencies
        container.csv_loader.return_value = Mock()
        container.parquet_loader.return_value = Mock()
        container.feature_validator.return_value = Mock()

        mock_get_container.return_value = container
        yield dataset_repo


@pytest.fixture
def mock_detector_service():
    """Mock detector service for CLI tests."""
    with patch(
        "pynomaly.presentation.cli.detectors.get_cli_container"
    ) as mock_get_container:
        container = Mock()

        # Mock detector service
        detector_service = Mock()
        detector_service.list_detectors.return_value = []
        detector_service.get_detector.return_value = None
        detector_service.create_detector.return_value = Mock(id="test-detector")
        detector_service.delete_detector.return_value = True
        detector_service.validate_detector.return_value = {"valid": True}

        container.detector_service.return_value = detector_service
        mock_get_container.return_value = container
        yield detector_service


@pytest.fixture
def mock_training_service():
    """Mock training service for CLI tests."""
    with patch(
        "pynomaly.presentation.cli.training.get_cli_container"
    ) as mock_get_container:
        container = Mock()

        # Mock training service
        training_service = Mock()
        training_service.start_training.return_value = {
            "job_id": "test-job",
            "status": "started",
        }
        training_service.get_training_status.return_value = {
            "status": "running",
            "progress": 50,
        }
        training_service.list_training_jobs.return_value = []
        training_service.cancel_training.return_value = True

        container.training_service.return_value = training_service
        mock_get_container.return_value = container
        yield training_service


@pytest.fixture
def mock_detection_service():
    """Mock detection service for CLI tests."""
    with patch(
        "pynomaly.presentation.cli.detection.get_cli_container"
    ) as mock_get_container:
        container = Mock()

        # Mock detection service
        detection_service = Mock()
        detection_service.run_detection.return_value = {
            "results": [],
            "anomalies_found": 0,
        }
        detection_service.get_detection_results.return_value = {"results": []}
        detection_service.list_detection_runs.return_value = []

        container.detection_service.return_value = detection_service
        mock_get_container.return_value = container
        yield detection_service


@pytest.fixture
def sample_dataset_mock():
    """Create a sample dataset mock."""
    dataset = Mock()
    dataset.id = "test-dataset"
    dataset.name = "Test Dataset"
    dataset.n_samples = 100
    dataset.n_features = 3
    dataset.has_target = True
    dataset.memory_usage = 1024 * 1024  # 1MB
    dataset.created_at = Mock()
    dataset.created_at.strftime.return_value = "2024-01-01 00:00"
    dataset.shape = (100, 3)
    dataset.target_column = "target"
    dataset.feature_names = ["feature1", "feature2"]
    dataset.description = "A test dataset"
    dataset.to_dict.return_value = {
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
    return dataset


@pytest.fixture
def sample_detector_mock():
    """Create a sample detector mock."""
    detector = Mock()
    detector.id = "test-detector"
    detector.name = "Test Detector"
    detector.algorithm_name = "IsolationForest"
    detector.parameters = {"n_estimators": 100, "contamination": 0.1}
    detector.contamination_rate = 0.1
    detector.created_at = Mock()
    detector.created_at.strftime.return_value = "2024-01-01 00:00"
    detector.is_trained = True
    detector.training_dataset_id = "test-dataset"
    detector.to_dict.return_value = {
        "id": "test-detector",
        "name": "Test Detector",
        "algorithm": "IsolationForest",
        "parameters": {"n_estimators": 100, "contamination": 0.1},
        "created_at": "2024-01-01T00:00:00Z",
        "is_trained": True,
    }
    return detector
