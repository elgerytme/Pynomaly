"""Integration test configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
from anomaly_detection.infrastructure.config.settings import Settings
from anomaly_detection.infrastructure.monitoring import (
    get_metrics_collector,
    get_health_checker,
    get_performance_monitor
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """Create test datasets for integration tests."""
    np.random.seed(42)
    
    # Normal data (2D Gaussian)
    normal_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
    
    # Anomalous data (outliers)
    anomaly_data = np.random.multivariate_normal([5, 5], [[0.5, 0], [0, 0.5]], 10)
    
    # Combined dataset
    all_data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([np.ones(100), -np.ones(10)])  # 1 for normal, -1 for anomaly
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['feature_1', 'feature_2'])
    df['label'] = labels
    
    return {
        'data_only': all_data.astype(np.float64),
        'data_with_labels': df,
        'normal_data': normal_data.astype(np.float64),
        'anomaly_data': anomaly_data.astype(np.float64),
        'labels': labels,
        'n_samples': 110,
        'n_features': 2,
        'n_anomalies': 10,
        'contamination': 0.09  # 10/110
    }


@pytest.fixture
def detection_service() -> DetectionService:
    """Create a detection service for testing."""
    return DetectionService()


@pytest.fixture
def ensemble_service() -> EnsembleService:
    """Create an ensemble service for testing."""
    return EnsembleService()


@pytest.fixture
def model_repository(temp_dir: Path) -> ModelRepository:
    """Create a model repository with temporary storage."""
    return ModelRepository(str(temp_dir / "models"))


@pytest.fixture
def test_settings(temp_dir: Path) -> Settings:
    """Create test settings with temporary directories."""
    return Settings(
        environment="test",
        debug=True,
        database=MagicMock(),
        logging=MagicMock(),
        detection=MagicMock(),
        api=MagicMock()
    )


@pytest.fixture
def sample_csv_file(test_data: Dict[str, Any], temp_dir: Path) -> Path:
    """Create a sample CSV file for CLI testing."""
    csv_file = temp_dir / "test_data.csv"
    test_data['data_with_labels'].to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_json_file(test_data: Dict[str, Any], temp_dir: Path) -> Path:
    """Create a sample JSON file for CLI testing."""
    json_file = temp_dir / "test_data.json"
    test_data['data_with_labels'].to_json(json_file, orient='records')
    return json_file


@pytest.fixture
def metrics_collector():
    """Get the global metrics collector for testing."""
    collector = get_metrics_collector()
    # Clear any existing metrics
    collector.clear_all_metrics()
    return collector


@pytest.fixture
def health_checker():
    """Get the global health checker for testing."""
    return get_health_checker()


@pytest.fixture
def performance_monitor():
    """Get the global performance monitor for testing."""
    monitor = get_performance_monitor()
    # Clear any existing profiles
    monitor.clear_profiles()
    return monitor


@pytest.fixture
async def mock_fastapi_client():
    """Create a test client for FastAPI endpoints."""
    from fastapi.testclient import TestClient
    from anomaly_detection.server import create_app
    
    app = create_app()
    return TestClient(app)


@pytest.fixture
def large_dataset() -> np.ndarray:
    """Create a larger dataset for performance testing."""
    np.random.seed(42)
    
    # Create a larger dataset (1000 samples, 10 features)
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(10),
        cov=np.eye(10),
        size=900
    )
    
    # Add some anomalies
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(10) * 3,
        cov=np.eye(10) * 0.5,
        size=100
    )
    
    data = np.vstack([normal_data, anomaly_data])
    np.random.shuffle(data)
    
    return data.astype(np.float64)


@pytest.fixture
def streaming_data() -> Generator[np.ndarray, None, None]:
    """Generate streaming data for streaming tests."""
    np.random.seed(42)
    
    for i in range(100):
        # Generate single sample
        if i % 20 == 0:  # Every 20th sample is anomalous
            sample = np.random.multivariate_normal([3, 3], [[0.1, 0], [0, 0.1]], 1)
        else:
            sample = np.random.multivariate_normal([0, 0], [[1, 0.2], [0.2, 1]], 1)
        
        yield sample.astype(np.float64)


# Utility functions for tests

def assert_detection_result_valid(result, expected_samples: int, algorithm: str = None):
    """Assert that a detection result is valid."""
    assert result is not None
    assert hasattr(result, 'predictions')
    assert hasattr(result, 'success')
    assert hasattr(result, 'total_samples')
    assert hasattr(result, 'anomaly_count')
    assert hasattr(result, 'normal_count')
    assert hasattr(result, 'anomaly_rate')
    
    assert result.success is True
    assert result.total_samples == expected_samples
    assert len(result.predictions) == expected_samples
    assert result.anomaly_count + result.normal_count == expected_samples
    assert 0 <= result.anomaly_rate <= 1
    
    if algorithm:
        assert result.algorithm == algorithm
    
    # Check predictions are in correct format (sklearn: -1 for anomaly, 1 for normal)
    assert all(pred in [-1, 1] for pred in result.predictions)


def assert_model_saved_correctly(model_path: Path, model_id: str):
    """Assert that a model was saved correctly."""
    assert model_path.exists()
    
    model_dir = model_path / model_id
    assert model_dir.exists()
    assert (model_dir / "metadata.json").exists()
    assert (model_dir / "model.pkl").exists() or (model_dir / "model.joblib").exists()


def create_test_model_metadata(model_id: str = "test-model") -> Dict[str, Any]:
    """Create test model metadata."""
    from datetime import datetime
    
    return {
        "model_id": model_id,
        "name": "Test Model",
        "algorithm": "isolation_forest",
        "version": "1.0.0",
        "status": "trained",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "training_samples": 100,
        "training_features": 2,
        "contamination_rate": 0.1,
        "accuracy": 0.95,
        "precision": 0.9,
        "recall": 0.85,
        "f1_score": 0.875
    }