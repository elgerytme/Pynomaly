"""Integration testing configuration and fixtures."""

import asyncio
import logging
import os
import tempfile
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from pynomaly.infrastructure.config import Container, create_container
from pynomaly.infrastructure.config.settings import Settings
from pynomaly.presentation.api.app import create_app


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings."""
    # Create temporary directory for test data
    temp_dir = tempfile.mkdtemp()

    # Override settings for testing
    os.environ.update(
        {
            "PYNOMALY_ENVIRONMENT": "testing",
            "PYNOMALY_DATABASE_URL": f"sqlite:///{temp_dir}/test.db",
            "PYNOMALY_CACHE_ENABLED": "false",
            "PYNOMALY_AUTH_ENABLED": "false",
            "PYNOMALY_DOCS_ENABLED": "true",
            "PYNOMALY_CORS_ENABLED": "true",
            "PYNOMALY_MONITORING_METRICS_ENABLED": "false",
            "PYNOMALY_MONITORING_TRACING_ENABLED": "false",
            "PYNOMALY_MONITORING_PROMETHEUS_ENABLED": "false",
            "PYNOMALY_LOG_LEVEL": "DEBUG",
            "PYNOMALY_DATA_DIR": temp_dir,
            "PYNOMALY_MODEL_DIR": f"{temp_dir}/models",
            "PYNOMALY_UPLOAD_DIR": f"{temp_dir}/uploads",
        }
    )

    container = create_container()
    settings = container.config()

    return settings


@pytest.fixture(scope="session")
def test_container(test_settings: Settings) -> Container:
    """Create test dependency injection container."""
    container = create_container()
    return container


@pytest.fixture(scope="session")
def test_app(test_container: Container):
    """Create test FastAPI application."""
    app = create_app(test_container)
    return app


@pytest.fixture(scope="session")
def test_client(test_app) -> Generator[TestClient, None, None]:
    """Create test client for synchronous testing."""
    with TestClient(test_app) as client:
        yield client


@pytest_asyncio.fixture(scope="session")
async def async_test_client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for asynchronous testing."""
    async with AsyncClient(app=test_app, base_url="http://testserver") as client:
        yield client


@pytest.fixture(scope="session")
def sample_dataset_csv(test_settings: Settings) -> str:
    """Create sample CSV dataset for testing."""
    import numpy as np
    import pandas as pd

    # Generate sample anomaly detection dataset
    np.random.seed(42)
    n_samples = 1000
    n_anomalies = 50

    # Normal data (multivariate normal distribution)
    normal_data = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]],
        size=n_samples - n_anomalies,
    )

    # Anomalous data (outliers)
    anomaly_data = np.random.multivariate_normal(
        mean=[3, 3, 3], cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], size=n_anomalies
    )

    # Combine data
    data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([np.zeros(n_samples - n_anomalies), np.ones(n_anomalies)])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])
    df["label"] = labels
    df["timestamp"] = pd.date_range("2024-01-01", periods=n_samples, freq="1H")

    # Save to CSV
    csv_path = os.path.join(test_settings.data_dir, "sample_dataset.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture(scope="session")
def sample_time_series_csv(test_settings: Settings) -> str:
    """Create sample time series dataset for testing."""
    import numpy as np
    import pandas as pd

    # Generate time series with anomalies
    np.random.seed(42)
    n_samples = 2000
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="5T")

    # Base trend and seasonality
    t = np.arange(n_samples)
    trend = 0.01 * t
    daily_seasonal = 10 * np.sin(
        2 * np.pi * t / (24 * 12)
    )  # 24 hours * 12 (5-min intervals)
    weekly_seasonal = 5 * np.sin(2 * np.pi * t / (7 * 24 * 12))  # 7 days
    noise = np.random.normal(0, 2, n_samples)

    # Normal time series
    values = 50 + trend + daily_seasonal + weekly_seasonal + noise

    # Inject anomalies
    anomaly_indices = np.random.choice(n_samples, size=100, replace=False)
    anomaly_multipliers = np.random.choice([0.3, 2.5, 3.0], size=100)
    values[anomaly_indices] *= anomaly_multipliers

    # Create labels
    labels = np.zeros(n_samples)
    labels[anomaly_indices] = 1

    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": values,
            "cpu_usage": 20 + 30 * np.random.random(n_samples),
            "memory_usage": 40 + 40 * np.random.random(n_samples),
            "network_io": np.random.exponential(1000, n_samples),
            "label": labels,
        }
    )

    # Save to CSV
    csv_path = os.path.join(test_settings.data_dir, "time_series_dataset.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture(scope="function")
def cleanup_test_data(test_settings: Settings):
    """Clean up test data after each test."""
    yield

    # Clean up any test files created during the test
    import shutil

    # Remove uploaded files
    upload_dir = test_settings.upload_dir
    if os.path.exists(upload_dir):
        for file in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    # Remove model files
    model_dir = test_settings.model_dir
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)


@pytest.fixture(scope="function")
def auth_headers() -> dict[str, str]:
    """Create authentication headers for testing."""
    # In a real implementation, this would create a valid JWT token
    # For testing purposes, we'll use a mock token or skip auth
    return {"Authorization": "Bearer test-token", "Content-Type": "application/json"}


@pytest.fixture(scope="function")
def disable_auth(test_settings: Settings):
    """Disable authentication for testing."""
    # Temporarily disable auth for testing
    original_auth_enabled = test_settings.auth_enabled
    test_settings.auth_enabled = False

    yield

    # Restore original auth setting
    test_settings.auth_enabled = original_auth_enabled


# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Suppress noisy loggers during testing
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


class IntegrationTestHelper:
    """Helper class for integration testing."""

    def __init__(self, client: AsyncClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.created_resources = {
            "datasets": [],
            "detectors": [],
            "models": [],
            "sessions": [],
            "experiments": [],
        }

    async def upload_dataset(self, file_path: str, name: str = None) -> dict:
        """Upload a dataset for testing."""
        if name is None:
            name = f"test_dataset_{os.path.basename(file_path)}"

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "text/csv")}
            data = {"name": name, "description": "Test dataset"}

            response = await self.client.post(
                "/api/datasets/upload", files=files, data=data
            )

        response.raise_for_status()
        dataset = response.json()["data"]
        self.created_resources["datasets"].append(dataset["id"])
        return dataset

    async def create_detector(
        self, dataset_id: str, algorithm: str = "isolation_forest"
    ) -> dict:
        """Create a detector for testing."""
        detector_data = {
            "name": f"test_detector_{algorithm}",
            "description": "Test detector",
            "algorithm": algorithm,
            "parameters": {"contamination": 0.1, "random_state": 42},
            "feature_columns": ["feature1", "feature2", "feature3"],
        }

        response = await self.client.post(
            f"/api/detectors/create?dataset_id={dataset_id}", json=detector_data
        )

        response.raise_for_status()
        detector = response.json()["data"]
        self.created_resources["detectors"].append(detector["id"])
        return detector

    async def train_detector(self, detector_id: str) -> dict:
        """Train a detector for testing."""
        response = await self.client.post(f"/api/detection/train/{detector_id}")
        response.raise_for_status()
        return response.json()["data"]

    async def cleanup_resources(self):
        """Clean up all created resources."""
        # Clean up in reverse order of dependencies

        # Clean up sessions
        for session_id in self.created_resources["sessions"]:
            try:
                await self.client.delete(f"/api/streaming/sessions/{session_id}")
            except:
                pass

        # Clean up experiments
        for experiment_id in self.created_resources["experiments"]:
            try:
                await self.client.delete(f"/api/experiments/{experiment_id}")
            except:
                pass

        # Clean up models
        for model_id in self.created_resources["models"]:
            try:
                await self.client.delete(f"/api/models/{model_id}")
            except:
                pass

        # Clean up detectors
        for detector_id in self.created_resources["detectors"]:
            try:
                await self.client.delete(f"/api/detectors/{detector_id}")
            except:
                pass

        # Clean up datasets
        for dataset_id in self.created_resources["datasets"]:
            try:
                await self.client.delete(f"/api/datasets/{dataset_id}")
            except:
                pass


@pytest.fixture(scope="function")
async def integration_helper(
    async_test_client: AsyncClient, test_settings: Settings
) -> AsyncGenerator[IntegrationTestHelper, None]:
    """Create integration test helper."""
    helper = IntegrationTestHelper(async_test_client, test_settings)
    yield helper
    await helper.cleanup_resources()
