"""
End-to-End Integration Test Configuration
"""

import asyncio
import os
import tempfile
import time
from typing import Dict, Any, AsyncGenerator
import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

# Import test client and dependencies
try:
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
except ImportError:
    pytest.skip("FastAPI test dependencies not available", allow_module_level=True)

# Pynomaly imports
import sys
sys.path.insert(0, '/mnt/c/Users/andre/Pynomaly/src')

from pynomaly.presentation.api.app import create_app
from pynomaly.infrastructure.persistence.database import get_database_url
from pynomaly.infrastructure.config.container import ApplicationContainer


class E2ETestConfig:
    """Configuration for end-to-end tests."""
    
    # Test database configuration
    TEST_DATABASE_URL = "sqlite:///:memory:"
    
    # API configuration
    API_BASE_URL = "http://testserver"
    API_TIMEOUT = 30.0
    
    # Test data configuration
    SAMPLE_DATA_SIZE = 1000
    ANOMALY_RATE = 0.05
    FEATURE_COUNT = 5
    
    # Performance test configuration
    LOAD_TEST_DURATION = 10  # seconds
    CONCURRENT_REQUESTS = 5
    
    # Security test configuration
    INVALID_API_KEYS = ["invalid-key", "", "wrong-format"]
    
    # Multi-tenant test configuration
    TEST_TENANTS = ["tenant-1", "tenant-2", "tenant-3"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return E2ETestConfig()


@pytest.fixture(scope="session")
def test_database():
    """Create test database engine."""
    engine = create_engine(
        E2ETestConfig.TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine


@pytest.fixture(scope="session")
def app_container(test_database):
    """Create application container for testing."""
    container = ApplicationContainer()
    
    # Override database settings for testing
    container.config.database.url.from_value(E2ETestConfig.TEST_DATABASE_URL)
    container.config.database.echo.from_value(False)
    
    # Wire container
    container.wire(modules=["pynomaly.presentation.api.app"])
    
    return container


@pytest.fixture(scope="session")
def app(app_container):
    """Create FastAPI application for testing."""
    return create_app()


@pytest.fixture(scope="session")
def test_client(app):
    """Create test client for API testing."""
    return TestClient(app)


@pytest.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for API testing."""
    async with AsyncClient(app=app, base_url=E2ETestConfig.API_BASE_URL) as client:
        yield client


@pytest.fixture
def sample_dataset():
    """Generate sample dataset for testing."""
    np.random.seed(42)
    
    # Generate normal data
    normal_size = int(E2ETestConfig.SAMPLE_DATA_SIZE * (1 - E2ETestConfig.ANOMALY_RATE))
    normal_data = np.random.multivariate_normal(
        mean=[0] * E2ETestConfig.FEATURE_COUNT,
        cov=np.eye(E2ETestConfig.FEATURE_COUNT),
        size=normal_size
    )
    
    # Generate anomalous data
    anomaly_size = E2ETestConfig.SAMPLE_DATA_SIZE - normal_size
    anomaly_data = np.random.multivariate_normal(
        mean=[5] * E2ETestConfig.FEATURE_COUNT,
        cov=np.eye(E2ETestConfig.FEATURE_COUNT) * 2,
        size=anomaly_size
    )
    
    # Combine data
    data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([
        np.zeros(normal_size),
        np.ones(anomaly_size)
    ])
    
    # Shuffle
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(E2ETestConfig.FEATURE_COUNT)]
    df = pd.DataFrame(data, columns=feature_names)
    df['is_anomaly'] = labels
    
    return {
        'dataframe': df,
        'features': df[feature_names],
        'labels': labels,
        'feature_names': feature_names,
        'anomaly_rate': E2ETestConfig.ANOMALY_RATE
    }


@pytest.fixture
def multimodal_dataset():
    """Generate multimodal dataset for advanced testing."""
    np.random.seed(123)
    
    # Multiple normal clusters
    cluster1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 300)
    cluster2 = np.random.multivariate_normal([5, 5], [[1, 0.5], [0.5, 1]], 300)
    cluster3 = np.random.multivariate_normal([-3, 4], [[2, -0.5], [-0.5, 2]], 300)
    
    # Anomalies scattered throughout
    anomalies = np.random.uniform(-10, 15, (100, 2))
    
    # Combine data
    normal_data = np.vstack([cluster1, cluster2, cluster3])
    all_data = np.vstack([normal_data, anomalies])
    labels = np.hstack([np.zeros(900), np.ones(100)])
    
    # Shuffle
    indices = np.random.permutation(len(all_data))
    all_data = all_data[indices]
    labels = labels[indices]
    
    df = pd.DataFrame(all_data, columns=['x', 'y'])
    df['is_anomaly'] = labels
    
    return {
        'dataframe': df,
        'features': df[['x', 'y']],
        'labels': labels,
        'feature_names': ['x', 'y'],
        'anomaly_rate': 0.1
    }


@pytest.fixture
def time_series_dataset():
    """Generate time series dataset for temporal testing."""
    np.random.seed(456)
    
    # Generate time series with trend and seasonality
    time_points = 1000
    time = np.arange(time_points)
    
    # Base signal with trend and seasonality
    trend = 0.01 * time
    seasonality = 2 * np.sin(2 * np.pi * time / 100)
    noise = np.random.normal(0, 0.5, time_points)
    base_signal = trend + seasonality + noise
    
    # Add anomalies at specific points
    anomaly_indices = np.random.choice(time_points, 50, replace=False)
    anomalous_signal = base_signal.copy()
    anomalous_signal[anomaly_indices] += np.random.normal(5, 2, 50)
    
    labels = np.zeros(time_points)
    labels[anomaly_indices] = 1
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=time_points, freq='H'),
        'value': anomalous_signal,
        'is_anomaly': labels
    })
    
    return {
        'dataframe': df,
        'features': df[['value']],
        'labels': labels,
        'feature_names': ['value'],
        'anomaly_rate': 0.05
    }


@pytest.fixture
def api_headers():
    """Generate API headers for testing."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer test-api-key"
    }


@pytest.fixture
def invalid_api_headers():
    """Generate invalid API headers for security testing."""
    return [
        {},  # No headers
        {"Content-Type": "application/json"},  # Missing auth
        {"Authorization": "Bearer invalid-key"},  # Invalid key
        {"Authorization": "InvalidFormat"},  # Wrong format
    ]


@pytest.fixture
def performance_monitor():
    """Performance monitoring utility."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            
        def start_timer(self, name: str):
            self.metrics[name] = {"start": time.time()}
            
        def end_timer(self, name: str):
            if name in self.metrics:
                self.metrics[name]["end"] = time.time()
                self.metrics[name]["duration"] = (
                    self.metrics[name]["end"] - self.metrics[name]["start"]
                )
                
        def get_duration(self, name: str) -> float:
            return self.metrics.get(name, {}).get("duration", 0.0)
            
        def get_all_metrics(self) -> Dict[str, Any]:
            return self.metrics.copy()
    
    return PerformanceMonitor()


@pytest.fixture
def test_workspace():
    """Create temporary workspace for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = {
            'root': temp_dir,
            'data': os.path.join(temp_dir, 'data'),
            'models': os.path.join(temp_dir, 'models'),
            'results': os.path.join(temp_dir, 'results'),
            'logs': os.path.join(temp_dir, 'logs')
        }
        
        # Create subdirectories
        for path in workspace.values():
            if path != workspace['root']:
                os.makedirs(path, exist_ok=True)
                
        yield workspace


@pytest.fixture
def error_simulator():
    """Utility for simulating various error conditions."""
    class ErrorSimulator:
        def __init__(self):
            self.original_methods = {}
            
        def simulate_network_error(self, client, method_name: str):
            """Simulate network connectivity issues."""
            def raise_network_error(*args, **kwargs):
                raise ConnectionError("Simulated network error")
            
            original = getattr(client, method_name)
            self.original_methods[method_name] = original
            setattr(client, method_name, raise_network_error)
            
        def simulate_timeout(self, client, method_name: str):
            """Simulate request timeout."""
            def raise_timeout(*args, **kwargs):
                raise TimeoutError("Simulated timeout")
                
            original = getattr(client, method_name)
            self.original_methods[method_name] = original
            setattr(client, method_name, raise_timeout)
            
        def restore_method(self, client, method_name: str):
            """Restore original method."""
            if method_name in self.original_methods:
                setattr(client, method_name, self.original_methods[method_name])
                del self.original_methods[method_name]
                
        def restore_all(self, client):
            """Restore all modified methods."""
            for method_name in list(self.original_methods.keys()):
                self.restore_method(client, method_name)
    
    return ErrorSimulator()


# Utility functions for tests
def assert_detection_quality(result: Dict[str, Any], expected_anomaly_rate: float, tolerance: float = 0.1):
    """Assert that detection results meet quality standards."""
    actual_rate = result['n_anomalies'] / result['n_samples']
    assert abs(actual_rate - expected_anomaly_rate) <= tolerance, (
        f"Detection rate {actual_rate:.3f} differs from expected {expected_anomaly_rate:.3f} "
        f"by more than tolerance {tolerance}"
    )


def assert_performance_within_limits(duration: float, max_duration: float):
    """Assert that operation completed within performance limits."""
    assert duration <= max_duration, (
        f"Operation took {duration:.2f}s, exceeding limit of {max_duration:.2f}s"
    )


def assert_api_response_valid(response, expected_status: int = 200):
    """Assert that API response is valid."""
    assert response.status_code == expected_status, (
        f"Expected status {expected_status}, got {response.status_code}. "
        f"Response: {response.text}"
    )
    
    if expected_status == 200:
        assert response.headers.get("content-type", "").startswith("application/json")
        
        
# Markers for test categorization
pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e
]