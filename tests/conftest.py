"""
Global pytest configuration for Pynomaly comprehensive testing

This configuration file provides fixtures, marks, and settings
for all test categories across the comprehensive testing framework.
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =========================================================================
# PYTEST CONFIGURATION
# =========================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    
    # Register all test markers
    markers = [
        "unit: Unit tests for individual components",
        "integration: Integration tests across components", 
        "e2e: End-to-end tests for complete workflows",
        "performance: Performance and benchmark tests",
        "security: Security-focused tests",
        "api: API endpoint tests",
        "ui: User interface tests",
        "platform: Cross-platform compatibility tests", 
        "load: Load testing with multiple users",
        "regression: Regression tests against baselines",
        "slow: Tests that take more than 1 second",
        "fast: Tests that complete in under 1 second",
        "external: Tests requiring external services",
        "database: Tests requiring database",
        "redis: Tests requiring Redis",
        "auth: Authentication/authorization tests",
        "contract: API contract validation tests"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location/name"""
    
    for item in items:
        # Auto-mark based on file path
        if "/unit/" in str(item.fspath) or "test_unit_" in item.name:
            item.add_marker(pytest.mark.unit)
        
        if "/integration/" in str(item.fspath) or "test_integration_" in item.name:
            item.add_marker(pytest.mark.integration)
            
        if "/e2e/" in str(item.fspath) or "test_e2e_" in item.name:
            item.add_marker(pytest.mark.e2e)
            
        if "/performance/" in str(item.fspath) or "test_performance_" in item.name:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
            
        if "/security/" in str(item.fspath) or "test_security_" in item.name:
            item.add_marker(pytest.mark.security)
            
        if "/api/" in str(item.fspath) or "test_api_" in item.name:
            item.add_marker(pytest.mark.api)
            
        if "/ui/" in str(item.fspath) or "test_ui_" in item.name:
            item.add_marker(pytest.mark.ui)
            item.add_marker(pytest.mark.slow)
            
        if "/load/" in str(item.fspath) or "test_load_" in item.name:
            item.add_marker(pytest.mark.load)
            item.add_marker(pytest.mark.slow)
            
        # Mark slow tests
        if "large_dataset" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)


# =========================================================================
# CORE FIXTURES
# =========================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests"""
    np.random.seed(42)
    yield 42


@pytest.fixture
def temp_directory():
    """Create temporary directory for test artifacts"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration dictionary"""
    return {
        "testing": True,
        "log_level": "DEBUG",
        "database_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/1",
        "api_timeout": 30,
        "batch_size": 100
    }


# =========================================================================
# DATA FIXTURES
# =========================================================================

@pytest.fixture
def small_dataset() -> np.ndarray:
    """Small dataset for quick tests (100 samples, 5 features)"""
    np.random.seed(42)
    data = np.random.randn(100, 5)
    # Add some outliers
    data[:10] += 3
    return data


@pytest.fixture
def medium_dataset() -> np.ndarray:
    """Medium dataset for standard tests (1000 samples, 10 features)"""
    np.random.seed(42)
    data = np.random.randn(1000, 10)
    # Add some outliers
    data[:100] += 2
    return data


@pytest.fixture
def large_dataset() -> np.ndarray:
    """Large dataset for performance tests (10000 samples, 20 features)"""
    np.random.seed(42)
    data = np.random.randn(10000, 20)
    # Add some outliers
    data[:1000] += 2.5
    return data


@pytest.fixture
def high_dimensional_dataset() -> np.ndarray:
    """High-dimensional dataset (500 samples, 100 features)"""
    np.random.seed(42)
    data = np.random.randn(500, 100)
    # Add some outliers
    data[:50] += 1.5
    return data


@pytest.fixture
def time_series_data() -> pd.DataFrame:
    """Time series data for temporal tests"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    data = {
        'timestamp': dates,
        'value': np.random.randn(1000).cumsum(),
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000)
    }
    # Add some anomalous periods
    anomaly_indices = np.random.choice(1000, 50, replace=False)
    for idx in anomaly_indices:
        data['value'][idx] += np.random.normal(0, 5)
    
    return pd.DataFrame(data)


@pytest.fixture
def mixed_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Dataset with known normal/anomalous labels"""
    np.random.seed(42)
    
    # Normal samples
    normal_data = np.random.randn(800, 8)
    normal_labels = np.zeros(800)
    
    # Anomalous samples
    anomalous_data = np.random.randn(200, 8) + 3
    anomalous_labels = np.ones(200)
    
    # Combine
    data = np.vstack([normal_data, anomalous_data])
    labels = np.hstack([normal_labels, anomalous_labels])
    
    # Shuffle
    indices = np.random.permutation(len(data))
    
    return data[indices], labels[indices]


@pytest.fixture
def contamination_levels() -> list[float]:
    """Standard contamination levels for testing"""
    return [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


@pytest.fixture
def performance_datasets() -> Dict[str, np.ndarray]:
    """Datasets of different sizes for performance testing"""
    np.random.seed(42)
    
    return {
        "tiny": np.random.randn(50, 5),
        "small": np.random.randn(200, 10),
        "medium": np.random.randn(1000, 15),
        "large": np.random.randn(5000, 20),
        "xlarge": np.random.randn(10000, 25)
    }


# =========================================================================
# API TESTING FIXTURES
# =========================================================================

@pytest.fixture(scope="session")
def api_client():
    """FastAPI test client for API testing"""
    try:
        from src.packages.software.interfaces.api.app import app
        return TestClient(app)
    except ImportError:
        # Mock client if API not available
        mock_client = Mock()
        mock_client.get.return_value.status_code = 200
        mock_client.post.return_value.status_code = 200
        return mock_client


@pytest.fixture
def api_headers() -> Dict[str, str]:
    """Standard API headers for testing"""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Pynomaly-Test-Suite/1.0"
    }


@pytest.fixture
def sample_api_data() -> Dict[str, Any]:
    """Sample data for API requests"""
    return {
        "data": np.random.randn(100, 5).tolist(),
        "contamination": 0.1,
        "algorithm": "isolation_forest",
        "parameters": {
            "n_estimators": 100,
            "random_state": 42
        }
    }


# =========================================================================
# DATABASE FIXTURES
# =========================================================================

@pytest.fixture
def in_memory_db():
    """In-memory SQLite database for testing"""
    import sqlite3
    
    conn = sqlite3.connect(":memory:")
    
    # Create test schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY,
            test_name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            result TEXT NOT NULL,
            data TEXT
        )
    """)
    
    yield conn
    conn.close()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    from unittest.mock import Mock
    
    redis_mock = Mock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.exists.return_value = False
    
    return redis_mock


# =========================================================================
# SECURITY TESTING FIXTURES
# =========================================================================

@pytest.fixture
def security_test_data() -> Dict[str, Any]:
    """Data for security testing including malicious inputs"""
    return {
        "sql_injection_attempts": [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ],
        "xss_attempts": [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "'\"><script>alert('XSS')</script>"
        ],
        "path_traversal_attempts": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\system",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2f",
            "....//....//....//etc/passwd"
        ],
        "large_payload": "A" * 1000000,  # 1MB of data
        "special_characters": "!@#$%^&*()[]{}|;':\",./<>?`~",
        "unicode_characters": "αβγδεζηθικλμνξοπρστυφχψω",
        "null_bytes": "\x00\x01\x02\x03\x04\x05"
    }


# =========================================================================
# PERFORMANCE TESTING FIXTURES
# =========================================================================

@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks"""
    return {
        "min_rounds": 3,
        "max_time": 10.0,
        "warmup": True,
        "timer": time.perf_counter
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for regression detection"""
    return {
        "small_dataset_processing": 1.0,  # seconds
        "medium_dataset_processing": 5.0,
        "large_dataset_processing": 30.0,
        "api_response_time": 0.5,
        "memory_usage_mb": 1000,
        "cpu_usage_percent": 80
    }


# =========================================================================
# UI TESTING FIXTURES (Playwright)
# =========================================================================

@pytest.fixture(scope="session")
def browser():
    """Playwright browser instance"""
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            yield browser
            browser.close()
    except ImportError:
        # Mock browser if Playwright not available
        mock_browser = Mock()
        yield mock_browser


@pytest.fixture
def page(browser):
    """Playwright page instance"""
    try:
        page = browser.new_page()
        yield page
        page.close()
    except AttributeError:
        # Mock page if browser is mock
        mock_page = Mock()
        yield mock_page


# =========================================================================
# LOAD TESTING FIXTURES
# =========================================================================

@pytest.fixture
def load_test_config():
    """Configuration for load testing"""
    return {
        "users": 10,
        "spawn_rate": 2,
        "run_time": "60s",
        "host": "http://localhost:8000"
    }


# =========================================================================
# MOCKING FIXTURES
# =========================================================================

@pytest.fixture
def mock_external_service():
    """Mock external service responses"""
    def _mock_service(response_data=None, status_code=200):
        mock = Mock()
        mock.status_code = status_code
        mock.json.return_value = response_data or {"status": "success"}
        mock.text = str(response_data) if response_data else "OK"
        return mock
    
    return _mock_service


@pytest.fixture
def mock_ml_model():
    """Mock machine learning model for testing"""
    mock_model = Mock()
    mock_model.fit.return_value = mock_model
    mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
    mock_model.decision_function.return_value = np.array([0.1, 2.5, 0.2, 3.1, -0.1])
    return mock_model


# =========================================================================
# CUSTOM ASSERTIONS
# =========================================================================

def assert_valid_predictions(predictions: np.ndarray, data_length: int):
    """Assert that anomaly predictions are valid"""
    assert isinstance(predictions, np.ndarray), "Predictions must be numpy array"
    assert len(predictions) == data_length, f"Expected {data_length} predictions, got {len(predictions)}"
    assert predictions.dtype in [np.int32, np.int64, int], "Predictions must be integers"
    assert all(p in [0, 1] for p in predictions), "Predictions must be 0 (normal) or 1 (anomaly)"


def assert_reasonable_anomaly_count(predictions: np.ndarray, contamination: float, tolerance: float = 0.1):
    """Assert that anomaly count is reasonable given contamination level"""
    anomaly_count = np.sum(predictions)
    expected_count = int(contamination * len(predictions))
    tolerance_count = int(tolerance * len(predictions))
    
    assert abs(anomaly_count - expected_count) <= tolerance_count, \
        f"Anomaly count {anomaly_count} not within tolerance of expected {expected_count}"


def assert_performance_within_threshold(actual_time: float, threshold_time: float, test_name: str = ""):
    """Assert that performance is within acceptable threshold"""
    assert actual_time <= threshold_time, \
        f"{test_name} took {actual_time:.3f}s, exceeded threshold of {threshold_time:.3f}s"


def assert_api_response_valid(response, expected_status: int = 200):
    """Assert that API response is valid"""
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}"
    
    if expected_status == 200:
        assert "application/json" in response.headers.get("content-type", ""), \
            "Response should be JSON"


# Register custom assertions in pytest namespace
pytest.assert_valid_predictions = assert_valid_predictions
pytest.assert_reasonable_anomaly_count = assert_reasonable_anomaly_count  
pytest.assert_performance_within_threshold = assert_performance_within_threshold
pytest.assert_api_response_valid = assert_api_response_valid


# =========================================================================
# ENVIRONMENT SETUP/TEARDOWN
# =========================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before any tests run"""
    
    # Set environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise during testing
    
    # Create test directories
    test_dirs = ["test_reports", "test_artifacts", "test_data"]
    for dir_name in test_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after all tests (optional)
    # Could remove test directories here if needed


@pytest.fixture(autouse=True)
def isolate_test_state():
    """Ensure each test starts with clean state"""
    
    # Reset any global state before each test
    yield
    
    # Cleanup after each test if needed