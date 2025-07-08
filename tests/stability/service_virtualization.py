"""Service virtualization and mocking system for external APIs - Phase 1 Stability."""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, MagicMock, patch

import pytest

# Optional imports for HTTP mocking
try:
    import requests
    import responses
    RESPONSES_AVAILABLE = True
except ImportError:
    RESPONSES_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


@dataclass
class APIEndpoint:
    """Configuration for a mocked API endpoint."""
    method: str
    url: str
    response_body: Union[Dict, List, str]
    status_code: int = 200
    headers: Optional[Dict[str, str]] = None
    delay: float = 0.0
    failure_rate: float = 0.0
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Type": "application/json"}


class ServiceVirtualization:
    """Service virtualization system for external APIs."""
    
    def __init__(self):
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.call_logs: List[Dict] = []
        self.enabled = True
        self.default_timeout = 30
        self.mock_external_services = True
        
        # Load configuration from environment
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment variables."""
        self.enabled = os.getenv("MOCK_EXTERNAL_SERVICES", "true").lower() == "true"
        self.default_timeout = int(os.getenv("MOCK_API_TIMEOUT", "30"))
        self.mock_external_services = os.getenv("VIRTUALIZE_APIS", "true").lower() == "true"
    
    def register_endpoint(self, endpoint: APIEndpoint):
        """Register a mocked API endpoint."""
        key = f"{endpoint.method.upper()}:{endpoint.url}"
        self.endpoints[key] = endpoint
    
    def register_endpoints(self, endpoints: List[APIEndpoint]):
        """Register multiple mocked API endpoints."""
        for endpoint in endpoints:
            self.register_endpoint(endpoint)
    
    def get_endpoint(self, method: str, url: str) -> Optional[APIEndpoint]:
        """Get a registered endpoint."""
        key = f"{method.upper()}:{url}"
        return self.endpoints.get(key)
    
    def log_call(self, method: str, url: str, **kwargs):
        """Log an API call."""
        self.call_logs.append({
            "method": method,
            "url": url,
            "timestamp": time.time(),
            "kwargs": kwargs
        })
    
    def get_call_logs(self) -> List[Dict]:
        """Get all logged API calls."""
        return self.call_logs.copy()
    
    def clear_call_logs(self):
        """Clear all logged API calls."""
        self.call_logs.clear()
    
    def setup_requests_mock(self):
        """Set up requests library mocking."""
        if not RESPONSES_AVAILABLE:
            return None
        
        mock = responses.RequestsMock()
        
        for key, endpoint in self.endpoints.items():
            method, url = key.split(":", 1)
            
            def callback(request, endpoint=endpoint):
                # Simulate delay
                if endpoint.delay > 0:
                    time.sleep(endpoint.delay)
                
                # Simulate failure
                if endpoint.failure_rate > 0:
                    import random
                    if random.random() < endpoint.failure_rate:
                        return (500, {}, "Service temporarily unavailable")
                
                # Log the call
                self.log_call(method, url, headers=dict(request.headers))
                
                # Return response
                if isinstance(endpoint.response_body, (dict, list)):
                    body = json.dumps(endpoint.response_body)
                else:
                    body = endpoint.response_body
                
                return (endpoint.status_code, endpoint.headers, body)
            
            mock.add_callback(
                getattr(responses, method),
                url,
                callback=callback
            )
        
        return mock
    
    def setup_httpx_mock(self):
        """Set up httpx library mocking."""
        if not HTTPX_AVAILABLE:
            return None
        
        # This would need httpx-mock library for full implementation
        # For now, we'll use a basic mock
        mock = MagicMock()
        return mock
    
    def create_database_mock(self):
        """Create a mock database connection."""
        db_mock = MagicMock()
        
        # Mock common database operations
        db_mock.execute.return_value = None
        db_mock.fetchone.return_value = None
        db_mock.fetchall.return_value = []
        db_mock.commit.return_value = None
        db_mock.rollback.return_value = None
        db_mock.close.return_value = None
        
        return db_mock
    
    def create_redis_mock(self):
        """Create a mock Redis connection."""
        redis_mock = MagicMock()
        
        # Mock Redis operations
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True
        redis_mock.delete.return_value = 1
        redis_mock.exists.return_value = False
        redis_mock.expire.return_value = True
        redis_mock.ping.return_value = True
        
        return redis_mock
    
    def create_s3_mock(self):
        """Create a mock S3 client."""
        s3_mock = MagicMock()
        
        # Mock S3 operations
        s3_mock.upload_file.return_value = None
        s3_mock.download_file.return_value = None
        s3_mock.list_objects_v2.return_value = {"Contents": []}
        s3_mock.delete_object.return_value = None
        s3_mock.head_object.return_value = {"ContentLength": 0}
        
        return s3_mock
    
    def create_ml_model_mock(self):
        """Create a mock ML model."""
        model_mock = MagicMock()
        
        # Mock ML model operations
        model_mock.fit.return_value = None
        model_mock.predict.return_value = [0, 1, 0, 1]
        model_mock.predict_proba.return_value = [[0.9, 0.1], [0.3, 0.7]]
        model_mock.score.return_value = 0.95
        model_mock.save.return_value = None
        model_mock.load.return_value = None
        
        return model_mock
    
    @contextmanager
    def mock_external_services_context(self):
        """Context manager for mocking external services."""
        if not self.enabled:
            yield
            return
        
        patches = []
        
        try:
            # Mock requests
            if RESPONSES_AVAILABLE:
                requests_mock = self.setup_requests_mock()
                if requests_mock:
                    requests_mock.start()
                    patches.append(("requests_mock", requests_mock))
            
            # Mock database connections
            db_mock = self.create_database_mock()
            db_patch = patch('psycopg2.connect', return_value=db_mock)
            db_patch.start()
            patches.append(("db_patch", db_patch))
            
            # Mock Redis
            redis_mock = self.create_redis_mock()
            redis_patch = patch('redis.Redis', return_value=redis_mock)
            redis_patch.start()
            patches.append(("redis_patch", redis_patch))
            
            # Mock S3
            s3_mock = self.create_s3_mock()
            s3_patch = patch('boto3.client', return_value=s3_mock)
            s3_patch.start()
            patches.append(("s3_patch", s3_patch))
            
            # Mock ML models
            model_mock = self.create_ml_model_mock()
            sklearn_patch = patch('sklearn.ensemble.IsolationForest', return_value=model_mock)
            sklearn_patch.start()
            patches.append(("sklearn_patch", sklearn_patch))
            
            yield
            
        finally:
            # Clean up patches
            for name, patch_obj in patches:
                if hasattr(patch_obj, 'stop'):
                    patch_obj.stop()
                elif hasattr(patch_obj, 'reset'):
                    patch_obj.reset()


# Global service virtualization instance
service_virtualizer = ServiceVirtualization()


def setup_common_api_mocks():
    """Set up common API mocks for typical external services."""
    # Mock health check endpoint
    service_virtualizer.register_endpoint(APIEndpoint(
        method="GET",
        url="http://localhost:8000/health",
        response_body={"status": "ok", "timestamp": time.time()},
        status_code=200
    ))
    
    # Mock authentication endpoint
    service_virtualizer.register_endpoint(APIEndpoint(
        method="POST",
        url="http://localhost:8000/auth/login",
        response_body={"token": "mock_jwt_token", "expires_in": 3600},
        status_code=200
    ))
    
    # Mock data API endpoint
    service_virtualizer.register_endpoint(APIEndpoint(
        method="GET",
        url="http://localhost:8000/api/data",
        response_body={"data": [1, 2, 3, 4, 5], "count": 5},
        status_code=200
    ))
    
    # Mock external ML service
    service_virtualizer.register_endpoint(APIEndpoint(
        method="POST",
        url="http://external-ml-service.com/predict",
        response_body={"predictions": [0.1, 0.9, 0.2, 0.8], "model_version": "1.0.0"},
        status_code=200,
        delay=0.1  # Simulate network delay
    ))
    
    # Mock metrics endpoint
    service_virtualizer.register_endpoint(APIEndpoint(
        method="GET",
        url="http://localhost:8000/metrics",
        response_body={"cpu_usage": 45.2, "memory_usage": 78.5, "disk_usage": 23.1},
        status_code=200
    ))


# Pytest fixtures
@pytest.fixture(scope="function")
def mock_external_services():
    """Fixture to mock external services for a test."""
    setup_common_api_mocks()
    
    with service_virtualizer.mock_external_services_context():
        yield service_virtualizer


@pytest.fixture(scope="function")
def api_endpoint_mock():
    """Fixture to create custom API endpoint mocks."""
    def _create_mock(method: str, url: str, response_body: Any, status_code: int = 200, **kwargs):
        endpoint = APIEndpoint(
            method=method,
            url=url,
            response_body=response_body,
            status_code=status_code,
            **kwargs
        )
        service_virtualizer.register_endpoint(endpoint)
        return endpoint
    
    return _create_mock


@pytest.fixture(scope="function")
def requests_mock_fixture():
    """Fixture for requests mocking."""
    if not RESPONSES_AVAILABLE:
        pytest.skip("responses library not available")
    
    with responses.RequestsMock() as rsps:
        yield rsps


@pytest.fixture(scope="function")
def database_mock():
    """Fixture for database mocking."""
    return service_virtualizer.create_database_mock()


@pytest.fixture(scope="function")
def redis_mock():
    """Fixture for Redis mocking."""
    return service_virtualizer.create_redis_mock()


@pytest.fixture(scope="function")
def s3_mock():
    """Fixture for S3 mocking."""
    return service_virtualizer.create_s3_mock()


@pytest.fixture(scope="function")
def ml_model_mock():
    """Fixture for ML model mocking."""
    return service_virtualizer.create_ml_model_mock()


# Utility functions
def create_stable_mock_data(size: int = 100) -> Dict:
    """Create stable mock data for testing."""
    return {
        "data": list(range(size)),
        "metadata": {
            "count": size,
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "1.0.0"
        }
    }


def create_deterministic_response(seed: int = 42) -> Dict:
    """Create deterministic response data for consistent testing."""
    import random
    random.seed(seed)
    
    return {
        "predictions": [random.random() for _ in range(10)],
        "confidence": random.uniform(0.8, 0.95),
        "model_id": f"model_{seed}"
    }


# Test markers
pytest.mark.mock_external = pytest.mark.mock_external
pytest.mark.service_virtualization = pytest.mark.service_virtualization
pytest.mark.no_external_deps = pytest.mark.no_external_deps


# Example usage and test utilities
class MockedExternalService:
    """Helper class for creating mocked external services."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.endpoints = {}
    
    def add_endpoint(self, path: str, method: str = "GET", response: Any = None, status: int = 200):
        """Add a mocked endpoint."""
        full_url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        endpoint = APIEndpoint(
            method=method,
            url=full_url,
            response_body=response or {"status": "ok"},
            status_code=status
        )
        service_virtualizer.register_endpoint(endpoint)
        self.endpoints[path] = endpoint
    
    def get_call_logs(self) -> List[Dict]:
        """Get call logs for this service."""
        return [
            log for log in service_virtualizer.get_call_logs()
            if log["url"].startswith(self.base_url)
        ]


def create_mock_anomaly_detection_service():
    """Create a complete mock anomaly detection service."""
    service = MockedExternalService("http://anomaly-service.com")
    
    # Add endpoints
    service.add_endpoint("/health", "GET", {"status": "healthy"})
    service.add_endpoint("/models", "GET", {"models": ["isolation_forest", "one_class_svm"]})
    service.add_endpoint("/predict", "POST", {
        "anomalies": [False, True, False, True],
        "scores": [0.1, 0.9, 0.2, 0.8],
        "threshold": 0.5
    })
    service.add_endpoint("/train", "POST", {"model_id": "model_123", "status": "training"})
    service.add_endpoint("/models/model_123/status", "GET", {"status": "completed", "accuracy": 0.95})
    
    return service


if __name__ == "__main__":
    # Test the service virtualization
    setup_common_api_mocks()
    
    # Test mock external services
    with service_virtualizer.mock_external_services_context():
        if RESPONSES_AVAILABLE:
            import requests
            
            # Test a mocked endpoint
            response = requests.get("http://localhost:8000/health")
            print(f"Health check response: {response.json()}")
            
            # Test call logging
            logs = service_virtualizer.get_call_logs()
            print(f"API call logs: {logs}")
        
        # Test database mock
        db_mock = service_virtualizer.create_database_mock()
        result = db_mock.execute("SELECT * FROM users")
        print(f"Database query result: {result}")
        
        # Test ML model mock
        model_mock = service_virtualizer.create_ml_model_mock()
        predictions = model_mock.predict([[1, 2, 3]])
        print(f"ML predictions: {predictions}")
    
    print("Service virtualization test completed")
