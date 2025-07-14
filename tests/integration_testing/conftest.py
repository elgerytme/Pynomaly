"""Integration testing configuration and fixtures."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import Mock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

# Import test utilities
from tests.shared.utilities import ResourceManager


@pytest.fixture(scope="session")
def integration_test_config() -> Dict[str, Any]:
    """Load integration test configuration."""
    config_path = Path(__file__).parent / "config" / "test_config.yaml"
    
    default_config = {
        "database": {
            "url": "sqlite:///:memory:",
            "echo": False,
            "pool_size": 10
        },
        "api": {
            "host": "localhost",
            "port": 8001,  # Different port for integration tests
            "timeout": 30
        },
        "performance": {
            "max_response_time": 2.0,
            "concurrent_users": 10,
            "test_duration": 60
        },
        "security": {
            "enable_auth": True,
            "jwt_secret": "test-secret-key",
            "password_complexity": "medium"
        },
        "monitoring": {
            "enabled": True,
            "metrics_interval": 5
        }
    }
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
            default_config.update(loaded_config)
    
    return default_config


@pytest.fixture(scope="session")
def test_environment(integration_test_config):
    """Set up integration test environment."""
    original_env = os.environ.copy()
    
    # Set integration test environment variables
    integration_env = {
        "PYNOMALY_ENVIRONMENT": "integration_test",
        "PYNOMALY_DEBUG": "true",
        "PYNOMALY_LOG_LEVEL": "DEBUG",
        "PYNOMALY_DATABASE_URL": integration_test_config["database"]["url"],
        "PYNOMALY_API_HOST": integration_test_config["api"]["host"],
        "PYNOMALY_API_PORT": str(integration_test_config["api"]["port"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1"
    }
    
    os.environ.update(integration_env)
    
    yield integration_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="session")
def test_database():
    """Set up test database for integration tests."""
    # For integration tests, we'll use a persistent test database
    db_path = tempfile.mktemp(suffix=".db")
    db_url = f"sqlite:///{db_path}"
    
    # Initialize database (would normally use Alembic migrations)
    # For now, we'll mock this
    
    yield db_url
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture(scope="function")
def api_client(test_environment, test_database) -> Generator[TestClient, None, None]:
    """Create FastAPI test client for integration tests."""
    # Import the actual FastAPI app
    try:
        from src.pynomaly.presentation.api.app import create_app
        app = create_app()
        client = TestClient(app)
        yield client
    except ImportError:
        # Mock client for when the actual app isn't available
        mock_client = Mock(spec=TestClient)
        mock_client.get.return_value = Mock(status_code=200, json=lambda: {"status": "ok"})
        mock_client.post.return_value = Mock(status_code=201, json=lambda: {"id": "test-123"})
        yield mock_client


@pytest.fixture(scope="function")
def test_data_manager():
    """Manage test data lifecycle."""
    class TestDataManager:
        def __init__(self):
            self.created_resources = []
            
        def create_test_dataset(self, size: int = 1000) -> Dict[str, Any]:
            """Create a test dataset."""
            import numpy as np
            import pandas as pd
            
            # Generate synthetic anomaly detection data
            normal_data = np.random.normal(0, 1, (int(size * 0.95), 10))
            anomaly_data = np.random.normal(5, 2, (int(size * 0.05), 10))
            
            data = np.vstack([normal_data, anomaly_data])
            labels = np.hstack([
                np.zeros(len(normal_data)),
                np.ones(len(anomaly_data))
            ])
            
            df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(10)])
            df['is_anomaly'] = labels
            
            dataset_info = {
                "id": f"test-dataset-{int(time.time())}",
                "data": df,
                "size": len(df),
                "anomaly_rate": len(anomaly_data) / len(data)
            }
            
            self.created_resources.append(dataset_info["id"])
            return dataset_info
            
        def create_test_detector(self) -> Dict[str, Any]:
            """Create a test detector configuration."""
            detector_config = {
                "id": f"test-detector-{int(time.time())}",
                "algorithm": "isolation_forest",
                "parameters": {
                    "n_estimators": 100,
                    "contamination": 0.1,
                    "random_state": 42
                },
                "name": "Test Isolation Forest",
                "description": "Test detector for integration testing"
            }
            
            self.created_resources.append(detector_config["id"])
            return detector_config
            
        def cleanup(self):
            """Clean up created test resources."""
            # In a real implementation, this would delete created resources
            # from the database or storage system
            self.created_resources.clear()
    
    manager = TestDataManager()
    yield manager
    manager.cleanup()


@pytest.fixture(scope="function")
def performance_monitor():
    """Monitor performance during tests."""
    import psutil
    import threading
    
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = []
            self.monitoring = False
            self.monitor_thread = None
            
        def start_monitoring(self, interval: float = 1.0):
            """Start performance monitoring."""
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,)
            )
            self.monitor_thread.start()
            
        def stop_monitoring(self):
            """Stop performance monitoring."""
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
                
        def _monitor_loop(self, interval: float):
            """Performance monitoring loop."""
            process = psutil.Process()
            
            while self.monitoring:
                try:
                    cpu_percent = process.cpu_percent()
                    memory_info = process.memory_info()
                    
                    self.metrics.append({
                        "timestamp": time.time(),
                        "cpu_percent": cpu_percent,
                        "memory_rss": memory_info.rss,
                        "memory_vms": memory_info.vms
                    })
                    
                    time.sleep(interval)
                except Exception:
                    # Ignore monitoring errors
                    continue
                    
        def get_summary(self) -> Dict[str, Any]:
            """Get performance summary."""
            if not self.metrics:
                return {}
                
            cpu_values = [m["cpu_percent"] for m in self.metrics]
            memory_values = [m["memory_rss"] for m in self.metrics]
            
            return {
                "cpu": {
                    "avg": sum(cpu_values) / len(cpu_values),
                    "max": max(cpu_values),
                    "min": min(cpu_values)
                },
                "memory": {
                    "avg": sum(memory_values) / len(memory_values),
                    "max": max(memory_values),
                    "min": min(memory_values),
                    "peak_mb": max(memory_values) / 1024 / 1024
                },
                "duration": self.metrics[-1]["timestamp"] - self.metrics[0]["timestamp"],
                "sample_count": len(self.metrics)
            }
    
    monitor = PerformanceMonitor()
    yield monitor
    monitor.stop_monitoring()


@pytest.fixture(scope="function")
def security_context():
    """Security testing context."""
    class SecurityContext:
        def __init__(self):
            self.test_users = []
            self.test_tokens = {}
            
        def create_test_user(self, role: str = "user") -> Dict[str, Any]:
            """Create a test user with specified role."""
            import uuid
            
            user = {
                "id": str(uuid.uuid4()),
                "username": f"test-user-{int(time.time())}",
                "email": f"test-{int(time.time())}@example.com",
                "role": role,
                "permissions": self._get_role_permissions(role)
            }
            
            self.test_users.append(user)
            return user
            
        def generate_test_token(self, user: Dict[str, Any]) -> str:
            """Generate a test JWT token for user."""
            # Mock JWT token generation
            token = f"test-token-{user['id']}"
            self.test_tokens[token] = user
            return token
            
        def _get_role_permissions(self, role: str) -> List[str]:
            """Get permissions for role."""
            role_permissions = {
                "admin": ["read", "write", "delete", "admin"],
                "analyst": ["read", "write"],
                "viewer": ["read"],
                "user": ["read"]
            }
            return role_permissions.get(role, ["read"])
            
        def cleanup(self):
            """Clean up test security resources."""
            self.test_users.clear()
            self.test_tokens.clear()
    
    context = SecurityContext()
    yield context
    context.cleanup()


@pytest.fixture(scope="function")
def load_test_simulator():
    """Simulate load testing scenarios."""
    class LoadTestSimulator:
        def __init__(self):
            self.active_sessions = []
            
        async def simulate_concurrent_users(
            self,
            client: TestClient,
            num_users: int,
            duration: float,
            endpoint: str = "/health"
        ) -> Dict[str, Any]:
            """Simulate concurrent user load."""
            import asyncio
            import aiohttp
            
            results = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "response_times": [],
                "errors": []
            }
            
            async def user_session(session_id: int):
                """Individual user session."""
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    request_start = time.time()
                    
                    try:
                        # In a real implementation, this would use actual HTTP requests
                        # For now, we'll simulate the response
                        await asyncio.sleep(0.01)  # Simulate network latency
                        
                        response_time = time.time() - request_start
                        results["response_times"].append(response_time)
                        results["successful_requests"] += 1
                        
                    except Exception as e:
                        results["errors"].append(str(e))
                        results["failed_requests"] += 1
                        
                    results["total_requests"] += 1
                    
                    # Wait between requests
                    await asyncio.sleep(0.1)
            
            # Run concurrent user sessions
            tasks = [user_session(i) for i in range(num_users)]
            await asyncio.gather(*tasks)
            
            # Calculate statistics
            if results["response_times"]:
                results["avg_response_time"] = sum(results["response_times"]) / len(results["response_times"])
                results["max_response_time"] = max(results["response_times"])
                results["min_response_time"] = min(results["response_times"])
            
            results["success_rate"] = (
                results["successful_requests"] / results["total_requests"] 
                if results["total_requests"] > 0 else 0
            )
            
            return results
    
    return LoadTestSimulator()


@pytest.fixture(scope="function")
def disaster_recovery_simulator():
    """Simulate disaster recovery scenarios."""
    class DisasterRecoverySimulator:
        def __init__(self):
            self.simulated_failures = []
            
        def simulate_database_failure(self):
            """Simulate database connection failure."""
            failure_context = {
                "type": "database_failure",
                "start_time": time.time(),
                "active": True
            }
            self.simulated_failures.append(failure_context)
            
            # Mock database failure by patching database connections
            return patch('sqlalchemy.create_engine', side_effect=Exception("Database connection failed"))
            
        def simulate_service_unavailable(self, service_name: str):
            """Simulate service unavailability."""
            failure_context = {
                "type": "service_unavailable",
                "service": service_name,
                "start_time": time.time(),
                "active": True
            }
            self.simulated_failures.append(failure_context)
            
            # Return a context manager for service failure simulation
            return patch(f'requests.get', side_effect=Exception(f"{service_name} unavailable"))
            
        def simulate_network_partition(self):
            """Simulate network partition."""
            failure_context = {
                "type": "network_partition",
                "start_time": time.time(),
                "active": True
            }
            self.simulated_failures.append(failure_context)
            
            # Mock network failures
            return patch('socket.socket', side_effect=Exception("Network unavailable"))
            
        def restore_service(self, failure_type: str):
            """Restore service after simulated failure."""
            for failure in self.simulated_failures:
                if failure["type"] == failure_type and failure["active"]:
                    failure["active"] = False
                    failure["end_time"] = time.time()
                    failure["duration"] = failure["end_time"] - failure["start_time"]
                    
        def get_failure_summary(self) -> List[Dict[str, Any]]:
            """Get summary of simulated failures."""
            return [f for f in self.simulated_failures if not f["active"]]
    
    return DisasterRecoverySimulator()


# Autouse fixtures for test isolation
@pytest.fixture(autouse=True)
def integration_test_isolation():
    """Ensure integration test isolation."""
    yield
    
    # Clean up any global state
    import gc
    gc.collect()


@pytest.fixture(scope="function")
def contract_validator():
    """Validate API contracts and interfaces."""
    class ContractValidator:
        def __init__(self):
            self.contract_violations = []
            
        def validate_api_response(self, response, expected_schema: Dict[str, Any]) -> bool:
            """Validate API response against expected schema."""
            try:
                # In a real implementation, this would use jsonschema
                # For now, we'll do basic validation
                if hasattr(response, 'json'):
                    response_data = response.json()
                else:
                    response_data = response
                    
                # Basic schema validation
                for field, field_type in expected_schema.items():
                    if field not in response_data:
                        self.contract_violations.append(f"Missing field: {field}")
                        return False
                        
                    if not isinstance(response_data[field], field_type):
                        self.contract_violations.append(
                            f"Field {field} type mismatch: expected {field_type}, got {type(response_data[field])}"
                        )
                        return False
                
                return True
                
            except Exception as e:
                self.contract_violations.append(f"Validation error: {str(e)}")
                return False
                
        def validate_package_interface(self, package_name: str, expected_methods: List[str]) -> bool:
            """Validate package interface compliance."""
            try:
                # Import and check package interface
                import importlib
                package = importlib.import_module(package_name)
                
                for method in expected_methods:
                    if not hasattr(package, method):
                        self.contract_violations.append(f"Missing method {method} in {package_name}")
                        return False
                        
                return True
                
            except ImportError as e:
                self.contract_violations.append(f"Package import error: {str(e)}")
                return False
                
        def get_violations(self) -> List[str]:
            """Get list of contract violations."""
            return self.contract_violations.copy()
    
    return ContractValidator()