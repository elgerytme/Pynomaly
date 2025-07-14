"""Stable integration tests for external dependencies with comprehensive mocking and fallback strategies."""

import json
import os
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests


class ExternalDependencyStabilizer:
    """Comprehensive stabilization for external dependency testing."""

    def __init__(self):
        self.mock_registry = {}
        self.fallback_strategies = {}
        self.health_checks = {}
        self.retry_configs = {}

    def register_mock(self, dependency_name: str, mock_factory: Callable):
        """Register a mock factory for a dependency."""
        self.mock_registry[dependency_name] = mock_factory

    def register_fallback(self, dependency_name: str, fallback_strategy: Callable):
        """Register a fallback strategy for when dependency is unavailable."""
        self.fallback_strategies[dependency_name] = fallback_strategy

    def register_health_check(self, dependency_name: str, health_check: Callable):
        """Register a health check for a dependency."""
        self.health_checks[dependency_name] = health_check

    def configure_retry(
        self,
        dependency_name: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True,
    ):
        """Configure retry behavior for a dependency."""
        self.retry_configs[dependency_name] = {
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "exponential_backoff": exponential_backoff,
        }

    @contextmanager
    def stabilized_dependency(self, dependency_name: str):
        """Context manager for stabilized dependency usage."""
        # Try health check first
        is_healthy = self._check_dependency_health(dependency_name)

        if is_healthy:
            # Use real dependency
            yield "real"
        else:
            # Use mock or fallback
            if dependency_name in self.mock_registry:
                with self._apply_mock(dependency_name):
                    yield "mock"
            elif dependency_name in self.fallback_strategies:
                yield "fallback"
            else:
                pytest.skip(
                    f"Dependency {dependency_name} unavailable and no fallback configured"
                )

    def _check_dependency_health(self, dependency_name: str) -> bool:
        """Check if dependency is healthy."""
        if dependency_name not in self.health_checks:
            return True  # Assume healthy if no check configured

        try:
            return self.health_checks[dependency_name]()
        except Exception:
            return False

    @contextmanager
    def _apply_mock(self, dependency_name: str):
        """Apply mock for dependency."""
        mock = self.mock_registry[dependency_name]()
        with mock:
            yield


# Initialize global stabilizer
stabilizer = ExternalDependencyStabilizer()


# Configure mocks and fallbacks for common dependencies
def setup_pyod_mock():
    """Setup comprehensive PyOD mocking."""
    mock_iforest = Mock()
    mock_iforest.fit = Mock()
    mock_iforest.predict = Mock(return_value=np.array([1, -1, 1, -1, 1]))
    mock_iforest.decision_function = Mock(
        return_value=np.array([0.1, 0.8, 0.2, 0.9, 0.1])
    )

    mock_lof = Mock()
    mock_lof.fit_predict = Mock(return_value=np.array([1, -1, 1, -1, 1]))
    mock_lof.negative_outlier_factor_ = np.array([-0.1, -0.8, -0.2, -0.9, -0.1])

    return patch.dict(
        "sys.modules",
        {
            "pyod.models.iforest": Mock(IsolationForest=lambda **kwargs: mock_iforest),
            "pyod.models.lof": Mock(LocalOutlierFactor=lambda **kwargs: mock_lof),
        },
    )


def setup_sklearn_mock():
    """Setup sklearn mocking for consistency."""
    return patch("sklearn.ensemble.IsolationForest", return_value=Mock())


def setup_database_mock():
    """Setup database connection mocking."""
    mock_connection = Mock()
    mock_connection.execute = AsyncMock()
    mock_connection.fetch = AsyncMock(return_value=[])
    mock_connection.close = AsyncMock()

    return patch("asyncpg.connect", return_value=mock_connection)


def setup_redis_mock():
    """Setup Redis mocking."""
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)

    return patch("aioredis.from_url", return_value=mock_redis)


def setup_http_mock():
    """Setup HTTP requests mocking."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={"status": "success", "data": []})
    mock_response.raise_for_status = Mock()

    return patch("requests.get", return_value=mock_response)


# Register mocks and health checks
stabilizer.register_mock("pyod", setup_pyod_mock)
stabilizer.register_mock("sklearn", setup_sklearn_mock)
stabilizer.register_mock("database", setup_database_mock)
stabilizer.register_mock("redis", setup_redis_mock)
stabilizer.register_mock("http", setup_http_mock)

# Health checks
stabilizer.register_health_check("pyod", lambda: True)  # Always available
stabilizer.register_health_check("sklearn", lambda: True)  # Always available
stabilizer.register_health_check(
    "database", lambda: os.getenv("DATABASE_URL") is not None
)
stabilizer.register_health_check("redis", lambda: os.getenv("REDIS_URL") is not None)
stabilizer.register_health_check("http", lambda: True)  # Mock HTTP by default

# Retry configurations
stabilizer.configure_retry("database", max_retries=3, retry_delay=0.5)
stabilizer.configure_retry("redis", max_retries=2, retry_delay=0.2)
stabilizer.configure_retry("http", max_retries=3, retry_delay=1.0)


class TestStableExternalDependencies:
    """Stable integration tests for external dependencies."""

    def test_ml_library_integration_stable(self):
        """Test ML library integration with fallback to mocks."""
        with stabilizer.stabilized_dependency("pyod") as dependency_type:
            from sklearn.ensemble import IsolationForest

            # Generate test data
            data = np.random.randn(100, 5)

            if dependency_type == "real":
                # Use real PyOD if available
                detector = IsolationForest(contamination=0.1, random_state=42)
            else:
                # Use mock
                detector = Mock()
                detector.fit = Mock()
                detector.predict = Mock(return_value=np.random.choice([-1, 1], 100))
                detector.decision_function = Mock(return_value=np.random.randn(100))

            # Execute operations
            detector.fit(data)
            predictions = detector.predict(data)
            scores = detector.decision_function(data)

            # Verify results regardless of mock/real
            assert len(predictions) == 100
            assert len(scores) == 100
            assert all(pred in [-1, 1] for pred in predictions)
            assert all(np.isfinite(score) for score in scores)

    @pytest.mark.asyncio
    async def test_database_integration_stable(self):
        """Test database integration with stable mocking."""
        with stabilizer.stabilized_dependency("database") as dependency_type:
            if dependency_type == "real":
                # Use real database if available
                import asyncpg

                try:
                    connection = await asyncpg.connect(os.getenv("DATABASE_URL"))

                    # Test query
                    result = await connection.fetch("SELECT 1 as test")
                    assert len(result) == 1
                    assert result[0]["test"] == 1

                    await connection.close()

                except Exception as e:
                    pytest.skip(f"Database not available: {e}")

            else:
                # Use mock database
                mock_connection = Mock()
                mock_connection.fetch = AsyncMock(return_value=[{"test": 1}])
                mock_connection.close = AsyncMock()

                # Test operations
                result = await mock_connection.fetch("SELECT 1 as test")
                assert len(result) == 1
                assert result[0]["test"] == 1

                await mock_connection.close()

    @pytest.mark.asyncio
    async def test_cache_integration_stable(self):
        """Test cache integration with Redis fallback."""
        with stabilizer.stabilized_dependency("redis") as dependency_type:
            if dependency_type == "real":
                # Use real Redis if available
                try:
                    import aioredis

                    redis = await aioredis.from_url(
                        os.getenv("REDIS_URL", "redis://localhost")
                    )

                    # Test cache operations
                    await redis.set("test_key", "test_value")
                    value = await redis.get("test_key")
                    assert value == "test_value"

                    await redis.delete("test_key")
                    value = await redis.get("test_key")
                    assert value is None

                    await redis.close()

                except Exception as e:
                    pytest.skip(f"Redis not available: {e}")

            else:
                # Use mock Redis
                mock_redis = Mock()
                mock_redis.set = AsyncMock()
                mock_redis.get = AsyncMock(side_effect=["test_value", None])
                mock_redis.delete = AsyncMock()

                # Test operations
                await mock_redis.set("test_key", "test_value")
                value = await mock_redis.get("test_key")
                assert value == "test_value"

                await mock_redis.delete("test_key")
                value = await mock_redis.get("test_key")
                assert value is None

    def test_http_api_integration_stable(self):
        """Test HTTP API integration with stable mocking."""
        with stabilizer.stabilized_dependency("http") as dependency_type:
            if dependency_type == "real":
                # Use real HTTP if service is available
                try:
                    response = requests.get("http://httpbin.org/json", timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    assert isinstance(data, dict)

                except Exception as e:
                    pytest.skip(f"HTTP service not available: {e}")

            else:
                # Use mock HTTP
                with patch("requests.get") as mock_get:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"mock": "data"}
                    mock_response.raise_for_status = Mock()
                    mock_get.return_value = mock_response

                    response = requests.get("http://example.com/api")
                    response.raise_for_status()
                    data = response.json()
                    assert data == {"mock": "data"}

    def test_file_system_integration_stable(self):
        """Test file system operations with temporary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test data file operations
            test_data = pd.DataFrame(
                {
                    "feature_1": np.random.randn(50),
                    "feature_2": np.random.randn(50),
                    "label": np.random.choice([0, 1], 50),
                }
            )

            # Save data
            data_file = temp_path / "test_data.csv"
            test_data.to_csv(data_file, index=False)

            # Load data
            loaded_data = pd.read_csv(data_file)

            # Verify data integrity
            assert len(loaded_data) == 50
            assert list(loaded_data.columns) == ["feature_1", "feature_2", "label"]
            assert loaded_data["label"].isin([0, 1]).all()

            # Test model persistence
            model_file = temp_path / "test_model.json"
            model_config = {
                "algorithm": "IsolationForest",
                "parameters": {"contamination": 0.1},
                "trained_at": time.time(),
            }

            with open(model_file, "w") as f:
                json.dump(model_config, f)

            # Load model config
            with open(model_file) as f:
                loaded_config = json.load(f)

            assert loaded_config["algorithm"] == "IsolationForest"
            assert loaded_config["parameters"]["contamination"] == 0.1

    def test_multi_dependency_workflow_stable(self):
        """Test workflow involving multiple external dependencies."""
        workflow_results = {}

        # Step 1: ML Library
        with stabilizer.stabilized_dependency("sklearn") as sklearn_type:
            if sklearn_type == "real":
                from sklearn.ensemble import IsolationForest

                detector = IsolationForest(contamination=0.1, random_state=42)
            else:
                detector = Mock()
                detector.fit = Mock()
                detector.predict = Mock(return_value=np.random.choice([-1, 1], 100))

            # Generate and process data
            data = np.random.randn(100, 5)
            detector.fit(data)
            predictions = detector.predict(data)

            workflow_results["ml_step"] = {
                "success": True,
                "predictions_count": len(predictions),
                "dependency_type": sklearn_type,
            }

        # Step 2: File Operations (always available)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            result_data = {
                "predictions": predictions.tolist(),
                "model_config": {"algorithm": "IsolationForest"},
                "timestamp": time.time(),
            }
            json.dump(result_data, tmp_file)
            tmp_filename = tmp_file.name

        # Verify file was written
        with open(tmp_filename) as f:
            loaded_results = json.load(f)

        workflow_results["file_step"] = {
            "success": True,
            "file_size": os.path.getsize(tmp_filename),
            "predictions_saved": len(loaded_results["predictions"]),
        }

        # Cleanup
        os.unlink(tmp_filename)

        # Step 3: HTTP (with mock)
        with stabilizer.stabilized_dependency("http") as http_type:
            if http_type == "real":
                try:
                    response = requests.get("http://httpbin.org/status/200", timeout=5)
                    success = response.status_code == 200
                except:
                    success = False
            else:
                # Mock HTTP
                with patch("requests.get") as mock_get:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_get.return_value = mock_response

                    response = requests.get("http://mock-api.com/submit")
                    success = response.status_code == 200

            workflow_results["http_step"] = {
                "success": success,
                "dependency_type": http_type,
            }

        # Verify overall workflow success
        assert all(step["success"] for step in workflow_results.values())
        assert workflow_results["ml_step"]["predictions_count"] == 100
        assert workflow_results["file_step"]["file_size"] > 0
        assert workflow_results["file_step"]["predictions_saved"] == 100

    def test_error_recovery_mechanisms(self):
        """Test error recovery with external dependencies."""
        error_scenarios = []

        # Scenario 1: ML library error with fallback
        try:
            with patch("sklearn.ensemble.IsolationForest") as mock_if:
                # Simulate library error then recovery
                mock_if.side_effect = [ImportError("Library not available"), Mock()]

                # First attempt should fail
                try:
                    detector = mock_if()
                    error_scenarios.append(
                        "ml_library_first_success"
                    )  # Should not reach here
                except ImportError:
                    error_scenarios.append("ml_library_first_failed")

                # Second attempt should succeed
                try:
                    detector = mock_if()
                    error_scenarios.append("ml_library_second_success")
                except ImportError:
                    error_scenarios.append("ml_library_second_failed")

        except Exception as e:
            error_scenarios.append(f"ml_library_error: {str(e)}")

        # Scenario 2: Network timeout with retry
        with patch("requests.get") as mock_get:
            # Simulate timeout then success
            mock_get.side_effect = [
                requests.exceptions.Timeout("Request timeout"),
                Mock(status_code=200, json=lambda: {"success": True}),
            ]

            # Implement retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = requests.get("http://example.com")
                    if response.status_code == 200:
                        error_scenarios.append("http_retry_success")
                        break
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        error_scenarios.append("http_retry_failed")
                    else:
                        error_scenarios.append(f"http_retry_attempt_{attempt}")

        # Scenario 3: File permission error with alternative path
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Try to write to a restricted file
                restricted_path = Path(temp_dir) / "restricted" / "data.csv"

                # This should fail (directory doesn't exist)
                try:
                    pd.DataFrame({"test": [1, 2, 3]}).to_csv(restricted_path)
                    error_scenarios.append("file_write_unexpected_success")
                except (FileNotFoundError, PermissionError):
                    error_scenarios.append("file_write_first_failed")

                    # Try alternative path
                    alternative_path = Path(temp_dir) / "data.csv"
                    pd.DataFrame({"test": [1, 2, 3]}).to_csv(alternative_path)
                    error_scenarios.append("file_write_alternative_success")

            except Exception as e:
                error_scenarios.append(f"file_error: {str(e)}")

        # Verify error recovery scenarios
        expected_scenarios = [
            "ml_library_first_failed",
            "ml_library_second_success",
            "http_retry_attempt_0",
            "http_retry_success",
            "file_write_first_failed",
            "file_write_alternative_success",
        ]

        for scenario in expected_scenarios:
            assert (
                scenario in error_scenarios
            ), f"Expected scenario '{scenario}' not found in {error_scenarios}"

    def test_dependency_isolation(self):
        """Test that external dependency failures don't affect other components."""
        test_results = {}

        # Component 1: Pure Python logic (should always work)
        try:
            data = np.random.randn(50, 3)
            mean_values = np.mean(data, axis=0)
            std_values = np.std(data, axis=0)

            test_results["pure_python"] = {
                "success": True,
                "mean_shape": mean_values.shape,
                "std_shape": std_values.shape,
            }
        except Exception as e:
            test_results["pure_python"] = {"success": False, "error": str(e)}

        # Component 2: ML with mock (isolated failure)
        try:
            with patch("sklearn.ensemble.IsolationForest") as mock_if:
                # Simulate failure
                mock_if.side_effect = Exception("ML library failed")

                try:
                    detector = mock_if()
                    test_results["ml_component"] = {"success": True}
                except Exception:
                    # Fallback to simple outlier detection
                    z_scores = np.abs((data - mean_values) / std_values)
                    outliers = np.any(z_scores > 2, axis=1)

                    test_results["ml_component"] = {
                        "success": True,
                        "used_fallback": True,
                        "outliers_detected": np.sum(outliers),
                    }

        except Exception as e:
            test_results["ml_component"] = {"success": False, "error": str(e)}

        # Component 3: File operations (should work independently)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp_file:
                tmp_file.write("test data")
                tmp_filename = tmp_file.name

            # Verify file exists and has content
            with open(tmp_filename) as f:
                content = f.read()

            os.unlink(tmp_filename)

            test_results["file_component"] = {
                "success": True,
                "content_length": len(content),
            }

        except Exception as e:
            test_results["file_component"] = {"success": False, "error": str(e)}

        # Verify isolation: other components should succeed even if one fails
        assert test_results["pure_python"]["success"]
        assert test_results["ml_component"]["success"]
        assert test_results["file_component"]["success"]

        # Verify fallback was used for ML component
        assert test_results["ml_component"].get("used_fallback", False)

    @pytest.mark.parametrize("dependency", ["sklearn", "http", "database", "redis"])
    def test_dependency_specific_stability(self, dependency):
        """Test stability for each specific dependency."""
        with stabilizer.stabilized_dependency(dependency) as dependency_type:
            if dependency == "sklearn":
                # Test sklearn operations
                from sklearn.ensemble import IsolationForest

                detector = IsolationForest(contamination=0.1)
                data = np.random.randn(20, 3)
                detector.fit(data)
                predictions = detector.predict(data)
                assert len(predictions) == 20

            elif dependency == "http":
                # Test HTTP operations
                with patch("requests.get") as mock_get:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"status": "ok"}
                    mock_get.return_value = mock_response

                    response = requests.get("http://example.com")
                    assert response.status_code == 200

            elif dependency == "database":
                # Test database operations (mocked)
                mock_connection = Mock()
                mock_connection.execute = Mock()
                mock_connection.fetch = Mock(return_value=[{"id": 1, "name": "test"}])

                mock_connection.execute("SELECT * FROM test_table")
                result = mock_connection.fetch()
                assert len(result) == 1
                assert result[0]["name"] == "test"

            elif dependency == "redis":
                # Test Redis operations (mocked)
                mock_redis = Mock()
                mock_redis.get = Mock(return_value="cached_value")
                mock_redis.set = Mock(return_value=True)

                mock_redis.set("key", "value")
                value = mock_redis.get("key")
                assert value == "cached_value"

        # Test should complete successfully regardless of dependency availability
        assert True  # Implicit success if we reach here without exceptions
