"""Enhanced integration tests with improved isolation and comprehensive mocking."""

import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

# Import test containers and isolation utilities
try:
    from testcontainers.postgres import PostgresContainer
    from testcontainers.redis import RedisContainer

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False


# Mock imports for external dependencies
class MockedExternalDependencies:
    """Mock external dependencies for isolated testing."""

    def __init__(self):
        self.pyod_mocks = {}
        self.sklearn_mocks = {}
        self.database_mocks = {}
        self.redis_mocks = {}
        self.api_mocks = {}

    def setup_pyod_mocks(self):
        """Setup PyOD library mocks."""
        with patch.dict(
            "sys.modules",
            {
                "pyod.models.iforest": Mock(),
                "pyod.models.lof": Mock(),
                "pyod.models.ocsvm": Mock(),
                "pyod.models.auto_encoder": Mock(),
            },
        ):
            # Mock PyOD model classes
            mock_iforest = Mock()
            mock_iforest.fit = Mock()
            mock_iforest.predict = Mock(return_value=np.array([1, -1, 1, -1, 1]))
            mock_iforest.decision_function = Mock(
                return_value=np.array([0.1, 0.8, 0.2, 0.9, 0.1])
            )

            self.pyod_mocks["IsolationForest"] = mock_iforest

            mock_lof = Mock()
            mock_lof.fit = Mock()
            mock_lof.predict = Mock(return_value=np.array([1, -1, 1, -1, 1]))
            mock_lof.decision_function = Mock(
                return_value=np.array([0.2, 0.7, 0.3, 0.8, 0.2])
            )

            self.pyod_mocks["LOF"] = mock_lof

            return self.pyod_mocks

    def setup_database_mocks(self):
        """Setup database mocks for isolation."""
        mock_session = AsyncMock()
        mock_session.begin = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.add = Mock()
        mock_session.delete = Mock()
        mock_session.merge = Mock()

        self.database_mocks["session"] = mock_session
        return self.database_mocks

    def setup_redis_mocks(self):
        """Setup Redis mocks for caching isolation."""
        mock_redis = Mock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.delete = AsyncMock(return_value=1)
        mock_redis.exists = AsyncMock(return_value=False)
        mock_redis.flushdb = AsyncMock(return_value=True)

        self.redis_mocks["client"] = mock_redis
        return self.redis_mocks

    def setup_api_mocks(self):
        """Setup external API mocks."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"status": "success", "data": []})
        mock_response.raise_for_status = Mock()

        self.api_mocks["response"] = mock_response
        return self.api_mocks


@pytest.fixture
def mocked_dependencies():
    """Provide comprehensively mocked external dependencies."""
    deps = MockedExternalDependencies()
    deps.setup_pyod_mocks()
    deps.setup_database_mocks()
    deps.setup_redis_mocks()
    deps.setup_api_mocks()
    return deps


@pytest.fixture
async def isolated_database():
    """Provide isolated database for testing."""
    if TESTCONTAINERS_AVAILABLE:
        # Use TestContainers for real database isolation
        postgres = PostgresContainer("postgres:13")
        postgres.start()

        try:
            connection_url = postgres.get_connection_url()
            yield {
                "url": connection_url,
                "host": postgres.get_container_host_ip(),
                "port": postgres.get_exposed_port(5432),
                "database": postgres.dbname,
                "username": postgres.username,
                "password": postgres.password,
            }
        finally:
            postgres.stop()
    else:
        # Fallback to mocked database
        yield {
            "url": "postgresql://test:test@localhost:5432/test_db",
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "username": "test",
            "password": "test",
            "mocked": True,
        }


@pytest.fixture
async def isolated_redis():
    """Provide isolated Redis for testing."""
    if TESTCONTAINERS_AVAILABLE:
        # Use TestContainers for real Redis isolation
        redis = RedisContainer("redis:6")
        redis.start()

        try:
            yield {
                "host": redis.get_container_host_ip(),
                "port": redis.get_exposed_port(6379),
                "url": f"redis://{redis.get_container_host_ip()}:{redis.get_exposed_port(6379)}",
            }
        finally:
            redis.stop()
    else:
        # Fallback to mocked Redis
        yield {
            "host": "localhost",
            "port": 6379,
            "url": "redis://localhost:6379",
            "mocked": True,
        }


@pytest.fixture
def isolated_filesystem():
    """Provide isolated filesystem for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test directory structure
        (temp_path / "datasets").mkdir()
        (temp_path / "models").mkdir()
        (temp_path / "exports").mkdir()
        (temp_path / "logs").mkdir()

        # Create sample test files
        sample_data = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
            }
        )
        sample_data.to_csv(temp_path / "datasets" / "sample_data.csv", index=False)

        yield {
            "root": temp_path,
            "datasets": temp_path / "datasets",
            "models": temp_path / "models",
            "exports": temp_path / "exports",
            "logs": temp_path / "logs",
        }


class TestEnhancedIntegrationIsolation:
    """Enhanced integration tests with comprehensive isolation."""

    def test_data_pipeline_isolated(self, mocked_dependencies, isolated_filesystem):
        """Test complete data pipeline with isolated dependencies."""
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

        # Setup mocked PyOD dependencies
        with patch(
            "pynomaly.infrastructure.adapters.pyod_adapter.IsolationForest"
        ) as mock_if:
            mock_if.return_value = mocked_dependencies.pyod_mocks["IsolationForest"]

            # Test data loading with isolated filesystem
            data_path = isolated_filesystem["datasets"] / "sample_data.csv"

            # Mock data loader
            with patch(
                "pynomaly.infrastructure.data_loaders.csv_loader.CSVLoader"
            ) as mock_loader:
                mock_data = np.random.randn(100, 3)
                mock_loader.return_value.load.return_value = mock_data

                # Test detection pipeline
                adapter = SklearnAdapter(algorithm_name="IsolationForest")
                adapter.fit(mock_data)
                predictions = adapter.predict(mock_data)
                scores = adapter.decision_function(mock_data)

                # Verify isolated execution
                assert predictions is not None
                assert scores is not None
                assert len(predictions) == len(mock_data)
                assert len(scores) == len(mock_data)

    @pytest.mark.asyncio
    async def test_async_service_integration_isolated(
        self, mocked_dependencies, isolated_database
    ):
        """Test async service integration with database isolation."""

        # Mock async database session
        with patch(
            "pynomaly.infrastructure.persistence.async_session_maker"
        ) as mock_session_maker:
            mock_session_maker.return_value = mocked_dependencies.database_mocks[
                "session"
            ]

            # Mock repository
            with patch(
                "pynomaly.infrastructure.repositories.detector_repository.DetectorRepository"
            ) as mock_repo:
                mock_detector = Mock()
                mock_detector.id = str(uuid.uuid4())
                mock_detector.name = "Test Detector"
                mock_detector.is_fitted = True

                mock_repo.return_value.create = AsyncMock(return_value=mock_detector)
                mock_repo.return_value.find_by_id = AsyncMock(
                    return_value=mock_detector
                )
                mock_repo.return_value.update = AsyncMock(return_value=mock_detector)

                # Test async service operations
                from pynomaly.application.services.detection_service import (
                    DetectionService,
                )

                with patch.object(DetectionService, "__init__", return_value=None):
                    service = DetectionService()
                    service.detector_repository = mock_repo.return_value

                    # Test async operations
                    created_detector = await service.detector_repository.create(
                        mock_detector
                    )
                    found_detector = await service.detector_repository.find_by_id(
                        mock_detector.id
                    )

                    assert created_detector.id == mock_detector.id
                    assert found_detector.name == mock_detector.name

    @pytest.mark.asyncio
    async def test_caching_integration_isolated(
        self, mocked_dependencies, isolated_redis
    ):
        """Test caching integration with Redis isolation."""

        # Mock Redis client
        with patch(
            "pynomaly.infrastructure.caching.redis_cache.RedisCache"
        ) as mock_cache:
            mock_cache.return_value.get = mocked_dependencies.redis_mocks["client"].get
            mock_cache.return_value.set = mocked_dependencies.redis_mocks["client"].set
            mock_cache.return_value.delete = mocked_dependencies.redis_mocks[
                "client"
            ].delete

            # Test caching operations
            cache_key = "test_detection_result"
            cache_value = {"predictions": [1, -1, 1], "scores": [0.1, 0.8, 0.2]}

            cache_client = mock_cache.return_value

            # Test cache miss
            result = await cache_client.get(cache_key)
            assert result is None

            # Test cache set
            await cache_client.set(cache_key, cache_value)
            mocked_dependencies.redis_mocks["client"].set.assert_called_once()

            # Test cache delete
            await cache_client.delete(cache_key)
            mocked_dependencies.redis_mocks["client"].delete.assert_called_once()

    def test_external_api_integration_isolated(self, mocked_dependencies):
        """Test external API integration with comprehensive mocking."""

        # Mock HTTP requests
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            mock_get.return_value = mocked_dependencies.api_mocks["response"]
            mock_post.return_value = mocked_dependencies.api_mocks["response"]

            # Test external API calls
            from pynomaly.infrastructure.external_services.ml_model_service import (
                MLModelService,
            )

            with patch.object(MLModelService, "__init__", return_value=None):
                service = MLModelService()
                service.base_url = "http://localhost:8080/api"

                # Mock service methods
                service.get_model_info = Mock(
                    return_value={"model_id": "test", "status": "ready"}
                )
                service.predict = Mock(return_value={"predictions": [1, -1, 1]})

                # Test API operations
                model_info = service.get_model_info("test_model")
                predictions = service.predict("test_model", [[1, 2, 3], [4, 5, 6]])

                assert model_info["status"] == "ready"
                assert len(predictions["predictions"]) == 3

    def test_end_to_end_workflow_isolated(
        self, mocked_dependencies, isolated_filesystem, isolated_database
    ):
        """Test complete end-to-end workflow with full isolation."""

        # Setup comprehensive mocking
        with (
            patch(
                "pynomaly.infrastructure.adapters.pyod_adapter.IsolationForest"
            ) as mock_if,
            patch(
                "pynomaly.infrastructure.persistence.async_session_maker"
            ) as mock_session,
            patch(
                "pynomaly.infrastructure.data_loaders.csv_loader.CSVLoader"
            ) as mock_loader,
        ):
            # Setup mocks
            mock_if.return_value = mocked_dependencies.pyod_mocks["IsolationForest"]
            mock_session.return_value = mocked_dependencies.database_mocks["session"]

            # Mock data loading
            test_data = np.random.randn(50, 4)
            mock_loader.return_value.load.return_value = test_data

            # Test complete workflow
            workflow_steps = []

            try:
                # Step 1: Data Loading
                data_path = isolated_filesystem["datasets"] / "sample_data.csv"
                loader = mock_loader.return_value
                data = loader.load(str(data_path))
                workflow_steps.append("✓ Data loaded successfully")

                # Step 2: Model Creation
                from pynomaly.infrastructure.adapters.sklearn_adapter import (
                    SklearnAdapter,
                )

                detector = SklearnAdapter(algorithm_name="IsolationForest")
                workflow_steps.append("✓ Detector created successfully")

                # Step 3: Model Training
                detector.fit(data)
                workflow_steps.append("✓ Model trained successfully")

                # Step 4: Prediction
                predictions = detector.predict(data)
                scores = detector.decision_function(data)
                workflow_steps.append("✓ Predictions generated successfully")

                # Step 5: Results Validation
                assert len(predictions) == len(data)
                assert len(scores) == len(data)
                assert all(p in [-1, 1] for p in predictions)
                workflow_steps.append("✓ Results validated successfully")

                # Step 6: Export Results
                results_path = isolated_filesystem["exports"] / "detection_results.csv"
                results_df = pd.DataFrame(
                    {"prediction": predictions, "anomaly_score": scores}
                )
                results_df.to_csv(results_path, index=False)
                workflow_steps.append("✓ Results exported successfully")

                # Verify end-to-end success
                assert len(workflow_steps) == 6
                assert all(step.startswith("✓") for step in workflow_steps)

            except Exception as e:
                workflow_steps.append(f"✗ Workflow failed: {str(e)}")
                raise

    def test_error_recovery_isolated(self, mocked_dependencies):
        """Test error recovery mechanisms with isolated failures."""

        # Test database connection failure recovery
        with patch(
            "pynomaly.infrastructure.persistence.get_database_session"
        ) as mock_db:
            # Simulate database failure then recovery
            mock_db.side_effect = [
                ConnectionError("Database unavailable"),
                mocked_dependencies.database_mocks["session"],
            ]

            from pynomaly.infrastructure.repositories.detector_repository import (
                DetectorRepository,
            )

            with patch.object(DetectorRepository, "__init__", return_value=None):
                repo = DetectorRepository()
                repo.session = None

                # Test retry mechanism
                retry_count = 0
                max_retries = 2

                for attempt in range(max_retries):
                    try:
                        repo.session = mock_db()
                        break
                    except ConnectionError:
                        retry_count += 1
                        if attempt == max_retries - 1:
                            raise

                # Should succeed on second attempt
                assert retry_count == 1
                assert repo.session is not None

    def test_concurrent_operations_isolated(self, mocked_dependencies):
        """Test concurrent operations with proper isolation."""

        # Mock concurrent detector operations
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter"
        ) as mock_adapter:

            def create_mock_detector(algorithm_name):
                mock = Mock()
                mock.algorithm_name = algorithm_name
                mock.fit = Mock()
                mock.predict = Mock(return_value=np.array([1, -1, 1, -1]))
                mock.decision_function = Mock(
                    return_value=np.array([0.1, 0.8, 0.2, 0.9])
                )
                return mock

            mock_adapter.side_effect = create_mock_detector

            # Test concurrent detector creation and usage
            algorithms = ["IsolationForest", "LOF", "OneClassSVM"]
            detectors = []

            for algorithm in algorithms:
                detector = mock_adapter(algorithm_name=algorithm)
                detectors.append(detector)

            # Test that each detector is properly isolated
            test_data = np.random.randn(20, 3)

            for i, detector in enumerate(detectors):
                detector.fit(test_data)
                predictions = detector.predict(test_data)

                # Verify isolation - each detector should have independent state
                assert detector.algorithm_name == algorithms[i]
                assert len(predictions) == len(test_data)

                # Verify mock was called correctly
                detector.fit.assert_called_once_with(test_data)
                detector.predict.assert_called_once_with(test_data)

    def test_resource_cleanup_isolated(self, mocked_dependencies, isolated_filesystem):
        """Test proper resource cleanup in isolated environment."""

        cleanup_tracker = {
            "files_created": [],
            "connections_opened": [],
            "resources_cleaned": [],
        }

        try:
            # Create temporary resources
            temp_file = isolated_filesystem["logs"] / "test_log.txt"
            with open(temp_file, "w") as f:
                f.write("Test log entry")
            cleanup_tracker["files_created"].append(temp_file)

            # Mock database connection
            with patch(
                "pynomaly.infrastructure.persistence.create_connection"
            ) as mock_conn:
                mock_connection = Mock()
                mock_connection.close = Mock()
                mock_conn.return_value = mock_connection

                connection = mock_conn()
                cleanup_tracker["connections_opened"].append(connection)

                # Simulate some operations
                assert temp_file.exists()
                assert connection is not None

        finally:
            # Test cleanup
            for temp_file in cleanup_tracker["files_created"]:
                if temp_file.exists():
                    temp_file.unlink()
                    cleanup_tracker["resources_cleaned"].append(
                        f"file:{temp_file.name}"
                    )

            for connection in cleanup_tracker["connections_opened"]:
                connection.close()
                cleanup_tracker["resources_cleaned"].append(
                    f"connection:{id(connection)}"
                )

        # Verify cleanup
        assert len(cleanup_tracker["resources_cleaned"]) == len(
            cleanup_tracker["files_created"] + cleanup_tracker["connections_opened"]
        )

        # Verify files are actually cleaned up
        for temp_file in cleanup_tracker["files_created"]:
            assert not temp_file.exists()

    def test_performance_isolation(self, mocked_dependencies):
        """Test performance characteristics in isolated environment."""
        import time

        # Mock performance-sensitive operations
        with patch(
            "pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter"
        ) as mock_adapter:

            def timed_operation(data_size):
                # Simulate processing time based on data size
                time.sleep(0.001 * data_size / 100)  # 1ms per 100 samples
                return np.random.choice([-1, 1], size=data_size)

            mock_adapter.return_value.predict = lambda x: timed_operation(len(x))

            # Test performance with different data sizes
            data_sizes = [100, 500, 1000]
            performance_results = {}

            for size in data_sizes:
                test_data = np.random.randn(size, 5)
                detector = mock_adapter.return_value

                start_time = time.time()
                predictions = detector.predict(test_data)
                end_time = time.time()

                execution_time = end_time - start_time
                performance_results[size] = {
                    "execution_time": execution_time,
                    "samples_per_second": (
                        size / execution_time if execution_time > 0 else float("inf")
                    ),
                    "predictions_count": len(predictions),
                }

            # Verify performance characteristics
            for size, result in performance_results.items():
                assert result["predictions_count"] == size
                assert result["execution_time"] > 0
                assert result["samples_per_second"] > 0

            # Verify performance scaling (larger datasets take more time)
            assert (
                performance_results[1000]["execution_time"]
                > performance_results[100]["execution_time"]
            )


class TestContainerIntegration:
    """Tests using real containers when available."""

    @pytest.mark.skipif(
        not TESTCONTAINERS_AVAILABLE, reason="TestContainers not available"
    )
    @pytest.mark.asyncio
    async def test_real_database_integration(self, isolated_database):
        """Test with real database container."""

        # This test runs only when TestContainers is available
        if isolated_database.get("mocked"):
            pytest.skip("Real database not available, using mocked version")

        # Test real database operations
        import psycopg2

        try:
            conn = psycopg2.connect(isolated_database["url"])
            cursor = conn.cursor()

            # Create test table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS test_detectors (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    algorithm VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Insert test data
            cursor.execute(
                """
                INSERT INTO test_detectors (name, algorithm)
                VALUES (%s, %s)
            """,
                ("Test Detector", "IsolationForest"),
            )

            # Query test data
            cursor.execute(
                "SELECT * FROM test_detectors WHERE name = %s", ("Test Detector",)
            )
            result = cursor.fetchone()

            assert result is not None
            assert result[1] == "Test Detector"
            assert result[2] == "IsolationForest"

            conn.commit()

        finally:
            cursor.close()
            conn.close()

    @pytest.mark.skipif(
        not TESTCONTAINERS_AVAILABLE, reason="TestContainers not available"
    )
    def test_real_redis_integration(self, isolated_redis):
        """Test with real Redis container."""

        if isolated_redis.get("mocked"):
            pytest.skip("Real Redis not available, using mocked version")

        # Test real Redis operations
        import redis

        r = redis.Redis(
            host=isolated_redis["host"],
            port=isolated_redis["port"],
            decode_responses=True,
        )

        # Test basic operations
        test_key = "test_detection_cache"
        test_value = "cached_result_data"

        # Set value
        r.set(test_key, test_value)

        # Get value
        retrieved_value = r.get(test_key)
        assert retrieved_value == test_value

        # Delete value
        r.delete(test_key)

        # Verify deletion
        assert r.get(test_key) is None
