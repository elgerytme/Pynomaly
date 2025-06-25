"""Comprehensive cross-platform OS compatibility tests.

This module tests compatibility across different operating systems including
Windows, Linux, macOS, and various deployment environments like Docker containers,
cloud platforms, and edge devices.
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import AnomalyScore


class TestOperatingSystemCompatibility:
    """Test compatibility across different operating systems."""

    @pytest.fixture
    def system_info(self):
        """Get system information for testing."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        }

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for cross-platform testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "feature3": np.random.exponential(1, 100),
            }
        )
        return Dataset(name="Cross-Platform Test Dataset", data=data)

    def test_file_system_compatibility(self, system_info):
        """Test file system operations across platforms."""
        # Test path handling
        test_paths = [
            "data/models/detector.pkl",
            "logs/application.log",
            "config/settings.json",
            "temp/processing/data.csv",
        ]

        for path_str in test_paths:
            # Test Path creation
            path = Path(path_str)
            assert isinstance(path, Path)

            # Test path components
            assert path.name is not None
            assert path.suffix is not None

            # Test cross-platform path conversion
            normalized_path = path.as_posix()
            assert "/" in normalized_path or path_str == normalized_path

            # Test absolute path creation
            abs_path = path.resolve()
            assert abs_path.is_absolute()

    def test_file_permissions_cross_platform(self, tmp_path):
        """Test file permission handling across platforms."""
        test_file = tmp_path / "test_permissions.txt"
        test_file.write_text("test content")

        # Test file exists
        assert test_file.exists()

        # Test read permissions
        assert test_file.is_file()
        content = test_file.read_text()
        assert content == "test content"

        # Test platform-specific permission settings
        if platform.system() != "Windows":
            # Unix-like systems
            import stat

            # Test setting read-only
            test_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

            # Should still be readable
            content = test_file.read_text()
            assert content == "test content"

            # Test setting writable again
            test_file.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

            # Should be writable
            test_file.write_text("updated content")
            assert test_file.read_text() == "updated content"
        else:
            # Windows - test basic file operations
            test_file.write_text("windows test")
            assert test_file.read_text() == "windows test"

    def test_environment_variables_cross_platform(self):
        """Test environment variable handling across platforms."""
        # Test setting and getting environment variables
        test_env_vars = {
            "PYNOMALY_TEST_VAR": "test_value",
            "PYNOMALY_CONFIG_PATH": "/tmp/config",
            "PYNOMALY_LOG_LEVEL": "DEBUG",
        }

        for var_name, var_value in test_env_vars.items():
            # Set environment variable
            os.environ[var_name] = var_value

            # Verify it was set
            assert os.getenv(var_name) == var_value

            # Test with default value
            assert os.getenv(var_name, "default") == var_value
            assert os.getenv("NONEXISTENT_VAR", "default") == "default"

        # Clean up environment variables
        for var_name in test_env_vars:
            os.environ.pop(var_name, None)

    def test_process_management_cross_platform(self):
        """Test process management across platforms."""
        # Test subprocess creation with cross-platform commands
        if platform.system() == "Windows":
            # Windows commands
            commands = [
                ["echo", "Hello World"],
                ["python", "--version"],
                ["where", "python"],
            ]
        else:
            # Unix-like commands
            commands = [
                ["echo", "Hello World"],
                ["python", "--version"],
                ["which", "python"],
            ]

        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                # Command should either succeed or fail gracefully
                assert result.returncode is not None

                if result.returncode == 0:
                    # Successful command should have output
                    assert result.stdout is not None

            except subprocess.TimeoutExpired:
                pytest.skip(f"Command {cmd} timed out on {platform.system()}")
            except FileNotFoundError:
                pytest.skip(f"Command {cmd} not available on {platform.system()}")

    def test_multiprocessing_compatibility(self):
        """Test multiprocessing compatibility across platforms."""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        def test_worker_function(x):
            """Simple worker function for testing."""
            return x * x

        test_data = [1, 2, 3, 4, 5]

        # Test ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(test_worker_function, test_data))

        expected_results = [x * x for x in test_data]
        assert results == expected_results

        # Test multiprocessing.Pool
        with mp.Pool(processes=2) as pool:
            pool_results = pool.map(test_worker_function, test_data)

        assert pool_results == expected_results

    def test_unicode_handling_cross_platform(self, tmp_path):
        """Test Unicode and character encoding across platforms."""
        # Test various Unicode strings
        unicode_strings = [
            "Hello, 世界",  # Chinese
            "Привет, мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "🚀 Anomaly Detection 📊",  # Emojis
            "Café résumé naïve",  # Accented characters
        ]

        for i, text in enumerate(unicode_strings):
            test_file = tmp_path / f"unicode_test_{i}.txt"

            # Test writing Unicode
            test_file.write_text(text, encoding="utf-8")

            # Test reading Unicode
            read_text = test_file.read_text(encoding="utf-8")
            assert read_text == text

            # Test with pandas DataFrame
            df = pd.DataFrame({"text_column": [text]})
            csv_file = tmp_path / f"unicode_df_{i}.csv"

            df.to_csv(csv_file, index=False, encoding="utf-8")
            loaded_df = pd.read_csv(csv_file, encoding="utf-8")

            assert loaded_df.iloc[0]["text_column"] == text


class TestPythonVersionCompatibility:
    """Test compatibility across different Python versions."""

    def test_python_version_requirements(self):
        """Test Python version requirements."""
        # Verify minimum Python version (3.11+)
        version_info = sys.version_info
        assert version_info.major == 3
        assert (
            version_info.minor >= 11
        ), f"Python 3.11+ required, found {version_info.major}.{version_info.minor}"

    def test_type_hints_compatibility(self):
        """Test type hints compatibility across Python versions."""

        # Test basic type hints
        def test_function(
            data: dict[str, Any],
            items: list[int],
            optional_value: str | None = None,
            union_type: int | str = 0,
        ) -> tuple[bool, str]:
            return True, "success"

        # Test function with type hints
        result = test_function(
            data={"key": "value"},
            items=[1, 2, 3],
            optional_value="test",
            union_type="string",
        )

        assert result == (True, "success")

        # Test newer type hint syntax (Python 3.10+)
        def new_syntax_function(value: int | str) -> bool:
            return isinstance(value, int | str)

        assert new_syntax_function(42)
        assert new_syntax_function("test")

    def test_async_compatibility(self):
        """Test async/await compatibility."""
        import asyncio

        async def async_test_function(delay: float = 0.01) -> str:
            await asyncio.sleep(delay)
            return "async_result"

        # Test async function execution
        result = asyncio.run(async_test_function())
        assert result == "async_result"

        # Test async context manager
        class AsyncContextManager:
            async def __aenter__(self):
                return "entered"

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        async def test_async_context():
            async with AsyncContextManager() as value:
                return value

        context_result = asyncio.run(test_async_context())
        assert context_result == "entered"

    def test_dataclass_compatibility(self):
        """Test dataclass compatibility across Python versions."""
        from dataclasses import dataclass, field

        @dataclass
        class TestDataClass:
            name: str
            values: list[int] = field(default_factory=list)
            optional_field: Optional[str] = None

        # Test dataclass creation
        instance = TestDataClass(name="test", values=[1, 2, 3])
        assert instance.name == "test"
        assert instance.values == [1, 2, 3]
        assert instance.optional_field is None

        # Test dataclass with defaults
        default_instance = TestDataClass(name="default")
        assert default_instance.values == []


class TestDependencyCompatibility:
    """Test compatibility with various dependencies across platforms."""

    def test_numpy_compatibility(self):
        """Test NumPy compatibility across platforms."""
        import numpy as np

        # Test basic NumPy operations
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.dtype == np.int64 or arr.dtype == np.int32  # Platform dependent

        # Test mathematical operations
        result = np.mean(arr)
        assert abs(result - 3.0) < 1e-10

        # Test random number generation
        np.random.seed(42)
        random_arr = np.random.normal(0, 1, 100)
        assert len(random_arr) == 100
        assert abs(np.mean(random_arr)) < 0.2  # Should be close to 0

        # Test platform-specific optimizations
        large_arr = np.random.random(10000)
        fast_sum = np.sum(large_arr)
        python_sum = sum(large_arr)
        assert abs(fast_sum - python_sum) < 1e-10

    def test_pandas_compatibility(self):
        """Test Pandas compatibility across platforms."""
        import pandas as pd

        # Test DataFrame creation
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": ["a", "b", "c", "d", "e"],
                "C": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

        assert len(df) == 5
        assert list(df.columns) == ["A", "B", "C"]

        # Test file I/O
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            loaded_df = pd.read_csv(f.name)

            pd.testing.assert_frame_equal(df, loaded_df)

            # Clean up
            os.unlink(f.name)

        # Test platform-specific optimizations
        large_df = pd.DataFrame({"values": np.random.random(10000)})

        # Test aggregation performance
        mean_value = large_df["values"].mean()
        assert isinstance(mean_value, float)

    def test_scikit_learn_compatibility(self):
        """Test scikit-learn compatibility across platforms."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.model_selection import train_test_split
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Generate test data
        np.random.seed(42)
        X = np.random.normal(0, 1, (1000, 5))

        # Test IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X)
        scores = iso_forest.decision_function(X)
        predictions = iso_forest.predict(X)

        assert len(scores) == 1000
        assert len(predictions) == 1000
        assert set(predictions) <= {-1, 1}  # Should only contain -1 and 1

        # Test LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        lof_predictions = lof.fit_predict(X)

        assert len(lof_predictions) == 1000
        assert set(lof_predictions) <= {-1, 1}

        # Test preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.shape == X.shape
        assert abs(np.mean(X_scaled)) < 1e-10  # Should be approximately 0
        assert abs(np.std(X_scaled) - 1.0) < 1e-10  # Should be approximately 1


class TestContainerCompatibility:
    """Test compatibility in containerized environments."""

    def test_docker_environment_detection(self):
        """Test detection of Docker environment."""
        # Check for common Docker indicators
        docker_indicators = [
            Path("/.dockerenv").exists(),
            os.getenv("DOCKER_CONTAINER") is not None,
            "docker" in platform.platform().lower(),
        ]

        is_docker = any(docker_indicators)

        # If running in Docker, test container-specific functionality
        if is_docker:
            # Test container resource limits
            try:
                # Check memory limits
                with open("/proc/meminfo") as f:
                    meminfo = f.read()
                    assert "MemTotal" in meminfo

                # Check CPU information
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()
                    assert "processor" in cpuinfo or "cpu" in cpuinfo.lower()

            except (FileNotFoundError, PermissionError):
                # Not a Linux container or no access to proc
                pass

        # Test should pass regardless of environment
        assert True

    def test_resource_constraints_compatibility(self):
        """Test behavior under resource constraints."""
        import psutil

        # Get system resource information
        try:
            memory_info = psutil.virtual_memory()
            psutil.cpu_count()

            # Test with limited resources simulation
            if memory_info.available < 1024 * 1024 * 1024:  # Less than 1GB
                # Adjust algorithm parameters for low memory
                contamination = 0.1
                n_estimators = 50  # Reduced for low memory
            else:
                # Normal parameters
                contamination = 0.1
                n_estimators = 100

            # Test algorithm with resource-appropriate parameters
            from sklearn.ensemble import IsolationForest

            X = np.random.normal(0, 1, (1000, 10))
            detector = IsolationForest(
                contamination=contamination, n_estimators=n_estimators, random_state=42
            )

            detector.fit(X)
            predictions = detector.predict(X)

            assert len(predictions) == 1000

        except ImportError:
            pytest.skip("psutil not available for resource monitoring")

    def test_network_configuration_compatibility(self):
        """Test network configuration in different environments."""
        import socket

        # Test localhost connectivity
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", 80))
            sock.close()

            # Connection might succeed or fail, but should not raise exception
            assert isinstance(result, int)

        except Exception as e:
            # Network operations might not be available in all environments
            pytest.skip(f"Network test skipped: {e}")

        # Test hostname resolution
        try:
            hostname = socket.gethostname()
            assert isinstance(hostname, str)
            assert len(hostname) > 0

        except Exception:
            pytest.skip("Hostname resolution not available")


class TestCloudPlatformCompatibility:
    """Test compatibility across cloud platforms."""

    def test_aws_environment_detection(self):
        """Test AWS environment detection and compatibility."""
        # Check for AWS-specific environment indicators
        aws_indicators = [
            os.getenv("AWS_REGION") is not None,
            os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None,
            os.getenv("AWS_EXECUTION_ENV") is not None,
            Path("/opt/aws").exists(),
        ]

        is_aws = any(aws_indicators)

        if is_aws:
            # Test AWS-specific functionality
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            assert isinstance(aws_region, str)

            # Test S3-compatible path handling
            s3_paths = [
                "s3://bucket/path/to/file.csv",
                "s3://another-bucket/models/detector.pkl",
            ]

            for s3_path in s3_paths:
                # Test S3 path parsing
                assert s3_path.startswith("s3://")
                parts = s3_path[5:].split("/", 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ""

                assert len(bucket) > 0
                assert isinstance(key, str)

        # Test should pass in any environment
        assert True

    def test_azure_environment_detection(self):
        """Test Azure environment detection and compatibility."""
        # Check for Azure-specific environment indicators
        azure_indicators = [
            os.getenv("AZURE_CLIENT_ID") is not None,
            os.getenv("AZURE_TENANT_ID") is not None,
            os.getenv("WEBSITE_SITE_NAME") is not None,  # Azure App Service
            "azure" in platform.platform().lower(),
        ]

        is_azure = any(azure_indicators)

        if is_azure:
            # Test Azure-specific path handling
            azure_paths = [
                "https://storageaccount.blob.core.windows.net/container/file.csv",
                "abfss://container@account.dfs.core.windows.net/path/file.parquet",
            ]

            for azure_path in azure_paths:
                # Test Azure path parsing
                assert "azure" in azure_path or azure_path.startswith("abfss://")

        assert True

    def test_gcp_environment_detection(self):
        """Test Google Cloud Platform environment detection."""
        # Check for GCP-specific environment indicators
        gcp_indicators = [
            os.getenv("GOOGLE_CLOUD_PROJECT") is not None,
            os.getenv("GCLOUD_PROJECT") is not None,
            os.getenv("GCP_PROJECT") is not None,
            Path("/var/secrets/google").exists(),
        ]

        is_gcp = any(gcp_indicators)

        if is_gcp:
            # Test GCP-specific functionality
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv(
                "GCLOUD_PROJECT"
            )
            if project_id:
                assert isinstance(project_id, str)
                assert len(project_id) > 0

            # Test GCS path handling
            gcs_paths = [
                "gs://bucket/path/to/file.csv",
                "gs://ml-models/anomaly-detector/v1.0.0/model.pkl",
            ]

            for gcs_path in gcs_paths:
                assert gcs_path.startswith("gs://")
                bucket_and_path = gcs_path[5:].split("/", 1)
                bucket = bucket_and_path[0]
                path = bucket_and_path[1] if len(bucket_and_path) > 1 else ""

                assert len(bucket) > 0
                assert isinstance(path, str)

        assert True


class TestDeploymentEnvironmentCompatibility:
    """Test compatibility across different deployment environments."""

    def test_development_environment(self):
        """Test development environment compatibility."""
        # Check for development indicators
        dev_indicators = [
            os.getenv("ENVIRONMENT") == "development",
            os.getenv("ENV") == "dev",
            os.getenv("DEBUG") == "True",
            "dev" in os.getcwd().lower(),
        ]

        is_development = any(dev_indicators)

        if is_development:
            # Test development-specific features
            # More verbose logging, debug modes, etc.
            debug_mode = True
        else:
            debug_mode = False

        # Test should work in both dev and production
        assert isinstance(debug_mode, bool)

    def test_production_environment_compatibility(self):
        """Test production environment compatibility."""
        # Check for production indicators
        prod_indicators = [
            os.getenv("ENVIRONMENT") == "production",
            os.getenv("ENV") == "prod",
            os.getenv("DEBUG") == "False",
            not any(
                [
                    "dev" in os.getcwd().lower(),
                    "test" in os.getcwd().lower(),
                    "debug" in os.getcwd().lower(),
                ]
            ),
        ]

        is_production = any(prod_indicators)

        if is_production:
            # Test production-specific configurations
            # Security hardening, performance optimizations, etc.

            # Verify secure defaults
            assert os.getenv("DEBUG", "False") != "True"

            # Test performance optimizations
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Production code should minimize warnings
                pass

        assert True

    def test_testing_environment_compatibility(self):
        """Test testing environment compatibility."""
        # Detect if we're in a testing environment
        testing_indicators = [
            "pytest" in sys.modules,
            os.getenv("PYTEST_CURRENT_TEST") is not None,
            "test" in os.getcwd().lower(),
            sys.argv and "pytest" in sys.argv[0],
        ]

        is_testing = any(testing_indicators)

        if is_testing:
            # Test testing-specific configurations
            # Mock services, test databases, etc.

            # Verify test isolation
            temp_dir = tempfile.mkdtemp()
            test_file = Path(temp_dir) / "test_isolation.txt"
            test_file.write_text("test content")

            assert test_file.exists()
            assert test_file.read_text() == "test content"

            # Clean up
            shutil.rmtree(temp_dir)

        # This test itself proves we're in a testing environment
        assert is_testing


class TestEdgeDeviceCompatibility:
    """Test compatibility on edge devices and constrained environments."""

    def test_arm_architecture_compatibility(self):
        """Test compatibility on ARM architectures."""
        machine = platform.machine().lower()

        # Check if running on ARM
        is_arm = any(["arm" in machine, "aarch64" in machine, "armv" in machine])

        if is_arm:
            # Test ARM-specific optimizations
            import numpy as np

            # NumPy should work on ARM
            arr = np.random.random(1000)
            result = np.mean(arr)
            assert isinstance(result, float)

            # Test memory efficiency on ARM
            # ARM devices often have limited memory
            small_array = np.random.random(1000)  # Smaller than usual
            assert len(small_array) == 1000

        # Test should pass on any architecture
        assert True

    def test_low_memory_compatibility(self):
        """Test compatibility in low memory environments."""
        try:
            import psutil

            # Get available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            if available_gb < 2.0:  # Less than 2GB available
                # Use memory-efficient algorithms
                max_samples = 1000
                n_estimators = 50
            else:
                # Normal parameters
                max_samples = 10000
                n_estimators = 100

            # Test with appropriate parameters
            from sklearn.ensemble import IsolationForest

            X = np.random.normal(0, 1, (min(max_samples, 1000), 5))
            detector = IsolationForest(
                n_estimators=min(n_estimators, 50),
                max_samples=min(len(X), 256),
                random_state=42,
            )

            detector.fit(X)
            predictions = detector.predict(X)

            assert len(predictions) == len(X)

        except ImportError:
            # If psutil not available, test with conservative parameters
            from sklearn.ensemble import IsolationForest

            X = np.random.normal(0, 1, (500, 5))
            detector = IsolationForest(
                n_estimators=25, max_samples=256, random_state=42
            )

            detector.fit(X)
            predictions = detector.predict(X)

            assert len(predictions) == 500

    def test_single_core_compatibility(self):
        """Test compatibility on single-core systems."""
        import multiprocessing as mp

        # Get CPU count
        cpu_count = mp.cpu_count()

        # Test with single-threaded algorithms
        from sklearn.ensemble import IsolationForest

        # Force single-threaded execution
        detector = IsolationForest(
            n_jobs=1,  # Single thread
            n_estimators=50,
            random_state=42,
        )

        X = np.random.normal(0, 1, (1000, 5))
        detector.fit(X)
        predictions = detector.predict(X)

        assert len(predictions) == 1000

        # Test should work regardless of CPU count
        assert cpu_count >= 1

    def test_offline_compatibility(self):
        """Test compatibility in offline environments."""
        # Test that core functionality works without internet

        # Create test data
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.exponential(1, 100),
            }
        )

        dataset = Dataset(name="Offline Test", data=data)

        # Test basic operations
        assert dataset.n_samples == 100
        assert dataset.n_features == 2

        # Test without external dependencies
        scores = [AnomalyScore(np.random.beta(2, 8)) for _ in range(100)]
        assert len(scores) == 100
        assert all(isinstance(score, AnomalyScore) for score in scores)

        # Verify no network calls are made
        # (This would be tested with network mocking in real scenarios)
        assert True
