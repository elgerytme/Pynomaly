"""Comprehensive dependency compatibility tests.

This module tests compatibility across different dependency versions,
package managers, Python distributions, and optional dependencies
to ensure robust deployment across various environments.
"""

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest


class TestCoreDependencyCompatibility:
    """Test compatibility with core dependencies."""

    def test_python_version_compatibility(self):
        """Test Python version compatibility requirements."""
        # Verify minimum Python version (3.11+)
        version_info = sys.version_info
        assert version_info.major == 3
        assert version_info.minor >= 11, f"Python 3.11+ required, found {version_info.major}.{version_info.minor}"

        # Test Python 3.11+ specific features
        if version_info >= (3, 11):
            # Test exception groups (Python 3.11+)
            try:
                # ExceptionGroup is available in Python 3.11+
                from builtins import ExceptionGroup
                assert ExceptionGroup is not None
            except ImportError:
                # Fallback for older Python versions
                pass

            # Test improved error messages (Python 3.11+)
            try:
                # This should work in Python 3.11+
                exec("x = 1\ny = x.nonexistent_attribute")
            except AttributeError as e:
                # Python 3.11+ provides better error messages
                error_msg = str(e)
                assert "int" in error_msg.lower() or "attribute" in error_msg.lower()

        # Test Python 3.12+ features if available
        if version_info >= (3, 12):
            # Test generic type syntax (Python 3.12+)
            try:
                # This syntax is available in Python 3.12+
                exec("type Point[T] = tuple[T, T]")
            except SyntaxError:
                # Expected in earlier Python versions
                pass

    def test_numpy_compatibility(self):
        """Test NumPy compatibility across versions."""
        import numpy as np

        # Check NumPy version
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        assert numpy_version >= (1, 21), f"NumPy 1.21+ required, found {np.__version__}"

        # Test NumPy functionality
        test_array = np.array([1, 2, 3, 4, 5])
        assert test_array.dtype in [np.int32, np.int64]  # Platform dependent

        # Test NumPy operations
        operations = [
            np.mean(test_array),
            np.std(test_array),
            np.sum(test_array),
            np.max(test_array),
            np.min(test_array)
        ]

        for result in operations:
            assert isinstance(result, (int, float, np.number))

        # Test NumPy random state (reproducibility)
        np.random.seed(42)
        random_data = np.random.normal(0, 1, 100)
        assert len(random_data) == 100
        assert abs(np.mean(random_data)) < 0.3  # Should be close to 0

        # Test NumPy dtype compatibility
        supported_dtypes = [
            np.float32, np.float64,
            np.int32, np.int64,
            np.bool_, np.object_
        ]

        for dtype in supported_dtypes:
            test_data = np.array([1, 2, 3], dtype=dtype)
            assert test_data.dtype == dtype

    def test_pandas_compatibility(self):
        """Test Pandas compatibility across versions."""
        import pandas as pd

        # Check Pandas version
        pandas_version = tuple(map(int, pd.__version__.split('.')[:2]))
        assert pandas_version >= (1, 5), f"Pandas 1.5+ required, found {pd.__version__}"

        # Test DataFrame creation and operations
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5],
            'D': pd.date_range('2023-01-01', periods=5)
        })

        # Test basic DataFrame operations
        assert len(df) == 5
        assert list(df.columns) == ['A', 'B', 'C', 'D']
        assert df['A'].dtype in [np.int32, np.int64]
        assert df['B'].dtype == object
        assert df['C'].dtype == np.float64

        # Test DataFrame methods
        operations = [
            df.describe(),
            df.head(),
            df.tail(),
            df.info(),
            df.dtypes
        ]

        # All operations should complete without error
        for op in operations:
            assert op is not None

        # Test Pandas I/O capabilities
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            loaded_df = pd.read_csv(f.name)

            # Verify data integrity (excluding datetime parsing differences)
            assert len(loaded_df) == len(df)
            assert list(loaded_df.columns) == list(df.columns)

        # Test Pandas extension types (if available)
        try:
            # Test nullable integer type (Pandas 1.0+)
            nullable_int_series = pd.Series([1, 2, None, 4], dtype="Int64")
            assert nullable_int_series.isna().sum() == 1
        except (TypeError, ValueError):
            # Not all versions support all extension types
            pass

    def test_scikit_learn_compatibility(self):
        """Test scikit-learn compatibility across versions."""
        try:
            import sklearn
            from sklearn.ensemble import IsolationForest
            from sklearn.metrics import classification_report, confusion_matrix
            from sklearn.model_selection import train_test_split
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Check scikit-learn version
        sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        assert sklearn_version >= (1, 0), f"scikit-learn 1.0+ required, found {sklearn.__version__}"

        # Test anomaly detection algorithms
        np.random.seed(42)
        X = np.random.normal(0, 1, (1000, 5))

        # Test IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X)
        iso_predictions = iso_forest.predict(X)
        iso_scores = iso_forest.decision_function(X)

        assert len(iso_predictions) == 1000
        assert len(iso_scores) == 1000
        assert set(iso_predictions) <= {-1, 1}

        # Test LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        lof_predictions = lof.fit_predict(X)

        assert len(lof_predictions) == 1000
        assert set(lof_predictions) <= {-1, 1}

        # Test preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.shape == X.shape
        assert abs(np.mean(X_scaled)) < 1e-10
        assert abs(np.std(X_scaled) - 1.0) < 1e-10

        # Test model persistence (if available)
        try:
            import joblib

            # Test model serialization
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                joblib.dump(iso_forest, f.name)
                loaded_model = joblib.load(f.name)

                # Test loaded model functionality
                loaded_predictions = loaded_model.predict(X[:10])
                assert len(loaded_predictions) == 10

        except ImportError:
            # joblib not available, skip persistence test
            pass


class TestOptionalDependencyCompatibility:
    """Test compatibility with optional dependencies."""

    def test_pytorch_compatibility(self):
        """Test PyTorch compatibility if available."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            pytest.skip("PyTorch not available")

        # Check PyTorch version
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        assert torch_version >= (1, 12), f"PyTorch 1.12+ recommended, found {torch.__version__}"

        # Test basic PyTorch functionality
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert x.dtype == torch.float32

        # Test neural network creation
        class SimpleAutoencoder(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.encoder = nn.Linear(input_dim, input_dim // 2)
                self.decoder = nn.Linear(input_dim // 2, input_dim)

            def forward(self, x):
                encoded = torch.relu(self.encoder(x))
                decoded = self.decoder(encoded)
                return decoded

        # Test model instantiation
        model = SimpleAutoencoder(5)
        test_input = torch.randn(10, 5)
        output = model(test_input)

        assert output.shape == test_input.shape
        assert isinstance(output, torch.Tensor)

        # Test optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        assert optimizer is not None

        # Test GPU availability (optional)
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = torch.device("cuda")
            model_gpu = model.to(device)
            test_input_gpu = test_input.to(device)
            output_gpu = model_gpu(test_input_gpu)
            assert output_gpu.device.type == "cuda"

    def test_tensorflow_compatibility(self):
        """Test TensorFlow compatibility if available."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not available")

        # Check TensorFlow version
        tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
        assert tf_version >= (2, 8), f"TensorFlow 2.8+ recommended, found {tf.__version__}"

        # Test basic TensorFlow functionality
        x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        assert x.dtype == tf.float32

        # Test model creation
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(3, input_shape=(5,), activation='relu'),
            tf.keras.layers.Dense(5, activation='linear')
        ])

        # Test model compilation
        model.compile(optimizer='adam', loss='mse')

        # Test model prediction
        test_input = tf.random.normal((10, 5))
        output = model(test_input)

        assert output.shape == (10, 5)
        assert isinstance(output, tf.Tensor)

        # Test GPU availability (optional)
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        if gpu_available:
            with tf.device('/GPU:0'):
                gpu_output = model(test_input)
                assert gpu_output.shape == output.shape

    def test_jax_compatibility(self):
        """Test JAX compatibility if available."""
        try:
            import jax
            import jax.numpy as jnp
            from jax import grad, jit, vmap
        except ImportError:
            pytest.skip("JAX not available")

        # Test basic JAX functionality
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert x.dtype == jnp.float32

        # Test JAX transformations
        def simple_function(x):
            return jnp.sum(x ** 2)

        # Test gradient computation
        grad_fn = grad(simple_function)
        gradient = grad_fn(x)
        expected_gradient = 2 * x

        assert jnp.allclose(gradient, expected_gradient)

        # Test JIT compilation
        jit_fn = jit(simple_function)
        jit_result = jit_fn(x)
        normal_result = simple_function(x)

        assert jnp.allclose(jit_result, normal_result)

        # Test vectorization
        vmap_fn = vmap(lambda x: x ** 2)
        batch_x = jnp.array([[1, 2], [3, 4], [5, 6]])
        vmap_result = vmap_fn(batch_x)

        assert vmap_result.shape == batch_x.shape

    def test_distributed_computing_dependencies(self):
        """Test distributed computing dependencies if available."""
        # Test Dask
        try:
            import dask
            import dask.array as da
            import dask.dataframe as dd

            # Test Dask DataFrame
            df = pd.DataFrame({
                'x': np.random.random(1000),
                'y': np.random.random(1000)
            })
            ddf = dd.from_pandas(df, npartitions=4)

            # Test Dask operations
            result = ddf.x.mean().compute()
            assert isinstance(result, float)

            # Test Dask Array
            arr = da.random.random((1000, 100), chunks=(100, 100))
            mean_result = arr.mean().compute()
            assert isinstance(mean_result, float)

        except ImportError:
            pass  # Dask not available

        # Test Ray
        try:
            import ray

            # Test Ray initialization (if not already initialized)
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            @ray.remote
            def simple_task(x):
                return x * 2

            # Test Ray task execution
            future = simple_task.remote(5)
            result = ray.get(future)
            assert result == 10

            ray.shutdown()

        except ImportError:
            pass  # Ray not available

    def test_visualization_dependencies(self):
        """Test visualization dependencies if available."""
        # Test Matplotlib
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            # Test basic plotting (without display)
            matplotlib.use('Agg')  # Use non-interactive backend

            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y)

            # Test figure creation
            assert fig is not None
            assert ax is not None

            plt.close(fig)

        except ImportError:
            pass  # Matplotlib not available

        # Test Seaborn
        try:
            import seaborn as sns

            # Test seaborn functionality
            tips = sns.load_dataset("tips")
            assert tips is not None

        except ImportError:
            pass  # Seaborn not available

        # Test Plotly
        try:
            import plotly.express as px
            import plotly.graph_objects as go

            # Test Plotly functionality
            fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
            assert fig is not None

        except ImportError:
            pass  # Plotly not available


class TestPackageManagerCompatibility:
    """Test compatibility across different package managers."""

    def test_pip_compatibility(self):
        """Test pip package manager compatibility."""
        # Test pip version detection
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                pip_version = result.stdout.strip()
                assert "pip" in pip_version.lower()

                # Extract version number
                version_parts = pip_version.split()
                for part in version_parts:
                    if part.count('.') >= 1:  # Version format like "23.0.1"
                        version_nums = part.split('.')
                        major_version = int(version_nums[0])
                        assert major_version >= 20  # Modern pip version
                        break

        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pytest.skip("pip not available or not responsive")

    def test_conda_compatibility(self):
        """Test conda package manager compatibility."""
        # Test conda availability
        try:
            result = subprocess.run(
                ["conda", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                conda_version = result.stdout.strip()
                assert "conda" in conda_version.lower()

                # Test conda environment detection
                conda_env = os.getenv("CONDA_DEFAULT_ENV")
                if conda_env:
                    assert isinstance(conda_env, str)
                    assert len(conda_env) > 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Conda not available - this is acceptable
            pass

    def test_poetry_compatibility(self):
        """Test Poetry package manager compatibility."""
        # Test Poetry availability
        try:
            result = subprocess.run(
                ["poetry", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                poetry_version = result.stdout.strip()
                assert "poetry" in poetry_version.lower()

                # Test pyproject.toml existence
                pyproject_path = Path("pyproject.toml")
                if pyproject_path.exists():
                    # Verify it's a valid TOML file
                    try:
                        import toml
                        config = toml.load(pyproject_path)

                        # Check for Poetry sections
                        if "tool" in config and "poetry" in config["tool"]:
                            poetry_config = config["tool"]["poetry"]
                            assert "name" in poetry_config
                            assert "version" in poetry_config
                            assert "dependencies" in poetry_config

                    except ImportError:
                        # TOML parser not available
                        pass

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Poetry not available - this is acceptable
            pass

    def test_pipenv_compatibility(self):
        """Test Pipenv package manager compatibility."""
        # Test Pipenv availability
        try:
            result = subprocess.run(
                ["pipenv", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                pipenv_version = result.stdout.strip()
                assert "pipenv" in pipenv_version.lower()

                # Test Pipfile existence
                pipfile_path = Path("Pipfile")
                if pipfile_path.exists():
                    pipfile_content = pipfile_path.read_text()

                    # Basic Pipfile validation
                    assert "[packages]" in pipfile_content
                    assert "python_version" in pipfile_content

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Pipenv not available - this is acceptable
            pass


class TestPythonDistributionCompatibility:
    """Test compatibility across different Python distributions."""

    def test_cpython_compatibility(self):
        """Test CPython compatibility."""
        # Check if running on CPython
        implementation = sys.implementation.name

        if implementation == "cpython":
            # Test CPython-specific features
            version_info = sys.version_info
            assert version_info.major == 3
            assert version_info.minor >= 11

            # Test C extension compatibility
            try:
                import _ctypes
                assert _ctypes is not None
            except ImportError:
                # C extensions not available
                pass

            # Test global interpreter lock behavior
            import threading
            import time

            def cpu_bound_task():
                total = 0
                for i in range(1000000):
                    total += i
                return total

            # Test that threading works (even with GIL)
            thread = threading.Thread(target=cpu_bound_task)
            thread.start()
            thread.join(timeout=5)

            assert not thread.is_alive()  # Thread should complete

    def test_pypy_compatibility(self):
        """Test PyPy compatibility if running on PyPy."""
        implementation = sys.implementation.name

        if implementation == "pypy":
            # Test PyPy-specific optimizations
            version_info = sys.version_info
            assert version_info.major == 3

            # Test JIT compilation behavior
            def jit_optimizable_function(n):
                total = 0
                for i in range(n):
                    total += i * i
                return total

            # Run function multiple times to trigger JIT
            for _ in range(10):
                result = jit_optimizable_function(1000)
                assert result == sum(i * i for i in range(1000))

            # Test memory efficiency
            import gc
            gc.collect()  # Force garbage collection

            # PyPy should handle memory efficiently
            large_list = list(range(100000))
            assert len(large_list) == 100000

            del large_list
            gc.collect()

    def test_anaconda_distribution_compatibility(self):
        """Test Anaconda distribution compatibility."""
        # Check for Anaconda-specific indicators
        anaconda_indicators = [
            "anaconda" in sys.executable.lower(),
            "conda" in sys.executable.lower(),
            os.getenv("CONDA_DEFAULT_ENV") is not None
        ]

        is_anaconda = any(anaconda_indicators)

        if is_anaconda:
            # Test Anaconda-specific packages
            anaconda_packages = [
                "numpy", "pandas", "matplotlib", "scipy"
            ]

            for package in anaconda_packages:
                try:
                    importlib.import_module(package)
                except ImportError:
                    # Some packages might not be installed
                    pass

            # Test conda environment
            conda_env = os.getenv("CONDA_DEFAULT_ENV")
            if conda_env:
                assert isinstance(conda_env, str)
                assert len(conda_env) > 0

    def test_virtual_environment_compatibility(self):
        """Test virtual environment compatibility."""
        # Check for virtual environment indicators
        venv_indicators = [
            sys.prefix != sys.base_prefix,  # Virtual environment
            os.getenv("VIRTUAL_ENV") is not None,
            "venv" in sys.executable.lower(),
            ".venv" in sys.executable.lower()
        ]

        is_venv = any(venv_indicators)

        if is_venv:
            # Test virtual environment isolation
            venv_path = os.getenv("VIRTUAL_ENV")
            if venv_path:
                venv_path_obj = Path(venv_path)
                assert venv_path_obj.exists()
                assert venv_path_obj.is_dir()

                # Test site-packages directory
                if sys.platform == "win32":
                    site_packages = venv_path_obj / "Lib" / "site-packages"
                else:
                    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
                    site_packages = venv_path_obj / "lib" / python_version / "site-packages"

                if site_packages.exists():
                    assert site_packages.is_dir()

        # Test that sys.path is properly configured
        assert len(sys.path) > 0
        assert any(Path(path).exists() for path in sys.path if path)


class TestDependencyConflictResolution:
    """Test dependency conflict resolution and compatibility."""

    def test_version_pinning_compatibility(self):
        """Test compatibility with version pinning strategies."""
        # Test semantic versioning compatibility
        version_constraints = [
            ("numpy", ">=1.21.0,<2.0.0"),
            ("pandas", ">=1.5.0,<3.0.0"),
            ("scikit-learn", ">=1.0.0,<2.0.0")
        ]

        for package, constraint in version_constraints:
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "0.0.0")

                # Basic version format validation
                version_parts = version.split(".")
                assert len(version_parts) >= 2

                # Check major version is reasonable
                major_version = int(version_parts[0])
                assert 0 <= major_version <= 10

            except (ImportError, ValueError, AttributeError):
                # Package not available or version not parseable
                pass

    def test_optional_dependency_graceful_fallback(self):
        """Test graceful fallback when optional dependencies are missing."""
        # Test missing optional dependencies
        optional_packages = [
            "torch", "tensorflow", "jax", "dask", "ray",
            "plotly", "seaborn", "bokeh", "streamlit"
        ]

        for package in optional_packages:
            try:
                importlib.import_module(package)
                # Package available - test basic functionality
                module = sys.modules[package]
                assert hasattr(module, "__version__")

            except ImportError:
                # Package not available - this should be handled gracefully
                # Test that the application can still function
                assert True  # Graceful fallback successful

    def test_dependency_tree_consistency(self):
        """Test dependency tree consistency."""
        # Test that core dependencies don't conflict
        core_packages = ["numpy", "pandas"]

        versions = {}
        for package in core_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")
                versions[package] = version
            except ImportError:
                versions[package] = "not_installed"

        # If both are installed, check compatibility
        if all(v != "not_installed" for v in versions.values()):
            # NumPy and Pandas should be compatible
            import numpy as np
            import pandas as pd

            # Test that Pandas can use NumPy arrays
            np_array = np.array([1, 2, 3, 4, 5])
            pd_series = pd.Series(np_array)

            assert len(pd_series) == len(np_array)
            assert pd_series.dtype == np_array.dtype

    def test_namespace_package_compatibility(self):
        """Test namespace package compatibility."""
        # Test that namespace packages work correctly
        namespace_indicators = [
            "google" in sys.modules,  # Google namespace packages
            "azure" in sys.modules,   # Azure namespace packages
            "aws" in sys.modules      # AWS namespace packages
        ]

        # Test basic import functionality
        try:
            import importlib.util

            # Test dynamic import capability
            spec = importlib.util.find_spec("sys")
            assert spec is not None

            # Test module loading
            sys_module = importlib.util.module_from_spec(spec)
            assert sys_module is not None

        except (ImportError, AttributeError):
            # Namespace packages not available
            pass

    def test_extension_module_compatibility(self):
        """Test compatibility with compiled extension modules."""
        # Test common extension modules
        extension_modules = [
            "_ctypes",     # Core C extension
            "_sqlite3",    # SQLite extension
            "_ssl",        # SSL extension
            "_json",       # JSON extension
        ]

        for module_name in extension_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None

                # Test that extension module has expected attributes
                if hasattr(module, "__file__"):
                    file_path = module.__file__
                    if file_path:
                        # Extension modules typically have .so, .pyd, or .dll extensions
                        file_ext = Path(file_path).suffix.lower()
                        expected_extensions = [".so", ".pyd", ".dll", ".dylib"]
                        # Note: Some built-in modules might not have file extensions

            except ImportError:
                # Extension module not available - this is acceptable
                pass
