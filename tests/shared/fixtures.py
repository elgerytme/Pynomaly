"""Common test fixtures shared across all test modules."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator, Any
from unittest.mock import patch

import pytest

from .utilities import ResourceManager


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_file(temp_dir) -> Generator[Path, None, None]:
    """Create temporary file for testing."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    yield temp_file_path


@pytest.fixture(scope="function")
def resource_manager() -> Generator[ResourceManager, None, None]:
    """Provide a resource manager for test cleanup."""
    manager = ResourceManager()
    yield manager
    manager.cleanup_all()


@pytest.fixture(scope="function")
def test_environment():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "PYNOMALY_ENVIRONMENT": "test",
        "PYNOMALY_DEBUG": "false",
        "PYNOMALY_LOG_LEVEL": "INFO",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1"
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def isolated_imports():
    """Isolate imports for testing module loading."""
    import sys
    original_modules = sys.modules.copy()
    
    yield
    
    # Clean up imported test modules
    test_modules_to_clear = [
        mod for mod in sys.modules.keys()
        if "pynomaly" in mod and any(test_path in mod for test_path in ["test_", "_test"])
    ]
    
    for module in test_modules_to_clear:
        sys.modules.pop(module, None)


@pytest.fixture(scope="function")
def mock_logging():
    """Mock logging for tests."""
    with patch('logging.getLogger') as mock_logger:
        mock_logger.return_value.debug.return_value = None
        mock_logger.return_value.info.return_value = None
        mock_logger.return_value.warning.return_value = None
        mock_logger.return_value.error.return_value = None
        mock_logger.return_value.critical.return_value = None
        yield mock_logger


@pytest.fixture(scope="function")
def mock_datetime():
    """Mock datetime for deterministic testing."""
    from datetime import datetime
    from unittest.mock import Mock
    
    fixed_datetime = datetime(2023, 1, 1, 12, 0, 0)
    
    with patch('datetime.datetime') as mock_dt:
        mock_dt.now.return_value = fixed_datetime
        mock_dt.utcnow.return_value = fixed_datetime
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield mock_dt


@pytest.fixture(scope="function")
def capture_warnings():
    """Capture warnings during test execution."""
    import warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield warning_list


@pytest.fixture(scope="function")
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


@pytest.fixture(scope="function", autouse=True)
def test_isolation():
    """Ensure test isolation by cleaning up after each test."""
    yield
    
    # Clean up any global state that might leak between tests
    import gc
    gc.collect()


@pytest.fixture(scope="function")
def no_network():
    """Disable network access for tests."""
    import socket
    
    original_socket = socket.socket
    
    def disabled_socket(*args, **kwargs):
        raise RuntimeError("Network access disabled in tests")
    
    socket.socket = disabled_socket
    
    yield
    
    socket.socket = original_socket


@pytest.fixture(scope="function")
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    yield
    
    final_memory = process.memory_info().rss
    memory_diff = final_memory - initial_memory
    
    # Warn if memory usage increased significantly (more than 100MB)
    if memory_diff > 100 * 1024 * 1024:
        print(f"Warning: Test increased memory usage by {memory_diff / 1024 / 1024:.2f} MB")


@pytest.fixture(scope="function")
def async_timeout():
    """Provide timeout for async operations."""
    return 30.0  # 30 seconds timeout for async operations


@pytest.fixture(scope="function")
def test_config():
    """Provide test configuration."""
    return {
        "database": {
            "url": "sqlite:///:memory:",
            "echo": False
        },
        "api": {
            "host": "localhost",
            "port": 8000,
            "debug": True
        },
        "storage": {
            "type": "memory",
            "path": "/tmp/test_storage"
        },
        "cache": {
            "type": "memory",
            "ttl": 300
        },
        "monitoring": {
            "enabled": False
        }
    }


@pytest.fixture(scope="function")
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "min_rounds": 5,
        "max_time": 10.0,
        "min_time": 0.001,
        "timer": "perf_counter"
    }