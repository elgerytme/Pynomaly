"""Configuration for benchmark tests."""

import pytest


def pytest_configure(config):
    """Configure pytest for benchmarks."""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "performance: mark test as performance-related")


@pytest.fixture(scope="session")
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "warmup_rounds": 1,
        "min_rounds": 3,
        "max_time": 30.0,
        "timer": "time.perf_counter"
    }


def pytest_benchmark_update_machine_info(config, machine_info):
    """Update machine info for benchmark reports."""
    import platform
    import psutil
    
    machine_info.update({
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total // (1024 ** 3),  # GB
    })


def pytest_runtest_setup(item):
    """Setup for benchmark tests."""
    if "benchmark" in item.keywords and not hasattr(item.config.option, "benchmark_skip"):
        # Ensure benchmark plugin is available
        try:
            import pytest_benchmark
        except ImportError:
            pytest.skip("pytest-benchmark not available")


@pytest.fixture
def benchmark_group():
    """Group related benchmarks together."""
    return "anomaly_detection_algorithms"