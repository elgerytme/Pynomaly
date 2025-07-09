"""Benchmark-specific fixtures that extend the root conftest.py."""

# Import all fixtures from root conftest
from ..conftest import *


# Additional benchmark-specific hooks
def pytest_benchmark_update_machine_info(config, machine_info):
    """Update machine info for benchmark reports."""
    import platform

    try:
        import psutil

        machine_info.update(
            {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total // (1024**3),  # GB
            }
        )
    except ImportError:
        machine_info.update(
            {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            }
        )


def pytest_runtest_setup(item):
    """Setup for benchmark tests."""
    if "benchmark" in item.keywords and not hasattr(
        item.config.option, "benchmark_skip"
    ):
        # Ensure benchmark plugin is available
        try:
            import pytest_benchmark
        except ImportError:
            pytest.skip("pytest-benchmark not available")
