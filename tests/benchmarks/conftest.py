"""Benchmark-specific fixtures that extend the root conftest.py."""

# Import specific fixtures from root conftest
try:
    from ..conftest import container, sample_data, sample_dataset, large_dataset, performance_data
except ImportError:
    # Fallback for when running benchmarks in isolation
    import pytest
    
    @pytest.fixture
    def container():
        return None
    
    @pytest.fixture
    def sample_data():
        return None
    
    @pytest.fixture
    def sample_dataset():
        return None
    
    @pytest.fixture
    def large_dataset():
        return None
    
    @pytest.fixture
    def performance_data():
        return None


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
