"""Standard test markers for consistent test categorization."""

import pytest

# Define standard test markers
UNIT_TEST_MARKER = pytest.mark.unit
INTEGRATION_TEST_MARKER = pytest.mark.integration
E2E_TEST_MARKER = pytest.mark.e2e
PERFORMANCE_TEST_MARKER = pytest.mark.performance
SECURITY_TEST_MARKER = pytest.mark.security
SLOW_TEST_MARKER = pytest.mark.slow

# Domain-specific markers
ML_TEST_MARKER = pytest.mark.ml
DATA_TEST_MARKER = pytest.mark.data
API_TEST_MARKER = pytest.mark.api
CLI_TEST_MARKER = pytest.mark.cli
DATABASE_TEST_MARKER = pytest.mark.database
EXTERNAL_SERVICE_MARKER = pytest.mark.external_service

# Environment markers
LOCAL_ONLY_MARKER = pytest.mark.local_only
CI_ONLY_MARKER = pytest.mark.ci_only
DOCKER_REQUIRED_MARKER = pytest.mark.docker_required
GPU_REQUIRED_MARKER = pytest.mark.gpu_required

# Standard marker definitions for pytest registration
STANDARD_MARKERS = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "e2e: End-to-end tests",
    "performance: Performance tests",
    "security: Security tests",
    "slow: Slow running tests (>1 second)",
    "ml: Machine learning specific tests",
    "data: Data processing tests",
    "api: API endpoint tests",
    "cli: Command line interface tests",
    "database: Database interaction tests",
    "external_service: Tests requiring external services",
    "local_only: Tests that only run in local environment",
    "ci_only: Tests that only run in CI environment",
    "docker_required: Tests requiring Docker",
    "gpu_required: Tests requiring GPU acceleration",
]


def pytest_configure(config):
    """Register standard markers with pytest."""
    for marker in STANDARD_MARKERS:
        config.addinivalue_line("markers", marker)


# Decorator functions for easy test marking
def unit_test(func):
    """Mark a test as a unit test."""
    return UNIT_TEST_MARKER(func)


def integration_test(func):
    """Mark a test as an integration test."""
    return INTEGRATION_TEST_MARKER(func)


def e2e_test(func):
    """Mark a test as an end-to-end test.""" 
    return E2E_TEST_MARKER(func)


def performance_test(func):
    """Mark a test as a performance test."""
    return PERFORMANCE_TEST_MARKER(func)


def security_test(func):
    """Mark a test as a security test."""
    return SECURITY_TEST_MARKER(func)


def slow_test(func):
    """Mark a test as slow running."""
    return SLOW_TEST_MARKER(func)


def ml_test(func):
    """Mark a test as ML-specific."""
    return ML_TEST_MARKER(func)


def data_test(func):
    """Mark a test as data processing specific."""
    return DATA_TEST_MARKER(func)


def api_test(func):
    """Mark a test as API-specific."""
    return API_TEST_MARKER(func)


def cli_test(func):
    """Mark a test as CLI-specific."""
    return CLI_TEST_MARKER(func)


def database_test(func):
    """Mark a test as database-specific."""
    return DATABASE_TEST_MARKER(func)


def external_service_test(func):
    """Mark a test as requiring external services."""
    return EXTERNAL_SERVICE_MARKER(func)


def requires_docker(func):
    """Mark a test as requiring Docker."""
    return DOCKER_REQUIRED_MARKER(func)


def requires_gpu(func):
    """Mark a test as requiring GPU."""
    return GPU_REQUIRED_MARKER(func)


def local_only(func):
    """Mark a test to run only in local environment."""
    return LOCAL_ONLY_MARKER(func)


def ci_only(func):
    """Mark a test to run only in CI environment."""
    return CI_ONLY_MARKER(func)