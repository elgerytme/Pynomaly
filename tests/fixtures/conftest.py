"""Pytest configuration and fixtures for test data management."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tests.fixtures.test_data_generator import (
    HIGH_DIM_DATASET_PARAMS,
    LARGE_DATASET_PARAMS,
    MEDIUM_DATASET_PARAMS,
    SMALL_DATASET_PARAMS,
    TestDataManager,
    TestScenarioFactory,
)


@pytest.fixture(scope="session")
def test_data_manager():
    """Provide a test data manager for the entire test session."""
    manager = TestDataManager()
    yield manager
    # Cleanup after session
    manager.clear_cache()


@pytest.fixture(scope="session")
def test_scenario_factory():
    """Provide a test scenario factory for the entire test session."""
    return TestScenarioFactory()


@pytest.fixture
def small_dataset(test_data_manager):
    """Provide a small test dataset."""
    return test_data_manager.get_dataset("simple", **SMALL_DATASET_PARAMS)


@pytest.fixture
def medium_dataset(test_data_manager):
    """Provide a medium test dataset."""
    return test_data_manager.get_dataset("simple", **MEDIUM_DATASET_PARAMS)


@pytest.fixture
def large_dataset(test_data_manager):
    """Provide a large test dataset."""
    return test_data_manager.get_dataset("simple", **LARGE_DATASET_PARAMS)


@pytest.fixture
def high_dimensional_dataset(test_data_manager):
    """Provide a high-dimensional test dataset."""
    return test_data_manager.get_dataset("high_dimensional", **HIGH_DIM_DATASET_PARAMS)


@pytest.fixture
def clustered_dataset(test_data_manager):
    """Provide a clustered test dataset."""
    return test_data_manager.get_dataset(
        "clustered", n_samples=800, n_features=10, n_clusters=4, contamination=0.1
    )


@pytest.fixture
def time_series_dataset(test_data_manager):
    """Provide a time series test dataset."""
    return test_data_manager.get_dataset(
        "timeseries",
        n_timestamps=500,
        n_features=8,
        anomaly_periods=[(100, 120), (300, 350)],
    )


@pytest.fixture
def mixed_types_dataset(test_data_manager):
    """Provide a mixed data types test dataset."""
    return test_data_manager.get_dataset(
        "mixed_types", n_samples=600, contamination=0.12
    )


@pytest.fixture
def basic_scenario(test_scenario_factory):
    """Provide a basic anomaly detection scenario."""
    return test_scenario_factory.create_basic_anomaly_detection_scenario()


@pytest.fixture
def high_dim_scenario(test_scenario_factory):
    """Provide a high-dimensional scenario."""
    return test_scenario_factory.create_high_dimensional_scenario()


@pytest.fixture
def time_series_scenario(test_scenario_factory):
    """Provide a time series scenario."""
    return test_scenario_factory.create_time_series_scenario()


@pytest.fixture
def mixed_types_scenario(test_scenario_factory):
    """Provide a mixed data types scenario."""
    return test_scenario_factory.create_mixed_data_types_scenario()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Create test data directory if it doesn't exist
    test_data_dir = Path("test_data_cache")
    test_data_dir.mkdir(exist_ok=True)

    yield

    # Cleanup after all tests are done
    # Note: Individual fixtures handle their own cleanup


@pytest.fixture
def dataset_entities(test_data_manager):
    """Provide domain entities created from test data."""
    return test_data_manager.create_domain_entities(
        "simple", n_samples=200, n_features=6, contamination=0.15
    )


@pytest.fixture
def test_detector(test_data_manager):
    """Provide a test detector entity."""
    return test_data_manager.create_test_detector(
        algorithm_name="TestAlgorithm", contamination=0.1
    )


# Parameterized fixtures for testing multiple scenarios
@pytest.fixture(
    params=[
        ("simple", SMALL_DATASET_PARAMS),
        (
            "clustered",
            {"n_samples": 300, "n_features": 8, "n_clusters": 3, "contamination": 0.1},
        ),
        ("mixed_types", {"n_samples": 400, "contamination": 0.1}),
    ]
)
def various_datasets(request, test_data_manager):
    """Provide various types of datasets for parameterized tests."""
    dataset_type, params = request.param
    return test_data_manager.get_dataset(dataset_type, **params)


@pytest.fixture(params=[0.05, 0.1, 0.15, 0.2])
def contamination_rates(request):
    """Provide various contamination rates for testing."""
    return request.param


@pytest.fixture(
    params=[
        ("IsolationForest", {"n_estimators": 50}),
        ("LocalOutlierFactor", {"n_neighbors": 10}),
        ("OneClassSVM", {"nu": 0.1}),
    ]
)
def algorithm_configs(request):
    """Provide various algorithm configurations for testing."""
    algorithm_name, params = request.param
    return algorithm_name, params


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring test data"
    )
    config.addinivalue_line("markers", "small_data: mark test as using small datasets")
    config.addinivalue_line(
        "markers", "medium_data: mark test as using medium datasets"
    )
    config.addinivalue_line("markers", "large_data: mark test as using large datasets")
    config.addinivalue_line(
        "markers", "synthetic_data: mark test as using synthetic data"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    for item in items:
        # Auto-mark tests that use data fixtures
        data_fixtures = [
            "small_dataset",
            "medium_dataset",
            "large_dataset",
            "high_dimensional_dataset",
            "clustered_dataset",
            "time_series_dataset",
            "mixed_types_dataset",
        ]

        if any(fixture in item.fixturenames for fixture in data_fixtures):
            item.add_marker(pytest.mark.requires_data)
            item.add_marker(pytest.mark.synthetic_data)

        # Add size markers based on fixture names
        if any(
            name in item.fixturenames for name in ["small_dataset", "basic_scenario"]
        ):
            item.add_marker(pytest.mark.small_data)
        elif any(name in item.fixturenames for name in ["medium_dataset"]):
            item.add_marker(pytest.mark.medium_data)
        elif any(name in item.fixturenames for name in ["large_dataset"]):
            item.add_marker(pytest.mark.large_data)
