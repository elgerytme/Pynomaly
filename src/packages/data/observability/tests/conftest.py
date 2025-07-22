"""
Pytest configuration for data observability package tests.
"""

import pytest
from typing import Generator
from uuid import uuid4

from src.data_observability.infrastructure.di.container import Container
from src.data_observability.application.facades.observability_facade import DataObservabilityFacade


@pytest.fixture
def container() -> Container:
    """Provide dependency injection container for tests."""
    return Container()


@pytest.fixture
def observability_facade(container: Container) -> DataObservabilityFacade:
    """Provide observability facade for tests."""
    return container.get_observability_facade()


@pytest.fixture
def sample_asset_id() -> str:
    """Provide a sample asset ID for tests."""
    return str(uuid4())


@pytest.fixture
def sample_asset_data() -> dict:
    """Provide sample asset data for tests."""
    return {
        "name": "test_dataset",
        "asset_type": "dataset",
        "location": "s3://test-bucket/test-dataset",
        "data_format": "parquet",
        "description": "Test dataset for observability tests",
        "owner": "test_user",
        "domain": "test_domain"
    }


@pytest.fixture(scope="session")
def test_database() -> Generator[str, None, None]:
    """Provide test database for integration tests."""
    # This would set up a test database
    db_url = "sqlite:///test_observability.db"
    
    # Setup
    yield db_url
    
    # Cleanup
    # Clean up test database
    pass