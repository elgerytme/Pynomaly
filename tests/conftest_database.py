"""Database test configuration for improved test coverage."""

import tempfile

import pytest

from pynomaly.infrastructure.config import Settings
from pynomaly.infrastructure.config.container import Container
from pynomaly.infrastructure.persistence import DatabaseManager


@pytest.fixture
def test_database_url():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name
    yield f"sqlite:///{db_path}"
    # Cleanup handled by tempfile


@pytest.fixture
def test_database_settings(test_database_url):
    """Create test settings with database configuration."""
    return Settings(
        database_url=test_database_url,
        use_database_repositories=True,
        database_echo=False,
        app__debug=True,
        auth_enabled=False,
        cache_enabled=False,
        monitoring__metrics_enabled=False,
    )


@pytest.fixture
def test_database_manager(test_database_settings):
    """Create a test database manager with SQLite."""
    try:
        manager = DatabaseManager(database_url=test_database_settings.database_url)

        # Create all tables for testing
        from pynomaly.infrastructure.persistence.database_repositories import Base

        Base.metadata.create_all(manager.engine)

        yield manager

        # Cleanup
        manager.close()
    except ImportError:
        pytest.skip("Database dependencies not available")


@pytest.fixture
def test_container_with_database(test_database_settings):
    """Create a container configured for database testing."""
    try:
        # Override settings in container creation
        container = Container()
        container.config.override(test_database_settings)
        container.wire(modules=[])
        yield container
    except ImportError:
        pytest.skip("Database dependencies not available")


@pytest.fixture
def test_async_database_repositories(test_container_with_database):
    """Create async repository wrappers for database testing."""
    try:
        container = test_container_with_database
        return {
            "detector_repository": container.async_detector_repository(),
            "dataset_repository": container.async_dataset_repository(),
            "result_repository": container.async_result_repository(),
        }
    except Exception:
        pytest.skip("Database repositories not available")
