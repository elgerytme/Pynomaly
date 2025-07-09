"""Test to verify pytest configuration is working correctly."""

import pytest


@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker is registered."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test that integration marker is registered."""
    assert True


@pytest.mark.ui
def test_ui_marker():
    """Test that UI marker is registered."""
    assert True


@pytest.mark.performance
def test_performance_marker():
    """Test that performance marker is registered."""
    assert True


def test_random_seed_fixture(random_seed):
    """Test that random seed fixture is available."""
    assert isinstance(random_seed, int)
    assert random_seed == 42  # Default seed


def test_isolated_db_session_fixture(isolated_db_session):
    """Test that isolated_db_session fixture is available."""
    # This should be a SQLAlchemy session object
    assert isolated_db_session is not None
    # Try to execute a simple query
    result = isolated_db_session.execute("SELECT 1")
    assert result.scalar() == 1
