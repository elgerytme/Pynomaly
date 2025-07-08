"""Integration test for Issue #33 - Demonstrate conftest.py works with pytest

This test verifies that conftest.py can be loaded by pytest and that
test fixtures work properly after the syntax fix.
"""

from pathlib import Path

import pandas as pd
import pytest


def test_conftest_fixtures_work(sample_data):
    """Test that conftest.py fixtures work after the syntax fix."""
    # This test uses the sample_data fixture from conftest.py
    # If conftest.py has syntax errors, this test won't run

    assert isinstance(sample_data, pd.DataFrame)
    assert not sample_data.empty
    assert "target" in sample_data.columns
    assert len(sample_data) > 0


def test_conftest_mock_fixtures_work(mock_model):
    """Test that mock fixtures from conftest.py work."""
    # This test uses the mock_model fixture from conftest.py

    assert mock_model is not None
    assert hasattr(mock_model, 'fit')
    assert hasattr(mock_model, 'predict')
    assert hasattr(mock_model, 'decision_function')


def test_conftest_temp_dir_fixture_works(temp_dir):
    """Test that temp_dir fixture works (this was using shutil.rmtree)."""
    # This test uses the temp_dir fixture that was using shutil.rmtree

    temp_path = Path(temp_dir)
    assert temp_path.exists()
    assert temp_path.is_dir()

    # Create a test file
    test_file = temp_path / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()


def test_pytest_collect_works():
    """Test that pytest can collect tests from this file."""
    # If conftest.py has syntax errors, pytest won't be able to collect tests
    # The fact that this test is running means conftest.py is working
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
